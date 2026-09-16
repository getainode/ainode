"""#83: a multimodal request routes to an instance that actually accepts images.

The same model id can be served twice with different multimodal limits (Ornith
1.5 is served text-only on one node and with vision on another), so routing on
the model id alone lands an image on the text-only instance and returns vLLM's
"At most 0 image(s) may be provided in one prompt". These tests pin the three
halves of the fix: detection, capability-aware ordering, and failover on that
specific 400 (and on nothing else).
"""

from __future__ import annotations

import asyncio
import json

import ainode.api.server as server
from ainode.api.chat_routes import (
    caps_cache,
    instance_caps_index,
    is_multimodal_limit_error,
    is_multimodal_request,
    order_by_vision,
)
from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeStatus
from ainode.discovery.cluster import ClusterNode, ClusterState

MODEL = "cerebras/Ornith-1.5-35B-A3B-NVFP4"

LOCAL = ("localhost", 8000)
REMOTE = ("10.100.0.15", 8000)

_MM_400 = {"error": {"message": "At most 0 image(s) may be provided in one prompt. "
                                "(parameter=image)",
                     "type": "BadRequestError", "param": "image", "code": 400}}


# ---------------------------------------------------------------- fixtures --

def _node(nid, name=None, model=MODEL, fabric="", api_port=8000,
          status=NodeStatus.ONLINE, instances=None):
    return ClusterNode(node_id=nid, node_name=name or nid, gpu_name="NVIDIA GB10",
                       gpu_memory_gb=121.7, unified_memory=True, model=model,
                       status=status, api_port=api_port, web_port=3000,
                       last_seen=0.0, fabric_ip=fabric, instances=instances or [])


def _cluster(nodes):
    c = ClusterState()
    for n in nodes:
        c.add_node(n)
    return c


class _Collector:
    def __init__(self):
        self.calls: list = []

    def record_request(self, model, ms, error=False):
        self.calls.append((model, error))


class _Up:
    """One upstream answer."""

    def __init__(self, status=200, payload=None):
        self.status = status
        self.headers = {"Content-Type": "application/json"}
        self._payload = payload if payload is not None else {"ok": 1}

    async def read(self):
        return json.dumps(self._payload).encode()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    """Answers per host from a {host: (status, payload)} map, recording each URL."""

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.tried: list = []

    def request(self, method, url, **kwargs):
        self.tried.append(url)
        for host, answer in self.answers.items():
            if f"//{host}:" in url:
                if isinstance(answer, Exception):
                    raise answer
                status, payload = answer
                return _Up(status, payload)
        return _Up(200, {"ok": 1})


def _app(cluster, session=None, caps=None, node_name="Spark-1-DGX"):
    app = {
        "config": NodeConfig(node_id="spark1", node_name=node_name,
                             api_port=8000, web_port=3000, model=MODEL),
        "cluster_state": cluster,
        "client_session": session or _Session(),
        "metrics_collector": _Collector(),
    }
    if caps:
        caps_cache(app).update(caps)
    return app


def _proxy(app, body):
    class _R:
        method = "POST"
        path = "/v1/chat/completions"
        headers: dict = {}

        def __init__(self):
            self.app = app

        async def read(self):
            return json.dumps(body).encode()

    resp = asyncio.run(server.proxy_to_vllm(_R()))
    payload = json.loads(resp.body.decode()) if resp.body else {}
    return resp.status, payload


def _image_body():
    return {"model": MODEL, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}


def _two_nodes():
    return _cluster([
        _node("spark1", "Spark-1-DGX"),                              # local
        _node("spark3", "Spark-3-DGX", fabric="10.100.0.15"),        # remote
    ])


# --------------------------------------------------------------- detection --

def test_detects_an_image_part():
    assert is_multimodal_request(_image_body()) is True


def test_detects_an_audio_part():
    body = {"model": MODEL, "messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}]}]}
    assert is_multimodal_request(body) is True


def test_detects_a_video_part():
    body = {"model": MODEL, "messages": [{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": "https://example/clip.mp4"}}]}]}
    assert is_multimodal_request(body) is True


def test_detects_a_part_that_carries_only_the_key():
    """Some clients omit `type` and send the modality key alone."""
    body = {"model": MODEL, "messages": [{"role": "user", "content": [
        {"image_url": {"url": "data:image/png;base64,AAAA"}}]}]}
    assert is_multimodal_request(body) is True


def test_a_text_request_is_not_multimodal():
    assert is_multimodal_request({"model": MODEL, "messages": [
        {"role": "user", "content": "hi"}]}) is False
    assert is_multimodal_request({"model": MODEL, "messages": [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}]}) is False
    assert is_multimodal_request({}) is False
    assert is_multimodal_request(None) is False


def test_recognises_the_vllm_multimodal_limit_400():
    assert is_multimodal_limit_error(
        "At most 0 image(s) may be provided in one prompt. (parameter=image)") is True
    assert is_multimodal_limit_error("At most 0 video(s) may be provided in one prompt.") is True
    assert is_multimodal_limit_error("This model's maximum context length is 4096 tokens") is False
    assert is_multimodal_limit_error("") is False


# ---------------------------------------------------------------- ordering --

def test_caps_index_is_keyed_the_way_the_proxy_addresses_a_candidate():
    app = _app(_two_nodes(), caps={("spark3", 8000, MODEL): {"vision": True}})
    index = instance_caps_index(app, MODEL)
    assert index[LOCAL]["vision"] is None          # never probed
    assert index[REMOTE]["vision"] is True
    assert index[REMOTE]["node_name"] == "Spark-3-DGX"


def test_ordering_puts_vision_first_then_unknown_and_drops_refusers():
    index = {
        ("a", 8000): {"vision": None},
        ("b", 8000): {"vision": True},
        ("c", 8000): {"vision": False},
    }
    ordered, refused = order_by_vision(
        [("a", 8000), ("b", 8000), ("c", 8000)], index)
    assert ordered == [("b", 8000), ("a", 8000)]
    assert refused == [("c", 8000)]


def test_an_image_skips_the_local_text_only_instance():
    """Local hop first is the rule for text; caps override it for an image."""
    session = _Session({"10.100.0.15": (200, {"ok": 1})})
    app = _app(_two_nodes(), session=session, caps={
        ("spark1", 8000, MODEL): {"vision": False},
        ("spark3", 8000, MODEL): {"vision": True},
    })
    status, _ = _proxy(app, _image_body())
    assert status == 200
    assert session.tried == ["http://10.100.0.15:8000/v1/chat/completions"]


def test_a_text_request_keeps_the_local_first_order():
    session = _Session()
    app = _app(_two_nodes(), session=session, caps={
        ("spark1", 8000, MODEL): {"vision": False},
        ("spark3", 8000, MODEL): {"vision": True},
    })
    status, _ = _proxy(app, {"model": MODEL, "messages": [
        {"role": "user", "content": "hi"}]})
    assert status == 200
    assert session.tried == ["http://localhost:8000/v1/chat/completions"]


# ---------------------------------------------------------------- failover --

def test_fails_over_past_the_zero_image_400_and_remembers_it():
    session = _Session({"localhost": (400, _MM_400), "10.100.0.15": (200, {"ok": 1})})
    app = _app(_two_nodes(), session=session)      # nothing probed yet
    status, body = _proxy(app, _image_body())
    assert status == 200 and body == {"ok": 1}
    assert len(session.tried) == 2                 # text-only first, then vision
    # The refusal is remembered in the ONE caps cache, so the next image request
    # skips that instance and the chat view's badge learns it too.
    caps = caps_cache(app)[("spark1", 8000, MODEL)]
    assert caps["vision"] is False
    assert "may be provided in one prompt" in caps["vision_error"]
    assert caps["probed"] is False


def test_a_remembered_refusal_is_not_tried_again():
    session = _Session({"localhost": (400, _MM_400), "10.100.0.15": (200, {"ok": 1})})
    app = _app(_two_nodes(), session=session)
    _proxy(app, _image_body())
    _proxy(app, _image_body())
    # 2 calls for the first request (refusal + failover), 1 for the second.
    assert len(session.tried) == 3
    assert session.tried[-1] == "http://10.100.0.15:8000/v1/chat/completions"


def test_another_400_is_returned_as_is_and_never_retried():
    other = {"error": {"message": "This model's maximum context length is 4096 tokens",
                       "type": "BadRequestError"}}
    session = _Session({"localhost": (400, other), "10.100.0.15": (200, {"ok": 1})})
    app = _app(_two_nodes(), session=session)
    status, body = _proxy(app, _image_body())
    assert status == 400
    assert body == other                           # the engine's answer, untouched
    assert len(session.tried) == 1                 # no failover
    assert ("spark1", 8000, MODEL) not in caps_cache(app)


def test_a_404_from_the_engine_is_not_retried():
    session = _Session({"localhost": (404, {"error": "nope"}),
                        "10.100.0.15": (200, {"ok": 1})})
    app = _app(_two_nodes(), session=session)
    status, _ = _proxy(app, _image_body())
    assert status == 404 and len(session.tried) == 1


# --------------------------------------------------------------- exhausted --

def test_every_instance_refusing_images_returns_a_400_naming_the_nodes():
    session = _Session({"localhost": (400, _MM_400), "10.100.0.15": (400, _MM_400)})
    app = _app(_two_nodes(), session=session)
    status, body = _proxy(app, _image_body())
    assert status == 400
    msg = body["error"]["message"]
    assert "no instance accepts images" in msg
    assert "Spark-1-DGX" in msg and "Spark-3-DGX" in msg
    assert "limit-mm-per-prompt" in msg
    assert body["error"]["code"] == "no_multimodal_instance"


def test_all_candidates_known_text_only_answers_without_a_request():
    session = _Session()
    app = _app(_two_nodes(), session=session, caps={
        ("spark1", 8000, MODEL): {"vision": False},
        ("spark3", 8000, MODEL): {"vision": False},
    })
    status, body = _proxy(app, _image_body())
    assert status == 400
    assert session.tried == []                     # no engine was bothered
    assert "no instance accepts images" in body["error"]["message"]
    assert app["metrics_collector"].calls == [(MODEL, True)]


def test_an_unreachable_instance_still_502s_a_text_request():
    """The multimodal 400 must not swallow a genuine transport failure."""
    session = _Session({"localhost": server.aiohttp.ClientError("ghost")})
    app = _app(_cluster([_node("spark1", "Spark-1-DGX")]), session=session)
    status, body = _proxy(app, {"model": MODEL, "messages": [
        {"role": "user", "content": "hi"}]})
    assert status == 502
    assert "no reachable node" in body["error"]["message"]
