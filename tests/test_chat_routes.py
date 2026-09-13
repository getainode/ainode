"""Chat view routes: /api/models/card and /api/models/caps.

The card must never invent a field: everything it reports comes from cluster
state, a per-instance config snapshot, the engine's own /v1/models, or the
catalog, and anything we cannot see stays null. The caps route must report what
the engine ACCEPTED, carrying the engine's own refusal text.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from ainode.api.chat_routes import (
    _pick_instance,
    _quant_from_model_id,
    _speculative,
    catalog_entry,
    fleet_instances,
    handle_model_caps,
    handle_model_card,
)
from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeStatus
from ainode.discovery.cluster import ClusterNode, ClusterState
from ainode.discovery.instance import InstanceRecord
from ainode.engine.instance_manager import InstanceManager


# ---------------------------------------------------------------- fixtures --

def _node(nid, name=None, model="", fabric="", instances=None,
          status=NodeStatus.ONLINE, api_port=8000, gpu="NVIDIA GB10", vram=121.7):
    return ClusterNode(node_id=nid, node_name=name or nid, gpu_name=gpu,
                       gpu_memory_gb=vram, unified_memory=True, model=model,
                       status=status, api_port=api_port, web_port=3000,
                       last_seen=0.0, fabric_ip=fabric, instances=instances or [])


def _cluster(nodes):
    c = ClusterState()
    for n in nodes:
        c.add_node(n)
    return c


class _Req:
    """Minimal stand-in for web.Request: the handlers only read app + query."""

    def __init__(self, app, query=None):
        self.app = app
        self.query = query or {}


class _Resp:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def json(self, content_type=None):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    """Records every call and answers from a url->(status, payload) map."""

    def __init__(self, gets=None, posts=None):
        self.gets = gets or {}
        self.posts = posts or []
        self.get_calls: list = []
        self.post_calls: list = []

    def get(self, url, **kwargs):
        self.get_calls.append(url)
        status, payload = self.gets.get(url, (404, {}))
        return _Resp(status, payload)

    def post(self, url, json=None, **kwargs):
        self.post_calls.append((url, json))
        if not self.posts:
            return _Resp(200, {})
        status, payload = self.posts.pop(0)
        return _Resp(status, payload)


def _engine_models(model, max_model_len=262144):
    return (200, {"object": "list",
                  "data": [{"id": model, "object": "model",
                            "max_model_len": max_model_len}]})


def _app(cluster, config=None, session=None, manager=None):
    app = {
        "config": config or NodeConfig(node_id="spark1", node_name="Spark-1-DGX",
                                       api_port=8000, web_port=3000, model=""),
        "cluster_state": cluster,
        "client_session": session,
    }
    if manager is not None:
        app["instances"] = manager
    return app


def _card(app, **query):
    resp = asyncio.run(handle_model_card(_Req(app, query)))
    return resp.status, json.loads(resp.body)


def _caps(app, **query):
    resp = asyncio.run(handle_model_caps(_Req(app, query)))
    return resp.status, json.loads(resp.body)


# ------------------------------------------------------------ fleet truth --

def test_fleet_lists_local_primary_and_remote_instances():
    cluster = _cluster([
        _node("spark1", "Spark-1-DGX", model="A"),
        _node("spark4", "Spark-4-GX10", model="B", fabric="10.100.0.17"),
    ])
    entries = fleet_instances(_app(cluster))
    got = {(e["model"], e["node_name"], e["host"], e["port"]) for e in entries}
    assert got == {("A", "Spark-1-DGX", "localhost", 8000),
                   ("B", "Spark-4-GX10", "10.100.0.17", 8000)}


def test_fleet_includes_stacked_instances_on_their_own_port():
    cluster = _cluster([_node("spark1", model="A", instances=[
        {"model": "B", "api_port": 8001, "status": "serving"}])])
    entries = fleet_instances(_app(cluster))
    assert ("B", 8001) in {(e["model"], e["port"]) for e in entries}


def test_fleet_skips_offline_nodes_and_unroutable_peers():
    cluster = _cluster([
        _node("spark2", model="B", fabric="10.100.0.13", status=NodeStatus.OFFLINE),
        _node("spark3", model="C", fabric=""),  # remote with no fabric IP
    ])
    assert fleet_instances(_app(cluster)) == []


def test_fleet_merges_the_local_instance_manager():
    # A model that just launched is in the manager before the next broadcast.
    manager = InstanceManager(base_port=8000)
    manager.add(InstanceRecord(instance_id="i1", model="Fresh", api_port=8002),
                object())
    entries = fleet_instances(_app(_cluster([]), manager=manager))
    assert [(e["model"], e["port"], e["local"]) for e in entries] == [("Fresh", 8002, True)]


def test_pick_instance_disambiguates_on_node_and_port():
    entries = [
        {"model": "A", "node_id": "spark1", "port": 8000, "local": True},
        {"model": "A", "node_id": "spark4", "port": 8001, "local": False},
    ]
    assert _pick_instance(entries, "A")["node_id"] == "spark1"      # local first
    assert _pick_instance(entries, "A", "spark4")["port"] == 8001
    assert _pick_instance(entries, "A", "", "8001")["node_id"] == "spark4"
    assert _pick_instance(entries, "B") is None


# -------------------------------------------------------------------- card --

def test_card_requires_a_model():
    status, body = _card(_app(_cluster([])))
    assert status == 400 and "model" in body["error"]["message"]


def test_card_404s_on_a_model_nobody_serves():
    status, body = _card(_app(_cluster([_node("spark1", model="A")])), model="Ghost")
    assert status == 404 and body["error"]["type"] == "model_not_found"


def test_card_reports_the_node_and_its_gpu():
    cluster = _cluster([_node("spark4", "Spark-4-GX10", model="Nemo",
                              fabric="10.100.0.17", gpu="NVIDIA GB10", vram=121.6)])
    session = FakeSession({"http://10.100.0.17:8000/v1/models": _engine_models("Nemo")})
    status, body = _card(_app(cluster, session=session), model="Nemo")
    assert status == 200
    assert body["instance"]["node_name"] == "Spark-4-GX10"
    assert body["instance"]["local"] is False
    assert body["hardware"] == {"gpu_name": "NVIDIA GB10", "gpu_count": 1,
                                "gpu_memory_gb": 121.6, "unified_memory": True}


def test_card_takes_context_length_from_the_engine():
    cluster = _cluster([_node("spark1", model="Qwen")])
    session = FakeSession({"http://localhost:8000/v1/models": _engine_models("Qwen", 262144)})
    _, body = _card(_app(cluster, session=session), model="Qwen")
    assert body["serving"]["max_model_len"] == 262144
    assert body["serving"]["max_model_len_source"] == "engine"


def test_card_context_length_is_null_when_the_engine_does_not_answer():
    cluster = _cluster([_node("spark1", model="Qwen")])
    _, body = _card(_app(cluster, session=FakeSession()), model="Qwen")
    assert body["serving"]["max_model_len"] is None
    assert body["serving"]["max_model_len_source"] is None


def test_card_reads_engine_image_and_speculative_from_the_local_instance():
    cfg = NodeConfig(node_id="spark1", api_port=8001, model="Nemo",
                     engine_image="vllm/vllm-openai:v0.27.1", kv_cache_dtype="fp8",
                     gpu_memory_utilization=0.91,
                     extra_vllm_args=["--moe-backend", "marlin",
                                      "--speculative_config.model", "draft-repo",
                                      "--speculative_config.num_speculative_tokens", "3"])
    backend = type("B", (), {"config": cfg})()
    manager = InstanceManager(base_port=8000)
    manager.add(InstanceRecord(instance_id="i1", model="Nemo", api_port=8001), backend)
    cluster = _cluster([_node("spark1", model="", instances=[
        {"model": "Nemo", "api_port": 8001, "status": "serving",
         "tensor_parallel_size": 2, "member_node_ids": ["spark1", "spark2"]}])])
    session = FakeSession({"http://localhost:8001/v1/models": _engine_models("Nemo", 65536)})
    _, body = _card(_app(cluster, session=session, manager=manager), model="Nemo")
    s = body["serving"]
    assert s["config_source"] == "instance"
    assert s["engine_image"] == "vllm/vllm-openai:v0.27.1"
    assert s["kv_cache_dtype"] == "fp8"
    assert s["gpu_memory_utilization"] == 0.91
    assert s["tensor_parallel_size"] == 2
    assert "draft-repo" in s["speculative"] and "3" in s["speculative"]
    assert body["instance"]["nodes"] == ["spark1", "spark2"]
    assert body["hardware"]["gpu_count"] == 2


def test_card_engine_image_is_null_for_a_remote_node_that_does_not_answer():
    cluster = _cluster([_node("spark4", model="Nemo", fabric="10.100.0.17")])
    _, body = _card(_app(cluster, session=FakeSession()), model="Nemo")
    assert body["serving"]["engine_image"] is None
    assert body["serving"]["config_source"] is None
    assert body["serving"]["extra_vllm_args"] is None


def test_card_uses_a_remote_nodes_config_only_when_it_describes_this_model():
    cluster = _cluster([_node("spark4", model="Nemo", fabric="10.100.0.17")])
    # The peer's config describes a DIFFERENT model: ignore it rather than
    # showing another model's engine image on this card.
    session = FakeSession({
        "http://10.100.0.17:3000/api/config": (200, {"model": "Other",
                                                     "engine_image": "wrong:1"}),
    })
    _, body = _card(_app(cluster, session=session), model="Nemo")
    assert body["serving"]["engine_image"] is None and body["serving"]["config_source"] is None

    session = FakeSession({
        "http://10.100.0.17:3000/api/config": (200, {"model": "Nemo",
                                                     "engine_image": "right:1",
                                                     "kv_cache_dtype": "fp8"}),
    })
    _, body = _card(_app(cluster, session=session), model="Nemo")
    assert body["serving"]["engine_image"] == "right:1"
    assert body["serving"]["config_source"] == "node_config"


def test_card_quantization_prefers_the_catalog_then_the_model_id():
    cluster = _cluster([_node("spark1", model="acme/Thing-NVFP4")])
    _, body = _card(_app(cluster, session=FakeSession()), model="acme/Thing-NVFP4")
    assert body["serving"]["quantization"] == "NVFP4"
    assert body["serving"]["quantization_source"] == "model_id"


def test_card_quantization_is_null_when_nothing_states_it():
    cluster = _cluster([_node("spark1", model="acme/Thing")])
    _, body = _card(_app(cluster, session=FakeSession()), model="acme/Thing")
    assert body["serving"]["quantization"] is None
    assert body["serving"]["quantization_source"] is None


def test_card_links_hugging_face_from_a_repo_shaped_id():
    cluster = _cluster([_node("spark1", model="acme/Thing")])
    _, body = _card(_app(cluster, session=FakeSession()), model="acme/Thing")
    assert body["hf_url"] == "https://huggingface.co/acme/Thing"
    assert body["hf_source"] == "model_id"


def test_card_has_no_hugging_face_link_for_a_bare_alias():
    cluster = _cluster([_node("spark1", model="my-alias")])
    _, body = _card(_app(cluster, session=FakeSession()), model="my-alias")
    assert body["hf_url"] is None and body["hf_repo"] is None


def test_card_carries_the_catalog_description_when_the_model_is_curated():
    from ainode.models.registry import CURATED_CLUSTER_MODELS
    info = next(iter(CURATED_CLUSTER_MODELS.values()))
    cluster = _cluster([_node("spark1", model=info.hf_repo)])
    _, body = _card(_app(cluster, session=FakeSession()), model=info.hf_repo)
    assert body["catalog"]["description"] == info.description
    assert body["hf_repo"] == info.hf_repo and body["hf_source"] == "catalog"


def test_card_catalog_is_null_for_an_unknown_model():
    cluster = _cluster([_node("spark1", model="acme/Nothing-Like-This")])
    _, body = _card(_app(cluster, session=FakeSession()), model="acme/Nothing-Like-This")
    assert body["catalog"] is None


@pytest.mark.parametrize("model,expected", [
    ("unsloth/Qwen3.8-27B-NVFP4", "NVFP4"),
    ("org/Model-AWQ", "AWQ"),
    ("org/Model.gguf", "GGUF"),
    ("org/Model-FP8-Dynamic", "FP8"),
    ("org/Plain-Model", None),
    ("org/awqward-name", None),          # substring must not count as a format
])
def test_quant_from_model_id(model, expected):
    assert _quant_from_model_id(model) == expected


@pytest.mark.parametrize("args,hit", [
    (["--speculative-config", "{}"], True),
    (["--speculative_config.num_speculative_tokens", "3"], True),
    (["--num-speculative-tokens", "4"], True),
    (["--moe-backend", "marlin"], False),
    ([], False),
])
def test_speculative_detection(args, hit):
    assert (_speculative(args) is not None) is hit


def test_catalog_entry_matches_id_or_repo():
    from ainode.models.registry import CURATED_CLUSTER_MODELS
    info = next(iter(CURATED_CLUSTER_MODELS.values()))
    assert catalog_entry(info.hf_repo) is info
    assert catalog_entry(info.id) is info
    assert catalog_entry("nope/nope") is None


# -------------------------------------------------------------------- caps --

def test_caps_requires_a_model():
    status, body = _caps(_app(_cluster([])))
    assert status == 400


def test_caps_404s_on_a_model_nobody_serves():
    status, _ = _caps(_app(_cluster([_node("spark1", model="A")])), model="Ghost")
    assert status == 404


def test_caps_reports_accepted_capabilities():
    cluster = _cluster([_node("spark1", model="Qwen")])
    session = FakeSession(posts=[(200, {"choices": []}), (200, {"choices": []})])
    status, body = _caps(_app(cluster, session=session), model="Qwen")
    assert status == 200
    assert body["vision"] is True and body["tools"] is True
    assert body["vision_error"] is None


def test_caps_probe_sends_a_16px_image_and_a_tools_array():
    cluster = _cluster([_node("spark1", model="Qwen")])
    session = FakeSession(posts=[(200, {}), (200, {})])
    _caps(_app(cluster, session=session), model="Qwen")
    (vision_url, vision_body), (tools_url, tools_body) = session.post_calls
    assert vision_url == "http://localhost:8000/v1/chat/completions"
    parts = vision_body["messages"][0]["content"]
    img = [p for p in parts if p["type"] == "image_url"][0]
    assert img["image_url"]["url"].startswith("data:image/png;base64,")
    # 16x16, not 1x1: the IHDR width byte is 0x10.
    import base64
    png = base64.b64decode(img["image_url"]["url"].split(",", 1)[1])
    assert png[16:24] == bytes([0, 0, 0, 16, 0, 0, 0, 16])
    assert vision_body["max_tokens"] == 1
    assert tools_body["tools"][0]["function"]["name"] == "ping"
    assert tools_body["max_tokens"] == 1
    assert tools_url == vision_url


def test_caps_reports_a_refusal_with_the_engines_own_words():
    cluster = _cluster([_node("spark1", model="Nemo")])
    session = FakeSession(posts=[
        (400, {"error": {"message": "nvidia/Nemo is not a multimodal model"}}),
        (400, {"error": {"message": "tool-call-parser not configured"}}),
    ])
    _, body = _caps(_app(cluster, session=session), model="Nemo")
    assert body["vision"] is False
    assert body["vision_error"] == "nvidia/Nemo is not a multimodal model"
    assert body["tools"] is False and "tool-call-parser" in body["tools_error"]


def test_caps_are_null_when_the_engine_is_unreachable():
    cluster = _cluster([_node("spark1", model="Qwen")])
    status, body = _caps(_app(cluster, session=None), model="Qwen")
    assert status == 200 and body["vision"] is None and body["tools"] is None


def test_caps_never_probes_reasoning():
    cluster = _cluster([_node("spark1", model="Qwen")])
    session = FakeSession(posts=[(200, {}), (200, {})])
    _, body = _caps(_app(cluster, session=session), model="Qwen")
    assert body["reasoning"] is None
    assert "observed" in body["reasoning_note"]
    assert len(session.post_calls) == 2          # exactly two probes, no third


def test_caps_are_cached_per_instance_and_refreshed_on_demand():
    cluster = _cluster([
        _node("spark1", model="Qwen"),
        _node("spark4", model="Qwen", fabric="10.100.0.17"),
    ])
    session = FakeSession(posts=[(200, {}), (200, {})])
    app = _app(cluster, session=session)
    _caps(app, model="Qwen")
    assert len(session.post_calls) == 2
    _caps(app, model="Qwen")                     # served from cache
    assert len(session.post_calls) == 2

    # A different instance of the same model id is a different cache entry.
    session.posts = [(400, {"error": {"message": "not a multimodal model"}}), (200, {})]
    _, other = _caps(app, model="Qwen", node_id="spark4")
    assert len(session.post_calls) == 4 and other["vision"] is False

    # fresh=1 re-probes the first instance (an engine relaunch changes flags).
    session.posts = [(200, {}), (200, {})]
    _caps(app, model="Qwen", fresh="1")
    assert len(session.post_calls) == 6


# ------------------------------------------------------- route registration --

@pytest.mark.asyncio
async def test_card_and_caps_are_reachable_through_the_real_app():
    """Wiring: /api/models/{model_id} is registered later and would swallow both
    paths if register_chat_routes ran after register_model_routes."""
    import socket

    import pytest_asyncio  # noqa: F401  (import guard: the app fixture style)
    from aiohttp.test_utils import TestClient, TestServer

    from ainode.api.server import create_app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    app = create_app(config=NodeConfig(node_id="n1", node_name="N1", model="",
                                       api_port=free_port), engine=None)
    async with TestClient(TestServer(app)) as client:
        # No model param: our handler answers 400. A shadowed route would 404
        # or hand back a catalog entry for the literal id "card".
        resp = await client.get("/api/models/card")
        assert resp.status == 400
        assert "model" in (await resp.json())["error"]["message"]
        resp = await client.get("/api/models/caps")
        assert resp.status == 400


def test_fleet_keeps_the_instance_record_for_a_primary_listed_twice():
    """A node advertises its primary both as `model` and in `instances`. The
    entry that survives must be the one carrying TP and the member node set."""
    cluster = _cluster([_node("spark4", "Spark-4-GX10", model="Nemo",
                              fabric="10.100.0.17", instances=[
                                  {"model": "Nemo", "api_port": 8000, "status": "serving",
                                   "tensor_parallel_size": 2}])])
    entries = fleet_instances(_app(cluster))
    assert len(entries) == 1
    assert entries[0]["record"]["tensor_parallel_size"] == 2

    session = FakeSession({"http://10.100.0.17:8000/v1/models": _engine_models("Nemo")})
    _, body = _card(_app(cluster, session=session), model="Nemo")
    assert body["serving"]["tensor_parallel_size"] == 2


# ----------------------------------------------------------- chat view UI --

def test_chat_view_markup_carries_the_card_and_the_controls():
    from ainode.web.serve import get_index_html
    html = get_index_html()
    for needle in ("model-card-body", "model-card-toggle", "chat-thinking",
                   "chat-temp", "chat-maxtok", "chat-system", "chat-stop"):
        assert needle in html, needle


def test_chat_client_asks_for_usage_and_reads_the_two_routes():
    from ainode.web.serve import STATIC_DIR
    js = (STATIC_DIR / "js" / "app.js").read_text()
    # Decode rate must come from the server's usage block, not chunk counting.
    assert "stream_options" in js and "include_usage" in js
    # The thinking toggle only sends the kwarg when thinking is OFF.
    assert "enable_thinking" in js
    assert "/api/models/card" in js and "/api/models/caps" in js
    # The picker is built from what the fleet is actually serving.
    assert "/api/server/status" in js
