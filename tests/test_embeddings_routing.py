"""`POST /v1/embeddings` asks the fleet before it asks this process.

The feature was dead on our hardware: `sentence-transformers` is not in the AINode
container image, so every embeddings request the master answered itself came back
`dependency_missing`, and nothing in the fleet served embeddings at all. It does now,
as an ordinary stacked vLLM instance with `--runner pooling`, which means the model id
is enough to say where the vectors are, exactly as it is for a chat completion.

Four things are pinned here, and they are the whole of the routing rule:

  * a model a fleet instance serves is forwarded to it, body verbatim,
  * a candidate that will not connect fails over to the next one,
  * a model nothing serves falls back to the in-process manager, unchanged,
  * a stacked instance on a peer's own engine port is found, which is the shape an
    embedding model is actually launched in.

The fake upstream is the one `tests/test_routing_caps.py` uses on the chat proxy, for
the same reason: the point is which URL was called with which bytes, and a real
server would only add a port number to the test.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types

import pytest

from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeStatus
from ainode.discovery.cluster import ClusterNode, ClusterState
from ainode.embeddings.api_routes import (
    fleet_candidates,
    forward_to_fleet,
    handle_v1_embeddings,
)
from ainode.embeddings.manager import EmbeddingManager

EMBED = "Qwen/Qwen3-Embedding-0.6B"
CHAT = "fraserprice/DeepSeek-V4-Flash-DSpark"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"

#: What a pooling engine answers with. Two dimensions is enough: the handler passes
#: the body through and never looks inside it.
_OK = {"object": "list", "model": EMBED, "usage": {"prompt_tokens": 3},
       "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}]}


# ---------------------------------------------------------------- fixtures --

class _FakeSentenceTransformer:
    """The in-process fallback's library, faked, so no test reaches HuggingFace."""

    def __init__(self, model_id, *args, **kwargs):
        self.model_id = model_id
        self.max_seq_length = 256

    def get_sentence_embedding_dimension(self):
        return 8

    def encode(self, texts, convert_to_numpy=True, **kwargs):
        return [[float(i + j) for j in range(8)] for i, _ in enumerate(texts)]


@pytest.fixture(autouse=True)
def _fake_sentence_transformers(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    yield


def _node(node_id, name, model="", fabric="", api_port=8000, instances=None,
          status=NodeStatus.ONLINE):
    return ClusterNode(node_id=node_id, node_name=name, gpu_name="NVIDIA GB10",
                       gpu_memory_gb=121.7, unified_memory=True, model=model,
                       status=status, api_port=api_port, web_port=3000,
                       last_seen=0.0, fabric_ip=fabric, instances=instances or [])


def _cluster(nodes):
    state = ClusterState()
    for node in nodes:
        state.add_node(node)
    return state


class _Up:
    """One upstream answer, or the exception raised instead of connecting."""

    def __init__(self, status=200, payload=None):
        self.status = status
        self.headers = {"Content-Type": "application/json; charset=utf-8"}
        self._payload = _OK if payload is None else payload

    async def read(self):
        return json.dumps(self._payload).encode()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    """Answers per host from {host: (status, payload) | Exception}, recording calls."""

    #: The app's cleanup hook closes whatever sits at app["client_session"], so the
    #: fake has to answer the two things it asks for.
    closed = False

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.tried: list = []
        self.bodies: list = []
        self.headers: list = []

    async def close(self):
        self.closed = True

    def post(self, url, **kwargs):
        self.tried.append(url)
        self.bodies.append(kwargs.get("data"))
        self.headers.append(kwargs.get("headers") or {})
        for host, answer in self.answers.items():
            if f"//{host}:" in url:
                if isinstance(answer, Exception):
                    raise answer
                status, payload = answer
                return _Up(status, payload)
        return _Up(200, _OK)


class _Request:
    """The two things the handler reads off a request, plus the app it hangs on."""

    method = "POST"

    def __init__(self, app, body, headers=None):
        self.app = app
        self._body = json.dumps(body).encode() if isinstance(body, dict) else body
        self.headers = headers or {"Content-Type": "application/json",
                                   "Authorization": "Bearer ainode"}
        self.tags: dict = {}

    async def read(self):
        return self._body

    async def json(self):
        return json.loads(self._body)

    def __setitem__(self, key, value):
        self.tags[key] = value


def _app(cluster, session=None, node_id="spark1", api_port=8000):
    return {
        "config": NodeConfig(node_id=node_id, node_name="Spark-1-DGX",
                             api_port=api_port, web_port=3000, model=CHAT),
        "cluster_state": cluster,
        "client_session": session if session is not None else _Session(),
        "embedding_manager": EmbeddingManager(),
    }


def _post(app, body):
    request = _Request(app, body)
    response = asyncio.run(handle_v1_embeddings(request))
    payload = json.loads(response.body.decode()) if response.body else {}
    return response, payload, request


#: Spark-2 heads a chat model on its own port and stacks the embedding model on
#: :8001, which is how the real one is served. The local node serves neither.
def _fleet_with_stacked_embedder():
    return _cluster([
        _node("spark1", "Spark-1-DGX", model=CHAT),                   # local
        _node("spark2", "Spark-2-DGX", model=CHAT, fabric="10.100.0.12",
              instances=[{"model": EMBED, "api_port": 8001,
                          "status": "serving"}]),
    ])


# -------------------------------------------------------- candidate lookup --

def test_a_stacked_embedding_instance_on_a_peer_is_a_candidate():
    """The shape that matters: the model is not any node's primary, it is a second
    engine on a port of its own."""
    cands = fleet_candidates(_app(_fleet_with_stacked_embedder()), EMBED)
    assert cands == [("10.100.0.12", 8001)]


def test_a_model_nothing_serves_has_no_candidates():
    assert fleet_candidates(_app(_fleet_with_stacked_embedder()), MINILM) == []


def test_the_local_node_is_the_first_candidate_for_its_own_model():
    """A worker routes by the same list the proxy uses, so the local hop is first."""
    cluster = _cluster([
        _node("spark1", "Spark-1-DGX", model=EMBED),
        _node("spark2", "Spark-2-DGX", model=EMBED, fabric="10.100.0.12"),
    ])
    assert fleet_candidates(_app(cluster), EMBED) == [("localhost", 8000),
                                                     ("10.100.0.12", 8000)]


def test_an_app_with_no_cluster_state_has_no_candidates():
    """A bare app must fall through to the in-process path, not raise."""
    app = _app(None)
    app.pop("cluster_state")
    assert fleet_candidates(app, EMBED) == []


# ------------------------------------------------------------- forwarding ---

def test_a_fleet_instance_gets_the_body_verbatim():
    session = _Session({"10.100.0.12": (200, _OK)})
    app = _app(_fleet_with_stacked_embedder(), session)
    body = {"model": EMBED, "input": ["hello", "world"], "encoding_format": "float"}
    response, payload, request = _post(app, body)

    assert response.status == 200
    assert payload == _OK
    assert session.tried == ["http://10.100.0.12:8001/v1/embeddings"]
    # Verbatim: the same bytes, so a field this handler has never heard of (and a
    # caller's exact float formatting) reaches the engine untouched.
    assert json.loads(session.bodies[0]) == body
    assert request.tags["_log_model"] == EMBED


def test_the_upstream_status_and_content_type_come_back():
    """An engine's 400 is the caller's answer, charset stripped off the type the way
    the chat proxy strips it (web.Response rejects one)."""
    session = _Session({"10.100.0.12": (400, {"error": {"message": "too long"}})})
    app = _app(_fleet_with_stacked_embedder(), session)
    response, payload, _ = _post(app, {"model": EMBED, "input": "x" * 10})
    assert response.status == 400
    assert response.content_type == "application/json"
    assert payload["error"]["message"] == "too long"


def test_the_proxy_headers_are_passed_on_without_the_hop_by_hop_ones():
    session = _Session()
    app = _app(_fleet_with_stacked_embedder(), session)
    request = _Request(app, {"model": EMBED, "input": "hi"},
                       headers={"Authorization": "Bearer sekret",
                                "Host": "master:3000",
                                "Content-Length": "999",
                                "Transfer-Encoding": "chunked"})
    asyncio.run(handle_v1_embeddings(request))
    sent = {k.lower() for k in session.headers[0]}
    assert "authorization" in sent
    assert "host" not in sent and "content-length" not in sent
    assert "transfer-encoding" not in sent


def test_a_dead_candidate_fails_over_to_the_next_one():
    """Two nodes serve it, the first will not connect. A stale claim from a crashed
    node must not 502 a request another node can answer."""
    import aiohttp

    cluster = _cluster([
        _node("spark1", "Spark-1-DGX", model=CHAT),
        _node("spark2", "Spark-2-DGX", model=CHAT, fabric="10.100.0.12",
              instances=[{"model": EMBED, "api_port": 8001, "status": "serving"}]),
        _node("spark3", "Spark-3-DGX", model=CHAT, fabric="10.100.0.13",
              instances=[{"model": EMBED, "api_port": 8001, "status": "serving"}]),
    ])
    session = _Session({"10.100.0.12": aiohttp.ClientConnectionError("refused"),
                        "10.100.0.13": (200, _OK)})
    response, payload, _ = _post(_app(cluster, session), {"model": EMBED,
                                                          "input": "hi"})
    assert response.status == 200
    assert payload == _OK
    assert session.tried == ["http://10.100.0.12:8001/v1/embeddings",
                             "http://10.100.0.13:8001/v1/embeddings"]


def test_every_candidate_dead_is_a_502_and_not_a_silent_cpu_answer():
    """The fleet claims to serve this model. Quietly answering with MiniLM vectors
    from the in-process manager would hand the caller a different model's numbers
    under the id they asked for."""
    import aiohttp

    session = _Session({"10.100.0.12": aiohttp.ClientConnectionError("refused")})
    response, payload, _ = _post(_app(_fleet_with_stacked_embedder(), session),
                                 {"model": EMBED, "input": "hi"})
    assert response.status == 502
    assert EMBED in payload["error"]["message"]


def test_a_node_with_no_client_session_says_so_rather_than_crashing():
    app = _app(_fleet_with_stacked_embedder())
    app["client_session"] = None
    response = asyncio.run(forward_to_fleet(_Request(app, {"model": EMBED}),
                                            EMBED, b"{}", [("h", 8001)]))
    assert response.status == 503


# --------------------------------------------------------- local fallback ---

def test_a_model_no_node_serves_is_answered_in_process():
    """Unchanged behaviour, which is the point: the CPU path is the fallback, not the
    thing that was removed."""
    session = _Session()
    response, payload, request = _post(_app(_fleet_with_stacked_embedder(), session),
                                       {"model": MINILM, "input": ["a", "b"]})
    assert response.status == 200
    assert session.tried == []                      # nothing left this process
    assert payload["model"] == MINILM
    assert len(payload["data"]) == 2
    assert len(payload["data"][0]["embedding"]) == 8  # the fake's width
    assert request.tags["_log_model"] == MINILM


def test_the_body_is_still_validated_on_the_in_process_path():
    app = _app(_fleet_with_stacked_embedder())
    assert _post(app, {"input": "hi"})[0].status == 400            # no model
    assert _post(app, {"model": MINILM})[0].status == 400          # no input
    assert _post(app, {"model": MINILM, "input": [1, 2]})[0].status == 400

    request = _Request(app, b"not json at all")
    assert asyncio.run(handle_v1_embeddings(request)).status == 400


# ------------------------------------- the whole way through the real app ---
#
# The tests above drive the handler directly, the way `tests/test_routing_caps.py`
# drives the chat proxy. This one goes through the registered route on a real
# `create_app`, because the routing rule is only true if the route table reaches it:
# `/v1/embeddings` is registered by the embeddings module and NOT by the proxy's route
# table, so a regression there would pass every test above.

@pytest.mark.asyncio
async def test_the_registered_route_forwards_to_the_fleet():
    from aiohttp.test_utils import TestClient, TestServer

    from ainode.api.server import create_app

    config = NodeConfig(node_id="spark1", node_name="Spark-1-DGX",
                        api_port=8000, web_port=3000, model=CHAT)
    app = create_app(config=config, engine=None)
    session = _Session({"10.100.0.12": (200, _OK)})
    async with TestClient(TestServer(app)) as client:
        # Swapped after startup, and the real ClientSession the app opened is closed
        # here rather than left dangling for the loop to complain about.
        real = app["client_session"]
        app["client_session"] = session
        app["cluster_state"] = _fleet_with_stacked_embedder()
        await real.close()

        served = await (await client.get("/v1/models")).json()
        assert EMBED in [m["id"] for m in served["data"]], \
            "the master's federated /v1/models must list the embedding instance"

        resp = await client.post("/v1/embeddings",
                                 json={"model": EMBED, "input": ["hello"]})
        assert resp.status == 200
        assert await resp.json() == _OK
        assert session.tried == ["http://10.100.0.12:8001/v1/embeddings"]


# ------------------------------------- what the fleet view calls the model ---
#
# Nothing about HOW an embedding model is launched distinguishes it from a chat
# engine: same image, same container shape, and vLLM's own /v1/models says nothing
# about which runner is behind an id. The curated catalog entry's `capabilities` is
# the only place that knowledge lives, so `/api/server/status` reads it, and the
# browser reads the `type` it produces to keep the model out of the chat picker
# (`app.js::refreshChatFleet` filters `type !== 'embed'`) and off the card's chat
# controls.

def test_serving_kind_reads_the_capability_off_the_catalog():
    from ainode.api.server_routes import serving_kind

    assert serving_kind(EMBED) == ("embed", ["embeddings"])
    assert serving_kind("qwen3-embedding-0.6b") == ("embed", ["embeddings"])
    assert serving_kind(CHAT) == ("llm", ["chat", "completions"])
    # A model the catalog does not describe is a chat engine, which is what every
    # instance was before embeddings could be served this way.
    assert serving_kind("someone/unknown-model") == ("llm", ["chat", "completions"])


def test_the_returned_capability_list_cannot_be_mutated_into_the_next_caller():
    from ainode.api.server_routes import serving_kind

    _kind, caps = serving_kind(EMBED)
    caps.append("chat")
    assert serving_kind(EMBED)[1] == ["embeddings"]


@pytest.mark.asyncio
async def test_the_fleet_view_marks_a_served_embedding_instance_as_embed():
    """A peer's stacked embedding instance is reported as an embedding, so the chat
    picker drops it instead of offering a model that 400s every message."""
    from aiohttp.test_utils import TestClient, TestServer

    from ainode.api.server import create_app

    config = NodeConfig(node_id="spark1", node_name="Spark-1-DGX",
                        api_port=8000, web_port=3000, model=CHAT)
    app = create_app(config=config, engine=None)
    async with TestClient(TestServer(app)) as client:
        real = app["client_session"]
        app["client_session"] = _Session()
        app["cluster_state"] = _fleet_with_stacked_embedder()
        await real.close()

        rows = (await (await client.get("/api/server/status")).json())["loaded_models"]
        by_id = {row["id"]: row for row in rows}

        assert by_id[EMBED]["type"] == "embed"
        assert by_id[EMBED]["capabilities"] == ["embeddings"]
        assert by_id[EMBED]["port"] == 8001
        # The chat model on the same node is untouched.
        assert by_id[CHAT]["type"] == "llm"
        assert by_id[CHAT]["capabilities"] == ["chat", "completions"]
        # What the browser's picker does with them.
        offered = [r["id"] for r in rows if r.get("id") and r.get("type") != "embed"]
        assert EMBED not in offered and CHAT in offered
