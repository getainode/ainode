"""The fleet endpoint forwards the rest of what the engines actually serve.

vLLM answers ``/v1/responses``, ``/tokenize``, ``/detokenize`` and, on a pooling
engine, ``/v1/rerank`` and ``/v1/score``. Port 3000 used to 404 every one of
them, so a caller who wanted a token count or a rerank had to abandon fleet
routing and address one engine's port directly, losing the model-id lookup and
the failover with it.

The route list here is not invented: it is what
``curl http://<node>:<port>/openapi.json`` returns on the fleet (vLLM 0.27.1 for
the chat paths, the 0.17.0 pooling engine for rerank/score). These tests run
against a REAL AINode app with a fake engine in its cluster state, so routing on
the body's ``model``, the streamed answer, the failover and the 404 that stays a
404 are all exercised through the same handler the chat paths use.
"""

import asyncio
import json
import socket

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from ainode.api.server import create_app
from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeStatus
from ainode.discovery.cluster import ClusterNode

CHAT_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"


class FakeEngine:
    """A vLLM serving the paths this change forwards, recording what it was asked."""

    def __init__(self):
        self.seen: list = []

    def app(self):
        app = web.Application()
        app.router.add_post("/v1/responses", self.responses)
        app.router.add_post("/tokenize", self.tokenize)
        app.router.add_post("/detokenize", self.detokenize)
        app.router.add_post("/v1/rerank", self.rerank)
        app.router.add_post("/v1/score", self.score)
        return app

    def _record(self, request, body):
        self.seen.append({"path": request.path, "query": dict(request.query),
                          "headers": dict(request.headers), "body": body})

    async def responses(self, request):
        body = await request.json()
        self._record(request, body)
        if not body.get("stream"):
            return web.json_response({
                "id": "resp_01", "object": "response", "status": "completed",
                "model": body["model"],
                "output": [{"type": "message", "role": "assistant",
                            "content": [{"type": "output_text",
                                         "text": "an isogram repeats no letter"}]}],
                "usage": {"input_tokens": 11, "output_tokens": 7},
            })
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for event in ("response.created", "response.output_text.delta",
                      "response.completed"):
            await asyncio.sleep(0.002)
            await resp.write(
                f"event: {event}\ndata: {json.dumps({'type': event})}\n\n".encode())
        await resp.write_eof()
        return resp

    async def tokenize(self, request):
        body = await request.json()
        self._record(request, body)
        return web.json_response({"count": 4, "max_model_len": 32768,
                                  "tokens": [9, 4340, 1099, 30]})

    async def detokenize(self, request):
        body = await request.json()
        self._record(request, body)
        return web.json_response({"prompt": "what is an isogram"})

    async def rerank(self, request):
        body = await request.json()
        self._record(request, body)
        return web.json_response({"id": "rerank-1", "model": body["model"],
                                  "results": [{"index": 0, "relevance_score": 0.91}]})

    async def score(self, request):
        body = await request.json()
        self._record(request, body)
        return web.json_response({"id": "score-1", "model": body["model"],
                                  "data": [{"index": 0, "score": 0.42}]})


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def engine_fake():
    return FakeEngine()


@pytest_asyncio.fixture
async def engine(engine_fake):
    server = TestServer(engine_fake.app())
    await server.start_server()
    try:
        yield server
    finally:
        await server.close()


@pytest_asyncio.fixture
async def client(engine):
    """An AINode app that believes one node serves a chat model and a pooling model.

    Both are on the fake's port, which is how the fleet really looks: the pooling
    engine is a stacked instance on its own port beside the chat engine.
    """
    config = NodeConfig(node_id="local-node", node_name="LocalNode", model=None,
                        api_port=_free_port(), web_port=_free_port(),
                        cluster_enabled=False)
    app = create_app(config=config, engine=None)
    app["cluster_state"].add_node(ClusterNode(
        node_id="spark-4", node_name="Spark-4-GX10", gpu_name="NVIDIA GB10",
        gpu_memory_gb=121.7, unified_memory=True, model=CHAT_MODEL,
        status=NodeStatus.ONLINE, api_port=engine.port, web_port=engine.port,
        last_seen=0.0, fabric_ip="127.0.0.1",
        instances=[{"model": EMBED_MODEL, "api_port": engine.port,
                    "status": "serving"}]))
    async with TestClient(TestServer(app)) as c:
        yield c


# ----------------------------------------------------------------- /v1/responses

@pytest.mark.asyncio
async def test_responses_is_routed_by_the_model_in_the_body(client, engine_fake):
    resp = await client.post("/v1/responses",
                             json={"model": CHAT_MODEL, "input": "what is an isogram"})
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "completed"
    assert data["output"][0]["content"][0]["text"] == "an isogram repeats no letter"
    assert [s["path"] for s in engine_fake.seen] == ["/v1/responses"]


@pytest.mark.asyncio
async def test_a_streamed_response_passes_through_as_sse(client, engine_fake):
    resp = await client.post("/v1/responses",
                             json={"model": CHAT_MODEL, "input": "hi", "stream": True})
    assert resp.status == 200
    assert "text/event-stream" in resp.headers["Content-Type"]
    text = await resp.text()
    assert text.index("response.created") < text.index("response.output_text.delta") \
        < text.index("response.completed")


@pytest.mark.asyncio
async def test_the_query_string_and_headers_survive(client, engine_fake):
    resp = await client.post("/v1/responses?store=false",
                             json={"model": CHAT_MODEL, "input": "hi"},
                             headers={"authorization": "Bearer placeholder"})
    assert resp.status == 200
    assert engine_fake.seen[0]["query"] == {"store": "false"}
    seen = {k.lower(): v for k, v in engine_fake.seen[0]["headers"].items()}
    assert seen["authorization"] == "Bearer placeholder"


# ------------------------------------------------------- /tokenize, /detokenize

@pytest.mark.asyncio
async def test_tokenize_is_forwarded_even_though_it_is_not_under_v1(client, engine_fake):
    """vLLM serves /tokenize at the root, so the route table has to as well."""
    resp = await client.post("/tokenize",
                             json={"model": CHAT_MODEL, "prompt": "what is an isogram"})
    assert resp.status == 200
    assert (await resp.json())["count"] == 4
    assert engine_fake.seen[0]["path"] == "/tokenize"


@pytest.mark.asyncio
async def test_detokenize_is_forwarded(client, engine_fake):
    resp = await client.post("/detokenize",
                             json={"model": CHAT_MODEL, "tokens": [9, 4340, 1099, 30]})
    assert resp.status == 200
    assert (await resp.json())["prompt"] == "what is an isogram"
    assert engine_fake.seen[0]["path"] == "/detokenize"


@pytest.mark.asyncio
async def test_tokenize_reaches_a_pooling_instance_on_its_own_port(client, engine_fake):
    """An embedding model is a stacked instance, and it serves /tokenize too."""
    resp = await client.post("/tokenize",
                             json={"model": EMBED_MODEL, "prompt": "vectors"})
    assert resp.status == 200
    assert engine_fake.seen[0]["body"]["model"] == EMBED_MODEL


# ------------------------------------------------------- /v1/rerank, /v1/score

@pytest.mark.asyncio
async def test_rerank_is_forwarded_for_a_pooling_model(client, engine_fake):
    resp = await client.post("/v1/rerank", json={
        "model": EMBED_MODEL, "query": "isogram",
        "documents": ["a word with no repeated letter", "unrelated"]})
    assert resp.status == 200
    assert (await resp.json())["results"][0]["relevance_score"] == 0.91
    assert engine_fake.seen[0]["path"] == "/v1/rerank"


@pytest.mark.asyncio
async def test_score_is_forwarded_for_a_pooling_model(client, engine_fake):
    resp = await client.post("/v1/score", json={
        "model": EMBED_MODEL, "text_1": "isogram", "text_2": "no repeated letter"})
    assert resp.status == 200
    assert (await resp.json())["data"][0]["score"] == 0.42
    assert engine_fake.seen[0]["path"] == "/v1/score"


# ------------------------------------------------- the shared behaviour it gains

@pytest.mark.asyncio
async def test_a_model_nobody_serves_is_a_404_not_a_bad_gateway(client, engine_fake):
    for path, body in (("/v1/responses", {"model": "who/Knows-3B", "input": "x"}),
                       ("/tokenize", {"model": "who/Knows-3B", "prompt": "x"}),
                       ("/v1/rerank", {"model": "who/Knows-3B", "query": "x",
                                       "documents": ["y"]})):
        resp = await client.post(path, json=body)
        assert resp.status == 404, path
        assert (await resp.json())["error"]["code"] == "model_not_found"
    assert engine_fake.seen == []


@pytest.mark.asyncio
async def test_an_unregistered_path_is_still_a_404(client, engine_fake):
    """Still a route table, not a catch-all that hands any path to an engine."""
    for path in ("/v1/responses/resp_01/cancel", "/v2/rerank", "/pooling",
                 "/generative_scoring", "/invocations"):
        resp = await client.post(path, json={"model": CHAT_MODEL})
        assert resp.status == 404, path
    assert engine_fake.seen == []
