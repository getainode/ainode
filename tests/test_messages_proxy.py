"""The fleet endpoint forwards the Anthropic Messages API.

vLLM answers ``POST /v1/messages`` natively, but AINode's port 3000 used to 404 it:
the proxy registered the OpenAI paths only, so a client that speaks Messages and
nothing else (Claude Code) had to be pointed at one engine's port and lost fleet
routing, failover and the model-id lookup. These tests run against a REAL AINode
app whose cluster state holds a fake engine on a real port, so the whole path is
exercised: routing on the body's ``model``, header passthrough, the query string
Claude Code sends, SSE streaming, ``count_tokens``, and a 404 that stays a 404.
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

MODEL = "unsloth/Qwen3.8-27B-NVFP4"
ANSWER = "an isogram has no repeated letter"


class FakeEngine:
    """A vLLM that serves the Messages API, recording exactly what it was asked."""

    def __init__(self):
        self.seen: list = []

    def app(self):
        app = web.Application()
        app.router.add_post("/v1/messages", self.messages)
        app.router.add_post("/v1/messages/count_tokens", self.count_tokens)
        return app

    def _record(self, request, body):
        self.seen.append({"path": request.path, "query": dict(request.query),
                          "headers": dict(request.headers), "body": body})

    async def messages(self, request):
        body = await request.json()
        self._record(request, body)
        if not body.get("stream"):
            return web.json_response({
                "id": "msg_01", "type": "message", "role": "assistant",
                "model": body["model"],
                "content": [{"type": "thinking", "thinking": "letters..."},
                            {"type": "text", "text": ANSWER}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 11, "output_tokens": 9},
            })
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for event, data in (
            ("message_start", {"type": "message_start", "message": {"id": "msg_01"}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": ANSWER}}),
            ("message_stop", {"type": "message_stop"}),
        ):
            await asyncio.sleep(0.002)
            await resp.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
        await resp.write_eof()
        return resp

    async def count_tokens(self, request):
        body = await request.json()
        self._record(request, body)
        return web.json_response({"input_tokens": 11})


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
    """A real AINode app that believes a peer serves MODEL on the fake's port."""
    config = NodeConfig(node_id="local-node", node_name="LocalNode", model=None,
                        api_port=_free_port(), web_port=_free_port(),
                        cluster_enabled=False)
    app = create_app(config=config, engine=None)
    app["cluster_state"].add_node(ClusterNode(
        node_id="spark-1", node_name="Spark-1-DGX", gpu_name="NVIDIA GB10",
        gpu_memory_gb=121.7, unified_memory=True, model=MODEL,
        status=NodeStatus.ONLINE, api_port=engine.port, web_port=engine.port,
        last_seen=0.0, fabric_ip="127.0.0.1"))
    async with TestClient(TestServer(app)) as c:
        yield c


def _body(**over):
    body = {"model": MODEL, "max_tokens": 128,
            "messages": [{"role": "user", "content": "what is an isogram"}]}
    body.update(over)
    return body


# ------------------------------------------------------------------- routing

@pytest.mark.asyncio
async def test_messages_is_routed_by_the_model_in_the_body(client, engine_fake):
    resp = await client.post("/v1/messages", json=_body(),
                             headers={"x-api-key": "ainode",
                                      "anthropic-version": "2023-06-01"})
    assert resp.status == 200
    data = await resp.json()
    assert data["type"] == "message"
    assert data["content"][-1]["text"] == ANSWER
    # It reached the node that advertises this model, at the path it was sent to.
    assert [s["path"] for s in engine_fake.seen] == ["/v1/messages"]
    assert engine_fake.seen[0]["body"]["model"] == MODEL


@pytest.mark.asyncio
async def test_the_anthropic_headers_reach_the_engine_untouched(client, engine_fake):
    await client.post("/v1/messages", json=_body(),
                      headers={"x-api-key": "secret-placeholder",
                               "anthropic-version": "2023-06-01",
                               "anthropic-beta": "fine-grained-tool-streaming-2025-05-14"})
    seen = {k.lower(): v for k, v in engine_fake.seen[0]["headers"].items()}
    assert seen["x-api-key"] == "secret-placeholder"
    assert seen["anthropic-version"] == "2023-06-01"
    assert seen["anthropic-beta"] == "fine-grained-tool-streaming-2025-05-14"


@pytest.mark.asyncio
async def test_the_query_string_claude_code_sends_survives(client, engine_fake):
    """Claude Code posts to /v1/messages?beta=true. A proxy that drops a caller's
    query string is answering a question it was not asked."""
    resp = await client.post("/v1/messages?beta=true", json=_body())
    assert resp.status == 200
    assert engine_fake.seen[0]["query"] == {"beta": "true"}


@pytest.mark.asyncio
async def test_a_model_nobody_serves_is_a_404_not_a_bad_gateway(client, engine_fake):
    resp = await client.post("/v1/messages", json=_body(model="who/Knows-3B"))
    assert resp.status == 404
    data = await resp.json()
    assert data["error"]["code"] == "model_not_found"
    assert engine_fake.seen == []


# ----------------------------------------------------------------- streaming

@pytest.mark.asyncio
async def test_a_streamed_answer_passes_through_as_sse(client, engine_fake):
    resp = await client.post("/v1/messages", json=_body(stream=True))
    assert resp.status == 200
    assert "text/event-stream" in resp.headers["Content-Type"]
    text = await resp.text()
    assert text.index("message_start") < text.index("content_block_delta") \
        < text.index("message_stop")
    assert ANSWER in text
    assert engine_fake.seen[0]["body"]["stream"] is True


# --------------------------------------------------------------- count_tokens

@pytest.mark.asyncio
async def test_count_tokens_is_forwarded_the_same_way(client, engine_fake):
    resp = await client.post("/v1/messages/count_tokens", json=_body())
    assert resp.status == 200
    assert await resp.json() == {"input_tokens": 11}
    assert engine_fake.seen[0]["path"] == "/v1/messages/count_tokens"


# ------------------------------------------------------------- what stays 404

@pytest.mark.asyncio
async def test_an_unregistered_v1_path_is_still_a_404(client, engine_fake):
    """Only the paths the proxy registers are forwarded: this is a route table,
    not a catch-all that hands anything under /v1 to an engine."""
    for path in ("/v1/messages/batches", "/v1/messages/count", "/v1/anthropic"):
        resp = await client.post(path, json=_body())
        assert resp.status == 404, path
    assert engine_fake.seen == []
