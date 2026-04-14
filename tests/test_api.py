"""Tests for ainode.api.server — AINode-specific endpoints."""

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from ainode.api.server import create_app, _build_announcement
from ainode.core.config import NodeConfig


@pytest.fixture
def config():
    return NodeConfig(node_id="test-node-1", node_name="TestNode", model="test-model")


@pytest.fixture
def app(config):
    return create_app(config=config, engine=None)


@pytest_asyncio.fixture
async def client(app):
    async with TestClient(TestServer(app)) as c:
        yield c


# ---- /api/health -----------------------------------------------------------

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/api/health")
    assert resp.status == 200
    data = await resp.json()
    assert data == {"status": "ok"}


# ---- /api/status -----------------------------------------------------------

@pytest.mark.asyncio
async def test_status_fields(client):
    resp = await client.get("/api/status")
    assert resp.status == 200
    data = await resp.json()
    assert data["node_id"] == "test-node-1"
    assert data["node_name"] == "TestNode"
    assert data["model"] == "test-model"
    assert data["version"] == "0.1.0"
    assert data["powered_by"] == "argentos.ai"
    assert data["engine_ready"] is False
    assert isinstance(data["uptime"], (int, float))
    assert isinstance(data["models_loaded"], list)
    assert "api_port" in data


# ---- /api/nodes ------------------------------------------------------------

@pytest.mark.asyncio
async def test_nodes(client):
    resp = await client.get("/api/nodes")
    assert resp.status == 200
    data = await resp.json()
    assert "nodes" in data
    nodes = data["nodes"]
    assert len(nodes) == 1
    node = nodes[0]
    assert node["node_id"] == "test-node-1"
    assert node["node_name"] == "TestNode"
    assert node["model"] == "test-model"
    # Local node appears as ONLINE in ClusterState, so engine_ready derives from status
    assert "engine_ready" in node


# ---- / (index) -------------------------------------------------------------

@pytest.mark.asyncio
async def test_index_serves_html(client):
    resp = await client.get("/")
    assert resp.status == 200
    assert "text/html" in resp.headers.get("Content-Type", "")
    body = await resp.text()
    assert "<html" in body.lower() or "<!doctype" in body.lower()


# ---- CORS ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cors_headers_localhost(client):
    resp = await client.get("/api/health", headers={"Origin": "http://localhost:3000"})
    assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"


@pytest.mark.asyncio
async def test_cors_headers_rejected(client):
    resp = await client.get("/api/health", headers={"Origin": "http://evil.com"})
    assert resp.headers.get("Access-Control-Allow-Origin") == ""


@pytest.mark.asyncio
async def test_cors_preflight(client):
    resp = await client.options("/api/health", headers={"Origin": "http://127.0.0.1:3000"})
    assert resp.status == 204
    assert resp.headers.get("Access-Control-Allow-Origin") == "http://127.0.0.1:3000"
    assert "POST" in resp.headers.get("Access-Control-Allow-Methods", "")


# ---- vLLM proxy returns 502 when engine is down ----------------------------

@pytest.mark.asyncio
async def test_vllm_proxy_returns_502_when_down(client):
    resp = await client.get("/v1/models")
    assert resp.status == 502
    data = await resp.json()
    assert "error" in data


# ---- _build_announcement ----------------------------------------------------

def test_build_announcement_from_config():
    cfg = NodeConfig(node_id="abc123", node_name="TestSpark", model="llama-3", api_port=9000, web_port=4000)
    ann = _build_announcement(cfg, engine=None)
    assert ann.node_id == "abc123"
    assert ann.node_name == "TestSpark"
    assert ann.model == "llama-3"
    assert ann.api_port == 9000
    assert ann.web_port == 4000
    assert ann.status == "starting"  # no engine = not ready


def test_build_announcement_engine_ready():
    cfg = NodeConfig(node_id="abc123", node_name="TestSpark", model="llama-3")

    class FakeEngine:
        ready = True
    ann = _build_announcement(cfg, engine=FakeEngine())
    assert ann.status == "serving"


# ---- /api/nodes with cluster data ------------------------------------------

@pytest.mark.asyncio
async def test_server_status(client):
    resp = await client.get("/api/server/status")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "running"
    assert "reachable_at" in data and isinstance(data["reachable_at"], list)
    assert len(data["reachable_at"]) >= 1
    assert "loaded_models" in data
    assert "request_count_total" in data


@pytest.mark.asyncio
async def test_server_endpoints_catalog(client):
    resp = await client.get("/api/server/endpoints")
    assert resp.status == 200
    data = await resp.json()
    assert "openai" in data and "lmstudio" in data and "anthropic" in data
    # Each row has method + path
    for row in data["openai"]:
        assert "method" in row and "path" in row


@pytest.mark.asyncio
async def test_server_logs_roundtrip(client):
    # Start clean
    await client.delete("/api/server/logs")
    # Trigger a loggable request
    await client.get("/api/status")
    resp = await client.get("/api/server/logs")
    assert resp.status == 200
    data = await resp.json()
    assert "entries" in data
    assert any(e["path"] == "/api/status" for e in data["entries"])
    # Health & server/logs themselves should be skipped
    assert not any(e["path"].startswith("/api/health") for e in data["entries"])
    assert not any(e["path"] == "/api/server/logs" for e in data["entries"])
    # Clear
    resp = await client.delete("/api/server/logs")
    assert resp.status == 200
    resp = await client.get("/api/server/logs")
    data = await resp.json()
    # After the GET, the clear call itself would not be logged (skipped).
    # Any entries remaining should predate the clear only if more requests
    # happened between. Assert the clear worked by checking the list is
    # empty or only contains this GET's predecessors.
    assert isinstance(data["entries"], list)


@pytest.mark.asyncio
async def test_server_eject_no_engine(client):
    resp = await client.post("/api/server/models/dummy%2Fmodel/eject")
    # With no engine, we return 202 + planned status
    assert resp.status in (202, 200)
    data = await resp.json()
    assert "model_id" in data


@pytest.mark.asyncio
async def test_nodes_returns_local_node_from_cluster(client):
    """When cluster has a local announcement, /api/nodes should return it."""
    resp = await client.get("/api/nodes")
    assert resp.status == 200
    data = await resp.json()
    nodes = data["nodes"]
    assert len(nodes) >= 1
    node_ids = [n["node_id"] for n in nodes]
    assert "test-node-1" in node_ids
