"""Tests for onboarding API routes."""

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from ainode.api.server import create_app
from ainode.core.config import NodeConfig


@pytest.fixture
def fresh_config(tmp_path):
    """A config that has NOT completed onboarding, using a temp dir for saves."""
    config = NodeConfig(
        node_id="test-node-1",
        node_name="TestNode",
        model="test-model",
        onboarded=False,
    )
    # Override save to write to tmp instead of ~/.ainode
    config_file = tmp_path / "config.json"
    original_save = config.save

    def save_to_tmp():
        import json
        from dataclasses import asdict
        config_file.write_text(json.dumps(asdict(config), indent=2))

    config.save = save_to_tmp
    return config


@pytest.fixture
def onboarded_config():
    """A config that HAS completed onboarding."""
    return NodeConfig(
        node_id="test-node-2",
        node_name="MyNode",
        model="meta-llama/Llama-3.1-8B-Instruct",
        onboarded=True,
    )


@pytest.fixture
def app_fresh(fresh_config):
    return create_app(config=fresh_config, engine=None)


@pytest.fixture
def app_onboarded(onboarded_config):
    return create_app(config=onboarded_config, engine=None)


@pytest_asyncio.fixture
async def client_fresh(app_fresh):
    async with TestClient(TestServer(app_fresh)) as c:
        yield c


@pytest_asyncio.fixture
async def client_onboarded(app_onboarded):
    async with TestClient(TestServer(app_onboarded)) as c:
        yield c


# ---- GET /api/onboarding/status --------------------------------------------

@pytest.mark.asyncio
async def test_status_needs_setup(client_fresh):
    resp = await client_fresh.get("/api/onboarding/status")
    assert resp.status == 200
    data = await resp.json()
    assert data["onboarded"] is False
    assert data["needs_setup"] is True


@pytest.mark.asyncio
async def test_status_already_onboarded(client_onboarded):
    resp = await client_onboarded.get("/api/onboarding/status")
    assert resp.status == 200
    data = await resp.json()
    assert data["onboarded"] is True
    assert data["needs_setup"] is False


# ---- GET /api/onboarding/suggestions ---------------------------------------

@pytest.mark.asyncio
async def test_suggestions_returns_hostname_and_models(client_fresh):
    resp = await client_fresh.get("/api/onboarding/suggestions")
    assert resp.status == 200
    data = await resp.json()
    assert "hostname" in data
    assert isinstance(data["hostname"], str)
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0
    # Each model should have required fields
    m = data["models"][0]
    assert "id" in m
    assert "name" in m
    assert "hf_repo" in m
    assert "size_gb" in m
    assert "fits_gpu" in m


# ---- POST /api/onboarding/complete -----------------------------------------

@pytest.mark.asyncio
async def test_complete_saves_config(client_fresh, fresh_config):
    payload = {
        "node_name": "my-spark",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "email": "test@example.com",
    }
    resp = await client_fresh.post("/api/onboarding/complete", json=payload)
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "ok"
    assert data["onboarded"] is True
    assert data["node_name"] == "my-spark"
    assert data["model"] == "meta-llama/Llama-3.1-8B-Instruct"

    # Config object should be updated
    assert fresh_config.onboarded is True
    assert fresh_config.node_name == "my-spark"
    assert fresh_config.model == "meta-llama/Llama-3.1-8B-Instruct"
    assert fresh_config.email == "test@example.com"


@pytest.mark.asyncio
async def test_complete_sets_awq_quantization(client_fresh, fresh_config):
    payload = {
        "node_name": "awq-node",
        "model": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    }
    resp = await client_fresh.post("/api/onboarding/complete", json=payload)
    assert resp.status == 200
    assert fresh_config.quantization == "awq"


@pytest.mark.asyncio
async def test_complete_without_email(client_fresh, fresh_config):
    payload = {
        "node_name": "no-email-node",
        "model": "meta-llama/Llama-3.2-3B-Instruct",
    }
    resp = await client_fresh.post("/api/onboarding/complete", json=payload)
    assert resp.status == 200
    assert fresh_config.onboarded is True
    assert fresh_config.email is None


@pytest.mark.asyncio
async def test_complete_rejects_invalid_json(client_fresh):
    resp = await client_fresh.post(
        "/api/onboarding/complete",
        data=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status == 400
    data = await resp.json()
    assert "error" in data


# ---- Index redirect logic ---------------------------------------------------

@pytest.mark.asyncio
async def test_index_redirects_to_onboarding_when_not_setup(client_fresh):
    resp = await client_fresh.get("/", allow_redirects=False)
    assert resp.status == 302
    assert "/onboarding" in resp.headers.get("Location", "")


@pytest.mark.asyncio
async def test_index_serves_dashboard_when_onboarded(client_onboarded):
    resp = await client_onboarded.get("/")
    assert resp.status == 200
    text = await resp.text()
    assert "AINode" in text


@pytest.mark.asyncio
async def test_onboarding_page_serves_wizard_when_not_setup(client_fresh):
    resp = await client_fresh.get("/onboarding")
    assert resp.status == 200
    text = await resp.text()
    assert "Let's get you set up" in text or "onboarding" in text.lower()


@pytest.mark.asyncio
async def test_onboarding_page_redirects_when_already_done(client_onboarded):
    resp = await client_onboarded.get("/onboarding", allow_redirects=False)
    assert resp.status == 302
    assert resp.headers.get("Location", "").endswith("/")
