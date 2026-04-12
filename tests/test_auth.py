"""Tests for ainode.auth — middleware, key management, enable/disable."""

import json
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from ainode.auth.middleware import AuthConfig, auth_middleware, AUTH_FILE
from ainode.auth.api_routes import register_auth_routes
from ainode.core.config import NodeConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def tmp_auth_file(tmp_path, monkeypatch):
    """Redirect AUTH_FILE to a temp directory."""
    auth_file = tmp_path / "auth.json"
    monkeypatch.setattr("ainode.auth.middleware.AUTH_FILE", auth_file)
    monkeypatch.setattr("ainode.auth.middleware.AINODE_HOME", tmp_path)
    return auth_file


@pytest.fixture
def auth_config(tmp_auth_file):
    """Fresh AuthConfig with temp storage."""
    return AuthConfig()


def _make_app(auth_config):
    """Create a minimal aiohttp app with auth middleware and test routes."""

    @web.middleware
    async def cors_middleware(request, handler):
        return await handler(request)

    app = web.Application(middlewares=[cors_middleware, auth_middleware])
    app["config"] = NodeConfig(node_id="test", node_name="Test", onboarded=True)
    app["auth_config"] = auth_config

    # Test routes
    app.router.add_get("/", _handle_index)
    app.router.add_get("/api/health", _handle_health)
    app.router.add_get("/api/status", _handle_status)
    app.router.add_get("/api/onboarding/config", _handle_onboarding)
    app.router.add_get("/v1/models", _handle_protected)
    app.router.add_post("/v1/chat/completions", _handle_protected)
    app.router.add_static("/static", Path(__file__).parent, name="static")

    # Auth management routes
    register_auth_routes(app)

    return app


async def _handle_index(_req):
    return web.Response(text="<html>dashboard</html>", content_type="text/html")

async def _handle_health(_req):
    return web.json_response({"status": "ok"})

async def _handle_status(_req):
    return web.json_response({"node_id": "test"})

async def _handle_onboarding(_req):
    return web.json_response({"step": 1})

async def _handle_protected(_req):
    return web.json_response({"data": "secret"})


# =============================================================================
# AuthConfig unit tests
# =============================================================================

class TestAuthConfig:
    def test_default_disabled(self, auth_config):
        assert auth_config.enabled is False
        assert auth_config.api_keys == []

    def test_generate_key(self, auth_config):
        entry = auth_config.generate_key()
        assert len(entry["key"]) == 32
        assert len(entry["id"]) == 8
        assert len(auth_config.api_keys) == 1
        assert "key_hash" in auth_config.api_keys[0]

    def test_enable_generates_key(self, auth_config):
        entry = auth_config.enable()
        assert auth_config.enabled is True
        assert len(auth_config.api_keys) == 1
        assert auth_config.validate_token(entry["key"])

    def test_enable_reuses_existing_key(self, auth_config):
        first = auth_config.generate_key()
        entry = auth_config.enable()
        assert entry["id"] == auth_config.api_keys[0]["id"]
        assert len(auth_config.api_keys) == 1

    def test_disable(self, auth_config):
        auth_config.enable()
        auth_config.disable()
        assert auth_config.enabled is False
        # Keys are kept
        assert len(auth_config.api_keys) == 1

    def test_revoke_key(self, auth_config):
        entry = auth_config.generate_key()
        assert auth_config.revoke_key(entry["id"]) is True
        assert len(auth_config.api_keys) == 0

    def test_revoke_nonexistent(self, auth_config):
        assert auth_config.revoke_key("nope") is False

    def test_save_and_load(self, auth_config, tmp_auth_file):
        auth_config.enable()
        key_hash = auth_config.api_keys[0]["key_hash"]

        # Load from disk
        loaded = AuthConfig.load()
        assert loaded.enabled is True
        assert loaded.api_keys[0]["key_hash"] == key_hash

    def test_load_missing_file(self, tmp_auth_file):
        cfg = AuthConfig.load()
        assert cfg.enabled is False
        assert cfg.api_keys == []


# =============================================================================
# Middleware tests — auth disabled (default)
# =============================================================================

class TestMiddlewareDisabled:
    @pytest_asyncio.fixture
    async def client(self, auth_config):
        app = _make_app(auth_config)
        async with TestClient(TestServer(app)) as c:
            yield c

    @pytest.mark.asyncio
    async def test_passthrough_no_auth(self, client):
        """When auth is disabled, all requests pass through."""
        resp = await client.get("/v1/models")
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_health_no_auth(self, client):
        resp = await client.get("/api/health")
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_index_no_auth(self, client):
        resp = await client.get("/")
        assert resp.status == 200


# =============================================================================
# Middleware tests — auth enabled
# =============================================================================

class TestMiddlewareEnabled:
    @pytest_asyncio.fixture
    async def client_and_key(self, auth_config):
        entry = auth_config.enable()
        app = _make_app(auth_config)
        async with TestClient(TestServer(app)) as c:
            yield c, entry["key"]

    @pytest.mark.asyncio
    async def test_protected_without_key_returns_401(self, client_and_key):
        client, _ = client_and_key
        resp = await client.get("/v1/models")
        assert resp.status == 401
        data = await resp.json()
        assert "auth_error" in str(data)

    @pytest.mark.asyncio
    async def test_protected_with_invalid_key_returns_401(self, client_and_key):
        client, _ = client_and_key
        resp = await client.get("/v1/models", headers={"Authorization": "Bearer badkey"})
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_protected_with_valid_key(self, client_and_key):
        client, key = client_and_key
        resp = await client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})
        assert resp.status == 200
        data = await resp.json()
        assert data["data"] == "secret"

    @pytest.mark.asyncio
    async def test_post_with_valid_key(self, client_and_key):
        client, key = client_and_key
        resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "test"},
        )
        assert resp.status == 200

    # -- Skip paths -----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_index_skips_auth(self, client_and_key):
        client, _ = client_and_key
        resp = await client.get("/")
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_health_skips_auth(self, client_and_key):
        client, _ = client_and_key
        resp = await client.get("/api/health")
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_onboarding_skips_auth(self, client_and_key):
        client, _ = client_and_key
        resp = await client.get("/api/onboarding/config")
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_status_requires_auth(self, client_and_key):
        """Non-exempt paths require auth."""
        client, _ = client_and_key
        resp = await client.get("/api/status")
        assert resp.status == 401


# =============================================================================
# Auth API routes
# =============================================================================

class TestAuthRoutes:
    @pytest_asyncio.fixture
    async def client_and_key(self, auth_config):
        entry = auth_config.enable()
        app = _make_app(auth_config)
        async with TestClient(TestServer(app)) as c:
            yield c, entry["key"]

    def _auth_headers(self, key):
        return {"Authorization": f"Bearer {key}"}

    @pytest.mark.asyncio
    async def test_auth_status(self, client_and_key):
        client, key = client_and_key
        resp = await client.get("/api/auth/status", headers=self._auth_headers(key))
        assert resp.status == 200
        data = await resp.json()
        assert data["enabled"] is True
        assert data["key_count"] == 1

    @pytest.mark.asyncio
    async def test_auth_disable(self, client_and_key):
        client, key = client_and_key
        resp = await client.post("/api/auth/disable", headers=self._auth_headers(key))
        assert resp.status == 200
        data = await resp.json()
        assert data["enabled"] is False

    @pytest.mark.asyncio
    async def test_auth_enable(self, auth_config):
        # Start disabled
        app = _make_app(auth_config)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/auth/enable")
            assert resp.status == 200
            data = await resp.json()
            assert data["enabled"] is True
            assert "api_key" in data
            assert len(data["api_key"]) == 32

    @pytest.mark.asyncio
    async def test_create_key(self, client_and_key):
        client, key = client_and_key
        resp = await client.post("/api/auth/keys", headers=self._auth_headers(key))
        assert resp.status == 200
        data = await resp.json()
        assert "api_key" in data
        assert len(data["api_key"]) == 32

    @pytest.mark.asyncio
    async def test_revoke_key(self, client_and_key):
        client, key = client_and_key
        # Create a second key, then revoke it
        resp = await client.post("/api/auth/keys", headers=self._auth_headers(key))
        new_key_data = await resp.json()
        key_id = new_key_data["key_id"]

        resp = await client.delete(f"/api/auth/keys/{key_id}", headers=self._auth_headers(key))
        assert resp.status == 200
        data = await resp.json()
        assert data["revoked"] is True

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_key(self, client_and_key):
        client, key = client_and_key
        resp = await client.delete("/api/auth/keys/nope", headers=self._auth_headers(key))
        assert resp.status == 404
