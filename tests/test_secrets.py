"""Tests for ainode.secrets.SecretsManager and its HTTP routes."""

import json
import os
import stat
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from ainode.api.server import create_app
from ainode.core.config import NodeConfig
from ainode.secrets.manager import KNOWN_SECRETS, SecretsManager, _mask


@pytest.fixture
def tmp_secrets(tmp_path, monkeypatch):
    path = tmp_path / "secrets.json"
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.secrets.manager.SECRETS_FILE", path)
    return path


class TestSecretsManager:
    def test_set_and_get_known_secret(self, tmp_secrets):
        mgr = SecretsManager(tmp_secrets)
        mgr.set("huggingface_token", "hf_abcdef123456")
        assert mgr.get("huggingface_token") == "hf_abcdef123456"
        assert mgr.has("huggingface_token")

    def test_unknown_key_rejected(self, tmp_secrets):
        mgr = SecretsManager(tmp_secrets)
        with pytest.raises(KeyError):
            mgr.set("not_real", "x")

    def test_empty_value_rejected(self, tmp_secrets):
        mgr = SecretsManager(tmp_secrets)
        with pytest.raises(ValueError):
            mgr.set("huggingface_token", "")

    def test_delete(self, tmp_secrets):
        mgr = SecretsManager(tmp_secrets)
        mgr.set("openai_api_key", "sk-xxxxxxxx1234")
        assert mgr.delete("openai_api_key") is True
        assert mgr.get("openai_api_key") is None
        assert mgr.delete("openai_api_key") is False

    def test_all_masks_by_default(self, tmp_secrets):
        mgr = SecretsManager(tmp_secrets)
        mgr.set("huggingface_token", "hf_abcdef123456")
        data = mgr.all()
        assert "value" not in data["huggingface_token"]
        assert data["huggingface_token"]["is_set"] is True
        assert "3456" in data["huggingface_token"]["masked"]

    def test_all_with_values(self, tmp_secrets):
        mgr = SecretsManager(tmp_secrets)
        mgr.set("huggingface_token", "hf_abcdef123456")
        data = mgr.all(include_values=True)
        assert data["huggingface_token"]["value"] == "hf_abcdef123456"

    def test_persist_across_instances(self, tmp_secrets):
        mgr = SecretsManager(tmp_secrets)
        mgr.set("huggingface_token", "hf_abcdef123456")
        # New manager instance reading the same file
        mgr2 = SecretsManager(tmp_secrets)
        assert mgr2.get("huggingface_token") == "hf_abcdef123456"

    def test_file_is_not_plaintext(self, tmp_secrets):
        mgr = SecretsManager(tmp_secrets)
        mgr.set("huggingface_token", "hf_SECRETVALUE_NEEDLE")
        raw = tmp_secrets.read_text()
        assert "hf_SECRETVALUE_NEEDLE" not in raw

    def test_file_mode_0600(self, tmp_secrets):
        mgr = SecretsManager(tmp_secrets)
        mgr.set("huggingface_token", "hf_abcdef123456")
        mode = stat.S_IMODE(os.stat(tmp_secrets).st_mode)
        # Owner rw only; group/other must be 0
        assert mode & 0o077 == 0

    def test_custom_secrets(self, tmp_secrets):
        mgr = SecretsManager(tmp_secrets)
        mgr.custom_set("my_api", "abc123")
        assert mgr.get("my_api") == "abc123"
        assert "my_api" in mgr.custom_all()
        assert mgr.custom_delete("my_api") is True
        assert mgr.get("my_api") is None

    def test_custom_name_validation(self, tmp_secrets):
        mgr = SecretsManager(tmp_secrets)
        with pytest.raises(ValueError):
            mgr.custom_set("bad name with spaces", "x")
        with pytest.raises(ValueError):
            mgr.custom_set("huggingface_token", "x")  # reserved

    def test_mask_short_value(self):
        assert _mask("") == ""
        assert _mask("abc") == "•••"

    def test_repr_does_not_leak_values(self, tmp_secrets):
        mgr = SecretsManager(tmp_secrets)
        mgr.set("huggingface_token", "hf_SECRETVALUE_NEEDLE")
        assert "SECRETVALUE" not in repr(mgr)


# ---- HTTP routes ------------------------------------------------------------

@pytest.fixture
def config():
    return NodeConfig(node_id="node-a", node_name="A", model="x")


@pytest.fixture
def app(config, tmp_path, monkeypatch):
    # Redirect all AINode state writes into tmp_path
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr("ainode.secrets.manager.SECRETS_FILE", tmp_path / "secrets.json")
    monkeypatch.setattr("ainode.auth.middleware.AUTH_FILE", tmp_path / "auth.json")
    return create_app(config=config, engine=None)


@pytest_asyncio.fixture
async def client(app):
    async with TestClient(TestServer(app)) as c:
        yield c


@pytest.mark.asyncio
async def test_list_secrets_empty(client):
    resp = await client.get("/api/secrets")
    assert resp.status == 200
    data = await resp.json()
    assert "known" in data
    assert "custom" in data
    assert "huggingface_token" in data["known"]
    assert data["known"]["huggingface_token"]["is_set"] is False


@pytest.mark.asyncio
async def test_set_and_get_secret(client):
    resp = await client.put(
        "/api/secrets/huggingface_token",
        json={"value": "hf_abcdef123456"},
    )
    assert resp.status == 200
    resp = await client.get("/api/secrets")
    data = await resp.json()
    assert data["known"]["huggingface_token"]["is_set"] is True
    # Raw value must never appear in API responses
    raw_body = json.dumps(data)
    assert "hf_abcdef123456" not in raw_body


@pytest.mark.asyncio
async def test_unknown_secret_rejected(client):
    resp = await client.put("/api/secrets/nope", json={"value": "x"})
    assert resp.status == 400


@pytest.mark.asyncio
async def test_delete_secret(client):
    await client.put("/api/secrets/openai_api_key", json={"value": "sk-xxxxxxxx1234"})
    resp = await client.delete("/api/secrets/openai_api_key")
    assert resp.status == 200
    resp = await client.get("/api/secrets")
    data = await resp.json()
    assert data["known"]["openai_api_key"]["is_set"] is False


@pytest.mark.asyncio
async def test_custom_secret_flow(client):
    resp = await client.put("/api/secrets/custom/my_key", json={"value": "xyz123"})
    assert resp.status == 200
    resp = await client.get("/api/secrets/custom")
    data = await resp.json()
    assert "my_key" in data["custom"]
    raw_body = json.dumps(data)
    assert "xyz123" not in raw_body
    resp = await client.delete("/api/secrets/custom/my_key")
    assert resp.status == 200
