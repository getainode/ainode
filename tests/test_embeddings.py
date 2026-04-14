"""Tests for the embedding manager and embedding API routes."""

from __future__ import annotations

import sys
import types
from typing import List

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from ainode.api.server import create_app
from ainode.core.config import NodeConfig
from ainode.embeddings.manager import EmbeddingManager, KNOWN_EMBEDDING_MODELS


# ---------------------------------------------------------------------------
# Fake sentence-transformers for tests — we never want to hit HF during CI.
# ---------------------------------------------------------------------------


class _FakeSentenceTransformer:
    """Deterministic fake that mimics the subset of SentenceTransformer used."""

    # Track every instantiation so tests can assert load behavior
    instantiated: List[str] = []

    def __init__(self, model_id: str, *args, **kwargs) -> None:
        self.model_id = model_id
        self.max_seq_length = 256
        _FakeSentenceTransformer.instantiated.append(model_id)

    def get_sentence_embedding_dimension(self) -> int:
        return 8  # small fake dimension for test speed

    def encode(self, texts, convert_to_numpy=True, **kwargs):
        # Return a simple deterministic vector per text (a plain list-of-lists
        # works — the manager handles both numpy and list returns).
        return [[float(i + j) for j in range(8)] for i, _ in enumerate(texts)]


@pytest.fixture(autouse=True)
def _install_fake_sentence_transformers(monkeypatch):
    """Inject a fake ``sentence_transformers`` module for the duration of the test."""
    _FakeSentenceTransformer.instantiated = []
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    yield


# ---------------------------------------------------------------------------
# Manager tests
# ---------------------------------------------------------------------------


def test_embedding_manager_list_known():
    mgr = EmbeddingManager()
    known = mgr.list_known()
    ids = {m["id"] for m in known}

    # Required models from the spec
    assert "sentence-transformers/all-MiniLM-L6-v2" in ids
    assert "nomic-ai/nomic-embed-text-v1.5" in ids
    assert "BAAI/bge-large-en-v1.5" in ids
    assert "mixedbread-ai/mxbai-embed-large-v1" in ids

    # Required fields
    for entry in known:
        for field in ("id", "hf_repo", "dimensions", "max_seq_length", "size_mb", "description"):
            assert field in entry, f"missing {field} in {entry['id']}"

    # Catalog dict is the source of truth
    assert set(KNOWN_EMBEDDING_MODELS.keys()) == ids


def test_embedding_manager_load_unload_lifecycle():
    mgr = EmbeddingManager()
    model_id = "sentence-transformers/all-MiniLM-L6-v2"

    assert mgr.is_loaded(model_id) is False
    meta = mgr.load(model_id)
    assert meta["id"] == model_id
    assert meta["dimensions"] == 8  # from fake
    assert mgr.is_loaded(model_id) is True
    assert [m["id"] for m in mgr.list_loaded()] == [model_id]

    # Second load returns cached metadata without re-instantiating
    mgr.load(model_id)
    assert _FakeSentenceTransformer.instantiated.count(model_id) == 1

    # Force reload re-instantiates
    mgr.load(model_id, force=True)
    assert _FakeSentenceTransformer.instantiated.count(model_id) == 2

    assert mgr.unload(model_id) is True
    assert mgr.is_loaded(model_id) is False
    assert mgr.unload(model_id) is False  # no-op on a second call


def test_embedding_manager_embed_auto_loads_and_returns_shape():
    mgr = EmbeddingManager()
    model_id = "sentence-transformers/all-MiniLM-L6-v2"

    vectors = mgr.embed(model_id, ["hello", "world", "foo"])
    assert mgr.is_loaded(model_id)  # auto-loaded
    assert len(vectors) == 3
    assert all(len(v) == 8 for v in vectors)
    assert all(isinstance(x, float) for v in vectors for x in v)


def test_embedding_manager_embed_requires_list():
    mgr = EmbeddingManager()
    with pytest.raises(TypeError):
        mgr.embed("sentence-transformers/all-MiniLM-L6-v2", "not-a-list")  # type: ignore[arg-type]


def test_embedding_manager_missing_dependency_raises_runtime(monkeypatch):
    # Remove the fake so the import fails
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)

    # Block any real import too
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    mgr = EmbeddingManager()
    with pytest.raises(RuntimeError) as excinfo:
        mgr.load("sentence-transformers/all-MiniLM-L6-v2")
    assert "sentence-transformers" in str(excinfo.value)


# ---------------------------------------------------------------------------
# HTTP route tests
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return NodeConfig(node_id="test-embed", node_name="EmbedNode", model="x")


@pytest.fixture
def app(config):
    return create_app(config=config, engine=None)


@pytest_asyncio.fixture
async def client(app):
    async with TestClient(TestServer(app)) as c:
        yield c


@pytest.mark.asyncio
async def test_embedding_routes_models_list(client):
    resp = await client.get("/api/embeddings/models")
    assert resp.status == 200
    data = await resp.json()
    assert data["count"] >= 4
    ids = {m["id"] for m in data["models"]}
    assert "sentence-transformers/all-MiniLM-L6-v2" in ids
    # Every entry exposes a "loaded" flag
    assert all("loaded" in m for m in data["models"])
    # None are loaded on a fresh app
    assert all(m["loaded"] is False for m in data["models"])


@pytest.mark.asyncio
async def test_embedding_routes_v1_embeddings_single(client):
    resp = await client.post(
        "/v1/embeddings",
        json={"model": "sentence-transformers/all-MiniLM-L6-v2", "input": "hello"},
    )
    assert resp.status == 200
    data = await resp.json()
    assert data["object"] == "list"
    assert data["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert len(data["data"]) == 1
    item = data["data"][0]
    assert item["object"] == "embedding"
    assert item["index"] == 0
    assert isinstance(item["embedding"], list)
    assert len(item["embedding"]) == 8
    assert data["usage"]["prompt_tokens"] >= 1
    assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"]


@pytest.mark.asyncio
async def test_embedding_routes_v1_embeddings_batch(client):
    resp = await client.post(
        "/v1/embeddings",
        json={
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "input": ["alpha beta", "gamma", "delta epsilon zeta"],
        },
    )
    assert resp.status == 200
    data = await resp.json()
    assert len(data["data"]) == 3
    assert [d["index"] for d in data["data"]] == [0, 1, 2]
    # Token usage reflects all three texts
    assert data["usage"]["prompt_tokens"] == 2 + 1 + 3


@pytest.mark.asyncio
async def test_embedding_routes_v1_embeddings_validates_body(client):
    resp = await client.post("/v1/embeddings", json={"input": "hi"})
    assert resp.status == 400
    resp = await client.post(
        "/v1/embeddings", json={"model": "x", "input": [1, 2, 3]}
    )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_embedding_routes_load_endpoint(client):
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    resp = await client.post(f"/api/embeddings/models/{model_id}/load")
    assert resp.status == 200
    data = await resp.json()
    assert data["ok"] is True
    assert data["status"] == "loaded"
    assert data["model"]["dimensions"] == 8

    # Second call is idempotent
    resp = await client.post(f"/api/embeddings/models/{model_id}/load")
    assert resp.status == 200

    # List now reports it as loaded
    resp = await client.get("/api/embeddings/models")
    data = await resp.json()
    loaded_flags = {m["id"]: m["loaded"] for m in data["models"]}
    assert loaded_flags[model_id] is True


@pytest.mark.asyncio
async def test_embedding_routes_unload_endpoint(client):
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    await client.post(f"/api/embeddings/models/{model_id}/load")
    resp = await client.post(f"/api/embeddings/models/{model_id}/unload")
    assert resp.status == 200
    data = await resp.json()
    assert data["ok"] is True
    assert data["status"] == "unloaded"


@pytest.mark.asyncio
async def test_server_status_lists_embedding_models(client, app):
    mgr: EmbeddingManager = app["embedding_manager"]
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    mgr.load(model_id)

    resp = await client.get("/api/server/status")
    assert resp.status == 200
    data = await resp.json()
    embed_models = [m for m in data["loaded_models"] if m["type"] == "embed"]
    assert any(m["id"] == model_id for m in embed_models)
    entry = next(m for m in embed_models if m["id"] == model_id)
    assert entry["capabilities"] == ["embeddings"]
    assert entry["dimensions"] == 8


@pytest.mark.asyncio
async def test_server_endpoints_marks_embeddings_available(client):
    resp = await client.get("/api/server/endpoints")
    assert resp.status == 200
    data = await resp.json()
    openai = data["openai"]
    emb = next(ep for ep in openai if ep["path"] == "/v1/embeddings")
    assert emb.get("status") != "planned"


@pytest.mark.asyncio
async def test_embedding_routes_eject_via_server_endpoint(client, app):
    mgr: EmbeddingManager = app["embedding_manager"]
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    mgr.load(model_id)
    assert mgr.is_loaded(model_id)

    resp = await client.post(
        f"/api/server/models/{model_id}/eject"
    )
    assert resp.status == 200
    data = await resp.json()
    assert data["ok"] is True
    assert mgr.is_loaded(model_id) is False
