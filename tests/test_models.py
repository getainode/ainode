"""Tests for the model registry, manager, and API routes.

The catalog is now dynamic (fetched from HuggingFace Hub, Ollama, NVIDIA NIM
with a 24h cache). Tests use the in-memory FALLBACK_CATALOG by mocking the
aggregator so they are deterministic and don't hit the network.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from ainode.models.registry import (
    FALLBACK_CATALOG,
    MODEL_CATALOG,
    CatalogAggregator,
    ModelInfo,
    ModelManager,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_models_dir(tmp_path: Path) -> Path:
    """Provide a temporary models directory."""
    d = tmp_path / "models"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def _force_fallback_catalog(tmp_path, monkeypatch):
    """Default: all tests get the small deterministic fallback catalog.

    This both avoids network calls and keeps legacy model-id assertions happy.
    Tests that want to exercise the live aggregator can opt in.
    """
    # Redirect the cache file away from the user's home dir
    monkeypatch.setattr(
        CatalogAggregator, "CACHE_FILE", tmp_path / "catalog-cache.json"
    )
    # Force fetch() to return [] so ModelManager falls back.
    monkeypatch.setattr(CatalogAggregator, "fetch", lambda self, force_refresh=False: [])


@pytest.fixture
def manager(tmp_models_dir: Path) -> ModelManager:
    return ModelManager(models_dir=tmp_models_dir)


def _fake_downloaded(manager: ModelManager, model_id: str) -> Path:
    """Create a fake downloaded model directory with a dummy file."""
    info = manager.get_catalog_map()[model_id]
    dirname = ModelManager._repo_to_dirname(info.hf_repo)
    model_dir = manager.models_dir / dirname
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text('{"test": true}')
    (model_dir / "model.safetensors").write_bytes(b"\x00" * 1024)
    return model_dir


# ---------------------------------------------------------------------------
# Catalog tests
# ---------------------------------------------------------------------------

class TestCatalog:
    def test_fallback_catalog_has_core_models(self):
        expected = {
            "llama-3.2-3b",
            "qwen-2.5-7b",
            "mistral-7b",
            "phi-3-mini",
            "gemma-2-9b",
        }
        assert expected.issubset(set(FALLBACK_CATALOG.keys()))

    def test_model_catalog_alias(self):
        # MODEL_CATALOG is kept as a backward-compat alias for FALLBACK_CATALOG.
        assert MODEL_CATALOG is FALLBACK_CATALOG

    def test_all_fallback_entries_are_model_info(self):
        for model_id, info in FALLBACK_CATALOG.items():
            assert isinstance(info, ModelInfo), f"{model_id} is not ModelInfo"
            assert info.id == model_id
            assert info.hf_repo
            assert info.size_gb > 0
            assert info.min_memory_gb > 0

    def test_model_info_to_dict(self):
        info = FALLBACK_CATALOG["llama-3.2-3b"]
        d = info.to_dict()
        assert d["id"] == "llama-3.2-3b"
        assert "hf_repo" in d
        assert "size_gb" in d


# ---------------------------------------------------------------------------
# Aggregator tests (unit — mocked, no network)
# ---------------------------------------------------------------------------

class TestAggregator:
    def test_estimate_size_from_id(self):
        agg = CatalogAggregator()
        m = MagicMock(id="meta-llama/Llama-3.1-8B-Instruct", safetensors=None)
        assert agg._estimate_size_gb(m) == pytest.approx(16.0)

    def test_estimate_size_awq(self):
        agg = CatalogAggregator()
        m = MagicMock(id="Some/Llama-70B-AWQ", safetensors=None)
        # 70 * 0.6 = 42.0
        assert agg._estimate_size_gb(m) == pytest.approx(42.0)

    def test_detect_quantization(self):
        agg = CatalogAggregator()
        assert agg._detect_quantization("foo/model-AWQ") == "awq"
        assert agg._detect_quantization("foo/model-gptq-int4") == "gptq"
        assert agg._detect_quantization("foo/plain-model") is None

    def test_is_recommended(self):
        agg = CatalogAggregator()
        assert agg._is_recommended("meta-llama/Llama-3.2-3B-Instruct", 1_000_000) is True
        assert agg._is_recommended("random/base-model", 1_000_000) is False
        # Known family, but base (non-instruct) → not recommended
        assert agg._is_recommended("meta-llama/Llama-3.1-8B", 1_000_000) is False

    def test_cache_roundtrip(self, tmp_path):
        agg = CatalogAggregator()
        agg.CACHE_FILE = tmp_path / "cache.json"
        sample = [FALLBACK_CATALOG["llama-3.2-3b"]]
        agg._save_cache(sample)
        assert agg._cache_valid()
        loaded = agg._load_cache()
        assert len(loaded) == 1
        assert loaded[0].id == "llama-3.2-3b"


# ---------------------------------------------------------------------------
# Manager — list / info tests
# ---------------------------------------------------------------------------

class TestManagerList:
    def test_list_available_all_not_downloaded(self, manager: ModelManager):
        models = manager.list_available()
        assert len(models) == len(FALLBACK_CATALOG)
        for m in models:
            assert m["downloaded"] is False

    def test_list_available_with_downloaded(self, manager: ModelManager):
        _fake_downloaded(manager, "llama-3.2-3b")
        models = manager.list_available()
        by_id = {m["id"]: m for m in models}
        assert by_id["llama-3.2-3b"]["downloaded"] is True
        assert by_id["mistral-7b"]["downloaded"] is False

    def test_list_downloaded_empty(self, manager: ModelManager):
        assert manager.list_downloaded() == []

    def test_list_downloaded_with_model(self, manager: ModelManager):
        _fake_downloaded(manager, "phi-3-mini")
        downloaded = manager.list_downloaded()
        assert len(downloaded) == 1
        assert downloaded[0]["id"] == "phi-3-mini"
        assert downloaded[0]["downloaded"] is True
        assert downloaded[0]["local_size_gb"] >= 0

    def test_list_downloaded_unknown_dir(self, manager: ModelManager):
        """Directories not in catalog show as user-added."""
        custom_dir = manager.models_dir / "some-org--custom-model"
        custom_dir.mkdir()
        (custom_dir / "weights.bin").write_bytes(b"\x00" * 512)
        downloaded = manager.list_downloaded()
        assert len(downloaded) == 1
        assert downloaded[0]["id"] == "some-org/custom-model"
        assert downloaded[0]["hf_repo"] == "some-org/custom-model"
        assert downloaded[0]["downloaded"] is True


class TestManagerInfo:
    def test_get_model_info_exists(self, manager: ModelManager):
        info = manager.get_model_info("mistral-7b")
        assert info is not None
        assert info["id"] == "mistral-7b"
        assert info["downloaded"] is False

    def test_get_model_info_downloaded(self, manager: ModelManager):
        _fake_downloaded(manager, "mistral-7b")
        info = manager.get_model_info("mistral-7b")
        assert info["downloaded"] is True
        assert "local_size_gb" in info

    def test_get_model_info_unknown(self, manager: ModelManager):
        assert manager.get_model_info("nonexistent") is None


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

class TestRecommendations:
    def test_recommend_small_gpu(self, manager: ModelManager):
        recs = manager.recommend_for_gpu(8)
        ids = {r["id"] for r in recs}
        assert "llama-3.2-3b" in ids
        assert "phi-3-mini" in ids

    def test_recommend_large_gpu(self, manager: ModelManager):
        huge = max(info.min_memory_gb for info in FALLBACK_CATALOG.values()) + 1
        recs = manager.recommend_for_gpu(huge)
        assert len(recs) == len(FALLBACK_CATALOG)

    def test_recommend_tiny_gpu(self, manager: ModelManager):
        recs = manager.recommend_for_gpu(1)
        assert len(recs) == 0

    def test_recommend_sorted_by_size_desc(self, manager: ModelManager):
        huge = max(info.min_memory_gb for info in FALLBACK_CATALOG.values()) + 1
        recs = manager.recommend_for_gpu(huge)
        sizes = [r["size_gb"] for r in recs]
        assert sizes == sorted(sizes, reverse=True)


# ---------------------------------------------------------------------------
# Download (mocked)
# ---------------------------------------------------------------------------

class TestDownload:
    def test_download_unknown_model_raises(self, manager: ModelManager):
        with pytest.raises(ValueError, match="Unknown model"):
            manager.download_model("nonexistent")

    def test_download_missing_huggingface_hub(self, manager: ModelManager):
        """If huggingface_hub is not installed, raise RuntimeError."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "huggingface_hub":
                raise ImportError("no module")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RuntimeError, match="huggingface_hub"):
                manager.download_model("llama-3.2-3b")

    @patch("ainode.models.registry.snapshot_download", create=True)
    def test_download_calls_snapshot_download(self, mock_sd, manager: ModelManager):
        """Verify download_model calls huggingface_hub.snapshot_download correctly."""
        expected_dir = manager.models_dir / "meta-llama--Llama-3.2-3B-Instruct"
        mock_sd.return_value = str(expected_dir)

        with patch.dict("sys.modules", {"huggingface_hub": MagicMock(snapshot_download=mock_sd)}):
            result = manager.download_model("llama-3.2-3b")

        mock_sd.assert_called_once_with(
            repo_id="meta-llama/Llama-3.2-3B-Instruct",
            local_dir=str(expected_dir),
            local_dir_use_symlinks=False,
        )
        assert result == expected_dir


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_downloaded_model(self, manager: ModelManager):
        model_dir = _fake_downloaded(manager, "mistral-7b")
        assert model_dir.exists()
        assert manager.delete_model("mistral-7b") is True
        assert not model_dir.exists()

    def test_delete_not_downloaded(self, manager: ModelManager):
        assert manager.delete_model("mistral-7b") is False

    def test_delete_unknown_raises(self, manager: ModelManager):
        with pytest.raises(ValueError, match="Unknown model"):
            manager.delete_model("nonexistent")


# ---------------------------------------------------------------------------
# API routes (aiohttp test client)
# ---------------------------------------------------------------------------

pytest.importorskip("aiohttp")
from aiohttp import web
from ainode.models.api_routes import register_model_routes


class TestModelAPI:
    """Test API routes using aiohttp test client via pytest-aiohttp."""

    @pytest.fixture
    def app(self, manager: ModelManager) -> web.Application:
        app = web.Application()
        register_model_routes(app, manager=manager)
        return app

    @pytest.mark.asyncio
    async def test_list_models(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/api/models")
        assert resp.status == 200
        data = await resp.json()
        assert "models" in data
        assert len(data["models"]) == len(FALLBACK_CATALOG)
        assert data["count"] == len(FALLBACK_CATALOG)

    @pytest.mark.asyncio
    async def test_get_model(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/api/models/llama-3.2-3b")
        assert resp.status == 200
        data = await resp.json()
        assert data["id"] == "llama-3.2-3b"

    @pytest.mark.asyncio
    async def test_get_model_not_found(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/api/models/nonexistent")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_download_model(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.post("/api/models/llama-3.2-3b/download")
        assert resp.status == 202
        data = await resp.json()
        assert data["status"] == "downloading"
        assert "job_id" in data

    @pytest.mark.asyncio
    async def test_download_unknown(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.post("/api/models/nonexistent/download")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_delete_model_not_downloaded(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.delete("/api/models/llama-3.2-3b")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_delete_model_downloaded(self, app, aiohttp_client, manager):
        _fake_downloaded(manager, "llama-3.2-3b")
        client = await aiohttp_client(app)
        resp = await client.delete("/api/models/llama-3.2-3b")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_recommended_no_gpu(self, app, aiohttp_client):
        with patch("ainode.models.api_routes.detect_gpu", return_value=None):
            client = await aiohttp_client(app)
            resp = await client.get("/api/models/recommended")
            assert resp.status == 200
            data = await resp.json()
            assert "error" in data

    @pytest.mark.asyncio
    async def test_refresh_catalog(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.post("/api/models/refresh")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "refreshed"
        assert "count" in data
