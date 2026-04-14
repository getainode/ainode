"""Tests for ainode.datasets — manager and HTTP routes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from ainode.datasets.manager import (
    Dataset,
    DatasetFormat,
    DatasetManager,
    DatasetSource,
)
from ainode.datasets.api_routes import setup_dataset_routes


# =============================================================================
# DatasetManager
# =============================================================================


class TestDatasetManager:
    def test_empty_manager(self, tmp_path):
        mgr = DatasetManager(root=tmp_path)
        assert mgr.list() == []

    def test_add_upload_jsonl(self, tmp_path):
        mgr = DatasetManager(root=tmp_path)
        content = b'{"instruction":"hi","output":"hello"}\n{"instruction":"a","output":"b"}\n'
        ds = mgr.add_upload("alpaca.jsonl", content, name="alpaca")
        assert ds.name == "alpaca"
        assert ds.source == DatasetSource.UPLOAD.value
        assert ds.format == DatasetFormat.JSONL.value
        assert ds.samples == 2
        assert ds.size_bytes == len(content)
        assert Path(ds.path).exists()

    def test_add_upload_rejects_unsupported_ext(self, tmp_path):
        mgr = DatasetManager(root=tmp_path)
        with pytest.raises(ValueError, match="Unsupported file type"):
            mgr.add_upload("data.exe", b"malicious", name="x")

    def test_add_upload_rejects_oversized(self, tmp_path, monkeypatch):
        # Set the max to something tiny for the test
        from ainode.datasets import manager as m
        monkeypatch.setattr(m, "MAX_UPLOAD_BYTES", 10)
        mgr = DatasetManager(root=tmp_path)
        with pytest.raises(ValueError, match="too large"):
            mgr.add_upload("data.jsonl", b"x" * 100, name="x")

    def test_add_local(self, tmp_path):
        mgr = DatasetManager(root=tmp_path)
        data_file = tmp_path / "raw.jsonl"
        data_file.write_text('{"a":1}\n{"a":2}\n')
        ds = mgr.add_local(str(data_file), name="raw")
        assert ds.source == DatasetSource.LOCAL.value
        assert ds.samples == 2

    def test_add_local_missing(self, tmp_path):
        mgr = DatasetManager(root=tmp_path)
        with pytest.raises(ValueError, match="does not exist"):
            mgr.add_local(str(tmp_path / "missing.jsonl"))

    def test_add_huggingface(self, tmp_path):
        mgr = DatasetManager(root=tmp_path)
        ds = mgr.add_huggingface("tatsu-lab/alpaca", name="alpaca-hf")
        assert ds.source == DatasetSource.HUGGINGFACE.value
        assert ds.path == "tatsu-lab/alpaca"

    def test_add_huggingface_invalid(self, tmp_path):
        mgr = DatasetManager(root=tmp_path)
        with pytest.raises(ValueError, match="repo_id"):
            mgr.add_huggingface("not-a-valid-id")

    def test_persistence(self, tmp_path):
        mgr1 = DatasetManager(root=tmp_path)
        mgr1.add_upload("a.jsonl", b'{"t":"x"}\n', name="a")
        # Reopen — should see it
        mgr2 = DatasetManager(root=tmp_path)
        assert len(mgr2.list()) == 1
        assert mgr2.list()[0]["name"] == "a"

    def test_delete_upload(self, tmp_path):
        mgr = DatasetManager(root=tmp_path)
        ds = mgr.add_upload("d.jsonl", b'{"t":"x"}\n', name="d")
        p = Path(ds.path)
        assert p.exists()
        assert mgr.delete(ds.id) is True
        # File removed and registry updated
        assert not p.exists()
        assert mgr.list() == []

    def test_delete_local_preserves_file(self, tmp_path):
        mgr = DatasetManager(root=tmp_path)
        f = tmp_path / "keep.jsonl"
        f.write_text('{"t":"x"}\n')
        ds = mgr.add_local(str(f))
        assert mgr.delete(ds.id) is True
        # Local file must be preserved — we only referenced it
        assert f.exists()

    def test_delete_not_found(self, tmp_path):
        mgr = DatasetManager(root=tmp_path)
        assert mgr.delete("nope") is False

    def test_preview_jsonl(self, tmp_path):
        mgr = DatasetManager(root=tmp_path)
        content = '\n'.join(json.dumps({"i": i}) for i in range(5)).encode()
        ds = mgr.add_upload("p.jsonl", content)
        prev = mgr.preview(ds.id, limit=3)
        assert prev["total_samples"] == 5
        assert len(prev["samples"]) == 3
        assert prev["samples"][0]["i"] == 0

    def test_preview_missing(self, tmp_path):
        mgr = DatasetManager(root=tmp_path)
        with pytest.raises(KeyError):
            mgr.preview("nope")


# =============================================================================
# HTTP routes
# =============================================================================


@pytest.fixture
def dataset_app(tmp_path):
    app = web.Application()
    mgr = DatasetManager(root=tmp_path)
    setup_dataset_routes(app, mgr)
    return app


@pytest_asyncio.fixture
async def dataset_client(dataset_app):
    async with TestClient(TestServer(dataset_app)) as c:
        yield c


class TestDatasetAPI:
    @pytest.mark.asyncio
    async def test_list_empty(self, dataset_client):
        resp = await dataset_client.get("/api/datasets")
        assert resp.status == 200
        data = await resp.json()
        assert data == {"datasets": []}

    @pytest.mark.asyncio
    async def test_create_hf(self, dataset_client):
        resp = await dataset_client.post(
            "/api/datasets",
            json={"source": "huggingface", "repo_id": "tatsu-lab/alpaca", "name": "alpaca"},
        )
        assert resp.status == 201
        data = await resp.json()
        assert data["source"] == "huggingface"
        assert data["name"] == "alpaca"

    @pytest.mark.asyncio
    async def test_create_hf_invalid(self, dataset_client):
        resp = await dataset_client.post(
            "/api/datasets",
            json={"source": "huggingface", "repo_id": "bad"},
        )
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_create_unsupported_source(self, dataset_client):
        resp = await dataset_client.post(
            "/api/datasets",
            json={"source": "ftp", "path": "ftp://example"},
        )
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_get_missing(self, dataset_client):
        resp = await dataset_client.get("/api/datasets/nope")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_delete_missing(self, dataset_client):
        resp = await dataset_client.delete("/api/datasets/nope")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_preview_missing(self, dataset_client):
        resp = await dataset_client.get("/api/datasets/nope/preview")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_create_and_get(self, dataset_client):
        resp = await dataset_client.post(
            "/api/datasets",
            json={"source": "huggingface", "repo_id": "x/y", "name": "xy"},
        )
        data = await resp.json()
        ds_id = data["id"]
        resp = await dataset_client.get(f"/api/datasets/{ds_id}")
        assert resp.status == 200
        got = await resp.json()
        assert got["id"] == ds_id
