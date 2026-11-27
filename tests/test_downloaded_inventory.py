"""Catalog inventory reads the configured stores and each answering fleet member."""

from __future__ import annotations

import json
import shutil
import socket
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from ainode.auth.middleware import AuthConfig, auth_middleware
from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeStatus
from ainode.discovery.cluster import ClusterNode, ClusterState
from ainode.discovery.instance import InstanceRecord
from ainode.engine.instance_manager import InstanceManager
from ainode.models.api_routes import register_model_routes
from ainode.models.inventory import scan_model_copies

FLEET = "/api/models/downloaded/fleet"
LOCAL = "/api/models/downloaded?inventory=1"
GLM = "sample/GLM-5.3-Flash-NVFP4"
SECRET = "synthetic-inventory-fleet"


def checkpoint(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"weights")
    return path


def cached_model(root, repo=GLM, revision="a" * 40):
    base = root / ("models--" + repo.replace("/", "--"))
    snapshot = base / "snapshots" / revision
    snapshot.mkdir(parents=True)
    blobs = base / "blobs"
    blobs.mkdir(exist_ok=True)
    files = {
        "config.json": "{}",
        "model.safetensors.index.json": json.dumps({
            "weight_map": {"layers.1.weight": "model-00001-of-00002.safetensors",
                           "layers.2.weight": "model-00002-of-00002.safetensors"},
        }),
        "model-00001-of-00002.safetensors": "weights",
        "model-00002-of-00002.safetensors": "more weights",
    }
    for index, (name, contents) in enumerate(files.items()):
        blob = blobs / f"{revision}-{index}"
        blob.write_text(contents)
        (snapshot / name).symlink_to(f"../../blobs/{blob.name}")
    return snapshot


def node(node_id, port, status=NodeStatus.ONLINE, cluster_id="inventory", **kwargs):
    return ClusterNode(node_id=node_id, node_name=node_id, gpu_name="", gpu_memory_gb=0,
                       unified_memory=False, model="", status=status, api_port=8000,
                       web_port=port, last_seen=time.time(), peer_ip="127.0.0.1",
                       cluster_id=cluster_id, **kwargs)


@asynccontextmanager
async def model_node(root, name="local", *, auth=None, engine_port=None, cache=None):
    app = web.Application(middlewares=[auth_middleware])
    config = NodeConfig(node_id=name, node_name=name, models_dir=str(root),
                        cluster_id="inventory", cluster_secret=SECRET,
                        hf_cache_dir=str(cache) if cache else None)
    app["config"] = config
    app["auth_config"] = auth
    app["cluster_state"] = ClusterState()
    app["instances"] = InstanceManager()
    if engine_port is not None:
        backend_config = SimpleNamespace(served_model_name=["synthetic-alias"])
        app["instances"].add(InstanceRecord(instance_id="local:loaded", api_port=engine_port,
                                             model="sample/loaded"), SimpleNamespace(config=backend_config))
    register_model_routes(app, models_dir=str(root))
    async with aiohttp.ClientSession() as session:
        app["client_session"] = session
        async with TestClient(TestServer(app)) as client:
            yield app, client


def test_scan_existing_store_layouts_and_distinct_variants(tmp_path):
    models = tmp_path / "models"
    cache = tmp_path / "custom-hf-cache"
    checkpoint(models / "sample--weights--variant")
    checkpoint(models / "internal-quant")
    (models / "internal-alias").symlink_to(models / "internal-quant", target_is_directory=True)
    checkpoint(models / "sample" / "nested")
    checkpoint(models / "models--sample--flat")
    checkpoint(models / "hub" / "models--sample--in-hub")
    cached_model(models / "hf-cache" / "hub")
    cached_model(cache / "hub", revision="b" * 40)
    cached_model(cache / "hub", revision="c" * 40)
    (models / "local.gguf").write_bytes(b"GGUF")
    gguf = models / "sample--mixed"
    gguf.mkdir()
    (gguf / "Q4.gguf").write_bytes(b"GGUF")
    (gguf / "Q8.gguf").write_bytes(b"GGUF")
    rows = scan_model_copies(models, str(cache))
    assert {(row.id, row.layout, row.revision, row.file) for row in rows} == {
        ("sample/weights--variant", "Model store", "", ""),
        ("internal-quant", "Model store", "", ""),
        ("internal-alias", "Model store", "", ""),
        ("sample/nested", "Nested store", "", ""),
        ("sample/flat", "Flat HF cache", "", ""),
        ("sample/in-hub", "Hub cache", "", ""),
        (GLM, "Engine cache", "a" * 40, ""),
        (GLM, "Engine cache", "b" * 40, ""),
        (GLM, "Engine cache", "c" * 40, ""),
        ("local.gguf", "Model store", "", ""),
        ("sample/mixed", "Model store", "", "Q4.gguf"),
        ("sample/mixed", "Model store", "", "Q8.gguf"),
    }
    assert {row.status for row in rows} == {"available"}


def test_partial_or_unsafe_checkpoints_cannot_confirm_availability(tmp_path):
    models = tmp_path / "models"
    snapshot = cached_model(models / "hf-cache" / "hub")
    (snapshot / "model-00002-of-00002.safetensors").resolve().unlink()
    orphan = models / "hub" / "models--sample--orphan" / "refs"
    orphan.mkdir(parents=True)
    (orphan / "main").write_text("a" * 40)
    outside = checkpoint(tmp_path / "outside")
    (models / "sample--escape").symlink_to(outside, target_is_directory=True)
    unsafe = models / "sample--unsafe"
    unsafe.mkdir()
    (unsafe / "config.json").write_text("{}")
    (unsafe / "model.safetensors").symlink_to(outside / "model.safetensors")
    (unsafe / "unrelated").symlink_to(outside, target_is_directory=True)
    live = checkpoint(models / "sample--partial")
    partial = live / ".cache" / "huggingface" / "download"
    partial.mkdir(parents=True)
    (partial / "tokenizer.json.incomplete").write_bytes(b"partial")
    unknown = models / "unrelated"
    unknown.mkdir()
    (unknown / "README.md").write_text("not a model")
    split = models / "sample--split"
    split.mkdir()
    (split / "Q4-00001-of-00002.gguf").write_bytes(b"GGUF")
    (models / "loose-00001-of-00002.gguf").write_bytes(b"GGUF")
    missing_shard = models / "sample--indexed"
    missing_shard.mkdir()
    (missing_shard / "config.json").write_text("{}")
    (missing_shard / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"layer": "missing.safetensors"},
    }))
    rows = {row.id: row.status for row in scan_model_copies(models)}
    assert rows == {
        GLM: "incomplete",
        "loose-of-00002.gguf": "incomplete",
        "sample/indexed": "incomplete",
        "sample/orphan": "missing",
        "sample/partial": "incomplete",
        "sample/split": "incomplete",
        "sample/unsafe": "incomplete",
    }


@pytest.mark.asyncio
async def test_refresh_reports_download_completion_removal_and_scan_failure(tmp_path):
    root = tmp_path / "models"
    async with model_node(root) as (app, client):
        async def rows():
            response = await client.get(LOCAL)
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store"
            return (await response.json())["models"]

        assert await rows() == []
        job = {"model_id": "sample/fresh", "status": "downloading", "finished_at": None}
        app["download_jobs"]["active"] = job
        assert [(row["id"], row["status"]) for row in await rows()] == [
            ("sample/fresh", "downloading"),
        ]
        downloaded = checkpoint(root / "sample--fresh")
        job.update(status="completed", finished_at=time.time())
        assert [(row["status"], row["loaded"]) for row in await rows()] == [("available", False)]
        shutil.rmtree(downloaded)
        assert [(row["status"], row["loaded"]) for row in await rows()] == [("missing", None)]
        job["finished_at"] -= 3602
        assert await rows() == []
        root.rmdir()
        response = await client.get(LOCAL)
        assert response.status == 503
        assert (await response.json())["status"] == "unavailable"
        response = await client.get(FLEET)
        assert (await response.json())["nodes"][0]["status"] == "unavailable"
        assert not root.exists()


@pytest.mark.asyncio
async def test_fleet_confirms_local_and_authenticated_peer_with_explicit_gaps(tmp_path, monkeypatch):
    monkeypatch.setattr("ainode.auth.middleware.AUTH_FILE", tmp_path / "auth.json")
    monkeypatch.setenv("AINODE_ENGINE_API_KEY", "synthetic-engine-key")
    engine_calls = []

    async def engine_models(request):
        engine_calls.append(request.headers.get("Authorization"))
        return web.json_response({"data": [{"id": "synthetic-alias"}]})

    engine = web.Application()
    engine.router.add_get("/v1/models", engine_models)
    checkpoint(tmp_path / "local-models" / "sample--loaded")
    cached_model(tmp_path / "peer-models" / "hf-cache" / "hub")
    auth = AuthConfig(enabled=True)
    key = auth.generate_key("fixture")["key"]
    scoped_key = auth.generate_key("decisions", scope="decide")["key"]
    headers = {"Authorization": f"Bearer {key}"}
    with socket.socket() as unreachable:
        unreachable.bind(("127.0.0.1", 0))
        async with TestServer(engine) as probe, model_node(
            tmp_path / "peer-models", "peer", auth=auth
        ) as (_, peer):
            async with model_node(tmp_path / "local-models", auth=auth, engine_port=probe.port) as (app, client):
                app["cluster_state"].add_node(node("peer", peer.port))
                app["cluster_state"].add_node(node("stale", peer.port, NodeStatus.STALE))
                app["cluster_state"].add_node(node("unreachable", unreachable.getsockname()[1]))
                app["cluster_state"].add_node(node("excluded", peer.port, cluster_id="other-cluster"))
                assert (await client.get(FLEET)).status == 401
                assert (await client.get(FLEET, headers={"Authorization": f"Bearer {scoped_key}"})).status == 403
                response = await client.get(FLEET, headers=headers)
                assert response.status == 200
                nodes = {item["node_id"]: item for item in (await response.json())["nodes"]}
                assert {name: row["status"] for name, row in nodes.items()} == {
                    "local": "ok", "peer": "ok", "stale": "stale", "unreachable": "unavailable",
                }
                assert [(row["id"], row["loaded"]) for row in nodes["local"]["models"]] == [
                    ("sample/loaded", True),
                ]
                assert [(row["id"], row["loaded"]) for row in nodes["peer"]["models"]] == [(GLM, False)]
                assert all(nodes[name]["models"] == [] for name in ("stale", "unreachable"))
                assert engine_calls == ["Bearer synthetic-engine-key"]


@pytest.mark.asyncio
async def test_open_entry_does_not_expose_protected_peer_or_accept_legacy_empty_reply(tmp_path, monkeypatch):
    monkeypatch.setattr("ainode.auth.middleware.AUTH_FILE", tmp_path / "auth.json")
    legacy = web.Application()
    legacy.router.add_get("/api/models/downloaded", lambda _: web.json_response({"models": []}))
    async with TestServer(legacy) as old:
        async with model_node(tmp_path / "peer-models", "peer", auth=AuthConfig(enabled=True)) as (_, peer):
            async with model_node(tmp_path / "local-models") as (app, client):
                app["cluster_state"].add_node(node("peer", peer.port))
                app["cluster_state"].add_node(node("old", old.port))
                response = await client.get(FLEET)
                assert [(row["node_id"], row["status"]) for row in (await response.json())["nodes"]] == [
                    ("local", "ok"), ("old", "unavailable"), ("peer", "unavailable"),
                ]


@pytest.mark.asyncio
async def test_unverified_runtime_or_ambiguous_revision_keeps_load_state_unknown(tmp_path):
    root = tmp_path / "models"
    primary = checkpoint(root / "sample--loaded")
    response = {"data": [{"id": "unrelated-model"}]}

    async def engine_models(_):
        return web.json_response(response)

    engine_app = web.Application()
    engine_app.router.add_get("/v1/models", engine_models)
    async with TestServer(engine_app) as engine:
        async with model_node(root, engine_port=engine.port) as (_, client):
            rows = (await (await client.get(LOCAL)).json())["models"]
            assert [(row["status"], row["loaded"]) for row in rows] == [("available", None)]
            response["data"][0]["id"] = "synthetic-alias"
            cached_model(root / "hf-cache" / "hub", repo="sample/loaded")
            rows = (await (await client.get(LOCAL)).json())["models"]
            assert {row["layout"] for row in rows} == {"Model store", "Engine cache"}
            assert len(rows) == 2 and all(row["loaded"] is None for row in rows)
            shutil.rmtree(primary)
            rows = (await (await client.get(LOCAL)).json())["models"]
            assert [(row["layout"], row["loaded"]) for row in rows] == [("Engine cache", True)]
