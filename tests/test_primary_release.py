"""Losing the primary never promotes a survivor (the Spark-4 Whisper leak).

The primary is the instance on the node's OWN api_port, the one config.json boots.
Unloading it used to repoint ``app["engine"]`` and ``config.model`` at
``survivors[0]``. On Spark-4 that made Whisper, a stacked instance on :8001, the
node's model while config.json still held the old primary's image, vLLM flags and
memory share: the node advertised Whisper on :8000, where nothing served it, and
the next boot would have launched Whisper with another model's engine parameters.

Pinned here, for both ways the primary can be taken down (``/api/models/unload``
and the server view's eject):

* the slot is EMPTY afterwards: ``app["engine"]`` and ``config.model`` are None;
* the old primary's per-load overrides are gone from the config, reset to the
  NodeConfig defaults;
* the survivor is untouched: still running, still on its own port, still in the
  manifest with its OWN parameters.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import ainode.models.api_routes as mr
from ainode.core.config import NodeConfig
from ainode.discovery.cluster import ClusterState

from tests.test_cluster_load import _patch_backend, _Req

PRIMARY = "Qwen/Qwen3.8-27B"
WHISPER = "openai/whisper-large-v3-turbo"
PRIMARY_LOAD = {
    "model": PRIMARY,
    "gpu_memory_utilization": 0.55,
    "max_model_len": 65536,
    "engine_image": "vllm/vllm-openai:v0.27.1",
    "extra_vllm_args": ["--reasoning-parser", "qwen3"],
    "served_model_name": ["qwen"],
}


@pytest.fixture
def node(monkeypatch, tmp_path):
    """A Spark-4 shaped node: a pinned primary on :8000, Whisper stacked on :8001."""
    import ainode.engine.reconcile as reconcile

    monkeypatch.setattr(mr, "_manifest_path", lambda: tmp_path / "instances.json")
    monkeypatch.setattr(reconcile, "record_path", lambda: tmp_path / "distributed.json")
    _patch_backend(monkeypatch)
    cfg = NodeConfig(node_id="spark4", api_port=8000)
    saves = []
    cfg.save = lambda: saves.append(cfg.model)
    app = {"engine": None, "config": cfg, "cluster_state": ClusterState(),
           "ray_autostart_state": None}
    asyncio.run(mr.handle_model_load(_Req(app, dict(PRIMARY_LOAD))))
    asyncio.run(mr.handle_model_load(_Req(app, {
        "model": WHISPER, "gpu_memory_utilization": 0.2})))
    manager = app["instances"]
    assert manager.by_model(PRIMARY).record.api_port == 8000
    assert manager.by_model(WHISPER).record.api_port == 8001
    assert cfg.model == PRIMARY and cfg.engine_image == PRIMARY_LOAD["engine_image"]
    return app, cfg, manager, saves, tmp_path


def _assert_slot_released(app, cfg, manager, tmp_path):
    defaults = NodeConfig()
    assert app["engine"] is None, "a survivor was promoted into the primary slot"
    assert cfg.model is None, f"config.model leaked to {cfg.model!r}"
    # The old primary's launch parameters do not outlive it.
    assert cfg.engine_image == defaults.engine_image
    assert cfg.extra_vllm_args == defaults.extra_vllm_args
    assert cfg.max_model_len == defaults.max_model_len
    assert cfg.served_model_name == defaults.served_model_name
    assert cfg.gpu_memory_utilization == defaults.gpu_memory_utilization

    # Whisper stays what it was: stacked, on its own port, running.
    whisper = manager.by_model(WHISPER)
    assert whisper is not None and whisper.backend.stopped is False
    assert whisper.record.api_port == 8001
    assert whisper.backend.config.gpu_memory_utilization == 0.2
    assert manager.by_model(PRIMARY) is None

    # The restart replays Whisper from the manifest with ITS parameters, and
    # nothing boots on the primary port.
    entries = json.loads((tmp_path / "instances.json").read_text())["instances"]
    assert [e["model"] for e in entries] == [WHISPER]
    assert entries[0]["gpu_memory_utilization"] == 0.2
    assert entries[0].get("engine_image") != PRIMARY_LOAD["engine_image"]
    assert entries[0].get("extra_vllm_args") != PRIMARY_LOAD["extra_vllm_args"]


def test_unloading_the_primary_does_not_promote_whisper(node):
    app, cfg, manager, saves, tmp_path = node

    resp = asyncio.run(mr.handle_model_unload(_Req(app, {"model": PRIMARY})))

    body = json.loads(resp.body)
    assert body["stopped"] is True and body["remaining"] == 1
    _assert_slot_released(app, cfg, manager, tmp_path)
    assert saves and saves[-1] is None, "the cleared primary was not persisted"


def test_unloading_the_primary_by_port_does_not_promote_whisper(node):
    app, cfg, manager, _saves, tmp_path = node

    asyncio.run(mr.handle_model_unload(_Req(app, {"api_port": 8000})))

    _assert_slot_released(app, cfg, manager, tmp_path)


def test_ejecting_the_primary_does_not_promote_whisper(node):
    from ainode.api.server_routes import handle_server_eject

    app, cfg, manager, _saves, tmp_path = node

    class _EjectReq:
        def __init__(self, app, model_id):
            self.app = app
            self.match_info = {"model_id": model_id}

    resp = asyncio.run(handle_server_eject(_EjectReq(app, PRIMARY)))

    assert json.loads(resp.body)["ok"] is True
    _assert_slot_released(app, cfg, manager, tmp_path)


def test_unloading_the_stacked_whisper_leaves_the_primary_alone(node):
    app, cfg, manager, _saves, _tmp = node
    primary_backend = manager.by_model(PRIMARY).backend

    asyncio.run(mr.handle_model_unload(_Req(app, {"model": WHISPER})))

    assert app["engine"] is primary_backend
    assert cfg.model == PRIMARY
    assert cfg.engine_image == PRIMARY_LOAD["engine_image"]
    assert cfg.extra_vllm_args == PRIMARY_LOAD["extra_vllm_args"]


def test_unloading_the_last_instance_still_empties_the_node(node):
    app, cfg, manager, _saves, _tmp = node

    asyncio.run(mr.handle_model_unload(_Req(app, {"model": WHISPER})))
    asyncio.run(mr.handle_model_unload(_Req(app, {"model": PRIMARY})))

    assert manager.is_empty()
    assert app["engine"] is None and cfg.model is None
    assert cfg.engine_image == ""
