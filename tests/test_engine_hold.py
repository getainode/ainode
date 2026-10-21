"""Timed holds and recovery use intent, positive death evidence and the launch gate."""

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ainode.discovery.instance import InstanceRecord
from ainode.engine import lifecycle, maintenance
from ainode.engine.instance_manager import InstanceManager


@pytest.fixture(autouse=True)
def hold_file(monkeypatch, tmp_path):
    monkeypatch.setattr(maintenance.core_config, "AINODE_HOME", tmp_path)
    monkeypatch.setattr(maintenance, "hold_path", lambda: tmp_path / "maintenance.json")


def test_hold_persists_expired_deadline_and_clear():
    deadline = time.time() + 60
    maintenance.set_maintenance({"reason": " driver reload ", "until": deadline})
    assert maintenance.held()
    assert maintenance.maintenance(deadline + 1) == {
        "reason": "driver reload", "until": deadline, "active": False,
    }
    assert maintenance.hold_path().stat().st_mode & 0o777 == 0o600
    maintenance.set_maintenance(None)
    assert maintenance.maintenance() is None


@pytest.mark.parametrize("value", [[], {}, {"reason": "", "until": 1},
    {"reason": "x", "until": float("inf")}, {"reason": "x", "until": True},
    {"reason": "x", "until": 0}])
def test_invalid_hold(value):
    with pytest.raises(ValueError):
        maintenance.set_maintenance(value)
    assert not maintenance.hold_path().exists()


@pytest.fixture
def managed(monkeypatch):
    import ainode.api.server as server
    import ainode.models.api_routes as routes
    manager = InstanceManager()
    backend = SimpleNamespace(start=Mock(return_value=True))
    record = InstanceRecord(instance_id="node:model", model="model",
                            head_node_id="node", api_port=8000, status="serving")
    manager.add(record, backend)
    monkeypatch.setattr(server, "_engine_serving", AsyncMock(return_value=False))
    monkeypatch.setattr(routes, "_engine_exited", lambda b: True)
    monkeypatch.setattr(routes, "_GPU_RELEASE_SECONDS", 0)
    monkeypatch.setattr(routes, "_log_failed_engine_output", AsyncMock())
    monkeypatch.setattr(routes, "_wait_for_bind", AsyncMock(return_value=(True, "", 1)))
    return {"instances": manager, "engine_recovery": {}}, record, backend


@pytest.mark.asyncio
async def test_dead_serving_engine_recovers(managed):
    app, record, backend = managed
    await lifecycle.recover_engines(app)
    backend.start.assert_called_once()
    assert record.status == "serving"


@pytest.mark.asyncio
async def test_hold_marks_failed_then_recovers_when_cleared(managed):
    app, record, backend = managed
    maintenance.set_maintenance({"reason": "reload", "until": time.time() + 60})
    await lifecycle.recover_engines(app)
    assert record.status == "failed"
    backend.start.assert_not_called()
    maintenance.set_maintenance(None)
    await lifecycle.recover_engines(app)
    backend.start.assert_called_once()


@pytest.mark.asyncio
async def test_live_container_and_loading_engine_never_restarted(managed, monkeypatch):
    import ainode.models.api_routes as routes
    app, record, backend = managed
    monkeypatch.setattr(routes, "_engine_exited", lambda b: False)
    await lifecycle.recover_engines(app)
    backend.start.assert_not_called()
    monkeypatch.setattr(routes, "_engine_exited", lambda b: True)
    record.status = "starting"
    await lifecycle.recover_engines(app)
    backend.start.assert_not_called()


@pytest.mark.asyncio
async def test_failed_recovery_does_not_loop(managed):
    app, record, backend = managed
    backend.start.return_value = False
    await lifecycle.recover_engines(app)
    await lifecycle.recover_engines(app)
    assert record.status == "failed"
    backend.start.assert_called_once()


@pytest.mark.asyncio
async def test_busy_launch_and_unloaded_instance_not_restarted(managed, monkeypatch):
    import ainode.models.api_routes as routes
    app, record, backend = managed
    async with routes.launch_slot("operator load"):
        await lifecycle.recover_engines(app)
    backend.start.assert_not_called()

    def exited(b):
        app["instances"].remove(record.instance_id)
        return True
    monkeypatch.setattr(routes, "_engine_exited", exited)
    await lifecycle.recover_engines(app)
    backend.start.assert_not_called()


@pytest.mark.asyncio
async def test_cluster_hold_set_and_clear():
    from ainode.api.server import handle_cluster_hold
    class Request:
        method = "POST"
        remote = "127.0.0.1"
        headers = {"X-AINode-Client": "dashboard"}
        app = {"config": SimpleNamespace(node_id="node")}
        async def json(self):
            return self.body
    req = Request()
    req.body = {"node_id": "node", "reason": "reload", "until": time.time() + 60}
    response = await handle_cluster_hold(req)
    assert response.status == 200
    assert json.loads(response.body)["maintenance"]["active"]
    req.body = {"node_id": "node", "maintenance": None}
    response = await handle_cluster_hold(req)
    assert response.status == 200
    assert not maintenance.held()


@pytest.mark.asyncio
async def test_boot_deferred_by_hold_resumes_without_stacked_manifest(monkeypatch):
    import ainode.models.api_routes as routes
    backend = SimpleNamespace(_maintenance_deferred=True, start=Mock(return_value=True))
    app = {"config": SimpleNamespace(model="model"), "engine": backend}
    monkeypatch.setattr(routes, "load_instance_manifest", lambda: [])
    monkeypatch.setattr(routes, "adopt_running_engines", AsyncMock(return_value=[]))
    monkeypatch.setattr(routes, "ensure_startup_sweep", AsyncMock())
    monkeypatch.setattr(routes, "replay_distributed_if_needed", AsyncMock(return_value={}))
    monkeypatch.setattr(routes, "_REPLAY_SETTLE_SECONDS", 0)
    replay = AsyncMock()
    monkeypatch.setattr(routes, "_replay_serialized", replay)
    await routes.replay_instances_on_startup(app)
    backend.start.assert_called_once()
    replay.assert_awaited_once_with(app, app["config"], [])
    assert not backend._maintenance_deferred


@pytest.mark.asyncio
async def test_window_can_expire_before_dead_engine_recovery(managed, monkeypatch):
    app, record, backend = managed
    maintenance.set_maintenance({"reason": "reload", "until": time.time() + 60})
    monkeypatch.setattr(maintenance.time, "time", lambda: 10**12)
    await lifecycle.recover_engines(app)
    backend.start.assert_called_once()
    assert maintenance.maintenance()["active"] is False


@pytest.mark.asyncio
async def test_hold_api_and_nodes_over_http(monkeypatch):
    from aiohttp.test_utils import TestClient, TestServer
    from ainode.api.server import create_app
    from ainode.core.config import NodeConfig
    cfg = NodeConfig(node_id="held-node", node_name="held-node", cluster_enabled=False)
    app = create_app(cfg)
    async with TestClient(TestServer(app), headers={"X-AINode-Client": "dashboard"}) as client:
        deadline = time.time() + 60
        response = await client.post("/api/engine/hold", json={"reason": "reload", "until": deadline})
        assert response.status == 200
        response = await client.get("/api/nodes")
        row = (await response.json())["nodes"][0]
        assert row["maintenance"] == {"reason": "reload", "until": deadline, "active": True}
        response = await client.post("/api/cluster/hold", json={"node_id": "held-node", "maintenance": None})
        assert response.status == 200
        assert (await response.json())["maintenance"] is None
        response = await client.post("/api/engine/hold", json={"reason": "bad", "until": 0})
        assert response.status == 400


def test_discovery_carries_window_to_peer():
    from ainode.discovery.broadcast import NodeAnnouncement, NodeStatus
    from ainode.discovery.cluster import ClusterNode
    window = {"reason": "reload", "until": time.time() + 60, "active": True}
    ann = NodeAnnouncement(node_id="peer", node_name="peer", gpu_name="CPU",
                           gpu_memory_gb=0, unified_memory=False, model="", status="stopped",
                           api_port=8000, web_port=3000, maintenance=window)
    decoded = NodeAnnouncement.from_json(ann.to_json())
    peer = ClusterNode.from_announcement(decoded, NodeStatus.ONLINE)
    assert peer.maintenance == window


@pytest.mark.asyncio
async def test_failed_primary_phase_is_visible_while_held(managed):
    from ainode.api.server import node_load_phase
    app, record, backend = managed
    app["config"] = SimpleNamespace(api_port=8000)
    app["engine"] = backend
    maintenance.set_maintenance({"reason": "reload", "until": time.time() + 60})
    await lifecycle.recover_engines(app)
    assert node_load_phase(app, False) == "failed"
    assert node_load_phase(app, True) == "ready"


@pytest.mark.asyncio
async def test_hold_suppresses_startup_retry(managed, monkeypatch):
    import ainode.models.api_routes as routes
    app, record, backend = managed
    maintenance.set_maintenance({"reason": "reload", "until": time.time() + 60})
    monkeypatch.setattr(routes, "_wait_for_bind", AsyncMock(return_value=(False, "exited", 1)))
    monkeypatch.setattr(routes, "_GPU_RELEASE_SECONDS", 0)
    assert not await routes._ensure_serving(app, 8000, backend.start, "model", backend=backend)
    backend.start.assert_not_called()


@pytest.mark.asyncio
async def test_unload_applies_window_before_stopping(managed):
    from ainode.models.api_routes import handle_model_unload
    app, record, backend = managed
    observed = []
    backend.stop = lambda: observed.append(maintenance.held())
    app["config"] = SimpleNamespace(model=None, api_port=8000)
    class Request:
        method = "POST"
        remote = "127.0.0.1"
        headers = {"X-AINode-Client": "dashboard"}
        query = {}
        async def json(self):
            return {"model": "model", "maintenance": {"reason": "reload", "until": time.time() + 60}}
    req = Request()
    req.app = app
    response = await handle_model_unload(req)
    assert response.status == 200
    assert observed == [True]
    assert app["instances"].is_empty()


@pytest.mark.asyncio
async def test_hold_pauses_stacked_replay_without_holding_launch_slot(monkeypatch):
    import ainode.models.api_routes as routes
    maintenance.set_maintenance({"reason": "reload", "until": time.time() + 60})
    monkeypatch.setattr(routes, "_await_primary_bind", AsyncMock(return_value=True))
    launch = Mock()
    monkeypatch.setattr(routes, "append_solo_instance", launch)
    cfg = SimpleNamespace(model=None)
    assert await routes._replay_serialized({"config": cfg}, cfg, [{"model": "model"}]) is False
    launch.assert_not_called()


@pytest.mark.asyncio
async def test_death_is_reported_while_startup_owns_recovery(managed):
    app, record, backend = managed
    await lifecycle.recover_engines(app, allow_recovery=False)
    assert record.status == "failed"
    backend.start.assert_not_called()
