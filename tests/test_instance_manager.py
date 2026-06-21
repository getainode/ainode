"""P2-2: InstanceManager + per-instance NvidiaBackend + append-launch + eject-by-model."""

from __future__ import annotations

import asyncio
import dataclasses as dc
import json

import ainode.engine.backends as backends_mod
from ainode.api.server_routes import handle_server_eject
from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeStatus
from ainode.discovery.cluster import ClusterNode, ClusterState
from ainode.discovery.instance import InstanceRecord
from ainode.engine.backends.nvidia import NvidiaBackend
from ainode.engine.instance_manager import InstanceManager
from ainode.engine.sharding_routes import handle_sharding_launch


class FakeBackend:
    """Stand-in for a started engine — no docker, records lifecycle."""
    created: list = []

    def __init__(self, config, on_ready=None, instance_id=""):
        self.config = config
        self.instance_id = instance_id
        self.started = False
        self.stopped = False
        FakeBackend.created.append(self)

    def start_distributed(self):
        self.started = True
        return True

    def is_running(self):
        return self.started and not self.stopped

    def stop(self):
        self.stopped = True


class _Req:
    def __init__(self, app, body=None, match_info=None):
        self.app = app
        self._body = body or {}
        self.match_info = match_info or {}

    async def json(self):
        return self._body


def _cn(nid, fab):
    # peers must advertise distributed_mode="member" to be selectable as workers
    return ClusterNode(node_id=nid, node_name=nid, gpu_name="NVIDIA GB10", gpu_memory_gb=128.0,
                       unified_memory=True, model="", status=NodeStatus.ONLINE,
                       api_port=8000, web_port=3000, last_seen=0.0, fabric_ip=fab,
                       distributed_mode="member")


# --- InstanceManager ---------------------------------------------------------

def test_manager_allocate_port_and_lifecycle():
    m = InstanceManager(base_port=8000)
    assert m.is_empty() and m.allocate_port() == 8000
    m.add(InstanceRecord(instance_id="h:A", model="A", api_port=8000), object())
    assert m.allocate_port() == 8001          # 8000 taken
    m.add(InstanceRecord(instance_id="h:B", model="B", api_port=8001), object())
    assert m.allocate_port() == 8002
    assert m.by_model("B").record.api_port == 8001
    m.remove("h:A")
    assert m.allocate_port() == 8000          # freed
    assert m.by_model("A") is None


# --- per-instance NvidiaBackend ---------------------------------------------

def test_backend_per_instance_names_and_ports():
    primary = NvidiaBackend(NodeConfig(node_id="h", api_port=8000))
    second = NvidiaBackend(dc.replace(NodeConfig(node_id="h"), api_port=8001), instance_id="8001")
    # distinct names; primary keeps legacy unsuffixed name (back-compat)
    assert primary._head_container_name() == "ainode-vllm-head"
    assert second._head_container_name() == "ainode-vllm-head-8001"
    assert primary._worker_container_name("10.0.0.2") != second._worker_container_name("10.0.0.2")
    # distinct Ray + torch rendezvous ports (collide otherwise on a shared head)
    assert (primary._ray_port(), primary._master_port()) == (6379, "29501")
    assert (second._ray_port(), second._master_port()) == (6380, "29502")


# --- append-launch -----------------------------------------------------------

def test_launch_appends_without_teardown(monkeypatch):
    FakeBackend.created = []
    monkeypatch.setattr(backends_mod, "get_backend",
                        lambda c, on_ready=None, instance_id="": FakeBackend(c, instance_id=instance_id))
    config = NodeConfig(node_id="head", api_port=8000, engine_backend="nvidia")
    config.save = lambda: None
    cluster = ClusterState()
    for nid, fab in [("head", "10.0.0.1"), ("p2", "10.0.0.2"), ("p3", "10.0.0.3")]:
        cluster.add_node(_cn(nid, fab))
    app = {"config": config, "cluster_state": cluster}

    r1 = asyncio.run(handle_sharding_launch(_Req(app, {"model": "A", "node_ids": ["head", "p2"], "strategy": "tensor"})))
    d1 = json.loads(r1.body)
    assert d1["api_port"] == 8000 and d1["model"] == "A"
    mgr = app["instances"]
    assert len(mgr.records()) == 1
    assert app["engine"] is FakeBackend.created[0]      # primary wired for proxy/status

    r2 = asyncio.run(handle_sharding_launch(_Req(app, {"model": "B", "node_ids": ["head", "p3"], "strategy": "tensor"})))
    d2 = json.loads(r2.body)
    assert d2["api_port"] == 8001                        # second instance, own port
    assert len(mgr.records()) == 2                       # APPENDED, not swapped
    assert app["engine"] is FakeBackend.created[0]       # primary unchanged
    assert FakeBackend.created[0].stopped is False       # A NOT torn down
    assert FakeBackend.created[1].instance_id == "8001"  # name token = port


# --- eject by model ----------------------------------------------------------

def test_eject_stops_one_instance():
    mgr = InstanceManager(base_port=8000)
    b = FakeBackend(NodeConfig(node_id="h"))
    mgr.add(InstanceRecord(instance_id="h:A", model="A", api_port=8000), b)
    app = {"instances": mgr, "engine": b}
    resp = asyncio.run(handle_server_eject(_Req(app, match_info={"model_id": "A"})))
    assert json.loads(resp.body)["ok"] is True
    assert b.stopped is True
    assert mgr.by_model("A") is None
    assert app["engine"] is None                         # primary cleared
