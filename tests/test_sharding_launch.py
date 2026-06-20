"""BUG D + node-selection: /api/sharding/launch must resolve participating peers
to their FABRIC IPs (never the mgmt-LAN UDP peer_ip) and honor explicit node_ids.

Calls the handler directly with a fake request (like test_distributed) and a
patched backend so nothing actually launches.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeAnnouncement, NodeStatus
from ainode.discovery.cluster import ClusterNode, ClusterState
from ainode.engine.sharding_routes import handle_sharding_launch
import ainode.engine.backends as backends


def _member(node_id, fabric_ip, peer_ip="192.168.0.99"):
    return ClusterNode(
        node_id=node_id, node_name=f"host-{node_id}", gpu_name="NVIDIA GB10",
        gpu_memory_gb=128.0, unified_memory=True, model="", status=NodeStatus.ONLINE,
        api_port=8000, web_port=3000, last_seen=0.0,
        distributed_mode="member", peer_ip=peer_ip, fabric_ip=fabric_ip,
    )


def _local_ann(node_id="head"):
    return NodeAnnouncement(
        node_id=node_id, node_name="head", gpu_name="NVIDIA GB10", gpu_memory_gb=128.0,
        unified_memory=True, model="", status="starting", api_port=8000, web_port=3000,
        distributed_mode="head",
    )


class _FakeBackend:
    """Stand-in for NvidiaBackend — records launch, never touches docker."""
    last = {}

    def __init__(self, config):
        _FakeBackend.last = {"config": config, "launched": False}

    def is_running(self):
        return False

    def start_distributed(self):
        _FakeBackend.last["launched"] = True
        _FakeBackend.last["peer_ips"] = list(self.config.peer_ips)
        return True

    @property
    def config(self):
        return _FakeBackend.last["config"]


def _run(body, members):
    _FakeBackend.last = {}  # reset cross-test state
    config = NodeConfig(node_id="head")
    config.save = lambda *a, **k: None  # no disk writes
    cluster = ClusterState(local_announcement=_local_ann("head"))
    for m in members:
        cluster.add_node(m)
    app = {"cluster_state": cluster, "config": config, "engine": None}

    class _Req:
        def __init__(self):
            self.app = app
        async def json(self):
            return body

    with patch.object(backends, "get_backend", _FakeBackend):
        resp = asyncio.run(handle_sharding_launch(_Req()))
    return config, resp


def test_count_mode_uses_fabric_not_mgmt():
    # min_nodes=2 → 1 peer; it must be the member's FABRIC ip, not its mgmt peer_ip.
    config, resp = _run(
        {"model": "nvidia/Llama-3.3-70B-Instruct-NVFP4", "min_nodes": 2},
        [_member("m1", fabric_ip="10.100.0.13", peer_ip="192.168.0.13")],
    )
    assert resp.status == 200
    assert config.peer_ips == ["10.100.0.13"]
    assert _FakeBackend.last["launched"] is True


def test_explicit_node_ids_selects_exactly_those():
    # Choose head + m2 → peer_ips is exactly m2's fabric IP (not m1's).
    config, resp = _run(
        {"model": "m", "node_ids": ["head", "m2"]},
        [_member("m1", "10.100.0.13"), _member("m2", "10.100.0.15"), _member("m3", "10.100.0.17")],
    )
    assert resp.status == 200
    assert config.peer_ips == ["10.100.0.15"]


def test_missing_fabric_ip_is_rejected():
    # A selected node with no fabric IP must 422, not silently fall back to mgmt.
    config, resp = _run(
        {"model": "m", "node_ids": ["head", "m1"]},
        [_member("m1", fabric_ip="", peer_ip="192.168.0.13")],
    )
    assert resp.status == 422
    assert _FakeBackend.last.get("launched") is not True


def test_unknown_node_id_is_rejected():
    config, resp = _run(
        {"model": "m", "node_ids": ["head", "ghost"]},
        [_member("m1", "10.100.0.13")],
    )
    assert resp.status == 422


def test_proxy_returns_503_loading_during_swap():
    """While the engine is mid-swap (not ready), :3000 returns a clear loading
    state with the phase — not a hang/opaque proxy error."""
    import json as _json
    from ainode.api.server import proxy_to_vllm

    class _Engine:
        ready = False
        load_phase = "loading_weights"

    config = NodeConfig(node_id="head")
    config.model = "nvidia/Llama-3.3-70B-Instruct-NVFP4"
    app = {"config": config, "client_session": None, "metrics_collector": None, "engine": _Engine()}

    class _Req:
        method = "GET"
        path = "/v1/models"
        headers = {}
        def __init__(self):
            self.app = app

    resp = asyncio.run(proxy_to_vllm(_Req()))
    assert resp.status == 503
    data = _json.loads(resp.body.decode())
    assert data["error"]["load_phase"] == "loading_weights"
    assert resp.headers.get("Retry-After") == "10"


def test_distributed_instance_resolves_peers_and_model():
    """cluster/resources must report the running model (not stale "") and resolve
    fabric-IP peers back to member nodes so the UI shows DISTRIBUTED, not SINGLE."""
    import json as _json
    from ainode.api.server import handle_cluster_resources
    from ainode.engine.ray_autostart import RayAutostartState

    head = ClusterNode(
        node_id="headid", node_name="Spark-1-DGX", gpu_name="NVIDIA GB10",
        gpu_memory_gb=128.0, unified_memory=True, model="",  # stale (idle-start)
        status=NodeStatus.ONLINE, api_port=8000, web_port=3000, last_seen=0.0,
        distributed_mode="head",
        distributed_instance_id="headid:nvidia/Llama-3.3-70B-Instruct-NVFP4",
        distributed_peers=["10.100.0.13"],  # FABRIC IP
    )
    member = _member("memberid", fabric_ip="10.100.0.13")
    cluster = ClusterState()
    cluster.add_node(head)
    cluster.add_node(member)
    app = {"cluster_state": cluster,
           "ray_autostart_state": RayAutostartState(is_head=True, head_address="x:6379")}

    class _Req:
        def __init__(self):
            self.app = app

    resp = asyncio.run(handle_cluster_resources(_Req()))
    di = _json.loads(resp.body.decode())["distributed_instance"]
    assert di["model"] == "nvidia/Llama-3.3-70B-Instruct-NVFP4"  # from iid, not ""
    assert "memberid" in di["peer_node_ids"]                      # fabric IP → node_id
    assert di["tensor_parallel_size"] == 2
    assert "Spark-1-DGX" in di["member_names"] and "host-memberid" in di["member_names"]
