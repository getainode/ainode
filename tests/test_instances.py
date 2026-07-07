"""P2-1: InstanceRecord + the `instances` list through broadcast/cluster/cluster-resources."""

from __future__ import annotations

import asyncio
import json

from ainode.api.server import handle_cluster_resources
from ainode.discovery.broadcast import NodeAnnouncement, NodeStatus
from ainode.discovery.cluster import ClusterNode, ClusterState
from ainode.discovery.instance import InstanceRecord
from ainode.engine.ray_autostart import RayAutostartState


def _node(node_id, name=None, fabric_ip="", instances=None, distributed_mode="solo"):
    return ClusterNode(
        node_id=node_id, node_name=name or node_id, gpu_name="NVIDIA GB10",
        gpu_memory_gb=128.0, unified_memory=True, model="", status=NodeStatus.ONLINE,
        api_port=8000, web_port=3000, last_seen=0.0,
        distributed_mode=distributed_mode, fabric_ip=fabric_ip, instances=instances or [],
    )


def _resources(nodes):
    cluster = ClusterState()
    for n in nodes:
        cluster.add_node(n)
    app = {"cluster_state": cluster,
           "ray_autostart_state": RayAutostartState(is_head=True, head_address="x:6379")}

    class _Req:
        def __init__(self):
            self.app = app

    resp = asyncio.run(handle_cluster_resources(_Req()))
    return json.loads(resp.body.decode())


def test_instance_record_roundtrip():
    r = InstanceRecord(instance_id="h:m", model="m", head_node_id="h",
                       peer_ips=["10.0.0.2"], api_port=8001, tensor_parallel_size=2)
    d = r.to_dict()
    assert d["api_port"] == 8001 and d["peer_ips"] == ["10.0.0.2"]
    # from_dict ignores unknown keys (forward-compatible wire parsing)
    assert InstanceRecord.from_dict({**d, "future_field": 1}) == r


def test_cluster_resources_zero_instances():
    out = _resources([_node("solo1")])
    assert out["distributed_instances"] == []
    assert out["distributed_instance"] is None


def test_cluster_resources_one_instance_resolves_members():
    inst = InstanceRecord(instance_id="h:nvidia/Llama-70B", model="nvidia/Llama-70B",
                          head_node_id="h", peer_ips=["10.100.0.13"],
                          api_port=8000, tensor_parallel_size=2).to_dict()
    head = _node("h", "Spark-1", fabric_ip="10.100.0.11", instances=[inst], distributed_mode="head")
    member = _node("m2", "Spark-2", fabric_ip="10.100.0.13", distributed_mode="member")
    out = _resources([head, member])
    di = out["distributed_instances"]
    assert len(di) == 1
    assert di[0]["model"] == "nvidia/Llama-70B"
    assert di[0]["peer_node_ids"] == ["m2"]              # fabric IP resolved to node_id
    assert "Spark-2" in di[0]["member_names"]
    assert di[0]["tensor_parallel_size"] == 2
    assert out["distributed_instance"] == di[0]          # singular back-compat = first


def test_cluster_resources_two_disjoint_instances():
    i1 = InstanceRecord(instance_id="h1:A", model="A", head_node_id="h1", peer_ips=["10.0.0.2"]).to_dict()
    i2 = InstanceRecord(instance_id="h3:B", model="B", head_node_id="h3", peer_ips=["10.0.0.4"]).to_dict()
    out = _resources([
        _node("h1", fabric_ip="10.0.0.1", instances=[i1], distributed_mode="head"),
        _node("m2", fabric_ip="10.0.0.2", distributed_mode="member"),
        _node("h3", fabric_ip="10.0.0.3", instances=[i2], distributed_mode="head"),
        _node("m4", fabric_ip="10.0.0.4", distributed_mode="member"),
    ])
    assert sorted(x["model"] for x in out["distributed_instances"]) == ["A", "B"]


def test_cluster_resources_backcompat_singular_fields():
    """A node advertising only the legacy distributed_instance_id still surfaces."""
    n = _node("h", "Spark-1", fabric_ip="10.100.0.11", distributed_mode="head")
    n.distributed_instance_id = "h:nvidia/Old"
    n.distributed_peers = ["10.100.0.13"]
    member = _node("m2", "Spark-2", fabric_ip="10.100.0.13", distributed_mode="member")
    out = _resources([n, member])
    assert len(out["distributed_instances"]) == 1
    assert out["distributed_instances"][0]["model"] == "nvidia/Old"


def test_announcement_carries_instances_roundtrip():
    inst = InstanceRecord(instance_id="h:m", model="m", head_node_id="h", peer_ips=["10.0.0.2"]).to_dict()
    a = NodeAnnouncement(node_id="h", node_name="h", gpu_name="NVIDIA GB10", gpu_memory_gb=128.0,
                         unified_memory=True, model="m", status="serving",
                         api_port=8000, web_port=3000, instances=[inst])
    assert NodeAnnouncement.from_json(a.to_json()).instances == [inst]


# ---------------------------------------------------------------------------
# F3: instance status flips 'starting' -> 'serving' once the engine answers,
# and the flipped status rides the announcement (via _live_instance_records).
# ---------------------------------------------------------------------------

class _FakeBackend:
    def __init__(self, responding: bool):
        self._responding = responding

    def health_check(self) -> dict:
        return {"api_responding": self._responding}


def test_live_instance_records_flips_starting_to_serving():
    from ainode.api.server import _live_instance_records
    from ainode.engine.instance_manager import InstanceManager

    manager = InstanceManager(base_port=8000)
    # A stacked instance stamped 'starting' at load time whose engine now answers.
    rec = InstanceRecord(instance_id="h:qwen", model="qwen", head_node_id="h",
                         api_port=8001, status="starting")
    manager.add(rec, _FakeBackend(responding=True))

    loop = asyncio.new_event_loop()
    try:
        live = loop.run_until_complete(_live_instance_records(manager, loop))
    finally:
        loop.close()

    # The record object itself is mutated (so /api/server/status + the
    # announcement both see the truth) and it is advertised as serving.
    assert rec.status == "serving"
    assert [r.to_dict()["status"] for r in live] == ["serving"]


def test_live_instance_records_drops_dead_instance():
    from ainode.api.server import _live_instance_records
    from ainode.engine.instance_manager import InstanceManager

    manager = InstanceManager(base_port=8000)
    rec = InstanceRecord(instance_id="h:dead", model="dead", head_node_id="h",
                         api_port=8001, status="starting")
    manager.add(rec, _FakeBackend(responding=False))

    loop = asyncio.new_event_loop()
    try:
        live = loop.run_until_complete(_live_instance_records(manager, loop))
    finally:
        loop.close()

    # A non-answering engine is neither advertised nor promoted to serving.
    assert live == []
    assert rec.status == "starting"
