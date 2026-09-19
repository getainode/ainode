"""F1: federated master router — route-by-model table + /v1/models union."""

from __future__ import annotations

import asyncio
import json

from ainode.api.server import _routing_table, handle_v1_models
from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeStatus
from ainode.discovery.cluster import ClusterNode, ClusterState


def _node(nid, model="", fabric="", instances=None, status=NodeStatus.ONLINE, api_port=8000):
    return ClusterNode(node_id=nid, node_name=nid, gpu_name="NVIDIA GB10", gpu_memory_gb=128.0,
                       unified_memory=True, model=model, status=status, api_port=api_port,
                       web_port=3000, last_seen=0.0, fabric_ip=fabric, instances=instances or [])


def _cluster(nodes):
    c = ClusterState()
    for n in nodes:
        c.add_node(n)
    return c


class _Req:
    def __init__(self, app):
        self.app = app


def _cfg(nid, model=""):
    return NodeConfig(node_id=nid, api_port=8000, model=model)


def test_routing_table_local_and_remote():
    c = _cluster([_node("spark1", model="A"), _node("spark2", model="B", fabric="10.100.0.13")])
    t = _routing_table(c, "spark1", 8000)
    assert t["A"] == ("localhost", 8000)            # local → localhost
    assert t["B"] == ("10.100.0.13", 8000)          # remote → fabric IP


def test_routing_table_skips_offline_and_no_fabric():
    c = _cluster([
        _node("spark2", model="B", fabric="10.100.0.13", status=NodeStatus.OFFLINE),
        _node("spark3", model="C", fabric=""),       # remote with no fabric IP → unroutable
    ])
    t = _routing_table(c, "spark1", 8000)
    assert "B" not in t and "C" not in t


def test_routing_table_includes_instances():
    inst = {"model": "BigMoE", "api_port": 8001}
    c = _cluster([_node("spark1", instances=[inst])])
    t = _routing_table(c, "spark1", 8000)
    assert t["BigMoE"] == ("localhost", 8001)        # stacked/distributed instance, own port


def test_v1_models_unions_the_fleet():
    c = _cluster([_node("spark1", model="A"), _node("spark2", model="B", fabric="10.100.0.13")])
    out = asyncio.run(handle_v1_models(_Req({"config": _cfg("spark1"), "cluster_state": c})))
    ids = [m["id"] for m in json.loads(out.body)["data"]]
    assert ids == ["A", "B"]


def test_v1_models_local_fallback_when_no_cluster_entry():
    # empty cluster table but a local model is configured → still listed
    out = asyncio.run(handle_v1_models(_Req({"config": _cfg("spark1", model="Local"),
                                             "cluster_state": _cluster([])})))
    ids = [m["id"] for m in json.loads(out.body)["data"]]
    assert ids == ["Local"]


# ------------- #162: the head is an instance, even with a stack beside it ----
#
# The announcement's `instances` list was EITHER the InstanceManager's records OR a
# head synthesised from config, never both. So loading a second model on a node that
# heads a distributed launch made the head disappear from the broadcast: the master
# drew the peer as an empty node and the head instance as SINGLE, and its fleet view
# lost a live TP=2 engine. These pin the merge.

def _head_config(model="fraserprice/DeepSeek-V4-Flash-DSpark",
                 peers=("10.100.0.13",)):
    from ainode.core.config import NodeConfig

    cfg = NodeConfig(node_id="spark2", node_name="Spark-2-DGX", api_port=8000)
    cfg.model = model
    cfg.distributed_mode = "head"
    cfg.peer_ips = list(peers)
    cfg.distributed_executor = "mp"
    return cfg


def _stacked_record(model="Qwen/Qwen3-Embedding-0.6B", port=8001):
    from ainode.discovery.instance import InstanceRecord

    return InstanceRecord(instance_id=f"spark2:{model}", model=model,
                          head_node_id="spark2", peer_ips=[], api_port=port,
                          tensor_parallel_size=1, status="serving")


def test_a_head_with_no_manager_records_still_announces_itself():
    """The shape after a restart: the head container serves, the manager is empty
    because a distributed launch is never written to instances.json."""
    import ainode.api.server as server

    out = server.announced_instances(_head_config(), [], "head", True)
    assert [(i["model"], i["api_port"], i["tensor_parallel_size"]) for i in out] == [
        ("fraserprice/DeepSeek-V4-Flash-DSpark", 8000, 2)]
    assert out[0]["peer_ips"] == ["10.100.0.13"]


def test_a_head_and_a_stacked_instance_are_both_announced():
    """The bug: the stacked record used to REPLACE the head instead of joining it."""
    import ainode.api.server as server

    out = server.announced_instances(_head_config(), [_stacked_record()], "head", True)
    by_port = {i["api_port"]: i for i in out}
    assert set(by_port) == {8000, 8001}
    assert by_port[8000]["model"] == "fraserprice/DeepSeek-V4-Flash-DSpark"
    assert by_port[8000]["tensor_parallel_size"] == 2
    assert by_port[8000]["peer_ips"] == ["10.100.0.13"]
    assert by_port[8001]["model"] == "Qwen/Qwen3-Embedding-0.6B"
    assert by_port[8001]["tensor_parallel_size"] == 1


def test_the_managers_own_head_record_is_not_doubled():
    """Launched in this process, the head is in the manager too. One engine per port,
    so the manager's record wins: it carries the launch's real peer set."""
    import ainode.api.server as server
    from ainode.discovery.instance import InstanceRecord

    head = InstanceRecord(instance_id="spark2:head",
                          model="fraserprice/DeepSeek-V4-Flash-DSpark",
                          head_node_id="spark2", peer_ips=["10.100.0.13"],
                          api_port=8000, tensor_parallel_size=2, status="serving",
                          distributed_executor="mp")
    out = server.announced_instances(_head_config(), [head, _stacked_record()],
                                     "head", True)
    assert len(out) == 2
    assert [i["api_port"] for i in out] == [8000, 8001]
    assert sum(1 for i in out if i["api_port"] == 8000) == 1


def test_a_head_whose_engine_is_not_serving_announces_only_its_stack():
    """Same liveness rule as everything else on the wire: an engine that does not
    answer is not advertised, or the master routes to a ghost."""
    import ainode.api.server as server

    out = server.announced_instances(_head_config(), [_stacked_record()],
                                     "head", False)
    assert [i["api_port"] for i in out] == [8001]


def test_a_solo_node_announces_exactly_its_manager_records():
    """Nothing changes for a node that heads nothing, which is most of them."""
    import ainode.api.server as server

    cfg = _head_config()
    cfg.distributed_mode = "solo"
    cfg.peer_ips = []
    out = server.announced_instances(cfg, [_stacked_record(port=8000),
                                           _stacked_record("X", port=8001)],
                                    "solo", True)
    assert [i["api_port"] for i in out] == [8000, 8001]


def test_a_head_with_no_peers_announces_no_head_instance():
    """`_head_instances` returns nothing without peers, and a TP=1 "head" is a solo
    engine already covered by the node's own `model` field."""
    import ainode.api.server as server

    out = server.announced_instances(_head_config(peers=()), [], "head", True)
    assert out == []


def test_the_master_routes_to_an_announced_head_and_its_stack():
    """The point of announcing it: both models on that node are reachable by id from
    the master, the head on :8000 and the stacked one on :8001."""
    import ainode.api.server as server

    instances = server.announced_instances(_head_config(), [_stacked_record()],
                                           "head", True)
    cluster = _cluster([_node("spark2", model="", fabric="10.100.0.12",
                              instances=instances)])
    assert server._routing_candidates(cluster, "fraserprice/DeepSeek-V4-Flash-DSpark",
                                      "spark1", 3000) == [("10.100.0.12", 8000)]
    assert server._routing_candidates(cluster, "Qwen/Qwen3-Embedding-0.6B",
                                      "spark1", 3000) == [("10.100.0.12", 8001)]
