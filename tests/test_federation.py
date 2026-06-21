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
