"""Tests for cluster state, leader election, model routing, and summary."""

import time
import pytest

from ainode.discovery.broadcast import (
    NodeAnnouncement,
    NodeStatus,
    DiscoveredNode,
)
from ainode.discovery.cluster import ClusterState, ClusterNode


def _make_announcement(**overrides) -> NodeAnnouncement:
    defaults = dict(
        node_id="node-aaa",
        node_name="spark-1",
        gpu_name="GB10",
        gpu_memory_gb=128.0,
        unified_memory=True,
        model="meta-llama/Llama-3.2-3B-Instruct",
        status="serving",
        api_port=8000,
        web_port=3000,
        timestamp=time.time(),
    )
    defaults.update(overrides)
    return NodeAnnouncement(**defaults)


def _make_cluster_node(**overrides) -> ClusterNode:
    defaults = dict(
        node_id="node-aaa",
        node_name="spark-1",
        gpu_name="GB10",
        gpu_memory_gb=128.0,
        unified_memory=True,
        model="meta-llama/Llama-3.2-3B-Instruct",
        status=NodeStatus.ONLINE,
        api_port=8000,
        web_port=3000,
        last_seen=time.time(),
    )
    defaults.update(overrides)
    return ClusterNode(**defaults)


class TestClusterState:
    def test_local_node_added_on_init(self):
        ann = _make_announcement(node_id="local-1")
        cluster = ClusterState(local_announcement=ann)
        assert cluster.get_node("local-1") is not None

    def test_empty_cluster(self):
        cluster = ClusterState()
        assert cluster.get_nodes() == []
        assert cluster.get_leader() is None

    def test_add_and_get_node(self):
        cluster = ClusterState()
        node = _make_cluster_node(node_id="n1")
        cluster.add_node(node)
        assert cluster.get_node("n1") is not None
        assert cluster.get_node("n1").gpu_memory_gb == 128.0

    def test_remove_node(self):
        cluster = ClusterState()
        cluster.add_node(_make_cluster_node(node_id="n1"))
        cluster.remove_node("n1")
        assert cluster.get_node("n1") is None

    def test_remove_nonexistent_node_no_error(self):
        cluster = ClusterState()
        cluster.remove_node("nope")

    def test_get_nodes_excludes_offline(self):
        cluster = ClusterState()
        cluster.add_node(_make_cluster_node(node_id="online", status=NodeStatus.ONLINE))
        cluster.add_node(_make_cluster_node(node_id="offline", status=NodeStatus.OFFLINE))
        nodes = cluster.get_nodes(include_offline=False)
        assert len(nodes) == 1
        assert nodes[0].node_id == "online"

    def test_get_nodes_includes_offline(self):
        cluster = ClusterState()
        cluster.add_node(_make_cluster_node(node_id="online", status=NodeStatus.ONLINE))
        cluster.add_node(_make_cluster_node(node_id="offline", status=NodeStatus.OFFLINE))
        assert len(cluster.get_nodes(include_offline=True)) == 2


class TestLeaderElection:
    def test_single_node_is_leader(self):
        cluster = ClusterState()
        cluster.add_node(_make_cluster_node(node_id="node-aaa"))
        leader = cluster.get_leader()
        assert leader is not None
        assert leader.node_id == "node-aaa"

    def test_lowest_id_wins(self):
        cluster = ClusterState()
        cluster.add_node(_make_cluster_node(node_id="node-ccc"))
        cluster.add_node(_make_cluster_node(node_id="node-aaa"))
        cluster.add_node(_make_cluster_node(node_id="node-bbb"))
        leader = cluster.get_leader()
        assert leader.node_id == "node-aaa"

    def test_offline_nodes_excluded_from_election(self):
        cluster = ClusterState()
        cluster.add_node(_make_cluster_node(node_id="node-aaa", status=NodeStatus.OFFLINE))
        cluster.add_node(_make_cluster_node(node_id="node-bbb", status=NodeStatus.ONLINE))
        leader = cluster.get_leader()
        assert leader.node_id == "node-bbb"

    def test_stale_nodes_excluded_from_election(self):
        cluster = ClusterState()
        cluster.add_node(_make_cluster_node(node_id="node-aaa", status=NodeStatus.STALE))
        cluster.add_node(_make_cluster_node(node_id="node-bbb", status=NodeStatus.ONLINE))
        leader = cluster.get_leader()
        assert leader.node_id == "node-bbb"

    def test_no_leader_when_all_offline(self):
        cluster = ClusterState()
        cluster.add_node(_make_cluster_node(node_id="node-aaa", status=NodeStatus.OFFLINE))
        assert cluster.get_leader() is None

    def test_leader_changes_when_node_goes_offline(self):
        cluster = ClusterState()
        cluster.add_node(_make_cluster_node(node_id="node-aaa", status=NodeStatus.ONLINE))
        cluster.add_node(_make_cluster_node(node_id="node-bbb", status=NodeStatus.ONLINE))
        assert cluster.get_leader().node_id == "node-aaa"
        cluster.add_node(_make_cluster_node(node_id="node-aaa", status=NodeStatus.OFFLINE))
        assert cluster.get_leader().node_id == "node-bbb"


class TestModelRouting:
    def test_find_model_single_match(self):
        cluster = ClusterState()
        cluster.add_node(_make_cluster_node(node_id="n1", model="llama-3"))
        cluster.add_node(_make_cluster_node(node_id="n2", model="mistral-7b"))
        results = cluster.find_model("llama-3")
        assert len(results) == 1
        assert results[0].node_id == "n1"

    def test_find_model_multiple_matches(self):
        cluster = ClusterState()
        cluster.add_node(_make_cluster_node(node_id="n1", model="llama-3"))
        cluster.add_node(_make_cluster_node(node_id="n2", model="llama-3"))
        results = cluster.find_model("llama-3")
        assert len(results) == 2

    def test_find_model_no_match(self):
        cluster = ClusterState()
        cluster.add_node(_make_cluster_node(node_id="n1", model="llama-3"))
        assert cluster.find_model("gpt-4") == []

    def test_find_model_excludes_offline(self):
        cluster = ClusterState()
        cluster.add_node(_make_cluster_node(node_id="n1", model="llama-3", status=NodeStatus.ONLINE))
        cluster.add_node(_make_cluster_node(node_id="n2", model="llama-3", status=NodeStatus.OFFLINE))
        results = cluster.find_model("llama-3")
        assert len(results) == 1
        assert results[0].node_id == "n1"

    def test_find_model_includes_stale(self):
        cluster = ClusterState()
        cluster.add_node(_make_cluster_node(node_id="n1", model="llama-3", status=NodeStatus.STALE))
        results = cluster.find_model("llama-3")
        assert len(results) == 1


class TestClusterSummary:
    def test_summary_two_sparks(self):
        cluster = ClusterState()
        cluster.add_node(_make_cluster_node(
            node_id="spark-1", gpu_name="GB10", gpu_memory_gb=128.0,
            model="meta-llama/Llama-3.2-70B-Instruct",
        ))
        cluster.add_node(_make_cluster_node(
            node_id="spark-2", gpu_name="GB10", gpu_memory_gb=128.0,
            model="meta-llama/Llama-3.2-3B-Instruct",
        ))
        summary = cluster.cluster_summary()
        assert summary["total_nodes"] == 2
        assert summary["total_gpus"] == 2
        assert summary["total_memory_gb"] == 256.0
        assert len(summary["models_available"]) == 2
        assert summary["leader_id"] == "spark-1"

    def test_summary_excludes_offline(self):
        cluster = ClusterState()
        cluster.add_node(_make_cluster_node(node_id="n1", gpu_memory_gb=128.0))
        cluster.add_node(_make_cluster_node(node_id="n2", gpu_memory_gb=128.0, status=NodeStatus.OFFLINE))
        summary = cluster.cluster_summary()
        assert summary["total_nodes"] == 1
        assert summary["total_memory_gb"] == 128.0

    def test_summary_empty_cluster(self):
        cluster = ClusterState()
        summary = cluster.cluster_summary()
        assert summary["total_nodes"] == 0
        assert summary["total_gpus"] == 0
        assert summary["total_memory_gb"] == 0
        assert summary["models_available"] == []
        assert summary["leader_id"] is None

    def test_summary_models_sorted(self):
        cluster = ClusterState()
        cluster.add_node(_make_cluster_node(node_id="n1", model="zebra-model"))
        cluster.add_node(_make_cluster_node(node_id="n2", model="alpha-model"))
        summary = cluster.cluster_summary()
        assert summary["models_available"] == ["alpha-model", "zebra-model"]

    def test_update_from_discovered(self):
        local_ann = _make_announcement(node_id="local-1")
        cluster = ClusterState(local_announcement=local_ann)
        discovered = {
            "remote-1": DiscoveredNode(
                announcement=_make_announcement(node_id="remote-1", model="mistral"),
                last_seen=time.time(),
            ),
        }
        cluster.update_from_discovered(discovered)
        assert cluster.get_node("local-1") is not None
        assert cluster.get_node("remote-1") is not None
        assert cluster.get_node("remote-1").model == "mistral"

    def test_cluster_node_from_discovered(self):
        ann = _make_announcement(node_id="test-node")
        discovered = DiscoveredNode(announcement=ann, last_seen=time.time())
        cn = ClusterNode.from_discovered(discovered)
        assert cn.node_id == "test-node"
        assert cn.status == NodeStatus.ONLINE
