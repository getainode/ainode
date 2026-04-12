"""Tests for UDP broadcast discovery -- announcement serialization, registry, health."""

import time
import json
import pytest
from unittest.mock import MagicMock

from ainode.discovery.broadcast import (
    NodeAnnouncement,
    NodeStatus,
    DiscoveredNode,
    BroadcastSender,
    BroadcastListener,
    ONLINE_THRESHOLD,
    STALE_THRESHOLD,
)


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
        timestamp=1700000000.0,
    )
    defaults.update(overrides)
    return NodeAnnouncement(**defaults)


class TestNodeAnnouncement:
    def test_to_json_roundtrip(self):
        ann = _make_announcement()
        raw = ann.to_json()
        restored = NodeAnnouncement.from_json(raw)
        assert restored.node_id == ann.node_id
        assert restored.gpu_memory_gb == 128.0
        assert restored.unified_memory is True
        assert restored.model == ann.model

    def test_json_contains_all_fields(self):
        ann = _make_announcement()
        data = json.loads(ann.to_json())
        expected_keys = {
            "node_id", "node_name", "gpu_name", "gpu_memory_gb",
            "unified_memory", "model", "status", "api_port", "web_port", "timestamp",
        }
        assert set(data.keys()) == expected_keys

    def test_default_timestamp(self):
        ann = NodeAnnouncement(
            node_id="x", node_name="x", gpu_name="x",
            gpu_memory_gb=0, unified_memory=False, model="",
            status="idle", api_port=8000, web_port=3000,
        )
        assert abs(ann.timestamp - time.time()) < 2


class TestDiscoveredNodeHealth:
    def test_online_when_fresh(self):
        node = DiscoveredNode(announcement=_make_announcement(), last_seen=time.time())
        assert node.health == NodeStatus.ONLINE

    def test_stale_after_threshold(self):
        node = DiscoveredNode(
            announcement=_make_announcement(),
            last_seen=time.time() - (ONLINE_THRESHOLD + 1),
        )
        assert node.health == NodeStatus.STALE

    def test_offline_after_threshold(self):
        node = DiscoveredNode(
            announcement=_make_announcement(),
            last_seen=time.time() - (STALE_THRESHOLD + 1),
        )
        assert node.health == NodeStatus.OFFLINE

    def test_age_property(self):
        now = time.time()
        node = DiscoveredNode(announcement=_make_announcement(), last_seen=now - 10)
        assert abs(node.age - 10) < 1

    def test_health_transition_online_to_stale(self):
        node = DiscoveredNode(announcement=_make_announcement(), last_seen=time.time())
        assert node.health == NodeStatus.ONLINE
        node.last_seen = time.time() - (ONLINE_THRESHOLD + 1)
        assert node.health == NodeStatus.STALE

    def test_health_transition_stale_to_offline(self):
        node = DiscoveredNode(
            announcement=_make_announcement(),
            last_seen=time.time() - (ONLINE_THRESHOLD + 1),
        )
        assert node.health == NodeStatus.STALE
        node.last_seen = time.time() - (STALE_THRESHOLD + 1)
        assert node.health == NodeStatus.OFFLINE

    def test_health_recovery(self):
        node = DiscoveredNode(
            announcement=_make_announcement(),
            last_seen=time.time() - (ONLINE_THRESHOLD + 1),
        )
        assert node.health == NodeStatus.STALE
        node.last_seen = time.time()
        assert node.health == NodeStatus.ONLINE


class TestBroadcastListenerRegistry:
    def test_process_announcement_adds_node(self):
        listener = BroadcastListener(local_node_id="local-node")
        ann = _make_announcement(node_id="remote-1")
        listener._process_announcement(ann)
        assert "remote-1" in listener._registry
        assert listener._registry["remote-1"].announcement.node_id == "remote-1"

    def test_ignores_own_announcements(self):
        listener = BroadcastListener(local_node_id="local-node")
        ann = _make_announcement(node_id="local-node")
        listener._process_announcement(ann)
        assert "local-node" not in listener._registry

    def test_updates_existing_node(self):
        listener = BroadcastListener(local_node_id="local-node")
        ann1 = _make_announcement(node_id="remote-1", model="model-a")
        listener._process_announcement(ann1)
        ann2 = _make_announcement(node_id="remote-1", model="model-b")
        listener._process_announcement(ann2)
        assert listener._registry["remote-1"].announcement.model == "model-b"

    def test_on_node_found_callback(self):
        callback = MagicMock()
        listener = BroadcastListener(local_node_id="local-node", on_node_found=callback)
        ann = _make_announcement(node_id="remote-1")
        listener._process_announcement(ann)
        callback.assert_called_once_with(ann)

    def test_on_node_found_not_called_on_update(self):
        callback = MagicMock()
        listener = BroadcastListener(local_node_id="local-node", on_node_found=callback)
        ann = _make_announcement(node_id="remote-1")
        listener._process_announcement(ann)
        listener._process_announcement(ann)
        callback.assert_called_once()

    def test_get_nodes_excludes_offline(self):
        listener = BroadcastListener(local_node_id="local-node")
        listener._process_announcement(_make_announcement(node_id="online-node"))
        listener._process_announcement(_make_announcement(node_id="old-node"))
        listener._registry["old-node"].last_seen = time.time() - (STALE_THRESHOLD + 1)
        nodes = listener.get_nodes(include_offline=False)
        assert len(nodes) == 1
        assert nodes[0].announcement.node_id == "online-node"

    def test_get_nodes_includes_offline(self):
        listener = BroadcastListener(local_node_id="local-node")
        listener._process_announcement(_make_announcement(node_id="online-node"))
        listener._process_announcement(_make_announcement(node_id="old-node"))
        listener._registry["old-node"].last_seen = time.time() - (STALE_THRESHOLD + 1)
        nodes = listener.get_nodes(include_offline=True)
        assert len(nodes) == 2

    def test_get_node(self):
        listener = BroadcastListener(local_node_id="local-node")
        ann = _make_announcement(node_id="remote-1")
        listener._process_announcement(ann)
        node = listener.get_node("remote-1")
        assert node is not None
        assert node.announcement.node_name == "spark-1"
        assert listener.get_node("nonexistent") is None


class TestBroadcastSenderAnnouncement:
    """Tests for BroadcastSender announcement creation and updates."""

    def test_sender_holds_announcement(self):
        ann = _make_announcement(node_id="sender-1", model="llama-3")
        sender = BroadcastSender(announcement=ann)
        assert sender.announcement.node_id == "sender-1"
        assert sender.announcement.model == "llama-3"

    def test_sender_update_announcement(self):
        ann = _make_announcement(status="starting")
        sender = BroadcastSender(announcement=ann)
        sender.update_announcement(status="serving", model="new-model")
        assert sender.announcement.status == "serving"
        assert sender.announcement.model == "new-model"

    def test_sender_ignores_unknown_fields(self):
        ann = _make_announcement()
        sender = BroadcastSender(announcement=ann)
        sender.update_announcement(nonexistent_field="value")
        assert not hasattr(sender.announcement, "nonexistent_field")


class TestListenerUpdatesClusterState:
    """Tests that BroadcastListener feeds into ClusterState correctly."""

    def test_listener_feeds_cluster_state(self):
        from ainode.discovery.cluster import ClusterState
        local_ann = _make_announcement(node_id="local-1")
        cluster = ClusterState(local_announcement=local_ann)
        listener = BroadcastListener(local_node_id="local-1")

        # Simulate receiving a remote announcement
        remote_ann = _make_announcement(node_id="remote-1", node_name="spark-2")
        listener._process_announcement(remote_ann)

        # Sync into cluster
        cluster.update_from_discovered(listener.registry)

        nodes = cluster.get_nodes()
        node_ids = [n.node_id for n in nodes]
        assert "local-1" in node_ids
        assert "remote-1" in node_ids
        assert len(nodes) == 2
