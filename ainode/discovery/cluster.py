"""Cluster coordinator -- tracks nodes, elects leader, routes model requests."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ainode.discovery.broadcast import (
    DiscoveredNode,
    NodeAnnouncement,
    NodeStatus,
)


@dataclass
class ClusterNode:
    """Enriched view of a node within the cluster."""
    node_id: str
    node_name: str
    gpu_name: str
    gpu_memory_gb: float
    unified_memory: bool
    model: str
    status: NodeStatus
    api_port: int
    web_port: int
    last_seen: float

    @classmethod
    def from_discovered(cls, discovered: DiscoveredNode) -> "ClusterNode":
        a = discovered.announcement
        return cls(
            node_id=a.node_id,
            node_name=a.node_name,
            gpu_name=a.gpu_name,
            gpu_memory_gb=a.gpu_memory_gb,
            unified_memory=a.unified_memory,
            model=a.model,
            status=discovered.health,
            api_port=a.api_port,
            web_port=a.web_port,
            last_seen=discovered.last_seen,
        )

    @classmethod
    def from_announcement(cls, announcement: NodeAnnouncement, status: NodeStatus) -> "ClusterNode":
        return cls(
            node_id=announcement.node_id,
            node_name=announcement.node_name,
            gpu_name=announcement.gpu_name,
            gpu_memory_gb=announcement.gpu_memory_gb,
            unified_memory=announcement.unified_memory,
            model=announcement.model,
            status=status,
            api_port=announcement.api_port,
            web_port=announcement.web_port,
            last_seen=time.time(),
        )


class ClusterState:
    """Maintains cluster state: all nodes, their GPUs, loaded models, and leader election."""

    def __init__(self, local_announcement: Optional[NodeAnnouncement] = None):
        self._nodes: Dict[str, ClusterNode] = {}
        self._local_announcement = local_announcement

        # If we have a local announcement, add ourselves
        if local_announcement:
            self._nodes[local_announcement.node_id] = ClusterNode.from_announcement(
                local_announcement, NodeStatus.ONLINE
            )

    def update_from_discovered(self, discovered_nodes: Dict[str, DiscoveredNode]):
        """Sync cluster state from the listener registry."""
        local_id = self._local_announcement.node_id if self._local_announcement else None

        # Update remote nodes
        for node_id, discovered in discovered_nodes.items():
            self._nodes[node_id] = ClusterNode.from_discovered(discovered)

        # Remove nodes no longer in the registry (except local)
        remote_ids = set(discovered_nodes.keys())
        to_remove = [
            nid for nid in self._nodes
            if nid != local_id and nid not in remote_ids
        ]
        for nid in to_remove:
            del self._nodes[nid]

        # Refresh local node timestamp
        if local_id and self._local_announcement:
            self._nodes[local_id] = ClusterNode.from_announcement(
                self._local_announcement, NodeStatus.ONLINE
            )

    def add_node(self, node: ClusterNode):
        """Add or update a node directly."""
        self._nodes[node.node_id] = node

    def remove_node(self, node_id: str):
        """Remove a node from the cluster."""
        self._nodes.pop(node_id, None)

    def get_node(self, node_id: str) -> Optional[ClusterNode]:
        """Get info for a specific node."""
        return self._nodes.get(node_id)

    def get_nodes(self, include_offline: bool = False) -> List[ClusterNode]:
        """Get all nodes with their status."""
        nodes = list(self._nodes.values())
        if not include_offline:
            nodes = [n for n in nodes if n.status != NodeStatus.OFFLINE]
        return nodes

    def get_leader(self) -> Optional[ClusterNode]:
        """Return the current leader node. Leader = lowest node_id alphabetically among ONLINE nodes."""
        online = [n for n in self._nodes.values() if n.status == NodeStatus.ONLINE]
        if not online:
            return None
        return min(online, key=lambda n: n.node_id)

    def find_model(self, model_name: str) -> List[ClusterNode]:
        """Return which node(s) serve a given model. Only returns ONLINE or STALE nodes."""
        return [
            n for n in self._nodes.values()
            if n.model == model_name and n.status in (NodeStatus.ONLINE, NodeStatus.STALE)
        ]

    def cluster_summary(self) -> dict:
        """Return a summary dict of the cluster state."""
        active_nodes = [n for n in self._nodes.values() if n.status != NodeStatus.OFFLINE]
        models = set(n.model for n in active_nodes if n.model)
        leader = self.get_leader()

        return {
            "total_nodes": len(active_nodes),
            "total_gpus": len(active_nodes),
            "total_memory_gb": sum(n.gpu_memory_gb for n in active_nodes),
            "models_available": sorted(models),
            "leader_id": leader.node_id if leader else None,
        }
