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
    cluster_id: str = "default"
    role: str = "auto"  # raw config role
    is_master: bool = False  # broadcast-declared
    distributed_mode: str = "solo"
    distributed_instance_id: Optional[str] = None
    distributed_peers: list = field(default_factory=list)
    peer_ip: Optional[str] = None  # captured from UDP recvfrom on the head

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
            cluster_id=getattr(a, "cluster_id", "default"),
            role=getattr(a, "role", "auto"),
            is_master=getattr(a, "is_master", False),
            distributed_mode=getattr(a, "distributed_mode", "solo"),
            distributed_instance_id=getattr(a, "distributed_instance_id", None),
            distributed_peers=list(getattr(a, "distributed_peers", []) or []),
            peer_ip=getattr(discovered, "peer_ip", None),
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
            cluster_id=getattr(announcement, "cluster_id", "default"),
            role=getattr(announcement, "role", "auto"),
            is_master=getattr(announcement, "is_master", False),
            distributed_mode=getattr(announcement, "distributed_mode", "solo"),
            distributed_instance_id=getattr(announcement, "distributed_instance_id", None),
            distributed_peers=list(getattr(announcement, "distributed_peers", []) or []),
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
        self._nodes[node.node_id] = node

    def remove_node(self, node_id: str):
        self._nodes.pop(node_id, None)

    def get_node(self, node_id: str) -> Optional[ClusterNode]:
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

    # ------------------------------------------------------------------
    # Master election (role-aware, cluster-id scoped)
    # ------------------------------------------------------------------

    def _local_cluster_id(self) -> str:
        if self._local_announcement:
            return getattr(self._local_announcement, "cluster_id", "default")
        return "default"

    def _peers_in_cluster(self) -> List[ClusterNode]:
        """Online nodes (including self) sharing the local cluster_id."""
        cid = self._local_cluster_id()
        return [
            n for n in self._nodes.values()
            if n.status == NodeStatus.ONLINE and (n.cluster_id or "default") == cid
        ]

    def get_master(self) -> Optional[ClusterNode]:
        """Return the elected master node for the local cluster.

        Election rules:
          * If any online node in the cluster has ``role == "master"`` (explicit),
            the lowest such node_id wins.
          * Otherwise, the lowest node_id among online nodes whose role is
            ``"auto"`` becomes master.
          * Nodes with ``role == "worker"`` are NEVER elected.
          * Only nodes sharing the local ``cluster_id`` participate.
        """
        peers = self._peers_in_cluster()
        if not peers:
            return None

        explicit = [n for n in peers if n.role == "master"]
        if explicit:
            return min(explicit, key=lambda n: n.node_id)

        candidates = [n for n in peers if n.role == "auto"]
        if not candidates:
            return None
        return min(candidates, key=lambda n: n.node_id)

    def is_master_of_cluster(self) -> bool:
        """Return True if the local node is currently the elected master."""
        if not self._local_announcement:
            return False
        master = self.get_master()
        if master is None:
            return False
        return master.node_id == self._local_announcement.node_id

    def get_cluster_role_for(self, node_id: str) -> str:
        """Return the EFFECTIVE role for *node_id*: ``"master"`` or ``"worker"``.

        Nodes outside the local cluster, or unknown, return ``"worker"``.
        """
        master = self.get_master()
        if master is not None and master.node_id == node_id:
            return "master"
        return "worker"

    def members(self, cluster_id: Optional[str] = None) -> List[ClusterNode]:
        """Return all nodes that share the given (or local) cluster_id."""
        cid = cluster_id or self._local_cluster_id()
        return [
            n for n in self._nodes.values()
            if (n.cluster_id or "default") == cid
        ]

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
