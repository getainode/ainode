"""InstanceRecord — one running (or pending) distributed model instance.

Phase 2 spine. The cluster represents N concurrent instances on disjoint node
sets as a *list* of these. Today there's exactly one; this type lets the same
plumbing carry many without changing single-instance behavior.

Wire form (inside a NodeAnnouncement) carries the peer **fabric IPs** in
``peer_ips`` — ``member_node_ids`` is resolved downstream in
``/api/cluster/resources`` where the fabric_ip→node map is available, mirroring
the legacy ``distributed_instance_id`` / ``distributed_peers`` fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class InstanceRecord:
    instance_id: str = ""
    model: str = ""
    head_node_id: str = ""
    member_node_ids: List[str] = field(default_factory=list)  # resolved: head + peers, by node_id
    peer_ips: List[str] = field(default_factory=list)          # peer FABRIC IPs (wire form)
    api_port: int = 8000
    tensor_parallel_size: int = 1
    status: str = "serving"  # starting | distributing | serving | failed

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "InstanceRecord":
        """Parse a wire dict, ignoring unknown keys (forward-compatible)."""
        fields = (
            "instance_id", "model", "head_node_id", "member_node_ids",
            "peer_ips", "api_port", "tensor_parallel_size", "status",
        )
        return cls(**{k: d[k] for k in fields if k in d})
