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
    # Which distributed shape launched it: "ray" (ray containers + docker exec)
    # or "mp" (one vllm serve container per node, vLLM's own multi-node
    # executor). Carried so a replay relaunches the SAME shape. An image that
    # ships no ray cannot be brought back by the Ray path.
    distributed_executor: str = "ray"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "InstanceRecord":
        """Parse a wire dict, ignoring unknown keys (forward-compatible)."""
        fields = (
            "instance_id", "model", "head_node_id", "member_node_ids",
            "peer_ips", "api_port", "tensor_parallel_size", "status",
            "distributed_executor",
        )
        return cls(**{k: d[k] for k in fields if k in d})


def instance_parallel(inst) -> int:
    """How many GPUs an instance spans, read from a record OR a wire dict.

    The launch width lives only on the InstanceRecord, so every view that reports
    parallelism (``/api/server/status``'s ``loaded_models[].parallel``,
    ``/api/nodes``) must read it from there rather than assume 1 (#92). An older
    node does not send the field at all, and a missing, zero or unparseable value
    reads as 1 (one node, no sharding), which keeps the wire format backward
    compatible in both directions.
    """
    raw = (inst.get("tensor_parallel_size") if isinstance(inst, dict)
           else getattr(inst, "tensor_parallel_size", None))
    try:
        width = int(raw or 1)
    except (TypeError, ValueError):
        return 1
    return width if width > 0 else 1
