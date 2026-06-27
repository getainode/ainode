"""InstanceManager — the N concurrent distributed instances a head node runs.

Phase 2 (P2-2). The single ``app["engine"]`` slot baked in "exactly one instance";
this registry makes the set explicit so a head can run several models at once on
disjoint node sets. ``app["engine"]`` is kept = the *primary* (first) instance for
the not-yet-changed proxy/status path; this manager holds them all.

Ports are allocated from ``base_port`` (the node's configured api_port) upward, so
the first instance reuses the legacy port (8000) and the proxy/status keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ainode.discovery.instance import InstanceRecord


@dataclass
class Instance:
    record: InstanceRecord
    backend: object  # EngineBackend (avoids an import cycle)


class InstanceManager:
    def __init__(self, base_port: int = 8000):
        self._base_port = base_port
        self._instances: Dict[str, Instance] = {}

    def add(self, record: InstanceRecord, backend) -> None:
        self._instances[record.instance_id] = Instance(record=record, backend=backend)

    def get(self, instance_id: str) -> Optional[Instance]:
        return self._instances.get(instance_id)

    def by_model(self, model: str) -> Optional[Instance]:
        for inst in self._instances.values():
            if inst.record.model == model:
                return inst
        return None

    def remove(self, instance_id: str) -> Optional[Instance]:
        return self._instances.pop(instance_id, None)

    def records(self) -> List[InstanceRecord]:
        return [i.record for i in self._instances.values()]

    def instances(self) -> List[Instance]:
        return list(self._instances.values())

    def is_empty(self) -> bool:
        return not self._instances

    def used_ports(self) -> set:
        return {i.record.api_port for i in self._instances.values()}

    def allocate_port(self) -> int:
        """Lowest free port from base_port up, not held by an existing instance."""
        used = self.used_ports()
        port = self._base_port
        while port in used:
            port += 1
        return port
