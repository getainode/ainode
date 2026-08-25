"""InstanceManager — the N concurrent distributed instances a head node runs.

Phase 2 (P2-2). The single ``app["engine"]`` slot baked in "exactly one instance";
this registry makes the set explicit so a head can run several models at once on
disjoint node sets. ``app["engine"]`` is kept = the *primary* (first) instance for
the not-yet-changed proxy/status path; this manager holds them all.

Ports are allocated from ``base_port`` (the node's configured api_port) upward, so
the first instance reuses the legacy port (8000) and the proxy/status keep working.
"""

from __future__ import annotations

import socket
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

    @staticmethod
    def _port_bindable(port: int, host: str = "0.0.0.0") -> bool:
        """True if nothing on the HOST is already listening on this port.

        The engine container runs on the host network, so a port owned by any
        other process (not just another AINode instance) collides. Without this
        check the engine launches, fails deep in startup with
        ``OSError: [Errno 98] Address already in use``, and the caller only sees
        a generic launch failure. Observed 2026-08-25 on a node where an
        unrelated service had held 8000 for weeks.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
            return True
        except OSError:
            return False

    def allocate_port(self, probe: bool = True) -> int:
        """Lowest free port from base_port up.

        Skips ports held by an existing instance AND, unless ``probe`` is off,
        ports already bound by anything else on the host.
        """
        used = self.used_ports()
        port = self._base_port
        for _ in range(256):
            if port not in used and (not probe or self._port_bindable(port)):
                return port
            port += 1
        return port
