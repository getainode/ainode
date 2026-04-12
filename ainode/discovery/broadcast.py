"""UDP broadcast-based node discovery for automatic clustering."""

import json
import socket
import asyncio
import uuid
from dataclasses import dataclass, asdict
from typing import Callable, Optional


@dataclass
class NodeAnnouncement:
    """Broadcast message from a node."""
    node_id: str
    host: str
    api_port: int
    gpu_name: str
    gpu_memory_mb: int
    models_loaded: list
    version: str


class NodeDiscovery:
    """Discovers other AINode instances on the local network via UDP broadcast."""

    def __init__(
        self,
        node_id: str,
        host: str,
        api_port: int,
        discovery_port: int = 5678,
        broadcast_interval: float = 5.0,
        on_node_found: Optional[Callable] = None,
        on_node_lost: Optional[Callable] = None,
    ):
        self.node_id = node_id
        self.host = host
        self.api_port = api_port
        self.discovery_port = discovery_port
        self.broadcast_interval = broadcast_interval
        self.on_node_found = on_node_found
        self.on_node_lost = on_node_lost
        self.known_nodes: dict[str, NodeAnnouncement] = {}
        self._running = False

    async def start(self):
        """Start broadcasting and listening."""
        self._running = True
        asyncio.create_task(self._broadcast_loop())
        asyncio.create_task(self._listen_loop())

    async def stop(self):
        """Stop discovery."""
        self._running = False

    async def _broadcast_loop(self):
        """Periodically broadcast our presence."""
        from ainode.core.gpu import detect_gpu
        from ainode import __version__

        gpu = detect_gpu()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setblocking(False)

        announcement = NodeAnnouncement(
            node_id=self.node_id,
            host=self.host,
            api_port=self.api_port,
            gpu_name=gpu.name if gpu else "unknown",
            gpu_memory_mb=gpu.memory_total_mb if gpu else 0,
            models_loaded=[],
            version=__version__,
        )

        while self._running:
            try:
                data = json.dumps(asdict(announcement)).encode()
                sock.sendto(data, ("<broadcast>", self.discovery_port))
            except Exception:
                pass
            await asyncio.sleep(self.broadcast_interval)

        sock.close()

    async def _listen_loop(self):
        """Listen for broadcasts from other nodes."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", self.discovery_port))
        sock.setblocking(False)

        loop = asyncio.get_event_loop()

        while self._running:
            try:
                data = await loop.run_in_executor(None, lambda: sock.recv(4096))
                announcement = NodeAnnouncement(**json.loads(data.decode()))

                if announcement.node_id == self.node_id:
                    continue  # Ignore our own broadcasts

                if announcement.node_id not in self.known_nodes:
                    self.known_nodes[announcement.node_id] = announcement
                    if self.on_node_found:
                        self.on_node_found(announcement)
                else:
                    self.known_nodes[announcement.node_id] = announcement

            except Exception:
                await asyncio.sleep(1)

        sock.close()
