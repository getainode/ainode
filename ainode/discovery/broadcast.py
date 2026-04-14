"""UDP broadcast-based node discovery for automatic clustering."""

import json
import socket
import asyncio
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Callable, Dict, List, Optional


class NodeStatus(str, Enum):
    """Health status of a discovered node."""
    ONLINE = "online"
    STALE = "stale"
    OFFLINE = "offline"


# Thresholds in seconds
ONLINE_THRESHOLD = 15.0
STALE_THRESHOLD = 30.0


@dataclass
class NodeAnnouncement:
    """Broadcast message from a node."""
    node_id: str
    node_name: str
    gpu_name: str
    gpu_memory_gb: float
    unified_memory: bool
    model: str
    status: str
    api_port: int
    web_port: int
    timestamp: float = field(default_factory=time.time)
    # Cluster membership: only nodes with the same cluster_id see each other.
    cluster_id: str = "default"
    # Raw role from config: "auto" | "master" | "worker".
    role: str = "auto"
    # Runtime flag -- set to True if this node currently believes it is the
    # elected master for its cluster. Informational only; workers make their
    # own decision based on the full announcement set.
    is_master: bool = False
    # Distributed inference mode: "solo" (own vLLM) | "head" (runs sharded
    # vLLM across self + members) | "member" (GPU reserved for Ray workers
    # placed by the head, no local vLLM).
    distributed_mode: str = "solo"
    # When this node is a head with an active distributed instance, this is
    # the instance id + participating peer node_ids so the UI can render
    # "DISTRIBUTED across N nodes" and know which topology members are busy.
    distributed_instance_id: Optional[str] = None
    distributed_peers: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "NodeAnnouncement":
        """Deserialize from JSON string, tolerating unknown/missing fields."""
        raw = json.loads(data)
        fields = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**fields)


@dataclass
class DiscoveredNode:
    """A node discovered on the network, with health tracking."""
    announcement: NodeAnnouncement
    last_seen: float = field(default_factory=time.time)

    @property
    def age(self) -> float:
        """Seconds since last heartbeat."""
        return time.time() - self.last_seen

    @property
    def health(self) -> NodeStatus:
        """Determine node health based on heartbeat age."""
        age = self.age
        if age < ONLINE_THRESHOLD:
            return NodeStatus.ONLINE
        elif age < STALE_THRESHOLD:
            return NodeStatus.STALE
        return NodeStatus.OFFLINE


class BroadcastSender:
    """Sends UDP broadcast announcements on a regular interval."""

    def __init__(
        self,
        announcement: NodeAnnouncement,
        discovery_port: int = 5678,
        broadcast_interval: float = 5.0,
    ):
        self.announcement = announcement
        self.discovery_port = discovery_port
        self.broadcast_interval = broadcast_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the broadcast loop."""
        self._running = True
        self._task = asyncio.create_task(self._broadcast_loop())

    async def stop(self):
        """Stop broadcasting."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def update_announcement(self, **kwargs):
        """Update fields on the announcement (e.g. model, status)."""
        for key, value in kwargs.items():
            if hasattr(self.announcement, key):
                setattr(self.announcement, key, value)

    async def _broadcast_loop(self):
        """Periodically broadcast our presence."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setblocking(False)

        try:
            while self._running:
                try:
                    self.announcement.timestamp = time.time()
                    data = self.announcement.to_json().encode()
                    sock.sendto(data, ("<broadcast>", self.discovery_port))
                except Exception:
                    pass
                await asyncio.sleep(self.broadcast_interval)
        finally:
            sock.close()


class BroadcastListener:
    """Listens for UDP broadcast announcements and maintains a node registry."""

    def __init__(
        self,
        local_node_id: str,
        discovery_port: int = 5678,
        on_node_found: Optional[Callable[[NodeAnnouncement], None]] = None,
        on_node_lost: Optional[Callable[[str], None]] = None,
    ):
        self.local_node_id = local_node_id
        self.discovery_port = discovery_port
        self.on_node_found = on_node_found
        self.on_node_lost = on_node_lost
        self._registry: Dict[str, DiscoveredNode] = {}
        self._running = False
        self._listen_task: Optional[asyncio.Task] = None
        self._reaper_task: Optional[asyncio.Task] = None

    @property
    def registry(self) -> Dict[str, DiscoveredNode]:
        return dict(self._registry)

    def get_nodes(self, include_offline: bool = False) -> List[DiscoveredNode]:
        """Return discovered nodes, optionally filtering out offline ones."""
        nodes = list(self._registry.values())
        if not include_offline:
            nodes = [n for n in nodes if n.health != NodeStatus.OFFLINE]
        return nodes

    def get_node(self, node_id: str) -> Optional[DiscoveredNode]:
        """Get a specific discovered node by ID."""
        return self._registry.get(node_id)

    def _process_announcement(self, announcement: NodeAnnouncement):
        """Process a received announcement."""
        if announcement.node_id == self.local_node_id:
            return

        is_new = announcement.node_id not in self._registry
        self._registry[announcement.node_id] = DiscoveredNode(
            announcement=announcement,
            last_seen=time.time(),
        )

        if is_new and self.on_node_found:
            self.on_node_found(announcement)

    async def start(self):
        """Start listening for broadcasts."""
        self._running = True
        self._listen_task = asyncio.create_task(self._listen_loop())
        self._reaper_task = asyncio.create_task(self._reaper_loop())

    async def stop(self):
        """Stop listening."""
        self._running = False
        for task in [self._listen_task, self._reaper_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _listen_loop(self):
        """Listen for broadcast announcements."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass
        sock.bind(("", self.discovery_port))
        sock.setblocking(False)

        loop = asyncio.get_event_loop()

        try:
            while self._running:
                try:
                    data = await loop.run_in_executor(None, lambda: sock.recv(4096))
                    announcement = NodeAnnouncement.from_json(data.decode())
                    self._process_announcement(announcement)
                except Exception:
                    await asyncio.sleep(1)
        finally:
            sock.close()

    async def _reaper_loop(self):
        """Periodically check for offline nodes and fire on_node_lost."""
        while self._running:
            await asyncio.sleep(10)
            to_remove = []
            for node_id, node in self._registry.items():
                if node.health == NodeStatus.OFFLINE:
                    to_remove.append(node_id)

            for node_id in to_remove:
                del self._registry[node_id]
                if self.on_node_lost:
                    self.on_node_lost(node_id)
