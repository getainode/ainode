"""UDP broadcast-based node discovery for automatic clustering."""

import json
import logging
import socket
import asyncio
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Callable, Dict, List, Optional

from ainode.core.config import DEFAULT_DISCOVERY_PORT
from ainode.discovery.signing import ACCEPT, rejection, seal
from ainode.metrics.collector import optional_float


class NodeStatus(str, Enum):
    """Health status of a discovered node."""
    ONLINE = "online"
    STALE = "stale"
    OFFLINE = "offline"


# Thresholds in seconds
ONLINE_THRESHOLD = 15.0
STALE_THRESHOLD = 30.0

logger = logging.getLogger(__name__)

# The listener reads one datagram of this size. An announcement that outgrows it
# is TRUNCATED on arrival, so it fails to parse and the node silently vanishes
# from every peer's cluster view. Sender-side check below, and a test pins that a
# fully populated announcement still fits, because this payload has grown field
# by field (telemetry, instances, load progress) and nothing else would notice.
MAX_ANNOUNCEMENT_BYTES = 4096

# Both classes below take core.config.DEFAULT_DISCOVERY_PORT (5679) as their
# default, imported rather than respelled: each carried its own 5678 while the
# installer, the fleet and the docs used 5679, which is how a node could listen
# where nobody spoke (#181).


@dataclass
class NodeAnnouncement:
    """Broadcast message from a node."""
    node_id: str
    node_name: str
    gpu_name: str
    # The node's TOTAL memory across every NVIDIA device on it, not device 0's.
    # A four-V100 host announced 32 GB and counted as one GPU, which is what the
    # cluster's total VRAM, the topology's GPU count and every placement decision
    # were built on (#163).
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
    # How many NVIDIA devices this node has. One per node was true of every GB10
    # and false of the two x86 nodes in the fleet, so the fleet's GPU count was
    # its node count (#163). An older peer sends no field, which reads as 1, the
    # shape it was announcing anyway.
    gpu_count: int = 1
    # Live GPU telemetry (metrics fan-out): stamped fresh on every broadcast
    # tick so the head can render real per-peer VRAM/util on the cluster
    # graphic. None means "this node cannot measure it", which is the truth on a
    # part whose driver does not populate the counter, and it is NOT the same
    # claim as 0 (#176, #175): a permanent 0 reads as an idle, empty node. An
    # older peer sends 0.0 for all four, and a reader cannot tell that apart from
    # a real zero: one release of that, and nothing worse than today.
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    gpu_temp: Optional[float] = None
    # This node's IP on the cluster fabric (cluster_interface). The head uses
    # this to launch distributed peers over the fabric — NOT the mgmt-LAN UDP
    # source IP (peer_ip), which lands a Ray worker on a non-GPU address (BUG D).
    fabric_ip: str = ""
    # Phase 2: distributed instances this node HEADS, as wire dicts
    # (InstanceRecord.to_dict()). Empty for non-heads / solo. Same info as the
    # legacy distributed_instance_id/distributed_peers, but a list so a head can
    # run more than one. from_json drops unknown keys → older peers stay OK.
    instances: List[dict] = field(default_factory=list)
    # Live load progress, so a model coming up on THIS node can be drawn as
    # loading (elapsed, expected, "taking longer than usual") on any other
    # node's dashboard, not just as "not ready yet". ``status`` already says
    # starting vs serving; these say how far in and how long it should take.
    # All empty/None when nothing is loading, and on a peer too old to send
    # them, which every reader treats the same way: no bar to draw.
    load_phase: str = ""
    load_started_at: Optional[float] = None
    # The loading node's OWN arithmetic: the cluster shares no clock, so a
    # reader must not subtract a remote start stamp from its own now().
    load_elapsed_seconds: Optional[float] = None
    expected_ready_minutes: Optional[float] = None
    # The release this node is running (ainode.__version__). Without it a cluster
    # split across two releases is indistinguishable from one on a single release
    # (#171), which is the state a roll leaves every time a node is missed, and
    # the announcement IS the cross-version contract: 0.5.25 changed how a head's
    # instances are merged into ``instances``, so a 0.5.24 reader and a 0.5.25
    # sender were exchanging different data with nothing to say so. Empty on a
    # peer too old to send it, which every reader renders as "unknown" rather
    # than guessing.
    ainode_version: str = ""

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
    # IP address the announcement arrived from (captured via recvfrom).
    # This is authoritative for head→peer connectivity because the
    # announcement payload itself doesn't carry a routable IP.
    peer_ip: Optional[str] = None

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
        discovery_port: int = DEFAULT_DISCOVERY_PORT,
        broadcast_interval: float = 5.0,
        metrics_provider: Optional[Callable[[], dict]] = None,
        secret_provider: Optional[Callable[[], Optional[str]]] = None,
    ):
        self.announcement = announcement
        self.discovery_port = discovery_port
        self.broadcast_interval = broadcast_interval
        # Optional zero-arg callable returning a get_gpu_metrics()-shaped dict;
        # its values are stamped onto the announcement each tick so peers
        # broadcast live VRAM/util/temp.
        self.metrics_provider = metrics_provider
        # Optional zero-arg callable returning the cluster_secret, asked on EVERY
        # send rather than captured here: rotating the secret must not need a
        # restart of every node in the fleet (see signing.ClusterSecret).
        self.secret_provider = secret_provider
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._warned_oversize = False
        self._warned_secret_error = False

    def current_secret(self) -> Optional[str]:
        """The secret to sign with right now, or None to send unsigned.

        A provider that raises is treated as "no secret" and reported once: the
        node then keeps broadcasting, and a peer that HAS a secret drops it,
        which is visible in that peer's log rather than silent here.
        """
        if self.secret_provider is None:
            return None
        try:
            return self.secret_provider()
        except Exception:
            if not self._warned_secret_error:
                self._warned_secret_error = True
                logger.warning(
                    "could not read cluster_secret; broadcasting UNSIGNED "
                    "announcements, which peers that have a secret will drop",
                    exc_info=True)
            return None

    def datagram(self) -> bytes:
        """The bytes for one announcement, signed when a secret is configured."""
        return json.dumps(seal(asdict(self.announcement), self.current_secret())).encode()

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
                    # Refresh live GPU telemetry so each broadcast carries the
                    # node's current VRAM/util (the head renders it per-peer).
                    if self.metrics_provider is not None:
                        try:
                            m = self.metrics_provider() or {}
                            if not m.get("error"):
                                # A figure the node cannot measure travels as
                                # null, so a peer draws n/a rather than a zero it
                                # would read as an idle node (#176).
                                self.announcement.gpu_memory_used_mb = optional_float(m.get("memory_used_mb"))
                                self.announcement.gpu_memory_total_mb = optional_float(m.get("memory_total_mb"))
                                self.announcement.gpu_utilization = optional_float(m.get("utilization_percent"))
                                self.announcement.gpu_temp = optional_float(m.get("temperature_c"))
                                if m.get("gpu_count"):
                                    self.announcement.gpu_count = int(m["gpu_count"])
                        except Exception:
                            pass
                    # Signed here, not in to_json(): the signature is an envelope
                    # around the payload, and the size check below has to see the
                    # bytes that actually go on the wire.
                    data = self.datagram()
                    if len(data) > MAX_ANNOUNCEMENT_BYTES and not self._warned_oversize:
                        # Once per process: peers are dropping us and nothing
                        # else in the system can tell you why.
                        self._warned_oversize = True
                        logger.warning(
                            "discovery announcement is %d bytes, past the %d-byte "
                            "listener buffer: peers cannot parse it and will drop "
                            "this node. Shrink what the announcement carries.",
                            len(data), MAX_ANNOUNCEMENT_BYTES)
                    sock.sendto(data, ("<broadcast>", self.discovery_port))
                except Exception:
                    pass
                await asyncio.sleep(self.broadcast_interval)
        finally:
            sock.close()


class BroadcastListener:
    """Listens for UDP broadcast announcements and maintains a node registry."""

    # A rejected sender is reported once per (source, reason) so a forged flood
    # cannot fill the log, and capped so a spoofed source address cannot fill
    # memory either: at this many distinct pairs the listener says so once and
    # stops naming individual sources. ``dropped`` keeps counting either way.
    MAX_REPORTED_SOURCES = 50

    def __init__(
        self,
        local_node_id: str,
        discovery_port: int = DEFAULT_DISCOVERY_PORT,
        on_node_found: Optional[Callable[[NodeAnnouncement], None]] = None,
        on_node_lost: Optional[Callable[[str], None]] = None,
        secret_provider: Optional[Callable[[], Optional[str]]] = None,
    ):
        self.local_node_id = local_node_id
        self.discovery_port = discovery_port
        self.on_node_found = on_node_found
        self.on_node_lost = on_node_lost
        # Asked per datagram, so a rotated secret takes effect without a restart
        # (see signing.ClusterSecret). None, or a provider that returns nothing,
        # means this node has no secret and accepts what it always accepted.
        self.secret_provider = secret_provider
        self._registry: Dict[str, DiscoveredNode] = {}
        self._running = False
        self._listen_task: Optional[asyncio.Task] = None
        self._reaper_task: Optional[asyncio.Task] = None
        self._warned_unauthenticated = False
        self._warned_flood = False
        self._reported: set = set()
        self.dropped: Dict[str, int] = {}

    def current_secret(self) -> Optional[str]:
        """The secret to verify against right now, or None to verify nothing."""
        if self.secret_provider is None:
            return None
        try:
            return self.secret_provider()
        except Exception:
            # Cannot read the key: verifying would drop the whole cluster, so
            # accept as an unsigned fleet does and let the once-per-process
            # warning below say discovery is unauthenticated.
            logger.debug("cluster_secret provider raised", exc_info=True)
            return None

    def _report_drop(self, reason: str, source: str, node_id: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1
        key = (reason, source)
        if key in self._reported:
            return
        if len(self._reported) >= self.MAX_REPORTED_SOURCES:
            if not self._warned_flood:
                self._warned_flood = True
                logger.warning(
                    "more than %d distinct sources have sent discovery "
                    "announcements this node cannot verify (%s); no longer naming "
                    "them individually. Counts stay in the listener's `dropped`.",
                    self.MAX_REPORTED_SOURCES, self.dropped)
            return
        self._reported.add(key)
        logger.warning(
            "dropped a discovery announcement from %s claiming node_id %r: %s. "
            "This node has a cluster_secret, so an announcement it cannot verify "
            "is not cluster membership. Set the SAME cluster_secret on that node, "
            "or it stays invisible here.",
            source, node_id, reason)

    def handle_datagram(self, data: bytes, peer_ip: Optional[str] = None) -> str:
        """Verify and register one received datagram. Returns why it was dropped.

        The empty string means it was accepted. Split out of the receive loop so
        the accept/drop decision is one testable function of the bytes on the
        wire, which is the only place the forged-master attack in #169 can be
        stopped.
        """
        try:
            raw = json.loads(data.decode())
        except Exception:
            return "unparseable"
        if not isinstance(raw, dict):
            return "unparseable"

        secret = self.current_secret()
        if not secret and not self._warned_unauthenticated:
            self._warned_unauthenticated = True
            logger.warning(
                "discovery is UNAUTHENTICATED on port %d: no cluster_secret is "
                "set, so any host on this broadcast domain can join this cluster, "
                "claim to be its master and advertise instances this node will "
                "route inference to. Set the same cluster_secret on every node to "
                "sign announcements.",
                self.discovery_port)

        reason = rejection(raw, secret)
        if reason != ACCEPT:
            self._report_drop(reason, peer_ip or "an unknown address",
                              str(raw.get("node_id", "")))
            return reason

        try:
            announcement = NodeAnnouncement.from_json(data.decode())
        except Exception:
            return "unparseable"
        self._process_announcement(announcement, peer_ip=peer_ip)
        return ACCEPT

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

    def _process_announcement(self, announcement: NodeAnnouncement, peer_ip: Optional[str] = None):
        """Process a received announcement."""
        if announcement.node_id == self.local_node_id:
            return

        is_new = announcement.node_id not in self._registry
        self._registry[announcement.node_id] = DiscoveredNode(
            announcement=announcement,
            last_seen=time.time(),
            peer_ip=peer_ip,
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
                    # recvfrom exposes the sender's (ip, port) — we persist
                    # the IP on DiscoveredNode so the head can use it as the
                    # authoritative address for SSH + Ray bootstrap.
                    data, addr = await loop.run_in_executor(
                        None, lambda: sock.recvfrom(MAX_ANNOUNCEMENT_BYTES))
                    peer_ip = addr[0] if addr else None
                    self.handle_datagram(data, peer_ip=peer_ip)
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
