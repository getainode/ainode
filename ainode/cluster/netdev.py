"""Network-device enumeration and cluster-interface autodetection.

AINode binds NCCL / Ray / Gloo / UCX to one netdev: the cluster fabric
interface. That name is hardware specific. A DGX Spark's direct-connect
NIC is ``enP2p1s0f1np1``; an ASUS GX10 calls the same class of port
``enp1s0f0np0``; a plain workstation with a switched fabric might use
``eno1``. Historically AINode shipped a hardcoded guess, so on any box
that did not match, the engine came up bound to ``127.0.0.1`` (silently,
via :func:`ainode.cluster.hca_discovery.detect_fabric_ip` returning None)
or died with an opaque EngineCore failure, and the user had to discover
the real name with ``ip -br addr`` and hand-edit ``config.json``.

This module removes the guess. :func:`detect_cluster_interface` ranks the
host's real interfaces and :func:`resolve_cluster_interface` is the single
accessor every call site uses in place of ``config.cluster_interface``.

Everything here is read-only and side-effect free: two short ``ip``
invocations plus a few sysfs reads. Both the command runner and the sysfs
root are injectable so tests never touch a real host.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Injectable for tests: point at a fake tree instead of the real sysfs.
SYS_CLASS_NET = Path("/sys/class/net")

# Devices that are never the cluster fabric. Loopback by exact name; the
# rest by prefix, covering container bridges/veth pairs, libvirt bridges,
# VPN/overlay tunnels (Tailscale, WireGuard, ZeroTier, generic tun/tap),
# and CNI plugin devices (Flannel, Calico, kube-proxy dummies, LXC).
EXCLUDED_NAMES = frozenset({"lo"})
EXCLUDED_PREFIXES = (
    "docker",
    "br-",
    "veth",
    "virbr",
    "tailscale",
    "wg",
    "tun",
    "tap",
    "cni",
    "flannel",
    "cali",
    "kube",
    "lxc",
    "zt",
)

# IPv4 link-local (RFC 3927). An autoconfigured 169.254.x.x address means
# "no DHCP answered", not "here is your fabric".
_LINK_LOCAL_PREFIX = "169.254."

_ADDR_RE = re.compile(r"^\d+:\s+(\S+)\s+inet\s+(\d+\.\d+\.\d+\.\d+)")
_ROUTE_DEV_RE = re.compile(r"\bdev\s+(\S+)")

# A command runner takes an argv list and returns stdout, or "" on any
# failure (binary missing, non-zero exit, timeout).
CommandRunner = Callable[[Sequence[str]], str]

_SUBPROCESS_TIMEOUT = 5


@dataclass(frozen=True)
class NetDev:
    """One IPv4-carrying network device on this host."""

    name: str
    ipv4: str
    is_up: bool
    rdma: bool
    is_default_route: bool


def _run_command(argv: Sequence[str]) -> str:
    """Run ``argv`` and return stdout, or "" if it cannot be run.

    Tolerates a host with no ``ip`` binary (macOS dev boxes, slim
    containers) rather than raising: callers treat "" as "no data".
    """
    try:
        out = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("%s failed: %s", " ".join(argv), exc)
        return ""
    if out.returncode != 0:
        logger.debug("%s exited %d", " ".join(argv), out.returncode)
        return ""
    return out.stdout or ""


def is_virtual_interface(name: str) -> bool:
    """True when ``name`` is a loopback / bridge / overlay / tunnel device."""
    if name in EXCLUDED_NAMES:
        return True
    return name.startswith(EXCLUDED_PREFIXES)


def _parse_ipv4_addrs(stdout: str) -> Dict[str, str]:
    """Map device name to its first global IPv4 from ``ip -o -4 addr show``."""
    addrs: Dict[str, str] = {}
    for line in stdout.splitlines():
        match = _ADDR_RE.match(line.strip())
        if not match:
            continue
        name = match.group(1).split("@", 1)[0]
        ipv4 = match.group(2)
        if ipv4.startswith(_LINK_LOCAL_PREFIX):
            continue
        addrs.setdefault(name, ipv4)
    return addrs


def _parse_default_route_devs(stdout: str) -> set:
    """Device names carrying a default route (handles multipath nexthops)."""
    return set(_ROUTE_DEV_RE.findall(stdout))


def _operstate(name: str, sysfs_root: Path) -> Optional[str]:
    try:
        return (sysfs_root / name / "operstate").read_text().strip().lower()
    except OSError:
        return None


def _has_rdma(name: str, sysfs_root: Path) -> bool:
    """True when the netdev's parent device exposes an infiniband class dir."""
    try:
        return (sysfs_root / name / "device" / "infiniband").exists()
    except OSError:
        return False


def _enumerate(root: Path, run: CommandRunner):
    """Read the host once. Returns (every IPv4 addr, non-virtual NetDevs)."""
    addrs = _parse_ipv4_addrs(run(["ip", "-o", "-4", "addr", "show"]))
    default_devs = _parse_default_route_devs(run(["ip", "route", "show", "default"]))

    devices: List[NetDev] = []
    for name, ipv4 in addrs.items():
        if is_virtual_interface(name):
            continue
        state = _operstate(name, root)
        devices.append(
            NetDev(
                name=name,
                ipv4=ipv4,
                is_up=state not in ("down", "lowerlayerdown"),
                rdma=_has_rdma(name, root),
                is_default_route=name in default_devs,
            )
        )
    return addrs, sorted(devices, key=lambda d: d.name)


def list_ipv4_interfaces(
    sysfs_root: Optional[Path] = None,
    runner: Optional[CommandRunner] = None,
) -> List[NetDev]:
    """Return every non-virtual device that currently carries an IPv4.

    Sorted by name so callers and error messages are deterministic.
    ``is_up`` is read from sysfs ``operstate``: only an explicitly down
    (or ``lowerlayerdown``) device reports False, because many real
    devices report ``unknown`` and an unreadable sysfs must not make the
    whole host look down.
    """
    root = SYS_CLASS_NET if sysfs_root is None else sysfs_root
    run = _run_command if runner is None else runner
    return _enumerate(root, run)[1]


def detect_cluster_interface(
    preferred: str = "",
    sysfs_root: Optional[Path] = None,
    runner: Optional[CommandRunner] = None,
) -> Optional[str]:
    """Pick the interface most likely to be the cluster fabric.

    ``preferred`` (the configured name) wins whenever it exists and
    carries an IPv4, even if it looks virtual: an explicit pin is a
    decision, not a guess, so a user who deliberately runs the fabric over
    a tunnel is not overruled.

    Otherwise candidates are ranked, best first:

    1. up, RDMA capable, has an IPv4: a RoCE/IB fabric port,
    2. the default-route device, the one interface known to route,
    3. any up, non-virtual device with an IPv4.

    Ties inside a tier break on name, so repeated calls on the same host
    return the same answer. Returns None when nothing qualifies (no
    ``ip``, no addresses, nothing but virtual devices).
    """
    root = SYS_CLASS_NET if sysfs_root is None else sysfs_root
    run = _run_command if runner is None else runner

    all_addrs, devices = _enumerate(root, run)
    if preferred and preferred in all_addrs:
        return preferred

    candidates = [dev for dev in devices if dev.is_up]
    if not candidates:
        return None

    for tier in (
        lambda d: d.rdma,
        lambda d: d.is_default_route,
        lambda _d: True,
    ):
        matches = [d for d in candidates if tier(d)]
        if matches:
            return matches[0].name
    return None


# ---------------------------------------------------------------------------
# The accessor every call site uses instead of config.cluster_interface
# ---------------------------------------------------------------------------

# Resolution is cached per configured value: the engine, the NCCL env
# builder, and the discovery announcement loop all ask for it, and the
# announcement loop asks on a timer. One `ip` pair per process, not per
# broadcast. Also keeps the fallback warning to a single line in the log.
_RESOLVED: Dict[str, str] = {}


def reset_cache() -> None:
    """Forget the cached resolution. For tests, and for config reloads."""
    _RESOLVED.clear()


def _resolve_uncached(configured: str) -> str:
    detected = detect_cluster_interface(configured)

    if detected is not None and detected == configured:
        return configured

    if detected is None:
        # Nothing to offer. Return the configured name unchanged so the
        # downstream error still names the interface the user set.
        if configured:
            logger.warning(
                "cluster_interface %r is not present on this host and no "
                "other interface with an IPv4 address was found. Leaving it "
                "as configured; cross-node traffic will not bind.",
                configured,
            )
        else:
            logger.warning(
                "No cluster interface could be auto-detected on this host. "
                "Set cluster_interface in ~/.ainode/config.json."
            )
        return configured

    if configured:
        logger.warning(
            "cluster_interface %r is not present on this host; using %r. "
            "Set cluster_interface in ~/.ainode/config.json to pin it.",
            configured,
            detected,
        )
    else:
        logger.info("cluster_interface not set; auto-detected %r.", detected)
    return detected


def resolve_cluster_interface(config) -> str:
    """Return the cluster interface to bind, autodetecting when needed.

    Returns ``config.cluster_interface`` when that device exists on this
    host. When it does not (or when it is empty, meaning autodetect), a
    ranked detection runs and its answer is returned, with one warning per
    process. When detection also comes up empty the configured value is
    returned unchanged, so downstream errors still name it.
    """
    configured = (getattr(config, "cluster_interface", "") or "").strip()
    if configured in _RESOLVED:
        return _RESOLVED[configured]
    resolved = _resolve_uncached(configured)
    _RESOLVED[configured] = resolved
    return resolved


def describe_cluster_interface(config) -> str:
    """One-line human summary for the CLI banner.

    ``'enp1s0f0np0 (192.168.6.162)'`` when resolved and addressed,
    ``'eno1 (no IPv4)'`` when the configured name has no address, or
    ``'none detected'`` when there is nothing to report.
    """
    iface = resolve_cluster_interface(config)
    if not iface:
        return "none detected"
    for dev in list_ipv4_interfaces():
        if dev.name == iface:
            return f"{iface} ({dev.ipv4})"
    return f"{iface} (no IPv4)"


def interface_candidates_hint() -> str:
    """Interfaces that DO carry an IPv4, for a 'wrong NIC' error message."""
    devices = list_ipv4_interfaces()
    if not devices:
        return "no interface on this host currently carries an IPv4 address"
    return ", ".join(f"{d.name} ({d.ipv4})" for d in devices)
