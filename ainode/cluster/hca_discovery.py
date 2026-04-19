"""HCA + fabric auto-discovery for the NVIDIA engine backend.

This module encodes the Phase-1 NCCL floor-verification lessons into code so
the Phase-4 ``NvidiaBackend`` does not need to hardcode any RoCE names, GID
indices, or fabric IPs. The whole point of this module is to handle the
heterogeneous-cluster case (Sparks 1+2 expose HCAs as ``mlx5_*`` while the
ASUS GX10 / DGX-proper nodes expose them as ``rocep*`` / ``roceP*``) without
the operator having to know about the split.

Everything here reads from ``/sys/class/infiniband`` (kernel-provided, always
present on a node that has a RoCE NIC bound) or shells out to ``ip`` / ``ping``
with short timeouts so the module is safe to import on a Mac without any of
those paths existing.

See ``ops/slices/nvidia-vllm-engine/runbooks/01-nccl-floor-verification.md``
for the empirical grounding of the GID-index-3 + IPv4 RoCEv2 choice.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

# Base sysfs path where the kernel exposes RDMA devices. Shadowed in tests
# via ``monkeypatch.setattr`` so we can assert against fake trees without a
# live Spark.
SYS_INFINIBAND = Path("/sys/class/infiniband")

# HCA name pattern. Accepts both naming conventions we've observed across
# the cluster:
#   mlx5_*   — MOFED / non-DGX-proper Sparks
#   rocep*   — stock rdma-core, lower-case port variant
#   roceP*   — stock rdma-core, upper-case port variant
_HCA_NAME_RE = re.compile(r"^(mlx5_\d+|rocep\w+|roceP\w+)$")

# IPv4-mapped IPv6 RoCEv2 GIDs always start with this prefix
# (five zero groups + ``ffff``). ``fe80:...`` is RoCEv1 link-local and must
# be excluded. Case-insensitive match because some kernels emit mixed case.
_IPV4_MAPPED_RE = re.compile(
    r"^0000:0000:0000:0000:0000:ffff:[0-9a-f]{4}:[0-9a-f]{4}$",
    re.IGNORECASE,
)


def list_local_hcas() -> List[str]:
    """Enumerate InfiniBand / RoCE HCAs on this host.

    Reads ``/sys/class/infiniband/`` and returns the sorted list of HCA
    names that match our accepted naming pattern. Entries that do not
    look like a RoCE HCA (virtual devices, loopback, etc.) are filtered out.

    Returns an empty list if the sysfs path does not exist or cannot be
    read. That keeps the import-and-call flow safe on non-Linux / non-RDMA
    hosts (CI, Mac dev machines).

    Example return values::

        ['mlx5_0', 'mlx5_1', 'mlx5_2', 'mlx5_3']  # on a DGX Spark proper
        ['rocep1s0f0', 'rocep1s0f1', 'roceP2p1s0f0', 'roceP2p1s0f1']  # GX10
    """
    if not SYS_INFINIBAND.is_dir():
        return []
    try:
        entries = sorted(p.name for p in SYS_INFINIBAND.iterdir())
    except OSError as exc:
        logger.debug("Could not list %s: %s", SYS_INFINIBAND, exc)
        return []
    return [name for name in entries if _HCA_NAME_RE.match(name)]


def hca_has_ipv4_rocev2_gid(hca: str, gid_index: int = 3) -> bool:
    """Return True if the given HCA has a populated IPv4 RoCEv2 GID at ``gid_index``.

    Phase 1 determined that the cluster must use ``NCCL_IB_GID_INDEX=3``,
    which is the IPv4-mapped RoCEv2 slot on our hardware. An HCA is only
    useful for NCCL if *that specific slot* is populated with an IPv4-
    mapped address, not a MAC-based link-local (``fe80:...``).

    We read ``/sys/class/infiniband/<hca>/ports/1/gids/<gid_index>`` and
    pattern-match the contents. Missing file, permission error, or
    unpopulated / link-local value → False. Any IPv4-mapped IPv6 → True.
    """
    gid_path = SYS_INFINIBAND / hca / "ports" / "1" / "gids" / str(gid_index)
    try:
        value = gid_path.read_text().strip()
    except OSError:
        return False
    if not value:
        return False
    # Unpopulated slots often read as all-zero — explicitly reject.
    if value == "0000:0000:0000:0000:0000:0000:0000:0000":
        return False
    return bool(_IPV4_MAPPED_RE.match(value))


def build_nccl_ib_hca_whitelist(
    remote_hca_lists: Optional[List[List[str]]] = None,
    gid_index: int = 3,
) -> str:
    """Return the ``NCCL_IB_HCA`` env-var value for the cluster.

    Builds a comma-separated whitelist that:

    1. Includes every HCA on THIS host that has an IPv4 RoCEv2 GID at
       ``gid_index`` (the "functional" filter from Phase 1).
    2. Unions in every HCA name reported by peers via ``remote_hca_lists``
       (already filtered on the peer side). We cannot validate peer GID
       presence from here; the caller is expected to have done that.
    3. Deduplicates and sorts for determinism (so restarts produce stable
       values).

    The NCCL-side semantics: listing an HCA name that doesn't exist on a
    particular rank is harmless — NCCL uses whichever names match its
    local view. Listing one with a DOWN / MAC-based GID is harmful, which
    is why we filter at the source.

    Example return: ``'mlx5_1,mlx5_3,rocep1s0f1,roceP2p1s0f1'``.

    Returns the empty string when no HCA passes the filter. The caller
    should leave ``NCCL_IB_HCA`` unset in that case; a stale / wrong value
    is worse than letting NCCL auto-detect.
    """
    hcas: set[str] = set()
    for hca in list_local_hcas():
        if hca_has_ipv4_rocev2_gid(hca, gid_index=gid_index):
            hcas.add(hca)

    if remote_hca_lists:
        for remote in remote_hca_lists:
            for name in remote or []:
                if _HCA_NAME_RE.match(name):
                    hcas.add(name)

    return ",".join(sorted(hcas))


def detect_fabric_ip(interface: str) -> Optional[str]:
    """Return the IPv4 address bound to ``interface``, or None if unbound.

    Shells out to ``ip -4 addr show <iface>`` (a five-second-timeout
    subprocess) and extracts the first ``inet <addr>/<cidr>`` line, then
    strips the CIDR suffix.

    Returns None on any failure: missing interface, ``ip`` not in PATH,
    command non-zero, no address assigned.
    """
    if not interface:
        return None
    try:
        out = subprocess.run(
            ["ip", "-4", "addr", "show", interface],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("ip addr show %s failed: %s", interface, exc)
        return None
    if out.returncode != 0:
        return None
    match = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)(?:/\d+)?", out.stdout)
    return match.group(1) if match else None


def probe_path_mtu(peer_ip: str, mgmt_iface: str = "enP7s7") -> int:
    """Probe the path MTU between this node and ``peer_ip``.

    Uses ``ping -M do -s <size>`` binary search to find the largest
    payload (excluding the 28-byte IPv4 + ICMP headers) that reaches the
    peer without fragmentation. Returns the MTU in bytes (payload + 28)
    so callers get a comparable number.

    The search window is [500, 8972] — 500 covers pathological low-MTU
    PPPoE links, 8972 is the jumbo-frame payload ceiling (9000 - 28).

    Returns -1 if the peer is unreachable at the minimum size, which
    means any MTU probing is meaningless.

    Total wall-clock budget is capped at ~5 seconds: each individual
    ping gets 1-second deadline (``-W 1``) and the binary search is at
    most ~5 iterations inside this window.
    """
    if not peer_ip:
        return -1

    # Cheap liveness check first so we fail fast on a down peer.
    if not _ping(peer_ip, size=56, mgmt_iface=mgmt_iface, deadline=1):
        return -1

    lo = 500  # payload
    hi = 8972  # payload (9000 - 28 header bytes)
    # Ensure the upper bound is actually reachable; if not, do the search
    # between lo and the hi we fail at (still useful).
    best = lo
    if _ping(peer_ip, size=hi, mgmt_iface=mgmt_iface, deadline=1):
        return hi + 28  # Full jumbo path.

    while lo <= hi:
        mid = (lo + hi) // 2
        if _ping(peer_ip, size=mid, mgmt_iface=mgmt_iface, deadline=1):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best + 28


def _ping(peer_ip: str, size: int, mgmt_iface: str, deadline: int = 1) -> bool:
    """Single don't-fragment ping; return True on reply, False otherwise.

    ``mgmt_iface`` is used when present to force the outgoing interface;
    on hosts where the iface does not exist ``ping`` will error and we
    treat that as False.
    """
    cmd: List[str] = [
        "ping",
        "-c",
        "1",
        "-M",
        "do",
        "-s",
        str(size),
        "-W",
        str(deadline),
    ]
    if mgmt_iface:
        cmd.extend(["-I", mgmt_iface])
    cmd.append(peer_ip)
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=deadline + 2,
        )
        return out.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


__all__ = [
    "SYS_INFINIBAND",
    "build_nccl_ib_hca_whitelist",
    "detect_fabric_ip",
    "hca_has_ipv4_rocev2_gid",
    "list_local_hcas",
    "probe_path_mtu",
]
