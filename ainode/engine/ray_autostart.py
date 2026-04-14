"""Ray cluster autostart — wires node role → head/worker Ray lifecycle.

This module is intentionally decoupled from the aiohttp server so the CLI
``ainode start`` command can also drive Ray bring-up.

Behaviour:
    - Master node  → start_ray_head() if ray is installed
    - Worker node  → wait until a master is discovered, then join_ray_cluster()
    - If ``ray`` is not installed, log a warning and stay in single-node mode
"""

from __future__ import annotations

import asyncio
import logging
import socket
from dataclasses import dataclass, field
from typing import Optional

from ainode.discovery.cluster import ClusterState
from ainode.engine.ray_setup import (
    RAY_DEFAULT_PORT,
    is_ray_available,
    join_ray_cluster,
    start_ray_head,
)

logger = logging.getLogger(__name__)


@dataclass
class RayAutostartState:
    """Tracks the live Ray autostart status for the local node."""

    enabled: bool = True
    is_head: bool = False
    joined_as_worker: bool = False
    head_address: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0
    last_attempt_ts: float = 0.0

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "is_head": self.is_head,
            "joined_as_worker": self.joined_as_worker,
            "head_address": self.head_address,
            "error": self.error,
            "attempts": self.attempts,
        }


def _gpu_count() -> Optional[int]:
    """Best-effort detect number of GPUs to expose to Ray."""
    try:
        from ainode.core.gpu import detect_gpu  # local import
        gpu = detect_gpu()
        return 1 if gpu else 0
    except Exception:  # pragma: no cover - defensive
        return None


def _format_head_address(port: int = RAY_DEFAULT_PORT) -> str:
    """Figure out an advertisable head address (best guess)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = "127.0.0.1"
    return f"{ip}:{port}"


def start_head_if_needed(state: RayAutostartState) -> RayAutostartState:
    """Start a Ray head on this node. Idempotent."""
    if state.is_head and state.head_address:
        return state
    if not is_ray_available():
        state.enabled = False
        state.error = "ray not installed"
        logger.warning("Ray not installed — distributed inference disabled (pip install 'ray[default]')")
        return state
    try:
        addr = start_ray_head(num_gpus=_gpu_count())
        state.is_head = True
        state.head_address = addr or _format_head_address()
        state.error = None
        logger.info("Ray head started at %s", state.head_address)
    except Exception as exc:
        state.error = str(exc)
        logger.warning("Failed to start Ray head: %s", exc)
    return state


def join_worker_if_possible(state: RayAutostartState, head_address: str) -> RayAutostartState:
    """Join the given Ray head as a worker. Idempotent."""
    if state.joined_as_worker and state.head_address == head_address:
        return state
    if not is_ray_available():
        state.enabled = False
        state.error = "ray not installed"
        logger.warning("Ray not installed — cannot join cluster (pip install 'ray[default]')")
        return state
    try:
        join_ray_cluster(head_address, num_gpus=_gpu_count())
        state.joined_as_worker = True
        state.head_address = head_address
        state.error = None
        logger.info("Joined Ray cluster at %s", head_address)
    except Exception as exc:
        state.error = str(exc)
        logger.warning("Failed to join Ray cluster at %s: %s", head_address, exc)
    return state


async def autostart_loop(
    cluster_state: ClusterState,
    get_master_address,
    state: RayAutostartState,
    poll_seconds: float = 5.0,
) -> None:
    """Background coroutine — keep Ray aligned with cluster role.

    Parameters
    ----------
    cluster_state : ClusterState
        Live cluster state; used to check ``is_master_of_cluster()`` and
        discover the master peer.
    get_master_address : Callable[[], Optional[str]]
        Returns the discovered master's Ray address (``host:6379``) once known.
    state : RayAutostartState
        Mutable state object; callers can read ``state.to_dict()`` for status.
    """
    if not is_ray_available():
        state.enabled = False
        state.error = "ray not installed"
        logger.warning("Ray not installed — distributed inference disabled")
        return

    try:
        while True:
            try:
                if cluster_state.is_master_of_cluster():
                    if not state.is_head:
                        start_head_if_needed(state)
                else:
                    master_addr = get_master_address()
                    if master_addr and not state.joined_as_worker:
                        join_worker_if_possible(state, master_addr)
            except Exception as exc:  # pragma: no cover - defensive
                state.error = str(exc)
                logger.exception("Ray autostart loop error: %s", exc)
            state.attempts += 1
            await asyncio.sleep(poll_seconds)
    except asyncio.CancelledError:
        return
