"""Ray cluster helpers for multi-node distributed inference."""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

RAY_DEFAULT_PORT = 6379
RAY_DASHBOARD_PORT = 8265


@dataclass
class RayStatus:
    """Snapshot of Ray cluster health."""
    running: bool
    is_head: bool
    head_address: Optional[str]
    num_nodes: int
    total_cpus: float
    total_gpus: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "running": self.running,
            "is_head": self.is_head,
            "head_address": self.head_address,
            "num_nodes": self.num_nodes,
            "total_cpus": self.total_cpus,
            "total_gpus": self.total_gpus,
            "error": self.error,
        }


def _ray_executable() -> str:
    """Return path to ray CLI, or raise if not installed."""
    ray_path = shutil.which("ray")
    if not ray_path:
        raise RuntimeError(
            "Ray is not installed. Install with: pip install 'ray[default]'"
        )
    return ray_path


def is_ray_available() -> bool:
    """Return True if the ``ray`` CLI is present on PATH."""
    return shutil.which("ray") is not None


def start_ray_head(
    port: int = RAY_DEFAULT_PORT,
    dashboard_port: int = RAY_DASHBOARD_PORT,
    num_gpus: Optional[int] = None,
) -> str:
    """Start a Ray head node on this machine.

    Returns the address string (e.g. '192.168.1.10:6379') for workers to join.
    """
    ray_bin = _ray_executable()
    cmd = [
        ray_bin, "start", "--head",
        "--port", str(port),
        "--dashboard-port", str(dashboard_port),
    ]
    if num_gpus is not None:
        cmd.extend(["--num-gpus", str(num_gpus)])

    logger.info("Starting Ray head node: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start Ray head: {result.stderr.strip()}")

    # Parse the address from output
    for line in result.stdout.splitlines():
        if "ray start --address=" in line:
            parts = line.split("ray start --address=")
            if len(parts) > 1:
                addr = parts[1].strip().strip("'\"")
                logger.info("Ray head started at %s", addr)
                return addr

    # Fallback: construct address
    address = f"127.0.0.1:{port}"
    logger.info("Ray head started (fallback address: %s)", address)
    return address


def join_ray_cluster(
    head_address: str,
    num_gpus: Optional[int] = None,
) -> bool:
    """Join this node to an existing Ray cluster as a worker.

    Parameters
    ----------
    head_address : str
        The head node address, e.g. '192.168.1.10:6379'.
    num_gpus : int | None
        Number of GPUs to expose to Ray on this worker.

    Returns True on success.
    """
    ray_bin = _ray_executable()
    cmd = [ray_bin, "start", f"--address={head_address}"]
    if num_gpus is not None:
        cmd.extend(["--num-gpus", str(num_gpus)])

    logger.info("Joining Ray cluster at %s: %s", head_address, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to join Ray cluster: {result.stderr.strip()}")

    logger.info("Successfully joined Ray cluster at %s", head_address)
    return True


def get_ray_status() -> RayStatus:
    """Check Ray cluster health and return a status snapshot."""
    ray_bin_path = shutil.which("ray")
    if not ray_bin_path:
        return RayStatus(
            running=False, is_head=False, head_address=None,
            num_nodes=0, total_cpus=0, total_gpus=0,
            error="Ray not installed",
        )

    try:
        result = subprocess.run(
            [ray_bin_path, "status"],
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        return RayStatus(
            running=False, is_head=False, head_address=None,
            num_nodes=0, total_cpus=0, total_gpus=0,
            error="Ray status check timed out",
        )

    if result.returncode != 0:
        return RayStatus(
            running=False, is_head=False, head_address=None,
            num_nodes=0, total_cpus=0, total_gpus=0,
            error=result.stderr.strip() or "Ray not running",
        )

    # Parse output for basic info
    stdout = result.stdout
    num_nodes = 0
    total_cpus = 0.0
    total_gpus = 0.0

    for line in stdout.splitlines():
        line_stripped = line.strip()
        if "node" in line_stripped.lower() and "alive" in line_stripped.lower():
            parts = line_stripped.split()
            for p in parts:
                if p.isdigit():
                    num_nodes = int(p)
                    break
        if "CPU" in line_stripped:
            for token in line_stripped.split():
                try:
                    total_cpus = float(token)
                    break
                except ValueError:
                    continue
        if "GPU" in line_stripped:
            for token in line_stripped.split():
                try:
                    total_gpus = float(token)
                    break
                except ValueError:
                    continue

    return RayStatus(
        running=True,
        is_head=True,
        head_address=None,
        num_nodes=max(num_nodes, 1),
        total_cpus=total_cpus,
        total_gpus=total_gpus,
    )


def stop_ray() -> bool:
    """Gracefully shut down Ray on this node.

    Returns True on success.
    """
    ray_bin_path = shutil.which("ray")
    if not ray_bin_path:
        return True  # Nothing to stop

    logger.info("Stopping Ray...")
    try:
        result = subprocess.run(
            [ray_bin_path, "stop"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            logger.info("Ray stopped successfully")
            return True
        logger.warning("Ray stop returned non-zero: %s", result.stderr.strip())
        return False
    except subprocess.TimeoutExpired:
        logger.error("Ray stop timed out")
        return False
