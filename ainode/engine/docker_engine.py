"""DockerEngine — drive vLLM inside the AINode unified container image.

Running *inside* the container (which is how the image ships — systemd on the
host runs `docker run ... ainode` once), this class handles two modes:

* ``start_solo()`` — a single vLLM process on this host only. Direct
  ``vllm serve`` ``Popen``. Used when ``config.distributed_mode == "solo"``.
* ``start_distributed()`` — shells out to ``/opt/spark-vllm-docker/launch-
  cluster.sh`` (baked into the image) to SSH-orchestrate peer workers and
  form a Ray cluster across nodes. Used when ``config.distributed_mode ==
  "head"``. Refuses to run in any other mode.

Public surface mirrors :class:`ainode.engine.vllm_engine.VLLMEngine` —
``start``, ``stop``, ``wait_ready``, ``is_running``, ``health_check``,
``api_url``, ``log_path``, ``process`` — so ``cmd_start``/``cmd_status``
can dispatch polymorphically via a factory.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional

from ainode.core.config import LOGS_DIR, NodeConfig
from ainode.core.gpu import detect_gpu

logger = logging.getLogger(__name__)

# Path to eugr's launcher inside the unified image. See scripts/Dockerfile.ainode.
EUGR_LAUNCHER = Path("/opt/spark-vllm-docker/launch-cluster.sh")
EUGR_ENV_FILE = Path("/opt/spark-vllm-docker/.env")


class DockerEngineError(RuntimeError):
    """Raised when the engine cannot be driven (missing binary, bad config)."""


def build_engine(config: NodeConfig, on_ready: Optional[Callable] = None) -> "DockerEngine":
    """Factory — returns the right engine regardless of mode.

    ``cmd_start`` calls ``engine.start()`` and this factory picks solo vs
    head based on ``config.distributed_mode``. Keeps the call site clean.
    """
    return DockerEngine(config, on_ready=on_ready)


class DockerEngine:
    """Single class that handles solo + head invocation of vLLM.

    Stateful only on the current process instance — the actual engine state
    (GPU memory, Ray cluster, peer containers) lives in the OS.
    """

    def __init__(self, config: NodeConfig, on_ready: Optional[Callable] = None):
        self.config = config
        self.on_ready = on_ready
        self.process: Optional[subprocess.Popen] = None
        self._ready = False
        self._log_thread: Optional[threading.Thread] = None
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._log_file: Path = LOGS_DIR / "vllm.log"
        self._distributed_log: Path = LOGS_DIR / "distributed.log"

    # ------------------------------------------------------------------
    # Public API (lifecycle)
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start the engine in the mode dictated by ``config.distributed_mode``.

        Returns True if the launch sequence was kicked off successfully;
        False if validation failed or subprocess couldn't spawn. Use
        ``wait_ready()`` to block until the API serves.
        """
        mode = (self.config.distributed_mode or "solo").lower()
        if mode == "solo":
            return self.start_solo()
        if mode == "head":
            return self.start_distributed()
        raise DockerEngineError(
            f"Unknown distributed_mode={mode!r}; expected 'solo' or 'head'. "
            "Workers are launched by the head via eugr's launcher — they don't "
            "run a full ainode process directly."
        )

    def start_solo(self) -> bool:
        """Spawn a single-node ``vllm serve`` subprocess."""
        if self.is_running():
            return True

        cmd = self._build_solo_cmd()
        env = self._build_env()

        logger.info("Starting solo vLLM: %s", " ".join(cmd))
        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        self._log_thread = threading.Thread(
            target=self._stream_logs, args=(self.process, self._log_file), daemon=True
        )
        self._log_thread.start()
        return self.process.poll() is None

    def start_distributed(self) -> bool:
        """Invoke eugr's launcher for cross-node TP/PP.

        Requires ``config.distributed_mode == "head"``, a non-empty
        ``peer_ips`` list, and passwordless SSH from this node to every peer.
        """
        if self.config.distributed_mode != "head":
            raise DockerEngineError(
                "start_distributed() only runs when distributed_mode='head'. "
                f"Current mode: {self.config.distributed_mode!r}."
            )
        if not self.config.peer_ips:
            raise DockerEngineError(
                "peer_ips is empty; cannot launch distributed cluster without peers."
            )
        if not EUGR_LAUNCHER.exists():
            raise DockerEngineError(
                f"eugr launcher missing at {EUGR_LAUNCHER}. Is this running inside the ainode image?"
            )

        self._write_eugr_env()
        launch_script = self._write_distributed_launch_script()

        cmd = [str(EUGR_LAUNCHER), "--launch-script", str(launch_script)]
        extra_docker_args = [
            "-v",
            f"{self.config.models_dir or '/root/.ainode/models'}:/models",
        ]
        env = self._build_env()
        env["VLLM_SPARK_EXTRA_DOCKER_ARGS"] = " ".join(extra_docker_args)

        logger.info(
            "Starting distributed vLLM: TP=%d across %d peers via eugr launcher",
            self._tp_size(),
            len(self.config.peer_ips),
        )
        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            cwd=str(EUGR_LAUNCHER.parent),
        )
        self._log_thread = threading.Thread(
            target=self._stream_logs,
            args=(self.process, self._distributed_log),
            daemon=True,
        )
        self._log_thread.start()
        return self.process.poll() is None

    def stop(self) -> None:
        """Graceful shutdown. For distributed, also invokes eugr's ``stop``."""
        if self.process and self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            self.process = None

        if self.config.distributed_mode == "head" and EUGR_LAUNCHER.exists():
            try:
                subprocess.run(
                    [str(EUGR_LAUNCHER), "stop"],
                    cwd=str(EUGR_LAUNCHER.parent),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except Exception:  # pragma: no cover - best-effort teardown
                logger.exception("eugr launch-cluster.sh stop failed")

        self._ready = False

    def wait_ready(self, timeout: int = 300) -> bool:
        """Poll ``/v1/models`` on the API port until 2xx or timeout."""
        url = f"http://127.0.0.1:{self.config.api_port}/v1/models"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process and self.process.poll() is not None:
                # Subprocess died before ever serving.
                return False
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if 200 <= resp.status < 300:
                        self._ready = True
                        if self.on_ready:
                            try:
                                self.on_ready()
                            except Exception:  # pragma: no cover
                                logger.exception("on_ready callback failed")
                        return True
            except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, OSError):
                pass
            time.sleep(2)
        return False

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def health_check(self) -> dict:
        """Used by ``ainode status``. Mirrors VLLMEngine.health_check."""
        result = {
            "process_alive": self.is_running(),
            "api_responding": False,
            "models_loaded": [],
        }
        try:
            url = f"http://127.0.0.1:{self.config.api_port}/v1/models"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                result["api_responding"] = True
                result["models_loaded"] = [m["id"] for m in data.get("data", [])]
        except Exception:
            pass
        return result

    def logs(self, n: int = 100) -> str:
        log = self._distributed_log if self.config.distributed_mode == "head" else self._log_file
        if not log.exists():
            return ""
        try:
            lines = log.read_text().splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-n:])

    # ------------------------------------------------------------------
    # Compatibility properties (VLLMEngine parity)
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def api_url(self) -> str:
        return f"http://localhost:{self.config.api_port}/v1"

    @property
    def log_path(self) -> Path:
        return (
            self._distributed_log
            if self.config.distributed_mode == "head"
            else self._log_file
        )

    # ------------------------------------------------------------------
    # Command construction
    # ------------------------------------------------------------------

    def _build_solo_cmd(self) -> List[str]:
        """Assemble ``vllm serve`` args mirroring VLLMEngine.build_cmd."""
        gpu = detect_gpu()
        cmd: List[str] = [
            "vllm",
            "serve",
            self.config.model,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.config.api_port),
            "--gpu-memory-utilization",
            str(self.config.gpu_memory_utilization),
        ]
        if self.config.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.config.max_model_len:
            cmd.extend(["--max-model-len", str(self.config.max_model_len)])
        if self.config.quantization:
            cmd.extend(["--quantization", self.config.quantization])
        if self.config.models_dir:
            cmd.extend(["--download-dir", self.config.models_dir])
        if gpu and gpu.unified_memory:
            cmd.extend(["--dtype", "bfloat16"])
        return cmd

    def _build_env(self) -> dict:
        """Base environment for subprocess calls, including NCCL/RoCE tuning."""
        env = os.environ.copy()
        env.setdefault("HF_HOME", self.config.models_dir or "/root/.ainode/models")

        iface = self.config.cluster_interface
        if iface:
            env["NCCL_SOCKET_IFNAME"] = iface
            env["GLOO_SOCKET_IFNAME"] = iface
            env["UCX_NET_DEVICES"] = iface

        # Prefer RDMA over socket for cross-node; NCCL will fall back if
        # the device isn't actually RDMA-capable.
        ib_hca = self._detect_ib_hca() or "mlx5_0"
        env.setdefault("NCCL_IB_HCA", ib_hca)
        env.setdefault("NCCL_IB_DISABLE", "0")
        env.setdefault("NCCL_P2P_DISABLE", "0")
        env.setdefault("NCCL_NET_GDR_LEVEL", "5")
        env.setdefault("NCCL_IGNORE_CPU_AFFINITY", "1")

        return env

    @staticmethod
    def _detect_ib_hca() -> Optional[str]:
        """Return a comma-joined mlx5_* list from ``ibdev2netdev`` or None."""
        if shutil.which("ibdev2netdev") is None:
            return None
        try:
            out = subprocess.run(
                ["ibdev2netdev"], capture_output=True, text=True, timeout=5
            )
            if out.returncode != 0:
                return None
            devs = [
                line.split()[0]
                for line in out.stdout.splitlines()
                if line.strip().startswith("mlx5_")
            ]
            return ",".join(devs) or None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Distributed (eugr) wiring
    # ------------------------------------------------------------------

    def _tp_size(self) -> int:
        """Total TP = local GPUs × (1 + number of peers). One GPU per GB10 node."""
        return 1 + len(self.config.peer_ips)

    def _write_eugr_env(self) -> None:
        """Populate ``/opt/spark-vllm-docker/.env`` for the launcher."""
        iface = self.config.cluster_interface
        head_ip = _local_ip_for_interface(iface)
        cluster_nodes = ",".join([head_ip] + self.config.peer_ips)

        ib_hca = self._detect_ib_hca() or "mlx5_0"

        lines = [
            f"CLUSTER_NODES={cluster_nodes}",
            f"ETH_IF={iface}",
            f"IB_IF={iface}",
            "MASTER_PORT=29501",
            f"SSH_USER={self.config.ssh_user}",
            # CONTAINER_* vars are injected into every per-node container.
            "CONTAINER_NCCL_DEBUG=INFO",
            "CONTAINER_NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH",
            f"CONTAINER_NCCL_SOCKET_IFNAME={iface}",
            f"CONTAINER_NCCL_IB_HCA={ib_hca}",
            "CONTAINER_NCCL_IB_DISABLE=0",
            "CONTAINER_NCCL_IGNORE_CPU_AFFINITY=1",
            "CONTAINER_NCCL_NET_GDR_LEVEL=5",
            f"CONTAINER_UCX_NET_DEVICES={iface}",
        ]
        EUGR_ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
        EUGR_ENV_FILE.write_text("\n".join(lines) + "\n")

    def _write_distributed_launch_script(self) -> Path:
        """Emit a ``vllm serve`` script for eugr to execute inside the container."""
        gpu = detect_gpu()
        dtype_line = ""
        if gpu and gpu.unified_memory:
            dtype_line = "    --dtype bfloat16 \\\n"

        extra = ""
        if self.config.max_model_len:
            extra += f"    --max-model-len {self.config.max_model_len} \\\n"
        if self.config.quantization:
            extra += f"    --quantization {self.config.quantization} \\\n"

        # Note: --enforce-eager intentionally omitted. CUDA graphs add ~60s
        # to initial warmup but give 2-3x steady-state throughput, which
        # matters far more for the user than first-token latency. If the
        # graphs capture fails on a particular model, we can re-add eager
        # per-invocation via config.
        script = f"""#!/bin/bash
# Auto-generated by ainode DockerEngine.start_distributed. Do not edit.
vllm serve {self.config.model} \\
    --host 0.0.0.0 --port {self.config.api_port} \\
    --distributed-executor-backend ray \\
    --tensor-parallel-size {self._tp_size()} \\
    --pipeline-parallel-size 1 \\
    --gpu-memory-utilization {self.config.gpu_memory_utilization} \\
{dtype_line}{extra}    --download-dir {self.config.models_dir or '/models'}
"""
        target = EUGR_LAUNCHER.parent / "examples" / "ainode-distributed.sh"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(script)
        target.chmod(0o755)
        return target

    # ------------------------------------------------------------------
    # Log streaming
    # ------------------------------------------------------------------

    def _stream_logs(self, process: subprocess.Popen, target: Path) -> None:
        """Tee subprocess stdout to a log file, watching for readiness."""
        if not process.stdout:
            return
        with open(target, "a") as sink:
            for line in process.stdout:
                sink.write(line)
                sink.flush()
                if not self._ready and (
                    "Uvicorn running on" in line
                    or "Application startup complete" in line
                ):
                    self._ready = True
                    if self.on_ready:
                        try:
                            self.on_ready()
                        except Exception:  # pragma: no cover
                            logger.exception("on_ready callback failed")


def _local_ip_for_interface(iface: Optional[str]) -> str:
    """Return the IPv4 address on ``iface`` — or fall back to hostname."""
    if not iface:
        return "127.0.0.1"
    try:
        out = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", "dev", iface],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                parts = line.split()
                if "inet" in parts:
                    idx = parts.index("inet")
                    return parts[idx + 1].split("/")[0]
    except Exception:
        pass
    import socket as _socket
    try:
        return _socket.gethostbyname(_socket.gethostname())
    except Exception:
        return "127.0.0.1"
