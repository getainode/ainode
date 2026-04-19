"""NvidiaBackend — drive vLLM via NVIDIA's official ``nvcr.io/nvidia/vllm`` image.

Phase 4 implementation. Mirrors the public lifecycle surface of
:class:`ainode.engine.backends.eugr.EugrBackend` so ``cmd_start`` /
``cmd_status`` / the dashboard can dispatch polymorphically. Internally,
very different from eugr:

* **Solo mode** — a single ``docker run nvcr.io/nvidia/vllm:26.02-py3 \\
  vllm serve ...`` on this host. No Ray, no run_cluster.sh. Environment
  is populated from :mod:`ainode.cluster.hca_discovery` so NCCL sees the
  correct HCA + fabric IP without manual tuning.

* **Distributed (head) mode** — launches the head container via
  ``scripts/run_cluster.sh --head`` (vendored by Agent B), SSHes to each
  peer to run ``run_cluster.sh --worker``, then ``docker exec``s into
  the local head container to start ``vllm serve`` with
  ``--tensor-parallel-size N``. This mirrors runbook 02 § Steps 4-7
  exactly.

AINode's own process continues to run outside the vLLM container; the
backend only orchestrates docker + ssh + docker-exec. All env vars come
from :meth:`_build_nccl_env` which consults :mod:`hca_discovery` — no
hardcoded HCA names, no hardcoded fabric IPs.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ainode.cluster.hca_discovery import (
    build_nccl_ib_hca_whitelist,
    detect_fabric_ip,
    list_local_hcas,
)
from ainode.core.config import LOGS_DIR, NodeConfig
from ainode.engine.backends.base import EngineBackend

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Module-level constants
# -----------------------------------------------------------------------------

NVIDIA_VLLM_IMAGE = "nvcr.io/nvidia/vllm:26.02-py3"

# Agent B vendors ``scripts/run_cluster.sh`` into the AINode install at
# ``/opt/ainode/run_cluster.sh``. For dev environments where that install
# isn't present, fall back to ``/tmp/run_cluster.sh`` so operators can
# ``wget`` the script into /tmp and test without a full install.
RUN_CLUSTER_SCRIPT_SOURCE = Path("/opt/ainode/run_cluster.sh")
RUN_CLUSTER_SCRIPT_FALLBACK = Path("/tmp/run_cluster.sh")

# run_cluster.sh names its container ``node-<N>`` where N is a small index.
# We use a stable prefix so ``stop()`` can find the container to kill.
RAY_CONTAINER_NAME_PREFIX = "ainode-vllm-node"

# NCCL tuning from Phase 1 floor verification — see
# ops/slices/nvidia-vllm-engine/runbooks/01-nccl-floor-verification.md.
NCCL_IB_GID_INDEX = "3"
MASTER_PORT = "29501"


class NvidiaBackendError(RuntimeError):
    """Raised when the backend cannot be driven (missing image, bad config)."""


class NvidiaBackend(EngineBackend):
    """Drive NVIDIA's vLLM image via plain ``docker run`` + optional ssh/run_cluster.sh.

    Stateful only on the current process instance — the container state,
    Ray cluster, and peer containers live outside AINode. ``stop()``
    fans out to peers over SSH to tear them down.
    """

    def __init__(self, config: NodeConfig, on_ready: Optional[Callable] = None):
        self.config = config
        self.on_ready = on_ready
        self._process: Optional[subprocess.Popen] = None
        self._ready = False
        self._log_thread: Optional[threading.Thread] = None
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._log_file: Path = LOGS_DIR / "nvidia-vllm.log"
        self._distributed_log: Path = LOGS_DIR / "nvidia-distributed.log"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Dispatch to solo / distributed based on ``config.distributed_mode``."""
        mode = (self.config.distributed_mode or "solo").lower()
        if mode == "solo":
            return self.start_solo()
        if mode == "head":
            return self.start_distributed()
        raise NvidiaBackendError(
            f"Unknown distributed_mode={mode!r}; expected 'solo' or 'head'. "
            "Workers are launched via ssh+run_cluster.sh by the head — they "
            "don't run a full ainode process directly."
        )

    def start_solo(self) -> bool:
        """Launch a single-node vLLM container on this host.

        No Ray, no run_cluster.sh. Direct ``docker run
        nvcr.io/nvidia/vllm:26.02-py3 vllm serve <model> ...``.
        """
        if self.is_running():
            return True

        container_name = self._solo_container_name()
        cmd = self._build_solo_docker_cmd(container_name)
        env = self._build_env_for_subprocess()

        logger.info(
            "Starting NVIDIA solo vLLM: docker run %s vllm serve %s",
            NVIDIA_VLLM_IMAGE,
            self.config.model,
        )
        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        self._log_thread = threading.Thread(
            target=self._stream_logs,
            args=(self._process, self._log_file),
            daemon=True,
        )
        self._log_thread.start()
        return self._process.poll() is None

    def start_distributed(self) -> bool:
        """Launch a distributed TP/PP cluster across ``config.peer_ips``.

        Steps (mirrors runbook 02 § Steps 4-7):

        1. Validate we're ``distributed_mode == "head"`` with peers configured
           and ``run_cluster.sh`` is vendored.
        2. Run ``run_cluster.sh --head`` locally via ``Popen``. This does
           ``docker run -d`` of the NVIDIA image with ``ray start --head``
           as its PID-1. Env vars built via :meth:`_build_nccl_env`.
        3. SSH to each peer and run ``run_cluster.sh --worker`` with the
           same image + env, pointing it at our head fabric IP.
        4. ``docker exec`` into the local head container to invoke
           ``vllm serve --tensor-parallel-size N`` with N = 1 + len(peers).

        The ``Popen`` handle we keep is for the ``vllm serve`` exec (step 4);
        the Ray containers on head + peers are managed by docker itself and
        cleaned up in :meth:`stop`.
        """
        if self.config.distributed_mode != "head":
            raise NvidiaBackendError(
                "start_distributed() only runs when distributed_mode='head'. "
                f"Current mode: {self.config.distributed_mode!r}."
            )
        if not self.config.peer_ips:
            raise NvidiaBackendError(
                "peer_ips is empty; cannot launch distributed cluster without peers."
            )

        script = self._locate_run_cluster_script()
        if script is None:
            raise NvidiaBackendError(
                f"run_cluster.sh missing at both {RUN_CLUSTER_SCRIPT_SOURCE} and "
                f"{RUN_CLUSTER_SCRIPT_FALLBACK}. Agent B's vendoring step "
                "must land before NvidiaBackend distributed mode can run."
            )

        fabric_ip = self._head_fabric_ip()
        if fabric_ip is None:
            raise NvidiaBackendError(
                f"Could not detect fabric IP on interface "
                f"{self.config.cluster_interface!r}. Is the NIC up?"
            )

        hf_cache = self._head_hf_cache()

        # Step 2 — head Ray container.
        head_cmd = self._build_run_cluster_cmd(
            script=script,
            role="head",
            head_ip=fabric_ip,
            hf_cache_dir=hf_cache,
            fabric_ip=fabric_ip,
        )
        logger.info("Launching head Ray container: %s", " ".join(head_cmd))
        head_result = subprocess.run(
            head_cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if head_result.returncode != 0:
            raise NvidiaBackendError(
                "run_cluster.sh --head failed "
                f"(rc={head_result.returncode}): {head_result.stderr.strip()}"
            )

        # Step 3 — peer Ray workers over SSH.
        for peer_ip in self.config.peer_ips:
            self._ssh_launch_worker(
                script_path=script,
                peer_ip=peer_ip,
                head_ip=fabric_ip,
            )

        # Step 4 — docker exec into local head container to start vllm serve.
        vllm_cmd = self._build_vllm_exec_cmd(tp_size=self._tp_size())
        env = self._build_env_for_subprocess()

        logger.info(
            "Starting distributed vllm serve: TP=%d across head + %d peers",
            self._tp_size(),
            len(self.config.peer_ips),
        )
        self._process = subprocess.Popen(
            vllm_cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        self._log_thread = threading.Thread(
            target=self._stream_logs,
            args=(self._process, self._distributed_log),
            daemon=True,
        )
        self._log_thread.start()
        return self._process.poll() is None

    def launch_distributed(self, sharding_config=None) -> bool:
        """Shim for the ``/api/models/load`` dashboard path.

        Mirrors :meth:`EugrBackend.launch_distributed` so
        ``ainode/models/api_routes.py`` (which reaches through this method
        name regardless of backend) works when ``engine_backend='nvidia'``.
        Applies the sharding config's model + peer_ips onto ``self.config``
        and flips to head mode if needed, then delegates to
        :meth:`start_distributed`.
        """
        if sharding_config is not None:
            if getattr(sharding_config, "model", None):
                self.config.model = sharding_config.model
            if getattr(sharding_config, "peer_ips", None):
                self.config.peer_ips = sharding_config.peer_ips

        if self.config.distributed_mode != "head":
            self.config.distributed_mode = "head"
            try:
                self.config.save()
            except Exception:  # pragma: no cover - best-effort persist
                pass

        return self.start_distributed()

    def stop(self) -> None:
        """Stop the vllm serve process + fan out to peers to kill their Ray containers."""
        if self._process and self._process.poll() is None:
            self._process.send_signal(signal.SIGTERM)
            try:
                self._process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._process.kill()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            self._process = None

        # Tear down local Ray / vllm container by name. Best-effort.
        container = self._solo_container_name() if self.config.distributed_mode == "solo" else self._head_container_name()
        self._docker_stop_best_effort(container)

        # For distributed: SSH to each peer and stop their worker container.
        if self.config.distributed_mode == "head":
            for peer_ip in self.config.peer_ips:
                self._ssh_stop_peer_container(peer_ip)

        self._ready = False

    def wait_ready(self, timeout: float = 600.0) -> bool:
        """Poll ``/v1/models`` on the API port until 2xx or timeout."""
        url = f"http://127.0.0.1:{self.config.api_port}/v1/models"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._process and self._process.poll() is not None:
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
        return self._process is not None and self._process.poll() is None

    def health_check(self) -> dict:
        """Mirrors EugrBackend.health_check for dashboard parity."""
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

    # ------------------------------------------------------------------
    # Properties
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

    @property
    def process(self) -> Optional[subprocess.Popen]:
        return self._process

    @process.setter
    def process(self, value: Optional[subprocess.Popen]) -> None:
        # Preserve mutability so tests can inject mock Popen instances.
        self._process = value

    # ------------------------------------------------------------------
    # Env construction — the hca_discovery integration point
    # ------------------------------------------------------------------

    def _build_nccl_env(
        self,
        is_head: bool = True,
        head_fabric_ip: Optional[str] = None,
    ) -> Dict[str, str]:
        """Build the NCCL/Ray env vars for the NVIDIA container.

        This is the single source of truth for what env vars land in the
        docker run / docker exec commands. Values come from Phase 1
        runbook + hca_discovery:

        * ``VLLM_HOST_IP``, ``MASTER_ADDR`` — fabric IP of *this* node
          (for head) or the head (for peers). Never hardcoded.
        * Ray / UCX / Gloo / Torch socket iface — all set to
          ``config.cluster_interface`` so no process falls back to the
          default route.
        * ``NCCL_IB_HCA`` — whitelist built dynamically from local sysfs.
          Remote HCA lists are NOT threaded here yet; distributed mode
          uses the local view (it's what every peer also uses for their
          own view, so the union happens naturally in NCCL).
        * ``NCCL_IB_GID_INDEX=3`` — per Phase 1. Hardcoded because
          every DGX Spark + GX10 we've tested uses the same slot.
        * ``HF_HUB_ENABLE_HF_TRANSFER=1`` — always on, per install-UX spec.
        """
        iface = self.config.cluster_interface or ""
        fabric_ip = detect_fabric_ip(iface) or "127.0.0.1"
        hca = build_nccl_ib_hca_whitelist()

        master_addr = fabric_ip if is_head else (head_fabric_ip or fabric_ip)

        env: Dict[str, str] = {
            "VLLM_HOST_IP": fabric_ip,
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": MASTER_PORT,
            "UCX_NET_DEVICES": iface,
            "NCCL_SOCKET_IFNAME": iface,
            "OMPI_MCA_btl_tcp_if_include": iface,
            "GLOO_SOCKET_IFNAME": iface,
            "TP_SOCKET_IFNAME": iface,
            "RAY_memory_monitor_refresh_ms": "0",
            "NCCL_IB_GID_INDEX": NCCL_IB_GID_INDEX,
            "NCCL_IB_SUBNET_AWARE_ROUTING": "1",
            "NCCL_IB_DISABLE": "0",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_TOKEN": self.config.hf_token or "",
        }
        if hca:
            env["NCCL_IB_HCA"] = hca
        return env

    def _build_env_for_subprocess(self) -> Dict[str, str]:
        """OS-level env for the docker CLI + docker exec subprocess.

        The NCCL vars must land INSIDE the container; for ``docker run``
        we pass them via ``-e``. For ``docker exec`` we likewise pass
        ``-e``. This helper returns only the env the outer subprocess
        (``docker`` itself) needs — mostly inherited from ``os.environ``
        with ``HF_TOKEN`` forwarded so any prompt that reads from it
        works.
        """
        env = os.environ.copy()
        if self.config.hf_token:
            env["HF_TOKEN"] = self.config.hf_token
            env["HUGGING_FACE_HUB_TOKEN"] = self.config.hf_token
        return env

    # ------------------------------------------------------------------
    # Docker command builders
    # ------------------------------------------------------------------

    def _solo_container_name(self) -> str:
        return f"{RAY_CONTAINER_NAME_PREFIX}-solo"

    def _head_container_name(self) -> str:
        # run_cluster.sh names the container ``node-0`` on the head by
        # default. We publish that name via our own launcher wrapper so
        # ``docker exec`` / ``docker stop`` are deterministic.
        return f"{RAY_CONTAINER_NAME_PREFIX}-head"

    def _build_solo_docker_cmd(self, container_name: str) -> List[str]:
        """Single-container solo mode — ``docker run ... vllm serve ...``.

        Not using run_cluster.sh here; that script always wires up Ray,
        which is overkill (and adds ~30s boot time) for a single-node
        vLLM process.
        """
        nccl_env = self._build_nccl_env(is_head=True)
        hf_cache = self._head_hf_cache()

        cmd: List[str] = [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            container_name,
            "--gpus",
            "all",
            "--network",
            "host",
            "--ipc=host",
            "--pid=host",
            "--shm-size",
            "10.24g",
            "-v",
            f"{hf_cache}:/root/.cache/huggingface",
        ]
        for key, value in nccl_env.items():
            cmd.extend(["-e", f"{key}={value}"])

        cmd.extend([NVIDIA_VLLM_IMAGE, "vllm", "serve", self.config.model])
        cmd.extend(self._build_vllm_serve_args(tp_size=1))
        return cmd

    def _build_vllm_serve_args(self, tp_size: int) -> List[str]:
        """Assemble the positional ``vllm serve`` args after ``<model>``."""
        args: List[str] = [
            "--host", "0.0.0.0",
            "--port", str(self.config.api_port),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
        ]
        if tp_size > 1:
            args.extend(["--tensor-parallel-size", str(tp_size)])
            args.extend(["--distributed-executor-backend", "ray"])
        if self.config.max_model_len:
            args.extend(["--max-model-len", str(self.config.max_model_len)])
        if self.config.quantization:
            args.extend(["--quantization", self.config.quantization])
        if self.config.trust_remote_code:
            args.append("--trust-remote-code")
        return args

    def _build_run_cluster_cmd(
        self,
        script: Path,
        role: str,
        head_ip: str,
        hf_cache_dir: str,
        fabric_ip: str,
    ) -> List[str]:
        """Construct the ``bash run_cluster.sh`` invocation for head or worker.

        Matches runbook 02 § Step 4 (head) / Step 5 (worker) verbatim —
        the positional args are: IMAGE, HEAD_IP, --head|--worker, HF_CACHE,
        followed by any number of ``-e KEY=VALUE`` repeated pairs that
        the script forwards to ``docker run`` inside itself.

        The ``role`` must be ``"head"`` or ``"worker"``.
        """
        if role not in {"head", "worker"}:
            raise ValueError(f"role must be 'head' or 'worker', got {role!r}")

        is_head = role == "head"
        nccl_env = self._build_nccl_env(is_head=is_head, head_fabric_ip=head_ip)

        cmd: List[str] = [
            "bash", str(script),
            NVIDIA_VLLM_IMAGE,
            head_ip,
            f"--{role}",
            hf_cache_dir,
        ]
        for key, value in nccl_env.items():
            cmd.extend(["-e", f"{key}={value}"])
        return cmd

    def _build_vllm_exec_cmd(self, tp_size: int) -> List[str]:
        """Build the ``docker exec`` command that launches ``vllm serve``.

        Runs INSIDE the already-started head Ray container. Ray picks up
        the peer workers automatically via the cluster address embedded
        in the container env by run_cluster.sh.
        """
        head = self._head_container_name()
        inner = ["vllm", "serve", self.config.model]
        inner.extend(self._build_vllm_serve_args(tp_size=tp_size))

        # Wrap the command in bash so stdout/stderr line-buffer correctly.
        # docker exec -i lets us stream logs back; -d would detach.
        cmd: List[str] = ["docker", "exec", "-i", head, "bash", "-lc", " ".join(shlex.quote(p) for p in inner)]
        return cmd

    # ------------------------------------------------------------------
    # SSH helpers for distributed mode
    # ------------------------------------------------------------------

    def _ssh_launch_worker(
        self,
        script_path: Path,
        peer_ip: str,
        head_ip: str,
    ) -> None:
        """SSH to ``peer_ip`` and run ``run_cluster.sh --worker`` there.

        Assumes the script has been propagated to the peer at the same
        path (``/opt/ainode/run_cluster.sh``) via Agent B's install, and
        that ``ssh_user`` has passwordless SSH access to the peer.
        """
        nccl_env = self._build_nccl_env(is_head=False, head_fabric_ip=head_ip)
        env_args: List[str] = []
        for key, value in nccl_env.items():
            env_args.extend(["-e", f"{key}={value}"])

        # Reasonable per-peer HF cache path. Workers can't always write to
        # NFS (runbook 02 § Observations / gotcha 2), so default to a
        # home-directory path.
        peer_hf_cache = "~/ainode-nvidia-cache"

        remote_cmd_parts: List[str] = [
            "mkdir", "-p", peer_hf_cache, "&&",
            "bash", str(script_path),
            NVIDIA_VLLM_IMAGE,
            head_ip,
            "--worker",
            peer_hf_cache,
            *env_args,
        ]
        remote_cmd = " ".join(shlex.quote(p) for p in remote_cmd_parts)

        ssh_target = f"{self.config.ssh_user}@{peer_ip}"
        ssh_cmd = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            ssh_target,
            remote_cmd,
        ]
        logger.info("SSH-launching worker on %s", peer_ip)
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            raise NvidiaBackendError(
                f"run_cluster.sh --worker on {peer_ip} failed "
                f"(rc={result.returncode}): {result.stderr.strip()}"
            )

    def _ssh_stop_peer_container(self, peer_ip: str) -> None:
        """Best-effort ``docker stop`` on a peer's worker container."""
        ssh_target = f"{self.config.ssh_user}@{peer_ip}"
        # Stop any container whose name starts with our prefix.
        remote = (
            "docker ps -q --filter "
            f"'name={RAY_CONTAINER_NAME_PREFIX}' | xargs -r docker stop"
        )
        ssh_cmd = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            ssh_target,
            remote,
        ]
        try:
            subprocess.run(
                ssh_cmd, capture_output=True, text=True, timeout=30
            )
        except Exception:  # pragma: no cover - best-effort teardown
            logger.exception("ssh docker stop on %s failed", peer_ip)

    def _docker_stop_best_effort(self, container_name: str) -> None:
        try:
            subprocess.run(
                ["docker", "stop", container_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception:  # pragma: no cover - best-effort teardown
            logger.exception("docker stop %s failed", container_name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tp_size(self) -> int:
        """Total TP = 1 local GPU + N peer GPUs. One GPU per GB10 node."""
        return 1 + len(self.config.peer_ips)

    def _head_fabric_ip(self) -> Optional[str]:
        return detect_fabric_ip(self.config.cluster_interface or "")

    def _head_hf_cache(self) -> str:
        """Path mounted into the container at /root/.cache/huggingface.

        Prefer ``config.hf_cache_dir`` (user-overridden), fall back to
        a default under the AINode models dir. Guaranteed writable on
        the head — workers get their own local path (see
        :meth:`_ssh_launch_worker`).
        """
        return (
            self.config.hf_cache_dir
            or str(Path(self.config.models_dir or "/root/.ainode/models") / "hf-cache")
        )

    def _locate_run_cluster_script(self) -> Optional[Path]:
        """Return the resolved path to run_cluster.sh, or None if missing."""
        for candidate in (RUN_CLUSTER_SCRIPT_SOURCE, RUN_CLUSTER_SCRIPT_FALLBACK):
            if candidate.exists():
                return candidate
        return None

    def _stream_logs(self, process: subprocess.Popen, target: Path) -> None:
        """Tee subprocess stdout to ``target``, watch for readiness lines."""
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


__all__ = [
    "NVIDIA_VLLM_IMAGE",
    "NvidiaBackend",
    "NvidiaBackendError",
    "RUN_CLUSTER_SCRIPT_FALLBACK",
    "RUN_CLUSTER_SCRIPT_SOURCE",
    "RAY_CONTAINER_NAME_PREFIX",
]
