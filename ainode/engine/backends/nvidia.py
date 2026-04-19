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

# Agent B originally vendored ``scripts/run_cluster.sh`` into the AINode
# install at ``/opt/ainode/run_cluster.sh``. Phase 5 Bug 2 fix (Option α)
# removed run_cluster.sh from the hot path entirely — NvidiaBackend now
# drives ``docker run -d`` directly in Python. The constants are retained
# for back-compat: the vendored script still ships for manual debugging
# and eugr parity, and operators can still ``wget`` it into /tmp.
RUN_CLUSTER_SCRIPT_SOURCE = Path("/opt/ainode/run_cluster.sh")
RUN_CLUSTER_SCRIPT_FALLBACK = Path("/tmp/run_cluster.sh")

# Historical prefix used by run_cluster.sh-era containers. Retained for
# callers that import it (it's still in ``__all__``), but no longer used
# in the Option α launch path — see ``HEAD_CONTAINER_NAME`` /
# ``WORKER_CONTAINER_NAME_PREFIX`` below.
RAY_CONTAINER_NAME_PREFIX = "ainode-vllm-node"

# Option α — stable container names for head + workers. Stable so
# ``stop()`` (and operator ``docker stop``) can always find them, and
# collision-free across peers because worker names embed the peer IP.
HEAD_CONTAINER_NAME = "ainode-vllm-head"
WORKER_CONTAINER_NAME_PREFIX = "ainode-vllm-worker"

# How long to wait for the head Ray container to report Running after
# ``docker run -d`` returns. ``ray start --block`` binds :6379 in a few
# seconds on a pre-pulled image; 60s is generous.
HEAD_CONTAINER_READY_TIMEOUT = 60

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

        Phase 5 Bug 2 fix (Option α) — the prior implementation invoked
        the vendored ``scripts/run_cluster.sh`` via a blocking
        ``subprocess.run(..., timeout=120)``. That script does a
        *foreground* ``docker run`` ending in ``ray start --block`` and
        therefore never exits on its own, so the 120 s timeout always
        fired and AINode's main thread was stuck long enough that
        ``run_server()`` never bound port 3000. Option α replaces the
        script with an inline ``docker run -d`` in Python, giving us
        immediate return + a stable container handle for teardown.

        Steps:

        1. Validate we're ``distributed_mode == "head"`` with peers
           configured.
        2. Start the head Ray container via ``docker run -d`` (see
           :meth:`_launch_head_container`). Poll ``docker inspect`` until
           ``.State.Running`` is true (see
           :meth:`_wait_for_head_container_ready`).
        3. SSH to each peer and run ``docker run -d`` there too, pointing
           the workers at the head fabric IP.
        4. ``docker exec`` into the local head container to invoke
           ``vllm serve --tensor-parallel-size N`` with N = 1 + len(peers).

        The ``Popen`` handle we keep is for the ``vllm serve`` exec
        (step 4); the Ray containers on head + peers are managed by
        docker itself and cleaned up in :meth:`stop`.

        Assumes the NVIDIA vLLM image is pre-pulled on every node (our
        deploy pipeline does ``docker load`` from NFS before enabling
        the systemd unit). We deliberately do NOT pass ``--pull=always``
        — first-run pulls can be multi-GB and would blow the 30 s
        ``docker run -d`` timeout.
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

        fabric_ip = self._head_fabric_ip()
        if fabric_ip is None:
            raise NvidiaBackendError(
                f"Could not detect fabric IP on interface "
                f"{self.config.cluster_interface!r}. Is the NIC up?"
            )

        hf_cache = self._head_hf_cache()

        # Step 2 — head Ray container. Non-blocking: ``docker run -d``
        # returns as soon as the container is created.
        self._launch_head_container(
            fabric_ip=fabric_ip,
            hf_cache_dir=hf_cache,
        )

        # Step 2b — wait for Ray head to actually be up before we SSH
        # workers at it (otherwise they race to connect to an unbound
        # :6379 and error out).
        if not self._wait_for_head_container_ready(
            HEAD_CONTAINER_NAME, timeout=HEAD_CONTAINER_READY_TIMEOUT
        ):
            raise NvidiaBackendError(
                f"Head container {HEAD_CONTAINER_NAME!r} did not enter "
                f"Running state within {HEAD_CONTAINER_READY_TIMEOUT}s. "
                "Check ``docker logs`` on the head for Ray startup errors."
            )

        # Step 3 — peer Ray workers over SSH (``ssh <peer> docker run -d``).
        for peer_ip in self.config.peer_ips:
            self._ssh_launch_worker(
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
        """Stop the vllm serve process + fan out to peers to kill their Ray containers.

        Teardown is best-effort for every remote call — an unreachable
        peer should not block shutdown of the head. The local head
        container is removed as well as stopped, so the next
        ``start_distributed`` can re-create the named container without
        a conflict.
        """
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

        # Tear down local container by name. Best-effort.
        container = (
            self._solo_container_name()
            if self.config.distributed_mode == "solo"
            else self._head_container_name()
        )
        self._docker_stop_and_rm_best_effort(container)

        # For distributed: SSH to each peer and stop+rm their worker container.
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
            # NVIDIA's nvcr.io/nvidia/vllm:26.02-py3 does NOT ship hf_transfer.
            # If AINode's own container has HF_HUB_ENABLE_HF_TRANSFER=1 (our
            # install-UX default), that env var would inherit into the vllm
            # container via docker exec and crash vllm at first weight
            # download. Explicitly set to "0" so the NVIDIA image uses the
            # standard HF downloader. If NVIDIA bakes hf_transfer into a
            # future image, flip this to "1".
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
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
        """Stable name for the head Ray container.

        Under Option α we launch the head ourselves via ``docker run -d
        --name``, so this is simply the constant ``HEAD_CONTAINER_NAME``.
        """
        return HEAD_CONTAINER_NAME

    def _worker_container_name(self, peer_ip: str) -> str:
        """Stable name for a peer's worker container.

        Must be collision-free across peers so ``docker inspect`` / ``docker
        stop`` address the right container. We embed the peer IP (with
        dots replaced by dashes, since ``.`` is valid in docker names but
        confusing to read) so the name round-trips from config to running
        container deterministically.
        """
        safe_ip = peer_ip.replace(".", "-").replace(":", "-")
        return f"{WORKER_CONTAINER_NAME_PREFIX}-{safe_ip}"

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

    def _build_ray_docker_cmd(
        self,
        *,
        container_name: str,
        role: str,
        head_ip: str,
        node_ip: str,
        hf_cache_dir: str,
    ) -> List[str]:
        """Build the ``docker run -d ... ray start --block`` command.

        Used by both head and worker launches under Option α. The
        container is detached (``-d``), so the returned command finishes
        fast and we keep the handle via ``--name``.

        ``role`` must be ``"head"`` or ``"worker"``. On head, Ray binds
        :6379; on worker, Ray connects to ``<head_ip>:6379``. ``node_ip``
        is what each Ray process registers as its own address in the
        cluster — for head this equals ``head_ip``, for worker it's the
        peer's fabric IP.

        Note: we deliberately do NOT pass ``--rm`` so operators can
        ``docker logs <name>`` after a crash. ``stop()`` removes the
        container explicitly.
        """
        if role not in {"head", "worker"}:
            raise ValueError(f"role must be 'head' or 'worker', got {role!r}")

        nccl_env = self._build_nccl_env(
            is_head=(role == "head"),
            head_fabric_ip=head_ip,
        )

        if role == "head":
            ray_cmd = (
                f"ray start --block --head "
                f"--node-ip-address={shlex.quote(node_ip)} --port=6379"
            )
        else:
            ray_cmd = (
                f"ray start --block "
                f"--address={shlex.quote(head_ip)}:6379 "
                f"--node-ip-address={shlex.quote(node_ip)}"
            )

        cmd: List[str] = [
            "docker", "run", "-d",
            "--name", container_name,
            "--network", "host",
            "--gpus", "all",
            "--shm-size", "10.24g",
            "--entrypoint", "/bin/bash",
            "-v", f"{hf_cache_dir}:/root/.cache/huggingface",
        ]
        for key, value in nccl_env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([NVIDIA_VLLM_IMAGE, "-c", ray_cmd])
        return cmd

    def _launch_head_container(
        self,
        fabric_ip: str,
        hf_cache_dir: str,
    ) -> str:
        """Launch the head Ray container via ``docker run -d``.

        Returns the container ID (stdout of ``docker run -d``) on
        success. Raises :class:`NvidiaBackendError` if docker reports a
        non-zero exit — most commonly because a previous container of
        the same name already exists (we try to ``stop/rm`` it first to
        make this idempotent).

        The 30 s timeout is a safety net, not a normal-path bound:
        ``docker run -d`` returns as soon as the container is created,
        which should take well under a second on a pre-pulled image.
        If we hit the timeout, something is catastrophically wrong with
        the local docker daemon and raising is the right call.
        """
        # Idempotency: if a stale container from a previous run is
        # hanging around, remove it before trying to ``--name`` ours.
        self._docker_stop_and_rm_best_effort(HEAD_CONTAINER_NAME)

        cmd = self._build_ray_docker_cmd(
            container_name=HEAD_CONTAINER_NAME,
            role="head",
            head_ip=fabric_ip,
            node_ip=fabric_ip,
            hf_cache_dir=hf_cache_dir,
        )
        env = self._build_env_for_subprocess()

        logger.info("Launching head Ray container: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            raise NvidiaBackendError(
                f"docker run -d for head container {HEAD_CONTAINER_NAME!r} "
                f"timed out after {exc.timeout}s; the local docker daemon "
                "may be unresponsive."
            ) from exc

        if result.returncode != 0:
            raise NvidiaBackendError(
                f"docker run -d for head container {HEAD_CONTAINER_NAME!r} "
                f"failed (rc={result.returncode}): {result.stderr.strip()}"
            )
        return result.stdout.strip()

    def _wait_for_head_container_ready(
        self,
        container_name: str,
        timeout: int = HEAD_CONTAINER_READY_TIMEOUT,
    ) -> bool:
        """Poll ``docker inspect`` until the container is Running, or time out.

        Returns True once ``.State.Running == true``, False on timeout.
        We only check ``Running`` and not ``Health.Status`` — the
        NVIDIA vLLM image does not ship a HEALTHCHECK instruction, so
        ``Health`` is absent from ``docker inspect`` output. ``ray start``
        binds :6379 within a couple of seconds on a pre-pulled image, so
        Running-true is a reliable-enough proxy for "head is up".
        """
        deadline = time.time() + timeout
        inspect_fmt = "{{.State.Running}}"
        last_err: Optional[str] = None
        while time.time() < deadline:
            try:
                result = subprocess.run(
                    ["docker", "inspect", "-f", inspect_fmt, container_name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except subprocess.TimeoutExpired:
                last_err = "docker inspect timed out"
                time.sleep(1)
                continue
            if result.returncode == 0 and result.stdout.strip() == "true":
                return True
            last_err = result.stderr.strip() or result.stdout.strip()
            time.sleep(1)
        logger.warning(
            "Head container %s not ready within %ds: %s",
            container_name, timeout, last_err,
        )
        return False

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
        peer_ip: str,
        head_ip: str,
    ) -> None:
        """SSH to ``peer_ip`` and launch its Ray worker container.

        Phase 5 Bug 2 fix: previously this invoked
        ``bash run_cluster.sh --worker ...`` on the peer, inheriting the
        same foreground/trap-EXIT problem that caused the head to hang.
        Option α replaces it with a direct ``docker run -d`` over SSH
        using the same builder as the head, so the SSH call returns
        fast and leaves a detached container on the peer.

        Assumes passwordless SSH from this node as ``ssh_user`` to the
        peer, and that the NVIDIA vLLM image is pre-pulled on the peer
        (the deploy pipeline distributes it via ``docker load`` from NFS).
        """
        # Reasonable per-peer HF cache path. Workers can't always write
        # to NFS (runbook 02 § Observations / gotcha 2), so default to a
        # home-directory path under the ssh_user's home. We can't use /root
        # because we SSH in as the non-root ssh_user on the peer.
        peer_hf_cache = f"/home/{self.config.ssh_user}/ainode-nvidia-cache"

        worker_name = self._worker_container_name(peer_ip)

        docker_cmd = self._build_ray_docker_cmd(
            container_name=worker_name,
            role="worker",
            head_ip=head_ip,
            # The peer registers as its own IP, which is the IP we SSH to.
            # (We SSH over the fabric, so peer_ip here is the fabric IP.)
            node_ip=peer_ip,
            hf_cache_dir=peer_hf_cache,
        )

        # Remote shell command: clean up any stale worker container from
        # a prior run (stable name means we can always find it), make
        # the cache dir, then docker run -d. Chained with && so a failed
        # cleanup still lets docker run surface its own error.
        docker_cmd_str = " ".join(shlex.quote(p) for p in docker_cmd)
        remote_cmd = (
            f"docker rm -f {shlex.quote(worker_name)} >/dev/null 2>&1 || true; "
            f"mkdir -p {shlex.quote(peer_hf_cache)} && {docker_cmd_str}"
        )

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
        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            raise NvidiaBackendError(
                f"ssh docker run -d for worker on {peer_ip} timed out "
                f"after {exc.timeout}s."
            ) from exc
        if result.returncode != 0:
            raise NvidiaBackendError(
                f"ssh docker run -d for worker on {peer_ip} failed "
                f"(rc={result.returncode}): {result.stderr.strip()}"
            )

    def _ssh_stop_peer_container(self, peer_ip: str) -> None:
        """Best-effort ``docker stop && docker rm`` on a peer's worker container.

        Uses the deterministic container name (see
        :meth:`_worker_container_name`) so stop is targeted and can't
        accidentally clobber unrelated containers on the peer. Remote
        errors are swallowed — an unreachable peer should not block
        shutdown of the head.
        """
        worker_name = self._worker_container_name(peer_ip)
        ssh_target = f"{self.config.ssh_user}@{peer_ip}"
        # ``|| true`` so a missing container (peer never started) doesn't
        # fail the ssh. ``-f`` on rm covers still-running containers.
        remote = (
            f"docker stop {shlex.quote(worker_name)} >/dev/null 2>&1 || true; "
            f"docker rm -f {shlex.quote(worker_name)} >/dev/null 2>&1 || true"
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

    def _docker_stop_and_rm_best_effort(self, container_name: str) -> None:
        """Stop and remove a local container, swallowing all errors.

        Used both at teardown (``stop()``) and before launching a fresh
        head/solo container so the ``--name`` flag doesn't collide with
        a lingering stopped container from a previous run.
        """
        for args in (
            ["docker", "stop", container_name],
            ["docker", "rm", "-f", container_name],
        ):
            try:
                subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except Exception:  # pragma: no cover - best-effort teardown
                logger.exception("%s failed", " ".join(args))

    # Back-compat alias — older tests (and any outside caller) might
    # import the historical name. Kept so imports don't break.
    def _docker_stop_best_effort(self, container_name: str) -> None:
        self._docker_stop_and_rm_best_effort(container_name)

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
    "HEAD_CONTAINER_NAME",
    "HEAD_CONTAINER_READY_TIMEOUT",
    "NVIDIA_VLLM_IMAGE",
    "NvidiaBackend",
    "NvidiaBackendError",
    "RUN_CLUSTER_SCRIPT_FALLBACK",
    "RUN_CLUSTER_SCRIPT_SOURCE",
    "RAY_CONTAINER_NAME_PREFIX",
    "WORKER_CONTAINER_NAME_PREFIX",
]
