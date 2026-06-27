"""NvidiaBackend — drive vLLM in a container image (``$NVIDIA_VLLM_IMAGE``,
defaulting to the proven GB10/Spark build; see ``NVIDIA_VLLM_IMAGE`` below).

Phase 4 implementation. Mirrors the public lifecycle surface of
:class:`ainode.engine.backends.eugr.EugrBackend` so ``cmd_start`` /
``cmd_status`` / the dashboard can dispatch polymorphically. Internally,
very different from eugr:

* **Solo mode** — a single ``docker run $NVIDIA_VLLM_IMAGE \\
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
import shutil
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

# The vLLM container image to run. Defaults to the proven GB10/Spark build
# (vLLM 0.17.1, serves MoE on sm120); override with $NVIDIA_VLLM_IMAGE (e.g.
# nvcr.io/nvidia/vllm on non-Spark GPUs). Resolved here so a deployment never has
# to sed-repoint this source — the nvcr→scitrera drift that once broke a node.
NVIDIA_VLLM_IMAGE = os.environ.get("NVIDIA_VLLM_IMAGE") or "scitrera/dgx-spark-vllm:0.17.0-t5"

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

    def __init__(self, config: NodeConfig, on_ready: Optional[Callable] = None,
                 instance_id: str = ""):
        self.config = config
        self.on_ready = on_ready
        # Container-name disambiguator for concurrent instances (P2-2). Empty for
        # the primary instance → legacy unsuffixed names (back-compat); otherwise a
        # short safe token (the handler uses the per-instance port) so two heads on
        # the same node don't collide on `ainode-vllm-head`.
        self.instance_id = instance_id
        self._process: Optional[subprocess.Popen] = None
        self._ready = False
        # Coarse load-phase for the UI launching card (3c). Advances
        # monotonically as _stream_logs sees the engine's startup markers.
        self._load_phase = "idle"
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
        $NVIDIA_VLLM_IMAGE vllm serve <model> ...``.
        """
        if self.is_running():
            return True

        container_name = self._solo_container_name()
        # Idempotent launch: a leftover container with this name (from a prior run
        # that wasn't cleanly stopped) makes `docker run --name` fail with a
        # Conflict. The head path already does this (see _launch_head_container);
        # solo needs it too.
        self._docker_stop_and_rm_best_effort(container_name)
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
        self._load_phase = "idle"

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
            "load_phase": self._load_phase,
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
    def load_phase(self) -> str:
        """Coarse engine load phase for the UI launching card (3c)."""
        return self._load_phase

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
        peer_fabric_ip: Optional[str] = None,
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
        local_fabric_ip = detect_fabric_ip(iface) or "127.0.0.1"
        hca = build_nccl_ib_hca_whitelist()

        # For the head, VLLM_HOST_IP is this node's fabric IP. For a worker,
        # it must be THE WORKER's fabric IP (we pass `peer_fabric_ip` when
        # assembling the SSH-to-worker docker command from the head). If
        # not set, fall back to local detection — but that would only be
        # correct when the method runs on the worker itself, which is not
        # how `_ssh_launch_worker` invokes it today. Phase 5 Bug 5 fix.
        if is_head:
            fabric_ip = local_fabric_ip
        else:
            fabric_ip = peer_fabric_ip or local_fabric_ip

        master_addr = head_fabric_ip if not is_head else local_fabric_ip

        env: Dict[str, str] = {
            "VLLM_HOST_IP": fabric_ip,
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": self._master_port(),
            "UCX_NET_DEVICES": iface,
            "NCCL_SOCKET_IFNAME": iface,
            "OMPI_MCA_btl_tcp_if_include": iface,
            "GLOO_SOCKET_IFNAME": iface,
            "TP_SOCKET_IFNAME": iface,
            "RAY_memory_monitor_refresh_ms": "0",
            "NCCL_IB_GID_INDEX": NCCL_IB_GID_INDEX,
            "NCCL_IB_SUBNET_AWARE_ROUTING": "1",
            "NCCL_IB_DISABLE": "0",
            # The vLLM image does NOT ship hf_transfer. If AINode's own
            # container has HF_HUB_ENABLE_HF_TRANSFER=1 (our install-UX
            # default), that env var would inherit into the vllm container via
            # docker exec and crash vllm at first weight download. Explicitly
            # set to "0" so the image uses the standard HF downloader. If a
            # future image bakes hf_transfer in, flip this to "1".
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "HF_TOKEN": self.config.hf_token or "",
            # Attention backend. NOTE (verified 2026-06-17): this
            # scitrera/vLLM 0.17.1 build does NOT honor "TRITON_ATTN" — every
            # rank still logs "Using FLASHINFER attention", so this pin is
            # currently a NO-OP. The actual GB10/sm120 crash fix is
            # --enforce-eager (see _build_vllm_serve_args); FlashInfer's
            # prefill kernel is fine in eager, it only crashes under CUDA-graph
            # capture. Pin retained as an env-overridable hedge: if a future
            # build honors it, the correct value is likely "TRITON_ATTN_VLLM_V1"
            # — set VLLM_ATTENTION_BACKEND in the systemd unit to override.
            "VLLM_ATTENTION_BACKEND": os.environ.get(
                "VLLM_ATTENTION_BACKEND", "TRITON_ATTN"
            ),
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

    def _name_suffix(self) -> str:
        """Per-instance container-name suffix (empty for the primary)."""
        return f"-{self.instance_id}" if self.instance_id else ""

    def _port_offset(self) -> int:
        """0 for the primary (api_port 8000), 1+ for co-resident instances.

        When two instances are headed by the SAME node, their Ray heads + torch
        rendezvous can't share host ports under --network host, so each gets an
        offset keyed off its (unique) api_port.
        """
        return max(0, int(self.config.api_port) - 8000)

    def _ray_port(self) -> int:
        return 6379 + self._port_offset()

    def _master_port(self) -> str:
        return str(int(MASTER_PORT) + self._port_offset())

    def _solo_container_name(self) -> str:
        return f"{RAY_CONTAINER_NAME_PREFIX}-solo{self._name_suffix()}"

    def _head_container_name(self) -> str:
        """Stable name for the head Ray container, unique per instance.

        Legacy single-instance name is ``HEAD_CONTAINER_NAME``; concurrent
        instances append ``-<instance_id>`` so two heads on this node don't
        collide (and the `docker rm -f` before launch only hits this instance).
        """
        return f"{HEAD_CONTAINER_NAME}{self._name_suffix()}"

    def _worker_container_name(self, peer_ip: str) -> str:
        """Stable, instance-unique name for a peer's worker container.

        Embeds the peer IP (dots→dashes) AND the instance suffix, so two
        instances that share a peer node don't collide on the worker name.
        """
        safe_ip = peer_ip.replace(".", "-").replace(":", "-")
        return f"{WORKER_CONTAINER_NAME_PREFIX}-{safe_ip}{self._name_suffix()}"

    # Container-side mount point for AINode's on-disk model store (read-only).
    MODELS_MOUNT = "/ainode-models"

    def _host_path(self, container_path: str) -> str:
        """Translate a path under AINODE_HOME (this orchestrator's *container*
        view) to the equivalent *host* path, so a docker ``-v`` SOURCE resolves
        on the host daemon — not to a stray root-owned dir.

        AINode runs inside a container that bind-mounts a host dir at AINODE_HOME
        (the systemd unit: ``-v <host>/.ainode:/root/.ainode``). When we then spawn
        the vLLM container we pass ``-v <our-path>:...`` to the SAME host daemon,
        which reads the SOURCE literally — so it must be the host path, not ours.
        The unit sets ``AINODE_HOST_HOME`` to the host dir it mounted. No-op when
        unset (AINode running directly on the host, where the two paths coincide).
        """
        host_home = os.environ.get("AINODE_HOST_HOME")
        if not host_home:
            return container_path
        from ainode.core.config import AINODE_HOME
        home = str(AINODE_HOME)
        if container_path == home or container_path.startswith(home + os.sep):
            return host_home.rstrip("/") + container_path[len(home):]
        return container_path

    def _local_model_dir(self) -> Optional[str]:
        """This model's on-disk weight dir (flat ``org--name`` layout written by
        our downloader), container-side path — or None if not downloaded.

        When present we serve it DIRECTLY (mounted at MODELS_MOUNT) instead of
        passing the HF repo-id, so vLLM never re-downloads 10s–100s of GB it
        already has on disk (the wart that nuked the WAN on a TP=2 launch)."""
        if not self.config.model:
            return None
        slug = self.config.model.replace("/", "--")
        d = Path(self.config.models_dir) / slug
        try:
            if d.is_dir() and any(d.iterdir()):
                return str(d)
        except OSError:
            pass
        return None

    def _serve_target_and_name_args(self) -> tuple:
        """Return ``(serve_target, extra_args)`` for ``vllm serve``.

        If the model is on disk AND the host mount is trustworthy, serve the local
        mount path and pin the API id with ``--served-model-name <repo-id>`` so
        /v1/models is unchanged. Otherwise serve the repo-id (vLLM downloads it).

        "Trustworthy" = we're either running directly on the host (the -v source
        path coincides) or AINODE_HOST_HOME tells us the host path for the source.
        When AINode runs in a container WITHOUT that env, the -v source would
        resolve to an empty root-owned dir, so we must NOT point vLLM at it —
        falling back to the repo-id keeps the current (re-download) behaviour and
        guarantees no regression before the systemd unit sets AINODE_HOST_HOME."""
        in_container = os.environ.get("AINODE_IN_CONTAINER")
        mount_trustworthy = (not in_container) or bool(os.environ.get("AINODE_HOST_HOME"))
        if mount_trustworthy:
            local = self._local_model_dir()
            if local:
                slug = self.config.model.replace("/", "--")
                return f"{self.MODELS_MOUNT}/{slug}", ["--served-model-name", self.config.model]
        return self.config.model, []

    def _build_solo_docker_cmd(self, container_name: str) -> List[str]:
        """Single-container solo mode — ``docker run ... vllm serve ...``.

        Not using run_cluster.sh here; that script always wires up Ray,
        which is overkill (and adds ~30s boot time) for a single-node
        vLLM process.
        """
        nccl_env = self._build_nccl_env(is_head=True)
        hf_cache = self._host_path(self._head_hf_cache())
        models_src = self._host_path(str(Path(self.config.models_dir)))
        serve_target, name_args = self._serve_target_and_name_args()

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
            # Mount the on-disk model store read-only so an already-downloaded
            # model serves straight from disk (no re-download).
            "-v",
            f"{models_src}:{self.MODELS_MOUNT}:ro",
        ]
        for key, value in nccl_env.items():
            cmd.extend(["-e", f"{key}={value}"])

        cmd.extend([NVIDIA_VLLM_IMAGE, "vllm", "serve", serve_target])
        cmd.extend(self._build_vllm_serve_args(tp_size=1))
        cmd.extend(name_args)
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

        # When building the env for a peer (role == "worker"), pass the
        # peer's fabric IP explicitly so VLLM_HOST_IP is UNIQUE per node.
        # node_ip here is the peer's own fabric IP (set by the caller in
        # _ssh_launch_worker).
        nccl_env = self._build_nccl_env(
            is_head=(role == "head"),
            head_fabric_ip=head_ip,
            peer_fabric_ip=(node_ip if role != "head" else None),
        )

        ray_port = self._ray_port()
        if role == "head":
            # --include-dashboard is HEAD-ONLY (the worker `ray start --address`
            # PANICs on it). Disabling it avoids the 8265 dashboard-port collision
            # only relevant if heads ever co-reside on a node.
            ray_cmd = (
                f"ray start --block --head --include-dashboard=false "
                f"--node-ip-address={shlex.quote(node_ip)} --port={ray_port}"
            )
        else:
            ray_cmd = (
                f"ray start --block "
                f"--address={shlex.quote(head_ip)}:{ray_port} "
                f"--node-ip-address={shlex.quote(node_ip)}"
            )

        cmd: List[str] = [
            "docker", "run", "-d",
            "--name", container_name,
            "--network", "host",
            "--gpus", "all",
            "--shm-size", "10.24g",
            "--entrypoint", "/bin/bash",
            # host-path the SOURCE so the host docker daemon mounts the real
            # dir, not a stray root-owned path (see _host_path). A no-op for the
            # peer's home-dir cache, which isn't under AINODE_HOME.
            "-v", f"{self._host_path(hf_cache_dir)}:/root/.cache/huggingface",
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
            # THE GB10/sm120 fix (verified 2026-06-17). FlashInfer's prefill
            # kernel (BatchPrefillWithPagedKVCache) illegal-instructions under
            # CUDA-graph capture on GB10 (sm120) and kills EngineCore on the
            # first real prefill — the engine loads, reports READY, then
            # suicides (vLLM SIGTERMs its own Ray workers). --enforce-eager
            # disables graph capture and the same kernel runs clean (235B TP=4
            # survived a 3,513-token prefill). Re-enabling graphs for
            # throughput needs a working non-FlashInfer backend first.
            "--enforce-eager",
        ]
        # fp8 KV cache — the GB10 design default (engine/AGENTS.md): required for
        # long context or vLLM OOMs sizing the cache at bf16. Config-driven so a
        # model/quant that rejects fp8 can fall back via kv_cache_dtype="".
        if getattr(self.config, "kv_cache_dtype", ""):
            args.extend(["--kv-cache-dtype", self.config.kv_cache_dtype])
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

        # Phase 3a: ensure the peer actually has the model weights before its
        # worker starts — distribute from the head over the fabric if missing.
        self._ensure_peer_has_model(peer_ip, peer_hf_cache)

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

    def _ensure_peer_has_model(self, peer_ip: str, peer_hf_cache: str) -> None:
        """Distribute the model weights to a peer over the fabric if it's missing.

        The launch only succeeds if every node can read the model from its local
        HF cache. Rather than require manual pre-placement, the head streams the
        weights to any selected peer that lacks them. Uses tar-over-ssh (the image
        ships tar + ssh, not rsync) on the cluster fabric (``peer_ip``).
        Best-effort no-op when the peer already has it, or the head doesn't.
        """
        model = self.config.model or ""
        if not model:
            return
        model_dir = "models--" + model.replace("/", "--")
        head_hub = str(Path(self._head_hf_cache()) / "hub")
        if not (Path(head_hub) / model_dir).is_dir():
            return  # head doesn't have it either — engine will report clearly
        peer_hub = peer_hf_cache.rstrip("/") + "/hub"
        target = f"{peer_hub}/{model_dir}"
        ssh_target = f"{self.config.ssh_user}@{peer_ip}"
        ssh_opts = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]

        check = subprocess.run(
            ["ssh", *ssh_opts, ssh_target, f"test -d {shlex.quote(target)} && echo present || echo missing"],
            capture_output=True, text=True, timeout=30,
        )
        if "present" in (check.stdout or ""):
            return  # peer already has the weights

        logger.info("Distributing %s to %s over the fabric (not cached)...", model_dir, peer_ip)
        self._load_phase = "distributing"
        ssh_e = "ssh " + " ".join(ssh_opts)
        if shutil.which("rsync"):
            # Preferred: rsync is resumable (--partial) and incremental, so a
            # re-launch after a dropped transfer doesn't re-send the whole model.
            subprocess.run(["ssh", *ssh_opts, ssh_target, f"mkdir -p {shlex.quote(peer_hub)}"],
                           capture_output=True, text=True, timeout=30)
            result = subprocess.run(
                ["rsync", "-a", "--partial", "-e", ssh_e,
                 f"{head_hub}/{model_dir}/", f"{ssh_target}:{peer_hub}/{model_dir}/"],
                capture_output=True, text=True, timeout=7200,
            )
        else:
            # Fallback for images without rsync: tar-over-ssh (not resumable).
            tar = (
                f"tar -C {shlex.quote(head_hub)} -cf - {shlex.quote(model_dir)} | "
                f"{ssh_e} {shlex.quote(ssh_target)} "
                f"'mkdir -p {shlex.quote(peer_hub)} && tar -C {shlex.quote(peer_hub)} -xf -'"
            )
            result = subprocess.run(["bash", "-lc", tar], capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            raise NvidiaBackendError(
                f"Failed to distribute {model_dir} to {peer_ip} "
                f"(rc={result.returncode}): {result.stderr.strip()[:300]}"
            )
        logger.info("Distributed %s to %s", model_dir, peer_ip)

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

    # Ordered load phases (3c). Each engine startup log line is matched against
    # these markers; the phase only advances (monotonic by rank) so a coarse
    # progress card can show load → distributed-init → profiling → ready, and a
    # stall is visible as the phase that stops advancing.
    _LOAD_PHASE_ORDER = ["idle", "starting", "distributing", "loading_weights", "distributed_init", "profiling", "ready"]
    _LOAD_PHASE_MARKERS = [
        ("loading_weights", ("loading model weights", "loading weights", "loading safetensors")),
        ("distributed_init", ("nccl info", "init_process_group", "rayworkerwrapper", "ray worker")),
        ("profiling", ("memory profiling", "available kv cache", "gpu kv cache", "warming up", "autotuning")),
    ]

    def _advance_load_phase(self, phase: str) -> None:
        """Set _load_phase to `phase` only if it's later than the current one."""
        order = self._LOAD_PHASE_ORDER
        try:
            if order.index(phase) > order.index(self._load_phase):
                self._load_phase = phase
        except ValueError:
            pass

    def _stream_logs(self, process: subprocess.Popen, target: Path) -> None:
        """Tee subprocess stdout to ``target``, watch for readiness + load phase."""
        if not process.stdout:
            return
        # A fresh log stream means a fresh launch — start the phase clock over.
        self._load_phase = "starting"
        with open(target, "a") as sink:
            for line in process.stdout:
                sink.write(line)
                sink.flush()
                if not self._ready:
                    low = line.lower()
                    for phase, markers in self._LOAD_PHASE_MARKERS:
                        if any(m in low for m in markers):
                            self._advance_load_phase(phase)
                            break
                if not self._ready and (
                    "Uvicorn running on" in line
                    or "Application startup complete" in line
                ):
                    self._ready = True
                    self._load_phase = "ready"
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
