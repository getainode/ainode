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

* **Distributed (head) mode**: two shapes, picked per instance by
  ``NodeConfig.distributed_executor`` (see ``engine/AGENTS.md``):

  * ``"ray"`` (default): a ``ray start --head`` container here, an
    SSH-launched ``ray start`` worker container on each peer, then a
    ``docker exec`` into the local head to run ``vllm serve
    --distributed-executor-backend ray --tensor-parallel-size N``. Mirrors
    runbook 02 § Steps 4-7. REQUIRES the ``ray`` CLI in the engine image.
  * ``"mp"``: one ``vllm serve`` container per node using vLLM's own
    multi-node executor: rank 0 here, ``--node-rank k --headless`` on each
    peer, all rendezvousing on ``--master-addr``/``--master-port``. No Ray
    container, no ``docker exec``, nothing needed in the image beyond vLLM.
    the shape for a custom engine build (the GB10 DeepSeek V4 image and
    stock ``vllm/vllm-openai`` both ship no ray).

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
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ainode.cluster.hca_discovery import (
    build_nccl_ib_hca_whitelist,
    detect_fabric_ip,
)
from ainode.cluster.netdev import (
    interface_candidates_hint,
    resolve_cluster_interface,
)
from ainode.core.config import HF_CACHE_MOUNT, LOGS_DIR, NodeConfig
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

# Workarounds below (forced --enforce-eager, the NVFP4 MARLIN env) are bugs in
# the PINNED 0.17 build, not in vLLM generally. Newer engines (0.27.1, which
# Nemotron 3.5 Lightning and Qwen3.8 require) fix them upstream and are actively
# harmed by --enforce-eager, which disables CUDA graphs and costs throughput.
# So they apply only when the instance runs the pinned default image.

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
# A cold engine image is ~20 GB; a first pull on a slow link needs real room.
IMAGE_PULL_TIMEOUT = 3600

NCCL_IB_GID_INDEX = "3"
MASTER_PORT = "29501"

# Host device tree the RDMA verbs libraries open. Mapped into the engine
# container when it exists, so NCCL can use IB/RoCE verbs instead of falling
# back to sockets. A node without it (no RDMA NIC, or a non-Spark host) simply
# does not get the mapping. A missing path is never a launch failure.
INFINIBAND_DEVICE = "/dev/infiniband"

# Shared-memory + ulimits for the mp multi-node shape. vLLM's own multi-node
# executor puts every rank's NCCL/torch buffers in /dev/shm and pins them, so
# the 10.24g the Ray shape uses is not enough and an unlimited memlock is
# required for IB registration. These are the values the proven GB10 recipe ran.
MP_SHM_SIZE = "64g"
MP_ULIMIT_MEMLOCK = "memlock=-1"
MP_ULIMIT_STACK = "stack=67108864"


class NvidiaBackendError(RuntimeError):
    """Raised when the backend cannot be driven (missing image, bad config)."""


_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{12,64}$")


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
        # Epoch seconds of the last line this engine printed. The startup replay
        # reads it to tell a slow-but-progressing start from a wedged one.
        self._last_log_activity: Optional[float] = None
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
        # Before _build_solo_docker_cmd, because the argv prefix is derived from
        # the image's ENTRYPOINT and inspecting a missing image yields nothing.
        if not self.ensure_image(self._engine_image()):
            logger.error("Engine image %s unavailable; not launching %s",
                         self._engine_image(), self.config.model)
            return False
        cmd = self._build_solo_docker_cmd(container_name)
        env = self._build_env_for_subprocess()

        logger.info(
            "Starting NVIDIA solo vLLM: docker run %s vllm serve %s (extra args: %s)",
            self._engine_image(),
            self.config.model,
            " ".join(getattr(self.config, "extra_vllm_args", None) or []) or "none",
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
        return self._confirm_container_started(container_name)

    def _confirm_container_started(self, container_name: str, timeout: float = 25.0) -> bool:
        """Return True only if the container is actually RUNNING shortly after
        launch.

        ``docker run -d`` forks and returns immediately, so the old
        ``self._process.poll() is None`` check only proved the docker CLI had
        spawned — an engine that rejected its flags or failed the memory
        pre-check still reported a successful launch, and the caller registered
        a live instance that never existed (2026-08-14 phantom rows). Here we
        wait for the container to exist and report Running; a container that
        exited gets its last log lines surfaced so the failure has a reason.

        This is a LAUNCH check, not a readiness check — weights take minutes;
        the engine reports ready separately via ``_stream_logs``.
        """
        deadline = time.time() + timeout
        state = ""
        while time.time() < deadline:
            state = self._docker_container_state(container_name)
            if state == "running":
                return True
            if state in ("exited", "dead"):
                break
            time.sleep(1.0)
        tail = self._docker_logs_tail(container_name, lines=15)
        logger.error(
            "Engine container %s failed to start (state=%s). Last output:\n%s",
            container_name, state or "missing", tail or "(no output captured)",
        )
        return False

    def _docker_container_state(self, container_name: str) -> str:
        """``docker inspect`` state string ('running'/'exited'/...), '' if absent."""
        try:
            out = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Status}}", container_name],
                capture_output=True, text=True, timeout=10,
            )
            return out.stdout.strip() if out.returncode == 0 else ""
        except Exception:
            return ""

    def _docker_logs_tail(self, container_name: str, lines: int = 15) -> str:
        """Last N lines of a container's output — the failure reason for a
        crashed engine. Safe on a missing container (returns '')."""
        try:
            out = subprocess.run(
                ["docker", "logs", "--tail", str(lines), container_name],
                capture_output=True, text=True, timeout=15,
            )
            return ((out.stdout or "") + (out.stderr or "")).strip()
        except Exception:
            return ""

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
            # Name the interfaces that DO have an address so the user can fix
            # cluster_interface from this message alone. Issue #34's reporter
            # had to go find `ip -br addr` to discover the real NIC name.
            raise NvidiaBackendError(
                f"Could not detect fabric IP on interface "
                f"{resolve_cluster_interface(self.config)!r}. Is the NIC up? "
                f"Interfaces with an IPv4 address on this host: "
                f"{interface_candidates_hint()}. Set cluster_interface in "
                f"~/.ainode/config.json to one of those."
            )

        hf_cache = self._head_hf_cache()

        # Two distributed shapes, chosen per instance (see NodeConfig.
        # distributed_executor and engine/AGENTS.md). "mp" needs no ray in the
        # image, so it is the shape for a custom engine build; everything below
        # this branch is the Ray shape.
        if self._distributed_executor() == "mp":
            return self._start_distributed_mp(
                fabric_ip=fabric_ip, hf_cache_dir=hf_cache,
            )

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
        """Stop the local engine + fan out to peers to kill their containers.

        Shape-agnostic: the local process is either the ``docker exec``'d
        ``vllm serve`` (Ray shape) or the ``docker logs -f`` follower (mp
        shape); the local container is the head in both; and peer containers
        carry the same name in both, so one teardown covers them.

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
        if self._process is not None and self._process.poll() is None:
            return True
        # In the mp shape the head CONTAINER is the server; ``_process`` is only
        # the ``docker logs -f`` follower, so a dead follower (log stream closed,
        # AINode restarted) must not read as a dead engine. The Ray and solo
        # shapes keep the process-only answer and never pay for a docker call.
        if self._is_mp_distributed():
            return self._docker_container_state(self._head_container_name()) == "running"
        return False

    def health_check(self) -> dict:
        """Mirrors EugrBackend.health_check for dashboard parity."""
        result = {
            "process_alive": self.is_running(),
            "api_responding": False,
            "models_loaded": [],
            "load_phase": self.load_phase,
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
        """Coarse engine load phase for the UI launching card (3c).

        Derive 'ready' from the readiness latch: wait_ready() can set _ready via
        the API-poll path before _stream_logs sees the startup log line, which
        left _load_phase stuck at 'starting' on a model that is actually serving.
        """
        return "ready" if self._ready else self._load_phase

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
    def last_log_activity(self) -> Optional[float]:
        """Epoch seconds of the last line this engine printed (see base class).

        Fed by ``_stream_logs``, which reads this container's own stdout, so the
        value is per-instance even though stacked instances share a log file.
        """
        return self._last_log_activity

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
        * Ray / UCX / Gloo / Torch socket iface: all set to the resolved
          cluster interface (``config.cluster_interface`` when that device
          exists on this host, otherwise autodetected) so no process falls
          back to the default route.
        * ``NCCL_IB_HCA`` — whitelist built dynamically from local sysfs.
          Remote HCA lists are NOT threaded here yet; distributed mode
          uses the local view (it's what every peer also uses for their
          own view, so the union happens naturally in NCCL).
        * ``NCCL_IB_GID_INDEX=3`` — per Phase 1. Hardcoded because
          every DGX Spark + GX10 we've tested uses the same slot.
        * ``HF_HUB_ENABLE_HF_TRANSFER=1`` — always on, per install-UX spec.
        """
        iface = resolve_cluster_interface(self.config)
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
        env.update(self._nvfp4_serve_env())
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

    def _volume_args(self) -> List[str]:
        """``-v`` pairs for ``config.extra_volumes`` ("host:container[:ro]").

        The host side goes through :meth:`_host_path` for the same reason every
        other mount does: when AINode itself runs in a container, a SOURCE it
        can see is not necessarily the path the host daemon would resolve. A
        malformed entry (no container side) is skipped with a warning rather
        than failing the launch: a bad recipe should not cost the whole serve.
        """
        args: List[str] = []
        for raw in (getattr(self.config, "extra_volumes", None) or []):
            spec = str(raw).strip()
            if not spec:
                continue
            host, sep, rest = spec.partition(":")
            if not sep or not rest:
                logger.warning("Ignoring malformed extra_volumes entry %r "
                               "(expected host:container[:ro])", spec)
                continue
            args.extend(["-v", f"{self._host_path(host)}:{rest}"])
        return args

    def _infiniband_present(self) -> bool:
        """True when the host exposes the RDMA device tree.

        Its own method so the launch builders have one seam to check (and tests
        one place to fake). Never raises: a host without RDMA just misses the
        mapping, which is not an error.
        """
        try:
            return Path(self._host_path(INFINIBAND_DEVICE)).exists()
        except OSError:  # pragma: no cover - defensive
            return False

    def _infiniband_device_args(self) -> List[str]:
        """``--device /dev/infiniband:/dev/infiniband`` when the host has it."""
        if not self._infiniband_present():
            return []
        return ["--device", f"{INFINIBAND_DEVICE}:{INFINIBAND_DEVICE}"]

    def _distributed_executor(self) -> str:
        """Which distributed shape this instance launches: "ray" or "mp".

        An unrecognised value is not silently reinterpreted: a typo in a recipe
        would otherwise launch a shape the image cannot run and fail deep inside
        a container. ``stop()`` and ``is_running()`` must never raise on it,
        which is why they route through :meth:`_is_mp_distributed` instead.
        """
        value = (getattr(self.config, "distributed_executor", "") or "ray").strip().lower()
        if value not in ("ray", "mp"):
            raise NvidiaBackendError(
                f"Unknown distributed_executor={value!r}; expected 'ray' "
                "(ray containers + docker exec) or 'mp' (one vllm serve "
                "container per node, vLLM's own multi-node executor)."
            )
        return value

    def _is_mp_distributed(self) -> bool:
        """True for a head instance running the mp shape. Never raises."""
        if self.config.distributed_mode != "head":
            return False
        return (getattr(self.config, "distributed_executor", "") or "ray").strip().lower() == "mp"

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

    def _is_multimodal_model(self) -> bool:
        """Detect a vision/multimodal model from its on-disk ``config.json``.

        True when config.json has a ``vision_config`` key, or any
        ``architectures`` entry matches ``/VL|Vision|vision/``. Only the LOCAL
        model dir is inspected: if config.json is unreadable (e.g. a remote
        repo-id not yet downloaded), we return False so the caller keeps the fp8
        default — we deliberately never fetch remote config here (no network in
        the serve-args builder).

        CEILING: a not-yet-downloaded VLM served by repo-id won't be detected
        and will use the fp8 KV default until its weights are on disk.
        """
        local = self._local_model_dir()
        if not local:
            return False
        try:
            cfg = json.loads((Path(local) / "config.json").read_text())
        except Exception:
            return False
        if "vision_config" in cfg:
            return True
        import re
        archs = cfg.get("architectures") or []
        if isinstance(archs, str):
            archs = [archs]
        return any(re.search(r"VL|Vision|vision", str(a)) for a in archs)

    def _effective_kv_cache_dtype(self) -> str:
        """Resolve the KV-cache dtype for serve args.

        fp8 KV corrupts vision-model generation on GB10 (verified: Qwen2.5-VL
        emits garbage on fp8, clean output on auto; text models are unaffected).
        So the fp8 DEFAULT is downgraded to 'auto' for a multimodal model. Any
        EXPLICIT ``kv_cache_dtype`` always wins — whether it's a non-fp8 value
        (which isn't the default anyway) or an explicit 'fp8' flagged via
        ``kv_cache_dtype_explicit`` (the user's opt-back-in for a VLM/vLLM combo
        they know handles fp8 KV). A model whose config.json can't be read keeps
        the fp8 default.
        """
        dtype = getattr(self.config, "kv_cache_dtype", "") or ""
        explicit = getattr(self.config, "kv_cache_dtype_explicit", False)
        if dtype == "fp8" and not explicit and self._is_multimodal_model():
            return "auto"
        return dtype

    def _engine_env(self, nccl_env: Dict[str, str]) -> Dict[str, str]:
        """Env for the engine container: computed NCCL env + per-instance extras.

        ``extra_env`` is applied LAST so a recipe wins over an autodetected
        value. Some engine features have no CLI flag at all (the b12x FP4 path
        is selected purely by VLLM_NVFP4_GEMM_BACKEND and friends), so without
        this they can only be reached by hand-rolling a container — which is
        exactly what the launch path exists to avoid.
        """
        env = dict(nccl_env or {})
        for key, value in (getattr(self.config, "extra_env", None) or {}).items():
            env[str(key)] = str(value)
        return env

    def _engine_image(self) -> str:
        """Container image for THIS instance — per-load override, else the
        fleet default. Lets one node run a 0.17 model and a 0.27 model side by
        side (both images coexist; docker doesn't care)."""
        return (getattr(self.config, "engine_image", "") or "").strip() or NVIDIA_VLLM_IMAGE

    def _serve_argv_prefix(self, image: str) -> List[str]:
        """Tokens to place before ``<model>`` so the container runs
        ``vllm serve <model>`` exactly once.

        Engine images disagree on ENTRYPOINT, and a per-instance image makes
        that our problem:
          * the pinned default is ``/opt/nvidia/nvidia_entrypoint.sh`` — a
            passthrough shim that also sets up the CUDA env, so we must supply
            ``vllm serve`` ourselves (and must NOT override the entrypoint);
          * ``vllm/vllm-openai`` bakes ENTRYPOINT ``["vllm","serve"]``, so
            passing our own ``vllm serve`` produced
            ``vllm serve vllm serve <model>`` → "unrecognized arguments".

        Unknown/unreadable images fall back to the legacy prefix, so a docker
        hiccup can never silently change how the default image is launched.
        """
        ep = self._image_entrypoint(image)
        if not ep:
            return ["vllm", "serve"]
        tail = [str(t) for t in ep]
        if tail[-1] == "serve":          # e.g. ["vllm","serve"] — fully baked in
            return []
        if Path(tail[-1]).name == "vllm":  # entrypoint is vllm itself
            return ["serve"]
        return ["vllm", "serve"]           # shim/shell entrypoint

    def _image_present(self, image: str) -> bool:
        """True if the image is already on this host."""
        try:
            out = subprocess.run(["docker", "image", "inspect", image],
                                 capture_output=True, text=True, timeout=20)
            return out.returncode == 0
        except Exception:
            return False

    def ensure_image(self, image: str, timeout: float = IMAGE_PULL_TIMEOUT) -> bool:
        """Make sure ``image`` is on this host, pulling it if it isn't.

        Per-model ``engine_image`` means a node can be asked for an image it has
        never run. Without this the launch just fails: docker starts an implicit
        pull, the launch confirmation times out underneath it, and the caller
        gets a bare "Failed to launch engine" with no mention of an image. The
        entrypoint probe degrades too, since ``docker inspect`` on a missing
        image returns nothing and we silently fall back to a default prefix.
        Observed 2026-08-25 loading a recipe model onto a fresh node.
        """
        if self._image_present(image):
            return True
        logger.info("Engine image %s not present; pulling (this can take several "
                    "minutes for a ~20 GB image)", image)
        try:
            out = subprocess.run(["docker", "pull", image],
                                 capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.error("Timed out pulling engine image %s after %ss", image, timeout)
            return False
        except Exception as exc:
            logger.error("Could not pull engine image %s: %s", image, exc)
            return False
        if out.returncode != 0:
            logger.error("Failed to pull engine image %s: %s", image,
                         (out.stderr or out.stdout or "").strip()[-400:])
            return False
        logger.info("Pulled engine image %s", image)
        return True

    def _image_entrypoint(self, image: str) -> List[str]:
        """The image's configured ENTRYPOINT, or [] when unknown."""
        try:
            out = subprocess.run(
                ["docker", "inspect", "-f", "{{json .Config.Entrypoint}}", image],
                capture_output=True, text=True, timeout=15,
            )
            if out.returncode != 0:
                return []
            return json.loads((out.stdout or "").strip() or "null") or []
        except Exception:
            return []

    def _is_pinned_default_image(self) -> bool:
        """True when this instance runs the pinned default engine image, i.e.
        when the 0.17-era GB10 workarounds still apply. A caller who pins a
        different image is asserting they know that engine's requirements, and
        can always re-add a workaround explicitly via ``extra_vllm_args``."""
        return self._engine_image() == NVIDIA_VLLM_IMAGE

    def _is_nvfp4_model(self) -> bool:
        """Detect NVFP4 from the on-disk config.json quantization metadata (with
        the model id as a fallback) so the MARLIN serve env is applied only when
        needed."""
        mid = (self.config.model or "").lower()
        if "nvfp4" in mid:
            return True
        local = self._local_model_dir()
        if not local:
            return False
        try:
            cfg = json.loads((Path(local) / "config.json").read_text())
            blob = json.dumps(cfg.get("quantization_config") or {}).lower()
            return "nvfp4" in blob or "fp4" in blob
        except Exception:
            return False

    def _nvfp4_serve_env(self) -> Dict[str, str]:
        """GB10/sm121 NVFP4 serve fix: the default FlashInfer CUTLASS FP4 GEMM
        emits `cvt .e2m1x2` PTX that ptxas rejects on sm_121 (fatal even with
        --enforce-eager). Force the MARLIN backend — env-only, no image rebuild,
        and more KV-cache-memory-efficient. Applied only when serving an NVFP4
        model. See memory ainode-gb10-quant-nvfp4-serving."""
        if not self._is_nvfp4_model() or not self._is_pinned_default_image():
            return {}
        return {
            "VLLM_USE_FLASHINFER_MOE_FP4": "0",
            "VLLM_NVFP4_GEMM_BACKEND": "marlin",
            "VLLM_TEST_FORCE_FP8_MARLIN": "1",
        }

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
        # API id(s) clients address the model by. Custom aliases (e.g. "Aegis-14B")
        # win; otherwise pin the repo-id so /v1/models is stable even when serving
        # from a local mount path. A list emits multiple --served-model-name values.
        names = self.config.served_model_name or [self.config.model]
        name_args: List[str] = ["--served-model-name", *names]
        in_container = os.environ.get("AINODE_IN_CONTAINER")
        mount_trustworthy = (not in_container) or bool(os.environ.get("AINODE_HOST_HOME"))
        if mount_trustworthy:
            local = self._local_model_dir()
            if local:
                slug = self.config.model.replace("/", "--")
                return f"{self.MODELS_MOUNT}/{slug}", name_args
        # Remote (vLLM downloads the repo-id) — previously emitted NO name args, so
        # the served id was the full repo-id with no alias option. Now aliasable too.
        return self.config.model, name_args

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
            # NO --rm: an engine that dies during startup must leave a corpse.
            # With --rm the container deleted itself on crash, so a failed launch
            # left zero logs and nothing in `docker ps -a` — every startup failure
            # looked like silence (2026-08-14: ~18 self-erased corpses on spark4,
            # visible only as bare container-ID hashes in nvidia-vllm.log).
            # start_solo() already stop/rm's a leftover by name, so nothing leaks.
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
            f"{hf_cache}:{HF_CACHE_MOUNT}",
            # Mount the on-disk model store read-only so an already-downloaded
            # model serves straight from disk (no re-download).
            "-v",
            f"{models_src}:{self.MODELS_MOUNT}:ro",
        ]
        # Operator/recipe mounts (e.g. a writable JIT cache the image compiles
        # kernels into). Last, so they can never displace the two above.
        cmd.extend(self._volume_args())
        for key, value in self._engine_env(nccl_env).items():
            cmd.extend(["-e", f"{key}={value}"])

        image = self._engine_image()
        cmd.extend([image, *self._serve_argv_prefix(image), serve_target])
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
            "-v", f"{self._host_path(hf_cache_dir)}:{HF_CACHE_MOUNT}",
        ]
        cmd.extend(self._volume_args())
        for key, value in self._engine_env(nccl_env).items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self._engine_image(), "-c", ray_cmd])
        return cmd

    def _build_mp_docker_cmd(
        self,
        *,
        container_name: str,
        node_rank: int,
        nnodes: int,
        master_addr: str,
        hf_cache_dir: str,
        node_ip: str,
    ) -> List[str]:
        """Build the ``docker run -d ... vllm serve ...`` command for ONE node of
        the mp multi-node shape.

        Every node runs the same command in its own container, differing only in
        ``--node-rank`` and the ``--headless`` that rank >= 1 carries (rank 0 is
        the one that serves the OpenAI API). There is no Ray container and no
        ``docker exec``: this container IS the engine, which is what makes the
        shape work on an image that ships no ``ray`` (the GB10 DeepSeek build,
        and stock ``vllm/vllm-openai``).

        Serves the repo-id rather than the local mount path, exactly like the
        Ray shape's ``docker exec``: the model identifier has to resolve
        identically on every rank, and only the HF cache is mounted on a peer
        (``_ensure_peer_has_model`` fills it over the fabric).
        """
        is_head = node_rank == 0
        nccl_env = self._build_nccl_env(
            is_head=is_head,
            head_fabric_ip=master_addr,
            peer_fabric_ip=(None if is_head else node_ip),
        )

        cmd: List[str] = [
            "docker", "run", "-d",
            "--name", container_name,
            "--gpus", "all",
            "--network", "host",
            "--ipc", "host",
            "--shm-size", MP_SHM_SIZE,
            "--ulimit", MP_ULIMIT_MEMLOCK,
            "--ulimit", MP_ULIMIT_STACK,
        ]
        cmd.extend(self._infiniband_device_args())
        cmd.extend(["-v", f"{self._host_path(hf_cache_dir)}:{HF_CACHE_MOUNT}"])
        cmd.extend(self._volume_args())
        for key, value in self._engine_env(nccl_env).items():
            cmd.extend(["-e", f"{key}={value}"])

        image = self._engine_image()
        cmd.extend([image, *self._serve_argv_prefix(image), self.config.model])
        cmd.extend(self._build_vllm_serve_args(tp_size=nnodes, executor_backend="mp"))
        cmd.extend(self._mp_rendezvous_args(
            node_rank=node_rank, nnodes=nnodes, master_addr=master_addr,
        ))
        return cmd

    def _mp_rendezvous_args(self, *, node_rank: int, nnodes: int,
                            master_addr: str) -> List[str]:
        """The ``--nnodes/--node-rank/--master-addr/--master-port`` set (plus
        ``--headless`` on rank >= 1) that joins one container to the mp cluster.

        Honours the same duplicate-suppression rule as the serve args: a recipe
        that states one of these itself keeps its value and we stay quiet.
        """
        supplied = self._supplied_flags()
        args: List[str] = []
        if "--nnodes" not in supplied:
            args.extend(["--nnodes", str(nnodes)])
        if "--node-rank" not in supplied:
            args.extend(["--node-rank", str(node_rank)])
        if "--master-addr" not in supplied:
            args.extend(["--master-addr", master_addr])
        if "--master-port" not in supplied:
            args.extend(["--master-port", self._master_port()])
        # Rank 0 serves the API; every other rank is a worker with no HTTP server.
        if node_rank > 0 and "--headless" not in supplied:
            args.append("--headless")
        return args

    def _start_distributed_mp(self, *, fabric_ip: str, hf_cache_dir: str) -> bool:
        """Launch the mp multi-node shape: one ``vllm serve`` container per node.

        PEERS FIRST, then the head. vLLM's mp rendezvous is a torch
        ``init_process_group`` on ``--master-addr:--master-port``: rank 0 binds
        it and blocks until all ``--nnodes`` ranks have dialled in, so a peer
        launched afterwards is fine but a peer launched first costs nothing and
        removes the window where rank 0 is up with nobody to talk to. There is
        deliberately no wait on the head reaching Running before the peers go
        out (the Ray shape needs that because a worker cannot connect to an
        unbound :6379; here the rendezvous does the waiting).
        """
        nnodes = self._tp_size()
        image = self._engine_image()
        if not self.ensure_image(image):
            logger.error("Engine image %s unavailable on the head; not launching %s",
                         image, self.config.model)
            return False

        for rank, peer_ip in enumerate(self.config.peer_ips, start=1):
            self._ssh_launch_mp_worker(
                peer_ip=peer_ip, head_ip=fabric_ip, node_rank=rank, nnodes=nnodes,
            )

        head_name = self._head_container_name()
        self._docker_stop_and_rm_best_effort(head_name)
        cmd = self._build_mp_docker_cmd(
            container_name=head_name,
            node_rank=0,
            nnodes=nnodes,
            master_addr=fabric_ip,
            hf_cache_dir=hf_cache_dir,
            node_ip=fabric_ip,
        )
        logger.info(
            "Launching mp head container (rank 0 of %d, master %s:%s): %s",
            nnodes, fabric_ip, self._master_port(), " ".join(cmd),
        )
        try:
            result = subprocess.run(
                cmd, env=self._build_env_for_subprocess(),
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            self._teardown_mp_peers()
            raise NvidiaBackendError(
                f"docker run -d for mp head container {head_name!r} timed out "
                f"after {exc.timeout}s; the local docker daemon may be unresponsive."
            ) from exc
        if result.returncode != 0:
            self._teardown_mp_peers()
            raise NvidiaBackendError(
                f"docker run -d for mp head container {head_name!r} failed "
                f"(rc={result.returncode}): {(result.stderr or '').strip()}"
            )

        # The head container is the engine, so its stdout is the engine log, so
        # follow it so readiness, load phase and last_log_activity all keep
        # working for a shape with no docker-exec'd process to watch.
        self._follow_container_logs(head_name)

        if not self._confirm_container_started(head_name):
            # A rank-0 container that died on its flags (or a missing vllm in the
            # image) leaves the peers waiting on a rendezvous that will never
            # happen; take them down with it.
            self._teardown_mp_peers()
            return False
        return True

    def _follow_container_logs(self, container_name: str) -> None:
        """Stream ``docker logs -f <container>`` into the distributed log file.

        The resulting Popen becomes ``self._process``, which is what
        ``is_running`` / ``wait_ready`` / ``stop`` already watch, and feeds
        ``_stream_logs`` so ``last_log_activity`` (the adaptive bind wait's
        liveness signal) comes from the engine's own stdout.
        """
        self._process = subprocess.Popen(
            ["docker", "logs", "-f", container_name],
            env=self._build_env_for_subprocess(),
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

    def _teardown_mp_peers(self) -> None:
        """Best-effort removal of every peer container (failed head launch)."""
        for peer_ip in self.config.peer_ips:
            try:
                self._ssh_stop_peer_container(peer_ip)
            except Exception:  # pragma: no cover - best-effort teardown
                logger.exception("failed to tear down mp peer %s", peer_ip)

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

    def _extra_vllm_args(self) -> List[str]:
        """``config.extra_vllm_args`` as a list of strings (never None)."""
        return [str(a) for a in (getattr(self.config, "extra_vllm_args", None) or [])]

    def _supplied_flags(self) -> set:
        """Flag names the caller supplied in ``extra_vllm_args``, in both the
        ``--flag value`` and ``--flag=value`` forms.

        Anything in here SUPPRESSES the same built-in flag rather than appearing
        twice: vLLM errors on duplicates, and the caller's explicit value is the
        intent. Shared by the serve-arg builder and the mp rendezvous args.
        """
        return {a.split("=", 1)[0] for a in self._extra_vllm_args() if a.startswith("--")}

    def _build_vllm_serve_args(self, tp_size: int,
                               executor_backend: str = "ray") -> List[str]:
        """Assemble the positional ``vllm serve`` args after ``<model>``.

        ``config.extra_vllm_args`` is appended verbatim so a model's published
        recipe (spec-decode, MoE/mamba backends, reasoning + tool-call parsers)
        can be expressed without hand-rolling a container. A flag supplied there
        SUPPRESSES the same built-in flag rather than appearing twice — vLLM
        errors on duplicates, and the caller's explicit value is the intent.

        ``executor_backend`` is the value emitted for
        ``--distributed-executor-backend`` when TP > 1: "ray" for the Ray
        container shape, "mp" for vLLM's own multi-node executor.
        """
        extra: List[str] = self._extra_vllm_args()
        supplied = self._supplied_flags()

        def wanted(flag: str) -> bool:
            return flag not in supplied

        args: List[str] = [
            "--host", "0.0.0.0",
            "--port", str(self.config.api_port),
        ]
        if wanted("--gpu-memory-utilization"):
            args.extend(["--gpu-memory-utilization", str(self.config.gpu_memory_utilization)])
        args.extend(self._legacy_gb10_args(supplied))
        # fp8 KV cache — the GB10 design default (engine/AGENTS.md): required for
        # long context or vLLM OOMs sizing the cache at bf16. Config-driven so a
        # model/quant that rejects fp8 can fall back via kv_cache_dtype="".
        # _effective_kv_cache_dtype downgrades the fp8 DEFAULT to auto for
        # multimodal models (fp8 corrupts VLM generation on GB10).
        if wanted("--kv-cache-dtype"):
            kv_dtype = self._effective_kv_cache_dtype()
            if kv_dtype:
                args.extend(["--kv-cache-dtype", kv_dtype])
        if tp_size > 1 and wanted("--tensor-parallel-size"):
            args.extend(["--tensor-parallel-size", str(tp_size)])
            if wanted("--distributed-executor-backend"):
                args.extend(["--distributed-executor-backend", executor_backend])
        if self.config.max_model_len and wanted("--max-model-len"):
            args.extend(["--max-model-len", str(self.config.max_model_len)])
        if self.config.quantization and wanted("--quantization"):
            args.extend(["--quantization", self.config.quantization])
        if self.config.trust_remote_code and wanted("--trust-remote-code"):
            args.append("--trust-remote-code")
        args.extend(extra)
        return args

    def _legacy_gb10_args(self, supplied: set) -> List[str]:
        """The 0.17-era GB10 workaround flags — emitted ONLY for the pinned
        default image (see module header)."""
        if not self._is_pinned_default_image() or "--enforce-eager" in supplied:
            return []
        return [
            # THE GB10/sm120 fix (verified 2026-06-17). FlashInfer's prefill
            # kernel illegal-instructions under CUDA-graph capture on GB10 and
            # kills EngineCore on the first real prefill. Fixed upstream by
            # 0.27.1, where forcing eager only costs throughput.
            "--enforce-eager",
        ]

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
        for key, value in self._engine_env(nccl_env).items():
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
        peer_hf_cache = self._peer_hf_cache()

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
        self._ssh_run_worker_container(
            peer_ip=peer_ip, worker_name=worker_name,
            peer_hf_cache=peer_hf_cache, docker_cmd=docker_cmd,
        )

    def _peer_hf_cache(self) -> str:
        """HF cache path on a peer, mounted into its container at HF_CACHE_MOUNT.

        Workers can't always write to NFS (runbook 02 § Observations / gotcha 2),
        so default to a home-directory path under the ssh_user's home. We can't
        use /root because we SSH in as the non-root ssh_user on the peer.
        """
        return f"/home/{self.config.ssh_user}/ainode-nvidia-cache"

    def _ssh_launch_mp_worker(self, peer_ip: str, head_ip: str,
                              node_rank: int, nnodes: int) -> None:
        """SSH to ``peer_ip`` and launch its ``--headless`` mp rank.

        Same plumbing as :meth:`_ssh_launch_worker`: same ssh target, same
        stale-container removal, same container name, same weight distribution
        but the container runs ``vllm serve ... --node-rank <k> --headless``
        instead of ``ray start``.
        """
        peer_hf_cache = self._peer_hf_cache()
        self._ensure_peer_has_model(peer_ip, peer_hf_cache)
        worker_name = self._worker_container_name(peer_ip)
        docker_cmd = self._build_mp_docker_cmd(
            container_name=worker_name,
            node_rank=node_rank,
            nnodes=nnodes,
            master_addr=head_ip,
            hf_cache_dir=peer_hf_cache,
            node_ip=peer_ip,
        )
        logger.info("SSH-launching mp rank %d on %s", node_rank, peer_ip)
        self._ssh_run_worker_container(
            peer_ip=peer_ip, worker_name=worker_name,
            peer_hf_cache=peer_hf_cache, docker_cmd=docker_cmd,
        )

    def _ssh_run_worker_container(self, *, peer_ip: str, worker_name: str,
                                  peer_hf_cache: str,
                                  docker_cmd: List[str]) -> None:
        """Run ``docker_cmd`` on a peer over SSH (shared by both shapes)."""
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
        self._wait_for_container_name_to_clear(container_name)

    # How long to wait for the daemon to finish removing a container whose name
    # we are about to reuse. Engines run with ``--rm``, so after ``docker stop``
    # the daemon removes them asynchronously; ``docker rm -f`` returns while that
    # removal is still in flight, and a ``docker run --name`` issued in that gap
    # fails with "Conflict. The container name ... is already in use". Seen on
    # every engine of the 0.5.8 roll (2026-09-14): the first launch died at 0 s
    # and replay burned a second launch per engine. A 27B engine takes seconds
    # to tear down; 90 s is generous without hiding a truly stuck daemon.
    NAME_CLEAR_TIMEOUT_S = 90.0
    NAME_CLEAR_POLL_S = 1.0

    def _container_name_in_use(self, container_name: str) -> bool:
        """True while the daemon still knows a container by this exact name.

        Uses ``check_output`` rather than ``run`` on purpose: the launch tests
        fake ``subprocess.run`` and count its calls positionally (stop, rm, run),
        and a poll routed through the same seam would shift those counts. Only a
        line that looks like a container id counts; anything else, or any
        failure to ask the daemon, reads as "not in use" so nothing spins.
        """
        try:
            out = subprocess.check_output(
                ["docker", "ps", "-aq", "--filter", f"name=^/{container_name}$"],
                text=True, timeout=15, stderr=subprocess.DEVNULL,
            )
        except Exception:
            return False
        return any(_CONTAINER_ID_RE.match(line.strip()) for line in out.splitlines())

    def _wait_for_container_name_to_clear(self, container_name: str) -> bool:
        """Block until no container carries ``container_name``; False on timeout."""
        deadline = time.monotonic() + self.NAME_CLEAR_TIMEOUT_S
        waited = 0.0
        while self._container_name_in_use(container_name):
            if time.monotonic() >= deadline:
                logger.warning(
                    "container name %s still in use after %.0fs; launching anyway",
                    container_name, self.NAME_CLEAR_TIMEOUT_S)
                return False
            time.sleep(self.NAME_CLEAR_POLL_S)
            waited += self.NAME_CLEAR_POLL_S
        if waited:
            logger.info("container name %s cleared after %.0fs", container_name, waited)
        return True

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
        return detect_fabric_ip(resolve_cluster_interface(self.config))

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
        self._last_log_activity = time.time()
        with open(target, "a") as sink:
            for line in process.stdout:
                sink.write(line)
                sink.flush()
                self._last_log_activity = time.time()
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
