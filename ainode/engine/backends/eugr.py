"""EugrBackend — drive vLLM inside the AINode unified container image
via eugr/spark-vllm-docker's launch-cluster.sh.

Running *inside* the container (which is how the image ships — systemd on the
host runs ``docker run ... ainode`` once), this backend handles two modes:

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

import ipaddress
import json
import logging
import os
import re
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
from ainode.engine.backends.base import EngineBackend

logger = logging.getLogger(__name__)

# Path to eugr's launcher inside the unified image. See scripts/Dockerfile.ainode.
EUGR_LAUNCHER = Path("/opt/spark-vllm-docker/launch-cluster.sh")
EUGR_ENV_FILE = Path("/opt/spark-vllm-docker/.env")

# Per-node NCCL init shim. Baked into the ainode image via Dockerfile COPY;
# AINode copies it onto shared storage at launch so every peer's vllm_node
# container can mount it via ``-v`` and run it via ``--entrypoint``. The shim
# detects each node's local HCAs (mlx5_* or rocep*/roceP*), filters by Up
# state and cluster subnet, and exports NCCL_IB_HCA before exec-ing CMD.
#
# TODO(v0.4.10): bake the shim into ainode-base so every vllm_node has it at
# /usr/local/bin/nccl-env-init.sh without needing NFS distribution. Then the
# NCCL_INIT_SHARED_* paths and ``_publish_nccl_init_script`` become optional.
NCCL_INIT_IMAGE_PATH = Path("/usr/local/bin/nccl-env-init.sh")
NCCL_INIT_SHARED_DIR = Path("/mnt/shared-models/.ainode")
NCCL_INIT_SHARED_PATH = NCCL_INIT_SHARED_DIR / "nccl-env-init.sh"
# In-container path used by ``--entrypoint`` AND by the head's exec-script source hook.
NCCL_INIT_CONTAINER_PATH = "/mnt/shared-models/.ainode/nccl-env-init.sh"


class EugrBackendError(RuntimeError):
    """Raised when the backend cannot be driven (missing binary, bad config)."""


# Back-compat alias — old code imports ``DockerEngineError`` from the
# ``docker_engine`` shim, which in turn re-exports this name.
DockerEngineError = EugrBackendError


class EugrBackend(EngineBackend):
    """Single backend that handles solo + head invocation of vLLM via eugr.

    Stateful only on the current process instance — the actual engine state
    (GPU memory, Ray cluster, peer containers) lives in the OS.
    """

    def __init__(self, config: NodeConfig, on_ready: Optional[Callable] = None):
        self.config = config
        self.on_ready = on_ready
        self._process: Optional[subprocess.Popen] = None
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
        raise EugrBackendError(
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
        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        self._log_thread = threading.Thread(
            target=self._stream_logs, args=(self._process, self._log_file), daemon=True
        )
        self._log_thread.start()
        return self._process.poll() is None

    def start_distributed(self) -> bool:
        """Invoke eugr's launcher for cross-node TP/PP.

        Requires ``config.distributed_mode == "head"``, a non-empty
        ``peer_ips`` list, and passwordless SSH from this node to every peer.
        """
        if self.config.distributed_mode != "head":
            raise EugrBackendError(
                "start_distributed() only runs when distributed_mode='head'. "
                f"Current mode: {self.config.distributed_mode!r}."
            )
        if not self.config.peer_ips:
            raise EugrBackendError(
                "peer_ips is empty; cannot launch distributed cluster without peers."
            )
        if not EUGR_LAUNCHER.exists():
            raise EugrBackendError(
                f"eugr launcher missing at {EUGR_LAUNCHER}. Is this running inside the ainode image?"
            )

        self._write_eugr_env()
        launch_script = self._write_distributed_launch_script()

        # Bug 3 fix: publish per-node shim to shared storage so every peer's
        # vllm_node can mount + exec it as --entrypoint. Returns None if the
        # shim/shared-storage isn't available; falls back cleanly to head-only
        # detection (bugs 1/2/4 still fixed).
        shim_container_path = self._publish_nccl_init_script()

        cmd = [str(EUGR_LAUNCHER), "--launch-script", str(launch_script)]
        extra_docker_args = [
            "-v",
            f"{self.config.models_dir or '/root/.ainode/models'}:/models",
        ]
        if shim_container_path is not None:
            # Mount the shared dir read-only and replace the vllm_node
            # container's default entrypoint with the shim. The shim detects
            # local HCAs, exports NCCL_IB_HCA, and execs the original CMD
            # (typically ``sleep infinity`` from eugr's launcher).
            extra_docker_args.extend([
                "-v", "/mnt/shared-models:/mnt/shared-models:ro",
                "--entrypoint", shim_container_path,
            ])
        env = self._build_env()
        env["VLLM_SPARK_EXTRA_DOCKER_ARGS"] = " ".join(extra_docker_args)

        logger.info(
            "Starting distributed vLLM: TP=%d across %d peers via eugr launcher",
            self._tp_size(),
            len(self.config.peer_ips),
        )
        self._process = subprocess.Popen(
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
            args=(self._process, self._distributed_log),
            daemon=True,
        )
        self._log_thread.start()
        return self._process.poll() is None

    def launch_distributed(self, sharding_config=None) -> bool:
        """Launch distributed inference — compatibility shim for the /api/models/load path.

        Accepts an optional sharding_config (from VLLMEngine's interface) but
        delegates to start_distributed() which reads peer_ips and TP size from
        self.config. If the config isn't already in head mode, we flip it.
        """
        if sharding_config is not None:
            # Apply sharding config fields to our config
            if hasattr(sharding_config, "model") and sharding_config.model:
                self.config.model = sharding_config.model
            if hasattr(sharding_config, "peer_ips") and sharding_config.peer_ips:
                self.config.peer_ips = sharding_config.peer_ips
            if hasattr(sharding_config, "strategy"):
                pass  # TP vs PP handled by eugr launcher via env vars

        if self.config.distributed_mode != "head":
            self.config.distributed_mode = "head"
            try:
                self.config.save()
            except Exception:
                pass

        return self.start_distributed()

    def stop(self) -> None:
        """Graceful shutdown. For distributed, also invokes eugr's ``stop``."""
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

    def wait_ready(self, timeout: float = 300.0) -> bool:
        """Poll ``/v1/models`` on the API port until 2xx or timeout."""
        url = f"http://127.0.0.1:{self.config.api_port}/v1/models"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._process and self._process.poll() is not None:
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
        return self._process is not None and self._process.poll() is None

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

    @property
    def process(self) -> Optional[subprocess.Popen]:
        return self._process

    @process.setter
    def process(self, value: Optional[subprocess.Popen]) -> None:
        # Preserve the mutability of the original public ``process`` attr
        # — existing tests set it directly to mock instances.
        self._process = value

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
            quant = self.config.quantization
        elif self.config.model and "awq" in self.config.model.lower():
            # Model name contains AWQ — explicitly pin to awq (not awq_marlin).
            # vLLM auto-upgrades AWQ → awq_marlin which requires Marlin CUDA
            # kernels not compiled for GB10 (sm_12.1) in the eugr base image.
            quant = "awq"
        else:
            quant = None
        if quant:
            cmd.extend(["--quantization", quant])
        if self.config.models_dir:
            cmd.extend(["--download-dir", self.config.models_dir])
        if gpu and gpu.unified_memory:
            cmd.extend(["--dtype", "bfloat16"])
        return cmd

    def _build_env(self) -> dict:
        """Base environment for subprocess calls, including NCCL/RoCE tuning."""
        env = os.environ.copy()
        env.setdefault("HF_HOME", self.config.models_dir or "/root/.ainode/models")

        # Propagate HF token if configured — needed for gated repos (Llama etc.)
        if self.config.hf_token:
            env["HUGGING_FACE_HUB_TOKEN"] = self.config.hf_token
            env["HF_TOKEN"] = self.config.hf_token

        iface = self.config.cluster_interface
        if iface:
            env["NCCL_SOCKET_IFNAME"] = iface
            env["GLOO_SOCKET_IFNAME"] = iface
            env["UCX_NET_DEVICES"] = iface

        # Bugs 1/2/4 fix: accept both MOFED (mlx5_*) and stock rdma-core
        # (rocep*/roceP*) naming, filter by (Up) state, filter by cluster
        # subnet so direct-connect HCAs on dual-homed nodes are excluded.
        # If detection returns nothing, leave NCCL_IB_HCA unset — better for
        # NCCL to auto-detect locally than to pin to a hardcoded name that
        # may not exist on this host (the removed "mlx5_0" fallback).
        ib_hca = self._detect_ib_hca(subnet_cidr=self._cluster_subnet())
        if ib_hca:
            env.setdefault("NCCL_IB_HCA", ib_hca)
        env.setdefault("NCCL_IB_DISABLE", "0")
        env.setdefault("NCCL_P2P_DISABLE", "0")
        env.setdefault("NCCL_NET_GDR_LEVEL", "5")
        env.setdefault("NCCL_IGNORE_CPU_AFFINITY", "1")

        return env

    def _cluster_subnet(self) -> Optional[str]:
        """Return the CIDR of the ``cluster_interface`` netdev (e.g. '192.168.0.0/24').

        Needed to filter out vestigial direct-connect HCAs (10.0.0.x on
        dual-homed Sparks 1 & 2 in our reference cluster) vs the switched
        fabric HCAs (192.168.0.x). Without this, ``_detect_ib_hca`` would
        return direct-connect devices that don't exist on all cluster nodes,
        hanging NCCL ring init.
        """
        iface = self.config.cluster_interface
        if not iface:
            return None
        try:
            out = subprocess.run(
                ["ip", "-o", "-4", "addr", "show", "dev", iface],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode != 0:
                return None
            m = re.search(r"inet\s+(\S+)", out.stdout)
            if not m:
                return None
            return str(ipaddress.ip_network(m.group(1), strict=False))
        except Exception:
            return None

    @staticmethod
    def _detect_ib_hca(subnet_cidr: Optional[str] = None) -> Optional[str]:
        """Return a comma-joined list of Up, on-subnet HCAs, or None.

        Reads ``/sys/class/infiniband/`` directly. Prior implementation
        shelled out to ``ibdev2netdev``, but that tool ships in MOFED /
        infiniband-diags on the host and is NOT in our containers (neither
        the eugr base nor the ainode layer). sysfs is kernel-provided and
        always available.

        Accepts any HCA name the kernel exposes. Udev rules set the name:
        MOFED gives ``mlx5_*``, stock Ubuntu rdma-core gives ``rocep*`` /
        ``roceP*``. Regex limits to those patterns so the detector doesn't
        pick up unrelated entries (virtual HCAs, etc.).

        Port state is parsed as the leading integer and matched against the
        ``ib_port_state`` kernel enum (``include/rdma/ib_verbs.h``):
        0 = NOP, 1 = DOWN, 2 = INIT, 3 = ARMED, 4 = ACTIVE, 5 = ACTIVE_DEFER.
        Accept {4, 5}, reject everything else. Leading-integer parse is
        format-stable where substring matching on the textual label is not
        (kernels emit ``"4: ACTIVE"``, ``"4\\n"``, ``"4 : ACTIVE"``, etc.).

        Subnet filter: if ``subnet_cidr`` is given, exclude HCAs whose
        netdev does not have an IP inside that subnet. On nodes 1 & 2 in
        our reference cluster, this filters out the vestigial direct-
        connect fabric (10.0.0.x) so it never enters the NCCL ring.

        Returns None if no HCA passes the filters — caller should leave
        ``NCCL_IB_HCA`` unset rather than fall back to a guess.

        TODO(v0.4.10): eugr's ``autodiscover.sh`` also depends on
        ``ibdev2netdev``. When this function returns None, the launcher
        writes ``IB_IF=`` (empty) → eugr runs its own autodiscover →
        fails with the same "ibdev2netdev not found" error. Upstream a
        /sys-based autodiscover to eugr, or wrap it.
        """
        ib_base = Path("/sys/class/infiniband")
        if not ib_base.is_dir():
            return None

        name_re = re.compile(r"^(mlx5_\d+|rocep\w+|roceP\w+)$")

        subnet_obj = None
        if subnet_cidr:
            try:
                subnet_obj = ipaddress.ip_network(subnet_cidr, strict=False)
            except Exception:
                subnet_obj = None

        try:
            hca_dirs = sorted(ib_base.iterdir())
        except Exception:
            return None

        devs: List[str] = []
        for hca_dir in hca_dirs:
            hca = hca_dir.name
            if not name_re.match(hca):
                continue

            # Port state — parse leading integer and match against the
            # ib_port_state enum (include/rdma/ib_verbs.h):
            #   0 = NOP, 1 = DOWN, 2 = INIT, 3 = ARMED,
            #   4 = ACTIVE (normal up),
            #   5 = ACTIVE_DEFER (also functional for traffic).
            # Accept {4, 5}, reject everything else. File content on recent
            # kernels is "4: ACTIVE\n" but can be "4\n" or "4 : ACTIVE" on
            # others — leading-integer parse is format-stable where a
            # substring match on the textual label is not.
            try:
                state_text = (hca_dir / "ports" / "1" / "state").read_text().strip()
            except Exception:
                continue
            state_match = re.match(r"(\d+)", state_text)
            if not state_match or int(state_match.group(1)) not in (4, 5):
                continue

            # Netdev via /sys/class/infiniband/<hca>/device/net/<netdev>.
            # iterdir() returns empty if no netdev is associated (e.g.
            # a pure-IB port with no Ethernet overlay); skip those.
            try:
                netdev_dir = hca_dir / "device" / "net"
                netdevs = list(netdev_dir.iterdir())
            except Exception:
                continue
            if not netdevs:
                continue
            netdev = netdevs[0].name

            if subnet_obj is not None:
                try:
                    ip_out = subprocess.run(
                        ["ip", "-o", "-4", "addr", "show", "dev", netdev],
                        capture_output=True, text=True, timeout=3,
                    )
                    addr_m = re.search(r"inet\s+(\S+)", ip_out.stdout)
                    if not addr_m:
                        continue
                    netdev_ip = ipaddress.ip_interface(addr_m.group(1)).ip
                    if netdev_ip not in subnet_obj:
                        continue
                except Exception:
                    continue  # can't verify subnet → exclude, don't guess

            devs.append(hca)

        return ",".join(devs) or None

    # ------------------------------------------------------------------
    # Distributed (eugr) wiring
    # ------------------------------------------------------------------

    def _tp_size(self) -> int:
        """Total TP = local GPUs × (1 + number of peers). One GPU per GB10 node."""
        return 1 + len(self.config.peer_ips)

    def _publish_nccl_init_script(self) -> Optional[str]:
        """Publish the per-node shim to shared storage so every peer sees it.

        Bug 3 fix: per-node NCCL detection can't be broadcast from a single
        ``.env`` (eugr's launcher propagates one cluster-wide value), so we
        mount the shim into each peer's vllm_node via ``--entrypoint``. The
        shim itself is baked into the ainode image by Dockerfile COPY; here
        we stage it on the shared NFS path that every peer's ``docker run
        -v`` can bind-mount from.

        Returns the in-container path to pass to ``--entrypoint``, or None
        if publishing failed (shim missing from image, or shared storage
        unavailable). On None, ``start_distributed`` falls back to head-only
        detection — bugs 1, 2, 4 still fixed; bug 3 degrades to partial.

        TODO(v0.4.10): bake the shim into ainode-base so every vllm_node
        has it at ``/usr/local/bin/nccl-env-init.sh`` without NFS. Then this
        publish step and the ``/mnt/shared-models`` dependency go away.
        """
        if not NCCL_INIT_IMAGE_PATH.exists():
            logger.warning(
                "NCCL init shim %s missing from image; per-node NCCL env "
                "disabled (bugs 1/2/4 still fixed).",
                NCCL_INIT_IMAGE_PATH,
            )
            return None
        try:
            NCCL_INIT_SHARED_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(NCCL_INIT_IMAGE_PATH, NCCL_INIT_SHARED_PATH)
            NCCL_INIT_SHARED_PATH.chmod(0o755)
            logger.info("Published NCCL shim to %s", NCCL_INIT_SHARED_PATH)
            return NCCL_INIT_CONTAINER_PATH
        except Exception as exc:
            logger.warning(
                "Could not publish NCCL shim to %s: %s. Falling back to "
                "head-only detection (bugs 1/2/4 still fixed).",
                NCCL_INIT_SHARED_PATH, exc,
            )
            return None

    def _write_eugr_env(self) -> None:
        """Populate ``/opt/spark-vllm-docker/.env`` for the launcher."""
        iface = self.config.cluster_interface
        head_ip = _local_ip_for_interface(iface)
        cluster_nodes = ",".join([head_ip] + self.config.peer_ips)
        subnet = self._cluster_subnet()

        # Bug 4 fix: IB_IF carries the HCA device list (eugr maps it to
        # NCCL_IB_HCA at launch-cluster.sh:810). Prior code wrote the netdev
        # name here — ``IB_IF=enP2p1s0f1np1`` — which produced a garbage
        # ``NCCL_IB_HCA=enP2p1s0f1np1`` (a netdev is not an HCA device).
        ib_hca = self._detect_ib_hca(subnet_cidr=subnet) or ""

        lines = [
            f"CLUSTER_NODES={cluster_nodes}",
            f"ETH_IF={iface}",
            f"IB_IF={ib_hca}",
            "MASTER_PORT=29501",
            f"SSH_USER={self.config.ssh_user}",
            # CONTAINER_* vars are injected into every per-node container.
            "CONTAINER_NCCL_DEBUG=INFO",
            "CONTAINER_NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH",
            f"CONTAINER_NCCL_SOCKET_IFNAME={iface}",
            # Bug 3 fix: CONTAINER_NCCL_IB_HCA deliberately omitted. Each
            # vllm_node's per-node shim (mounted via --entrypoint) exports
            # its own node-local NCCL_IB_HCA at startup. A cluster-wide
            # value would be wrong for heterogeneous HCA naming.
            "CONTAINER_NCCL_IB_DISABLE=0",
            "CONTAINER_NCCL_IGNORE_CPU_AFFINITY=1",
            "CONTAINER_NCCL_NET_GDR_LEVEL=5",
            f"CONTAINER_UCX_NET_DEVICES={iface}",
            # Consumed by the per-node shim to filter HCAs by subnet.
            f"CONTAINER_AINODE_CLUSTER_SUBNET={subnet or ''}",
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

        # Bug 3 belt-and-suspenders: ``docker exec`` on the head does not
        # inherit PID-1 runtime env from ``--entrypoint``. Source the shim
        # in --export-only mode so NCCL_IB_HCA lands in THIS shell before
        # vllm starts, and thereby in every child (driver + Ray workers).
        # ``|| true`` and the presence check keep this safe if the shim
        # wasn't staged (operator without shared storage).
        env_init_hook = (
            f"if [ -x {NCCL_INIT_CONTAINER_PATH} ]; then\n"
            f'    eval "$({NCCL_INIT_CONTAINER_PATH} --export-only 2>/dev/null || true)"\n'
            f"fi\n"
        )

        script = f"""#!/bin/bash
# Auto-generated by ainode EugrBackend.start_distributed. Do not edit.
{env_init_hook}
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


# Back-compat alias — old code imports ``DockerEngine`` from the
# ``docker_engine`` shim, which in turn re-exports this name.
DockerEngine = EugrBackend


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
