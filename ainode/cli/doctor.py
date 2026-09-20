"""``ainode doctor``: the checks that say why a node is not working.

Every check here is one a human ran by hand during the 2026-09-19 fleet audit,
and nearly every gotcha that audit turned up was one command away from being
visible: a stale image pin, a cluster split across two releases, an
``engine_backend`` still on eugr, a discovery port mismatch, a member with no
fabric IP, a secrets store nobody had chmodded. So the command is deliberately
a *list of small facts* rather than a score: one line per check, OK / WARN /
FAIL, and a one-line fix hint wherever the answer is not OK.

Shape, so this stays testable and stays honest:

* Every check is a module-level function of plain values that returns
  ``list[Check]``. It never prints, never exits, and never raises: an
  unavailable probe is a WARN that says the probe was unavailable, never a
  dead run. ``tests/test_doctor.py`` drives each branch with fakes.
* Everything that touches the world goes through a module-level seam
  (:func:`run_command`, :func:`disk_usage`, :func:`tcp_listening`,
  :func:`udp_listeners`, :func:`http_json`, :func:`latest_image_tag`,
  :func:`probe_gpus`). Tests monkeypatch the seam, not the check.
* WARN means "this will bite you"; FAIL means "this node cannot do its job
  right now". Only a FAIL makes the command exit non-zero, because a fleet
  where every WARN is fatal is a fleet where nobody runs the doctor.
* ``--fix`` applies only fixes that cannot lose anything: create a missing
  directory, chmod the secrets store to 0600, write the fleet discovery port.
  Everything else is listed for a human.
* Secrets are reported by PRESENCE only. No value, no prefix, no length.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import stat
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ainode import __version__
from ainode.core.config import AINODE_HOME, DEFAULT_ENGINE_BACKEND, NodeConfig

OK = "ok"
WARN = "warn"
FAIL = "fail"

#: Rank order for the exit code and the summary line.
_SEVERITY = {OK: 0, WARN: 1, FAIL: 2}

#: The UDP port a fleet installed by ``scripts/install.sh`` announces on. It is
#: NOT ``NodeConfig.discovery_port``'s dataclass default (5678): the installer
#: writes 5679 and the shipped image EXPOSEs 5679/udp, so a config.json with the
#: key missing or carrying the old value leaves a node talking to nobody while
#: looking perfectly healthy. That mismatch is what this constant exists to
#: catch, and it is the one config value ``--fix`` will write.
FLEET_DISCOVERY_PORT = 5679

#: Free space below this fraction of a filesystem is a WARN. A GB10 node pays
#: for a model twice (download, then the engine's own cache), so "nearly full"
#: shows up as a launch that dies mid-pull rather than as a disk error.
DISK_WARN_FRACTION = 0.15

#: Above this, vLLM's KV cache leaves no room for a second engine, and the
#: stacked-load admission guard refuses anything that would pass 0.90.
GMU_WARN_AT = 0.9


@dataclass
class Check:
    """One fact about this node.

    ``name`` is a stable dotted id (machines group on it), ``detail`` is the one
    line a human reads, ``fix`` is the one line they act on, and ``data`` carries
    the raw values so ``--json`` is useful without re-parsing prose.
    """

    name: str
    status: str
    detail: str
    fix: str = ""
    data: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Seams. Everything that reads the world lives here so a test can replace it.
# ---------------------------------------------------------------------------

def run_command(argv: list[str], timeout: float = 10.0) -> tuple[int, str]:
    """Run ``argv`` and return ``(returncode, combined output)``.

    Never raises. A missing binary, a timeout or an OS error all come back as a
    non-zero code with the reason as the output, because every caller here wants
    to report the failure rather than handle it.
    """
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return 127, str(exc)
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return proc.returncode, out


def disk_usage(path) -> Optional[tuple[int, int]]:
    """``(total_bytes, free_bytes)`` for the filesystem holding ``path``, or None."""
    try:
        usage = shutil.disk_usage(str(path))
    except (OSError, ValueError):
        return None
    return usage.total, usage.free


def tcp_listening(port: int, host: str = "127.0.0.1", timeout: float = 0.8) -> bool:
    """True when something accepts a TCP connection on ``host:port``."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def udp_listeners() -> Optional[set[int]]:
    """UDP ports bound on this host, or None when we cannot tell.

    Read from ``ss -lunH`` rather than probed with a bind: the discovery
    listener sets SO_REUSEADDR, so a successful bind proves nothing about
    whether anything is listening. ``iproute2`` ships in the AINode image
    for exactly this (see scripts/Dockerfile.ainode).
    """
    code, out = run_command(["ss", "-lunH"], timeout=5)
    if code != 0:
        return None
    ports: set[int] = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        match = re.search(r":(\d+)$", parts[3])
        if match:
            ports.add(int(match.group(1)))
    return ports


def http_json(url: str, timeout: float = 3.0) -> Optional[dict]:
    """GET ``url`` and return the decoded JSON object, or None on any failure."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    return payload if isinstance(payload, dict) else None


def latest_image_tag() -> Optional[str]:
    """Highest numeric tag published for the AINode image, or None.

    One home for the GHCR query: this is the same resolver ``/api/version/check``
    answers from, imported late so ``ainode --help`` does not pay for aiohttp.
    """
    try:
        from ainode.api.server import _fetch_latest_ghcr_tag

        return _fetch_latest_ghcr_tag()
    except Exception:
        return None


def probe_gpus() -> list[dict]:
    """Every NVIDIA GPU this host enumerates, as plain dicts.

    Not :func:`ainode.core.gpu.detect_gpu`, which answers for device 0 only and
    caches: the doctor wants the COUNT as much as the names. ``unified`` is
    inferred the way detect_gpu infers it, from NVML refusing to report memory
    on GB10, and the figures then come from host RAM because that is what
    unified memory is.
    """
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", FutureWarning)
            import pynvml
        pynvml.nvmlInit()
    except Exception:
        return []

    found: list[dict] = []
    try:
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            total_mb = free_mb = 0
            unified = False
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_mb = int(mem.total) // (1024 * 1024)
                free_mb = int(mem.free) // (1024 * 1024)
            except Exception:
                unified = True
                try:
                    import psutil

                    vm = psutil.virtual_memory()
                    total_mb = int(vm.total) // (1024 * 1024)
                    free_mb = int(vm.available) // (1024 * 1024)
                except Exception:
                    pass
            found.append({
                "index": index,
                "name": str(name),
                "memory_total_mb": total_mb,
                "memory_free_mb": free_mb,
                "unified_memory": unified,
            })
    except Exception:
        pass
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return found


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def _gb(num_bytes: float) -> str:
    return f"{num_bytes / (1024 ** 3):.1f} GB"


def path_state(path) -> tuple[str, str]:
    """``("present"|"missing"|"denied", reason)`` for ``path``, without raising.

    ``Path.exists()`` RAISES on a path whose parent the caller may not read, and
    a doctor that dies on one unreadable directory is worse than no doctor.
    Found on Spark-2, whose ``models_dir`` is the container path
    ``/root/.ainode/models``: correct inside the engine container, unreadable by
    the operator on the host, and a crash rather than a finding.
    """
    try:
        return ("present", "") if Path(path).exists() else ("missing", "")
    except OSError as exc:
        return "denied", str(exc)


def _exists(path) -> bool:
    """``Path.exists()`` that answers False instead of raising."""
    return path_state(path)[0] == "present"


def model_dir_on_disk(models_dir, hf_repo: str) -> Optional[Path]:
    """Where ``hf_repo`` already lives under ``models_dir``, or None.

    One home for the layout list is :meth:`ainode.models.registry.find_model_dir`
    (our downloader writes ``org--name``, the Hub writes ``models--org--name``
    under ``hub/``); this only calls it, so a new layout is added there.
    """
    try:
        from ainode.models.registry import find_model_dir

        return find_model_dir(Path(models_dir), hf_repo)
    except Exception:
        return None


def read_image_env(home) -> Optional[str]:
    """The image ``$AINODE_HOME/image.env`` pins for the systemd unit, or None."""
    path = Path(home) / "image.env"
    try:
        text = path.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "AINODE_IMAGE" and value.strip():
            return value.strip()
    return None


def _version_tuple(tag: str) -> tuple:
    return tuple(int(part) for part in re.findall(r"\d+", tag or ""))


def load_config(config_path) -> tuple[NodeConfig, Optional[str]]:
    """``(config, error)``: defaults plus the reason when the file is unusable."""
    path = Path(config_path)
    if not _exists(path):
        return NodeConfig(), None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return NodeConfig(), str(exc)
    if not isinstance(data, dict):
        return NodeConfig(), "config.json is not a JSON object"
    known = {k: v for k, v in data.items() if k in NodeConfig.__dataclass_fields__}
    try:
        return NodeConfig(**known), None
    except TypeError as exc:
        return NodeConfig(), str(exc)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_sudo_trap(env: Optional[dict] = None, euid: Optional[int] = None) -> list[Check]:
    """The sudo trap: under sudo, ``$HOME`` is root's and every path moves.

    ``sudo ainode update`` wrote root's ``image.env`` and restarted a unit that
    reads the install user's, so the pull succeeded and the old image came back
    up with nothing saying so. Any doctor run under sudo without an explicit
    AINODE_HOME is reporting on the wrong home directory.
    """
    env = dict(os.environ if env is None else env)
    if euid is None:
        euid = getattr(os, "geteuid", lambda: -1)()
    sudo_user = env.get("SUDO_USER", "")
    pinned = env.get("AINODE_HOME", "")
    data = {"euid": euid, "sudo_user": sudo_user, "ainode_home_env": pinned}
    if euid != 0 or not sudo_user:
        return [Check("env.sudo", OK, "not running under sudo", data=data)]
    if pinned:
        return [Check("env.sudo", OK,
                      f"root under sudo with AINODE_HOME pinned to {pinned}",
                      data=data)]
    return [Check(
        "env.sudo", WARN,
        f"running as root under sudo (SUDO_USER={sudo_user}) with no AINODE_HOME, "
        f"so every path below resolves under root's home and not {sudo_user}'s",
        fix=f"re-run without sudo, or pass AINODE_HOME=~{sudo_user}/.ainode",
        data=data)]


def check_config_file(config_path, error: Optional[str]) -> list[Check]:
    path = Path(config_path)
    data = {"path": str(path), "exists": _exists(path)}
    if error:
        return [Check("config.file", FAIL, f"{path} could not be read: {error}",
                      fix=f"fix or delete {path}; AINode falls back to defaults without it",
                      data=dict(data, error=error))]
    if not _exists(path):
        return [Check("config.file", WARN,
                      f"no config.json at {path}, so every value below is a built-in default",
                      fix="ainode start writes one, or set values with ainode config",
                      data=data)]
    return [Check("config.file", OK, f"{path} parsed", data=data)]


def check_engine_backend(config: NodeConfig, config_path,
                         docker_ok: bool, image_present: Optional[bool]) -> list[Check]:
    """Is ``engine_backend`` a backend whose binary or image is actually here?

    This is the 0.5.26 install bug as a check: a node on the ``eugr`` backend
    needs a ``vllm`` on PATH, the shipped image is python:3.12-slim plus this
    package, and the first Launch click answered 500 with no way to guess why.
    """
    backend = (getattr(config, "engine_backend", "") or "").strip().lower() \
        or DEFAULT_ENGINE_BACKEND
    engine_image = _engine_image_for(config)
    data = {"engine_backend": backend, "engine_image": engine_image}

    if backend == "nvidia":
        if not docker_ok:
            return [Check("config.engine_backend", WARN,
                          f"nvidia backend wants image {engine_image}; docker is not "
                          f"answering here, so whether it is pulled is unknown",
                          fix="fix docker first (see the docker.daemon check)",
                          data=data)]
        if image_present:
            return [Check("config.engine_backend", OK,
                          f"nvidia backend, engine image {engine_image} present locally",
                          data=dict(data, image_present=True))]
        return [Check("config.engine_backend", WARN,
                      f"nvidia backend, engine image {engine_image} is not pulled on "
                      f"this node, so the first launch pays for it",
                      fix=f"docker pull {engine_image}",
                      data=dict(data, image_present=False))]

    if backend == "eugr":
        vllm = shutil.which("vllm")
        if vllm:
            return [Check("config.engine_backend", OK,
                          f"eugr backend with a vllm binary at {vllm}",
                          data=dict(data, vllm=vllm))]
        return [Check("config.engine_backend", FAIL,
                      "eugr backend needs a vllm binary on PATH and there is none, so "
                      "every launch fails before it starts",
                      fix=f'set "engine_backend": "nvidia" in {config_path}',
                      data=dict(data, vllm=None))]

    return [Check("config.engine_backend", FAIL,
                  f"engine_backend {backend!r} is not a backend AINode has "
                  f"(nvidia, eugr)",
                  fix=f'set "engine_backend": "nvidia" in {config_path}',
                  data=data)]


def _engine_image_for(config: NodeConfig) -> str:
    """The engine image this config launches. One home is the nvidia backend."""
    pinned = (getattr(config, "engine_image", "") or "").strip()
    if pinned:
        return pinned
    try:
        from ainode.engine.backends.nvidia import NVIDIA_VLLM_IMAGE

        return NVIDIA_VLLM_IMAGE
    except Exception:
        return ""


def check_gpu_memory_utilization(config: NodeConfig) -> list[Check]:
    try:
        gmu = float(getattr(config, "gpu_memory_utilization", 0) or 0)
    except (TypeError, ValueError):
        return [Check("config.gpu_memory_utilization", FAIL,
                      "gpu_memory_utilization is not a number",
                      fix="set it to 0.6", data={"value": None})]
    data = {"value": gmu}
    if gmu <= 0 or gmu > 1:
        return [Check("config.gpu_memory_utilization", FAIL,
                      f"gpu_memory_utilization is {gmu}, which is outside (0, 1]",
                      fix="set it to 0.6", data=data)]
    if gmu >= GMU_WARN_AT:
        return [Check("config.gpu_memory_utilization", WARN,
                      f"gpu_memory_utilization is {gmu}, so the stacked-load guard "
                      f"refuses every second model (the total may not pass 0.90)",
                      fix="set it to 0.6, which leaves 0.30 for one stacked neighbour",
                      data=data)]
    return [Check("config.gpu_memory_utilization", OK, f"{gmu}", data=data)]


def check_discovery_port(config: NodeConfig) -> list[Check]:
    port = int(getattr(config, "discovery_port", 0) or 0)
    data = {"value": port, "expected": FLEET_DISCOVERY_PORT, "fix_action": "discovery_port"}
    if port == FLEET_DISCOVERY_PORT:
        return [Check("config.discovery_port", OK, f"{port}", data=data)]
    return [Check("config.discovery_port", WARN,
                  f"discovery_port is {port}; a fleet installed the documented way "
                  f"announces on {FLEET_DISCOVERY_PORT}, so this node is alone on the wire",
                  fix=f"ainode doctor --fix writes {FLEET_DISCOVERY_PORT} (restart the service after)",
                  data=data)]


def check_cluster_id(config: NodeConfig, peers_seen: int = 0) -> list[Check]:
    cluster_id = str(getattr(config, "cluster_id", "") or "")
    peer_ips = list(getattr(config, "peer_ips", []) or [])
    dmode = str(getattr(config, "distributed_mode", "solo") or "solo")
    peers_expected = bool(peer_ips) or dmode != "solo" or peers_seen > 0
    data = {"cluster_id": cluster_id, "peer_ips": peer_ips,
            "distributed_mode": dmode, "peers_seen": peers_seen}
    if not cluster_id:
        return [Check("config.cluster_id", FAIL,
                      "cluster_id is empty, so discovery matches this node with nobody",
                      fix='set "cluster_id" to the name the rest of the fleet uses',
                      data=data)]
    if cluster_id == "default" and peers_expected:
        return [Check("config.cluster_id", WARN,
                      f"cluster_id is still \"default\" on a node that expects peers "
                      f"({len(peer_ips)} configured, {peers_seen} seen); two clusters on "
                      f"one LAN would merge",
                      fix='give the fleet its own cluster_id on every node',
                      data=data)]
    return [Check("config.cluster_id", OK, cluster_id, data=data)]


def check_model(config: NodeConfig) -> list[Check]:
    model = (getattr(config, "model", "") or "").strip()
    models_dir = getattr(config, "models_dir", "") or ""
    data = {"model": model or None, "models_dir": models_dir}
    if not model:
        return [Check("config.model", OK,
                      "no model pinned, so nothing loads at boot", data=data)]
    found = model_dir_on_disk(models_dir, model)
    if found is not None:
        return [Check("config.model", OK, f"{model} is on disk at {found}",
                      data=dict(data, path=str(found)))]
    return [Check("config.model", WARN,
                  f"{model} is pinned and not downloaded under {models_dir}, so the "
                  f"first launch pays the download (and answers 401 if it is gated)",
                  fix="download it from the Models view, or clear it with "
                      "ainode config --model ''",
                  data=dict(data, path=None))]


def check_docker(engine_image: str = "") -> list[Check]:
    """Is the docker daemon reachable, and is the engine image here?

    Returns the daemon check first; the engine-image answer rides in its
    ``data`` so :func:`check_engine_backend` does not shell out a second time.
    """
    code, out = run_command(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=15)
    first = (out.splitlines() or [""])[0][:200]
    if code == 127:
        return [Check("docker.daemon", FAIL, "no docker CLI on PATH",
                      fix="install docker; AINode runs both itself and the engine as containers",
                      data={"reachable": False, "image_present": None})]
    if code != 0:
        return [Check("docker.daemon", FAIL,
                      f"docker is installed and the daemon is not answering: {first}",
                      fix="sudo systemctl start docker, and check this user is in the docker group",
                      data={"reachable": False, "image_present": None, "error": first})]
    image_present: Optional[bool] = None
    if engine_image:
        img_code, _ = run_command(["docker", "image", "inspect", engine_image], timeout=15)
        image_present = img_code == 0
    return [Check("docker.daemon", OK, f"docker {first} reachable",
                  data={"reachable": True, "server_version": first,
                        "image_present": image_present})]


def check_gpus(gpus: Optional[list[dict]] = None) -> list[Check]:
    gpus = probe_gpus() if gpus is None else gpus
    if not gpus:
        return [Check("gpu.devices", FAIL,
                      "no NVIDIA GPU enumerated (NVML returned nothing)",
                      fix="check nvidia-smi and the driver; a node with no GPU cannot serve",
                      data={"count": 0, "gpus": []})]
    parts = []
    for gpu in gpus:
        mem = f"{gpu.get('memory_total_mb', 0) / 1024:.0f} GB"
        if gpu.get("unified_memory"):
            mem += " unified"
        parts.append(f"{gpu.get('name', '?')} ({mem})")
    return [Check("gpu.devices", OK,
                  f"{len(gpus)} GPU: " + ", ".join(parts),
                  data={"count": len(gpus), "gpus": gpus})]


def check_disk(home, models_dir) -> list[Check]:
    """Free space on the two directories a launch actually writes to."""
    checks: list[Check] = []
    for name, path in (("disk.home", Path(home)), ("disk.models", Path(models_dir))):
        state, reason = path_state(path)
        if state == "missing":
            checks.append(Check(name, WARN, f"{path} does not exist",
                                fix="ainode doctor --fix creates it",
                                data={"path": str(path), "exists": False,
                                      "fix_action": "mkdir"}))
            continue
        if state == "denied":
            # Usually a container path in a host-run doctor: Spark-2's
            # models_dir is /root/.ainode/models, which the engine container
            # sees and the operator on the host does not. Say that, rather than
            # claiming the directory is missing and offering to create it.
            checks.append(Check(name, WARN, f"cannot read {path}: {reason}",
                                fix="run the doctor where that path is readable, or point "
                                    "models_dir at a path this user can see",
                                data={"path": str(path), "exists": None,
                                      "error": reason}))
            continue
        usage = disk_usage(path)
        if usage is None:
            checks.append(Check(name, WARN,
                                f"cannot read the filesystem holding {path}",
                                data={"path": str(path), "exists": True}))
            continue
        total, free = usage
        fraction = (free / total) if total else 0.0
        data = {"path": str(path), "exists": True, "total_bytes": total,
                "free_bytes": free, "free_fraction": round(fraction, 4)}
        detail = f"{_gb(free)} free of {_gb(total)} ({fraction * 100:.0f}%) on {path}"
        if fraction < DISK_WARN_FRACTION:
            checks.append(Check(name, WARN, detail,
                                fix="free space or point models_dir at a bigger filesystem; "
                                    "a launch that runs out mid-pull looks like an engine crash",
                                data=data))
        else:
            checks.append(Check(name, OK, detail, data=data))
    return checks


def check_image_pin(home, running_version: str = "", docker_ok: bool = True) -> list[Check]:
    """What the unit will start, what is running, and what has been published."""
    pinned = read_image_env(home)
    running_image = None
    container_state = None
    if docker_ok:
        code, out = run_command(
            ["docker", "inspect", "-f", "{{.State.Status}} {{.Config.Image}}", "ainode"],
            timeout=15)
        if code == 0 and out:
            bits = out.split()
            container_state = bits[0] if bits else None
            running_image = bits[1] if len(bits) > 1 else None

    data = {"pinned": pinned, "running_image": running_image,
            "container_state": container_state}
    checks: list[Check] = []
    if pinned is None:
        checks.append(Check("image.pin", WARN,
                            f"no image.env under {home}, so the unit starts whatever tag "
                            f"was baked into it at install time",
                            fix="ainode update writes image.env",
                            data=data))
    elif running_image and pinned != running_image:
        checks.append(Check("image.pin", WARN,
                            f"image.env pins {pinned} and the running container is "
                            f"{running_image}, so a restart changes the version under you",
                            fix="restart the service to take the pin, or re-run ainode update "
                                "as the install user (sudo resolves image.env to root's home)",
                            data=data))
    elif running_image:
        checks.append(Check("image.pin", OK,
                            f"{running_image} running, image.env agrees", data=data))
    else:
        checks.append(Check("image.pin", WARN,
                            f"image.env pins {pinned} and no container named ainode is running",
                            fix="systemctl start ainode",
                            data=data))

    latest = latest_image_tag()
    current = running_version or __version__
    ldata = {"current": current, "latest": latest}
    if latest is None:
        checks.append(Check("image.latest", WARN,
                            f"running {current}; could not reach ghcr.io to compare",
                            data=ldata))
    elif _version_tuple(latest) > _version_tuple(current):
        checks.append(Check("image.latest", WARN,
                            f"running {current}, {latest} is published",
                            fix="ainode update (and roll every node in the fleet, not just this one)",
                            data=ldata))
    else:
        checks.append(Check("image.latest", OK,
                            f"running {current}, the newest published tag is {latest}",
                            data=ldata))
    return checks


def check_service(in_container: Optional[bool] = None) -> list[Check]:
    """Is the systemd unit installed and running?

    From inside the AINode container there is no systemd at all, and saying
    "not installed" there would be a lie: that answer is a WARN naming the
    reason instead.
    """
    if in_container is None:
        in_container = os.environ.get("AINODE_IN_CONTAINER") == "1"
    try:
        from ainode.service.systemd import SERVICE_NAME, SYSTEM_UNIT_DIR, USER_UNIT_DIR

        unit_paths = [SYSTEM_UNIT_DIR / SERVICE_NAME, USER_UNIT_DIR / SERVICE_NAME]
    except Exception:
        SERVICE_NAME = "ainode.service"
        unit_paths = [Path("/etc/systemd/system") / SERVICE_NAME]
    present = [p for p in unit_paths if _exists(p)]

    if in_container:
        return [Check("service.unit", WARN,
                      "running inside the AINode container, where there is no systemd to ask",
                      fix=f"run `systemctl status {SERVICE_NAME}` on the host",
                      data={"in_container": True})]

    code, out = run_command(["systemctl", "is-active", SERVICE_NAME], timeout=15)
    state = (out.splitlines() or [""])[0].strip() or "unknown"
    data = {"in_container": False, "unit_paths": [str(p) for p in present],
            "state": state}
    if code == 127:
        return [Check("service.unit", WARN,
                      "no systemctl on this host, so the service cannot be checked",
                      data=data)]
    if not present and state != "active":
        return [Check("service.unit", WARN,
                      "no ainode.service unit on this host; the node was started by hand",
                      fix="ainode service install",
                      data=data)]
    if state == "active":
        return [Check("service.unit", OK, f"{SERVICE_NAME} active", data=data)]
    return [Check("service.unit", FAIL,
                  f"{SERVICE_NAME} is installed and {state}",
                  fix=f"sudo systemctl start {SERVICE_NAME}, then journalctl -u {SERVICE_NAME} -n 50",
                  data=data)]


def check_ports(config: NodeConfig, udp_bound: Optional[set[int]] = None) -> list[Check]:
    """The three ports a node lives on, each judged against what it should be."""
    web_port = int(getattr(config, "web_port", 3000) or 3000)
    api_port = int(getattr(config, "api_port", 8000) or 8000)
    disc_port = int(getattr(config, "discovery_port", FLEET_DISCOVERY_PORT)
                    or FLEET_DISCOVERY_PORT)
    model = (getattr(config, "model", "") or "").strip()
    checks: list[Check] = []

    web_up = tcp_listening(web_port)
    if web_up:
        checks.append(Check("port.web", OK, f"{web_port} listening (dashboard and /api)",
                            data={"port": web_port, "listening": True}))
    else:
        checks.append(Check("port.web", FAIL,
                            f"nothing answers on {web_port}, so the dashboard and every "
                            f"/api route are down",
                            fix="systemctl start ainode (or ainode start on a host install)",
                            data={"port": web_port, "listening": False}))

    api_up = tcp_listening(api_port)
    if api_up:
        checks.append(Check("port.engine", OK, f"{api_port} listening (an engine is serving)",
                            data={"port": api_port, "listening": True}))
    elif model:
        checks.append(Check("port.engine", WARN,
                            f"{api_port} is free and {model} is pinned, so no engine is up yet",
                            fix="launch the model from the dashboard, or watch ainode logs -f",
                            data={"port": api_port, "listening": False, "model": model}))
    else:
        checks.append(Check("port.engine", OK,
                            f"{api_port} is free and no model is pinned, which is the "
                            f"idle shape",
                            data={"port": api_port, "listening": False}))

    bound = udp_listeners() if udp_bound is None else udp_bound
    if bound is None:
        checks.append(Check("port.discovery", WARN,
                            f"cannot tell whether {disc_port}/udp is bound (no ss here)",
                            data={"port": disc_port, "listening": None}))
    elif disc_port in bound:
        checks.append(Check("port.discovery", OK, f"{disc_port}/udp bound by discovery",
                            data={"port": disc_port, "listening": True}))
    else:
        checks.append(Check("port.discovery", WARN,
                            f"{disc_port}/udp is not bound, so this node neither announces "
                            f"itself nor hears a peer",
                            fix="start the service; discovery binds at boot",
                            data={"port": disc_port, "listening": False}))
    return checks


def check_peers(config: NodeConfig, version: str = "") -> list[Check]:
    """Peers this node's own discovery has seen, and whether they match our release.

    Asked of the LOCAL API rather than by binding the discovery port, because
    the running service is the thing that listens and a second listener would
    only see what it already knows. A fleet split across two releases is the
    failure the standing "roll every node" rule exists to prevent, so a version
    disagreement is a finding, not a note.
    """
    version = version or __version__
    web_port = int(getattr(config, "web_port", 3000) or 3000)
    payload = http_json(f"http://127.0.0.1:{web_port}/api/nodes")
    if payload is None:
        return [Check("cluster.peers", WARN,
                      f"cannot enumerate peers: the local API did not answer on {web_port}",
                      fix="bring the service up first, then re-run",
                      data={"reachable": False})]

    rows = [r for r in (payload.get("nodes") or []) if isinstance(r, dict)]
    local_id = getattr(config, "node_id", None)
    peers = [r for r in rows if r.get("node_id") != local_id]
    peer_ips = list(getattr(config, "peer_ips", []) or [])

    if not peers:
        if peer_ips:
            return [Check("cluster.peers", WARN,
                          f"config names {len(peer_ips)} peer(s) and discovery sees none",
                          fix="check each peer's service and that every node uses the same "
                              "discovery port and cluster_id",
                          data={"reachable": True, "seen": 0, "configured": peer_ips})]
        return [Check("cluster.peers", OK,
                      "no peers announced; this node is alone, which is the solo shape",
                      data={"reachable": True, "seen": 0})]

    versions: dict[str, Optional[str]] = {}
    unreachable: list[str] = []
    no_fabric: list[str] = []
    for row in peers:
        label = row.get("node_name") or row.get("node_id") or "?"
        host = (row.get("fabric_ip") or "").strip()
        if not host:
            no_fabric.append(label)
            versions[label] = None
            continue
        status = http_json(f"http://{host}:{row.get('web_port') or 3000}/api/status")
        if status is None:
            unreachable.append(label)
            versions[label] = None
            continue
        versions[label] = str(status.get("version") or "") or None

    known = {v for v in versions.values() if v}
    data = {"reachable": True, "seen": len(peers), "local_version": version,
            "peer_versions": versions}
    mismatched = sorted(v for v in known if v != version)
    checks = [Check("cluster.peers", OK,
                    f"{len(peers)} peer(s) on discovery: " +
                    ", ".join(sorted(str(k) for k in versions)),
                    data=data)]
    if mismatched:
        detail = ", ".join(f"{k} on {v}" for k, v in sorted(versions.items()) if v)
        checks.append(Check("cluster.versions", WARN,
                            f"this node is on {version} and the fleet is not: {detail}",
                            fix="roll every node to the same release in one pass "
                                "(sudo ainode update on each, as the install user)",
                            data=data))
    elif known:
        checks.append(Check("cluster.versions", OK,
                            f"every peer that answered is on {version}", data=data))
    else:
        checks.append(Check("cluster.versions", WARN,
                            "no peer reported a version, so release agreement is unknown",
                            data=data))
    if no_fabric:
        checks.append(Check("cluster.fabric_ip", WARN,
                            "announcing no fabric IP: " + ", ".join(sorted(no_fabric)) +
                            "; a distributed launch would place workers on the mgmt LAN",
                            fix="set cluster_interface on those nodes",
                            data={"nodes": sorted(no_fabric)}))
    if unreachable:
        checks.append(Check("cluster.reachable", WARN,
                            "no route to " + ", ".join(sorted(unreachable)) +
                            " on the fabric address they announce",
                            fix="check the fabric link and that each node's API is up",
                            data={"nodes": sorted(unreachable)}))
    return checks


def check_fabric(config: NodeConfig) -> list[Check]:
    """The RoCE fabric: an interface configured, an address on it, a live port."""
    iface = (getattr(config, "cluster_interface", "") or "").strip()
    try:
        from ainode.cluster.hca_discovery import (
            detect_fabric_ip,
            hca_port_active,
            list_local_hcas,
        )
    except Exception as exc:
        return [Check("fabric.interface", WARN,
                      f"cannot inspect the fabric: {exc}", data={"interface": iface})]

    hcas = list_local_hcas()
    active = [h for h in hcas if hca_port_active(h)]
    data = {"interface": iface, "hcas": hcas, "hcas_active": active}

    if not iface:
        if hcas:
            return [Check("fabric.interface", WARN,
                          f"no cluster_interface configured and this node has RoCE HCAs "
                          f"({', '.join(hcas)}), so the fast fabric is unused",
                          fix='set "cluster_interface" to the fabric NIC, or leave it empty '
                              'only if the mgmt LAN is the intended path',
                          data=data)]
        return [Check("fabric.interface", OK,
                      "no fabric configured and no RoCE HCA on this node", data=data)]

    address = detect_fabric_ip(iface)
    data["address"] = address
    if not address:
        return [Check("fabric.interface", FAIL,
                      f"cluster_interface {iface} has no IPv4 address, so a distributed "
                      f"launch would rendezvous on an address nothing answers",
                      fix=f"bring {iface} up and give it an address, or clear cluster_interface",
                      data=data)]
    suffix = f", HCAs {', '.join(active)} up" if active else ""
    return [Check("fabric.interface", OK, f"{iface} at {address}{suffix}", data=data)]


def check_secrets(home) -> list[Check]:
    """The secrets store exists and nobody but its owner can read it."""
    path = Path(home) / "secrets.json"
    if not _exists(path):
        return [Check("secrets.store", OK,
                      "no secrets store yet, so there is nothing to protect",
                      data={"path": str(path), "exists": False})]
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError as exc:
        return [Check("secrets.store", WARN, f"cannot stat {path}: {exc}",
                      data={"path": str(path), "exists": True})]
    data = {"path": str(path), "exists": True, "mode": oct(mode),
            "fix_action": "chmod600"}
    if mode == 0o600:
        return [Check("secrets.store", OK, f"{path} is mode 0600", data=data)]
    return [Check("secrets.store", WARN,
                  f"{path} is mode {oct(mode)}, so it is readable beyond its owner",
                  fix="ainode doctor --fix chmods it to 0600",
                  data=data)]


def check_hf_token(config: NodeConfig, home, env: Optional[dict] = None) -> list[Check]:
    """Is a Hugging Face token configured anywhere? Presence only, never a value."""
    env = dict(os.environ if env is None else env)
    sources: list[str] = []
    if (getattr(config, "hf_token", "") or "").strip():
        sources.append("config.json")
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if (env.get(name) or "").strip():
            sources.append(f"${name}")
    try:
        from ainode.secrets import SecretsManager

        manager = SecretsManager(path=Path(home) / "secrets.json")
        if manager.has("huggingface_token"):
            sources.append("secrets store")
    except Exception:
        pass
    data = {"sources": sources}
    if sources:
        return [Check("credentials.hf_token", OK,
                      "a Hugging Face token is configured (" + ", ".join(sources) + ")",
                      data=data)]
    return [Check("credentials.hf_token", WARN,
                  "no Hugging Face token anywhere, so a gated repo (Llama, Gemma) "
                  "answers 401 at download time",
                  fix="ainode config --hf-token <token>",
                  data=data)]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def run_checks(home=None, config_path=None) -> list[Check]:
    """Every check, in the order a human reads them."""
    home = Path(home or os.environ.get("AINODE_HOME") or AINODE_HOME)
    config_path = Path(config_path) if config_path else home / "config.json"
    config, error = load_config(config_path)

    checks: list[Check] = []
    checks += check_sudo_trap()
    checks += check_config_file(config_path, error)

    docker_checks = check_docker(_engine_image_for(config))
    docker_ok = bool(docker_checks[0].data.get("reachable"))
    image_present = docker_checks[0].data.get("image_present")

    checks += check_engine_backend(config, config_path, docker_ok, image_present)
    checks += check_gpu_memory_utilization(config)
    checks += check_discovery_port(config)
    checks += check_model(config)
    checks += docker_checks
    checks += check_gpus()
    checks += check_disk(home, getattr(config, "models_dir", "") or (home / "models"))
    checks += check_image_pin(home, __version__, docker_ok)
    checks += check_service()
    checks += check_ports(config)

    peer_checks = check_peers(config)
    seen = int(peer_checks[0].data.get("seen") or 0)
    checks += check_cluster_id(config, seen)
    checks += peer_checks
    checks += check_fabric(config)
    checks += check_secrets(home)
    checks += check_hf_token(config, home)
    return checks


def summarize(checks: list[Check]) -> dict:
    counts = {OK: 0, WARN: 0, FAIL: 0}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    counts["total"] = len(checks)
    return counts


def exit_code(checks: list[Check]) -> int:
    """Non-zero when anything FAILed. A WARN is not a failed run."""
    return 1 if any(c.status == FAIL for c in checks) else 0


def report_payload(checks: list[Check], config: Optional[NodeConfig] = None) -> dict:
    return {
        "doctor": 1,
        "version": __version__,
        "node_id": getattr(config, "node_id", None) if config else None,
        "node_name": getattr(config, "node_name", None) if config else None,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summarize(checks),
        "checks": [
            {"name": c.name, "status": c.status, "detail": c.detail,
             "fix": c.fix, "data": c.data}
            for c in checks
        ],
    }


# ---------------------------------------------------------------------------
# --fix: only what cannot lose anything
# ---------------------------------------------------------------------------

def apply_fixes(checks: list[Check], config_path) -> list[str]:
    """Apply the safe fixes named by ``data["fix_action"]``; return what was done.

    Safe means reversible or additive: create a directory, tighten a file mode,
    write one config key. Anything that pulls an image, restarts a service or
    edits a systemd unit is listed for a human instead.
    """
    done: list[str] = []
    for check in checks:
        if check.status == OK:
            continue
        action = check.data.get("fix_action")
        if action == "mkdir":
            path = Path(check.data.get("path", ""))
            try:
                path.mkdir(parents=True, exist_ok=True)
                done.append(f"created {path}")
            except OSError as exc:
                done.append(f"could not create {path}: {exc}")
        elif action == "chmod600":
            path = Path(check.data.get("path", ""))
            try:
                os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
                done.append(f"chmod 0600 {path}")
            except OSError as exc:
                done.append(f"could not chmod {path}: {exc}")
        elif action == "discovery_port":
            port = int(check.data.get("expected") or FLEET_DISCOVERY_PORT)
            try:
                written = _write_config_key(config_path, "discovery_port", port)
                done.append(f"wrote discovery_port={port} to {written} "
                            f"(restart the service to pick it up)")
            except OSError as exc:
                done.append(f"could not write discovery_port: {exc}")
    return done


def _write_config_key(config_path, key: str, value) -> Path:
    """Set one key in config.json, leaving every other key exactly as it was."""
    path = Path(config_path)
    data: dict = {}
    if _exists(path):
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict):
                data = loaded
        except ValueError:
            data = {}
    data[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.doctor-tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# --peer: the same report, from a node over SSH
# ---------------------------------------------------------------------------

#: The container FIRST, with a plain ``docker exec`` and no ``-it``. The order
#: matters and was found the hard way against Spark-3: a node installed the
#: documented way does have an ``ainode`` on PATH, but it is the host wrapper the
#: installer writes, and that wrapper runs ``docker exec -it``, which over SSH
#: dies with "the input device is not a TTY" and never reaches the doctor at all.
#: The host binary is the fallback, for a dev box running from a venv.
_PEER_COMMAND = "docker exec ainode ainode doctor --json || ainode doctor --json"


def peer_checks(peer: str) -> tuple[list[Check], Optional[dict]]:
    """Run the doctor on ``peer`` over SSH and return its checks.

    A peer we cannot reach, or that answers something other than a doctor
    payload, is ONE failed check naming what was tried rather than an exception
    or a silent empty report.
    """
    code, out = run_command(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", peer, _PEER_COMMAND],
        timeout=180)
    payload = None
    for line in (out or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict) and candidate.get("doctor"):
            payload = candidate
            break
    if payload is None:
        first = (out or "").strip().splitlines()
        said = first[0][:200] if first else "no output"
        # The common answer during a rollout, and worth naming: the peer answered,
        # its build's doctor is just the old stub that printed prose and exited 0.
        if "stub (coming in" in (out or ""):
            return ([Check("peer.version", FAIL,
                           f"{peer} is running a build whose doctor is still a stub, "
                           f"so it has no report to give",
                           fix=f"roll {peer} to a release that has the real doctor",
                           data={"peer": peer, "exit": code})], None)
        return ([Check("peer.ssh", FAIL,
                       f"{peer} returned no doctor report (exit {code}): {said}",
                       fix=f"check `ssh {peer}` works without a password and that either "
                           f"ainode or a container named ainode is there",
                       data={"peer": peer, "exit": code})], None)
    checks = [
        Check(str(c.get("name", "?")), str(c.get("status", WARN)),
              str(c.get("detail", "")), str(c.get("fix", "") or ""),
              c.get("data") or {})
        for c in (payload.get("checks") or [])
        if isinstance(c, dict)
    ]
    return checks, payload


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_STYLE = {OK: "green", WARN: "yellow", FAIL: "bold red"}
_LABEL = {OK: " OK ", WARN: "WARN", FAIL: "FAIL"}


def render(checks: list[Check], console=None, header: str = "") -> None:
    """One line per check, plus a fix line wherever the answer is not OK."""
    from rich.console import Console
    from rich.markup import escape

    console = console or Console()
    width = max((len(c.name) for c in checks), default=0)
    counts = summarize(checks)
    console.print(f"[bold cyan]AINode doctor[/bold cyan] v{__version__}"
                  f"{'  ' + escape(header) if header else ''}"
                  f"  [dim italic]Made in Texas[/dim italic]")
    console.print("")
    for check in checks:
        style = _STYLE.get(check.status, "yellow")
        console.print(f"[{style}]\\[{_LABEL.get(check.status, '????')}][/{style}] "
                      f"[bold white]{escape(check.name.ljust(width))}[/bold white]  "
                      f"{escape(check.detail)}", highlight=False, soft_wrap=True)
        if check.status != OK and check.fix:
            console.print(f"       {' ' * width}  [dim]fix: {escape(check.fix)}[/dim]",
                          highlight=False, soft_wrap=True)
    console.print("")
    tone = "bold red" if counts[FAIL] else ("yellow" if counts[WARN] else "green")
    console.print(f"[{tone}]{counts['total']} checks: {counts[OK]} OK, "
                  f"{counts[WARN]} WARN, {counts[FAIL]} FAIL[/{tone}]")


def _sorted_by_severity(checks: list[Check]) -> list[Check]:
    return sorted(checks, key=lambda c: -_SEVERITY.get(c.status, 1))


def cmd_doctor(args) -> None:
    """``ainode doctor``: report, optionally fix, exit non-zero on any FAIL."""
    as_json = bool(getattr(args, "json", False))
    peer = getattr(args, "peer", None)
    wants_fix = bool(getattr(args, "fix", False))

    if peer:
        checks, payload = peer_checks(peer)
        if as_json:
            print(json.dumps(payload or report_payload(checks), indent=2))
        else:
            render(checks, header=f"peer {peer}")
        raise SystemExit(exit_code(checks))

    home = Path(os.environ.get("AINODE_HOME") or AINODE_HOME)
    config_path = home / "config.json"
    checks = run_checks(home, config_path)

    applied: list[str] = []
    if wants_fix:
        applied = apply_fixes(checks, config_path)
        if applied:
            # Re-run so the report describes the node as it is NOW, not as it
            # was before the fixes landed.
            checks = run_checks(home, config_path)

    config, _ = load_config(config_path)
    if as_json:
        payload = report_payload(checks, config)
        if wants_fix:
            payload["fixes_applied"] = applied
        print(json.dumps(payload, indent=2))
        raise SystemExit(exit_code(checks))

    from rich.console import Console

    console = Console()
    label = config.node_name or config.node_id or ""
    render(checks, console=console, header=label)
    if wants_fix:
        console.print("")
        if applied:
            console.print("[bold cyan]fixed[/bold cyan]")
            for line in applied:
                console.print(f"  {line}")
        else:
            console.print("[dim]--fix had nothing safe to apply[/dim]")
        remaining = [c for c in _sorted_by_severity(checks) if c.status != OK]
        if remaining:
            console.print("")
            console.print("[bold cyan]left for a human[/bold cyan]")
            for check in remaining:
                console.print(f"  {check.name}: {check.fix or check.detail}")
    raise SystemExit(exit_code(checks))
