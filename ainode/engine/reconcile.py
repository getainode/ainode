"""Reconcile what this node is RUNNING with what it wrote down (#179, #240).

A restart of the orchestrator does not touch the engine containers: they are
siblings spawned through docker.sock. So after a `systemctl restart ainode` or an
`ainode update` the engine can still be serving while the new process has no
record of it, and on a distributed head that divergence was total: the launch was
deliberately never written to ``instances.json``, the ``InstanceManager`` started
empty, and the node's own view of what it ran came from whatever could be
reconstructed from ``config.json`` further downstream (``announced_instances``,
``_local_parallel``, ``_will_own_primary_port`` all carry a fallback for it). Once
the engine container did stop, for any reason, nothing brought the model back.

Four pieces live here, in the order a boot uses them:

0. **The boot decision** (:func:`adopt_boot_engines`): `ainode start` asks, BEFORE
   it sweeps and before it launches anything, which of those containers it should
   KEEP. Three gates: running, the same shape the configured recipe renders
   (:func:`shape_mismatch`), and answering (:func:`engine_unhealthy`). A kept
   container is skipped by the sweep and not relaunched, which is what turns a
   restart from a full model load into seconds; anything else reloads exactly as
   it did before, with one line saying which gate said no.
   :func:`keep_engines_on_shutdown` is the other end of the same idea: the
   orchestrator no longer stops a still-serving engine on its way out, because
   that is what left the next boot with nothing to adopt (#240).
1. **Adoption** (:func:`adopt_running_engines`): before anything launches or is
   swept, ask docker which engine containers this node still has, and put the
   live ones back in the ``InstanceManager`` with the shape the container is
   actually running (model, TP width, peers, port, executor, parsed from its own
   argv). Nothing is relaunched and no peer is contacted: adoption is a read.
   An adopted record carries ``adopted=True`` so every consumer can tell a
   reconstructed instance from one this process launched.
2. **The record** (:func:`write_distributed_record` and friends): the
   distributed shape, persisted to ``<AINODE_HOME>/distributed.json`` when a
   launch succeeds and removed when it is unloaded, so a restart knows what this
   node was serving EVEN WHEN THE CONTAINER IS GONE. It is a separate file from
   ``instances.json`` on purpose: the solo manifest is replayed entry by entry
   through ``append_solo_instance``, and a distributed shape in that list would
   be relaunched as a single-node load.
3. **The replay policy** (:func:`replay_distributed_if_needed`): record present
   and container gone is the only case that may relaunch, and only when every
   peer answers the probe the launch itself depends on (ssh + docker). Otherwise
   the record is marked ``degraded`` with the peer and its answer, and that shows
   up in ``ainode doctor``, in ``/api/status`` and as a dashboard banner. ONE
   attempt per process: never a loop, because a launch that needs a human is not
   improved by trying it again every ten seconds.

Everything that touches the world goes through a module-level seam
(:func:`inspect_container`, :func:`list_engine_containers`, :func:`probe_peer`,
:func:`port_serving`, :func:`port_health`, :func:`served_models`) so
``tests/test_distributed_replay.py`` and ``tests/test_adopt_primary.py`` drive
every branch with fakes and no docker, no ssh, no HTTP and no sleeps.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

#: Container-name prefixes this node owns, and what shape each one means. Kept in
#: step with ``models/api_routes.py``'s sweep filters: the two lists describe the
#: same containers from opposite ends (that module removes them, this one adopts
#: them), so a new engine name has to be added in both places.
PRIMARY_ENGINE_NAME = "ainode-vllm-node-solo"
STACKED_ENGINE_PREFIX = "ainode-vllm-node-solo-"
HEAD_ENGINE_PREFIX = "ainode-vllm-head"

#: The distributed shape, one file under AINODE_HOME.
RECORD_FILENAME = "distributed.json"

#: Record states. ``serving`` means the shape is up (or coming up) as written;
#: ``degraded`` means this node knows what it should be serving and cannot,
#: and says why.
SERVING = "serving"
DEGRADED = "degraded"

#: Names of containers adoption has claimed this process. The startup sweep reads
#: it so it cannot remove an engine we just decided to keep (the sweep is
#: name-prefixed and would otherwise take an adopted stacked instance with it).
_ADOPTED_CONTAINER_IDS: set = set()

#: One replay attempt per process (see the module docstring).
_REPLAY_ATTEMPTED = False

#: What the BOOT decided about the engine containers that were already running
#: when this process started: ``{"primary": entry|None, "stacked": [entry, ...],
#: "lines": [...]}``. Filled by :func:`adopt_boot_engines`, which `ainode start`
#: calls before the sweep, and read afterwards by ``create_app`` (so the primary's
#: seeded record carries ``adopted``) and by the startup replay (so it does not
#: wait on, or relaunch, an engine that is already serving). Empty on a boot that
#: never asked, which is every test that just builds an app.
_BOOT_DECISION: dict = {}


def reset_state_for_tests() -> None:
    """Forget the per-process adoption/replay state. Test seam only."""
    global _REPLAY_ATTEMPTED
    _ADOPTED_CONTAINER_IDS.clear()
    _BOOT_DECISION.clear()
    _REPLAY_ATTEMPTED = False


def adopted_container_ids() -> set:
    """Container ids adoption has claimed, so the startup sweep can skip them."""
    return set(_ADOPTED_CONTAINER_IDS)


# ---------------------------------------------------------------------------
# Seams. Everything that reads the world lives here.
# ---------------------------------------------------------------------------

def inspect_container(name: str) -> Optional[dict]:
    """``docker inspect <name>`` as a dict, or None when there is no such thing.

    Never raises: a missing docker binary, an unreachable daemon and an absent
    container are all "nothing to adopt", which is what every caller wants.
    """
    try:
        out = subprocess.check_output(
            ["docker", "inspect", name], text=True, timeout=20,
            stderr=subprocess.DEVNULL)
    except Exception:
        return None
    try:
        parsed = json.loads(out)
    except ValueError:
        return None
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        return parsed[0]
    return None


def list_engine_containers() -> list:
    """Names of every engine container this node owns that docker still knows.

    Includes exited ones: the caller decides what a non-running container means
    (adoption skips it; the sweep removes it).
    """
    names: list = []
    for prefix in (PRIMARY_ENGINE_NAME, HEAD_ENGINE_PREFIX):
        try:
            out = subprocess.check_output(
                ["docker", "ps", "-a", "--format", "{{.Names}}",
                 "--filter", f"name={prefix}"],
                text=True, timeout=20, stderr=subprocess.DEVNULL)
        except Exception:
            continue
        for line in out.splitlines():
            line = line.strip()
            if line and line not in names:
                names.append(line)
    return names


def probe_peer(peer_ip: str, ssh_user: str, timeout: float = 20.0) -> tuple:
    """Can this node still launch a rank on ``peer_ip``? ``(ok, answer)``.

    The same plumbing the launch depends on, asked as a question instead of a
    launch: the ssh the head uses to place a peer container
    (``BatchMode=yes``, ``StrictHostKeyChecking=no``, ``ConnectTimeout=10``),
    running ``docker version`` on the far end. A peer that refuses the key, has
    no route, or answers ssh with no usable docker all come back False with what
    it actually said, because "which peer, and what did it answer" is the whole
    value of the degraded report.
    """
    argv = [
        "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10", f"{ssh_user}@{peer_ip}",
        "docker version --format '{{.Server.Version}}'",
    ]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        return False, str(exc)
    answer = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        return False, answer or f"ssh exited {proc.returncode}"
    return True, answer or "docker answered"


def port_serving(port: int, timeout: float = 3.0) -> bool:
    """True when ``localhost:<port>/v1/models`` answers 200 with a model.

    Blocking on purpose (adoption offloads it): the same question
    ``/api/status`` asks, asked without an aiohttp session, because adoption runs
    before the app is fully up.
    """
    import urllib.request
    try:
        with urllib.request.urlopen(
                f"http://localhost:{port}/v1/models", timeout=timeout) as resp:
            if resp.status != 200:
                return False
            data = json.loads(resp.read().decode())
    except Exception:
        return False
    return bool(data.get("data"))


def port_health(port: int, timeout: float = 3.0) -> bool:
    """True when ``localhost:<port>/health`` answers 200.

    vLLM's own readiness path, and the cheapest question there is: it answers
    only once the engine has finished loading and the HTTP server is up. Kept
    separate from :func:`port_serving` because the adoption gate asks both, and
    an operator reading a refusal has to be told WHICH one said no.
    """
    import urllib.request
    try:
        with urllib.request.urlopen(
                f"http://localhost:{port}/health", timeout=timeout) as resp:
            return int(getattr(resp, "status", 0) or 0) == 200
    except Exception:
        return False


def served_models(port: int, timeout: float = 3.0) -> list:
    """The model ids ``localhost:<port>/v1/models`` reports, or ``[]``.

    The engine's own account of what it is serving, which is the claim adoption
    has to check: a container can be up and answering while serving a model this
    node's config no longer names (an operator changed it, a release moved a
    catalog recipe), and adopting that would make the node advertise a model
    nothing is serving.
    """
    import urllib.request
    try:
        with urllib.request.urlopen(
                f"http://localhost:{port}/v1/models", timeout=timeout) as resp:
            if int(getattr(resp, "status", 0) or 0) != 200:
                return []
            data = json.loads(resp.read().decode())
    except Exception:
        return []
    return [str(entry.get("id") or "") for entry in (data.get("data") or [])
            if isinstance(entry, dict) and entry.get("id")]


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

def record_path() -> Path:
    from ainode.core.config import AINODE_HOME
    return Path(AINODE_HOME) / RECORD_FILENAME


def load_distributed_record() -> Optional[dict]:
    """The persisted distributed shape, or None. A junk file reads as None."""
    try:
        path = record_path()
        if not path.exists():
            return None
        data = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict) or not data.get("model"):
        return None
    return data


def save_distributed_record(record: dict) -> None:
    """Write the distributed shape. Never raises: bookkeeping cannot fail a launch."""
    try:
        path = record_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, sort_keys=True))
    except Exception:
        logger.exception("could not write the distributed record")


def clear_distributed_record() -> None:
    """Remove the record: this node is no longer meant to serve that shape."""
    try:
        path = record_path()
        if path.exists():
            path.unlink()
            logger.info("distributed record cleared")
    except Exception:
        logger.exception("could not clear the distributed record")


def write_distributed_record(config, *, model: str, api_port: int, peer_ips: list,
                             tensor_parallel_size: int, distributed_executor: str,
                             instance_id: str, container: str,
                             overrides: Optional[dict] = None) -> dict:
    """Persist the shape a distributed launch just brought up, and return it."""
    record = {
        "model": model,
        "api_port": int(api_port),
        "peer_ips": [str(p) for p in (peer_ips or [])],
        "tensor_parallel_size": int(tensor_parallel_size or 1),
        "distributed_executor": (distributed_executor or "ray"),
        "head_node_id": getattr(config, "node_id", "") or "head",
        "instance_id": instance_id,
        "container": container,
        "ssh_user": getattr(config, "ssh_user", "") or "",
        "overrides": dict(overrides or {}),
        "written_at": _stamp(),
        "status": SERVING,
        "degraded_reason": "",
        "degraded_peers": [],
    }
    save_distributed_record(record)
    logger.info("distributed record written: %s TP=%d on :%d across %d peer(s) (%s)",
                model, record["tensor_parallel_size"], record["api_port"],
                len(record["peer_ips"]), record["distributed_executor"])
    return record


def mark_record_degraded(record: dict, reason: str, peers: Optional[list] = None) -> dict:
    """Stamp the record ``degraded`` with the reason, and persist it."""
    updated = dict(record)
    updated["status"] = DEGRADED
    updated["degraded_reason"] = reason
    updated["degraded_peers"] = list(peers or [])
    updated["degraded_at"] = _stamp()
    save_distributed_record(updated)
    logger.warning("distributed shape DEGRADED: %s", reason)
    return updated


def mark_record_serving(record: dict) -> dict:
    """Clear a degraded stamp: the shape is up again. No write when unchanged."""
    if record.get("status") == SERVING and not record.get("degraded_reason"):
        return record
    updated = dict(record)
    updated["status"] = SERVING
    updated["degraded_reason"] = ""
    updated["degraded_peers"] = []
    updated.pop("degraded_at", None)
    save_distributed_record(updated)
    return updated


def degraded_instances(record: Optional[dict] = None) -> list:
    """``/api/status``'s ``degraded_instances``: what this node cannot serve.

    A list because the shape of the answer must not change when a node can hold
    more than one distributed instance; today there is at most one record.
    """
    rec = record if record is not None else load_distributed_record()
    if not rec or rec.get("status") != DEGRADED:
        return []
    return [{
        "model": rec.get("model", ""),
        "api_port": rec.get("api_port"),
        "peer_ips": list(rec.get("peer_ips") or []),
        "tensor_parallel_size": rec.get("tensor_parallel_size", 1),
        "distributed_executor": rec.get("distributed_executor", "ray"),
        "reason": rec.get("degraded_reason", ""),
        "peers_unreachable": list(rec.get("degraded_peers") or []),
        "since": rec.get("degraded_at") or rec.get("written_at") or "",
    }]


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# Reading a container's own launch shape
# ---------------------------------------------------------------------------

#: vLLM serve flags worth recovering from a running container. The container's
#: argv is the only first-hand account of what an engine is actually serving: a
#: config file can have been edited since the launch, and for a stacked instance
#: whose manifest write never happened it is the ONLY account.
_FLAGS_WITH_VALUES = (
    "--tensor-parallel-size", "--port", "--gpu-memory-utilization",
    "--max-model-len", "--kv-cache-dtype", "--served-model-name",
    "--nnodes", "--node-rank", "--master-addr", "--master-port",
    "--distributed-executor-backend",
)


def container_argv(info: dict) -> list:
    """The full argv of a container, entrypoint included."""
    config = (info or {}).get("Config") or {}
    argv: list = []
    for key in ("Entrypoint", "Cmd"):
        part = config.get(key)
        if isinstance(part, list):
            argv.extend(str(a) for a in part)
        elif isinstance(part, str) and part.strip():
            argv.extend(shlex.split(part))
    return argv


def parse_engine_argv(argv: list) -> dict:
    """What a ``vllm serve`` argv says about the instance it is serving.

    Returns only keys the argv actually states: ``target`` (the serve target,
    the first non-flag token after ``serve``), ``served_model_name`` and the
    numeric/str flags in :data:`_FLAGS_WITH_VALUES`, plus ``headless``. A flag
    we do not recognise is skipped rather than guessed at.
    """
    out: dict = {}
    if not argv:
        return out
    serve_at = None
    for index, token in enumerate(argv):
        if token == "serve":
            serve_at = index
            break
    index = (serve_at + 1) if serve_at is not None else 0
    if serve_at is not None and index < len(argv) and not argv[index].startswith("-"):
        out["target"] = argv[index]
        index += 1
    while index < len(argv):
        token = argv[index]
        if token == "--headless":
            out["headless"] = True
            index += 1
            continue
        if token in _FLAGS_WITH_VALUES and index + 1 < len(argv):
            out[token.lstrip("-").replace("-", "_")] = argv[index + 1]
            index += 2
            continue
        index += 1
    return out


def _as_int(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def container_shape(info: dict) -> dict:
    """The instance shape a running engine container describes, from its argv.

    ``model`` prefers ``--served-model-name`` (the id clients and the router use)
    over the serve target, which on a model downloaded through AINode is a mount
    path rather than a repo id.
    """
    parsed = parse_engine_argv(container_argv(info))
    state = (info or {}).get("State") or {}
    shape = {
        "running": bool(state.get("Running")),
        "state": str(state.get("Status") or ""),
        "container_id": str((info or {}).get("Id") or ""),
        "image": str(((info or {}).get("Config") or {}).get("Image") or ""),
        "started_at": _epoch(state.get("StartedAt")),
        "model": parsed.get("served_model_name") or parsed.get("target") or "",
        "api_port": _as_int(parsed.get("port"), 0),
        "tensor_parallel_size": _as_int(parsed.get("tensor_parallel_size"), 0),
        "nnodes": _as_int(parsed.get("nnodes"), 0),
        "headless": bool(parsed.get("headless")),
        "gpu_memory_utilization": _as_float(parsed.get("gpu_memory_utilization")),
        "max_model_len": _as_int(parsed.get("max_model_len"), 0) or None,
        "kv_cache_dtype": parsed.get("kv_cache_dtype") or "",
    }
    backend = (parsed.get("distributed_executor_backend") or "").strip().lower()
    if backend:
        shape["distributed_executor"] = "mp" if backend == "mp" else backend
    elif shape["nnodes"] > 1:
        # The mp shape is the one that carries rendezvous flags; the Ray shape
        # runs vllm through a docker exec and never states --nnodes.
        shape["distributed_executor"] = "mp"
    return shape


def _epoch(stamp) -> Optional[float]:
    """Docker's RFC3339 ``StartedAt`` as epoch seconds, or None."""
    if not isinstance(stamp, str) or not stamp.strip():
        return None
    text = stamp.strip()
    if text.startswith("0001-01-01"):
        return None  # docker's "never started"
    text = text.replace("Z", "+00:00")
    # Docker prints nanoseconds; datetime takes at most microseconds.
    if "." in text:
        head, _, tail = text.partition(".")
        digits = "".join(ch for ch in tail if ch.isdigit())[:6]
        offset = tail[len(digits):] if len(tail) > len(digits) else ""
        offset = offset.lstrip("0123456789")
        text = f"{head}.{digits}{offset}" if digits else head + offset
    from datetime import datetime
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# What this node is SUPPOSED to be running: the configured recipe
# ---------------------------------------------------------------------------
#
# Adoption is only safe when the container that is up is the container a launch
# would have created. So the configured recipe is rendered to the handful of
# fields a running container can be compared on -- model, port, image, width,
# shape -- WITHOUT launching anything, and each one is read from the same place
# the backend reads it (``resolve_engine_image``, ``extra_vllm_args``,
# ``served_model_name``) so the two cannot drift.

def primary_container_name(config) -> str:
    """The container name the configured PRIMARY engine would have created."""
    dmode = (getattr(config, "distributed_mode", "solo") or "solo").lower()
    if dmode == "head":
        return head_container_name(config, _as_int(getattr(config, "api_port", 0), 0))
    return PRIMARY_ENGINE_NAME


def stacked_container_name(api_port: int) -> str:
    """The container name a stacked instance on ``api_port`` would have created."""
    return f"{STACKED_ENGINE_PREFIX}{int(api_port)}"


def _flag_value(argv: list, flag: str) -> str:
    """The value a flag list states for ``flag``, or ``""``."""
    tokens = [str(token) for token in (argv or [])]
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            return tokens[index + 1]
    return ""


def expected_width(config) -> int:
    """The ``--tensor-parallel-size`` a launch on this config would emit.

    A SOLO launch renders the flag only from ``extra_vllm_args``
    (``nvidia.py::_build_solo_docker_cmd`` passes ``tp_size=1``), which is how
    castor's four-V100 recipe states its width, so that list is read first. A head
    gets one rank per node.
    """
    stated = _flag_value(list(getattr(config, "extra_vllm_args", None) or []),
                         "--tensor-parallel-size")
    if stated:
        return _as_int(stated, 1) or 1
    dmode = (getattr(config, "distributed_mode", "solo") or "solo").lower()
    if dmode == "head":
        return 1 + len(list(getattr(config, "peer_ips", None) or []))
    return 1


def expected_served_model(config) -> str:
    """The id ``/v1/models`` would report for a launch on this config.

    Mirrors ``nvidia.py::_served_model_name_args``: a recipe that states the flag
    itself wins, then a ``served_model_name`` alias, then the model id. It is what
    a running container's argv states and what its ``/v1/models`` answers, so it
    is the one value both halves of the gate compare against.
    """
    stated = _flag_value(list(getattr(config, "extra_vllm_args", None) or []),
                         "--served-model-name")
    if stated:
        return stated
    names = [str(n) for n in (getattr(config, "served_model_name", None) or [])
             if str(n).strip()]
    return names[0] if names else (getattr(config, "model", "") or "")


def expected_shape(config, *, container: str = "") -> dict:
    """The shape a launch on ``config`` would produce, to compare a container to.

    A falsy field means "no expectation" and is not compared: an orphan stacked
    container has no on-disk record to compare against, and the container's own
    argv is then the only account there is (#179).
    """
    from ainode.engine.backends.nvidia import resolve_engine_image
    dmode = (getattr(config, "distributed_mode", "solo") or "solo").lower()
    return {
        "container": container or primary_container_name(config),
        "model": expected_served_model(config),
        "api_port": _as_int(getattr(config, "api_port", 0), 0),
        "image": resolve_engine_image(config),
        "tensor_parallel_size": expected_width(config),
        # Only a distributed launch renders one, so a solo container carrying no
        # executor flag is not a mismatch.
        "distributed_executor": ((getattr(config, "distributed_executor", "") or "")
                                 if dmode == "head" else ""),
    }


def shape_mismatch(expected: dict, shape: dict) -> str:
    """Why a running container is NOT the one this config describes, or ``""``.

    One line naming the field and both values, because the operator reading the
    boot log has to be able to tell "I changed the model" from "this release moved
    the engine image", and those are the two reasons a restart still has to
    reload (#240).
    """
    if not shape.get("model"):
        return "names no served model in its own argv, so there is nothing to match"
    if expected.get("model") and shape["model"] != expected["model"]:
        return f"is serving {shape['model']}, config says {expected['model']}"
    if (expected.get("api_port") and shape.get("api_port")
            and int(shape["api_port"]) != int(expected["api_port"])):
        return f"is on :{shape['api_port']}, config says :{expected['api_port']}"
    if (expected.get("image") and shape.get("image")
            and shape["image"] != expected["image"]):
        return f"runs {shape['image']}, config says {expected['image']}"
    want_width = _as_int(expected.get("tensor_parallel_size"), 0)
    got_width = _as_int(shape.get("tensor_parallel_size"), 0) or 1
    if want_width and want_width != got_width:
        return f"is TP={got_width}, config says TP={want_width}"
    want_exec = str(expected.get("distributed_executor") or "").lower()
    got_exec = str(shape.get("distributed_executor") or "").lower()
    if want_exec and got_exec and want_exec != got_exec:
        return f"is the {got_exec} shape, config says {want_exec}"
    return ""


def engine_unhealthy(port: int, model: str) -> str:
    """Why the engine on ``port`` cannot be trusted to be serving, or ``""``.

    Both halves are asked, in this order, because they fail for different
    reasons: ``/health`` is silent while an engine is still loading (vLLM binds
    its HTTP server last), and ``/v1/models`` is what says WHICH model a bound
    engine is serving.
    """
    if not port_health(port):
        return f"/health on :{port} did not answer"
    ids = served_models(port)
    if not ids:
        return f"/v1/models on :{port} named no model"
    if model and model not in ids:
        return f"/v1/models on :{port} names {', '.join(ids)}, not {model}"
    return ""


def uptime_phrase(started_at, now=None) -> str:
    """How long a container has been up, for the operator's one log line."""
    if not started_at:
        return "unknown"
    try:
        seconds = float(now if now is not None else time.time()) - float(started_at)
    except (TypeError, ValueError):
        return "unknown"
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.0f}m"
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h {rest // 60:02d}m"


# ---------------------------------------------------------------------------
# The boot decision: keep what is serving, relaunch the rest
# ---------------------------------------------------------------------------

def boot_decision() -> dict:
    """What this boot decided about the containers that were already running."""
    return dict(_BOOT_DECISION)


def adopted_boot_primary() -> Optional[dict]:
    """The primary this boot adopted, or None when it is launching its own."""
    return _BOOT_DECISION.get("primary") or None


def adopt_boot_engines(config) -> dict:
    """Decide, BEFORE the boot sweep, which running engine containers to keep.

    This is the half of #240 that has to run in ``ainode start``, because both
    things that destroy a still-serving engine happen there: the pre-launch sweep
    (``docker rm -f`` every engine container this node owns) and the boot
    primary's own launch. A restart of a solo node therefore paid a full model
    load -- 420 s on pollux, 834 s on castor -- to arrive back at the state it
    was already in.

    So each container this node would have created is inspected first, and kept
    when all three questions answer yes:

    * it is RUNNING (an exited one is a corpse and belongs to the sweep),
    * its argv matches the configured recipe (:func:`shape_mismatch`): same
      model, port, image, width and shape, so an update that moved the engine
      image still reloads,
    * and it is ANSWERING (:func:`engine_unhealthy`): ``/health`` plus
      ``/v1/models`` naming that model. A container that is up but not yet bound
      is relaunched exactly as today, so nothing is lost when adoption declines.

    Returns ``{"primary": entry|None, "stacked": [...], "lines": [...]}`` and
    records the same in :data:`_BOOT_DECISION`. ``lines`` is what the caller
    prints: one line per instance, either ``adopted <model> on :<port>`` or
    ``relaunching <model>: <reason>``, because the node's own logger is not
    configured in a service start and an operator reads the console.
    """
    from ainode.core.config import DEFAULT_ENGINE_BACKEND

    decision: dict = {"primary": None, "stacked": [], "lines": []}
    _BOOT_DECISION.clear()
    _BOOT_DECISION.update(decision)
    if config is None:
        return decision
    backend_name = (getattr(config, "engine_backend", "") or DEFAULT_ENGINE_BACKEND).lower()
    if backend_name != "nvidia":
        # No other backend runs the engine in a container of ours, so there is
        # nothing a restart could have left behind to adopt.
        return decision
    dmode = (getattr(config, "distributed_mode", "solo") or "solo").lower()
    if dmode == "member":
        # A member's containers were placed by whichever head SSH'd in.
        return decision
    if getattr(config, "_skip_replay", False):
        # start-clean: the operator asked for an idle node, so every engine
        # container goes to the sweep (the whole point of the knob).
        logger.info("start-clean: adopting nothing, the sweep frees this node")
        return decision

    primary = _consider_primary(config, decision["lines"])
    if primary is not None:
        decision["primary"] = primary
    decision["stacked"] = _consider_stacked(config, decision["lines"])
    _BOOT_DECISION.clear()
    _BOOT_DECISION.update(decision)
    return decision


def _keep(name: str, expected: dict, *, kind: str, lines: list,
          info: Optional[dict] = None) -> Optional[dict]:
    """Keep this container, or say in one line why it is being relaunched.

    The three gates in order, cheapest first. Keeping means two things: the id
    goes in :data:`_ADOPTED_CONTAINER_IDS` so the sweep cannot remove it, and the
    entry is handed back so the caller can skip the launch.
    """
    model = expected.get("model") or ""
    if info is None:
        info = inspect_container(name)
    if info is None:
        lines.append(f"relaunching {model}: no container {name} is running here")
        return None
    shape = container_shape(info)
    if not shape.get("running"):
        lines.append(f"relaunching {model}: container {name} is "
                     f"{shape.get('state') or 'gone'}")
        return None
    reason = shape_mismatch(expected, shape)
    if reason:
        lines.append(f"relaunching {model}: container {name} {reason}")
        return None
    port = _as_int(shape.get("api_port"), 0) or _as_int(expected.get("api_port"), 0)
    reason = engine_unhealthy(port, model or shape["model"])
    if reason:
        lines.append(f"relaunching {model}: container {name} is up but {reason}")
        return None
    entry = {
        "kind": kind, "container": name, "model": shape["model"], "api_port": port,
        "image": shape["image"], "container_id": shape["container_id"],
        "tensor_parallel_size": _as_int(shape.get("tensor_parallel_size"), 0) or 1,
        "started_at": shape["started_at"],
        "uptime": uptime_phrase(shape["started_at"]),
    }
    if shape["container_id"]:
        _ADOPTED_CONTAINER_IDS.add(shape["container_id"])
    lines.append(f"adopted {entry['model']} on :{port} (container {name}, "
                 f"up {entry['uptime']})")
    return entry


def _consider_primary(config, lines: list) -> Optional[dict]:
    """The boot primary: ``config.model`` on the node's own port."""
    if not (getattr(config, "model", "") or ""):
        return None  # nothing configured, so nothing to adopt or relaunch
    expected = expected_shape(config)
    entry = _keep(expected["container"], expected, kind="primary", lines=lines)
    if entry is not None:
        entry["distributed_mode"] = (getattr(config, "distributed_mode", "solo")
                                     or "solo").lower()
    return entry


def _consider_stacked(config, lines: list) -> list:
    """Every stacked instance the manifest names that is still serving.

    The manifest entry, not the container's argv, is the expectation: it is what
    a replay would have launched (its own engine image, context length and extra
    flags), so comparing the container against it is what tells a restart that can
    keep the engine from one that has to reload it.
    """
    from dataclasses import replace

    from ainode.models.api_routes import (
        _OVERRIDE_KEYS,
        _resolved_overrides,
        load_instance_manifest,
    )

    node_port = _as_int(getattr(config, "api_port", 8000), 8000)
    entries = {str(entry["model"]): entry for entry in load_instance_manifest()
               if isinstance(entry, dict) and entry.get("model")}
    kept: list = []
    for name in list_engine_containers():
        if not name.startswith(STACKED_ENGINE_PREFIX):
            continue
        port = _as_int(name[len(STACKED_ENGINE_PREFIX):], 0)
        if not port or port == node_port:
            continue
        info = inspect_container(name)
        if info is None:
            continue
        shape = container_shape(info)
        manifest_entry = entries.get(shape.get("model") or "")
        if manifest_entry is None:
            # Nothing on disk describes it. Adoption proper repairs that case
            # (#179) by writing it back into the manifest; here it is simply not
            # a container the boot has a recipe to compare, so leave the decision
            # to the sweep and the replay, as before.
            continue
        overrides = {k: manifest_entry[k] for k in _OVERRIDE_KEYS
                     if k in manifest_entry}
        inst_config = replace(
            config, model=shape["model"], distributed_mode="solo", peer_ips=[],
            api_port=port,
            **_resolved_overrides(manifest_entry.get("gpu_memory_utilization"),
                                  overrides))
        entry = _keep(name, expected_shape(inst_config, container=name),
                      kind="stacked", lines=lines, info=info)
        if entry is not None:
            kept.append(entry)
    return kept


def attach_adopted_backend(engine, entry: dict) -> None:
    """Point a backend handle at the container this boot adopted.

    The handle itself is already right: the primary's ``instance_id`` token is
    empty, so it names the unsuffixed container the launch would have created, and
    ``stop()`` and ``is_running()`` both reach it. What it lacks is a launch stamp,
    because this process never launched anything, so it is taken from the
    container's own ``StartedAt``: that is what makes ``is_running()`` ask docker
    rather than answer from a subprocess that does not exist, and it is the clock
    a bind wait and the UI report.
    """
    if engine is None:
        return
    try:
        engine._launched_at = entry.get("started_at") or time.time()
    except Exception:  # pragma: no cover - a backend without the attribute
        logger.debug("adopted backend takes no launch stamp")


def keep_engines_on_shutdown(engine, config) -> str:
    """Leave a still-running engine container alone when the orchestrator exits.

    The other half of #240, and the one the issue's own diagnosis missed: the
    sweep is not the only thing that frees the engine, because ``ainode start``
    also calls ``engine.stop()`` on its way out, which stops AND removes the
    container (and a head's peer containers with it). Measured on pollux: a
    ``systemctl restart ainode`` left no engine container for the next boot to
    find, so the pre-launch sweep freed nothing and the boot engine reloaded the
    model from scratch, every time.

    Engine containers are siblings spawned through docker.sock and every STACKED
    one already outlives the orchestrator; this makes the primary and the head
    behave the same way. A container that is NOT running is still stopped through
    the backend, which is what reaps a corpse and a head's peers.

    Returns the operator line when the engine was left up, else ``""``.
    """
    if engine is None:
        return ""
    name = primary_container_name(config)
    info = inspect_container(name)
    shape = container_shape(info) if info is not None else {}
    if not shape.get("running"):
        try:
            engine.stop()
        except Exception:
            logger.exception("stopping the engine on shutdown failed")
        return ""
    model = shape.get("model") or (getattr(config, "model", "") or "")
    port = _as_int(shape.get("api_port"), 0) or _as_int(getattr(config, "api_port", 0), 0)
    line = (f"engine left serving {model} on :{port} (container {name}); the next "
            f"start adopts it. To free the GPU: docker rm -f {name}")
    logger.info("%s", line)
    return line


# ---------------------------------------------------------------------------
# Adoption
# ---------------------------------------------------------------------------

def head_container_name(config, api_port: int) -> str:
    """The container name a distributed launch on ``api_port`` would have created.

    Mirrors ``NvidiaBackend._head_container_name``: the primary keeps the legacy
    bare name and a co-resident instance appends its port, which is the same
    ``instance_id`` token the launch passes to the backend.
    """
    node_port = _as_int(getattr(config, "api_port", 0), 0)
    if not api_port or api_port == node_port:
        return HEAD_ENGINE_PREFIX
    return f"{HEAD_ENGINE_PREFIX}-{api_port}"


def _instance_config(config, *, model: str, api_port: int, peer_ips: list,
                     distributed_mode: str, overrides: Optional[dict] = None):
    """A per-instance config snapshot for an adopted container's backend.

    Never the shared ``app["config"]``: a backend built on that object would
    cross-wire this instance with the node's own primary settings, which is the
    same rule every launch path follows.
    """
    from dataclasses import replace
    fields = {"model": model, "api_port": int(api_port),
              "distributed_mode": distributed_mode,
              "peer_ips": [str(p) for p in (peer_ips or [])]}
    for key, value in (overrides or {}).items():
        if hasattr(config, key) and value is not None:
            fields[key] = value
    return replace(config, **fields)


def _build_backend(inst_config, *, api_port: int, node_port: int, started_at):
    """A backend handle for a container that is ALREADY running.

    The instance_id token is what names the container, so it has to match the one
    the launch used or ``stop()`` would reach for a container that does not
    exist. ``_launched_at`` is stamped from the container's own start time: it is
    what makes ``is_running()`` ask docker instead of answering from a launch
    subprocess this process never had, and it is the clock a bind wait reports.
    """
    from ainode.engine.backends import get_backend
    name_token = "" if api_port == node_port else str(api_port)
    backend = get_backend(inst_config, instance_id=name_token)
    try:
        backend._launched_at = started_at if started_at else time.time()
    except Exception:  # pragma: no cover - a backend without the attribute
        pass
    return backend


async def adopt_running_engines(app) -> list:
    """Put every engine container this node is still running back in the manager.

    Runs BEFORE the startup sweep and before anything launches, and relaunches
    nothing: this is the half of #179 that needs no peer coordination. Returns
    the records adopted, newest state first for the log.

    Three kinds get adopted:

    * **the distributed head**, from the record if there is one, else from
      ``config.json``'s head shape (the mp head is a config state today: the
      shape is hand-written into config.json and the boot engine replays it, so
      there is no launch call to have written a record on this node's first boot
      under this code).
    * **the solo primary**, when the boot kept the container instead of
      relaunching it (:func:`adopt_boot_engines`, #240). ``create_app`` seeds that
      instance from the boot engine handle, so normally this is what flips the
      seeded record to ``adopted`` and ``serving``; with no handle to seed from it
      builds the whole record. A boot that never asked (no ``ainode start``, so no
      decision) adopts no primary and behaves as it did before #240.
    * **a stacked instance**, either one the boot kept or an orphan on a stacked
      port that no manager entry and no ``instances.json`` entry knows about,
      which is what a crash between a launch and the manifest write leaves
      behind. An orphan is written back into the manifest as it is adopted, so
      the next restart replays it.

    A container that is not running is left alone for the sweep to remove.
    """
    import asyncio

    config = app.get("config")
    if config is None:
        return []
    loop = asyncio.get_event_loop()
    adopted: list = []

    head = await loop.run_in_executor(None, lambda: _adopt_distributed_head(app, config))
    if head is not None:
        adopted.append(head)
    primary = await loop.run_in_executor(None, lambda: _adopt_solo_primary(app, config))
    if primary is not None:
        adopted.append(primary)
    stacked = await loop.run_in_executor(None, lambda: _adopt_stacked(app, config))
    adopted.extend(stacked)
    return adopted


def _manager(app, config):
    from ainode.engine.instance_manager import InstanceManager
    manager = app.get("instances")
    if manager is None:
        manager = InstanceManager(base_port=getattr(config, "api_port", 8000))
        app["instances"] = manager
    return manager


def _add_adopted(app, config, manager, *, model: str, api_port: int, peer_ips: list,
                 tensor_parallel_size: int, distributed_executor: str,
                 instance_id: str, backend, serving: bool):
    from ainode.discovery.instance import InstanceRecord
    record = InstanceRecord(
        instance_id=instance_id, model=model,
        head_node_id=getattr(config, "node_id", "") or "head",
        peer_ips=[str(p) for p in (peer_ips or [])], api_port=int(api_port),
        tensor_parallel_size=int(tensor_parallel_size or 1),
        status=("serving" if serving else "starting"),
        distributed_executor=(distributed_executor or "ray"),
        adopted=True)
    manager.add(record, backend)
    if app.get("engine") is None and int(api_port) == _as_int(
            getattr(config, "api_port", 0), 0):
        # Nothing else claims the node's own port, so the back-compat
        # status/proxy path (app["engine"]) should point at the engine that
        # actually holds it rather than at nothing.
        app["engine"] = backend
    return record


def _adopt_distributed_head(app, config) -> Optional[dict]:
    """Adopt the head engine container, if this node is still running one."""
    record = load_distributed_record()
    dmode = (getattr(config, "distributed_mode", "solo") or "solo").lower()
    if dmode == "member":
        # A member runs no engine of its own: its containers were placed by
        # whichever head SSH'd in, and belong to that head's instance.
        return None
    if record is None and dmode != "head":
        return None

    api_port = _as_int((record or {}).get("api_port"),
                       _as_int(getattr(config, "api_port", 8000), 8000))
    name = (record or {}).get("container") or head_container_name(config, api_port)
    info = inspect_container(name)
    if info is None:
        logger.info("no engine container %s to adopt on this node", name)
        return None
    shape = container_shape(info)
    if not shape["running"]:
        logger.info("engine container %s is %s, not adopting it", name,
                    shape["state"] or "gone")
        return None

    manager = _manager(app, config)
    port = shape["api_port"] or api_port
    if manager.by_port(port) is not None:
        return None  # this process already knows about it

    model = (record or {}).get("model") or shape["model"] or getattr(config, "model", "") or ""
    peer_ips = list((record or {}).get("peer_ips") or getattr(config, "peer_ips", []) or [])
    executor = (shape.get("distributed_executor")
                or (record or {}).get("distributed_executor")
                or getattr(config, "distributed_executor", "ray") or "ray")
    width = (shape["tensor_parallel_size"] or shape["nnodes"]
             or _as_int((record or {}).get("tensor_parallel_size"), 0)
             or 1 + len(peer_ips))
    instance_id = ((record or {}).get("instance_id")
                   or f"{getattr(config, 'node_id', '') or 'head'}:{model}")

    engine = app.get("engine")
    reuse = (engine is not None
             and _as_int(getattr(getattr(engine, "config", None), "api_port", 0), 0) == port)
    if reuse:
        backend = engine
    else:
        inst_config = _instance_config(
            config, model=model, api_port=port, peer_ips=peer_ips,
            distributed_mode="head",
            overrides=dict((record or {}).get("overrides") or {},
                           distributed_executor=executor))
        backend = _build_backend(inst_config, api_port=port,
                                 node_port=_as_int(getattr(config, "api_port", 0), 0),
                                 started_at=shape["started_at"])
    serving = port_serving(port)
    _add_adopted(app, config, manager, model=model, api_port=port, peer_ips=peer_ips,
                 tensor_parallel_size=width, distributed_executor=executor,
                 instance_id=instance_id, backend=backend, serving=serving)
    if shape["container_id"]:
        _ADOPTED_CONTAINER_IDS.add(shape["container_id"])
    logger.info(
        "adopted the distributed head container %s: %s TP=%d on :%d across "
        "%d peer(s) (%s), engine %s, backend %s",
        name, model, width, port, len(peer_ips), executor,
        "answering" if serving else "not answering yet",
        "reused from this boot" if reuse else "rebuilt for the running container")

    written = write_distributed_record(
        config, model=model, api_port=port, peer_ips=peer_ips,
        tensor_parallel_size=width, distributed_executor=executor,
        instance_id=instance_id, container=name,
        overrides=(record or {}).get("overrides") or _config_overrides(config))
    return {"kind": "head", "container": name, "record": written,
            "model": model, "api_port": port, "serving": serving}


def _config_overrides(config) -> dict:
    """The launch keys worth carrying in the record, read off a config snapshot.

    The mp head is a config state today, so on the first boot under this code
    there is no launch to have supplied them: ``config.json`` is where the proven
    recipe actually lives, and a relaunch has to render the same flags or it
    brings the model back on the wrong image.
    """
    keys = ("engine_image", "extra_vllm_args", "extra_env", "extra_volumes",
            "kv_cache_dtype", "kv_cache_dtype_explicit", "max_model_len",
            "trust_remote_code", "quantization", "served_model_name",
            "gpu_memory_utilization", "distributed_executor")
    out: dict = {}
    for key in keys:
        value = getattr(config, key, None)
        if value not in (None, "", [], {}):
            out[key] = value
    return out


def _adopt_solo_primary(app, config) -> Optional[dict]:
    """Put the solo primary container back in the manager, when there is one.

    The boot decision (:func:`adopt_boot_engines`) is what kept the container;
    this is what makes the node's own view of it an INSTANCE rather than a bare
    ``app["engine"]`` handle. Normally ``create_app`` has already seeded that
    record from the boot engine, and then all this does is flip it to
    ``adopted`` and to ``serving``, because it is: the container predates this
    process. An app built with no engine handle gets the whole record here,
    backend included. Nothing happens without a boot decision to read, so a
    process that never ran the gate cannot adopt a container it never checked.
    """
    dmode = (getattr(config, "distributed_mode", "solo") or "solo").lower()
    if dmode != "solo":
        return None  # the head has its own path, a member adopts nothing
    entry = adopted_boot_primary()
    if entry is None:
        return None
    port = _as_int(entry.get("api_port"), _as_int(getattr(config, "api_port", 8000), 8000))
    manager = _manager(app, config)
    existing = manager.by_port(port)
    if existing is not None:
        existing.record.adopted = True
        existing.record.status = "serving"
        return {"kind": "primary", "container": entry.get("container", ""),
                "model": existing.record.model, "api_port": port, "serving": True,
                "seeded": True}
    model = entry.get("model") or (getattr(config, "model", "") or "")
    backend = app.get("engine")
    if backend is None:
        backend = _build_backend(
            _instance_config(config, model=model, api_port=port, peer_ips=[],
                             distributed_mode="solo"),
            api_port=port, node_port=_as_int(getattr(config, "api_port", 0), 0),
            started_at=entry.get("started_at"))
    _add_adopted(app, config, manager, model=model, api_port=port, peer_ips=[],
                 tensor_parallel_size=_as_int(entry.get("tensor_parallel_size"), 1) or 1,
                 # A single-node record's executor is the InstanceRecord default,
                 # the same value ``append_solo_instance`` leaves on one: nothing
                 # reads it below TP=2, and the two paths must not differ.
                 distributed_executor="ray",
                 instance_id=f"{getattr(config, 'node_id', '') or 'head'}:{model}",
                 backend=backend, serving=True)
    logger.info("adopted the primary container %s: %s on :%d, up %s",
                entry.get("container", ""), model, port,
                entry.get("uptime") or "unknown")
    return {"kind": "primary", "container": entry.get("container", ""),
            "model": model, "api_port": port, "serving": True, "seeded": False}


def _adopt_stacked(app, config) -> list:
    """Adopt live stacked containers nothing in this process knows about.

    Two cases, told apart by whether anything on disk describes the container:

    * **the boot kept it** (#240): there is a manifest entry, and that entry is
      what a replay would have launched, so the adopted backend is built from it
      rather than from the argv. The entry is the full override set (the recipe's
      engine image, its extra flags, its context length) and a snapshot missing
      those would be written back OVER the entry by the next manifest save.
    * **an orphan**: the manifest write is the last step of a stacked launch, so a
      crash between the two leaves a container serving a model no restart will
      ever replay, on a port the manager thinks is free. The container's own argv
      is then the only account there is. Adoption puts it back AND writes it into
      the manifest, which is what makes the repair survive the next restart
      (#179).
    """
    from dataclasses import replace

    from ainode.models.api_routes import (
        _OVERRIDE_KEYS,
        _resolved_overrides,
        load_instance_manifest,
        save_instance_manifest,
    )

    node_port = _as_int(getattr(config, "api_port", 8000), 8000)
    manager = _manager(app, config)
    entries = {str(entry["model"]): entry for entry in load_instance_manifest()
               if isinstance(entry, dict) and entry.get("model")}
    adopted: list = []
    for name in list_engine_containers():
        if not name.startswith(STACKED_ENGINE_PREFIX):
            continue
        info = inspect_container(name)
        if info is None:
            continue
        shape = container_shape(info)
        if not shape["running"]:
            continue
        suffix = name[len(STACKED_ENGINE_PREFIX):]
        port = shape["api_port"] or _as_int(suffix, 0)
        if not port or port == node_port or manager.by_port(port) is not None:
            continue
        model = shape["model"]
        if not model:
            logger.warning("stacked container %s is running but its argv names no "
                           "model; leaving it alone", name)
            continue
        manifest_entry = entries.get(model)
        if manifest_entry is not None:
            overrides = {k: manifest_entry[k] for k in _OVERRIDE_KEYS
                         if k in manifest_entry}
            inst_config = replace(
                config, model=model, distributed_mode="solo", peer_ips=[],
                api_port=port,
                **_resolved_overrides(manifest_entry.get("gpu_memory_utilization"),
                                      overrides))
        else:
            inst_config = _instance_config(
                config, model=model, api_port=port, peer_ips=[],
                distributed_mode="solo",
                overrides={"gpu_memory_utilization": shape["gpu_memory_utilization"],
                           "max_model_len": shape["max_model_len"],
                           "kv_cache_dtype": shape["kv_cache_dtype"] or None})
        backend = _build_backend(inst_config, api_port=port, node_port=node_port,
                                 started_at=shape["started_at"])
        serving = port_serving(port)
        _add_adopted(app, config, manager, model=model, api_port=port, peer_ips=[],
                     tensor_parallel_size=_as_int(shape.get("tensor_parallel_size"), 0) or 1,
                     distributed_executor="ray",
                     instance_id=f"{getattr(config, 'node_id', '') or 'head'}:{model}",
                     backend=backend, serving=serving)
        if shape["container_id"]:
            _ADOPTED_CONTAINER_IDS.add(shape["container_id"])
        logger.info("adopted the %s stacked container %s: %s on :%d, engine %s",
                    "recorded" if manifest_entry is not None else "orphan",
                    name, model, port, "answering" if serving else "not answering yet")
        adopted.append({"kind": "stacked", "container": name, "model": model,
                        "api_port": port, "serving": serving})
        if manifest_entry is None:
            # It was never written down (the crash this repairs), so write it.
            save_instance_manifest(app)
            entries[model] = {"model": model}
    return adopted


# ---------------------------------------------------------------------------
# The replay policy
# ---------------------------------------------------------------------------

async def replay_distributed_if_needed(app) -> dict:
    """Bring the recorded distributed shape back, or say why it cannot come back.

    Called once per boot, after adoption and after the stacked replay. Four
    outcomes, each reported in the return value's ``action``:

    ``none``       no record, so this node was not serving a distributed shape.
    ``adopted``    the container is alive and already in the manager: nothing to do.
    ``relaunched`` the container was gone and every peer answered the probe.
    ``degraded``   the container was gone and a peer did not answer, or the
                   relaunch failed. The record says which peer and what it said,
                   and nothing retries until a human or the next restart.
    """
    global _REPLAY_ATTEMPTED
    import asyncio

    config = app.get("config")
    record = load_distributed_record()
    if config is None or record is None:
        return {"action": "none"}
    if (getattr(config, "distributed_mode", "solo") or "solo").lower() == "member":
        # This node has been turned into somebody else's peer since the record was
        # written. A member launches nothing of its own, so the record is stale
        # rather than actionable, and it is reported, not acted on.
        logger.info("distributed record present but this node is a member now; "
                    "not replaying %s", record.get("model"))
        return {"action": "none", "reason": "node is a member"}
    if _REPLAY_ATTEMPTED:
        return {"action": "none", "reason": "already attempted this boot"}
    _REPLAY_ATTEMPTED = True

    port = _as_int(record.get("api_port"), _as_int(getattr(config, "api_port", 8000), 8000))
    manager = _manager(app, config)
    if manager.by_port(port) is not None:
        return {"action": "adopted", "record": mark_record_serving(record)}

    loop = asyncio.get_event_loop()
    peers = [str(p) for p in (record.get("peer_ips") or [])]
    ssh_user = record.get("ssh_user") or getattr(config, "ssh_user", "") or "ubuntu"
    unreachable: list = []
    answers: list = []
    for peer in peers:
        ok, answer = await loop.run_in_executor(
            None, lambda p=peer: probe_peer(p, ssh_user))
        answers.append({"peer_ip": peer, "ok": ok, "answer": answer})
        if not ok:
            unreachable.append({"peer_ip": peer, "answer": answer})

    if unreachable:
        detail = ", ".join(f"{u['peer_ip']} answered {u['answer']!r}"
                           for u in unreachable)
        reason = (f"{record.get('model')} TP={record.get('tensor_parallel_size')} is "
                  f"not running here and {len(unreachable)} of {len(peers)} peer(s) "
                  f"did not answer: {detail}")
        return {"action": "degraded", "peers": answers,
                "record": mark_record_degraded(record, reason, unreachable)}

    logger.info("replaying the distributed shape %s TP=%s: every peer answered",
                record.get("model"), record.get("tensor_parallel_size"))
    # The per-node launch slot, held until this engine BINDS rather than until the
    # launch returns: vLLM sizes its KV cache from what is free when it profiles,
    # so a UI load that cut in here would under-provision one of the two (#96).
    from ainode.models.api_routes import (
        WAIT_FOREVER,
        acquire_launch_slot,
        hold_launch_slot_until_bound,
        release_launch_slot,
    )
    label = f"distributed replay {record.get('model')}"
    await acquire_launch_slot(label, wait=WAIT_FOREVER)
    handed_off = False
    try:
        ok, detail = await loop.run_in_executor(
            None, lambda: _relaunch_from_record(app, config, record))
        if not ok:
            reason = (f"{record.get('model')} could not be relaunched even though every "
                      f"peer answered: {detail}")
            return {"action": "degraded", "peers": answers,
                    "record": mark_record_degraded(record, reason, [])}
        inst = manager.by_port(port)
        asyncio.get_event_loop().create_task(hold_launch_slot_until_bound(
            app, port, inst.backend if inst is not None else None, label))
        handed_off = True
        return {"action": "relaunched", "peers": answers,
                "record": mark_record_serving(record)}
    finally:
        if not handed_off:
            release_launch_slot()


def _relaunch_from_record(app, config, record: dict) -> tuple:
    """One attempt at the recorded distributed launch. ``(ok, detail)``.

    The caller holds the node's launch slot around this and hands it to a bind
    watch when it succeeds, same as every other launch path (#96).
    """
    port = _as_int(record.get("api_port"), _as_int(getattr(config, "api_port", 8000), 8000))
    model = record.get("model") or ""
    peers = [str(p) for p in (record.get("peer_ips") or [])]
    executor = record.get("distributed_executor") or "ray"
    overrides = dict(record.get("overrides") or {})
    overrides["distributed_executor"] = executor
    inst_config = _instance_config(config, model=model, api_port=port,
                                  peer_ips=peers, distributed_mode="head",
                                  overrides=overrides)
    from ainode.engine.backends import get_backend
    node_port = _as_int(getattr(config, "api_port", 0), 0)
    backend = get_backend(inst_config, instance_id=("" if port == node_port else str(port)))
    try:
        started = backend.start_distributed()
    except Exception as exc:
        logger.exception("replay of the distributed shape failed")
        return False, str(exc)
    if not started:
        return False, "start_distributed() returned False"

    manager = _manager(app, config)
    width = _as_int(record.get("tensor_parallel_size"), 1 + len(peers))
    _add_adopted(app, config, manager, model=model, api_port=port, peer_ips=peers,
                 tensor_parallel_size=width, distributed_executor=executor,
                 instance_id=(record.get("instance_id")
                              or f"{getattr(config, 'node_id', '') or 'head'}:{model}"),
                 backend=backend, serving=False)
    inst = manager.by_port(port)
    if inst is not None:
        # Launched by this process, not reconstructed from a container.
        inst.record.adopted = False
    if port == node_port:
        app["engine"] = backend
    return True, "launched"
