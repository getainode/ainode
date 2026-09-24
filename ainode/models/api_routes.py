"""API route handlers for model management."""

from __future__ import annotations

import asyncio
import re
import aiohttp
import json
import logging
import shlex
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from aiohttp import web

from ainode.auth.fleet import fleet_headers
from ainode.auth.middleware import TRUST_REMOTE_CODE_RULE, is_authenticated
from ainode.core.gpu import detect_gpu
from ainode.engine.reconcile import (
    adopt_running_engines,
    adopted_boot_primary,
    replay_distributed_if_needed,
)
from ainode.models import fit as model_fit
from ainode.models.registry import ModelManager

logger = logging.getLogger(__name__)


# -- Disk fit: refuse a download that cannot land ------------------------------
# A pull with nowhere to go does not fail as a disk error; it dies part way and
# the engine waiting for those weights looks like an engine that crashed (#184
# point 4). The size, the free space and the decision live in
# ``ainode/models/fit.py``; this is the HTTP shape of the refusal.
#
# 507 Insufficient Storage, because that is what happened and no other code in
# this file means it: 400 would say the caller asked wrongly, and 409 is the
# launch-slot refusal the dashboard already reads as "another load is running".
FIT_REFUSED_STATUS = 507


def _wants_force(body: dict) -> bool:
    """Did the caller say "download it anyway"? Both spellings accepted."""
    for key in ("force", "force_download"):
        value = (body or {}).get(key)
        if isinstance(value, bool) and value:
            return True
        if isinstance(value, str) and value.strip().lower() in ("1", "true", "yes"):
            return True
    return False


def _model_on_disk(request: web.Request, model: str) -> bool:
    """Are ``model``'s weights already under this node's models directory?

    One home for the layout list is ``registry.find_model_dir``; this only asks
    it, and answers False on any failure so the fit check runs rather than being
    skipped by an unreadable directory.
    """
    manager = request.app.get("model_manager")
    if manager is None:
        return False
    try:
        from ainode.models.registry import find_model_dir

        return find_model_dir(Path(manager.models_dir), model) is not None
    except Exception:
        return False


async def _refuse_if_it_will_not_fit(request: web.Request, hf_repo: str, *,
                                     force: bool = False,
                                     catalog_size_gb=None,
                                     learn: bool = True):
    """None when the transfer may proceed; a 507 response when it may not.

    ``learn`` is False on the launch path: a launch must not wait on an HTTP
    round trip to the Hub before it starts, so it decides from what this node
    already knows (a remembered size, the catalog entry). A size nobody can
    learn is unknown, and unknown always proceeds -- the check refuses on
    evidence, never on the absence of it.
    """
    manager: Optional[ModelManager] = request.app.get("model_manager")
    if manager is None:
        return None
    loop = asyncio.get_event_loop()
    verdict = await loop.run_in_executor(
        None,
        lambda: model_fit.check_fit(hf_repo, manager.models_dir,
                                    catalog_size_gb=catalog_size_gb, learn=learn))
    if verdict.fits is False:
        if force:
            logger.warning("forced past the fit check: %s", verdict.message())
            return None
        logger.warning("refused for disk: %s", verdict.message())
        return web.json_response(model_fit.refusal_payload(verdict),
                                 status=FIT_REFUSED_STATUS)
    return None

# Serialize model downloads so two concurrent fat pulls can't gang up on the
# uplink. Concurrency is AINODE_MAX_CONCURRENT_DOWNLOADS (default 1). Lazily
# built so the asyncio primitive binds to the running loop.
_DOWNLOAD_SEM = None


def _download_gate():
    global _DOWNLOAD_SEM
    if _DOWNLOAD_SEM is None:
        import os
        try:
            n = max(1, int(os.environ.get("AINODE_MAX_CONCURRENT_DOWNLOADS", "1")))
        except (TypeError, ValueError):
            n = 1
        _DOWNLOAD_SEM = asyncio.Semaphore(n)
    return _DOWNLOAD_SEM


# --- Per-node launch slot -----------------------------------------------------
# ONE engine launches at a time on a node. vLLM sizes its KV cache from what is
# FREE when the engine profiles, so two engines profiling at once under-provision
# whichever finishes second: on the 0.5.11 roll a stacked Ornith launched 2 s
# behind the primary and died with "Available KV cache memory: 1.59 GiB" at the
# same gpu_memory_utilization that gave it a 600K-token cache on a settled node
# (#96). Every launch path takes this slot -- the startup replay, the boot
# primary's wait, POST /api/models/load, POST /api/sharding/launch,
# POST /api/engine/set-model -- and HOLDS it until the engine binds, so the next
# launch always profiles against memory the previous one has finished reserving.
_LAUNCH_LOCK = None
_LAUNCH_LOCK_LOOP = None
# Label of whatever holds the slot, reported in the 409 a refused launch gets.
_LAUNCH_OWNER = None
# How long a queued launch waits for the in-flight one before it is refused. Short
# on purpose: a bind can take 12 minutes on a GB10, and a request that hangs that
# long is worse than a 409 naming who is launching.
_LAUNCH_QUEUE_SECONDS = 5.0
# ``wait=WAIT_FOREVER`` queues instead of ever raising LaunchBusy. The startup
# replay uses it -- boot has nobody to report a refusal to.
WAIT_FOREVER = float("inf")


class LaunchBusy(Exception):
    """Raised when the node's launch slot is held and the queue wait expired."""

    def __init__(self, owner=None):
        self.owner = owner or "another launch"
        super().__init__(f"{self.owner} is still launching on this node")


def _launch_lock():
    """The node's launch lock, bound lazily to the running loop.

    Rebuilt when the running loop changes: an ``asyncio.Lock`` binds to the first
    loop that awaits it and raises on every other one, and the test suite runs a
    fresh loop per test. In the product there is exactly one loop per process.
    """
    global _LAUNCH_LOCK, _LAUNCH_LOCK_LOOP, _LAUNCH_OWNER
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if _LAUNCH_LOCK is None or _LAUNCH_LOCK_LOOP is not loop:
        _LAUNCH_LOCK = asyncio.Lock()
        _LAUNCH_LOCK_LOOP = loop
        _LAUNCH_OWNER = None
    return _LAUNCH_LOCK


def launch_owner():
    """Label of the launch that holds the slot right now, or None."""
    if _LAUNCH_LOCK is not None and _LAUNCH_LOCK.locked():
        return _LAUNCH_OWNER
    return None


async def acquire_launch_slot(label: str, wait=None) -> None:
    """Take the node's launch slot for ``label``, or raise :class:`LaunchBusy`.

    ``wait`` is how long to queue before refusing: the default
    ``_LAUNCH_QUEUE_SECONDS``, ``WAIT_FOREVER`` to queue indefinitely, 0 to refuse
    an in-flight launch immediately.
    """
    global _LAUNCH_OWNER
    lock = _launch_lock()
    timeout = _LAUNCH_QUEUE_SECONDS if wait is None else float(wait)
    if lock.locked() and timeout != WAIT_FOREVER:
        if timeout <= 0:
            raise LaunchBusy(_LAUNCH_OWNER)
        try:
            await asyncio.wait_for(lock.acquire(), timeout)
        except (asyncio.TimeoutError, TimeoutError):
            raise LaunchBusy(_LAUNCH_OWNER) from None
    else:
        await lock.acquire()
    _LAUNCH_OWNER = label


def release_launch_slot() -> None:
    """Hand the slot back. A no-op when it is not held."""
    global _LAUNCH_OWNER
    _LAUNCH_OWNER = None
    lock = _LAUNCH_LOCK
    if lock is not None and lock.locked():
        lock.release()


@asynccontextmanager
async def launch_slot(label: str, wait=None):
    """Hold the node's launch slot for the duration of the block."""
    await acquire_launch_slot(label, wait)
    try:
        yield
    finally:
        release_launch_slot()


def launch_busy_error(busy: LaunchBusy) -> dict:
    """The refusal a caller gets when the slot is taken: 409 + who holds it."""
    return {
        "ok": False,
        "status": 409,
        "error": (
            f"Another launch is in flight on this node: {busy.owner}. Wait for it "
            f"to finish, then retry. Two engines profiling at once split the node's "
            f"free memory, so the second one sizes its KV cache from what the first "
            f"has not reserved yet and fails engine init."
        ),
    }


def register_model_routes(app: web.Application, manager: Optional[ModelManager] = None,
                          models_dir: Optional[str] = None) -> None:
    """Register model management routes on the aiohttp app.

    ``models_dir`` is the node's configured store (``config.models_dir``), the
    SAME directory the engine backends mount. Built with neither argument the
    manager used the module constant, so changing Config > Models directory
    repointed launches and left downloads, Installed and Delete on the old path
    (#205).
    """
    if manager is None:
        manager = ModelManager(models_dir=models_dir)

    app["model_manager"] = manager
    app["download_jobs"] = {}

    app.router.add_post("/api/models/load", handle_model_load)
    app.router.add_post("/api/models/unload", handle_model_unload)
    app.router.add_get("/api/models", handle_list_models)
    app.router.add_post("/api/models/refresh", handle_refresh_catalog)
    app.router.add_get("/api/models/recommended", handle_recommended)
    app.router.add_get("/api/models/search", handle_search_models)
    app.router.add_get("/api/models/trending", handle_trending_models)
    app.router.add_get("/api/models/latest", handle_latest_models)
    app.router.add_get("/api/models/openrouter", handle_openrouter_models)
    app.router.add_get("/api/models/ollama", handle_ollama_models)
    # Every LITERAL path under /api/models goes above the dynamic
    # /api/models/{model_id}, which would otherwise swallow it: aiohttp 3.9+
    # happens to resolve a plain resource first, but under registration-order
    # resolution `downloaded` would answer "Model 'downloaded' not found" and the
    # Installed list, the launch dropdown and the on-disk cards would all go
    # quietly empty (#209). Same care as /api/models/card|caps in api/server.py.
    app.router.add_get("/api/models/downloaded", handle_list_downloaded)
    app.router.add_get("/api/models/{model_id}", handle_get_model)
    app.router.add_post("/api/models/download-repo", handle_download_repo)
    app.router.add_post("/api/models/download-cancel", handle_cancel_download)
    app.router.add_get("/api/models/download/status", handle_download_status)
    app.router.add_get("/api/models/downloads/active", handle_active_downloads)
    app.router.add_post("/api/models/delete-repo", handle_delete_repo)
    app.router.add_post("/api/models/{model_id}/download", handle_download_model)
    app.router.add_delete("/api/models/{model_id}", handle_delete_model)


# -- Instance persistence (always-on) ----------------------------------------
# A node's loaded solo instances live in a tiny on-disk manifest (one JSON file
# under AINODE_HOME — no DB) so a `systemctl restart ainode` brings the same
# model set back automatically, no manual reload. The manifest is just each
# model + its KV reservation; the engine rebuilds the container from that.

def _manifest_path() -> Path:
    from ainode.core.config import AINODE_HOME
    return Path(AINODE_HOME) / "instances.json"


def consume_start_clean() -> bool:
    """One-shot 'start clean' signal: skip replaying persisted models this boot.

    A node restart otherwise reloads config.model (boot engine) + the stacked
    manifest, so 'restart to free a node' just reloads. This lets an operator
    start a node idle. Triggered by:
      - env AINODE_START_CLEAN truthy (persists across restarts), or
      - a sentinel file <AINODE_HOME>/.start-clean — a single-use
        `touch ~/.ainode/.start-clean && systemctl restart ainode` knob,
        consumed (deleted) here so the next restart serves normally.
    Non-destructive: the on-disk config + manifest are left intact.
    """
    import os
    from ainode.core.config import AINODE_HOME
    env = str(os.environ.get("AINODE_START_CLEAN", "")).strip().lower() in ("1", "true", "yes", "on")
    sentinel_present = False
    try:
        sentinel = Path(AINODE_HOME) / ".start-clean"
        if sentinel.exists():
            sentinel_present = True
            sentinel.unlink()
    except Exception:
        pass
    return env or sentinel_present


def save_instance_manifest(app) -> None:
    """Write the current solo instance set (model + gpu_memory_utilization)."""
    manager = app.get("instances")
    if manager is None:
        return
    entries = []
    for inst in manager.instances():
        cfg = getattr(inst.backend, "config", None)
        # Only persist solo instances — distributed (head) instances are out of
        # scope for auto-replay (they need peer coordination).
        if getattr(cfg, "distributed_mode", "solo") not in ("solo", None):
            continue
        entry = {
            "model": inst.record.model,
            "gpu_memory_utilization": getattr(cfg, "gpu_memory_utilization", None),
        }
        # round-trip per-load overrides so restart-replay restores aliases + ctx len
        for k in _OVERRIDE_KEYS:
            v = getattr(cfg, k, None)
            if v is not None:
                entry[k] = v
        entries.append(entry)
    try:
        p = _manifest_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"instances": entries}))
    except Exception:
        pass


def load_instance_manifest() -> list:
    try:
        p = _manifest_path()
        if not p.exists():
            return []
        return json.loads(p.read_text()).get("instances", []) or []
    except Exception:
        return []


def forget_distributed_instance(model: str, api_port) -> bool:
    """Drop the persisted distributed shape when THIS unload was that instance.

    The other half of ``write_distributed_record``: a shape that has been unloaded
    must not be replayed by the next restart, and the record is what a restart
    reads (#179). Matched on the port, which is what tells two instances of one
    model apart, and on the model, so unloading a neighbour never clears it.
    Returns True when a record was removed.
    """
    from ainode.engine.reconcile import clear_distributed_record, load_distributed_record
    record = load_distributed_record()
    if record is None:
        return False
    try:
        same_port = int(record.get("api_port") or 0) == int(api_port or 0)
    except (TypeError, ValueError):
        same_port = False
    if not same_port or (model and record.get("model") != model):
        return False
    clear_distributed_record()
    return True


# -- Launch-time ledger (how long this model took to come up here) -----------
#
# The only trace a launch used to leave was one log line ("<label> bound on
# :<port> after <N>s"), which nothing reads and no browser can see, so the
# interface could not answer the first question a user asks about a 27B on a
# GB10: how long is this going to take? Each bind wait now appends its verdict
# to a small JSON file beside the instance manifest, same shape of storage (one
# file under AINODE_HOME, no DB).
#
# FAILURES are recorded too, with the reason. A ledger holding only the launches
# that worked would quietly describe a model that never comes up as a model that
# comes up fast, and the point of writing this down is that it is honest about
# this node.
#
# Capped at the last _LAUNCH_TIMES_CAP entries: this is a rolling record for the
# UI, not history. History is `bench/results/`.

_LAUNCH_TIMES_CAP = 200


def _launch_times_path() -> Path:
    from ainode.core.config import AINODE_HOME
    return Path(AINODE_HOME) / "launch-times.json"


def read_launch_times() -> list:
    """Every ledger entry on this node, oldest first. No file / junk file -> []."""
    try:
        p = _launch_times_path()
        if not p.exists():
            return []
        entries = json.loads(p.read_text()).get("launches", []) or []
        return [e for e in entries if isinstance(e, dict)]
    except Exception:
        return []


def append_launch_time(entry: dict) -> None:
    """Append one launch verdict, keeping only the last _LAUNCH_TIMES_CAP.

    Blocking (a read and a write): every caller on the event loop goes through
    ``record_launch_time``, which offloads it. Never raises: a bookkeeping file
    must not be able to fail a launch.
    """
    try:
        entries = read_launch_times()
        entries.append(entry)
        if len(entries) > _LAUNCH_TIMES_CAP:
            entries = entries[-_LAUNCH_TIMES_CAP:]
        p = _launch_times_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"launches": entries}))
    except Exception:
        pass


def _instance_tp(app, port: int):
    """Tensor-parallel size of the instance on ``port``, when a record says so."""
    manager = app.get("instances") if hasattr(app, "get") else None
    if manager is None:
        return None
    try:
        for inst in manager.instances():
            if getattr(inst.record, "api_port", None) == port:
                return getattr(inst.record, "tensor_parallel_size", None)
    except Exception:
        return None
    return None


def launch_time_entry(app, port: int, backend, *, seconds: float, outcome: str,
                      reason: str = "") -> Optional[dict]:
    """The ledger row for one bind verdict, or None when there is nothing honest
    to write.

    The model is the engine's OWN config, not the wait's label: a label is
    display text ("boot primary a/one") and the ledger is looked up by model id.
    A wait with no engine config and no matching node config cannot name a model,
    so it records nothing rather than a row keyed on "".
    """
    cfg = getattr(backend, "config", None)
    node_cfg = app.get("config") if hasattr(app, "get") else None
    model = str(getattr(cfg, "model", "") or "").strip()
    if not model and node_cfg is not None and getattr(node_cfg, "api_port", None) == port:
        model = str(getattr(node_cfg, "model", "") or "").strip()
    if not model:
        return None
    node_port = getattr(node_cfg, "api_port", None)
    src = cfg if cfg is not None else node_cfg
    entry = {
        "model": model,
        "node_id": getattr(src, "node_id", None) or getattr(node_cfg, "node_id", None),
        "node_name": (getattr(src, "node_name", None)
                      or getattr(node_cfg, "node_name", None)),
        "api_port": port,
        # A stacked instance is one on a port other than the node's own: the
        # primary keeps config.api_port, every stacked load gets an allocated one.
        "stacked": bool(node_port and port != node_port),
        "tensor_parallel_size": _instance_tp(app, port),
        "engine_image": getattr(src, "engine_image", "") or "",
        "seconds_to_ready": round(float(seconds), 1),
        "stamp": _utc_stamp(),
        "outcome": outcome,
    }
    if reason:
        entry["reason"] = reason
    return entry


def _utc_stamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


async def record_launch_time(app, port: int, backend, *, seconds: float,
                             outcome: str, reason: str = "") -> None:
    """Append a bind verdict to the ledger, off the event loop.

    The write is a read + a write of a small file, and this runs on the server's
    loop during boot replay, so it goes to the executor like every other file
    write on this path.
    """
    entry = launch_time_entry(app, port, backend, seconds=seconds,
                              outcome=outcome, reason=reason)
    if entry is None:
        return
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, append_launch_time, entry)
    except Exception:  # pragma: no cover - defensive
        logger.exception("could not record the launch time for %s", entry.get("model"))


def last_ready_launch(model: str, hf_repo: str = "", entries=None) -> Optional[dict]:
    """The most recent ledger entry where this model actually reached READY.

    Matched on the ledger's model string against BOTH the catalog id and the HF
    repo, because the engine is launched with the repo id while the catalog (and
    the launch dropdown) may ask by either.
    """
    wanted = {str(x).strip().lower() for x in (model, hf_repo) if str(x or "").strip()}
    if not wanted:
        return None
    rows = read_launch_times() if entries is None else entries
    for entry in reversed(list(rows)):
        if entry.get("outcome") != "ready":
            continue
        if str(entry.get("model") or "").strip().lower() in wanted:
            return entry
    return None


def launch_time_summary(model: str, hf_repo: str = "", entries=None) -> dict:
    """``last_ready_minutes`` + ``last_ready_on`` for a model, from this ledger.

    Both are None when this node has never brought the model up, which is what
    lets the UI fall back to the catalog's ``typical_ready_minutes`` seed and say
    nothing at all when there is neither.
    """
    hit = last_ready_launch(model, hf_repo, entries)
    if hit is None:
        return {"last_ready_minutes": None, "last_ready_on": None}
    secs = hit.get("seconds_to_ready")
    minutes = (round(float(secs) / 60.0, 1)
               if isinstance(secs, (int, float)) and not isinstance(secs, bool) else None)
    stamp = str(hit.get("stamp") or "")
    return {
        "last_ready_minutes": minutes,
        "last_ready_on": {
            "node_name": hit.get("node_name") or hit.get("node_id") or None,
            "stacked": bool(hit.get("stacked")),
            "date": stamp[:10] or None,
            "tensor_parallel_size": hit.get("tensor_parallel_size"),
        },
    }


def annotate_launch_times(entries: list) -> list:
    """Add this node's measured load times to catalog dicts, in place.

    Used by GET /api/models, which is what the launch dropdown reads. One ledger
    read for the whole list, not one per model.
    """
    ledger = read_launch_times()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry.update(launch_time_summary(entry.get("id") or "",
                                         entry.get("hf_repo") or "",
                                         entries=ledger))
    return entries


_OVERRIDE_KEYS = ("served_model_name", "max_model_len", "kv_cache_dtype",
                  "kv_cache_dtype_explicit", "quantization", "trust_remote_code",
                  "extra_vllm_args", "engine_image", "extra_env", "extra_volumes")


def catalog_recipe(model: str) -> dict:
    """Proven launch recipe for a curated model, matched on catalog id OR hf_repo.

    Some models only serve correctly on a specific engine build with a specific
    flag set (spec-decode, MoE/mamba backends, reasoning + tool-call parsers),
    and some only on a specific distributed shape (an image with no ``ray`` can
    only do the mp one). Carrying that in the catalog is what makes them a
    one-click load instead of a hand-rolled container. Returns {} for anything
    not curated. Keys: ``engine_image``, ``extra_vllm_args``, ``extra_env``,
    ``extra_volumes``, ``distributed_executor``, ``kv_cache_dtype``,
    ``max_model_len``, ``trust_remote_code``, ``gpu_memory_utilization``:
    each present only when the entry actually states it.

    A CATALOG ID WINS OVER AN HF REPO, in two passes rather than one. Two entries
    may share a checkpoint and pin different recipes for different hardware
    (``qwen3.8-flash-next-nvfp4`` on GB10 at TP=2 against
    ``qwen3.8-flash-next-nvfp4-v100`` on four Volta cards at TP=4, one nightly
    aarch64 image against one local SM70 build). The id is the only handle that
    names one of them, so resolving it must not depend on which entry the dict
    happens to reach first. The repo id stays ambiguous by nature and answers
    with the first entry declaring it, which is why an entry in that position
    says in its own description that its lane is addressed by id.
    """
    from ainode.models.registry import CURATED_CLUSTER_MODELS
    m = (model or "").strip()
    if not m:
        return {}
    # The id pass, which is one dict lookup, then the repo pass over everything.
    by_id = {i.id: i for i in CURATED_CLUSTER_MODELS.values()}
    exact = by_id.get(m)
    for info in ([exact] if exact is not None else CURATED_CLUSTER_MODELS.values()):
        if m in (info.id, info.hf_repo):
            recipe = {}
            if getattr(info, "engine_image", ""):
                recipe["engine_image"] = info.engine_image
            if getattr(info, "extra_vllm_args", None):
                recipe["extra_vllm_args"] = list(info.extra_vllm_args)
            if getattr(info, "extra_env", None):
                recipe["extra_env"] = dict(info.extra_env)
            if getattr(info, "extra_volumes", None):
                recipe["extra_volumes"] = list(info.extra_volumes)
            executor = (getattr(info, "distributed_executor", "") or "").strip()
            if executor:
                # Any shape the entry states, "ray" included. This used to drop
                # "ray" as if it had not been stated, which was invisible while
                # ray was also the node default and became a real gap when the
                # default moved to "mp" (#172).
                recipe["distributed_executor"] = executor
            if getattr(info, "kv_cache_dtype", ""):
                recipe["kv_cache_dtype"] = info.kv_cache_dtype
                # A recipe dtype is a stated value, not the node default, so the
                # multimodal fp8→auto downgrade must not second-guess it.
                recipe["kv_cache_dtype_explicit"] = True
            if getattr(info, "max_model_len", 0):
                recipe["max_model_len"] = int(info.max_model_len)
            if getattr(info, "trust_remote_code", False):
                recipe["trust_remote_code"] = True
            if getattr(info, "recommended_gmu", 0):
                recipe["gpu_memory_utilization"] = info.recommended_gmu
            return recipe
    return {}


def catalog_size_gb(model: str):
    """The curated catalog's own size for ``model`` (id or hf_repo), or None.

    Read from the in-process CURATED_CLUSTER_MODELS dict and nothing else: the
    launch path calls this and must not reach the network or refresh a catalog
    cache to answer a disk question. None for anything not curated, which the
    fit check reports as an unknown size rather than as zero.
    """
    from ainode.models.registry import CURATED_CLUSTER_MODELS

    m = (model or "").strip()
    if not m:
        return None
    for info in CURATED_CLUSTER_MODELS.values():
        if m in (info.id, info.hf_repo):
            size = getattr(info, "size_gb", 0) or 0
            return float(size) if size > 0 else None
    return None


# Recipe keys that are per-instance launch config (everything except the gmu,
# which travels on its own because both load paths clamp it differently).
RECIPE_CONFIG_KEYS = ("engine_image", "extra_vllm_args", "extra_env",
                      "extra_volumes", "distributed_executor", "kv_cache_dtype",
                      "kv_cache_dtype_explicit", "max_model_len",
                      "trust_remote_code")


def parse_launch_overrides(body: dict, *, distributed: bool = False,
                           authenticated: bool = False, model: str = "") -> tuple:
    """Parse the per-launch config overrides out of a load/launch body.

    Returns ``(overrides, error)``: ``error`` is a message string when the body
    is malformed (the caller answers 400 with it) and None otherwise. Rejecting
    rather than silently dropping matters because a typo here otherwise surfaces
    as a container that dies with no explanation.

    Shared by ``/api/models/load`` and ``/api/sharding/launch`` so a distributed
    launch accepts the same keys a solo load does. ``distributed=True`` also
    accepts ``distributed_executor`` (meaningless for a solo load).

    ``trust_remote_code`` is the one key that is not simply the caller's to set:
    it makes the engine run the repository's own Python, so turning it ON needs
    either ``authenticated=True`` (the request presented an API key) or a
    ``model`` whose curated catalog recipe already declares it (#168). Turning it
    off is always allowed.
    """
    overrides: dict = {}
    smn = body.get("served_model_name")
    if isinstance(smn, str):
        smn = [smn]
    if isinstance(smn, list) and smn:
        overrides["served_model_name"] = [str(s) for s in smn if str(s).strip()]
    if body.get("max_model_len") is not None:
        try:
            overrides["max_model_len"] = int(body["max_model_len"])
        except (TypeError, ValueError):
            pass
    for k in ("kv_cache_dtype", "quantization"):
        if body.get(k) is not None:
            overrides[k] = body[k]
    if "kv_cache_dtype" in overrides:
        # Mark provenance so the multimodal fp8→auto safety downgrade
        # (nvidia.py _effective_kv_cache_dtype) is skipped: an EXPLICIT fp8 KV
        # request on a VLM is honored, giving the user a way to opt back in.
        overrides["kv_cache_dtype_explicit"] = True
    if body.get("trust_remote_code") is not None:
        wanted = bool(body["trust_remote_code"])
        if wanted and not authenticated and not catalog_recipe(model).get("trust_remote_code"):
            return {}, (
                f"{TRUST_REMOTE_CODE_RULE} "
                f"{model or 'This model'} is not a curated entry that declares it."
            )
        overrides["trust_remote_code"] = wanted
    # Recipe passthrough: extra vLLM flags + the engine image to run them on.
    if body.get("extra_vllm_args") is not None:
        raw = body["extra_vllm_args"]
        if isinstance(raw, str):
            raw = shlex.split(raw)
        if not isinstance(raw, list) or not all(isinstance(a, (str, int, float)) for a in raw):
            return {}, ("extra_vllm_args must be a list of strings "
                        "(e.g. [\"--moe-backend\", \"marlin\"]) or a shell-style string")
        overrides["extra_vllm_args"] = [str(a) for a in raw]
    if body.get("extra_env") is not None:
        raw = body["extra_env"]
        if not isinstance(raw, dict) or not all(
                isinstance(k, str) and k and isinstance(v, (str, int, float, bool))
                for k, v in raw.items()):
            return {}, ("extra_env must be an object of NAME -> value "
                        "(e.g. {\"VLLM_NVFP4_GEMM_BACKEND\": \"flashinfer-b12x\"})")
        overrides["extra_env"] = {k: str(v) for k, v in raw.items()}
    if body.get("extra_volumes") is not None:
        raw = body["extra_volumes"]
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list) or not all(
                isinstance(v, str) and ":" in v for v in raw):
            return {}, ("extra_volumes must be a list of \"host:container\" or "
                        "\"host:container:ro\" strings")
        overrides["extra_volumes"] = [str(v) for v in raw]
    if body.get("engine_image") is not None:
        img = str(body["engine_image"]).strip()
        if " " in img:
            return {}, "engine_image must be a single image ref"
        overrides["engine_image"] = img
    if distributed and body.get("distributed_executor") is not None:
        ex = str(body["distributed_executor"]).strip().lower()
        if ex not in ("ray", "mp"):
            return {}, ("distributed_executor must be \"ray\" (ray containers "
                        "+ docker exec) or \"mp\" (one vllm serve container per "
                        "node, vLLM's own multi-node executor)")
        overrides["distributed_executor"] = ex
    return overrides, None


def _resolved_overrides(gmu, overrides) -> dict:
    """Resolve the FULL per-load override set to concrete values, defaulting
    every field the caller did NOT supply to its NodeConfig class default.

    Returned as a field→value dict covering ``gpu_memory_utilization`` and every
    ``_OVERRIDE_KEYS`` entry. This is the single source of truth applied to BOTH
    the per-instance launch config (``inst_config``) and the persisted primary
    ``NodeConfig`` so the live backend and config.json can never diverge. Absent
    fields RESET to their default rather than inheriting the previous load's
    value — critical because ``inst_config`` is built from the SHARED, mutable
    ``app["config"]`` (which still carries the prior primary load's overrides),
    so without an explicit reset here, loading model B after model A would leak
    A's kv_cache_dtype/quantization/served_model_name/etc. onto B (a bare
    ``{"model": ...}`` load would silently inherit A's --quantization,
    --trust-remote-code, and --served-model-name).
    """
    from ainode.core.config import NodeConfig
    defaults = NodeConfig()
    ov = overrides or {}
    resolved = {
        "gpu_memory_utilization":
            gmu if gmu is not None else defaults.gpu_memory_utilization,
    }
    for k in _OVERRIDE_KEYS:
        resolved[k] = ov[k] if k in ov else getattr(defaults, k)
    return resolved


def _persist_primary_overrides(config, gmu, overrides) -> None:
    """Persist per-load overrides onto the SHARED NodeConfig for the primary.

    The primary solo model boots from ``NodeConfig`` (config.json) after a
    `systemctl restart` — NOT from the stacked-instance manifest — so every
    per-load override (kv_cache_dtype, max_model_len, served_model_name,
    trust_remote_code, quantization, gpu_memory_utilization) must be written
    here or the boot engine serves the model with stale/default values (the
    live VLM-came-back-on-fp8 bug). Uses the SAME resolved set as ``inst_config``
    so the persisted config always matches the live backend just launched.
    Caller saves config.
    """
    for k, v in _resolved_overrides(gmu, overrides).items():
        setattr(config, k, v)


def _will_own_primary_port(manager, config, existing) -> bool:
    """Will this load end up on the node's OWN api_port, i.e. be the primary?

    The InstanceManager is not the only thing that can hold that port, and assuming
    it was is the bug this exists to close. A node that HEADS a distributed launch
    runs its head engine in a container on ``config.api_port``, and that launch is
    deliberately never written to ``instances.json`` (a distributed shape is not
    auto-replayed), so after a `systemctl restart` the head keeps serving while the
    manager starts empty. Counting instances alone then called the next
    ``POST /api/models/load`` the PRIMARY: it answered ``{"api_port": 8001,
    "stacked": false}`` (``allocate_port`` had already skipped the busy 8000), and
    then overwrote ``config.model``, flipped ``distributed_mode`` back to "solo" and
    dropped ``peer_ips``. The node stopped advertising the model its head container
    was still serving, so the master's federated ``/v1/models`` and its routing lost
    a live distributed engine.

    So the question is asked about the port instead, which is the same definition of
    "stacked" the node's own ``/api/models`` listing already uses (``port !=
    node_port``): a load is the primary only when the primary port is actually free
    for it to take. ``existing`` holding that port is a reload of the primary and
    stays one.
    """
    node_port = getattr(config, "api_port", 0) or 0
    if not node_port:
        return True  # no api_port to own; nothing to protect either
    if existing is not None and existing.record.api_port == node_port:
        return True
    return manager.port_free(node_port)


def _is_distributed_record(record) -> bool:
    """Does this instance record describe a MULTI-NODE engine?

    Either half is enough: the peer list is what the launch spanned and the width
    is what vLLM was told, and an adopted record can carry the width from the
    container's own argv before anything has resolved peers.
    """
    if list(getattr(record, "peer_ips", []) or []):
        return True
    try:
        return int(getattr(record, "tensor_parallel_size", 1) or 1) > 1
    except (TypeError, ValueError):
        return False


def append_solo_instance(app, model: str, gmu=None, *, overrides=None, persist: bool = True) -> dict:
    """APPEND a solo instance through the InstanceManager — the shared core of the
    /api/models/load solo path AND the startup replay. Returns a plain dict (no
    HTTP). Each model gets its own container/port/config snapshot so several stack
    on one node; the first becomes the primary wired to app["engine"]."""
    from dataclasses import replace
    from ainode.discovery.instance import InstanceRecord
    from ainode.engine.instance_manager import InstanceManager
    from ainode.engine.backends import get_backend

    config = app.get("config")
    if config is None:
        return {"ok": False, "error": "Engine not initialized", "status": 503}

    manager = app.get("instances")
    if manager is None:
        manager = InstanceManager(base_port=config.api_port)
        app["instances"] = manager

    # Re-loading a model already up replaces THAT instance — other stacked
    # instances are untouched. We must NOT stop it yet: the admission gate below
    # can still reject the request (400/409), and killing the live instance
    # BEFORE that check would leave the model unloaded on a "failed" request with
    # no automatic restore. So decide admission first, then destroy.
    existing = manager.by_model(model)
    # ...unless THAT instance is a multi-node one. A solo load would stop the
    # distributed engine and bring the model back on one node, which for a
    # frontier MoE is a model that cannot fit and a head that is now gone. Before
    # #179 this could not happen (the manager was empty after a restart, so the
    # head was invisible here); now that adoption puts the head back, the
    # downgrade has to be refused out loud instead.
    if existing is not None and _is_distributed_record(existing.record):
        peers = list(getattr(existing.record, "peer_ips", []) or [])
        return {"ok": False, "status": 409,
                "error": (f"{model} is already serving here as a DISTRIBUTED instance "
                          f"(TP={getattr(existing.record, 'tensor_parallel_size', 1)} "
                          f"across this node + {len(peers)} peer(s) "
                          f"{', '.join(peers) or 'unknown'}) on port "
                          f"{existing.record.api_port}. A solo load would tear that "
                          f"engine down and bring the model back on one node. Unload "
                          f"it first, or relaunch it across the cluster.")}
    replaced_primary = existing is not None and app.get("engine") is existing.backend

    def _existing_id():
        return existing.record.instance_id if existing is not None else None

    # This load becomes the primary iff no OTHER instance remains once `existing`
    # (if any) is replaced — i.e. reloading the sole instance, or the very first
    # load. Reloading a stacked model while the primary is up is NOT primary.
    others = [i for i in manager.instances() if i.record.instance_id != _existing_id()]
    is_primary = len(others) == 0 and _will_own_primary_port(manager, config, existing)

    # Stacked-load admission control (unified-memory safety). A 2nd+ model on a
    # busy node with the node default gpu_memory_utilization (0.5) can push total
    # reservation past capacity and crash the whole GB10 (host death, power cycle).
    # Reloading the primary keeps today's behavior. Reloading an existing stacked
    # model excludes its own (about-to-be-freed) reservation from the running
    # total. Runs BEFORE stop/remove so a rejection leaves the live model intact.
    if not is_primary and not replaced_primary:
        if gmu is None:
            return {"ok": False, "status": 400,
                    "error": ("A stacked load (2nd+ model on this node) must specify "
                              "gpu_memory_utilization explicitly (e.g. 0.4) — refusing "
                              "to inherit the node default and risk overcommitting "
                              "unified memory.")}
        existing_total = 0.0
        for inst in others:
            g = getattr(getattr(inst.backend, "config", None), "gpu_memory_utilization", None)
            if g is not None:
                existing_total += float(g)
        projected = existing_total + gmu
        if projected > 0.9:
            return {"ok": False, "status": 409,
                    "error": (f"Refusing stacked load: this node already reserves "
                              f"{existing_total:.2f} of GPU memory across "
                              f"{len(others)} instance(s); the requested "
                              f"{gmu:.2f} would total {projected:.2f} (> 0.90 cap). "
                              f"Unload a model or lower gpu_memory_utilization.")}

    # Admission passed (or N/A) — NOW it's safe to tear down the old instance.
    if existing is not None:
        try:
            existing.backend.stop()
        except Exception:
            pass
        manager.remove(existing.record.instance_id)

    port = manager.allocate_port()
    name_token = "" if port == config.api_port else str(port)  # primary keeps legacy names
    instance_id = f"{config.node_id or 'head'}:{model}"

    # Build the launch config from the RESOLVED override set. app["config"] is a
    # SHARED, mutable object that still carries the PREVIOUS primary's per-load
    # overrides, so `replace(config, ...)` alone would leak A's kv_cache_dtype /
    # quantization / served_model_name / trust_remote_code onto the next model B
    # (a bare {"model": ...} load). _resolved_overrides resets every unsupplied
    # field to its NodeConfig default, and is the SAME set persisted below, so the
    # live backend and config.json can never diverge.
    inst_config = replace(config, model=model, distributed_mode="solo",
                          peer_ips=[], api_port=port,
                          **_resolved_overrides(gmu, overrides))

    def _clear():
        # routing-truth: a failed primary launch must stop the node advertising a
        # model it isn't serving, or the federated router 502s on the ghost.
        if is_primary and config is not None:
            config.model = None
            try:
                config.save()
            except Exception:
                pass

    backend = get_backend(inst_config, instance_id=name_token)
    try:
        ok = backend.start()
    except Exception as exc:
        _clear()
        return {"ok": False, "error": f"Launch failed: {exc}", "status": 500}
    if not ok:
        _clear()
        # The backend's own words when it has them: a launch refused because the
        # recipe's engine image is not on this node and could not be pulled is
        # fixable, and only if the answer names the image (see
        # backends/base.py::launch_error).
        return {"ok": False,
                "error": (getattr(backend, "launch_error", "")
                          or "Failed to launch engine"),
                "status": 500}

    manager.add(InstanceRecord(
        instance_id=instance_id, model=model, head_node_id=config.node_id or "head",
        peer_ips=[], api_port=port, tensor_parallel_size=1, status="starting"), backend)

    if is_primary:
        # Back-compat: the proxy/status path reads app["config"] + app["engine"].
        config.model = model
        config.distributed_mode = "solo"
        config.peer_ips = []
        _persist_primary_overrides(config, gmu, overrides)
        try:
            config.save()
        except Exception:
            pass
        app["engine"] = backend
    elif replaced_primary:
        # Reloaded the primary while a stack exists: keep app["engine"] on the live
        # backend (not the stopped old one) so status/proxy don't dangle.
        config.model = model
        _persist_primary_overrides(config, gmu, overrides)
        try:
            config.save()
        except Exception:
            pass
        app["engine"] = backend

    if persist:
        save_instance_manifest(app)
    return {"ok": True, "model": model, "instance_id": instance_id,
            "api_port": port, "stacked": not is_primary}


# One poll of a bind wait. 3s matches the cadence the fixed window used.
_BIND_POLL_SECONDS = 3.0
# Pause before re-asking for the GPU on a relaunch: the previous engine's device
# release is what the failed attempt lost to (0.5.5).
_GPU_RELEASE_SECONDS = 30.0
# Used when no NodeConfig is reachable (a config written by an older release, or
# a caller that passes a bare dict). Mirrors the NodeConfig defaults.
_DEFAULT_BIND_LOG_SILENCE_SECONDS = 300.0
_DEFAULT_BIND_CEILING_SECONDS = 3600.0

# How many of the engine's own log lines are copied into the ainode log when it
# fails to bind. The container is about to be removed by the relaunch (a launch
# stop/rm's a leftover by name), and with it the only record of WHY it died: the
# 0.5.28 replay on Spark-4 said "never bound on :8001 after 79s (container
# exited)" and nothing anywhere said what the engine had printed (#235). Forty
# lines is a vLLM traceback plus the CUDA error above it.
REPLAY_LOG_TAIL_LINES = 40

# Each engine loading at the same time as this one buys the bind wait this much
# more of its base window, capped at BIND_WINDOW_MAX_SCALE. Concurrent loads on
# one node share its bandwidth, its CPU and its GPU, so a window sized for a
# single load is a kill switch when three are coming up: the same start that
# binds first time on a quiet node exits under memory pressure while the primary
# profiles (#235).
BIND_WINDOW_PER_EXTRA_INSTANCE = 0.5
BIND_WINDOW_MAX_SCALE = 3.0


def bind_window_scale(loading) -> float:
    """Multiplier for a bind window when ``loading`` engines are coming up.

    A pure function of the count: 1 engine is 1.0 (nothing changes for the
    single-load case), and every extra one adds
    ``BIND_WINDOW_PER_EXTRA_INSTANCE`` up to ``BIND_WINDOW_MAX_SCALE``. Anything
    unusable (None, 0, a negative, a string) reads as one engine.
    """
    try:
        count = int(loading)
    except (TypeError, ValueError):
        count = 1
    count = max(1, count)
    return min(BIND_WINDOW_MAX_SCALE,
               1.0 + (count - 1) * BIND_WINDOW_PER_EXTRA_INSTANCE)


async def _port_serving(port: int) -> bool:
    """One probe of http://localhost:<port>/v1/models, True on HTTP 200."""
    import urllib.request
    loop = asyncio.get_event_loop()

    def _probe() -> bool:
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=3) as r:
                return getattr(r, "status", r.getcode()) == 200
        except Exception:
            return False

    return await loop.run_in_executor(None, _probe)


async def _wait_port_ready(port: int, timeout: float = 300.0) -> bool:
    """Poll a port until it serves or a FIXED window expires.

    Only for waits with no engine handle to watch, i.e. a port whose engine
    this process did not launch. When we do hold the handle, ``_wait_for_bind``
    watches the engine instead of a clock.
    """
    step = _BIND_POLL_SECONDS
    for _ in range(max(1, int(timeout // step))):
        if await _port_serving(port):
            return True
        await asyncio.sleep(step)
    return False


def _engine_exited(backend) -> bool:
    """True only on positive evidence that the engine we launched is gone.

    Ask the backend about its CONTAINER first (``EngineBackend.engine_exited``):
    the solo launch is ``docker run -d``, whose client returns in about a second
    while the engine keeps loading, so a finished launch subprocess is not death.
    On 0.5.12 every solo launch read as "container exited" after 1 to 11 s and
    got a relaunch that failed on the container-name conflict with its own live
    engine. Only a backend with no container view (eugr's attached ``vllm
    serve``) falls back to the subprocess. Absence of any handle is NOT evidence
    of death: an engine that outlived a previous orchestrator must not read as a
    crash and earn an instant relaunch.
    """
    asker = getattr(backend, "engine_exited", None)
    if callable(asker):
        try:
            verdict = asker()
        except Exception:
            verdict = None
        if verdict is not None:
            return bool(verdict)
    proc = getattr(backend, "process", None)
    if proc is None:
        return False
    try:
        return proc.poll() is not None
    except Exception:
        return False


def _engine_log_mark(backend):
    """The engine's last-log-line stamp, or None if it publishes none.

    See ``EngineBackend.last_log_activity``. None means "no progress signal
    available": the wait then leans on container exit and the ceiling only,
    never on silence, or a backend that simply does not report would be
    relaunched for saying nothing.
    """
    ts = getattr(backend, "last_log_activity", None)
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        return None
    return ts


async def _engine_activity_mark(backend):
    """The engine's last-ACTIVITY stamp, or None if it publishes none.

    See ``EngineBackend.activity_mark``. This is the bind wait's primary proof
    of life: an engine loading weights, running torch.compile or autotuning
    FlashInfer prints nothing for minutes but burns CPU the whole time, so the
    log alone cannot tell that engine from a wedged one (#112).

    The probe may shell out (``docker stats`` costs about a second), so it runs
    in the executor: the bind loop is on the server's event loop during startup
    and must not block on docker. A backend that cannot see its engine's
    activity, or that raises, reads as None and the wait leans on the log.
    """
    probe = getattr(backend, "activity_mark", None)
    if not callable(probe):
        return None

    def _ask():
        try:
            return probe()
        except Exception:
            return None

    loop = asyncio.get_event_loop()
    ts = await loop.run_in_executor(None, _ask)
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        return None
    return ts


def _engine_launch_mark(backend):
    """Epoch seconds when this engine's container was launched, or None.

    The bind wait usually starts well AFTER the launch: the boot primary is
    launched by ``ainode start`` before the web server exists, and the replay only
    reaches its wait after the settle sleep and the pre-launch sweep. Timing the
    WAIT therefore understated the container's life badly -- a primary that lived
    47 s was logged as "never bound ... after 0s" on the 0.5.11 roll (#96). Prefer
    the backend's own launch stamp (``EngineBackend.launched_at``) and fall back to
    the start of the wait for a backend that publishes none.
    """
    ts = getattr(backend, "launched_at", None)
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        return None
    return ts


def _bind_limits(app, loading=1):
    """(silence seconds, ceiling seconds) from NodeConfig, with fallbacks.

    The first is how long BOTH liveness signals (engine activity, log) may be
    quiet before the wait calls the start dead -- not the log alone (#112),
    which is why the default came back down from 900 s to 300 s.

    ``loading`` is how many engines are coming up on this node at once, and it
    stretches both numbers by :func:`bind_window_scale`. One engine leaves the
    configured values exactly as they are.
    """
    config = app.get("config") if hasattr(app, "get") else None

    def _num(name: str, fallback: float) -> float:
        try:
            v = float(getattr(config, name, None) or 0)
        except (TypeError, ValueError):
            return fallback
        return v if v > 0 else fallback

    scale = bind_window_scale(loading)
    return (_num("engine_bind_log_silence_seconds",
                 _DEFAULT_BIND_LOG_SILENCE_SECONDS) * scale,
            _num("engine_bind_ceiling_seconds", _DEFAULT_BIND_CEILING_SECONDS) * scale)


async def _engine_log_tail(backend, lines: int = REPLAY_LOG_TAIL_LINES) -> str:
    """The engine's last ``lines`` log lines, or "" when it cannot say.

    A module-level seam over ``EngineBackend.log_tail`` so the replay can copy a
    dying engine's own words into the ainode log, and so a test can hand back
    canned output. Runs in the executor: the implementation shells out to
    ``docker logs`` and the caller is on the event loop during startup.
    """
    probe = getattr(backend, "log_tail", None)
    if not callable(probe):
        return ""

    def _ask() -> str:
        try:
            return str(probe(lines) or "")
        except Exception:
            return ""

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _ask)


async def _log_failed_engine_output(backend, label: str, reason: str) -> None:
    """Copy a failed engine's last output into the ainode log, before it is gone.

    The relaunch removes the container by name before starting a fresh one, and
    engines run without ``--rm`` precisely so a corpse survives to be read. This
    is the read: without it the only record of a replay that died during a
    concurrent load was the bind wait's one-line verdict (#235).
    """
    tail = await _engine_log_tail(backend)
    if not tail:
        return
    logger.warning("%s last %d log line(s) before the container is removed (%s):\n%s",
                   label, REPLAY_LOG_TAIL_LINES, reason, tail)


async def _wait_for_bind(app, port: int, backend, timeout: float = 300.0,
                         loading: int = 1):
    """Wait for ``port`` to serve. Returns (bound, reason, seconds_waited).

    Time-to-bind is not ours to predict. On vllm/vllm-openai:v0.27.1 a GB10 node
    spends minutes in FlashInfer fp4_gemm autotune and CUDA graph capture before
    the server listens: measured 2026-09-13 on spark-1, about 12 minutes for a
    27B NVFP4 model and 6 for a 35B-A3B. A fixed window shorter than that is a
    kill switch on healthy starts. That run's replay relaunched both engines,
    doubling a 14-minute boot to 28.

    So when we hold the engine's handle, wait on the ENGINE, not on a clock: it
    is alive while its container is up and it is visibly doing something. Two
    signals say that, and EITHER one resets the silence clock (#112):

    * ``activity_mark`` -- the container's CPU time advancing. The primary
      signal, because it holds through the phases that print nothing.
    * ``last_log_activity`` -- a new log line. Secondary: vLLM goes quiet for
      minutes inside weight load, torch.compile and FlashInfer autotune (206 s,
      363 s and one 48-minute autotune measured in two days), so log silence on
      its own is not evidence of a wedge.

    Give up only on evidence: the container exited, BOTH signals went quiet past
    the silence budget, or the absolute ceiling hit (a wedged-but-busy engine
    must not hold boot open forever). With no handle to watch, or a handle that
    reports nothing watchable, fall back to the fixed window.

    The seconds returned are how long the CONTAINER has been alive (from the
    backend's launch stamp when it publishes one), not how long this wait ran --
    the two differ by however late the wait started, which is what made a 47 s
    life read as "after 0s" (#96). The silence budget and the ceiling still measure
    the wait: they bound how long boot is held open, not the engine's life.

    ``loading`` is how many engines this node is bringing up at once; it stretches
    the silence budget, the ceiling and the fixed window through
    :func:`bind_window_scale`, because concurrent loads share one node's
    bandwidth and memory and a window sized for one of them kills the rest
    (#235).

    Every verdict this returns, bound or not, is appended to the node's
    launch-time ledger, the only place a load time is written down where the
    interface can read it back afterwards (see ``record_launch_time``).

    A bind is also where a decision model starts warming its answer grammar
    (``api/decide.py::schedule_decision_warmup``, #277), in the background and
    without holding the verdict: every launch path waits here, so no path can
    skip it.
    """
    bound, reason, alive = await _bind_wait(app, port, backend, timeout,
                                            loading=loading)
    await record_launch_time(app, port, backend, seconds=alive,
                             outcome="ready" if bound else "failed",
                             reason="" if bound else reason)
    if bound:
        _schedule_decision_warmup(app, port, backend)
    return bound, reason, alive


def _schedule_decision_warmup(app, port: int, backend) -> None:
    """Start a decision model's warm-up. Never lets a failure reach the bind."""
    try:
        from ainode.api.decide import schedule_decision_warmup
        schedule_decision_warmup(app, port, backend)
    except Exception:
        logger.exception("could not schedule the decision warm-up on :%s", port)


async def _bind_wait(app, port: int, backend, timeout: float = 300.0,
                     loading: int = 1):
    """The wait itself. The contract, the signals and the verdicts are documented
    on ``_wait_for_bind``, which is the entry point every caller uses and which
    records the verdict in the ledger."""
    launched = _engine_launch_mark(backend)
    began = time.monotonic()
    timeout = float(timeout) * bind_window_scale(loading)

    def _alive() -> float:
        """Seconds the container has been up, or the wait's own age as a fallback."""
        if launched is None:
            return time.monotonic() - began
        return max(0.0, time.time() - launched)

    # No handle, or a handle that publishes no activity probe, no log stamp and
    # no launch process, gives us nothing to be adaptive about -- so don't hold
    # the port open for the whole ceiling on it: that is the fixed-window case.
    activity = None if backend is None else await _engine_activity_mark(backend)
    if backend is None or (_engine_log_mark(backend) is None
                           and activity is None
                           and getattr(backend, "process", None) is None):
        ok = await _wait_port_ready(port, timeout=timeout)
        return ok, ("bound" if ok else f"fixed {timeout:.0f}s window expired"), _alive()

    silence, ceiling = _bind_limits(app, loading)
    mark = _engine_log_mark(backend)
    progress_at = began
    while True:
        if await _port_serving(port):
            return True, "bound", _alive()
        now = time.monotonic()
        waited = now - began
        if waited >= ceiling:
            return False, f"ceiling of {ceiling:.0f}s reached", _alive()
        if _engine_exited(backend):
            return False, "container exited", _alive()
        latest = _engine_log_mark(backend)
        if latest is not None and (mark is None or latest > mark):
            mark = latest
            progress_at = now
        # The engine's own activity, asked once per poll (the backend caches).
        # An advance here is proof of life even when the log says nothing.
        latest_activity = await _engine_activity_mark(backend)
        if latest_activity is not None and (activity is None or latest_activity > activity):
            activity = latest_activity
            progress_at = now
        silent_for = now - progress_at
        if (mark is not None or activity is not None) and silent_for >= silence:
            return False, f"no engine activity or log for {silent_for:.0f}s", _alive()
        await asyncio.sleep(_BIND_POLL_SECONDS)


async def _ensure_serving(app, port: int, relaunch, label: str, timeout: float = 300.0,
                          backend=None, loading: int = 1) -> bool:
    """Wait for an engine to bind, and if it died on the way up, relaunch ONCE.

    An engine can pass the launch check (its container reached Running) and then
    die minutes later during weight load. Observed 2026-08-19 on spark-3: the
    startup sweep killed the previous engine and the replacement launched while
    the driver was still releasing the GPU, so the nvidia hook handed it no
    device — `Can't initialize NVML`, `0 active driver(s) found` — and it exited
    during load. Nothing retried, so the node came back advertising nothing and
    the model stayed missing until a human re-loaded it.

    ``relaunch`` is a zero-arg callable that re-issues the launch. The retry is
    deliberately single: a model that fails twice has a real problem, and a retry
    loop would just hide it.

    ``backend`` is the engine handle whose liveness the wait watches (see
    ``_wait_for_bind``). Without it the wait is the old fixed ``timeout``, which
    relaunches a slow start that was never in trouble.

    ``loading`` is how many engines are coming up at once, and stretches the
    wait's windows accordingly (#235).

    Before the relaunch, the failed engine's own last lines are copied into the
    ainode log: the relaunch removes that container, and it was the only thing
    that knew why it died.
    """
    bound, reason, alive = await _wait_for_bind(app, port, backend, timeout,
                                                loading=loading)
    if bound:
        logger.info("%s bound on :%s after %.0fs", label, port, alive)
        return True
    # The seconds are the container's life, not this wait's: see _wait_for_bind.
    logger.warning("%s never bound on :%s after %.0fs (%s); relaunching once",
                   label, port, alive, reason)
    await _log_failed_engine_output(backend, label, reason)
    # Give the GPU time to finish releasing before asking for it again.
    await asyncio.sleep(_GPU_RELEASE_SECONDS)
    loop = asyncio.get_event_loop()
    try:
        ok = await loop.run_in_executor(None, relaunch)
    except Exception:
        logger.exception("%s relaunch raised", label)
        return False
    if not ok:
        logger.error("%s relaunch failed to start", label)
        return False
    served, reason, alive = await _wait_for_bind(app, port, backend, timeout,
                                                loading=loading)
    logger.info("%s relaunch %s after %.0fs%s", label,
                "is serving" if served else "still not serving", alive,
                "" if served else f" ({reason})")
    if not served:
        # The second corpse is read too: nothing else will remove it, so a human
        # can still docker logs it, but the reason belongs where the verdict is.
        await _log_failed_engine_output(backend, f"{label} relaunch", reason)
    return served


async def hold_launch_slot_until_bound(app, port: int, backend, label: str) -> bool:
    """Keep the node's launch slot until ``port`` binds, then release it.

    Launch-and-return is not enough: a second load two seconds behind this one
    profiles while this engine is still reserving memory, which is the #96 failure.
    The HTTP caller still gets its "launching" answer immediately -- this runs as a
    background task -- but until the engine binds, the next launch is queued and
    then refused with a 409 naming this one.
    """
    try:
        bound, reason, elapsed = await _wait_for_bind(app, port, backend)
        if bound:
            logger.info("%s bound on :%s after %.0fs", label, port, elapsed)
        else:
            logger.warning("%s never bound on :%s after %.0fs (%s); releasing the "
                           "launch slot", label, port, elapsed, reason)
        return bound
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("%s bind watch failed", label)
        return False
    finally:
        release_launch_slot()


async def launch_solo_serialized(app, model: str, gmu=None, *, overrides=None,
                                 persist: bool = True, label=None) -> dict:
    """``append_solo_instance`` under the node's launch slot (#96).

    Returns what ``append_solo_instance`` returns, or the 409 refusal shape when
    another launch is in flight. On a successful launch the slot is handed to a
    background bind watch, so the NEXT caller is refused until this engine binds
    instead of profiling alongside it.

    Not used by the startup replay: the replay already holds the slot for its whole
    serialized run, and the lock is not reentrant.
    """
    label = label or f"load {model}"
    try:
        await acquire_launch_slot(label)
    except LaunchBusy as busy:
        return launch_busy_error(busy)
    handed_off = False
    try:
        loop = asyncio.get_event_loop()
        # backend.start() shells out to docker -- run it off the event loop so a
        # concurrent request can still be answered (with a 409) while it runs.
        res = await loop.run_in_executor(
            None,
            lambda: append_solo_instance(app, model, gmu, overrides=overrides,
                                         persist=persist))
        if isinstance(res, dict) and res.get("ok") and res.get("api_port"):
            manager = app.get("instances")
            inst = manager.by_model(model) if manager is not None else None
            loop.create_task(hold_launch_slot_until_bound(
                app, res["api_port"], inst.backend if inst is not None else None, label))
            handed_off = True
        return res
    finally:
        if not handed_off:
            release_launch_slot()


# Bound on waiting for the daemon to finish removing swept engine containers
# before anything reuses their names (see the sweep below).
_ORPHAN_CLEAR_TIMEOUT_S = 90.0
_ORPHAN_CLEAR_POLL_S = 1.0


_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{12,64}$")

# Container-name prefixes for the engines a node launches LOCALLY. A solo engine
# is `ainode-vllm-node-solo` for the primary and `-<port>` per stacked instance; a
# distributed head is `ainode-vllm-head`. Peer worker containers
# (`ainode-vllm-worker-<ip>`) live on the peers and belong to whichever head
# placed them, so they are never swept from here.
_PRIMARY_ENGINE_NAME = "ainode-vllm-node-solo"
_STACKED_ENGINE_PREFIX = "ainode-vllm-node-solo-"
_HEAD_ENGINE_PREFIX = "ainode-vllm-head"


def _engine_name_filters(include_primary: bool) -> list:
    """`docker ps` name filters for this node's engine containers.

    ``include_primary`` widens the set from the stacked engines to EVERY engine
    this node owns, primary and distributed head included. Only the boot sweep,
    which runs before this process has launched anything, may widen it: a sweep
    that runs after a launch would remove the engine we just started.
    """
    prefixes = ([_PRIMARY_ENGINE_NAME, _HEAD_ENGINE_PREFIX] if include_primary
                else [_STACKED_ENGINE_PREFIX])
    out = []
    for prefix in prefixes:
        out += ["--filter", f"name={prefix}"]
    return out


def _engine_container_ids(include_primary: bool) -> list:
    """Ids of this node's engine containers the daemon still knows about.

    ``check_output`` rather than ``run`` on purpose: this is the poll seam, and
    tests that fake ``run`` positionally (stop, rm, run) must not be shifted by
    it. Only a line that looks like a container id counts, so a generic fake
    answer cannot read as "still present".
    """
    import subprocess
    try:
        out = subprocess.check_output(
            ["docker", "ps", "-aq", *_engine_name_filters(include_primary)],
            text=True, timeout=20, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    return [line.strip() for line in out.splitlines() if _CONTAINER_ID_RE.match(line.strip())]


def _engine_container_names(include_primary: bool) -> list:
    """Names of this node's surviving engine containers, for the timeout log.

    A sweep that gives up has to say WHICH container is still there, or the next
    person has only a count to go on.
    """
    import subprocess
    try:
        out = subprocess.check_output(
            ["docker", "ps", "-a", "--format", "{{.Names}}",
             *_engine_name_filters(include_primary)],
            text=True, timeout=20, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _without_adopted(ids: list) -> list:
    """``ids`` minus every container adoption has claimed (``reconcile.py``).

    One home for the comparison, because `docker ps -q` prints short ids and an
    inspect reports the full one, so the two are matched on either being a prefix
    of the other. Three callers need the same answer: the sweep must not REMOVE an
    adopted container, and neither poll may WAIT for one to disappear.
    """
    from ainode.engine.reconcile import adopted_container_ids
    keep = adopted_container_ids()
    if not keep:
        return list(ids)
    return [i for i in ids
            if i and not any(a.startswith(i) or i.startswith(a) for a in keep)]


def _orphan_engine_ids() -> list:
    """Stacked-engine poll seam. Kept zero-arg: tests fake it by name.

    An adopted container is not an orphan, so it is not what this poll is waiting
    to disappear: counting it would spend the whole clear timeout and then warn
    about a container the node is deliberately still running (#179).
    """
    return _without_adopted(_engine_container_ids(include_primary=False))


def _remove_engine_containers(include_primary: bool) -> list:
    """``docker rm -f`` this node's engine containers. Returns the ids removed.

    An engine ADOPTED by ``engine/reconcile.py`` is excluded: the container is
    still serving and has a record in the manager, so removing it here would take
    down the very instance the node just decided to keep (#179, #240). That now
    includes the BOOT sweep: `ainode start` asks ``adopt_boot_engines`` which
    containers match the configured recipe and are answering BEFORE it calls the
    sweep, which is what makes a restart keep its engine instead of reloading it.
    """
    import subprocess

    ps = subprocess.run(
        ["docker", "ps", "-aq", *_engine_name_filters(include_primary)],
        capture_output=True, text=True, timeout=20)
    ids = _without_adopted(ps.stdout.split())
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids],
                       capture_output=True, text=True, timeout=60)
    return ids


async def _sweep_orphan_engine_containers() -> None:
    """Remove stacked engine containers left over from a previous orchestrator.

    Stacked vLLM containers (ainode-vllm-node-solo-<port>) outlive the
    orchestrator restart, but the in-memory manager does not, so a surviving
    suffixed container is an orphan the replay is about to relaunch. Remove them
    first or the relaunch's `--name` collides (Conflict), and WAIT for the removal:
    `--rm` containers are deleted asynchronously, so `rm -f` returns while the
    daemon is still working and a `docker run --name` in that gap conflicts (every
    engine on the 0.5.8 roll, #80).

    Deliberately stacked-only. The primary and the head are swept by
    :func:`sweep_engines_before_boot`, which runs before this process has launched
    anything; widening the filter here would let a late sweep remove the primary
    this boot just started.
    """
    try:
        ids = _remove_engine_containers(include_primary=False)
        if not ids:
            return
        for _ in range(int(_ORPHAN_CLEAR_TIMEOUT_S / _ORPHAN_CLEAR_POLL_S)):
            if not _orphan_engine_ids():
                logger.info("orphan sweep removed %d stacked engine container(s)", len(ids))
                return
            await asyncio.sleep(_ORPHAN_CLEAR_POLL_S)
        logger.warning("orphan engine containers still present after %.0fs (%s); "
                       "replay continues", _ORPHAN_CLEAR_TIMEOUT_S,
                       ", ".join(_engine_container_names(False)) or "names unavailable")
    except Exception:
        logger.exception("orphan container sweep failed")


# One sweep per process, and only before the first launch. `ainode start` sweeps
# synchronously before it launches the boot primary; the replay's own sweep is the
# belt-and-braces for a boot that did not go through the CLI, and a no-op once the
# CLI has swept.
_BOOT_SWEEP_DONE = False


def _claim_boot_sweep() -> bool:
    """True for the first caller only; closes the pre-launch sweep window.

    Claimed BEFORE the work, not after: a second caller must not sweep a second
    time even if the first one failed, because by then an engine may be up.
    """
    global _BOOT_SWEEP_DONE
    if _BOOT_SWEEP_DONE:
        return False
    _BOOT_SWEEP_DONE = True
    return True


def sweep_engines_before_boot() -> list:
    """Free EVERY engine container this node owns, before anything launches.

    Engine containers are siblings spawned through docker.sock, so they survive
    the orchestrator restart that `ainode update` performs. On the 0.5.11 roll the
    old stacked engine was only reaped 14 s AFTER the new primary had started, so
    the primary profiled against a node that still had the previous model resident
    and both engines then under-sized their KV caches (#96). The fix is ordering:
    remove them, wait for the daemon to finish (names gone means the container and
    its process are gone), and only then launch.

    An engine container the boot ADOPTED is not swept and not waited for: it
    matched the configured recipe and is answering, so it is the engine this boot
    is going to keep serving (#240). Everything else goes, exactly as before.

    Synchronous by design -- it runs in `ainode start` before the event loop
    exists. Bounded by ``_ORPHAN_CLEAR_TIMEOUT_S``: a stuck container is logged by
    name and boot continues rather than hanging forever. Returns the ids removed.
    """
    if not _claim_boot_sweep():
        return []
    try:
        ids = _remove_engine_containers(include_primary=True)
    except Exception:
        logger.exception("pre-launch engine sweep failed")
        return []
    if not ids:
        return []
    deadline = time.monotonic() + _ORPHAN_CLEAR_TIMEOUT_S
    while True:
        still = _without_adopted(_engine_container_ids(include_primary=True))
        if not still:
            logger.info("pre-launch sweep freed %d engine container(s) from the "
                        "previous run", len(ids))
            return ids
        if time.monotonic() >= deadline:
            logger.warning("pre-launch sweep: %d engine container(s) still present "
                           "after %.0fs (%s); launching anyway", len(still),
                           _ORPHAN_CLEAR_TIMEOUT_S,
                           ", ".join(_engine_container_names(True)) or "names unavailable")
            return ids
        time.sleep(_ORPHAN_CLEAR_POLL_S)


async def ensure_startup_sweep() -> None:
    """The replay's half of the pre-launch sweep: stacked orphans, once per process.

    A no-op when `ainode start` already swept (the normal boot). Awaited BEFORE the
    replay launches anything, and before it waits on the boot primary, so no engine
    of a previous life is still holding memory while a new one profiles.
    """
    if not _claim_boot_sweep():
        return
    await _sweep_orphan_engine_containers()


# How long the replay lets the node settle before it touches anything: the web
# server and discovery come up first so the UI can show what is happening.
_REPLAY_SETTLE_SECONDS = 10.0


async def replay_instances_on_startup(app) -> None:
    """Always-on: after boot, re-load the persisted solo instance set so a node
    restart brings every previously-loaded model back with no manual step. The
    boot engine claims the primary (config.model); this replays the stacked rest.

    THE ORDER IS THE CONTRACT (#96):

    1. sweep every engine container this node owns from a previous life, and wait
       for the daemon to finish removing them (`ainode start` does this before it
       launches the boot primary; this is the belt-and-braces),
    2. let the boot primary bind, retrying it once if it died on the way up,
    3. launch the stacked instances one at a time, each waiting for the one before
       it to bind, all of it under the node's launch slot so a UI load cannot cut
       in.

    vLLM sizes its KV cache from what is free when it profiles, so an engine that
    profiles next to a still-loading neighbour -- or next to an old engine nobody
    has reaped yet -- under-provisions its cache and dies in engine init. A launch
    that fails after its one retry does not block the rest: it logs and the replay
    moves on to the next model.

    ADOPTION COMES FIRST, before the settle wait and before the sweep (#179). An
    orchestrator restart leaves the engine containers running, so the first thing
    this node has to do is find out what it is already serving: a container that
    is up is put back in the InstanceManager with its real shape, and never
    relaunched. `ainode start` has already made the KEEP-or-relaunch decision for
    the boot primary and the recorded stacked instances before it swept anything
    (#240), so by here those containers are still up and this step is what puts
    them in the manager. The distributed replay decision comes LAST, once the
    manager reflects reality, so a shape whose container is alive is recognised
    rather than launched a second time.
    """
    config = app.get("config")
    if config is None:
        return
    entries = load_instance_manifest()

    # What is this node already running? Asked before anything is swept or
    # launched, because the answer decides both. Both reconciliation steps are
    # module-level names here for the same reason ``ensure_startup_sweep`` is: a
    # test drives the replay's ordering with them faked, and neither may shell out
    # to a real docker on the machine running the suite.
    try:
        adopted = await adopt_running_engines(app)
        if adopted:
            logger.info("adopted %d running engine container(s) on startup", len(adopted))
    except Exception:
        logger.exception("engine adoption failed; continuing with the replay")

    await asyncio.sleep(_REPLAY_SETTLE_SECONDS)

    # Sweep BEFORE anything launches, and wait for it -- even with nothing to
    # replay, an engine from a previous life must not be left holding memory.
    await ensure_startup_sweep()

    # One launch at a time on this node, and the replay outranks nobody: it queues
    # rather than refusing, because boot has nobody to report a refusal to.
    if entries:
        async with launch_slot("startup replay", wait=WAIT_FOREVER):
            await _replay_serialized(app, config, entries)

    # The distributed shape last: one attempt, and only when the container is
    # gone AND every peer answers. Otherwise the record is marked degraded and
    # reported (ainode doctor, /api/status, the dashboard banner). It takes the
    # node's launch slot itself, and only around a real relaunch, because it holds
    # that slot until the engine BINDS (the lock is not reentrant, so it cannot be
    # wrapped in one here).
    try:
        outcome = await replay_distributed_if_needed(app)
        if outcome.get("action") not in (None, "none"):
            logger.info("distributed replay: %s", outcome.get("action"))
    except Exception:
        logger.exception("distributed replay failed")


#: The window for a primary this process has no engine handle for. It was a flat
#: 300 s, and on a node whose primary takes twelve minutes to bind that expired
#: while the primary was still profiling, which let the stacked replay launch into
#: the middle of its memory reservation (#235). Scaled by how many engines are
#: loading, like every other bind window here.
_PRIMARY_PORT_WAIT_SECONDS = 300.0


async def _await_primary_bind(app, config, loading: int = 1) -> bool:
    """Wait for the boot primary to SERVE. The gate the stacked replay waits on.

    Binding is the only signal that the primary has finished reserving memory and
    sizing its KV cache: vLLM profiles against what is free at that moment, so an
    engine that starts while the primary is still in loading_weights or profiling
    is the one that dies (#235, and #96 before it). "Started" is not that signal,
    and neither is a clock that ran out.

    Returns True when the port serves. With the engine's handle the wait watches
    the engine and retries it once; without one it is a fixed window, scaled by
    how many engines are coming up, and only when a primary is expected at all.
    """
    boot_engine = app.get("engine")
    port = int(getattr(config, "api_port", 8000) or 8000)
    model = getattr(config, "model", None)

    # An ADOPTED primary is already serving, so there is nothing to wait for and
    # nothing to relaunch (#240). It must also not go through the bind wait: that
    # wait reports the CONTAINER's life, so an engine adopted after fourteen hours
    # up would be written into the launch-time ledger as a fourteen-hour load and
    # every "how long does this model take here" answer would read it.
    adopted_primary = adopted_boot_primary()
    if adopted_primary is not None and await _port_serving(port):
        logger.info("boot primary %s is already serving on :%s, adopted from the "
                    "container that outlived the restart (up %s)",
                    adopted_primary.get("model") or model, port,
                    adopted_primary.get("uptime") or "unknown")
        # The warm flag died with the old process. An engine that already
        # compiled answers the warm-up at once, and one that did not needs it.
        _schedule_decision_warmup(app, port, boot_engine)
        return True
    # Otherwise wait for it to serve, retrying it once if it died on the way up:
    # a node that comes back with its main model silently missing is the #96 shape.
    if boot_engine is not None and model:
        # Pass the engine handle so the wait tracks ITS liveness (container up,
        # log advancing) instead of a fixed window a slow bind would blow past.
        return await _ensure_serving(app, port, boot_engine.start,
                                     f"boot primary {model}", backend=boot_engine,
                                     loading=loading)
    window = _PRIMARY_PORT_WAIT_SECONDS * (bind_window_scale(loading) if model else 1.0)
    if await _wait_port_ready(port, timeout=window):
        return True
    logger.warning("nothing serves on :%s after %.0fs and this process holds no "
                   "engine handle for it", port, window)
    return False


async def _replay_serialized(app, config, entries) -> None:
    """Boot primary, then one stacked instance at a time. Slot already held.

    The stacked instances wait for the primary's BIND, not for its start: see
    ``_await_primary_bind``. A primary that never binds does not cancel the
    replay, because a node has to come back serving whatever it can, but it is
    logged as what it is, so a stacked engine that then dies of memory pressure
    has the reason written down above it.
    """
    # How many engines this node is bringing up at once. Every bind window below
    # is scaled by it: concurrent loads share one node's bandwidth, CPU and
    # memory, and a window sized for a single load kills the rest (#235).
    loading = len(entries) + (1 if getattr(config, "model", None) else 0)

    primary_ready = await _await_primary_bind(app, config, loading)
    if not primary_ready and entries:
        logger.warning("the primary is not serving, so the %d stacked instance(s) "
                       "in the manifest are replayed without it: an engine that "
                       "profiles next to a primary still reserving memory is the "
                       "one that dies", len(entries))

    manager = app.get("instances")
    have = {i.record.model for i in manager.instances()} if manager is not None else set()
    if getattr(config, "model", None):
        have.add(config.model)
    loop = asyncio.get_event_loop()
    for e in entries:
        m = e.get("model")
        if not m or m in have:
            continue
        try:
            # backend.start() shells out to docker — run off the event loop.
            res = await loop.run_in_executor(
                None,
                lambda mm=m, g=e.get("gpu_memory_utilization"), ov={k: e[k] for k in _OVERRIDE_KEYS if k in e}: append_solo_instance(app, mm, g, overrides=ov, persist=False),
            )
            have.add(m)
            # Serialize: let this model bind before launching the next one, and
            # retry once if it died on the way up (same GPU-release race).
            if isinstance(res, dict) and res.get("ok") and res.get("api_port"):
                inst = manager.by_model(m) if manager is not None else None
                relaunch = (inst.backend.start if inst is not None
                            else (lambda mm=m, g=e.get("gpu_memory_utilization"),
                                  ov={k: e[k] for k in _OVERRIDE_KEYS if k in e}:
                                  bool(append_solo_instance(app, mm, g, overrides=ov,
                                                            persist=False).get("ok"))))
                await _ensure_serving(app, res["api_port"], relaunch, f"replay {m}",
                                      backend=inst.backend if inst is not None else None,
                                      loading=loading)
            else:
                # One model failing is not the next model's problem: say why and
                # carry on down the manifest.
                logger.error("replay load failed for %s: %s", m,
                             (res or {}).get("error") if isinstance(res, dict)
                             else "launch returned nothing")
        except Exception:
            logger.exception("replay load failed for %s", m)


# -- Handlers ------------------------------------------------------------------

async def handle_model_load(request: web.Request) -> web.Response:
    """POST /api/models/load — launch a model on this engine.

    Body: {"model": "<hf_repo>", "strategy": "auto|tensor_parallel|pipeline_parallel"}

    Behaviour:
      - If the local cluster has workers AND Ray is available, derive a
        ShardingConfig via ShardingPlanner and hand off to
        ``engine.launch_distributed``.
      - Otherwise, launch single-node (tensor_parallel = local GPU count).
      - Falls back gracefully when vLLM/Ray are missing.
    """
    from ainode.engine.sharding import ShardingPlanner, ShardingStrategy
    from ainode.engine.ray_autostart import RayAutostartState

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    model = (body.get("model") or "").strip()
    if not model:
        return web.json_response({"error": "model field required"}, status=400)

    strategy_str = body.get("strategy", "auto")
    try:
        strategy = ShardingStrategy(strategy_str)
    except ValueError:
        strategy = ShardingStrategy.AUTO

    engine = request.app.get("engine")
    cluster = request.app.get("cluster_state")
    config = request.app.get("config")
    ray_state: Optional[RayAutostartState] = request.app.get("ray_autostart_state")

    # Per-load KV-cache knob: lets a caller cap vLLM's GPU reservation so small
    # models don't hog a unified-memory node (and several can stack). Applied to
    # the per-instance config snapshot below — NOT the shared app config, which
    # would cross-wire a co-resident instance's reservation.
    gmu = None
    raw_gmu = body.get("gpu_memory_utilization")
    if raw_gmu is not None:
        try:
            gmu = max(0.05, min(0.95, float(raw_gmu)))
        except (TypeError, ValueError):
            gmu = None

    # Per-load config overrides applied to the per-instance snapshot only (NOT the
    # shared app config). served_model_name = API alias(es); the rest let stacked
    # models differ in context length / KV dtype / quant without cross-wiring.
    # Shared with /api/sharding/launch so both paths accept the same keys.
    overrides, err = parse_launch_overrides(
        body, authenticated=is_authenticated(request), model=model)
    if err:
        return web.json_response({"error": err}, status=400)

    # Curated models carry their proven recipe — apply it as DEFAULTS so a bare
    # {"model": "..."} load (i.e. clicking it in the dashboard) launches with the
    # engine image and flags it actually needs. Anything the caller stated
    # explicitly above wins; the recipe only fills the gaps.
    recipe = catalog_recipe(model)
    for key in RECIPE_CONFIG_KEYS:
        if key in recipe and key not in overrides:
            overrides[key] = recipe[key]
    if gmu is None and "gpu_memory_utilization" in recipe:
        gmu = recipe["gpu_memory_utilization"]

    # A launch whose weights are not on disk downloads them itself, inside the
    # engine container, so it pays the same disk question a download does and
    # fails the same way when the filesystem runs out: an engine that died
    # (#184 point 4). No Hub call on this path (learn=False) -- a launch must not
    # wait on huggingface.co -- so the size comes from what this node already
    # knows, and an unknown size launches exactly as it did before.
    if not _model_on_disk(request, model):
        refusal = await _refuse_if_it_will_not_fit(
            request, model, force=_wants_force(body), learn=False,
            catalog_size_gb=catalog_size_gb(model))
        if refusal is not None:
            return refusal

    # Decide: single-node or distributed?
    sharding_config = None
    if cluster is not None:
        try:
            worker_count = max(0, len(cluster.members()) - 1)
        except Exception:
            worker_count = 0
        ray_ready = bool(ray_state and (ray_state.is_head or ray_state.joined_as_worker))
        if worker_count > 0 and ray_ready:
            try:
                planner = ShardingPlanner()
                sharding_config = planner.plan_sharding(model, cluster, strategy)
                if ray_state and ray_state.head_address:
                    sharding_config.ray_head_address = ray_state.head_address
            except Exception as exc:
                return web.json_response(
                    {"error": f"Sharding plan failed: {exc}"}, status=422
                )

    def _clear_model_claim():
        # routing-truth: a failed launch must stop this node advertising a model
        # it isn't serving, or it becomes a ghost the federated router 502s on.
        if config is not None:
            config.model = None
            try:
                config.save()
            except Exception:
                pass

    # --- Distributed auto-shard path (Ray/TP) — singleton engine, unchanged ----
    # Under the node's launch slot: one engine profiles at a time (#96). The slot
    # is handed to a background bind watch once the launch is away, so the next
    # caller is refused (409) until this engine serves.
    if sharding_config is not None:
        label = f"distributed load {model}"
        try:
            await acquire_launch_slot(label)
        except LaunchBusy as busy:
            refused = launch_busy_error(busy)
            return web.json_response({"error": refused["error"]},
                                     status=refused["status"])
        handed_off = False
        try:
            if engine is None:
                # Lazy-create via get_backend (honors engine_backend=nvidia, not the
                # legacy host-venv VLLMEngine) for a node booted without an engine.
                try:
                    if config is None:
                        return web.json_response({"error": "Engine not initialized"},
                                                 status=503)
                    from ainode.engine.backends import get_backend
                    engine = get_backend(config)
                    request.app["engine"] = engine
                except Exception as exc:
                    return web.json_response({"error": f"Engine unavailable: {exc}"},
                                             status=503)
            if config is not None and getattr(config, "model", None) != model:
                config.model = model
                try:
                    config.save()
                except Exception:
                    pass
            try:
                if engine.is_running():
                    engine.stop()
            except Exception:
                pass
            try:
                success = engine.launch_distributed(sharding_config)
            except Exception as exc:
                _clear_model_claim()
                return web.json_response({"error": f"Launch failed: {exc}"}, status=500)
            if not success:
                _clear_model_claim()
                # Same reason as the solo path: report the backend's own sentence
                # when it has one, because a missing engine image names its fix.
                return web.json_response(
                    {"error": (getattr(engine, "launch_error", "")
                               or "Failed to launch engine")}, status=500)
            asyncio.get_event_loop().create_task(hold_launch_slot_until_bound(
                request.app, getattr(config, "api_port", 8000), engine, label))
            handed_off = True
            return web.json_response({
                "status": "launching", "model": model,
                "distributed": True, "plan": sharding_config.to_dict(),
            })
        finally:
            if not handed_off:
                release_launch_slot()

    # --- Solo path: APPEND an instance via the InstanceManager ------------------
    # A solo load no longer REPLACES the running model. Each model stacks (own
    # container/port/config snapshot); the set is persisted for auto-replay on
    # restart. Shared with the startup replay via append_solo_instance().
    if config is None:
        return web.json_response({"error": "Engine not initialized"}, status=503)

    # Serialized through the node's launch slot, which is held until this engine
    # binds: a second load 2 s behind this one would profile against memory this
    # one has not finished reserving (#96). A refused load comes back 409.
    result = await launch_solo_serialized(request.app, model, gmu, overrides=overrides)
    if not result.get("ok"):
        return web.json_response({"error": result.get("error")},
                                 status=result.get("status", 500))
    return web.json_response({
        "status": "launching",
        "model": result["model"],
        "instance_id": result["instance_id"],
        "api_port": result["api_port"],
        "stacked": result["stacked"],
        "distributed": False,
    })


async def handle_model_unload(request: web.Request) -> web.Response:
    """POST /api/models/unload {model, api_port?, all?} -- stop ONE instance.

    The dashboard UNLOAD button hits this endpoint through
    ``POST /api/cluster/unload`` with the card's own ``node_id`` and
    ``api_port``, so it stops the copy the card belongs to. Calls engine.stop(),
    which for EugrBackend tears down eugr's launch-cluster.sh, and for
    NvidiaBackend stops the head container + fan-outs `docker stop` to peer
    workers over SSH.

    ``api_port`` picks the instance when a node stacks several (two instances of
    the SAME model on one node are only told apart by their port).

    The fan-out to every peer serving the model id is OPT-IN, behind
    ``all: true``. It used to be what a plain ``{"model": X}`` did whenever this
    node was not serving it, so UNLOAD on one instance card took down every copy
    in the fleet (#207). A caller that wants one remote instance goes through
    ``/api/cluster/unload`` with its node id.

    For distributed (head) mode, flips config back to "solo" so a subsequent
    launch defaults sanely.

    `stopped` means "the instance is no longer serving after this call" — a
    dead/phantom/no-engine instance force-clears to stopped:true rather than
    requiring a live SIGTERM. `errors` still carries best-effort teardown detail.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    engine = request.app.get("engine")
    config = request.app["config"]
    errors = []

    # Instance-aware unload: stop the ONE instance serving `model`, leaving any
    # other stacked instances on this node serving. Falls through to the legacy
    # singleton teardown when no manager/model match (back-compat).
    manager = request.app.get("instances")
    if not isinstance(body, dict):
        body = {}
    model = (body.get("model") or "").strip()
    try:
        want_port = int(body.get("api_port") or body.get("port") or 0) or None
    except (TypeError, ValueError):
        want_port = None
    if manager is not None and (model or want_port):
        # The port wins when given: it names ONE instance even where the model id
        # does not. A port that matches nothing falls through rather than
        # stopping a neighbour by accident.
        inst = manager.by_port(want_port) if want_port else None
        if inst is not None and model and inst.record.model != model:
            inst = None
        if inst is None and not want_port:
            inst = manager.by_model(model)
        if inst is not None:
            model = model or inst.record.model
            try:
                inst.backend.stop()
            except Exception as exc:
                errors.append(f"instance.stop(): {exc}")
            manager.remove(inst.record.instance_id)
            # If the primary went away, repoint app["engine"]/config to a survivor
            # so the status/proxy back-compat path doesn't dangle on a dead backend.
            if request.app.get("engine") is inst.backend:
                survivors = manager.instances()
                if survivors:
                    keep = survivors[0]
                    request.app["engine"] = keep.backend
                    config.model = keep.record.model
                else:
                    request.app["engine"] = None
                    config.model = None
                    if getattr(config, "distributed_mode", "") == "head":
                        config.distributed_mode = "solo"
                try:
                    config.save()
                except Exception as exc:
                    errors.append(f"config.save: {exc}")
            # Persist the reduced set so a restart doesn't resurrect the unloaded one.
            save_instance_manifest(request.app)
            forget_distributed_instance(inst.record.model, inst.record.api_port)
            return web.json_response({
                "stopped": True, "model": model,
                "instance_id": inst.record.instance_id,
                "remaining": len(manager.instances()),
                "errors": errors,
            })

    if want_port:
        # A port was named and nothing on this node is serving on it. Falling
        # through to the legacy singleton teardown below would stop the node's
        # PRIMARY engine instead, which is the opposite of what naming a port
        # asks for.
        return web.json_response({
            "stopped": False, "model": model, "api_port": want_port,
            "scope": "port-miss",
            "errors": errors + [f"no instance is serving on port {want_port}"],
        })

    # Was THIS node actually serving the requested model? (back-compat: no model
    # given → stop whatever is local.) Only then is a local "stopped" truthful —
    # the old code returned stopped:true even when engine was None or serving a
    # different model, which is why the dashboard's Unload button silently no-op'd.
    served_here = engine is not None and (not model or getattr(config, "model", None) == model)
    if served_here:
        forget_distributed_instance(getattr(config, "model", "") or model,
                                    getattr(config, "api_port", 0))
        try:
            if engine.is_running():
                engine.stop()
        except Exception as exc:
            errors.append(f"engine.stop(): {exc}")
        try:
            engine._ready = False  # force the latch down so a phantom doesn't re-advertise
        except Exception:
            pass
        try:
            config.model = None
            if getattr(config, "distributed_mode", "") == "head":
                config.distributed_mode = "solo"
            config.save()
        except Exception as exc:
            errors.append(f"config clear: {exc}")
        return web.json_response({"stopped": True, "model": model, "scope": "local", "errors": errors})

    # Not serving here. This endpoint is node-scoped: it answers for THIS node and
    # stops there. `fanout=0` still means "do not recurse" (a fan-out child, or an
    # older head calling us).
    #
    # The fleet-wide fan-out lives behind `all: true`. It was the default whenever
    # this node was not serving the model, which made the UI's UNLOAD button stop
    # every copy in the fleet (#207); a caller that wants one remote instance goes
    # through /api/cluster/unload with its node id.
    fanned_out = bool(body.get("all")) and request.query.get("fanout") != "0"
    if not fanned_out:
        return web.json_response({"stopped": False, "model": model, "scope": "local-miss",
                                  "errors": errors})

    cluster = request.app.get("cluster_state")
    session = request.app.get("client_session")
    remote_stopped = False
    peers_reached = 0
    if cluster is not None and session is not None and model:
        for node in cluster.members():
            if node.node_id == config.node_id:
                continue
            host = node.fabric_ip or node.node_name
            if not host:
                continue
            url = f"http://{host}:{node.web_port}/api/models/unload?fanout=0"
            try:
                async with session.post(url, json={"model": model},
                                        headers=fleet_headers(request.app),
                                        timeout=aiohttp.ClientTimeout(total=30)) as r:
                    peers_reached += 1
                    jr = await r.json()
                    if jr.get("stopped"):
                        remote_stopped = True
                        errors.extend(jr.get("errors") or [])
            except Exception as exc:
                errors.append(f"peer {node.node_id}: {exc}")

    return web.json_response({
        "stopped": remote_stopped,
        "model": model,
        "scope": "remote-fanout",
        "peers_reached": peers_reached,
        "errors": errors,
    })


async def handle_list_models(request: web.Request) -> web.Response:
    """GET /api/models -- list the dynamic catalog with download status.

    Each entry also carries what this node measured the last time it brought the
    model up (``last_ready_minutes`` / ``last_ready_on``) alongside the catalog's
    ``typical_ready_minutes`` seed, because this is the list the launch dropdown
    reads and "how long will this take" is the question it has to answer before
    anybody clicks LAUNCH. The ledger is read once for the whole list, in the same
    executor call as the catalog scan.
    """
    manager: ModelManager = request.app["model_manager"]
    loop = asyncio.get_event_loop()

    def _listing():
        return annotate_launch_times(manager.list_available())

    models = await loop.run_in_executor(None, _listing)
    return web.json_response({"models": models, "count": len(models)})


async def handle_refresh_catalog(request: web.Request) -> web.Response:
    """POST /api/models/refresh -- force re-fetch of dynamic catalog."""
    manager: ModelManager = request.app["model_manager"]
    loop = asyncio.get_event_loop()
    # refresh=True bypasses both in-memory and on-disk caches
    models = await loop.run_in_executor(None, lambda: manager.get_catalog(refresh=True))
    return web.json_response({"status": "refreshed", "count": len(models)})


async def handle_get_model(request: web.Request) -> web.Response:
    """GET /api/models/:model_id -- info for a specific model."""
    model_id = request.match_info["model_id"]
    manager: ModelManager = request.app["model_manager"]
    info = manager.get_model_info(model_id)
    if info is None:
        return web.json_response(
            {"error": f"Model '{model_id}' not found in catalog"},
            status=404,
        )
    return web.json_response(info)


async def handle_download_model(request: web.Request) -> web.Response:
    """POST /api/models/:model_id/download -- start async download, return 202."""
    model_id = request.match_info["model_id"]
    manager: ModelManager = request.app["model_manager"]

    entry = manager.get_model_info(model_id)
    if entry is None:
        return web.json_response(
            {"error": f"Model '{model_id}' not found in catalog"},
            status=404,
        )

    # Does it fit? The catalog entry's own size is the fallback if the Hub will
    # not answer; a repo whose size nobody knows is downloaded as before.
    try:
        body = await request.json()
    except Exception:
        body = {}
    refusal = await _refuse_if_it_will_not_fit(
        request, entry.get("hf_repo") or model_id,
        force=_wants_force(body if isinstance(body, dict) else {}),
        catalog_size_gb=entry.get("size_gb"))
    if refusal is not None:
        return refusal

    job_id = str(uuid.uuid4())
    jobs: dict = request.app["download_jobs"]
    jobs[job_id] = {"model_id": model_id, "status": "downloading", "error": None, "finished_at": None}

    _cleanup_old_jobs(jobs)

    loop = asyncio.get_event_loop()
    loop.create_task(_run_download(manager, model_id, job_id, jobs))

    return web.json_response(
        {"job_id": job_id, "model_id": model_id, "status": "downloading"},
        status=202,
    )


async def handle_download_repo(request: web.Request) -> web.Response:
    """POST /api/models/download-repo -- download any HF repo directly."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    hf_repo = body.get("hf_repo") or body.get("repo") or body.get("model_id") or ""
    hf_repo = hf_repo.strip()
    if not hf_repo or "/" not in hf_repo:
        return web.json_response({"error": "hf_repo required (e.g. meta-llama/Llama-3.2-3B-Instruct)"}, status=400)

    # An arbitrary repo is the case the fit check exists for: nothing in the
    # catalog bounds its size, and a 400 GB pull onto a node with 70 GB free is
    # a wasted hour that ends in a partial blob cache.
    refusal = await _refuse_if_it_will_not_fit(request, hf_repo,
                                              force=_wants_force(body))
    if refusal is not None:
        return refusal

    manager: ModelManager = request.app["model_manager"]
    job_id = str(uuid.uuid4())
    jobs: dict = request.app["download_jobs"]
    jobs[job_id] = {"model_id": hf_repo, "status": "downloading", "error": None, "finished_at": None}
    _cleanup_old_jobs(jobs)

    loop = asyncio.get_event_loop()
    loop.create_task(_run_download_repo(manager, hf_repo, job_id, jobs))

    return web.json_response(
        {"job_id": job_id, "hf_repo": hf_repo, "status": "downloading"},
        status=202,
    )


async def handle_list_downloaded(request: web.Request) -> web.Response:
    """GET /api/models/downloaded — list all models present on disk."""
    manager: ModelManager = request.app["model_manager"]
    try:
        models = manager.list_downloaded()
    except Exception as exc:
        return web.json_response({"error": str(exc), "models": []}, status=500)
    return web.json_response({"models": models})


async def handle_download_status(request: web.Request) -> web.Response:
    """GET /api/models/download/status?job_id=... — returns job status."""
    job_id = request.query.get("job_id", "").strip()
    jobs: dict = request.app["download_jobs"]
    if job_id and job_id in jobs:
        payload = dict(jobs[job_id])
        payload["job_id"] = job_id
        return web.json_response(payload)
    return web.json_response({"error": "job not found", "status": "unknown"}, status=404)


async def handle_delete_repo(request: web.Request) -> web.Response:
    """POST /api/models/delete-repo — delete any downloaded hf_repo directory."""
    import shutil
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    hf_repo = (body.get("hf_repo") or body.get("model_id") or "").strip()
    if not hf_repo or "/" not in hf_repo:
        return web.json_response({"error": "hf_repo required"}, status=400)

    manager: ModelManager = request.app["model_manager"]
    slug = hf_repo.replace("/", "--")
    target = Path(manager.models_dir) / slug

    if not target.exists() or not target.is_dir():
        return web.json_response({"error": f"Model not downloaded: {hf_repo}"}, status=404)

    # Safety: ensure we're deleting inside models_dir
    try:
        target_resolved = target.resolve()
        models_resolved = Path(manager.models_dir).resolve()
        if not str(target_resolved).startswith(str(models_resolved)):
            return web.json_response({"error": "refusing to delete outside models_dir"}, status=400)
    except Exception:
        return web.json_response({"error": "path resolution failed"}, status=500)

    try:
        size_gb = manager._dir_size_gb(target)
        shutil.rmtree(target)
        return web.json_response({
            "status": "deleted",
            "hf_repo": hf_repo,
            "freed_gb": round(size_gb, 2),
        })
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


async def handle_active_downloads(request: web.Request) -> web.Response:
    """GET /api/models/downloads/active — list all download jobs (running + recently finished)."""
    jobs: dict = request.app["download_jobs"]
    active = []
    for job_id, job in jobs.items():
        entry = dict(job)
        entry["job_id"] = job_id
        active.append(entry)
    return web.json_response({"jobs": active, "count": len(active)})


def _get_repo_total_bytes(hf_repo: str) -> int:
    """Query HF API for the total byte size of all files in a repo."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info(hf_repo, files_metadata=True)
        total = 0
        siblings = getattr(info, "siblings", []) or []
        for f in siblings:
            size = getattr(f, "size", None) or getattr(f, "lfs", {}) or 0
            if isinstance(size, dict):
                size = size.get("size", 0) or 0
            if isinstance(size, (int, float)) and size > 0:
                total += int(size)
        return total
    except Exception:
        return 0


def _get_dir_bytes(path: Path) -> int:
    """Sum of sizes of all regular files under path (follows symlinks for LFS)."""
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file() or (p.is_symlink() and p.exists()):
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except Exception:
        pass
    return total


class _DownloadCancelled(Exception):
    pass


async def handle_cancel_download(request: web.Request) -> web.Response:
    """POST /api/models/download-cancel -- cancel an in-progress download."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    job_id = (body.get("job_id") or "").strip()
    jobs: dict = request.app["download_jobs"]

    if not job_id or job_id not in jobs:
        return web.json_response({"error": "job not found"}, status=404)

    job = jobs[job_id]
    if job.get("status") != "downloading":
        return web.json_response(
            {"error": f"job is {job.get('status')}, not downloading"}, status=409
        )

    # Signal the download thread to stop
    job["_cancel"] = True
    job["status"] = "cancelling"
    return web.json_response({"job_id": job_id, "status": "cancelling"})


async def _run_download_repo(manager: "ModelManager", hf_repo: str, job_id: str, jobs: dict) -> None:
    """Download an arbitrary HF repo that may not be in our catalog."""
    import shutil
    loop = asyncio.get_event_loop()
    target = Path(manager.models_dir) / hf_repo.replace("/", "--")
    target.mkdir(parents=True, exist_ok=True)

    # Fetch total size in background (don't block start)
    total_bytes = await loop.run_in_executor(None, _get_repo_total_bytes, hf_repo)
    if total_bytes > 0:
        # Remember it: the next fit check on this repo, including the one a
        # launch makes without touching the Hub, then has a real number.
        model_fit.remember_repo_size(
            hf_repo, model_fit.RepoSize(total_bytes=int(total_bytes),
                                        source="hub_files"))
    jobs[job_id]["total_bytes"] = total_bytes
    jobs[job_id]["downloaded_bytes"] = 0
    jobs[job_id]["target_dir"] = str(target)

    # Poller task: watch directory size and update job progress
    poll_stop = asyncio.Event()

    async def _poll_progress():
        while not poll_stop.is_set():
            try:
                downloaded = await loop.run_in_executor(None, _get_dir_bytes, target)
                jobs[job_id]["downloaded_bytes"] = downloaded
                if total_bytes > 0:
                    jobs[job_id]["progress"] = min(100.0, (downloaded / total_bytes) * 100)
                else:
                    jobs[job_id]["progress"] = None
            except Exception:
                pass
            try:
                await asyncio.wait_for(poll_stop.wait(), timeout=1.5)
            except asyncio.TimeoutError:
                pass

    poll_task = loop.create_task(_poll_progress())

    try:
        def _do_download():
            # File-by-file (NOT snapshot_download) so a cancel is real: snapshot_download
            # has no cancel hook, so the old _cancel flag was a no-op and a cancelled
            # pull ran to completion. We resolve the repo ONCE, then fetch each file via
            # hf_hub_download into the SAME local_dir — snapshot_download internally
            # does exactly this per file, so the on-disk layout is identical (load-
            # bearing for on-disk-serve) — checking the cancel flag between files.
            #
            # Two properties we must preserve from snapshot_download (it gave them for
            # free; a naive loop drops both):
            #   1. Commit pinning. snapshot_download resolves repo_info.sha ONCE and
            #      passes revision=<sha> to every per-file fetch, so the whole snapshot
            #      comes from one commit even if `main` moves mid-pull (these are often
            #      actively-maintained community/quant repos, and a big pull is a wide
            #      window). We do the same: repo_info once → pin its .sha → thread it
            #      through every hf_hub_download. Without this, a push mid-download could
            #      404 on a renamed file or silently mix artifacts from two commits.
            #   2. Intra-repo parallelism. snapshot_download fetches with a worker pool
            #      (max_workers). Serial file-by-file would multiply wall-clock on the
            #      multi-shard, multi-hundred-GB repos this endpoint targets. We keep a
            #      bounded ThreadPoolExecutor (_download_max_workers), still checking the
            #      cancel flag before each file is submitted.
            import os
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from huggingface_hub import HfApi, hf_hub_download

            from ainode.models.registry import _download_max_workers

            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

            def _check_cancel():
                if jobs.get(job_id, {}).get("_cancel"):
                    raise _DownloadCancelled("Download cancelled by user")

            _check_cancel()
            # Resolve files + pinned commit in one call so every file is from one commit.
            info = HfApi(token=token).repo_info(repo_id=hf_repo)
            revision = info.sha
            files = [s.rfilename for s in (info.siblings or [])]

            def _download_one(rfilename: str) -> None:
                _check_cancel()  # stop between files — no mid-file hook exists
                hf_hub_download(
                    repo_id=hf_repo,
                    filename=rfilename,
                    revision=revision,  # pin to the commit resolved above
                    local_dir=str(target),
                    token=token,
                )

            _check_cancel()
            max_workers = max(1, min(_download_max_workers(), len(files) or 1))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_download_one, f) for f in files]
                try:
                    for fut in as_completed(futures):
                        fut.result()  # propagate first error (incl. _DownloadCancelled)
                except BaseException:
                    for fut in futures:
                        fut.cancel()  # drop not-yet-started files; in-flight finish
                    raise
            _check_cancel()  # a cancel arriving after the last file still counts
            return str(target)

        # Serialize downloads (one fat pull at a time) so two concurrent model
        # downloads can't gang up on the link — what stacked Nemotron+MiniMax did.
        async with _download_gate():
            await loop.run_in_executor(None, _do_download)
        # Stop the poller BEFORE writing the final numbers. Its last directory
        # read can still be in flight in the executor, and on a slow box it lands
        # after this block and overwrites 100% with a stale partial (CI saw 0.15%).
        poll_stop.set()
        try:
            await poll_task
        except Exception:
            pass
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["finished_at"] = time.time()
        jobs[job_id]["progress"] = 100.0
        if total_bytes > 0:
            jobs[job_id]["downloaded_bytes"] = total_bytes
    except _DownloadCancelled:
        jobs[job_id]["status"] = "cancelled"
        jobs[job_id]["finished_at"] = time.time()
        # Clean up partial download
        try:
            if target.exists():
                shutil.rmtree(target)
        except Exception:
            pass
    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)
        jobs[job_id]["finished_at"] = time.time()
    finally:
        poll_stop.set()
        try:
            await poll_task
        except Exception:
            pass


async def handle_delete_model(request: web.Request) -> web.Response:
    """DELETE /api/models/:model_id -- delete a downloaded model."""
    model_id = request.match_info["model_id"]
    manager: ModelManager = request.app["model_manager"]

    if manager.get_model_info(model_id) is None:
        return web.json_response(
            {"error": f"Model '{model_id}' not found in catalog"},
            status=404,
        )

    try:
        deleted = manager.delete_model(model_id)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)

    if deleted:
        return web.json_response({"status": "deleted", "model_id": model_id})
    return web.json_response(
        {"error": f"Model '{model_id}' is not downloaded"},
        status=404,
    )


async def handle_search_models(request: web.Request) -> web.Response:
    """Search HuggingFace Hub for models."""
    manager: ModelManager = request.app["model_manager"]
    query = request.query.get("q", "").strip()
    if not query:
        return web.json_response({"models": []})
    limit = int(request.query.get("limit", "30"))
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, manager.search_huggingface, query, limit)
    return web.json_response({"models": results, "query": query})


async def handle_recommended(request: web.Request) -> web.Response:
    """GET /api/models/recommended -- models that fit this node's GPU."""
    gpu = detect_gpu()
    if gpu is None:
        return web.json_response(
            {"error": "No GPU detected", "models": []},
            status=200,
        )

    gpu_memory_gb = gpu.memory_total_mb / 1024
    manager: ModelManager = request.app["model_manager"]
    loop = asyncio.get_event_loop()
    models = await loop.run_in_executor(None, manager.recommend_for_gpu, gpu_memory_gb)
    return web.json_response({
        "gpu_memory_gb": round(gpu_memory_gb, 1),
        "models": models,
    })


async def handle_trending_models(request: web.Request) -> web.Response:
    """GET /api/models/trending -- HuggingFace trending models."""
    manager: ModelManager = request.app["model_manager"]
    loop = asyncio.get_event_loop()
    models = await loop.run_in_executor(
        None, lambda: manager._aggregator.fetch_trending(30)
    )
    payload = [m.to_dict() for m in models]
    return web.json_response({
        "models": payload,
        "source": "trending",
        "count": len(payload),
    })


async def handle_latest_models(request: web.Request) -> web.Response:
    """GET /api/models/latest -- most recently released HF models."""
    manager: ModelManager = request.app["model_manager"]
    loop = asyncio.get_event_loop()
    models = await loop.run_in_executor(
        None, lambda: manager._aggregator.fetch_latest(30)
    )
    payload = [m.to_dict() for m in models]
    return web.json_response({
        "models": payload,
        "source": "latest",
        "count": len(payload),
    })


async def handle_openrouter_models(request: web.Request) -> web.Response:
    """GET /api/models/openrouter -- OpenRouter popular models."""
    manager: ModelManager = request.app["model_manager"]
    loop = asyncio.get_event_loop()
    models = await loop.run_in_executor(
        None, lambda: manager._aggregator.fetch_openrouter_popular(30)
    )
    payload = [m.to_dict() for m in models]
    return web.json_response({
        "models": payload,
        "source": "openrouter",
        "count": len(payload),
    })


async def handle_ollama_models(request: web.Request) -> web.Response:
    """GET /api/models/ollama -- Ollama library models."""
    manager: ModelManager = request.app["model_manager"]
    loop = asyncio.get_event_loop()
    models = await loop.run_in_executor(
        None, lambda: manager._aggregator.fetch_ollama_library(30)
    )
    payload = [m.to_dict() for m in models]
    return web.json_response({
        "models": payload,
        "source": "ollama",
        "count": len(payload),
    })


# -- Background download task -------------------------------------------------

_DOWNLOAD_JOB_MAX_AGE = 3600

def _cleanup_old_jobs(jobs: dict) -> None:
    now = time.time()
    to_remove = [jid for jid, info in jobs.items() if info.get("finished_at") is not None and (now - info["finished_at"]) > _DOWNLOAD_JOB_MAX_AGE]
    for jid in to_remove:
        del jobs[jid]


async def _run_download(
    manager: ModelManager,
    model_id: str,
    job_id: str,
    jobs: dict,
) -> None:
    """Run model download in a thread so we don't block the event loop."""
    loop = asyncio.get_event_loop()
    try:
        async with _download_gate():  # serialize with other downloads
            await loop.run_in_executor(None, manager.download_model, model_id)
        jobs[job_id]["status"] = "complete"
    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)
    finally:
        jobs[job_id]["finished_at"] = time.time()
