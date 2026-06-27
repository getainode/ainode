"""API route handlers for model management."""

from __future__ import annotations

import asyncio
import aiohttp
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from aiohttp import web

from ainode.core.gpu import detect_gpu
from ainode.models.registry import ModelManager

logger = logging.getLogger(__name__)

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


def register_model_routes(app: web.Application, manager: Optional[ModelManager] = None) -> None:
    """Register model management routes on the aiohttp app."""
    if manager is None:
        manager = ModelManager()

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
    app.router.add_get("/api/models/{model_id}", handle_get_model)
    app.router.add_get("/api/models/downloaded", handle_list_downloaded)
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
        entries.append({
            "model": inst.record.model,
            "gpu_memory_utilization": getattr(cfg, "gpu_memory_utilization", None),
        })
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


def append_solo_instance(app, model: str, gmu=None, *, persist: bool = True) -> dict:
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

    # Re-loading a model already up replaces THAT instance (stop it first), not
    # the whole node — other stacked instances are untouched.
    existing = manager.by_model(model)
    replaced_primary = existing is not None and app.get("engine") is existing.backend
    if existing is not None:
        try:
            existing.backend.stop()
        except Exception:
            pass
        manager.remove(existing.record.instance_id)

    is_primary = manager.is_empty()
    port = manager.allocate_port()
    name_token = "" if port == config.api_port else str(port)  # primary keeps legacy names
    instance_id = f"{config.node_id or 'head'}:{model}"

    inst_config = replace(config, model=model, distributed_mode="solo",
                          peer_ips=[], api_port=port)
    if gmu is not None:
        inst_config = replace(inst_config, gpu_memory_utilization=gmu)

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
        return {"ok": False, "error": "Failed to launch engine", "status": 500}

    manager.add(InstanceRecord(
        instance_id=instance_id, model=model, head_node_id=config.node_id or "head",
        peer_ips=[], api_port=port, tensor_parallel_size=1, status="starting"), backend)

    if is_primary:
        # Back-compat: the proxy/status path reads app["config"] + app["engine"].
        config.model = model
        config.distributed_mode = "solo"
        config.peer_ips = []
        if gmu is not None:
            config.gpu_memory_utilization = gmu
        try:
            config.save()
        except Exception:
            pass
        app["engine"] = backend
    elif replaced_primary:
        # Reloaded the primary while a stack exists: keep app["engine"] on the live
        # backend (not the stopped old one) so status/proxy don't dangle.
        config.model = model
        try:
            config.save()
        except Exception:
            pass
        app["engine"] = backend

    if persist:
        save_instance_manifest(app)
    return {"ok": True, "model": model, "instance_id": instance_id,
            "api_port": port, "stacked": not is_primary}


async def _wait_port_ready(port: int, timeout: float = 300.0) -> bool:
    """Poll http://localhost:<port>/v1/models until it serves (200) or times out."""
    import urllib.request
    loop = asyncio.get_event_loop()

    def _probe() -> bool:
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=3) as r:
                return getattr(r, "status", r.getcode()) == 200
        except Exception:
            return False

    for _ in range(max(1, int(timeout // 3))):
        if await loop.run_in_executor(None, _probe):
            return True
        await asyncio.sleep(3)
    return False


async def replay_instances_on_startup(app) -> None:
    """Always-on: after boot, re-load the persisted solo instance set so a node
    restart brings every previously-loaded model back with no manual step. The
    boot engine claims the primary (config.model); this replays the stacked rest.

    Loads are SERIALIZED — the boot primary must serve before the first stack, and
    each stacked model must bind before the next — because concurrent vLLM loads on
    a unified-memory node race for memory and one gets OOM-killed."""
    config = app.get("config")
    if config is None:
        return
    entries = load_instance_manifest()
    if not entries:
        return
    await asyncio.sleep(10)

    # Orphan sweep: stacked vLLM containers (ainode-vllm-node-solo-<port>) outlive
    # the orchestrator restart, but the in-memory manager does not — so a surviving
    # suffixed container is an orphan the replay is about to relaunch. Remove them
    # first or the relaunch's `--name` collides (Conflict). The primary
    # `ainode-vllm-node-solo` (no suffix) is owned/pre-cleaned by the boot engine.
    try:
        import subprocess
        ps = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "name=ainode-vllm-node-solo-"],
            capture_output=True, text=True, timeout=20)
        ids = [i for i in ps.stdout.split() if i]
        if ids:
            subprocess.run(["docker", "rm", "-f", *ids],
                           capture_output=True, text=True, timeout=60)
    except Exception:
        logger.exception("orphan container sweep failed")

    # Wait for the boot primary to actually serve before stacking on top of it.
    await _wait_port_ready(config.api_port, timeout=300)

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
                lambda mm=m, g=e.get("gpu_memory_utilization"): append_solo_instance(app, mm, g, persist=False),
            )
            have.add(m)
            # Serialize: let this model bind before launching the next one.
            if isinstance(res, dict) and res.get("ok") and res.get("api_port"):
                await _wait_port_ready(res["api_port"], timeout=300)
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
    if sharding_config is not None:
        if engine is None:
            # Lazy-create via get_backend (honors engine_backend=nvidia, not the
            # legacy host-venv VLLMEngine) for a node booted without an engine.
            try:
                if config is None:
                    return web.json_response({"error": "Engine not initialized"}, status=503)
                from ainode.engine.backends import get_backend
                engine = get_backend(config)
                request.app["engine"] = engine
            except Exception as exc:
                return web.json_response({"error": f"Engine unavailable: {exc}"}, status=503)
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
            return web.json_response({"error": "Failed to launch engine"}, status=500)
        return web.json_response({
            "status": "launching", "model": model,
            "distributed": True, "plan": sharding_config.to_dict(),
        })

    # --- Solo path: APPEND an instance via the InstanceManager ------------------
    # A solo load no longer REPLACES the running model. Each model stacks (own
    # container/port/config snapshot); the set is persisted for auto-replay on
    # restart. Shared with the startup replay via append_solo_instance().
    if config is None:
        return web.json_response({"error": "Engine not initialized"}, status=503)

    result = append_solo_instance(request.app, model, gmu)
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
    """POST /api/models/unload -- stop the current model (solo or distributed).

    The dashboard DELETE button hits this endpoint. Calls engine.stop(), which
    for EugrBackend tears down eugr's launch-cluster.sh, and for NvidiaBackend
    stops the head container + fan-outs `docker stop` to peer workers over SSH.

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
    stopped = False
    errors = []

    # Instance-aware unload: stop the ONE instance serving `model`, leaving any
    # other stacked instances on this node serving. Falls through to the legacy
    # singleton teardown when no manager/model match (back-compat).
    manager = request.app.get("instances")
    model = (body.get("model") or "").strip() if isinstance(body, dict) else ""
    if manager is not None and model:
        inst = manager.by_model(model)
        if inst is not None:
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
            return web.json_response({
                "stopped": True, "model": model,
                "instance_id": inst.record.instance_id,
                "remaining": len(manager.instances()),
                "errors": errors,
            })

    # Was THIS node actually serving the requested model? (back-compat: no model
    # given → stop whatever is local.) Only then is a local "stopped" truthful —
    # the old code returned stopped:true even when engine was None or serving a
    # different model, which is why the dashboard's Unload button silently no-op'd.
    served_here = engine is not None and (not model or getattr(config, "model", None) == model)
    if served_here:
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

    # Not serving here. `fanout=0` marks a fan-out child — stop, don't recurse
    # (prevents an unload broadcast storm). Otherwise the model lives on another
    # node: fan the unload out to online peers (each peer's local unload is
    # idempotent) so the head can unload a remote-node instance.
    if request.query.get("fanout") == "0":
        return web.json_response({"stopped": False, "model": model, "scope": "local-miss", "errors": errors})

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
    """GET /api/models -- list the dynamic catalog with download status."""
    manager: ModelManager = request.app["model_manager"]
    loop = asyncio.get_event_loop()
    models = await loop.run_in_executor(None, manager.list_available)
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

    if manager.get_model_info(model_id) is None:
        return web.json_response(
            {"error": f"Model '{model_id}' not found in catalog"},
            status=404,
        )

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
            from huggingface_hub import snapshot_download

            def _progress_callback(info):
                # Check cancel flag on every chunk callback
                if jobs.get(job_id, {}).get("_cancel"):
                    raise _DownloadCancelled("Download cancelled by user")

            from ainode.models.registry import _download_max_workers
            try:
                snapshot_download(
                    repo_id=hf_repo,
                    local_dir=str(target),
                    tqdm_class=None,
                    max_workers=_download_max_workers(),  # don't monopolise the uplink
                )
            except _DownloadCancelled:
                raise
            return str(target)

        # Serialize downloads (one fat pull at a time) so two concurrent model
        # downloads can't gang up on the link — what stacked Nemotron+MiniMax did.
        async with _download_gate():
            await loop.run_in_executor(None, _do_download)
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
