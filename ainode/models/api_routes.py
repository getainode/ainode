"""API route handlers for model management."""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Optional

from aiohttp import web

from ainode.core.gpu import detect_gpu
from ainode.models.registry import ModelManager


def register_model_routes(app: web.Application, manager: Optional[ModelManager] = None) -> None:
    """Register model management routes on the aiohttp app."""
    if manager is None:
        manager = ModelManager()

    app["model_manager"] = manager
    app["download_jobs"] = {}

    app.router.add_get("/api/models", handle_list_models)
    app.router.add_post("/api/models/refresh", handle_refresh_catalog)
    app.router.add_get("/api/models/recommended", handle_recommended)
    app.router.add_get("/api/models/search", handle_search_models)
    app.router.add_get("/api/models/trending", handle_trending_models)
    app.router.add_get("/api/models/latest", handle_latest_models)
    app.router.add_get("/api/models/openrouter", handle_openrouter_models)
    app.router.add_get("/api/models/ollama", handle_ollama_models)
    app.router.add_get("/api/models/{model_id}", handle_get_model)
    app.router.add_post("/api/models/download-repo", handle_download_repo)
    app.router.add_get("/api/models/download/status", handle_download_status)
    app.router.add_get("/api/models/downloads/active", handle_active_downloads)
    app.router.add_post("/api/models/delete-repo", handle_delete_repo)
    app.router.add_post("/api/models/{model_id}/download", handle_download_model)
    app.router.add_delete("/api/models/{model_id}", handle_delete_model)


# -- Handlers ------------------------------------------------------------------

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


async def _run_download_repo(manager: "ModelManager", hf_repo: str, job_id: str, jobs: dict) -> None:
    """Download an arbitrary HF repo that may not be in our catalog."""
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
            snapshot_download(repo_id=hf_repo, local_dir=str(target))
            return str(target)
        await loop.run_in_executor(None, _do_download)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["finished_at"] = time.time()
        jobs[job_id]["progress"] = 100.0
        if total_bytes > 0:
            jobs[job_id]["downloaded_bytes"] = total_bytes
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
        await loop.run_in_executor(None, manager.download_model, model_id)
        jobs[job_id]["status"] = "complete"
    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)
    finally:
        jobs[job_id]["finished_at"] = time.time()
