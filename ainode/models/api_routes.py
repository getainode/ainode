"""API route handlers for model management."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Optional

from aiohttp import web

from ainode.core.gpu import detect_gpu
from ainode.models.registry import ModelManager, MODEL_CATALOG


def register_model_routes(app: web.Application, manager: Optional[ModelManager] = None) -> None:
    """Register model management routes on the aiohttp app."""
    if manager is None:
        manager = ModelManager()

    app["model_manager"] = manager
    app["download_jobs"] = {}

    app.router.add_get("/api/models", handle_list_models)
    app.router.add_get("/api/models/recommended", handle_recommended)
    app.router.add_get("/api/models/{model_id}", handle_get_model)
    app.router.add_post("/api/models/{model_id}/download", handle_download_model)
    app.router.add_delete("/api/models/{model_id}", handle_delete_model)


# -- Handlers ------------------------------------------------------------------

async def handle_list_models(request: web.Request) -> web.Response:
    """GET /api/models -- list all catalog models with download status."""
    manager: ModelManager = request.app["model_manager"]
    models = manager.list_available()
    return web.json_response({"models": models})


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

    if model_id not in MODEL_CATALOG:
        return web.json_response(
            {"error": f"Model '{model_id}' not found in catalog"},
            status=404,
        )

    job_id = str(uuid.uuid4())
    jobs: dict = request.app["download_jobs"]
    jobs[job_id] = {"model_id": model_id, "status": "downloading", "error": None, "finished_at": None}

    # Clean up old completed/failed jobs
    _cleanup_old_jobs(jobs)

    loop = asyncio.get_event_loop()
    loop.create_task(_run_download(manager, model_id, job_id, jobs))

    return web.json_response(
        {"job_id": job_id, "model_id": model_id, "status": "downloading"},
        status=202,
    )


async def handle_delete_model(request: web.Request) -> web.Response:
    """DELETE /api/models/:model_id -- delete a downloaded model."""
    model_id = request.match_info["model_id"]
    manager: ModelManager = request.app["model_manager"]

    if model_id not in MODEL_CATALOG:
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
    models = manager.recommend_for_gpu(gpu_memory_gb)
    return web.json_response({
        "gpu_memory_gb": round(gpu_memory_gb, 1),
        "models": models,
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
