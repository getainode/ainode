"""Onboarding API route handlers for browser-based setup wizard."""

import socket
from aiohttp import web

from ainode.core.config import NodeConfig
from ainode.core.gpu import detect_gpu
from ainode.models.registry import MODEL_CATALOG


def register_onboarding_routes(app: web.Application) -> None:
    """Register all onboarding API routes on the given app."""
    app.router.add_get("/api/onboarding/status", handle_onboarding_status)
    app.router.add_get("/api/onboarding/suggestions", handle_onboarding_suggestions)
    app.router.add_post("/api/onboarding/complete", handle_onboarding_complete)


async def handle_onboarding_status(request: web.Request) -> web.Response:
    """Return whether this node has completed onboarding."""
    config: NodeConfig = request.app["config"]
    return web.json_response({
        "onboarded": config.onboarded,
        "needs_setup": not config.onboarded,
    })


async def handle_onboarding_suggestions(request: web.Request) -> web.Response:
    """Return hostname and recommended models for the detected GPU."""
    hostname = socket.gethostname()
    gpu = detect_gpu()

    gpu_memory_gb = 0.0
    gpu_info = None
    if gpu:
        gpu_memory_gb = gpu.memory_total_mb / 1024
        gpu_info = {
            "name": gpu.name,
            "memory_gb": round(gpu_memory_gb, 1),
            "unified_memory": gpu.unified_memory,
        }

    # Build model list with fit indicators
    models = []
    for model_id, info in MODEL_CATALOG.items():
        entry = info.to_dict()
        entry["fits_gpu"] = gpu_memory_gb >= info.min_memory_gb if gpu else False
        models.append(entry)

    # Sort: fitting models first (by size desc), then non-fitting (by size asc)
    models.sort(key=lambda m: (-m["fits_gpu"], -m["size_gb"] if m["fits_gpu"] else m["size_gb"]))

    return web.json_response({
        "hostname": hostname,
        "gpu": gpu_info,
        "models": models,
    })


async def handle_onboarding_complete(request: web.Request) -> web.Response:
    """Accept onboarding data and save config."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": "Invalid JSON body"}, status=400,
        )

    config: NodeConfig = request.app["config"]

    node_name = body.get("node_name", "").strip()
    if node_name:
        config.node_name = node_name

    model = body.get("model", "").strip()
    if model:
        config.model = model
        # Set quantization for AWQ models
        if "awq" in model.lower():
            config.quantization = "awq"
        else:
            config.quantization = None

    email = body.get("email", "").strip()
    if email:
        config.email = email

    config.onboarded = True
    config.save()

    return web.json_response({
        "status": "ok",
        "node_name": config.node_name,
        "model": config.model,
        "onboarded": True,
    })
