"""API routes for model sharding — plan, launch, and monitor distributed inference."""

from __future__ import annotations

import json
import logging
from typing import Optional

from aiohttp import web

from ainode.discovery.cluster import ClusterState
from ainode.engine.sharding import ShardingPlanner, ShardingStrategy, ShardingConfig
from ainode.engine.ray_setup import get_ray_status, start_ray_head, join_ray_cluster, stop_ray

logger = logging.getLogger(__name__)

# Module-level state for active sharding session
_active_sharding: Optional[ShardingConfig] = None


def register_sharding_routes(app: web.Application) -> None:
    """Register sharding API endpoints on the aiohttp app."""
    app.router.add_get("/api/sharding/plan", handle_sharding_plan)
    app.router.add_post("/api/sharding/launch", handle_sharding_launch)
    app.router.add_get("/api/sharding/status", handle_sharding_status)


async def handle_sharding_plan(request: web.Request) -> web.Response:
    """GET /api/sharding/plan?model=X — preview sharding plan for a model.

    Query params:
        model (required): HuggingFace model ID
        strategy (optional): tensor_parallel, pipeline_parallel, auto (default: auto)
    """
    model = request.query.get("model")
    if not model:
        return web.json_response({"error": "model parameter required"}, status=400)

    strategy_str = request.query.get("strategy", "auto")
    try:
        strategy = ShardingStrategy(strategy_str)
    except ValueError:
        return web.json_response(
            {"error": f"Invalid strategy: {strategy_str}. Use: auto, tensor_parallel, pipeline_parallel"},
            status=400,
        )

    cluster: ClusterState = request.app["cluster_state"]
    planner = ShardingPlanner()

    try:
        config = planner.plan_sharding(model, cluster, strategy)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=422)

    return web.json_response({
        "plan": config.to_dict(),
        "can_fit": planner.can_fit_model(model, cluster),
        "cluster_nodes": len(cluster.get_nodes(include_offline=False)),
    })


async def handle_sharding_launch(request: web.Request) -> web.Response:
    """POST /api/sharding/launch — launch a model with sharding across the cluster.

    JSON body:
        model (required): HuggingFace model ID
        strategy (optional): auto | tensor_parallel | pipeline_parallel
    """
    global _active_sharding

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    model = body.get("model")
    if not model:
        return web.json_response({"error": "model field required"}, status=400)

    strategy_str = body.get("strategy", "auto")
    try:
        strategy = ShardingStrategy(strategy_str)
    except ValueError:
        return web.json_response({"error": f"Invalid strategy: {strategy_str}"}, status=400)

    cluster: ClusterState = request.app["cluster_state"]
    engine = request.app.get("engine")

    if engine is None:
        return web.json_response({"error": "Engine not initialized"}, status=503)

    planner = ShardingPlanner()

    try:
        config = planner.plan_sharding(model, cluster, strategy)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=422)

    # If distributed, set up Ray cluster
    if config.is_distributed:
        try:
            head_addr = start_ray_head()
            config.ray_head_address = head_addr
            logger.info("Ray head started at %s, workers should join", head_addr)
        except RuntimeError as exc:
            return web.json_response(
                {"error": f"Failed to start Ray cluster: {exc}"}, status=500
            )

    # Stop existing engine if running
    if engine.is_running():
        engine.stop()

    # Launch with sharding config
    success = engine.launch_distributed(config)
    if not success:
        return web.json_response({"error": "Failed to launch distributed engine"}, status=500)

    _active_sharding = config

    return web.json_response({
        "status": "launching",
        "plan": config.to_dict(),
    })


async def handle_sharding_status(request: web.Request) -> web.Response:
    """GET /api/sharding/status — current sharding state and Ray cluster health."""
    ray_status = get_ray_status()
    engine = request.app.get("engine")

    engine_running = False
    engine_ready = False
    if engine is not None:
        engine_running = engine.is_running()
        engine_ready = getattr(engine, "ready", False)

    result = {
        "active_sharding": _active_sharding.to_dict() if _active_sharding else None,
        "engine_running": engine_running,
        "engine_ready": engine_ready,
        "ray": ray_status.to_dict(),
    }

    return web.json_response(result)
