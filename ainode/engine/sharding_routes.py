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
    """POST /api/sharding/launch — launch a model distributed across the cluster.

    JSON body:
        model (required): HuggingFace model ID
        strategy (optional): auto | tensor_parallel | pipeline_parallel
        min_nodes (optional, default 1): nodes to span

    When min_nodes > 1, this endpoint flips the local engine into head mode:
    it discovers member nodes from the cluster state, takes their peer IPs
    from UDP recvfrom, writes them into config, stops the current (solo)
    engine, and starts the distributed engine via the configured backend
    (eugr's launch-cluster.sh, or NvidiaBackend's run_cluster path).
    When min_nodes == 1, it falls through to the existing single-node load
    path (/api/models/load).
    """
    from ainode.core.config import NodeConfig
    from ainode.engine.backends import get_backend

    global _active_sharding

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    model = body.get("model")
    if not model:
        return web.json_response({"error": "model field required"}, status=400)

    try:
        min_nodes = int(body.get("min_nodes", 1) or 1)
    except (TypeError, ValueError):
        min_nodes = 1

    strategy_str = body.get("strategy", "tensor_parallel")
    # We accept but don't gate on strategy here — vLLM picks TP vs PP via
    # CLI args in the launch script; for now any min_nodes > 1 triggers TP.

    cluster: ClusterState = request.app["cluster_state"]
    config: NodeConfig = request.app["config"]
    engine = request.app.get("engine")

    if min_nodes <= 1:
        # Delegate to the single-node load path so behaviour stays
        # consistent with what the UI called before.
        request._rewritten_body = {"model": model}  # for observability
        from ainode.models.api_routes import handle_model_load  # lazy import
        # Re-inject body so handle_model_load can read it
        class _ReqShim:
            def __init__(self, orig, body): self._o = orig; self._b = body
            def __getattr__(self, k): return getattr(self._o, k)
            async def json(self): return self._b
        shim = _ReqShim(request, {"model": model})
        return await handle_model_load(shim)

    # Distributed path. Find discovered member nodes on our cluster_interface
    # subnet and use their peer IPs as authoritative addresses.
    members = [
        n for n in cluster.members()
        if getattr(n, "distributed_mode", "solo") == "member"
        and (n.status.value if hasattr(n.status, "value") else str(n.status)) in ("online", "member-ready", "serving")
    ]
    peer_ips = [n.peer_ip for n in members if getattr(n, "peer_ip", None)]
    if len(peer_ips) < (min_nodes - 1):
        return web.json_response({
            "error": (
                f"Requested min_nodes={min_nodes} but only {len(peer_ips)+1} "
                f"node(s) available (1 head + {len(peer_ips)} member(s) with "
                f"known peer IPs). Ensure member nodes are running and "
                f"broadcasting on the cluster_interface subnet."
            ),
            "discovered_members": [
                {"node_id": n.node_id, "node_name": n.node_name,
                 "peer_ip": getattr(n, "peer_ip", None),
                 "status": n.status.value if hasattr(n.status, "value") else str(n.status)}
                for n in members
            ],
        }, status=422)

    chosen_peers = peer_ips[: min_nodes - 1]

    # Flip local config to head mode and persist.
    config.model = model
    config.distributed_mode = "head"
    config.peer_ips = chosen_peers
    try:
        config.save()
    except Exception:
        logger.exception("Failed to persist config.json before distributed launch")

    # Hot-swap the engine: stop the solo vLLM (if running) and build a
    # fresh DockerEngine bound to the updated config for the distributed
    # launch. We don't replace request.app["engine"] until success so the
    # status endpoint keeps working during the switch.
    try:
        if engine is not None and engine.is_running():
            engine.stop()
    except Exception:
        logger.exception("Engine.stop() failed during distributed hot-swap")

    new_engine = get_backend(config)
    try:
        started = new_engine.start_distributed()
    except Exception as exc:
        logger.exception("start_distributed raised")
        return web.json_response({"error": f"Distributed launch failed: {exc}"}, status=500)

    if not started:
        return web.json_response({"error": "Distributed launch returned False"}, status=500)

    request.app["engine"] = new_engine

    return web.json_response({
        "status": "launching",
        "model": model,
        "distributed_mode": "head",
        "peer_ips": chosen_peers,
        "tensor_parallel_size": 1 + len(chosen_peers),
        "strategy": strategy_str,
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
