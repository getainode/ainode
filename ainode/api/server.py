"""AINode API proxy server — aiohttp app that serves the web UI and proxies to vLLM."""

import asyncio
import logging
import socket
import time
from dataclasses import asdict
from typing import Optional

import aiohttp
from aiohttp import web

from ainode.core.config import NodeConfig
from ainode.core.gpu import detect_gpu, GPUInfo
from ainode.web.serve import get_index_html, get_onboarding_html, get_static_path
from ainode.models.api_routes import register_model_routes
from ainode.onboarding.api_routes import register_onboarding_routes
from ainode.auth.middleware import AuthConfig, auth_middleware
from ainode.auth.api_routes import register_auth_routes
from ainode.metrics.collector import MetricsCollector
from ainode.metrics.api_routes import register_metrics_routes
from ainode.training.engine import TrainingManager
from ainode.training.api_routes import setup_training_routes
from ainode.datasets.manager import DatasetManager
from ainode.datasets.api_routes import setup_dataset_routes
from ainode.discovery.broadcast import (
    BroadcastSender,
    BroadcastListener,
    NodeAnnouncement,
)
from ainode.discovery.cluster import ClusterState
from ainode.engine.sharding_routes import register_sharding_routes
from ainode.engine.ray_autostart import (
    RayAutostartState,
    autostart_loop as _ray_autostart_loop,
)
from ainode.secrets import SecretsManager
from ainode.secrets.api_routes import register_secrets_routes
from ainode.embeddings.manager import EmbeddingManager
from ainode.embeddings.api_routes import register_embedding_routes
from ainode.api.server_routes import (
    register_server_routes,
    request_log_middleware,
    init_server_state,
)

from ainode import __version__

logger = logging.getLogger(__name__)

def create_app(
    config: Optional[NodeConfig] = None,
    engine=None,
) -> web.Application:
    """Create and return the aiohttp application.

    Parameters
    ----------
    config : NodeConfig
        Node configuration (defaults created if None).
    engine : VLLMEngine | None
        Optional engine instance for health/status queries.
    """
    if config is None:
        config = NodeConfig()

    auth_config = AuthConfig.load()

    app = web.Application(middlewares=[cors_middleware, request_log_middleware, auth_middleware])
    init_server_state(app)
    # Instantiate shared services
    collector = MetricsCollector()
    dataset_manager = DatasetManager()
    manager = TrainingManager(dataset_manager=dataset_manager)

    # Build local node announcement for discovery
    announcement = _build_announcement(config, engine)
    cluster = ClusterState(local_announcement=announcement)

    app["config"] = config
    app["auth_config"] = auth_config
    app["engine"] = engine
    app["start_time"] = time.time()
    app["client_session"] = None  # lazy-init in startup
    app["metrics_collector"] = collector
    app["training_manager"] = manager
    app["dataset_manager"] = dataset_manager
    app["cluster_state"] = cluster
    app["announcement"] = announcement
    app["broadcast_sender"] = None
    app["broadcast_listener"] = None
    app["secrets_manager"] = SecretsManager()
    app["embedding_manager"] = EmbeddingManager()
    app["ray_autostart_state"] = RayAutostartState()

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    app.router.add_get("/", handle_index)
    app.router.add_get("/onboarding", handle_onboarding)
    app.router.add_get("/api/health", handle_health)
    app.router.add_get("/api/status", handle_status)
    app.router.add_get("/api/nodes", handle_nodes)
    app.router.add_get("/api/cluster/info", handle_cluster_info)
    app.router.add_get("/api/cluster/resources", handle_cluster_resources)
    app.router.add_post("/api/cluster/role", handle_cluster_set_role)
    app.router.add_post("/api/cluster/id", handle_cluster_set_id)
    app.router.add_get("/api/config", handle_get_config)
    app.router.add_patch("/api/config", handle_patch_config)

    app.router.add_get("/v1/models", proxy_to_vllm)
    app.router.add_post("/v1/chat/completions", proxy_to_vllm)
    app.router.add_post("/v1/completions", proxy_to_vllm)

    register_model_routes(app)

    register_onboarding_routes(app)

    register_auth_routes(app)

    # --- Metrics routes ------------------------------------------------------
    register_metrics_routes(app, collector)

    # --- Training routes -----------------------------------------------------
    setup_training_routes(app, manager)

    # --- Dataset routes ------------------------------------------------------
    setup_dataset_routes(app, dataset_manager)

    # --- Sharding routes ----------------------------------------------------
    register_sharding_routes(app)

    # --- Secrets routes ------------------------------------------------------
    register_secrets_routes(app)

    # --- Embedding routes ----------------------------------------------------
    register_embedding_routes(app)

    # --- Server view routes --------------------------------------------------
    register_server_routes(app)

    app.router.add_static("/static", get_static_path(), name="static")

    return app


def _build_announcement(config: NodeConfig, engine=None) -> NodeAnnouncement:
    """Create a NodeAnnouncement from current node state."""
    gpu: Optional[GPUInfo] = detect_gpu()
    gpu_name = gpu.name if gpu else "CPU"
    gpu_memory_gb = round(gpu.memory_total_mb / 1024, 1) if gpu else 0.0
    unified_memory = gpu.unified_memory if gpu else False

    engine_ready = False
    if engine is not None:
        engine_ready = getattr(engine, "ready", False)

    distributed_mode = getattr(config, "distributed_mode", "solo") or "solo"
    # Member nodes report "member-ready" so the UI can distinguish them
    # from solo nodes that just haven't loaded a model yet.
    if distributed_mode == "member":
        status = "member-ready"
    elif engine_ready:
        status = "serving"
    else:
        status = "starting"

    # Head nodes with an active distributed instance advertise the instance id
    # and the peer node ids it spans so the UI can paint "DISTRIBUTED TP=N".
    # We use peer_ips as the peer key here — the UI resolves them against
    # discovered members.
    distributed_instance_id = None
    distributed_peers: list = []
    if distributed_mode == "head" and engine_ready:
        peer_ips = list(getattr(config, "peer_ips", []) or [])
        if peer_ips:
            distributed_instance_id = f"{config.node_id or 'head'}:{config.model}"
            distributed_peers = peer_ips

    return NodeAnnouncement(
        node_id=config.node_id or "unknown",
        node_name=config.node_name or socket.gethostname() or "unknown",
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        unified_memory=unified_memory,
        model=config.model or "" if distributed_mode != "member" else "",
        status=status,
        api_port=config.api_port,
        web_port=config.web_port,
        cluster_id=getattr(config, "cluster_id", "default"),
        role=getattr(config, "cluster_role", "auto"),
        is_master=False,  # runtime flag, updated by sync loop
        distributed_mode=distributed_mode,
        distributed_instance_id=distributed_instance_id,
        distributed_peers=distributed_peers,
    )


async def _on_startup(app: web.Application) -> None:
    app["client_session"] = aiohttp.ClientSession()

    config: NodeConfig = app["config"]
    announcement: NodeAnnouncement = app["announcement"]
    cluster: ClusterState = app["cluster_state"]

    if config.cluster_enabled:
        # Start broadcast sender
        sender = BroadcastSender(
            announcement=announcement,
            discovery_port=config.discovery_port,
        )
        await sender.start()
        app["broadcast_sender"] = sender
        logger.info("Discovery sender started on port %d", config.discovery_port)

        # Start broadcast listener
        def on_node_found(ann: NodeAnnouncement):
            logger.info("Discovered node %s (%s)", ann.node_id, ann.node_name)

        def on_node_lost(node_id: str):
            logger.info("Lost node %s", node_id)
            cluster.remove_node(node_id)

        listener = BroadcastListener(
            local_node_id=announcement.node_id,
            discovery_port=config.discovery_port,
            on_node_found=on_node_found,
            on_node_lost=on_node_lost,
        )
        await listener.start()
        app["broadcast_listener"] = listener
        logger.info("Discovery listener started on port %d", config.discovery_port)

        # Start a periodic task to sync listener registry into ClusterState
        app["_cluster_sync_task"] = asyncio.get_event_loop().create_task(
            _cluster_sync_loop(app)
        )

        # Kick off Ray autostart — master starts head, workers join once a
        # master is discovered. Gracefully no-ops if ray is not installed.
        def _get_master_address() -> Optional[str]:
            master = cluster.get_master()
            if master is None:
                return None
            # Same node → no remote address needed
            if master.node_id == announcement.node_id:
                return None
            # node_name can be None/"unknown" (no hostname resolution yet); the
            # Ray join would hang for 60s on a bogus address and block the
            # asyncio event loop. Skip until we have real peer IP plumbing.
            name = master.node_name
            if not name or name in ("unknown", "localhost", "None"):
                return None
            return f"{name}:6379"

        app["_ray_autostart_task"] = asyncio.get_event_loop().create_task(
            _ray_autostart_loop(
                cluster_state=cluster,
                get_master_address=_get_master_address,
                state=app["ray_autostart_state"],
            )
        )


async def _cluster_sync_loop(app: web.Application) -> None:
    """Periodically sync the listener registry into ClusterState."""
    try:
        while True:
            await asyncio.sleep(5)
            listener: Optional[BroadcastListener] = app.get("broadcast_listener")
            cluster: ClusterState = app["cluster_state"]
            if listener:
                cluster.update_from_discovered(listener.registry)
                # Update sender announcement with current engine status + master flag
                sender: Optional[BroadcastSender] = app.get("broadcast_sender")
                engine = app.get("engine")
                config: NodeConfig = app["config"]
                is_master = cluster.is_master_of_cluster()
                updates: dict = {
                    "is_master": is_master,
                    "cluster_id": getattr(config, "cluster_id", "default"),
                    "role": getattr(config, "cluster_role", "auto"),
                    "distributed_mode": getattr(config, "distributed_mode", "solo") or "solo",
                }
                dmode = updates["distributed_mode"]
                engine_ready = bool(engine is not None and getattr(engine, "ready", False))
                if dmode == "member":
                    updates["status"] = "member-ready"
                elif engine is not None:
                    updates["status"] = "serving" if engine_ready else "starting"

                # Advertise distributed instance metadata once the head's
                # sharded engine is serving — the UI uses this to render
                # "DISTRIBUTED TP=N across X nodes".
                if dmode == "head" and engine_ready:
                    peer_ips = list(getattr(config, "peer_ips", []) or [])
                    if peer_ips:
                        updates["distributed_instance_id"] = f"{config.node_id or 'head'}:{config.model}"
                        updates["distributed_peers"] = peer_ips
                    else:
                        updates["distributed_instance_id"] = None
                        updates["distributed_peers"] = []
                elif dmode != "head":
                    updates["distributed_instance_id"] = None
                    updates["distributed_peers"] = []
                if sender:
                    sender.update_announcement(**updates)
                    # Keep the app-level announcement in sync so /api/status sees fresh values
                    for k, v in updates.items():
                        if hasattr(sender.announcement, k):
                            setattr(sender.announcement, k, v)
    except asyncio.CancelledError:
        pass


async def _on_cleanup(app: web.Application) -> None:
    # Stop cluster sync task
    sync_task = app.get("_cluster_sync_task")
    if sync_task:
        sync_task.cancel()
        try:
            await sync_task
        except asyncio.CancelledError:
            pass

    # Stop Ray autostart task
    ray_task = app.get("_ray_autostart_task")
    if ray_task:
        ray_task.cancel()
        try:
            await ray_task
        except asyncio.CancelledError:
            pass

    # Stop discovery sender and listener
    sender: Optional[BroadcastSender] = app.get("broadcast_sender")
    if sender:
        await sender.stop()
        logger.info("Discovery sender stopped")

    listener: Optional[BroadcastListener] = app.get("broadcast_listener")
    if listener:
        await listener.stop()
        logger.info("Discovery listener stopped")

    session: Optional[aiohttp.ClientSession] = app.get("client_session")
    if session and not session.closed:
        await session.close()

@web.middleware
async def cors_middleware(request: web.Request, handler):
    """Add CORS headers to every response so the dashboard can fetch freely."""
    if request.method == "OPTIONS":
        resp = web.Response(status=204)
    else:
        try:
            resp = await handler(request)
        except web.HTTPException as exc:
            resp = exc

    origin = request.headers.get("Origin", "")
    allowed = origin if origin.startswith(("http://localhost", "http://127.0.0.1")) else ""
    resp.headers["Access-Control-Allow-Origin"] = allowed
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return resp

async def handle_index(request: web.Request) -> web.Response:
    """Serve the dashboard, or redirect to onboarding if not set up."""
    config: NodeConfig = request.app["config"]
    if not config.onboarded:
        raise web.HTTPFound("/onboarding")
    html = get_index_html()
    return web.Response(text=html, content_type="text/html")

async def handle_onboarding(request: web.Request) -> web.Response:
    """Serve the onboarding wizard. Redirect to dashboard if already onboarded."""
    config: NodeConfig = request.app["config"]
    if config.onboarded:
        raise web.HTTPFound("/")
    html = get_onboarding_html()
    return web.Response(text=html, content_type="text/html")

async def handle_health(_request: web.Request) -> web.Response:
    """Simple liveness probe."""
    return web.json_response({"status": "ok"})

async def handle_status(request: web.Request) -> web.Response:
    """Return rich node status."""
    config: NodeConfig = request.app["config"]
    engine = request.app["engine"]
    start_time: float = request.app["start_time"]
    session: Optional[aiohttp.ClientSession] = request.app.get("client_session")

    gpu: Optional[GPUInfo] = detect_gpu()
    gpu_info = asdict(gpu) if gpu else None

    engine_ready = False
    models_loaded: list[str] = []

    # First try our managed engine
    if engine is not None:
        engine_ready = getattr(engine, "ready", False)
        try:
            hc = engine.health_check()
            models_loaded = hc.get("models_loaded", [])
        except Exception:
            pass

    # Fallback: probe vLLM directly (handles Docker-managed vLLM)
    if not models_loaded and session is not None:
        try:
            vllm_url = f"http://localhost:{config.api_port}/v1/models"
            async with session.get(vllm_url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models_loaded = [m.get("id", "") for m in data.get("data", [])]
                    engine_ready = len(models_loaded) > 0
        except Exception:
            pass

    cluster: ClusterState = request.app["cluster_state"]
    master = cluster.get_master()
    effective_role = cluster.get_cluster_role_for(config.node_id) if config.node_id else "worker"

    return web.json_response({
        "node_id": config.node_id,
        "node_name": config.node_name,
        "model": config.model,
        "gpu": gpu_info,
        "engine_ready": engine_ready,
        "uptime": round(time.time() - start_time, 1),
        "version": __version__,
        "powered_by": "argentos.ai",
        "models_loaded": models_loaded,
        "api_port": config.api_port,
        "cluster_role": effective_role,
        "cluster_id": getattr(config, "cluster_id", "default"),
        "master_node_id": master.node_id if master else None,
    })

async def handle_nodes(request: web.Request) -> web.Response:
    """Return the list of known cluster nodes."""
    config: NodeConfig = request.app["config"]
    engine = request.app["engine"]
    cluster: ClusterState = request.app["cluster_state"]

    cluster_nodes = cluster.get_nodes(include_offline=False)
    if cluster_nodes:
        nodes_list = []
        for n in cluster_nodes:
            status_str = n.status.value if hasattr(n.status, "value") else str(n.status)
            dmode = getattr(n, "distributed_mode", "solo") or "solo"
            # Members are "ready for work" once discovered, even though they
            # don't run a local vLLM.
            ready = status_str in ("online", "serving") or (
                dmode == "member" and status_str in ("online", "member-ready")
            )
            nodes_list.append({
                "node_id": n.node_id,
                "node_name": n.node_name,
                "host": "localhost",
                "api_port": n.api_port,
                "web_port": n.web_port,
                "model": n.model,
                "gpu_name": n.gpu_name,
                "gpu_memory_gb": n.gpu_memory_gb,
                "unified_memory": n.unified_memory,
                "status": status_str,
                "engine_ready": ready,
                "distributed_mode": dmode,
                "distributed_instance_id": getattr(n, "distributed_instance_id", None),
                "distributed_peers": list(getattr(n, "distributed_peers", []) or []),
            })
    else:
        # Fallback: return this node
        engine_ready = False
        if engine is not None:
            engine_ready = getattr(engine, "ready", False)
        dmode = getattr(config, "distributed_mode", "solo") or "solo"
        nodes_list = [{
            "node_id": config.node_id,
            "node_name": config.node_name,
            "host": config.host,
            "api_port": config.api_port,
            "web_port": config.web_port,
            "model": config.model,
            "engine_ready": engine_ready or dmode == "member",
            "distributed_mode": dmode,
        }]
    return web.json_response({"nodes": nodes_list})

async def proxy_to_vllm(request: web.Request) -> web.StreamResponse:
    """Forward the request to the local vLLM server and stream the response back."""
    config: NodeConfig = request.app["config"]
    session: aiohttp.ClientSession = request.app["client_session"]
    collector: MetricsCollector = request.app["metrics_collector"]
    vllm_url = f"http://localhost:{config.api_port}{request.path}"

    # Extract model name for metrics
    model = config.model or "unknown"
    body_bytes = None
    if request.method == "POST":
        body_bytes = await request.read()
        try:
            import json as _json
            body_json = _json.loads(body_bytes)
            model = body_json.get("model", model)
        except Exception:
            pass
    # Tag the request so the server-view log middleware can capture the model
    try:
        request["_log_model"] = model
    except Exception:
        pass

    # Build upstream request kwargs
    kwargs: dict = {
        "headers": {k: v for k, v in request.headers.items()
                    if k.lower() not in ("host", "transfer-encoding")},
    }
    if body_bytes is not None:
        kwargs["data"] = body_bytes

    start_time = time.time()
    try:
        async with session.request(request.method, vllm_url, **kwargs) as upstream:
            # Detect SSE streaming
            is_sse = "text/event-stream" in upstream.headers.get("Content-Type", "")

            if is_sse:
                resp = web.StreamResponse(
                    status=upstream.status,
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )
                await resp.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                latency_ms = (time.time() - start_time) * 1000
                collector.record_request(model, latency_ms, error=False)
                return resp
            else:
                body = await upstream.read()
                latency_ms = (time.time() - start_time) * 1000
                collector.record_request(model, latency_ms, error=False)
                return web.Response(
                    status=upstream.status,
                    body=body,
                    content_type=upstream.headers.get("Content-Type", "application/json"),
                )
    except aiohttp.ClientError:
        latency_ms = (time.time() - start_time) * 1000
        collector.record_request(model, latency_ms, error=True)
        return web.json_response(
            {"error": {"message": "vLLM engine not reachable", "type": "server_error"}},
            status=502,
        )

async def handle_cluster_info(request: web.Request) -> web.Response:
    """Return the current cluster topology from this node's perspective."""
    config: NodeConfig = request.app["config"]
    cluster: ClusterState = request.app["cluster_state"]

    master = cluster.get_master()
    members = cluster.members()
    master_address: Optional[str] = getattr(config, "master_address", None)
    if master and not master_address and master.node_id != config.node_id:
        master_address = f"{master.node_name}:{master.web_port}"

    return web.json_response({
        "my_role": cluster.get_cluster_role_for(config.node_id) if config.node_id else "worker",
        "my_node_id": config.node_id,
        "cluster_id": getattr(config, "cluster_id", "default"),
        "configured_role": getattr(config, "cluster_role", "auto"),
        "master_node_id": master.node_id if master else None,
        "master_address": master_address,
        "members": [
            {
                "node_id": m.node_id,
                "node_name": m.node_name,
                "api_port": m.api_port,
                "web_port": m.web_port,
                "role": m.role,
                "effective_role": "master" if (master and master.node_id == m.node_id) else "worker",
                "status": m.status.value if hasattr(m.status, "value") else str(m.status),
                "last_seen": m.last_seen,
                "gpu_name": m.gpu_name,
                "gpu_memory_gb": m.gpu_memory_gb,
            }
            for m in members
        ],
    })


async def handle_cluster_resources(request: web.Request) -> web.Response:
    """Return aggregated cluster resources (VRAM, GPUs) across ready nodes."""
    cluster: ClusterState = request.app["cluster_state"]
    ray_state: RayAutostartState = request.app.get("ray_autostart_state") or RayAutostartState()

    members = cluster.members()
    # Include member-ready too: those nodes have reserved their GPU for
    # Ray workers launched by the head, so they contribute to total VRAM
    # even though they don't run their own local vLLM.
    ready = [
        n for n in members
        if (n.status.value if hasattr(n.status, "value") else str(n.status))
        in ("online", "serving", "starting", "member-ready")
    ]
    total_vram = sum(float(n.gpu_memory_gb or 0) for n in ready)
    total_gpus = len(ready)  # one GPU per node today; future: per-node GPU count

    # Surface the distributed instance (if any) — the head broadcasts it;
    # everybody else can tell from cluster state.
    distributed_instance = None
    for n in ready:
        iid = getattr(n, "distributed_instance_id", None)
        if iid:
            peers = list(getattr(n, "distributed_peers", []) or [])
            distributed_instance = {
                "instance_id": iid,
                "head_node_id": n.node_id,
                "head_node_name": n.node_name,
                "peer_ips": peers,
                "tensor_parallel_size": 1 + len(peers),
                "model": n.model,
            }
            break

    nodes_payload = []
    for n in ready:
        nodes_payload.append({
            "node_id": n.node_id,
            "hostname": n.node_name,
            "vram_gb": round(float(n.gpu_memory_gb or 0), 1),
            "gpus": 1,
            "gpu_name": n.gpu_name,
            "unified_memory": n.unified_memory,
            "status": n.status.value if hasattr(n.status, "value") else str(n.status),
            "distributed_mode": getattr(n, "distributed_mode", "solo") or "solo",
            "ray_status": (
                "head" if (ray_state.is_head and cluster.is_master_of_cluster() and n.node_id == (cluster._local_announcement.node_id if cluster._local_announcement else ""))
                else ("joined" if ray_state.joined_as_worker and cluster._local_announcement and n.node_id == cluster._local_announcement.node_id
                      else "unknown")
            ),
        })

    return web.json_response({
        "total_vram_gb": round(total_vram, 1),
        "available_vram_gb": round(total_vram, 1),  # best-effort; same as total until live utilization is wired
        "total_gpus": total_gpus,
        "total_nodes": len(ready),
        "nodes": nodes_payload,
        "distributed_instance": distributed_instance,
        "ray": ray_state.to_dict(),
    })


def _rebuild_announcement(app: web.Application) -> None:
    """Apply the current config to the live broadcast announcement.

    Lets role/cluster-id changes take effect without a server restart.
    """
    config: NodeConfig = app["config"]
    sender: Optional[BroadcastSender] = app.get("broadcast_sender")
    if sender is not None:
        sender.update_announcement(
            cluster_id=getattr(config, "cluster_id", "default"),
            role=getattr(config, "cluster_role", "auto"),
        )
    # Also refresh the stored announcement
    announcement: NodeAnnouncement = app.get("announcement")
    if announcement is not None:
        announcement.cluster_id = getattr(config, "cluster_id", "default")
        announcement.role = getattr(config, "cluster_role", "auto")


async def handle_cluster_set_role(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": {"message": "Invalid JSON body", "type": "invalid_request"}},
            status=400,
        )
    role = body.get("role", "")
    if role not in ("auto", "master", "worker"):
        return web.json_response(
            {"error": {"message": "role must be one of auto|master|worker", "type": "invalid_request"}},
            status=400,
        )
    config: NodeConfig = request.app["config"]
    config.cluster_role = role
    config.save()
    _rebuild_announcement(request.app)
    return web.json_response({"ok": True, "cluster_role": role})


async def handle_cluster_set_id(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": {"message": "Invalid JSON body", "type": "invalid_request"}},
            status=400,
        )
    cluster_id = str(body.get("cluster_id", "")).strip() or "default"
    if len(cluster_id) > 64 or not all(c.isalnum() or c in "-_." for c in cluster_id):
        return web.json_response(
            {"error": {"message": "cluster_id must be alphanumeric (plus -_.), 1-64 chars",
                       "type": "invalid_request"}},
            status=400,
        )
    config: NodeConfig = request.app["config"]
    config.cluster_id = cluster_id
    config.save()
    _rebuild_announcement(request.app)
    return web.json_response({"ok": True, "cluster_id": cluster_id})


# Fields that can be updated via PATCH /api/config. Keep this list tight --
# never expose auth or secret-related fields here.
PATCHABLE_CONFIG_FIELDS = {
    "node_name",
    "email",
    "host",
    "api_port",
    "web_port",
    "discovery_port",
    "model",
    "models_dir",
    "max_model_len",
    "gpu_memory_utilization",
    "quantization",
    "trust_remote_code",
    "cluster_enabled",
    "cluster_role",
    "cluster_id",
    "master_address",
    "datasets_dir",
    "training_dir",
    "hf_cache_dir",
    "cors_origins",
    "telemetry",
    "training_default_method",
    "training_default_epochs",
    "training_default_batch_size",
    "training_default_learning_rate",
}


def _safe_config_dict(config: NodeConfig) -> dict:
    """Return a safely serializable view of the config (no secrets)."""
    data = asdict(config)
    # Scrub anything that might carry a credential.
    data.pop("cluster_secret", None)
    return data


async def handle_get_config(request: web.Request) -> web.Response:
    config: NodeConfig = request.app["config"]
    return web.json_response(_safe_config_dict(config))


async def handle_patch_config(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": {"message": "Invalid JSON body", "type": "invalid_request"}},
            status=400,
        )
    if not isinstance(body, dict):
        return web.json_response(
            {"error": {"message": "Body must be an object", "type": "invalid_request"}},
            status=400,
        )

    config: NodeConfig = request.app["config"]
    applied: dict = {}
    rejected: list = []

    for key, value in body.items():
        if key not in PATCHABLE_CONFIG_FIELDS:
            rejected.append(key)
            continue
        if not hasattr(config, key):
            rejected.append(key)
            continue
        # Basic validation on role / cluster_id
        if key == "cluster_role" and value not in ("auto", "master", "worker"):
            rejected.append(key)
            continue
        setattr(config, key, value)
        applied[key] = value

    config.save()
    if any(k in applied for k in ("cluster_id", "cluster_role")):
        _rebuild_announcement(request.app)

    return web.json_response({
        "ok": True,
        "applied": applied,
        "rejected": rejected,
        "config": _safe_config_dict(config),
    })


def run_server(config: Optional[NodeConfig] = None, engine=None) -> None:
    """Start the API server (blocking)."""
    if config is None:
        config = NodeConfig()
    app = create_app(config=config, engine=engine)
    web.run_app(app, host=config.host, port=config.web_port, print=None)
