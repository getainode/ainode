"""AINode API proxy server — aiohttp app that serves the web UI and proxies to vLLM."""

import asyncio
import json
import logging
import os
import signal
import socket
import ssl
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import aiohttp
from aiohttp import web

from ainode.core.config import (
    DEFAULT_DISTRIBUTED_EXECUTOR,
    DEFAULT_ENGINE_BACKEND,
    NodeConfig,
)
from ainode.core.gpu import detect_gpu, detect_gpus, GPUInfo
from ainode.web.serve import get_index_html, get_static_path
from ainode.models.api_routes import register_model_routes
from ainode.auth.middleware import (
    TRUST_REMOTE_CODE_RULE,
    AuthConfig,
    auth_middleware,
    is_authenticated,
)
from ainode.auth.api_routes import register_auth_routes
from ainode.ratelimit.middleware import (
    RateLimitConfig,
    RateLimiter,
    rate_limit_middleware,
    rate_limit_status_fields,
)
from ainode.tls.certs import certificate_info, ssl_context
from ainode.tls.config import TLSConfig, load_tls_config
from ainode.metrics.collector import MetricsCollector, optional_float
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
from ainode.discovery.instance import instance_parallel
from ainode.discovery.signing import ClusterSecret
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
    endpoint_nodes,
    peer_host,
    register_server_routes,
    request_log_middleware,
    init_server_state,
)
from ainode.api.chat_routes import (
    catalog_entry,
    instance_caps_index,
    is_multimodal_limit_error,
    is_multimodal_request,
    node_label,
    order_by_vision,
    record_vision_unsupported,
    register_chat_routes,
)
from ainode.api.cluster_join import register_cluster_join_routes
from ainode.api.decide import handle_decide
from ainode.bench.api_routes import register_bench_routes

from ainode import __version__

logger = logging.getLogger(__name__)

def _client_max_bytes(config) -> int:
    """Inbound request-body ceiling for the API server, in bytes.

    NOT optional: aiohttp defaults to 1 MB, which rejects long-context prompts
    at the proxy. A model advertising 262k context can only be fed ~190k tokens
    through our own endpoint before the caller gets an opaque 413 that says
    nothing about which hop refused it (found 2026-08-25 benchmarking decode
    against context depth). A bad value falls back to the default rather than
    producing a server that rejects every body.
    """
    try:
        mb = int(getattr(config, "max_request_mb", 64))
    except (TypeError, ValueError):
        mb = 64
    return max(1, mb) * 1024 * 1024


def create_app(
    config: Optional[NodeConfig] = None,
    engine=None,
) -> web.Application:
    """Create and return the aiohttp application.

    Parameters
    ----------
    config : NodeConfig
        Node configuration (defaults created if None).
    engine : EngineBackend | VLLMEngine | None
        Optional engine instance for health/status queries. May be any of
        the concrete backends in :mod:`ainode.engine.backends` (eugr,
        nvidia) or the legacy :class:`ainode.engine.vllm_engine.VLLMEngine`
        pip-venv engine.
    """
    if config is None:
        config = NodeConfig()

    auth_config = AuthConfig.load()

    # Order matters. The rate limiter is LAST, so it runs innermost: by then the
    # auth middleware has stamped the API key id, which is what the limiter keys
    # its buckets on, and a request with no key on a node that requires one has
    # already been refused without spending anybody's budget.
    app = web.Application(
        middlewares=[cors_middleware, request_log_middleware, auth_middleware,
                     rate_limit_middleware],
        client_max_size=_client_max_bytes(config),
    )
    init_server_state(app)
    # Instantiate shared services
    collector = MetricsCollector()
    # Both take the operator's configured directories, with the AINODE_HOME
    # subpaths as the fallback: Config > Storage saved datasets_dir and
    # training_dir and nothing read either of them (#204).
    dataset_manager = DatasetManager(root=(config.datasets_dir or None))
    manager = TrainingManager(dataset_manager=dataset_manager, config=config)

    # Build local node announcement for discovery
    announcement = _build_announcement(config, engine)
    cluster = ClusterState(local_announcement=announcement)

    app["config"] = config
    app["auth_config"] = auth_config
    # Per-client limits on /v1. Off unless config.json says otherwise, so a node
    # that never heard of the block behaves exactly as it did before.
    app["rate_limiter"] = RateLimiter(config=RateLimitConfig.from_config(config))
    app["engine"] = engine
    # Seed the InstanceManager with the boot engine as the PRIMARY instance, so
    # a later solo load APPENDS on the next port (8001…) instead of colliding
    # with the boot container on the legacy name/port. The boot engine is owned
    # by `ainode start` and binds `ainode-vllm-node-solo` + api_port; without
    # this seed the manager would hand the same name/port to a 2nd backend and
    # the two fight (repeated docker-name Conflict, neither serving).
    if engine is not None and getattr(config, "model", None) \
            and (getattr(config, "distributed_mode", "solo") or "solo") == "solo":
        from ainode.discovery.instance import InstanceRecord
        from ainode.engine.instance_manager import InstanceManager
        _seed = InstanceManager(base_port=config.api_port)
        _seed.add(InstanceRecord(
            instance_id=f"{config.node_id or 'head'}:{config.model}",
            model=config.model, head_node_id=config.node_id or "head",
            peer_ips=[], api_port=config.api_port, tensor_parallel_size=1,
            status="starting"), engine)
        app["instances"] = _seed
    app["start_time"] = time.time()
    app["client_session"] = None  # lazy-init in startup
    app["metrics_collector"] = collector
    # On a unified-memory node the collector has no usage figure of its own to
    # report, and host RAM is not VRAM (#175). Give it the one number this node
    # really knows: what its engines reserved.
    collector.set_reservation_provider(lambda: engine_reserved_fraction(app))
    app["training_manager"] = manager
    app["dataset_manager"] = dataset_manager
    app["cluster_state"] = cluster
    app["announcement"] = announcement
    app["broadcast_sender"] = None
    app["broadcast_listener"] = None
    app["secrets_manager"] = SecretsManager()
    app["embedding_manager"] = EmbeddingManager()
    # Ray autostart is only meaningful for the legacy eugr backend. The NVIDIA
    # backend manages its own Ray lifecycle via run_cluster.sh at model-load time;
    # running `ray start` here fights with that (session-name mismatch on peer
    # nodes, port conflicts on port 6379). Disable the autostart loop in that case.
    _engine_backend_for_ray = (
        getattr(config, "engine_backend", None) or DEFAULT_ENGINE_BACKEND
    ).lower()
    app["ray_autostart_state"] = RayAutostartState(enabled=(_engine_backend_for_ray == "eugr"))

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    app.router.add_get("/", handle_index)
    app.router.add_get("/api/health", handle_health)
    app.router.add_get("/api/status", handle_status)
    app.router.add_get("/api/nodes", handle_nodes)
    app.router.add_get("/api/cluster/info", handle_cluster_info)
    app.router.add_get("/api/cluster/resources", handle_cluster_resources)
    app.router.add_post("/api/cluster/role", handle_cluster_set_role)
    app.router.add_post("/api/cluster/id", handle_cluster_set_id)
    # Joining: POST /api/cluster/join (keyless, because a node that has not
    # joined cannot hold this cluster's key, so the token is the credential)
    # and POST /api/cluster/join-self (keyed, what the dashboard card calls).
    register_cluster_join_routes(app)
    app.router.add_post("/api/cluster/load", handle_cluster_load)
    app.router.add_post("/api/cluster/unload", handle_cluster_unload)
    app.router.add_post("/api/cluster/update-all", handle_cluster_update_all)
    app.router.add_get("/api/cluster/update-status", handle_cluster_update_status)
    app.router.add_get("/api/config", handle_get_config)
    app.router.add_post("/api/engine/set-model", handle_set_model)
    app.router.add_get("/api/version/check", handle_version_check)
    app.router.add_post("/api/engine/update", handle_engine_update)
    app.router.add_patch("/api/config", handle_patch_config)

    app.router.add_get("/v1/models", handle_v1_models)
    app.router.add_post("/v1/chat/completions", proxy_to_vllm)
    app.router.add_post("/v1/completions", proxy_to_vllm)
    # The Anthropic Messages API. vLLM serves it natively alongside the OpenAI
    # paths, so a client that speaks it (Claude Code) can be pointed at the fleet
    # endpoint instead of one engine's port, which it could not be while :3000
    # answered 404 here. Deliberately the SAME handler: the body's `model` picks
    # the node, an `{"type": "image"}` block orders candidates by capability, a
    # ghost node fails over and an SSE answer streams through, all of it code that
    # already existed and none of it protocol-specific.
    app.router.add_post("/v1/messages", proxy_to_vllm)
    app.router.add_post("/v1/messages/count_tokens", proxy_to_vllm)
    # The rest of what the engines actually serve. Every one of these is a path
    # vLLM answers on the port behind this proxy and :3000 was answering 404 for,
    # so a caller had to abandon fleet routing and address one engine directly to
    # use it. Same handler, for the same reason /v1/messages is: the body carries
    # a `model`, which is all the routing, failover and SSE passthrough below need.
    #   /v1/responses: the OpenAI Responses API (streams; 0.27.1 serves it)
    #   /tokenize, /detokenize: NOT under /v1 in vLLM's own route table, and
    #                    served by chat AND pooling engines alike
    #   /v1/rerank, /v1/score: pooling-model paths, live on the embedding engine
    #                    beside /v1/embeddings (which keeps its own handler
    #                    because it validates the body before forwarding)
    # Verified against vLLM 0.27.1's /openapi.json on Spark-4 (chat engine) and
    # the 0.17.0 pooling engine stacked beside it.
    app.router.add_post("/v1/responses", proxy_to_vllm)
    app.router.add_post("/tokenize", proxy_to_vllm)
    app.router.add_post("/detokenize", proxy_to_vllm)
    app.router.add_post("/v1/rerank", proxy_to_vllm)
    app.router.add_post("/v1/score", proxy_to_vllm)
    # The decision endpoint. NOT a forwarded path and deliberately not on
    # proxy_to_vllm: it composes N grammar-constrained chat completions of its
    # own out of one request, so there is no caller body to forward. It routes
    # with the proxy's own `_routing_candidates` and the shared client session,
    # so a head still reaches the node serving the requested model.
    app.router.add_post("/v1/decide", handle_decide)

    # Chat view: the per-instance model card + the capability probe. Registered
    # BEFORE the model routes because aiohttp resolves in registration order and
    # /api/models/{model_id} would otherwise swallow /api/models/card|caps.
    register_chat_routes(app)

    # The manager reads and writes the SAME directory the engine mounts. It used
    # to take the module constant, so changing Config > Models directory left
    # downloads, Installed and Delete on the old path while launches used the new
    # one (#205).
    register_model_routes(app, models_dir=config.models_dir)

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

    # --- Bench routes --------------------------------------------------------
    register_bench_routes(app)

    app.router.add_static("/static", get_static_path(), name="static")

    # Static assets revalidate on every load. aiohttp's static handler answers a
    # conditional request with 304 via Last-Modified/ETag, so "no-cache" costs a
    # round trip per asset and never serves a stale stylesheet after an update.
    async def _static_no_cache(request, response):
        if request.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
    app.on_response_prepare.append(_static_no_cache)

    return app


def _head_instances(config) -> list:
    """Phase 2: the instances list this node HEADS, as wire dicts.

    Today there's at most one (derived from config), so this returns a 0- or
    1-element list. When the engine layer owns multiple instances (P2-2) it will
    return all of them. Mirrors the legacy distributed_instance_id/peers.
    """
    from ainode.discovery.instance import InstanceRecord

    peer_ips = list(getattr(config, "peer_ips", []) or [])
    if not peer_ips:
        return []
    iid = f"{config.node_id or 'head'}:{config.model}"
    return [InstanceRecord(
        instance_id=iid,
        model=config.model or "",
        head_node_id=config.node_id or "unknown",
        peer_ips=peer_ips,
        api_port=config.api_port,
        tensor_parallel_size=1 + len(peer_ips),
        status="serving",
        distributed_executor=(getattr(config, "distributed_executor", "")
                              or DEFAULT_DISTRIBUTED_EXECUTOR),
    ).to_dict()]


def announced_instances(config, live_records, dmode: str, engine_serving: bool) -> list:
    """The `instances` list this node broadcasts: its managed ones AND its head.

    The head of a distributed launch is an instance like any other as far as the
    master is concerned: it has to be routable by model id and drawn as DISTRIBUTED
    TP=N across head plus peers. It used to be synthesised only when the
    InstanceManager was EMPTY, so the first stacked model loaded beside a head made
    the head vanish from the announcement: the peer was drawn as an empty node, the
    head instance as SINGLE, and the master's fleet view lost a live distributed
    engine (#162). The head is also absent from the manager altogether after a
    restart, because a distributed launch is deliberately kept out of
    ``instances.json`` and never replayed.

    So the two sources are merged rather than chosen between, keyed on the port a
    node can only have one engine on. A manager record wins where both describe the
    same port, because that one carries the launch's real peer set and executor
    instead of the shape reconstructed from config.
    """
    out = [record.to_dict() for record in (live_records or [])]
    if dmode != "head" or not engine_serving:
        return out
    ports = {entry.get("api_port") for entry in out}
    for entry in _head_instances(config):
        if entry.get("api_port") not in ports:
            out.append(entry)
    return out


def engine_reserved_fraction(app) -> Optional[float]:
    """How much of this node's memory the engines on it have RESERVED, 0.0 to 1.0.

    This is the honest answer to "how much of this node is committed" on a
    unified-memory part, where NVML reports no usage at all and the psutil
    figure that stood in for it was host RAM: page cache and every other
    process, published as VRAM used (#175).

    The number AINode actually knows is vLLM's ``gpu_memory_utilization``, set
    per instance at launch and already the basis of the stacked-load admission
    gate. Summed across the live instances, it is what the engines hold. None
    when there is no engine layer to ask, which reports the figure as unknown
    rather than as an empty node.
    """
    config = app.get("config")
    if config is None:
        return None
    default_gmu = getattr(config, "gpu_memory_utilization", None)

    manager = app.get("instances")
    records = []
    if manager is not None:
        try:
            records = list(manager.instances())
        except Exception:
            records = []

    if records:
        total = 0.0
        for inst in records:
            gmu = getattr(getattr(inst, "backend", None), "config", None)
            gmu = getattr(gmu, "gpu_memory_utilization", None) if gmu else None
            if gmu is None:
                gmu = default_gmu
            if gmu is not None:
                total += float(gmu)
        return total

    # A distributed head replayed from config.json is not in the manager, and
    # neither is the boot engine before the seed lands: an engine that is running
    # holds its node's configured fraction.
    engine = app.get("engine")
    if engine is not None and default_gmu is not None:
        return float(default_gmu)
    # No engine on this node reserves nothing. That is a measurement, not a gap.
    return 0.0


def _build_announcement(config: NodeConfig, engine=None) -> NodeAnnouncement:
    """Create a NodeAnnouncement from current node state."""
    # Every device on the host, not device 0: a four-V100 node announced one
    # 32 GB GPU, and that number was the cluster's total VRAM, the topology's
    # GPU count and every placement decision (#163).
    gpus = detect_gpus()
    if gpus is not None:
        gpu_name = gpus.name
        gpu_count = gpus.count
        gpu_memory_gb = round(gpus.memory_total_mb / 1024, 1)
        unified_memory = gpus.unified_memory
    else:
        gpu: Optional[GPUInfo] = detect_gpu()
        gpu_name = gpu.name if gpu else "CPU"
        gpu_count = 1 if gpu else 0
        gpu_memory_gb = round(gpu.memory_total_mb / 1024, 1) if gpu else 0.0
        unified_memory = gpu.unified_memory if gpu else False

    # This node's fabric IP, so a head can launch us over the cluster fabric
    # (BUG D fix — not the mgmt-LAN UDP source address).
    fabric_ip = ""
    try:
        from ainode.cluster.hca_discovery import detect_fabric_ip
        from ainode.cluster.netdev import resolve_cluster_interface
        fabric_ip = detect_fabric_ip(resolve_cluster_interface(config)) or ""
    except Exception:
        pass

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
        gpu_count=gpu_count,
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
        fabric_ip=fabric_ip,
        instances=(_head_instances(config) if (distributed_mode == "head" and engine_ready) else []),
        # The timing half of the load-progress fields needs the app (one shared
        # fallback stamp), so the cluster sync loop fills those in on its next
        # tick. The phase is free here and is what a peer reads first.
        load_phase=engine_load_phase(engine, engine_ready),
        # The release on the wire (#171). Set once: a node cannot change its own
        # version without restarting the process that reads it.
        ainode_version=__version__,
    )


async def _on_startup(app: web.Application) -> None:
    app["client_session"] = aiohttp.ClientSession()

    config: NodeConfig = app["config"]
    announcement: NodeAnnouncement = app["announcement"]
    cluster: ClusterState = app["cluster_state"]

    _start_metrics_retention(app, config)

    # Always-on: re-load the persisted solo instance set so a `systemctl restart
    # ainode` brings every previously-loaded model back with no manual step.
    # Skipped when the operator requested a clean boot (closet #310, set in the
    # CLI start path) so a restart can actually free the node.
    if getattr(config, "_skip_replay", False):
        logger.info("start-clean: skipping persisted-instance replay")
    else:
        try:
            from ainode.models.api_routes import replay_instances_on_startup
            app["_instance_replay_task"] = asyncio.get_event_loop().create_task(
                replay_instances_on_startup(app)
            )
        except Exception:
            logger.exception("Failed to schedule instance replay")

    if config.cluster_enabled:
        # Start broadcast sender
        _collector = app.get("metrics_collector")
        # One secret source for both directions, re-read from config.json when it
        # changes so rotating cluster_secret needs no restart (#169). Shared, so
        # this node never signs with one value while verifying against another.
        cluster_secret = ClusterSecret(config)
        app["cluster_secret"] = cluster_secret
        sender = BroadcastSender(
            announcement=announcement,
            discovery_port=config.discovery_port,
            # Stamp live GPU telemetry onto every broadcast so the head can
            # render real per-peer VRAM/util (metrics fan-out).
            metrics_provider=(_collector.get_gpu_metrics if _collector else None),
            secret_provider=cluster_secret,
        )
        await sender.start()
        app["broadcast_sender"] = sender
        # Port AND cluster id AND whether the wire is signed, in one greppable
        # line: a node on the wrong port (#181) or the only node in the fleet
        # without a secret (#169) is otherwise invisible at both ends.
        logger.info(
            "Discovery sender started on UDP port %d (cluster_id=%s, version=%s, "
            "announcements %s)",
            config.discovery_port, getattr(config, "cluster_id", "default"),
            __version__,
            "signed" if cluster_secret() else "UNSIGNED (no cluster_secret set)")

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
            secret_provider=cluster_secret,
        )
        await listener.start()
        app["broadcast_listener"] = listener
        logger.info(
            "Discovery listener started on UDP port %d (cluster_id=%s, %s)",
            config.discovery_port, getattr(config, "cluster_id", "default"),
            "verifying signatures" if cluster_secret()
            else "accepting UNSIGNED announcements from anyone on this broadcast "
                 "domain (no cluster_secret set)")

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

        if app["ray_autostart_state"].enabled:
            app["_ray_autostart_task"] = asyncio.get_event_loop().create_task(
                _ray_autostart_loop(
                    cluster_state=cluster,
                    get_master_address=_get_master_address,
                    state=app["ray_autostart_state"],
                )
            )


def _start_metrics_retention(app: web.Application, config: NodeConfig) -> None:
    """Open ``<AINODE_HOME>/metrics.db`` and start the sampler, if enabled.

    Started here and not in ``create_app`` so an application that is built and
    never run (every test that only inspects the route table) neither opens a
    database nor starts a thread. ``/api/metrics/history`` reads the store off
    the app at request time, so it simply reports retention off when this did
    not run or could not open the file.

    Nothing in here is allowed to stop the node coming up. A metrics file that
    cannot be opened is a node with no history, which is exactly where every
    node was before this existed.
    """
    from ainode.metrics.store import MetricsSettings, MetricsStore

    settings = MetricsSettings.from_config(config)
    if not settings.enabled:
        logger.info("metrics retention disabled by config")
        return
    collector = app.get("metrics_collector")
    if collector is None:
        return
    try:
        store = MetricsStore(settings=settings)
    except Exception:
        logger.exception("metrics retention: could not open the sample store")
        return
    if not store.available:
        return
    app["metrics_store"] = store
    collector.attach_store(store, interval_seconds=settings.interval_seconds)
    logger.info(
        "metrics retention on: %s, raw %dh, rolled up %dd, every %.0fs",
        store.path, settings.retention_hours, settings.retention_days,
        settings.interval_seconds,
    )


def _stop_metrics_retention(app: web.Application) -> None:
    """Stop the sampler and close the store. Never raises."""
    collector = app.get("metrics_collector")
    if collector is not None:
        try:
            collector.detach_store()
        except Exception:
            pass
    store = app.get("metrics_store")
    if store is not None:
        try:
            store.close()
        except Exception:
            pass


async def _engine_serving(backend, loop) -> bool:
    """True iff the engine's OpenAI API actually answers right now.

    The latched `ready` flag never flips False when an engine crashes or is
    killed out-of-band, so a dead engine reads READY forever (phantom-READY →
    ghost routing → 502s). An active localhost probe is the truthful signal —
    and it correctly reads False while a model is still loading (api not up
    yet), so we never advertise a not-yet-serving OR already-dead engine.

    health_check() uses a blocking 5s-timeout urlopen, so run it in the default
    executor to keep the event loop free (a hung engine must not stall sync).
    """
    if backend is None:
        return False
    try:
        hc = await loop.run_in_executor(None, backend.health_check)
        return bool(hc.get("api_responding"))
    except Exception:
        return False


async def _engine_port_serving(app, port: int) -> bool:
    """True iff ``localhost:<port>/v1/models`` answers 200 with a model right now.

    The readiness question ``engine_ready`` claims to answer, asked the cheap way:
    one localhost GET on the endpoint dashboards and clients actually use. Not
    ``_engine_serving``, which goes through ``health_check`` and shells out to
    ``docker inspect`` -- too expensive for the hottest polled endpoint -- and
    not the engine's latched ``ready`` flag, which reads True before a model is
    serving and never flips back when one dies.
    """
    session: Optional[aiohttp.ClientSession] = app.get("client_session")
    if session is None:
        # No HTTP session (a bare app, or shutdown): the managed engine's own
        # health check is the next best answer.
        return await _engine_serving(app.get("engine"), asyncio.get_event_loop())
    try:
        url = f"http://localhost:{port}/v1/models"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
            if resp.status != 200:
                return False
            data = await resp.json()
            return len(data.get("data", [])) > 0
    except Exception:
        return False


async def _live_instance_records(manager, loop) -> list:
    """Probe every managed instance; return the records whose engine answers.

    Also flips each live record's status to ``serving`` (F3). The record is
    stamped ``starting`` at load time and was never updated once the engine
    came up, so the announcement advertised a phantom ``starting`` forever and
    the dashboard kept painting a "STARTING · 8%" progress bar for an instance
    that was actually serving traffic. The localhost /v1/models probe (via
    ``_engine_serving``) is the truthful liveness signal, so a passing probe is
    exactly when the status should read ``serving``.
    """
    live = []
    for inst in manager.instances():
        if await _engine_serving(inst.backend, loop):
            if inst.record.status != "serving":
                inst.record.status = "serving"
            live.append(inst.record)
        elif inst.record.status == "serving":
            # Truthful reset (the other half of F3): the `serving` stamp is a
            # latch — set once when the engine first answered and, before this,
            # never cleared. An instance that was serving but whose engine no
            # longer answers has crashed or been killed out-of-band; leaving the
            # latch at `serving` makes every consumer that reads `record.status`
            # without its own probe (e.g. the Server view's LOADED MODELS list)
            # paint a dead instance as READY forever. Flip it back to `failed`
            # so status tracks liveness in both directions. If the engine
            # recovers, the next cycle re-flips it to `serving`.
            inst.record.status = "failed"
    return live


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
                loop = asyncio.get_event_loop()
                # Liveness: the latched `ready` flag never flips False when an engine
                # crashes or is killed out-of-band, so a dead engine reads READY forever
                # (phantom-READY → ghost routing → 502s, BUG A FIX 2). Probe the engine's
                # own API instead (see _engine_serving) — also reads False while loading,
                # so we never advertise a not-yet-serving OR already-dead engine.
                # ponytail: one localhost probe per instance per 5s cycle; a transient
                # blip drops the model for one cycle and self-heals on the next probe.
                engine_serving = await _engine_serving(engine, loop)
                engine_proc_alive = bool(engine is not None and engine.is_running())
                # Re-broadcast the live primary model every cycle, gated on real
                # liveness — fixes both the stale `model` field (BUG A) and the
                # phantom-READY-after-crash case (FIX 2). Members serve via the head's
                # sharded engine, not their own model.
                updates["model"] = "" if (dmode == "member" or not engine_serving) else (config.model or "")
                if dmode == "member":
                    updates["status"] = "member-ready"
                elif engine is not None:
                    updates["status"] = (
                        "serving" if engine_serving
                        else ("starting" if engine_proc_alive else "stopped")
                    )

                # Live load progress on the wire, so a peer loading a model is
                # drawn as loading on the master's dashboard rather than as a
                # node that is simply not ready. Same numbers /api/status serves
                # locally, from the same helpers, timed on this node's clock.
                load_phase = engine_load_phase(engine, engine_serving)
                updates["load_phase"] = load_phase
                progress = await load_progress_payload(
                    app, engine, load_phase, config.model or "")
                updates["load_started_at"] = progress["load_started_at"]
                updates["load_elapsed_seconds"] = progress["load_elapsed_seconds"]
                updates["expected_ready_minutes"] = progress["expected_ready_minutes"]

                # Advertise distributed instance metadata once the head's
                # sharded engine is serving — the UI uses this to render
                # "DISTRIBUTED TP=N across X nodes".
                if dmode == "head" and engine_serving:
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
                manager = app.get("instances")
                live_records = []
                if manager is not None and not manager.is_empty():
                    # Only advertise instances whose engine actually answers — a dead
                    # stacked instance drops out of the broadcast within one cycle —
                    # and flip each live record's status to `serving` so the UI stops
                    # showing a phantom `starting` progress bar (F3).
                    live_records = await _live_instance_records(manager, loop)
                updates["instances"] = announced_instances(
                    config, live_records, dmode, engine_serving)
                if sender:
                    sender.update_announcement(**updates)
                    # Keep the app-level announcement in sync so /api/status sees fresh values
                    for k, v in updates.items():
                        if hasattr(sender.announcement, k):
                            setattr(sender.announcement, k, v)
    except asyncio.CancelledError:
        pass


async def _on_cleanup(app: web.Application) -> None:
    # Stop the metrics sampler before anything else, so the last thing it writes
    # is a sample of a node that is still up rather than one mid-teardown.
    _stop_metrics_retention(app)

    # Stop the instance-replay task if still running
    replay_task = app.get("_instance_replay_task")
    if replay_task:
        replay_task.cancel()
        try:
            await replay_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

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

# ---------------------------------------------------------------------------
# Pinning a forwarded request to ONE instance (#197)
# ---------------------------------------------------------------------------
# The chat and bench pickers let a user choose an instance ("model @ node:port").
# Routing on the model id alone then sent the request to the local hop first, so
# with the same model on two nodes the answer, its stats and the "routing to"
# line came from an engine the user did not pick. A caller pins the target with
# these two headers (what the UI sends: a header costs nothing to forward and
# leaves the OpenAI body untouched) or with an ``ainode_target`` field in the
# body, which the proxy strips before forwarding. A pin is honored EXACTLY: one
# candidate, no failover, because the point of pinning is that the answer is
# attributable to that engine. Documented in the README under
# "Pinning a request to one instance".
TARGET_NODE_HEADER = "X-AINode-Node"
TARGET_PORT_HEADER = "X-AINode-Port"
TARGET_BODY_FIELD = "ainode_target"
# Which instance actually answered, on every forwarded response.
SERVED_BY_HEADER = "X-AINode-Served-By"


def pinned_target(headers, body_obj: dict) -> tuple:
    """The instance a caller pinned this request to, as ``(node_id, port)``.

    Either part may be absent: a port with no node pins a stacked instance on
    this node, a node with no port pins that node's primary engine port. Returns
    ``(None, None)`` when the caller pinned nothing.
    """
    node = (headers.get(TARGET_NODE_HEADER) or "").strip()
    port_raw = (headers.get(TARGET_PORT_HEADER) or "").strip()
    target = body_obj.get(TARGET_BODY_FIELD) if isinstance(body_obj, dict) else None
    if isinstance(target, str) and target.strip():
        # "<node_id>" or "<node_id>:<port>"
        head, _, tail = target.strip().rpartition(":")
        node = node or (head if head else tail)
        port_raw = port_raw or (tail if head else "")
    elif isinstance(target, dict):
        node = node or str(target.get("node_id") or target.get("node") or "").strip()
        port_raw = port_raw or str(target.get("port") or target.get("api_port") or "").strip()
    port = None
    if port_raw:
        try:
            port = int(port_raw)
        except ValueError:
            port = None
    return (node or None), port


def pinned_candidate(cluster, config, node_id: Optional[str], port: Optional[int]):
    """Resolve a pin to the single ``(host, port)`` to forward to, or None.

    None means the node id is not in this node's cluster view (or has no fabric
    IP), which the caller has to hear about: silently falling back to the local
    hop is exactly the mis-attribution the pin exists to prevent.
    """
    if node_id and node_id != config.node_id:
        node = cluster.get_node(node_id) if cluster is not None else None
        host = (getattr(node, "fabric_ip", "") or "") if node is not None else ""
        if not host:
            return None
        return (host, port or getattr(node, "api_port", None) or config.api_port)
    return ("localhost", port or config.api_port)


def cors_allowed_origin(config, origin: str) -> str:
    """The value for ``Access-Control-Allow-Origin``, or "" to allow nothing.

    localhost and 127.0.0.1 are always allowed: that is where the dashboard is
    served from. ``config.cors_origins`` is a comma-separated allow-list ON TOP
    of that, so an operator who types an origin into Config > Network actually
    gets it, and ``*`` allows any origin. The field used to save and never be
    read, which for a CORS setting is not a cosmetic bug (#204).
    """
    if not origin:
        return ""
    if origin.startswith(("http://localhost", "http://127.0.0.1")):
        return origin
    entries = [e.strip() for e in (getattr(config, "cors_origins", None) or "").split(",")]
    entries = [e for e in entries if e]
    if "*" in entries:
        return origin
    return origin if origin in entries else ""


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
    allowed = cors_allowed_origin(request.app.get("config"), origin)
    resp.headers["Access-Control-Allow-Origin"] = allowed
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = (
        f"Content-Type, Authorization, {TARGET_NODE_HEADER}, {TARGET_PORT_HEADER}")
    # The instance that actually answered is a response header, so a browser
    # client has to be told it may read it.
    resp.headers["Access-Control-Expose-Headers"] = SERVED_BY_HEADER
    return resp

async def handle_index(request: web.Request) -> web.Response:
    """Serve the dashboard.

    It used to redirect to /onboarding whenever ``config.onboarded`` was false,
    which was never: the installer writes ``"onboarded": true`` and a non-TTY
    start sets it before the server binds, so on every deployed node this branch
    answered a redirect to a page that redirected straight back (#208). The
    browser wizard it guarded is gone; joining a cluster is Config > Cluster or
    ``ainode join``.
    """
    html = get_index_html()
    return web.Response(text=html, content_type="text/html")

async def handle_health(_request: web.Request) -> web.Response:
    """Simple liveness probe."""
    return web.json_response({"status": "ok"})

# -- Live load progress: how far into a launch this node is right now ---------
#
# The launch-time ledger (models/api_routes.py) answers "how long did this model
# take LAST time". It cannot say anything about the load happening right now,
# and a load on this hardware runs for minutes: the interface used to show a
# fixed percent per phase, which stands still for twelve minutes and reads as a
# hang. These three fields are everything a client needs to draw an honest bar
# and to say "taking longer than usual": when the load started, how long it has
# been going, and how long it is expected to take.
#
# ONE key holds the API-layer fallback start stamp, so there is one place that
# records it and one place that clears it.
_LOAD_START_KEY = "load_start_fallback"

# Phases where no load is in flight. Both ends of a launch land here (a load
# that reached ready, a node with no engine), and both clear the stamp so the
# next launch times itself from scratch.
_LOAD_SETTLED_PHASES = ("", "idle", "ready")


def engine_load_phase(engine, engine_ready: bool) -> str:
    """Coarse load phase for this node: the live readiness probe wins.

    idle | starting | loading_weights | distributed_init | profiling | ready.
    The engine's own ``_ready`` latch can miss vLLM's startup log marker and
    stay False on a model that is actually serving, so a probe that says serving
    reports ready whatever the backend thinks.
    """
    if engine_ready:
        return "ready"
    if engine is None:
        return "idle"
    return str(getattr(engine, "load_phase", "idle") or "idle")


def _forget_load_start(app) -> None:
    """Drop the fallback stamp once a load has settled."""
    try:
        if app.get(_LOAD_START_KEY) is not None:
            app[_LOAD_START_KEY] = None
    except Exception:  # pragma: no cover - a mapping that refuses writes
        pass


def load_started_at(app, engine, load_phase: str,
                    *, now: Optional[float] = None) -> Optional[float]:
    """Epoch seconds the current load started, or None when nothing is loading.

    The backend's ``launched_at`` is the real answer: stamped when the container
    starts, cleared in ``stop()``, so elapsed time survives a browser reload and
    every client sees the same number. It is None for the mp distributed shape
    until the head stamps it (and for any backend that launches no container of
    its own), so a phase past idle with no stamp gets one from here instead,
    recorded on the app where every reader shares it. A load that is late to
    stamp is therefore reported a poll late, never as "no load at all".
    """
    phase = (load_phase or "idle").strip()
    if phase in _LOAD_SETTLED_PHASES:
        _forget_load_start(app)
        return None
    stamped = getattr(engine, "launched_at", None) if engine is not None else None
    if isinstance(stamped, (int, float)) and not isinstance(stamped, bool) and stamped > 0:
        return float(stamped)
    try:
        existing = app.get(_LOAD_START_KEY)
    except Exception:  # pragma: no cover - a mapping that refuses reads
        existing = None
    if isinstance(existing, (int, float)) and not isinstance(existing, bool) and existing > 0:
        return float(existing)
    started = float(now if now is not None else time.time())
    try:
        app[_LOAD_START_KEY] = started
    except Exception:  # pragma: no cover - a mapping that refuses writes
        pass
    return started


def expected_ready_minutes(model: str, hf_repo: str = "") -> Optional[float]:
    """Minutes this model is expected to need to come up here, or None.

    This node's own ledger first (what it actually did last time, on this
    hardware, at this parallelism), then the catalog's ``typical_ready_minutes``
    seed, then None. Never a figure derived from the weight size: a guess the UI
    then draws a progress bar from is worse than drawing no bar.

    Reads the small ledger file, so callers on the event loop go through
    ``load_progress_payload``.
    """
    m = (model or "").strip()
    if not m:
        return None
    try:
        from ainode.models.api_routes import last_ready_launch

        hit = last_ready_launch(m, hf_repo or "")
    except Exception:  # pragma: no cover - a ledger read must not fail a status read
        hit = None
    if hit is not None:
        secs = hit.get("seconds_to_ready")
        if isinstance(secs, (int, float)) and not isinstance(secs, bool) and secs > 0:
            return round(float(secs) / 60.0, 1)
    info = catalog_entry(m) or (catalog_entry(hf_repo) if hf_repo else None)
    typical = getattr(info, "typical_ready_minutes", None) if info is not None else None
    if isinstance(typical, (int, float)) and not isinstance(typical, bool) and typical > 0:
        return float(typical)
    return None


def load_progress_fields(app, engine, load_phase: str, model: str,
                         *, hf_repo: str = "", now: Optional[float] = None) -> dict:
    """The three live-load fields for one node. Blocking (reads the ledger).

    All three are None whenever nothing is loading, so a client can read null as
    "no bar to draw" without a special case per surface.
    """
    started = load_started_at(app, engine, load_phase, now=now)
    if started is None:
        return {"load_started_at": None, "load_elapsed_seconds": None,
                "expected_ready_minutes": None}
    stamp = float(now if now is not None else time.time())
    return {
        "load_started_at": round(started, 3),
        "load_elapsed_seconds": round(max(0.0, stamp - started), 1),
        "expected_ready_minutes": expected_ready_minutes(model, hf_repo),
    }


async def load_progress_payload(app, engine, load_phase: str, model: str,
                                *, hf_repo: str = "") -> dict:
    """``load_progress_fields`` with its ledger read off the event loop.

    An idle node answers on the loop: there is no ledger read to offload when
    nothing is loading, and /api/status is polled every few seconds.
    """
    if (load_phase or "idle").strip() in _LOAD_SETTLED_PHASES:
        return load_progress_fields(app, engine, load_phase, model, hf_repo=hf_repo)
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: load_progress_fields(app, engine, load_phase, model, hf_repo=hf_repo),
        )
    except Exception:  # pragma: no cover - never fail a status read on this
        logger.exception("could not read the load progress for %s", model)
        return {"load_started_at": None, "load_elapsed_seconds": None,
                "expected_ready_minutes": None}


def peer_load_progress(node) -> dict:
    """The three live-load fields a PEER announced, passed through as it said them.

    Elapsed is the peer's own arithmetic on the peer's own clock, which is the
    honest number here: the two nodes do not share a clock, so subtracting a
    remote ``load_started_at`` from our own would report skew as progress. It is
    at most one broadcast interval stale, and the browser adds the time since
    the poll on top of it.
    """
    started = getattr(node, "load_started_at", None)
    if not isinstance(started, (int, float)) or isinstance(started, bool) or started <= 0:
        return {"load_started_at": None, "load_elapsed_seconds": None,
                "expected_ready_minutes": None}
    elapsed = getattr(node, "load_elapsed_seconds", None)
    expected = getattr(node, "expected_ready_minutes", None)
    return {
        "load_started_at": float(started),
        "load_elapsed_seconds": (float(elapsed) if isinstance(elapsed, (int, float))
                                 and not isinstance(elapsed, bool) else None),
        "expected_ready_minutes": (float(expected) if isinstance(expected, (int, float))
                                   and not isinstance(expected, bool) else None),
    }


# Free space below this fraction of a filesystem is reported as a warning. A node
# pays for a model twice (the download, then whatever the engine caches beside
# it), and running out mid-pull surfaces as an engine that died rather than as a
# disk error, so the dashboard has to be able to see it coming. Same threshold
# ``ainode doctor`` warns at.
DISK_WARN_FRACTION = 0.15


def disk_fields(config: NodeConfig) -> dict:
    """Free and total space on the two directories a launch writes to.

    The AINode home holds config, logs, the bench ledger and the secrets store;
    the models dir holds the weights and is usually far bigger, and on our nodes
    it is often a different filesystem. So both are reported, each with the
    warning state already computed, rather than a single "disk" number that is
    true of neither. A path we cannot stat is reported as null rather than zero:
    "unknown" and "full" are different answers.
    """
    import shutil as _shutil

    home = _ainode_home_path()
    models = Path(getattr(config, "models_dir", "") or (home / "models"))
    out: dict = {}
    for key, path in (("home", home), ("models", models)):
        try:
            usage = _shutil.disk_usage(str(path))
        except (OSError, ValueError):
            out[key] = {"path": str(path), "total_gb": None, "free_gb": None,
                        "free_fraction": None, "warn": False}
            continue
        fraction = (usage.free / usage.total) if usage.total else 0.0
        out[key] = {
            "path": str(path),
            "total_gb": round(usage.total / (1024 ** 3), 1),
            "free_gb": round(usage.free / (1024 ** 3), 1),
            "free_fraction": round(fraction, 4),
            "warn": fraction < DISK_WARN_FRACTION,
        }
    return out


async def handle_status(request: web.Request) -> web.Response:
    """Return rich node status."""
    config: NodeConfig = request.app["config"]
    engine = request.app["engine"]
    start_time: float = request.app["start_time"]
    session: Optional[aiohttp.ClientSession] = request.app.get("client_session")

    gpu: Optional[GPUInfo] = detect_gpu()
    gpu_info = asdict(gpu) if gpu else None
    # The node's whole GPU inventory, because this block is what the browser
    # sizes models against: device 0's 32 GB stood for a four-V100 host's 128
    # (#163). Names and counts come from the set; the live figures from the
    # collector, the same source /api/nodes uses, so the two endpoints agree.
    gpus = detect_gpus()
    if gpu_info is not None and gpus is not None:
        gpu_info["name"] = gpus.name
        gpu_info["gpu_count"] = gpus.count
        gpu_info["memory_total_mb"] = gpus.memory_total_mb
        gpu_info["unified_memory"] = gpus.unified_memory
        gpu_info["memory_free_mb"] = None
    if gpu_info is not None:
        gpu_info.setdefault("gpu_count", 1)
        collector = request.app.get("metrics_collector")
        if collector is not None:
            try:
                m = collector.get_gpu_metrics() or {}
                if not m.get("error"):
                    total_mb = m.get("memory_total_mb") or gpu_info.get("memory_total_mb")
                    used_mb = m.get("memory_used_mb")
                    if total_mb:
                        gpu_info["memory_total_mb"] = round(total_mb)
                    gpu_info["gpu_count"] = int(m.get("gpu_count") or gpu_info["gpu_count"])
                    gpu_info["memory_kind"] = m.get("memory_kind")
                    # Free memory is null, not total, when nothing can say how
                    # much is in use: the dashboard draws n/a for an unknown and
                    # a real bar for a measurement (#175, #176).
                    gpu_info["memory_used_mb"] = (round(used_mb) if used_mb is not None
                                                  else None)
                    gpu_info["memory_free_mb"] = (
                        max(0, round(total_mb - used_mb))
                        if (total_mb and used_mb is not None) else None)
                    gpu_info["utilization_percent"] = m.get("utilization_percent")
                    gpu_info["temperature_c"] = m.get("temperature_c")
                    gpu_info["devices"] = m.get("devices") or []
                    if m.get("system_memory_used_mb") is not None:
                        gpu_info["system_memory_used_mb"] = m["system_memory_used_mb"]
            except Exception:
                pass

    engine_ready = False
    models_loaded: list[str] = []

    # Live-probe wins: a vLLM that answers /v1/models with >=1 model right now
    # is the single source of liveness. The latched engine.ready is no longer
    # trusted for status (wait_ready still uses it internally).
    if session is not None:
        try:
            vllm_url = f"http://localhost:{config.api_port}/v1/models"
            async with session.get(vllm_url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models_loaded = [m.get("id", "") for m in data.get("data", [])]
                    engine_ready = len(models_loaded) > 0
        except Exception:
            engine_ready = False
    elif engine is not None:
        # No HTTP session — fall back to the managed engine's health check.
        try:
            hc = engine.health_check()
            models_loaded = hc.get("models_loaded", [])
            engine_ready = bool(hc.get("api_responding")) and len(models_loaded) > 0
        except Exception:
            engine_ready = False

    cluster: ClusterState = request.app["cluster_state"]
    master = cluster.get_master()
    effective_role = cluster.get_cluster_role_for(config.node_id) if config.node_id else "worker"

    # The model being loaded is the ENGINE's own config wherever there is one: a
    # load started through /api/engine/set-model runs ahead of config.model.
    phase = engine_load_phase(engine, engine_ready)
    loading_model = str(getattr(getattr(engine, "config", None), "model", "")
                        or config.model or "")
    progress = await load_progress_payload(request.app, engine, phase, loading_model)

    # What this node knows it should be serving and cannot (#179). A distributed
    # shape whose container is gone and whose peers did not answer the launch's
    # own probe is recorded degraded rather than retried in a loop, and silence
    # was the part of that failure that needed fixing: it surfaces here, in
    # `ainode doctor` and as a dashboard banner. Empty on a healthy node.
    try:
        from ainode.engine.reconcile import degraded_instances
        degraded = degraded_instances()
    except Exception:
        logger.exception("could not read the distributed record")
        degraded = []

    return web.json_response({
        "node_id": config.node_id,
        "node_name": config.node_name,
        "model": config.model,
        "gpu": gpu_info,
        "engine_ready": engine_ready,
        # Coarse engine load phase for the UI launching card (3c), derived from
        # the live /v1/models probe above rather than the engine's own latch:
        # see engine_load_phase.
        "load_phase": phase,
        # How far into the launch this node is. The browser cannot time a load it
        # did not watch start (a page opened mid-launch, or a second browser), so
        # the clock comes from here and the page only ticks between polls. All
        # three are null when nothing is loading.
        "load_started_at": progress["load_started_at"],
        "load_elapsed_seconds": progress["load_elapsed_seconds"],
        "expected_ready_minutes": progress["expected_ready_minutes"],
        "uptime": round(time.time() - start_time, 1),
        # Free/total on the AINode home and the models dir, each with its own
        # warning state (see disk_fields). The dashboard draws it on the node the
        # topology has selected; a full models filesystem is the one resource
        # failure that otherwise only shows up as a launch that died.
        "disk": disk_fields(config),
        "version": __version__,
        "powered_by": "ainode.dev",
        "models_loaded": models_loaded,
        "api_port": config.api_port,
        "cluster_role": effective_role,
        "cluster_id": getattr(config, "cluster_id", "default"),
        "master_node_id": master.node_id if master else None,
        # Where else this fleet answers, for free, to a client that already polls
        # status. Routing is replicated across every node but the ADDRESS a client
        # holds is not, so a client that never reads this is stranded by an outage
        # on one node while five others could have served it. Same rows as
        # GET /api/cluster/endpoint (which needs no key, for the client that
        # cannot reach this node at all).
        "endpoint_hint": endpoint_nodes(request.app, request),
        # Say out loud how this port is protected. The default is open, which is
        # fine on a private network and is what Jason runs, but a dashboard that
        # never mentions it is a dashboard that lets you believe otherwise. The
        # header reads `label` straight out of here.
        "auth": auth_status_fields(request.app),
        # Recorded shapes this node cannot serve right now, with the reason.
        "degraded_instances": degraded,
        # Whether anything here speaks HTTPS, and until when. The dashboard's API
        # access panel used to state "there is no TLS on these ports" as a fact;
        # it reads this instead, so the panel cannot be wrong once a node has a
        # certificate.
        "tls": tls_status_fields(config),
        # The per-client limits, in the same words the doctor prints.
        "rate_limit": rate_limit_status_fields(request.app),
    })


def tls_status_fields(config: NodeConfig) -> dict:
    """Whether this node serves HTTPS, on which port, and until when.

    One place, so /api/status, the dashboard panel and ``ainode doctor`` cannot
    end up saying three different things. ``cert_expires`` is the earliest expiry
    in the certificate file (see ``tls.certs.certificate_info``) and is null
    whenever TLS is off or the file cannot be read, never a guess.
    """
    tls: TLSConfig = load_tls_config(config)
    fields: dict = {
        "enabled": bool(tls.enabled),
        "port": int(tls.port),
        "cert_file": tls.cert_file,
        "cert_expires": None,
        "cert_days_left": None,
        "self_signed": None,
        "label": "HTTP only, no TLS on this node",
    }
    if not tls.enabled:
        return fields
    if not tls.cert_file:
        fields["label"] = "TLS enabled with no certificate configured"
        return fields
    info = certificate_info(tls.cert_file)
    fields["cert_expires"] = info.get("expires")
    fields["cert_days_left"] = info.get("days_left")
    fields["self_signed"] = info.get("self_signed")
    if not info.get("exists"):
        fields["label"] = f"TLS enabled but {tls.cert_file} is not there"
    elif info.get("error"):
        fields["label"] = f"TLS enabled, certificate unreadable: {info['error']}"
    else:
        kind = "self-signed" if info.get("self_signed") else "CA-issued"
        fields["label"] = f"HTTPS on {tls.port}, {kind} certificate"
    return fields


def auth_status_fields(app: web.Application) -> dict:
    """How this node's API is protected, in the words the header shows.

    One place, so /api/status, the dashboard header and the installer's summary
    line cannot end up saying three different things.
    """
    auth_cfg: Optional[AuthConfig] = app.get("auth_config")
    enabled = bool(getattr(auth_cfg, "enabled", False))
    key_count = len(getattr(auth_cfg, "api_keys", []) or [])
    if enabled:
        label = "API key required"
    elif key_count:
        label = "API open, key set but not required"
    else:
        label = "API open, no key set"
    return {"enabled": enabled, "key_count": key_count, "label": label}


def _node_host(node, local_id: Optional[str]) -> str:
    """The address a caller can REACH this node on.

    ``/api/nodes`` hardcoded "localhost" for every row, so any link a consumer
    built from it pointed at the viewer's own machine (#178). "localhost" is
    right for exactly one node. For a peer the reachable address is the one its
    announcement arrived from (the listener captures it with ``recvfrom``, which
    is the management-LAN address a browser is on), and the fabric IP when
    nothing else is known: the address the master itself launches over.

    The peer half is ``server_routes.peer_host``, which is the same order plus
    the two guards a client endpoint needs (a loopback or placeholder value is
    dropped rather than published, and a node with neither address falls back to
    its name). One derivation, two readers: this row and /api/cluster/endpoint.
    """
    if local_id and node.node_id == local_id:
        return "localhost"
    return peer_host(node)


async def handle_nodes(request: web.Request) -> web.Response:
    """Return the list of known cluster nodes.

    ``engine_ready`` on each row means what it says: that node's engine answers
    ``/v1/models`` right now. It used to be derived from ``status``, which is
    HEARTBEAT health (online/stale/offline) and true of any node whose
    orchestrator is up -- so a node whose engine was still loading weights was
    published as engine_ready with an empty ``loaded_models`` (#112). The local
    node answers from a live localhost probe; a peer answers from the engine
    state it broadcasts, which its own sync loop sets from the same probe.
    """
    config: NodeConfig = request.app["config"]
    cluster: ClusterState = request.app["cluster_state"]

    cluster_nodes = cluster.get_nodes(include_offline=False)
    collector = request.app.get("metrics_collector")
    local_id = config.node_id
    master = cluster.get_master()
    if cluster_nodes:
        nodes_list = []
        for n in cluster_nodes:
            status_str = n.status.value if hasattr(n.status, "value") else str(n.status)
            dmode = getattr(n, "distributed_mode", "solo") or "solo"
            # Members run no local vLLM, so there is no engine port to answer:
            # they are "ready for work" once discovered.
            if dmode == "member":
                ready = True
            elif n.node_id == local_id:
                # Our own engine is one localhost probe away, so never answer
                # from anything latched or announced.
                ready = await _engine_port_serving(request.app, n.api_port)
            else:
                ready = (getattr(n, "engine_status", "") or "").lower() == "serving"
            # Live load progress per node, so an instance loading on ANOTHER node
            # shows as loading on this dashboard instead of just "not ready". Our
            # own row is timed here; a peer's row is what the peer announced (see
            # peer_load_progress for why its own arithmetic is the honest one).
            if n.node_id == local_id:
                load_phase = engine_load_phase(request.app.get("engine"), ready)
                load_progress = await load_progress_payload(
                    request.app, request.app.get("engine"), load_phase,
                    n.model or config.model or "")
            else:
                load_phase = str(getattr(n, "load_phase", "") or "")
                load_progress = peer_load_progress(n)
            # Live GPU telemetry: peers come from their broadcast; the local
            # node's ClusterNode is built once at startup, so read it fresh
            # from our own collector here. (metrics fan-out)
            #
            # Every one of these is Optional. A figure the node could not measure
            # travels as null and is drawn as n/a: a 0 published here read as an
            # idle GPU on a node that was serving, on every node in the fleet
            # (#176), and a percentage computed from host RAM read as a full node
            # whatever the user did (#175).
            used_mb = optional_float(getattr(n, "gpu_memory_used_mb", None))
            total_mb = optional_float(getattr(n, "gpu_memory_total_mb", None))
            util = optional_float(getattr(n, "gpu_utilization", None))
            temp = optional_float(getattr(n, "gpu_temp", None))
            gpu_count = int(getattr(n, "gpu_count", 1) or 1)
            memory_kind = "unified" if n.unified_memory else "dedicated"
            if n.node_id == local_id and collector is not None:
                try:
                    m = collector.get_gpu_metrics() or {}
                    if not m.get("error"):
                        used_mb = optional_float(m.get("memory_used_mb"))
                        total_mb = optional_float(m.get("memory_total_mb")) or total_mb
                        util = optional_float(m.get("utilization_percent"))
                        temp = optional_float(m.get("temperature_c"))
                        gpu_count = int(m.get("gpu_count") or gpu_count)
                        memory_kind = str(m.get("memory_kind") or memory_kind)
                except Exception:
                    pass
            if not total_mb and n.gpu_memory_gb:
                total_mb = float(n.gpu_memory_gb) * 1024
            used_pct = (round(used_mb / total_mb * 100)
                        if (used_mb is not None and total_mb) else None)
            nodes_list.append({
                "node_id": n.node_id,
                "node_name": n.node_name,
                # The release that node runs (#171). Our own row answers from the
                # process serving this request; a peer's is what it announced, and
                # "" means a peer too old to announce one, never agreement.
                "ainode_version": (__version__ if n.node_id == local_id
                                   else (getattr(n, "ainode_version", "") or "")),
                "fabric_ip": getattr(n, "fabric_ip", "") or "",
                # The address a caller can actually reach this node on. It was
                # "localhost" for every row, so anything built from it pointed at
                # the viewer's own machine (#178).
                "host": _node_host(n, local_id),
                "api_port": n.api_port,
                "web_port": n.web_port,
                "model": n.model,
                "gpu_name": n.gpu_name,
                "gpu_count": gpu_count,
                "gpu_memory_gb": n.gpu_memory_gb,
                "unified_memory": n.unified_memory,
                "memory_kind": memory_kind,
                "gpu_memory_used_pct": used_pct,
                "gpu_memory_used_mb": (round(used_mb) if used_mb is not None else None),
                "gpu_memory_total_mb": (round(total_mb) if total_mb else None),
                "gpu_utilization": (round(util) if util is not None else None),
                "gpu_temp": (round(temp) if temp is not None else None),
                "status": status_str,
                # The role the cluster ELECTED, from the same get_master() the
                # Config view's members table reads. Without it the topology
                # crowned whichever node you happened to open the dashboard on
                # (#203).
                "effective_role": cluster.get_cluster_role_for(n.node_id),
                "is_leader": bool(master is not None and master.node_id == n.node_id),
                "engine_ready": ready,
                "load_phase": load_phase,
                "load_started_at": load_progress["load_started_at"],
                "load_elapsed_seconds": load_progress["load_elapsed_seconds"],
                "expected_ready_minutes": load_progress["expected_ready_minutes"],
                "distributed_mode": dmode,
                "distributed_instance_id": getattr(n, "distributed_instance_id", None),
                "distributed_peers": list(getattr(n, "distributed_peers", []) or []),
                # Per-node instance list (primary + any stacked models on ports
                # 8001+). The node card renders a sub-row per stacked instance so
                # they're no longer invisible in the dashboard. Same source the
                # proxy's _routing_candidates uses, so views and routing agree.
                "instances": [
                    {"model": inst.get("model"),
                     "api_port": inst.get("api_port"),
                     "status": inst.get("status"),
                     # Real launch width (#92): a distributed instance spans
                     # several nodes and must not read as single-GPU here. An
                     # older peer sends no field, which reads as 1.
                     "tensor_parallel_size": instance_parallel(inst)}
                    for inst in (getattr(n, "instances", []) or [])
                    if isinstance(inst, dict) and inst.get("model")
                ],
            })
    else:
        # Fallback: return this node. Same rule as above -- a live probe, not the
        # `ready` latch, which never flips back False when an engine dies and
        # reads True before one is serving.
        dmode = getattr(config, "distributed_mode", "solo") or "solo"
        engine_ready = await _engine_port_serving(request.app, config.api_port)
        engine = request.app.get("engine")
        load_phase = engine_load_phase(engine, engine_ready)
        load_progress = await load_progress_payload(
            request.app, engine, load_phase, config.model or "")
        nodes_list = [{
            "node_id": config.node_id,
            "node_name": config.node_name,
            "ainode_version": __version__,
            "host": "localhost",
            "api_port": config.api_port,
            "web_port": config.web_port,
            "model": config.model,
            # This branch is the no-cluster case (discovery off, or nothing
            # discovered yet): one node, no election to report, and it is the
            # only node there is. Saying so beats leaving the topology to pick a
            # crown out of row order (#203).
            "effective_role": "master",
            "is_leader": True,
            "engine_ready": engine_ready or dmode == "member",
            "load_phase": load_phase,
            "load_started_at": load_progress["load_started_at"],
            "load_elapsed_seconds": load_progress["load_elapsed_seconds"],
            "expected_ready_minutes": load_progress["expected_ready_minutes"],
            "distributed_mode": dmode,
        }]
    return web.json_response({"nodes": nodes_list})

async def _cluster_dispatch(request: web.Request, path: str):
    """F2: forward a load/unload to a node's local /api/models endpoint.

    node_id == this node → call the local handler directly (back-compat). Remote →
    POST over the fabric to http://<fabric_ip>:<web_port><path>. Reuses each node's
    existing /api/models/load|unload; the master never SSHes.
    """
    config: NodeConfig = request.app["config"]
    cluster = request.app.get("cluster_state")
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    node_id = (body.get("node_id") or "").strip()
    if not node_id or node_id == config.node_id:
        # Local: hand the body to the local model handler unchanged.
        from ainode.models.api_routes import handle_model_load, handle_model_unload

        class _Shim:
            def __init__(self, orig, b):
                self._o, self._b = orig, b
            def __getattr__(self, k):
                return getattr(self._o, k)
            async def json(self):
                return self._b
        handler = handle_model_load if path.endswith("/load") else handle_model_unload
        return await handler(_Shim(request, body))

    node = cluster.get_node(node_id) if cluster is not None else None
    host = (getattr(node, "fabric_ip", "") or "") if node else ""
    if not host:
        return web.json_response(
            {"error": f"node '{node_id}' not found or has no fabric IP"}, status=404)
    url = f"http://{host}:{node.web_port}{path}"
    session: aiohttp.ClientSession = request.app["client_session"]
    fwd = {k: v for k, v in body.items() if k != "node_id"}
    try:
        async with session.post(url, json=fwd, timeout=aiohttp.ClientTimeout(total=60)) as up:
            data = await up.read()
            # web.Response rejects a content_type carrying a charset; the node's
            # json_response sends "application/json; charset=utf-8" — strip it.
            ctype = up.headers.get("Content-Type", "application/json").split(";")[0].strip()
            return web.Response(status=up.status, body=data, content_type=ctype)
    except aiohttp.ClientError as exc:
        return web.json_response(
            {"error": f"failed to reach node '{node_id}' at {url}: {exc}"}, status=502)


async def handle_cluster_load(request: web.Request) -> web.Response:
    """POST /api/cluster/load {node_id, model} — load a model on any node (F2)."""
    return await _cluster_dispatch(request, "/api/models/load")


async def handle_cluster_unload(request: web.Request) -> web.Response:
    """POST /api/cluster/unload {node_id, model} — unload on any node (F2)."""
    return await _cluster_dispatch(request, "/api/models/unload")


def _routing_candidates(cluster, model: str, local_node_id: str, local_port: int) -> list:
    """All (host, port) currently serving `model` (routing-truth).

    Returns a LIST so proxy_to_vllm can fail over when the first target is a
    stale/ghost claim (a node that crashed but still advertises the model). A
    crashed node is indistinguishable from a live one in cluster state, so
    failover — not ordering — is what makes routing robust. Local node first
    (cheapest hop), then remote peers.
    """
    local, remote = [], []
    for n in (cluster.members() if cluster is not None else []):
        status = n.status.value if hasattr(n.status, "value") else str(n.status)
        if status not in ("online", "serving", "member-ready"):
            continue
        is_local = n.node_id == local_node_id
        host = "localhost" if is_local else (getattr(n, "fabric_ip", "") or "")
        if not host:
            continue
        node_port = local_port if is_local else n.api_port
        bucket = local if is_local else remote
        seen = set()
        # The node's primary/solo model is served on its main api_port.
        if getattr(n, "model", "") == model and node_port not in seen:
            bucket.append((host, node_port))
            seen.add(node_port)
        # Each stacked instance is served on its OWN port — a co-resident 2nd
        # model on this node lives at :8001, not the node's main :8000.
        for inst in (getattr(n, "instances", []) or []):
            if inst.get("model") != model:
                continue
            iport = inst.get("api_port") or node_port
            if iport not in seen:
                bucket.append((host, iport))
                seen.add(iport)
    return local + remote


def _routing_table(cluster, local_node_id: str, local_port: int) -> dict:
    """model name → (host, port) for every model served across the fleet (F1).

    Built from cluster broadcast state: each node advertises its solo model and
    any instances it heads. The local node routes to localhost; remote nodes to
    their fabric IP (reachable from the master over the cluster fabric).
    """
    table: dict = {}
    for n in cluster.members():
        status = n.status.value if hasattr(n.status, "value") else str(n.status)
        if status not in ("online", "serving", "member-ready"):
            continue
        is_local = n.node_id == local_node_id
        host = "localhost" if is_local else (getattr(n, "fabric_ip", "") or "")
        if not host:
            continue
        port = local_port if is_local else n.api_port
        if getattr(n, "model", ""):
            table.setdefault(n.model, (host, port))
        for inst in (getattr(n, "instances", []) or []):
            m = inst.get("model")
            if m:
                table.setdefault(m, (host, inst.get("api_port") or port))
    return table


async def handle_v1_models(request: web.Request) -> web.Response:
    """Federated /v1/models — the UNION of models served across the fleet (F1)."""
    config: NodeConfig = request.app["config"]
    cluster = request.app.get("cluster_state")
    table = _routing_table(cluster, config.node_id, config.api_port) if cluster is not None else {}
    if not table and config.model:
        table = {config.model: ("localhost", config.api_port)}
    data = [{"id": m, "object": "model", "owned_by": "ainode"} for m in sorted(table)]
    return web.json_response({"object": "list", "data": data})


def _error_message(text: str) -> Optional[str]:
    """The engine's own words out of a JSON error body, trimmed for a cache note."""
    msg = None
    try:
        import json as _json
        payload = _json.loads(text)
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                msg = err.get("message")
            elif isinstance(err, str):
                msg = err
            msg = msg or payload.get("message")
    except Exception:
        pass
    msg = " ".join(str(msg or text or "").split())
    return msg[:300] or None


def _no_multimodal_instance(model: str, tried: list) -> web.Response:
    """400 for a multimodal request no instance of a loaded model will take.

    Named nodes, not a bare refusal: the fix is to relaunch one of them without
    the modality capped (``--limit-mm-per-prompt``), and the caller can only know
    that if we say which instances were tried.
    """
    where = ", ".join(dict.fromkeys(tried)) or "no instance"
    return web.json_response(
        {"error": {
            "message": (f"'{model}' is loaded but no instance accepts images: "
                        f"tried {where}. Relaunch one of those instances without "
                        f"the modality capped (--limit-mm-per-prompt), or load "
                        f"the model on a node that serves it with vision."),
            "type": "invalid_request_error",
            "param": "image",
            "code": "no_multimodal_instance"}},
        status=400,
    )


async def proxy_to_vllm(request: web.Request) -> web.StreamResponse:
    """Forward the request to the node serving the requested model (F1 federation)."""
    config: NodeConfig = request.app["config"]
    session: aiohttp.ClientSession = request.app["client_session"]
    collector: MetricsCollector = request.app["metrics_collector"]
    # Extract the model name first — it drives BOTH routing and metrics.
    model = config.model or "unknown"
    body_bytes = None
    body_obj: dict = {}
    if request.method == "POST":
        body_bytes = await request.read()
        try:
            import json as _json
            parsed = _json.loads(body_bytes)
            if isinstance(parsed, dict):
                body_obj = parsed
                model = parsed.get("model", model)
        except Exception:
            pass
    # Tag the request so the server-view log middleware can capture the model
    try:
        request["_log_model"] = model
    except Exception:
        pass

    # Federated routing with failover (F1 + routing-truth): try every node that
    # serves this model (ready ones first), so a stale/ghost claim from a crashed
    # node doesn't 502 a request another node can serve. Built from cluster state.
    cluster = request.app.get("cluster_state")
    # An explicit pin wins over routing-by-model-id: the caller picked an
    # instance, so that instance answers or nothing does (#197).
    pin_node, pin_port = pinned_target(request.headers, body_obj)
    pinned = None
    if pin_node or pin_port:
        pinned = pinned_candidate(cluster, config, pin_node, pin_port)
        if pinned is None:
            collector.record_request(model, 0.0, error=True)
            return web.json_response(
                {"error": {"type": "invalid_request_error",
                           "message": (f"pinned node '{pin_node}' is not a member of this "
                                       f"cluster, or has no fabric IP"),
                           "code": "unknown_node"}},
                status=404)
        # The body field is ours, not vLLM's: strip it before forwarding. Only a
        # request that actually used it pays for the re-serialize.
        if isinstance(body_obj, dict) and TARGET_BODY_FIELD in body_obj:
            body_obj.pop(TARGET_BODY_FIELD, None)
            body_bytes = json.dumps(body_obj).encode("utf-8")
    if pinned is not None:
        candidates = [pinned]
    else:
        candidates = _routing_candidates(cluster, model, config.node_id, config.api_port)
    if not candidates:
        if model and model != "unknown" and cluster is not None and cluster.members():
            return web.json_response(
                {"error": {"type": "model_not_found",
                           "message": f"Model '{model}' is not loaded on any node",
                           "code": "model_not_found"}},
                status=404)
        candidates = [("localhost", config.api_port)]  # back-compat: empty fleet → local

    # Capability-aware routing (#83). The same model id can be served text-only
    # on one node (launched with --limit-mm-per-prompt '{"image":0}') and with
    # vision on another, so a request carrying an image part cannot be routed on
    # the model id alone: the text-only instance answers 400 and the caller sees
    # a failure the fleet could have avoided. Consult the capability cache
    # /api/models/caps already fills: accepting instances first, never-probed
    # next, known refusers dropped. A request with no media keeps today's order
    # exactly (local hop first, then peers).
    multimodal = is_multimodal_request(body_obj)
    caps_index: dict = {}
    refused_labels: list = []
    if multimodal:
        caps_index = instance_caps_index(request.app, model)
        # A pinned request has one candidate by definition: reordering or
        # dropping it would substitute our capability cache for the user's own
        # choice. The engine's answer is the honest one there.
        if pinned is None:
            candidates, refused = order_by_vision(candidates, caps_index)
            refused_labels = [node_label(caps_index, c) for c in refused]
            if not candidates:
                collector.record_request(model, 0.0, error=True)
                return _no_multimodal_instance(model, refused_labels)

    # Build upstream request kwargs. Strip content-length: aiohttp recomputes it
    # from `data`, and forwarding the original alongside makes the upstream wait
    # for a body that never arrives (the proxy hangs). Strip host/transfer-encoding
    # for the usual reverse-proxy reasons.
    kwargs: dict = {
        "headers": {k: v for k, v in request.headers.items()
                    if k.lower() not in ("host", "transfer-encoding", "content-length")},
        # Fast failover: a dead/ghost node must fail the CONNECT quickly so the
        # loop moves on to the next candidate — but leave total uncapped so a live
        # node's slow cold-start generation (35s+) can still stream to completion.
        "timeout": aiohttp.ClientTimeout(total=None, sock_connect=5),
    }
    if body_bytes is not None:
        kwargs["data"] = body_bytes

    start_time = time.time()
    last_err = None
    # path_qs, not path: Claude Code posts to `/v1/messages?beta=true`, and a
    # proxy that drops a caller's query string is guessing on the caller's behalf.
    target = request.path_qs
    for host, port in candidates:
        vllm_url = f"http://{host}:{port}{target}"
        # Which engine answered, in the caller's hands rather than inferred from
        # the pick: routing can fail over, and a stats record that names the
        # wrong node is worse than one that names none (#197).
        served_by = f"{host}:{port}"
        try:
            async with session.request(request.method, vllm_url, **kwargs) as upstream:
                is_sse = "text/event-stream" in upstream.headers.get("Content-Type", "")
                if is_sse:
                    resp = web.StreamResponse(
                        status=upstream.status,
                        headers={
                            "Content-Type": "text/event-stream",
                            "Cache-Control": "no-cache",
                            "X-Accel-Buffering": "no",
                            SERVED_BY_HEADER: served_by,
                        },
                    )
                    await resp.prepare(request)
                    async for chunk in upstream.content.iter_any():
                        await resp.write(chunk)
                    await resp.write_eof()
                    collector.record_request(model, (time.time() - start_time) * 1000, error=False)
                    return resp
                body = await upstream.read()
                # A multimodal-limit 400 is a ROUTING miss, not a bad request:
                # this instance serves the model with the modality capped at
                # zero. Remember it in the caps cache and fail over, same as a
                # dead connection. Every OTHER 4xx is the caller's answer and is
                # returned untouched: we never retry those.
                if multimodal and upstream.status == 400:
                    text = body.decode("utf-8", "replace")
                    if is_multimodal_limit_error(text):
                        label = node_label(caps_index, (host, port))
                        record_vision_unsupported(
                            request.app, model, host, port,
                            reason=_error_message(text))
                        refused_labels.append(label)
                        last_err = f"{label} serves '{model}' without images"
                        continue
                collector.record_request(model, (time.time() - start_time) * 1000, error=False)
                return web.Response(
                    status=upstream.status, body=body,
                    headers={SERVED_BY_HEADER: served_by},
                    content_type=upstream.headers.get("Content-Type", "application/json").split(";")[0].strip(),
                )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_err = exc  # unreachable / connect-timeout (likely a ghost) — try the next
            continue
    # Every candidate failed.
    collector.record_request(model, (time.time() - start_time) * 1000, error=True)
    if multimodal and refused_labels:
        # The model IS loaded; no instance of it takes images. Say so, and name
        # the nodes tried, so the caller knows which engine to relaunch.
        return _no_multimodal_instance(model, refused_labels)
    return web.json_response(
        {"error": {"message": f"no reachable node is serving '{model}' ({last_err})",
                   "type": "server_error"}},
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
    # GPUs, not nodes. Counting nodes reported the fleet's nine GPUs as six,
    # because the two x86 boxes hold four and one V100 (#163).
    total_gpus = sum(int(getattr(n, "gpu_count", 1) or 1) for n in ready)

    # What is actually free. This was a copy of the total, so the cluster view
    # showed an empty fleet however many models were loaded (#174). Per node:
    # total minus what that node reports in use (NVML on a discrete GPU, the
    # engines' reservations on a unified-memory part). A node with no usable
    # figure is NOT counted as free. It is named in `vram_unknown_nodes`, and
    # the available total says it is a floor rather than the whole answer.
    local_metrics: dict = {}
    _collector = request.app.get("metrics_collector")
    if _collector is not None:
        try:
            local_metrics = _collector.get_gpu_metrics() or {}
        except Exception:
            local_metrics = {}
    local_id = getattr(request.app.get("config"), "node_id", None)

    def _node_usage(node) -> tuple[Optional[float], Optional[float]]:
        """(used_gb, total_gb) for one node, either None when unknown."""
        total_gb = float(getattr(node, "gpu_memory_gb", 0) or 0) or None
        if node.node_id == local_id and local_metrics and not local_metrics.get("error"):
            used_mb = optional_float(local_metrics.get("memory_used_mb"))
            total_mb = optional_float(local_metrics.get("memory_total_mb"))
            if total_mb:
                total_gb = total_mb / 1024
            return ((used_mb / 1024 if used_mb is not None else None), total_gb)
        used_mb = optional_float(getattr(node, "gpu_memory_used_mb", None))
        return ((used_mb / 1024 if used_mb is not None else None), total_gb)

    available_vram = 0.0
    vram_unknown_nodes: list = []
    node_usage: dict = {}
    for n in ready:
        used_gb, total_gb = _node_usage(n)
        node_usage[n.node_id] = (used_gb, total_gb)
        if used_gb is None or total_gb is None:
            vram_unknown_nodes.append(n.node_name or n.node_id)
            continue
        available_vram += max(0.0, total_gb - used_gb)

    # Phase 2: the LIST of distributed instances across all nodes — each head
    # advertises the instances it heads. Resolve each instance's peer FABRIC IPs
    # (BUG D) back to member node ids/names. `distributed_instance` (singular)
    # stays = the first one, for one release of back-compat.
    by_fabric = {
        (getattr(m, "fabric_ip", "") or ""): m
        for m in cluster.members() if getattr(m, "fabric_ip", "")
    }

    def _resolve_instance(head, inst):
        iid = inst.get("instance_id", "") or ""
        peers = list(inst.get("peer_ips", []) or [])
        peer_node_ids, member_names = [], [head.node_name]
        for ip in peers:
            m = by_fabric.get(ip)
            peer_node_ids.append(m.node_id if m else ip)
            member_names.append(m.node_name if m else ip)
        # model can be stale ("") if the head started idle; it's authoritative in
        # instance_id ("<node_id>:<model>").
        model = inst.get("model") or (iid.split(":", 1)[1] if ":" in iid else "") or head.model
        return {
            "instance_id": iid,
            "head_node_id": head.node_id,
            "head_node_name": head.node_name,
            "peer_ips": peers,
            "peer_node_ids": peer_node_ids,
            "member_names": member_names,
            "tensor_parallel_size": inst.get("tensor_parallel_size") or (1 + len(peers)),
            "model": model,
            "status": inst.get("status", "serving"),
        }

    distributed_instances = []
    for n in ready:
        node_instances = list(getattr(n, "instances", []) or [])
        if not node_instances:
            # Back-compat: an older node advertises only the singular fields.
            iid = getattr(n, "distributed_instance_id", None)
            if iid:
                node_instances = [{
                    "instance_id": iid,
                    "model": (iid.split(":", 1)[1] if ":" in iid else getattr(n, "model", "")),
                    "peer_ips": list(getattr(n, "distributed_peers", []) or []),
                }]
        for inst in node_instances:
            distributed_instances.append(_resolve_instance(n, inst))

    distributed_instance = distributed_instances[0] if distributed_instances else None

    local_node_id = (cluster._local_announcement.node_id
                     if cluster._local_announcement else "")
    nodes_payload = []
    for n in ready:
        used_gb, total_gb = node_usage.get(n.node_id, (None, None))
        nodes_payload.append({
            "node_id": n.node_id,
            "hostname": n.node_name,
            # Per-node release, so the cluster view can show a split fleet (#171).
            "ainode_version": (__version__ if local_node_id and n.node_id == local_node_id
                               else (getattr(n, "ainode_version", "") or "")),
            "fabric_ip": getattr(n, "fabric_ip", "") or "",
            "host": _node_host(n, local_id),
            "vram_gb": round(float(n.gpu_memory_gb or 0), 1),
            # Every device on the node, not one per node (#163).
            "gpus": int(getattr(n, "gpu_count", 1) or 1),
            "vram_used_gb": (round(used_gb, 1) if used_gb is not None else None),
            "vram_free_gb": (round(max(0.0, total_gb - used_gb), 1)
                             if (used_gb is not None and total_gb is not None) else None),
            "gpu_name": n.gpu_name,
            "unified_memory": n.unified_memory,
            "memory_kind": "unified" if n.unified_memory else "dedicated",
            "effective_role": cluster.get_cluster_role_for(n.node_id),
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
        # Summed from real per-node usage. None when NOT ONE node could say, so
        # the view shows nothing rather than the total twice (#174).
        "available_vram_gb": (round(available_vram, 1)
                              if len(vram_unknown_nodes) < len(ready) else None),
        # The nodes whose usage is unknown. While this list is non-empty the
        # figure above is a floor: those nodes contribute none of their free
        # memory to it.
        "vram_unknown_nodes": vram_unknown_nodes,
        "available_vram_is_floor": bool(vram_unknown_nodes),
        "total_gpus": total_gpus,
        "total_nodes": len(ready),
        "nodes": nodes_payload,
        "distributed_instance": distributed_instance,
        "distributed_instances": distributed_instances,
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


# Cluster update job state. In memory AND on disk, because the master self-stops
# as the last step of the job it is reporting on: the workers were updated and
# recorded, the container died, and the process that came back on the new image
# had an empty dict, so a UI polling update-status for a job that had fully
# succeeded got a 404 (#182). The file lives under AINODE_HOME next to
# instances.json, which is bind-mounted from the host, so it survives the
# restart the job itself causes.
_cluster_update_state: dict = {}

# Most recent jobs kept on disk. A job record is a few hundred bytes; this is
# only here so the file cannot grow without bound on a node that is updated
# weekly for a year.
MAX_PERSISTED_UPDATE_JOBS = 10


def _cluster_updates_path() -> Path:
    return _ainode_home_path() / "cluster-updates.json"


def _load_cluster_updates() -> dict:
    """Merge the on-disk jobs into the in-memory dict and return it.

    Disk loses to memory: a job this process is running has fresher rows than
    anything written before its last mutation.
    """
    try:
        raw = json.loads(_cluster_updates_path().read_text())
    except (OSError, ValueError):
        return _cluster_update_state
    if not isinstance(raw, dict):
        return _cluster_update_state
    for update_id, job in raw.items():
        if isinstance(job, dict) and update_id not in _cluster_update_state:
            _cluster_update_state[update_id] = job
    return _cluster_update_state


def _save_cluster_updates() -> None:
    """Write the job state out. Never raises: a job must not fail over its log."""
    try:
        keep = sorted(_cluster_update_state.keys())[-MAX_PERSISTED_UPDATE_JOBS:]
        snapshot = {k: _cluster_update_state[k] for k in keep}
        path = _cluster_updates_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(snapshot, indent=2))
        tmp.replace(path)
    except Exception:
        logger.debug("could not persist cluster update state", exc_info=True)


async def handle_cluster_update_all(request: web.Request) -> web.Response:
    """POST /api/cluster/update-all — pull latest image and restart on all nodes.

    Workers are updated via HTTP (POST /api/engine/update on each worker's
    API port) — no SSH required. The master updates itself last.
    Poll GET /api/cluster/update-status for live per-node progress.
    """
    import asyncio
    import subprocess as _sp
    cluster: "ClusterState" = request.app["cluster_state"]
    config: NodeConfig = request.app["config"]
    session: aiohttp.ClientSession = request.app["client_session"]

    # Optional target version, threaded to self and every peer so the whole
    # cluster converges on the same image.
    requested = None
    try:
        if request.can_read_body:
            body = await request.json()
            if isinstance(body, dict):
                requested = body.get("version")
    except Exception:
        requested = None

    nodes = cluster.members()
    all_nodes = [{"node_id": config.node_id, "node_name": config.node_id, "host": "localhost", "port": config.web_port or 3000, "is_self": True}]
    for n in nodes:
        if n.node_id != config.node_id:
            peer_ip = getattr(n, "peer_ip", None) or n.host
            port = getattr(n, "web_port", 3000) or 3000
            all_nodes.append({
                "node_id": n.node_id,
                "node_name": getattr(n, "node_name", n.node_id),
                "host": peer_ip,
                "port": port,
                "is_self": False,
            })

    if len(all_nodes) == 0:
        return web.json_response({"error": "No nodes in cluster"}, status=400)

    # Wall clock, not loop.time(): the id has to sort by age across the restart
    # this job causes, and a monotonic clock starts over with the new process.
    update_id = f"update-{int(time.time())}"
    _cluster_update_state[update_id] = {
        "id": update_id,
        "status": "running",
        "target": requested or "",
        "nodes": {n["node_id"]: {"node_name": n["node_name"], "status": "pending",
                                 "log": "", "version_before": _node_version(cluster, config, n["node_id"])}
                  for n in all_nodes},
        "started_at": time.time(),
    }
    _save_cluster_updates()

    # Resolve the target tag once for the whole cluster.
    target = requested
    if not target:
        try:
            target = await asyncio.get_event_loop().run_in_executor(
                None, _fetch_latest_ghcr_tag
            )
        except Exception:
            target = None
    if not target:
        # Mark the just-created job failed instead of leaving it stuck at
        # "running" forever in the in-memory dict (a poll would never resolve).
        job = _cluster_update_state.get(update_id)
        if job is not None:
            job["status"] = "failed"
            for n in job["nodes"].values():
                n["status"] = "failed"
                n["log"] = "Could not resolve a target version from GHCR"
            _save_cluster_updates()
        return web.json_response(
            {"error": "Could not resolve a target version from GHCR"}, status=502
        )
    image = f"{AINODE_GHCR_REPO}:{target}"
    _cluster_update_state[update_id]["target"] = target
    _save_cluster_updates()

    def _mark(state: dict, status: str, log: Optional[str] = None) -> None:
        """One node's outcome, written to disk as it happens.

        Every mutation persists, because the master's own row is written moments
        before it stops its own container: an in-memory-only record of the job
        dies with the process that is reporting on it (#182).
        """
        state["status"] = status
        if log is not None:
            state["log"] = log
        state["updated_at"] = time.time()
        _save_cluster_updates()

    async def _update_node(node: dict) -> None:
        nid = node["node_id"]
        state = _cluster_update_state[update_id]["nodes"][nid]
        _mark(state, "updating")

        if node["is_self"]:
            # Update self: docker pull → write image.env → self-stop (systemd
            # Restart=always relaunches on the new image). systemctl does not
            # work from inside the container.
            try:
                loop = asyncio.get_event_loop()
                try:
                    pull = await loop.run_in_executor(
                        None, lambda: _sp.run(
                            ["docker", "pull", image],
                            capture_output=True, text=True, timeout=600
                        )
                    )
                except _sp.TimeoutExpired:
                    _mark(state, "failed", "docker pull timed out after 600s")
                    return
                if pull.returncode != 0:
                    _mark(state, "failed", pull.stderr[:500])
                    return
                try:
                    _write_image_env(image)
                except Exception as exc:
                    _mark(state, "failed", f"image.env write failed: {exc}"[:200])
                    return
                # Same swappable-unit gate as handle_engine_update: never self-stop
                # a node whose unit predates the swappable image, or we'd drop it /
                # reboot the old image and still report "done". Report honestly.
                if not _unit_is_swappable():
                    _mark(state, "needs-migration", (
                        "Image pulled + pinned, but this node's systemd unit "
                        "predates the swappable unit — not restarted. Re-run the "
                        "installer on the host to migrate."
                    ))
                    return
                # "done" is the outcome of the ACTION (pulled, pinned, restart
                # triggered), written seconds before this container stops. Whether
                # the node came back on the target version is a separate question,
                # answered at poll time from the version on the wire (#182).
                _mark(state, "done", "Updated, restarting on the new image")

                async def _restart_self():
                    await asyncio.sleep(2)
                    await loop.run_in_executor(
                        None, lambda: _sp.run(
                            ["docker", "stop", "ainode"],
                            capture_output=True, text=True, timeout=60
                        )
                    )

                asyncio.get_event_loop().create_task(_restart_self())
            except Exception as exc:
                _mark(state, "failed", str(exc)[:200])
        else:
            # Update remote worker via HTTP — no SSH needed.
            # Each worker runs the same AINode container with /api/engine/update.
            url = f"http://{node['host']}:{node['port']}/api/engine/update"
            try:
                async with session.post(url, json={"version": target}, timeout=aiohttp.ClientTimeout(total=700)) as resp:
                    data = await resp.json()
                    if resp.status < 300:
                        _mark(state, "done",
                              data.get("message", "Updated and restarting"))
                    else:
                        _mark(state, "failed",
                              data.get("error", f"HTTP {resp.status}")[:300])
            except asyncio.TimeoutError:
                _mark(state, "failed",
                      "Timeout, worker may still be pulling the image")
            except Exception as exc:
                _mark(state, "failed", str(exc)[:200])

    async def _run_all():
        workers = [n for n in all_nodes if not n["is_self"]]
        self_node = next((n for n in all_nodes if n["is_self"]), None)

        await asyncio.gather(*[_update_node(n) for n in workers])

        if self_node:
            await _update_node(self_node)

        # "complete" means every node was told to update and none failed on the
        # way. Whether each one CAME BACK on the target version is a live question
        # answered at poll time against the version on the wire, not a claim this
        # record gets to make on its own.
        _cluster_update_state[update_id]["status"] = "complete"
        _cluster_update_state[update_id]["finished_at"] = time.time()
        _save_cluster_updates()

    asyncio.get_event_loop().create_task(_run_all())

    return web.json_response({
        "update_id": update_id,
        "nodes": list(_cluster_update_state[update_id]["nodes"].keys()),
        "message": f"Updating {len(all_nodes)} node(s) via HTTP. Poll /api/cluster/update-status?id={update_id} for progress.",
    }, status=202)


def _node_version(cluster: Optional[ClusterState], config: NodeConfig,
                  node_id: str) -> str:
    """The release *node_id* is running, from the wire. "" when it cannot be told.

    Our own row answers from the running process; a peer from its announcement.
    """
    if node_id and node_id == config.node_id:
        return __version__
    if cluster is None:
        return ""
    node = cluster.get_node(node_id)
    return (getattr(node, "ainode_version", "") or "") if node else ""


async def handle_cluster_update_status(request: web.Request) -> web.Response:
    """GET /api/cluster/update-status?id=... : poll update progress.

    Served from the on-disk record, so a poll that lands after the master
    restarted itself still answers for the job that caused the restart (#182),
    and enriched per node with the version on the wire RIGHT NOW: comparing the
    running release against the target is the real answer to "did the update
    work", and it does not depend on a record written by a process that then
    stopped itself.
    """
    config: NodeConfig = request.app["config"]
    cluster: Optional[ClusterState] = request.app.get("cluster_state")
    jobs = _load_cluster_updates()
    update_id = request.query.get("id", "")
    if not update_id or update_id not in jobs:
        # Return most recent if no ID given. Ids are "update-<unix seconds>", so
        # the newest sorts last for the next two centuries.
        if jobs:
            update_id = max(jobs.keys())
        else:
            return web.json_response({"error": "No update in progress"}, status=404)

    job = jobs[update_id]
    target = str(job.get("target") or "")
    nodes = {}
    on_target = 0
    for nid, row in (job.get("nodes") or {}).items():
        live = _node_version(cluster, config, nid)
        enriched = dict(row)
        enriched["ainode_version"] = live
        # None means "cannot tell": either no target was resolved or that node is
        # not announcing a version (too old, or not back on the wire yet). Never
        # False, which would read as "the update failed".
        matches = (live == target) if (target and live) else None
        enriched["on_target"] = matches
        if matches:
            on_target += 1
        nodes[nid] = enriched

    payload = dict(job)
    payload["nodes"] = nodes
    payload["target"] = target
    payload["nodes_on_target"] = on_target
    payload["nodes_total"] = len(nodes)
    # True only when every node in the job is verifiably on the target version.
    payload["verified"] = bool(target and nodes and on_target == len(nodes))
    payload["persisted"] = str(_cluster_updates_path())
    return web.json_response(payload)


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
# Fields a caller may only set by presenting an API key, even on a node running
# open. trust_remote_code is the whole list: with it on, a load of an
# attacker-named repo executes that repo's modeling_*.py inside the engine
# container (#168), so the unauthenticated PATCH that used to accept it was a
# remote code execution path on port 3000. It stays patchable for a caller that
# holds a key, and a curated catalog entry that declares it still applies per
# load (see models.api_routes.parse_launch_overrides).
AUTHENTICATED_ONLY_CONFIG_FIELDS = {"trust_remote_code"}

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


async def handle_set_model(request: web.Request) -> web.Response:
    """POST /api/engine/set-model: rewrite the BOOT model and restart that engine.

    DESTRUCTIVE, and not a launch route. It stops whatever ``app["engine"]`` is
    serving and restarts that one backend on the new model, with no
    InstanceManager awareness: a node running stacked instances loses the one
    attached to ``app["engine"]`` and keeps the rest unaccounted for. With no
    engine object at all it saves the config and starts NOTHING, while still
    answering ``{"status": "restarting"}``.

    It exists for the boot-config use it was written for (change what
    ``ainode start`` brings up next). Nothing in the UI calls it any more: the
    Models view's detail modal used to, which is how a click on "Launch Model"
    could stop a serving node or do nothing at all and report success either way
    (#189). Launch through ``POST /api/cluster/load`` (one node, stacks) or
    ``POST /api/sharding/launch`` (several).
    """
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    model = (body.get("model") or "").strip()
    if not model or "/" not in model:
        return web.json_response({"error": "model must be a HF repo ID (org/name)"}, status=400)

    config: NodeConfig = request.app["config"]
    engine = request.app.get("engine")

    # Persist the new model choice
    config.model = model
    config.save()

    # Stop current engine and start fresh with new model. Under the node's launch
    # slot: this is a launch like any other, and two engines profiling at once
    # under-provision the second one's KV cache (#96).
    if engine is not None:
        from ainode.models.api_routes import (
            LaunchBusy,
            acquire_launch_slot,
            hold_launch_slot_until_bound,
            launch_busy_error,
            release_launch_slot,
        )

        slot_label = f"set-model {model}"
        try:
            await acquire_launch_slot(slot_label)
        except LaunchBusy as busy:
            refused = launch_busy_error(busy)
            return web.json_response({"error": refused["error"]}, status=refused["status"])
        handed_off = False
        try:
            try:
                engine.stop()
            except Exception:
                pass
            try:
                engine.config.model = model
                engine.start()
            except Exception as exc:
                return web.json_response({"error": str(exc)}, status=500)
            asyncio.get_event_loop().create_task(hold_launch_slot_until_bound(
                request.app, getattr(config, "api_port", 8000), engine, slot_label))
            handed_off = True
        finally:
            if not handed_off:
                release_launch_slot()

    return web.json_response({"status": "restarting", "model": model})


AINODE_GHCR_REPO = "ghcr.io/getainode/ainode"


def _fetch_latest_ghcr_tag() -> Optional[str]:
    """Return the highest numeric GHCR version tag, or None.

    Blocking (uses urllib); call via ``run_in_executor``. Resolves anonymously
    against the public image — no auth required.
    """
    import urllib.request
    import json as _json

    token_url = "https://ghcr.io/token?service=ghcr.io&scope=repository:getainode/ainode:pull"
    with urllib.request.urlopen(token_url, timeout=5) as r:
        token = _json.loads(r.read())["token"]
    tags_url = "https://ghcr.io/v2/getainode/ainode/tags/list"
    req = urllib.request.Request(tags_url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=5) as r:
        data = _json.loads(r.read())
    # Highest numeric version tag (ignore 'latest' and non-numeric tags).
    versions = sorted(
        [t for t in data.get("tags", []) if t and t != "latest" and t[0].isdigit()],
        key=lambda v: tuple(int(x) for x in v.split(".") if x.isdigit()),
        reverse=True,
    )
    return versions[0] if versions else None


def _ainode_home_path() -> Path:
    """Host-mounted AINode home (``~/.ainode`` bind-mounted to /root/.ainode)."""
    return Path(os.environ.get("AINODE_HOME", str(Path.home() / ".ainode")))


def _write_image_env(image: str) -> None:
    """Persist the target image for the host systemd unit's EnvironmentFile.

    Inside the container this writes ``$AINODE_HOME/image.env`` which is the
    same file the host unit reads via the ``~/.ainode`` bind mount.

    Written atomically (temp file + rename) so the host unit never reads a
    half-written line if it happens to (re)load while we're mid-write.
    """
    home = _ainode_home_path()
    home.mkdir(parents=True, exist_ok=True)
    tmp = home / "image.env.tmp"
    tmp.write_text(f"AINODE_IMAGE={image}\n")
    tmp.replace(home / "image.env")


def _unit_is_swappable() -> bool:
    """True iff this container was launched by the swappable-image systemd unit.

    The swappable unit (the one this deploy path relies on) stamps
    ``AINODE_UNIT_SWAPPABLE=1`` into the container env. Its ABSENCE means the
    host is still on a pre-swappable unit (pinned ExecStart, no EnvironmentFile),
    where a self-``docker stop`` is destructive rather than an image swap: the
    old unit either won't relaunch at all (Restart=on-failure + clean exit) or
    relaunches the SAME pinned image (it never reads image.env). In that case we
    must NOT self-stop — pull + pin image.env, and tell the operator to migrate
    the unit on the host first.
    """
    return os.environ.get("AINODE_UNIT_SWAPPABLE") == "1"


def _version_key(value: str) -> tuple:
    """Sort key for a release string. Unparseable parts sort low, never raise."""
    parts = []
    for chunk in str(value or "").split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def cluster_versions(app: web.Application) -> dict:
    """Which release each node in the cluster is running, and whether they agree.

    The answer for THIS node is the version of the process serving the request; a
    peer's is what it put on the wire. A peer too old to announce one reports ""
    and is counted in ``unknown_versions`` rather than folded into agreement:
    absence is not a match. ``cluster_split`` is the thing a roll needs to see
    (#171) and the honest answer to "did the update land", which is why
    /api/version/check and /api/cluster/update-status both serve it.
    """
    config: NodeConfig = app["config"]
    cluster: Optional[ClusterState] = app.get("cluster_state")
    rows = []
    for n in (cluster.get_nodes(include_offline=False) if cluster else []):
        rows.append({
            "node_id": n.node_id,
            "node_name": n.node_name,
            "ainode_version": (__version__ if n.node_id == config.node_id
                               else (getattr(n, "ainode_version", "") or "")),
        })
    if not rows:
        rows = [{
            "node_id": config.node_id,
            "node_name": config.node_name,
            "ainode_version": __version__,
        }]
    known = sorted({r["ainode_version"] for r in rows if r["ainode_version"]},
                   key=_version_key)
    return {
        "nodes": rows,
        "versions": known,
        "unknown_versions": sum(1 for r in rows if not r["ainode_version"]),
        "cluster_split": len(known) > 1,
    }


async def handle_version_check(request: web.Request) -> web.Response:
    """GET /api/version/check : compare the local version against the latest GHCR tag.

    Also reports the version of every node in the cluster: the master knowing it
    is stale said nothing about the other five nodes disagreeing with each other
    (#171), which is exactly the state a partial roll leaves.
    """
    current = __version__
    try:
        loop = asyncio.get_event_loop()
        latest = await loop.run_in_executor(None, _fetch_latest_ghcr_tag)
    except Exception:
        latest = None

    update_available = False
    if latest and latest != current:
        try:
            cur_parts = tuple(int(x) for x in current.split(".") if x.isdigit())
            lat_parts = tuple(int(x) for x in latest.split(".") if x.isdigit())
            update_available = lat_parts > cur_parts
        except Exception:
            update_available = latest != current

    payload = {
        "current": current,
        "latest": latest,
        "update_available": update_available,
    }
    payload.update(cluster_versions(request.app))
    return web.json_response(payload)


async def handle_engine_update(request: web.Request) -> web.Response:
    """POST /api/engine/update — pull a target image and swap to it.

    Body (optional): ``{"version": "X.Y.Z"}``. When omitted, targets the highest
    numeric GHCR tag. On a successful ``docker pull`` we write ``image.env``
    (which the host systemd unit reads via EnvironmentFile) and then, after a
    short delay, ``docker stop ainode`` on the mounted socket — systemd's
    ``Restart=always`` relaunches the container on the new image. ``systemctl``
    is deliberately NOT used: it does not work from inside the container.

    On pull failure we do NOT write image.env and do NOT restart — the node
    keeps running the current image.
    """
    import subprocess as _sp
    loop = asyncio.get_event_loop()

    # Optional requested version from the body.
    requested = None
    try:
        if request.can_read_body:
            body = await request.json()
            if isinstance(body, dict):
                requested = body.get("version")
    except Exception:
        requested = None

    target = requested
    if not target:
        try:
            target = await loop.run_in_executor(None, _fetch_latest_ghcr_tag)
        except Exception:
            target = None
    if not target:
        return web.json_response(
            {"error": "Could not resolve a target version from GHCR"}, status=502
        )

    image = f"{AINODE_GHCR_REPO}:{target}"

    try:
        pull = await loop.run_in_executor(
            None,
            lambda: _sp.run(
                ["docker", "pull", image],
                capture_output=True, text=True, timeout=600
            )
        )
    except _sp.TimeoutExpired:
        # Same clean no-env-write failure path as a nonzero-return pull — the
        # node keeps running the current image; nothing was pinned or restarted.
        return web.json_response(
            {"error": "docker pull failed", "detail": "docker pull timed out after 600s"},
            status=502,
        )
    if pull.returncode != 0:
        return web.json_response(
            {"error": "docker pull failed", "detail": (pull.stderr or "")[-500:]},
            status=502,
        )

    # Pull succeeded — pin the new image so the swappable unit boots it.
    try:
        _write_image_env(image)
    except Exception as exc:
        return web.json_response(
            {"error": f"failed to write image.env: {exc}"}, status=500
        )

    # Only self-stop when THIS container was launched by the swappable unit. On a
    # node still running a pre-swappable unit, `docker stop` is destructive (see
    # _unit_is_swappable): the pull + pin above are harmless and ready the node,
    # but restarting would either drop the node or reboot the SAME old image, so
    # we refuse and point the operator at the host-side migration.
    if not _unit_is_swappable():
        return web.json_response({
            "status": "pulled",
            "restarted": False,
            "target": target,
            "image": image,
            "message": (
                "Image pulled and pinned, but this node's systemd unit predates "
                "the swappable-image unit and will not pick it up. Migrate it on "
                "the host (re-run the installer: curl -fsSL https://ainode.dev/"
                "install | bash) to boot the new image."
            ),
        })

    async def _self_restart():
        await asyncio.sleep(2)
        await loop.run_in_executor(
            None,
            lambda: _sp.run(
                ["docker", "stop", "ainode"],
                capture_output=True, text=True, timeout=60
            )
        )

    asyncio.get_event_loop().create_task(_self_restart())
    return web.json_response(
        {"status": "updating", "target": target, "image": image, "restarted": True}
    )


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
    rejected_reasons: dict = {}
    authenticated = is_authenticated(request)

    for key, value in body.items():
        if key not in PATCHABLE_CONFIG_FIELDS:
            rejected.append(key)
            continue
        if not hasattr(config, key):
            rejected.append(key)
            continue
        # An escalation needs a key. Turning the field OFF is a de-escalation, so
        # it stays open and a client can always put the node back.
        if key in AUTHENTICATED_ONLY_CONFIG_FIELDS and bool(value) and not authenticated:
            rejected.append(key)
            rejected_reasons[key] = TRUST_REMOTE_CODE_RULE
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
        # Why a field was refused, wherever there is a rule to state. A bare
        # "rejected" is the one answer a caller cannot act on.
        "rejected_reasons": rejected_reasons,
        "config": _safe_config_dict(config),
    })


def listener_plan(config: NodeConfig) -> list[tuple[str, int, Optional[ssl.SSLContext]]]:
    """The sockets this node should open: ``[(host, port, ssl context or None)]``.

    The HTTP entry is always first and always there. TLS adds a SECOND entry and
    never replaces it: every client in the fleet, the peer proxy included, talks
    to :3000, so a node that switched ports when TLS came on would drop out of
    its own cluster.

    TLS is skipped, loudly, when the block is enabled but unusable (a missing
    pair, an unreadable key, the same port as HTTP). A node with a broken
    certificate serves HTTP and logs why; it does not fail to boot, because the
    dashboard is how an operator fixes the certificate.
    """
    plan: list[tuple[str, int, Optional[ssl.SSLContext]]] = [
        (config.host, int(config.web_port), None)
    ]
    tls: TLSConfig = load_tls_config(config)
    if not tls.enabled:
        return plan
    if int(tls.port) == int(config.web_port):
        logger.error("TLS port %d is the HTTP port: serving HTTP only", tls.port)
        return plan
    if not tls.files_present():
        logger.error(
            "TLS is enabled but the pair is missing (cert=%s key=%s): serving "
            "HTTP only. Run `ainode tls enable` to make one.",
            tls.cert_file or "unset", tls.key_file or "unset",
        )
        return plan
    try:
        context = ssl_context(tls.cert_file, tls.key_file)
    except (ssl.SSLError, OSError, ValueError) as exc:
        logger.error("TLS is enabled but %s / %s cannot be loaded (%s): serving "
                     "HTTP only", tls.cert_file, tls.key_file, exc)
        return plan
    plan.append((config.host, int(tls.port), context))
    return plan


async def start_sites(runner: web.AppRunner,
                      plan: list[tuple[str, int, Optional[ssl.SSLContext]]]) -> int:
    """Open every socket in ``plan`` on ``runner``; return how many opened.

    A TLS port that cannot be bound (something else is already on it) is logged
    and skipped, for the same reason a bad certificate is: the HTTP port is how
    the operator reaches the dashboard that fixes it. A failure on the HTTP entry
    is raised, exactly as ``web.run_app`` would: a node with no HTTP port is not
    a node.
    """
    started = 0
    for host, port, context in plan:
        site = web.TCPSite(runner, host, port, ssl_context=context)
        try:
            await site.start()
        except OSError as exc:
            if context is None:
                raise
            logger.error("TLS port %d could not be opened (%s): serving HTTP only",
                         port, exc)
            continue
        started += 1
        logger.info("Serving %s on %s:%d",
                    "HTTPS" if context else "HTTP", host or "0.0.0.0", port)
    return started


async def _serve_forever(app: web.Application,
                         plan: list[tuple[str, int, Optional[ssl.SSLContext]]]) -> None:
    """Run ``app`` on every socket in ``plan`` until a signal says stop.

    aiohttp's ``run_app`` opens exactly one site, and TLS needs a second with its
    own context, so this is the multi-listener path. It installs the same signal
    handling ``run_app`` does, because the cleanup hooks (discovery sender, client
    session) have to run on the SIGTERM that `docker stop` sends.
    """
    runner = web.AppRunner(app)
    await runner.setup()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError, ValueError):
            pass  # not POSIX, or not the main thread
    try:
        await start_sites(runner, plan)
        await stop.wait()
    finally:
        await runner.cleanup()


def run_server(config: Optional[NodeConfig] = None, engine=None) -> None:
    """Start the API server (blocking)."""
    if config is None:
        config = NodeConfig()
    app = create_app(config=config, engine=engine)
    plan = listener_plan(config)
    if len(plan) == 1:
        # The path every node without TLS takes, unchanged.
        web.run_app(app, host=config.host, port=config.web_port, print=None)
        return
    try:
        asyncio.run(_serve_forever(app, plan))
    except KeyboardInterrupt:
        pass
