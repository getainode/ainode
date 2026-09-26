"""Server view routes — LM Studio-style server console.

Provides:
- In-memory request log (ring buffer) + middleware to capture API traffic
- GET/DELETE /api/server/logs
- GET /api/server/endpoints (catalog)
- GET /api/server/status (reachable URLs, loaded models, request counters)
- POST /api/server/models/{model_id}/eject
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from collections import deque
from typing import Optional

import aiohttp
from aiohttp import web

from ainode import __version__
from ainode.core.config import NodeConfig
from ainode.discovery.instance import instance_parallel

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

LOG_BUFFER_MAX = 500

# Paths we never log (noise reduction)
LOG_SKIP_PREFIXES = (
    "/static/",
    "/api/health",
    "/api/metrics",
    "/api/models/downloads/active",
    "/api/cluster/info",
    "/api/server/logs",
    "/favicon.ico",
)


def _should_log(path: str) -> bool:
    for p in LOG_SKIP_PREFIXES:
        if path == p or path.startswith(p):
            return False
    return True


# ------------------------------------------------------------------
# Middleware
# ------------------------------------------------------------------

@web.middleware
async def request_log_middleware(request: web.Request, handler):
    """Capture API requests into the app's in-memory ring buffer."""
    path = request.path
    start = time.time()
    status = 500
    resp: Optional[web.StreamResponse] = None
    try:
        resp = await handler(request)
        status = getattr(resp, "status", 200)
        return resp
    except web.HTTPException as exc:
        status = exc.status
        raise
    finally:
        if _should_log(path):
            try:
                duration_ms = round((time.time() - start) * 1000, 2)
                client_ip = ""
                peername = request.transport.get_extra_info("peername") if request.transport else None
                if peername:
                    client_ip = peername[0]
                forwarded = request.headers.get("X-Forwarded-For")
                if forwarded:
                    client_ip = forwarded.split(",")[0].strip()

                content_length = 0
                if resp is not None:
                    try:
                        content_length = int(resp.headers.get("Content-Length", "0") or 0)
                    except Exception:
                        content_length = 0

                level = "INFO"
                if status >= 500:
                    level = "ERROR"
                elif status >= 400:
                    level = "WARN"

                # Try to extract model from body for chat/completions paths
                model = request.get("_log_model")

                entry = {
                    "timestamp": time.time(),
                    "method": request.method,
                    "path": path,
                    "status": status,
                    "duration_ms": duration_ms,
                    "client_ip": client_ip,
                    "content_length": content_length,
                    "level": level,
                    "model": model,
                }

                buf: Optional[deque] = request.app.get("api_log")
                if buf is not None:
                    buf.append(entry)

                # Bump counters
                counters = request.app.get("api_log_counters")
                if counters is not None:
                    counters["total"] = counters.get("total", 0) + 1
                    counters["recent"].append(entry["timestamp"])
            except Exception:  # pragma: no cover - logging must never fail a request
                logger.exception("request_log_middleware: failed to capture entry")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

#: How long a looked-up address list is reused. The answer only changes when an
#: interface does, and a lookup is the one part of the Server view that can hang.
HOST_ADDRS_TTL_SECONDS = 300.0
#: How long a request waits on a lookup before answering without it.
HOST_ADDRS_WAIT_SECONDS = 1.0

# (expires_at, addresses) of the last finished lookup, and the lookup in flight.
_host_addrs_cache: Optional[tuple[float, list[str]]] = None
_host_addrs_pending: Optional[asyncio.Future] = None


def _lookup_host_addrs() -> list[str]:
    """This host's IPv4 addresses by name. BLOCKING: resolver time, unbounded."""
    try:
        _, _, addrs = socket.gethostbyname_ex(socket.gethostname())
        return list(addrs)
    except Exception:
        return []


async def _host_addrs() -> list[str]:
    """This host's addresses, without ever holding the event loop on DNS.

    ``gethostbyname_ex`` goes through the resolver when the hostname is not in
    /etc/hosts, and a resolver with a dead nameserver takes 10 to 20 s to give
    up. Run on the loop, that froze every request on the node for that long, on
    every poll of /api/server/status: on 2026-09-26 the master's heartbeats went
    stale and it stopped routing to its peers. The lookup now runs in a thread,
    one at a time, and its answer is kept for HOST_ADDRS_TTL_SECONDS. A request
    waits at most HOST_ADDRS_WAIT_SECONDS and otherwise answers with the last
    known list (empty the first time); a slow lookup still fills the cache when
    it finishes.
    """
    global _host_addrs_cache, _host_addrs_pending
    now = time.monotonic()
    cached = _host_addrs_cache
    if cached is not None and cached[0] > now:
        return cached[1]
    stale = cached[1] if cached is not None else []

    if _host_addrs_pending is None or _host_addrs_pending.done():
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, _lookup_host_addrs)

        def _store(fut: asyncio.Future) -> None:
            global _host_addrs_cache, _host_addrs_pending
            try:
                addrs = fut.result()
            except Exception:
                addrs = []
            _host_addrs_cache = (time.monotonic() + HOST_ADDRS_TTL_SECONDS, addrs)
            _host_addrs_pending = None

        future.add_done_callback(_store)
        _host_addrs_pending = future

    try:
        return list(await asyncio.wait_for(asyncio.shield(_host_addrs_pending),
                                           timeout=HOST_ADDRS_WAIT_SECONDS))
    except Exception:
        return stale


async def _reachable_urls(host: str, port: int) -> list[str]:
    """Return a list of URLs this server can be reached at."""
    urls: list[str] = [f"http://localhost:{port}"]
    for addr in await _host_addrs():
        url = f"http://{addr}:{port}"
        if url not in urls:
            urls.append(url)
    # Also include the configured bind host if it is a specific IP
    if host and host not in ("0.0.0.0", "127.0.0.1", "localhost"):
        url = f"http://{host}:{port}"
        if url not in urls:
            urls.append(url)
    return urls


async def _probe_loaded_models(
    session: Optional[aiohttp.ClientSession],
    api_port: int,
) -> list[str]:
    """Query local vLLM /v1/models to list currently-loaded model IDs."""
    if session is None:
        return []
    try:
        url = f"http://localhost:{api_port}/v1/models"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return [m.get("id", "") for m in data.get("data", []) if m.get("id")]
    except Exception:
        pass
    return []


# ------------------------------------------------------------------
# Endpoint catalog
# ------------------------------------------------------------------

ENDPOINT_CATALOG = {
    "lmstudio": [
        # No /api/v1/* route is registered anywhere, so the GET row that used to
        # sit here (unmarked, next to three `planned` siblings) handed the reader
        # a curl that 404s (#206). The federated list is GET /v1/models, in the
        # OpenAI tab.
        {"method": "POST", "path": "/api/v1/chat/completions", "description": "Chat completion", "status": "planned"},
        {"method": "POST", "path": "/api/v1/completions", "description": "Text completion", "status": "planned"},
        {"method": "POST", "path": "/api/v1/embeddings", "description": "Generate embeddings", "status": "planned"},
    ],
    "openai": [
        {"method": "GET", "path": "/v1/models", "description": "List models (OpenAI-compatible)"},
        {"method": "POST", "path": "/v1/chat/completions", "description": "Chat completions (OpenAI)"},
        {"method": "POST", "path": "/v1/completions", "description": "Text completions (OpenAI)"},
        {"method": "POST", "path": "/v1/embeddings", "description": "Generate embeddings (OpenAI-compatible): routed by model id to the node serving it, or in-process via sentence-transformers when no node does"},
        {"method": "POST", "path": "/v1/audio/transcriptions", "description": "Speech to text (OpenAI-compatible): multipart, with the model id as a form field beside the audio file, routed to the node serving that model"},
        {"method": "POST", "path": "/v1/audio/translations", "description": "Speech to English text, for the ASR models that translate (whisper-large-v3-turbo transcribes only)"},
    ],
    "anthropic": [
        {"method": "POST", "path": "/v1/messages", "description": "Anthropic Messages API, forwarded to the node serving the requested model"},
        {"method": "POST", "path": "/v1/messages/count_tokens", "description": "Token count for a Messages body (Anthropic)"},
    ],
    "ainode": [
        {"method": "POST", "path": "/v1/decide", "description": "Typed questions in, calibrated probabilities out: every question answered at once by the node serving the model"},
        {"method": "POST", "path": "/v1/systemone", "description": "The same decisions in TypeSafe's System One (Jev) wire format, so a client written for the hosted endpoint answers off a model on this fleet"},
    ],
}


# ------------------------------------------------------------------
# Parallelism (#92)
# ------------------------------------------------------------------
#
# `loaded_models[].parallel` used to be a hard-coded 1 on every row, so a model
# launched across two nodes (tensor_parallel_size 2) read as single-GPU in the
# Server view, in the fleet dashboard and in a bench record's placement.tp. The
# launch width lives on the InstanceRecord; these two helpers are the only places
# that read it, one for a local instance and one for a peer's announcement.


def _local_parallel(app, model: str, port: int) -> int:
    """TP width of a LOCAL instance, from the record that launched it."""
    manager = app.get("instances")
    if manager is not None:
        try:
            for inst in manager.instances():
                rec = inst.record
                if rec.api_port == port and (not model or rec.model == model):
                    return instance_parallel(rec)
        except Exception:  # pragma: no cover - defensive
            logger.exception("failed to read local instance parallelism")
    # A distributed primary that booted from config.json (replayed head) is not
    # always in the manager yet; its peer list is the same truth.
    config = app.get("config")
    if config is not None and getattr(config, "model", None) == model \
            and (getattr(config, "distributed_mode", "solo") or "solo") == "head":
        peers = list(getattr(config, "peer_ips", []) or [])
        if peers:
            return 1 + len(peers)
    return 1


#: What a served instance is for, and what it can answer. An embedding model is
#: an ordinary stacked vLLM instance on this hardware (same image, `--runner
#: pooling`), so nothing about how it was launched distinguishes it from a chat
#: engine: the curated catalog entry's `capabilities` is the only place that
#: knowledge lives, and the browser reads `type` to keep the model out of the chat
#: picker (`app.js::refreshChatFleet`) and out of the card's chat controls.
_LLM_KIND = ("llm", ("chat", "completions"))
_EMBED_KIND = ("embed", ("embeddings",))
#: A speech-to-text instance is the same story: an ordinary stacked vLLM instance
#: whose model happens to answer the two audio paths and no chat path, so only the
#: catalog entry knows, and the browser needs `type` to keep it out of the chat
#: and bench pickers the way it keeps an embedding model out.
_SPEECH_KIND = ("speech", ("transcriptions", "translations"))


def serving_kind(model: str) -> tuple[str, list]:
    """``(type, capabilities)`` for a model id the fleet is serving.

    Read from the curated catalog rather than probed: vLLM's ``/v1/models`` says
    nothing about which runner is behind an id, and a pooling instance answers a
    chat completion with a 400 nobody can act on. A model the catalog does not
    describe is reported as a chat engine, which is what every instance was before
    embeddings could be served this way.
    """
    try:
        from ainode.models.registry import CURATED_CLUSTER_MODELS, FALLBACK_CATALOG

        for table in (CURATED_CLUSTER_MODELS, FALLBACK_CATALOG):
            for cid, info in (table or {}).items():
                if model in (getattr(info, "hf_repo", ""), cid):
                    caps = list(getattr(info, "capabilities", None) or [])
                    if "embedding" in caps:
                        kind = _EMBED_KIND
                    elif "speech" in caps:
                        kind = _SPEECH_KIND
                    else:
                        kind = _LLM_KIND
                    return kind[0], list(kind[1])
    except Exception:  # pragma: no cover - defensive
        logger.exception("failed to read the catalog for %s", model)
    return _LLM_KIND[0], list(_LLM_KIND[1])


def model_disk_facts(model: str) -> tuple:
    """``(size_bytes, quantization, quantization_source)`` for a served model id.

    The Server view published ``size_bytes: 0`` and ``quantization: null`` for
    every row (#180): nothing resolved a served id back to the weights on disk.
    Both are read here, from the snapshot directory and the catalog recipe or the
    model's own config.json, and both are None when this node cannot tell:
    never a 0 the interface would draw as a measured size.
    """
    try:
        from ainode.models.registry import model_disk_size_bytes, model_quantization

        size = model_disk_size_bytes(model)
        quant, source = model_quantization(model)
        return size, quant, source
    except Exception:  # pragma: no cover - defensive
        logger.exception("failed to read on-disk facts for %s", model)
        return None, None, None


def _remote_parallel(node, model: str, port: int) -> int:
    """TP width of a PEER's instance, from the instance list on its announcement.

    A peer's announcement already carries every instance it heads as a wire dict
    (``InstanceRecord.to_dict()``), tensor_parallel_size included, so the master
    reads it there. An older peer that sends no instances falls back to its
    legacy distributed_peers list, then to 1.
    """
    for inst in (getattr(node, "instances", []) or []):
        if not isinstance(inst, dict):
            continue
        if inst.get("model") != model:
            continue
        inst_port = inst.get("api_port") or port
        if inst_port == port:
            return instance_parallel(inst)
    if (getattr(node, "distributed_mode", "solo") or "solo") == "head" \
            and getattr(node, "model", "") == model:
        peers = list(getattr(node, "distributed_peers", []) or [])
        if peers:
            return 1 + len(peers)
    return 1


# ------------------------------------------------------------------
# Handlers
# ------------------------------------------------------------------

async def handle_server_status(request: web.Request) -> web.Response:
    """Return a rich server status block for the Server view."""
    config: NodeConfig = request.app["config"]
    start_time: float = request.app.get("start_time", time.time())
    session: Optional[aiohttp.ClientSession] = request.app.get("client_session")

    uptime = round(time.time() - start_time, 1)
    web_port = getattr(config, "web_port", 3000)
    host = getattr(config, "host", "0.0.0.0")

    # Collect loaded models (local primary + local stacked + cluster members).
    primary_port = getattr(config, "api_port", 8000)
    local_models = await _probe_loaded_models(session, primary_port)
    loaded_models: list[dict] = []
    for mid in local_models:
        kind, caps = serving_kind(mid)
        size_bytes, quant, quant_source = model_disk_facts(mid)
        loaded_models.append({
            "id": mid,
            "node_hostname": config.node_name or "local",
            "node_id": config.node_id or "local",
            "port": primary_port,
            "ready": True,
            "ejectable": True,
            "type": kind,
            "format": "SafeTensors",
            "quantization": quant,
            "quantization_source": quant_source,
            "size_bytes": size_bytes,
            "parallel": _local_parallel(request.app, mid, primary_port),
            "capabilities": caps,
            "loaded_at": start_time,
        })

    # Local STACKED instances (2nd+ model on this node, ports 8001+) live in the
    # InstanceManager, not on the primary vLLM port the probe above hits — so
    # they were invisible in the Server view (F2). Add a row per stacked
    # instance. These are local, so the eject endpoint can target them.
    manager = request.app.get("instances")
    if manager is not None:
        try:
            for inst in manager.instances():
                rec = inst.record
                if not rec.model or rec.api_port == primary_port:
                    continue  # primary already covered by the probe above
                # Truthful readiness: probe the stacked instance's own OpenAI
                # port live rather than trusting rec.status. rec.status is a
                # latch stamped `serving` when the engine first answered; a
                # stacked engine that later crashes or is killed out-of-band
                # keeps reading `serving` (so it would show a green READY badge
                # and be chat-targetable indefinitely). The /v1/models probe is
                # the same liveness signal the primary uses two blocks up, so
                # views and routing agree. The row stays ejectable either way so
                # a dead instance can still be cleaned up from the UI.
                stacked_live = await _probe_loaded_models(session, rec.api_port)
                kind, caps = serving_kind(rec.model)
                size_bytes, quant, quant_source = model_disk_facts(rec.model)
                loaded_models.append({
                    "id": rec.model,
                    "node_hostname": config.node_name or "local",
                    "node_id": config.node_id or "local",
                    "port": rec.api_port,
                    "ready": bool(stacked_live),
                    "ejectable": True,
                    "type": kind,
                    "format": "SafeTensors",
                    "quantization": quant,
                    "quantization_source": quant_source,
                    "size_bytes": size_bytes,
                    "parallel": instance_parallel(rec),
                    "capabilities": caps,
                    "loaded_at": start_time,
                })
        except Exception:
            logger.exception("failed to list local stacked instances")

    # Include loaded embedding models (in-process, via EmbeddingManager)
    embedding_manager = request.app.get("embedding_manager")
    if embedding_manager is not None:
        try:
            hostname = socket.gethostname()
            for emb in embedding_manager.list_loaded():
                size_mb = emb.get("size_mb") or 0
                loaded_models.append({
                    "id": emb["id"],
                    "node_hostname": hostname,
                    "node_id": config.node_id or "local",
                    "port": primary_port,
                    "ready": True,
                    "ejectable": True,
                    "type": "embed",
                    "format": "SafeTensors",
                    "quantization": None,
                    "quantization_source": None,
                    "size_bytes": int(size_mb) * 1024 * 1024,
                    "parallel": 1,
                    "capabilities": ["embeddings"],
                    "loaded_at": emb.get("loaded_at", start_time),
                    "dimensions": emb.get("dimensions"),
                    "max_seq_length": emb.get("max_seq_length"),
                })
        except Exception:
            logger.exception("failed to list loaded embedding models")

    # Include models from cluster members (via ClusterState announcements)
    cluster = request.app.get("cluster_state")
    if cluster is not None:
        try:
            members = cluster.members()
            for m in members:
                if m.node_id == config.node_id:
                    continue
                member_port = getattr(m, "api_port", 8000) or 8000
                # Remote instances can't be ejected from here — the eject
                # endpoint only targets this node's local InstanceManager.
                if m.model:
                    kind, caps = serving_kind(m.model)
                    size_bytes, quant, quant_source = model_disk_facts(m.model)
                    loaded_models.append({
                        "id": m.model,
                        "node_hostname": m.node_name,
                        "node_id": m.node_id,
                        "port": member_port,
                        "ready": True,  # model is only broadcast once serving
                        "ejectable": False,
                        "type": kind,
                        "format": "SafeTensors",
                        "quantization": quant,
                        "quantization_source": quant_source,
                        "size_bytes": size_bytes,
                        "parallel": _remote_parallel(m, m.model, member_port),
                        "capabilities": caps,
                        "loaded_at": getattr(m, "last_seen", start_time),
                    })
                # Remote STACKED instances (ports 8001+) carried on the peer's
                # announcement — same source /api/nodes exposes (F2).
                for inst in (getattr(m, "instances", []) or []):
                    if not isinstance(inst, dict):
                        continue
                    im = inst.get("model")
                    if not im:
                        continue
                    inst_port = inst.get("api_port") or member_port
                    if im == m.model and inst_port == member_port:
                        continue  # primary already added above
                    kind, caps = serving_kind(im)
                    size_bytes, quant, quant_source = model_disk_facts(im)
                    loaded_models.append({
                        "id": im,
                        "node_hostname": m.node_name,
                        "node_id": m.node_id,
                        "port": inst_port,
                        "ready": inst.get("status") == "serving",
                        "ejectable": False,
                        "type": kind,
                        "format": "SafeTensors",
                        "quantization": quant,
                        "quantization_source": quant_source,
                        "size_bytes": size_bytes,
                        "parallel": instance_parallel(inst),
                        "capabilities": caps,
                        "loaded_at": getattr(m, "last_seen", start_time),
                    })
        except Exception:
            logger.exception("failed to list cluster-member instances")

    # Request counters
    counters = request.app.get("api_log_counters") or {}
    total = counters.get("total", 0)
    recent: deque = counters.get("recent") if counters else None
    now = time.time()
    last_minute = 0
    if recent is not None:
        # Prune old entries (> 60s)
        while recent and (now - recent[0] > 60):
            recent.popleft()
        last_minute = len(recent)

    # The Server view's status dot reads this. It was the literal "running",
    # true by tautology (we are answering the request), so the dot stayed green
    # with every engine in the fleet dead (#206). Report what the instance
    # records already say instead: "running" when at least one model is serving,
    # "loading" when models are loaded but none are ready yet, "idle" when none
    # are loaded at all.
    ready = sum(1 for m in loaded_models if m.get("ready"))
    if ready:
        status = "running"
    elif loaded_models:
        status = "loading"
    else:
        status = "idle"

    return web.json_response({
        "status": status,
        "host": host,
        "port": web_port,
        "reachable_at": await _reachable_urls(host, web_port),
        "uptime_seconds": uptime,
        "loaded_models": loaded_models,
        "models_ready": ready,
        "request_count_total": total,
        "request_count_last_minute": last_minute,
    })


async def handle_server_endpoints(_request: web.Request) -> web.Response:
    return web.json_response(ENDPOINT_CATALOG)


async def handle_server_logs_get(request: web.Request) -> web.Response:
    buf: Optional[deque] = request.app.get("api_log")
    entries: list[dict] = list(buf) if buf else []
    since_raw = request.query.get("since")
    if since_raw:
        try:
            since = float(since_raw)
            entries = [e for e in entries if e.get("timestamp", 0) > since]
        except ValueError:
            pass
    limit_raw = request.query.get("limit")
    if limit_raw:
        try:
            limit = int(limit_raw)
            entries = entries[-limit:]
        except ValueError:
            pass
    return web.json_response({"entries": entries, "now": time.time()})


async def handle_server_logs_clear(request: web.Request) -> web.Response:
    buf: Optional[deque] = request.app.get("api_log")
    if buf is not None:
        buf.clear()
    counters = request.app.get("api_log_counters")
    if counters is not None:
        counters["total"] = 0
        recent = counters.get("recent")
        if recent is not None:
            recent.clear()
    return web.json_response({"ok": True})


async def handle_server_eject(request: web.Request) -> web.Response:
    """Eject (unload) a model from the engine.

    Current implementation: best-effort — delegates to the model manager if
    available, otherwise returns a planned-status response so the UI can
    show a toast.
    """
    model_id = request.match_info.get("model_id", "")
    engine = request.app.get("engine")

    # P2-2: if this model is a running distributed instance, stop just that one
    # and drop it from the manager (leaving any other instances untouched).
    manager = request.app.get("instances")
    if manager is not None:
        inst = manager.by_model(model_id)
        if inst is not None:
            try:
                inst.backend.stop()
            except Exception:  # best-effort — still drop it from the registry
                pass
            manager.remove(inst.record.instance_id)
            if request.app.get("engine") is inst.backend:
                # The primary went away. routing-truth: the node must stop
                # claiming a model it no longer serves, or the master keeps
                # advertising a ghost, and its launch parameters go with it.
                request.app["engine"] = None
                config = request.app.get("config")
                if config is not None:
                    from ainode.models.api_routes import release_primary
                    try:
                        release_primary(request.app, config)
                    except Exception:
                        logger.warning("eject: failed to persist the cleared primary",
                                       exc_info=True)
            # Persist the shrunken instance set. Without this the eject was
            # memory-only: startup replay reads the manifest, so the ejected model
            # came BACK on the next reboot (spark-4, 2026-08-13 — an ejected 0.5B
            # reappeared and then blocked a later load via admission control).
            try:
                from ainode.models.api_routes import save_instance_manifest
                save_instance_manifest(request.app)
            except Exception:
                logger.warning("eject: failed to persist instance manifest", exc_info=True)
            return web.json_response({"ok": True, "model_id": model_id,
                                      "message": "Instance stopped"})

    # Delegate to embedding manager if this is a loaded embedding model
    embedding_manager = request.app.get("embedding_manager")
    if embedding_manager is not None and embedding_manager.is_loaded(model_id):
        unloaded = embedding_manager.unload(model_id)
        return web.json_response({
            "ok": bool(unloaded),
            "model_id": model_id,
            "message": "Embedding model unloaded" if unloaded else "Not loaded",
        })

    # Try calling engine.unload if present
    unloaded = False
    message = ""
    if engine is not None and hasattr(engine, "unload"):
        try:
            result = engine.unload(model_id)
            # allow sync or awaitable
            if hasattr(result, "__await__"):
                result = await result
            unloaded = bool(result)
            message = "Model ejected"
        except Exception as exc:  # pragma: no cover - defensive
            message = f"Failed to eject: {exc}"

    if not unloaded:
        return web.json_response({
            "ok": False,
            "model_id": model_id,
            "status": "planned",
            "message": message or "Eject not yet implemented for the running engine. Stop the container or restart AINode to release the model.",
        }, status=202)

    return web.json_response({"ok": True, "model_id": model_id, "message": message})


# ------------------------------------------------------------------
# The client endpoint: every node can say where the fleet answers
# ------------------------------------------------------------------
#
# Routing is already replicated, every node proxies every fleet model, but the
# ADDRESS is not: a client holds one, so an outage on the node it holds strands
# it even though every other node could have served it. These helpers answer
# GET /api/cluster/endpoint, and fill /api/status's endpoint_hint, with the
# addresses a client can fail over to, so a client learns its own fallbacks from
# the node it is already talking to.
#
# Two rules run through all of it:
#   * "localhost" is never an answer. It is what /api/nodes reports for every
#     row, and it is the one address guaranteed wrong on the caller's machine, so
#     any loopback spelling is dropped here rather than handed out.
#   * A host is only echoed back when this node really answers on it. The Host
#     header is caller-controlled, so accepting it unchecked would let one
#     request make a node advertise any address at all to the next reader.
#
# And one rule about the scheme, added when TLS grew up: a row's `url` carries
# the scheme and port this node ACTUALLY serves, so a client learns "prefer
# https on 3443 here" from the same payload that tells it where to knock. The
# `port` field stays the HTTP port in every row, because the peer proxy and
# every client build `http://host:port` from it and a node that moved that port
# when TLS came on would leave its own cluster. Only this node's own rows can
# say `tls: true`: a peer's TLS state is not on the discovery wire.

#: Spellings that only ever mean "the machine making the request", so they can
#: never be handed to a client as a fleet address. 127.* is matched by prefix.
LOOPBACK_HOSTS = frozenset({
    "localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback",
    "0.0.0.0", "::", "::1", "127.0.0.1",
})

#: Placeholders a node can carry instead of a name (``node_name`` is Optional,
#: and the Ray join path already guards against these exact values), which must
#: never become a URL.
UNRESOLVED_HOSTS = frozenset({"unknown", "none", "null", ""})

#: How long this node's own addresses are cached, in seconds.
#:
#: NOT an optimization to skip: deriving them reads the host (an ``ip``
#: subprocess and a socket the kernel has to route), and this runs on
#: /api/status, which every dashboard polls every few seconds for every node it
#: draws, so uncached it would put a fork per poll on the fleet's hottest route,
#: on the event loop. A machine's addresses change on the timescale of a DHCP
#: lease, so a minute of staleness costs nothing.
ADDRESS_CACHE_SECONDS = 60.0

#: ``{"at": <monotonic>, "key": <config fields>, "own": frozenset, "lan": str}``.
#: Both answers are refreshed together because both come from one read of the host.
_address_cache: dict = {"at": 0.0, "key": None, "own": frozenset(), "lan": ""}


def reset_address_cache() -> None:
    """Forget the cached addresses. For tests."""
    _address_cache.update({"at": 0.0, "key": None, "own": frozenset(), "lan": ""})


def _is_usable_host(value) -> bool:
    """True when *value* is an address worth handing to another machine."""
    if not isinstance(value, str):
        return False
    host = value.strip().strip("[]").lower()
    if host in UNRESOLVED_HOSTS or host in LOOPBACK_HOSTS:
        return False
    return not host.startswith("127.")


def _first_usable(*candidates) -> str:
    for candidate in candidates:
        if _is_usable_host(candidate):
            return str(candidate).strip()
    return ""


def host_header_host(request) -> str:
    """The host part of this request's Host header, port stripped, or ""."""
    try:
        header = (request.headers.get("Host") or "").strip()
    except AttributeError:  # pragma: no cover - a stub request with no headers
        return ""
    if not header:
        return ""
    if header.startswith("["):  # [::1]:3000, an IPv6 literal keeps its colons
        closing = header.find("]")
        return header[: closing + 1] if closing != -1 else header
    return header.split(":", 1)[0].strip()


def _local_ipv4s() -> list:
    """Every IPv4 bound on this host, in ``ip`` order, or [].

    Deliberately the raw address list rather than ``netdev.list_ipv4_interfaces``:
    that one drops virtual devices while it hunts for a fabric port, and a tailnet
    or bridge address is a perfectly good way to reach this dashboard.
    """
    try:
        from ainode.cluster import netdev

        return list(netdev._parse_ipv4_addrs(
            netdev._run_command(["ip", "-o", "-4", "addr", "show"])
        ).values())
    except Exception:  # pragma: no cover - no ip(8), or a host that will not say
        logger.debug("could not enumerate this node's addresses", exc_info=True)
        return []


def _outbound_ipv4() -> str:
    """The address the kernel would source an outbound packet from, or "".

    A UDP ``connect()`` sends nothing, so this is a routing-table lookup rather
    than a network round trip. No name resolution anywhere in here: the DNS
    lookup this used to do could stall for seconds on a host whose own hostname
    does not resolve, and it would have done it on the /api/status path.
    """
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("8.8.8.8", 80))
            return probe.getsockname()[0]
        finally:
            probe.close()
    except Exception:  # pragma: no cover - no route at all
        logger.debug("could not source an outbound address", exc_info=True)
        return ""


def _derive_addresses(config) -> tuple:
    """``(own_addresses, lan_address)`` read fresh from the host."""
    own: set = set()

    def _add(value) -> None:
        if isinstance(value, str) and value.strip():
            own.add(value.strip().lower())

    addrs = _local_ipv4s()
    for addr in addrs:
        _add(addr)
    hostname = ""
    try:
        hostname = socket.gethostname()
    except Exception:  # pragma: no cover - a host with no name
        hostname = ""
    _add(hostname)
    _add(hostname.split(".", 1)[0] if hostname else "")
    if config is not None:
        _add(getattr(config, "node_name", ""))
        _add(getattr(config, "host", ""))

    # The address to advertise: an explicitly configured bind host, the interface
    # carrying the default route, then any other real address on the box, then
    # this node's name. Loopback is rejected at every step, so a node that can
    # offer nothing routable answers "" and its url comes back null, which a
    # client can act on where "localhost" cannot.
    configured = getattr(config, "host", "") if config is not None else ""
    lan = ""
    if _is_usable_host(configured):
        lan = str(configured).strip()
    else:
        lan = _first_usable(_outbound_ipv4(), *addrs,
                            getattr(config, "node_name", "") if config else "",
                            hostname)
    return frozenset(own), lan


def _addresses(config) -> tuple:
    """``(own_addresses, lan_address)``, cached for ADDRESS_CACHE_SECONDS.

    Keyed by the two config fields that feed it, so a renamed node or a changed
    bind host answers correctly at once rather than after the TTL, and so two
    apps in one process (which is every test module) cannot read each other's.
    """
    now = time.monotonic()
    key = (getattr(config, "node_name", None), getattr(config, "host", None))
    if (_address_cache.get("key") == key
            and now - float(_address_cache.get("at") or 0.0) < ADDRESS_CACHE_SECONDS):
        return _address_cache["own"], _address_cache["lan"]
    own, lan = _derive_addresses(config)
    _address_cache.update({"at": now, "key": key, "own": own, "lan": lan})
    return own, lan


def own_addresses(config) -> frozenset:
    """Every spelling of an address THIS node answers on, lowercased.

    The guard on the Host header: a caller-supplied host is echoed back only when
    it is in here. Built from the node's own interfaces, its hostname, and its
    configured identity. A DNS alias of this node that is none of those is not
    echoed back and falls through to the LAN address, which is a correct answer
    too: resolving the hostname here cost seconds on hosts where that lookup
    fails, on a route the dashboard polls constantly.
    """
    return _addresses(config)[0]


def lan_address(config) -> str:
    """The address this node tells other machines to use, or "".

    See ``_derive_addresses`` for the order. Cached with ``own_addresses``.
    """
    return _addresses(config)[1]


def self_host(request, config) -> str:
    """The address to report for the node answering this request.

    The Host header wins when this node really answers on it: that is the address
    the caller just proved it can reach, tailnet or LAN or a name out of its own
    hosts file, and it is a better answer than anything derived here. A header
    naming something else (a proxy's name, a spoof, or ``localhost``) is ignored
    in favour of the LAN address.
    """
    header_host = host_header_host(request)
    if _is_usable_host(header_host) and header_host.strip().lower() in own_addresses(config):
        return header_host.strip()
    return lan_address(config)


def peer_host(node) -> str:
    """The address to report for a PEER, or "".

    The UDP source address first: the announcement payload carries no routable
    address of its own, and the source IP is the one address the peer has proved
    it sends from. Then its fabric IP, then its name.
    """
    return _first_usable(
        getattr(node, "peer_ip", None),
        getattr(node, "fabric_ip", ""),
        getattr(node, "node_name", ""),
    )


def endpoint_url(host: str, port: int, tls: bool = False) -> Optional[str]:
    """The URL a client should prefer for this node, or None for an unusable host.

    ``https://host:<tls port>`` when the node really serves HTTPS, plain
    ``http://host:<web port>`` otherwise. Never a loopback URL either way.
    """
    if not _is_usable_host(host):
        return None
    scheme = "https" if tls else "http"
    return f"{scheme}://{host}:{int(port or 3000)}"


def local_scheme(config) -> tuple[bool, int]:
    """``(does THIS node serve HTTPS, on which port)``, for its own rows.

    Only ever asked about this node. A peer's TLS state is not on the discovery
    wire (the announcement is one datagram under a hard size ceiling, and TLS
    between peers is deliberately out of scope), so a peer row says ``http`` and
    a client that wants that node's HTTPS learns it by asking that node. Nothing
    here probes anything.

    Cheap when TLS is off, which is every node until an operator turns it on:
    ``serves_https`` answers from the config block alone and touches no files.
    """
    from ainode.tls.certs import serves_https

    try:
        return serves_https(config)
    except Exception:  # pragma: no cover - defensive, this is on /api/status
        logger.debug("could not read this node's TLS state", exc_info=True)
        return False, int(getattr(config, "web_port", 3000) or 3000)


def endpoint_nodes(app, request) -> list:
    """Every node of this cluster a client could talk to instead, master first.

    Ordered so a client that has lost its address reaches the coordinator while
    it is up, then the rest by name for a stable list. Offline nodes are left
    out: this is a list of addresses to try, not a roster. In-memory only, no
    node is probed, so a client already polling /api/status pays nothing for it.
    """
    config: Optional[NodeConfig] = app.get("config")
    cluster = app.get("cluster_state")
    if cluster is None:
        return []
    local_id = getattr(config, "node_id", None)
    web_port = int(getattr(config, "web_port", 3000) or 3000)

    try:
        from ainode.discovery.broadcast import NodeStatus

        members = [n for n in cluster.members() if n.status != NodeStatus.OFFLINE]
    except Exception:  # pragma: no cover - defensive
        logger.exception("could not read the cluster's members")
        return []

    master = None
    try:
        master = cluster.get_master()
    except Exception:  # pragma: no cover - defensive
        logger.exception("could not read the cluster's master")
    master_id = getattr(master, "node_id", None)

    local_tls, local_tls_port = local_scheme(config)

    rows: list[dict] = []
    for node in members:
        is_self = node.node_id == local_id
        host = self_host(request, config) if is_self else peer_host(node)
        port = web_port if is_self else int(getattr(node, "web_port", 3000) or 3000)
        # HTTPS is reported for this node only, for the reason in local_scheme.
        row_tls = bool(is_self and local_tls)
        rows.append({
            "name": node.node_name or node.node_id,
            "host": host,
            # The HTTP port, always, exactly as before: a peer proxy and every
            # older client build `http://host:port` out of these two fields, and
            # moving this to the TLS port would take a node out of its own
            # cluster. `url` below is the address a CLIENT should prefer, and
            # `tls_port` is where the second listener is; they are different
            # questions and this payload answers all three.
            "port": port,
            "tls": row_tls,
            "tls_port": local_tls_port if row_tls else None,
            # Our own version from the running process, a peer's from the wire
            # (``ainode_version`` on the announcement). A peer that announces
            # none reports "", the same spelling every other node-listing view
            # uses, and NEVER this node's version: filling it in is how a split
            # fleet goes on looking like a healthy one (#171). The key is
            # ``version`` here because every field in this payload is about the
            # node it describes; it is the same value /api/nodes calls
            # ``ainode_version``.
            "version": (__version__ if is_self
                        else (getattr(node, "ainode_version", "") or "")),
            "role": "master" if node.node_id == master_id else "worker",
            "url": endpoint_url(host, local_tls_port if row_tls else port, row_tls),
        })
    rows.sort(key=lambda row: (row["role"] != "master", (row["name"] or "").lower()))
    return rows


def endpoint_payload(app, request) -> dict:
    """The body of GET /api/cluster/endpoint.

    Addresses and nothing else, which is what lets it answer without an API key:
    a client that cannot reach its configured node has to be able to ask a
    reachable one where the fleet is, and the key it holds is no use to it if the
    node that went down is the one holding the fleet's addresses.
    """
    config: Optional[NodeConfig] = app.get("config")
    cluster = app.get("cluster_state")
    web_port = int(getattr(config, "web_port", 3000) or 3000)
    host = self_host(request, config)
    local_id = getattr(config, "node_id", None)

    master = None
    if cluster is not None:
        try:
            master = cluster.get_master()
        except Exception:  # pragma: no cover - defensive
            logger.exception("could not read the cluster's master")

    local_tls, local_tls_port = local_scheme(config)

    master_block = None
    if master is not None:
        is_self = master.node_id == local_id
        master_host = host if is_self else peer_host(master)
        master_port = web_port if is_self else int(getattr(master, "web_port", 3000) or 3000)
        master_tls = bool(is_self and local_tls)
        master_block = {
            "name": master.node_name or master.node_id,
            "host": master_host,
            "port": master_port,
            "tls": master_tls,
            "tls_port": local_tls_port if master_tls else None,
            "url": endpoint_url(master_host,
                                local_tls_port if master_tls else master_port,
                                master_tls),
        }

    return {
        "self": {
            "name": getattr(config, "node_name", None) or local_id,
            "host": host,
            "port": web_port,
            # Whether the node answering this request serves HTTPS, and where.
            # A client reading only this block still learns the scheme to prefer.
            "tls": local_tls,
            "tls_port": local_tls_port if local_tls else None,
            "url": endpoint_url(host, local_tls_port if local_tls else web_port,
                                local_tls),
            "version": __version__,
            "role": ("master" if master is not None and master.node_id == local_id
                     else "worker"),
        },
        # Null, not an empty dict and not this node: a member that has seen no
        # master says so, so a client can tell "no coordinator" apart from "the
        # coordinator is the node I am asking".
        "master": master_block,
        "nodes": endpoint_nodes(app, request),
        "generated_at": time.time(),
    }


async def handle_cluster_endpoint(request: web.Request) -> web.Response:
    """GET /api/cluster/endpoint: where this fleet answers, asked of any node."""
    return web.json_response(endpoint_payload(request.app, request))


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------

def init_server_state(app: web.Application) -> None:
    """Initialize in-memory buffers used by the Server view."""
    if "api_log" not in app:
        app["api_log"] = deque(maxlen=LOG_BUFFER_MAX)
    if "api_log_counters" not in app:
        app["api_log_counters"] = {"total": 0, "recent": deque(maxlen=1000)}


def register_server_routes(app: web.Application) -> None:
    init_server_state(app)
    app.router.add_get("/api/server/status", handle_server_status)
    app.router.add_get("/api/server/endpoints", handle_server_endpoints)
    app.router.add_get("/api/server/logs", handle_server_logs_get)
    app.router.add_delete("/api/server/logs", handle_server_logs_clear)
    app.router.add_post("/api/server/models/{model_id:.+}/eject", handle_server_eject)
    app.router.add_get("/api/cluster/endpoint", handle_cluster_endpoint)
