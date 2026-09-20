"""Server view routes — LM Studio-style server console.

Provides:
- In-memory request log (ring buffer) + middleware to capture API traffic
- GET/DELETE /api/server/logs
- GET /api/server/endpoints (catalog)
- GET /api/server/status (reachable URLs, loaded models, request counters)
- POST /api/server/models/{model_id}/eject
"""

from __future__ import annotations

import logging
import socket
import time
from collections import deque
from typing import Optional

import aiohttp
from aiohttp import web

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

def _reachable_urls(host: str, port: int) -> list[str]:
    """Return a list of URLs this server can be reached at."""
    urls: list[str] = [f"http://localhost:{port}"]
    try:
        hostname = socket.gethostname()
        _, _, addrs = socket.gethostbyname_ex(hostname)
        for addr in addrs:
            url = f"http://{addr}:{port}"
            if url not in urls:
                urls.append(url)
    except Exception:
        pass
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
        "reachable_at": _reachable_urls(host, web_port),
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
                request.app["engine"] = None  # the primary went away
                # routing-truth: the node must stop claiming a model it no longer
                # serves, or the master keeps advertising a ghost.
                config = request.app.get("config")
                if config is not None and getattr(config, "model", None) == model_id:
                    config.model = None
                    try:
                        config.save()
                    except Exception:
                        pass
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
