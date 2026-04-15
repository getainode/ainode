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
        {"method": "GET", "path": "/api/v1/models", "description": "List loaded models"},
        {"method": "POST", "path": "/api/v1/chat/completions", "description": "Chat completion", "status": "planned"},
        {"method": "POST", "path": "/api/v1/completions", "description": "Text completion", "status": "planned"},
        {"method": "POST", "path": "/api/v1/embeddings", "description": "Generate embeddings", "status": "planned"},
    ],
    "openai": [
        {"method": "GET", "path": "/v1/models", "description": "List models (OpenAI-compatible)"},
        {"method": "POST", "path": "/v1/chat/completions", "description": "Chat completions (OpenAI)"},
        {"method": "POST", "path": "/v1/completions", "description": "Text completions (OpenAI)"},
        {"method": "POST", "path": "/v1/embeddings", "description": "Generate embeddings — OpenAI-compatible, served in-process via sentence-transformers"},
    ],
    "anthropic": [
        {"method": "POST", "path": "/v1/messages", "description": "Anthropic Messages API", "status": "planned"},
    ],
}


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

    # Collect loaded models (local + cluster members)
    local_models = await _probe_loaded_models(session, getattr(config, "api_port", 8000))
    loaded_models: list[dict] = []
    for mid in local_models:
        loaded_models.append({
            "id": mid,
            "node_hostname": config.node_name or "local",
            "node_id": config.node_id or "local",
            "type": "llm",
            "format": "SafeTensors",
            "quantization": None,
            "size_bytes": 0,
            "parallel": 1,
            "capabilities": ["chat", "completions"],
            "loaded_at": start_time,
        })

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
                    "type": "embed",
                    "format": "SafeTensors",
                    "quantization": None,
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
                if m.model:
                    loaded_models.append({
                        "id": m.model,
                        "node_hostname": m.node_name,
                        "node_id": m.node_id,
                        "type": "llm",
                        "format": "SafeTensors",
                        "quantization": None,
                        "size_bytes": 0,
                        "parallel": 1,
                        "capabilities": ["chat", "completions"],
                        "loaded_at": getattr(m, "last_seen", start_time),
                    })
        except Exception:
            pass

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

    return web.json_response({
        "status": "running",
        "host": host,
        "port": web_port,
        "reachable_at": _reachable_urls(host, web_port),
        "uptime_seconds": uptime,
        "loaded_models": loaded_models,
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
