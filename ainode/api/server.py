"""AINode API proxy server — aiohttp app that serves the web UI and proxies to vLLM."""

import time
from dataclasses import asdict
from typing import Optional

import aiohttp
from aiohttp import web

from ainode.core.config import NodeConfig
from ainode.core.gpu import detect_gpu, GPUInfo
from ainode.web.serve import get_index_html, get_static_path

__version__ = "0.1.0"


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

    app = web.Application(middlewares=[cors_middleware])
    app["config"] = config
    app["engine"] = engine
    app["start_time"] = time.time()
    app["client_session"] = None  # lazy-init in startup

    # --- Lifecycle -----------------------------------------------------------
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    # --- AINode routes -------------------------------------------------------
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/health", handle_health)
    app.router.add_get("/api/status", handle_status)
    app.router.add_get("/api/nodes", handle_nodes)

    # --- OpenAI-compatible proxy routes --------------------------------------
    app.router.add_get("/v1/models", proxy_to_vllm)
    app.router.add_post("/v1/chat/completions", proxy_to_vllm)
    app.router.add_post("/v1/completions", proxy_to_vllm)

    # --- Static files --------------------------------------------------------
    app.router.add_static("/static", get_static_path(), name="static")

    return app


# =============================================================================
# Lifecycle helpers
# =============================================================================

async def _on_startup(app: web.Application) -> None:
    app["client_session"] = aiohttp.ClientSession()


async def _on_cleanup(app: web.Application) -> None:
    session: Optional[aiohttp.ClientSession] = app.get("client_session")
    if session and not session.closed:
        await session.close()


# =============================================================================
# Middleware
# =============================================================================

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

    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return resp


# =============================================================================
# Web UI
# =============================================================================

async def handle_index(_request: web.Request) -> web.Response:
    """Serve the embedded dashboard HTML."""
    html = get_index_html()
    return web.Response(text=html, content_type="text/html")


# =============================================================================
# AINode API endpoints
# =============================================================================

async def handle_health(_request: web.Request) -> web.Response:
    """Simple liveness probe."""
    return web.json_response({"status": "ok"})


async def handle_status(request: web.Request) -> web.Response:
    """Return rich node status."""
    config: NodeConfig = request.app["config"]
    engine = request.app["engine"]
    start_time: float = request.app["start_time"]

    gpu: Optional[GPUInfo] = detect_gpu()
    gpu_info = asdict(gpu) if gpu else None

    engine_ready = False
    models_loaded: list[str] = []
    if engine is not None:
        engine_ready = getattr(engine, "ready", False)
        try:
            hc = engine.health_check()
            models_loaded = hc.get("models_loaded", [])
        except Exception:
            pass

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
    })


async def handle_nodes(request: web.Request) -> web.Response:
    """Return the list of known cluster nodes (stub: just this node)."""
    config: NodeConfig = request.app["config"]
    engine = request.app["engine"]
    engine_ready = False
    if engine is not None:
        engine_ready = getattr(engine, "ready", False)

    node = {
        "node_id": config.node_id,
        "node_name": config.node_name,
        "host": config.host,
        "api_port": config.api_port,
        "web_port": config.web_port,
        "model": config.model,
        "engine_ready": engine_ready,
    }
    return web.json_response({"nodes": [node]})


# =============================================================================
# vLLM proxy
# =============================================================================

async def proxy_to_vllm(request: web.Request) -> web.StreamResponse:
    """Forward the request to the local vLLM server and stream the response back."""
    config: NodeConfig = request.app["config"]
    session: aiohttp.ClientSession = request.app["client_session"]
    vllm_url = f"http://localhost:{config.api_port}{request.path}"

    # Build upstream request kwargs
    kwargs: dict = {
        "headers": {k: v for k, v in request.headers.items()
                    if k.lower() not in ("host", "transfer-encoding")},
    }
    if request.method == "POST":
        kwargs["data"] = await request.read()

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
                return resp
            else:
                body = await upstream.read()
                return web.Response(
                    status=upstream.status,
                    body=body,
                    content_type=upstream.headers.get("Content-Type", "application/json"),
                )
    except aiohttp.ClientError:
        return web.json_response(
            {"error": {"message": "vLLM engine not reachable", "type": "server_error"}},
            status=502,
        )


# =============================================================================
# Convenience runner
# =============================================================================

def run_server(config: Optional[NodeConfig] = None, engine=None) -> None:
    """Start the API server (blocking)."""
    if config is None:
        config = NodeConfig()
    app = create_app(config=config, engine=engine)
    web.run_app(app, host=config.host, port=config.web_port, print=None)
