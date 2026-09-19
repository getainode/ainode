"""API route handlers for embedding models.

Exposes:
- ``POST /v1/embeddings`` — OpenAI-compatible embeddings endpoint
- ``GET /api/embeddings/models`` — catalog of known embedding models
- ``POST /api/embeddings/models/{model_id}/load`` — eagerly load a model
- ``POST /api/embeddings/models/{model_id}/unload`` — drop from memory

``POST /v1/embeddings`` asks the FLEET first. An embedding model served by vLLM's
pooling runner is an ordinary stacked instance on some node's engine port, so the
model id alone says where the vectors come from, exactly as it does for a chat
completion: the handler routes on ``body["model"]`` through the proxy's own
``_routing_candidates``, forwards the caller's body verbatim, and fails over to the
next instance when one is unreachable. The in-process sentence-transformers manager
below is the fallback for a model nothing in the fleet serves, and it is the reason
this path used to be dead on our hardware: the library is not in the container image,
so every request the master answered itself was a ``dependency_missing`` 503.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

import aiohttp
from aiohttp import web

from ainode.embeddings.manager import (
    EmbeddingManager,
    KNOWN_EMBEDDING_MODELS,
)

logger = logging.getLogger(__name__)

#: Same transport settings ``proxy_to_vllm`` forwards with: a dead or ghost
#: instance must fail its CONNECT fast so the loop reaches the next candidate,
#: while total stays uncapped because a large batch on a busy node is slow but
#: alive.
CONNECT_TIMEOUT_S = 5

#: Headers a reverse proxy must not pass upstream. aiohttp recomputes
#: content-length from ``data``, and forwarding the caller's alongside makes the
#: upstream wait for a body that never arrives.
_DROP_HEADERS = ("host", "transfer-encoding", "content-length")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_embedding_routes(app: web.Application) -> None:
    """Attach embedding routes to the aiohttp app.

    The caller must have already placed an :class:`EmbeddingManager` at
    ``app["embedding_manager"]``.
    """
    app.router.add_post("/v1/embeddings", handle_v1_embeddings)
    app.router.add_get("/api/embeddings/models", handle_list_embedding_models)
    app.router.add_post(
        "/api/embeddings/models/{model_id:.+}/load", handle_load_embedding_model
    )
    app.router.add_post(
        "/api/embeddings/models/{model_id:.+}/unload", handle_unload_embedding_model
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    # Cheap whitespace approximation — matches OpenAI's "usage" stat well
    # enough for clients that just want a non-zero number.
    return max(1, len(text.split()))


def _error(message: str, *, code: str = "invalid_request_error", status: int = 400) -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": code}}, status=status
    )


def fleet_candidates(app, model: str) -> list:
    """Every ``(host, port)`` in the fleet serving ``model`` right now.

    The proxy's own list, not a second one. ``_routing_candidates`` already knows
    that a stacked instance answers on its OWN engine port rather than the node's
    main one, which is exactly the shape an embedding model is launched in, and it
    already puts the local hop first and peers after. Imported lazily because
    ``ainode.api.server`` is what registers these routes, so a module-level import
    back into it would be a cycle.

    Returns ``[]`` when nothing serves the model, which is the one case that falls
    through to the in-process manager. A worker behaves the same as the master
    here: the candidate list is built from cluster state, which every node has.
    """
    try:
        from ainode.api.server import _routing_candidates
    except Exception:  # pragma: no cover - defensive, the import is in-package
        logger.exception("embeddings: could not load the proxy's routing candidates")
        return []
    config = app.get("config")
    if config is None:
        return []
    return _routing_candidates(app.get("cluster_state"), model,
                               getattr(config, "node_id", ""),
                               getattr(config, "api_port", 8000))


async def forward_to_fleet(request: web.Request, model: str, body_bytes: bytes,
                           candidates: list) -> web.Response:
    """Post the caller's body to each candidate until one answers.

    The response goes back with the upstream's status and content type, the same
    two things ``proxy_to_vllm`` passes through for a non-streamed answer: an
    embeddings reply is one JSON document, never an SSE stream. A transport
    failure moves to the next candidate, because a node that advertises a model
    it can no longer serve is indistinguishable from a live one in cluster state.
    Every HTTP status the engine returns is the caller's answer and is handed back
    untouched, including a 400: the body came from the caller, so an engine that
    rejects it would reject it on the next instance too.
    """
    session: Optional[aiohttp.ClientSession] = request.app.get("client_session")
    if session is None:
        return _error("no HTTP client session on this node", code="server_error",
                      status=503)
    kwargs = {
        "headers": {k: v for k, v in request.headers.items()
                    if k.lower() not in _DROP_HEADERS},
        "timeout": aiohttp.ClientTimeout(total=None, sock_connect=CONNECT_TIMEOUT_S),
        "data": body_bytes,
    }
    last_err = None
    for host, port in candidates:
        url = f"http://{host}:{port}/v1/embeddings"
        try:
            async with session.post(url, **kwargs) as upstream:
                body = await upstream.read()
                ctype = upstream.headers.get("Content-Type", "application/json")
                return web.Response(status=upstream.status, body=body,
                                    content_type=ctype.split(";")[0].strip())
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_err = exc  # unreachable, probably a ghost claim; try the next
            continue
    # The fleet says it serves this model and no instance of it answered. That is
    # a 502 and not a quiet fall back to the CPU model: a caller asking for
    # Qwen3-Embedding vectors must not silently receive MiniLM ones.
    return web.json_response(
        {"error": {"message": f"no reachable node is serving '{model}' ({last_err})",
                   "type": "server_error"}},
        status=502,
    )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def handle_v1_embeddings(request: web.Request) -> web.Response:
    """OpenAI-compatible embeddings endpoint, fleet first.

    Order matters. The body is read as bytes before it is parsed so the fleet hop
    forwards it verbatim rather than a re-serialised copy, and only the two things
    routing needs are validated before that hop (it is a JSON object, and it names
    a model). Everything else about the request is the serving engine's to accept
    or refuse. The in-process path keeps its own validation because the manager
    takes a list of strings, not a body.
    """
    manager: EmbeddingManager = request.app["embedding_manager"]

    body_bytes = await request.read()
    try:
        import json as _json

        body = _json.loads(body_bytes) if body_bytes else None
    except Exception:
        return _error("Invalid JSON body")

    if not isinstance(body, dict):
        return _error("Body must be a JSON object")

    model_id = body.get("model")
    if not model_id or not isinstance(model_id, str):
        return _error("'model' is required and must be a string")

    # Tag the request so the server-view log shows the embedding model, whichever
    # of the two paths answers it.
    try:
        request["_log_model"] = model_id
    except Exception:
        pass

    # --- the fleet, if anything in it serves this model id --------------------
    candidates = fleet_candidates(request.app, model_id)
    if candidates:
        return await forward_to_fleet(request, model_id, body_bytes, candidates)

    # --- otherwise this process, on the CPU ----------------------------------
    raw_input = body.get("input")
    if raw_input is None:
        return _error("'input' is required (string or array of strings)")

    if isinstance(raw_input, str):
        texts: List[str] = [raw_input]
    elif isinstance(raw_input, list):
        if not all(isinstance(x, str) for x in raw_input):
            return _error("'input' array must contain only strings")
        texts = raw_input
    else:
        return _error("'input' must be a string or array of strings")

    try:
        vectors = await manager.aembed(model_id, texts)
    except RuntimeError as exc:
        return _error(str(exc), code="dependency_missing", status=503)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("embedding failure for %s", model_id)
        return _error(f"embedding failed: {exc}", code="server_error", status=500)

    total_tokens = sum(_approx_tokens(t) for t in texts)
    data = [
        {"object": "embedding", "embedding": vec, "index": idx}
        for idx, vec in enumerate(vectors)
    ]
    return web.json_response(
        {
            "object": "list",
            "data": data,
            "model": model_id,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        }
    )


async def handle_list_embedding_models(request: web.Request) -> web.Response:
    manager: EmbeddingManager = request.app["embedding_manager"]
    loaded_ids = {m["id"] for m in manager.list_loaded()}
    models = []
    for entry in manager.list_known():
        info = dict(entry)
        info["loaded"] = info["id"] in loaded_ids
        models.append(info)
    return web.json_response({"models": models, "count": len(models)})


async def handle_load_embedding_model(request: web.Request) -> web.Response:
    manager: EmbeddingManager = request.app["embedding_manager"]
    model_id = request.match_info.get("model_id", "")
    if not model_id:
        return _error("model_id required")

    if manager.is_loaded(model_id):
        meta = next(
            (m for m in manager.list_loaded() if m["id"] == model_id), None
        )
        return web.json_response(
            {"ok": True, "model_id": model_id, "status": "loaded", "model": meta}
        )

    try:
        meta = await manager.aload(model_id)
    except RuntimeError as exc:
        return _error(str(exc), code="dependency_missing", status=503)
    except Exception as exc:
        logger.exception("failed to load embedding model %s", model_id)
        return _error(f"failed to load: {exc}", code="server_error", status=500)

    return web.json_response(
        {"ok": True, "model_id": model_id, "status": "loaded", "model": meta}
    )


async def handle_unload_embedding_model(request: web.Request) -> web.Response:
    manager: EmbeddingManager = request.app["embedding_manager"]
    model_id = request.match_info.get("model_id", "")
    if not model_id:
        return _error("model_id required")
    unloaded = manager.unload(model_id)
    return web.json_response(
        {
            "ok": unloaded,
            "model_id": model_id,
            "status": "unloaded" if unloaded else "not_loaded",
        }
    )


__all__ = [
    "register_embedding_routes",
    "fleet_candidates",
    "forward_to_fleet",
    "KNOWN_EMBEDDING_MODELS",
]
