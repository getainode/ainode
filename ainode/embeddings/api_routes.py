"""API route handlers for embedding models.

Exposes:
- ``POST /v1/embeddings`` — OpenAI-compatible embeddings endpoint
- ``GET /api/embeddings/models`` — catalog of known embedding models
- ``POST /api/embeddings/models/{model_id}/load`` — eagerly load a model
- ``POST /api/embeddings/models/{model_id}/unload`` — drop from memory
"""

from __future__ import annotations

import logging
from typing import List

from aiohttp import web

from ainode.embeddings.manager import (
    EmbeddingManager,
    KNOWN_EMBEDDING_MODELS,
)

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def handle_v1_embeddings(request: web.Request) -> web.Response:
    """OpenAI-compatible embeddings endpoint."""
    manager: EmbeddingManager = request.app["embedding_manager"]

    try:
        body = await request.json()
    except Exception:
        return _error("Invalid JSON body")

    if not isinstance(body, dict):
        return _error("Body must be a JSON object")

    model_id = body.get("model")
    if not model_id or not isinstance(model_id, str):
        return _error("'model' is required and must be a string")

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

    # Tag the request so the server-view log shows the embedding model.
    try:
        request["_log_model"] = model_id
    except Exception:
        pass

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
    "KNOWN_EMBEDDING_MODELS",
]
