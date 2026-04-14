"""HTTP routes for secret management.

Secrets are always masked in responses. The only way to retrieve a raw value
is via :meth:`SecretsManager.all(include_values=True)` from Python code (e.g.
the training or download pipeline).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

import aiohttp
from aiohttp import web

from ainode.secrets.manager import KNOWN_SECRETS, SecretsManager


logger = logging.getLogger(__name__)


def _manager(request: web.Request) -> SecretsManager:
    mgr = request.app.get("secrets_manager")
    if mgr is None:
        raise web.HTTPInternalServerError(reason="SecretsManager not configured")
    return mgr


async def list_secrets(request: web.Request) -> web.Response:
    """Return all known secrets, MASKED."""
    mgr = _manager(request)
    known: Dict[str, Any] = mgr.all(include_values=False)
    custom: Dict[str, Any] = mgr.custom_all(include_values=False)
    return web.json_response({"known": known, "custom": custom})


async def set_secret(request: web.Request) -> web.Response:
    mgr = _manager(request)
    key = request.match_info["key"]
    if key not in KNOWN_SECRETS:
        return web.json_response(
            {"error": {"message": f"Unknown secret key '{key}'", "type": "invalid_request"}},
            status=400,
        )
    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": {"message": "Invalid JSON body", "type": "invalid_request"}},
            status=400,
        )
    value = body.get("value", "")
    try:
        mgr.set(key, value)
    except (KeyError, ValueError) as exc:
        # NOTE: exc never contains the value itself (only a generic message).
        return web.json_response(
            {"error": {"message": str(exc), "type": "invalid_request"}},
            status=400,
        )
    return web.json_response({"ok": True, "key": key, "masked": mgr.all()[key]["masked"]})


async def delete_secret(request: web.Request) -> web.Response:
    mgr = _manager(request)
    key = request.match_info["key"]
    if key not in KNOWN_SECRETS:
        return web.json_response(
            {"error": {"message": f"Unknown secret key '{key}'", "type": "invalid_request"}},
            status=400,
        )
    removed = mgr.delete(key)
    return web.json_response({"ok": True, "removed": removed})


async def list_custom_secrets(request: web.Request) -> web.Response:
    mgr = _manager(request)
    return web.json_response({"custom": mgr.custom_all(include_values=False)})


async def set_custom_secret(request: web.Request) -> web.Response:
    mgr = _manager(request)
    name = request.match_info["name"]
    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": {"message": "Invalid JSON body", "type": "invalid_request"}},
            status=400,
        )
    value = body.get("value", "")
    try:
        mgr.custom_set(name, value)
    except ValueError as exc:
        return web.json_response(
            {"error": {"message": str(exc), "type": "invalid_request"}},
            status=400,
        )
    return web.json_response({"ok": True, "name": name, "masked": mgr.custom_all()[name]["masked"]})


async def delete_custom_secret(request: web.Request) -> web.Response:
    mgr = _manager(request)
    name = request.match_info["name"]
    removed = mgr.custom_delete(name)
    return web.json_response({"ok": True, "removed": removed})


async def test_secret(request: web.Request) -> web.Response:
    """Attempt to validate a secret against its service. Never echoes the value."""
    mgr = _manager(request)
    key = request.match_info["key"]
    if key not in KNOWN_SECRETS:
        return web.json_response(
            {"error": {"message": f"Unknown secret key '{key}'", "type": "invalid_request"}},
            status=400,
        )
    meta = KNOWN_SECRETS[key]
    if not meta.get("testable"):
        return web.json_response(
            {"ok": False, "message": "No test handler configured for this key"},
            status=400,
        )
    raw = mgr.get(key)
    if not raw:
        return web.json_response({"ok": False, "message": "Secret is not set"}, status=400)

    if meta["testable"] == "huggingface":
        return await _test_huggingface(raw)

    return web.json_response({"ok": False, "message": "Unknown test handler"}, status=400)


async def _test_huggingface(token: str) -> web.Response:
    """Call HF whoami with the token. Returns identity on success."""
    url = "https://huggingface.co/api/whoami-v2"
    headers = {"Authorization": f"Bearer {token}"}
    timeout = aiohttp.ClientTimeout(total=8)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return web.json_response({
                        "ok": True,
                        "service": "huggingface",
                        "identity": data.get("name") or data.get("fullname") or "authenticated",
                    })
                return web.json_response(
                    {"ok": False, "service": "huggingface", "message": f"HTTP {resp.status}"},
                    status=200,
                )
    except asyncio.TimeoutError:
        return web.json_response(
            {"ok": False, "service": "huggingface", "message": "Timed out"},
            status=200,
        )
    except Exception as exc:
        # Never log the token; only log the exception type.
        logger.warning("HF token test failed: %s", type(exc).__name__)
        return web.json_response(
            {"ok": False, "service": "huggingface", "message": "Connection error"},
            status=200,
        )


def register_secrets_routes(app: web.Application) -> None:
    """Attach /api/secrets/* routes to the given aiohttp app."""
    app.router.add_get("/api/secrets", list_secrets)
    app.router.add_get("/api/secrets/custom", list_custom_secrets)
    app.router.add_put("/api/secrets/custom/{name}", set_custom_secret)
    app.router.add_delete("/api/secrets/custom/{name}", delete_custom_secret)
    app.router.add_get("/api/secrets/{key}/test", test_secret)
    app.router.add_put("/api/secrets/{key}", set_secret)
    app.router.add_delete("/api/secrets/{key}", delete_secret)
