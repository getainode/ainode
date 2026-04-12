"""API routes for auth management (enable/disable, key CRUD)."""

from __future__ import annotations

from aiohttp import web

from ainode.auth.middleware import AuthConfig


def register_auth_routes(app: web.Application) -> None:
    """Register auth management routes on the aiohttp app."""
    app.router.add_get("/api/auth/status", handle_auth_status)
    app.router.add_post("/api/auth/enable", handle_auth_enable)
    app.router.add_post("/api/auth/disable", handle_auth_disable)
    app.router.add_post("/api/auth/keys", handle_create_key)
    app.router.add_delete("/api/auth/keys/{key_id}", handle_revoke_key)


# -- Handlers ------------------------------------------------------------------

async def handle_auth_status(request: web.Request) -> web.Response:
    """GET /api/auth/status -- return auth state."""
    auth_cfg: AuthConfig = request.app["auth_config"]
    return web.json_response({
        "enabled": auth_cfg.enabled,
        "key_count": len(auth_cfg.api_keys),
    })


async def handle_auth_enable(request: web.Request) -> web.Response:
    """POST /api/auth/enable -- enable auth, return the (first) API key."""
    auth_cfg: AuthConfig = request.app["auth_config"]
    entry = auth_cfg.enable()
    return web.json_response({
        "enabled": True,
        "api_key": entry["key"],
        "key_id": entry["id"],
    })


async def handle_auth_disable(request: web.Request) -> web.Response:
    """POST /api/auth/disable -- disable auth."""
    auth_cfg: AuthConfig = request.app["auth_config"]
    auth_cfg.disable()
    return web.json_response({"enabled": False})


async def handle_create_key(request: web.Request) -> web.Response:
    """POST /api/auth/keys -- generate a new API key."""
    auth_cfg: AuthConfig = request.app["auth_config"]
    entry = auth_cfg.generate_key()
    return web.json_response({
        "api_key": entry["key"],
        "key_id": entry["id"],
    })


async def handle_revoke_key(request: web.Request) -> web.Response:
    """DELETE /api/auth/keys/:key_id -- revoke a key."""
    key_id = request.match_info["key_id"]
    auth_cfg: AuthConfig = request.app["auth_config"]
    revoked = auth_cfg.revoke_key(key_id)
    if not revoked:
        return web.json_response(
            {"error": f"Key '{key_id}' not found"},
            status=404,
        )
    return web.json_response({"revoked": True, "key_id": key_id})
