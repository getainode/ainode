"""API routes for auth management (enable/disable, key CRUD).

``GET /api/auth/status`` is the one route here the middleware leaves open: the
dashboard asks it before anything else so it can tell "this node wants a key"
apart from "this node is broken". Everything else needs the key once auth is on.

Every route here except that one is ADMINISTRATION, so since #261 it takes the
same gate the account routes take (``session_routes.admin_refusal``): an admin
session, an operator API key, or the fleet key. A member session is refused,
because a key is the whole access control on this node and "log in as a member"
must not be a way to mint one, revoke everybody else's, or switch the wall off.
"""

from __future__ import annotations

from aiohttp import web

from ainode.auth.middleware import AuthConfig, is_authenticated
from ainode.auth.session_routes import admin_refusal


def register_auth_routes(app: web.Application) -> None:
    """Register auth management routes on the aiohttp app."""
    app.router.add_get("/api/auth/status", handle_auth_status)
    app.router.add_post("/api/auth/enable", handle_auth_enable)
    app.router.add_post("/api/auth/disable", handle_auth_disable)
    app.router.add_get("/api/auth/keys", handle_list_keys)
    app.router.add_post("/api/auth/keys", handle_create_key)
    app.router.add_delete("/api/auth/keys/{key_id}", handle_revoke_key)


# -- Handlers ------------------------------------------------------------------

async def handle_auth_status(request: web.Request) -> web.Response:
    """GET /api/auth/status -- return auth state.

    Open with no key on purpose (see the module docstring). It says whether a
    key is wanted and whether the one this caller sent works; it never says
    anything about the keys themselves.
    """
    auth_cfg: AuthConfig = request.app["auth_config"]
    return web.json_response({
        "enabled": auth_cfg.enabled,
        "key_count": len(auth_cfg.api_keys),
        # Whether THIS request's key is good. The dashboard uses it to tell a
        # stale stored key from a missing one.
        "authenticated": is_authenticated(request),
    })


async def handle_auth_enable(request: web.Request) -> web.Response:
    """POST /api/auth/enable -- enable auth, return the key if one was minted.

    ``api_key`` is null when the node already had keys: only hashes are stored,
    so an existing key cannot be shown again. The caller is told to use the key
    it has, or to create a new one.
    """
    refused = admin_refusal(request)
    if refused is not None:
        return refused
    auth_cfg: AuthConfig = request.app["auth_config"]
    entry = auth_cfg.enable()
    payload = {
        "enabled": True,
        "key_id": entry["id"],
        "api_key": entry["key"],
        "key_count": len(auth_cfg.api_keys),
    }
    if not entry["key"]:
        payload["message"] = (
            "Auth enabled using the keys this node already has. They are stored "
            "hashed and cannot be shown again: use the key you have, or create "
            "a new one."
        )
    return web.json_response(payload)


async def handle_auth_disable(request: web.Request) -> web.Response:
    """POST /api/auth/disable -- disable auth."""
    refused = admin_refusal(request)
    if refused is not None:
        return refused
    auth_cfg: AuthConfig = request.app["auth_config"]
    auth_cfg.disable()
    return web.json_response({"enabled": False, "key_count": len(auth_cfg.api_keys)})


async def handle_list_keys(request: web.Request) -> web.Response:
    """GET /api/auth/keys -- id, name and age per key, so the UI can revoke one.

    Never a hash and never a plaintext: a key is shown once, at mint time, and
    after that the only operations are "list" and "revoke".
    """
    refused = admin_refusal(request)
    if refused is not None:
        return refused
    auth_cfg: AuthConfig = request.app["auth_config"]
    return web.json_response({
        "enabled": auth_cfg.enabled,
        "key_count": len(auth_cfg.api_keys),
        "keys": auth_cfg.key_ids(),
    })


async def handle_create_key(request: web.Request) -> web.Response:
    """POST /api/auth/keys -- generate a new API key.

    An optional ``name`` in the body says which client the key is for, so
    ``GET /api/auth/keys`` and ``ainode auth key list`` can name it later. A
    request with no body is the old shape and still mints an unnamed key.
    """
    refused = admin_refusal(request)
    if refused is not None:
        return refused
    auth_cfg: AuthConfig = request.app["auth_config"]
    name = ""
    try:
        if request.can_read_body:
            body = await request.json()
            if isinstance(body, dict):
                name = str(body.get("name") or "")
    except Exception:
        name = ""
    entry = auth_cfg.generate_key(name)
    return web.json_response({
        "api_key": entry["key"],
        "key_id": entry["id"],
        "name": entry["name"],
        "key_count": len(auth_cfg.api_keys),
    })


async def handle_revoke_key(request: web.Request) -> web.Response:
    """DELETE /api/auth/keys/:key_id -- revoke a key."""
    refused = admin_refusal(request)
    if refused is not None:
        return refused
    key_id = request.match_info["key_id"]
    auth_cfg: AuthConfig = request.app["auth_config"]
    revoked = auth_cfg.revoke_key(key_id)
    if not revoked:
        return web.json_response(
            {"error": f"Key '{key_id}' not found"},
            status=404,
        )
    return web.json_response({
        "revoked": True,
        "key_id": key_id,
        "key_count": len(auth_cfg.api_keys),
    })
