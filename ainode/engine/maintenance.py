"""Persistent, timed suppression of automatic engine launches."""

import json
import math
import os
import tempfile
import time

from aiohttp import web

from ainode.core import config as core_config


def hold_path():
    return core_config.AINODE_HOME / "maintenance.json"


def maintenance(now=None):
    """Keep expired windows visible so a watchdog can report a missed deadline."""
    try:
        value = json.loads(hold_path().read_text())
        until = float(value["until"])
        reason = value["reason"]
        if not math.isfinite(until) or not isinstance(reason, str):
            return None
        return {
            "reason": reason, "until": until,
            "active": until > (time.time() if now is None else now),
        }
    except (OSError, ValueError, TypeError, KeyError):
        return None


def held():
    return bool((maintenance() or {}).get("active"))


def set_maintenance(value):
    if value is None:
        hold_path().unlink(missing_ok=True)
        return
    if not isinstance(value, dict):
        raise ValueError("maintenance must be an object or null")
    reason = value.get("reason")
    until = value.get("until")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 256:
        raise ValueError("maintenance.reason must contain 1 to 256 characters")
    if isinstance(until, bool) or not isinstance(until, (int, float)):
        raise ValueError("maintenance.until must be an epoch timestamp")
    if not math.isfinite(until) or until <= time.time():
        raise ValueError("maintenance.until must be a finite future timestamp")
    path = hold_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".maintenance-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"reason": reason.strip(), "until": until}, stream)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def apply_maintenance(body):
    if "maintenance" in body:
        set_maintenance(body["maintenance"])


def maintenance_refusal(request, body=None, *, direct=False):
    """Authorize the original caller before any hold mutation or forwarding."""
    from ainode.auth.accounts import may_control_host
    from ainode.auth.fleet import FLEET_KEY_ID

    if not direct and (not isinstance(body, dict) or "maintenance" not in body):
        return None
    getter = getattr(request, "get", lambda key, default=None: default)
    allowed = may_control_host(request)
    if getter("api_key_id") == FLEET_KEY_ID:
        allowed = allowed and request.headers.get("X-AINode-Maintenance-Authorized") == "true"
    if not allowed:
        return web.json_response({"error": "Maintenance host controls require an admin or operator"}, status=403)
    return None


async def handle_maintenance(request):
    """POST /api/engine/hold {reason, until}; null clears. GET reads the window."""
    if request.method == "POST":
        refused = maintenance_refusal(request, direct=True)
        if refused is not None:
            return refused
        try:
            value = await request.json()
            if isinstance(value, dict) and "maintenance" in value:
                value = value["maintenance"]
            set_maintenance(value)
        except (ValueError, TypeError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"maintenance": maintenance()})
