"""API key authentication middleware for aiohttp."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field, asdict
from aiohttp import web

from ainode.core.config import AINODE_HOME

AUTH_FILE = AINODE_HOME / "auth.json"

# Paths that never require authentication
SKIP_PATHS: set[str] = {
    "/",
    "/onboarding",
    "/api/health",
}
SKIP_PREFIXES: tuple[str, ...] = (
    "/static/",
    "/api/onboarding/",
)


@dataclass
class AuthConfig:
    """Persisted auth state."""

    enabled: bool = False
    api_keys: list[dict] = field(default_factory=list)
    # Each key entry: {"id": "<short-id>", "key": "<hex>"}

    # -- persistence ----------------------------------------------------------

    def save(self) -> None:
        AINODE_HOME.mkdir(parents=True, exist_ok=True)
        AUTH_FILE.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls) -> "AuthConfig":
        if AUTH_FILE.exists():
            data = json.loads(AUTH_FILE.read_text())
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        return cls()

    # -- key management -------------------------------------------------------

    def generate_key(self) -> dict:
        """Create a new API key, append it, save, and return the entry."""
        key = secrets.token_hex(16)  # 32-char hex string
        key_id = key[:8]
        entry = {"id": key_id, "key": key}
        self.api_keys.append(entry)
        self.save()
        return entry

    def revoke_key(self, key_id: str) -> bool:
        """Remove a key by its id. Returns True if found and removed."""
        before = len(self.api_keys)
        self.api_keys = [k for k in self.api_keys if k["id"] != key_id]
        if len(self.api_keys) != before:
            self.save()
            return True
        return False

    def valid_keys(self) -> set[str]:
        """Return the set of currently valid raw key strings."""
        return {k["key"] for k in self.api_keys}

    def enable(self) -> dict:
        """Enable auth. Generates a default key if none exist. Returns the key entry."""
        self.enabled = True
        if not self.api_keys:
            entry = self.generate_key()  # also saves
        else:
            entry = self.api_keys[0]
            self.save()
        return entry

    def disable(self) -> None:
        """Disable auth (keys are kept but not enforced)."""
        self.enabled = False
        self.save()


def _should_skip(request: web.Request) -> bool:
    """Return True if this request should bypass auth checks."""
    path = request.path
    if path in SKIP_PATHS:
        return True
    if path.startswith(SKIP_PREFIXES):
        return True
    # GET requests to onboarding pages
    if request.method == "GET" and path.startswith("/api/onboarding"):
        return True
    return False


@web.middleware
async def auth_middleware(request: web.Request, handler):
    """Check Bearer token when auth is enabled."""
    auth_cfg: AuthConfig | None = request.app.get("auth_config")

    # If auth is not configured or disabled, pass through
    if auth_cfg is None or not auth_cfg.enabled:
        return await handler(request)

    # Skip exempt paths
    if _should_skip(request):
        return await handler(request)

    # Check Authorization header
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return web.json_response(
            {"error": {"message": "Missing or invalid Authorization header", "type": "auth_error"}},
            status=401,
        )

    token = auth_header[7:]  # strip "Bearer "
    if token not in auth_cfg.valid_keys():
        return web.json_response(
            {"error": {"message": "Invalid API key", "type": "auth_error"}},
            status=401,
        )

    return await handler(request)
