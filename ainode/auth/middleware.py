"""API key authentication middleware for aiohttp.

The rule, in one place: **when auth is enabled every path under ``/api`` and
``/v1`` needs the key.** The exceptions are the four things a caller with no key
must still be able to reach:

* the static shell (``/``, ``/static/*``),
* ``/api/health`` (liveness, for a probe that has no key),
* ``/api/auth/status`` (so the UI can say "this node wants a key" instead of
  rendering an empty page),
* ``POST /api/cluster/join`` (a node joining this cluster does not have this
  cluster's key yet, so a single-use expiring join token is the credential
  instead; see ``api/cluster_join.py`` for the rate limit that replaces the key).

The browser onboarding wizard used to be a fifth exemption, open while
``config.onboarded`` was false. The wizard is gone (#208): it was unreachable on
every deployed node, because the installer and every non-TTY start set
``onboarded`` before the server came up, and it never joined a cluster even when
reached.

Every request is stamped with ``request["authenticated"]`` -- True only when a
Bearer token matched a stored key hash. Handlers read it through
``is_authenticated()`` to gate the few fields that are dangerous even when auth
is switched off (``trust_remote_code``; see ``TRUST_REMOTE_CODE_RULE``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass, field, asdict

from aiohttp import web

from ainode.core.config import AINODE_HOME


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


AUTH_FILE = AINODE_HOME / "auth.json"

SKIP_PATHS: set[str] = {"/", "/api/health", "/api/auth/status", "/api/cluster/join"}
SKIP_PREFIXES: tuple[str, ...] = ("/static/",)

#: ``request`` key carrying the outcome of token validation for this request.
AUTHENTICATED_KEY = "authenticated"

#: Printed back to any caller refused a ``trust_remote_code`` escalation, and
#: quoted in the CHANGELOG. One sentence of rule, one of how to comply.
TRUST_REMOTE_CODE_RULE = (
    "trust_remote_code makes the engine execute Python from the model "
    "repository inside the engine container, so it can only be set by a "
    "request that presents an API key (dashboard: Config > API access, or "
    "Authorization: Bearer <key>), or by loading a curated catalog model whose "
    "recipe already declares it."
)

#: What a 401 tells the caller. The dashboard turns this into its API access
#: panel; a curl user gets the header to send.
MISSING_KEY_MESSAGE = (
    "This node requires an API key. Send Authorization: Bearer <key>, or paste "
    "the key into the dashboard under Config > API access."
)


@dataclass
class AuthConfig:
    enabled: bool = False
    api_keys: list[dict] = field(default_factory=list)
    # Each key entry: {"id": "<short-id>", "key_hash": "<sha256 hex>"}

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

    def generate_key(self) -> dict:
        key = secrets.token_hex(16)
        key_id = secrets.token_hex(4)
        key_hash = _hash_key(key)
        self.api_keys.append({"id": key_id, "key_hash": key_hash})
        self.save()
        return {"id": key_id, "key": key}

    def revoke_key(self, key_id: str) -> bool:
        before = len(self.api_keys)
        self.api_keys = [k for k in self.api_keys if k["id"] != key_id]
        if len(self.api_keys) != before:
            self.save()
            return True
        return False

    def key_ids(self) -> list[dict]:
        """The stored keys as the UI may see them: ids only, never a hash."""
        return [{"id": entry.get("id", "")} for entry in self.api_keys]

    def validate_token(self, token: str) -> bool:
        token_hash = _hash_key(token)
        for entry in self.api_keys:
            stored = entry.get("key_hash") or entry.get("key", "")
            if hmac.compare_digest(token_hash, stored):
                return True
        return False

    def enable(self) -> dict:
        """Turn auth on, minting a first key only when there is none.

        Returns ``{"id": ..., "key": <plaintext> | None}``. The plaintext is
        there only for a key minted by this call: the file stores hashes, so an
        existing key cannot be shown again and the caller has to say so rather
        than hand back a ``KeyError`` (it used to: ``entry["key"]`` on a reused
        entry 500'd both ``POST /api/auth/enable`` and ``ainode auth enable``).
        """
        self.enabled = True
        if not self.api_keys:
            entry = self.generate_key()
            self.save()
            return {"id": entry["id"], "key": entry["key"]}
        self.save()
        return {"id": self.api_keys[0].get("id", ""), "key": None}

    def disable(self) -> None:
        self.enabled = False
        self.save()


def bearer_token(request: web.Request) -> str:
    """The Bearer token on this request, or "" when there is none."""
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return header[7:].strip()
    return ""


def is_authenticated(request) -> bool:
    """True when this request presented a token matching a stored key.

    Independent of whether auth is ENABLED: a node running open can still have
    a key, and presenting it is what separates the operator from anyone else who
    can reach the port.

    Tolerant of a request-like object with no mapping interface, because several
    tests call handlers with a stub request; a stub reads as not authenticated,
    which is the safe direction.
    """
    getter = getattr(request, "get", None)
    if not callable(getter):
        return False
    return bool(getter(AUTHENTICATED_KEY, False))


def _should_skip(request: web.Request) -> bool:
    path = request.path
    if path in SKIP_PATHS:
        return True
    return path.startswith(SKIP_PREFIXES)


@web.middleware
async def auth_middleware(request: web.Request, handler):
    auth_cfg: AuthConfig | None = request.app.get("auth_config")
    token = bearer_token(request)
    # Stamped on every request, enabled or not: handlers gate on it (see
    # is_authenticated) even when the node is running open.
    request[AUTHENTICATED_KEY] = bool(
        token and auth_cfg is not None and auth_cfg.validate_token(token)
    )
    if auth_cfg is None or not auth_cfg.enabled:
        return await handler(request)
    if _should_skip(request):
        return await handler(request)
    if not token:
        return web.json_response(
            {"error": {"message": MISSING_KEY_MESSAGE, "type": "auth_error"}},
            status=401,
        )
    if not request[AUTHENTICATED_KEY]:
        return web.json_response(
            {"error": {"message": "Invalid API key", "type": "auth_error"}},
            status=401,
        )
    return await handler(request)
