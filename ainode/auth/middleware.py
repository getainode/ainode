"""Authentication middleware for aiohttp: an API key, a login, or the fleet.

The rule, in one place: **when auth is enabled every path under ``/api`` and
``/v1`` needs a credential.** The exceptions are the seven things a caller with
none must still be able to reach:

* the static shell (``/``, ``/static/*``),
* ``/api/health`` (liveness, for a probe that has no key),
* ``/api/auth/status`` (so the UI can say "this node wants a key" instead of
  rendering an empty page),
* ``/api/auth/login`` (the door: a caller with no credential is exactly who
  posts to it),
* ``/api/auth/me`` (which answers ``{"user": null}`` to a caller with no
  session, so the dashboard can draw the login page instead of guessing),
* ``/api/cluster/endpoint`` (node names, addresses and ports, nothing else: a
  client whose configured node is down has to be able to ask a reachable one
  where the rest of the fleet is, and the key it holds does not help it find an
  address. It carries no model, no telemetry, no config and no key material.)
* ``POST /api/cluster/join`` (a node joining this cluster does not have this
  cluster's key yet, so a single-use expiring join token is the credential
  instead; see ``api/cluster_join.py`` for the rate limit that replaces the key).

The browser onboarding wizard used to be one more exemption, open while
``config.onboarded`` was false. The wizard is gone (#208): it was unreachable on
every deployed node, because the installer and every non-TTY start set
``onboarded`` before the server came up, and it never joined a cluster even when
reached.

Every request is stamped with ``request["authenticated"]`` -- True when a
credential matched -- and with ``request["api_key_id"]``, who matched. Handlers
read the first through ``is_authenticated()`` to gate the few fields that are
dangerous even when auth is switched off (``trust_remote_code``; see
``TRUST_REMOTE_CODE_RULE``); the rate limiter reads the second so a keyed caller
gets its own budget.

**Three credentials, in one pass.**

1. An operator API key, from ``Authorization: Bearer``, stamped as its key id.
2. **The fleet.** A peer presenting the key derived from this node's
   ``cluster_secret`` is accepted as ``api_key_id == "fleet"``, which is what
   lets a cluster run with auth on everywhere (see ``auth/fleet.py`` for the
   derivation, and why it is a derivation). It is checked against the LIVE
   secret on every request, so rotating ``cluster_secret`` rotates the fleet's
   access with no restart, and a node whose secret differs from the rest of the
   cluster refuses them exactly as it drops their unverifiable discovery
   datagrams.
3. **A login session**, from the ``ainode_session`` cookie (#261), stamped as
   ``api_key_id == "user:<name>"`` with ``request["user"]`` and
   ``request["user_role"]`` beside it. ``auth/accounts.py`` owns the accounts
   and the sessions; this module only decides whether one is presented.

A Bearer token is tried FIRST and a cookie only when no key matched. That is
"the key wins" for a caller holding both, without locking out the one case that
really happens: a browser that logged in while a stale key was still sitting in
``localStorage`` would otherwise 401 on every request with a valid session in
the jar.

**The cookie carries a CSRF rule the other two do not need.** A browser attaches
a cookie to a cross-site request on its own, so a cookie alone would let any page
on the internet POST to a node the operator is logged into. A Bearer token cannot
be attached by anybody but the caller, so it needs nothing. The rule: a
cookie-authenticated request whose method is not GET, HEAD or OPTIONS must carry
``X-AINode-Client: dashboard``. A custom header cannot be set cross-origin
without a CORS preflight this node never approves, so the header IS the proof
that the caller is the dashboard's own code and not somebody else's page. Without
it the cookie is ignored: the request proceeds as unauthenticated, which with
auth on is a 401 whose message names the missing header rather than claiming the
login was bad.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import stat
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from aiohttp import web

from ainode.auth.accounts import USER_KEY_PREFIX, UsersStore
from ainode.auth.fleet import FLEET_KEY_ID, cluster_secret_of, is_fleet_key
from ainode.core.config import AINODE_HOME


logger = logging.getLogger(__name__)


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _file_stamp(path) -> Optional[tuple]:
    """``(mtime_ns, size)`` for *path*, or None when it is not there.

    The same cheap identity ``discovery/signing.py::ClusterSecret`` uses to tell
    "this file changed" from "this file is the one I already read", so a per
    request freshness check costs a stat rather than a parse.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


AUTH_FILE = AINODE_HOME / "auth.json"

SKIP_PATHS: set[str] = {"/", "/api/health", "/api/auth/status",
                        "/api/auth/login", "/api/auth/me",
                        "/api/cluster/endpoint", "/api/cluster/join"}
SKIP_PREFIXES: tuple[str, ...] = ("/static/",)

#: The cookie a logged-in browser carries. HttpOnly, so no script on the page can
#: read it (``auth/session_routes.py`` writes it and owns the attributes).
SESSION_COOKIE = "ainode_session"

#: The header a cookie-authenticated write must carry, and its one accepted
#: value. See the module docstring for why a custom header is the CSRF proof.
CLIENT_HEADER = "X-AINode-Client"
DASHBOARD_CLIENT = "dashboard"

#: Methods a cookie alone is allowed to authenticate. The three that must not
#: change anything, so a cross-site GET buys an attacker nothing it did not
#: already have from the browser's own address bar.
SAFE_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS"})

#: ``request`` key carrying the outcome of token validation for this request.
AUTHENTICATED_KEY = "authenticated"

#: ``request`` keys carrying the logged-in account, or "" when the caller is a
#: key, the fleet, or nobody. Handlers read them through ``request["user"]`` /
#: ``request["user_role"]``; ``accounts.require_admin`` is the one place that
#: turns them into a yes or no.
USER_KEY = "user"
USER_ROLE_KEY = "user_role"

#: ``request`` key carrying the id of the API key this request presented, or "".
#: Set by the same validation pass as AUTHENTICATED_KEY, so anything downstream
#: can tell WHICH key called without hashing the token again. The rate limiter
#: keys its buckets on it (ainode/ratelimit/middleware.py::client_key): two
#: callers behind one NAT are two clients when they hold different keys.
API_KEY_ID_KEY = "api_key_id"

#: Printed back to any caller refused a ``trust_remote_code`` escalation, and
#: quoted in the CHANGELOG. One sentence of rule, one of how to comply.
TRUST_REMOTE_CODE_RULE = (
    "trust_remote_code makes the engine execute Python from the model "
    "repository inside the engine container, so it can only be set by a "
    "request that presents an API key (dashboard: Config > API access, or "
    "Authorization: Bearer <key>), or by loading a curated catalog model whose "
    "recipe already declares it."
)

#: What a 401 tells the caller. The dashboard turns this into its login page or
#: its API access panel; a curl user gets the header to send. Both doors are
#: named, because a person who is told only about a Bearer token pastes a machine
#: key into a browser, which is the thing #261 exists to stop.
MISSING_KEY_MESSAGE = (
    "This node requires a login or an API key. Sign in on the dashboard, or send "
    "Authorization: Bearer <key>."
)

#: What a 401 tells a caller whose session cookie was ignored for want of the
#: client header. Distinct from MISSING_KEY_MESSAGE on purpose: the credential
#: was fine and the request shape was not, and a message blaming the login sends
#: somebody to re-type a password that was never wrong.
CSRF_HEADER_MESSAGE = (
    f"A session cookie only authenticates a write that carries "
    f"{CLIENT_HEADER}: {DASHBOARD_CLIENT}. Send that header, or use "
    f"Authorization: Bearer <key>."
)


@dataclass
class AuthConfig:
    enabled: bool = False
    api_keys: list[dict] = field(default_factory=list)
    # Each key entry: {"id", "key_hash", "name", "created_at"}. `name` is who the
    # key is FOR ("laptop", "n8n", "the installer"): a list of eight hex ids is
    # not a list an operator can revoke from with any confidence about what they
    # are switching off. Both of the newer fields are optional on read, because
    # a key minted before this release has neither and must keep working.

    # -- persistence ----------------------------------------------------------

    def __post_init__(self) -> None:
        # Identity of the file this state came from, so a live reload can tell
        # "changed" from "already read". Not a dataclass field: `asdict` walks
        # fields, and this must never be written into auth.json.
        self._stamp: Optional[tuple] = None

    def save(self) -> None:
        """Write the store 0600, through a temp file in the same directory.

        0600 because this file decides who may call the node, and the installer
        and the server both run as root, where the default umask leaves it
        world-readable. Atomic because the server re-reads it on change
        (:meth:`reload_if_changed`) and must never see half a document.
        """
        AINODE_HOME.mkdir(parents=True, exist_ok=True)
        tmp = AUTH_FILE.with_name(AUTH_FILE.name + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        os.chmod(tmp, 0o600)
        tmp.replace(AUTH_FILE)
        self._stamp = _file_stamp(AUTH_FILE)

    @classmethod
    def load(cls) -> "AuthConfig":
        cfg = cls()
        if AUTH_FILE.exists():
            # A malformed store still raises here, as it always has: a node whose
            # access control cannot be read must fail loudly at boot rather than
            # come up open. The tolerant path is reload_if_changed, which keeps
            # the state it has.
            data = json.loads(AUTH_FILE.read_text())
            for key, value in data.items():
                if key in cls.__dataclass_fields__:
                    setattr(cfg, key, value)
            cfg._stamp = _file_stamp(AUTH_FILE)
            cfg.tighten_file_mode()
        return cfg

    def tighten_file_mode(self) -> bool:
        """chmod the store to 0600 when it is wider. True when it changed.

        Once, at load: every install before this one wrote it under the default
        umask, so a node that has been running for months has a 0644 file naming
        its keys. Hashes, not keys, but the list of who may call this node is not
        public either.
        """
        try:
            mode = stat.S_IMODE(os.stat(AUTH_FILE).st_mode)
        except OSError:
            return False
        if not mode & 0o077:
            return False
        try:
            os.chmod(AUTH_FILE, 0o600)
        except OSError as exc:  # pragma: no cover - a read-only home
            logger.warning("could not chmod %s to 0600: %s", AUTH_FILE, exc)
            return False
        logger.info("tightened %s from %s to 0600", AUTH_FILE, oct(mode))
        return True

    def reload_if_changed(self) -> bool:
        """Adopt what ``auth.json`` says now. True when this changed something.

        The middleware calls it per request, so ``ainode auth enable``,
        ``disable``, ``key create`` and ``key revoke`` are LIVE on a running node:
        the CLI runs in the same container over the same bind mount, and before
        this the server held whatever the file said at ``create_app`` time, so
        turning auth on from the CLI did nothing at all until somebody restarted
        the service (and nothing said so).

        Deliberately conservative in both directions. A file that cannot be
        stat'ed or parsed leaves the state alone: a half-written document or a
        deleted file must never be the thing that drops a node's access control.
        Cost is one stat per request, the same trade ClusterSecret makes per
        datagram.
        """
        stamp = _file_stamp(AUTH_FILE)
        if stamp is None or stamp == self._stamp:
            return False
        try:
            data = json.loads(AUTH_FILE.read_text())
        except (OSError, ValueError):
            logger.warning("could not re-read %s; keeping the auth state in memory",
                           AUTH_FILE)
            return False
        if not isinstance(data, dict):
            return False
        before = (self.enabled, self.api_keys)
        self.enabled = bool(data.get("enabled", False))
        self.api_keys = list(data.get("api_keys") or [])
        self._stamp = stamp
        changed = before != (self.enabled, self.api_keys)
        if changed:
            logger.info("auth.json changed on disk: auth is now %s with %d key(s)",
                        "required" if self.enabled else "not required",
                        len(self.api_keys))
        return changed

    def generate_key(self, name: str = "") -> dict:
        """Mint a key, store its hash, and return the plaintext ONCE.

        ``name`` is free text the operator picks and nothing validates: it is a
        label for a human reading ``ainode auth key list``, never a lookup key
        (two clients may share a name; the id is the identity).
        """
        key = secrets.token_hex(16)
        key_id = secrets.token_hex(4)
        key_hash = _hash_key(key)
        self.api_keys.append({
            "id": key_id,
            "key_hash": key_hash,
            "name": str(name or "").strip(),
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        self.save()
        return {"id": key_id, "key": key, "name": str(name or "").strip()}

    def revoke_key(self, key_id: str) -> bool:
        before = len(self.api_keys)
        self.api_keys = [k for k in self.api_keys if k["id"] != key_id]
        if len(self.api_keys) != before:
            self.save()
            return True
        return False

    def key_ids(self) -> list[dict]:
        """The stored keys as a caller may see them: never a hash.

        The id, the name it was minted under and when: everything needed to
        decide which key to revoke, and nothing that helps present one.
        """
        return [{"id": entry.get("id", ""),
                 "name": str(entry.get("name") or ""),
                 "created_at": str(entry.get("created_at") or "")}
                for entry in self.api_keys]

    def identify_token(self, token: str) -> tuple[bool, str]:
        """``(matched, key id)`` for ``token``, in one constant-time pass.

        The id is what lets anything downstream tell WHICH key called, which is
        how the rate limiter gives each key its own budget. It comes back "" for
        a key entry stored without an id, so the two answers are independent: a
        match with no id is still a match.
        """
        if not token:
            return False, ""
        token_hash = _hash_key(token)
        for entry in self.api_keys:
            stored = entry.get("key_hash") or entry.get("key", "")
            if hmac.compare_digest(token_hash, stored):
                return True, str(entry.get("id") or "")
        return False, ""

    def validate_token(self, token: str) -> bool:
        return self.identify_token(token)[0]

    def key_id_for_token(self, token: str) -> str:
        """The id of the stored key ``token`` matches, or "" for no match."""
        return self.identify_token(token)[1]

    def enable(self, name: str = "") -> dict:
        """Turn auth on, minting a first key only when there is none.

        ``name`` labels the key it mints, and is ignored when there is already a
        key (nothing is minted, so there is nothing to name).

        Returns ``{"id": ..., "key": <plaintext> | None}``. The plaintext is
        there only for a key minted by this call: the file stores hashes, so an
        existing key cannot be shown again and the caller has to say so rather
        than hand back a ``KeyError`` (it used to: ``entry["key"]`` on a reused
        entry 500'd both ``POST /api/auth/enable`` and ``ainode auth enable``).
        """
        self.enabled = True
        if not self.api_keys:
            entry = self.generate_key(name)
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


def identify_caller(app, auth_cfg: "AuthConfig | None",
                    token: str) -> tuple[bool, str]:
    """``(matched, key id)`` for *token*: an operator key, or the fleet key.

    One pass over both credentials this node accepts, so every reader of
    ``request["api_key_id"]`` sees the same two answers. Operator keys are tried
    first because they are the common case and the cheaper check (a hash and a
    list walk against a file already in memory, versus an HMAC over the live
    secret, which stats ``config.json``).

    The fleet key is checked whether or not auth is ENABLED, for the same reason
    an operator key is: ``is_authenticated`` gates ``trust_remote_code`` on a node
    running open, and a load forwarded by the cluster is as much an authenticated
    act as one typed into the dashboard. It is the head that talked to the
    operator; this node is being told by a member of its own cluster.
    """
    if not token:
        return False, ""
    if auth_cfg is not None:
        matched, key_id = auth_cfg.identify_token(token)
        if matched:
            return True, key_id
    if is_fleet_key(token, cluster_secret_of(app)):
        return True, FLEET_KEY_ID
    return False, ""


def session_token(request) -> str:
    """The session token in this request's cookie jar, or "".

    ``request.cookies`` is tolerant of a request-like stub with none, because the
    handlers in this package are called with one in several tests.
    """
    cookies = getattr(request, "cookies", None) or {}
    try:
        return str(cookies.get(SESSION_COOKIE) or "")
    except Exception:  # pragma: no cover - a stub with a hostile mapping
        return ""


def csrf_header_ok(request) -> bool:
    """Is this request allowed to authenticate on a cookie alone?

    True for a safe method, and for any method carrying
    ``X-AINode-Client: dashboard``. See the module docstring: the header is the
    proof that the caller is the dashboard's own code, because a cross-origin
    page cannot set one without a preflight this node never approves.
    """
    method = str(getattr(request, "method", "GET") or "GET").upper()
    if method in SAFE_METHODS:
        return True
    headers = getattr(request, "headers", None) or {}
    try:
        sent = str(headers.get(CLIENT_HEADER) or "")
    except Exception:  # pragma: no cover - a stub with a hostile mapping
        sent = ""
    return sent.strip().casefold() == DASHBOARD_CLIENT


def identify_session(request) -> tuple[Optional[dict], bool]:
    """``(session, refused for want of the client header)`` for this request.

    The session is None whenever the cookie does not name a live one: no cookie,
    a token nobody holds, or an account that has since been removed or disabled
    (``UsersStore.session_for_token`` owns that last rule).

    The second value is True only in the one case worth a different message: the
    cookie DID name a live session and the method needed the client header, which
    was not there. The caller is refused either way; the flag is how the 401 says
    which thing to fix.
    """
    app = getattr(request, "app", None)
    getter = getattr(app, "get", None)
    store: Optional[UsersStore] = getter("users_store") if callable(getter) else None
    if store is None:
        return None, False
    token = session_token(request)
    if not token:
        return None, False
    if not csrf_header_ok(request):
        # Look the session up anyway, so the message can tell "your cookie is
        # fine, the header is missing" from "that cookie is stale". Without
        # touching last_seen: this request is about to be refused, and a refused
        # request is not activity.
        return None, store.session_for_token(token, touch=False) is not None
    return store.session_for_token(token), False


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
    if auth_cfg is not None:
        # One stat, so an `ainode auth ...` on this box is live rather than
        # waiting for a restart nobody was told to do.
        auth_cfg.reload_if_changed()
    users: UsersStore | None = request.app.get("users_store")
    if users is not None:
        # The same one stat, for the same reason: `ainode auth user add`, a
        # password change and a revoked session all land on a running server.
        users.reload_if_changed()
    token = bearer_token(request)
    # Stamped on every request, enabled or not: handlers gate on it (see
    # is_authenticated) even when the node is running open. The key id goes on
    # with it so the rate limiter can count a keyed caller as itself rather than
    # as its address (see API_KEY_ID_KEY), so a peer reads as "fleet", and so a
    # logged-in person reads as "user:<name>".
    matched, key_id = identify_caller(request.app, auth_cfg, token)
    user_name = ""
    user_role = ""
    csrf_refused = False
    if not matched:
        # Only when no key matched: a Bearer token wins, and a stale one in a
        # browser's localStorage must not cost a valid session its request.
        session, csrf_refused = identify_session(request)
        if session is not None and users is not None:
            user_name = str(session.get("user") or "")
            user_role = users.role_of(user_name)
            matched = True
            key_id = f"{USER_KEY_PREFIX}{user_name}"
    request[AUTHENTICATED_KEY] = matched
    request[API_KEY_ID_KEY] = key_id
    request[USER_KEY] = user_name
    request[USER_ROLE_KEY] = user_role
    if auth_cfg is None or not auth_cfg.enabled:
        return await handler(request)
    if _should_skip(request):
        return await handler(request)
    if matched:
        return await handler(request)
    if csrf_refused:
        return web.json_response(
            {"error": {"message": CSRF_HEADER_MESSAGE, "type": "auth_error"}},
            status=401,
        )
    if token:
        return web.json_response(
            {"error": {"message": "Invalid API key", "type": "auth_error"}},
            status=401,
        )
    return web.json_response(
        {"error": {"message": MISSING_KEY_MESSAGE, "type": "auth_error"}},
        status=401,
    )
