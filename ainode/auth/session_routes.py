"""The login routes: the door, the session, and who may hold the keys to it.

``auth/api_routes.py`` manages API keys, which are machine credentials.
This module is the half a PERSON uses (#261): ``POST /api/auth/login`` takes a
name and a password and hands back an ``HttpOnly`` cookie, ``GET /api/auth/me``
is what the dashboard asks before it decides whether to draw a login page, and
the rest is the account and session management around them. ``auth/accounts.py``
owns the storage and the rules; this module owns the HTTP.

Four things here are decisions rather than plumbing:

* **Login and ``/api/auth/me`` are open with no credential** (they are in
  ``middleware.SKIP_PATHS``), and they are safe open for opposite reasons. Login
  is the door: the caller with no credential is exactly who knocks. ``me``
  answers ``{"user": null}`` to a caller it does not recognise, and nothing else,
  so a stranger learns only what the login page already tells them. Everything
  else in this module needs a credential like every other ``/api`` path.
* **A failed login is throttled per source address**, not per name. Ten failures
  in five minutes and the address waits, because a name-keyed limiter is a
  lockout an attacker can aim at somebody else's account, and an unlimited login
  route in front of scrypt is both a guessing oracle and a way to make a GPU node
  spend its CPU on hashes.
* **The 401 says the same thing for a wrong password and a name nobody has.**
  ``UsersStore.verify_password`` pays the same scrypt cost either way, so neither
  the message nor the timing tells a stranger which accounts exist.
* **Administration accepts three callers, and the second and third are why the
  first can exist** (``accounts.require_admin``): an admin session, an operator
  API key, or the fleet key. A node with no accounts has nobody to authorise the
  first admin, so the key the installer already minted is what authorises it, and
  the fleet key is what lets a head replicate its accounts to a member. A member
  session is refused, which is the whole point of having two roles.

Replication is two routes, both fleet-key only: ``GET /api/auth/users/export``
hands out the records WITH their hashes so a member can verify a password without
asking the head, and ``POST /api/auth/users/sync`` takes them. Sessions are never
part of it (see ``accounts.py``). Every mutation calls ``app["users_changed"]``
when something registered one, which is how a head learns it has replicating to
do without this module knowing anything about the fleet.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Optional

from aiohttp import web

from ainode.auth.accounts import (
    MIN_PASSWORD_LENGTH,
    ROLE_MEMBER,
    UsersStore,
    normalize_name,
    public_session,
    public_user,
    require_admin,
)
from ainode.auth.fleet import FLEET_KEY_ID
from ainode.auth.middleware import (
    CLIENT_HEADER,
    DASHBOARD_CLIENT,
    SESSION_COOKIE,
)


logger = logging.getLogger(__name__)

#: How long the cookie lives in the browser: 400 days, the longest a browser will
#: honour. The SESSION itself has no expiry at all (see ``accounts.py``), so this
#: number is only about how long a browser keeps offering the token, not about how
#: long the node accepts it. A Max-Age far in the future is the cookie spelling of
#: "you are in until you log out".
COOKIE_MAX_AGE = 34560000

#: Failed logins one source address may make per window, and the window.
LOGIN_RATE_LIMIT = 10
LOGIN_RATE_WINDOW = 300.0
#: ``app`` key holding the per-address failure log.
LOGIN_RATE_STATE_KEY = "login_failures"
#: Addresses tracked at once. A flood from forged addresses must not grow this
#: dict without bound; past the cap the least recently active source is dropped,
#: which at worst gives an attacker back failures it had already spent.
LOGIN_RATE_MAX_SOURCES = 512

#: The one answer a failed login gets, whatever was wrong with it.
LOGIN_REFUSED_MESSAGE = "Wrong name or password"

#: What a caller is told when there is nobody to log in as. Names the command,
#: because the person reading it is looking at a login page on a node that cannot
#: let anybody in and needs a shell, not a support article.
NO_ACCOUNTS_MESSAGE = (
    "No accounts yet. Create the first admin on this node: "
    "ainode auth user add <name> --admin"
)

#: Clients a session may be opened for. Free text would make the session list
#: unreadable, and these two are the only things that log in.
CLIENTS = ("dashboard", "cli")


# =============================================================================
# Registration
# =============================================================================

def register_session_routes(app: web.Application) -> None:
    """Register the login, session and account routes.

    The throttle log is seeded HERE and not on first use: an aiohttp Application
    is read-only once it has started, so a handler that created the key would be
    mutating a started app (the same reason ``api/cluster_join.py`` seeds its
    own).

    ``/api/auth/users/export`` and ``/api/auth/users/sync`` are registered BEFORE
    ``/api/auth/users/{name}``: aiohttp resolves in registration order, and a
    variable segment registered first would swallow a literal one that follows it.
    """
    app[LOGIN_RATE_STATE_KEY] = {}

    app.router.add_post("/api/auth/login", handle_login)
    app.router.add_post("/api/auth/logout", handle_logout)
    app.router.add_get("/api/auth/me", handle_me)

    app.router.add_get("/api/auth/sessions", handle_list_sessions)
    app.router.add_delete("/api/auth/sessions/{id}", handle_revoke_session)
    app.router.add_post("/api/auth/password", handle_change_own_password)

    app.router.add_get("/api/auth/users/export", handle_export_users)
    app.router.add_post("/api/auth/users/sync", handle_sync_users)
    app.router.add_get("/api/auth/users", handle_list_users)
    app.router.add_post("/api/auth/users", handle_add_user)
    app.router.add_delete("/api/auth/users/{name}", handle_remove_user)
    app.router.add_post("/api/auth/users/{name}/password", handle_set_user_password)
    app.router.add_post("/api/auth/users/{name}/disable", handle_disable_user)
    app.router.add_post("/api/auth/users/{name}/enable", handle_enable_user)


# =============================================================================
# Small shared pieces
# =============================================================================

def _error(message: str, kind: str, status: int, **kwargs) -> web.Response:
    """The error shape every route in this module answers with."""
    return web.json_response({"error": {"message": message, "type": kind}},
                             status=status, **kwargs)


async def _body(request: web.Request) -> dict:
    """The request's JSON object, or ``{}``. Never raises.

    A handler validates the fields it needs and says which one is missing, which
    is a better answer than a 400 about JSON for a body that was simply empty.
    """
    if not request.can_read_body:
        return {}
    try:
        data = await request.json()
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _store(request: web.Request) -> UsersStore:
    """This node's account store, creating an empty one only as a fallback.

    ``create_app`` always installs one. The fallback keeps a hand-built test app
    (and any embedder that only wired the middleware) from 500ing on an
    ``AttributeError`` instead of answering "there are no accounts".
    """
    store = request.app.get("users_store")
    if store is None:
        store = UsersStore()
    return store


def _auth_enabled(request: web.Request) -> bool:
    return bool(getattr(request.app.get("auth_config"), "enabled", False))


def client_address(request: web.Request) -> str:
    """The peer address, as the socket reports it.

    Deliberately NOT ``X-Forwarded-For``: that header is caller-supplied, so
    keying a throttle on it would let one client mint itself a fresh budget per
    attempt. Same rule, same words, as ``api/cluster_join.py``.
    """
    peer = request.transport.get_extra_info("peername") if request.transport else None
    if isinstance(peer, tuple) and peer:
        return str(peer[0])
    return str(request.remote or "unknown")


def request_is_https(request: web.Request) -> bool:
    """Is the caller on HTTPS, directly or through a terminating proxy?

    Both halves matter for the cookie's ``Secure`` flag. A node with TLS on serves
    a second listener itself (``api/server.py::listener_plan``), and a node behind
    a reverse proxy sees plain HTTP with ``X-Forwarded-Proto: https``. Marking the
    cookie ``Secure`` on a plain-HTTP node would be a cookie the browser then
    refuses to send, so this cannot simply always be true.
    """
    if bool(getattr(request, "secure", False)):
        return True
    forwarded = str(request.headers.get("X-Forwarded-Proto", "") or "")
    return forwarded.split(",")[0].strip().casefold() == "https"


def session_cookie(token: str, secure: bool,
                   max_age: int = COOKIE_MAX_AGE) -> str:
    """The ``Set-Cookie`` value for a session token.

    ``HttpOnly`` so no script on the page can read it, which is what keeps an XSS
    bug from turning into a stolen session that outlives the fix. ``SameSite=Lax``
    rather than ``Strict`` because a node is reached from links, bookmarks and the
    installer's printed URL, and ``Strict`` drops the cookie on the first
    navigation from any of them: the operator lands on the dashboard logged out
    and logs in again, every time. ``Lax`` still withholds the cookie from
    cross-site POSTs, and the ``X-AINode-Client`` rule in the middleware is what
    covers the rest.
    """
    parts = [f"{SESSION_COOKIE}={token}", "Path=/", "HttpOnly", "SameSite=Lax",
             f"Max-Age={int(max_age)}"]
    if secure:
        parts.append("Secure")
    return "; ".join(parts)


def _clearing_cookie(secure: bool) -> str:
    return session_cookie("", secure, max_age=0)


def _log_hook_failure(task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning("the users_changed hook failed: %s", exc)


async def notify_users_changed(app) -> None:
    """Tell whoever registered ``app["users_changed"]`` that accounts changed.

    Optional by design: nothing in this module knows how a fleet replicates, so
    the CLI and fleet half of #261 registers a broadcaster here and this module
    stays a no-op until it does. An async hook is scheduled rather than awaited,
    because the operator's 201 must not wait on a fan-out to every peer, and a
    hook that raises is one warning rather than a failed mutation: the accounts
    are already on disk by the time it runs.
    """
    getter = getattr(app, "get", None)
    hook = getter("users_changed") if callable(getter) else None
    if not callable(hook):
        return
    try:
        result = hook()
    except Exception as exc:
        logger.warning("the users_changed hook raised: %s", exc)
        return
    if inspect.isawaitable(result):
        task = asyncio.ensure_future(result)
        task.add_done_callback(_log_hook_failure)


# =============================================================================
# The login throttle: pure functions over a mutable log, so a test drives time
# =============================================================================

def login_locked_out(state: dict, source: str, now: float,
                     limit: int = LOGIN_RATE_LIMIT,
                     window: float = LOGIN_RATE_WINDOW) -> bool:
    """Has *source* spent its failed logins for this window?

    Reads and trims; it never records. A successful login must not count against
    anybody, so recording is :func:`record_login_failure`'s job alone.
    """
    failures = [t for t in state.get(source, []) if now - t < window]
    if failures:
        state[source] = failures
    else:
        state.pop(source, None)
    return len(failures) >= limit


def record_login_failure(state: dict, source: str, now: float,
                         window: float = LOGIN_RATE_WINDOW,
                         max_sources: int = LOGIN_RATE_MAX_SOURCES) -> None:
    """Log one failed attempt from *source*, trimming the window as it goes."""
    failures = [t for t in state.get(source, []) if now - t < window]
    failures.append(now)
    state[source] = failures
    if len(state) > max_sources:
        ordered = sorted(state.items(), key=lambda kv: max(kv[1] or [0]))
        for key, _ in ordered[: len(state) - max_sources]:
            if key != source:
                state.pop(key, None)


def clear_login_failures(state: dict, source: str) -> None:
    """Forget *source*'s failures. Called on a success.

    Somebody who mistyped twice and then got it right is not halfway to a
    lockout, and leaving the failures on the log would make an honest person's
    next bad day start from eight.
    """
    state.pop(source, None)


# =============================================================================
# The door
# =============================================================================

async def handle_login(request: web.Request) -> web.Response:
    """POST /api/auth/login {name, password} -- open a session.

    200 with the account, the session id and the ``Set-Cookie``. 400 for a body
    with no name or password, 401 for anything wrong with them, 409 when this node
    has no accounts to log into, 429 when this address has spent its attempts.
    """
    store = _store(request)
    store.reload_if_changed()
    state = request.app.get(LOGIN_RATE_STATE_KEY)
    if state is None:  # an app built without register_session_routes
        state = {}
    source = client_address(request)
    if login_locked_out(state, source, time.time()):
        logger.warning("login refused: %s is over %d failures in %.0fs",
                       source, LOGIN_RATE_LIMIT, LOGIN_RATE_WINDOW)
        return _error(
            f"Too many failed logins from this address. {LOGIN_RATE_LIMIT} per "
            f"{int(LOGIN_RATE_WINDOW)} seconds.",
            "rate_limited", 429,
            headers={"Retry-After": str(int(LOGIN_RATE_WINDOW))},
        )

    if not store.has_users():
        # Not a credential failure, so it does not count against the throttle.
        # Answered whether or not auth is enabled: on a node with no accounts
        # there is no name for "wrong name or password" to be about, and the
        # caller needs the command, not a guess.
        return _error(NO_ACCOUNTS_MESSAGE, "no_accounts", 409)

    body = await _body(request)
    name = str(body.get("name") or "")
    password = str(body.get("password") or "")
    if not name or not password:
        return _error("Send a name and a password.", "invalid_request", 400)

    if not store.verify_password(name, password):
        record_login_failure(state, source, time.time())
        logger.info("login refused for %r from %s", normalize_name(name) or name,
                    source)
        return _error(LOGIN_REFUSED_MESSAGE, "auth_error", 401)

    clear_login_failures(state, source)
    token, session = store.create_session(
        name,
        client=_requested_client(request, body),
        agent=request.headers.get("User-Agent", ""),
    )
    return web.json_response(
        {"user": {"name": session["user"], "role": store.role_of(session["user"])},
         "session_id": session["id"]},
        headers={"Set-Cookie": session_cookie(token, request_is_https(request))},
    )


def _requested_client(request: web.Request, body: dict) -> str:
    """Which client this session is for: the body says, or the header, or the UI.

    The browser is the default because it is the client that cannot easily add a
    field; ``ainode auth login`` passes ``{"client": "cli"}``.
    """
    asked = str(body.get("client") or "").strip().casefold()
    if asked in CLIENTS:
        return asked
    sent = str(request.headers.get(CLIENT_HEADER, "") or "").strip().casefold()
    if sent in CLIENTS:
        return sent
    return DASHBOARD_CLIENT


async def handle_logout(request: web.Request) -> web.Response:
    """POST /api/auth/logout -- revoke this cookie's session and clear it.

    200 even when there was no session to revoke, because "log me out" has one
    meaningful answer and a browser holding a stale cookie is exactly the caller
    that needs it cleared. Not in ``SKIP_PATHS``, so on a node with auth ON a
    caller presenting nothing at all is still refused by the middleware first; a
    dashboard should read that 401 as "already logged out".
    """
    store = _store(request)
    cookie = str(request.cookies.get(SESSION_COOKIE) or "")
    revoked = False
    if cookie:
        session = store.session_for_token(cookie, touch=False)
        if session is not None:
            revoked = store.revoke_session(session.get("id"))
    return web.json_response(
        {"ok": True, "revoked": revoked},
        headers={"Set-Cookie": _clearing_cookie(request_is_https(request))},
    )


async def handle_me(request: web.Request) -> web.Response:
    """GET /api/auth/me -- who this request is, if anybody.

    The dashboard's first question, and the reason it can draw a login page
    without guessing: ``auth_enabled`` says whether a credential is required at
    all, ``has_users`` says whether logging in is even possible on this node, and
    ``user`` is null for a caller with no session (including one holding a
    perfectly good API key, which is not a person).
    """
    store = _store(request)
    payload = {
        "user": None,
        "auth_enabled": _auth_enabled(request),
        "has_users": store.has_users(),
    }
    name = str(request.get("user", "") or "")
    if not name:
        return web.json_response(payload)
    session = store.session_for_token(
        str(request.cookies.get(SESSION_COOKIE) or ""), touch=False)
    payload["user"] = {
        "name": name,
        "role": str(request.get("user_role", "") or ""),
        "session": public_session(session) if session else None,
    }
    return web.json_response(payload)


# =============================================================================
# Sessions
# =============================================================================

async def handle_list_sessions(request: web.Request) -> web.Response:
    """GET /api/auth/sessions[?user=<name>] -- sessions, mine or an account's.

    Own sessions by default. ``?user=`` names somebody else's, which takes the
    same administration check as the account routes; naming yourself is always
    allowed. A caller with no session and no query (an API key, say) gets an empty
    list rather than everybody's: a machine credential is not a person, and this
    route is a person's view of their own logins.
    """
    store = _store(request)
    me = str(request.get("user", "") or "")
    asked = normalize_name(request.query.get("user", ""))
    if request.query.get("user") and not asked:
        return _error("That is not a usable account name.", "invalid_request", 400)
    target = asked or me
    if target and target != me and not require_admin(request):
        return _error("Only an admin can read another account's sessions.",
                      "forbidden", 403)
    return web.json_response({
        "user": target or None,
        "sessions": store.sessions_for(target) if target else [],
    })


async def handle_revoke_session(request: web.Request) -> web.Response:
    """DELETE /api/auth/sessions/{id} -- end one session.

    Mine, or anybody's for an admin. 404 when there is no such session in the
    scope this caller may see, which is deliberately the same answer as "that id
    belongs to somebody else": a member must not be able to use this route to
    learn which session ids exist.
    """
    store = _store(request)
    session_id = request.match_info["id"]
    me = str(request.get("user", "") or "")
    if require_admin(request):
        owner = None
    elif me:
        owner = me
    else:
        return _error("Sign in to manage sessions.", "forbidden", 403)
    if not store.revoke_session(session_id, user=owner):
        return _error(f"No session '{session_id}' on this node.", "not_found", 404)
    return web.json_response({"revoked": True, "session_id": session_id})


async def handle_change_own_password(request: web.Request) -> web.Response:
    """POST /api/auth/password {current, new} -- change my own password.

    403 when ``current`` is wrong, which is what stops a borrowed browser from
    becoming a taken-over account. Every OTHER session of this account is revoked
    (``UsersStore.set_password``) and this one is replaced with a fresh cookie, so
    a password change logs out the laptop somebody else has and not the browser
    doing the changing.
    """
    store = _store(request)
    me = str(request.get("user", "") or "")
    if not me:
        return _error(
            "Sign in to change your own password. An operator key changes an "
            "account's password through POST /api/auth/users/<name>/password.",
            "forbidden", 403)
    body = await _body(request)
    current = str(body.get("current") or "")
    new = str(body.get("new") or "")
    if not current or not new:
        return _error("Send the current password and the new one.",
                      "invalid_request", 400)
    if not store.verify_password(me, current):
        return _error("That is not the current password.", "forbidden", 403)
    if len(new) < MIN_PASSWORD_LENGTH:
        return _error(f"A password is at least {MIN_PASSWORD_LENGTH} characters.",
                      "invalid_request", 400)
    try:
        store.set_password(me, new)
    except ValueError as exc:
        return _error(str(exc), "invalid_request", 400)
    token, session = store.create_session(
        me,
        client=_requested_client(request, body),
        agent=request.headers.get("User-Agent", ""),
    )
    await notify_users_changed(request.app)
    return web.json_response(
        {"ok": True, "session_id": session["id"]},
        headers={"Set-Cookie": session_cookie(token, request_is_https(request))},
    )


# =============================================================================
# Accounts (admin, an operator key, or the fleet key)
# =============================================================================

def admin_refusal(request) -> Optional[web.Response]:
    """403 when *request* must not administer this node, else None.

    The one gate in front of every account route, ``/api/auth/keys`` and
    ``/api/auth/enable|disable``. Three callers pass (see
    ``accounts.require_admin``), and exactly one is refused: a signed-in account
    whose role is not admin.

    A caller with no account and no key reaches here only on a node running with
    auth OFF, because the middleware refuses every ``/api`` path on a node with
    auth on. Such a node is open to anybody who can reach the port, so inventing
    a wall here would protect nothing and would make it impossible to create the
    first admin from a fresh install.
    """
    if require_admin(request):
        return None
    getter = getattr(request, "get", None)
    who = str((getter("user", "") if callable(getter) else "") or "")
    if who:
        return _error(
            f"'{who}' is not an admin on this node. An admin, an API key or a "
            f"peer of this cluster can do this.", "forbidden", 403)
    return None


def fleet_only_refusal(request) -> Optional[web.Response]:
    """403 unless *request* presented the FLEET key.

    The replication pair hands out and takes in password hashes, so it is not
    gated on "an admin" but on "a node of this cluster": the credential is derived
    from ``cluster_secret`` (``auth/fleet.py``), which is the same thing as being
    a member of the cluster. An operator who wants the accounts has the file.
    """
    getter = getattr(request, "get", None)
    key_id = str((getter("api_key_id", "") if callable(getter) else "") or "")
    if key_id == FLEET_KEY_ID:
        return None
    return _error(
        "This route is for nodes of this cluster. It authenticates with the "
        "fleet key derived from cluster_secret.", "forbidden", 403)


async def handle_list_users(request: web.Request) -> web.Response:
    """GET /api/auth/users -- every account, with no hash and no salt."""
    refused = admin_refusal(request)
    if refused is not None:
        return refused
    store = _store(request)
    return web.json_response({
        "users": store.list_users(),
        "admin_count": store.admin_count(),
    })


async def handle_add_user(request: web.Request) -> web.Response:
    """POST /api/auth/users {name, password, role} -- create an account.

    201 with the account. 400 with the rule it broke, which is the same string
    the CLI prints, because ``UsersStore.add_user`` raises it once for both.
    """
    refused = admin_refusal(request)
    if refused is not None:
        return refused
    store = _store(request)
    body = await _body(request)
    try:
        record = store.add_user(body.get("name"), body.get("password"),
                                str(body.get("role") or ROLE_MEMBER))
    except ValueError as exc:
        return _error(str(exc), "invalid_request", 400)
    await notify_users_changed(request.app)
    return web.json_response({"user": record}, status=201)


async def handle_remove_user(request: web.Request) -> web.Response:
    """DELETE /api/auth/users/{name} -- remove an account and its sessions.

    409 when it is the only admin left: a node nobody can administer is not a
    state an API should be able to put an operator in.
    """
    refused = admin_refusal(request)
    if refused is not None:
        return refused
    store = _store(request)
    name = request.match_info["name"]
    try:
        removed = store.remove_user(name)
    except ValueError as exc:
        return _error(str(exc), "conflict", 409)
    if not removed:
        return _error(f"No account named '{name}' on this node.", "not_found", 404)
    await notify_users_changed(request.app)
    return web.json_response({"removed": True, "name": normalize_name(name)})


async def handle_set_user_password(request: web.Request) -> web.Response:
    """POST /api/auth/users/{name}/password {password} -- reset a password.

    Every session that account holds is revoked, because this is the route an
    operator uses when somebody has lost a password or should no longer be logged
    in, and both cases mean the open browsers have to go.
    """
    refused = admin_refusal(request)
    if refused is not None:
        return refused
    store = _store(request)
    name = request.match_info["name"]
    body = await _body(request)
    try:
        changed = store.set_password(name, body.get("password"))
    except ValueError as exc:
        return _error(str(exc), "invalid_request", 400)
    if not changed:
        return _error(f"No account named '{name}' on this node.", "not_found", 404)
    await notify_users_changed(request.app)
    return web.json_response({"ok": True, "name": normalize_name(name)})


async def _set_disabled(request: web.Request, disabled: bool) -> web.Response:
    refused = admin_refusal(request)
    if refused is not None:
        return refused
    store = _store(request)
    name = request.match_info["name"]
    try:
        changed = store.set_disabled(name, disabled)
    except ValueError as exc:
        return _error(str(exc), "conflict", 409)
    if not changed:
        return _error(f"No account named '{name}' on this node.", "not_found", 404)
    record = store.find_user(name)
    await notify_users_changed(request.app)
    return web.json_response({"user": public_user(record or {})})


async def handle_disable_user(request: web.Request) -> web.Response:
    """POST /api/auth/users/{name}/disable -- lock an account out.

    Its sessions go with it: an account that can still act through a cookie it
    already holds is not disabled. 409 when it is the only admin left.
    """
    return await _set_disabled(request, True)


async def handle_enable_user(request: web.Request) -> web.Response:
    """POST /api/auth/users/{name}/enable -- let an account back in.

    It has no sessions afterwards, because disabling revoked them: the person logs
    in again, which is the right amount of ceremony for being let back in.
    """
    return await _set_disabled(request, False)


# =============================================================================
# Fleet replication (the fleet key only)
# =============================================================================

async def handle_export_users(request: web.Request) -> web.Response:
    """GET /api/auth/users/export -- the accounts, hashes included, for a peer.

    ``stamp`` is a content digest of exactly what ``users`` holds, so a caller can
    compare two nodes without shipping either list twice.
    """
    refused = fleet_only_refusal(request)
    if refused is not None:
        return refused
    store = _store(request)
    store.reload_if_changed()
    return web.json_response({"users": store.export_users(),
                              "stamp": store.export_stamp()})


async def handle_sync_users(request: web.Request) -> web.Response:
    """POST /api/auth/users/sync {users} -- adopt a peer's account list.

    All or nothing: a malformed record is a 400 and this node's accounts are
    untouched, because a half-adopted list is a node whose logins quietly differ
    from the rest of the cluster. ``changed`` is false for a list this node
    already has, which is what makes a replication pass cheap to repeat.

    An EMPTY list is refused with a 409 and this node keeps its accounts. The
    import replaces the whole list, so an empty push is a fleet-wide logout, and
    the sender that makes one is a master whose own store is empty (promoted
    before users.json reached it), or a release from before it refused to.
    """
    refused = fleet_only_refusal(request)
    if refused is not None:
        return refused
    store = _store(request)
    store.reload_if_changed()
    body = await _body(request)
    if "users" not in body:
        return _error("Send a 'users' list.", "invalid_request", 400)
    if isinstance(body.get("users"), list) and not body["users"]:
        logger.warning("refused an empty account list from a peer; this node keeps "
                       "its %d account(s)", len(store.users))
        return _error("Refusing an empty account list: it would delete every account "
                      "on this node. The sender's account store is empty; copy "
                      "users.json to it first.", "empty_account_list", 409)
    try:
        changed = store.import_users(body.get("users"))
    except ValueError as exc:
        return _error(str(exc), "invalid_request", 400)
    return web.json_response({"changed": changed, "count": len(store.users)})
