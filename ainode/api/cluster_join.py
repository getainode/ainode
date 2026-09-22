"""The two join routes: the one a joiner calls, and the one this node's UI calls.

``POST /api/cluster/join`` runs on the MASTER. It is the only route besides
``/`` and ``/static/*``, ``/api/health``, ``/api/auth/status``,
``/api/auth/login``, ``/api/auth/me`` and ``/api/cluster/endpoint`` that answers
without an API key, and the
reason is structural: the node calling it has not joined yet, so it cannot hold
this cluster's key. The join token IS the credential (32 random bytes, stored
hashed, single use, expiring), and everything that follows from that lives here:

* a wrong, expired and spent token get the SAME 403 body, so the route cannot be
  used to tell a real token from a guess,
* the route is rate limited to 5 attempts per minute per source IP INSIDE the
  handler, because there is no key in front of it to do it,
* a success is logged with the token id and the joining node's name, because
  handing over the cluster secret is the one event on this node an operator will
  want to find afterwards.

``POST /api/cluster/join-self`` runs on the JOINER, behind the normal API-key
rule. It is what the dashboard's "Join a cluster" card calls: it performs the same
join in-process, writes the same six config keys, and deliberately restarts
nothing. Restarting the service from inside the request that asked for it would
kill the response before the browser could read it, so the answer says what to run
instead.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import aiohttp
from aiohttp import web

from ainode import __version__
from ainode.cluster.join import (
    DEFAULT_WEB_PORT,
    REFUSED_MESSAGE,
    JoinTokenStore,
    apply_join,
    join_url,
    parse_host_port,
    version_refusal,
)
from ainode.core.config import NodeConfig

logger = logging.getLogger(__name__)

#: Attempts one source IP may make against ``/api/cluster/join`` per window.
JOIN_RATE_LIMIT = 5
#: The window those attempts are counted over, in seconds.
JOIN_RATE_WINDOW = 60.0
#: ``app`` key holding the per-IP attempt log.
RATE_STATE_KEY = "join_rate_limit"
#: Sources tracked at once. A forged flood from random addresses must not be able
#: to grow this dict without bound; past the cap the oldest source is dropped,
#: which at worst gives an attacker back the attempts it had already spent.
RATE_MAX_SOURCES = 512


def register_cluster_join_routes(app: web.Application) -> None:
    """Register both join routes on the app.

    The rate-limit log is seeded HERE, not on first use: an aiohttp Application is
    read-only once it has started, so a handler that created the key would be
    mutating a started app (a DeprecationWarning today, an error later).
    """
    app[RATE_STATE_KEY] = {}
    app.router.add_post("/api/cluster/join", handle_cluster_join)
    app.router.add_post("/api/cluster/join-self", handle_cluster_join_self)


# =============================================================================
# Rate limiting: a pure function over a mutable log, so a test can drive time
# =============================================================================

def rate_limit_check(state: dict, source: str, now: float,
                     limit: int = JOIN_RATE_LIMIT,
                     window: float = JOIN_RATE_WINDOW) -> bool:
    """Record an attempt from ``source``; True when it is allowed.

    ``state`` maps a source to the timestamps of its recent attempts. Timestamps
    outside the window are dropped on every call, so the log is self-trimming and
    a quiet source costs nothing.
    """
    attempts = [t for t in state.get(source, []) if now - t < window]
    if len(attempts) >= limit:
        state[source] = attempts
        return False
    attempts.append(now)
    state[source] = attempts
    if len(state) > RATE_MAX_SOURCES:
        # Drop the least recently active sources rather than growing forever.
        ordered = sorted(state.items(), key=lambda kv: max(kv[1] or [0]))
        for key, _ in ordered[: len(state) - RATE_MAX_SOURCES]:
            if key != source:
                state.pop(key, None)
    return True


def _source_ip(request: web.Request) -> str:
    """The peer address, as the socket reports it.

    Deliberately NOT X-Forwarded-For: that header is caller-supplied, so trusting
    it would let one client mint itself a fresh rate-limit bucket per request.
    """
    peer = request.transport.get_extra_info("peername") if request.transport else None
    if isinstance(peer, tuple) and peer:
        return str(peer[0])
    return str(request.remote or "unknown")


def _refused() -> web.Response:
    """The one answer a failed join gets, whatever the reason."""
    return web.json_response(
        {"error": {"message": REFUSED_MESSAGE, "type": "join_refused"}},
        status=403,
    )


# =============================================================================
# The master's side
# =============================================================================

def master_address_for(request: web.Request) -> str:
    """The address a joiner should write down for this master.

    In order: an explicitly configured ``master_address`` (an operator who pinned
    one meant it), then the Host header the joiner actually reached us on (the one
    address known to work from where the joiner is standing), then this node's own
    detected address. A Host header with no port gets this node's web port.
    """
    config: NodeConfig = request.app["config"]
    configured = (getattr(config, "master_address", None) or "").strip()
    if configured:
        return configured
    web_port = int(getattr(config, "web_port", DEFAULT_WEB_PORT) or DEFAULT_WEB_PORT)
    host_header = (request.headers.get("Host") or "").strip()
    if host_header:
        try:
            host, port = parse_host_port(host_header, default_port=web_port)
            bracketed = f"[{host}]" if ":" in host else host
            return f"{bracketed}:{port}"
        except ValueError:
            pass
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            local = sock.getsockname()[0]
        finally:
            sock.close()
    except OSError:
        local = socket.gethostname()
    return f"{local}:{web_port}"


async def handle_cluster_join(request: web.Request) -> web.Response:
    """POST /api/cluster/join {token, node_name} -- hand a joiner the cluster.

    Keyless by design (see the module docstring). Answers 200 with everything the
    joiner needs to write, or 403 with one message for every kind of bad token.
    """
    state = request.app.get(RATE_STATE_KEY)
    if state is None:  # an app built without register_cluster_join_routes
        state = {}
    source = _source_ip(request)
    if not rate_limit_check(state, source, time.time()):
        logger.warning(
            "join refused: %s is over %d attempts in %.0fs",
            source, JOIN_RATE_LIMIT, JOIN_RATE_WINDOW,
        )
        return web.json_response(
            {"error": {
                "message": (
                    f"Too many join attempts from this address. "
                    f"{JOIN_RATE_LIMIT} per {int(JOIN_RATE_WINDOW)} seconds."
                ),
                "type": "rate_limited",
            }},
            status=429,
            headers={"Retry-After": str(int(JOIN_RATE_WINDOW))},
        )

    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": {"message": "Invalid JSON body", "type": "invalid_request"}},
            status=400,
        )
    if not isinstance(body, dict):
        return web.json_response(
            {"error": {"message": "Body must be an object", "type": "invalid_request"}},
            status=400,
        )

    token = str(body.get("token") or "").strip()
    node_name = str(body.get("node_name") or "").strip()[:64]

    store = JoinTokenStore()
    record = store.consume(token, used_by=node_name or source)
    if record is None:
        logger.warning("join refused from %s (node_name=%r)", source, node_name)
        return _refused()

    config: NodeConfig = request.app["config"]
    secret = (getattr(config, "cluster_secret", None) or "").strip()
    master_address = master_address_for(request)
    logger.info(
        "join accepted: token %s spent by %s from %s, cluster_id=%s, signed=%s",
        record.get("id"), node_name or "an unnamed node", source,
        getattr(config, "cluster_id", "default"), "yes" if secret else "no",
    )
    return web.json_response({
        "cluster_id": getattr(config, "cluster_id", "default"),
        # Empty when this master has no secret: the joiner still joins, and its
        # own output says the cluster's discovery is unauthenticated.
        "cluster_secret": secret,
        "discovery_port": int(getattr(config, "discovery_port", 0) or 0),
        "master_address": master_address,
        "master_node_name": getattr(config, "node_name", None),
        "ainode_version": __version__,
    })


# =============================================================================
# The joiner's side, in-process (what the dashboard card calls)
# =============================================================================

async def fetch_join_payload(session: aiohttp.ClientSession, target: str,
                             token: str, node_name: str = "",
                             timeout: float = 15.0) -> tuple[Optional[dict], Optional[str], int]:
    """Call a master's join route. Returns ``(payload, error, status)``.

    Split out so the route below is only policy: a test drives this with a fake
    session, and the CLI has its own stdlib version for the same call.
    """
    url = join_url(target)
    try:
        async with session.post(
            url,
            json={"token": token, "node_name": node_name},
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            status = resp.status
            try:
                data = await resp.json()
            except Exception:
                data = None
            if status == 200 and isinstance(data, dict):
                return data, None, status
            message = ""
            if isinstance(data, dict):
                err = data.get("error")
                if isinstance(err, dict):
                    message = str(err.get("message") or "")
                elif err:
                    message = str(err)
            return None, message or f"the master answered HTTP {status}", status
    except aiohttp.ClientError as exc:
        return None, f"could not reach {url}: {exc}", 0
    except TimeoutError:
        return None, f"timed out reaching {url}", 0


async def handle_cluster_join_self(request: web.Request) -> web.Response:
    """POST /api/cluster/join-self {host, token, name?, interface?} -- join, in-process.

    Behind the API-key rule, because it is this node's own operator asking this
    node to join something. Writes the six config keys and restarts NOTHING: the
    service restart that applies them would kill this response, so the answer
    carries the command to run instead.
    """
    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": {"message": "Invalid JSON body", "type": "invalid_request"}},
            status=400,
        )
    if not isinstance(body, dict):
        return web.json_response(
            {"error": {"message": "Body must be an object", "type": "invalid_request"}},
            status=400,
        )

    target = str(body.get("host") or body.get("master") or "").strip()
    token = str(body.get("token") or "").strip()
    node_name = str(body.get("name") or body.get("node_name") or "").strip()[:64]
    interface = str(body.get("interface") or "").strip()[:32]
    allow_mismatch = bool(body.get("allow_version_mismatch"))

    if not target or not token:
        return web.json_response(
            {"error": {
                "message": "Both a master address and a join token are required.",
                "type": "invalid_request",
            }},
            status=400,
        )
    try:
        parse_host_port(target)
    except ValueError as exc:
        return web.json_response(
            {"error": {"message": str(exc), "type": "invalid_request"}},
            status=400,
        )

    session: Optional[aiohttp.ClientSession] = request.app.get("client_session")
    if session is None:
        return web.json_response(
            {"error": {"message": "This node has no HTTP client yet; try again.",
                       "type": "unavailable"}},
            status=503,
        )

    payload, error, status = await fetch_join_payload(
        session, target, token, node_name=node_name)
    if payload is None:
        # 403 from the master is the joiner's 403 too: the token was refused.
        return web.json_response(
            {"error": {"message": error or REFUSED_MESSAGE, "type": "join_failed"}},
            status=403 if status == 403 else 502,
        )

    refusal = version_refusal(__version__, payload.get("ainode_version", ""),
                              allow_mismatch=allow_mismatch)
    if refusal:
        return web.json_response(
            {"error": {"message": refusal, "type": "version_mismatch"},
             "master_version": payload.get("ainode_version", ""),
             "node_version": __version__},
            status=409,
        )

    written = apply_join(payload, node_name=node_name, interface=interface)
    logger.info("joined cluster %s via %s; wrote %s",
                written.get("cluster_id"), target, ", ".join(sorted(written)))
    return web.json_response({
        "ok": True,
        "cluster_id": written.get("cluster_id"),
        "master_address": written.get("master_address"),
        "signed_discovery": bool(written.get("cluster_secret")),
        # Never the secret itself, on a route that answers a browser.
        "written": sorted(written),
        "restart_required": True,
        "restart_command": "sudo systemctl restart ainode",
        "message": (
            "Config written. This node joins the cluster on its next start: run "
            "sudo systemctl restart ainode on the host."
        ),
    })
