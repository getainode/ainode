"""Token bucket plus a concurrency cap, on ``/v1`` only.

The rule, in one place:

* **Only ``/v1`` is limited.** The dashboard's own polling lives under ``/api``
  and would trip any limit worth setting; ``/api/health`` is what a probe and a
  peer's liveness check call; the static shell is files. None of those touch an
  engine, so none of them are limited, and a browser left open on the Config page
  can never rate-limit the operator out of their own node.
* **A client is an API key id when the request carries one, else a remote IP.**
  The key id comes from the auth middleware, which has already validated the
  token by the time this runs (see ``AuthConfig.key_id_for_token``), so two
  people sharing one NAT are two clients when they hold different keys, and one
  person with one key is one client from five machines.
* **``X-Forwarded-For`` is deliberately ignored.** It is caller-supplied: a
  limiter keyed on it is a limiter anybody bypasses by varying the header. Behind
  a reverse proxy every unkeyed caller therefore collapses into one bucket, which
  is why a node behind a proxy should require API keys.
* **Two limits.** ``requests_per_minute`` with a ``burst`` is the token bucket;
  ``max_inflight`` is the concurrency cap, and it is the one that stops a single
  client from monopolising the engines. A streaming answer stays in flight until
  the stream ends, because the handler only returns after ``write_eof``.
* **A refusal is a 429 with ``Retry-After``** and a body that names which limit
  was hit, since "too many requests" without a number is not something a client
  author can act on.

Example block in ``config.json`` (off by default; these are sane values for one
GB10 node serving a handful of people):

    "rate_limit": {
      "enabled": true,
      "requests_per_minute": 600,
      "burst": 60,
      "max_inflight": 8
    }

No locking anywhere: aiohttp runs one event loop in one thread, and every state
change here happens between awaits.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from aiohttp import web

logger = logging.getLogger(__name__)

#: Paths this middleware may refuse. Everything else is exempt, ``/api`` and the
#: static shell included (see the module docstring).
LIMITED_PREFIXES: tuple[str, ...] = ("/v1/",)

#: ``error.type`` on a refusal. Matches the auth middleware's envelope shape so
#: one client-side error handler covers both.
RATE_LIMIT_TYPE = "rate_limit_error"

#: Retry-After on a concurrency refusal. A bucket refusal can be timed exactly;
#: an in-flight refusal cannot (it clears when somebody else's request finishes),
#: so the caller is told to come back in a second rather than given a made-up
#: figure.
INFLIGHT_RETRY_AFTER = 1

#: Buckets are per client and clients are usually few, but an open port on the
#: internet can invent an IP per request. Past this many tracked clients the
#: idle ones are dropped (a client at full tokens with nothing in flight has no
#: state worth keeping: recreating it gives exactly the same answer).
MAX_TRACKED_CLIENTS = 4096


@dataclass
class RateLimitConfig:
    """The ``rate_limit`` block, with the defaults an unconfigured node has."""

    enabled: bool = False
    requests_per_minute: int = 600
    burst: int = 60
    max_inflight: int = 8

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "RateLimitConfig":
        """Read the block, tolerating every shape a hand-edited file can have.

        A bad number falls back to the default instead of raising: this is read
        on the boot path, and a typo must not be the reason a node has no API.
        """
        if not isinstance(data, dict):
            return cls()
        base = cls()

        def _int(key: str, fallback: int) -> int:
            try:
                value = int(data.get(key, fallback))
            except (TypeError, ValueError):
                return fallback
            return value if value >= 0 else fallback

        return cls(
            enabled=bool(data.get("enabled", False)),
            requests_per_minute=_int("requests_per_minute", base.requests_per_minute),
            burst=_int("burst", base.burst),
            max_inflight=_int("max_inflight", base.max_inflight),
        )

    @classmethod
    def from_config(cls, config) -> "RateLimitConfig":
        return cls.from_dict(getattr(config, "rate_limit", None))

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def refill_per_second(self) -> float:
        return max(0.0, self.requests_per_minute / 60.0)

    @property
    def capacity(self) -> float:
        """Bucket size. ``burst`` 0 means "no bucket, only the concurrency cap"."""
        return float(max(0, self.burst))

    def label(self) -> str:
        """One line for the dashboard and the doctor."""
        if not self.enabled:
            return "no rate limit, one client can occupy every engine"
        parts = []
        if self.requests_per_minute and self.capacity:
            parts.append(f"{self.requests_per_minute}/min, burst {int(self.capacity)}")
        if self.max_inflight:
            parts.append(f"{self.max_inflight} in flight per client")
        if not parts:
            return "rate limiting on with no limit set, so nothing is refused"
        return ", ".join(parts)


@dataclass
class ClientState:
    """One client's bucket and in-flight count."""

    tokens: float
    updated: float
    inflight: int = 0

    def refill(self, now: float, rate: float, capacity: float) -> None:
        elapsed = max(0.0, now - self.updated)
        self.updated = now
        if rate <= 0:
            self.tokens = capacity
            return
        self.tokens = min(capacity, self.tokens + elapsed * rate)

    def idle(self) -> bool:
        return self.inflight <= 0


@dataclass
class Decision:
    """The answer for one request: allowed, or which limit refused it."""

    allowed: bool
    limit: str = ""
    limit_value: int = 0
    retry_after: int = 0
    key: str = ""


@dataclass
class RateLimiter:
    """Per-client buckets and in-flight counts for one app.

    ``clock`` is injectable so the tests drive a bucket without sleeping, and it
    is ``time.monotonic`` by default: a wall clock that steps (NTP, a suspended
    laptop) would hand out a windfall of tokens or stall every client.
    """

    config: RateLimitConfig = field(default_factory=RateLimitConfig)
    clock: Callable[[], float] = time.monotonic
    clients: dict[str, ClientState] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def _state(self, key: str, now: float) -> ClientState:
        state = self.clients.get(key)
        if state is None:
            if len(self.clients) >= MAX_TRACKED_CLIENTS:
                self._prune()
            state = ClientState(tokens=self.config.capacity, updated=now)
            self.clients[key] = state
        else:
            state.refill(now, self.config.refill_per_second, self.config.capacity)
        return state

    def _prune(self) -> None:
        """Drop clients that are idle and back at full tokens."""
        capacity = self.config.capacity
        for key in [k for k, s in self.clients.items()
                    if s.idle() and s.tokens >= capacity]:
            self.clients.pop(key, None)

    def admit(self, key: str) -> Decision:
        """Take a slot and a token for ``key``, or say which limit refused.

        The concurrency cap is checked FIRST so a refused request does not also
        cost a token: the request never ran, and charging it would punish a
        client twice for one queue being full.
        """
        now = self.clock()
        state = self._state(key, now)
        cap = self.config.max_inflight
        if cap and state.inflight >= cap:
            return Decision(False, "max_inflight", cap, INFLIGHT_RETRY_AFTER, key)
        capacity = self.config.capacity
        if capacity:
            if state.tokens < 1.0:
                rate = self.config.refill_per_second
                wait = (1.0 - state.tokens) / rate if rate > 0 else 60.0
                return Decision(False, "requests_per_minute",
                                self.config.requests_per_minute,
                                max(1, int(math.ceil(wait))), key)
            state.tokens -= 1.0
        state.inflight += 1
        return Decision(True, key=key)

    def release(self, key: str) -> None:
        """Give back an in-flight slot. Safe to call for an unknown key."""
        state = self.clients.get(key)
        if state is None:
            return
        state.inflight = max(0, state.inflight - 1)

    def inflight_total(self) -> int:
        return sum(state.inflight for state in self.clients.values())

    def status_fields(self) -> dict:
        data = self.config.to_dict()
        data["label"] = self.config.label()
        data["clients_tracked"] = len(self.clients)
        data["inflight"] = self.inflight_total()
        return data


# ---------------------------------------------------------------------------
# Request plumbing
# ---------------------------------------------------------------------------

def is_limited_path(path: str) -> bool:
    """True for the paths a limit may refuse. See LIMITED_PREFIXES."""
    return path.startswith(LIMITED_PREFIXES)


def client_key(request) -> str:
    """Who this request counts against: ``key:<id>`` or ``ip:<addr>``.

    The API key id wins when there is one, so a keyed client keeps its own budget
    from anywhere. ``X-Forwarded-For`` is never consulted (see the module
    docstring). A request with no peer address at all (a test transport) counts as
    one anonymous client rather than crashing the middleware.
    """
    getter = getattr(request, "get", None)
    key_id = getter("api_key_id", "") if callable(getter) else ""
    if key_id:
        return f"key:{key_id}"
    remote = ""
    transport = getattr(request, "transport", None)
    if transport is not None:
        try:
            peer = transport.get_extra_info("peername")
        except Exception:
            peer = None
        if peer:
            remote = str(peer[0])
    if not remote:
        remote = str(getattr(request, "remote", "") or "")
    return f"ip:{remote or 'unknown'}"


def too_many_requests(decision: Decision, config: RateLimitConfig) -> web.Response:
    """The 429: a Retry-After header and a body that names the limit."""
    if decision.limit == "max_inflight":
        message = (
            f"too many requests in flight for this client (limit "
            f"{decision.limit_value}). AINode caps concurrency per client so one "
            f"caller cannot occupy every engine. Retry in "
            f"{decision.retry_after}s."
        )
    else:
        message = (
            f"rate limit exceeded: {config.requests_per_minute} requests per "
            f"minute with a burst of {int(config.capacity)}. Retry in "
            f"{decision.retry_after}s."
        )
    return web.json_response(
        {
            "error": {
                "message": message,
                "type": RATE_LIMIT_TYPE,
                "limit": decision.limit,
                "limit_value": decision.limit_value,
                "retry_after": decision.retry_after,
            }
        },
        status=429,
        headers={"Retry-After": str(decision.retry_after)},
    )


@web.middleware
async def rate_limit_middleware(request: web.Request, handler):
    """Refuse a client that is over its limit; otherwise hold a slot for it.

    Registered INSIDE the auth middleware so ``api_key_id`` is already stamped: a
    request with no key never reaches here on a node that requires one, and a
    keyed request is counted against its key rather than its address.
    """
    limiter: Optional[RateLimiter] = request.app.get("rate_limiter")
    if limiter is None or not limiter.enabled or not is_limited_path(request.path):
        return await handler(request)
    key = client_key(request)
    decision = limiter.admit(key)
    if not decision.allowed:
        logger.info("rate limit: %s refused on %s (%s)",
                    key, request.path, decision.limit)
        return too_many_requests(decision, limiter.config)
    try:
        # A streaming answer is written inside the handler (SSE passthrough calls
        # write_eof before returning), so the slot is held for the whole stream
        # and not just until the first byte.
        return await handler(request)
    finally:
        limiter.release(key)


def rate_limit_status_fields(app) -> dict:
    """The limiter's state for ``/api/status``, in the words the doctor prints."""
    limiter: Optional[RateLimiter] = app.get("rate_limiter")
    if limiter is None:
        config = RateLimitConfig()
        return dict(config.to_dict(), label=config.label(),
                    clients_tracked=0, inflight=0)
    return limiter.status_fields()
