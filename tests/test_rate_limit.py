"""Per-client limits on /v1: the keying, the bucket, the cap, the 429, the exemptions.

What is pinned here:

* **Who a client is.** An API key id when the request carries one, else the peer
  address, and ``X-Forwarded-For`` never. A limiter keyed on a caller-supplied
  header is a limiter anybody bypasses.
* **The bucket.** ``burst`` requests go straight through, the next one is a 429
  with a ``Retry-After`` a client can act on, and time refills it. Driven with an
  injected clock: a test that sleeps for a bucket is a test nobody keeps.
* **The concurrency cap**, which is the limit that actually stops a client from
  monopolising the engines: with ``max_inflight`` 1, a second request that arrives
  while the first is still streaming is refused, and the slot comes back when the
  stream ends rather than when the handler was entered.
* **The exemptions.** ``/api/health``, ``/api/status``, the static shell and every
  other ``/api`` path are never refused. The dashboard polls several of them every
  few seconds, so a limit that covered them would lock the operator out of the
  page they fix it from.
* **The wiring**, through the real ``create_app``: a middleware that is not in the
  app's list is code that does nothing.
"""

import asyncio
import socket

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from ainode.api.server import create_app
from ainode.auth.middleware import API_KEY_ID_KEY, AuthConfig
from ainode.core.config import NodeConfig
from ainode.ratelimit.middleware import (
    INFLIGHT_RETRY_AFTER,
    MAX_TRACKED_CLIENTS,
    RATE_LIMIT_TYPE,
    ClientState,
    RateLimitConfig,
    RateLimiter,
    client_key,
    is_limited_path,
    rate_limit_middleware,
    rate_limit_status_fields,
)


class FakeClock:
    """A clock the test moves by hand."""

    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class StubRequest(dict):
    """Enough of a request for ``client_key``: the mapping plus a transport."""

    def __init__(self, path="/v1/chat/completions", peer="10.0.0.9",
                 headers=None, **stamps):
        super().__init__(stamps)
        self.path = path
        self.headers = headers or {}
        self.transport = _Transport(peer)
        self.remote = peer


class _Transport:
    def __init__(self, peer):
        self.peer = peer

    def get_extra_info(self, name):
        return (self.peer, 51234) if name == "peername" and self.peer else None


def _limiter(clock=None, **block):
    config = RateLimitConfig.from_dict(dict({"enabled": True}, **block))
    return RateLimiter(config=config, clock=clock or FakeClock())


# =============================================================================
# The config block
# =============================================================================

def test_the_default_block_is_off():
    limits = RateLimitConfig()
    assert limits.enabled is False
    assert NodeConfig().rate_limit == {}
    assert RateLimitConfig.from_config(NodeConfig()).enabled is False
    assert "one client can occupy every engine" in limits.label()


def test_a_hand_edited_block_falls_back_instead_of_raising():
    """Read on the boot path: a typo must not be why a node has no API at all."""
    base = RateLimitConfig()
    limits = RateLimitConfig.from_dict({
        "enabled": True, "requests_per_minute": "lots",
        "burst": -5, "max_inflight": None,
    })
    assert limits.enabled is True
    assert limits.requests_per_minute == base.requests_per_minute
    assert limits.burst == base.burst
    assert limits.max_inflight == base.max_inflight
    assert RateLimitConfig.from_dict(None).enabled is False
    assert RateLimitConfig.from_dict("nonsense").enabled is False


def test_the_documented_example_block_reads_back_exactly():
    limits = RateLimitConfig.from_dict({
        "enabled": True, "requests_per_minute": 600, "burst": 60,
        "max_inflight": 8,
    })
    assert limits.to_dict() == {"enabled": True, "requests_per_minute": 600,
                                "burst": 60, "max_inflight": 8}
    assert limits.refill_per_second == 10.0
    assert limits.capacity == 60.0
    assert limits.label() == "600/min, burst 60, 8 in flight per client"


# =============================================================================
# Keying
# =============================================================================

def test_a_keyed_request_counts_against_its_key_id():
    request = StubRequest(**{API_KEY_ID_KEY: "3f2a91bb"})
    assert client_key(request) == "key:3f2a91bb"


def test_an_unkeyed_request_counts_against_its_address():
    assert client_key(StubRequest(peer="10.0.0.9")) == "ip:10.0.0.9"


def test_x_forwarded_for_is_ignored():
    """It is caller-supplied: keying on it makes the limiter opt-in for abusers."""
    request = StubRequest(peer="10.0.0.9",
                          headers={"X-Forwarded-For": "1.2.3.4, 5.6.7.8"})
    assert client_key(request) == "ip:10.0.0.9"


def test_two_keys_behind_one_address_are_two_clients():
    a = client_key(StubRequest(peer="10.0.0.9", **{API_KEY_ID_KEY: "aaaa"}))
    b = client_key(StubRequest(peer="10.0.0.9", **{API_KEY_ID_KEY: "bbbb"}))
    assert a != b


def test_a_request_with_no_peer_at_all_still_gets_a_key():
    assert client_key(StubRequest(peer=None)) == "ip:unknown"


def test_the_auth_middleware_stamps_the_key_id_the_limiter_reads(tmp_path,
                                                                monkeypatch):
    monkeypatch.setattr("ainode.auth.middleware.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.auth.middleware.AUTH_FILE", tmp_path / "auth.json")
    auth = AuthConfig()
    entry = auth.generate_key()
    matched, key_id = auth.identify_token(entry["key"])
    assert matched is True
    assert key_id == entry["id"]
    assert auth.identify_token("not a key") == (False, "")
    assert auth.validate_token(entry["key"]) is True
    assert auth.key_id_for_token(entry["key"]) == entry["id"]


# =============================================================================
# The bucket
# =============================================================================

def test_the_burst_goes_through_and_the_next_request_is_refused():
    limiter = _limiter(requests_per_minute=60, burst=3, max_inflight=0)
    for _ in range(3):
        decision = limiter.admit("ip:1.1.1.1")
        assert decision.allowed is True
        limiter.release("ip:1.1.1.1")
    refused = limiter.admit("ip:1.1.1.1")
    assert refused.allowed is False
    assert refused.limit == "requests_per_minute"
    assert refused.limit_value == 60
    assert refused.retry_after >= 1


def test_time_refills_the_bucket():
    clock = FakeClock()
    limiter = _limiter(clock, requests_per_minute=60, burst=2, max_inflight=0)
    for _ in range(2):
        limiter.admit("ip:1.1.1.1")
    assert limiter.admit("ip:1.1.1.1").allowed is False
    clock.advance(1.0)  # 60/min is one token a second
    assert limiter.admit("ip:1.1.1.1").allowed is True


def test_the_bucket_never_fills_past_the_burst():
    clock = FakeClock()
    limiter = _limiter(clock, requests_per_minute=600, burst=2, max_inflight=0)
    clock.advance(3600)
    assert limiter.admit("k").allowed is True
    assert limiter.admit("k").allowed is True
    assert limiter.admit("k").allowed is False


def test_one_client_over_its_limit_does_not_refuse_another():
    limiter = _limiter(requests_per_minute=60, burst=1, max_inflight=0)
    assert limiter.admit("ip:1.1.1.1").allowed is True
    assert limiter.admit("ip:1.1.1.1").allowed is False
    assert limiter.admit("ip:2.2.2.2").allowed is True


def test_retry_after_is_the_wait_for_one_token():
    clock = FakeClock()
    limiter = _limiter(clock, requests_per_minute=60, burst=1, max_inflight=0)
    limiter.admit("k")
    refused = limiter.admit("k")
    # 60/min is one token a second, so a whole second from empty.
    assert refused.retry_after == 1
    limiter = _limiter(FakeClock(), requests_per_minute=6, burst=1, max_inflight=0)
    limiter.admit("k")
    assert limiter.admit("k").retry_after == 10


# =============================================================================
# The concurrency cap
# =============================================================================

def test_the_cap_refuses_a_second_request_while_the_first_is_in_flight():
    limiter = _limiter(max_inflight=1, requests_per_minute=600, burst=60)
    first = limiter.admit("ip:1.1.1.1")
    assert first.allowed is True
    second = limiter.admit("ip:1.1.1.1")
    assert second.allowed is False
    assert second.limit == "max_inflight"
    assert second.limit_value == 1
    assert second.retry_after == INFLIGHT_RETRY_AFTER
    limiter.release("ip:1.1.1.1")
    assert limiter.admit("ip:1.1.1.1").allowed is True


def test_a_refusal_by_the_cap_does_not_also_cost_a_token():
    """The request never ran; charging it would punish one client twice."""
    limiter = _limiter(max_inflight=1, requests_per_minute=600, burst=5)
    limiter.admit("k")
    before = limiter.clients["k"].tokens
    limiter.admit("k")
    assert limiter.clients["k"].tokens == before


def test_releasing_an_unknown_client_is_not_an_error():
    limiter = _limiter(max_inflight=1)
    limiter.release("ip:nobody")  # a 429'd request still runs the finally block
    assert limiter.inflight_total() == 0


def test_zero_max_inflight_means_no_concurrency_cap():
    limiter = _limiter(max_inflight=0, requests_per_minute=600, burst=60)
    for _ in range(20):
        assert limiter.admit("k").allowed is True


def test_idle_clients_are_pruned_so_an_ip_per_request_cannot_grow_forever():
    """An open port can invent an address per request; a full idle bucket is
    identical to no bucket, so it is safe to forget."""
    limiter = _limiter(max_inflight=2, requests_per_minute=600, burst=1)
    limiter.clients = {
        f"ip:seen-{index}": ClientState(tokens=limiter.config.capacity, updated=0.0)
        for index in range(MAX_TRACKED_CLIENTS)
    }
    busy = ClientState(tokens=0.0, updated=0.0, inflight=1)
    limiter.clients["ip:busy"] = busy
    limiter.admit("ip:new")
    assert len(limiter.clients) < MAX_TRACKED_CLIENTS
    # A client with a request in flight is never forgotten: its slot has to be
    # released against the same state.
    assert limiter.clients["ip:busy"] is busy
    assert "ip:new" in limiter.clients


# =============================================================================
# Exemptions
# =============================================================================

@pytest.mark.parametrize("path", [
    "/v1/chat/completions", "/v1/completions", "/v1/models", "/v1/messages",
    "/v1/embeddings", "/v1/decide",
])
def test_the_inference_paths_are_limited(path):
    assert is_limited_path(path) is True


@pytest.mark.parametrize("path", [
    "/", "/onboarding", "/static/js/app.js", "/api/health", "/api/status",
    "/api/nodes", "/api/models", "/api/auth/status", "/favicon.ico",
    "/v1", "/v1models",
])
def test_health_static_and_api_are_never_limited(path):
    assert is_limited_path(path) is False


# =============================================================================
# The middleware, against a real aiohttp app
# =============================================================================

def _mini_app(limiter):
    async def ok(_request):
        return web.json_response({"ok": True})

    async def slow(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await request.app["gate"].wait()
        await response.write(b"data: done\n\n")
        await response.write_eof()
        return response

    app = web.Application(middlewares=[rate_limit_middleware])
    app["rate_limiter"] = limiter
    app["gate"] = asyncio.Event()
    app.router.add_get("/v1/models", ok)
    app.router.add_get("/v1/slow", slow)
    app.router.add_get("/api/status", ok)
    app.router.add_get("/api/health", ok)
    return app


@pytest_asyncio.fixture
async def mini_client():
    limiter = _limiter(requests_per_minute=60, burst=2, max_inflight=0)
    app = _mini_app(limiter)
    async with TestClient(TestServer(app)) as client:
        yield client


@pytest.mark.asyncio
async def test_a_burst_past_the_limit_is_a_429_naming_the_limit(mini_client):
    assert (await mini_client.get("/v1/models")).status == 200
    assert (await mini_client.get("/v1/models")).status == 200
    resp = await mini_client.get("/v1/models")
    assert resp.status == 429
    assert resp.headers["Retry-After"] == "1"
    body = await resp.json()
    assert body["error"]["type"] == RATE_LIMIT_TYPE
    assert body["error"]["limit"] == "requests_per_minute"
    assert body["error"]["limit_value"] == 60
    assert body["error"]["retry_after"] == 1
    assert "60 requests per minute" in body["error"]["message"]


@pytest.mark.asyncio
async def test_the_dashboard_paths_answer_while_v1_is_refused(mini_client):
    for _ in range(5):
        await mini_client.get("/v1/models")
    assert (await mini_client.get("/v1/models")).status == 429
    assert (await mini_client.get("/api/status")).status == 200
    assert (await mini_client.get("/api/health")).status == 200


@pytest.mark.asyncio
async def test_a_disabled_limiter_refuses_nothing(mini_client):
    mini_client.app["rate_limiter"].config.enabled = False
    for _ in range(10):
        assert (await mini_client.get("/v1/models")).status == 200


@pytest.mark.asyncio
async def test_a_stream_holds_its_slot_until_the_stream_ends():
    """The whole point of max_inflight: a long answer occupies an engine."""
    limiter = _limiter(max_inflight=1, requests_per_minute=600, burst=60)
    app = _mini_app(limiter)
    async with TestClient(TestServer(app)) as client:
        streaming = asyncio.create_task(client.get("/v1/slow"))
        await asyncio.sleep(0.05)  # let the handler prepare the response
        assert limiter.inflight_total() == 1
        refused = await client.get("/v1/models")
        assert refused.status == 429
        body = await refused.json()
        assert body["error"]["limit"] == "max_inflight"
        assert body["error"]["limit_value"] == 1
        assert refused.headers["Retry-After"] == str(INFLIGHT_RETRY_AFTER)
        # End the stream: the slot comes back and the next request is served.
        app["gate"].set()
        resp = await streaming
        assert resp.status == 200
        await resp.read()
        assert limiter.inflight_total() == 0
        assert (await client.get("/v1/models")).status == 200


# =============================================================================
# Wiring and /api/status
# =============================================================================

@pytest.fixture
def config():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        free_port = sock.getsockname()[1]
    return NodeConfig(node_id="rl-node", node_name="RLNode", api_port=free_port,
                      cluster_enabled=False)


def test_the_middleware_is_registered_on_the_real_app(config):
    app = create_app(config=config, engine=None)
    assert rate_limit_middleware in app.middlewares
    # Innermost, and after auth: the limiter reads the key id auth stamps.
    names = [getattr(m, "__name__", "") for m in app.middlewares]
    assert names[-1] == "rate_limit_middleware"
    assert names.index("auth_middleware") < names.index("rate_limit_middleware")


def test_the_limiter_is_built_from_the_config_block(config):
    config.rate_limit = {"enabled": True, "requests_per_minute": 120,
                         "burst": 10, "max_inflight": 2}
    app = create_app(config=config, engine=None)
    limiter = app["rate_limiter"]
    assert limiter.enabled is True
    assert limiter.config.requests_per_minute == 120
    assert limiter.config.max_inflight == 2


def test_status_fields_report_the_state():
    limiter = _limiter(requests_per_minute=600, burst=60, max_inflight=8)
    limiter.admit("ip:1.1.1.1")
    fields = rate_limit_status_fields({"rate_limiter": limiter})
    assert fields["enabled"] is True
    assert fields["requests_per_minute"] == 600
    assert fields["burst"] == 60
    assert fields["max_inflight"] == 8
    assert fields["clients_tracked"] == 1
    assert fields["inflight"] == 1
    assert fields["label"] == "600/min, burst 60, 8 in flight per client"


def test_status_fields_on_a_node_with_no_limiter_say_off():
    fields = rate_limit_status_fields({})
    assert fields["enabled"] is False
    assert "one client can occupy every engine" in fields["label"]


@pytest.mark.asyncio
async def test_api_status_carries_both_blocks(config):
    """The dashboard's API access panel reads TLS and the limits from here."""
    config.rate_limit = {"enabled": True, "max_inflight": 4}
    app = create_app(config=config, engine=None)
    async with TestClient(TestServer(app)) as client:
        data = await (await client.get("/api/status")).json()
    assert data["tls"]["enabled"] is False
    assert data["tls"]["port"] == 3443
    assert data["rate_limit"]["enabled"] is True
    assert data["rate_limit"]["max_inflight"] == 4
