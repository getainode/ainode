"""A decision request is answered by the node that owns the model.

The owner holds the model's ``temperatures.json`` and its warm state. A node that
does not serve the model (the master, which is the endpoint clients use) used to
call the owner's ENGINE itself and answered raw with ``temperatures: null``,
while the same request sent to the owner was tempered (2026-09-26,
jebadiah-9b-v2 through Spark-1). The receiving node now hands the whole request
to the owner's AINode under the fleet key.
"""

from types import SimpleNamespace

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from ainode.api import decide
from ainode.auth.fleet import FORWARDED_BY_HEADER
from ainode.ratelimit.middleware import is_forwarded_by_fleet
from tests.test_decide import MODEL, QUESTIONS, TICKET, FakeEngine, _app, _free_port
from tests.test_decide_calibration import TEMPS, _store
from tests.test_systemone import QUESTIONS as JEV_QUESTIONS

SECRET = "fleet-secret-for-tests"


def _decide_body(**over):
    body = {"model": MODEL, "state": TICKET, "questions": QUESTIONS}
    body.update(over)
    return body


def _jev_body(**over):
    body = {"model": MODEL, "state": TICKET, "questions": JEV_QUESTIONS}
    body.update(over)
    return body


@pytest_asyncio.fixture
async def engine():
    fake = FakeEngine()
    server = TestServer(fake.app())
    await server.start_server()
    try:
        yield fake, server.port
    finally:
        await server.close()


def _node(engine_port, models_dir, web_port=None):
    """An AINode whose cluster view says a peer at 127.0.0.1 serves MODEL."""
    app = _app(engine_port)
    app["config"].models_dir = str(models_dir)
    app["config"].cluster_secret = SECRET
    if web_port is not None:
        for member in app["cluster_state"].members():
            member.web_port = web_port
    return app


@pytest_asyncio.fixture
async def owner(engine, tmp_path):
    """The node that holds MODEL's temperatures, as a real AINode app."""
    _, engine_port = engine
    store = tmp_path / "owner-store"
    store.mkdir()
    _store(store)
    server = TestServer(_node(engine_port, store))
    await server.start_server()
    try:
        yield server
    finally:
        await server.close()


async def _entry(engine_port, owner_port, tmp_path):
    """The node the caller reached: no copy of the model, routes to the owner."""
    store = tmp_path / "entry-store"
    store.mkdir(exist_ok=True)
    client = TestClient(TestServer(_node(engine_port, store, web_port=owner_port)))
    await client.start_server()
    return client


@pytest.mark.asyncio
async def test_the_master_answers_with_the_owner_s_calibration(engine, owner, tmp_path):
    fake, engine_port = engine
    client = await _entry(engine_port, owner.port, tmp_path)
    try:
        resp = await client.post("/v1/decide", json=_decide_body())
        assert resp.status == 200
        assert (await resp.json())["calibration"] == {"applied": True,
                                                      "temperatures": TEMPS}

        resp = await client.post("/v1/systemone", json=_jev_body())
        assert resp.status == 200
        assert (await resp.json())["calibration"] == {"applied": True,
                                                      "temperatures": TEMPS}

        resp = await client.post("/v1/decide", json=_decide_body(calibration="raw"))
        assert (await resp.json())["calibration"] == {"applied": False,
                                                      "temperatures": TEMPS}
    finally:
        await client.close()
    assert fake.seen, "the owner reached its engine"


@pytest.mark.asyncio
async def test_the_owner_s_own_refusal_comes_back_as_it_is(engine, owner, tmp_path):
    _, engine_port = engine
    client = await _entry(engine_port, owner.port, tmp_path)
    try:
        resp = await client.post("/v1/systemone", json=_jev_body(calibration=1))
        assert resp.status == 422
        assert "'calibration'" in (await resp.json())["error"]["message"]
    finally:
        await client.close()


class Owner:
    """An owner's web port that records what it was sent and answers as told."""

    def __init__(self, status=200):
        self.status = status
        self.seen: list = []

    def app(self):
        app = web.Application()
        app.router.add_post("/v1/decide", self.handle)
        app.router.add_post("/v1/systemone", self.handle)
        return app

    async def handle(self, request):
        self.seen.append((request.path, dict(request.headers), await request.json()))
        if self.status != 200:
            return web.json_response({"error": {"message": "engine down"}},
                                     status=self.status)
        return web.json_response({"model": MODEL, "node": "owner", "decisions": {},
                                  "calibration": {"applied": True,
                                                  "temperatures": TEMPS}})


@pytest.mark.asyncio
async def test_a_down_owner_fails_over_to_the_next_replica(engine, tmp_path,
                                                           monkeypatch):
    _, engine_port = engine
    broken, healthy = Owner(status=503), Owner()
    servers = [TestServer(broken.app()), TestServer(healthy.app())]
    for s in servers:
        await s.start_server()
    dead_port = _free_port()  # nothing listens: a node that is gone
    monkeypatch.setattr(decide, "owner_web_ports", lambda _app, _c: [
        ("127.0.0.1", dead_port), ("127.0.0.1", servers[0].port),
        ("127.0.0.1", servers[1].port)])
    client = await _entry(engine_port, servers[1].port, tmp_path)
    try:
        resp = await client.post("/v1/decide", json=_decide_body())
        assert resp.status == 200
        data = await resp.json()
        assert data["node"] == "owner"
    finally:
        await client.close()
        for s in servers:
            await s.close()
    assert len(broken.seen) == 1 and len(healthy.seen) == 1
    path, headers, body = healthy.seen[0]
    assert path == "/v1/decide"
    assert headers.get("Authorization", "").startswith("Bearer "), "the fleet key"
    assert headers.get(FORWARDED_BY_HEADER) == "local-node"
    assert body["model"] == MODEL


@pytest.mark.asyncio
async def test_no_owner_answering_falls_back_to_the_engines(engine, tmp_path):
    """A peer on a release without the route, or every owner down: answer raw
    off the engine rather than not at all."""
    fake, engine_port = engine
    old = Owner(status=404)
    server = TestServer(old.app())
    await server.start_server()
    client = await _entry(engine_port, server.port, tmp_path)
    try:
        resp = await client.post("/v1/decide", json=_decide_body())
        assert resp.status == 200
        assert (await resp.json())["calibration"] == {"applied": False,
                                                      "temperatures": None}
    finally:
        await client.close()
        await server.close()
    assert len(old.seen) == 1
    assert fake.seen, "the fallback called the engine directly"


@pytest.mark.asyncio
async def test_a_forwarded_request_is_answered_where_it_lands(engine, tmp_path):
    """Never forwarded twice: the owner answers even if its own view names
    somebody else."""
    fake, engine_port = engine
    never = Owner()
    server = TestServer(never.app())
    await server.start_server()
    client = await _entry(engine_port, server.port, tmp_path)
    try:
        from ainode.auth.fleet import fleet_key
        resp = await client.post("/v1/decide", json=_decide_body(), headers={
            FORWARDED_BY_HEADER: "spark-1",
            "Authorization": f"Bearer {fleet_key(SECRET)}"})
        assert resp.status == 200
    finally:
        await client.close()
        await server.close()
    assert never.seen == []
    assert fake.seen


@pytest.mark.asyncio
async def test_a_model_served_here_takes_the_local_path():
    request = SimpleNamespace(headers={}, app={}, path="/v1/decide")
    assert await decide.forward_to_owner(
        request, {"model": MODEL}, [("localhost", 8002), ("10.0.0.5", 8002)]) is None


def test_owner_ports_come_from_what_each_node_announces():
    members = [SimpleNamespace(fabric_ip="10.100.0.17", web_port=3000),
               SimpleNamespace(fabric_ip="10.100.0.20", web_port=3100)]
    app = {"cluster_state": SimpleNamespace(members=lambda: members)}
    cands = [("localhost", 8000), ("10.100.0.17", 8002), ("10.100.0.17", 8003),
             ("10.100.0.20", 8100), ("10.9.9.9", 8000)]
    assert decide.owner_web_ports(app, cands) == [("10.100.0.17", 3000),
                                                  ("10.100.0.20", 3100)]


def test_only_the_fleet_key_skips_the_owner_s_rate_limit():
    forwarded = {FORWARDED_BY_HEADER: "spark-1"}
    fleet = SimpleNamespace(headers=forwarded, get=lambda k, d="": "fleet")
    client_claiming = SimpleNamespace(headers=forwarded, get=lambda k, d="": "k-123")
    plain_fleet = SimpleNamespace(headers={}, get=lambda k, d="": "fleet")
    assert is_forwarded_by_fleet(fleet)
    assert not is_forwarded_by_fleet(client_claiming)
    assert not is_forwarded_by_fleet(plain_fleet)
