"""Warming a decision engine's answer grammar on bind (#277).

The first grammar-constrained request per question shape costs a cold compile,
so a decision model (a store directory with ``prompt_contract.json`` or
``temperatures.json``) is sent one minimal question per kind as soon as its
engine binds, and ``/api/status`` reports ``warm`` per instance. A fake vLLM
records what the warm-up actually sent.
"""

import asyncio
import json
import socket
from types import SimpleNamespace

import aiohttp
import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from ainode.api import decide
from ainode.api.decide import (
    ask_one,
    schedule_decision_warmup,
    warm_decision_engine,
)
from ainode.api.server import create_app
from ainode.core.config import NodeConfig
from ainode.discovery.instance import InstanceRecord
from ainode.engine.instance_manager import InstanceManager
from ainode.models import api_routes
from tests.test_decide import MODEL, FakeEngine


@pytest_asyncio.fixture
async def fake():
    engine = FakeEngine()
    server = TestServer(engine.app())
    await server.start_server()
    try:
        yield engine, server.port
    finally:
        await server.close()


def _decision_store(models_dir, marker="temperatures.json"):
    directory = models_dir / MODEL.replace("/", "--")
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    (directory / marker).write_text(json.dumps({"temperatures": {"choice": 1.2}}))
    return directory


def _backend(models_dir, served=None):
    return SimpleNamespace(config=SimpleNamespace(model=MODEL, models_dir=str(models_dir),
                                                  served_model_name=served))


async def _drain():
    await asyncio.gather(*list(decide._WARM_TASKS))


@pytest.mark.asyncio
async def test_the_warm_up_sends_one_constrained_request_per_kind(fake):
    engine, port = fake
    status: dict = {}
    async with aiohttp.ClientSession() as session:
        assert await warm_decision_engine(session, port, MODEL, status)
    assert len(engine.seen) == 3
    for body in engine.seen:
        # The same body the routes send: constrained, logprobs on, thinking off.
        assert body["model"] == MODEL
        assert body["structured_outputs"] == {"choice": ["A", "B"]}
        assert body["logprobs"] is True
        assert body["chat_template_kwargs"]["enable_thinking"] is False
    noul = engine.seen[1]["messages"][-1]["content"]
    assert "A. true" in noul and "B. false" in noul
    assert status["warm"] is True and status["warming"] is False
    assert list(status["compile_seconds"]) == ["choice", "noul", "score"]
    assert status["error"] is None


@pytest.mark.asyncio
async def test_a_warm_up_that_fails_says_where_and_stays_cold():
    engine = FakeEngine(status=500)
    server = TestServer(engine.app())
    await server.start_server()
    try:
        status: dict = {}
        async with aiohttp.ClientSession() as session:
            assert not await warm_decision_engine(session, server.port, MODEL, status)
    finally:
        await server.close()
    assert status["warm"] is False
    assert status["error"].startswith("choice:")
    assert len(engine.seen) == 1, "the first failure stops the warm-up"


@pytest.mark.asyncio
async def test_a_bind_warms_a_decision_model_and_records_the_flag(fake, tmp_path,
                                                                  monkeypatch):
    engine, port = fake
    _decision_store(tmp_path, marker="prompt_contract.json")

    async def bound(*_a, **_k):
        return True, "bound", 1.0

    async def no_ledger(*_a, **_k):
        return None

    monkeypatch.setattr(api_routes, "_bind_wait", bound)
    monkeypatch.setattr(api_routes, "record_launch_time", no_ledger)
    app = {"decision_warm": {}, "client_session": None}
    ok, _, _ = await api_routes._wait_for_bind(app, port, _backend(tmp_path))
    assert ok
    assert app["decision_warm"][port]["warm"] is False, "warming, not yet warm"
    await _drain()
    assert app["decision_warm"][port]["warm"] is True
    assert len(engine.seen) == 3


@pytest.mark.asyncio
async def test_a_model_that_is_not_a_decision_model_is_left_alone(fake, tmp_path):
    engine, port = fake
    (tmp_path / MODEL.replace("/", "--")).mkdir()
    (tmp_path / MODEL.replace("/", "--") / "config.json").write_text("{}")
    app = {"decision_warm": {port: {"model": "old", "warm": True}}}
    assert not schedule_decision_warmup(app, port, _backend(tmp_path))
    assert port not in app["decision_warm"], "a stale entry on the port is cleared"
    assert engine.seen == []


@pytest.mark.asyncio
async def test_the_warm_up_asks_by_the_served_model_name(fake, tmp_path):
    engine, port = fake
    _decision_store(tmp_path)
    app = {"decision_warm": {}, "client_session": None}
    assert schedule_decision_warmup(app, port, _backend(tmp_path, served=["judge"]))
    await _drain()
    assert {body["model"] for body in engine.seen} == {"judge"}


@pytest.mark.asyncio
async def test_a_route_timeout_names_the_grammar_compile_not_an_unreachable_node():
    class Slow(FakeEngine):
        async def completions(self, request):
            await asyncio.sleep(1.0)
            return await super().completions(request)

    server = TestServer(Slow().app())
    await server.start_server()
    try:
        async with aiohttp.ClientSession() as session:
            payload, _, err = await ask_one(session, [("127.0.0.1", server.port)],
                                            {"model": MODEL}, timeout_s=0.1)
    finally:
        await server.close()
    assert payload is None
    assert "compiling the answer grammar, retry" in err
    assert "unreachable" not in err


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.asyncio
async def test_status_reports_warm_per_instance(tmp_path):
    config = NodeConfig(node_id="n1", node_name="TestNode", model=None,
                        api_port=_free_port(), web_port=_free_port(),
                        models_dir=str(tmp_path / "models"))
    app = create_app(config=config, engine=None)
    manager = InstanceManager(base_port=config.api_port)
    manager.add(InstanceRecord(instance_id="n1:judge", model=MODEL, api_port=8001,
                               status="serving"), None)
    manager.add(InstanceRecord(instance_id="n1:chat", model="org/chat", api_port=8002,
                               status="serving"), None)
    app["instances"] = manager
    app["decision_warm"][8001] = {"model": MODEL, "warm": False,
                                  "compile_seconds": {"choice": 71.2}}
    async with TestClient(TestServer(app)) as client:
        rows = {row["api_port"]: row
                for row in (await (await client.get("/api/status")).json())["instances"]}
        assert rows[8001]["warm"] is False
        assert rows[8001]["warm_compile_seconds"] == {"choice": 71.2}
        assert rows[8002]["warm"] is None, "nothing to warm on a chat model"
        app["decision_warm"][8001]["warm"] = True
        rows = {row["api_port"]: row
                for row in (await (await client.get("/api/status")).json())["instances"]}
        assert rows[8001]["warm"] is True


@pytest.mark.asyncio
async def test_an_adopted_stacked_decision_model_is_warmed(fake, tmp_path):
    """A stacked engine kept across a restart never binds, so the hook there never
    saw it and /api/status reported warm: null (Spark-4, 2026-09-26)."""
    engine, port = fake
    _decision_store(tmp_path)
    manager = InstanceManager(base_port=8000)
    record = InstanceRecord(instance_id="n1:judge", model=MODEL, api_port=port,
                            status="serving", adopted=True)
    manager.add(record, _backend(tmp_path))
    app = {"decision_warm": {}, "client_session": None, "instances": manager}
    api_routes._warm_adopted_stacked(app, NodeConfig(api_port=8000), [record])
    assert app["decision_warm"][port]["warming"] is True
    await _drain()
    assert app["decision_warm"][port]["warm"] is True
    assert len(engine.seen) == 3


@pytest.mark.asyncio
async def test_adoption_does_not_warm_the_primary_port_or_a_starting_engine(fake, tmp_path):
    engine, port = fake
    _decision_store(tmp_path)
    manager = InstanceManager(base_port=port)
    primary = InstanceRecord(instance_id="n1:primary", model=MODEL, api_port=port,
                             status="serving", adopted=True)
    starting = InstanceRecord(instance_id="n1:late", model=MODEL, api_port=port + 1,
                              status="starting", adopted=True)
    manager.add(primary, _backend(tmp_path))
    manager.add(starting, _backend(tmp_path))
    app = {"decision_warm": {}, "client_session": None, "instances": manager}
    # The primary is warmed by _await_primary_bind; warming it here would do it twice.
    api_routes._warm_adopted_stacked(app, NodeConfig(api_port=port), [primary, starting])
    assert app["decision_warm"] == {}
    assert engine.seen == []
