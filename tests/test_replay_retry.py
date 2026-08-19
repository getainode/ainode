"""Startup replay retries an engine that dies on the way up.

An engine can pass the launch check (container reached Running) and then die
minutes later during weight load. Observed 2026-08-19 on spark-3: the startup
sweep killed the previous engine, the replacement launched while the driver was
still releasing the GPU, got no device, and exited during load. Nothing retried,
so the node came back advertising nothing.
"""

from unittest import mock

import pytest

from ainode.models import api_routes


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    # The retry deliberately waits 30s for the GPU to release; don't in tests.
    async def _fast(_):
        return None
    monkeypatch.setattr(api_routes.asyncio, "sleep", _fast)


def _ready_sequence(*results):
    """_wait_port_ready stub returning the given results in order."""
    seq = list(results)

    async def _stub(port, timeout=300.0):
        return seq.pop(0) if seq else False
    return _stub


@pytest.mark.asyncio
async def test_no_retry_when_the_engine_binds_first_time(monkeypatch):
    monkeypatch.setattr(api_routes, "_wait_port_ready", _ready_sequence(True))
    relaunch = mock.Mock(return_value=True)
    ok = await api_routes._ensure_serving({}, 8000, relaunch, "x")
    assert ok is True
    relaunch.assert_not_called(), "a healthy engine must never be relaunched"


@pytest.mark.asyncio
async def test_relaunches_once_when_the_engine_never_binds(monkeypatch):
    monkeypatch.setattr(api_routes, "_wait_port_ready", _ready_sequence(False, True))
    relaunch = mock.Mock(return_value=True)
    ok = await api_routes._ensure_serving({}, 8000, relaunch, "x")
    assert ok is True
    assert relaunch.call_count == 1


@pytest.mark.asyncio
async def test_gives_up_after_one_retry(monkeypatch):
    # A model that fails twice has a real problem; looping would hide it.
    monkeypatch.setattr(api_routes, "_wait_port_ready", _ready_sequence(False, False))
    relaunch = mock.Mock(return_value=True)
    ok = await api_routes._ensure_serving({}, 8000, relaunch, "x")
    assert ok is False
    assert relaunch.call_count == 1


@pytest.mark.asyncio
async def test_relaunch_that_fails_to_start_is_reported(monkeypatch):
    monkeypatch.setattr(api_routes, "_wait_port_ready", _ready_sequence(False))
    relaunch = mock.Mock(return_value=False)
    assert await api_routes._ensure_serving({}, 8000, relaunch, "x") is False


@pytest.mark.asyncio
async def test_a_raising_relaunch_does_not_crash_replay(monkeypatch):
    monkeypatch.setattr(api_routes, "_wait_port_ready", _ready_sequence(False))
    relaunch = mock.Mock(side_effect=RuntimeError("docker gone"))
    assert await api_routes._ensure_serving({}, 8000, relaunch, "x") is False


@pytest.mark.asyncio
async def test_retry_waits_before_reasking_for_the_gpu(monkeypatch):
    # The delay is the point: the previous engine's device release is what the
    # first attempt lost to.
    slept = []

    async def _record(sec):
        slept.append(sec)
    monkeypatch.setattr(api_routes.asyncio, "sleep", _record)
    monkeypatch.setattr(api_routes, "_wait_port_ready", _ready_sequence(False, True))
    await api_routes._ensure_serving({}, 8000, mock.Mock(return_value=True), "x")
    assert slept and max(slept) >= 30
