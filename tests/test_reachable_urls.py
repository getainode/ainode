"""The Server view's reachable-URL list never holds the event loop on DNS.

``_reachable_urls`` resolved this host's name with ``gethostbyname_ex`` on the
event loop. With the hostname missing from /etc/hosts and a nameserver that does
not answer, that took 20 s per call, and every poll of /api/server/status froze
the whole node for that long: on 2026-09-26 the master's heartbeats went stale
and it stopped routing to its peers. The lookup now runs in a thread with a
short wait and a cached answer.
"""

import asyncio
import threading
import time

import pytest

from ainode.api import server_routes


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    monkeypatch.setattr(server_routes, "_host_addrs_cache", None)
    monkeypatch.setattr(server_routes, "_host_addrs_pending", None)
    monkeypatch.setattr(server_routes, "HOST_ADDRS_WAIT_SECONDS", 0.2)
    yield


@pytest.mark.asyncio
async def test_a_hanging_resolver_does_not_hold_the_event_loop(monkeypatch):
    release = threading.Event()
    calls = []

    def slow_lookup():
        calls.append(1)
        release.wait(5)
        return ["10.0.0.5"]

    monkeypatch.setattr(server_routes, "_lookup_host_addrs", slow_lookup)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    tick_task = asyncio.create_task(ticker())
    started = time.monotonic()
    urls = await server_routes._reachable_urls("0.0.0.0", 3000)
    waited = time.monotonic() - started
    tick_task.cancel()

    assert urls == ["http://localhost:3000"], "answers without the addresses"
    assert waited < 1.0
    assert ticks >= 5, "the loop kept running while the resolver hung"

    # A second request while the first lookup is still stuck starts no new one.
    await server_routes._reachable_urls("0.0.0.0", 3000)
    assert len(calls) == 1

    # When the slow lookup finishes, its answer is kept for the next request.
    release.set()
    for _ in range(50):
        if server_routes._host_addrs_cache is not None:
            break
        await asyncio.sleep(0.02)
    assert await server_routes._reachable_urls("0.0.0.0", 3000) == [
        "http://localhost:3000", "http://10.0.0.5:3000"]


@pytest.mark.asyncio
async def test_the_answer_is_cached_until_it_expires(monkeypatch):
    calls = []

    def lookup():
        calls.append(1)
        return ["192.168.0.10", "100.64.0.1"]

    monkeypatch.setattr(server_routes, "_lookup_host_addrs", lookup)
    first = await server_routes._reachable_urls("10.9.9.9", 3000)
    second = await server_routes._reachable_urls("10.9.9.9", 3000)
    assert first == second == ["http://localhost:3000", "http://192.168.0.10:3000",
                               "http://100.64.0.1:3000", "http://10.9.9.9:3000"]
    assert len(calls) == 1

    expires, addrs = server_routes._host_addrs_cache
    monkeypatch.setattr(server_routes, "_host_addrs_cache", (time.monotonic() - 1, addrs))
    await server_routes._reachable_urls("0.0.0.0", 3000)
    assert len(calls) == 2, "an expired answer is looked up again"


@pytest.mark.asyncio
async def test_a_failing_lookup_still_answers(monkeypatch):
    monkeypatch.setattr(server_routes.socket, "gethostbyname_ex",
                        lambda _name: (_ for _ in ()).throw(OSError("no resolver")))
    assert await server_routes._reachable_urls("0.0.0.0", 3000) == ["http://localhost:3000"]
    assert server_routes._host_addrs_cache[1] == []
