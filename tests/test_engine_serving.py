"""Tests for _engine_serving — the liveness probe behind BUG A FIX 2.

The latched `ready` flag never flips False on crash, so a dead engine reads
READY forever (phantom). _engine_serving probes the engine's own API instead:
truthful when serving, loading (api not up yet), crashed, or erroring.
"""

import asyncio

from ainode.api.server import _engine_serving


class _Backend:
    def __init__(self, health):
        self._health = health

    def health_check(self):
        if isinstance(self._health, Exception):
            raise self._health
        return self._health


def _run(backend):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_engine_serving(backend, loop))
    finally:
        loop.close()


def test_serving_when_api_responds():
    assert _run(_Backend({"api_responding": True})) is True


def test_not_serving_while_loading():
    # process up, api not yet answering — must NOT be advertised
    assert _run(_Backend({"api_responding": False, "process_alive": True})) is False


def test_not_serving_when_crashed():
    assert _run(_Backend({"api_responding": False, "process_alive": False})) is False


def test_not_serving_on_probe_error():
    assert _run(_Backend(RuntimeError("connection refused"))) is False


def test_no_backend():
    assert _run(None) is False


if __name__ == "__main__":
    test_serving_when_api_responds()
    test_not_serving_while_loading()
    test_not_serving_when_crashed()
    test_not_serving_on_probe_error()
    test_no_backend()
    print("ok")
