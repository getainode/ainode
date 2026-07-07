"""Tests for _engine_serving — the liveness probe behind BUG A FIX 2.

The latched `ready` flag never flips False on crash, so a dead engine reads
READY forever (phantom). _engine_serving probes the engine's own API instead:
truthful when serving, loading (api not up yet), crashed, or erroring.
"""

import asyncio

from ainode.api.server import _engine_serving, _live_instance_records


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


# --- _live_instance_records: status must track liveness in BOTH directions ---


class _Record:
    def __init__(self, status):
        self.status = status


class _Instance:
    def __init__(self, health, status):
        self.backend = _Backend(health)
        self.record = _Record(status)


class _Manager:
    def __init__(self, insts):
        self._insts = insts

    def instances(self):
        return self._insts


def _run_live(manager):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_live_instance_records(manager, loop))
    finally:
        loop.close()


def test_live_flips_starting_to_serving():
    inst = _Instance({"api_responding": True}, "starting")
    live = _run_live(_Manager([inst]))
    assert inst.record.status == "serving"
    assert live == [inst.record]


def test_live_resets_stale_serving_latch_to_failed():
    # F3 both-directions: an instance stamped `serving` whose engine no longer
    # answers must NOT keep reading `serving` (that painted a dead stacked
    # instance READY forever in the Server view). It drops out of the live list
    # AND its status latch is cleared.
    inst = _Instance({"api_responding": False, "process_alive": False}, "serving")
    live = _run_live(_Manager([inst]))
    assert inst.record.status == "failed"
    assert live == []


def test_live_leaves_never_served_starting_untouched():
    # A still-loading instance (never reached serving) is not falsely failed —
    # it's simply excluded from the live list until its api answers.
    inst = _Instance({"api_responding": False, "process_alive": True}, "starting")
    live = _run_live(_Manager([inst]))
    assert inst.record.status == "starting"
    assert live == []


if __name__ == "__main__":
    test_serving_when_api_responds()
    test_not_serving_while_loading()
    test_not_serving_when_crashed()
    test_not_serving_on_probe_error()
    test_no_backend()
    test_live_flips_starting_to_serving()
    test_live_resets_stale_serving_latch_to_failed()
    test_live_leaves_never_served_starting_untouched()
    print("ok")
