"""NVML is never read on the caller's thread (#238).

Seen on castor (Dell C4130, 4 x V100) on 2026-09-20, between one engine stopping
and the next starting: every route on the node's API, ``/api/health`` included,
timed out. ``wchan`` on the server process was ``nvidia_unlocked_ioctl`` and the
metrics thread was reopening ``/dev/nvidia0..3`` at about 22 ms each, because with
persistence mode off and no process holding the GPUs every sample pays a full GPU
init per device. The load POST eventually went through, after 32 s.

So the read happens on a worker thread and the caller waits a bounded time for
it. Three properties:

* A caller never waits longer than its timeout, whatever the driver is doing.
* While a slow read is in flight, the last COMPLETED sample is served, marked
  ``stale`` with the age it really has. A stale measurement is a measurement; a
  hung event loop is a node that looks dead.
* A stale sample is not stored in the retained series. ``metrics/store.py``'s rule
  is that a figure nobody measured is a NULL row, never the previous tick carried
  forward, and a stale sample IS the previous tick.
"""

from __future__ import annotations

import threading
import time

from ainode.metrics.collector import MetricsCollector
from ainode.metrics.store import series_from_snapshot


def _fixed(**extra):
    """A plain GPU sample, as ``read_gpu_metrics`` would return one."""
    payload = {"gpu_count": 1, "memory_kind": "dedicated", "utilization_percent": 42,
               "memory_used_mb": 8192, "memory_total_mb": 32768,
               "temperature_c": 55, "devices": []}
    payload.update(extra)
    return payload


def test_a_healthy_read_is_returned_as_it_is():
    collector = MetricsCollector()
    collector.read_gpu_metrics = lambda: _fixed()
    sample = collector.get_gpu_metrics()
    assert sample["utilization_percent"] == 42
    assert "stale" not in sample


def test_a_second_call_inside_the_freshness_window_does_not_read_again():
    """The dashboard polls several routes every few seconds; each of them must not
    cost the driver a fresh GPU init."""
    collector = MetricsCollector()
    reads = []

    def _read():
        reads.append(1)
        return _fixed()

    collector.read_gpu_metrics = _read
    collector.get_gpu_metrics()
    collector.get_gpu_metrics()
    collector.get_gpu_metrics()
    assert len(reads) == 1


def test_a_slow_driver_does_not_hold_the_caller(monkeypatch):
    """The regression itself: the read takes seconds, the caller waits its timeout
    and gets on with answering the request."""
    collector = MetricsCollector()
    release = threading.Event()

    def _slow():
        release.wait(30)
        return _fixed()

    collector.read_gpu_metrics = _slow
    began = time.monotonic()
    try:
        sample = collector.get_gpu_metrics(timeout=0.05)
        waited = time.monotonic() - began
        assert waited < 5.0, "a slow NVML read must not hold the caller"
        # Nothing has ever completed, so there is no measurement to serve.
        assert sample["pending"] is True
        assert "persistence" in sample["error"]
    finally:
        release.set()


def test_the_last_good_sample_is_served_while_a_read_is_slow(monkeypatch):
    collector = MetricsCollector()
    monkeypatch.setattr("ainode.metrics.collector.NVML_SAMPLE_MAX_AGE_SECONDS", 0.0)
    release = threading.Event()
    calls = []

    def _read():
        calls.append(1)
        if len(calls) == 1:
            return _fixed()
        release.wait(30)
        return _fixed(utilization_percent=99)

    collector.read_gpu_metrics = _read
    assert collector.get_gpu_metrics(timeout=1.0)["utilization_percent"] == 42
    try:
        served = collector.get_gpu_metrics(timeout=0.05)
        assert served["utilization_percent"] == 42, "the last good sample"
        assert served["stale"] is True
        assert served["sample_age_seconds"] is not None
    finally:
        release.set()


def test_one_read_at_a_time_however_many_callers_ask(monkeypatch):
    """Single flight: ten concurrent requests must not become ten GPU inits."""
    collector = MetricsCollector()
    monkeypatch.setattr("ainode.metrics.collector.NVML_SAMPLE_MAX_AGE_SECONDS", 0.0)
    started = threading.Event()
    release = threading.Event()
    calls = []

    def _read():
        calls.append(1)
        started.set()
        release.wait(30)
        return _fixed()

    collector.read_gpu_metrics = _read
    try:
        threads = [threading.Thread(target=lambda: collector.get_gpu_metrics(timeout=0.05))
                   for _ in range(10)]
        for t in threads:
            t.start()
        started.wait(5)
        for t in threads:
            t.join(5)
        assert len(calls) == 1
    finally:
        release.set()


def test_a_read_that_raises_is_reported_and_not_a_crash():
    collector = MetricsCollector()

    def _boom():
        raise RuntimeError("NVML_ERROR_DRIVER_NOT_LOADED")

    collector.read_gpu_metrics = _boom
    sample = collector.get_gpu_metrics(timeout=1.0)
    assert "NVML_ERROR_DRIVER_NOT_LOADED" in sample["error"]


def test_no_pynvml_is_still_an_error_dict():
    """The Mac and CI case, unchanged: an error dict, not an exception."""
    collector = MetricsCollector()
    sample = collector.get_gpu_metrics(timeout=2.0)
    assert isinstance(sample, dict)


def test_the_snapshot_carries_whatever_the_gpu_read_answered():
    collector = MetricsCollector()
    collector.read_gpu_metrics = lambda: _fixed()
    snapshot = collector.get_snapshot()
    assert snapshot["gpu"]["utilization_percent"] == 42
    assert "requests" in snapshot and "uptime_seconds" in snapshot


def test_the_retention_snapshot_waits_longer_than_a_request_would():
    """The sampler has its own thread and nothing waiting on it, so it waits for a
    real read rather than storing the previous tick again."""
    collector = MetricsCollector()
    seen = {}
    collector.get_snapshot = lambda gpu_timeout=None: seen.setdefault("timeout", gpu_timeout)
    collector.snapshot_for_retention()
    assert seen["timeout"] == MetricsCollector.RETENTION_GPU_TIMEOUT_SECONDS
    assert seen["timeout"] > 5.0


def test_a_stale_sample_is_stored_as_nothing_measured():
    """The retention rule: never the previous tick carried forward."""
    rows = series_from_snapshot({"gpu": _fixed(stale=True, sample_age_seconds=12.0),
                                 "requests": {"total": 0}})
    assert rows["gpu.utilization_percent"] is None
    assert rows["gpu.memory_used_mb"] is None


def test_a_fresh_sample_is_stored_normally():
    rows = series_from_snapshot({"gpu": _fixed(), "requests": {"total": 0}})
    assert rows["gpu.utilization_percent"] == 42.0
    assert rows["gpu.memory_used_mb"] == 8192.0


def test_the_read_thread_is_a_daemon_so_a_wedged_driver_cannot_hold_shutdown():
    collector = MetricsCollector()
    seen = {}
    release = threading.Event()

    def _read():
        seen["thread"] = threading.current_thread()
        release.wait(5)
        return _fixed()

    collector.read_gpu_metrics = _read
    try:
        collector.get_gpu_metrics(timeout=0.05)
        assert seen["thread"].daemon is True
        assert seen["thread"] is not threading.current_thread()
    finally:
        release.set()


def test_completed_reads_are_counted():
    collector = MetricsCollector()
    collector.read_gpu_metrics = lambda: _fixed()
    assert collector.gpu_reads == 0
    collector.get_gpu_metrics(timeout=1.0)
    assert collector.gpu_reads == 1
