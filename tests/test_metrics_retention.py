"""Tests for on-disk metrics retention: the store, the history route, the sampler.

The one rule every test here exists to hold: a figure nobody measured is stored
as NULL, read back as ``None``, served as ``null``, and never turned into a
number by anything in between. A zero in this path is a claim that the GPU was
idle, that the node was empty or that requests answered instantly, and the whole
telemetry surface has already been burned by that once (root AGENTS.md, #174,
#175, #176).

The second rule: none of this may take a node down. A read-only disk, a full
disk, a store that raises on every call, all of them cost samples and nothing
else.
"""

from __future__ import annotations

import sqlite3
import time
from types import SimpleNamespace

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from ainode.metrics import prometheus
from ainode.metrics.api_routes import register_metrics_routes
from ainode.metrics.collector import MetricsCollector
from ainode.metrics.store import (
    BUCKET_SECONDS,
    MetricsSampler,
    MetricsSettings,
    MetricsStore,
    default_store_path,
    series_from_snapshot,
)


def _minute(value: float) -> float:
    """Floor a timestamp to a minute boundary, so buckets are predictable."""
    return float(int(value // BUCKET_SECONDS) * BUCKET_SECONDS)


# A recent minute boundary. Recent on purpose: the history route picks raw or
# rolled-up data by how far back the window reaches, so a base from 2023 would
# be served from the roll-up table in every test that did not say otherwise.
BASE = _minute(time.time()) - 900

# The same instant floored to two minutes, for the tests that read a two minute
# step. The grid is aligned to the step, so a base that is not a multiple of it
# would split one bucket across two slots.
BASE_2M = float(int(BASE // 120) * 120)


@pytest.fixture
def store(tmp_path):
    s = MetricsStore(tmp_path / "metrics.db")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class TestSettings:
    def test_defaults_are_on_with_48h_and_30d(self):
        s = MetricsSettings()
        assert s.enabled is True
        assert s.retention_hours == 48
        assert s.retention_days == 30
        assert s.interval_seconds == 15.0

    def test_read_from_a_config_block(self):
        config = SimpleNamespace(metrics={
            "enabled": True, "retention_hours": 6,
            "retention_days": 2, "interval_seconds": 30,
        })
        s = MetricsSettings.from_config(config)
        assert (s.retention_hours, s.retention_days, s.interval_seconds) == (6, 2, 30.0)

    def test_missing_block_is_the_default(self):
        assert MetricsSettings.from_config(SimpleNamespace()) == MetricsSettings()
        assert MetricsSettings.from_config(SimpleNamespace(metrics=None)) \
            == MetricsSettings()

    def test_disabled_is_honoured(self):
        s = MetricsSettings.from_config(SimpleNamespace(metrics={"enabled": False}))
        assert s.enabled is False

    def test_nonsense_is_clamped_not_obeyed(self):
        # A hand-edited config. Zero retention would delete everything on the
        # next pass and a zero interval would be a busy loop.
        s = MetricsSettings.from_mapping({
            "retention_hours": 0, "retention_days": -5,
            "interval_seconds": 0, "enabled": True,
        })
        assert s.retention_hours == 1
        assert s.retention_days == 1
        assert s.interval_seconds >= 1.0

    def test_unparseable_values_keep_the_default(self):
        s = MetricsSettings.from_mapping({"retention_hours": "many", "retention_days": None})
        assert s.retention_hours == 48
        assert s.retention_days == 30

    def test_disabled_store_opens_nothing(self, tmp_path):
        path = tmp_path / "metrics.db"
        s = MetricsStore(path, MetricsSettings(enabled=False))
        assert s.available is False
        assert s.write({"a": 1.0}) == 0
        assert not path.exists()
        assert s.stats()["enabled"] is False


# ---------------------------------------------------------------------------
# What gets stored
# ---------------------------------------------------------------------------

class TestSeriesFromSnapshot:
    def test_flattens_a_good_snapshot(self):
        series = series_from_snapshot({
            "uptime_seconds": 120.5,
            "gpu": {
                "gpu_count": 4, "utilization_percent": 61,
                "memory_used_mb": 8192, "memory_total_mb": 131072,
                "temperature_c": 55, "engine_reserved_fraction": None,
                "system_memory_used_mb": 4096,
            },
            "requests": {
                "total": 3, "errors": 1, "tokens_generated": 90,
                "tokens_per_second": 12.5,
                "latency_ms": {"p50": 100.0, "p95": 200.0, "p99": 300.0},
            },
        })
        assert series["gpu.utilization_percent"] == 61.0
        assert series["gpu.memory_total_mb"] == 131072.0
        assert series["requests.total"] == 3.0
        assert series["requests.latency_ms.p95"] == 200.0
        assert series["uptime_seconds"] == 120.5
        # A key the node itself reported as unknown stays unknown.
        assert series["gpu.engine_reserved_fraction"] is None

    def test_a_gpu_error_makes_every_gpu_series_none(self):
        series = series_from_snapshot({
            "gpu": {"error": "pynvml not available"},
            "requests": {"total": 0, "latency_ms": {"p50": 0, "p95": 0, "p99": 0}},
        })
        for name, value in series.items():
            if name.startswith("gpu."):
                assert value is None, name

    def test_latency_is_none_until_something_has_been_timed(self):
        # /api/metrics returns 0 for the percentiles on a collector that has
        # timed nothing, and that shape does not change. Stored, that 0 would
        # claim requests are answering instantly.
        series = series_from_snapshot(MetricsCollector().get_snapshot())
        assert series["requests.latency_ms.p50"] is None
        assert series["requests.latency_ms.p99"] is None
        assert series["requests.total"] == 0.0

    def test_latency_appears_once_a_request_is_recorded(self):
        collector = MetricsCollector()
        collector.record_request("m", 42.0, tokens_generated=7)
        series = series_from_snapshot(collector.get_snapshot())
        assert series["requests.latency_ms.p50"] == 42.0
        assert series["requests.tokens_generated"] == 7.0


# ---------------------------------------------------------------------------
# Writing and reading
# ---------------------------------------------------------------------------

class TestWriteRead:
    def test_round_trip(self, store):
        store.write({"gpu.temperature_c": 41.0}, ts=BASE + 5)
        store.write({"gpu.temperature_c": 43.0}, ts=BASE + 20)
        out = store.history(["gpu.temperature_c"], since=BASE, until=BASE + 30,
                            step=15, resolution="raw")
        values = [p["value"] for p in out["series"]["gpu.temperature_c"]]
        assert values[0] == 41.0
        assert values[1] == 43.0
        assert out["resolution"] == "raw"

    def test_wal_mode_is_on(self, store):
        with sqlite3.connect(str(store.path)) as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert str(mode).lower() == "wal"

    def test_a_null_sample_stays_null(self, store):
        store.write({"gpu.utilization_percent": None}, ts=BASE + 5)
        out = store.history(["gpu.utilization_percent"], since=BASE, until=BASE + 15,
                            step=15, resolution="raw")
        assert out["series"]["gpu.utilization_percent"][0]["value"] is None

    def test_null_is_stored_as_sql_null_not_zero(self, store):
        store.write({"s": None}, ts=BASE)
        with sqlite3.connect(str(store.path)) as conn:
            rows = conn.execute("SELECT value FROM samples").fetchall()
        assert rows == [(None,)]

    def test_a_gap_reads_back_as_null_and_is_never_interpolated(self, store):
        # 41 at t+0, nothing at all for two steps, 45 at t+45. The middle two
        # slots must be null: filling them in would draw a line through minutes
        # the node did not measure.
        store.write({"gpu.temperature_c": 41.0}, ts=BASE)
        store.write({"gpu.temperature_c": 45.0}, ts=BASE + 45)
        out = store.history(["gpu.temperature_c"], since=BASE, until=BASE + 45,
                            step=15, resolution="raw")
        values = [p["value"] for p in out["series"]["gpu.temperature_c"]]
        assert values == [41.0, None, None, 45.0]

    def test_every_series_shares_one_grid(self, store):
        store.write({"a": 1.0, "b": None}, ts=BASE)
        out = store.history(["a", "b"], since=BASE, until=BASE + 60, step=15,
                            resolution="raw")
        a, b = out["series"]["a"], out["series"]["b"]
        assert len(a) == len(b) == out["points"]
        assert [p["ts"] for p in a] == [p["ts"] for p in b]

    def test_an_unknown_series_is_all_null_not_an_error(self, store):
        store.write({"a": 1.0}, ts=BASE)
        out = store.history(["nope"], since=BASE, until=BASE + 30, step=15,
                            resolution="raw")
        assert all(p["value"] is None for p in out["series"]["nope"])

    def test_series_names_lists_what_was_written(self, store):
        store.write({"b": 1.0, "a": None}, ts=BASE)
        assert store.series_names() == ["a", "b"]

    def test_history_survives_a_close_and_reopen(self, tmp_path):
        path = tmp_path / "metrics.db"
        first = MetricsStore(path)
        first.write({"gpu.temperature_c": 44.0}, ts=BASE)
        first.close()

        second = MetricsStore(path)
        try:
            out = second.history(["gpu.temperature_c"], since=BASE, until=BASE + 15,
                                 step=15, resolution="raw")
            assert out["series"]["gpu.temperature_c"][0]["value"] == 44.0
        finally:
            second.close()


# ---------------------------------------------------------------------------
# Roll-up
# ---------------------------------------------------------------------------

class TestDownsample:
    def test_one_minute_mean_min_max(self, store):
        for offset, value in ((0, 10.0), (15, 20.0), (30, 30.0), (45, 40.0)):
            store.write({"g": value}, ts=BASE + offset)
        assert store.downsample(now=BASE + 120) > 0

        out = store.history(["g"], since=BASE, until=BASE + 60, step=60,
                            resolution="1m")
        point = out["series"]["g"][0]
        assert point["value"] == pytest.approx(25.0)
        assert point["min"] == 10.0
        assert point["max"] == 40.0
        assert point["samples"] == 4

    def test_a_minute_nobody_measured_rolls_up_as_null(self, store):
        store.write({"g": None}, ts=BASE + 5)
        store.write({"g": None}, ts=BASE + 20)
        store.downsample(now=BASE + 120)
        with sqlite3.connect(str(store.path)) as conn:
            row = conn.execute(
                "SELECT mean_value, min_value, max_value, measured, missing "
                "FROM samples_1m WHERE series = 'g'"
            ).fetchone()
        assert row[:3] == (None, None, None)
        assert row[3] == 0      # nothing measured
        assert row[4] == 2      # and it recorded that two samples said nothing

    def test_a_half_measured_minute_averages_only_what_was_measured(self, store):
        store.write({"g": 10.0}, ts=BASE)
        store.write({"g": None}, ts=BASE + 15)
        store.write({"g": 20.0}, ts=BASE + 30)
        store.downsample(now=BASE + 120)
        out = store.history(["g"], since=BASE, until=BASE + 60, step=60,
                            resolution="1m")
        point = out["series"]["g"][0]
        assert point["value"] == pytest.approx(15.0)
        assert point["samples"] == 2

    def test_the_minute_in_progress_is_left_alone(self, store):
        # Rolling up a partial minute would publish its first sample as the whole
        # minute, and the next pass would have to correct it.
        store.write({"g": 10.0}, ts=BASE + 5)
        assert store.downsample(now=BASE + 30) == 0
        with sqlite3.connect(str(store.path)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM samples_1m").fetchone()[0] == 0

    def test_a_second_pass_only_does_the_new_minutes(self, store):
        store.write({"g": 1.0}, ts=BASE + 5)
        assert store.downsample(now=BASE + 120) == 1
        # Nothing new happened, so nothing to do.
        assert store.downsample(now=BASE + 120) == 0
        store.write({"g": 2.0}, ts=BASE + 125)
        assert store.downsample(now=BASE + 240) == 1

    def test_the_watermark_survives_a_reopen(self, tmp_path):
        path = tmp_path / "metrics.db"
        first = MetricsStore(path)
        first.write({"g": 1.0}, ts=BASE + 5)
        assert first.downsample(now=BASE + 120) == 1
        first.close()

        second = MetricsStore(path)
        try:
            # The same minute is not rolled up again after a restart.
            assert second.downsample(now=BASE + 120) == 0
        finally:
            second.close()

    def test_a_weighted_mean_across_several_minutes(self, store):
        # Minute one holds a single 100; minute two holds three 10s. A step of
        # two minutes must weigh them 1:3, not 1:1.
        store.write({"g": 100.0}, ts=BASE_2M + 5)
        for offset in (65, 80, 95):
            store.write({"g": 10.0}, ts=BASE_2M + offset)
        store.downsample(now=BASE_2M + 180)
        out = store.history(["g"], since=BASE_2M, until=BASE_2M + 119, step=120,
                            resolution="1m")
        assert out["series"]["g"][0]["value"] == pytest.approx(130.0 / 4)
        assert out["series"]["g"][0]["samples"] == 4


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

class TestRetention:
    def test_raw_samples_past_the_window_go(self, tmp_path):
        store = MetricsStore(tmp_path / "m.db",
                             MetricsSettings(retention_hours=2, retention_days=30))
        try:
            now = BASE + 100_000
            store.write({"g": 1.0}, ts=now - 5 * 3600)   # 5 hours old
            store.write({"g": 2.0}, ts=now - 600)        # 10 minutes old
            deleted = store.prune(now=now)
            assert deleted["samples"] == 1
            with sqlite3.connect(str(store.path)) as conn:
                assert conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 1
        finally:
            store.close()

    def test_rollups_past_the_window_go(self, tmp_path):
        store = MetricsStore(tmp_path / "m.db",
                             MetricsSettings(retention_hours=48, retention_days=1))
        try:
            now = BASE + 400_000
            store.write({"g": 1.0}, ts=now - 3 * 86400)
            store.write({"g": 2.0}, ts=now - 600)
            store.downsample(now=now)
            deleted = store.prune(now=now)
            assert deleted["downsampled"] == 1
            with sqlite3.connect(str(store.path)) as conn:
                assert conn.execute(
                    "SELECT COUNT(*) FROM samples_1m").fetchone()[0] == 1
        finally:
            store.close()

    def test_maintain_rolls_up_before_it_prunes(self, tmp_path):
        # Order matters: pruning first would drop raw rows the roll-up had not
        # read, leaving a hole at the front of the 30 day series.
        store = MetricsStore(tmp_path / "m.db",
                             MetricsSettings(retention_hours=1, retention_days=30))
        try:
            now = BASE + 100_000
            store.write({"g": 7.0}, ts=now - 3 * 3600)   # past raw retention
            store.maintain(now=now)
            with sqlite3.connect(str(store.path)) as conn:
                assert conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 0
                rolled = conn.execute(
                    "SELECT mean_value FROM samples_1m").fetchone()
            assert rolled[0] == 7.0
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Choosing a resolution
# ---------------------------------------------------------------------------

class TestResolution:
    def test_a_short_window_is_served_raw(self, store):
        _, _, _, resolution = store.resolve_window(
            since=BASE, until=BASE + 600, now=BASE + 600)
        assert resolution == "raw"

    def test_a_window_past_the_raw_retention_is_served_rolled_up(self, store):
        now = BASE + 10 * 86400
        _, _, _, resolution = store.resolve_window(
            since=now - 7 * 86400, until=now, now=now)
        assert resolution == "1m"

    def test_a_coarse_step_is_served_rolled_up(self, store):
        _, _, step, resolution = store.resolve_window(
            since=BASE, until=BASE + 3600, step=300, now=BASE + 3600)
        assert resolution == "1m"
        assert step == 300

    def test_a_rolled_up_step_never_goes_below_a_minute(self, store):
        _, _, step, resolution = store.resolve_window(
            since=BASE, until=BASE + 3600, step=5, resolution="1m", now=BASE + 3600)
        assert step == float(BUCKET_SECONDS)

    def test_the_grid_is_aligned_to_the_step(self, store):
        start, _, step, _ = store.resolve_window(
            since=BASE + 7, until=BASE + 600, step=15, now=BASE + 600)
        assert start % step == 0

    def test_a_huge_window_gets_a_coarser_step_not_a_truncated_one(self, store):
        since, until = BASE, BASE + 30 * 86400
        start, end, step, _ = store.resolve_window(
            since=since, until=until, step=1, now=until)
        assert end == until                      # the window asked for is kept
        assert (end - start) / step <= 5001      # and the payload stays drawable

    def test_the_budget_is_split_across_the_series_asked_for(self, store):
        store.write({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}, ts=BASE)
        out = store.history(None, since=BASE, until=BASE + 86400, step=1,
                            resolution="raw", max_points=400)
        assert len(out["series"]) == 4
        assert out["points"] <= 101


# ---------------------------------------------------------------------------
# Self-reporting
# ---------------------------------------------------------------------------

class TestStats:
    def test_an_empty_store_reports_no_oldest_sample(self, store):
        stats = store.stats()
        assert stats["samples"] == 0
        # A timestamp of 0 would read as January 1970 on every chart.
        assert stats["oldest_sample"] is None
        assert stats["newest_sample"] is None

    def test_counts_and_bounds(self, store):
        store.write({"a": 1.0, "b": None}, ts=BASE)
        store.write({"a": 2.0, "b": None}, ts=BASE + 60)
        store.downsample(now=BASE + 180)
        stats = store.stats()
        assert stats["samples"] == 4
        assert stats["downsampled"] == 4
        assert stats["oldest_sample"] == BASE
        assert stats["newest_sample"] == BASE + 60
        assert stats["db_bytes"] > 0
        assert stats["degraded"] is False


# ---------------------------------------------------------------------------
# The failure path: a broken store never takes the node down
# ---------------------------------------------------------------------------

class _ExplodingStore:
    """A store whose every call raises, as a full disk eventually does."""

    settings = MetricsSettings()

    def write(self, samples, ts=None):
        raise OSError("disk full")

    def maintain(self, now=None):
        raise OSError("disk full")

    def stats(self):
        raise OSError("disk full")


class TestFailurePath:
    def test_a_store_that_cannot_be_opened_does_not_raise(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        store = MetricsStore(blocker / "metrics.db")
        assert store.available is False
        assert store.degraded is True
        assert store.write({"a": 1.0}) == 0
        assert store.history(["a"], since=BASE, until=BASE + 60,
                             step=15)["series"]["a"]
        assert store.downsample() == 0
        assert store.prune() == {"samples": 0, "downsampled": 0}
        assert store.series_names() == []
        store.close()

    def test_a_write_to_a_closed_store_returns_zero(self, store):
        store.close()
        assert store.write({"a": 1.0}) == 0

    def test_one_write_failure_is_logged_once(self, tmp_path, caplog):
        store = MetricsStore(tmp_path / "m.db")
        store._conn.close()          # the handle is now dead under the store
        with caplog.at_level("WARNING"):
            assert store.write({"a": 1.0}) == 0
            assert store.write({"a": 2.0}) == 0
            assert store.write({"a": 3.0}) == 0
        warnings = [r for r in caplog.records if "metrics retention" in r.message]
        assert len(warnings) == 1
        assert store.degraded is True

    def test_a_sampler_over_a_broken_store_keeps_going(self):
        collector = MetricsCollector()
        sampler = MetricsSampler(_ExplodingStore(), collector.get_snapshot)
        # Three ticks, no exception, and the collector is untouched.
        assert sampler.tick() == 0
        assert sampler.tick() == 0
        sampler.maintain()
        assert collector.get_snapshot()["requests"]["total"] == 0

    def test_a_snapshot_provider_that_raises_costs_one_tick(self, store):
        def boom():
            raise RuntimeError("nvml exploded")

        sampler = MetricsSampler(store, boom)
        assert sampler.tick() == 0
        assert store.stats()["samples"] == 0

    def test_the_collector_survives_attaching_a_broken_store(self):
        collector = MetricsCollector()
        collector.attach_store(_ExplodingStore(), interval_seconds=1)
        try:
            assert collector.store_sampler is not None
            collector.record_request("m", 10.0, tokens_generated=1)
            assert collector.get_request_stats()["total"] == 1
        finally:
            collector.detach_store()
        assert collector.store_sampler is None


# ---------------------------------------------------------------------------
# The collector writing through
# ---------------------------------------------------------------------------

class TestCollectorWritesThrough:
    def test_a_tick_lands_every_series(self, store):
        collector = MetricsCollector()
        collector.record_request("m", 25.0, tokens_generated=10)
        sampler = MetricsSampler(store, collector.get_snapshot)
        written = sampler.tick(ts=BASE)
        assert written == len(series_from_snapshot(collector.get_snapshot()))
        out = store.history(["requests.total", "requests.latency_ms.p50"],
                            since=BASE, until=BASE + 15, step=15, resolution="raw")
        assert out["series"]["requests.total"][0]["value"] == 1.0
        assert out["series"]["requests.latency_ms.p50"][0]["value"] == 25.0

    def test_attach_starts_a_thread_that_writes(self, store):
        collector = MetricsCollector()
        collector.attach_store(store, interval_seconds=1)
        try:
            deadline = time.time() + 5
            while time.time() < deadline and store.stats()["samples"] == 0:
                time.sleep(0.05)
            assert store.stats()["samples"] > 0
            assert collector.store_sampler.running is True
        finally:
            collector.detach_store()
        assert collector.store_sampler is None

    def test_attach_twice_leaves_one_sampler(self, store):
        collector = MetricsCollector()
        first = collector.attach_store(store, interval_seconds=30)
        second = collector.attach_store(store, interval_seconds=30)
        try:
            assert first is not second
            assert first.running is False
            assert collector.store_sampler is second
        finally:
            collector.detach_store()

    def test_detach_is_safe_with_nothing_attached(self):
        collector = MetricsCollector()
        collector.detach_store()
        collector.detach_store()
        assert collector.store_sampler is None


# ---------------------------------------------------------------------------
# The history route
# ---------------------------------------------------------------------------

@pytest.fixture
def collector():
    return MetricsCollector()


@pytest.fixture
def route_app(collector, store):
    app = web.Application()
    app["config"] = SimpleNamespace(node_id="node-abc", node_name="Spark-3")
    register_metrics_routes(app, collector, store)
    return app


@pytest_asyncio.fixture
async def route_client(route_app):
    async with TestClient(TestServer(route_app)) as client:
        yield client


@pytest.mark.asyncio
class TestHistoryRoute:
    async def test_shape(self, route_client, store):
        store.write({"gpu.temperature_c": 41.0}, ts=BASE)
        resp = await route_client.get(
            f"/api/metrics/history?series=gpu.temperature_c&since={BASE}"
            f"&until={BASE + 60}&step=15"
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["resolution"] == "raw"
        assert body["step"] == 15
        assert body["since"] == BASE
        points = body["series"]["gpu.temperature_c"]
        assert len(points) == body["points"]
        assert points[0] == {"ts": BASE, "value": 41.0}
        assert body["store"]["retention_hours"] == 48
        assert body["store"]["degraded"] is False
        assert body["store"]["samples"] == 1
        assert body["store"]["downsampled"] == 0

    async def test_null_is_json_null_not_zero(self, route_client, store):
        store.write({"gpu.utilization_percent": None}, ts=BASE)
        resp = await route_client.get(
            f"/api/metrics/history?series=gpu.utilization_percent"
            f"&since={BASE}&until={BASE + 30}&step=15"
        )
        points = (await resp.json())["series"]["gpu.utilization_percent"]
        assert points[0]["value"] is None
        assert points[1]["value"] is None

    async def test_several_series_comma_separated(self, route_client, store):
        store.write({"a": 1.0, "b": 2.0}, ts=BASE)
        resp = await route_client.get(
            f"/api/metrics/history?series=a,b&since={BASE}&until={BASE + 15}&step=15"
        )
        body = await resp.json()
        assert sorted(body["series"]) == ["a", "b"]

    async def test_the_parameter_may_be_repeated(self, route_client, store):
        store.write({"a": 1.0, "b": 2.0}, ts=BASE)
        resp = await route_client.get(
            f"/api/metrics/history?series=a&series=b&since={BASE}&until={BASE + 15}"
        )
        assert sorted((await resp.json())["series"]) == ["a", "b"]

    async def test_no_series_means_all_of_them(self, route_client, store):
        store.write({"a": 1.0, "b": None}, ts=BASE)
        resp = await route_client.get(
            f"/api/metrics/history?since={BASE}&until={BASE + 15}")
        assert sorted((await resp.json())["series"]) == ["a", "b"]

    async def test_a_relative_window(self, route_client, store):
        now = time.time()
        store.write({"a": 5.0}, ts=now - 30)
        resp = await route_client.get("/api/metrics/history?series=a&since=-10m&step=15")
        body = await resp.json()
        values = [p["value"] for p in body["series"]["a"]]
        assert 5.0 in values
        assert body["resolution"] == "raw"

    async def test_units_on_the_step(self, route_client, store):
        resp = await route_client.get(
            f"/api/metrics/history?series=a&since={BASE}&until={BASE + 7200}&step=5m")
        body = await resp.json()
        assert body["step"] == 300
        assert body["resolution"] == "1m"

    async def test_milliseconds_are_accepted(self, route_client, store):
        store.write({"a": 1.0}, ts=BASE)
        resp = await route_client.get(
            f"/api/metrics/history?series=a&since={int(BASE * 1000)}"
            f"&until={int((BASE + 30) * 1000)}&step=15"
        )
        body = await resp.json()
        assert body["since"] == BASE
        assert body["series"]["a"][0]["value"] == 1.0

    async def test_a_bad_step_is_a_400(self, route_client):
        resp = await route_client.get("/api/metrics/history?step=soon")
        assert resp.status == 400
        assert "step" in (await resp.json())["error"]

    async def test_a_bare_small_number_is_a_400(self, route_client):
        # 3600 is not a timestamp, and reading it as one would answer with 1970.
        resp = await route_client.get("/api/metrics/history?since=3600")
        assert resp.status == 400

    async def test_a_bad_resolution_is_a_400(self, route_client):
        resp = await route_client.get("/api/metrics/history?resolution=hourly")
        assert resp.status == 400

    async def test_rolled_up_points_carry_min_and_max(self, route_client, store):
        for offset, value in ((0, 10.0), (30, 30.0)):
            store.write({"g": value}, ts=BASE + offset)
        store.downsample(now=BASE + 120)
        resp = await route_client.get(
            f"/api/metrics/history?series=g&since={BASE}&until={BASE + 60}&step=60")
        point = (await resp.json())["series"]["g"][0]
        assert point["value"] == 20.0
        assert (point["min"], point["max"], point["samples"]) == (10.0, 30.0, 2)


@pytest.mark.asyncio
async def test_history_says_so_when_retention_is_off(collector):
    app = web.Application()
    register_metrics_routes(app, collector)          # no store
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/metrics/history")
        assert resp.status == 200
        body = await resp.json()
        assert body["store"]["enabled"] is False
        assert body["series"] == {}


@pytest.mark.asyncio
async def test_the_existing_metrics_shapes_are_untouched(route_client):
    """The three JSON routes are what the dashboard and the broadcast read."""
    snapshot = await (await route_client.get("/api/metrics")).json()
    assert set(snapshot) == {"uptime_seconds", "requests", "gpu"}
    assert set(snapshot["requests"]) == {
        "total", "errors", "by_model", "tokens_generated",
        "latency_ms", "tokens_per_second",
    }
    gpu = await (await route_client.get("/api/metrics/gpu")).json()
    assert "error" in gpu or "gpu_count" in gpu
    stats = await (await route_client.get("/api/metrics/requests")).json()
    assert stats["total"] == 0


# ---------------------------------------------------------------------------
# Prometheus
# ---------------------------------------------------------------------------

class TestPrometheusLabels:
    def test_labels_are_stamped_on_every_series(self):
        collector = MetricsCollector()
        text = prometheus.render(collector, labels={"node": "Spark-3",
                                                   "node_id": "abc"})
        assert 'ainode_uptime_seconds{node="Spark-3",node_id="abc"}' in text
        assert 'ainode_requests_total{node="Spark-3",node_id="abc"} 0' in text
        assert 'ainode_build_info{node="Spark-3",node_id="abc",version=' in text

    def test_no_labels_renders_exactly_as_before(self):
        text = prometheus.render(MetricsCollector())
        assert "ainode_requests_total 0" in text
        assert "ainode_uptime_seconds{" not in text

    def test_identity_and_model_labels_together(self):
        collector = MetricsCollector()
        collector.record_request("meta-llama/Llama-3.2-3B", 10.0)
        text = prometheus.render(collector, labels={"node": "Spark-1"})
        assert ('ainode_requests_by_model_total{node="Spark-1",'
                'model="meta-llama/Llama-3.2-3B"} 1') in text

    def test_loaded_models_gauge(self):
        text = prometheus.render(
            MetricsCollector(),
            labels={"node": "Spark-1"},
            models=[{"model": "deepseek/V4-Flash", "status": "serving", "port": 8001}],
        )
        assert ('ainode_model_loaded{node="Spark-1",model="deepseek/V4-Flash",'
                'status="serving",port="8001"} 1') in text

    def test_no_models_emits_no_gauge(self):
        text = prometheus.render(MetricsCollector(), models=[])
        assert "ainode_model_loaded" not in text


class TestPrometheusRetention:
    def test_store_gauges(self, store):
        store.write({"a": 1.0}, ts=BASE)
        text = prometheus.render(MetricsCollector(), store=store)
        assert "ainode_metrics_retention_enabled 1" in text
        assert "ainode_metrics_retention_degraded 0" in text
        assert "ainode_metrics_retention_samples 1" in text
        assert f"ainode_metrics_retention_oldest_sample_timestamp_seconds {BASE}" in text
        assert "# TYPE ainode_metrics_retention_db_bytes gauge" in text

    def test_an_empty_store_omits_the_oldest_sample(self, store):
        text = prometheus.render(MetricsCollector(), store=store)
        assert "ainode_metrics_retention_samples 0" in text
        # Never 0: that is January 1970 on every dashboard that touches it.
        assert "ainode_metrics_retention_oldest_sample_timestamp_seconds" not in text

    def test_no_store_means_no_retention_series(self):
        text = prometheus.render(MetricsCollector())
        assert "ainode_metrics_retention" not in text

    def test_a_store_that_raises_does_not_break_the_scrape(self):
        text = prometheus.render(MetricsCollector(), store=_ExplodingStore())
        assert "ainode_requests_total 0" in text
        assert "ainode_metrics_retention" not in text

    def test_the_grammar_still_holds_with_everything_on(self, store):
        import re

        collector = MetricsCollector()
        collector.record_request("m", 12.0, tokens_generated=3)
        store.write({"a": 1.0}, ts=BASE)
        text = prometheus.render(
            collector,
            labels={"node": "Spark-3", "node_id": "abc"},
            store=store,
            models=[{"model": "m", "status": "serving", "port": 8000}],
        )
        assert text.endswith("\n")
        for line in text.split("\n"):
            if not line or line.startswith("#"):
                continue
            assert re.match(
                r"^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})?\s+-?\d+(\.\d+)?$", line
            ), f"line does not match Prometheus grammar: {line!r}"


@pytest.mark.asyncio
class TestPrometheusRoute:
    async def test_the_scrape_route_answers(self, route_client):
        """GET /metrics answers 200 with the exposition content type.

        Not a formality. The route passed both ``content_type=`` and a
        Content-Type header to ``web.Response``, which aiohttp refuses, so every
        scrape of every node since 0.4 got a 500. The only tests were against
        ``prometheus.render``, which never goes through the response object.
        """
        resp = await route_client.get("/metrics")
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "text/plain; version=0.0.4; charset=utf-8"
        text = await resp.text()
        assert text.endswith("\n")
        assert 'ainode_uptime_seconds{node="Spark-3",node_id="node-abc"}' in text

    async def test_a_node_with_no_identity_emits_no_empty_labels(self, collector):
        # node="" is indistinguishable from a missing label in PromQL, and would
        # collide every unnamed node in a fleet into one series.
        app = web.Application()
        app["config"] = SimpleNamespace(node_id=None, node_name=None)
        register_metrics_routes(app, collector)
        async with TestClient(TestServer(app)) as client:
            text = await (await client.get("/metrics")).text()
        assert "ainode_uptime_seconds " in text
        assert 'node=""' not in text


# ---------------------------------------------------------------------------
# Wiring: the app opens a store and the sampler fills it
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_app_wires_retention_and_tears_it_down(tmp_path, monkeypatch):
    from ainode.api.server import create_app
    from ainode.core.config import NodeConfig
    from ainode.metrics import store as store_module

    path = tmp_path / "wired.db"
    monkeypatch.setattr(store_module, "default_store_path", lambda: path)

    config = NodeConfig(
        node_id="wire-1", node_name="Wired", cluster_enabled=False,
        metrics={"interval_seconds": 1, "retention_hours": 3, "retention_days": 2},
    )
    config._skip_replay = True
    app = create_app(config=config, engine=None)

    async with TestClient(TestServer(app)) as client:
        store = app["metrics_store"]
        assert store.path == path
        deadline = time.time() + 5
        while time.time() < deadline and store.stats()["samples"] == 0:
            time.sleep(0.05)
        assert store.stats()["samples"] > 0

        resp = await client.get("/api/metrics/history?since=-5m&step=1s")
        body = await resp.json()
        assert body["store"]["retention_hours"] == 3
        assert "uptime_seconds" in body["series"]

        scrape = await (await client.get("/metrics")).text()
        assert 'ainode_uptime_seconds{node="Wired",node_id="wire-1"}' in scrape
        assert ('ainode_metrics_retention_enabled{node="Wired",node_id="wire-1"} 1'
                in scrape)

    # Cleanup ran: the sampler is stopped and the file is still on disk.
    assert app["metrics_collector"].store_sampler is None
    assert path.exists()


@pytest.mark.asyncio
async def test_a_config_can_turn_retention_off(tmp_path, monkeypatch):
    from ainode.api.server import create_app
    from ainode.core.config import NodeConfig
    from ainode.metrics import store as store_module

    path = tmp_path / "off.db"
    monkeypatch.setattr(store_module, "default_store_path", lambda: path)

    config = NodeConfig(node_id="off-1", cluster_enabled=False,
                        metrics={"enabled": False})
    config._skip_replay = True
    app = create_app(config=config, engine=None)
    async with TestClient(TestServer(app)) as client:
        assert "metrics_store" not in app
        body = await (await client.get("/api/metrics/history")).json()
        assert body["store"]["enabled"] is False
    assert not path.exists()


def test_the_store_path_is_under_ainode_home(monkeypatch, tmp_path):
    from ainode.core import config as core_config

    monkeypatch.setattr(core_config, "AINODE_HOME", tmp_path)
    # Read at call time, not at import: the container and the test suite both
    # set AINODE_HOME to something the developer's shell does not.
    assert default_store_path() == tmp_path / "metrics.db"
