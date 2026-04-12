"""Tests for ainode.metrics."""

import time
from unittest.mock import patch, MagicMock

import pytest

from ainode.metrics.collector import MetricsCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_pynvml():
    """Return a mock pynvml module with realistic GPU stats."""
    mod = MagicMock()

    util = MagicMock()
    util.gpu = 42
    mod.nvmlDeviceGetUtilizationRates.return_value = util

    mem = MagicMock()
    mem.used = 8 * 1024 * 1024 * 1024  # 8 GB
    mem.total = 128 * 1024 * 1024 * 1024  # 128 GB
    mod.nvmlDeviceGetMemoryInfo.return_value = mem

    mod.NVML_TEMPERATURE_GPU = 0
    mod.nvmlDeviceGetTemperature.return_value = 55

    return mod


# ---------------------------------------------------------------------------
# MetricsCollector — recording & snapshot
# ---------------------------------------------------------------------------

class TestRecordAndSnapshot:
    def test_initial_snapshot_is_empty(self):
        mc = MetricsCollector()
        snap = mc.get_snapshot()
        assert snap["requests"]["total"] == 0
        assert snap["requests"]["errors"] == 0
        assert snap["requests"]["latency_ms"]["p50"] == 0
        assert snap["uptime_seconds"] >= 0

    def test_record_single_request(self):
        mc = MetricsCollector()
        mc.record_request("llama-3", 120.5, tokens_generated=50)
        stats = mc.get_request_stats()
        assert stats["total"] == 1
        assert stats["errors"] == 0
        assert stats["by_model"]["llama-3"] == 1
        assert stats["tokens_generated"] == 50
        assert stats["latency_ms"]["p50"] == 120.5

    def test_record_error(self):
        mc = MetricsCollector()
        mc.record_request("llama-3", 500.0, error=True)
        stats = mc.get_request_stats()
        assert stats["total"] == 1
        assert stats["errors"] == 1

    def test_multiple_models(self):
        mc = MetricsCollector()
        mc.record_request("llama-3", 100.0)
        mc.record_request("llama-3", 110.0)
        mc.record_request("mistral-7b", 80.0)
        stats = mc.get_request_stats()
        assert stats["by_model"]["llama-3"] == 2
        assert stats["by_model"]["mistral-7b"] == 1
        assert stats["total"] == 3

    def test_latency_percentiles(self):
        mc = MetricsCollector()
        # Record 100 requests with latencies 1..100
        for i in range(1, 101):
            mc.record_request("m", float(i))
        stats = mc.get_request_stats()
        lat = stats["latency_ms"]
        assert lat["p50"] == pytest.approx(50.5, abs=1)
        assert lat["p95"] == pytest.approx(95.05, abs=1)
        assert lat["p99"] == pytest.approx(99.01, abs=1)

    def test_tokens_per_second(self):
        mc = MetricsCollector()
        # Backdate start_time to get predictable tps
        mc._start_time = time.time() - 10.0
        mc.record_request("m", 50.0, tokens_generated=500)
        stats = mc.get_request_stats()
        # 500 tokens / 10s = 50 tps (approximately)
        assert stats["tokens_per_second"] == pytest.approx(50.0, abs=5)


# ---------------------------------------------------------------------------
# GPU metrics
# ---------------------------------------------------------------------------

class TestGPUMetrics:
    def test_gpu_metrics_with_pynvml(self):
        mock_mod = _mock_pynvml()
        with patch.dict("sys.modules", {"pynvml": mock_mod}):
            mc = MetricsCollector()
            gpu = mc.get_gpu_metrics()
        assert gpu["utilization_percent"] == 42
        assert gpu["memory_used_mb"] == 8192
        assert gpu["memory_total_mb"] == 131072
        assert gpu["temperature_c"] == 55

    def test_gpu_metrics_without_pynvml(self):
        mc = MetricsCollector()
        with patch.dict("sys.modules", {"pynvml": None}):
            gpu = mc.get_gpu_metrics()
        assert "error" in gpu


# ---------------------------------------------------------------------------
# Thread safety (smoke test)
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_recording(self):
        import threading

        mc = MetricsCollector()
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(100):
                    mc.record_request("m", 10.0, tokens_generated=1)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        stats = mc.get_request_stats()
        assert stats["total"] == 800
        assert stats["tokens_generated"] == 800
