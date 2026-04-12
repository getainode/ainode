"""Metrics collector — GPU stats, request counters, latency tracking."""

import threading
import time
from collections import defaultdict
from typing import Any, Optional


class MetricsCollector:
    """Thread-safe metrics collector for AINode.

    Tracks GPU utilization, request counts/latency, tokens per second, and uptime.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.time()

        # Request counters
        self._total_requests: int = 0
        self._error_count: int = 0
        self._requests_by_model: dict[str, int] = defaultdict(int)

        # Latency tracking (store raw values for percentile computation)
        self._latencies: list[float] = []

        # Token tracking
        self._total_tokens: int = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_request(
        self,
        model: str,
        latency_ms: float,
        tokens_generated: int = 0,
        error: bool = False,
    ) -> None:
        """Record a completed inference request.

        Parameters
        ----------
        model : str
            Model name that served the request.
        latency_ms : float
            End-to-end latency in milliseconds.
        tokens_generated : int
            Number of tokens produced (0 if unknown).
        error : bool
            Whether the request resulted in an error.
        """
        with self._lock:
            self._total_requests += 1
            self._requests_by_model[model] += 1
            self._latencies.append(latency_ms)
            self._total_tokens += tokens_generated
            if error:
                self._error_count += 1

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict[str, Any]:
        """Return a full metrics snapshot (requests, latency, GPU, uptime)."""
        with self._lock:
            request_stats = self._request_stats_locked()
        gpu = self.get_gpu_metrics()
        return {
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "requests": request_stats,
            "gpu": gpu,
        }

    def get_request_stats(self) -> dict[str, Any]:
        """Return request-only stats (count, latency percentiles, errors)."""
        with self._lock:
            return self._request_stats_locked()

    def get_gpu_metrics(self) -> dict[str, Any]:
        """Query real-time GPU stats via pynvml.

        Returns a dict with utilization_percent, memory_used_mb, memory_total_mb,
        and temperature_c.  Returns an error dict if pynvml is unavailable.
        """
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            pynvml.nvmlShutdown()

            return {
                "utilization_percent": util.gpu,
                "memory_used_mb": round(mem.used / (1024 * 1024)),
                "memory_total_mb": round(mem.total / (1024 * 1024)),
                "temperature_c": temp,
            }
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Internal helpers (caller must hold self._lock)
    # ------------------------------------------------------------------

    def _request_stats_locked(self) -> dict[str, Any]:
        latencies = sorted(self._latencies) if self._latencies else []
        uptime = time.time() - self._start_time

        stats: dict[str, Any] = {
            "total": self._total_requests,
            "errors": self._error_count,
            "by_model": dict(self._requests_by_model),
            "tokens_generated": self._total_tokens,
        }

        if latencies:
            stats["latency_ms"] = {
                "p50": self._percentile(latencies, 50),
                "p95": self._percentile(latencies, 95),
                "p99": self._percentile(latencies, 99),
            }
        else:
            stats["latency_ms"] = {"p50": 0, "p95": 0, "p99": 0}

        if uptime > 0 and self._total_tokens > 0:
            stats["tokens_per_second"] = round(self._total_tokens / uptime, 2)
        else:
            stats["tokens_per_second"] = 0

        return stats

    @staticmethod
    def _percentile(sorted_data: list[float], pct: int) -> float:
        """Compute the *pct*-th percentile from pre-sorted data."""
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * (pct / 100)
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        d = k - f
        return round(sorted_data[f] + d * (sorted_data[c] - sorted_data[f]), 2)
