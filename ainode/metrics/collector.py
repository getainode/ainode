"""Metrics collector — GPU stats, request counters, latency tracking."""

import threading
import time
from collections import defaultdict, deque
from typing import Any, Optional


def optional_float(value: Any) -> Optional[float]:
    """``float(value)``, or None for anything that is not a number.

    One home for the rule every telemetry surface follows: a figure the node
    could not measure stays None, all the way from NVML to the browser. Written
    as ``float(x or 0)`` it becomes a claim instead, and nothing downstream can
    tell that zero apart from a measurement: an idle GPU on a node that is
    serving (#176), a full node on a part that reports host RAM (#175), a fleet
    with all its memory free (#174), a bench record asserting all three.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class MetricsCollector:
    """Thread-safe metrics collector for AINode.

    Tracks GPU utilization, request counts/latency, tokens per second, and uptime.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.time()

        # How much of this node's memory the engines running on it have RESERVED,
        # as a fraction of the total (vLLM's gpu_memory_utilization, summed over
        # the instances). On a unified-memory part NVML reports no per-device
        # usage at all, and the psutil figure that stood in for it was host RAM
        # (#175): page cache and every non-GPU process read as VRAM, so a GB10
        # showed 84 to 100 percent used forever. The reservation is the honest
        # answer to "how much of this node is committed to engines", and it is
        # something AINode itself knows. None until a provider is wired, which
        # reports the figure as unknown rather than inventing one.
        self._reservation_provider: Any = None

        # Request counters
        self._total_requests: int = 0
        self._error_count: int = 0
        self._requests_by_model: dict[str, int] = defaultdict(int)

        # Latency tracking (bounded to prevent unbounded growth)
        self._latencies: deque[float] = deque(maxlen=10000)

        # Token tracking
        self._total_tokens: int = 0

        # The on-disk retention sampler, when one is attached. None means every
        # figure below lives only in this process, which is how it was until
        # 0.5.28: a restart reset all of it and nothing kept a copy.
        self._store_sampler: Any = None

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
        """Record a completed inference request."""
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

    def set_reservation_provider(self, provider: Any) -> None:
        """Wire the "how much have the engines reserved" source.

        *provider* is a zero-arg callable returning a fraction of this node's
        memory (0.0 to 1.0), or None when it cannot be known. Only consulted on a
        unified-memory node, where NVML reports no usage of its own.
        """
        self._reservation_provider = provider

    # ------------------------------------------------------------------
    # On-disk retention
    # ------------------------------------------------------------------

    def attach_store(self, store: Any, interval_seconds: Optional[float] = None) -> Any:
        """Start ticking snapshots into *store* and return the sampler.

        The collector had no cadence of its own before this: every figure it
        holds was read on demand by ``/api/metrics``, by the discovery broadcast
        and by the dashboard's 3 second poll, and none of those is a cadence a
        node can be held to (the poll needs a browser open, the broadcast needs
        clustering on). So the sampler owns the clock, and the store gets an
        evenly spaced series whether or not anybody is watching.

        Safe to call twice: the previous sampler is stopped first.
        """
        # Imported here and not at module scope: ``ainode.metrics.store`` imports
        # ``optional_float`` from this module, so a top-level import either way
        # round would be a cycle.
        from ainode.metrics.store import MetricsSampler

        self.detach_store()
        sampler = MetricsSampler(
            store, self.get_snapshot, interval_seconds=interval_seconds
        )
        self._store_sampler = sampler
        sampler.start()
        return sampler

    def detach_store(self) -> None:
        """Stop the retention sampler, if one is running. Safe to call twice."""
        sampler, self._store_sampler = self._store_sampler, None
        if sampler is None:
            return
        try:
            sampler.stop()
        except Exception:
            # Shutdown is not a place to raise: a sampler that will not stop
            # cleanly is a daemon thread the interpreter will drop anyway.
            pass

    @property
    def store_sampler(self) -> Any:
        """The attached sampler, or None."""
        return self._store_sampler

    def _reserved_fraction(self) -> Optional[float]:
        provider = self._reservation_provider
        if provider is None:
            return None
        try:
            value = provider()
        except Exception:
            return None
        if value is None:
            return None
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return None

    def get_gpu_metrics(self) -> dict[str, Any]:
        """Query real-time GPU stats via pynvml, for EVERY device on the node.

        Returns utilization_percent, memory_used_mb, memory_total_mb and
        temperature_c for the node as a whole, plus ``gpu_count``, ``devices``
        (the per-device list) and ``memory_kind``. Returns an error dict if
        pynvml is unavailable.

        Three rules make these numbers true rather than merely present:

        * **Every device counts.** Device 0 alone reported a four-V100 host as
          one 32 GB GPU (#163); memory here is the sum across devices.
        * **Unknown is None, never 0.** A driver that does not populate the
          utilization counter returns 0, which renders as "idle" on a node that
          is serving (#176). A figure we cannot read is None, and the interface
          draws it as n/a.
        * **Host RAM is not VRAM.** On a unified-memory part (GB10) the old
          psutil fallback published page cache and every other process as GPU
          memory used (#175). Usage there is what the engines RESERVED, from
          ``set_reservation_provider``; the host reading is reported separately
          as system memory and never as VRAM.
        """
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                warnings.simplefilter("ignore", FutureWarning)
                import pynvml

            pynvml.nvmlInit()
            try:
                count = int(pynvml.nvmlDeviceGetCount())
            except Exception:
                count = 1

            devices: list[dict[str, Any]] = []
            for index in range(max(0, count)):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                except Exception:
                    continue
                devices.append(self._device_metrics(pynvml, handle, index))

            pynvml.nvmlShutdown()

            if not devices:
                return {"error": "no NVIDIA device reported by NVML"}

            return self._node_metrics(devices)
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # GPU helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _device_metrics(pynvml: Any, handle: Any, index: int) -> dict[str, Any]:
        """One device's live figures. Anything unreadable comes back None."""
        try:
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
        except Exception:
            name = ""

        # Unified memory (DGX Spark / GB10): NVML does not raise here, it returns
        # a struct of zeros (the same reason `nvidia-smi` prints N/A). A zero
        # total is therefore the unified case, not a 0 GB card.
        used_mb: Optional[int] = None
        total_mb = 0
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mb = round(int(mem.total) / (1024 * 1024))
            if total_mb:
                used_mb = round(int(mem.used) / (1024 * 1024))
        except Exception:
            total_mb = 0
        unified = not total_mb

        # A device whose memory the driver will not report does not populate the
        # utilization counter either: the 0 it answers with is the struct's
        # default, not a measurement (`nvidia-smi` on GB10 prints `0 %` next to
        # `[N/A]` memory). Reporting that 0 is what made every node in the fleet
        # read as idle while it was serving (#176), so the unified case is
        # unknown by construction, and a call that raises is unknown too.
        util: Optional[int] = None
        if not unified:
            try:
                util = int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            except Exception:
                util = None

        temp: Optional[int] = None
        try:
            temp = int(pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            temp = None

        return {
            "index": index,
            "name": str(name),
            "memory_used_mb": used_mb,
            "memory_total_mb": total_mb,
            "utilization_percent": util,
            "temperature_c": temp,
            "unified_memory": unified,
        }

    def _node_metrics(self, devices: list[dict[str, Any]]) -> dict[str, Any]:
        """Roll per-device figures up to the node, keeping unknowns unknown."""
        unified = any(d["unified_memory"] for d in devices)

        utils = [d["utilization_percent"] for d in devices
                 if d["utilization_percent"] is not None]
        temps = [d["temperature_c"] for d in devices if d["temperature_c"] is not None]

        payload: dict[str, Any] = {
            "gpu_count": len(devices),
            "memory_kind": "unified" if unified else "dedicated",
            # Mean across the devices that answered: one number for the node, and
            # None when no device would say. Never a 0 standing in for silence.
            "utilization_percent": (round(sum(utils) / len(utils)) if utils else None),
            # The hottest device is the node's thermal story.
            "temperature_c": (max(temps) if temps else None),
            "devices": devices,
        }

        if unified:
            # One pool, shared by every device and by the host: counted once.
            import psutil
            vm = psutil.virtual_memory()
            total_mb = round(vm.total / (1024 * 1024))
            fraction = self._reserved_fraction()
            payload.update({
                "memory_total_mb": total_mb,
                "memory_used_mb": (round(total_mb * fraction)
                                   if fraction is not None else None),
                "memory_used_source": ("engine_reservations"
                                       if fraction is not None else None),
                "engine_reserved_fraction": fraction,
                # The host reading, labelled as what it is. It is NOT VRAM: on
                # this hardware it counts page cache and every other process.
                "system_memory_used_mb": round(vm.used / (1024 * 1024)),
                "system_memory_total_mb": total_mb,
            })
        else:
            payload.update({
                "memory_total_mb": sum(d["memory_total_mb"] for d in devices),
                "memory_used_mb": sum(d["memory_used_mb"] or 0 for d in devices),
                "memory_used_source": "nvml",
                "engine_reserved_fraction": None,
            })
        return payload

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
