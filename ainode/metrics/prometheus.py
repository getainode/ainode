"""Prometheus text-format exporter for AINode metrics.

Renders the same data that MetricsCollector.get_snapshot() returns, but in
the Prometheus exposition format (text/plain; version=0.0.4) so it can be
scraped by any Prometheus-compatible TSDB (Prometheus, VictoriaMetrics,
Grafana Mimir, Cortex, Thanos).

Deliberately does *not* depend on ``prometheus_client``. The exposition
format is small, well-specified, and inline-able — adding a third-party
dep would cost more than it saves.

Format reference: https://prometheus.io/docs/instrumenting/exposition_formats/
"""

from __future__ import annotations

from typing import Any

from ainode.metrics.collector import MetricsCollector

_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def render(collector: MetricsCollector) -> str:
    """Render the collector's current snapshot as Prometheus text format."""
    snapshot = collector.get_snapshot()
    lines: list[str] = []

    # -- Uptime --------------------------------------------------------------
    uptime = float(snapshot.get("uptime_seconds", 0.0))
    lines += [
        "# HELP ainode_uptime_seconds Seconds since the AINode process started.",
        "# TYPE ainode_uptime_seconds counter",
        f"ainode_uptime_seconds {uptime}",
        "",
    ]

    # -- Requests ------------------------------------------------------------
    requests = snapshot.get("requests", {}) or {}
    total = int(requests.get("total", 0))
    errors = int(requests.get("errors", 0))
    tokens_generated = int(requests.get("tokens_generated", 0))
    tokens_per_second = float(requests.get("tokens_per_second", 0.0))

    lines += [
        "# HELP ainode_requests_total Total inference requests processed.",
        "# TYPE ainode_requests_total counter",
        f"ainode_requests_total {total}",
        "",
        "# HELP ainode_request_errors_total Total inference requests that failed.",
        "# TYPE ainode_request_errors_total counter",
        f"ainode_request_errors_total {errors}",
        "",
        "# HELP ainode_tokens_generated_total Total tokens generated across all requests.",
        "# TYPE ainode_tokens_generated_total counter",
        f"ainode_tokens_generated_total {tokens_generated}",
        "",
        "# HELP ainode_tokens_per_second Average tokens-per-second over the process lifetime.",
        "# TYPE ainode_tokens_per_second gauge",
        f"ainode_tokens_per_second {tokens_per_second}",
        "",
    ]

    # Per-model request counts (labelled)
    by_model = requests.get("by_model", {}) or {}
    if by_model:
        lines += [
            "# HELP ainode_requests_by_model_total Request count broken down by model.",
            "# TYPE ainode_requests_by_model_total counter",
        ]
        for model_name, count in by_model.items():
            lines.append(
                f'ainode_requests_by_model_total{{model="{_escape(model_name)}"}} {int(count)}'
            )
        lines.append("")

    # Latency summary (p50 / p95 / p99 are pre-computed by the collector)
    latency = requests.get("latency_ms", {}) or {}
    lines += [
        "# HELP ainode_request_latency_milliseconds Request latency percentiles (milliseconds).",
        "# TYPE ainode_request_latency_milliseconds summary",
    ]
    for pct_key, quantile in (("p50", "0.5"), ("p95", "0.95"), ("p99", "0.99")):
        value = float(latency.get(pct_key, 0.0) or 0.0)
        lines.append(
            f'ainode_request_latency_milliseconds{{quantile="{quantile}"}} {value}'
        )
    # Also emit count + sum so Prometheus recording rules can compute
    # additional aggregates (avg = sum/count) if a scraper wants them.
    lines.append(f"ainode_request_latency_milliseconds_count {total}")
    lines.append(f"ainode_request_latency_milliseconds_sum 0")  # exact sum not tracked
    lines.append("")

    # -- GPU -----------------------------------------------------------------
    gpu = snapshot.get("gpu", {}) or {}
    if isinstance(gpu, dict) and "error" not in gpu:
        util = gpu.get("utilization_percent")
        used = gpu.get("memory_used_mb")
        total_mem = gpu.get("memory_total_mb")
        temp = gpu.get("temperature_c")

        if util is not None:
            lines += [
                "# HELP ainode_gpu_utilization_percent GPU utilization (0-100).",
                "# TYPE ainode_gpu_utilization_percent gauge",
                f"ainode_gpu_utilization_percent {float(util)}",
                "",
            ]
        if used is not None:
            lines += [
                "# HELP ainode_gpu_memory_used_bytes GPU memory in use (bytes).",
                "# TYPE ainode_gpu_memory_used_bytes gauge",
                f"ainode_gpu_memory_used_bytes {int(used) * 1024 * 1024}",
                "",
            ]
        if total_mem is not None:
            lines += [
                "# HELP ainode_gpu_memory_total_bytes Total GPU memory (bytes).",
                "# TYPE ainode_gpu_memory_total_bytes gauge",
                f"ainode_gpu_memory_total_bytes {int(total_mem) * 1024 * 1024}",
                "",
            ]
        if temp is not None:
            lines += [
                "# HELP ainode_gpu_temperature_celsius GPU temperature (C).",
                "# TYPE ainode_gpu_temperature_celsius gauge",
                f"ainode_gpu_temperature_celsius {float(temp)}",
                "",
            ]
    else:
        # Surface the pynvml error as a gauge=0 stat so Grafana panels don't
        # silently show "No data" when the GPU is unreachable.
        lines += [
            "# HELP ainode_gpu_available Whether the GPU is queryable via pynvml (1=yes, 0=no).",
            "# TYPE ainode_gpu_available gauge",
            "ainode_gpu_available 0",
            "",
        ]

    # Build info — useful for Grafana dashboards to pin panels to a version.
    from ainode import __version__

    lines += [
        "# HELP ainode_build_info AINode build metadata (value is always 1).",
        "# TYPE ainode_build_info gauge",
        f'ainode_build_info{{version="{_escape(__version__)}"}} 1',
        "",
    ]

    return "\n".join(lines) + "\n"


def content_type() -> str:
    """Return the Prometheus exposition Content-Type header value."""
    return _CONTENT_TYPE


def _escape(value: Any) -> str:
    """Escape a label value per the Prometheus exposition format."""
    text = str(value)
    # Order matters — backslash first.
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
