"""Prometheus text-format exporter for AINode metrics.

Renders the same data that MetricsCollector.get_snapshot() returns, but in
the Prometheus exposition format (text/plain; version=0.0.4) so it can be
scraped by any Prometheus-compatible TSDB (Prometheus, VictoriaMetrics,
Grafana Mimir, Cortex, Thanos).

Deliberately does *not* depend on ``prometheus_client``. The exposition
format is small, well-specified, and inline-able — adding a third-party
dep would cost more than it saves.

Every series takes the identity labels the route passes in (``node``,
``node_id``). Without them a fleet scraped into one Prometheus produces four
``ainode_gpu_temperature_celsius`` series told apart only by the scraper's own
``instance`` label, which is an address and not a machine. The labels are an
argument rather than something read here, so ``render(collector)`` still renders
exactly what it always did.

Format reference: https://prometheus.io/docs/instrumenting/exposition_formats/
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ainode.metrics.collector import MetricsCollector, optional_float

_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def render(
    collector: MetricsCollector,
    labels: Optional[Mapping[str, Any]] = None,
    store: Any = None,
    models: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    """Render the collector's current snapshot as Prometheus text format.

    *labels* are stamped on every series (see the module docstring). *store* is a
    :class:`~ainode.metrics.store.MetricsStore`, whose own size and oldest sample
    are exported as ``ainode_metrics_retention_*``: a retention layer nobody can
    see the state of is one that silently stops. *models* is what this node is
    serving, so a fleet scrape can answer which box has which model loaded.
    """
    snapshot = collector.get_snapshot()
    base = dict(labels or {})
    lines: list[str] = []

    # -- Uptime --------------------------------------------------------------
    uptime = float(snapshot.get("uptime_seconds", 0.0))
    lines += [
        "# HELP ainode_uptime_seconds Seconds since the AINode process started.",
        "# TYPE ainode_uptime_seconds counter",
        _fmt("ainode_uptime_seconds", uptime, base),
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
        _fmt("ainode_requests_total", total, base),
        "",
        "# HELP ainode_request_errors_total Total inference requests that failed.",
        "# TYPE ainode_request_errors_total counter",
        _fmt("ainode_request_errors_total", errors, base),
        "",
        "# HELP ainode_tokens_generated_total Total tokens generated across all requests.",
        "# TYPE ainode_tokens_generated_total counter",
        _fmt("ainode_tokens_generated_total", tokens_generated, base),
        "",
        "# HELP ainode_tokens_per_second Average tokens-per-second over the process lifetime.",
        "# TYPE ainode_tokens_per_second gauge",
        _fmt("ainode_tokens_per_second", tokens_per_second, base),
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
            lines.append(_fmt(
                "ainode_requests_by_model_total", int(count),
                {**base, "model": model_name},
            ))
        lines.append("")

    # Latency summary (p50 / p95 / p99 are pre-computed by the collector).
    #
    # ABSENT until a request has been timed. The collector answers 0 for all
    # three percentiles on a node that has served nothing, because that is the
    # shape /api/metrics has always had, and exported that read as "every request
    # answers instantly": the one reading nobody can tell from a very fast node.
    # The store already refuses to persist those zeros (store.series_from_snapshot
    # stores None until `total` moves), and the exposition agrees with it: no
    # HELP, no TYPE, no samples, so a dashboard shows no data and an alert can
    # say absent() instead of == 0.
    #
    # There is no _sum. The collector keeps a window of latencies for the
    # percentiles and no running total, so the only sum available was a literal 0,
    # which makes avg = sum/count read as zero latency on every node in the
    # fleet. A summary without a sum is a summary a scraper cannot average; a
    # summary with a fake one is a wrong answer, and tracking a real one means
    # adding a field to a snapshot shape that is pinned by test and read by the
    # discovery broadcast. _count is real (it is the request counter) and stays.
    latency = requests.get("latency_ms", {}) or {}
    if total > 0:
        lines += [
            "# HELP ainode_request_latency_milliseconds Request latency percentiles "
            "(milliseconds). Absent until a request has been timed.",
            "# TYPE ainode_request_latency_milliseconds summary",
        ]
        for pct_key, quantile in (("p50", "0.5"), ("p95", "0.95"), ("p99", "0.99")):
            value = optional_float(latency.get(pct_key))
            if value is None:
                continue
            lines.append(_fmt(
                "ainode_request_latency_milliseconds", float(value),
                {**base, "quantile": quantile},
            ))
        lines.append(_fmt("ainode_request_latency_milliseconds_count", total, base))
        lines.append("")

    # -- GPU -----------------------------------------------------------------
    #
    # ainode_gpu_available is emitted in BOTH directions. It only ever carried the
    # 0, so the healthy case published no series at all and a dashboard could not
    # tell a working node from one that has never been scraped, although the HELP
    # text promised "1=yes, 0=no". The individual readings stay absent-when-
    # unreadable (a GB10 exposes no utilisation counter, so that series is simply
    # not here); this one is a yes/no about NVML itself and has an answer either
    # way.
    gpu = snapshot.get("gpu", {}) or {}
    gpu_ok = isinstance(gpu, dict) and "error" not in gpu
    lines += [
        "# HELP ainode_gpu_available Whether the GPU is queryable via pynvml (1=yes, 0=no).",
        "# TYPE ainode_gpu_available gauge",
        _fmt("ainode_gpu_available", 1 if gpu_ok else 0, base),
        "",
    ]
    if gpu_ok:
        util = gpu.get("utilization_percent")
        used = gpu.get("memory_used_mb")
        total_mem = gpu.get("memory_total_mb")
        temp = gpu.get("temperature_c")

        if util is not None:
            lines += [
                "# HELP ainode_gpu_utilization_percent GPU utilization (0-100).",
                "# TYPE ainode_gpu_utilization_percent gauge",
                _fmt("ainode_gpu_utilization_percent", float(util), base),
                "",
            ]
        if used is not None:
            lines += [
                "# HELP ainode_gpu_memory_used_bytes GPU memory in use (bytes).",
                "# TYPE ainode_gpu_memory_used_bytes gauge",
                _fmt("ainode_gpu_memory_used_bytes", int(used) * 1024 * 1024, base),
                "",
            ]
        if total_mem is not None:
            lines += [
                "# HELP ainode_gpu_memory_total_bytes Total GPU memory (bytes).",
                "# TYPE ainode_gpu_memory_total_bytes gauge",
                _fmt("ainode_gpu_memory_total_bytes", int(total_mem) * 1024 * 1024, base),
                "",
            ]
        if temp is not None:
            lines += [
                "# HELP ainode_gpu_temperature_celsius GPU temperature (C).",
                "# TYPE ainode_gpu_temperature_celsius gauge",
                _fmt("ainode_gpu_temperature_celsius", float(temp), base),
                "",
            ]

    lines += _model_lines(models, base)
    lines += _retention_lines(store, base)

    # Build info — useful for Grafana dashboards to pin panels to a version.
    from ainode import __version__

    lines += [
        "# HELP ainode_build_info AINode build metadata (value is always 1).",
        "# TYPE ainode_build_info gauge",
        _fmt("ainode_build_info", 1, {**base, "version": __version__}),
        "",
    ]

    return "\n".join(lines) + "\n"


def content_type() -> str:
    """Return the Prometheus exposition Content-Type header value."""
    return _CONTENT_TYPE


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _model_lines(
    models: Optional[Sequence[Mapping[str, Any]]],
    base: Mapping[str, Any],
) -> list[str]:
    """One gauge per model this node is serving, or nothing.

    The counters above answer "how many requests did this model take". Nothing
    answered "which node has it loaded right now", which is the question a fleet
    dashboard asks first, so an operator had to open four dashboards to find out.
    A node serving nothing emits no series rather than a zero: there is no model
    to label, and a placeholder label value would invent one.
    """
    if not models:
        return []
    lines = [
        "# HELP ainode_model_loaded A model this node currently has an engine for "
        "(value is always 1).",
        "# TYPE ainode_model_loaded gauge",
    ]
    for entry in models:
        model = str(entry.get("model") or "")
        if not model:
            continue
        extra = {**base, "model": model}
        status = str(entry.get("status") or "")
        if status:
            extra["status"] = status
        port = optional_float(entry.get("port"))
        if port is not None:
            extra["port"] = str(int(port))
        # How wide the instance is. It was collected and then dropped, so a
        # four node TP=4 instance and a solo TP=1 one produced identical series
        # and a fleet dashboard could not tell them apart. Absent, not 1, when
        # the record does not say: a default would claim solo of a shape nobody
        # reported.
        tp = optional_float(entry.get("tensor_parallel_size"))
        if tp is not None:
            extra["tp"] = str(int(tp))
        lines.append(_fmt("ainode_model_loaded", 1, extra))
    lines.append("")
    return lines


def _retention_lines(store: Any, base: Mapping[str, Any]) -> list[str]:
    """What the on-disk sample store is holding and costing.

    Cheap and honest: four numbers off one SQLite query, answering whether
    retention is on, whether it is broken, how big the file has become and how
    far back it reaches. ``oldest_sample`` is omitted on an empty store rather
    than exported as 0, which every dashboard would draw as January 1970.
    """
    if store is None:
        return []
    try:
        stats = store.stats()
    except Exception:
        return []
    if not isinstance(stats, Mapping):
        return []

    lines = [
        "# HELP ainode_metrics_retention_enabled Whether samples are being kept on "
        "disk (1=yes, 0=no).",
        "# TYPE ainode_metrics_retention_enabled gauge",
        _fmt("ainode_metrics_retention_enabled",
             1 if stats.get("available") else 0, base),
        "",
        "# HELP ainode_metrics_retention_degraded Whether a read or write to the "
        "sample store has failed (1=yes, 0=no).",
        "# TYPE ainode_metrics_retention_degraded gauge",
        _fmt("ainode_metrics_retention_degraded",
             1 if stats.get("degraded") else 0, base),
        "",
        "# HELP ainode_metrics_retention_db_bytes Size on disk of the sample store, "
        "including its write-ahead log.",
        "# TYPE ainode_metrics_retention_db_bytes gauge",
        _fmt("ainode_metrics_retention_db_bytes",
             int(optional_float(stats.get("db_bytes")) or 0), base),
        "",
        "# HELP ainode_metrics_retention_samples Raw samples currently held.",
        "# TYPE ainode_metrics_retention_samples gauge",
        _fmt("ainode_metrics_retention_samples",
             int(optional_float(stats.get("samples")) or 0), base),
        "",
        "# HELP ainode_metrics_retention_downsampled_samples One-minute roll-up rows "
        "currently held.",
        "# TYPE ainode_metrics_retention_downsampled_samples gauge",
        _fmt("ainode_metrics_retention_downsampled_samples",
             int(optional_float(stats.get("downsampled")) or 0), base),
        "",
    ]

    oldest = optional_float(stats.get("oldest_sample"))
    if oldest is not None:
        lines += [
            "# HELP ainode_metrics_retention_oldest_sample_timestamp_seconds Unix "
            "time of the oldest sample still held.",
            "# TYPE ainode_metrics_retention_oldest_sample_timestamp_seconds gauge",
            _fmt("ainode_metrics_retention_oldest_sample_timestamp_seconds",
                 oldest, base),
            "",
        ]
    return lines


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt(name: str, value: Any, labels: Optional[Mapping[str, Any]] = None) -> str:
    """One exposition line. No braces at all when there are no labels."""
    if not labels:
        return f"{name} {value}"
    rendered = ",".join(f'{key}="{_escape(val)}"' for key, val in labels.items())
    return f"{name}{{{rendered}}} {value}"


def _escape(value: Any) -> str:
    """Escape a label value per the Prometheus exposition format."""
    text = str(value)
    # Order matters — backslash first.
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
