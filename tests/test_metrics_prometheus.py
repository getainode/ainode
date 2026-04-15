"""Tests for ainode.metrics.prometheus — text exposition format."""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from ainode.metrics import prometheus
from ainode.metrics.collector import MetricsCollector


@pytest.fixture
def collector_no_gpu():
    """A collector that reports no GPU (pynvml import fails)."""
    c = MetricsCollector()
    with patch.object(c, "get_gpu_metrics", return_value={"error": "pynvml not available"}):
        yield c


@pytest.fixture
def collector_with_gpu():
    c = MetricsCollector()
    with patch.object(
        c,
        "get_gpu_metrics",
        return_value={
            "utilization_percent": 72,
            "memory_used_mb": 61_440,
            "memory_total_mb": 131_072,
            "temperature_c": 55,
        },
    ):
        yield c


def test_content_type_version_matches_prometheus_spec():
    ct = prometheus.content_type()
    assert "text/plain" in ct
    assert "version=0.0.4" in ct
    assert "charset=utf-8" in ct


def test_render_includes_required_sections(collector_no_gpu):
    text = prometheus.render(collector_no_gpu)
    # Every exported series must have a HELP and TYPE line.
    assert "# HELP ainode_uptime_seconds" in text
    assert "# TYPE ainode_uptime_seconds counter" in text
    assert "# HELP ainode_requests_total" in text
    assert "# TYPE ainode_requests_total counter" in text
    assert "# HELP ainode_tokens_per_second" in text
    assert "# TYPE ainode_tokens_per_second gauge" in text
    assert "# HELP ainode_build_info" in text
    # Build info always emits a labelled "1" value.
    assert re.search(r'ainode_build_info\{version="[^"]+"\} 1', text)


def test_render_gpu_available_flag_when_no_gpu(collector_no_gpu):
    text = prometheus.render(collector_no_gpu)
    assert "ainode_gpu_available 0" in text
    # Must NOT include utilization series when GPU is absent.
    assert "ainode_gpu_utilization_percent" not in text


def test_render_gpu_series_present_when_gpu_ok(collector_with_gpu):
    text = prometheus.render(collector_with_gpu)
    assert "ainode_gpu_utilization_percent 72.0" in text
    # 61_440 MB * 1024 * 1024 = 64_424_509_440 bytes
    assert "ainode_gpu_memory_used_bytes 64424509440" in text
    assert "ainode_gpu_memory_total_bytes 137438953472" in text
    assert "ainode_gpu_temperature_celsius 55.0" in text
    # And the "no GPU" fallback should NOT be present.
    assert "ainode_gpu_available 0" not in text


def test_render_per_model_counts_with_label_escaping(collector_no_gpu):
    collector_no_gpu.record_request(model="meta-llama/Llama-3.1-8B", latency_ms=42.0, tokens_generated=100)
    collector_no_gpu.record_request(model='quirky"model\\with\nchars', latency_ms=11.0, tokens_generated=10)

    text = prometheus.render(collector_no_gpu)

    assert 'ainode_requests_by_model_total{model="meta-llama/Llama-3.1-8B"} 1' in text
    # Escapes: \\ first, then \" and \n
    assert (
        'ainode_requests_by_model_total{model="quirky\\"model\\\\with\\nchars"} 1'
        in text
    )


def test_render_latency_summary_has_standard_quantiles(collector_no_gpu):
    for latency in [5.0, 10.0, 50.0, 100.0, 200.0, 500.0]:
        collector_no_gpu.record_request("m", latency_ms=latency, tokens_generated=1)

    text = prometheus.render(collector_no_gpu)

    assert 'ainode_request_latency_milliseconds{quantile="0.5"}' in text
    assert 'ainode_request_latency_milliseconds{quantile="0.95"}' in text
    assert 'ainode_request_latency_milliseconds{quantile="0.99"}' in text
    assert "ainode_request_latency_milliseconds_count 6" in text


def test_render_trailing_newline(collector_no_gpu):
    """Prometheus parsers strictly require a trailing newline."""
    text = prometheus.render(collector_no_gpu)
    assert text.endswith("\n")


def test_render_is_parseable_key_value_lines(collector_no_gpu):
    """Every non-comment, non-blank line must be `name{labels?} value`."""
    collector_no_gpu.record_request("m", latency_ms=12.3, tokens_generated=5)
    text = prometheus.render(collector_no_gpu)

    for line in text.split("\n"):
        if not line or line.startswith("#"):
            continue
        # metric_name{...}? value
        assert re.match(r"^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})?\s+-?\d+(\.\d+)?$", line), (
            f"line does not match Prometheus grammar: {line!r}"
        )
