"""API route handlers for metrics endpoints (JSON + Prometheus)."""

from aiohttp import web

from ainode.metrics import prometheus
from ainode.metrics.collector import MetricsCollector


def register_metrics_routes(app: web.Application, collector: MetricsCollector) -> None:
    """Register metrics endpoints on the aiohttp application.

    Registers:
      - /api/metrics — full JSON snapshot
      - /api/metrics/gpu — JSON GPU subset
      - /api/metrics/requests — JSON request subset
      - /metrics — Prometheus text exposition (the standard scrape path)
    """
    app["metrics_collector"] = collector

    app.router.add_get("/api/metrics", handle_metrics)
    app.router.add_get("/api/metrics/gpu", handle_metrics_gpu)
    app.router.add_get("/api/metrics/requests", handle_metrics_requests)
    app.router.add_get("/metrics", handle_prometheus)


async def handle_metrics(request: web.Request) -> web.Response:
    """GET /api/metrics — full metrics snapshot."""
    collector: MetricsCollector = request.app["metrics_collector"]
    return web.json_response(collector.get_snapshot())


async def handle_metrics_gpu(request: web.Request) -> web.Response:
    """GET /api/metrics/gpu — real-time GPU stats."""
    collector: MetricsCollector = request.app["metrics_collector"]
    return web.json_response(collector.get_gpu_metrics())


async def handle_metrics_requests(request: web.Request) -> web.Response:
    """GET /api/metrics/requests — request stats with latency percentiles."""
    collector: MetricsCollector = request.app["metrics_collector"]
    return web.json_response(collector.get_request_stats())


async def handle_prometheus(request: web.Request) -> web.Response:
    """GET /metrics — Prometheus text exposition format."""
    collector: MetricsCollector = request.app["metrics_collector"]
    return web.Response(
        text=prometheus.render(collector),
        content_type="text/plain",
        charset="utf-8",
        headers={"Content-Type": prometheus.content_type()},
    )
