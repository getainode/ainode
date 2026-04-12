"""API route handlers for /api/metrics endpoints."""

from aiohttp import web

from ainode.metrics.collector import MetricsCollector


def register_metrics_routes(app: web.Application, collector: MetricsCollector) -> None:
    """Register metrics endpoints on the aiohttp application.

    Parameters
    ----------
    app : web.Application
        The aiohttp app to add routes to.
    collector : MetricsCollector
        Shared collector instance used by these handlers.
    """
    app["metrics_collector"] = collector

    app.router.add_get("/api/metrics", handle_metrics)
    app.router.add_get("/api/metrics/gpu", handle_metrics_gpu)
    app.router.add_get("/api/metrics/requests", handle_metrics_requests)


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
