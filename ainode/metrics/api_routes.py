"""API route handlers for metrics endpoints (JSON + Prometheus)."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from aiohttp import web

from ainode.metrics import prometheus
from ainode.metrics.collector import MetricsCollector
from ainode.metrics.store import MAX_HISTORY_POINTS, MetricsStore

#: Suffixes accepted on a duration or a relative instant in the query string.
_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def register_metrics_routes(
    app: web.Application,
    collector: MetricsCollector,
    store: Optional[MetricsStore] = None,
) -> None:
    """Register metrics endpoints on the aiohttp application.

    Registers:
      - /api/metrics — full JSON snapshot
      - /api/metrics/gpu — JSON GPU subset
      - /api/metrics/requests — JSON request subset
      - /api/metrics/history: retained series from <AINODE_HOME>/metrics.db
      - /metrics — Prometheus text exposition (the standard scrape path)

    The first three shapes are unchanged and deliberately so: they are what the
    dashboard, the discovery broadcast and every existing scrape read, and the
    history route is additive.
    """
    app["metrics_collector"] = collector
    if store is not None:
        app["metrics_store"] = store

    app.router.add_get("/api/metrics", handle_metrics)
    app.router.add_get("/api/metrics/gpu", handle_metrics_gpu)
    app.router.add_get("/api/metrics/requests", handle_metrics_requests)
    app.router.add_get("/api/metrics/history", handle_metrics_history)
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


async def handle_metrics_history(request: web.Request) -> web.Response:
    """GET /api/metrics/history: what this node measured, from disk.

    Query parameters, all optional:

    ``series``
        Comma-separated names, or the parameter repeated. Omitted means every
        series the store holds. Names are the flattened snapshot keys, for
        instance ``gpu.temperature_c`` or ``requests.latency_ms.p95``.
    ``since`` / ``until``
        A unix timestamp in seconds (milliseconds are accepted and divided, so a
        browser can pass ``Date.now()``), or a relative offset from now with a
        leading minus and an optional unit: ``-30m``, ``-6h``, ``-2d``. Defaults
        are one hour ago and now.
    ``step``
        Grid width in seconds, with the same optional unit suffix. Defaults to
        the sampling interval for raw data and one minute for roll-ups. The
        response reports the step it actually used, which may be coarser than
        asked for so the payload stays drawable.
    ``resolution``
        ``raw`` or ``1m``, to override the automatic choice. The automatic choice
        is the roll-up table whenever the window reaches back past the raw
        retention or the step is a minute or more.

    Every series comes back as the same evenly spaced grid, ``value: null`` in
    any slot where nothing was measured. Null is never filled in: a gap is a gap,
    whether the sampler was down or the driver would not answer.
    """
    store: Optional[MetricsStore] = request.app.get("metrics_store")
    if store is None:
        # Retention off, or the store could not be opened at boot. Answer the
        # shape rather than an error, so a caller seeding a chart gets an empty
        # chart and a reason instead of having to special-case a status code.
        return web.json_response({
            "store": {"enabled": False, "available": False, "degraded": False},
            "series": {},
            "points": 0,
        })

    try:
        query = _parse_history_query(request)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)

    names = query["series"]
    budget = MAX_HISTORY_POINTS
    loop = asyncio.get_event_loop()
    # SQLite on the event loop is a few milliseconds for a window this size, but
    # a month of roll-ups on a busy node is not, and this server is also proxying
    # inference. Read in the executor, the same call the sampler thread makes.
    payload = await loop.run_in_executor(
        None,
        lambda: store.history(
            series=names,
            since=query["since"],
            until=query["until"],
            step=query["step"],
            resolution=query["resolution"],
            max_points=budget,
        ),
    )
    stats = store.stats()
    payload["store"] = {
        "enabled": stats["enabled"],
        "available": stats["available"],
        "degraded": stats["degraded"],
        "retention_hours": stats["retention_hours"],
        "retention_days": stats["retention_days"],
        "interval_seconds": stats["interval_seconds"],
        "oldest_sample": stats["oldest_sample"],
        "newest_sample": stats["newest_sample"],
        "db_bytes": stats["db_bytes"],
        "samples": stats["samples"],
        "downsampled": stats["downsampled"],
    }
    return web.json_response(payload)


async def handle_prometheus(request: web.Request) -> web.Response:
    """GET /metrics, the Prometheus text exposition format.

    Answers with ``body=`` and one explicit header, NOT with ``text=`` plus
    ``content_type=``. aiohttp refuses a response that carries both a
    Content-Type header and the content_type/charset arguments, so this route
    raised ValueError on every request and answered 500: the endpoint has
    existed since 0.4 and could not be scraped once. Nothing caught it because
    the only tests were against ``prometheus.render`` and never the route.

    The header has to be set by hand because the Prometheus content type carries
    a parameter (``version=0.0.4``) that aiohttp's ``content_type`` argument does
    not accept, and a scraper uses it to pick its parser.
    """
    collector: MetricsCollector = request.app["metrics_collector"]
    rendered = prometheus.render(
        collector,
        labels=node_labels(request.app),
        store=request.app.get("metrics_store"),
        models=loaded_models(request.app),
    )
    return web.Response(
        body=rendered.encode("utf-8"),
        headers={"Content-Type": prometheus.content_type()},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def node_labels(app: web.Application) -> dict[str, str]:
    """Labels that make a scraped series mean something across a fleet.

    Without these, four Sparks scraped into one Prometheus produce four
    ``ainode_gpu_temperature_celsius`` series told apart only by the ``instance``
    label the scraper adds, which is an address: it changes when a node moves and
    says nothing about which box it is. ``node`` is the name an operator uses and
    ``node_id`` is the stable key, so a dashboard can label by one and join on the
    other.

    Absent when the config has neither, rather than emitting ``node=""``: an
    empty label value is indistinguishable from a missing one in PromQL, and
    inventing a placeholder would collide every unnamed node into one series.
    """
    config = app.get("config")
    labels: dict[str, str] = {}
    node_id = getattr(config, "node_id", None)
    node_name = getattr(config, "node_name", None)
    display = node_name or node_id
    if display:
        labels["node"] = str(display)
    if node_id:
        labels["node_id"] = str(node_id)
    return labels


def loaded_models(app: web.Application) -> list[dict[str, Any]]:
    """Which models this node is serving, for the ``ainode_model_loaded`` gauge.

    Read defensively off the instance manager: a scrape must not 500 because the
    manager is not seeded yet, and ``/metrics`` is the one route a monitoring
    system hits every fifteen seconds forever.
    """
    manager = app.get("instances")
    if manager is None:
        return []
    try:
        records = manager.records()
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for record in records or []:
        model = getattr(record, "model", "") or ""
        if not model:
            continue
        out.append({
            "model": str(model),
            "status": str(getattr(record, "status", "") or ""),
            "port": getattr(record, "api_port", None),
            "tensor_parallel_size": getattr(record, "tensor_parallel_size", None),
        })
    return out


def _parse_history_query(request: web.Request) -> dict[str, Any]:
    """Pull the window out of the query string, or raise ValueError."""
    raw_series: list[str] = []
    for value in request.query.getall("series", []):
        raw_series.extend(part.strip() for part in value.split(","))
    names = [name for name in raw_series if name]

    now = time.time()
    since = _parse_instant(request.query.get("since"), now, "since")
    until = _parse_instant(request.query.get("until"), now, "until")
    step = _parse_duration(request.query.get("step"), "step")

    resolution = (request.query.get("resolution") or "").strip().lower() or None
    if resolution is not None and resolution not in ("raw", "1m"):
        raise ValueError("resolution must be raw or 1m")

    return {
        "series": names or None,
        "since": since,
        "until": until,
        "step": step,
        "resolution": resolution,
    }


def _parse_duration(text: Optional[str], field: str) -> Optional[float]:
    """Seconds from ``"90"``, ``"90s"``, ``"5m"``, ``"6h"``, ``"2d"``."""
    if text is None:
        return None
    body = text.strip().lower()
    if not body:
        return None
    unit = 1.0
    if body[-1] in _UNITS:
        unit = _UNITS[body[-1]]
        body = body[:-1]
    try:
        return float(body) * unit
    except ValueError:
        raise ValueError(f"{field} is not a duration: {text!r}") from None


def _parse_instant(text: Optional[str], now: float, field: str) -> Optional[float]:
    """A unix timestamp, or an offset from now when it leads with a sign."""
    if text is None:
        return None
    body = text.strip()
    if not body:
        return None
    if body[0] in "+-":
        offset = _parse_duration(body, field)
        return now + (offset or 0.0)
    value = _parse_duration(body, field)
    if value is None:
        return None
    if value >= 1e12:
        # A browser passing Date.now(). Accepted rather than read as the year
        # 33658, which is what treating it as seconds would mean.
        value = value / 1000.0
    if value < 1e9:
        raise ValueError(
            f"{field} must be a unix timestamp in seconds, or a relative offset "
            f"such as -1h; got {text!r}"
        )
    return value
