"""API routes for the bench - mounted under /api/bench/.

The contract, in one place:

  POST   /api/bench/runs              start a run, returns a run id
  GET    /api/bench/runs              list runs, live and on disk
  GET    /api/bench/runs/{id}         status, progress, log tail, result
  POST   /api/bench/runs/{id}/cancel  stop a run
  DELETE /api/bench/runs/{id}         forget a run and delete its result file
  GET    /api/bench/results/{id}.json the result file, the one a leaderboard eats
  GET    /api/bench/report            every local result rendered as one page

Rules the routes enforce, all of them measurement rules rather than plumbing:

  * One run at a time per node. A second request gets 409 and the running id,
    because two benchmarks against one engine measure each other's queue.
  * Refuse to start when the target instance is not serving the model: the
    readiness check is a live ``/v1/models`` probe of the engine, not an instance
    record's status latch, which stays "serving" after an engine dies.
  * Warn, never refuse, when the node is busy. Benching a node under load is a
    legitimate measurement; reporting it as if the node were idle is not.
  * Inference only. Nothing here loads, unloads or restarts anything.
"""
from __future__ import annotations

import asyncio
import json

from aiohttp import web

from ainode.bench.fleet import (
    busy_warnings,
    cluster_nodes_reader,
    describe_from_app,
    probe_ready,
    resolve_target,
)
from ainode.bench.measure import SECTIONS, BenchOptions
from ainode.bench.report import render_dir
from ainode.bench.runner import BenchBusy, BenchManager

# Sanity ceilings. Not policy, just a guard so a typo in the form cannot ask the
# node for a 400-million-token prompt or 4000 concurrent streams.
MAX_DEPTH = 2_000_000
MAX_STREAMS = 64
MAX_SWEEP = 12
MAX_TOKENS = 8192


def register_bench_routes(app: web.Application, manager: BenchManager | None = None) -> None:
    """Register the bench API on the aiohttp app."""
    app["bench_manager"] = manager if manager is not None else BenchManager()

    app.router.add_post("/api/bench/runs", handle_start_run)
    app.router.add_get("/api/bench/runs", handle_list_runs)
    app.router.add_get("/api/bench/runs/{run_id}", handle_get_run)
    app.router.add_post("/api/bench/runs/{run_id}/cancel", handle_cancel_run)
    app.router.add_delete("/api/bench/runs/{run_id}", handle_delete_run)
    app.router.add_get("/api/bench/results/{run_id}.json", handle_download_result)
    app.router.add_get("/api/bench/report", handle_report)
    app.router.add_get("/api/bench/sections", handle_sections)


# ---------------------------------------------------------------- body parsing

def _int_list(value, default, ceiling, name):
    """A sweep from the request body. Accepts a list or a comma string, because
    the form sends presets as text and an API caller sends JSON numbers."""
    if value is None or value == "":
        return list(default)
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise ValueError(f"{name} must be a list of integers")
    out = []
    for p in parts:
        try:
            n = int(p)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be integers, got {p!r}")
        if n < 1 or n > ceiling:
            raise ValueError(f"{name} values must be between 1 and {ceiling}, got {n}")
        out.append(n)
    if not out:
        raise ValueError(f"{name} is empty")
    if len(out) > MAX_SWEEP:
        raise ValueError(f"{name} has {len(out)} points; {MAX_SWEEP} is the ceiling")
    return out


def _options_from_body(body: dict) -> BenchOptions:
    """Validate a request body into BenchOptions. Raises ValueError with a
    message meant to be shown to whoever submitted the form."""
    model = (body.get("model") or "").strip()
    if not model:
        raise ValueError("model is required: pick a loaded instance")

    raw_sections = body.get("sections")
    if raw_sections in (None, "", []):
        sections = list(SECTIONS)
    else:
        if isinstance(raw_sections, str):
            raw_sections = [s.strip() for s in raw_sections.split(",") if s.strip()]
        sections = [str(s) for s in raw_sections]
        bad = [s for s in sections if s not in SECTIONS]
        if bad:
            raise ValueError(f"unknown section(s) {', '.join(bad)}; "
                             f"pick from {', '.join(SECTIONS)}")
        # Keep the canonical order regardless of what order the form ticked them
        # in, so the progress bar and the result file read the same way every run.
        sections = [s for s in SECTIONS if s in sections]

    depths = _int_list(body.get("depths"), [4000, 16000, 32000, 64000, 120000],
                       MAX_DEPTH, "depths")
    streams = _int_list(body.get("streams"), [1, 2, 4, 8, 16], MAX_STREAMS, "streams")

    def _tokens(key, default):
        v = body.get(key, default)
        try:
            n = int(v)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an integer")
        if n < 1 or n > MAX_TOKENS:
            raise ValueError(f"{key} must be between 1 and {MAX_TOKENS}")
        return n

    label = (str(body.get("label") or "").strip() or "web")
    if len(label) > 60:
        raise ValueError("label is too long (60 characters max)")

    return BenchOptions(
        url="",                      # filled from the resolved target
        model=model,
        label=label,
        sections=sections,
        depths=depths,
        streams=streams,
        no_think=bool(body.get("no_think")),
        max_tokens=_tokens("max_tokens", 200),
        sustained_tokens=_tokens("sustained_tokens", 1500),
        reasoning_tokens=_tokens("reasoning_tokens", 600),
    )


# ---------------------------------------------------------------- handlers

async def handle_sections(_request: web.Request) -> web.Response:
    """GET /api/bench/sections - what the form can tick, named by the API."""
    from ainode.bench.measure import SECTION_TITLES

    return web.json_response({"sections": [
        {"key": k, "title": SECTION_TITLES.get(k, k)} for k in SECTIONS]})


async def handle_start_run(request: web.Request) -> web.Response:
    """POST /api/bench/runs - start a run against one loaded instance."""
    manager: BenchManager = request.app["bench_manager"]
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    if not isinstance(body, dict):
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    # Cheap refusal first: a second run is rejected before we probe anything.
    running = manager.active_id
    if running:
        return web.json_response(
            {"error": "A bench run is already in progress on this node. One at a time: "
                      "two benchmarks against the same engine measure each other's "
                      "queue depth.",
             "running": running}, status=409)

    try:
        opts = _options_from_body(body)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)

    # The instance the form picked, carried through instead of re-derived. A bench
    # record is kept and compared, so its node attribution has to be the node the
    # user aimed at rather than whichever candidate routing would have taken first
    # (#197). Absent from the body (a script, an older client), the resolver falls
    # back to routing order as before.
    pick_node = str(body.get("node_id") or "").strip()
    try:
        pick_port = int(body.get("port") or 0) or None
    except (TypeError, ValueError):
        return web.json_response({"error": "port must be an integer"}, status=400)
    target = resolve_target(request.app, opts.model, node_id=pick_node, port=pick_port)
    if target is None:
        if pick_node or pick_port:
            return web.json_response(
                {"error": f"node '{pick_node or 'local'}' is not a member of this cluster, "
                          "or has no fabric IP. Pick the instance again: the fleet view "
                          "has moved on since the form was drawn."},
                status=404)
        return web.json_response(
            {"error": f"'{opts.model}' is not loaded on any node. The bench measures "
                      "what is already serving; it never loads a model."},
            status=404)
    ok, why = await probe_ready(request.app, target)
    if not ok:
        return web.json_response(
            {"error": f"The target instance is not ready: {why}"}, status=409)

    opts.url = target.url
    warnings = await busy_warnings(request.app, target)

    async def describe():
        return await describe_from_app(request.app, target)

    try:
        run = await manager.submit(
            opts, target, describe,
            telemetry_read=cluster_nodes_reader(request.app, target.node_id),
            warnings=warnings)
    except BenchBusy as exc:
        return web.json_response({"error": str(exc), "running": exc.running_id}, status=409)

    return web.json_response({
        "run_id": run.run_id,
        "status": run.status,
        "model": opts.model,
        "label": opts.label,
        "sections": list(opts.sections),
        "target": {"node": target.node_name, "node_id": target.node_id,
                   "host": target.host, "port": target.port, "endpoint": target.url},
        "warnings": warnings,
    }, status=202)


async def handle_list_runs(request: web.Request) -> web.Response:
    """GET /api/bench/runs - every run this node knows about, newest first."""
    manager: BenchManager = request.app["bench_manager"]
    rows = await asyncio.to_thread(manager.list_runs)
    return web.json_response({"runs": rows, "running": manager.active_id,
                              "results_dir": str(manager.dir)})


async def handle_get_run(request: web.Request) -> web.Response:
    """GET /api/bench/runs/{id} - status, progress, log tail, result when done."""
    manager: BenchManager = request.app["bench_manager"]
    run_id = request.match_info["run_id"]
    run = manager.get(run_id)
    if run is not None:
        try:
            tail = max(0, min(2000, int(request.query.get("log", 200))))
        except ValueError:
            tail = 200
        return web.json_response(run.get_status(log_tail=tail))
    # Not in memory: a completed run from before a restart is still a run.
    path = manager.result_path(run_id)
    if path is None:
        return web.json_response({"error": f"no bench run '{run_id}'"}, status=404)
    for row in await asyncio.to_thread(manager.list_runs):
        if row["run_id"] == run_id:
            try:
                row["result"] = json.loads(path.read_text())
            except (ValueError, OSError):
                row["result"] = None
            return web.json_response(row)
    return web.json_response({"error": f"no bench run '{run_id}'"}, status=404)


async def handle_cancel_run(request: web.Request) -> web.Response:
    """POST /api/bench/runs/{id}/cancel - stop a run in flight."""
    manager: BenchManager = request.app["bench_manager"]
    run_id = request.match_info["run_id"]
    run = manager.get(run_id)
    if run is None:
        return web.json_response({"error": f"no bench run '{run_id}'"}, status=404)
    if not await manager.cancel(run_id):
        return web.json_response(
            {"error": f"run '{run_id}' already finished ({run.status})",
             "status": run.status}, status=409)
    return web.json_response({"run_id": run_id, "status": run.status,
                              "cancelling": True})


async def handle_delete_run(request: web.Request) -> web.Response:
    """DELETE /api/bench/runs/{id} - forget the run and delete its result file."""
    manager: BenchManager = request.app["bench_manager"]
    run_id = request.match_info["run_id"]
    try:
        deleted = await asyncio.to_thread(manager.delete, run_id)
    except BenchBusy as exc:
        return web.json_response(
            {"error": "That run is still going. Cancel it first.",
             "running": exc.running_id}, status=409)
    if not deleted:
        return web.json_response({"error": f"no bench run '{run_id}'"}, status=404)
    return web.json_response({"run_id": run_id, "deleted": True})


async def handle_download_result(request: web.Request) -> web.Response:
    """GET /api/bench/results/{id}.json - the schema-1 file, verbatim.

    Served as a download rather than re-serialised: this is the file the public
    leaderboard consumes, so what a user saves has to be byte-identical to what
    is on the node.
    """
    manager: BenchManager = request.app["bench_manager"]
    run_id = request.match_info["run_id"]
    path = manager.result_path(run_id)
    if path is None:
        return web.json_response({"error": f"no bench result '{run_id}'"}, status=404)
    body = await asyncio.to_thread(path.read_bytes)
    return web.Response(body=body, content_type="application/json",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{path.name}"'})


async def handle_report(request: web.Request) -> web.Response:
    """GET /api/bench/report - every local result as one self-contained page.

    Rendered on demand rather than cached: the page is a few hundred KB of string
    building, and a stale report is worse than a slow one. The Bench view drops it
    straight into an iframe, which works because the page carries no CDN
    references and no JavaScript.
    """
    manager: BenchManager = request.app["bench_manager"]
    html = await asyncio.to_thread(render_dir, manager.dir)
    return web.Response(text=html, content_type="text/html",
                        headers={"Cache-Control": "no-store"})
