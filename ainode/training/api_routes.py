"""API routes for the training engine — mounted under /api/training/."""

from __future__ import annotations

from aiohttp import web

from ainode.training.engine import TrainingConfig, TrainingManager


def setup_training_routes(app: web.Application, manager: TrainingManager) -> None:
    """Register training API routes on the aiohttp app."""
    app["training_manager"] = manager

    app.router.add_post("/api/training/jobs", handle_submit_job)
    app.router.add_get("/api/training/jobs", handle_list_jobs)
    app.router.add_get("/api/training/jobs/{job_id}", handle_get_job)
    app.router.add_delete("/api/training/jobs/{job_id}", handle_cancel_job)
    app.router.add_get("/api/training/jobs/{job_id}/logs", handle_get_logs)


async def handle_submit_job(request: web.Request) -> web.Response:
    """POST /api/training/jobs — submit a new training job."""
    manager: TrainingManager = request.app["training_manager"]

    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": "Invalid JSON body"}, status=400
        )

    try:
        config = TrainingConfig.from_dict(body)
    except Exception as exc:
        return web.json_response(
            {"error": f"Invalid config: {exc}"}, status=400
        )

    try:
        job = manager.submit_job(config)
    except ValueError as exc:
        return web.json_response(
            {"error": str(exc)}, status=400
        )

    # Attempt to start if nothing is running
    await manager.start_next()

    return web.json_response(job.get_status(), status=201)


async def handle_list_jobs(request: web.Request) -> web.Response:
    """GET /api/training/jobs — list all training jobs."""
    manager: TrainingManager = request.app["training_manager"]
    return web.json_response({"jobs": manager.list_jobs()})


async def handle_get_job(request: web.Request) -> web.Response:
    """GET /api/training/jobs/:job_id — get job status + progress."""
    manager: TrainingManager = request.app["training_manager"]
    job_id = request.match_info["job_id"]

    job = manager.get_job(job_id)
    if job is None:
        return web.json_response(
            {"error": f"Job not found: {job_id}"}, status=404
        )

    return web.json_response(job.get_status())


async def handle_cancel_job(request: web.Request) -> web.Response:
    """DELETE /api/training/jobs/:job_id — cancel a running or pending job."""
    manager: TrainingManager = request.app["training_manager"]
    job_id = request.match_info["job_id"]

    job = manager.get_job(job_id)
    if job is None:
        return web.json_response(
            {"error": f"Job not found: {job_id}"}, status=404
        )

    cancelled = await manager.cancel_job(job_id)
    if not cancelled:
        return web.json_response(
            {"error": f"Job {job_id} cannot be cancelled (status: {job.status.value})"},
            status=409,
        )

    return web.json_response({"status": "cancelled", "job_id": job_id})


async def handle_get_logs(request: web.Request) -> web.Response:
    """GET /api/training/jobs/:job_id/logs — return training logs."""
    manager: TrainingManager = request.app["training_manager"]
    job_id = request.match_info["job_id"]

    job = manager.get_job(job_id)
    if job is None:
        return web.json_response(
            {"error": f"Job not found: {job_id}"}, status=404
        )

    # Support ?tail=N to get only the last N log lines
    tail = request.query.get("tail")
    logs = job.logs
    if tail is not None:
        try:
            n = int(tail)
            logs = logs[-n:]
        except ValueError:
            pass

    return web.json_response({
        "job_id": job_id,
        "status": job.status.value,
        "logs": logs,
        "total_lines": len(job.logs),
    })
