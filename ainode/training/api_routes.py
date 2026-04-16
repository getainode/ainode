"""API routes for the training engine — mounted under /api/training/."""

from __future__ import annotations

from pathlib import Path

from aiohttp import web

from ainode.training.engine import (
    TrainingConfig,
    TrainingManager,
    get_training_templates,
)


def setup_training_routes(app: web.Application, manager: TrainingManager) -> None:
    """Register training API routes on the aiohttp app."""
    app["training_manager"] = manager

    app.router.add_post("/api/training/jobs", handle_submit_job)
    app.router.add_get("/api/training/jobs", handle_list_jobs)
    app.router.add_get("/api/training/jobs/{job_id}", handle_get_job)
    app.router.add_delete("/api/training/jobs/{job_id}", handle_cancel_job)
    app.router.add_get("/api/training/jobs/{job_id}/logs", handle_get_logs)
    app.router.add_get("/api/training/jobs/{job_id}/output", handle_get_output)
    app.router.add_get("/api/training/jobs/{job_id}/output/{filename}", handle_download_artifact)
    app.router.add_post("/api/training/jobs/{job_id}/merge", handle_merge_adapter)
    app.router.add_post("/api/training/jobs/{job_id}/resume", handle_resume_job)
    app.router.add_post("/api/training/templates", handle_save_template)
    app.router.add_get("/api/training/templates", handle_templates)
    app.router.add_get("/api/training/stats", handle_stats)
    app.router.add_post("/api/training/estimate", handle_estimate)


async def handle_templates(_request: web.Request) -> web.Response:
    """GET /api/training/templates — return built-in + custom templates."""
    import json as _json
    built_in = get_training_templates()
    # Load persisted custom templates
    try:
        from ainode.core.config import AINODE_HOME
        templates_path = AINODE_HOME / "training" / "custom_templates.json"
        custom = _json.loads(templates_path.read_text()) if templates_path.exists() else []
    except Exception:
        custom = []
    return web.json_response({"templates": built_in + custom})


async def handle_stats(request: web.Request) -> web.Response:
    """GET /api/training/stats — summary counters for the overview dashboard."""
    manager: TrainingManager = request.app["training_manager"]
    return web.json_response(manager.stats())


async def handle_estimate(request: web.Request) -> web.Response:
    """POST /api/training/estimate — rough time/memory/throughput estimates."""
    manager: TrainingManager = request.app["training_manager"]
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    sample_count = body.pop("sample_count", None)
    try:
        cfg = TrainingConfig.from_dict(body)
    except Exception as exc:
        return web.json_response({"error": f"Invalid config: {exc}"}, status=400)
    return web.json_response(manager.estimate(cfg, sample_count=sample_count))


async def handle_submit_job(request: web.Request) -> web.Response:
    """POST /api/training/jobs — submit a new training job."""
    manager: TrainingManager = request.app["training_manager"]

    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": "Invalid JSON body"}, status=400
        )

    # Propagate HF token from NodeConfig if the request doesn't supply one.
    # This lets users set it once via `ainode config --hf-token` and have
    # it automatically available for all training jobs on gated models.
    if not body.get("hf_token"):
        node_config = request.app.get("config")
        if node_config and getattr(node_config, "hf_token", None):
            body["hf_token"] = node_config.hf_token

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
    logs = list(job.logs)
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


async def handle_get_output(request: web.Request) -> web.Response:
    """GET /api/training/jobs/:job_id/output — list artifact files from the output dir."""
    import os
    manager: TrainingManager = request.app["training_manager"]
    job_id = request.match_info["job_id"]

    job = manager.get_job(job_id)
    if job is None:
        return web.json_response({"error": f"Job not found: {job_id}"}, status=404)

    output_dir = Path(job.config.output_dir) if job.config.output_dir else None
    if output_dir is None or not output_dir.exists():
        return web.json_response({
            "job_id": job_id,
            "output_dir": str(output_dir) if output_dir else None,
            "files": [],
            "status": job.status.value,
        })

    files = []
    for entry in sorted(output_dir.iterdir()):
        if entry.is_file():
            stat = entry.stat()
            files.append({
                "name": entry.name,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "download_url": f"/api/training/jobs/{job_id}/output/{entry.name}",
            })

    return web.json_response({
        "job_id": job_id,
        "status": job.status.value,
        "output_dir": str(output_dir),
        "files": files,
        "total_files": len(files),
        "total_size_mb": round(sum(f["size_mb"] for f in files), 2),
    })


async def handle_download_artifact(request: web.Request) -> web.Response:
    """GET /api/training/jobs/:job_id/output/:filename — stream an artifact file."""
    import mimetypes
    manager: TrainingManager = request.app["training_manager"]
    job_id = request.match_info["job_id"]
    filename = request.match_info["filename"]

    # Path traversal guard
    if ".." in filename or "/" in filename or "\\" in filename:
        return web.json_response({"error": "Invalid filename"}, status=400)

    job = manager.get_job(job_id)
    if job is None:
        return web.json_response({"error": f"Job not found: {job_id}"}, status=404)

    output_dir = Path(job.config.output_dir) if job.config.output_dir else None
    if output_dir is None or not output_dir.exists():
        return web.json_response({"error": "Output directory not found"}, status=404)

    file_path = output_dir / filename
    if not file_path.exists() or not file_path.is_file():
        return web.json_response({"error": f"File not found: {filename}"}, status=404)

    content_type, _ = mimetypes.guess_type(str(file_path))
    content_type = content_type or "application/octet-stream"

    return web.FileResponse(
        path=file_path,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": content_type,
        },
    )


async def handle_merge_adapter(request: web.Request) -> web.Response:
    """POST /api/training/jobs/:job_id/merge — merge a LoRA/QLoRA adapter into the base model.

    Runs in the background (blocking ~5-20 min depending on model size).
    Returns a job-like status object with a merge_job_id the caller can poll
    via GET /api/training/jobs/{merge_job_id}.
    """
    import asyncio
    manager: TrainingManager = request.app["training_manager"]
    job_id = request.match_info["job_id"]

    job = manager.get_job(job_id)
    if job is None:
        return web.json_response({"error": f"Job not found: {job_id}"}, status=404)

    if job.status.value not in ("completed",):
        return web.json_response(
            {"error": f"Job must be completed to merge. Current status: {job.status.value}"},
            status=409,
        )

    if job.config.method not in ("lora", "qlora"):
        return web.json_response(
            {"error": f"Merge only applies to LoRA/QLoRA jobs. Method: {job.config.method}"},
            status=400,
        )

    try:
        body = await request.json()
    except Exception:
        body = {}

    adapter_dir = Path(job.config.output_dir)
    if not adapter_dir.exists():
        return web.json_response(
            {"error": f"Output directory not found: {adapter_dir}"},
            status=404,
        )

    # Determine merged output location
    merged_dir_name = body.get("output_dir") or str(adapter_dir.parent / "merged")
    merged_dir = Path(merged_dir_name)

    # Submit merge as a background task via a synthetic TrainingConfig
    from ainode.training.engine import TrainingConfig, JobStatus
    merge_config = TrainingConfig(
        base_model=job.config.base_model,
        dataset_path="__merge__",  # sentinel — no dataset needed
        method="__merge__",
        output_dir=merged_dir_name,
        run_name=f"merge-{job_id}",
        description=f"LoRA merge from job {job_id}",
        hf_token=job.config.hf_token,
    )

    merge_job = manager.submit_job(merge_config)
    merge_job._adapter_dir = str(adapter_dir)  # carried through to the runner

    # Run merge inline in executor (can take minutes)
    loop = asyncio.get_event_loop()

    async def _do_merge():
        merge_job.status = JobStatus.RUNNING
        import time
        merge_job.start_time = time.time()
        try:
            def _merge_blocking():
                from peft import PeftModel
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                base = AutoModelForCausalLM.from_pretrained(
                    job.config.base_model,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                model = PeftModel.from_pretrained(base, str(adapter_dir))
                merged = model.merge_and_unload()
                merged_dir.mkdir(parents=True, exist_ok=True)
                merged.save_pretrained(str(merged_dir))
                tokenizer = AutoTokenizer.from_pretrained(job.config.base_model, trust_remote_code=True)
                tokenizer.save_pretrained(str(merged_dir))

            await loop.run_in_executor(None, _merge_blocking)
            merge_job.status = JobStatus.COMPLETED
            merge_job.end_time = time.time()
            merge_job.progress = 100.0
        except Exception as exc:
            merge_job.status = JobStatus.FAILED
            merge_job.end_time = time.time()
            merge_job._log(f"Merge failed: {exc}")

    loop.create_task(_do_merge())

    return web.json_response({
        "merge_job_id": merge_job.job_id,
        "source_job_id": job_id,
        "adapter_dir": str(adapter_dir),
        "output_dir": merged_dir_name,
        "status": "running",
        "message": "Merge started. Poll GET /api/training/jobs/{merge_job_id} for status.",
    }, status=202)


async def handle_resume_job(request: web.Request) -> web.Response:
    """POST /api/training/jobs/:job_id/resume — resume training from a checkpoint.

    Creates a new job that resumes from the latest (or specified) checkpoint
    saved in the original job's output directory.
    """
    manager: TrainingManager = request.app["training_manager"]
    job_id = request.match_info["job_id"]

    job = manager.get_job(job_id)
    if job is None:
        return web.json_response({"error": f"Job not found: {job_id}"}, status=404)

    if job.status.value not in ("failed", "cancelled", "completed"):
        return web.json_response(
            {"error": f"Can only resume failed/cancelled/completed jobs. Status: {job.status.value}"},
            status=409,
        )

    try:
        body = await request.json()
    except Exception:
        body = {}

    # Find checkpoint directory in original output_dir
    output_dir = Path(job.config.output_dir) if job.config.output_dir else None
    if output_dir is None or not output_dir.exists():
        return web.json_response({"error": "No output directory found — nothing to resume from"}, status=404)

    # Find latest checkpoint (checkpoint-N dirs)
    checkpoint_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]) if d.name.split("-")[-1].isdigit() else 0,
        reverse=True,
    )
    if not checkpoint_dirs:
        return web.json_response({"error": "No checkpoints found in output directory"}, status=404)

    # Use specified checkpoint or latest
    checkpoint_name = body.get("checkpoint")
    if checkpoint_name:
        checkpoint_path = output_dir / checkpoint_name
        if not checkpoint_path.exists():
            return web.json_response(
                {"error": f"Checkpoint not found: {checkpoint_name}. Available: {[d.name for d in checkpoint_dirs]}"},
                status=404,
            )
    else:
        checkpoint_path = checkpoint_dirs[0]

    # Create a new job config with resume_from_checkpoint set
    from ainode.training.engine import TrainingConfig
    import dataclasses
    resume_config_dict = dataclasses.asdict(job.config)
    resume_config_dict["run_name"] = f"resume-{job_id}"
    resume_config_dict["description"] = f"Resumed from {checkpoint_path.name} of job {job_id}"
    # HF Trainer honours TRAINING_RESUME_FROM_CHECKPOINT — pass via env/config
    resume_config_dict["_resume_from_checkpoint"] = str(checkpoint_path)

    try:
        resume_config = TrainingConfig.from_dict(resume_config_dict)
    except Exception as exc:
        return web.json_response({"error": f"Failed to create resume config: {exc}"}, status=500)

    resume_job = manager.submit_job(resume_config)
    resume_job._resume_checkpoint = str(checkpoint_path)

    await manager.start_next()

    return web.json_response({
        "resume_job_id": resume_job.job_id,
        "source_job_id": job_id,
        "checkpoint": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path),
        "available_checkpoints": [d.name for d in checkpoint_dirs],
        "status": resume_job.status.value,
    }, status=201)


# ---------------------------------------------------------------------------
# Custom templates
# ---------------------------------------------------------------------------

_CUSTOM_TEMPLATES: list[dict] = []  # in-memory store; persisted to disk below


async def handle_save_template(request: web.Request) -> web.Response:
    """POST /api/training/templates — save a custom training template."""
    import uuid as _uuid
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    name = (body.get("name") or "").strip()
    if not name:
        return web.json_response({"error": "name is required"}, status=400)

    template = {
        "id": f"custom-{_uuid.uuid4().hex[:8]}",
        "name": name,
        "description": body.get("description", ""),
        "method": body.get("method", "lora"),
        "default_params": body.get("default_params", {}),
        "estimated_time": body.get("estimated_time", "varies"),
        "custom": True,
    }
    _CUSTOM_TEMPLATES.append(template)

    # Persist to disk alongside built-in templates
    from ainode.core.config import AINODE_HOME
    import json as _json
    templates_path = AINODE_HOME / "training" / "custom_templates.json"
    templates_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = _json.loads(templates_path.read_text()) if templates_path.exists() else []
        existing.append(template)
        templates_path.write_text(_json.dumps(existing, indent=2))
    except Exception:
        pass  # in-memory fallback is fine

    return web.json_response(template, status=201)
