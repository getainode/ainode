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
    app["autodata_runs"] = {}  # run_id -> status dict (in-memory; MVP)

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
    app.router.add_post("/api/training/autodata", handle_autodata_run)
    app.router.add_get("/api/training/autodata/{run_id}", handle_autodata_status)


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

    # Quantize jobs: default a clean on-disk output name, and refuse if this node
    # is already serving a model — quantization needs the node's full unified
    # memory, so running it against a live model OOMs (operator guardrail; pass
    # force=true to override).
    if body.get("method") == "quantize":
        if not body.get("out_slug"):
            base = (body.get("base_model") or "").rstrip("/").replace("/", "--")
            if base:
                body["out_slug"] = f"{base}-{(body.get('scheme') or 'awq').lower()}"
        if body.get("push_to_hf"):
            from ainode.models.hf_upload import resolve_hf_token, assert_write_scope
            tok = resolve_hf_token(request.app, body.get("hf_token"))
            try:
                assert_write_scope(tok)
            except Exception as exc:
                return web.json_response({"error": f"push_to_hf: {exc}"}, status=400)
            # Carry the resolved (write-preferred) token into the job so the
            # post-quant push uses it — NodeConfig.hf_token may be unset while the
            # write token lives in the Secrets store.
            body["hf_token"] = tok
        if not body.get("force"):
            instances = request.app.get("instances")
            engine = request.app.get("engine")
            cfg = request.app.get("config")
            busy = (instances is not None and not instances.is_empty()) or (
                engine is not None and getattr(cfg, "model", None)
            )
            if busy:
                return web.json_response({
                    "error": "This node is serving a model. Quantization needs the full "
                             "unified memory — unload all models on this node first "
                             "(or resubmit with force=true)."
                }, status=409)

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
        # "lora": validate() only allows lora/full/qlora/quantize, and method is
        # never read again — the merge itself runs via _do_merge() below, not
        # through the manager's normal _build_command() job-start path.
        method="lora",
        output_dir=merged_dir_name,
        run_name=f"merge-{job_id}",
        description=f"LoRA merge from job {job_id}",
        hf_token=job.config.hf_token,
    )

    merge_job = manager.submit_job(merge_config)
    merge_job._adapter_dir = str(adapter_dir)  # carried through to the runner
    # The merge container is spawned below via create_subprocess_exec (outside the
    # normal TrainingJob.start() path), so merge_job._process is never set and
    # stop() can't signal it. Register the container's deterministic name so a
    # cancel (DELETE) can `docker kill` it instead of no-op'ing.
    import os as _os
    if _os.environ.get("AINODE_IN_CONTAINER"):
        merge_job._container_name_override = f"ainode-merge-{merge_job.job_id}"

    # Run merge inline in executor (can take minutes)
    loop = asyncio.get_event_loop()

    async def _merge_in_container() -> None:
        """Slim orchestrator (no peft/torch): spawn a GPU container to merge,
        streaming its stdout so the AINODE_PROGRESS protocol keeps working."""
        from ainode.training.engine import build_merge_command
        cmd = build_merge_command(merge_job, job.config.base_model, adapter_dir, merged_dir, job.config.hf_token)
        merge_job._log("Merge command: " + " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            if line:
                merge_job._log(line)
                merge_job._parse_progress(line)
        rc = await proc.wait()
        if rc != 0:
            raise RuntimeError(f"merge container exited with code {rc}")

    async def _do_merge():
        import os
        import time
        # A cancel may already have landed while the job sat PENDING — don't
        # resurrect it into RUNNING and run the full merge anyway.
        if merge_job.status == JobStatus.CANCELLED:
            return
        merge_job.status = JobStatus.RUNNING
        merge_job.start_time = time.time()
        try:
            if os.environ.get("AINODE_IN_CONTAINER"):
                await _merge_in_container()
            else:
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
        except Exception as exc:
            # A cancel (DELETE) flips status to CANCELLED and `docker kill`s the
            # merge container, which makes _merge_in_container raise — don't
            # clobber the user's cancellation with FAILED.
            if merge_job.status == JobStatus.CANCELLED:
                merge_job._log("Merge cancelled")
            else:
                merge_job.status = JobStatus.FAILED
                merge_job.end_time = time.time()
                merge_job._log(f"Merge failed: {exc}")
            return
        # Success path — but a cancel may have landed right as the container
        # exited 0; honour it rather than overwriting CANCELLED with COMPLETED.
        if merge_job.status == JobStatus.CANCELLED:
            merge_job._log("Merge cancelled")
            return
        merge_job.status = JobStatus.COMPLETED
        merge_job.end_time = time.time()
        merge_job.progress = 100.0

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


# ---------------------------------------------------------------------------
# AutoData — Δ-filtered synthetic-data generation -> registered dataset
# ---------------------------------------------------------------------------
# Closes the loop: an AutoData run (pure-HTTP, no torch) writes ShareGPT JSONL,
# we register it with the DatasetManager, and the returned dataset_id drops
# straight into a training job (POST /api/training/jobs {dataset_id}).


async def handle_autodata_run(request: web.Request) -> web.Response:
    """POST /api/training/autodata — start a background AutoData run.

    Body: {config: {AutoData config}, meta?: bool, target_yield?, max_rounds?,
    name?, description?}. Returns 202 + run_id; poll GET /api/training/autodata/{run_id}.
    The run executes in a thread (it's blocking HTTP I/O) so the event loop never stalls.
    """
    import asyncio
    import uuid
    from pathlib import Path as _Path

    from ainode.training.autodata.core import AutoDataConfig

    dsm = request.app.get("dataset_manager")
    if dsm is None:
        return web.json_response({"error": "Dataset manager unavailable"}, status=503)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    config = body.get("config")
    if not isinstance(config, dict):
        return web.json_response({"error": "config (object) is required"}, status=400)
    try:
        cfg = AutoDataConfig.from_dict(config)
    except Exception as exc:
        return web.json_response({"error": f"Invalid AutoData config: {exc}"}, status=400)

    is_meta = bool(body.get("meta"))
    target_yield = float(body.get("target_yield", 30))
    max_rounds = int(body.get("max_rounds", 4))

    run_id = uuid.uuid4().hex[:12]
    # Write under the dataset store so add_local registers it in place (and the
    # resulting absolute path satisfies TrainingConfig's datasets-dir guard).
    out_path = _Path(dsm.root) / f"autodata-{run_id}.jsonl"
    cfg.out = str(out_path)
    name = (body.get("name") or f"autodata-{run_id}").strip()
    description = body.get("description") or "AutoData Δ-filtered synthetic dataset"

    runs = request.app["autodata_runs"]
    runs[run_id] = {
        "run_id": run_id, "status": "running", "meta": is_meta,
        "dataset_id": None, "report": None, "rounds": None,
        "out": str(out_path), "error": None,
    }

    loop = asyncio.get_event_loop()

    async def _do_run() -> None:
        try:
            def _blocking() -> dict:
                if is_meta:
                    from ainode.training.autodata.meta import meta_optimize
                    res = meta_optimize(cfg, target_yield=target_yield, max_rounds=max_rounds)
                    report = {"kept": len(res["dataset"]), "best_yield": res["best_yield"],
                              "rounds": len(res["rounds"])}
                    return {"report": report, "rounds": res["rounds"]}
                from ainode.training.autodata.core import run as _run
                res = _run(cfg)
                return {"report": res["report"], "rounds": None}

            result = await loop.run_in_executor(None, _blocking)
            ds = dsm.add_local(str(out_path), name=name, description=description) \
                if out_path.exists() else None
            runs[run_id].update(
                status="completed", dataset_id=(ds.id if ds else None),
                report=result["report"], rounds=result["rounds"],
            )
        except Exception as exc:  # noqa: BLE001 — surface any run failure to the poller
            runs[run_id].update(status="failed", error=str(exc))

    loop.create_task(_do_run())
    return web.json_response(
        {"run_id": run_id, "status": "running",
         "poll": f"/api/training/autodata/{run_id}"},
        status=202,
    )


async def handle_autodata_status(request: web.Request) -> web.Response:
    """GET /api/training/autodata/{run_id} — poll an AutoData run."""
    runs = request.app.get("autodata_runs", {})
    run = runs.get(request.match_info["run_id"])
    if run is None:
        return web.json_response({"error": "AutoData run not found"}, status=404)
    return web.json_response(run)
