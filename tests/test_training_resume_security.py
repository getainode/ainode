"""Resume requests cannot turn a training container into an arbitrary host mount."""

import json
import os
from pathlib import Path
import socket
import stat
from unittest.mock import AsyncMock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest
import pytest_asyncio

from ainode.training import engine
from ainode.training.api_routes import setup_training_routes
from ainode.training.engine import JobStatus, TrainingConfig, TrainingJob, TrainingManager


@pytest.fixture(autouse=True)
def isolate_training(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "AINODE_HOME", tmp_path)
    monkeypatch.setattr(engine, "JOBS_DIR", tmp_path / "training" / "jobs")
    monkeypatch.setitem(engine._DIR_OVERRIDES, "training", None)
    monkeypatch.setattr(engine, "_image_present", lambda image: False)
    monkeypatch.setenv("AINODE_HOST_HOME", "/host")
    monkeypatch.setenv("AINODE_NO_WHEEL_FETCH", "1")


@pytest_asyncio.fixture
async def client(monkeypatch):
    manager = TrainingManager()
    # Exercise the real HTTP parsing and queueing, without starting any process.
    monkeypatch.setattr(manager, "start_next", AsyncMock())
    app = web.Application()
    setup_training_routes(app, manager)
    async with TestClient(TestServer(app)) as client:
        yield client


def finished_job(manager):
    job = manager.submit_job(TrainingConfig(base_model="m", dataset_path="d.jsonl"))
    job.status = JobStatus.COMPLETED
    checkpoint = Path(job.config.output_dir) / "checkpoint-36"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}")
    manager._queue.clear()
    return job, checkpoint


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["jobs", "estimate"])
@pytest.mark.parametrize("field", ["_resume_from_checkpoint", "_future_internal"])
async def test_public_training_rejects_internal_fields(client, route, field):
    response = await client.post(f"/api/training/{route}", json={
        "base_model": "m", "dataset_path": "d.jsonl",
        field: "/var/run/docker.sock/checkpoint-1",
    })
    assert response.status == 400
    assert field in (await response.json())["error"]
    manager = client.app["training_manager"]
    assert manager.list_jobs() == []
    manager.start_next.assert_not_called()


@pytest.mark.asyncio
async def test_public_training_remains_open_without_auth(client):
    response = await client.post("/api/training/jobs", json={
        "base_model": "m", "dataset_path": "d.jsonl", "num_epochs": 2,
        "force": True,
    })
    assert response.status == 201
    assert (await response.json())["config"]["num_epochs"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", [
    "/var/run/docker.sock", "../checkpoint-36", "../../other/output/checkpoint-1",
    "./checkpoint-36", "checkpoint-36/..", "checkpoint-36/", "", [], 36,
])
async def test_resume_rejects_paths_instead_of_checkpoint_ids(client, selection):
    manager = client.app["training_manager"]
    job, _ = finished_job(manager)
    response = await client.post(f"/api/training/jobs/{job.job_id}/resume",
                                 json={"checkpoint": selection})
    assert response.status == 400
    assert len(manager.list_jobs()) == 1
    manager.start_next.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("absolute", [False, True])
async def test_resume_rejects_existing_absolute_and_escaping_checkpoint(client, absolute):
    manager = client.app["training_manager"]
    job, _ = finished_job(manager)
    _, other_checkpoint = finished_job(manager)
    selection = str(other_checkpoint) if absolute else os.path.relpath(
        other_checkpoint, job.config.output_dir,
    )
    response = await client.post(f"/api/training/jobs/{job.job_id}/resume",
                                 json={"checkpoint": selection})
    assert response.status == 400
    assert len(manager.list_jobs()) == 2
    manager.start_next.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_resume_rejects_symlinks_even_for_latest(client, tmp_path, explicit):
    manager = client.app["training_manager"]
    job, checkpoint = finished_job(manager)
    outside = tmp_path / "outside"
    outside.mkdir()
    (checkpoint / "trainer_state.json").unlink()
    checkpoint.rmdir()
    checkpoint.symlink_to(outside, target_is_directory=True)
    body = {"checkpoint": checkpoint.name} if explicit else {}
    response = await client.post(f"/api/training/jobs/{job.job_id}/resume", json=body)
    assert response.status == 400
    assert len(manager.list_jobs()) == 1


@pytest.mark.asyncio
async def test_resume_cannot_select_another_jobs_output(client):
    manager = client.app["training_manager"]
    job, _ = finished_job(manager)
    other, _ = finished_job(manager)
    job.config.output_dir = other.config.output_dir
    response = await client.post(f"/api/training/jobs/{job.job_id}/resume", json={})
    assert response.status == 400
    assert len(manager.list_jobs()) == 2


@pytest.mark.asyncio
async def test_resume_uses_a_discovered_id_and_keeps_output_separate(client):
    manager = client.app["training_manager"]
    job, checkpoint = finished_job(manager)
    response = await client.post(f"/api/training/jobs/{job.job_id}/resume",
                                 json={"checkpoint": checkpoint.name})
    assert response.status == 201
    resumed = manager.get_job((await response.json())["resume_job_id"])
    assert resumed.config._resume_from_checkpoint == str(checkpoint)
    assert resumed.config.output_dir != job.config.output_dir
    command = resumed._build_container_command()
    mounts = [command[i + 1] for i, arg in enumerate(command) if arg == "-v"]
    assert f"/host/training/jobs/{job.job_id}:/src:ro" in mounts
    cfg = json.loads((resumed._job_dir / "config.container.json").read_text())
    assert cfg["_resume_from_checkpoint"] == "/src/output/checkpoint-36"


@pytest.mark.asyncio
async def test_queued_resume_rechecks_source_at_mount_construction(client, tmp_path):
    manager = client.app["training_manager"]
    job, checkpoint = finished_job(manager)
    response = await client.post(f"/api/training/jobs/{job.job_id}/resume", json={})
    assert response.status == 201
    resumed = manager.get_job((await response.json())["resume_job_id"])
    (checkpoint / "trainer_state.json").unlink()
    checkpoint.rmdir()
    checkpoint.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(RuntimeError, match="Unsafe resume checkpoint"):
        resumed._build_container_command()


def test_docker_socket_can_never_be_a_resume_mount():
    # No socket is contacted. The original fallback rendered the socket itself
    # as a -v source without checking whether this checkpoint existed.
    config = TrainingConfig.from_dict({
        "base_model": "m", "dataset_path": "d.jsonl",
        "_resume_from_checkpoint": "/var/run/docker.sock/checkpoint-1",
    })
    job = TrainingJob(config)
    with pytest.raises(RuntimeError, match="Unsafe resume checkpoint"):
        job._build_container_command()


@pytest.mark.parametrize("kind", ["socket", "fifo", "symlink", "char", "block"])
def test_resume_rejects_special_files_anywhere_in_mounted_job(tmp_path, monkeypatch, kind):
    job, checkpoint = finished_job(TrainingManager())
    special = job._job_dir / "special"
    sock = None
    if kind == "socket":
        sock = socket.socket(socket.AF_UNIX)
        sock.bind(str(special))
    elif kind == "fifo":
        os.mkfifo(special)
    elif kind == "symlink":
        special.symlink_to(checkpoint, target_is_directory=True)
    else:
        special.touch()
        real_lstat = Path.lstat

        def device_lstat(path, *args, **kwargs):
            result = real_lstat(path, *args, **kwargs)
            if path == special:
                values = list(result)
                values[0] = stat.S_IFCHR if kind == "char" else stat.S_IFBLK
                return os.stat_result(values)
            return result

        monkeypatch.setattr(Path, "lstat", device_lstat)
    try:
        with pytest.raises(RuntimeError, match="symlink or special file"):
            engine._resume_mount_for(checkpoint)
    finally:
        if sock is not None:
            sock.close()


@pytest.mark.parametrize("part", ["source", "output", "checkpoint"])
def test_resume_rejects_symlink_components_even_inside_jobs(part):
    job, checkpoint = finished_job(TrainingManager())
    path = {"source": job._job_dir, "output": checkpoint.parent, "checkpoint": checkpoint}[part]
    target = path.with_name(path.name + "-real")
    path.rename(target)
    path.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="real directories"):
        engine._resume_mount_for(checkpoint)


def test_resume_rejects_lexical_escape_to_another_job():
    first, _ = finished_job(TrainingManager())
    _, second_checkpoint = finished_job(TrainingManager())
    escaped = first._job_dir / ".." / second_checkpoint.relative_to(engine.jobs_dir())
    with pytest.raises(RuntimeError, match="Unsafe resume checkpoint"):
        engine._resume_mount_for(escaped)
