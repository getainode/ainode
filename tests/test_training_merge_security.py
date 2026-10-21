"""Merge paths must never expand a model container's access to the node home."""

import asyncio
import json
import os
from pathlib import Path
import socket
from unittest.mock import AsyncMock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest
import pytest_asyncio

from ainode.training import engine
from ainode.training.api_routes import setup_training_routes
from ainode.training.engine import JobStatus, TrainingConfig, TrainingJob, TrainingManager
from ainode.training.engine import build_merge_command


@pytest.fixture(autouse=True)
def isolate_training(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "AINODE_HOME", tmp_path)
    monkeypatch.setattr(engine, "JOBS_DIR", tmp_path / "training" / "jobs")
    monkeypatch.setitem(engine._DIR_OVERRIDES, "training", None)
    monkeypatch.setitem(engine._DIR_OVERRIDES, "datasets", None)
    monkeypatch.setattr(engine, "_image_present", lambda image: False)
    monkeypatch.setenv("AINODE_HOST_HOME", "/host")
    monkeypatch.setenv("AINODE_NO_WHEEL_FETCH", "1")


def merge_paths():
    job = TrainingJob(TrainingConfig(base_model="m", dataset_path="__merge__"))
    adapter = job._job_dir / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    return job, adapter, job._job_dir / "merged"


@pytest.mark.parametrize("suffix", ["-escape", "", " ", "/../training-escape"])
def test_output_validation_rejects_prefix_collision_and_root(suffix):
    config = TrainingConfig(base_model="m", dataset_path="d",
                            output_dir=str(engine.training_dir()) + suffix)
    assert any("output_dir" in error for error in config.validate())


@pytest.mark.parametrize("suffix", ["-escape", "", " ", "/../training-escape"])
def test_merge_builder_rejects_prefix_collision_and_root(suffix):
    job, adapter, _ = merge_paths()
    output = Path(str(engine.training_dir()) + suffix)
    with pytest.raises(ValueError, match="Unsafe training directory"):
        engine.build_merge_command(job, "m", adapter, output)
    assert not (job._job_dir / "merge_config.json").exists()


def test_merge_binds_only_the_output_and_keeps_secrets_outside(tmp_path):
    job, adapter, output = merge_paths()
    (tmp_path / "auth.json").write_text('{"synthetic": true}')
    command = engine.build_merge_command(job, "m", adapter, output)
    mounts = [command[i + 1] for i, arg in enumerate(command) if arg == "-v"]
    assert f"/host/training/jobs/{job.job_id}/merged:/out" in mounts
    assert "/host:/out" not in mounts
    assert f"/host/training/jobs/{job.job_id}:/out" not in mounts
    config = json.loads((job._job_dir / "merge_config.json").read_text())
    assert config["output_dir"] == "/out"
    assert output.is_dir()


@pytest.mark.parametrize("target", ["output", "adapter", "ancestor", "root"])
def test_merge_rechecks_symlinks_before_binding(tmp_path, target):
    job, adapter, output = merge_paths()
    assert not TrainingConfig(base_model="m", dataset_path="d",
                              output_dir=str(output)).validate()
    if target == "output":
        output.symlink_to(tmp_path, target_is_directory=True)
    elif target == "adapter":
        (adapter / "config.json").symlink_to(tmp_path / "auth.json")
    elif target == "ancestor":
        original = job._job_dir / "real"
        original.mkdir()
        link = job._job_dir / "link"
        link.symlink_to(original, target_is_directory=True)
        output = link / "merged"
    else:
        root = engine.training_dir()
        moved = tmp_path / "real-training"
        root.rename(moved)
        root.symlink_to(moved, target_is_directory=True)
    with pytest.raises((ValueError, RuntimeError), match="Unsafe .*directory"):
        engine.build_merge_command(job, "m", adapter, output)


@pytest.mark.parametrize("tree", ["output", "adapter"])
@pytest.mark.parametrize("kind", ["fifo", "socket", "file-root"])
def test_merge_rejects_special_files_in_bound_trees(tree, kind, monkeypatch):
    job, adapter, output = merge_paths()
    path = output if tree == "output" else adapter
    if kind == "file-root":
        if path.exists():
            (path / "adapter_config.json").unlink()
            path.rmdir()
        path.write_text("not a directory")
    else:
        path.mkdir(exist_ok=True)
    with socket.socket(socket.AF_UNIX) as sock:
        if kind == "fifo":
            os.mkfifo(path / "pipe")
        elif kind == "socket":
            monkeypatch.chdir(path)
            sock.bind("sock")
        with pytest.raises((ValueError, RuntimeError), match="Unsafe .*directory"):
            engine.build_merge_command(job, "m", adapter, output)


def test_relative_and_operator_training_roots_remain_supported(tmp_path, monkeypatch):
    root = tmp_path / "operator-runs"
    monkeypatch.setitem(engine._DIR_OVERRIDES, "training", root)
    cfg = TrainingConfig(base_model="m", dataset_path="d", output_dir="custom/merged")
    assert cfg.validate() == []
    job = TrainingJob(cfg)
    assert Path(job.config.output_dir) == root / "custom" / "merged"
    adapter = job._job_dir / "adapter"
    adapter.mkdir()
    command = engine.build_merge_command(job, "m", adapter, Path(job.config.output_dir))
    assert "/host/operator-runs/custom/merged:/out" in command


def test_merge_rechecks_after_blocking_preparation(tmp_path, monkeypatch):
    job, adapter, output = merge_paths()

    def replace_output(*args, **kwargs):
        output.rmdir()
        output.symlink_to(tmp_path, target_is_directory=True)
        return "true"

    monkeypatch.setattr(engine, "_pip_install_step", replace_output)
    with pytest.raises((ValueError, RuntimeError), match="Unsafe .*directory"):
        engine.build_merge_command(job, "m", adapter, output)


def test_dataset_prefix_collision_is_rejected(tmp_path):
    cfg = TrainingConfig(base_model="m", dataset_path=str(tmp_path / "datasets-escape/data"))
    assert any("dataset_path" in error for error in cfg.validate())


@pytest.mark.parametrize("builder", ["merge", "training", "quantize"])
def test_job_bind_cannot_follow_a_symlink_to_node_home(tmp_path, builder):
    job, adapter, output = merge_paths()
    # Keep both caller-selected paths safe while replacing only the /job source.
    adapter = engine.training_dir() / "safe-adapter"
    adapter.mkdir()
    output = engine.training_dir() / "safe-output"
    job._job_dir.rename(job._job_dir.with_name("moved-job"))
    job._job_dir.symlink_to("../../..", target_is_directory=True)
    with pytest.raises(RuntimeError, match="Unsafe job directory"):
        if builder == "merge":
            engine.build_merge_command(job, "m", adapter, output)
        elif builder == "training":
            job._build_container_command()
        else:
            job._build_quant_command(job._job_dir / "config.json")
    assert not (tmp_path / "_run_merge.py").exists()
    assert not (tmp_path / "_run_training.py").exists()
    assert not (tmp_path / "config.container.json").exists()


@pytest.mark.parametrize("builder", ["merge", "training", "quantize"])
def test_job_bind_rejects_special_files(builder):
    job, adapter, output = merge_paths()
    os.mkfifo(job._job_dir / "pipe")
    with pytest.raises(RuntimeError, match="Unsafe job directory"):
        if builder == "merge":
            engine.build_merge_command(job, "m", adapter, output)
        elif builder == "training":
            job._build_container_command()
        else:
            job._build_quant_command(job._job_dir / "config.json")


@pytest_asyncio.fixture
async def client(monkeypatch):
    manager = TrainingManager()
    monkeypatch.setattr(manager, "start_next", AsyncMock())
    monkeypatch.setenv("AINODE_IN_CONTAINER", "1")
    monkeypatch.setattr(TrainingJob, "_assert_job_image", lambda self: None)

    def refuse_launch(*args):
        raise RuntimeError("No Docker in this test")

    monkeypatch.setattr(engine, "build_merge_command", refuse_launch)
    app = web.Application()
    setup_training_routes(app, manager)
    async with TestClient(TestServer(app)) as client:
        yield client


def completed_job(client):
    manager = client.app["training_manager"]
    job = manager.submit_job(TrainingConfig(base_model="m", dataset_path="d"))
    Path(job.config.output_dir).mkdir()
    job.status = JobStatus.COMPLETED
    manager._queue.clear()
    return job


@pytest.mark.asyncio
@pytest.mark.parametrize("body", ["collision", "root", [], {"output_dir": 7}, {"output_dir": ""}])
async def test_merge_request_rejects_unsafe_output_before_queueing(client, body):
    source = completed_job(client)
    if body == "collision":
        body = {"output_dir": str(engine.training_dir()) + "-escape"}
    elif body == "root":
        body = {"output_dir": str(engine.training_dir())}
    response = await client.post(f"/api/training/jobs/{source.job_id}/merge", json=body)
    assert response.status == 400
    assert len(client.app["training_manager"].list_jobs()) == 1


@pytest.mark.asyncio
async def test_default_merge_output_belongs_to_new_job_without_auth(client):
    source = completed_job(client)
    response = await client.post(f"/api/training/jobs/{source.job_id}/merge", json={})
    assert response.status == 202
    body = await response.json()
    merged = client.app["training_manager"].get_job(body["merge_job_id"])
    assert Path(body["output_dir"]) == merged._job_dir / "output"
    assert source.job_id not in body["output_dir"]
    await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_merge_rechecks_after_image_preflight(client, tmp_path, monkeypatch):
    source = completed_job(client)
    output = source._job_dir / "merged"
    output.mkdir()

    def replace_output(self):
        output.rmdir()
        output.symlink_to(tmp_path, target_is_directory=True)

    monkeypatch.setattr(TrainingJob, "_assert_job_image", replace_output)
    # Exercise the real builder and record any attempted process launch.
    monkeypatch.setattr(engine, "build_merge_command", build_merge_command)
    launch = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    response = await client.post(f"/api/training/jobs/{source.job_id}/merge",
                                 json={"output_dir": str(output)})
    assert response.status == 202
    merged = client.app["training_manager"].get_job((await response.json())["merge_job_id"])
    for _ in range(100):
        if merged.status == JobStatus.FAILED:
            break
        await asyncio.sleep(0.01)
    assert merged.status == JobStatus.FAILED
    assert any("Unsafe training directory" in line for line in merged.logs)
    launch.assert_not_awaited()
