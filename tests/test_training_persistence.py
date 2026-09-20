"""Training jobs survive a restart, resume for real, and the queue moves on its own.

Three defects the product shipped until 0.5.27, each with its own section here:

* ``TrainingManager._jobs`` was memory-only, so every restart emptied the Runs
  table, zeroed the stats tiles and 404'd merge, resume, logs and artifact
  download for every job that came before, while ten real job dirs sat on disk
  (issue #191).
* Resume handed the container an orchestrator checkpoint path with the source job
  dir not mounted, so HF answered "Can't find a valid checkpoint", and the resumed
  run reused the source job's output dir (issue #193).
* The queue only advanced from ``POST /api/training/jobs`` and the resume route,
  so a second queued job waited for a human to submit a third.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import ainode.training.engine as engine
from ainode.training.api_routes import setup_training_routes
from ainode.training.engine import (
    JobStatus,
    TrainingConfig,
    TrainingJob,
    TrainingManager,
)


@pytest.fixture(autouse=True)
def isolate_jobs_dir(tmp_path, monkeypatch):
    """Job dirs under a tmp AINODE_HOME, never the developer's real one."""
    monkeypatch.setattr(engine, "AINODE_HOME", tmp_path)
    monkeypatch.setattr(engine, "JOBS_DIR", tmp_path / "training" / "jobs")
    return tmp_path


@pytest.fixture(autouse=True)
def no_local_images(monkeypatch):
    """No candidate job image on this machine, so nothing shells out to docker."""
    monkeypatch.setattr(engine, "_image_present", lambda image: False)


def _job_dir(name: str) -> Path:
    path = engine.JOBS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_status(name: str, **overrides) -> Path:
    """Write a status.json of the shape the engine writes."""
    job_dir = _job_dir(name)
    payload = {
        "job_id": name,
        "status": "completed",
        "progress": 100.0,
        "current_epoch": 1,
        "current_loss": 2.13,
        "start_time": 1000.0,
        "end_time": 1100.0,
        "note": None,
        "restored": False,
        "written_at": 1100.0,
        "schema": 1,
        "config": {
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "dataset_path": "proof40.jsonl",
            "method": "lora",
            "output_dir": str(job_dir / "output"),
        },
    }
    payload.update(overrides)
    (job_dir / "status.json").write_text(json.dumps(payload, indent=2))
    return job_dir


def _write_legacy(name: str, *, adapter: bool, config: dict | None = None) -> Path:
    """A job dir as AINode 0.5.26 and earlier left it: a config, no status file."""
    job_dir = _job_dir(name)
    data = {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset_path": "proof40.jsonl",
        "method": "lora",
        "output_dir": str(job_dir / "output"),
    }
    data.update(config or {})
    (job_dir / "config.json").write_text(json.dumps(data, indent=2))
    out = job_dir / "output"
    out.mkdir(parents=True, exist_ok=True)
    if adapter:
        (out / "adapter_model.safetensors").write_bytes(b"weights")
        (out / "adapter_config.json").write_text("{}")
    return job_dir


# =============================================================================
# Rebuilding the registry from disk (issue #191)
# =============================================================================


class TestRegistryRebuild:

    def test_a_recorded_job_comes_back_as_recorded(self):
        _write_status("aaaa11112222")
        jobs = engine.load_jobs_from_disk()
        assert [j.job_id for j in jobs] == ["aaaa11112222"]
        job = jobs[0]
        assert job.status == JobStatus.COMPLETED
        assert job.progress == 100.0
        assert job.current_loss == 2.13
        assert job.start_time == 1000.0
        assert job.end_time == 1100.0
        assert job.config.base_model == "Qwen/Qwen2.5-0.5B-Instruct"
        # A reader must be able to tell a rebuilt record from a live one.
        assert job.restored is True
        assert job.get_status()["restored"] is True

    def test_a_job_that_was_running_comes_back_failed(self):
        """Its process died with the restart. A phantom RUNNING job also blocks
        the queue for every job submitted after it."""
        _write_status("bbbb11112222", status="running", progress=42.0, end_time=None)
        job = engine.load_jobs_from_disk()[0]
        assert job.status == JobStatus.FAILED
        assert "did not survive the restart" in job.note
        # And the corrected status is on disk, so the next restart does not guess.
        on_disk = json.loads((engine.JOBS_DIR / "bbbb11112222" / "status.json").read_text())
        assert on_disk["status"] == "failed"
        assert on_disk["note"]

    def test_a_pending_job_also_comes_back_failed(self):
        _write_status("cccc11112222", status="pending", start_time=None, end_time=None)
        job = engine.load_jobs_from_disk()[0]
        assert job.status == JobStatus.FAILED

    def test_a_legacy_dir_with_an_adapter_is_reported_completed(self):
        """The only LoRA run that ever finished is on Spark-4 with no status file."""
        _write_legacy("dddd11112222", adapter=True)
        job = engine.load_jobs_from_disk()[0]
        assert job.status == JobStatus.COMPLETED
        assert "no status file" in job.note
        assert "adapter_model.safetensors" in job.note
        # A duration nobody recorded is not invented: stats() counts GPU hours
        # from start_time, and this job has none.
        assert job.start_time is None
        assert job.end_time is not None

    def test_a_legacy_dir_with_no_weights_is_reported_failed(self):
        _write_legacy("eeee11112222", adapter=False)
        job = engine.load_jobs_from_disk()[0]
        assert job.status == JobStatus.FAILED
        assert "no weights" in job.note

    def test_a_legacy_dir_with_only_a_checkpoint_still_counts(self):
        job_dir = _write_legacy("ffff11112222", adapter=False)
        checkpoint = job_dir / "output" / "checkpoint-36"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_bytes(b"w")
        job = engine.load_jobs_from_disk()[0]
        assert job.status == JobStatus.COMPLETED

    def test_an_empty_directory_is_not_a_job(self):
        """6,000 empty job dirs accumulated from the suite before 0.5.26. None of
        them is a run, and resurrecting them as failures would be invented history."""
        _job_dir("0000empty000")
        (engine.JOBS_DIR / "0000empty000" / "hf.env").write_text("")
        assert engine.load_jobs_from_disk() == []

    def test_an_unparseable_status_file_falls_back_to_the_config(self):
        job_dir = _write_legacy("1111broken11", adapter=True)
        (job_dir / "status.json").write_text("{not json")
        job = engine.load_jobs_from_disk()[0]
        assert job.status == JobStatus.COMPLETED

    def test_a_legacy_merge_job_with_nothing_in_its_dir_says_why(self):
        """A merge before 0.5.27 wrote into the SOURCE run's dir and recorded only
        a container path, so its own dir cannot show the merged model. The note has
        to say that rather than let a reader read "failed" as "the merge broke"."""
        job_dir = _job_dir("6666mergefail")
        (job_dir / "merge_config.json").write_text(json.dumps({
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "adapter_dir": "/adapter",
            "output_dir": "/out/merged",
        }))
        job = engine.load_jobs_from_disk()[0]
        assert job.status == JobStatus.FAILED
        assert "merge job" in job.note
        assert "source run's directory" in job.note

    def test_a_merge_job_dir_is_rebuilt_from_its_merge_config(self):
        """A merge job writes merge_config.json and never a config.json."""
        job_dir = _job_dir("2222merge222")
        merged = job_dir / "merged"
        merged.mkdir()
        (merged / "config.json").write_text("{}")
        (job_dir / "merge_config.json").write_text(json.dumps({
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "adapter_dir": "/adapter",
            "output_dir": "/out/merged",
        }))
        job = engine.load_jobs_from_disk()[0]
        assert job.status == JobStatus.COMPLETED
        assert job.config.dataset_path == "__merge__"

    def test_a_legacy_quantize_job_is_judged_by_its_checkpoint_in_the_models_store(
        self, isolate_jobs_dir
    ):
        """A quantize job writes into ~/.ainode/models/<out_slug>, not its job dir,
        so looking only in the job dir would report every one of them failed. Six
        of the ten real job dirs on the fleet are quantize jobs."""
        _write_legacy("3333quant333", adapter=False, config={
            "method": "quantize", "dataset_path": "", "scheme": "awq",
            "out_slug": "qwen--qwen2.5-0.5b-instruct-awq",
        })
        checkpoint = isolate_jobs_dir / "models" / "qwen--qwen2.5-0.5b-instruct-awq"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}")

        job = engine.load_jobs_from_disk()[0]
        assert job.status == JobStatus.COMPLETED
        assert "awq" in job.note

    def test_a_legacy_quantize_job_with_no_checkpoint_is_failed(self):
        _write_legacy("4444quant444", adapter=False, config={
            "method": "quantize", "dataset_path": "", "scheme": "nvfp4",
            "out_slug": "never-written",
        })
        assert engine.load_jobs_from_disk()[0].status == JobStatus.FAILED

    def test_a_rebuilt_verdict_is_not_guessed_again_next_time(self):
        """The first rebuild writes the record, so the second reads it."""
        _write_legacy("5555stable55", adapter=True)
        first = engine.load_jobs_from_disk()[0]
        assert "no status file" in first.note
        second = engine.load_jobs_from_disk()[0]
        assert second.status == first.status
        assert second.note == first.note

    def test_jobs_come_back_oldest_first(self):
        _write_status("4444late4444", start_time=5000.0, end_time=5100.0)
        _write_status("5555early555", start_time=100.0, end_time=200.0)
        assert [j.job_id for j in engine.load_jobs_from_disk()] == [
            "5555early555", "4444late4444",
        ]


class TestManagerRehydration:

    def test_the_manager_rebuilds_at_construction(self):
        _write_status("6666aaaa6666")
        _write_legacy("7777bbbb7777", adapter=True)
        manager = TrainingManager()
        assert set(manager._jobs) == {"6666aaaa6666", "7777bbbb7777"}
        # The empty Runs table and the zeroed tiles were the same bug.
        stats = manager.stats()
        assert stats["total"] == 2
        assert stats["completed"] == 2
        assert stats["running"] == 0

    def test_rehydration_can_be_turned_off(self):
        _write_status("8888cccc8888")
        assert TrainingManager(rehydrate=False)._jobs == {}

    def test_a_rebuilt_job_is_not_started_and_holds_no_slot(self):
        _write_status("9999dddd9999", status="running", end_time=None)
        manager = TrainingManager()
        assert manager._active_job_id is None
        assert manager.queue_size == 0
        assert manager._jobs["9999dddd9999"].status == JobStatus.FAILED


class TestStatusFileIsWrittenAtEveryTransition:

    def test_submit_writes_a_pending_record(self):
        manager = TrainingManager()
        job = manager.submit_job(TrainingConfig(base_model="m", dataset_path="d.jsonl"))
        data = json.loads((job._job_dir / "status.json").read_text())
        assert data["status"] == "pending"
        assert data["job_id"] == job.job_id

    @pytest.mark.asyncio
    async def test_running_then_completed_are_both_recorded(self, monkeypatch):
        monkeypatch.setattr(
            TrainingJob, "_build_command",
            lambda self, config_path: [sys.executable, "-c", "pass"],
        )
        manager = TrainingManager()
        job = manager.submit_job(TrainingConfig(base_model="m", dataset_path="d.jsonl"))
        await manager.start_next()
        recorded = json.loads((job._job_dir / "status.json").read_text())
        assert recorded["status"] == "running"
        assert recorded["start_time"]

        await asyncio.wait_for(job._monitor_task, timeout=30)
        recorded = json.loads((job._job_dir / "status.json").read_text())
        assert recorded["status"] == "completed"
        assert recorded["end_time"]
        assert recorded["progress"] == 100.0

    @pytest.mark.asyncio
    async def test_a_failed_start_is_recorded(self, monkeypatch):
        def boom(self, config_path):
            raise RuntimeError("no launch path for this job")

        monkeypatch.setattr(TrainingJob, "_build_command", boom)
        manager = TrainingManager()
        job = manager.submit_job(TrainingConfig(base_model="m", dataset_path="d.jsonl"))
        with pytest.raises(RuntimeError):
            await manager.start_next()
        recorded = json.loads((job._job_dir / "status.json").read_text())
        assert recorded["status"] == "failed"

    def test_the_hf_token_never_reaches_the_status_file(self):
        """status.json lives in a directory the job API reads, same as config.json."""
        manager = TrainingManager()
        job = manager.submit_job(TrainingConfig(
            base_model="m", dataset_path="d.jsonl", hf_token="hf_secret_value",
        ))
        text = (job._job_dir / "status.json").read_text()
        assert "hf_secret_value" not in text
        assert "hf_token" not in json.loads(text)["config"]
        # And the rebuilt job does not come back with a masked string as a token.
        assert engine.load_job_from_dir(job._job_dir).config.hf_token is None

    def test_a_status_file_that_cannot_be_written_does_not_take_the_job_down(
        self, monkeypatch
    ):
        job = TrainingJob(TrainingConfig(base_model="m", dataset_path="d.jsonl"))

        def deny(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(Path, "write_text", deny)
        job.status = JobStatus.RUNNING  # must not raise
        assert job.status == JobStatus.RUNNING
        assert any("could not write status.json" in line for line in job.logs)


# =============================================================================
# The API serves a job that outlived the process that ran it
# =============================================================================


@pytest.fixture
def rebuilt_app():
    app = web.Application()
    setup_training_routes(app, TrainingManager())
    return app


@pytest_asyncio.fixture
async def rebuilt_client(rebuilt_app):
    async with TestClient(TestServer(rebuilt_app)) as client:
        yield client


class TestApiAfterARestart:

    @pytest.mark.asyncio
    async def test_a_prior_job_is_listed_with_its_artifacts(self, rebuilt_client):
        """Before 0.5.27 all three of these were 404 or [] after a restart."""
        listing = await (await rebuilt_client.get("/api/training/jobs")).json()
        assert [j["job_id"] for j in listing["jobs"]] == ["aaaa11112222"]

        detail = await rebuilt_client.get("/api/training/jobs/aaaa11112222")
        assert detail.status == 200
        assert (await detail.json())["status"] == "completed"

        logs = await rebuilt_client.get("/api/training/jobs/aaaa11112222/logs")
        assert logs.status == 200
        assert any("restored from" in line for line in (await logs.json())["logs"])

        output = await rebuilt_client.get("/api/training/jobs/aaaa11112222/output")
        assert output.status == 200
        names = [f["name"] for f in (await output.json())["files"]]
        assert "adapter_model.safetensors" in names

    @pytest.fixture(autouse=True)
    def _a_finished_job_on_disk(self, isolate_jobs_dir):
        job_dir = _write_status("aaaa11112222")
        out = job_dir / "output"
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_model.safetensors").write_bytes(b"weights")

    @pytest.mark.asyncio
    async def test_the_stats_tiles_count_the_rebuilt_jobs(self, rebuilt_client):
        stats = await (await rebuilt_client.get("/api/training/stats")).json()
        assert stats["total"] == 1
        assert stats["completed"] == 1


# =============================================================================
# The queue advances on its own when a job exits
# =============================================================================


class TestQueueAdvance:

    @pytest.mark.asyncio
    async def test_the_next_job_starts_when_the_first_one_exits(self, monkeypatch):
        monkeypatch.setattr(
            TrainingJob, "_build_command",
            lambda self, config_path: [sys.executable, "-c", "pass"],
        )
        manager = TrainingManager()
        first = manager.submit_job(TrainingConfig(base_model="m", dataset_path="a.jsonl"))
        second = manager.submit_job(TrainingConfig(base_model="m", dataset_path="b.jsonl"))

        await manager.start_next()
        assert manager._active_job_id == first.job_id
        assert second.status == JobStatus.PENDING

        # Nothing else is submitted: the monitor is what moves the queue.
        await asyncio.wait_for(first._monitor_task, timeout=30)
        deadline = time.monotonic() + 30
        while second.status == JobStatus.PENDING and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

        assert first.status == JobStatus.COMPLETED
        assert second.status in (JobStatus.RUNNING, JobStatus.COMPLETED)
        assert manager._active_job_id == second.job_id
        assert manager.queue_size == 0
        if second._monitor_task is not None:
            await asyncio.wait_for(second._monitor_task, timeout=30)

    @pytest.mark.asyncio
    async def test_a_next_job_that_cannot_start_is_reported_not_raised(self, monkeypatch):
        """The monitor task must survive it: the first job really did finish."""
        commands = {}

        def build(self, config_path):
            if self.config.dataset_path == "b.jsonl":
                raise RuntimeError("no launch path for this job")
            commands[self.job_id] = True
            return [sys.executable, "-c", "pass"]

        monkeypatch.setattr(TrainingJob, "_build_command", build)
        manager = TrainingManager()
        first = manager.submit_job(TrainingConfig(base_model="m", dataset_path="a.jsonl"))
        second = manager.submit_job(TrainingConfig(base_model="m", dataset_path="b.jsonl"))

        await manager.start_next()
        await asyncio.wait_for(first._monitor_task, timeout=30)
        deadline = time.monotonic() + 30
        while second.status == JobStatus.PENDING and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

        assert first.status == JobStatus.COMPLETED
        assert second.status == JobStatus.FAILED
        assert any("could not start the next queued job" in line for line in first.logs)
        assert manager._active_job_id is None


# =============================================================================
# Resume (issue #193)
# =============================================================================


class TestResumeMount:

    def test_the_source_job_dir_is_mounted_and_the_path_rewritten(self, monkeypatch):
        """The checkpoint belongs to another job whose dir was never mounted, and
        its orchestrator path meant nothing inside the container."""
        monkeypatch.setenv("AINODE_IN_CONTAINER", "1")
        monkeypatch.setenv("AINODE_HOST_HOME", "/host")
        monkeypatch.setenv("AINODE_NO_WHEEL_FETCH", "1")
        monkeypatch.setattr(engine, "AINODE_HOST_HOME", "/host", raising=False)

        source = _job_dir("aaaabbbbcccc")
        checkpoint = source / "output" / "checkpoint-36"
        checkpoint.mkdir(parents=True)

        job = TrainingJob(TrainingConfig(
            base_model="m", dataset_path="d.jsonl",
            _resume_from_checkpoint=str(checkpoint),
        ))
        cmd = job._build_container_command()

        mounts = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-v"]
        assert any(m.endswith(":/src:ro") for m in mounts), mounts
        # The SOURCE JOB dir, not the checkpoint dir: a checkpoint is resumed
        # together with the sibling files of its run.
        src_mount = next(m for m in mounts if m.endswith(":/src:ro"))
        assert src_mount.split(":")[0].endswith("aaaabbbbcccc")
        # Read-only, and the new job's own dir is still the writable /job mount.
        assert any(m.endswith(":/job") for m in mounts)

        config = json.loads((job._job_dir / "config.container.json").read_text())
        assert config["_resume_from_checkpoint"] == "/src/output/checkpoint-36"

    def test_no_resume_means_no_extra_mount(self, monkeypatch):
        monkeypatch.setenv("AINODE_IN_CONTAINER", "1")
        monkeypatch.setenv("AINODE_HOST_HOME", "/host")
        monkeypatch.setenv("AINODE_NO_WHEEL_FETCH", "1")
        job = TrainingJob(TrainingConfig(base_model="m", dataset_path="d.jsonl"))
        cmd = job._build_container_command()
        assert not any(arg.endswith(":/src:ro") for arg in cmd)

    def test_a_checkpoint_outside_the_jobs_tree_mounts_its_own_parent(self, tmp_path):
        checkpoint = tmp_path / "elsewhere" / "checkpoint-9"
        source, container_path = engine._resume_mount_for(checkpoint)
        assert source == checkpoint.parent
        assert container_path == "/src/checkpoint-9"

    def test_a_checkpoint_in_a_job_dir_maps_through_the_job_dir(self):
        checkpoint = engine.JOBS_DIR / "abcdef123456" / "output" / "checkpoint-12"
        source, container_path = engine._resume_mount_for(checkpoint)
        assert source == engine.JOBS_DIR / "abcdef123456"
        assert container_path == "/src/output/checkpoint-12"


class TestResumeApi:

    @pytest_asyncio.fixture
    async def client(self, monkeypatch):
        monkeypatch.setattr(
            TrainingJob, "_build_command",
            lambda self, config_path: [sys.executable, "-c", "pass"],
        )
        app = web.Application()
        setup_training_routes(app, TrainingManager())
        async with TestClient(TestServer(app)) as client:
            yield client

    @staticmethod
    def _finished_job_with_a_checkpoint(manager: TrainingManager) -> TrainingJob:
        job = manager.submit_job(TrainingConfig(base_model="m", dataset_path="d.jsonl"))
        job.status = JobStatus.COMPLETED
        checkpoint = Path(job.config.output_dir) / "checkpoint-36"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text("{}")
        manager._queue.clear()
        return job

    @pytest.mark.asyncio
    async def test_the_resumed_job_gets_its_own_output_dir(self, client):
        manager: TrainingManager = client.app["training_manager"]
        source = self._finished_job_with_a_checkpoint(manager)

        resp = await client.post(f"/api/training/jobs/{source.job_id}/resume")
        assert resp.status == 201
        body = await resp.json()
        resumed = manager.get_job(body["resume_job_id"])

        # Writing into the source job's output dir (what this did until 0.5.27)
        # left neither run's output meaning anything.
        assert resumed.config.output_dir != source.config.output_dir
        assert resumed.job_id in resumed.config.output_dir
        assert body["output_dir"] == resumed.config.output_dir
        assert body["checkpoint"] == "checkpoint-36"
        assert resumed.config._resume_from_checkpoint == str(
            Path(source.config.output_dir) / "checkpoint-36"
        )
        if resumed._monitor_task is not None:
            await asyncio.wait_for(resumed._monitor_task, timeout=30)

    @pytest.mark.asyncio
    async def test_a_resume_the_engine_refuses_answers_400(self, client, monkeypatch):
        manager: TrainingManager = client.app["training_manager"]
        source = self._finished_job_with_a_checkpoint(manager)

        def boom(self, config_path):
            raise RuntimeError("no launch path for this job")

        monkeypatch.setattr(TrainingJob, "_build_command", boom)
        resp = await client.post(f"/api/training/jobs/{source.job_id}/resume")
        assert resp.status == 400
        body = await resp.json()
        assert "no launch path" in body["error"]
        assert manager.get_job(body["resume_job_id"]).status == JobStatus.FAILED
        assert manager._active_job_id is None

    @pytest.mark.asyncio
    async def test_a_job_rebuilt_from_disk_can_be_resumed(self, client):
        """Resume 404'd for every job submitted before the last restart."""
        job_dir = _write_status("cafe12345678")
        checkpoint = job_dir / "output" / "checkpoint-36"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text("{}")

        manager = TrainingManager()
        client.app["training_manager"] = manager
        resp = await client.post("/api/training/jobs/cafe12345678/resume")
        assert resp.status == 201
        body = await resp.json()
        assert body["source_job_id"] == "cafe12345678"
        resumed = manager.get_job(body["resume_job_id"])
        if resumed._monitor_task is not None:
            await asyncio.wait_for(resumed._monitor_task, timeout=30)
