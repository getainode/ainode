"""Tests for ainode.training — config validation, job lifecycle, manager queue, API routes."""

import asyncio
import json
import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from ainode.training.engine import (
    TrainingConfig,
    TrainingJob,
    TrainingManager,
    JobStatus,
)
from ainode.training.api_routes import setup_training_routes


# =============================================================================
# TrainingConfig validation
# =============================================================================


class TestTrainingConfig:
    def test_valid_config(self):
        config = TrainingConfig(
            base_model="meta-llama/Llama-3.2-3B-Instruct",
            dataset_path="user/my-dataset",
        )
        assert config.validate() == []

    def test_missing_base_model(self):
        config = TrainingConfig(base_model="", dataset_path="user/my-dataset")
        errors = config.validate()
        assert any("base_model" in e for e in errors)

    def test_missing_dataset_path(self):
        config = TrainingConfig(base_model="some-model", dataset_path="")
        errors = config.validate()
        assert any("dataset_path" in e for e in errors)

    def test_invalid_method(self):
        config = TrainingConfig(
            base_model="model", dataset_path="user/dataset", method="bogus"
        )
        errors = config.validate()
        assert any("method" in e for e in errors)

    def test_qlora_method_valid(self):
        config = TrainingConfig(
            base_model="model", dataset_path="user/dataset", method="qlora"
        )
        assert config.validate() == []

    def test_invalid_num_epochs(self):
        config = TrainingConfig(
            base_model="model", dataset_path="user/dataset", num_epochs=0
        )
        errors = config.validate()
        assert any("num_epochs" in e for e in errors)

    def test_invalid_batch_size(self):
        config = TrainingConfig(
            base_model="model", dataset_path="user/dataset", batch_size=-1
        )
        errors = config.validate()
        assert any("batch_size" in e for e in errors)

    def test_invalid_learning_rate(self):
        config = TrainingConfig(
            base_model="model", dataset_path="user/dataset", learning_rate=0
        )
        errors = config.validate()
        assert any("learning_rate" in e for e in errors)

    def test_invalid_lora_rank(self):
        config = TrainingConfig(
            base_model="model", dataset_path="user/dataset", lora_rank=0
        )
        errors = config.validate()
        assert any("lora_rank" in e for e in errors)

    def test_invalid_lora_alpha(self):
        config = TrainingConfig(
            base_model="model", dataset_path="user/dataset", lora_alpha=0
        )
        errors = config.validate()
        assert any("lora_alpha" in e for e in errors)

    def test_invalid_max_seq_length(self):
        config = TrainingConfig(
            base_model="model", dataset_path="user/dataset", max_seq_length=0
        )
        errors = config.validate()
        assert any("max_seq_length" in e for e in errors)

    def test_multiple_errors(self):
        config = TrainingConfig(
            base_model="",
            dataset_path="",
            method="bad",
            num_epochs=0,
        )
        errors = config.validate()
        assert len(errors) >= 3

    def test_defaults(self):
        config = TrainingConfig(base_model="m", dataset_path="user/d")
        assert config.method == "lora"
        assert config.num_epochs == 3
        assert config.batch_size == 4
        assert config.learning_rate == 2e-4
        assert config.lora_rank == 16
        assert config.lora_alpha == 32
        assert config.max_seq_length == 2048

    def test_to_dict(self):
        config = TrainingConfig(base_model="m", dataset_path="user/d")
        d = config.to_dict()
        assert d["base_model"] == "m"
        assert d["dataset_path"] == "user/d"
        assert isinstance(d, dict)

    def test_from_dict(self):
        data = {
            "base_model": "model-x",
            "dataset_path": "/data/file.json",
            "method": "full",
            "num_epochs": 5,
            "extra_field": "ignored",
        }
        config = TrainingConfig.from_dict(data)
        assert config.base_model == "model-x"
        assert config.method == "full"
        assert config.num_epochs == 5

    def test_from_dict_defaults(self):
        config = TrainingConfig.from_dict({
            "base_model": "m",
            "dataset_path": "/d",
        })
        assert config.lora_rank == 16


# =============================================================================
# TrainingJob lifecycle
# =============================================================================


class TestTrainingJob:
    def test_initial_state(self):
        config = TrainingConfig(base_model="m", dataset_path="user/d")
        job = TrainingJob(config)
        assert job.status == JobStatus.PENDING
        assert job.progress == 0.0
        assert job.current_epoch == 0
        assert job.current_loss is None
        assert job.start_time is None
        assert job.end_time is None
        assert len(job.job_id) == 12

    def test_custom_job_id(self):
        config = TrainingConfig(base_model="m", dataset_path="user/d")
        job = TrainingJob(config, job_id="custom123")
        assert job.job_id == "custom123"

    def test_get_status(self):
        config = TrainingConfig(base_model="m", dataset_path="user/d")
        job = TrainingJob(config)
        status = job.get_status()
        assert status["job_id"] == job.job_id
        assert status["status"] == "pending"
        assert status["progress"] == 0.0
        assert status["config"]["base_model"] == "m"

    def test_output_dir_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ainode.training.engine.JOBS_DIR", tmp_path / "jobs")
        config = TrainingConfig(base_model="m", dataset_path="user/d")
        job = TrainingJob(config)
        assert "output" in job.config.output_dir
        assert job.job_id in job.config.output_dir

    def test_output_dir_custom(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ainode.training.engine.JOBS_DIR", tmp_path / "jobs")
        config = TrainingConfig(
            base_model="m", dataset_path="user/d", output_dir="/custom/out"
        )
        job = TrainingJob(config)
        assert job.config.output_dir == "/custom/out"

    @pytest.mark.asyncio
    async def test_cancel_pending_job(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ainode.training.engine.JOBS_DIR", tmp_path / "jobs")
        config = TrainingConfig(base_model="m", dataset_path="user/d")
        job = TrainingJob(config)
        assert job.status == JobStatus.PENDING
        await job.stop()
        assert job.status == JobStatus.CANCELLED
        assert job.end_time is not None

    def test_parse_progress(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ainode.training.engine.JOBS_DIR", tmp_path / "jobs")
        config = TrainingConfig(base_model="m", dataset_path="user/d")
        job = TrainingJob(config)
        line = 'AINODE_PROGRESS:{"epoch":2,"loss":0.45,"progress":66.7}'
        job._parse_progress(line)
        assert job.current_epoch == 2
        assert job.current_loss == 0.45
        assert job.progress == 66.7

    def test_parse_progress_invalid_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ainode.training.engine.JOBS_DIR", tmp_path / "jobs")
        config = TrainingConfig(base_model="m", dataset_path="user/d")
        job = TrainingJob(config)
        job._parse_progress("AINODE_PROGRESS:{invalid}")
        assert job.current_epoch == 0  # Unchanged

    def test_parse_progress_no_marker(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ainode.training.engine.JOBS_DIR", tmp_path / "jobs")
        config = TrainingConfig(base_model="m", dataset_path="user/d")
        job = TrainingJob(config)
        job._parse_progress("some random log line")
        assert job.current_epoch == 0  # Unchanged


# =============================================================================
# TrainingManager queue operations
# =============================================================================


class TestTrainingManager:
    def _make_config(self, model="m", dataset="user/d"):
        return TrainingConfig(base_model=model, dataset_path=dataset)

    def test_submit_job(self):
        mgr = TrainingManager()
        job = mgr.submit_job(self._make_config())
        assert job.status == JobStatus.PENDING
        assert mgr.get_job(job.job_id) is job

    def test_submit_invalid_config(self):
        mgr = TrainingManager()
        config = TrainingConfig(base_model="", dataset_path="")
        with pytest.raises(ValueError, match="Invalid training config"):
            mgr.submit_job(config)

    def test_list_jobs(self):
        mgr = TrainingManager()
        mgr.submit_job(self._make_config())
        mgr.submit_job(self._make_config(model="model-2"))
        jobs = mgr.list_jobs()
        assert len(jobs) == 2
        assert all(j["status"] == "pending" for j in jobs)

    def test_list_jobs_empty(self):
        mgr = TrainingManager()
        assert mgr.list_jobs() == []

    def test_get_job_not_found(self):
        mgr = TrainingManager()
        assert mgr.get_job("nonexistent") is None

    @pytest.mark.asyncio
    async def test_cancel_pending_job(self):
        mgr = TrainingManager()
        job = mgr.submit_job(self._make_config())
        result = await mgr.cancel_job(job.job_id)
        assert result is True
        assert job.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_job(self):
        mgr = TrainingManager()
        result = await mgr.cancel_job("nope")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_already_cancelled(self):
        mgr = TrainingManager()
        job = mgr.submit_job(self._make_config())
        await mgr.cancel_job(job.job_id)
        result = await mgr.cancel_job(job.job_id)
        assert result is False

    def test_queue_size(self):
        mgr = TrainingManager()
        mgr.submit_job(self._make_config())
        mgr.submit_job(self._make_config())
        assert mgr.queue_size == 2

    @pytest.mark.asyncio
    async def test_queue_size_after_cancel(self):
        mgr = TrainingManager()
        j1 = mgr.submit_job(self._make_config())
        mgr.submit_job(self._make_config())
        await mgr.cancel_job(j1.job_id)
        assert mgr.queue_size == 1

    def test_active_job_none(self):
        mgr = TrainingManager()
        assert mgr.active_job is None

    def test_stats_empty(self):
        mgr = TrainingManager()
        s = mgr.stats()
        assert s["total"] == 0 and s["running"] == 0

    def test_estimate_returns_expected_keys(self):
        mgr = TrainingManager()
        cfg = TrainingConfig(base_model="x/Llama-3.1-8B", dataset_path="user/d")
        est = mgr.estimate(cfg, sample_count=500)
        for key in ("params_b", "memory_gb_per_node", "samples_per_sec", "estimated_seconds"):
            assert key in est

    def test_estimate_distributed_scales(self):
        mgr = TrainingManager()
        c1 = TrainingConfig(base_model="m", dataset_path="user/d", distributed=False)
        c2 = TrainingConfig(base_model="m", dataset_path="user/d", distributed=True, num_nodes=4)
        s1 = mgr.estimate(c1)["samples_per_sec"]
        s2 = mgr.estimate(c2)["samples_per_sec"]
        assert s2 > s1

    def test_dataset_id_resolution(self, tmp_path, monkeypatch):
        """TrainingManager should resolve dataset_id to an absolute path via DatasetManager."""
        from ainode.datasets.manager import DatasetManager
        monkeypatch.setattr("ainode.training.engine.JOBS_DIR", tmp_path / "jobs")
        ds_mgr = DatasetManager(root=tmp_path / "ds")
        ds = ds_mgr.add_upload("x.jsonl", b'{"text":"a"}\n', name="x")
        mgr = TrainingManager(dataset_manager=ds_mgr)
        config = TrainingConfig(
            base_model="m",
            dataset_path="placeholder",  # will be overwritten
            dataset_id=ds.id,
        )
        job = mgr.submit_job(config)
        assert job.config.dataset_path == ds.path
        assert job.config.dataset_id == ds.id

    def test_submit_preserves_order(self):
        mgr = TrainingManager()
        j1 = mgr.submit_job(self._make_config(model="first"))
        j2 = mgr.submit_job(self._make_config(model="second"))
        assert mgr._queue == [j1.job_id, j2.job_id]


# =============================================================================
# Progress parsing edge cases
# =============================================================================


class TestProgressParsing:
    """Test _parse_progress with various line formats."""

    def _make_job(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ainode.training.engine.JOBS_DIR", tmp_path / "jobs")
        config = TrainingConfig(base_model="m", dataset_path="user/d")
        return TrainingJob(config)

    def test_progress_with_timestamp_prefix(self, tmp_path, monkeypatch):
        """The _log() method prepends [HH:MM:SS] -- progress lines in logs have this."""
        job = self._make_job(tmp_path, monkeypatch)
        line = '[12:34:56] AINODE_PROGRESS:{"epoch":1,"loss":0.32,"progress":50.0}'
        job._parse_progress(line)
        assert job.current_epoch == 1
        assert job.current_loss == 0.32
        assert job.progress == 50.0

    def test_progress_partial_fields(self, tmp_path, monkeypatch):
        """Only some fields present in payload."""
        job = self._make_job(tmp_path, monkeypatch)
        line = 'AINODE_PROGRESS:{"loss":1.23}'
        job._parse_progress(line)
        assert job.current_loss == 1.23
        assert job.current_epoch == 0  # unchanged
        assert job.progress == 0.0  # unchanged

    def test_progress_step_field(self, tmp_path, monkeypatch):
        """Step field from actual trainer callback should not break anything."""
        job = self._make_job(tmp_path, monkeypatch)
        line = 'AINODE_PROGRESS:{"epoch":2,"loss":0.1,"progress":80.0,"step":150}'
        job._parse_progress(line)
        assert job.current_epoch == 2
        assert job.progress == 80.0

    def test_progress_complete(self, tmp_path, monkeypatch):
        """Final progress line at 100%."""
        job = self._make_job(tmp_path, monkeypatch)
        line = 'AINODE_PROGRESS:{"epoch":3,"loss":0,"progress":100.0}'
        job._parse_progress(line)
        assert job.progress == 100.0


# =============================================================================
# Integration: submit -> manager lifecycle
# =============================================================================


class TestManagerIntegration:
    """Integration tests for the full submit -> queue -> state lifecycle."""

    def test_submit_and_verify_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ainode.training.engine.JOBS_DIR", tmp_path / "jobs")
        mgr = TrainingManager()
        config = TrainingConfig(
            base_model="meta-llama/Llama-3.2-3B-Instruct",
            dataset_path="user/alpaca.jsonl",
            method="lora",
            num_epochs=2,
            batch_size=8,
        )
        job = mgr.submit_job(config)

        # Job is pending and in the queue
        assert job.status == JobStatus.PENDING
        assert mgr.queue_size == 1
        assert mgr.get_job(job.job_id) is job

        # Status dict has all expected fields
        status = job.get_status()
        assert status["job_id"] == job.job_id
        assert status["status"] == "pending"
        assert status["config"]["base_model"] == "meta-llama/Llama-3.2-3B-Instruct"
        assert status["config"]["num_epochs"] == 2
        assert status["config"]["batch_size"] == 8

    @pytest.mark.asyncio
    async def test_submit_cancel_verify(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ainode.training.engine.JOBS_DIR", tmp_path / "jobs")
        mgr = TrainingManager()
        config = TrainingConfig(base_model="m", dataset_path="user/d")
        job = mgr.submit_job(config)

        cancelled = await mgr.cancel_job(job.job_id)
        assert cancelled is True
        assert job.status == JobStatus.CANCELLED
        assert mgr.queue_size == 0

        # List should still contain the job
        jobs = mgr.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["status"] == "cancelled"

    def test_submit_multiple_and_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ainode.training.engine.JOBS_DIR", tmp_path / "jobs")
        mgr = TrainingManager()
        j1 = mgr.submit_job(TrainingConfig(base_model="a", dataset_path="user/d1"))
        j2 = mgr.submit_job(TrainingConfig(base_model="b", dataset_path="user/d2"))
        j3 = mgr.submit_job(TrainingConfig(base_model="c", dataset_path="user/d3"))

        jobs = mgr.list_jobs()
        assert len(jobs) == 3
        ids = {j["job_id"] for j in jobs}
        assert ids == {j1.job_id, j2.job_id, j3.job_id}


# =============================================================================
# API route tests (aiohttp test client)
# =============================================================================


@pytest.fixture
def training_app():
    """Create a minimal aiohttp app with training routes."""
    app = web.Application()
    manager = TrainingManager()
    setup_training_routes(app, manager)
    return app


@pytest_asyncio.fixture
async def training_client(training_app):
    async with TestClient(TestServer(training_app)) as c:
        yield c


class TestTrainingAPI:

    @pytest.mark.asyncio
    async def test_list_jobs_empty(self, training_client):
        resp = await training_client.get("/api/training/jobs")
        assert resp.status == 200
        data = await resp.json()
        assert data == {"jobs": []}

    @pytest.mark.asyncio
    async def test_submit_job(self, training_client):
        payload = {
            "base_model": "meta-llama/Llama-3.2-3B-Instruct",
            "dataset_path": "user/my-dataset.jsonl",
            "method": "lora",
            "num_epochs": 2,
        }
        resp = await training_client.post(
            "/api/training/jobs",
            json=payload,
        )
        assert resp.status == 201
        data = await resp.json()
        assert "job_id" in data
        assert data["config"]["base_model"] == "meta-llama/Llama-3.2-3B-Instruct"
        assert data["config"]["num_epochs"] == 2

    @pytest.mark.asyncio
    async def test_submit_invalid_config(self, training_client):
        resp = await training_client.post(
            "/api/training/jobs",
            json={"base_model": "", "dataset_path": ""},
        )
        assert resp.status == 400
        data = await resp.json()
        assert "error" in data

    @pytest.mark.asyncio
    async def test_submit_invalid_json(self, training_client):
        resp = await training_client.post(
            "/api/training/jobs",
            data=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_get_job(self, training_client):
        resp = await training_client.post(
            "/api/training/jobs",
            json={"base_model": "m", "dataset_path": "user/d"},
        )
        job_id = (await resp.json())["job_id"]
        resp = await training_client.get(f"/api/training/jobs/{job_id}")
        assert resp.status == 200
        data = await resp.json()
        assert data["job_id"] == job_id

    @pytest.mark.asyncio
    async def test_get_job_not_found(self, training_client):
        resp = await training_client.get("/api/training/jobs/nonexistent")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_cancel_job(self, training_client):
        resp = await training_client.post(
            "/api/training/jobs",
            json={"base_model": "m", "dataset_path": "user/d"},
        )
        job_id = (await resp.json())["job_id"]
        resp = await training_client.delete(f"/api/training/jobs/{job_id}")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self, training_client):
        resp = await training_client.delete("/api/training/jobs/nope")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_get_logs(self, training_client):
        resp = await training_client.post(
            "/api/training/jobs",
            json={"base_model": "m", "dataset_path": "user/d"},
        )
        job_id = (await resp.json())["job_id"]
        resp = await training_client.get(f"/api/training/jobs/{job_id}/logs")
        assert resp.status == 200
        data = await resp.json()
        assert "logs" in data
        assert isinstance(data["logs"], list)
        assert "total_lines" in data

    @pytest.mark.asyncio
    async def test_get_logs_with_tail(self, training_client):
        resp = await training_client.post(
            "/api/training/jobs",
            json={"base_model": "m", "dataset_path": "user/d"},
        )
        job_id = (await resp.json())["job_id"]
        resp = await training_client.get(f"/api/training/jobs/{job_id}/logs?tail=5")
        assert resp.status == 200
        data = await resp.json()
        assert isinstance(data["logs"], list)

    @pytest.mark.asyncio
    async def test_get_logs_not_found(self, training_client):
        resp = await training_client.get("/api/training/jobs/nope/logs")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_list_after_submit(self, training_client):
        await training_client.post(
            "/api/training/jobs",
            json={"base_model": "a", "dataset_path": "user/d"},
        )
        await training_client.post(
            "/api/training/jobs",
            json={"base_model": "b", "dataset_path": "user/d"},
        )
        resp = await training_client.get("/api/training/jobs")
        data = await resp.json()
        assert len(data["jobs"]) == 2

    @pytest.mark.asyncio
    async def test_templates_endpoint(self, training_client):
        resp = await training_client.get("/api/training/templates")
        assert resp.status == 200
        data = await resp.json()
        assert "templates" in data
        assert isinstance(data["templates"], list)
        assert len(data["templates"]) >= 1
        assert all("id" in t and "name" in t for t in data["templates"])

    @pytest.mark.asyncio
    async def test_stats_endpoint_empty(self, training_client):
        resp = await training_client.get("/api/training/stats")
        assert resp.status == 200
        data = await resp.json()
        assert data["total"] == 0
        assert data["running"] == 0
        assert data["total_gpu_hours"] == 0.0

    @pytest.mark.asyncio
    async def test_estimate_endpoint(self, training_client):
        resp = await training_client.post(
            "/api/training/estimate",
            json={
                "base_model": "meta-llama/Llama-3.1-8B-Instruct",
                "dataset_path": "ignored",
                "method": "lora",
                "num_epochs": 1,
                "batch_size": 4,
                "max_seq_length": 2048,
                "sample_count": 1000,
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["params_b"] >= 1.0
        assert data["memory_gb_per_node"] > 0
        assert data["samples_per_sec"] > 0

    @pytest.mark.asyncio
    async def test_estimate_invalid(self, training_client):
        resp = await training_client.post(
            "/api/training/estimate", data=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_cancel_already_cancelled(self, training_client):
        resp = await training_client.post(
            "/api/training/jobs",
            json={"base_model": "m", "dataset_path": "user/d"},
        )
        job_id = (await resp.json())["job_id"]
        await training_client.delete(f"/api/training/jobs/{job_id}")
        resp = await training_client.delete(f"/api/training/jobs/{job_id}")
        assert resp.status == 409
