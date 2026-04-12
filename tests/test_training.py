"""Tests for ainode.training — config validation, job lifecycle, manager queue."""

import asyncio
import pytest

from ainode.training.engine import (
    TrainingConfig,
    TrainingJob,
    TrainingManager,
    JobStatus,
)


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
            base_model="model", dataset_path="user/dataset", method="qlora"
        )
        errors = config.validate()
        assert any("method" in e for e in errors)

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

    def test_submit_preserves_order(self):
        mgr = TrainingManager()
        j1 = mgr.submit_job(self._make_config(model="first"))
        j2 = mgr.submit_job(self._make_config(model="second"))
        assert mgr._queue == [j1.job_id, j2.job_id]
