"""Training engine — run fine-tuning jobs on local GPUs using HuggingFace + PEFT."""

from __future__ import annotations

import asyncio
import collections
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

from ainode.core.config import AINODE_HOME


TRAINING_DIR = AINODE_HOME / "training"
JOBS_DIR = TRAINING_DIR / "jobs"


class TrainingMethod(str, Enum):
    LORA = "lora"
    FULL = "full"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingConfig:
    """Configuration for a training/fine-tuning job."""

    base_model: str
    dataset_path: str
    output_dir: Optional[str] = None
    method: str = "lora"
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    max_seq_length: int = 2048

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty means valid)."""
        errors: list[str] = []

        if not self.base_model or not self.base_model.strip():
            errors.append("base_model is required")

        if not self.dataset_path or not self.dataset_path.strip():
            errors.append("dataset_path is required")
        else:
            ds = self.dataset_path.strip()
            if ".." in ds:
                errors.append("dataset_path must not contain '..'")
            elif ds.startswith("/"):
                datasets_dir = str(AINODE_HOME / "datasets")
                if not ds.startswith(datasets_dir):
                    errors.append(f"dataset_path absolute paths must be under {datasets_dir}")

        if self.method not in ("lora", "full"):
            errors.append(f"method must be 'lora' or 'full', got '{self.method}'")

        if self.num_epochs < 1:
            errors.append("num_epochs must be >= 1")

        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")

        if self.learning_rate <= 0:
            errors.append("learning_rate must be > 0")

        if self.lora_rank < 1:
            errors.append("lora_rank must be >= 1")

        if self.lora_alpha < 1:
            errors.append("lora_alpha must be >= 1")

        if self.max_seq_length < 1:
            errors.append("max_seq_length must be >= 1")

        if self.output_dir is not None:
            out = self.output_dir.strip()
            if ".." in out:
                errors.append("output_dir must not contain '..'")
            elif out.startswith("/"):
                allowed_prefix = str(AINODE_HOME / "training")
                if not out.startswith(allowed_prefix):
                    errors.append(f"output_dir absolute paths must be under {allowed_prefix}")

        return errors

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


class TrainingJob:
    """Represents a single training job with lifecycle management."""

    def __init__(self, config: TrainingConfig, job_id: Optional[str] = None):
        self.job_id: str = job_id or uuid.uuid4().hex[:12]
        self.config = config
        self.status: JobStatus = JobStatus.PENDING
        self.progress: float = 0.0
        self.current_epoch: int = 0
        self.current_loss: Optional[float] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.logs: collections.deque[str] = collections.deque(maxlen=5000)
        self._process: Optional[subprocess.Popen] = None
        self._monitor_task: Optional[asyncio.Task] = None

        # Set output directory
        if self.config.output_dir is None:
            self.config.output_dir = str(JOBS_DIR / self.job_id / "output")

        # Job working directory
        self._job_dir = JOBS_DIR / self.job_id
        self._job_dir.mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        """Launch the training subprocess."""
        if self.status != JobStatus.PENDING:
            raise RuntimeError(f"Cannot start job in '{self.status.value}' state")

        self.status = JobStatus.RUNNING
        self.start_time = time.time()
        self._log(f"Starting {self.config.method} training on {self.config.base_model}")

        # Write config to job directory for the training script
        config_path = self._job_dir / "config.json"
        config_path.write_text(json.dumps(self.config.to_dict(), indent=2))

        # Build the training command
        cmd = self._build_command(config_path)
        self._log(f"Command: {' '.join(cmd)}")

        try:
            # Ensure output dir exists
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(self._job_dir),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            # Start monitoring in background
            self._monitor_task = asyncio.create_task(self._monitor())
        except Exception as exc:
            self.status = JobStatus.FAILED
            self.end_time = time.time()
            self._log(f"Failed to start: {exc}")
            raise

    async def stop(self) -> None:
        """Gracefully cancel a running job."""
        if self.status == JobStatus.PENDING:
            self.status = JobStatus.CANCELLED
            self.end_time = time.time()
            self._log("Job cancelled before start")
            return

        if self.status != JobStatus.RUNNING:
            return

        self._log("Cancelling job...")
        if self._process and self._process.poll() is None:
            # Send SIGTERM for graceful shutdown
            self._process.send_signal(signal.SIGTERM)
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)

        self.status = JobStatus.CANCELLED
        self.end_time = time.time()
        self._log("Job cancelled")

        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()

    def get_status(self) -> dict:
        """Return a summary of the current job state."""
        elapsed = None
        if self.start_time:
            end = self.end_time or time.time()
            elapsed = round(end - self.start_time, 1)

        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": round(self.progress, 1),
            "current_epoch": self.current_epoch,
            "current_loss": self.current_loss,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_seconds": elapsed,
            "config": self.config.to_dict(),
        }

    def _build_command(self, config_path: Path) -> list[str]:
        """Build the CLI command to run training."""
        c = self.config

        if c.method == "lora":
            # Use a Python training script via module invocation
            return [
                sys.executable, "-m", "ainode.training._run_training",
                "--config", str(config_path),
            ]
        else:
            # Full fine-tuning — use torchrun for potential multi-GPU
            return [
                sys.executable, "-m", "torch.distributed.run",
                "--nproc_per_node=1",
                "-m", "ainode.training._run_training",
                "--config", str(config_path),
            ]

    async def _monitor(self) -> None:
        """Read subprocess output and update progress."""
        proc = self._process
        if proc is None or proc.stdout is None:
            return

        loop = asyncio.get_event_loop()
        try:
            while True:
                line = await loop.run_in_executor(None, proc.stdout.readline)
                if not line and proc.poll() is not None:
                    break
                if line:
                    line = line.rstrip()
                    self._log(line)
                    self._parse_progress(line)

            rc = proc.wait()
            if self.status == JobStatus.RUNNING:
                if rc == 0:
                    self.status = JobStatus.COMPLETED
                    self.progress = 100.0
                    self._log("Training completed successfully")
                else:
                    self.status = JobStatus.FAILED
                    self._log(f"Training process exited with code {rc}")
        except asyncio.CancelledError:
            pass
        finally:
            self.end_time = time.time()

    def _parse_progress(self, line: str) -> None:
        """Parse structured progress output from the training script.

        Expected format: AINODE_PROGRESS:{"epoch":1,"loss":0.5,"progress":33.3}
        """
        marker = "AINODE_PROGRESS:"
        if marker in line:
            try:
                payload = json.loads(line.split(marker, 1)[1])
                if "epoch" in payload:
                    self.current_epoch = payload["epoch"]
                if "loss" in payload:
                    self.current_loss = payload["loss"]
                if "progress" in payload:
                    self.progress = payload["progress"]
            except (json.JSONDecodeError, IndexError):
                pass

    def _log(self, msg: str) -> None:
        """Append a timestamped log entry."""
        ts = time.strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {msg}")


class TrainingManager:
    """Manage training jobs — one active at a time (GPU shared with inference)."""

    def __init__(self):
        self._jobs: dict[str, TrainingJob] = {}
        self._queue: list[str] = []  # job_ids in queue order
        self._active_job_id: Optional[str] = None

    def submit_job(self, config: TrainingConfig) -> TrainingJob:
        """Validate config and queue a new training job.

        Returns the created TrainingJob.
        Raises ValueError if config is invalid.
        """
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid training config: {'; '.join(errors)}")

        job = TrainingJob(config)
        self._jobs[job.job_id] = job
        self._queue.append(job.job_id)
        return job

    def list_jobs(self) -> list[dict]:
        """Return all jobs with their current status."""
        return [job.get_status() for job in self._jobs.values()]

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a specific job by ID."""
        return self._jobs.get(job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running or pending job. Returns True if cancelled."""
        job = self._jobs.get(job_id)
        if job is None:
            return False

        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False

        await job.stop()

        # Remove from queue if pending
        if job_id in self._queue:
            self._queue.remove(job_id)

        # Clear active if this was the running job
        if self._active_job_id == job_id:
            self._active_job_id = None

        return True

    async def start_next(self) -> Optional[TrainingJob]:
        """Start the next pending job if no job is currently running.

        Returns the started job, or None if nothing to start.
        """
        if self._active_job_id is not None:
            active = self._jobs.get(self._active_job_id)
            if active and active.status == JobStatus.RUNNING:
                return None  # Something is already running
            # Active job finished — clear it
            self._active_job_id = None

        # Find next pending job in queue
        while self._queue:
            job_id = self._queue[0]
            job = self._jobs.get(job_id)
            if job and job.status == JobStatus.PENDING:
                self._queue.pop(0)
                self._active_job_id = job_id
                await job.start()
                return job
            else:
                self._queue.pop(0)  # Skip cancelled/missing jobs

        return None

    @property
    def active_job(self) -> Optional[TrainingJob]:
        """Return the currently running job, if any."""
        if self._active_job_id:
            return self._jobs.get(self._active_job_id)
        return None

    @property
    def queue_size(self) -> int:
        """Number of pending jobs in the queue."""
        return len([
            jid for jid in self._queue
            if jid in self._jobs and self._jobs[jid].status == JobStatus.PENDING
        ])
