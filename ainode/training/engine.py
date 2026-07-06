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
    QLORA = "qlora"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


QUANT_IMAGE = os.environ.get("AINODE_QUANT_IMAGE") or "ainode-quant:0.17.0-t5"
# LoRA / adapter-merge jobs run in a spawned GPU container too — the slim
# orchestrator image has no torch/peft/datasets. Default to the quant image
# (torch 2.10 / transformers 5.10 / datasets 5.0 / accelerate 1.13; peft is
# pip-shimmed at launch, see _build_container_command). Override per-deploy.
TRAIN_IMAGE = os.environ.get("AINODE_TRAIN_IMAGE") or QUANT_IMAGE


def _host_path(container_path: str) -> str:
    """Translate an AINODE_HOME path (orchestrator *container* view) to the host
    path so a docker ``-v`` SOURCE resolves on the host daemon. Mirrors
    NvidiaBackend._host_path. No-op when AINODE_HOST_HOME is unset (AINode running
    directly on the host, where the two paths coincide)."""
    host_home = os.environ.get("AINODE_HOST_HOME")
    if not host_home:
        return container_path
    home = str(AINODE_HOME)
    if container_path == home or container_path.startswith(home + os.sep):
        return host_home.rstrip("/") + container_path[len(home):]
    return container_path


def _loadable_dir(d: Path) -> Optional[Path]:
    """Return the directory ``from_pretrained`` should actually load from, or None.

    A direct-download / flat dir holds ``config.json`` at its top level — load it
    as-is. An HF-cache-format dir (``models--org--name``) instead nests the real
    weights under ``snapshots/<hash>/``; return that snapshot subdir (its relative
    symlinks into ``../../blobs`` still resolve because the whole models tree is
    mounted). Prefer a snapshot that actually carries a ``config.json``."""
    if not d.is_dir():
        return None
    if (d / "config.json").exists():
        return d
    snap = d / "snapshots"
    if snap.is_dir():
        subs = sorted(s for s in snap.iterdir() if s.is_dir())
        for s in subs:
            if (s / "config.json").exists():
                return s
        if subs:
            return subs[0]
    return None


def _resolve_base_model_mount(base_model: str) -> Optional[str]:
    """If ``base_model`` names a model already on disk under the models store,
    return its CONTAINER mount path under ``/ainode-models/...``; else None.

    The training wizard's downloaded-model cards submit the ON-DISK slug (e.g.
    ``qwen--qwen2.5-0.5b-instruct``). Handed straight to
    ``AutoTokenizer.from_pretrained`` that raises HFValidationError ("Cannot have
    -- or .. in repo_id") and the job dies instantly. Rewriting it to the mounted
    directory path makes HF load from local weights (also offline-safe — no hub
    round-trip). Accepts both the raw slug and a canonical HF repo id
    (``Qwen/Qwen2.5-0.5B-Instruct``).

    Recognizes ALL FOUR on-disk layouts the registry tracks (mirrors
    ``ModelManager._find_model_dir``): direct ``org--name`` (our downloader), flat
    HF ``models--org--name``, HF cache ``hub/models--org--name``, and out-of-band
    ``hf-cache/hub/models--org--name`` (HF_HOME downloads, e.g. from the eugr
    distributed-serving backend). Missing the cache layouts silently fell back to
    a live hub round-trip that fails on air-gapped nodes for models that ARE on
    disk. Returns None for a plain hub repo id with no local copy so it passes
    through to load from the hub as before."""
    if not base_model:
        return None
    models_root = AINODE_HOME / "models"

    def _to_mount(p: Path) -> Optional[str]:
        try:
            rel = p.relative_to(models_root)
        except ValueError:
            return None
        return "/ainode-models/" + str(rel).replace(os.sep, "/")

    # Flat/direct forms: the raw slug, and (for a repo id) its org--name dir.
    # Lenient — the weights of a direct download sit at the dir's top level, so a
    # bare existing dir maps straight to its mount (matches the downloader layout
    # even before any config.json probe).
    flat_slug = base_model.replace("/", "--")
    for slug in (base_model, flat_slug):
        if not slug or "/" in slug or slug.startswith("."):
            continue
        d = models_root / slug
        if d.is_dir():
            return _to_mount(_loadable_dir(d) or d)

    # HF-cache forms: models--org--name under the store root, hub/, and
    # hf-cache/hub/. `flat_slug` is already org--name here (repo id or slug).
    hf_slug = "models--" + flat_slug
    for cache_dir in (models_root / hf_slug,
                      models_root / "hub" / hf_slug,
                      models_root / "hf-cache" / "hub" / hf_slug):
        loadable = _loadable_dir(cache_dir)
        if loadable is not None:
            return _to_mount(loadable)
    return None


def _vendor_wheel(pkg: str, job_dir: Path) -> Optional[str]:
    """Ensure a wheel for ``pkg`` is available in ``job_dir`` (mounted at /job) so
    the spawned container can ``pip install --no-index`` it with NO network.

    Wheels are cached once under ``AINODE_HOME/wheels`` and copied into each job
    dir. When the cache is empty we fetch it with the orchestrator's own pip
    (``pip download --no-deps``) — best-effort, short timeout — unless
    ``AINODE_NO_WHEEL_FETCH`` is set (air-gapped nodes pre-seed the cache).
    Returns the wheel FILENAME (basename) if vendored, else None so the caller
    falls back to online pip. Only sensible for pure-python packages (peft)."""
    import glob
    import shutil

    cache = AINODE_HOME / "wheels"
    norm = pkg.replace("-", "_")

    def _find(directory: Path) -> Optional[str]:
        for pat in (f"{norm}-*.whl", f"{pkg}-*.whl"):
            hits = sorted(glob.glob(str(directory / pat)))
            if hits:
                return hits[0]
        return None

    try:
        cache.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None

    wheel = _find(cache)
    if wheel is None and not os.environ.get("AINODE_NO_WHEEL_FETCH"):
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "download", "--no-deps",
                 "--dest", str(cache), pkg],
                capture_output=True, text=True, timeout=120,
            )
        except Exception:
            pass
        wheel = _find(cache)
    if wheel is None:
        return None
    try:
        dest = job_dir / Path(wheel).name
        if not dest.exists():
            shutil.copy2(wheel, dest)
        return dest.name
    except Exception:
        return None


def _pip_install_step(pkg: str, job_dir: Path, *, vendor: bool) -> str:
    """One tolerant shell step that makes ``pkg`` importable in the spawned
    container. Import-guarded (a future baked image satisfies it with no install),
    then — for vendored pure-python deps — an offline ``--no-index`` install from
    the mounted wheel, falling back to online pip only if the wheel is absent.
    Joined with ``;`` (never ``&&``) so a failed install never blocks the runner;
    the runner itself reports a clean error if the dep is truly missing."""
    mod = pkg.replace("-", "_")
    guard = f"python3 -c 'import {mod}' 2>/dev/null"
    if vendor:
        wheel = _vendor_wheel(pkg, job_dir)
        if wheel:
            install = (f"pip install -q --no-index --find-links /job /job/{wheel} "
                       f"|| pip install -q --no-deps {pkg}")
        else:
            install = f"pip install -q --no-deps {pkg}"
    else:
        # Network-only best-effort (e.g. bitsandbytes has no pure-python wheel).
        install = f"pip install -q --no-deps {pkg}"
    return f"{guard} || {install}"


@dataclass
class TrainingConfig:
    """Configuration for a training/fine-tuning job."""

    base_model: str
    dataset_path: str = ""  # required for training (enforced in validate); quantize omits it
    output_dir: Optional[str] = None
    method: str = "lora"
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    max_seq_length: int = 2048
    # Extended (optional) fields — enable premium UI & richer runs.
    dataset_id: Optional[str] = None  # references a Dataset in DatasetManager
    run_name: Optional[str] = None
    description: str = ""
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    weight_decay: float = 0.0
    use_gradient_checkpointing: bool = False
    distributed: bool = False
    num_nodes: int = 1
    template_id: Optional[str] = None  # training template used
    hf_token: Optional[str] = None                  # Hugging Face token for gated models
    _resume_from_checkpoint: Optional[str] = None  # internal: checkpoint path for resume
    eval_split: float = 0.1          # fraction of dataset to hold out for evaluation (0 = no eval)
    eval_steps: int = 0              # evaluate every N steps (0 = once per epoch)
    wandb_project: Optional[str] = None  # if set, enable W&B logging to this project
    # Quantize-job fields (method == "quantize") — runs llm-compressor in a GPU
    # container, producing a servable AWQ/NVFP4 checkpoint. See _run_quant.py.
    scheme: Optional[str] = None                      # "awq" | "nvfp4"
    calib_dataset: str = "HuggingFaceH4/ultrachat_200k"
    calib_samples: int = 256
    out_slug: Optional[str] = None                    # output dir name under ~/.ainode/models
    push_to_hf: bool = False
    hf_repo: Optional[str] = None                     # target repo; namespace defaults to whoami

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty means valid)."""
        errors: list[str] = []

        if not self.base_model or not self.base_model.strip():
            errors.append("base_model is required")

        if self.method == "quantize":
            pass  # quantize calibrates on calib_dataset, not a training dataset_path
        elif not self.dataset_path or not self.dataset_path.strip():
            errors.append("dataset_path is required")
        else:
            ds = self.dataset_path.strip()
            if ".." in ds:
                errors.append("dataset_path must not contain '..'")
            elif ds.startswith("/") and not self.dataset_id:
                # Absolute paths are only accepted under the known datasets dir
                # unless the path was resolved via a registered dataset_id.
                datasets_dir = str(AINODE_HOME / "datasets")
                if not ds.startswith(datasets_dir):
                    errors.append(f"dataset_path absolute paths must be under {datasets_dir}")

        if self.method not in ("lora", "full", "qlora", "quantize"):
            errors.append(f"method must be 'lora', 'qlora', 'full' or 'quantize', got '{self.method}'")
        if self.method == "quantize" and self.scheme not in ("awq", "nvfp4"):
            errors.append(f"scheme must be 'awq' or 'nvfp4' for a quantize job, got '{self.scheme}'")

        if self.num_nodes < 1:
            errors.append("num_nodes must be >= 1")
        if self.gradient_accumulation_steps < 1:
            errors.append("gradient_accumulation_steps must be >= 1")
        if self.warmup_steps < 0:
            errors.append("warmup_steps must be >= 0")
        if self.weight_decay < 0:
            errors.append("weight_decay must be >= 0")

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
        # Explicit override for the spawned GPU container name. Set by callers
        # whose container is spawned outside the normal start() path (merge jobs),
        # so stop() can still `docker kill` it. None → derive from method/job_id.
        self._container_name_override: Optional[str] = None

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

        # Build the training command OFF the event loop. _build_command can
        # shell out to a blocking `pip download` (peft wheel vendoring, up to a
        # 120s timeout) on a cold cache — running that inline on aiohttp's single
        # loop would freeze every concurrent request (live inference proxying
        # included) until it returns. run_in_executor keeps the loop responsive.
        loop = asyncio.get_event_loop()
        cmd = await loop.run_in_executor(None, self._build_command, config_path)
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

        # For container-spawn jobs (quantize always; lora/qlora/full in-container;
        # merge) self._process above is only the local `docker run` CLIENT — a
        # SIGKILL to it is NOT relayed to the `--gpus all` container, which would
        # keep running and holding the GPU with no record anywhere. Explicitly
        # remove the named container to free the device (mirrors
        # NvidiaBackend.stop()). Best-effort — swallow all errors.
        name = self._container_name()
        if name:
            for args in (["docker", "stop", name], ["docker", "rm", "-f", name]):
                try:
                    subprocess.run(args, capture_output=True, text=True, timeout=30)
                except Exception:
                    self._log(f"{' '.join(args)} failed (best-effort)")

        self.status = JobStatus.CANCELLED
        self.end_time = time.time()
        self._log("Job cancelled")

        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()

    def _container_name(self) -> Optional[str]:
        """Deterministic name of this job's spawned GPU container, or None for the
        in-process host-venv path (nothing to ``docker kill``).

        Quantize always runs in a container; lora/qlora/full only in
        container-spawn mode (``AINODE_IN_CONTAINER``); merge jobs register an
        explicit override because their container is spawned outside ``start()``.
        """
        if self._container_name_override:
            return self._container_name_override
        if self.config.method == "quantize":
            return f"ainode-quant-{self.job_id}"
        if os.environ.get("AINODE_IN_CONTAINER"):
            return f"ainode-train-{self.job_id}"
        return None

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
        """Build the CLI command to run training.

        Single-GPU (solo or LoRA/QLoRA on one card) runs as plain Python.
        Multi-GPU / multi-node runs go through ``torch.distributed.run``
        (aka ``torchrun``) so the HF Trainer picks up RANK/LOCAL_RANK/
        WORLD_SIZE and does DDP automatically.
        """
        c = self.config

        if c.method == "quantize":
            return self._build_quant_command(config_path)

        nproc = max(1, int(_detect_local_gpu_count()))
        needs_ddp = c.distributed or c.num_nodes > 1 or (c.method == "full" and nproc > 1)

        # In the shipped (slim) orchestrator container there is no torch/peft, so
        # the in-process `python -m ...` path is dead on arrival. Spawn a GPU
        # container from TRAIN_IMAGE instead — same pattern as quantize.
        if os.environ.get("AINODE_IN_CONTAINER"):
            if needs_ddp:
                raise RuntimeError(
                    "Distributed training (DDP / multi-node) is not supported in "
                    "container-spawn mode — run AINode in host-venv mode for multi-node "
                    f"DDP. (distributed={c.distributed}, num_nodes={c.num_nodes}, "
                    f"method={c.method}, local_gpus={nproc})"
                )
            return self._build_container_command()

        if not needs_ddp:
            return [
                sys.executable, "-m", "ainode.training._run_training",
                "--config", str(config_path),
            ]

        # Multi-GPU / multi-node path. ``torch.distributed.run`` handles
        # --nproc_per_node locally; cross-node rendezvous is the caller's
        # responsibility (set MASTER_ADDR / MASTER_PORT / NODE_RANK /
        # NNODES in the environment before spawning).
        return [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            f"--nnodes={max(1, c.num_nodes)}",
            "-m", "ainode.training._run_training",
            "--config", str(config_path),
        ]

    def _build_quant_command(self, config_path: Path) -> list[str]:
        """Quantization runs in a spawned GPU container — the slim orchestrator has
        no torch. Mirror the inference docker-run pattern (--gpus all, host-translated
        mounts), but mount the model store READ-WRITE so the runner reads the base
        weights and writes the quantized checkpoint to ~/.ainode/models/<out-slug>.
        Foreground (no -d): the existing Popen monitor streams AINODE_PROGRESS and
        the container exit code signals completion. Single-node, single-GPU."""
        c = self.config
        # Host-path prereq (contract tripwire): in-container without AINODE_HOST_HOME
        # the RW model mount resolves to an empty root-owned host dir and the output
        # is written into a throwaway --rm layer (lost on exit). Fail loud.
        if os.environ.get("AINODE_IN_CONTAINER") and not os.environ.get("AINODE_HOST_HOME"):
            raise RuntimeError(
                "quantize requires AINODE_HOST_HOME (the host path mounted at AINODE_HOME) "
                "so the output mount is host-backed — refusing to run, the checkpoint would be lost."
            )
        models_host = _host_path(str(AINODE_HOME / "models"))
        jobdir_host = _host_path(str(self._job_dir))
        token = c.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
        cmd: list[str] = [
            "docker", "run", "--rm",
            "--name", f"ainode-quant-{self.job_id}",
            "--gpus", "all", "--network", "host", "--ipc=host", "--shm-size", "16g",
            "-v", f"{models_host}:/ainode-models",            # RW: read base + write output
            "-v", f"{jobdir_host}:/job:ro",                   # config.json
            "-e", "HF_HUB_CACHE=/ainode-models/hf-cache",     # persist HF pulls into the store
        ]
        if token:
            cmd += ["-e", f"HF_TOKEN={token}", "-e", f"HUGGING_FACE_HUB_TOKEN={token}"]
        cmd += [QUANT_IMAGE, "python3", "/opt/ainode/run_quant.py", "--config", "/job/config.json"]
        return cmd

    def _build_container_command(self) -> list[str]:
        """LoRA/QLoRA/full (single-GPU) training in a spawned GPU container — the
        slim orchestrator has no torch/peft. Mirror _build_quant_command: --gpus all,
        host-translated mounts, foreground so the existing Popen monitor streams
        AINODE_PROGRESS and the container exit code signals completion.

        The train image bakes no training runner, so we copy _run_training.py into
        the job dir (mounted at /job) and rewrite a container-view config whose
        output_dir + dataset_path point at the mounts below — otherwise checkpoints
        land in the --rm layer and vanish on exit."""
        import shutil

        c = self.config
        # Host-path prereq (contract tripwire) — same guard as quantize: without
        # AINODE_HOST_HOME the RW mounts resolve to empty root-owned host dirs and
        # the adapter is written into a throwaway --rm layer (lost on exit).
        if not os.environ.get("AINODE_HOST_HOME"):
            raise RuntimeError(
                "container training requires AINODE_HOST_HOME (the host path mounted "
                "at AINODE_HOME) so the output mount is host-backed — refusing to run, "
                "the adapter/checkpoints would be lost with the --rm container."
            )

        # The train image knows nothing of the ainode package — copy the runner in.
        shutil.copy2(Path(__file__).parent / "_run_training.py", self._job_dir / "_run_training.py")

        # Container-view config: remap absolute output_dir + dataset_path onto the
        # mounts. job.config.output_dir stays the orchestrator path (same host inode
        # via the /job mount) so handle_get_output/download resolve unchanged.
        container_cfg = dict(c.to_dict())
        container_cfg["output_dir"] = "/job/output"
        # base_model may be an on-disk slug (what the GUI submits) — rewrite it to
        # the mounted weights path so AutoTokenizer.from_pretrained loads locally
        # instead of raising HFValidationError on the '--' in the slug.
        base_mount = _resolve_base_model_mount(c.base_model)
        if base_mount:
            container_cfg["base_model"] = base_mount
        datasets_dir = str(AINODE_HOME / "datasets")
        ds = c.dataset_path or ""
        if ds.startswith(datasets_dir):
            container_cfg["dataset_path"] = "/ainode-datasets/" + ds[len(datasets_dir):].lstrip("/")
        elif ds and not ds.startswith("/") and not ds.startswith("~"):
            # Relative dataset_path (e.g. "alpaca.jsonl" — exactly what the New Run
            # wizard's placeholder suggests) resolves against ~/.ainode/datasets on
            # the host. The runner's own resolver would look under AINODE_HOME=/job
            # (the container's job dir), where the datasets aren't mounted, so remap
            # it here onto the /ainode-datasets mount IF the file exists there.
            # Leave it untouched otherwise, so a HF hub repo id ("tatsu-lab/alpaca")
            # still passes through to load_dataset(). Mirrors _run_training.py's own
            # exists()-gated resolution.
            if (AINODE_HOME / "datasets" / ds).exists():
                container_cfg["dataset_path"] = "/ainode-datasets/" + ds.lstrip("/")
        (self._job_dir / "config.container.json").write_text(json.dumps(container_cfg, indent=2))

        models_host = _host_path(str(AINODE_HOME / "models"))
        datasets_host = _host_path(datasets_dir)
        jobdir_host = _host_path(str(self._job_dir))
        token = c.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""

        cmd: list[str] = [
            "docker", "run", "--rm",
            "--name", f"ainode-train-{self.job_id}",
            "--gpus", "all", "--network", "host", "--ipc=host", "--shm-size", "16g",
            "-v", f"{models_host}:/ainode-models",             # RW: HF cache + on-disk weights
            "-v", f"{datasets_host}:/ainode-datasets:ro",      # training data
            "-v", f"{jobdir_host}:/job",                       # runner + config + output
            "-e", "HF_HUB_CACHE=/ainode-models/hf-cache",      # persist HF pulls into the store
            "-e", "AINODE_HOME=/job",                          # runner config fallback (relative datasets)
        ]
        if token:
            cmd += ["-e", f"HF_TOKEN={token}", "-e", f"HUGGING_FACE_HUB_TOKEN={token}"]
        # ponytail: peft (and bitsandbytes for qlora) aren't baked into the train
        # image yet — pip-shim them at launch. TODO(ponytail): bake peft +
        # bitsandbytes into the next quant/train-image build and drop this shim.
        # peft is pure-python → vendor a wheel (offline-safe); bitsandbytes is
        # network-only best-effort (no aarch64 pure-python wheel). Steps are ';'
        # separated so a failed install never blocks the runner (a fatal '&&' here
        # killed jobs on nodes with broken DNS — the merge/train couldn't pip peft).
        steps = [_pip_install_step("peft", self._job_dir, vendor=True)]
        if c.method == "qlora":
            steps.append(_pip_install_step("bitsandbytes", self._job_dir, vendor=False))
        prep = " ; ".join(steps)
        cmd += [
            TRAIN_IMAGE, "sh", "-c",
            f"{prep} ; python3 /job/_run_training.py --config /job/config.container.json",
        ]
        return cmd

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
                    self._log("Job completed successfully")
                    if getattr(self.config, "method", "") == "quantize" and getattr(self.config, "push_to_hf", False):
                        await self._push_to_hf()
                else:
                    self.status = JobStatus.FAILED
                    self._log(f"Training process exited with code {rc}")
        except asyncio.CancelledError:
            pass
        finally:
            self.end_time = time.time()

    async def _push_to_hf(self) -> None:
        """After a quantize job completes, push the on-disk checkpoint to HF.
        Pure huggingface_hub (no torch) — run the blocking upload off the loop."""
        c = self.config
        out_dir = str(AINODE_HOME / "models" / (c.out_slug or ""))
        repo = c.hf_repo or c.out_slug
        token = c.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not repo or not token:
            self._log("push_to_hf: missing repo or token — skipped")
            return
        try:
            from ainode.models.hf_upload import upload_checkpoint
            loop = asyncio.get_event_loop()
            url = await loop.run_in_executor(
                None, lambda: upload_checkpoint(out_dir, repo, token, None, True)
            )
            self._log(f"push_to_hf: uploaded to {url}")
        except Exception as exc:
            self._log(f"push_to_hf failed: {exc}")

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
                if "pct" in payload:  # quantize runner emits {phase, pct, msg}
                    self.progress = payload["pct"]
            except (json.JSONDecodeError, IndexError):
                pass

    def _log(self, msg: str) -> None:
        """Append a timestamped log entry."""
        ts = time.strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {msg}")


def build_merge_command(
    merge_job: "TrainingJob",
    base_model: str,
    adapter_dir: Path,
    merged_dir: Path,
    hf_token: Optional[str] = None,
) -> list[str]:
    """Spawn a GPU container to merge a LoRA/QLoRA adapter into its base model.

    The slim orchestrator has no peft/torch, so — like training and quantize —
    the merge runs in TRAIN_IMAGE. Copies the self-contained _run_merge.py into
    the merge job dir, mounts the adapter RO, the merged-output parent RW, and the
    models store (HF cache), and pip-shims peft at launch. Foreground: the caller
    streams AINODE_PROGRESS and the exit code signals completion."""
    import shutil

    # Same host-backing tripwire as training/quantize.
    if not os.environ.get("AINODE_HOST_HOME"):
        raise RuntimeError(
            "container merge requires AINODE_HOST_HOME (the host path mounted at "
            "AINODE_HOME) so the merged model is host-backed — refusing to run, it "
            "would be lost with the --rm container."
        )

    job_dir = merge_job._job_dir
    adapter_dir = Path(adapter_dir)
    merged_dir = Path(merged_dir)
    merged_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).parent / "_run_merge.py", job_dir / "_run_merge.py")

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    # Same slug→mount rewrite as training: a downloaded base passed as its on-disk
    # slug must load from /ainode-models/<slug>, not choke AutoTokenizer on the '--'.
    merge_cfg = {
        "base_model": _resolve_base_model_mount(base_model) or base_model,
        "adapter_dir": "/adapter",
        "output_dir": f"/out/{merged_dir.name}",
        "hf_token": token,
    }
    (job_dir / "merge_config.json").write_text(json.dumps(merge_cfg, indent=2))

    cmd: list[str] = [
        "docker", "run", "--rm",
        "--name", f"ainode-merge-{merge_job.job_id}",
        "--gpus", "all", "--network", "host", "--ipc=host", "--shm-size", "16g",
        "-v", f"{_host_path(str(job_dir))}:/job",                          # runner + config
        "-v", f"{_host_path(str(adapter_dir))}:/adapter:ro",               # LoRA adapter
        "-v", f"{_host_path(str(merged_dir.parent))}:/out",                # RW: merged model
        "-v", f"{_host_path(str(AINODE_HOME / 'models'))}:/ainode-models",  # base weights / HF cache
        "-e", "HF_HUB_CACHE=/ainode-models/hf-cache",
    ]
    if token:
        cmd += ["-e", f"HF_TOKEN={token}", "-e", f"HUGGING_FACE_HUB_TOKEN={token}"]
    # ponytail: peft pip-shim — bake it into the next train-image build and drop this.
    # Vendor the (pure-python) peft wheel so the merge runs offline; ';' not '&&'
    # so a pip hiccup never blocks the runner (broken DNS killed a live merge here).
    peft_step = _pip_install_step("peft", job_dir, vendor=True)
    cmd += [
        TRAIN_IMAGE, "sh", "-c",
        f"{peft_step} ; python3 /job/_run_merge.py --config /job/merge_config.json",
    ]
    return cmd


TRAINING_TEMPLATES: list[dict] = [
    {
        "id": "alpaca-instruct",
        "name": "Alpaca-style instruction tuning",
        "description": "Fine-tune on instruction/output pairs. Classic Alpaca format.",
        "method": "lora",
        "sample_shape": {"instruction": "str", "input": "str (optional)", "output": "str"},
        "recommended_epochs": 3,
        "recommended_batch_size": 4,
        "recommended_lr": 2e-4,
        "estimated_time": "20-60 min (3B model, ~1k samples)",
    },
    {
        "id": "sharegpt-chat",
        "name": "Chat fine-tune (ShareGPT format)",
        "description": "Multi-turn conversation tuning — human/gpt turns.",
        "method": "lora",
        "sample_shape": {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]},
        "recommended_epochs": 2,
        "recommended_batch_size": 2,
        "recommended_lr": 1e-4,
        "estimated_time": "30-90 min (3B model, ~1k samples)",
    },
    {
        "id": "classification-head",
        "name": "Classification head",
        "description": "Train a lightweight classifier on labeled text.",
        "method": "lora",
        "sample_shape": {"text": "str", "label": "str"},
        "recommended_epochs": 5,
        "recommended_batch_size": 8,
        "recommended_lr": 3e-4,
        "estimated_time": "10-30 min (small dataset)",
    },
    {
        "id": "dpo-preference",
        "name": "DPO / Preference learning",
        "description": "Align a model with chosen/rejected preference pairs.",
        "method": "lora",
        "sample_shape": {"prompt": "str", "chosen": "str", "rejected": "str"},
        "recommended_epochs": 1,
        "recommended_batch_size": 2,
        "recommended_lr": 5e-5,
        "estimated_time": "1-3 hours",
    },
    {
        "id": "distributed-ddp",
        "name": "Distributed DDP (multi-node)",
        "description": "Multi-node data-parallel full or LoRA fine-tune via torchrun.",
        "method": "lora",
        "sample_shape": {"text": "str"},
        "recommended_epochs": 1,
        "recommended_batch_size": 2,
        "recommended_lr": 2e-4,
        "distributed": True,
        "estimated_time": "varies — scales with nodes",
    },
]


def get_training_templates() -> list[dict]:
    """Return the hard-coded list of training templates shown in the UI."""
    return list(TRAINING_TEMPLATES)


def _detect_local_gpu_count() -> int:
    """Return the number of CUDA-visible GPUs on this host.

    Never raises — falls back to 1 when torch is missing or CUDA is
    unavailable, so command construction stays deterministic on
    CPU-only dev boxes.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return max(1, torch.cuda.device_count())
    except Exception:
        pass
    return 1


class TrainingManager:
    """Manage training jobs — one active at a time (GPU shared with inference)."""

    def __init__(self, dataset_manager=None):
        self._jobs: dict[str, TrainingJob] = {}
        self._queue: list[str] = []  # job_ids in queue order
        self._active_job_id: Optional[str] = None
        self.dataset_manager = dataset_manager

    # ------------------------------------------------------------------
    # Stats / estimates
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """Return aggregate counters for the overview dashboard."""
        total = len(self._jobs)
        running = completed = failed = cancelled = pending = 0
        completed_today = 0
        total_gpu_seconds = 0.0
        now = time.time()
        for j in self._jobs.values():
            if j.status == JobStatus.RUNNING:
                running += 1
            elif j.status == JobStatus.COMPLETED:
                completed += 1
                if j.end_time and (now - j.end_time) < 86400:
                    completed_today += 1
            elif j.status == JobStatus.FAILED:
                failed += 1
            elif j.status == JobStatus.CANCELLED:
                cancelled += 1
            else:
                pending += 1
            if j.start_time:
                end = j.end_time or now
                total_gpu_seconds += max(0.0, end - j.start_time)
        return {
            "total": total,
            "running": running,
            "completed": completed,
            "completed_today": completed_today,
            "failed": failed,
            "cancelled": cancelled,
            "pending": pending,
            "total_gpu_hours": round(total_gpu_seconds / 3600.0, 2),
            "active_job_id": self._active_job_id,
            "queue_size": self.queue_size,
        }

    @staticmethod
    def estimate(config: TrainingConfig, sample_count: Optional[int] = None) -> dict:
        """Cheap heuristic estimates for time / memory / throughput.

        These are intentionally coarse — meant for UI hints, not billing.
        """
        # Pull a rough param count from the model string
        model = (config.base_model or "").lower()
        params_b = 3.0  # default to ~3B
        for key, val in (("405b", 405.0), ("70b", 70.0), ("34b", 34.0),
                          ("8b", 8.0), ("7b", 7.0), ("3b", 3.0), ("1b", 1.0)):
            if key in model:
                params_b = val
                break

        # Memory (GB) — very approximate
        bytes_per_param = 2  # fp16
        base_mem = params_b * bytes_per_param  # weights in GB
        if config.method == "lora" or config.method == "qlora":
            training_mem = base_mem * 1.2  # small overhead for LoRA adapters + activations
            if config.method == "qlora":
                training_mem = base_mem * 0.35  # 4-bit quantized
        else:
            training_mem = base_mem * 4.0  # weights + grads + optimizer state

        # Throughput — samples/sec (handwave on GB10)
        tokens_per_sec = max(500.0, 50000.0 / max(1.0, params_b))
        tokens_per_sample = config.max_seq_length
        samples_per_sec = tokens_per_sec / max(1, tokens_per_sample)
        if config.distributed and config.num_nodes > 1:
            samples_per_sec *= config.num_nodes * 0.85  # imperfect scaling

        # Time estimate
        if sample_count and sample_count > 0:
            total_samples = sample_count * config.num_epochs
            effective_batch = max(1, config.batch_size * config.gradient_accumulation_steps)
            steps = total_samples / effective_batch
            seconds = steps / max(0.01, samples_per_sec / max(1, effective_batch))
        else:
            seconds = None

        return {
            "params_b": params_b,
            "memory_gb_per_node": round(training_mem, 1),
            "samples_per_sec": round(samples_per_sec, 2),
            "estimated_seconds": round(seconds, 0) if seconds else None,
            "distributed": config.distributed,
            "num_nodes": config.num_nodes,
        }

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------
    def _resolve_dataset(self, config: TrainingConfig) -> None:
        """Resolve ``dataset_id`` (if set) to an absolute dataset_path."""
        if not config.dataset_id or self.dataset_manager is None:
            return
        ds = self.dataset_manager.get(config.dataset_id)
        if ds is not None and ds.path:
            config.dataset_path = ds.path

    def submit_job(self, config: TrainingConfig) -> TrainingJob:
        """Validate config and queue a new training job.

        Returns the created TrainingJob.
        Raises ValueError if config is invalid.
        """
        # Resolve dataset_id -> dataset_path BEFORE validating so the path is set.
        self._resolve_dataset(config)

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
