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
# Per-job state file, written at every status transition and read back at
# startup. The job registry is rebuilt from these.
STATUS_FILENAME = "status.json"


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


# Every training, quantize and merge job runs in a spawned GPU container: the
# slim orchestrator image has no torch/peft/datasets. Which image that is has
# three answers, tried in this order (see job_image_candidates):
#
#   1. AINODE_TRAIN_IMAGE / AINODE_QUANT_IMAGE, an operator's explicit override.
#   2. ghcr.io/getainode/ainode-train:<this AINode version>, the release image,
#      published by .github/workflows/publish-train-image.yml on a train-v* tag.
#   3. ainode-quant:0.17.0-t5, the tag the four Sparks built by hand in June
#      2026 and nothing else in the world has.
#
# Until 0.5.27 only (3) existed and CI published nothing, so every training,
# quantize and merge job on a node that had not built it died with docker exit
# 125 and a log the operator had to decode.
LOCAL_TRAIN_IMAGE = "ainode-quant:0.17.0-t5"
GHCR_TRAIN_REPO = "ghcr.io/getainode/ainode-train"


def ghcr_train_image() -> str:
    """The release train image for the AINode version this process is running."""
    try:
        from ainode import __version__
    except ImportError:  # pragma: no cover - the package importing itself
        return f"{GHCR_TRAIN_REPO}:latest"
    return f"{GHCR_TRAIN_REPO}:{__version__}"


def job_image_candidates(method: str = "lora") -> list[str]:
    """Container images a job of ``method`` should try, best first.

    A pure function of the environment and the running version, so the order is
    testable without a docker daemon. The quantize and training images are the
    same build; their env overrides are read in the order that matches the job."""
    env_keys = (
        ("AINODE_QUANT_IMAGE", "AINODE_TRAIN_IMAGE") if method == "quantize"
        else ("AINODE_TRAIN_IMAGE", "AINODE_QUANT_IMAGE")
    )
    candidates: list[str] = []
    for key in env_keys:
        value = (os.environ.get(key) or "").strip()
        if value and value not in candidates:
            candidates.append(value)
    for value in (ghcr_train_image(), LOCAL_TRAIN_IMAGE):
        if value not in candidates:
            candidates.append(value)
    return candidates


SECRET_ENV_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
# Config keys that must never be written to a job's on-disk config.
SECRET_CONFIG_KEYS = ("hf_token",)


def scrub_command(cmd: list[str]) -> list[str]:
    """Return ``cmd`` with any secret env value replaced by ``***``.

    Job logs are served verbatim by ``GET /api/training/jobs/{id}/logs``, and the
    launch line used to carry ``-e HF_TOKEN=<real token>`` into them. Scrub at the
    one place every launch line is logged rather than trusting each caller."""
    out: list[str] = []
    for arg in cmd:
        masked = arg
        for key in SECRET_ENV_KEYS:
            if arg.startswith(f"{key}=") and len(arg) > len(key) + 1:
                masked = f"{key}=***"
                break
        out.append(masked)
    return out


def _config_without_secrets(config: "TrainingConfig") -> dict:
    """Job config as a dict with secrets stripped, for writing to disk."""
    data = config.to_dict()
    for key in SECRET_CONFIG_KEYS:
        data.pop(key, None)
    return data


def _config_for_api(config: "TrainingConfig") -> dict:
    """Job config as a dict with secrets masked, for an API response.

    The job status is served by GET /api/training/jobs (auth is off by default),
    so the real token must not ride along. The key stays present, and truthy, so a
    caller can still see that the job HAS a token."""
    data = config.to_dict()
    for key in SECRET_CONFIG_KEYS:
        if data.get(key):
            data[key] = "***"
    return data


def _write_token_env_file(job_dir: Path, token: str) -> Optional[Path]:
    """Write a 0600 docker ``--env-file`` carrying the HF token, or None.

    The token used to travel two ways that both leaked it: serialized into the
    job's ``config.json`` (mode 644, and read back by the API) and spelled out in
    the ``-e HF_TOKEN=...`` argument of the launch line appended to the job log.
    An env file keeps it off both, readable only by the service user."""
    if not token:
        return None
    path = job_dir / "hf.env"
    # Create restricted, then write, never a world-readable window.
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        for key in SECRET_ENV_KEYS:
            fh.write(f"{key}={token}\n")
    return path


def _assert_image_present(image: str) -> None:
    """Raise RuntimeError unless ``image`` is present in the local docker daemon.

    Without this preflight a node that never built the training image answers
    every training, quantize and merge job with ``docker`` exit 125 and a log the
    operator has to decode."""
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "docker is not installed or not on PATH, so no GPU job container can "
            f"be spawned (needed image: {image})."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"docker did not answer within 30s while checking for image {image}."
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"Training image '{image}' is not present on this node, so the job "
            "cannot start. Build it with: "
            "docker build -f scripts/Dockerfile.quant -t "
            f"{image} . (or point AINODE_TRAIN_IMAGE / AINODE_QUANT_IMAGE at an "
            "image this node already has)."
        )


def _image_present(image: str) -> bool:
    """True when the local docker daemon already has ``image``.

    A broken docker (absent, or not answering) is NOT a missing image and is
    re-raised by the caller, so an operator never reads "no training image" when
    the real problem is the daemon."""
    _assert_image_present(image)
    return True


def resolve_job_image(method: str = "lora") -> str:
    """First candidate image this node actually has, else the preferred one.

    Never raises: it is called while BUILDING the launch command, and a command
    with a missing image in it is caught a moment later by
    ``assert_job_image_present``. Returning the preferred candidate keeps the
    argv (and every test that pins it) meaningful on a machine with no docker."""
    candidates = job_image_candidates(method)
    for image in candidates:
        try:
            if _image_present(image):
                return image
        except RuntimeError:
            continue
    return candidates[0]


def assert_job_image_present(method: str = "lora") -> str:
    """Return the image a ``method`` job will run, or raise naming every candidate.

    The preflight that replaced ``docker`` exit 125. The message names all three
    resolution steps, because which one an operator should fix depends on the
    node: a release node pulls the ghcr tag, a Spark built the local tag by hand,
    and a custom deploy sets the override."""
    candidates = job_image_candidates(method)
    for image in candidates:
        try:
            if _image_present(image):
                return image
        except RuntimeError as exc:
            message = str(exc)
            if "docker is not installed" in message or "did not answer" in message:
                raise
    tried = "\n".join(f"  {n}. {image}" for n, image in enumerate(candidates, 1))
    raise RuntimeError(
        "No training image is present on this node, so the job cannot start. "
        f"Tried, in order:\n{tried}\n"
        f"Fix it with one of: docker pull {ghcr_train_image()} (the release image, "
        "published by .github/workflows/publish-train-image.yml); docker build -f "
        f"scripts/Dockerfile.quant -t {LOCAL_TRAIN_IMAGE} . (about 22 GB, an hour "
        "on a Spark); or set AINODE_TRAIN_IMAGE / AINODE_QUANT_IMAGE to an image "
        "this node already has."
    )


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


# Where a resumed job sees the job directory it is resuming FROM.
RESUME_MOUNT = "/src"


def _resume_mount_for(checkpoint: Path) -> tuple[Path, str]:
    """Return (directory to mount at ``/src``, container path of the checkpoint).

    Resume handed the container the orchestrator's own checkpoint path with
    nothing mounted behind it, so HF answered "Can't find a valid checkpoint" and
    the button could not work in container mode. Mount the whole SOURCE JOB dir,
    not just the checkpoint: a checkpoint is resumed together with the sibling
    files of its run, and mounting one level up costs nothing (read-only).

    A checkpoint outside the jobs tree (a job with a hand-set ``output_dir``)
    falls back to its own parent, which is the least that still resolves."""
    try:
        relative = checkpoint.relative_to(JOBS_DIR)
    except ValueError:
        return checkpoint.parent, f"{RESUME_MOUNT}/{checkpoint.name}"
    parts = relative.parts
    if len(parts) < 2:
        return checkpoint.parent, f"{RESUME_MOUNT}/{checkpoint.name}"
    return JOBS_DIR / parts[0], RESUME_MOUNT + "/" + "/".join(parts[1:])


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
    # Attention kernel the runner loads the model with. "eager" is the default
    # because the memory-efficient SDPA kernels in the training image are built
    # for sm80-sm100 and silently produce zeros forward / NaN backward on a GB10
    # (sm121); see ainode/training/_run_training.py. "auto" hands the choice
    # back to transformers.
    attn_implementation: str = "eager"
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
        self._status: JobStatus = JobStatus.PENDING
        self.progress: float = 0.0
        self.current_epoch: int = 0
        self.current_loss: Optional[float] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.logs: collections.deque[str] = collections.deque(maxlen=5000)
        self._process: Optional[subprocess.Popen] = None
        self._monitor_task: Optional[asyncio.Task] = None
        # Set when this job was rebuilt from its directory rather than submitted
        # in this process, with a sentence saying what the rebuild concluded and
        # from what. Served with the job so nothing reads a reconstruction as a
        # measurement.
        self.note: Optional[str] = None
        self.restored: bool = False
        # Resolved container image, memoized: resolution asks the docker daemon,
        # and the command builder and the preflight must agree on the answer.
        self._image: Optional[str] = None
        # Called with this job when its monitored process exits, so the manager
        # can release the GPU slot and start the next queued job without waiting
        # for a submit (until 0.5.27 the queue only advanced from an HTTP call).
        self._on_exit = None
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

    @property
    def status(self) -> JobStatus:
        return self._status

    @status.setter
    def status(self, value: JobStatus) -> None:
        """Every status change writes the job's status file.

        A setter rather than a call at each transition because there are eight
        places that move a job's status (engine, monitor, merge runner) and one of
        them forgetting is exactly how the registry would drift from the disk
        again."""
        changed = value != self._status
        self._status = value
        if changed:
            self._write_status()

    def _write_status(self) -> None:
        """Persist this job to ``<job dir>/status.json``.

        The registry is rebuilt from these files at startup. Until 0.5.27 the job
        table lived only in ``TrainingManager._jobs``, so a restart emptied the
        Runs table, zeroed the stats tiles and 404'd merge, resume, logs and
        artifact download for every job that came before, while the job dirs sat
        on disk the whole time.

        Secrets are stripped exactly as ``config.json`` strips them: this file
        lives in a directory the job API reads."""
        payload = self.get_status()
        payload["config"] = _config_without_secrets(self.config)
        payload["written_at"] = time.time()
        payload["schema"] = 1
        try:
            self._job_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._job_dir / (STATUS_FILENAME + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self._job_dir / STATUS_FILENAME)
        except OSError as exc:
            # A job that cannot write its status file still runs; it just will not
            # survive a restart. Never take the run down for it.
            self.logs.append(f"WARNING: could not write {STATUS_FILENAME}: {exc}")

    async def start(self) -> None:
        """Launch the training subprocess.

        The job is marked RUNNING only once ``Popen`` has actually returned a
        process. Setting RUNNING first (what this did until 0.5.26) left a
        phantom RUNNING job with no process behind it whenever the command could
        not be built. A DDP submit in container mode did exactly that, and the
        phantom then blocked every later job in the queue.
        """
        if self.status != JobStatus.PENDING:
            raise RuntimeError(f"Cannot start job in '{self.status.value}' state")

        self._log(f"Starting {self.config.method} training on {self.config.base_model}")

        # Write config to job directory for the training script. The HF token is
        # deliberately NOT serialized: this file is world-readable in the job dir
        # and served indirectly through the job API. The spawned container gets
        # the token through a 0600 env file instead.
        config_path = self._job_dir / "config.json"
        config_path.write_text(json.dumps(_config_without_secrets(self.config), indent=2))

        # Build the training command OFF the event loop. _build_command can
        # shell out to a blocking `pip download` (peft wheel vendoring, up to a
        # 120s timeout) on a cold cache — running that inline on aiohttp's single
        # loop would freeze every concurrent request (live inference proxying
        # included) until it returns. run_in_executor keeps the loop responsive.
        loop = asyncio.get_event_loop()
        try:
            cmd = await loop.run_in_executor(None, self._build_command, config_path)
            # Never log a token: this log is served by GET /api/training/jobs/{id}/logs.
            self._log(f"Command: {' '.join(scrub_command(cmd))}")

            # Preflight the image before Popen. Without it a missing train image is
            # a docker exit 125 the operator has to decode from the job log.
            if cmd and cmd[0] == "docker":
                await loop.run_in_executor(None, self._assert_job_image)

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
        except Exception as exc:
            self.status = JobStatus.FAILED
            self.end_time = time.time()
            self._log(f"Failed to start: {exc}")
            self._write_status()
            raise

        # Popen returned: the job really is running now.
        self.status = JobStatus.RUNNING
        self.start_time = time.time()
        self._write_status()
        self._monitor_task = asyncio.create_task(self._monitor())

    def _job_image(self) -> str:
        """Container image this job's spawned container runs, memoized.

        Resolution asks the docker daemon which candidate is present, so the image
        baked into the launch command and the one the preflight checks must be the
        same answer, not two probes that could disagree."""
        if self._image is None:
            self._image = resolve_job_image(self.config.method)
        return self._image

    def _assert_job_image(self) -> str:
        """Preflight this job's image, raising if the node has none of them."""
        self._image = assert_job_image_present(self.config.method)
        return self._image

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
            "config": _config_for_api(self.config),
            # Set only on a job rebuilt from its directory: what the rebuild
            # concluded and from what. A reader must be able to tell a recorded
            # status from a reconstructed one.
            "restored": self.restored,
            "note": self.note,
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
        # container from the resolved train image instead, same pattern as quantize.
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
        # The token goes in through a 0600 env file, never as an -e argument: the
        # launch line is appended to a job log the API serves.
        env_file = _write_token_env_file(self._job_dir, token)
        if env_file:
            cmd += ["--env-file", str(env_file)]
        cmd += [
            self._job_image(), "python3", "/opt/ainode/run_quant.py",
            "--config", "/job/config.json",
        ]
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
        container_cfg = _config_without_secrets(c)
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
        # Resume: the checkpoint belongs to ANOTHER job, whose directory is not
        # mounted and whose path is an orchestrator path this container cannot
        # resolve. Mount that job dir read-only at /src and rewrite the path into
        # it. Without both halves HF raises "Can't find a valid checkpoint" and
        # resume simply cannot work in container mode (it never did until 0.5.27).
        resume_mount: list[str] = []
        checkpoint = (c._resume_from_checkpoint or "").strip()
        if checkpoint:
            source_dir, container_checkpoint = _resume_mount_for(Path(checkpoint))
            container_cfg["_resume_from_checkpoint"] = container_checkpoint
            resume_mount = ["-v", f"{_host_path(str(source_dir))}:{RESUME_MOUNT}:ro"]

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
            *resume_mount,                                     # RO: source job dir of a resume
            "-e", "HF_HUB_CACHE=/ainode-models/hf-cache",      # persist HF pulls into the store
            "-e", "AINODE_HOME=/job",                          # runner config fallback (relative datasets)
        ]
        # 0600 env file, not -e: the launch line lands in a log the API serves.
        env_file = _write_token_env_file(self._job_dir, token)
        if env_file:
            cmd += ["--env-file", str(env_file)]
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
            self._job_image(), "sh", "-c",
            f"{prep} ; python3 /job/_run_training.py --config /job/config.container.json",
        ]
        return cmd

    async def _monitor(self) -> None:
        """Read subprocess output and update progress.

        Also the one place that knows a job has ENDED, so it is where the queue
        advances: until 0.5.27 ``start_next`` only ran from an HTTP submit or
        resume, so a queued job sat pending until somebody submitted another one.
        """
        proc = self._process
        if proc is None or proc.stdout is None:
            return

        cancelled = False
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
            cancelled = True
        finally:
            self.end_time = time.time()
            # The token env file exists only for the life of the container.
            try:
                (self._job_dir / "hf.env").unlink(missing_ok=True)
            except OSError:
                pass
            # end_time and the final progress land after the status transition
            # that wrote the file, so write the finished record once more.
            self._write_status()

        # A cancel already released the slot through cancel_job, and awaiting
        # anything in a task that is being cancelled is asking for trouble.
        if not cancelled:
            await self._notify_exit()

    async def _notify_exit(self) -> None:
        """Tell the manager this job's process is gone, so the queue can move on."""
        callback = self._on_exit
        if callback is None:
            return
        try:
            await callback(self)
        except Exception as exc:
            # A job that ended is ended: a failure to start the NEXT one is
            # reported on this job's log, never raised out of the monitor task.
            self._log(f"WARNING: could not start the next queued job: {exc}")

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
    the merge runs in the resolved train image. Copies the self-contained _run_merge.py into
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
    # No hf_token in this file: the runner reads it from the env (see the 0600
    # env file below); merge_config.json sits in a job dir the API can serve.
    merge_cfg = {
        "base_model": _resolve_base_model_mount(base_model) or base_model,
        "adapter_dir": "/adapter",
        "output_dir": f"/out/{merged_dir.name}",
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
    env_file = _write_token_env_file(job_dir, token)
    if env_file:
        cmd += ["--env-file", str(env_file)]
    # ponytail: peft pip-shim — bake it into the next train-image build and drop this.
    # Vendor the (pure-python) peft wheel so the merge runs offline; ';' not '&&'
    # so a pip hiccup never blocks the runner (broken DNS killed a live merge here).
    peft_step = _pip_install_step("peft", job_dir, vendor=True)
    cmd += [
        merge_job._job_image(), "sh", "-c",
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
        # Truthful as of 0.5.27: the runner has a conversations branch that renders
        # each row with the tokenizer's own chat template (from/value and
        # role/content both accepted). Before that this tile advertised a shape
        # tokenization raised IndexError on, and it is what AutoData writes.
        "description": (
            "Multi-turn conversation tuning on human/gpt turns, rendered with the "
            "model's own chat template. This is what AutoData writes."
        ),
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
]

# Removed rather than shipped as decoration (0.5.26). Both offered a run the
# product cannot do, and the DPO one was actively harmful:
#
#   dpo-preference  : there is no DPO trainer here. The runner would have
#                     space-joined prompt/chosen/rejected into one SFT string and
#                     trained the model ON the rejected answer.
#   distributed-ddp : DDP has no launch path at all (multi-node raises in
#                     container-spawn mode, and the host-venv torchrun path has
#                     no rendezvous), so the tile only ever produced a failure.
#
# Put either back only together with an implementation and a proven run.
RETIRED_TEMPLATE_IDS = ("dpo-preference", "distributed-ddp")


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


# ---------------------------------------------------------------------------
# Rebuilding the job registry from disk
# ---------------------------------------------------------------------------

# Files that say a run actually produced weights. Checked in the job's output dir
# for a job dir written before status.json existed.
ARTIFACT_MARKERS = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_config.json",
    "model.safetensors",
    "pytorch_model.bin",
)


def _completed_artifact(job_dir: Path, config: TrainingConfig) -> Optional[str]:
    """Path of the artifact that says this job finished, or None.

    Best-effort, and only used for a job directory written before there was a
    status file. A LoRA / full run leaves its weights under ``output/``; a merge
    leaves a ``merged*/`` directory; a quantize job writes into the models store
    rather than its job dir, so its checkpoint is looked up by ``out_slug``."""
    if config.method == "quantize":
        if not config.out_slug:
            return None
        checkpoint = AINODE_HOME / "models" / config.out_slug
        return str(checkpoint) if (checkpoint / "config.json").exists() else None

    candidates = []
    if config.output_dir:
        candidates.append(Path(config.output_dir))
    candidates.append(job_dir / "output")
    for out in candidates:
        if not out.is_dir():
            continue
        for name in ARTIFACT_MARKERS:
            if (out / name).exists():
                return str(out / name)
        # A checkpoint-N subdir is a run that got somewhere, even if the final
        # save never happened.
        for pattern in ("*.safetensors", "checkpoint-*/*.safetensors"):
            for hit in sorted(out.glob(pattern)):
                return str(hit)
    for merged in sorted(job_dir.glob("merged*")):
        if (merged / "config.json").exists():
            return str(merged)
    return None


def _newest_mtime(job_dir: Path) -> Optional[float]:
    """Newest modification time in the job dir, one level deep.

    The only timestamp a legacy job dir really carries. Used as ``end_time`` so
    the Runs table can sort; ``start_time`` stays unset, because a duration
    nobody recorded must not be invented (``stats()`` counts GPU hours from it)."""
    newest: Optional[float] = None
    try:
        entries = [job_dir, *job_dir.iterdir()]
    except OSError:
        return None
    for entry in entries:
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if newest is None or mtime > newest:
            newest = mtime
    return newest


def _config_data_from_disk(job_dir: Path) -> Optional[dict]:
    """Config of a job dir with no status file, from whatever it wrote.

    ``config.json`` is written by ``start()``; a merge job only ever writes
    ``merge_config.json``, so reconstruct the shape the merge path submits."""
    config_path = job_dir / "config.json"
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text())
            if isinstance(data, dict) and data.get("base_model"):
                return data
        except (OSError, json.JSONDecodeError):
            return None
    merge_path = job_dir / "merge_config.json"
    if merge_path.exists():
        try:
            data = json.loads(merge_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if isinstance(data, dict):
            return {
                "base_model": data.get("base_model") or "unknown",
                "dataset_path": "__merge__",
                "method": "lora",
                "output_dir": data.get("output_dir"),
                "run_name": f"merge-{job_dir.name}",
                "description": "merge job, rebuilt from merge_config.json",
            }
    return None


def load_job_from_dir(job_dir: Path) -> Optional[TrainingJob]:
    """Rebuild one job from its directory, or None if that directory is not a job.

    A directory with neither a status file nor a config is NOT a run: the suite
    left 6,000 empty job dirs under the developer's ~/.ainode before 0.5.26, and
    resurrecting those as failed jobs would be a fabricated history.

    Three cases:

    * a status file with a terminal status: restored as recorded;
    * a status file saying RUNNING: that process died with the restart, so the
      job is FAILED with a note (a phantom RUNNING job also blocks the queue);
    * no status file: a best-effort status from what is on disk, COMPLETED if
      the run left weights behind, FAILED otherwise, always with a note saying
      so. `start_time` stays unset so no invented duration reaches GPU hours.
    """
    if not job_dir.is_dir():
        return None
    data: dict = {}
    status_path = job_dir / STATUS_FILENAME
    if status_path.exists():
        try:
            loaded = json.loads(status_path.read_text())
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, json.JSONDecodeError):
            data = {}

    config_data = data.get("config") or _config_data_from_disk(job_dir)
    if not isinstance(config_data, dict) or not config_data.get("base_model"):
        return None
    try:
        config = TrainingConfig.from_dict(config_data)
    except (TypeError, ValueError):
        return None

    job = TrainingJob(config, job_id=job_dir.name)
    job.restored = True
    recorded = data.get("status")
    note: Optional[str]

    if recorded:
        try:
            status = JobStatus(recorded)
        except ValueError:
            status = JobStatus.FAILED
        note = data.get("note")
        job.progress = float(data.get("progress") or 0.0)
        job.current_epoch = int(data.get("current_epoch") or 0)
        loss = data.get("current_loss")
        job.current_loss = float(loss) if isinstance(loss, (int, float)) else None
        job.start_time = data.get("start_time")
        job.end_time = data.get("end_time")
        if status in (JobStatus.RUNNING, JobStatus.PENDING):
            # The process behind it is gone: this instance did not start it.
            note = (
                f"was {status.value} when AINode last stopped; the process did not "
                "survive the restart, so this run is recorded as failed"
            )
            status = JobStatus.FAILED
            job.end_time = job.end_time or _newest_mtime(job_dir)
    else:
        artifact = _completed_artifact(job_dir, config)
        if artifact:
            status = JobStatus.COMPLETED
            note = (
                "rebuilt from disk: no status file (this job ran before AINode "
                f"0.5.27 wrote one), reported completed because {artifact} exists"
            )
            job.progress = 100.0
        else:
            status = JobStatus.FAILED
            note = (
                "rebuilt from disk: no status file (this job ran before AINode "
                "0.5.27 wrote one) and no weights in its output dir, so it is "
                "reported failed"
            )
            if config.dataset_path == "__merge__":
                # A merge before 0.5.27 wrote into the SOURCE run's directory and
                # its merge_config.json records only the container path, so its
                # own job dir cannot say whether it worked.
                note += (
                    ". This is a merge job, and a merge before 0.5.27 wrote into "
                    "the source run's directory, so its own job dir cannot show "
                    "the merged model even if the merge succeeded"
                )
        job.end_time = _newest_mtime(job_dir)

    job.note = note
    job.logs.append(
        f"[rebuilt] {job.job_id} restored from {job_dir}"
        + (f": {note}" if note else "")
    )
    job.logs.append(
        "[rebuilt] the live log of this run belonged to the process that wrote it "
        "and is not on disk"
    )
    # Assigning through the property persists the corrected record, so the next
    # restart reads a real status file instead of guessing again.
    job.status = status
    job._write_status()
    return job


def load_jobs_from_disk(jobs_dir: Optional[Path] = None) -> list[TrainingJob]:
    """Rebuild every job under ``jobs_dir`` (default ``JOBS_DIR``), oldest first."""
    root = jobs_dir or JOBS_DIR
    if not root.is_dir():
        return []
    jobs: list[TrainingJob] = []
    for entry in sorted(root.iterdir()):
        try:
            job = load_job_from_dir(entry)
        except Exception:  # a single unreadable job dir must not break startup
            job = None
        if job is not None:
            jobs.append(job)
    jobs.sort(key=lambda j: (j.start_time or j.end_time or 0.0))
    return jobs


class TrainingManager:
    """Manage training jobs — one active at a time (GPU shared with inference)."""

    def __init__(self, dataset_manager=None, rehydrate: bool = True):
        self._jobs: dict[str, TrainingJob] = {}
        self._queue: list[str] = []  # job_ids in queue order
        self._active_job_id: Optional[str] = None
        self.dataset_manager = dataset_manager
        # The registry is the job dirs on disk, not this process's memory. A
        # manager built without rehydration is a manager whose history is gone,
        # which is what every restart used to do.
        if rehydrate:
            self.rebuild_from_disk()

    def rebuild_from_disk(self) -> int:
        """Load every job under JOBS_DIR into the registry. Returns how many.

        Called at construction. Nothing here is started or adopted: a job that was
        RUNNING when the process stopped comes back FAILED, because its process
        went with the restart (and a phantom RUNNING job blocks the queue)."""
        found = 0
        for job in load_jobs_from_disk():
            if job.job_id in self._jobs:
                continue
            self._jobs[job.job_id] = job
            found += 1
        return found

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
        # Queued is a state worth surviving a restart: write the record now rather
        # than at the first transition.
        job._write_status()
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
                # The monitor calls this back when the process exits, which is
                # what makes the queue advance on its own.
                job._on_exit = self._job_exited
                try:
                    await job.start()
                except Exception:
                    # A job that never started is not the active job. Claiming the
                    # slot first (what this did until 0.5.26) wedged the queue: the
                    # phantom "active" id made every later start_next() return None.
                    self._active_job_id = None
                    raise
                self._active_job_id = job_id
                return job
            else:
                self._queue.pop(0)  # Skip cancelled/missing jobs

        return None

    async def _job_exited(self, job: "TrainingJob") -> None:
        """Release the GPU slot a finished job held and start the next queued one.

        Called from the job's own monitor task when its process exits. Before
        0.5.27 ``start_next`` ran only from ``POST /api/training/jobs`` and the
        resume route, so a queue with two jobs in it ran the first and then sat
        there: the second only started if a human submitted a third."""
        if self._active_job_id == job.job_id:
            self._active_job_id = None
        await self.start_next()

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
