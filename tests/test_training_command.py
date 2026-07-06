"""Tests for TrainingJob._build_command — DDP dispatch + method selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

from ainode.training import engine as tr_engine
from ainode.training.engine import (
    TrainingConfig,
    TrainingJob,
    _resolve_base_model_mount,
    build_merge_command,
)


def _job(**overrides) -> TrainingJob:
    defaults = dict(
        base_model="test/model",
        dataset_path="user/data.jsonl",
    )
    defaults.update(overrides)
    return TrainingJob(TrainingConfig(**defaults))


def _container_job(monkeypatch, tmp_path, **overrides) -> TrainingJob:
    """A TrainingJob wired to a tmp AINODE_HOME for container-command tests."""
    monkeypatch.setenv("AINODE_HOST_HOME", str(tmp_path / "host"))
    monkeypatch.setenv("AINODE_NO_WHEEL_FETCH", "1")  # hermetic: no pip download
    monkeypatch.setattr(tr_engine, "AINODE_HOME", tmp_path)
    monkeypatch.setattr(tr_engine, "JOBS_DIR", tmp_path / "training" / "jobs")
    defaults = dict(base_model="org/model", dataset_path="user/data.jsonl", method="lora")
    defaults.update(overrides)
    return TrainingJob(TrainingConfig(**defaults))


def _seed_peft_wheel(tmp_path) -> str:
    wheels = tmp_path / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    name = "peft-0.11.0-py3-none-any.whl"
    (wheels / name).write_bytes(b"fake-wheel")
    return name


# ---- D1: on-disk base_model slug → mounted path -----------------------------

def test_container_command_rewrites_ondisk_slug_base_model(monkeypatch, tmp_path):
    """The GUI submits the on-disk slug; the runner must load it from the mount,
    not choke AutoTokenizer on the '--' (HFValidationError)."""
    slug = "qwen--qwen2.5-0.5b-instruct"
    (tmp_path / "models" / slug).mkdir(parents=True)
    job = _container_job(monkeypatch, tmp_path, base_model=slug)
    job._build_container_command()
    cfg = json.loads((job._job_dir / "config.container.json").read_text())
    assert cfg["base_model"] == "/ainode-models/" + slug


def test_container_command_rewrites_repo_id_when_slug_on_disk(monkeypatch, tmp_path):
    """A canonical HF repo id whose org--name dir exists on disk also maps to the mount."""
    slug = "Qwen--Qwen2.5-0.5B-Instruct"
    (tmp_path / "models" / slug).mkdir(parents=True)
    job = _container_job(monkeypatch, tmp_path, base_model="Qwen/Qwen2.5-0.5B-Instruct")
    job._build_container_command()
    cfg = json.loads((job._job_dir / "config.container.json").read_text())
    assert cfg["base_model"] == "/ainode-models/" + slug


def test_container_command_passes_through_hub_repo_id(monkeypatch, tmp_path):
    """A plain hub repo id with no local copy is left untouched (loads from hub)."""
    job = _container_job(monkeypatch, tmp_path, base_model="meta-llama/Llama-3.2-3B-Instruct")
    job._build_container_command()
    cfg = json.loads((job._job_dir / "config.container.json").read_text())
    assert cfg["base_model"] == "meta-llama/Llama-3.2-3B-Instruct"


# ---- D1b: on-disk layout coverage — all 4 layouts the registry tracks -------

def test_resolve_mount_hf_cache_hub_snapshot(monkeypatch, tmp_path):
    """A model cached via the HF cache layout (hub/models--org--name/snapshots/
    <hash>) — populated by the eugr distributed backend's HF_HOME=models_dir —
    must resolve to the SNAPSHOT dir, not fall through to a live hub round-trip."""
    monkeypatch.setattr(tr_engine, "AINODE_HOME", tmp_path)
    hf_slug = "models--Qwen--Qwen2.5-0.5B-Instruct"
    snap = tmp_path / "models" / "hub" / hf_slug / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    assert (_resolve_base_model_mount("Qwen/Qwen2.5-0.5B-Instruct")
            == "/ainode-models/hub/" + hf_slug + "/snapshots/abc123")


def test_resolve_mount_hf_cache_out_of_band_snapshot(monkeypatch, tmp_path):
    """Out-of-band HF_HOME layout: hf-cache/hub/models--org--name/snapshots/<hash>."""
    monkeypatch.setattr(tr_engine, "AINODE_HOME", tmp_path)
    hf_slug = "models--meta-llama--Llama-3.2-3B-Instruct"
    snap = tmp_path / "models" / "hf-cache" / "hub" / hf_slug / "snapshots" / "deadbeef"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    assert (_resolve_base_model_mount("meta-llama/Llama-3.2-3B-Instruct")
            == "/ainode-models/hf-cache/hub/" + hf_slug + "/snapshots/deadbeef")


def test_resolve_mount_flat_hf_models_snapshot(monkeypatch, tmp_path):
    """Flat HF cache dir at the store root: models--org--name/snapshots/<hash>."""
    monkeypatch.setattr(tr_engine, "AINODE_HOME", tmp_path)
    hf_slug = "models--Qwen--Qwen2.5-0.5B-Instruct"
    snap = tmp_path / "models" / hf_slug / "snapshots" / "cafef00d"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    assert (_resolve_base_model_mount("Qwen/Qwen2.5-0.5B-Instruct")
            == "/ainode-models/" + hf_slug + "/snapshots/cafef00d")


def test_resolve_mount_absent_returns_none(monkeypatch, tmp_path):
    """A hub repo id with no local copy still passes through (None → load from hub)."""
    monkeypatch.setattr(tr_engine, "AINODE_HOME", tmp_path)
    (tmp_path / "models").mkdir(parents=True)
    assert _resolve_base_model_mount("meta-llama/Llama-3.2-3B-Instruct") is None


# ---- D2: offline peft — vendored wheel + tolerant fallback ------------------

def test_container_command_vendors_peft_wheel_offline(monkeypatch, tmp_path):
    wheel = _seed_peft_wheel(tmp_path)
    job = _container_job(monkeypatch, tmp_path)
    cmd = job._build_container_command()
    sh = cmd[-1]
    assert f"pip install -q --no-index --find-links /job /job/{wheel}" in sh
    assert "|| pip install -q --no-deps peft" in sh          # fallback if offline install fails
    assert " ; python3 /job/_run_training.py" in sh          # ';' not '&&' — never blocks runner
    assert " && python3 /job/_run_training.py" not in sh
    assert sh.startswith("python3 -c 'import peft'")          # tolerant: skip if already baked in
    assert (job._job_dir / wheel).exists()                    # wheel copied into the job dir


def test_container_command_falls_back_to_online_pip_without_wheel(monkeypatch, tmp_path):
    """No vendored wheel + fetch disabled → plain online pip, still ';'-joined."""
    job = _container_job(monkeypatch, tmp_path)  # no wheel seeded
    cmd = job._build_container_command()
    sh = cmd[-1]
    assert "--no-index" not in sh
    assert "pip install -q --no-deps peft" in sh
    assert " ; python3 /job/_run_training.py" in sh
    assert sh.startswith("python3 -c 'import peft'")


def test_container_command_qlora_adds_bitsandbytes_step(monkeypatch, tmp_path):
    job = _container_job(monkeypatch, tmp_path, method="qlora")
    sh = job._build_container_command()[-1]
    assert "import bitsandbytes" in sh
    assert "pip install -q --no-deps bitsandbytes" in sh


def test_merge_command_rewrites_slug_and_vendors_peft(monkeypatch, tmp_path):
    slug = "qwen--qwen2.5-0.5b-instruct"
    (tmp_path / "models" / slug).mkdir(parents=True)
    wheel = _seed_peft_wheel(tmp_path)
    merge_job = _container_job(monkeypatch, tmp_path)
    cmd = build_merge_command(
        merge_job, slug, tmp_path / "adapter", tmp_path / "out" / "merged"
    )
    cfg = json.loads((merge_job._job_dir / "merge_config.json").read_text())
    assert cfg["base_model"] == "/ainode-models/" + slug
    sh = cmd[-1]
    assert f"pip install -q --no-index --find-links /job /job/{wheel}" in sh
    assert " ; python3 /job/_run_merge.py" in sh
    assert " && python3 /job/_run_merge.py" not in sh


def test_lora_single_gpu_uses_plain_python():
    """LoRA on one GPU does NOT need torchrun — it bloats logs."""
    job = _job(method="lora")
    with patch("ainode.training.engine._detect_local_gpu_count", return_value=1):
        cmd = job._build_command(Path("/tmp/cfg.json"))
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["-m", "ainode.training._run_training"]
    assert "torch.distributed.run" not in cmd


def test_qlora_single_gpu_uses_plain_python():
    job = _job(method="qlora")
    with patch("ainode.training.engine._detect_local_gpu_count", return_value=1):
        cmd = job._build_command(Path("/tmp/cfg.json"))
    assert "torch.distributed.run" not in cmd


def test_full_single_gpu_still_plain_python():
    """Even full fine-tune stays single-process when there's only one GPU."""
    job = _job(method="full")
    with patch("ainode.training.engine._detect_local_gpu_count", return_value=1):
        cmd = job._build_command(Path("/tmp/cfg.json"))
    assert "torch.distributed.run" not in cmd


def test_full_multi_gpu_switches_to_torchrun():
    """Full fine-tune with 4 GPUs -> torchrun --nproc_per_node=4."""
    job = _job(method="full")
    with patch("ainode.training.engine._detect_local_gpu_count", return_value=4):
        cmd = job._build_command(Path("/tmp/cfg.json"))
    assert "torch.distributed.run" in cmd
    assert "--nproc_per_node=4" in cmd
    assert "--nnodes=1" in cmd


def test_distributed_flag_forces_ddp_even_for_lora():
    """distributed=True overrides single-GPU fast path."""
    job = _job(method="lora", distributed=True)
    with patch("ainode.training.engine._detect_local_gpu_count", return_value=1):
        cmd = job._build_command(Path("/tmp/cfg.json"))
    assert "torch.distributed.run" in cmd


def test_multi_node_sets_nnodes():
    """num_nodes > 1 wires --nnodes correctly."""
    job = _job(method="full", distributed=True, num_nodes=3)
    with patch("ainode.training.engine._detect_local_gpu_count", return_value=2):
        cmd = job._build_command(Path("/tmp/cfg.json"))
    assert "--nnodes=3" in cmd
    assert "--nproc_per_node=2" in cmd


def test_command_always_points_at_correct_config_path():
    cfg_path = Path("/tmp/specific-config-1234.json")
    job = _job(method="lora")
    with patch("ainode.training.engine._detect_local_gpu_count", return_value=1):
        cmd = job._build_command(cfg_path)
    assert cmd[-2:] == ["--config", str(cfg_path)]


def test_detect_gpu_count_falls_back_to_one_without_torch():
    """No torch installed -> safe fallback of 1."""
    from ainode.training import engine as tr_engine

    # Simulate ImportError from torch
    with patch.dict(sys.modules, {"torch": None}):
        assert tr_engine._detect_local_gpu_count() == 1
