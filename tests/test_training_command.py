"""Tests for TrainingJob._build_command — DDP dispatch + method selection."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from ainode.training.engine import TrainingConfig, TrainingJob


def _job(**overrides) -> TrainingJob:
    defaults = dict(
        base_model="test/model",
        dataset_path="user/data.jsonl",
    )
    defaults.update(overrides)
    return TrainingJob(TrainingConfig(**defaults))


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
