"""GPU smoke test: a real LoRA run must produce a finite loss and NaN-free weights.

Skipped everywhere there is no CUDA GPU with torch/transformers/peft importable,
which is every CI runner and every dev Mac, so the suite stays green off-GPU.
On a training node (inside the train image) it is the only test that executes the
runner for real. Run it with:

    AINODE_SMOKE_BASE_MODEL=/path/to/Qwen2.5-0.5B-Instruct \\
        pytest tests/test_training_gpu.py -m gpu -q

``AINODE_SMOKE_BASE_MODEL`` must point at a small local model directory (a 0.5B
is plenty); without it the test skips rather than pulling weights from the hub.

The numbers it enforces are the ones the 2026-07-06 run failed: a finite loss at
every logged step, a gradient norm that is not astronomical, and an adapter with
zero non-finite values. That run logged loss 12.61 (the uniform-random baseline
for this vocab) with grad_norm 3.3e6, went NaN, saved a 100 percent NaN adapter
and reported COMPLETED.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

RUNNER = Path(__file__).resolve().parents[1] / "ainode" / "training" / "_run_training.py"


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _deps_available() -> bool:
    for mod in ("torch", "transformers", "datasets", "peft", "safetensors"):
        try:
            __import__(mod)
        except ImportError:
            return False
    return True


requires_gpu = pytest.mark.skipif(
    not _cuda_available() or not _deps_available(),
    reason="needs a CUDA GPU with torch/transformers/datasets/peft installed",
)
requires_base_model = pytest.mark.skipif(
    not os.environ.get("AINODE_SMOKE_BASE_MODEL"),
    reason="set AINODE_SMOKE_BASE_MODEL to a small local model directory",
)


def _write_dataset(path: Path, rows: int = 24) -> None:
    lines = []
    for i in range(rows):
        lines.append(json.dumps({
            "instruction": f"Give definition number {i} of a node in an AINode cluster.",
            "output": "A node is one machine running AINode, serving alone or with others.",
        }))
    path.write_text("\n".join(lines) + "\n")


@requires_gpu
@requires_base_model
def test_lora_run_produces_finite_loss_and_clean_weights(tmp_path):
    dataset = tmp_path / "smoke.jsonl"
    _write_dataset(dataset)
    output_dir = tmp_path / "output"
    config = {
        "base_model": os.environ["AINODE_SMOKE_BASE_MODEL"],
        "dataset_path": str(dataset),
        "output_dir": str(output_dir),
        "method": "lora",
        "num_epochs": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-4,
        "lora_rank": 8,
        "lora_alpha": 16,
        "max_seq_length": 256,
        "eval_split": 0,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--config", str(config_path)],
        capture_output=True, text=True, timeout=1800,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    assert "AINODE_ERROR" not in proc.stdout + proc.stderr

    # cuda must actually have been used: a CPU fallback is not a passing run.
    assert "cuda_available=True" in proc.stdout

    # Every progress line the UI draws must carry a finite, non-zero loss.
    losses = [
        json.loads(line.split("AINODE_PROGRESS:", 1)[1])
        for line in proc.stdout.splitlines() if "AINODE_PROGRESS:" in line
    ]
    step_losses = [p["loss"] for p in losses if p.get("step")]
    assert len(step_losses) >= 5, proc.stdout[-2000:]
    for loss in step_losses:
        assert loss == loss, "NaN loss"  # NaN is the only value that fails this
        assert 0.0 < loss < 100.0, f"implausible loss {loss}"

    # And the artifact itself must be clean.
    from ainode.training._run_training import _scan_saved_weights
    assert (output_dir / "adapter_model.safetensors").exists()
    assert _scan_saved_weights(output_dir) == []


@requires_gpu
@requires_base_model
def test_sdpa_on_this_gpu_either_trains_or_fails_loudly(tmp_path):
    """The default is eager because SDPA's memory-efficient kernels in the train
    image are built for sm80-sm100 and return zeros forward / NaN backward on a
    GB10. An explicit attn_implementation=sdpa is allowed, but if it diverges the
    run must FAIL with AINODE_ERROR:NAN_LOSS rather than save NaN weights."""
    dataset = tmp_path / "smoke.jsonl"
    _write_dataset(dataset)
    output_dir = tmp_path / "output"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "base_model": os.environ["AINODE_SMOKE_BASE_MODEL"],
        "dataset_path": str(dataset),
        "output_dir": str(output_dir),
        "method": "lora",
        "num_epochs": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_seq_length": 256,
        "eval_split": 0,
        "attn_implementation": "sdpa",
    }))

    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--config", str(config_path)],
        capture_output=True, text=True, timeout=1800,
    )
    combined = proc.stdout + proc.stderr
    if proc.returncode == 0:
        # SDPA is healthy on this GPU: then the weights must be clean.
        from ainode.training._run_training import _scan_saved_weights
        assert _scan_saved_weights(output_dir) == []
    else:
        assert "AINODE_ERROR:NAN_LOSS" in combined, combined[-4000:]
        assert not (output_dir / "adapter_model.safetensors").exists()
