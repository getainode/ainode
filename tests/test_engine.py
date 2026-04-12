"""Tests for ainode.engine.vllm_engine."""

import sys
from unittest.mock import patch, MagicMock
from ainode.core.config import NodeConfig
from ainode.core.gpu import GPUInfo
from ainode.engine.vllm_engine import VLLMEngine


def _make_engine(**config_overrides):
    config = NodeConfig(**config_overrides)
    return VLLMEngine(config)


def test_build_cmd_defaults():
    engine = _make_engine()
    with patch("ainode.engine.vllm_engine.detect_gpu", return_value=None):
        cmd = engine.build_cmd()

    assert sys.executable in cmd[0]
    assert "-m" in cmd
    assert "vllm.entrypoints.openai.api_server" in cmd
    assert "--model" in cmd
    assert "meta-llama/Llama-3.2-3B-Instruct" in cmd
    assert "--port" in cmd
    assert "8000" in cmd
    assert "--trust-remote-code" in cmd


def test_build_cmd_with_quantization():
    engine = _make_engine(quantization="awq")
    with patch("ainode.engine.vllm_engine.detect_gpu", return_value=None):
        cmd = engine.build_cmd()
    assert "--quantization" in cmd
    assert "awq" in cmd


def test_build_cmd_with_max_model_len():
    engine = _make_engine(max_model_len=4096)
    with patch("ainode.engine.vllm_engine.detect_gpu", return_value=None):
        cmd = engine.build_cmd()
    assert "--max-model-len" in cmd
    assert "4096" in cmd


def test_build_cmd_unified_memory_adds_dtype():
    gpu = GPUInfo(
        name="NVIDIA GB10", memory_total_mb=131072, memory_free_mb=120000,
        cuda_version="12.8", driver_version="570.86",
        compute_capability="10.0", unified_memory=True,
    )
    engine = _make_engine()
    with patch("ainode.engine.vllm_engine.detect_gpu", return_value=gpu):
        cmd = engine.build_cmd()
    assert "--dtype" in cmd
    assert "bfloat16" in cmd


def test_build_cmd_discrete_gpu_no_dtype():
    gpu = GPUInfo(
        name="RTX 4090", memory_total_mb=24576, memory_free_mb=20000,
        cuda_version="12.4", driver_version="550.00",
        compute_capability="8.9", unified_memory=False,
    )
    engine = _make_engine()
    with patch("ainode.engine.vllm_engine.detect_gpu", return_value=gpu):
        cmd = engine.build_cmd()
    assert "--dtype" not in cmd


def test_api_url():
    engine = _make_engine(api_port=9999)
    assert engine.api_url == "http://localhost:9999/v1"


def test_is_running_no_process():
    engine = _make_engine()
    assert engine.is_running() is False


def test_is_running_with_live_process():
    engine = _make_engine()
    proc = MagicMock()
    proc.poll.return_value = None
    engine.process = proc
    assert engine.is_running() is True


def test_is_running_with_dead_process():
    engine = _make_engine()
    proc = MagicMock()
    proc.poll.return_value = 1
    engine.process = proc
    assert engine.is_running() is False


def test_health_check_api_down():
    engine = _make_engine()
    health = engine.health_check()
    assert health["process_alive"] is False
    assert health["api_responding"] is False
    assert health["models_loaded"] == []


def test_ready_default_false():
    engine = _make_engine()
    assert engine.ready is False


def test_stop_no_process():
    """Stop is a no-op when no process exists."""
    engine = _make_engine()
    engine.stop()  # Should not raise
