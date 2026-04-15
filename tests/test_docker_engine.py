"""Tests for ainode.engine.docker_engine.DockerEngine (v0.4.0 container-native).

The v0.3.x DockerEngine drove ``docker compose`` from the host. v0.4.0 runs
*inside* the AINode unified image and either spawns ``vllm serve`` directly
(solo) or shells out to eugr's ``launch-cluster.sh`` (head). These tests
mock ``subprocess`` so they run anywhere — no docker, no GPU, no eugr image
required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ainode.core.config import NodeConfig
from ainode.engine import docker_engine as de


def _cfg(**overrides) -> NodeConfig:
    defaults = dict(
        node_id="test-node",
        model="test/model",
        api_port=8000,
        web_port=3000,
        gpu_memory_utilization=0.5,
        distributed_mode="solo",
        peer_ips=[],
        ssh_user="test",
        cluster_interface="eno1",
    )
    defaults.update(overrides)
    return NodeConfig(**defaults)


def test_build_engine_returns_docker_engine():
    assert isinstance(de.build_engine(_cfg()), de.DockerEngine)


def test_start_rejects_unknown_mode():
    engine = de.DockerEngine(_cfg(distributed_mode="worker"))
    with pytest.raises(de.DockerEngineError):
        engine.start()


def test_start_distributed_refuses_in_solo_mode():
    engine = de.DockerEngine(_cfg(distributed_mode="solo", peer_ips=["10.0.0.2"]))
    with pytest.raises(de.DockerEngineError):
        engine.start_distributed()


def test_start_distributed_requires_peer_ips():
    engine = de.DockerEngine(_cfg(distributed_mode="head", peer_ips=[]))
    with pytest.raises(de.DockerEngineError):
        engine.start_distributed()


def test_start_distributed_requires_launcher():
    engine = de.DockerEngine(_cfg(distributed_mode="head", peer_ips=["10.0.0.2"]))
    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(de.DockerEngineError):
            engine.start_distributed()


def test_build_solo_cmd_minimum():
    cfg = _cfg(model="Qwen/Qwen2.5-1.5B-Instruct")
    with patch("ainode.engine.docker_engine.detect_gpu", return_value=None):
        cmd = de.DockerEngine(cfg)._build_solo_cmd()
    assert cmd[:3] == ["vllm", "serve", "Qwen/Qwen2.5-1.5B-Instruct"]
    assert "--host" in cmd and "0.0.0.0" in cmd
    assert "--port" in cmd and "8000" in cmd


def test_build_solo_cmd_adds_bfloat16_for_unified_memory():
    gpu = MagicMock(unified_memory=True)
    with patch("ainode.engine.docker_engine.detect_gpu", return_value=gpu):
        cmd = de.DockerEngine(_cfg())._build_solo_cmd()
    assert "--dtype" in cmd
    assert cmd[cmd.index("--dtype") + 1] == "bfloat16"


def test_build_solo_cmd_respects_quantization_and_max_len():
    cfg = _cfg(quantization="awq", max_model_len=4096, trust_remote_code=True)
    with patch("ainode.engine.docker_engine.detect_gpu", return_value=None):
        cmd = de.DockerEngine(cfg)._build_solo_cmd()
    assert "--quantization" in cmd and "awq" in cmd
    assert "--max-model-len" in cmd and "4096" in cmd
    assert "--trust-remote-code" in cmd


def test_build_env_sets_nccl_interface_from_config():
    cfg = _cfg(cluster_interface="enp1s0f0np0")
    with patch("ainode.engine.docker_engine.shutil.which", return_value=None):
        env = de.DockerEngine(cfg)._build_env()
    assert env["NCCL_SOCKET_IFNAME"] == "enp1s0f0np0"
    assert env["GLOO_SOCKET_IFNAME"] == "enp1s0f0np0"
    assert env["UCX_NET_DEVICES"] == "enp1s0f0np0"


def test_build_env_defaults_nccl_ib_disable_zero():
    with patch("ainode.engine.docker_engine.shutil.which", return_value=None):
        env = de.DockerEngine(_cfg())._build_env()
    assert env["NCCL_IB_DISABLE"] == "0"


def test_tp_size_one_plus_peers():
    assert de.DockerEngine(_cfg(peer_ips=["a"]))._tp_size() == 2
    assert de.DockerEngine(_cfg(peer_ips=["a", "b", "c"]))._tp_size() == 4


def test_start_solo_launches_subprocess():
    engine = de.DockerEngine(_cfg())
    with patch("ainode.engine.docker_engine.subprocess.Popen") as popen, \
         patch("ainode.engine.docker_engine.detect_gpu", return_value=None), \
         patch("ainode.engine.docker_engine.shutil.which", return_value=None):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        popen.return_value = mock_proc
        assert engine.start_solo() is True
        popen.assert_called_once()
        invoked_cmd = popen.call_args[0][0]
        assert invoked_cmd[:2] == ["vllm", "serve"]


def test_is_running_requires_live_process():
    engine = de.DockerEngine(_cfg())
    assert engine.is_running() is False
    engine.process = MagicMock()
    engine.process.poll.return_value = None
    assert engine.is_running() is True
    engine.process.poll.return_value = 0
    assert engine.is_running() is False


def test_api_url_uses_config_port():
    assert de.DockerEngine(_cfg(api_port=8080)).api_url == "http://localhost:8080/v1"


def test_log_path_switches_by_mode(tmp_path):
    with patch("ainode.engine.docker_engine.LOGS_DIR", tmp_path):
        solo = de.DockerEngine(_cfg(distributed_mode="solo"))
        head = de.DockerEngine(_cfg(distributed_mode="head", peer_ips=["x"]))
        assert solo.log_path.name == "vllm.log"
        assert head.log_path.name == "distributed.log"
