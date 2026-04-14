"""Tests for ainode.engine.docker_engine.DockerEngine."""

from __future__ import annotations

import io
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ainode.core.config import NodeConfig
from ainode.engine import docker_engine as de_mod
from ainode.engine.docker_engine import CONTAINER_NAME, DockerEngine, DockerEngineError


@pytest.fixture
def tmp_ainode_home(tmp_path, monkeypatch):
    """Redirect AINODE_HOME / COMPOSE_FILE / ENV_FILE / LOGS_DIR into a tmp dir."""
    monkeypatch.setattr(de_mod, "AINODE_HOME", tmp_path)
    monkeypatch.setattr(de_mod, "COMPOSE_FILE", tmp_path / "docker-compose.yml")
    monkeypatch.setattr(de_mod, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(de_mod, "LOGS_DIR", tmp_path / "logs")
    (tmp_path / "logs").mkdir()
    # Seed a compose file so start() doesn't bail
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    return tmp_path


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_start_runs_compose_up(tmp_ainode_home):
    engine = DockerEngine(NodeConfig())
    with patch("ainode.engine.docker_engine.shutil.which", return_value="/usr/bin/docker"), \
         patch("ainode.engine.docker_engine.subprocess.run", return_value=_completed()) as mock_run:
        assert engine.start() is True
        args = mock_run.call_args[0][0]
        assert args[:3] == ["docker", "compose", "-f"]
        assert "up" in args and "-d" in args


def test_start_missing_docker_raises(tmp_ainode_home):
    engine = DockerEngine(NodeConfig())
    with patch("ainode.engine.docker_engine.shutil.which", return_value=None):
        with pytest.raises(DockerEngineError):
            engine.start()


def test_start_missing_compose_raises(tmp_ainode_home):
    # Remove the seed compose file.
    (tmp_ainode_home / "docker-compose.yml").unlink()
    engine = DockerEngine(NodeConfig())
    with patch("ainode.engine.docker_engine.shutil.which", return_value="/usr/bin/docker"):
        with pytest.raises(DockerEngineError):
            engine.start()


def test_stop_runs_compose_down(tmp_ainode_home):
    engine = DockerEngine(NodeConfig())
    with patch("ainode.engine.docker_engine.shutil.which", return_value="/usr/bin/docker"), \
         patch("ainode.engine.docker_engine.subprocess.run", return_value=_completed()) as mock_run:
        engine.stop()
        args = mock_run.call_args[0][0]
        assert "down" in args


def test_is_running_true_when_container_present(tmp_ainode_home):
    engine = DockerEngine(NodeConfig())
    with patch("ainode.engine.docker_engine.shutil.which", return_value="/usr/bin/docker"), \
         patch("ainode.engine.docker_engine.subprocess.run",
               return_value=_completed(stdout=f"{CONTAINER_NAME}\n")):
        assert engine.is_running() is True


def test_is_running_false_when_absent(tmp_ainode_home):
    engine = DockerEngine(NodeConfig())
    with patch("ainode.engine.docker_engine.shutil.which", return_value="/usr/bin/docker"), \
         patch("ainode.engine.docker_engine.subprocess.run", return_value=_completed(stdout="\n")):
        assert engine.is_running() is False


def test_wait_ready_returns_true_on_200(tmp_ainode_home):
    engine = DockerEngine(NodeConfig())
    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_resp.__enter__ = lambda self: self
    fake_resp.__exit__ = lambda *a: None
    with patch("ainode.engine.docker_engine.urllib.request.urlopen", return_value=fake_resp):
        assert engine.wait_ready(timeout=5) is True


def test_wait_ready_times_out(tmp_ainode_home):
    import urllib.error
    engine = DockerEngine(NodeConfig())
    with patch("ainode.engine.docker_engine.urllib.request.urlopen",
               side_effect=urllib.error.URLError("nope")), \
         patch("ainode.engine.docker_engine.time.sleep"), \
         patch("ainode.engine.docker_engine.time.time",
               side_effect=[0, 0, 1, 2, 100, 200, 300]):
        assert engine.wait_ready(timeout=1) is False


def test_launch_rewrites_env_and_recreates(tmp_ainode_home):
    engine = DockerEngine(NodeConfig())
    (tmp_ainode_home / ".env").write_text("AINODE_MODEL=old\nAINODE_TP_SIZE=1\n")

    with patch("ainode.engine.docker_engine.shutil.which", return_value="/usr/bin/docker"), \
         patch("ainode.engine.docker_engine.subprocess.run", return_value=_completed()) as mock_run, \
         patch.object(NodeConfig, "save", lambda self: None):
        assert engine.launch("meta-llama/Llama-3.1-8B-Instruct") is True

    env_text = (tmp_ainode_home / ".env").read_text()
    assert "AINODE_MODEL=meta-llama/Llama-3.1-8B-Instruct" in env_text
    assert "AINODE_TP_SIZE=1" in env_text  # preserved
    args = mock_run.call_args[0][0]
    assert "up" in args and "-d" in args


def test_launch_distributed_sets_tp_and_ray(tmp_ainode_home):
    engine = DockerEngine(NodeConfig())
    (tmp_ainode_home / ".env").write_text("AINODE_MODEL=m\n")
    sharding = SimpleNamespace(
        tensor_parallel_size=4,
        ray_head_address="10.0.0.1:6379",
        model=None,
    )
    with patch("ainode.engine.docker_engine.shutil.which", return_value="/usr/bin/docker"), \
         patch("ainode.engine.docker_engine.subprocess.run", return_value=_completed()):
        assert engine.launch_distributed(sharding) is True

    env_text = (tmp_ainode_home / ".env").read_text()
    assert "AINODE_TP_SIZE=4" in env_text
    assert "AINODE_RAY_ADDRESS=10.0.0.1:6379" in env_text


def test_logs_invokes_docker_logs(tmp_ainode_home):
    engine = DockerEngine(NodeConfig())
    with patch("ainode.engine.docker_engine.shutil.which", return_value="/usr/bin/docker"), \
         patch("ainode.engine.docker_engine.subprocess.run",
               return_value=_completed(stdout="logline\n")) as mock_run:
        out = engine.logs(n=42)
    assert out == "logline\n"
    args = mock_run.call_args[0][0]
    assert args == ["docker", "logs", "--tail", "42", CONTAINER_NAME]


def test_cmd_start_dispatches_to_docker_engine(tmp_ainode_home, monkeypatch):
    """cmd_start should instantiate DockerEngine when engine_strategy == 'docker'."""
    from ainode.cli import main as cli_main
    from ainode.core import config as config_mod

    # Build a config that says "docker"
    cfg = NodeConfig(engine_strategy="docker", onboarded=True, node_id="testnode")

    monkeypatch.setattr(config_mod.NodeConfig, "load", classmethod(lambda cls: cfg))
    monkeypatch.setattr(cli_main, "ensure_dirs", lambda: None)
    monkeypatch.setattr(cli_main, "_write_pid", lambda: None)
    monkeypatch.setattr(cli_main, "_remove_pid", lambda: None)
    monkeypatch.setattr("ainode.core.gpu.detect_gpu", lambda: None)

    fake_engine = MagicMock()
    fake_engine.start.return_value = True
    fake_engine.wait_ready.return_value = True
    fake_engine.log_path = None
    fake_engine.process.wait.side_effect = KeyboardInterrupt()

    with patch("ainode.engine.docker_engine.DockerEngine", return_value=fake_engine) as mock_de, \
         patch("ainode.engine.vllm_engine.VLLMEngine") as mock_vllm, \
         patch("threading.Thread"):
        args = SimpleNamespace(model=None, port=None)
        cli_main.cmd_start(args)

    mock_de.assert_called_once()
    mock_vllm.assert_not_called()
    fake_engine.start.assert_called_once()
