"""Tests for ainode.cli.main — argument parsing and command dispatch."""

import os
import sys
from unittest.mock import patch, MagicMock
from ainode.cli.main import (
    main, _banner, _write_pid, _read_pid, _remove_pid, _pid_alive,
    _tail_log, _gpu_info_table,
    cmd_config, cmd_logs, cmd_stop,
)


def test_version(capsys):
    from ainode import __version__

    with patch.object(sys, "argv", ["ainode", "--version"]):
        try:
            main()
        except SystemExit:
            pass
    captured = capsys.readouterr()
    assert __version__ in captured.out
    assert captured.out.startswith("ainode ")


def test_no_args_calls_start(monkeypatch):
    """Running `ainode` with no subcommand calls cmd_start."""
    called = {}
    def fake_start(args):
        called["start"] = True
    monkeypatch.setattr("ainode.cli.main.cmd_start", fake_start)
    with patch.object(sys, "argv", ["ainode"]):
        main()
    assert called.get("start") is True


def test_status_subcommand(monkeypatch):
    called = {}
    def fake_status(args):
        called["status"] = True
    monkeypatch.setattr("ainode.cli.main.cmd_status", fake_status)
    with patch.object(sys, "argv", ["ainode", "status"]):
        main()
    assert called.get("status") is True


def test_models_subcommand(monkeypatch):
    called = {}
    def fake_models(args):
        called["models"] = True
    monkeypatch.setattr("ainode.cli.main.cmd_models", fake_models)
    with patch.object(sys, "argv", ["ainode", "models"]):
        main()
    assert called.get("models") is True


def test_stop_subcommand(monkeypatch):
    called = {}
    def fake_stop(args):
        called["stop"] = True
    monkeypatch.setattr("ainode.cli.main.cmd_stop", fake_stop)
    with patch.object(sys, "argv", ["ainode", "stop"]):
        main()
    assert called.get("stop") is True


def test_config_subcommand(monkeypatch):
    called = {}
    def fake_config(args):
        called["config"] = True
    monkeypatch.setattr("ainode.cli.main.cmd_config", fake_config)
    with patch.object(sys, "argv", ["ainode", "config", "--show"]):
        main()
    assert called.get("config") is True


def test_logs_subcommand(monkeypatch):
    called = {}
    def fake_logs(args):
        called["logs"] = True
    monkeypatch.setattr("ainode.cli.main.cmd_logs", fake_logs)
    with patch.object(sys, "argv", ["ainode", "logs"]):
        main()
    assert called.get("logs") is True


def test_logs_follow_flag(monkeypatch):
    """Ensure --follow flag is parsed correctly."""
    captured_args = {}
    def fake_logs(args):
        captured_args["follow"] = args.follow
    monkeypatch.setattr("ainode.cli.main.cmd_logs", fake_logs)
    with patch.object(sys, "argv", ["ainode", "logs", "--follow"]):
        main()
    assert captured_args["follow"] is True


def test_config_model_flag(monkeypatch):
    """Ensure --model flag is parsed for config subcommand."""
    captured_args = {}
    def fake_config(args):
        captured_args["model"] = args.model
    monkeypatch.setattr("ainode.cli.main.cmd_config", fake_config)
    with patch.object(sys, "argv", ["ainode", "config", "--model", "llama-3.1-8b"]):
        main()
    assert captured_args["model"] == "llama-3.1-8b"


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def test_banner_returns_panel():
    from rich.panel import Panel
    panel = _banner()
    assert isinstance(panel, Panel)


# ---------------------------------------------------------------------------
# PID helpers
# ---------------------------------------------------------------------------

def test_pid_write_read_remove(tmp_path, monkeypatch):
    pid_file = tmp_path / "ainode.pid"
    monkeypatch.setattr("ainode.cli.main.PID_FILE", pid_file)
    monkeypatch.setattr("ainode.cli.main.AINODE_HOME", tmp_path)

    _write_pid()
    assert pid_file.exists()
    assert _read_pid() == os.getpid()

    _remove_pid()
    assert not pid_file.exists()


def test_read_pid_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("ainode.cli.main.PID_FILE", tmp_path / "nope.pid")
    assert _read_pid() is None


def test_pid_alive_current():
    assert _pid_alive(os.getpid()) is True


def test_pid_alive_none():
    assert _pid_alive(None) is False


def test_pid_alive_dead():
    # PID 99999999 almost certainly doesn't exist
    assert _pid_alive(99999999) is False


# ---------------------------------------------------------------------------
# _tail_log
# ---------------------------------------------------------------------------

def test_tail_log(tmp_path):
    log = tmp_path / "test.log"
    log.write_text("\n".join(f"line {i}" for i in range(100)))
    lines = _tail_log(log, lines=5)
    assert len(lines) == 5
    assert "line 99" in lines[-1]


def test_tail_log_missing():
    lines = _tail_log("/nonexistent/file.log")
    assert lines == []


# ---------------------------------------------------------------------------
# GPU info table
# ---------------------------------------------------------------------------

def test_gpu_info_table():
    from ainode.core.gpu import GPUInfo
    from rich.table import Table
    gpu = GPUInfo(
        name="NVIDIA GH200",
        memory_total_mb=131072,
        memory_free_mb=120000,
        cuda_version="12.4",
        driver_version="550.54",
        compute_capability="9.0",
        unified_memory=True,
    )
    table = _gpu_info_table(gpu)
    assert isinstance(table, Table)


# ---------------------------------------------------------------------------
# cmd_config
# ---------------------------------------------------------------------------

def test_cmd_config_show(tmp_path, monkeypatch):
    """config --show renders a table with config keys."""
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "config.json")

    args = MagicMock()
    args.model = None
    args.port = None
    args.show = True

    cmd_config(args)


def test_cmd_config_set_model(tmp_path, monkeypatch):
    """config --model writes the new model to disk."""
    config_file = tmp_path / "config.json"
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", config_file)

    args = MagicMock()
    args.model = "deepseek-r1-7b"
    args.port = None

    cmd_config(args)

    from ainode.core.config import NodeConfig
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", config_file)
    loaded = NodeConfig.load()
    assert loaded.model == "deepseek-r1-7b"


def test_cmd_config_set_port(tmp_path, monkeypatch):
    """config --port writes the new port to disk."""
    config_file = tmp_path / "config.json"
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", config_file)

    args = MagicMock()
    args.model = None
    args.port = 9000

    cmd_config(args)

    from ainode.core.config import NodeConfig
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", config_file)
    loaded = NodeConfig.load()
    assert loaded.api_port == 9000


# ---------------------------------------------------------------------------
# cmd_logs
# ---------------------------------------------------------------------------

def test_cmd_logs_no_file(tmp_path, monkeypatch):
    """logs command with no log file prints a helpful message."""
    monkeypatch.setattr("ainode.cli.main.VLLM_LOG", tmp_path / "nope.log")

    args = MagicMock()
    args.follow = False
    args.lines = 50

    cmd_logs(args)


def test_cmd_logs_reads_lines(tmp_path, monkeypatch):
    """logs command reads last N lines from log."""
    log_file = tmp_path / "vllm.log"
    log_file.write_text("\n".join(f"log line {i}" for i in range(100)))
    monkeypatch.setattr("ainode.cli.main.VLLM_LOG", log_file)

    args = MagicMock()
    args.follow = False
    args.lines = 10

    cmd_logs(args)


# ---------------------------------------------------------------------------
# cmd_stop
# ---------------------------------------------------------------------------

def test_cmd_stop_no_running_instance(tmp_path, monkeypatch):
    """stop command with no running instance prints informational message."""
    monkeypatch.setattr("ainode.cli.main.PID_FILE", tmp_path / "ainode.pid")
    monkeypatch.setattr("ainode.cli.main.AINODE_HOME", tmp_path)

    # Mock pgrep to return nothing
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: MagicMock(stdout="", returncode=1))

    args = MagicMock()
    cmd_stop(args)
