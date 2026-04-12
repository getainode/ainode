"""Tests for ainode.cli.main — argument parsing and command dispatch."""

import sys
from unittest.mock import patch
from ainode.cli.main import main


def test_version(capsys):
    with patch.object(sys, "argv", ["ainode", "--version"]):
        try:
            main()
        except SystemExit:
            pass
    captured = capsys.readouterr()
    assert "0.1.0" in captured.out


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
