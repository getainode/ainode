"""Tests for ainode.core.config."""

import json
from pathlib import Path
from ainode.core.config import NodeConfig


def test_config_defaults():
    """Default config has expected values."""
    config = NodeConfig()
    assert config.api_port == 8000
    assert config.web_port == 3000
    assert config.host == "0.0.0.0"
    assert config.model == "meta-llama/Llama-3.2-3B-Instruct"
    assert config.gpu_memory_utilization == 0.9
    assert config.onboarded is False
    assert config.node_id is None


def test_config_save_load(tmp_path, monkeypatch):
    """Config round-trips through save/load."""
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "config.json")

    config = NodeConfig(node_id="abc123", model="test-model", api_port=9000)
    config.save()

    loaded = NodeConfig.load()
    assert loaded.node_id == "abc123"
    assert loaded.model == "test-model"
    assert loaded.api_port == 9000


def test_config_load_missing(tmp_path, monkeypatch):
    """Loading from nonexistent file returns defaults."""
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "nope.json")
    config = NodeConfig.load()
    assert config.api_port == 8000
    assert config.node_id is None


def test_config_load_ignores_unknown_fields(tmp_path, monkeypatch):
    """Unknown fields in config file are silently ignored."""
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"node_id": "x", "unknown_field": True}))
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", config_file)

    config = NodeConfig.load()
    assert config.node_id == "x"
    assert not hasattr(config, "unknown_field")
