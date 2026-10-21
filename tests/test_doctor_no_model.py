"""``ainode doctor`` passes on a node that loads no model under AINode.

A source-installed routing-only master may have GPUs used by other services and need no
engine container; absent Docker and engine images are not failures when no model is
configured.
"""

from __future__ import annotations

import json

import pytest

from ainode.cli import doctor as doc

A40 = {"index": 0, "name": "NVIDIA A40", "memory_total_mb": 46068,
       "memory_free_mb": 3400, "unified_memory": False, "persistence_mode": False}


@pytest.fixture
def primary_node(tmp_path, monkeypatch):
    """Fake a routing-only source installation with a GPU, no available Docker or systemctl
    result, a reachable dashboard and a free engine port."""
    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    monkeypatch.setattr(doc, "running_in_container", lambda *a, **kw: False)
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0: (127, "absent"))
    monkeypatch.setattr(doc, "disk_usage", lambda path: (1000, 900))
    monkeypatch.setattr(doc, "tcp_listening", lambda port, **kw: port == 3000)
    monkeypatch.setattr(doc, "udp_listeners", lambda: {5679})
    monkeypatch.setattr(doc, "http_json",
                        lambda url, timeout=3.0, headers=None: {"nodes": []})
    monkeypatch.setattr(doc, "latest_image_tag", lambda: None)
    monkeypatch.setattr(doc, "probe_gpus", lambda: [dict(A40)])
    (tmp_path / "models").mkdir()

    def write(**config):
        base = {"node_id": "primary_node", "node_name": "node-1", "model": None,
                "cluster_role": "master", "discovery_port": 5679,
                "models_dir": str(tmp_path / "models")}
        base.update(config)
        (tmp_path / "config.json").write_text(json.dumps(base))
        return {c.name: c for c in doc.run_checks(tmp_path, tmp_path / "config.json")}

    return write, tmp_path


def test_a_node_that_loads_no_model_passes(primary_node):
    write, _ = primary_node
    checks = write()

    failed = [f"{c.name}: {c.detail}" for c in checks.values() if c.status == doc.FAIL]
    assert failed == []
    assert doc.exit_code(list(checks.values())) == 0
    assert checks["gpu.devices"].status == doc.OK, "the GPU is still reported"


def test_no_model_loaded_is_info(primary_node):
    write, _ = primary_node
    checks = write()

    assert checks["config.model"].status == doc.INFO
    assert "no model loaded" in checks["config.model"].detail
    assert checks["port.engine"].status == doc.INFO


def test_the_engine_launch_checks_are_info_with_the_reason(primary_node):
    write, _ = primary_node
    checks = write()

    for name in ("docker.daemon", "config.engine_backend"):
        check = checks[name]
        assert check.status == doc.INFO, f"{name} is {check.status}"
        assert doc.NO_MODEL_NOTE in check.detail
        assert check.data["engine_expected"] is False
        assert check.fix, "the fix is still printed for when a model is loaded"
    # What it WOULD be on a node that loads one, kept for --json readers.
    assert checks["docker.daemon"].data["status_with_a_model"] == doc.FAIL


def test_an_eugr_backend_with_no_vllm_does_not_fail_a_node_with_no_model(primary_node,
                                                                          monkeypatch):
    write, _ = primary_node
    monkeypatch.setattr(doc.shutil, "which", lambda name: None)
    checks = write(engine_backend="eugr")

    assert checks["config.engine_backend"].status == doc.INFO
    assert doc.exit_code(list(checks.values())) == 0


def test_a_node_that_pins_a_model_still_fails_without_docker(primary_node):
    """The regression guard: the relaxation is for nodes that serve nothing."""
    write, _ = primary_node
    checks = write(model="Qwen/Qwen3.8-27B")

    assert checks["docker.daemon"].status == doc.FAIL
    assert doc.NO_MODEL_NOTE not in checks["docker.daemon"].detail
    assert doc.exit_code(list(checks.values())) == 1


def test_stacked_instances_on_disk_count_as_a_model(primary_node):
    write, home = primary_node
    (home / doc.INSTANCE_MANIFEST_NAME).write_text(json.dumps(
        {"instances": [{"model": "openai/whisper-large-v3-turbo",
                        "gpu_memory_utilization": 0.2}]}))
    checks = write()

    assert checks["docker.daemon"].status == doc.FAIL


def test_engine_expected_reads_the_pin_the_manifest_and_the_distributed_record(tmp_path):
    from ainode.core.config import NodeConfig

    idle = NodeConfig(model=None)
    assert doc.engine_expected(idle, tmp_path) is False
    assert doc.engine_expected(NodeConfig(model="Qwen/Q"), tmp_path) is True

    (tmp_path / doc.INSTANCE_MANIFEST_NAME).write_text('{"instances": []}')
    assert doc.engine_expected(idle, tmp_path) is False
    (tmp_path / doc.INSTANCE_MANIFEST_NAME).write_text("not json")
    assert doc.engine_expected(idle, tmp_path) is False

    (tmp_path / doc.DISTRIBUTED_RECORD_NAME).write_text("{}")
    assert doc.engine_expected(idle, tmp_path) is True


def test_the_file_names_match_the_ones_the_node_writes():
    """One home per name: the doctor reads what the launch path writes."""
    from pathlib import Path

    import ainode.models.api_routes as mr
    from ainode.engine import reconcile

    assert Path(mr._manifest_path()).name == doc.INSTANCE_MANIFEST_NAME
    assert reconcile.RECORD_FILENAME == doc.DISTRIBUTED_RECORD_NAME
