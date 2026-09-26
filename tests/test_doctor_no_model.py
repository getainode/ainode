"""``ainode doctor`` passes on a node that loads no model under AINode.

Atlas is the shape: a bare-metal source install with a GPU (an A40 that its own
llama.cpp services hold), running AINode as a routing-only master. Nothing on it
will launch an engine container, so neither docker nor the engine backend nor its
image is a reason for it to fail, and "no model loaded" is an INFO line. A node
that DOES load a model keeps every one of those findings exactly as before.
"""

from __future__ import annotations

import json

import pytest

from ainode.cli import doctor as doc

A40 = {"index": 0, "name": "NVIDIA A40", "memory_total_mb": 46068,
       "memory_free_mb": 3400, "unified_memory": False, "persistence_mode": False}


@pytest.fixture
def atlas(tmp_path, monkeypatch):
    """Every seam faked as Atlas answers it: GPU present, no docker for this user,
    no systemctl answer, dashboard up, engine port free, nobody else seen yet."""
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
        base = {"node_id": "atlas", "node_name": "Atlas", "model": None,
                "cluster_role": "master", "discovery_port": 5679,
                "models_dir": str(tmp_path / "models")}
        base.update(config)
        (tmp_path / "config.json").write_text(json.dumps(base))
        return {c.name: c for c in doc.run_checks(tmp_path, tmp_path / "config.json")}

    return write, tmp_path


def test_a_node_that_loads_no_model_passes(atlas):
    write, _ = atlas
    checks = write()

    failed = [f"{c.name}: {c.detail}" for c in checks.values() if c.status == doc.FAIL]
    assert failed == []
    assert doc.exit_code(list(checks.values())) == 0
    assert checks["gpu.devices"].status == doc.OK, "the GPU is still reported"


def test_no_model_loaded_is_info(atlas):
    write, _ = atlas
    checks = write()

    assert checks["config.model"].status == doc.INFO
    assert "no model loaded" in checks["config.model"].detail
    assert checks["port.engine"].status == doc.INFO


def test_the_engine_launch_checks_are_info_with_the_reason(atlas):
    write, _ = atlas
    checks = write()

    for name in ("docker.daemon", "config.engine_backend"):
        check = checks[name]
        assert check.status == doc.INFO, f"{name} is {check.status}"
        assert doc.NO_MODEL_NOTE in check.detail
        assert check.data["engine_expected"] is False
        assert check.fix, "the fix is still printed for when a model is loaded"
    # What it WOULD be on a node that loads one, kept for --json readers.
    assert checks["docker.daemon"].data["status_with_a_model"] == doc.FAIL


def test_an_eugr_backend_with_no_vllm_does_not_fail_a_node_with_no_model(atlas,
                                                                          monkeypatch):
    write, _ = atlas
    monkeypatch.setattr(doc.shutil, "which", lambda name: None)
    checks = write(engine_backend="eugr")

    assert checks["config.engine_backend"].status == doc.INFO
    assert doc.exit_code(list(checks.values())) == 0


def test_a_node_that_pins_a_model_still_fails_without_docker(atlas):
    """The regression guard: the relaxation is for nodes that serve nothing."""
    write, _ = atlas
    checks = write(model="Qwen/Qwen3.8-27B")

    assert checks["docker.daemon"].status == doc.FAIL
    assert doc.NO_MODEL_NOTE not in checks["docker.daemon"].detail
    assert doc.exit_code(list(checks.values())) == 1


def test_stacked_instances_on_disk_count_as_a_model(atlas):
    write, home = atlas
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
