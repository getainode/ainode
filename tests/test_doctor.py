"""``ainode doctor`` - every check's OK / WARN / FAIL branch, driven with fakes.

The command's whole value is that a human can trust the line it prints, so the
branches are pinned one by one rather than by running the real thing and reading
it. Nothing here touches the network, the docker socket, NVML or systemd: every
world-facing call in ``ainode.cli.doctor`` is a module-level seam, and each test
replaces the seam it cares about.

Also pinned: the ``--json`` shape (a machine reads it), and the exit code rule
(non-zero on any FAIL, zero when the worst answer is a WARN, because a doctor
that fails on every warning is a doctor nobody runs).
"""

import json
import os
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from ainode.cli import doctor as doc
from ainode.core.config import NodeConfig

FAIL, OK, WARN = doc.FAIL, doc.OK, doc.WARN


def _by_name(checks, name):
    for check in checks:
        if check.name == name:
            return check
    raise AssertionError(f"no check named {name} in {[c.name for c in checks]}")


def _one(checks):
    assert len(checks) == 1, [c.name for c in checks]
    return checks[0]


# ------------------------------------------------------------------ sudo trap

def test_a_plain_run_is_not_under_sudo():
    check = _one(doc.check_sudo_trap(env={}, euid=501))
    assert (check.name, check.status) == ("env.sudo", OK)


def test_root_under_sudo_with_no_ainode_home_is_the_trap():
    """The 0.5.26 update bug: under sudo every path moves to root's home."""
    check = _one(doc.check_sudo_trap(env={"SUDO_USER": "sem"}, euid=0))
    assert check.status == WARN
    assert "sem" in check.detail and "AINODE_HOME" in check.detail
    assert check.fix


def test_root_under_sudo_with_ainode_home_pinned_is_fine():
    check = _one(doc.check_sudo_trap(
        env={"SUDO_USER": "sem", "AINODE_HOME": "/home/sem/.ainode"}, euid=0))
    assert check.status == OK
    assert "/home/sem/.ainode" in check.detail


# ---------------------------------------------------------------- config file

def test_a_parsed_config_file_is_ok(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{}")
    assert _one(doc.check_config_file(path, None)).status == OK


def test_a_missing_config_file_warns_that_everything_is_a_default(tmp_path):
    check = _one(doc.check_config_file(tmp_path / "config.json", None))
    assert check.status == WARN
    assert "default" in check.detail


def test_an_unreadable_config_file_fails(tmp_path):
    check = _one(doc.check_config_file(tmp_path / "config.json", "bad json"))
    assert check.status == FAIL
    assert check.data["error"] == "bad json"


def test_load_config_survives_junk_and_unknown_keys(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"api_port": 9001, "not_a_field": 1}')
    config, error = doc.load_config(path)
    assert error is None
    assert config.api_port == 9001

    path.write_text("{not json")
    config, error = doc.load_config(path)
    assert error is not None
    assert config.api_port == NodeConfig().api_port


# ------------------------------------------------------------- engine backend

def test_the_nvidia_backend_is_ok_when_its_image_is_pulled():
    config = NodeConfig(engine_backend="nvidia", engine_image="scitrera/x:1")
    check = _one(doc.check_engine_backend(config, "/c.json", True, True))
    assert check.status == OK
    assert "scitrera/x:1" in check.detail


def test_the_nvidia_backend_warns_when_its_image_is_not_pulled():
    config = NodeConfig(engine_backend="nvidia", engine_image="scitrera/x:1")
    check = _one(doc.check_engine_backend(config, "/c.json", True, False))
    assert check.status == WARN
    assert check.fix == "docker pull scitrera/x:1"


def test_an_unreachable_docker_makes_the_image_unknown_not_absent():
    config = NodeConfig(engine_backend="nvidia", engine_image="scitrera/x:1")
    check = _one(doc.check_engine_backend(config, "/c.json", False, None))
    assert check.status == WARN
    assert "unknown" in check.detail


def test_the_eugr_backend_fails_without_a_vllm_binary(monkeypatch):
    """The 0.5.26 install bug, as a check: the shipped image has no vLLM in it."""
    monkeypatch.setattr(doc.shutil, "which", lambda name: None)
    config = NodeConfig(engine_backend="eugr")
    check = _one(doc.check_engine_backend(config, "/c.json", True, None))
    assert check.status == FAIL
    assert "nvidia" in check.fix


def test_the_eugr_backend_is_ok_with_a_vllm_binary(monkeypatch):
    monkeypatch.setattr(doc.shutil, "which", lambda name: "/usr/local/bin/vllm")
    config = NodeConfig(engine_backend="eugr")
    check = _one(doc.check_engine_backend(config, "/c.json", True, None))
    assert check.status == OK


def test_a_backend_that_does_not_exist_fails():
    config = NodeConfig(engine_backend="wishful")
    check = _one(doc.check_engine_backend(config, "/c.json", True, None))
    assert check.status == FAIL
    assert "wishful" in check.detail


# -------------------------------------------------------------------- numbers

@pytest.mark.parametrize("value,status", [
    (0.5, OK), (0.6, OK), (0.89, OK),
    (0.9, WARN), (0.95, WARN),
    (0.0, FAIL), (1.5, FAIL), (-0.1, FAIL),
])
def test_gpu_memory_utilization_branches(value, status):
    config = NodeConfig(gpu_memory_utilization=value)
    assert _one(doc.check_gpu_memory_utilization(config)).status == status


def test_a_non_numeric_gpu_memory_utilization_fails():
    config = NodeConfig()
    config.gpu_memory_utilization = "lots"
    assert _one(doc.check_gpu_memory_utilization(config)).status == FAIL


def test_the_fleet_discovery_port_is_ok_and_anything_else_warns():
    ok = _one(doc.check_discovery_port(NodeConfig(discovery_port=5679)))
    assert ok.status == OK
    # 5678 is NodeConfig's dataclass default and NOT what the installer writes,
    # which is exactly how a node ends up alone on the wire looking healthy.
    warn = _one(doc.check_discovery_port(NodeConfig(discovery_port=5678)))
    assert warn.status == WARN
    assert warn.data["fix_action"] == "discovery_port"
    assert warn.data["expected"] == 5679


def test_cluster_id_default_only_warns_when_peers_are_expected():
    alone = _one(doc.check_cluster_id(NodeConfig(cluster_id="default")))
    assert alone.status == OK
    with_peers = _one(doc.check_cluster_id(
        NodeConfig(cluster_id="default", peer_ips=["10.0.0.2"])))
    assert with_peers.status == WARN
    seen = _one(doc.check_cluster_id(NodeConfig(cluster_id="default"), peers_seen=2))
    assert seen.status == WARN
    named = _one(doc.check_cluster_id(NodeConfig(cluster_id="titanium",
                                                peer_ips=["10.0.0.2"])))
    assert named.status == OK


def test_an_empty_cluster_id_fails():
    assert _one(doc.check_cluster_id(NodeConfig(cluster_id=""))).status == FAIL


# ---------------------------------------------------------------------- model

def test_no_model_pinned_is_info():
    """A node that loads no model is a fact, not a finding (Atlas serves nothing)."""
    check = _one(doc.check_model(NodeConfig(model=None)))
    assert check.status == doc.INFO
    assert "no model loaded" in check.detail
    assert "nothing loads at boot" in check.detail


def test_a_pinned_model_that_is_on_disk_is_ok(tmp_path):
    repo = tmp_path / "models--org--m"
    repo.mkdir(parents=True)
    (repo / "config.json").write_text("{}")
    config = NodeConfig(model="org/m", models_dir=str(tmp_path))
    check = _one(doc.check_model(config))
    assert check.status == OK
    assert str(repo) in check.detail


def test_a_pinned_model_that_is_not_downloaded_warns(tmp_path):
    config = NodeConfig(model="meta-llama/Llama-3.2-3B-Instruct",
                        models_dir=str(tmp_path))
    check = _one(doc.check_model(config))
    assert check.status == WARN
    assert "401" in check.detail  # a gated repo is the expensive version of this


# --------------------------------------------------------------------- docker

def test_docker_ok_reports_the_server_version_and_the_engine_image(monkeypatch):
    calls = []

    def fake_run(argv, timeout=10.0):
        calls.append(argv)
        if argv[1] == "info":
            return 0, "28.3.2"
        return 0, "[{}]"

    monkeypatch.setattr(doc, "run_command", fake_run)
    check = _one(doc.check_docker("scitrera/x:1"))
    assert check.status == OK
    assert check.data == {"reachable": True, "server_version": "28.3.2",
                          "image_present": True}
    assert calls[1] == ["docker", "image", "inspect", "scitrera/x:1"]


def test_no_docker_cli_is_a_fail(monkeypatch):
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0: (127, "not found"))
    check = _one(doc.check_docker("scitrera/x:1"))
    assert check.status == FAIL
    assert check.data["reachable"] is False


def test_a_daemon_that_will_not_answer_is_a_fail(monkeypatch):
    monkeypatch.setattr(doc, "run_command",
                        lambda argv, timeout=10.0: (1, "Cannot connect to the Docker daemon"))
    check = _one(doc.check_docker(""))
    assert check.status == FAIL
    assert "Cannot connect" in check.detail


def test_a_missing_engine_image_is_reported_as_absent(monkeypatch):
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0:
                        (0, "28.3.2") if argv[1] == "info" else (1, "No such image"))
    assert _one(doc.check_docker("scitrera/x:1")).data["image_present"] is False


# ------------------------------------------------------------------------ gpu

def test_no_gpu_is_a_fail():
    check = _one(doc.check_gpus(gpus=[]))
    assert check.status == FAIL
    assert check.data["count"] == 0


def test_a_gb10_reports_its_count_name_and_unified_memory():
    check = _one(doc.check_gpus(gpus=[{
        "index": 0, "name": "NVIDIA GB10", "memory_total_mb": 124620,
        "memory_free_mb": 90000, "unified_memory": True}]))
    assert check.status == OK
    assert "1 GPU" in check.detail
    assert "NVIDIA GB10" in check.detail
    assert "unified" in check.detail


def test_two_discrete_gpus_are_both_named():
    check = _one(doc.check_gpus(gpus=[
        {"index": 0, "name": "Tesla V100", "memory_total_mb": 32768,
         "memory_free_mb": 32000, "unified_memory": False},
        {"index": 1, "name": "Tesla V100", "memory_total_mb": 32768,
         "memory_free_mb": 32000, "unified_memory": False}]))
    assert check.data["count"] == 2
    assert "unified" not in check.detail


# ----------------------------------------------------------------------- disk

def test_plenty_of_disk_is_ok(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(doc, "disk_usage", lambda path: (1000 * 1024 ** 3, 800 * 1024 ** 3))
    checks = doc.check_disk(tmp_path, models)
    assert [c.status for c in checks] == [OK, OK]
    assert "800.0 GB free of 1000.0 GB (80%)" in checks[0].detail


def test_under_fifteen_percent_free_warns(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(doc, "disk_usage", lambda path: (1000 * 1024 ** 3, 100 * 1024 ** 3))
    checks = doc.check_disk(tmp_path, models)
    assert [c.status for c in checks] == [WARN, WARN]
    assert checks[1].data["free_fraction"] == 0.1


def test_a_missing_models_dir_warns_and_is_fixable(tmp_path, monkeypatch):
    monkeypatch.setattr(doc, "disk_usage", lambda path: (1000, 900))
    checks = doc.check_disk(tmp_path, tmp_path / "models")
    assert _by_name(checks, "disk.models").status == WARN
    assert _by_name(checks, "disk.models").data["fix_action"] == "mkdir"


def test_a_directory_we_may_not_read_warns_instead_of_crashing(tmp_path, monkeypatch):
    """Found on Spark-2: models_dir is the CONTAINER path, unreadable on the host.

    ``Path.exists()`` raises PermissionError there, and the doctor died on the
    way to reporting 20 other facts. A denied path is a finding, and it does not
    offer to mkdir a directory the mkdir would fail on either.
    """
    monkeypatch.setattr(doc, "disk_usage", lambda path: (1000, 900))
    monkeypatch.setattr(doc, "path_state", lambda path: (
        ("denied", "[Errno 13] Permission denied: '/root/.ainode/models'")
        if "root" in str(path) else ("present", "")))
    checks = doc.check_disk(tmp_path, "/root/.ainode/models")
    denied = _by_name(checks, "disk.models")
    assert denied.status == WARN
    assert "Permission denied" in denied.detail
    assert denied.data["exists"] is None
    assert "fix_action" not in denied.data


def test_path_state_never_raises_on_an_unreadable_parent(monkeypatch):
    def boom(self):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "exists", boom)
    assert doc.path_state("/root/.ainode/models")[0] == "denied"
    assert doc._exists("/root/.ainode/models") is False


def test_a_filesystem_we_cannot_stat_warns_rather_than_reporting_zero(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(doc, "disk_usage", lambda path: None)
    checks = doc.check_disk(tmp_path, models)
    assert [c.status for c in checks] == [WARN, WARN]
    assert "cannot read" in checks[0].detail


# ---------------------------------------------------------------- image + pin

def test_image_env_agreeing_with_the_running_container_is_ok(tmp_path, monkeypatch):
    (tmp_path / "image.env").write_text("AINODE_IMAGE=ghcr.io/getainode/ainode:0.5.26\n")
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0:
                        (0, "running ghcr.io/getainode/ainode:0.5.26"))
    monkeypatch.setattr(doc, "latest_image_tag", lambda: "0.5.26")
    checks = doc.check_image_pin(tmp_path, "0.5.26")
    assert _by_name(checks, "image.pin").status == OK
    assert _by_name(checks, "image.latest").status == OK


def test_a_pin_that_disagrees_with_the_running_container_warns(tmp_path, monkeypatch):
    """The sudo-update gotcha: the pull wrote a pin nothing is running."""
    (tmp_path / "image.env").write_text("AINODE_IMAGE=ghcr.io/getainode/ainode:0.5.26\n")
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0:
                        (0, "running ghcr.io/getainode/ainode:0.5.24"))
    monkeypatch.setattr(doc, "latest_image_tag", lambda: "0.5.26")
    check = _by_name(doc.check_image_pin(tmp_path, "0.5.24"), "image.pin")
    assert check.status == WARN
    assert "0.5.26" in check.detail and "0.5.24" in check.detail


def test_no_image_env_warns(tmp_path, monkeypatch):
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0: (1, ""))
    monkeypatch.setattr(doc, "latest_image_tag", lambda: "0.5.26")
    check = _by_name(doc.check_image_pin(tmp_path, "0.5.26"), "image.pin")
    assert check.status == WARN
    assert check.data["pinned"] is None


def test_a_newer_published_tag_warns_and_names_the_fleet(tmp_path, monkeypatch):
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0: (1, ""))
    monkeypatch.setattr(doc, "latest_image_tag", lambda: "0.5.30")
    check = _by_name(doc.check_image_pin(tmp_path, "0.5.26"), "image.latest")
    assert check.status == WARN
    assert "every node" in check.fix


def test_ghcr_being_unreachable_warns_rather_than_claiming_up_to_date(tmp_path, monkeypatch):
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0: (1, ""))
    monkeypatch.setattr(doc, "latest_image_tag", lambda: None)
    check = _by_name(doc.check_image_pin(tmp_path, "0.5.26"), "image.latest")
    assert check.status == WARN
    assert "could not reach" in check.detail


def test_read_image_env_ignores_everything_but_the_image_line(tmp_path):
    (tmp_path / "image.env").write_text("# a comment\nOTHER=1\nAINODE_IMAGE=x:1\n")
    assert doc.read_image_env(tmp_path) == "x:1"
    assert doc.read_image_env(tmp_path / "nope") is None


# -------------------------------------------------------------------- service

def test_an_active_unit_is_ok(monkeypatch):
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0: (0, "active"))
    assert _one(doc.check_service(in_container=False)).status == OK


def test_an_installed_but_dead_unit_is_a_fail(monkeypatch):
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0: (3, "failed"))
    monkeypatch.setattr(Path, "exists", lambda self: True)
    check = _one(doc.check_service(in_container=False))
    assert check.status == FAIL
    assert "failed" in check.detail


def test_no_unit_at_all_warns(monkeypatch):
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0: (3, "inactive"))
    monkeypatch.setattr(Path, "exists", lambda self: False)
    check = _one(doc.check_service(in_container=False))
    assert check.status == WARN
    assert "started by hand" in check.detail


def test_inside_the_container_the_service_answer_says_it_cannot_tell():
    """INFO, not WARN: the documented deployment runs the CLI inside the
    container, so a WARN here was a permanent yellow line on every node in the
    fleet about something nobody standing in there could fix (#225). The wired
    behaviour, including the state the host wrapper passes in, is
    tests/test_doctor_container.py."""
    check = _one(doc.check_service(in_container=True, host_state=""))
    assert check.status == doc.INFO
    assert check.data == {"in_container": True, "state": None, "state_source": None}


def test_no_systemctl_warns_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0: (127, "not found"))
    check = _one(doc.check_service(in_container=False))
    assert check.status == WARN
    assert "systemctl" in check.detail


# ---------------------------------------------------------------------- ports

def test_the_expected_port_shape_on_an_idle_node(monkeypatch):
    monkeypatch.setattr(doc, "tcp_listening", lambda port, **kw: port == 3000)
    checks = doc.check_ports(NodeConfig(model=None, discovery_port=5679),
                             udp_bound={5679})
    assert [c.status for c in checks] == [OK, doc.INFO, OK]
    assert "idle shape" in _by_name(checks, "port.engine").detail


def test_a_dead_dashboard_port_is_a_fail(monkeypatch):
    monkeypatch.setattr(doc, "tcp_listening", lambda port, **kw: False)
    checks = doc.check_ports(NodeConfig(model=None), udp_bound={5679})
    assert _by_name(checks, "port.web").status == FAIL


def test_a_pinned_model_with_no_engine_port_warns(monkeypatch):
    monkeypatch.setattr(doc, "tcp_listening", lambda port, **kw: port == 3000)
    checks = doc.check_ports(NodeConfig(model="org/m"), udp_bound={5679})
    assert _by_name(checks, "port.engine").status == WARN


def test_an_unbound_discovery_port_warns(monkeypatch):
    monkeypatch.setattr(doc, "tcp_listening", lambda port, **kw: True)
    checks = doc.check_ports(NodeConfig(discovery_port=5679), udp_bound=set())
    assert _by_name(checks, "port.discovery").status == WARN


def test_not_knowing_whether_udp_is_bound_warns(monkeypatch):
    monkeypatch.setattr(doc, "tcp_listening", lambda port, **kw: True)
    monkeypatch.setattr(doc, "udp_listeners", lambda: None)
    checks = doc.check_ports(NodeConfig())
    assert _by_name(checks, "port.discovery").data["listening"] is None


def test_udp_listeners_parses_ss_output(monkeypatch):
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0: (0, (
        "UNCONN 0 0 0.0.0.0:5679 0.0.0.0:*\n"
        "UNCONN 0 0 127.0.0.1:323  0.0.0.0:*\n"
        "UNCONN 0 0    [::]:5353    [::]:*\n")))
    assert doc.udp_listeners() == {5679, 323, 5353}


def test_udp_listeners_is_none_when_ss_is_missing(monkeypatch):
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0: (127, "nope"))
    assert doc.udp_listeners() is None


# ---------------------------------------------------------------------- peers

def _nodes_payload(*rows):
    return {"nodes": list(rows)}


def test_a_solo_node_with_no_peers_is_ok(monkeypatch):
    monkeypatch.setattr(doc, "http_json", lambda url, timeout=3.0, headers=None:
                        _nodes_payload({"node_id": "me"}))
    check = _one(doc.check_peers(NodeConfig(node_id="me"), "0.5.26"))
    assert check.status == OK
    assert "solo shape" in check.detail


def test_configured_peers_that_discovery_cannot_see_warn(monkeypatch):
    monkeypatch.setattr(doc, "http_json", lambda url, timeout=3.0, headers=None:
                        _nodes_payload({"node_id": "me"}))
    check = _one(doc.check_peers(
        NodeConfig(node_id="me", peer_ips=["10.0.0.2", "10.0.0.3"]), "0.5.26"))
    assert check.status == WARN
    assert "sees none" in check.detail


def test_a_fleet_on_one_release_is_ok(monkeypatch):
    def fake(url, timeout=3.0, headers=None):
        if url.endswith("/api/nodes"):
            return _nodes_payload(
                {"node_id": "me"},
                {"node_id": "s3", "node_name": "Spark-3", "fabric_ip": "10.0.0.3",
                 "web_port": 3000})
        return {"version": "0.5.26"}

    monkeypatch.setattr(doc, "http_json", fake)
    checks = doc.check_peers(NodeConfig(node_id="me"), "0.5.26")
    assert _by_name(checks, "cluster.peers").status == OK
    assert _by_name(checks, "cluster.versions").status == OK


def test_a_fleet_split_across_two_releases_warns(monkeypatch):
    """The standing rule is one release across every node; a split is a finding."""
    def fake(url, timeout=3.0, headers=None):
        if url.endswith("/api/nodes"):
            return _nodes_payload(
                {"node_id": "me"},
                {"node_id": "s3", "node_name": "Spark-3", "fabric_ip": "10.0.0.3",
                 "web_port": 3000})
        return {"version": "0.5.25"}

    monkeypatch.setattr(doc, "http_json", fake)
    check = _by_name(doc.check_peers(NodeConfig(node_id="me"), "0.5.26"),
                     "cluster.versions")
    assert check.status == WARN
    assert "Spark-3 on 0.5.25" in check.detail


def test_a_peer_announcing_no_fabric_ip_is_its_own_finding(monkeypatch):
    monkeypatch.setattr(doc, "http_json", lambda url, timeout=3.0, headers=None: (
        _nodes_payload({"node_id": "me"},
                       {"node_id": "s3", "node_name": "Spark-3", "fabric_ip": ""})
        if url.endswith("/api/nodes") else {"version": "0.5.26"}))
    checks = doc.check_peers(NodeConfig(node_id="me"), "0.5.26")
    assert _by_name(checks, "cluster.fabric_ip").status == WARN
    assert _by_name(checks, "cluster.versions").status == WARN  # unknown, not equal


def test_an_unreachable_peer_is_reported_as_unreachable(monkeypatch):
    def fake(url, timeout=3.0, headers=None):
        if url.endswith("/api/nodes"):
            return _nodes_payload(
                {"node_id": "me"},
                {"node_id": "s3", "node_name": "Spark-3", "fabric_ip": "10.0.0.3",
                 "web_port": 3000})
        return None

    monkeypatch.setattr(doc, "http_json", fake)
    checks = doc.check_peers(NodeConfig(node_id="me"), "0.5.26")
    assert _by_name(checks, "cluster.reachable").status == WARN


def test_a_local_api_that_is_down_cannot_enumerate_peers(monkeypatch):
    monkeypatch.setattr(doc, "http_json", lambda url, timeout=3.0, headers=None: None)
    check = _one(doc.check_peers(NodeConfig(node_id="me"), "0.5.26"))
    assert check.status == WARN
    assert check.data == {"reachable": False}


# --------------------------------------------------------------------- fabric

def test_no_fabric_and_no_hca_is_ok(monkeypatch):
    monkeypatch.setattr("ainode.cluster.hca_discovery.list_local_hcas", lambda: [])
    check = _one(doc.check_fabric(NodeConfig(cluster_interface="")))
    assert check.status == OK


def test_an_unused_fabric_warns(monkeypatch):
    monkeypatch.setattr("ainode.cluster.hca_discovery.list_local_hcas",
                        lambda: ["mlx5_0", "mlx5_1"])
    monkeypatch.setattr("ainode.cluster.hca_discovery.hca_port_active", lambda h: True)
    check = _one(doc.check_fabric(NodeConfig(cluster_interface="")))
    assert check.status == WARN
    assert "mlx5_0" in check.detail


def test_a_configured_interface_with_no_address_fails(monkeypatch):
    monkeypatch.setattr("ainode.cluster.hca_discovery.list_local_hcas", lambda: ["mlx5_0"])
    monkeypatch.setattr("ainode.cluster.hca_discovery.hca_port_active", lambda h: True)
    monkeypatch.setattr("ainode.cluster.hca_discovery.detect_fabric_ip", lambda i: None)
    check = _one(doc.check_fabric(NodeConfig(cluster_interface="enp1s0f0np0")))
    assert check.status == FAIL
    assert "enp1s0f0np0" in check.detail


def test_a_configured_interface_with_an_address_is_ok(monkeypatch):
    monkeypatch.setattr("ainode.cluster.hca_discovery.list_local_hcas", lambda: ["mlx5_0"])
    monkeypatch.setattr("ainode.cluster.hca_discovery.hca_port_active", lambda h: True)
    monkeypatch.setattr("ainode.cluster.hca_discovery.detect_fabric_ip",
                        lambda i: "10.0.0.2")
    check = _one(doc.check_fabric(NodeConfig(cluster_interface="enp1s0f0np0")))
    assert check.status == OK
    assert "10.0.0.2" in check.detail and "mlx5_0" in check.detail


# ---------------------------------------------------------- distributed shape

def _record(tmp_path, **kw) -> Path:
    record = {"model": "fraserprice/DeepSeek-V4-Flash-DSpark", "api_port": 8000,
              "peer_ips": ["10.100.0.15"], "tensor_parallel_size": 2,
              "distributed_executor": "mp", "status": "serving"}
    record.update(kw)
    path = tmp_path / "distributed.json"
    path.write_text(json.dumps(record))
    return path


def test_no_distributed_record_is_ok(tmp_path):
    check = _one(doc.check_distributed(tmp_path))
    assert check.status == OK
    assert check.data["exists"] is False


def test_a_healthy_distributed_record_reports_the_shape(tmp_path):
    _record(tmp_path)
    check = _one(doc.check_distributed(tmp_path))
    assert check.status == OK
    assert "TP=2" in check.detail
    assert check.data["peer_ips"] == ["10.100.0.15"]


def test_a_degraded_distributed_record_warns_with_the_peer_named(tmp_path):
    """Nothing retries a degraded shape, so the doctor is where a human sees it."""
    _record(tmp_path, status="degraded",
            degraded_reason=("DeepSeek TP=2 is not running here and 1 of 1 peer(s) "
                             "did not answer: 10.100.0.15 answered 'No route to host'"),
            degraded_peers=[{"peer_ip": "10.100.0.15", "answer": "No route to host"}])
    check = _one(doc.check_distributed(tmp_path))
    assert check.status == WARN
    assert "No route to host" in check.detail
    assert check.data["peers_unreachable"] == ["10.100.0.15"]
    assert "10.100.0.15" in check.fix


def test_an_unreadable_distributed_record_is_a_warn_not_a_crash(tmp_path):
    (tmp_path / "distributed.json").write_text("{not json")
    check = _one(doc.check_distributed(tmp_path))
    assert check.status == WARN
    assert "cannot read" in check.detail


# -------------------------------------------------------------------- secrets

def test_no_secrets_store_is_ok(tmp_path):
    assert _one(doc.check_secrets(tmp_path)).status == OK


def test_a_secrets_store_at_0600_is_ok(tmp_path):
    path = tmp_path / "secrets.json"
    path.write_text("{}")
    os.chmod(path, 0o600)
    assert _one(doc.check_secrets(tmp_path)).status == OK


def test_a_world_readable_secrets_store_warns_and_is_fixable(tmp_path):
    path = tmp_path / "secrets.json"
    path.write_text("{}")
    os.chmod(path, 0o644)
    check = _one(doc.check_secrets(tmp_path))
    assert check.status == WARN
    assert check.data["fix_action"] == "chmod600"


# -------------------------------------------------------------------- login

def _write_users(home, records, mode=0o600):
    """``users.json`` as the account store writes it: hashes included, 0600."""
    path = home / "users.json"
    path.write_text(json.dumps({"users": records}))
    os.chmod(path, mode)
    return path


def _admin(name="jason", disabled=False):
    return {"name": name, "role": "admin", "disabled": disabled,
            "password_hash": "hash"}


def _member(name="ops"):
    return {"name": name, "role": "member", "disabled": False,
            "password_hash": "hash"}


def _auth_on(home, enabled=True):
    (home / "auth.json").write_text(json.dumps(
        {"enabled": enabled, "api_keys": [{"id": "k1", "key_hash": "h"}]}))


def test_auth_on_with_an_admin_account_is_the_pass(tmp_path):
    _auth_on(tmp_path)
    _write_users(tmp_path, [_admin(), _member()])
    checks = doc.check_login(NodeConfig(), tmp_path)

    state = _by_name(checks, "login.state")
    assert state.status == OK
    assert state.data["users"] == 2 and state.data["admins"] == 1
    assert _by_name(checks, "login.store").status == OK


def test_auth_off_is_a_pass_whether_or_not_anybody_has_an_account(tmp_path):
    """Nothing is refused without a credential, so a missing login is not a finding."""
    check = _by_name(doc.check_login(NodeConfig(), tmp_path), "login.state")
    assert check.status == OK
    assert "without a login" in check.detail

    _write_users(tmp_path, [_admin()])
    check = _by_name(doc.check_login(NodeConfig(), tmp_path), "login.state")
    assert check.status == OK
    assert "unused until auth is on" in check.detail


def test_auth_on_with_no_account_at_all_is_the_fail(tmp_path):
    """The dashboard is then reachable only by pasting an API key, which is the
    thing the login exists to replace."""
    _auth_on(tmp_path)
    check = _by_name(doc.check_login(NodeConfig(), tmp_path), "login.state")

    assert check.status == FAIL
    assert "only be opened by pasting an API key" in check.detail
    assert "ainode auth user add" in check.fix
    assert doc.exit_code([check]) == 1


def test_auth_on_with_accounts_but_no_enabled_admin_warns(tmp_path):
    _auth_on(tmp_path)
    _write_users(tmp_path, [_admin(disabled=True), _member()])
    check = _by_name(doc.check_login(NodeConfig(), tmp_path), "login.state")

    assert check.status == WARN
    assert "no enabled admin" in check.detail


def test_a_world_readable_users_file_warns_and_is_fixable(tmp_path):
    """It holds password hashes, so it is 0600 like auth.json and secrets.json."""
    _auth_on(tmp_path)
    _write_users(tmp_path, [_admin()], mode=0o644)
    check = _by_name(doc.check_login(NodeConfig(), tmp_path), "login.store")

    assert check.status == WARN
    assert check.data["fix_action"] == "chmod600"
    assert "password hashes" in check.detail


def test_the_login_fix_action_is_applied_by_fix(tmp_path):
    _auth_on(tmp_path)
    path = _write_users(tmp_path, [_admin()], mode=0o644)
    checks = doc.check_login(NodeConfig(), tmp_path)

    done = doc.apply_fixes(checks, tmp_path / "config.json")

    assert any("chmod 0600" in line for line in done)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_an_unreadable_users_file_warns_rather_than_crashing(tmp_path):
    _auth_on(tmp_path)
    (tmp_path / "users.json").write_text("{not json")
    check = _by_name(doc.check_login(NodeConfig(), tmp_path), "login.state")

    assert check.status == WARN
    assert "cannot read" in check.detail


def test_a_users_file_in_any_of_the_stores_shapes_is_counted(tmp_path):
    """The file belongs to the account store, so the doctor reads it tolerantly
    rather than importing a module that may not be there."""
    for raw in ({"users": [_admin()]},
                {"users": {"jason": {"role": "admin"}}},
                [_admin()]):
        assert [r["role"] for r in doc.users_file_records(raw)] == ["admin"]
    assert doc.users_file_records({"users": []}) == []
    assert doc.users_file_records("nonsense") == []


def test_a_worker_whose_accounts_match_the_master_is_ok(tmp_path, monkeypatch):
    from ainode.auth import replication as rep

    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    _auth_on(tmp_path)
    _write_users(tmp_path, [_admin()])
    rep.write_sync_state("master-stamp-1", master="http://10.0.0.1:3000", users=1,
                         home=tmp_path)
    monkeypatch.setattr(doc, "http_json",
                        lambda url, timeout=3.0, headers=None:
                        {"users": [], "stamp": "master-stamp-1"})
    config = NodeConfig(cluster_role="worker", master_address="10.0.0.1:3000",
                        cluster_secret="s" * 32)

    check = _by_name(doc.check_login(config, tmp_path), "login.sync")

    assert check.status == OK


def test_a_worker_behind_the_masters_stamp_warns(tmp_path, monkeypatch):
    """The password somebody just set on the master does not work here yet, and the
    two stamps compared are both the MASTER's own, so they cannot drift."""
    from ainode.auth import replication as rep
    from ainode.auth.fleet import fleet_key

    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    _auth_on(tmp_path)
    _write_users(tmp_path, [_admin()])
    rep.write_sync_state("master-stamp-1", master="http://10.0.0.1:3000", users=1,
                         home=tmp_path)
    seen = []

    def fake_http_json(url, timeout=3.0, headers=None):
        seen.append({"url": url, "headers": headers or {}})
        return {"users": [], "stamp": "master-stamp-2"}

    monkeypatch.setattr(doc, "http_json", fake_http_json)
    config = NodeConfig(cluster_role="worker", master_address="10.0.0.1:3000",
                        cluster_secret="s" * 32)

    check = _by_name(doc.check_login(config, tmp_path), "login.sync")

    assert check.status == WARN
    assert "older than the master's" in check.detail
    # Asked of the master, with the fleet key, like every other node-to-node read.
    assert seen[0]["url"] == "http://10.0.0.1:3000/api/auth/users/export"
    assert seen[0]["headers"]["Authorization"] == f"Bearer {fleet_key('s' * 32)}"


def test_a_worker_that_has_never_pulled_warns(tmp_path, monkeypatch):
    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    _auth_on(tmp_path)
    monkeypatch.setattr(doc, "http_json",
                        lambda url, timeout=3.0, headers=None:
                        {"users": [_admin()], "stamp": "master-stamp-1"})
    config = NodeConfig(cluster_role="worker", master_address="10.0.0.1:3000")

    check = _by_name(doc.check_login(config, tmp_path), "login.sync")

    assert check.status == WARN
    assert "never imported" in check.detail


def test_a_worker_whose_master_does_not_answer_warns(tmp_path, monkeypatch):
    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    _auth_on(tmp_path)
    _write_users(tmp_path, [_admin()])
    monkeypatch.setattr(doc, "http_json", lambda url, timeout=3.0, headers=None: None)
    config = NodeConfig(cluster_role="worker", master_address="10.0.0.1:3000")

    check = _by_name(doc.check_login(config, tmp_path), "login.sync")

    assert check.status == WARN
    assert "did not answer" in check.detail


def test_a_worker_with_no_master_named_anywhere_warns(tmp_path, monkeypatch):
    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    monkeypatch.setattr(doc, "http_json",
                        lambda url, timeout=3.0, headers=None:
                        pytest.fail("there is nobody to ask"))
    config = NodeConfig(cluster_role="worker")

    check = _by_name(doc.check_login(config, tmp_path), "login.sync")

    assert check.status == WARN
    assert "nothing names its master" in check.detail
    assert "ainode join" in check.fix


def test_a_master_and_a_solo_node_ask_nobody_anything(tmp_path, monkeypatch):
    """The sync check is the one part of this that leaves the box, so it runs on a
    worker and nowhere else."""
    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    monkeypatch.setattr(doc, "http_json",
                        lambda url, timeout=3.0, headers=None:
                        pytest.fail("a master pulls from nobody"))
    for config in (NodeConfig(cluster_role="master", peer_ips=["10.0.0.2"]),
                   NodeConfig()):
        names = [c.name for c in doc.check_login(config, tmp_path)]
        assert "login.sync" not in names


# ------------------------------------------------------------------ hf token

def test_an_hf_token_in_config_is_reported_without_its_value(tmp_path):
    secret = "hf_" + "x" * 34
    check = _one(doc.check_hf_token(NodeConfig(hf_token=secret), tmp_path, env={}))
    assert check.status == OK
    assert secret not in check.detail
    assert secret not in json.dumps(check.data)
    assert check.data["sources"] == ["config.json"]


def test_an_hf_token_in_the_environment_counts(tmp_path):
    check = _one(doc.check_hf_token(NodeConfig(hf_token=None), tmp_path,
                                   env={"HF_TOKEN": "hf_abc"}))
    assert check.status == OK
    assert check.data["sources"] == ["$HF_TOKEN"]


def test_no_hf_token_anywhere_warns(tmp_path):
    check = _one(doc.check_hf_token(NodeConfig(hf_token=None), tmp_path, env={}))
    assert check.status == WARN
    assert "401" in check.detail


# ----------------------------------------------------------------- exit codes

def test_exit_code_is_zero_when_the_worst_answer_is_a_warning():
    checks = [doc.Check("a", OK, ""), doc.Check("b", WARN, "")]
    assert doc.exit_code(checks) == 0


def test_exit_code_is_non_zero_on_any_fail():
    checks = [doc.Check("a", OK, ""), doc.Check("b", FAIL, "")]
    assert doc.exit_code(checks) == 1


def test_summarize_counts_every_status():
    checks = [doc.Check("a", OK, ""), doc.Check("b", WARN, ""), doc.Check("c", FAIL, ""),
              doc.Check("d", WARN, "")]
    assert doc.summarize(checks) == {OK: 1, WARN: 2, FAIL: 1, "total": 4}


# ------------------------------------------------------------------ json shape

def test_the_json_payload_shape_is_a_contract():
    checks = [doc.Check("config.file", OK, "parsed", data={"path": "/x"}),
              doc.Check("gpu.devices", FAIL, "none", fix="check nvidia-smi")]
    payload = doc.report_payload(checks, NodeConfig(node_id="n1", node_name="Spark-2"))
    assert payload["doctor"] == 1
    assert payload["node_id"] == "n1" and payload["node_name"] == "Spark-2"
    assert payload["summary"] == {OK: 1, WARN: 0, FAIL: 1, "total": 2}
    assert payload["checks"][1] == {
        "name": "gpu.devices", "status": FAIL, "detail": "none",
        "fix": "check nvidia-smi", "data": {}}
    assert payload["generated_at"].endswith("Z")
    json.dumps(payload)  # it has to actually serialize


# ------------------------------------------------------------------ --fix

def test_fix_creates_a_missing_dir_chmods_the_store_and_writes_the_port(tmp_path):
    secrets = tmp_path / "secrets.json"
    secrets.write_text("{}")
    os.chmod(secrets, 0o644)
    config_path = tmp_path / "config.json"
    config_path.write_text('{"node_id": "keep-me", "discovery_port": 5678}')
    checks = [
        doc.Check("disk.models", WARN, "", data={"path": str(tmp_path / "models"),
                                                 "fix_action": "mkdir"}),
        doc.Check("secrets.store", WARN, "", data={"path": str(secrets),
                                                  "fix_action": "chmod600"}),
        doc.Check("config.discovery_port", WARN, "", data={"expected": 5679,
                                                           "fix_action": "discovery_port"}),
    ]
    done = doc.apply_fixes(checks, config_path)
    assert len(done) == 3
    assert (tmp_path / "models").is_dir()
    assert stat.S_IMODE(secrets.stat().st_mode) == 0o600
    written = json.loads(config_path.read_text())
    # It writes ONE key. Everything else in the file survives untouched.
    assert written == {"node_id": "keep-me", "discovery_port": 5679}


def test_fix_touches_nothing_that_is_already_ok(tmp_path):
    checks = [doc.Check("disk.models", OK, "", data={"path": str(tmp_path / "nope"),
                                                     "fix_action": "mkdir"})]
    assert doc.apply_fixes(checks, tmp_path / "config.json") == []
    assert not (tmp_path / "nope").exists()


def test_fix_never_touches_a_check_with_no_fix_action(tmp_path):
    checks = [doc.Check("image.latest", WARN, "", fix="ainode update", data={})]
    assert doc.apply_fixes(checks, tmp_path / "config.json") == []


# ------------------------------------------------------------------- --peer

_PEER_REPORT = {
    "doctor": 1, "version": "0.5.26", "summary": {"ok": 1, "warn": 0, "fail": 1,
                                                  "total": 2},
    "checks": [{"name": "docker.daemon", "status": "ok", "detail": "docker 28 reachable"},
               {"name": "gpu.devices", "status": "fail", "detail": "none",
                "fix": "check nvidia-smi"}],
}


def test_peer_runs_the_doctor_over_ssh_and_parses_its_report(monkeypatch):
    seen = {}

    def fake_run(argv, timeout=10.0):
        seen["argv"] = argv
        return 1, "some ssh banner\n" + json.dumps(_PEER_REPORT)

    monkeypatch.setattr(doc, "run_command", fake_run)
    checks, payload = doc.peer_checks("spark3")
    assert seen["argv"][0] == "ssh" and "spark3" in seen["argv"]
    # The container FIRST and with no -it: the host wrapper the installer writes
    # runs `docker exec -it`, which over SSH dies on "the input device is not a
    # TTY" before it ever reaches the doctor (found against Spark-3).
    remote = seen["argv"][-1]
    assert remote.startswith("docker exec ainode ainode doctor --json")
    assert "-it" not in remote
    assert remote.endswith("|| ainode doctor --json")
    assert payload == _PEER_REPORT
    assert [(c.name, c.status) for c in checks] == [("docker.daemon", "ok"),
                                                   ("gpu.devices", "fail")]
    assert doc.exit_code(checks) == 1


def test_a_peer_still_running_the_old_stub_is_named_as_such(monkeypatch):
    """What every peer answered on 2026-09-19: the released doctor was a stub."""
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0:
                        (0, "ainode doctor - stub (coming in v0.5.0)"))
    checks, payload = doc.peer_checks("spark3")
    assert payload is None
    assert _one(checks).name == "peer.version"
    assert _one(checks).status == FAIL
    assert "roll spark3" in _one(checks).fix


def test_a_peer_that_answers_nothing_useful_is_one_failed_check(monkeypatch):
    monkeypatch.setattr(doc, "run_command",
                        lambda argv, timeout=10.0: (255, "Permission denied (publickey)."))
    checks, payload = doc.peer_checks("spark9")
    assert payload is None
    assert _one(checks).name == "peer.ssh"
    assert _one(checks).status == FAIL
    assert "publickey" in _one(checks).detail


# ---------------------------------------------------------- the command itself

def test_run_checks_produces_one_check_per_name_and_no_exceptions(tmp_path, monkeypatch):
    """The assembly runs end to end with every seam faked, and never repeats a name."""
    (tmp_path / "config.json").write_text('{"node_id": "n1", "model": null}')
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0: (127, "absent"))
    monkeypatch.setattr(doc, "disk_usage", lambda path: (1000, 900))
    monkeypatch.setattr(doc, "tcp_listening", lambda port, **kw: False)
    monkeypatch.setattr(doc, "udp_listeners", lambda: None)
    monkeypatch.setattr(doc, "http_json", lambda url, timeout=3.0, headers=None: None)
    monkeypatch.setattr(doc, "latest_image_tag", lambda: None)
    monkeypatch.setattr(doc, "probe_gpus", lambda: [])
    checks = doc.run_checks(tmp_path, tmp_path / "config.json")
    names = [c.name for c in checks]
    assert len(names) == len(set(names))
    for expected in ("env.sudo", "config.file", "config.engine_backend",
                     "config.gpu_memory_utilization", "config.discovery_port",
                     "config.model", "docker.daemon", "gpu.devices", "disk.home",
                     "disk.models", "image.pin", "image.latest", "service.unit",
                     "port.web", "port.engine", "port.discovery", "config.cluster_id",
                     "cluster.peers", "fabric.interface", "cluster.distributed",
                     "secrets.store", "credentials.hf_token"):
        assert expected in names


def _fake_world(monkeypatch, tmp_path):
    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0: (127, "absent"))
    monkeypatch.setattr(doc, "disk_usage", lambda path: (1000, 900))
    monkeypatch.setattr(doc, "tcp_listening", lambda port, **kw: False)
    monkeypatch.setattr(doc, "udp_listeners", lambda: None)
    monkeypatch.setattr(doc, "http_json", lambda url, timeout=3.0, headers=None: None)
    monkeypatch.setattr(doc, "latest_image_tag", lambda: None)
    monkeypatch.setattr(doc, "probe_gpus", lambda: [{
        "index": 0, "name": "NVIDIA GB10", "memory_total_mb": 124620,
        "memory_free_mb": 90000, "unified_memory": True}])


def test_the_command_prints_one_line_per_check(tmp_path, monkeypatch, capsys):
    _fake_world(monkeypatch, tmp_path)
    (tmp_path / "config.json").write_text('{"node_name": "Spark-2-DGX", "model": null}')
    with patch.object(sys, "argv", ["ainode", "doctor"]):
        from ainode.cli.main import main
        with pytest.raises(SystemExit) as exc:
            main()
    out = capsys.readouterr().out
    assert "AINode doctor" in out
    assert "Made in Texas" in out  # the brand rule applies to CLI output too
    assert "Spark-2-DGX" in out
    assert "config.discovery_port" in out
    assert "checks:" in out
    # port.web is a FAIL with nothing listening, so the run is non-zero.
    assert exc.value.code == 1


def test_the_command_emits_json_with_json(tmp_path, monkeypatch, capsys):
    _fake_world(monkeypatch, tmp_path)
    (tmp_path / "config.json").write_text('{"node_id": "n1", "model": null}')
    with patch.object(sys, "argv", ["ainode", "doctor", "--json"]):
        from ainode.cli.main import main
        with pytest.raises(SystemExit):
            main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["doctor"] == 1
    assert payload["node_id"] == "n1"
    assert payload["summary"]["total"] == len(payload["checks"])


def test_fix_reports_what_it_applied_and_what_is_left(tmp_path, monkeypatch, capsys):
    _fake_world(monkeypatch, tmp_path)
    (tmp_path / "config.json").write_text('{"model": null, "discovery_port": 5678}')
    with patch.object(sys, "argv", ["ainode", "doctor", "--fix"]):
        from ainode.cli.main import main
        with pytest.raises(SystemExit):
            main()
    out = capsys.readouterr().out
    assert "fixed" in out
    assert "left for a human" in out
    assert json.loads((tmp_path / "config.json").read_text())["discovery_port"] == 5679


def test_json_and_fix_together_report_the_fixes_in_the_payload(tmp_path, monkeypatch,
                                                              capsys):
    _fake_world(monkeypatch, tmp_path)
    (tmp_path / "config.json").write_text('{"model": null, "discovery_port": 5678}')
    with patch.object(sys, "argv", ["ainode", "doctor", "--json", "--fix"]):
        from ainode.cli.main import main
        with pytest.raises(SystemExit):
            main()
    payload = json.loads(capsys.readouterr().out)
    assert any("discovery_port=5679" in line for line in payload["fixes_applied"])


def test_the_peer_flag_renders_the_peers_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(doc, "run_command",
                        lambda argv, timeout=10.0: (0, json.dumps(_PEER_REPORT)))
    with patch.object(sys, "argv", ["ainode", "doctor", "--peer", "spark3"]):
        from ainode.cli.main import main
        with pytest.raises(SystemExit) as exc:
            main()
    out = capsys.readouterr().out
    assert "peer spark3" in out
    assert "gpu.devices" in out
    assert exc.value.code == 1
