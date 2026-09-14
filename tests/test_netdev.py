"""Unit tests for ainode.cluster.netdev: cluster-interface autodetection.

No real subprocess, no real sysfs, no sleeps. Both seams are injected: a
fake ``/sys/class/net`` tree under tmp_path and a command runner that
returns canned ``ip`` output.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ainode.cluster import netdev
from ainode.core.config import NodeConfig


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _make_sysfs(root: Path, devices: dict) -> Path:
    """Build a fake /sys/class/net.

    ``devices`` maps name -> {"operstate": str, "rdma": bool}.
    """
    root.mkdir(parents=True, exist_ok=True)
    for name, spec in devices.items():
        dev = root / name
        dev.mkdir(parents=True, exist_ok=True)
        state = spec.get("operstate")
        if state is not None:
            (dev / "operstate").write_text(state + "\n")
        if spec.get("rdma"):
            (dev / "device" / "infiniband").mkdir(parents=True, exist_ok=True)
    return root


def _make_runner(addr_lines: list, default_route: str = ""):
    """Return a CommandRunner serving canned ``ip`` output."""
    addr_out = "\n".join(addr_lines) + ("\n" if addr_lines else "")

    def run(argv):
        argv = list(argv)
        if "addr" in argv:
            return addr_out
        if "route" in argv:
            return default_route
        return ""

    return run


def _addr(index: int, name: str, ip: str, cidr: int = 24) -> str:
    return (
        f"{index}: {name}    inet {ip}/{cidr} brd 10.255.255.255 "
        f"scope global {name}\\       valid_lft forever preferred_lft forever"
    )


# ---------------------------------------------------------------------------
# list_ipv4_interfaces
# ---------------------------------------------------------------------------

def test_list_ipv4_interfaces_parses_name_ip_state_rdma(tmp_path):
    sysfs = _make_sysfs(tmp_path / "net", {
        "enp1s0f0np0": {"operstate": "up", "rdma": True},
        "eth0": {"operstate": "up"},
    })
    runner = _make_runner(
        [_addr(5, "enp1s0f0np0", "192.168.6.162"), _addr(2, "eth0", "10.1.0.9")],
        default_route="default via 10.1.0.1 dev eth0 proto dhcp metric 100\n",
    )

    devices = netdev.list_ipv4_interfaces(sysfs_root=sysfs, runner=runner)

    assert [d.name for d in devices] == ["enp1s0f0np0", "eth0"]  # sorted by name
    fabric, mgmt = devices
    assert fabric.ipv4 == "192.168.6.162"
    assert fabric.is_up is True
    assert fabric.rdma is True
    assert fabric.is_default_route is False
    assert mgmt.ipv4 == "10.1.0.9"
    assert mgmt.rdma is False
    assert mgmt.is_default_route is True


def test_list_ipv4_interfaces_excludes_virtual_devices(tmp_path):
    virtual = [
        "lo", "docker0", "br-abc123", "veth1a2b", "virbr0", "tailscale0",
        "wg0", "tun0", "tap0", "cni0", "flannel.1", "cali123", "kube-ipvs0",
        "lxcbr0", "ztabcdefgh",
    ]
    spec = {name: {"operstate": "up"} for name in virtual}
    spec["enp1s0f0np0"] = {"operstate": "up"}
    sysfs = _make_sysfs(tmp_path / "net", spec)
    lines = [
        _addr(i + 1, name, f"172.30.{i}.1") for i, name in enumerate(virtual)
    ] + [_addr(99, "enp1s0f0np0", "192.168.6.162")]

    devices = netdev.list_ipv4_interfaces(sysfs_root=sysfs, runner=_make_runner(lines))

    assert [d.name for d in devices] == ["enp1s0f0np0"]


def test_list_ipv4_interfaces_skips_link_local_addresses(tmp_path):
    sysfs = _make_sysfs(tmp_path / "net", {"eth0": {"operstate": "up"}})
    runner = _make_runner([_addr(2, "eth0", "169.254.7.7", cidr=16)])

    assert netdev.list_ipv4_interfaces(sysfs_root=sysfs, runner=runner) == []


def test_list_ipv4_interfaces_tolerates_missing_ip_binary(tmp_path):
    sysfs = _make_sysfs(tmp_path / "net", {"eth0": {"operstate": "up"}})

    assert netdev.list_ipv4_interfaces(sysfs_root=sysfs, runner=lambda argv: "") == []


def test_down_interface_reports_is_up_false(tmp_path):
    sysfs = _make_sysfs(tmp_path / "net", {"eth0": {"operstate": "down"}})
    runner = _make_runner([_addr(2, "eth0", "10.1.0.9")])

    (dev,) = netdev.list_ipv4_interfaces(sysfs_root=sysfs, runner=runner)
    assert dev.is_up is False


def test_unreadable_operstate_is_treated_as_up(tmp_path):
    """No sysfs entry must not make a device carrying an IPv4 look down."""
    sysfs = _make_sysfs(tmp_path / "net", {})
    runner = _make_runner([_addr(2, "eth0", "10.1.0.9")])

    (dev,) = netdev.list_ipv4_interfaces(sysfs_root=sysfs, runner=runner)
    assert dev.is_up is True


def test_multipath_default_route_marks_every_nexthop_device(tmp_path):
    sysfs = _make_sysfs(tmp_path / "net", {
        "eth0": {"operstate": "up"}, "eth1": {"operstate": "up"},
    })
    runner = _make_runner(
        [_addr(2, "eth0", "10.1.0.9"), _addr(3, "eth1", "10.2.0.9")],
        default_route=(
            "default \n"
            "\tnexthop via 10.1.0.1 dev eth0 weight 1\n"
            "\tnexthop via 10.2.0.1 dev eth1 weight 1\n"
        ),
    )

    devices = netdev.list_ipv4_interfaces(sysfs_root=sysfs, runner=runner)
    assert all(d.is_default_route for d in devices)


# ---------------------------------------------------------------------------
# detect_cluster_interface ranking
# ---------------------------------------------------------------------------

def test_rdma_interface_beats_the_default_route(tmp_path):
    sysfs = _make_sysfs(tmp_path / "net", {
        "enp1s0f0np0": {"operstate": "up", "rdma": True},
        "eth0": {"operstate": "up"},
    })
    runner = _make_runner(
        [_addr(5, "enp1s0f0np0", "192.168.6.162"), _addr(2, "eth0", "10.1.0.9")],
        default_route="default via 10.1.0.1 dev eth0 proto dhcp metric 100\n",
    )

    assert netdev.detect_cluster_interface(
        sysfs_root=sysfs, runner=runner
    ) == "enp1s0f0np0"


def test_default_route_wins_when_no_rdma_device(tmp_path):
    sysfs = _make_sysfs(tmp_path / "net", {
        "eth0": {"operstate": "up"}, "eth9": {"operstate": "up"},
    })
    runner = _make_runner(
        [_addr(2, "eth9", "10.9.0.9"), _addr(3, "eth0", "10.1.0.9")],
        default_route="default via 10.1.0.1 dev eth0 proto dhcp metric 100\n",
    )

    assert netdev.detect_cluster_interface(sysfs_root=sysfs, runner=runner) == "eth0"


def test_falls_back_to_any_up_interface_with_an_ipv4(tmp_path):
    sysfs = _make_sysfs(tmp_path / "net", {"eth7": {"operstate": "up"}})
    runner = _make_runner([_addr(2, "eth7", "10.7.0.9")])

    assert netdev.detect_cluster_interface(sysfs_root=sysfs, runner=runner) == "eth7"


def test_down_interfaces_are_never_candidates(tmp_path):
    sysfs = _make_sysfs(tmp_path / "net", {
        "enp1s0f0np0": {"operstate": "down", "rdma": True},
        "eth0": {"operstate": "up"},
    })
    runner = _make_runner(
        [_addr(5, "enp1s0f0np0", "192.168.6.162"), _addr(2, "eth0", "10.1.0.9")],
    )

    assert netdev.detect_cluster_interface(sysfs_root=sysfs, runner=runner) == "eth0"


def test_tie_break_is_deterministic_by_name(tmp_path):
    sysfs = _make_sysfs(tmp_path / "net", {
        "enp9s0f0np0": {"operstate": "up", "rdma": True},
        "enp1s0f0np0": {"operstate": "up", "rdma": True},
    })
    runner = _make_runner([
        _addr(9, "enp9s0f0np0", "192.168.9.9"),
        _addr(5, "enp1s0f0np0", "192.168.6.162"),
    ])

    for _ in range(3):
        assert netdev.detect_cluster_interface(
            sysfs_root=sysfs, runner=runner
        ) == "enp1s0f0np0"


def test_detect_returns_none_when_nothing_qualifies(tmp_path):
    sysfs = _make_sysfs(tmp_path / "net", {"docker0": {"operstate": "up"}})
    runner = _make_runner([_addr(3, "docker0", "172.17.0.1", cidr=16)])

    assert netdev.detect_cluster_interface(sysfs_root=sysfs, runner=runner) is None


def test_preferred_interface_is_honored_when_it_has_an_ipv4(tmp_path):
    sysfs = _make_sysfs(tmp_path / "net", {
        "enp1s0f0np0": {"operstate": "up", "rdma": True},
        "eth0": {"operstate": "up"},
    })
    runner = _make_runner(
        [_addr(5, "enp1s0f0np0", "192.168.6.162"), _addr(2, "eth0", "10.1.0.9")],
    )

    assert netdev.detect_cluster_interface(
        "eth0", sysfs_root=sysfs, runner=runner
    ) == "eth0"


def test_an_explicit_pin_survives_the_virtual_exclusion(tmp_path):
    """A user who deliberately runs the fabric over a tunnel is not overruled."""
    sysfs = _make_sysfs(tmp_path / "net", {
        "tailscale0": {"operstate": "up"}, "eth0": {"operstate": "up"},
    })
    runner = _make_runner(
        [_addr(4, "tailscale0", "100.84.108.16", cidr=32), _addr(2, "eth0", "10.1.0.9")],
    )

    assert netdev.detect_cluster_interface(
        "tailscale0", sysfs_root=sysfs, runner=runner
    ) == "tailscale0"


def test_missing_preferred_interface_falls_through_to_ranking(tmp_path):
    sysfs = _make_sysfs(tmp_path / "net", {
        "enp1s0f0np0": {"operstate": "up", "rdma": True},
    })
    runner = _make_runner([_addr(5, "enp1s0f0np0", "192.168.6.162")])

    assert netdev.detect_cluster_interface(
        "enP2p1s0f1np1", sysfs_root=sysfs, runner=runner
    ) == "enp1s0f0np0"


# ---------------------------------------------------------------------------
# resolve_cluster_interface
# ---------------------------------------------------------------------------

@pytest.fixture
def gx10_host(monkeypatch, tmp_path):
    """A GX10-shaped host: enp1s0f0np0 is the RDMA fabric, eth0 routes."""
    sysfs = _make_sysfs(tmp_path / "net", {
        "enp1s0f0np0": {"operstate": "up", "rdma": True},
        "eth0": {"operstate": "up"},
    })
    runner = _make_runner(
        [_addr(5, "enp1s0f0np0", "192.168.6.162"), _addr(2, "eth0", "10.1.0.9")],
        default_route="default via 10.1.0.1 dev eth0 proto dhcp metric 100\n",
    )
    monkeypatch.setattr(netdev, "SYS_CLASS_NET", sysfs)
    monkeypatch.setattr(netdev, "_run_command", runner)
    netdev.reset_cache()
    return sysfs


def test_resolve_returns_the_configured_interface_when_present(gx10_host):
    config = NodeConfig(cluster_interface="eth0")
    assert netdev.resolve_cluster_interface(config) == "eth0"


def test_resolve_falls_back_when_the_configured_interface_is_absent(gx10_host):
    """Issue #34: config said eno1 / enP2p1s0f1np1, the box has neither."""
    config = NodeConfig(cluster_interface="eno1")
    assert netdev.resolve_cluster_interface(config) == "enp1s0f0np0"


def test_resolve_autodetects_when_unset(gx10_host):
    config = NodeConfig(cluster_interface="")
    assert netdev.resolve_cluster_interface(config) == "enp1s0f0np0"


def test_resolve_warns_once_per_process(gx10_host, caplog):
    config = NodeConfig(cluster_interface="eno1")
    with caplog.at_level(logging.WARNING, logger="ainode.cluster.netdev"):
        for _ in range(5):
            netdev.resolve_cluster_interface(config)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].getMessage() == (
        "cluster_interface 'eno1' is not present on this host; using "
        "'enp1s0f0np0'. Set cluster_interface in ~/.ainode/config.json to pin it."
    )


def test_resolve_does_not_warn_for_a_valid_interface(gx10_host, caplog):
    with caplog.at_level(logging.WARNING, logger="ainode.cluster.netdev"):
        netdev.resolve_cluster_interface(NodeConfig(cluster_interface="eth0"))
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_resolve_keeps_the_configured_name_when_nothing_is_detected(tmp_path, monkeypatch):
    """Downstream errors must still name the interface the user set."""
    monkeypatch.setattr(netdev, "SYS_CLASS_NET", _make_sysfs(tmp_path / "net", {}))
    monkeypatch.setattr(netdev, "_run_command", lambda argv: "")
    netdev.reset_cache()

    config = NodeConfig(cluster_interface="enP2p1s0f1np1")
    assert netdev.resolve_cluster_interface(config) == "enP2p1s0f1np1"


def test_resolve_returns_empty_when_unset_and_undetectable(tmp_path, monkeypatch):
    monkeypatch.setattr(netdev, "SYS_CLASS_NET", _make_sysfs(tmp_path / "net", {}))
    monkeypatch.setattr(netdev, "_run_command", lambda argv: "")
    netdev.reset_cache()

    assert netdev.resolve_cluster_interface(NodeConfig(cluster_interface="")) == ""


def test_resolve_caches_so_the_host_is_read_once(gx10_host, monkeypatch):
    calls = []
    real = netdev._run_command

    def counting(argv):
        calls.append(list(argv))
        return real(argv)

    monkeypatch.setattr(netdev, "_run_command", counting)
    config = NodeConfig(cluster_interface="eno1")
    for _ in range(10):
        netdev.resolve_cluster_interface(config)

    # One addr + one route invocation, not ten of each.
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------

def test_describe_cluster_interface_shows_name_and_ip(gx10_host):
    assert netdev.describe_cluster_interface(
        NodeConfig(cluster_interface="eno1")
    ) == "enp1s0f0np0 (192.168.6.162)"


def test_describe_cluster_interface_says_none_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(netdev, "SYS_CLASS_NET", _make_sysfs(tmp_path / "net", {}))
    monkeypatch.setattr(netdev, "_run_command", lambda argv: "")
    netdev.reset_cache()

    assert netdev.describe_cluster_interface(
        NodeConfig(cluster_interface="")
    ) == "none detected"


def test_interface_candidates_hint_lists_names_and_ips(gx10_host):
    hint = netdev.interface_candidates_hint()
    assert "enp1s0f0np0 (192.168.6.162)" in hint
    assert "eth0 (10.1.0.9)" in hint


def test_interface_candidates_hint_when_nothing_has_an_address(tmp_path, monkeypatch):
    monkeypatch.setattr(netdev, "SYS_CLASS_NET", _make_sysfs(tmp_path / "net", {}))
    monkeypatch.setattr(netdev, "_run_command", lambda argv: "")

    assert netdev.interface_candidates_hint() == (
        "no interface on this host currently carries an IPv4 address"
    )


# ---------------------------------------------------------------------------
# Config default round-trip
# ---------------------------------------------------------------------------

def test_cluster_interface_defaults_to_autodetect():
    assert NodeConfig().cluster_interface == ""


def test_empty_cluster_interface_round_trips_through_disk(tmp_path, monkeypatch):
    from ainode.core import config as config_mod

    config_file = tmp_path / "config.json"
    monkeypatch.setattr(config_mod, "AINODE_HOME", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_FILE", config_file)

    NodeConfig(node_id="abc123").save()
    assert '"cluster_interface": ""' in config_file.read_text()

    reloaded = NodeConfig.load()
    assert reloaded.cluster_interface == ""
    assert reloaded.node_id == "abc123"


def test_pinned_cluster_interface_round_trips_through_disk(tmp_path, monkeypatch):
    from ainode.core import config as config_mod

    config_file = tmp_path / "config.json"
    monkeypatch.setattr(config_mod, "AINODE_HOME", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_FILE", config_file)

    NodeConfig(cluster_interface="enp1s0f0np0").save()
    assert NodeConfig.load().cluster_interface == "enp1s0f0np0"
