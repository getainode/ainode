"""Unit tests for ainode.cluster.hca_discovery.

All tests run on a Mac  -  no live /sys/class/infiniband, no ip, no ping.
We patch ``SYS_INFINIBAND`` at the module level to point at a tmpdir-
backed fake sysfs tree, and patch ``subprocess.run`` for the ip/ping
calls.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from ainode.cluster import hca_discovery


# ---------------------------------------------------------------------------
# Fake sysfs tree builder
# ---------------------------------------------------------------------------


def _make_hca(
    root: Path,
    name: str,
    port_state: str = "4: ACTIVE",
    gids: dict[int, str] | None = None,
) -> None:
    """Create a fake /sys/class/infiniband/<name> tree under ``root``."""
    port = root / name / "ports" / "1"
    port.mkdir(parents=True, exist_ok=True)
    (port / "state").write_text(port_state + "\n")
    gid_dir = port / "gids"
    gid_dir.mkdir(parents=True, exist_ok=True)
    for idx, value in (gids or {}).items():
        (gid_dir / str(idx)).write_text(value + "\n")


@pytest.fixture
def fake_sysfs(tmp_path, monkeypatch):
    """Re-point ``SYS_INFINIBAND`` at a tmpdir and return the Path."""
    root = tmp_path / "sys_infiniband"
    root.mkdir()
    monkeypatch.setattr(hca_discovery, "SYS_INFINIBAND", root)
    return root


# ---------------------------------------------------------------------------
# list_local_hcas
# ---------------------------------------------------------------------------


class TestListLocalHcas:
    def test_empty_when_sysfs_missing(self, tmp_path, monkeypatch):
        missing = tmp_path / "does-not-exist"
        monkeypatch.setattr(hca_discovery, "SYS_INFINIBAND", missing)
        assert hca_discovery.list_local_hcas() == []

    def test_mlx5_naming_sparks_1_and_2(self, fake_sysfs):
        for name in ["mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3"]:
            _make_hca(fake_sysfs, name)
        assert hca_discovery.list_local_hcas() == [
            "mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3",
        ]

    def test_rocep_naming_rdma_core(self, fake_sysfs):
        for name in ["rocep8s0f0", "rocep8s0f1", "roceP8p1s0f0", "roceP8p1s0f1"]:
            _make_hca(fake_sysfs, name)
        # Result is sorted; capital-P sorts before lower-p in ASCII.
        assert hca_discovery.list_local_hcas() == sorted(
            ["rocep8s0f0", "rocep8s0f1", "roceP8p1s0f0", "roceP8p1s0f1"]
        )

    def test_filters_unrecognized_names(self, fake_sysfs):
        _make_hca(fake_sysfs, "mlx5_0")
        _make_hca(fake_sysfs, "loopback")
        _make_hca(fake_sysfs, "ibp0")
        _make_hca(fake_sysfs, "rocep8s0f0")
        assert hca_discovery.list_local_hcas() == ["mlx5_0", "rocep8s0f0"]


# ---------------------------------------------------------------------------
# hca_has_ipv4_rocev2_gid
# ---------------------------------------------------------------------------


class TestHcaHasIpv4Rocev2Gid:
    # Kernel emits IPv4-mapped RoCEv2 GIDs for 198.51.100.3 as
    # 0000:0000:0000:0000:0000:ffff:c633:640b (10=0a, 100=64, 0=00, 11=0b).
    IPV4_GID = "0000:0000:0000:0000:0000:ffff:c633:640b"
    # RoCEv1 / MAC-based link-local. These have ``fe80:`` prefix.
    MAC_GID = "fe80:0000:0000:0000:1234:5678:9abc:def0"
    ZERO_GID = "0000:0000:0000:0000:0000:0000:0000:0000"

    def test_ipv4_mapped_returns_true(self, fake_sysfs):
        _make_hca(fake_sysfs, "mlx5_1", gids={3: self.IPV4_GID})
        assert hca_discovery.hca_has_ipv4_rocev2_gid("mlx5_1") is True

    def test_mac_based_returns_false(self, fake_sysfs):
        _make_hca(fake_sysfs, "mlx5_1", gids={3: self.MAC_GID})
        assert hca_discovery.hca_has_ipv4_rocev2_gid("mlx5_1") is False

    def test_zero_gid_returns_false(self, fake_sysfs):
        _make_hca(fake_sysfs, "mlx5_2", gids={3: self.ZERO_GID})
        assert hca_discovery.hca_has_ipv4_rocev2_gid("mlx5_2") is False

    def test_missing_gid_file_returns_false(self, fake_sysfs):
        _make_hca(fake_sysfs, "mlx5_1", gids={})  # no index 3 file
        assert hca_discovery.hca_has_ipv4_rocev2_gid("mlx5_1") is False

    def test_missing_hca_returns_false(self, fake_sysfs):
        assert hca_discovery.hca_has_ipv4_rocev2_gid("nonexistent") is False

    def test_case_insensitive_match(self, fake_sysfs):
        upper = self.IPV4_GID.upper()
        _make_hca(fake_sysfs, "mlx5_1", gids={3: upper})
        assert hca_discovery.hca_has_ipv4_rocev2_gid("mlx5_1") is True

    def test_honours_gid_index_argument(self, fake_sysfs):
        _make_hca(
            fake_sysfs,
            "mlx5_1",
            gids={1: self.MAC_GID, 3: self.IPV4_GID},
        )
        # Default index 3 → IPv4 → True
        assert hca_discovery.hca_has_ipv4_rocev2_gid("mlx5_1") is True
        # Index 1 → MAC-based → False
        assert hca_discovery.hca_has_ipv4_rocev2_gid("mlx5_1", gid_index=1) is False


# ---------------------------------------------------------------------------
# build_nccl_ib_hca_whitelist  -  the heterogeneous-cluster case
# ---------------------------------------------------------------------------


class TestBuildNcclIbHcaWhitelist:
    IPV4 = "0000:0000:0000:0000:0000:ffff:c633:640b"
    MAC = "fe80:0000:0000:0000:1234:5678:9abc:def0"

    def test_empty_when_no_hcas(self, fake_sysfs):
        assert hca_discovery.build_nccl_ib_hca_whitelist() == ""

    def test_local_only_mlx5(self, fake_sysfs):
        # Only IPv4-mapped GIDs qualify; MAC-only ports must be excluded.
        _make_hca(fake_sysfs, "mlx5_0", gids={3: self.MAC})
        _make_hca(fake_sysfs, "mlx5_1", gids={3: self.IPV4})
        _make_hca(fake_sysfs, "mlx5_2", gids={3: self.MAC})
        _make_hca(fake_sysfs, "mlx5_3", gids={3: self.IPV4})
        assert hca_discovery.build_nccl_ib_hca_whitelist() == "mlx5_1,mlx5_3"

    def test_local_only_rocep(self, fake_sysfs):
        # Both rdma-core naming variants must be accepted.
        _make_hca(fake_sysfs, "rocep8s0f0", gids={3: self.MAC})
        _make_hca(fake_sysfs, "rocep8s0f1", gids={3: self.IPV4})
        _make_hca(fake_sysfs, "roceP8p1s0f0", gids={3: self.MAC})
        _make_hca(fake_sysfs, "roceP8p1s0f1", gids={3: self.IPV4})
        result = hca_discovery.build_nccl_ib_hca_whitelist()
        assert set(result.split(",")) == {"rocep8s0f1", "roceP8p1s0f1"}

    def test_heterogeneous_cluster_merges_local_and_remote(self, fake_sysfs):
        """Merge heterogeneous driver names across synthetic peers."""
        # Include every IPv4-GID HCA, deduped and sorted.
        _make_hca(fake_sysfs, "mlx5_1", gids={3: self.IPV4})
        _make_hca(fake_sysfs, "mlx5_3", gids={3: self.IPV4})
        remote_lists = [
            ["rocep8s0f1", "roceP8p1s0f1"],  # node-3
            ["rocep8s0f1", "roceP8p1s0f1"],  # node-4 (same names, dedup)
        ]
        result = hca_discovery.build_nccl_ib_hca_whitelist(
            remote_hca_lists=remote_lists
        )
        # Result is ASCII-sorted (capital-P < lowercase-p), deduped across
        # all participating nodes. Content matters, not exact ordering, but we
        # commit to *some* deterministic order  -  sorted-ASCII is it.
        assert set(result.split(",")) == {
            "mlx5_1", "mlx5_3", "rocep8s0f1", "roceP8p1s0f1",
        }
        assert result == ",".join(sorted(result.split(",")))

    def test_remote_lists_filter_invalid_names(self, fake_sysfs):
        # Remote lists might have stale entries (old names, virtual devs).
        # The combine step must not pass them through.
        _make_hca(fake_sysfs, "mlx5_1", gids={3: self.IPV4})
        remote_lists = [["rocep8s0f1", "bogus_device", "lo", "ibp0"]]
        result = hca_discovery.build_nccl_ib_hca_whitelist(
            remote_hca_lists=remote_lists
        )
        assert set(result.split(",")) == {"mlx5_1", "rocep8s0f1"}

    def test_returns_empty_when_no_gid_3_present(self, fake_sysfs):
        # All HCAs exist but only have MAC-based GIDs. No IPv4 → empty.
        _make_hca(fake_sysfs, "mlx5_1", gids={3: self.MAC})
        _make_hca(fake_sysfs, "mlx5_3", gids={3: self.MAC})
        assert hca_discovery.build_nccl_ib_hca_whitelist() == ""


# ---------------------------------------------------------------------------
# detect_fabric_ip
# ---------------------------------------------------------------------------


class _FakeCompleted:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


IP_ADDR_OUTPUT = """\
4: rdma87: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 9000 qdisc mq state UP
    inet 198.51.100.3/24 brd 198.51.100.11 scope global rdma87
       valid_lft forever preferred_lft forever
"""


class TestDetectFabricIp:
    def test_happy_path(self):
        with mock.patch(
            "ainode.cluster.hca_discovery.subprocess.run",
            return_value=_FakeCompleted(stdout=IP_ADDR_OUTPUT),
        ) as run:
            result = hca_discovery.detect_fabric_ip("rdma87")
        assert result == "198.51.100.3"
        assert run.call_args.args[0] == [
            "ip", "-4", "addr", "show", "rdma87",
        ]

    def test_no_iface_argument_returns_none(self):
        assert hca_discovery.detect_fabric_ip("") is None

    def test_command_failure_returns_none(self):
        with mock.patch(
            "ainode.cluster.hca_discovery.subprocess.run",
            return_value=_FakeCompleted(returncode=1, stderr="does not exist"),
        ):
            assert hca_discovery.detect_fabric_ip("bogus0") is None

    def test_unassigned_iface_returns_none(self):
        # Output has no "inet X.X.X.X/Y" line.
        with mock.patch(
            "ainode.cluster.hca_discovery.subprocess.run",
            return_value=_FakeCompleted(stdout="5: eth0: <NO-CARRIER>\n"),
        ):
            assert hca_discovery.detect_fabric_ip("eth0") is None

    def test_ip_binary_missing_returns_none(self):
        with mock.patch(
            "ainode.cluster.hca_discovery.subprocess.run",
            side_effect=FileNotFoundError("ip"),
        ):
            assert hca_discovery.detect_fabric_ip("rdma87") is None

    def test_strips_cidr_suffix(self):
        # detect_fabric_ip must not return '198.51.100.3/24'.
        with mock.patch(
            "ainode.cluster.hca_discovery.subprocess.run",
            return_value=_FakeCompleted(stdout=IP_ADDR_OUTPUT),
        ):
            ip = hca_discovery.detect_fabric_ip("rdma87")
        assert ip is not None and "/" not in ip


# ---------------------------------------------------------------------------
# probe_path_mtu
# ---------------------------------------------------------------------------


class TestProbePathMtu:
    def test_unreachable_peer_returns_minus_one(self):
        # First liveness ping fails → -1.
        with mock.patch(
            "ainode.cluster.hca_discovery.subprocess.run",
            return_value=_FakeCompleted(returncode=1),
        ):
            assert hca_discovery.probe_path_mtu("198.51.100.14") == -1

    def test_full_jumbo_path_returns_9000(self):
        # Every ping succeeds  -  the early hi probe hits 8972 and we return
        # 8972 + 28 = 9000 without binary searching.
        with mock.patch(
            "ainode.cluster.hca_discovery.subprocess.run",
            return_value=_FakeCompleted(returncode=0),
        ) as run:
            mtu = hca_discovery.probe_path_mtu("198.51.100.5")
        assert mtu == 9000
        # Liveness ping + one jumbo ping = 2 calls max.
        assert run.call_count == 2

    def test_standard_1500_mtu_found_by_bisect(self):
        """Mgmt NIC path: ping succeeds up to payload 1472, fails above.

        Search should converge on best=1472 → MTU=1500.
        """

        def fake_run(cmd, **kwargs):
            # First ping is the liveness probe (size 56)  -  always succeeds.
            # Remaining calls have size as the 7th arg (index 6):
            #   ['ping', '-c', '1', '-M', 'do', '-s', '<size>', ...]
            size_idx = cmd.index("-s") + 1
            size = int(cmd[size_idx])
            # Liveness uses size 56; treat <= 1472 as success.
            return _FakeCompleted(returncode=0 if size <= 1472 else 1)

        with mock.patch(
            "ainode.cluster.hca_discovery.subprocess.run",
            side_effect=fake_run,
        ):
            mtu = hca_discovery.probe_path_mtu("192.0.2.26")
        assert mtu == 1500

    def test_empty_peer_ip_returns_minus_one(self):
        assert hca_discovery.probe_path_mtu("") == -1

    def test_ping_missing_returns_minus_one(self):
        with mock.patch(
            "ainode.cluster.hca_discovery.subprocess.run",
            side_effect=FileNotFoundError("ping"),
        ):
            assert hca_discovery.probe_path_mtu("198.51.100.5") == -1


# ---------------------------------------------------------------------------
# Subnet filter: select the configured fabric address
# ---------------------------------------------------------------------------

def _gid(ip: str) -> str:
    a, b, c, d = (int(x) for x in ip.split("."))
    return f"0000:0000:0000:0000:0000:ffff:{a:02x}{b:02x}:{c:02x}{d:02x}"


class TestHcaIpv4FromGid:
    def test_reads_the_address_out_of_the_gid(self, fake_sysfs):
        _make_hca(fake_sysfs, "roceP8p1s0f1", gids={3: _gid("198.51.100.5")})
        assert hca_discovery.hca_ipv4_from_gid("roceP8p1s0f1") == "198.51.100.5"

    def test_mac_gid_has_no_address(self, fake_sysfs):
        _make_hca(fake_sysfs, "mlx5_0", gids={3: "fe80:0000:0000:0000:1234:5678:9abc:def0"})
        assert hca_discovery.hca_ipv4_from_gid("mlx5_0") is None


class TestFabricIpFilter:
    """Synthetic layout: two fabric ports on 198.51.100.0/24, one live
    direct-connect port on 192.0.2.0/24, one link-local autoconf port. Only
    the port that backs the cluster interface (198.51.100.5) may reach NCCL."""

    def _synthetic_ports(self, root):
        _make_hca(root, "roceP8p1s0f0", gids={3: _gid("169.254.1.1")})
        _make_hca(root, "roceP8p1s0f1", gids={3: _gid("198.51.100.5")})
        _make_hca(root, "rocep8s0f0", gids={3: _gid("192.0.2.2")})
        _make_hca(root, "rocep8s0f1", gids={3: _gid("198.51.100.4")})

    def test_without_fabric_ip_everything_with_an_ipv4_gid_is_listed(self, fake_sysfs):
        self._synthetic_ports(fake_sysfs)
        assert set(hca_discovery.build_nccl_ib_hca_whitelist().split(",")) == {
            "roceP8p1s0f0", "roceP8p1s0f1", "rocep8s0f0", "rocep8s0f1"}

    def test_fabric_ip_keeps_only_the_port_behind_the_cluster_interface(self, fake_sysfs):
        self._synthetic_ports(fake_sysfs)
        assert hca_discovery.build_nccl_ib_hca_whitelist(fabric_ip="198.51.100.5") == "roceP8p1s0f1"

    def test_a_down_port_is_dropped_even_when_the_address_matches(self, fake_sysfs):
        _make_hca(fake_sysfs, "roceP8p1s0f1", port_state="1: DOWN", gids={3: _gid("198.51.100.6")})
        assert hca_discovery.build_nccl_ib_hca_whitelist(fabric_ip="198.51.100.6") == ""

    def test_remote_lists_are_not_filtered(self, fake_sysfs):
        _make_hca(fake_sysfs, "roceP8p1s0f1", gids={3: _gid("198.51.100.5")})
        result = hca_discovery.build_nccl_ib_hca_whitelist(
            remote_hca_lists=[["mlx5_1"]], fabric_ip="198.51.100.5")
        assert result == "mlx5_1,roceP8p1s0f1"

    def test_no_match_yields_empty_so_the_caller_leaves_the_var_unset(self, fake_sysfs):
        self._synthetic_ports(fake_sysfs)
        assert hca_discovery.build_nccl_ib_hca_whitelist(fabric_ip="192.0.2.36") == ""
