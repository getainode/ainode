"""Unit tests for ainode.engine.backends.nvidia.NvidiaBackend.

All tests mock the Docker / SSH / hca_discovery surface so they pass on a
Mac without a live Spark. For the docker-run-args style of testing, we
inspect the ``Popen`` / ``subprocess.run`` argv directly rather than
actually invoking docker.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, List
from unittest import mock

import pytest

from ainode.core.config import NodeConfig
from ainode.engine.backends import (
    EngineBackend,
    NvidiaBackend,
    get_backend,
)
from ainode.engine.backends.nvidia import (
    NVIDIA_VLLM_IMAGE,
    RAY_CONTAINER_NAME_PREFIX,
    RUN_CLUSTER_SCRIPT_FALLBACK,
    RUN_CLUSTER_SCRIPT_SOURCE,
    NvidiaBackendError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> NodeConfig:
    """Build a NodeConfig preset for NvidiaBackend tests."""
    defaults = dict(
        engine_backend="nvidia",
        model="meta-llama/Llama-3.2-3B-Instruct",
        api_port=8000,
        cluster_interface="enP2p1s0f1np1",
        gpu_memory_utilization=0.85,
        hf_token="hf_testtoken",
        ssh_user="ubuntu",
        models_dir="/tmp/ainode-models",
    )
    defaults.update(overrides)
    return NodeConfig(**defaults)


class _FakePopen:
    """Stand-in for subprocess.Popen that records argv + returns controllable state."""

    def __init__(self, *args, **kwargs):
        self.args = args[0] if args else kwargs.get("args", [])
        self.kwargs = kwargs
        self.stdout = None
        self._returncode: Any = None  # None = still running
        self._signals: List[int] = []

    def poll(self):
        return self._returncode

    def send_signal(self, sig):
        self._signals.append(sig)
        self._returncode = 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self._returncode = -9


class _FakeCompleted:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


class TestFactoryDispatch:
    def test_factory_returns_nvidia_backend(self):
        config = _make_config()
        backend = get_backend(config)
        assert isinstance(backend, NvidiaBackend)
        assert isinstance(backend, EngineBackend)

    def test_nvidia_backend_is_no_longer_notimplemented(self):
        """Phase 4 implementation: start_solo() must do real work, not raise
        NotImplementedError. (If you see NotImplementedError, someone
        reverted the Phase 4 nvidia.py back to a stub.)"""
        config = _make_config(distributed_mode="solo")
        backend = get_backend(config)
        with mock.patch(
            "ainode.engine.backends.nvidia.subprocess.Popen",
            return_value=_FakePopen(),
        ), mock.patch(
            "ainode.engine.backends.nvidia.detect_fabric_ip",
            return_value="10.100.0.11",
        ), mock.patch(
            "ainode.engine.backends.nvidia.build_nccl_ib_hca_whitelist",
            return_value="mlx5_1,mlx5_3",
        ):
            result = backend.start()
        assert result is True


# ---------------------------------------------------------------------------
# _build_nccl_env
# ---------------------------------------------------------------------------


class TestBuildNcclEnv:
    EXPECTED_KEYS = {
        "VLLM_HOST_IP",
        "MASTER_ADDR",
        "MASTER_PORT",
        "UCX_NET_DEVICES",
        "NCCL_SOCKET_IFNAME",
        "OMPI_MCA_btl_tcp_if_include",
        "GLOO_SOCKET_IFNAME",
        "TP_SOCKET_IFNAME",
        "RAY_memory_monitor_refresh_ms",
        "NCCL_IB_GID_INDEX",
        "NCCL_IB_SUBNET_AWARE_ROUTING",
        "NCCL_IB_DISABLE",
        "HF_HUB_ENABLE_HF_TRANSFER",
        "HF_TOKEN",
    }

    def test_env_contains_all_required_keys(self):
        config = _make_config()
        backend = NvidiaBackend(config)
        with mock.patch(
            "ainode.engine.backends.nvidia.detect_fabric_ip",
            return_value="10.100.0.11",
        ), mock.patch(
            "ainode.engine.backends.nvidia.build_nccl_ib_hca_whitelist",
            return_value="mlx5_1,mlx5_3",
        ):
            env = backend._build_nccl_env(is_head=True)

        assert self.EXPECTED_KEYS.issubset(env.keys())
        # Phase-1-derived invariants
        assert env["NCCL_IB_GID_INDEX"] == "3"
        assert env["NCCL_IB_SUBNET_AWARE_ROUTING"] == "1"
        assert env["NCCL_IB_DISABLE"] == "0"
        assert env["HF_HUB_ENABLE_HF_TRANSFER"] == "1"
        # MASTER_PORT is a constant at the module level; ensure it matches
        assert env["MASTER_PORT"] == "29501"

    def test_head_env_uses_local_fabric_ip(self):
        config = _make_config()
        backend = NvidiaBackend(config)
        with mock.patch(
            "ainode.engine.backends.nvidia.detect_fabric_ip",
            return_value="10.100.0.11",
        ), mock.patch(
            "ainode.engine.backends.nvidia.build_nccl_ib_hca_whitelist",
            return_value="",
        ):
            env = backend._build_nccl_env(is_head=True)
        assert env["VLLM_HOST_IP"] == "10.100.0.11"
        assert env["MASTER_ADDR"] == "10.100.0.11"

    def test_worker_env_uses_head_fabric_ip_for_master(self):
        config = _make_config()
        backend = NvidiaBackend(config)
        with mock.patch(
            "ainode.engine.backends.nvidia.detect_fabric_ip",
            return_value="10.100.0.13",  # this node's IP
        ), mock.patch(
            "ainode.engine.backends.nvidia.build_nccl_ib_hca_whitelist",
            return_value="",
        ):
            env = backend._build_nccl_env(
                is_head=False, head_fabric_ip="10.100.0.11"
            )
        # VLLM_HOST_IP = this node. MASTER_ADDR = head.
        assert env["VLLM_HOST_IP"] == "10.100.0.13"
        assert env["MASTER_ADDR"] == "10.100.0.11"

    def test_env_propagates_hca_whitelist(self):
        config = _make_config()
        backend = NvidiaBackend(config)
        whitelist = "mlx5_1,mlx5_3,rocep1s0f1,roceP2p1s0f1"
        with mock.patch(
            "ainode.engine.backends.nvidia.detect_fabric_ip",
            return_value="10.100.0.11",
        ), mock.patch(
            "ainode.engine.backends.nvidia.build_nccl_ib_hca_whitelist",
            return_value=whitelist,
        ):
            env = backend._build_nccl_env(is_head=True)
        assert env["NCCL_IB_HCA"] == whitelist

    def test_env_omits_nccl_ib_hca_when_no_hcas(self):
        """Empty whitelist means leave NCCL_IB_HCA unset — let NCCL autodetect."""
        config = _make_config()
        backend = NvidiaBackend(config)
        with mock.patch(
            "ainode.engine.backends.nvidia.detect_fabric_ip",
            return_value="10.100.0.11",
        ), mock.patch(
            "ainode.engine.backends.nvidia.build_nccl_ib_hca_whitelist",
            return_value="",
        ):
            env = backend._build_nccl_env(is_head=True)
        assert "NCCL_IB_HCA" not in env

    def test_env_socket_interfaces_match_cluster_interface(self):
        config = _make_config(cluster_interface="enp1s0f1np1")
        backend = NvidiaBackend(config)
        with mock.patch(
            "ainode.engine.backends.nvidia.detect_fabric_ip",
            return_value="10.100.0.11",
        ), mock.patch(
            "ainode.engine.backends.nvidia.build_nccl_ib_hca_whitelist",
            return_value="",
        ):
            env = backend._build_nccl_env(is_head=True)
        for key in (
            "UCX_NET_DEVICES",
            "NCCL_SOCKET_IFNAME",
            "OMPI_MCA_btl_tcp_if_include",
            "GLOO_SOCKET_IFNAME",
            "TP_SOCKET_IFNAME",
        ):
            assert env[key] == "enp1s0f1np1", f"{key} mismatch"

    def test_env_forwards_hf_token(self):
        config = _make_config(hf_token="hf_fakesecret")
        backend = NvidiaBackend(config)
        with mock.patch(
            "ainode.engine.backends.nvidia.detect_fabric_ip",
            return_value="10.100.0.11",
        ), mock.patch(
            "ainode.engine.backends.nvidia.build_nccl_ib_hca_whitelist",
            return_value="",
        ):
            env = backend._build_nccl_env(is_head=True)
        assert env["HF_TOKEN"] == "hf_fakesecret"


# ---------------------------------------------------------------------------
# start_solo — docker run invocation
# ---------------------------------------------------------------------------


class TestStartSolo:
    def test_start_solo_invokes_docker_run_with_expected_args(self):
        config = _make_config(distributed_mode="solo")
        backend = NvidiaBackend(config)
        with mock.patch(
            "ainode.engine.backends.nvidia.subprocess.Popen",
            return_value=_FakePopen(),
        ) as popen, mock.patch(
            "ainode.engine.backends.nvidia.detect_fabric_ip",
            return_value="10.100.0.11",
        ), mock.patch(
            "ainode.engine.backends.nvidia.build_nccl_ib_hca_whitelist",
            return_value="mlx5_1,mlx5_3",
        ):
            result = backend.start_solo()

        assert result is True
        popen.assert_called_once()
        argv: List[str] = popen.call_args.args[0]

        # Structural checks — order-independent where possible
        assert argv[0] == "docker"
        assert argv[1] == "run"
        assert "--gpus" in argv and argv[argv.index("--gpus") + 1] == "all"
        assert "--network" in argv and argv[argv.index("--network") + 1] == "host"
        assert "--shm-size" in argv
        assert NVIDIA_VLLM_IMAGE in argv
        # ``vllm serve <model>`` appears after the image
        image_idx = argv.index(NVIDIA_VLLM_IMAGE)
        assert argv[image_idx + 1 : image_idx + 4] == [
            "vllm", "serve", config.model,
        ]
        # API port present in the tail args
        assert "--port" in argv
        assert argv[argv.index("--port") + 1] == str(config.api_port)
        # Env vars pass through ``-e KEY=VALUE`` pairs
        env_pairs = [
            argv[i + 1]
            for i in range(len(argv))
            if argv[i] == "-e" and i + 1 < len(argv)
        ]
        assert any(p.startswith("VLLM_HOST_IP=") for p in env_pairs)
        assert any(p.startswith("NCCL_IB_GID_INDEX=3") for p in env_pairs)
        assert any(p.startswith("HF_HUB_ENABLE_HF_TRANSFER=1") for p in env_pairs)

    def test_start_solo_skips_when_already_running(self):
        config = _make_config(distributed_mode="solo")
        backend = NvidiaBackend(config)
        backend._process = _FakePopen()  # still running
        with mock.patch(
            "ainode.engine.backends.nvidia.subprocess.Popen"
        ) as popen:
            assert backend.start_solo() is True
            popen.assert_not_called()

    def test_solo_does_not_request_tp(self):
        """Solo mode must NOT pass --tensor-parallel-size (TP=1 is implicit)."""
        config = _make_config(distributed_mode="solo")
        backend = NvidiaBackend(config)
        with mock.patch(
            "ainode.engine.backends.nvidia.subprocess.Popen",
            return_value=_FakePopen(),
        ) as popen, mock.patch(
            "ainode.engine.backends.nvidia.detect_fabric_ip",
            return_value="10.100.0.11",
        ), mock.patch(
            "ainode.engine.backends.nvidia.build_nccl_ib_hca_whitelist",
            return_value="",
        ):
            backend.start_solo()

        argv = popen.call_args.args[0]
        assert "--tensor-parallel-size" not in argv


# ---------------------------------------------------------------------------
# start_distributed — SSH fan-out + docker exec
# ---------------------------------------------------------------------------


class TestStartDistributed:
    def test_raises_when_mode_is_not_head(self):
        config = _make_config(distributed_mode="solo", peer_ips=["10.100.0.13"])
        backend = NvidiaBackend(config)
        with pytest.raises(NvidiaBackendError, match="distributed_mode='head'"):
            backend.start_distributed()

    def test_raises_when_no_peers(self):
        config = _make_config(distributed_mode="head", peer_ips=[])
        backend = NvidiaBackend(config)
        with pytest.raises(NvidiaBackendError, match="peer_ips is empty"):
            backend.start_distributed()

    def test_raises_when_run_cluster_script_missing(self, monkeypatch):
        config = _make_config(distributed_mode="head", peer_ips=["10.100.0.13"])
        backend = NvidiaBackend(config)
        # Point both candidate paths at something guaranteed missing.
        monkeypatch.setattr(
            "ainode.engine.backends.nvidia.RUN_CLUSTER_SCRIPT_SOURCE",
            Path("/no/such/opt/ainode/run_cluster.sh"),
        )
        monkeypatch.setattr(
            "ainode.engine.backends.nvidia.RUN_CLUSTER_SCRIPT_FALLBACK",
            Path("/no/such/tmp/run_cluster.sh"),
        )
        with pytest.raises(NvidiaBackendError, match="run_cluster.sh missing"):
            backend.start_distributed()

    def test_distributed_launches_head_ssh_workers_and_vllm_exec(
        self, tmp_path, monkeypatch
    ):
        # Vendor a stub run_cluster.sh so _locate_run_cluster_script finds it.
        script = tmp_path / "run_cluster.sh"
        script.write_text("#!/bin/bash\nexit 0\n")
        monkeypatch.setattr(
            "ainode.engine.backends.nvidia.RUN_CLUSTER_SCRIPT_SOURCE", script
        )

        config = _make_config(
            distributed_mode="head",
            peer_ips=["10.100.0.13", "10.100.0.15"],
        )
        backend = NvidiaBackend(config)

        run_calls: List[List[str]] = []

        def fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            return _FakeCompleted(returncode=0)

        with mock.patch(
            "ainode.engine.backends.nvidia.subprocess.run",
            side_effect=fake_run,
        ), mock.patch(
            "ainode.engine.backends.nvidia.subprocess.Popen",
            return_value=_FakePopen(),
        ) as popen, mock.patch(
            "ainode.engine.backends.nvidia.detect_fabric_ip",
            return_value="10.100.0.11",
        ), mock.patch(
            "ainode.engine.backends.nvidia.build_nccl_ib_hca_whitelist",
            return_value="mlx5_1,mlx5_3,rocep1s0f1,roceP2p1s0f1",
        ):
            result = backend.start_distributed()

        assert result is True

        # 1 head run_cluster.sh invocation + 2 peer SSH invocations = 3 run calls.
        assert len(run_calls) == 3

        # Head call: first command is bash <script>, --head, image, head_ip present.
        head_call = run_calls[0]
        assert head_call[0] == "bash"
        assert head_call[1] == str(script)
        assert NVIDIA_VLLM_IMAGE in head_call
        assert "10.100.0.11" in head_call
        assert "--head" in head_call

        # Peer SSH calls: ssh user@peer ... --worker ...
        peer_calls = run_calls[1:]
        peer_ips_seen = set()
        for call in peer_calls:
            assert call[0] == "ssh"
            # The last element is the remote command string
            remote = call[-1]
            assert "--worker" in remote
            assert NVIDIA_VLLM_IMAGE in remote
            # SSH target is user@ip
            target = next(a for a in call if a.startswith(f"{config.ssh_user}@"))
            peer_ips_seen.add(target.split("@")[1])
        assert peer_ips_seen == set(config.peer_ips)

        # docker exec call for vllm serve — TP=3 (1 head + 2 peers)
        popen.assert_called_once()
        exec_argv = popen.call_args.args[0]
        assert exec_argv[0] == "docker"
        assert exec_argv[1] == "exec"
        # Wrapped in bash -lc, so scan the whole command string
        joined = " ".join(exec_argv)
        assert "vllm" in joined and "serve" in joined
        assert "--tensor-parallel-size 3" in joined
        assert "--distributed-executor-backend ray" in joined


# ---------------------------------------------------------------------------
# launch_distributed shim
# ---------------------------------------------------------------------------


class _FakeShardingConfig:
    def __init__(self, model=None, peer_ips=None):
        self.model = model
        self.peer_ips = peer_ips


class TestLaunchDistributedShim:
    """Mirrors EugrBackend.launch_distributed. Called by /api/models/load."""

    def test_applies_sharding_config_and_flips_to_head(self, tmp_path, monkeypatch):
        script = tmp_path / "run_cluster.sh"
        script.write_text("#!/bin/bash\nexit 0\n")
        monkeypatch.setattr(
            "ainode.engine.backends.nvidia.RUN_CLUSTER_SCRIPT_SOURCE", script
        )

        config = _make_config(distributed_mode="solo", peer_ips=[])
        backend = NvidiaBackend(config)

        sharding = _FakeShardingConfig(
            model="nvidia/Llama-3.3-70B-Instruct-NVFP4",
            peer_ips=["10.100.0.13"],
        )

        with mock.patch.object(config, "save"), mock.patch(
            "ainode.engine.backends.nvidia.subprocess.run",
            return_value=_FakeCompleted(returncode=0),
        ), mock.patch(
            "ainode.engine.backends.nvidia.subprocess.Popen",
            return_value=_FakePopen(),
        ), mock.patch(
            "ainode.engine.backends.nvidia.detect_fabric_ip",
            return_value="10.100.0.11",
        ), mock.patch(
            "ainode.engine.backends.nvidia.build_nccl_ib_hca_whitelist",
            return_value="",
        ):
            result = backend.launch_distributed(sharding)

        assert result is True
        assert config.distributed_mode == "head"
        assert config.model == "nvidia/Llama-3.3-70B-Instruct-NVFP4"
        assert config.peer_ips == ["10.100.0.13"]


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


class TestStop:
    def test_stop_sends_sigterm_to_process(self):
        config = _make_config(distributed_mode="solo")
        backend = NvidiaBackend(config)
        fake = _FakePopen()
        backend._process = fake
        with mock.patch(
            "ainode.engine.backends.nvidia.subprocess.run",
            return_value=_FakeCompleted(returncode=0),
        ):
            backend.stop()
        # process cleared + sigterm recorded
        assert backend._process is None
        import signal as _sig
        assert _sig.SIGTERM in fake._signals

    def test_stop_fans_out_to_peers_in_head_mode(self):
        config = _make_config(
            distributed_mode="head",
            peer_ips=["10.100.0.13", "10.100.0.15"],
        )
        backend = NvidiaBackend(config)

        run_calls: List[List[str]] = []

        def fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            return _FakeCompleted(returncode=0)

        with mock.patch(
            "ainode.engine.backends.nvidia.subprocess.run",
            side_effect=fake_run,
        ):
            backend.stop()

        # One local docker stop + 2 ssh docker-stop calls = 3 total.
        ssh_calls = [c for c in run_calls if c[0] == "ssh"]
        docker_stop_calls = [
            c for c in run_calls if c[0] == "docker" and c[1] == "stop"
        ]
        assert len(ssh_calls) == 2
        assert len(docker_stop_calls) == 1


# ---------------------------------------------------------------------------
# is_running / health_check / properties
# ---------------------------------------------------------------------------


class TestIsRunningAndHealth:
    def test_is_running_false_when_process_none(self):
        backend = NvidiaBackend(_make_config())
        assert backend.is_running() is False

    def test_is_running_true_when_process_alive(self):
        backend = NvidiaBackend(_make_config())
        backend._process = _FakePopen()  # poll() returns None → alive
        assert backend.is_running() is True

    def test_is_running_false_when_process_exited(self):
        backend = NvidiaBackend(_make_config())
        fake = _FakePopen()
        fake._returncode = 0  # exited cleanly
        backend._process = fake
        assert backend.is_running() is False

    def test_health_check_shape(self):
        backend = NvidiaBackend(_make_config())
        # API not up → api_responding False, models_loaded empty
        with mock.patch(
            "ainode.engine.backends.nvidia.urllib.request.urlopen",
            side_effect=OSError("connection refused"),
        ):
            result = backend.health_check()
        assert set(result.keys()) == {
            "process_alive", "api_responding", "models_loaded"
        }
        assert result["process_alive"] is False
        assert result["api_responding"] is False
        assert result["models_loaded"] == []

    def test_api_url_matches_config_port(self):
        config = _make_config(api_port=8765)
        backend = NvidiaBackend(config)
        assert backend.api_url == "http://localhost:8765/v1"

    def test_log_path_switches_by_mode(self):
        solo_backend = NvidiaBackend(_make_config(distributed_mode="solo"))
        head_backend = NvidiaBackend(_make_config(distributed_mode="head"))
        assert solo_backend.log_path.name == "nvidia-vllm.log"
        assert head_backend.log_path.name == "nvidia-distributed.log"

    def test_process_property_getter_and_setter(self):
        backend = NvidiaBackend(_make_config())
        assert backend.process is None
        fake = _FakePopen()
        backend.process = fake
        assert backend.process is fake
