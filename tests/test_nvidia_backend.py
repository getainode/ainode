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
    HEAD_CONTAINER_NAME,
    NVIDIA_VLLM_IMAGE,
    RAY_CONTAINER_NAME_PREFIX,
    RUN_CLUSTER_SCRIPT_FALLBACK,
    RUN_CLUSTER_SCRIPT_SOURCE,
    WORKER_CONTAINER_NAME_PREFIX,
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

    def test_raises_when_head_container_does_not_become_ready(self):
        """If ``docker inspect`` never reports Running=true, we must raise.

        Option α replaced the old ``run_cluster.sh`` blocking subprocess
        with a ``docker run -d`` + poll-readiness flow. The bad path is
        now "head container never reports Running" — we surface it as an
        NvidiaBackendError instead of hanging.
        """
        config = _make_config(
            distributed_mode="head",
            peer_ips=["10.100.0.13"],
        )
        backend = NvidiaBackend(config)
        with mock.patch.object(
            backend, "_launch_head_container", return_value="fake_container_id"
        ), mock.patch.object(
            backend, "_wait_for_head_container_ready", return_value=False
        ), mock.patch(
            "ainode.engine.backends.nvidia.detect_fabric_ip",
            return_value="10.100.0.11",
        ), mock.patch(
            "ainode.engine.backends.nvidia.build_nccl_ib_hca_whitelist",
            return_value="",
        ):
            with pytest.raises(NvidiaBackendError, match="did not enter Running"):
                backend.start_distributed()

    def test_distributed_launches_head_ssh_workers_and_vllm_exec(self):
        """Orchestration test — ``start_distributed`` calls the three helpers
        in order (head launch → ssh workers → docker exec vllm) with the
        right arguments, and produces a TP=N ``vllm serve`` command where
        N = 1 + len(peers).

        We mock the helpers rather than the raw subprocess layer so the
        test isn't brittle to internals of ``docker run`` / ``docker
        inspect`` polling.
        """
        config = _make_config(
            distributed_mode="head",
            peer_ips=["10.100.0.13", "10.100.0.15"],
        )
        backend = NvidiaBackend(config)

        ssh_launch_calls: List[dict] = []

        def fake_ssh_launch(peer_ip, head_ip):
            ssh_launch_calls.append({"peer_ip": peer_ip, "head_ip": head_ip})

        with mock.patch.object(
            backend, "_launch_head_container", return_value="ctr_abc"
        ) as launch_head, mock.patch.object(
            backend, "_wait_for_head_container_ready", return_value=True
        ) as wait_ready, mock.patch.object(
            backend, "_ssh_launch_worker", side_effect=fake_ssh_launch
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

        # Head launched once, readiness polled once.
        launch_head.assert_called_once()
        wait_ready.assert_called_once()
        _, kwargs = launch_head.call_args
        assert kwargs["fabric_ip"] == "10.100.0.11"

        # SSH launches: exactly one per peer, each carrying the head IP.
        assert {c["peer_ip"] for c in ssh_launch_calls} == set(config.peer_ips)
        assert all(c["head_ip"] == "10.100.0.11" for c in ssh_launch_calls)

        # docker exec call for vllm serve — TP=3 (1 head + 2 peers).
        popen.assert_called_once()
        exec_argv = popen.call_args.args[0]
        assert exec_argv[0] == "docker"
        assert exec_argv[1] == "exec"
        joined = " ".join(exec_argv)
        assert "vllm" in joined and "serve" in joined
        assert "--tensor-parallel-size 3" in joined
        assert "--distributed-executor-backend ray" in joined

    def test_start_distributed_returns_within_reasonable_time(self):
        """Regression test for Bug 2: ``start_distributed`` must not block
        the main thread for 120+ seconds. With helpers mocked to succeed
        immediately, the call should return in well under a second."""
        import time

        config = _make_config(
            distributed_mode="head",
            peer_ips=["10.100.0.13"],
        )
        backend = NvidiaBackend(config)
        with mock.patch.object(
            backend, "_launch_head_container", return_value="ctr_abc"
        ), mock.patch.object(
            backend, "_wait_for_head_container_ready", return_value=True
        ), mock.patch.object(
            backend, "_ssh_launch_worker", return_value=None
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
            t0 = time.time()
            result = backend.start_distributed()
            elapsed = time.time() - t0

        assert result is True
        # Generous: real path will be dominated by docker run +
        # docker inspect poll. Mocked path should be nearly instant.
        assert elapsed < 5.0, (
            f"start_distributed took {elapsed:.2f}s with all helpers mocked; "
            "something is re-introducing a blocking call."
        )


# ---------------------------------------------------------------------------
# Option α helper methods — _launch_head_container, _wait_for_head_container_ready,
# _ssh_launch_worker
# ---------------------------------------------------------------------------


class TestLaunchHeadContainer:
    def test_issues_docker_run_d_with_ray_head_command(self):
        """`docker run -d` must go out with --name HEAD_CONTAINER_NAME and
        the ray head start command embedded as the container argv."""
        config = _make_config(
            distributed_mode="head",
            peer_ips=["10.100.0.13"],
        )
        backend = NvidiaBackend(config)

        run_calls: List[List[str]] = []

        def fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            # First call will be docker stop, second docker rm (cleanup of
            # any stale container) — return ok. Third is docker run -d.
            return _FakeCompleted(returncode=0, stdout="ctr_abc\n")

        with mock.patch(
            "ainode.engine.backends.nvidia.subprocess.run",
            side_effect=fake_run,
        ), mock.patch(
            "ainode.engine.backends.nvidia.detect_fabric_ip",
            return_value="10.100.0.11",
        ), mock.patch(
            "ainode.engine.backends.nvidia.build_nccl_ib_hca_whitelist",
            return_value="mlx5_1",
        ):
            container_id = backend._launch_head_container(
                fabric_ip="10.100.0.11",
                hf_cache_dir="/tmp/hf",
            )

        assert container_id == "ctr_abc"

        # Last call is the docker run -d; earlier are the pre-cleanup
        # stop+rm (best-effort).
        run_cmd = run_calls[-1]
        assert run_cmd[0] == "docker"
        assert run_cmd[1] == "run"
        assert "-d" in run_cmd
        assert "--name" in run_cmd
        assert run_cmd[run_cmd.index("--name") + 1] == HEAD_CONTAINER_NAME
        assert "--network" in run_cmd
        assert run_cmd[run_cmd.index("--network") + 1] == "host"
        assert "--gpus" in run_cmd
        assert "--entrypoint" in run_cmd
        assert NVIDIA_VLLM_IMAGE in run_cmd

        # Last two tokens carry the shell-wrapped ray start command.
        # Structure: [..., NVIDIA_VLLM_IMAGE, "-c", "ray start ..."]
        assert run_cmd[-2] == "-c"
        ray_cmd = run_cmd[-1]
        assert ray_cmd.startswith("ray start --block --head")
        assert "--port=6379" in ray_cmd

    def test_raises_on_docker_failure(self):
        """Non-zero returncode from docker must raise NvidiaBackendError
        with the stderr so operators can see what broke."""
        config = _make_config(distributed_mode="head", peer_ips=["10.0.0.2"])
        backend = NvidiaBackend(config)

        call_n = {"n": 0}

        def fake_run(cmd, **kwargs):
            # Let the pre-cleanup succeed so we isolate the failure on
            # the actual docker run -d call.
            call_n["n"] += 1
            if call_n["n"] <= 2:
                return _FakeCompleted(returncode=0)
            return _FakeCompleted(
                returncode=125,
                stderr="docker: Error response from daemon: Conflict.",
            )

        with mock.patch(
            "ainode.engine.backends.nvidia.subprocess.run",
            side_effect=fake_run,
        ), mock.patch(
            "ainode.engine.backends.nvidia.build_nccl_ib_hca_whitelist",
            return_value="",
        ):
            with pytest.raises(NvidiaBackendError, match="docker run -d"):
                backend._launch_head_container(
                    fabric_ip="10.0.0.1",
                    hf_cache_dir="/tmp/hf",
                )

    def test_returns_fast(self):
        """`_launch_head_container` must finish well under its safety timeout —
        it is just a `docker run -d`, which returns as soon as the
        container is created. Regression guard for Bug 2."""
        import time

        config = _make_config(distributed_mode="head", peer_ips=["10.0.0.2"])
        backend = NvidiaBackend(config)

        with mock.patch(
            "ainode.engine.backends.nvidia.subprocess.run",
            return_value=_FakeCompleted(returncode=0, stdout="ctr_abc\n"),
        ), mock.patch(
            "ainode.engine.backends.nvidia.build_nccl_ib_hca_whitelist",
            return_value="",
        ):
            t0 = time.time()
            backend._launch_head_container(
                fabric_ip="10.0.0.1",
                hf_cache_dir="/tmp/hf",
            )
            elapsed = time.time() - t0

        assert elapsed < 2.0


class TestWaitForHeadContainerReady:
    def test_returns_true_when_container_running(self):
        config = _make_config(distributed_mode="head", peer_ips=["10.0.0.2"])
        backend = NvidiaBackend(config)

        with mock.patch(
            "ainode.engine.backends.nvidia.subprocess.run",
            return_value=_FakeCompleted(returncode=0, stdout="true\n"),
        ):
            ok = backend._wait_for_head_container_ready(
                HEAD_CONTAINER_NAME, timeout=5
            )

        assert ok is True

    def test_returns_false_on_timeout(self):
        config = _make_config(distributed_mode="head", peer_ips=["10.0.0.2"])
        backend = NvidiaBackend(config)

        # Return "false" every time — so the poll loop times out.
        with mock.patch(
            "ainode.engine.backends.nvidia.subprocess.run",
            return_value=_FakeCompleted(returncode=0, stdout="false\n"),
        ), mock.patch(
            "ainode.engine.backends.nvidia.time.sleep",
            return_value=None,
        ):
            ok = backend._wait_for_head_container_ready(
                HEAD_CONTAINER_NAME, timeout=2
            )

        assert ok is False

    def test_tolerates_inspect_errors_and_retries(self):
        """`docker inspect` errors early (container not yet visible) must
        not abort the poll — only the final timeout should."""
        config = _make_config(distributed_mode="head", peer_ips=["10.0.0.2"])
        backend = NvidiaBackend(config)

        call_n = {"n": 0}

        def fake_run(cmd, **kwargs):
            call_n["n"] += 1
            if call_n["n"] <= 2:
                return _FakeCompleted(
                    returncode=1,
                    stderr="No such object: ainode-vllm-head",
                )
            return _FakeCompleted(returncode=0, stdout="true\n")

        with mock.patch(
            "ainode.engine.backends.nvidia.subprocess.run",
            side_effect=fake_run,
        ), mock.patch(
            "ainode.engine.backends.nvidia.time.sleep",
            return_value=None,
        ):
            ok = backend._wait_for_head_container_ready(
                HEAD_CONTAINER_NAME, timeout=10
            )

        assert ok is True
        assert call_n["n"] >= 3


class TestSshLaunchWorker:
    def test_ssh_issues_docker_run_d_on_peer(self):
        """The remote command sent over SSH must include a `docker run -d`
        with the worker's stable name and a `ray start --address=HEAD:6379`
        — not `bash run_cluster.sh --worker` (that was the Bug 2 path)."""
        config = _make_config(
            distributed_mode="head",
            peer_ips=["10.100.0.13"],
        )
        backend = NvidiaBackend(config)

        ssh_cmds: List[List[str]] = []

        def fake_run(cmd, **kwargs):
            ssh_cmds.append(cmd)
            return _FakeCompleted(returncode=0)

        with mock.patch(
            "ainode.engine.backends.nvidia.subprocess.run",
            side_effect=fake_run,
        ), mock.patch(
            "ainode.engine.backends.nvidia.build_nccl_ib_hca_whitelist",
            return_value="",
        ), mock.patch(
            "ainode.engine.backends.nvidia.detect_fabric_ip",
            return_value="10.100.0.11",
        ):
            backend._ssh_launch_worker(
                peer_ip="10.100.0.13",
                head_ip="10.100.0.11",
            )

        assert len(ssh_cmds) == 1
        ssh = ssh_cmds[0]
        assert ssh[0] == "ssh"
        assert f"{config.ssh_user}@10.100.0.13" in ssh

        remote = ssh[-1]
        # Must not shell out to run_cluster.sh anymore.
        assert "run_cluster.sh" not in remote
        # Must be a docker run -d with the deterministic worker name.
        assert "docker run" in remote
        assert "-d" in remote
        assert f"{WORKER_CONTAINER_NAME_PREFIX}-10-100-0-13" in remote
        # And a ray worker connect string pointing at the head.
        assert "ray start" in remote
        assert "--address=10.100.0.11:6379" in remote

    def test_raises_on_ssh_failure(self):
        config = _make_config(
            distributed_mode="head",
            peer_ips=["10.100.0.13"],
        )
        backend = NvidiaBackend(config)

        with mock.patch(
            "ainode.engine.backends.nvidia.subprocess.run",
            return_value=_FakeCompleted(
                returncode=255,
                stderr="Permission denied (publickey).",
            ),
        ), mock.patch(
            "ainode.engine.backends.nvidia.build_nccl_ib_hca_whitelist",
            return_value="",
        ):
            with pytest.raises(NvidiaBackendError, match="ssh docker run"):
                backend._ssh_launch_worker(
                    peer_ip="10.100.0.13",
                    head_ip="10.100.0.11",
                )


# ---------------------------------------------------------------------------
# launch_distributed shim
# ---------------------------------------------------------------------------


class _FakeShardingConfig:
    def __init__(self, model=None, peer_ips=None):
        self.model = model
        self.peer_ips = peer_ips


class TestLaunchDistributedShim:
    """Mirrors EugrBackend.launch_distributed. Called by /api/models/load."""

    def test_applies_sharding_config_and_flips_to_head(self):
        config = _make_config(distributed_mode="solo", peer_ips=[])
        backend = NvidiaBackend(config)

        sharding = _FakeShardingConfig(
            model="nvidia/Llama-3.3-70B-Instruct-NVFP4",
            peer_ips=["10.100.0.13"],
        )

        with mock.patch.object(config, "save"), mock.patch.object(
            backend, "_launch_head_container", return_value="ctr_abc"
        ), mock.patch.object(
            backend, "_wait_for_head_container_ready", return_value=True
        ), mock.patch.object(
            backend, "_ssh_launch_worker", return_value=None
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

        # Locally we stop+rm the head container (2 docker calls). We SSH
        # to each peer once to do both stop+rm inside a single remote
        # shell (2 ssh calls for 2 peers).
        ssh_calls = [c for c in run_calls if c[0] == "ssh"]
        docker_stop_calls = [
            c for c in run_calls if c[0] == "docker" and c[1] == "stop"
        ]
        docker_rm_calls = [
            c for c in run_calls if c[0] == "docker" and c[1] == "rm"
        ]
        assert len(ssh_calls) == 2
        assert len(docker_stop_calls) == 1
        assert len(docker_rm_calls) == 1

        # The local docker stop must target HEAD_CONTAINER_NAME — if we
        # ever drift from the Option α naming, stop() will leak
        # containers, so pin the invariant here.
        assert HEAD_CONTAINER_NAME in docker_stop_calls[0]
        # And each ssh must target one of our peers with a remote command
        # that references the deterministic per-peer worker container.
        for call in ssh_calls:
            remote = call[-1]
            assert WORKER_CONTAINER_NAME_PREFIX in remote
            assert "docker stop" in remote and "docker rm" in remote

    def test_stop_tolerates_unreachable_peer(self):
        """An SSH failure to one peer must not prevent teardown of the head
        container or of the other (reachable) peer."""
        config = _make_config(
            distributed_mode="head",
            peer_ips=["10.100.0.13", "10.100.0.15"],
        )
        backend = NvidiaBackend(config)

        calls: List[List[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            # First ssh call explodes; everything else succeeds.
            if cmd[0] == "ssh" and "10.100.0.13" in cmd:
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=30)
            return _FakeCompleted(returncode=0)

        with mock.patch(
            "ainode.engine.backends.nvidia.subprocess.run",
            side_effect=fake_run,
        ):
            # Must not raise.
            backend.stop()

        # Both SSH attempts were still made, and so were the local docker
        # stop + rm — ordering shouldn't short-circuit on peer failure.
        ssh_calls = [c for c in calls if c[0] == "ssh"]
        assert len(ssh_calls) == 2
        assert any(
            c[0] == "docker" and c[1] == "stop" and HEAD_CONTAINER_NAME in c
            for c in calls
        )


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
