"""The mp multi-node distributed shape (#84): one ``vllm serve`` container per
node, no Ray anywhere.

Why it exists: the only vLLM build that serves DeepSeek V4 Flash correctly on
GB10 (sm121) ships no ``ray`` CLI, and neither does stock
``vllm/vllm-openai`` — the Ray shape's head container exits 127 on it. vLLM's
own multi-node executor (``--nnodes/--node-rank/--master-addr/--master-port``,
``--headless`` on rank >= 1) needs nothing but vLLM.

Fakes only: no docker, no ssh, no sleeps.
"""

from __future__ import annotations

import asyncio
from typing import Any, List
from unittest import mock
from unittest.mock import patch

import pytest

from ainode.core.config import HF_CACHE_MOUNT, NodeConfig
from ainode.engine.backends.nvidia import (
    HEAD_CONTAINER_NAME,
    INFINIBAND_DEVICE,
    WORKER_CONTAINER_NAME_PREFIX,
    NvidiaBackend,
    NvidiaBackendError,
)

DSPARK_ID = "deepseek-v4-flash-dspark"
DSPARK_REPO = "fraserprice/DeepSeek-V4-Flash-DSpark"
DSPARK_IMAGE = "vllm-dspark-runtime:dspark-nvfp4-stage-c"


class _FakePopen:
    def __init__(self, *args, **kwargs):
        self.args = args[0] if args else kwargs.get("args", [])
        self.stdout = None
        self._returncode: Any = None

    def poll(self):
        return self._returncode

    def send_signal(self, sig):
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


def _mp_config(**overrides) -> NodeConfig:
    defaults = dict(
        engine_backend="nvidia",
        model=DSPARK_REPO,
        api_port=8000,
        cluster_interface="enP2p1s0f1np1",
        distributed_mode="head",
        distributed_executor="mp",
        peer_ips=["10.100.0.13"],
        engine_image=DSPARK_IMAGE,
        gpu_memory_utilization=0.80,
        kv_cache_dtype="nvfp4_ds_mla",
        trust_remote_code=True,
        ssh_user="ubuntu",
        models_dir="/tmp/ainode-models",
    )
    defaults.update(overrides)
    return NodeConfig(**defaults)


def _backend(config: NodeConfig, *, infiniband: bool = True) -> NvidiaBackend:
    """A backend with every host probe faked: no docker, no sysfs, no NIC."""
    b = NvidiaBackend(config)
    # Empty ENTRYPOINT is what the custom image actually has, so the launch must
    # spell out `vllm serve` itself (it resolves through the recipe's PATH).
    b._image_entrypoint = lambda image: []  # type: ignore[assignment]
    b._infiniband_present = lambda: infiniband  # type: ignore[assignment]
    return b


def _nccl_free():
    """Patch the fabric/HCA probes the env builder reads off the host."""
    return (
        mock.patch("ainode.engine.backends.nvidia.detect_fabric_ip",
                   return_value="10.100.0.11"),
        mock.patch("ainode.engine.backends.nvidia.build_nccl_ib_hca_whitelist",
                   return_value=""),
    )


def _head_cmd(config: NodeConfig, **kw) -> List[str]:
    b = _backend(config, **kw)
    fabric, hca = _nccl_free()
    with fabric, hca:
        return b._build_mp_docker_cmd(
            container_name=b._head_container_name(), node_rank=0, nnodes=2,
            master_addr="10.100.0.11", hf_cache_dir="/root/.ainode/models/hf-cache",
            node_ip="10.100.0.11",
        )


def _peer_cmd(config: NodeConfig, **kw) -> List[str]:
    b = _backend(config, **kw)
    fabric, hca = _nccl_free()
    with fabric, hca:
        return b._build_mp_docker_cmd(
            container_name=b._worker_container_name("10.100.0.13"), node_rank=1,
            nnodes=2, master_addr="10.100.0.11",
            hf_cache_dir="/home/ubuntu/ainode-nvidia-cache", node_ip="10.100.0.13",
        )


def _after(argv: List[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


# ---------------------------------------------------------------------------
# Head command shape
# ---------------------------------------------------------------------------


def test_mp_head_command_shape():
    argv = _head_cmd(_mp_config())

    assert argv[:3] == ["docker", "run", "-d"]
    assert _after(argv, "--name") == HEAD_CONTAINER_NAME
    # Container shape the proven recipe ran.
    assert _after(argv, "--gpus") == "all"
    assert _after(argv, "--network") == "host"
    assert _after(argv, "--ipc") == "host"
    assert _after(argv, "--shm-size") == "64g"
    ulimits = [argv[i + 1] for i, a in enumerate(argv) if a == "--ulimit"]
    assert ulimits == ["memlock=-1", "stack=67108864"]
    assert f"{HF_CACHE_MOUNT}" in " ".join(argv)

    # Order: image, entrypoint prefix, model, serve args, rendezvous args.
    i = argv.index(DSPARK_IMAGE)
    assert argv[i + 1 : i + 4] == ["vllm", "serve", DSPARK_REPO]
    tail = argv[i + 4 :]
    assert tail.index("--tensor-parallel-size") < tail.index("--nnodes")
    assert _after(tail, "--tensor-parallel-size") == "2"
    assert _after(tail, "--distributed-executor-backend") == "mp"
    assert _after(tail, "--nnodes") == "2"
    assert _after(tail, "--node-rank") == "0"
    assert _after(tail, "--master-addr") == "10.100.0.11"
    assert _after(tail, "--master-port") == "29501"
    # Rank 0 serves the API; only a peer is headless.
    assert "--headless" not in tail
    assert _after(tail, "--host") == "0.0.0.0"
    assert _after(tail, "--port") == "8000"
    # No ray anywhere — that is the whole point of this shape.
    assert "ray" not in " ".join(tail)


def test_mp_head_serves_no_docker_exec_and_no_ray_container():
    """The head container IS the engine: no `ray start`, no `--entrypoint bash`."""
    argv = _head_cmd(_mp_config())
    assert "--entrypoint" not in argv
    assert "exec" not in argv


def test_mp_peer_command_is_the_same_command_with_rank_and_headless():
    head = _head_cmd(_mp_config())
    peer = _peer_cmd(_mp_config())

    i = peer.index(DSPARK_IMAGE)
    tail = peer[i + 4 :]
    assert _after(tail, "--node-rank") == "1"
    assert tail[-1] == "--headless"
    assert _after(tail, "--nnodes") == "2"
    assert _after(tail, "--master-addr") == "10.100.0.11"
    assert _after(peer, "--name") == f"{WORKER_CONTAINER_NAME_PREFIX}-10-100-0-13"

    # Everything else is identical to rank 0 (the proven shape runs one command).
    def strip(argv):
        out, skip = [], {"--node-rank", "--name", "-v", "-e"}
        i = 0
        while i < len(argv):
            if argv[i] in skip:
                i += 2
                continue
            if argv[i] == "--headless":
                i += 1
                continue
            out.append(argv[i])
            i += 1
        return out

    assert strip(head) == strip(peer)


def test_mp_peer_env_uses_its_own_host_ip_and_the_head_as_master():
    peer = _peer_cmd(_mp_config())
    env = {p.split("=", 1)[0]: p.split("=", 1)[1]
           for i, p in enumerate(peer) if peer[i - 1] == "-e"}
    assert env["VLLM_HOST_IP"] == "10.100.0.13"
    assert env["MASTER_ADDR"] == "10.100.0.11"
    assert env["MASTER_PORT"] == "29501"


# ---------------------------------------------------------------------------
# /dev/infiniband is conditional; extra_volumes render
# ---------------------------------------------------------------------------


def test_infiniband_device_mapped_when_the_host_has_it():
    argv = _head_cmd(_mp_config(), infiniband=True)
    devices = [argv[i + 1] for i, a in enumerate(argv) if a == "--device"]
    assert devices == [f"{INFINIBAND_DEVICE}:{INFINIBAND_DEVICE}"]


def test_infiniband_absent_is_not_a_failure():
    argv = _head_cmd(_mp_config(), infiniband=False)
    assert "--device" not in argv


def test_infiniband_probe_reads_the_host_path_seam(tmp_path):
    b = NvidiaBackend(_mp_config())
    with mock.patch.object(NvidiaBackend, "_host_path",
                           lambda self, p: str(tmp_path / "missing")):
        assert b._infiniband_present() is False
    (tmp_path / "infiniband").mkdir()
    with mock.patch.object(NvidiaBackend, "_host_path",
                           lambda self, p: str(tmp_path / "infiniband")):
        assert b._infiniband_present() is True


def test_extra_volumes_render_in_the_mp_command():
    argv = _head_cmd(_mp_config(extra_volumes=["/data/vllm-cache:/vllm-cache",
                                               "/srv/models:/models:ro"]))
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert "/data/vllm-cache:/vllm-cache" in mounts
    assert "/srv/models:/models:ro" in mounts


def test_extra_volumes_render_in_the_solo_command():
    b = _backend(_mp_config(distributed_mode="solo", peer_ips=[],
                            extra_volumes=["/data/jit:/vllm-cache"]))
    fabric, hca = _nccl_free()
    with fabric, hca:
        argv = b._build_solo_docker_cmd("c")
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert "/data/jit:/vllm-cache" in mounts


def test_malformed_extra_volume_is_skipped_not_fatal():
    b = _backend(_mp_config(extra_volumes=["nonsense", "/a:/b"]))
    assert b._volume_args() == ["-v", "/a:/b"]


# ---------------------------------------------------------------------------
# Launch orchestration: peers first, then the head
# ---------------------------------------------------------------------------


def test_mp_launch_starts_peers_before_the_head():
    config = _mp_config(peer_ips=["10.100.0.13", "10.100.0.15"])
    b = _backend(config)
    order: List[str] = []

    def fake_peer(peer_ip, head_ip, node_rank, nnodes):
        order.append(f"peer:{peer_ip}:rank{node_rank}:of{nnodes}")

    def fake_run(cmd, **kwargs):
        order.append("head" if cmd[:3] == ["docker", "run", "-d"] else cmd[1])
        return _FakeCompleted(returncode=0, stdout="ctr_head")

    fabric, hca = _nccl_free()
    with fabric, hca, mock.patch.object(
        b, "_ssh_launch_mp_worker", side_effect=fake_peer
    ), mock.patch.object(
        b, "ensure_image", return_value=True
    ), mock.patch.object(
        b, "_docker_stop_and_rm_best_effort"
    ), mock.patch.object(
        b, "_docker_container_state", return_value="running"
    ), mock.patch(
        "ainode.engine.backends.nvidia.subprocess.run", side_effect=fake_run
    ), mock.patch(
        "ainode.engine.backends.nvidia.subprocess.Popen", return_value=_FakePopen()
    ):
        assert b.start_distributed() is True

    assert order == ["peer:10.100.0.13:rank1:of3", "peer:10.100.0.15:rank2:of3", "head"]


def test_mp_launch_never_waits_for_head_running_before_the_peers():
    """The Ray shape must poll the head before SSHing workers at :6379. The mp
    rendezvous does that waiting itself, so this path must not."""
    b = _backend(_mp_config())
    fabric, hca = _nccl_free()
    with fabric, hca, mock.patch.object(
        b, "_ssh_launch_mp_worker"
    ), mock.patch.object(
        b, "ensure_image", return_value=True
    ), mock.patch.object(
        b, "_docker_stop_and_rm_best_effort"
    ), mock.patch.object(
        b, "_docker_container_state", return_value="running"
    ), mock.patch.object(
        b, "_wait_for_head_container_ready"
    ) as wait_head, mock.patch(
        "ainode.engine.backends.nvidia.subprocess.run",
        return_value=_FakeCompleted(returncode=0, stdout="ctr"),
    ), mock.patch(
        "ainode.engine.backends.nvidia.subprocess.Popen", return_value=_FakePopen()
    ):
        assert b.start_distributed() is True
    wait_head.assert_not_called()


def test_mp_launch_follows_the_head_container_logs():
    """`last_log_activity` (the adaptive bind wait's liveness signal) has to come
    from the engine's own stdout, and in this shape that is `docker logs -f`."""
    b = _backend(_mp_config())
    fabric, hca = _nccl_free()
    with fabric, hca, mock.patch.object(
        b, "_ssh_launch_mp_worker"
    ), mock.patch.object(
        b, "ensure_image", return_value=True
    ), mock.patch.object(
        b, "_docker_stop_and_rm_best_effort"
    ), mock.patch.object(
        b, "_docker_container_state", return_value="running"
    ), mock.patch(
        "ainode.engine.backends.nvidia.subprocess.run",
        return_value=_FakeCompleted(returncode=0, stdout="ctr"),
    ), mock.patch(
        "ainode.engine.backends.nvidia.subprocess.Popen", return_value=_FakePopen()
    ) as popen:
        b.start_distributed()

    assert popen.call_args.args[0] == ["docker", "logs", "-f", HEAD_CONTAINER_NAME]
    assert b.log_path.name == "nvidia-distributed.log"


def test_mp_head_that_dies_takes_the_peers_down_with_it():
    """A rank-0 container that exits on its flags leaves the peers waiting on a
    rendezvous that will never happen."""
    b = _backend(_mp_config(peer_ips=["10.100.0.13", "10.100.0.15"]))
    stopped: List[str] = []
    fabric, hca = _nccl_free()
    with fabric, hca, mock.patch.object(
        b, "_ssh_launch_mp_worker"
    ), mock.patch.object(
        b, "ensure_image", return_value=True
    ), mock.patch.object(
        b, "_docker_stop_and_rm_best_effort"
    ), mock.patch.object(
        b, "_docker_container_state", return_value="exited"
    ), mock.patch.object(
        b, "_docker_logs_tail", return_value="vllm: command not found"
    ), mock.patch.object(
        b, "_ssh_stop_peer_container", side_effect=stopped.append
    ), mock.patch(
        "ainode.engine.backends.nvidia.subprocess.run",
        return_value=_FakeCompleted(returncode=0, stdout="ctr"),
    ), mock.patch(
        "ainode.engine.backends.nvidia.subprocess.Popen", return_value=_FakePopen()
    ):
        assert b.start_distributed() is False
    assert stopped == ["10.100.0.13", "10.100.0.15"]


def test_mp_peer_launch_goes_over_ssh_with_the_same_plumbing_as_ray():
    b = _backend(_mp_config())
    ssh_cmds: List[List[str]] = []

    def fake_run(cmd, **kwargs):
        ssh_cmds.append(cmd)
        return _FakeCompleted(returncode=0)

    fabric, hca = _nccl_free()
    with fabric, hca, mock.patch(
        "ainode.engine.backends.nvidia.subprocess.run", side_effect=fake_run
    ):
        b._ssh_launch_mp_worker(peer_ip="10.100.0.13", head_ip="10.100.0.11",
                                node_rank=1, nnodes=2)

    assert len(ssh_cmds) == 1
    ssh = ssh_cmds[0]
    assert ssh[0] == "ssh"
    assert "ubuntu@10.100.0.13" in ssh
    remote = ssh[-1]
    worker = f"{WORKER_CONTAINER_NAME_PREFIX}-10-100-0-13"
    assert f"docker rm -f {worker}" in remote
    assert "docker run -d" in remote
    assert "--node-rank 1" in remote
    assert "--headless" in remote
    assert "ray start" not in remote
    # Peer HF cache is created before the container mounts it.
    assert "mkdir -p /home/ubuntu/ainode-nvidia-cache" in remote


def test_mp_ssh_failure_raises():
    b = _backend(_mp_config())
    fabric, hca = _nccl_free()
    with fabric, hca, mock.patch(
        "ainode.engine.backends.nvidia.subprocess.run",
        return_value=_FakeCompleted(returncode=255, stderr="Permission denied"),
    ):
        with pytest.raises(NvidiaBackendError, match="ssh docker run"):
            b._ssh_launch_mp_worker(peer_ip="10.100.0.13", head_ip="10.100.0.11",
                                    node_rank=1, nnodes=2)


def test_mp_launch_refuses_when_the_image_is_unavailable():
    b = _backend(_mp_config())
    fabric, hca = _nccl_free()
    with fabric, hca, mock.patch.object(
        b, "ensure_image", return_value=False
    ), mock.patch.object(b, "_ssh_launch_mp_worker") as peers:
        assert b.start_distributed() is False
    peers.assert_not_called()


# ---------------------------------------------------------------------------
# Readiness / liveness / teardown
# ---------------------------------------------------------------------------


def test_is_running_falls_back_to_the_head_container_for_mp():
    b = _backend(_mp_config())
    with mock.patch.object(b, "_docker_container_state", return_value="running") as st:
        assert b.is_running() is True
    st.assert_called_once_with(HEAD_CONTAINER_NAME)
    with mock.patch.object(b, "_docker_container_state", return_value="exited"):
        assert b.is_running() is False


def test_is_running_does_not_probe_docker_for_the_ray_shape():
    b = _backend(_mp_config(distributed_executor="ray"))
    with mock.patch.object(b, "_docker_container_state") as st:
        assert b.is_running() is False
    st.assert_not_called()


def test_stop_removes_the_head_container_and_every_peer():
    b = _backend(_mp_config(peer_ips=["10.100.0.13", "10.100.0.15"]))
    b._process = _FakePopen()
    with mock.patch.object(b, "_docker_stop_and_rm_best_effort") as rm, \
            mock.patch.object(b, "_ssh_stop_peer_container") as ssh_stop:
        b.stop()
    rm.assert_called_once_with(HEAD_CONTAINER_NAME)
    assert [c.args[0] for c in ssh_stop.call_args_list] == ["10.100.0.13", "10.100.0.15"]


def test_unknown_distributed_executor_is_rejected_loudly():
    b = _backend(_mp_config(distributed_executor="mpi"))
    fabric, hca = _nccl_free()
    with fabric, hca:
        with pytest.raises(NvidiaBackendError, match="Unknown distributed_executor"):
            b.start_distributed()


# ---------------------------------------------------------------------------
# The Ray shape must be untouched
# ---------------------------------------------------------------------------


def test_ray_shape_is_still_the_default_and_unchanged():
    config = _mp_config(distributed_executor="ray", engine_image="")
    b = NvidiaBackend(config)
    assert NodeConfig().distributed_executor == "ray"
    args = b._build_vllm_serve_args(tp_size=2)
    assert args[args.index("--distributed-executor-backend") + 1] == "ray"
    assert "--nnodes" not in args


def test_ray_launch_still_waits_for_the_head_then_ssh_launches_ray_workers():
    b = NvidiaBackend(_mp_config(distributed_executor="ray", engine_image=""))
    fabric, hca = _nccl_free()
    with fabric, hca, mock.patch.object(
        b, "_launch_head_container", return_value="ctr"
    ) as head, mock.patch.object(
        b, "_wait_for_head_container_ready", return_value=True
    ) as wait_head, mock.patch.object(
        b, "_ssh_launch_worker"
    ) as ray_worker, mock.patch.object(
        b, "_ssh_launch_mp_worker"
    ) as mp_worker, mock.patch(
        "ainode.engine.backends.nvidia.subprocess.Popen", return_value=_FakePopen()
    ) as popen:
        assert b.start_distributed() is True
    head.assert_called_once()
    wait_head.assert_called_once()
    ray_worker.assert_called_once()
    mp_worker.assert_not_called()
    assert popen.call_args.args[0][:2] == ["docker", "exec"]


def test_a_recipe_supplied_rendezvous_flag_is_not_duplicated():
    # vLLM errors on a duplicate flag, so a recipe that states one of the
    # rendezvous args keeps its value and we emit nothing for it.
    argv = _head_cmd(_mp_config(extra_vllm_args=["--master-port", "25000"]))
    tail = argv[argv.index(DSPARK_IMAGE) + 4 :]
    assert tail.count("--master-port") == 1
    assert _after(tail, "--master-port") == "25000"
    assert "29501" not in tail


# ---------------------------------------------------------------------------
# Catalog entry
# ---------------------------------------------------------------------------


def test_deepseek_catalog_entry_is_complete():
    from ainode.models.registry import CURATED_CLUSTER_MODELS

    info = CURATED_CLUSTER_MODELS[DSPARK_ID]
    assert info.hf_repo == DSPARK_REPO
    assert info.name == "DeepSeek V4 Flash (DSpark, FP8)"
    assert (info.size_gb, info.params_b) == (159.0, 284.0)
    assert info.context_length == 1048576
    assert info.license == "MIT"
    assert info.family == "deepseek"
    assert info.quantization == "FP8"
    assert info.format == "safetensors"
    assert info.proven_tp == 2
    assert info.verified is False       # flipped only after a live two-node proof
    assert info.recommended is False
    assert info.curated is True
    assert info.distributed_executor == "mp"
    assert info.engine_image == DSPARK_IMAGE
    assert info.kv_cache_dtype == "nvfp4_ds_mla"
    assert info.max_model_len == 1048576
    assert info.trust_remote_code is True
    assert info.recommended_gmu == 0.80
    # The recipe states the flags the backend does NOT emit, and none it does.
    for flag in ("--block-size", "--max-num-seqs", "--max-num-batched-tokens",
                 "--enable-prefix-caching", "--async-scheduling",
                 "--enable-chunked-prefill", "--speculative-config",
                 "--tokenizer-mode", "--tool-call-parser",
                 "--enable-auto-tool-choice", "--reasoning-parser",
                 "--reasoning-config", "--default-chat-template-kwargs",
                 "--generation-config", "--enable-flashinfer-autotune"):
        assert flag in info.extra_vllm_args, flag
    for emitted in ("--tensor-parallel-size", "--nnodes", "--node-rank",
                    "--master-addr", "--master-port", "--kv-cache-dtype",
                    "--max-model-len", "--gpu-memory-utilization", "--host",
                    "--port", "--trust-remote-code"):
        assert emitted not in info.extra_vllm_args, emitted
    assert "13B active" in info.description
    assert "1M-token context" in info.description


def test_deepseek_recipe_env_points_hf_and_the_jit_caches_at_the_mount():
    from ainode.models.registry import CURATED_CLUSTER_MODELS

    env = CURATED_CLUSTER_MODELS[DSPARK_ID].extra_env
    # The image's HOME is /tmp, so without HF_HOME it would re-download 159 GB.
    assert env["HF_HOME"] == HF_CACHE_MOUNT
    assert env["HF_HUB_OFFLINE"] == "1"
    # vllm lives at /opt/env/bin/vllm and the ENTRYPOINT is empty.
    assert env["PATH"].startswith("/opt/env/bin:")
    assert env["DG_JIT_NVCC_COMPILER"] == "/opt/env/bin/nvcc"
    # JIT caches persist per node inside the one directory AINode always mounts.
    for key in ("VLLM_CACHE_ROOT", "DG_JIT_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE",
                "TILELANG_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
                "TORCH_EXTENSIONS_DIR"):
        assert env[key].startswith(f"{HF_CACHE_MOUNT}/.vllm-jit"), key
    # AINode derives these per node; a recipe must not pin them.
    for derived in ("NCCL_IB_HCA", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME",
                    "TP_SOCKET_IFNAME", "VLLM_HOST_IP"):
        assert derived not in env, derived
    assert env["TORCH_CUDA_ARCH_LIST"] == "12.1a"


def test_model_info_recipe_fields_round_trip_through_the_catalog_cache():
    from ainode.models.registry import CURATED_CLUSTER_MODELS, ModelInfo

    info = CURATED_CLUSTER_MODELS[DSPARK_ID]
    assert ModelInfo(**info.to_dict()) == info


def test_deepseek_recipe_reaches_the_launch_config():
    from ainode.models.api_routes import catalog_recipe

    recipe = catalog_recipe(DSPARK_REPO)
    assert recipe == catalog_recipe(DSPARK_ID)
    assert recipe["distributed_executor"] == "mp"
    assert recipe["engine_image"] == DSPARK_IMAGE
    assert recipe["kv_cache_dtype"] == "nvfp4_ds_mla"
    assert recipe["kv_cache_dtype_explicit"] is True
    assert recipe["max_model_len"] == 1048576
    assert recipe["trust_remote_code"] is True
    assert recipe["gpu_memory_utilization"] == 0.80
    # A model with no shape stated stays on the Ray default.
    assert "distributed_executor" not in catalog_recipe("qwen3.8-27b-nvfp4")


def test_rendered_deepseek_commands_carry_the_whole_proven_recipe():
    """End to end on the argv: the catalog entry plus the mp shape reproduce the
    command that served this model on two GB10 nodes."""
    from ainode.models.api_routes import RECIPE_CONFIG_KEYS, catalog_recipe

    recipe = catalog_recipe(DSPARK_REPO)
    cfg_kwargs = {k: recipe[k] for k in RECIPE_CONFIG_KEYS if k in recipe}
    config = _mp_config(model=DSPARK_REPO,
                        gpu_memory_utilization=recipe["gpu_memory_utilization"],
                        **cfg_kwargs)
    head, peer = _head_cmd(config), _peer_cmd(config)

    for argv, rank in ((head, "0"), (peer, "1")):
        joined = " ".join(argv)
        assert "--kv-cache-dtype nvfp4_ds_mla" in joined
        assert "--max-model-len 1048576" in joined
        assert "--gpu-memory-utilization 0.8" in joined
        assert "--trust-remote-code" in joined
        assert "--block-size 256" in joined
        assert '{"method":"dspark","num_speculative_tokens":5' in joined
        assert "--tokenizer-mode deepseek_v4" in joined
        assert f"--node-rank {rank}" in joined
        assert "-e HF_HOME=/root/.cache/huggingface" in joined
        # 0.17-era GB10 workarounds must not ride along on a custom image, and
        # that includes the attention-backend pin: this fork HONORS it, and a
        # dense backend on V4's sparse MLA path is how a serve talks nonsense.
        assert "--enforce-eager" not in joined
        assert "VLLM_ATTENTION_BACKEND" not in joined
    assert "--headless" not in " ".join(head)
    assert " ".join(peer).endswith("--headless")


# ---------------------------------------------------------------------------
# /api/sharding/launch — recipe first, body override on top
# ---------------------------------------------------------------------------


class _FakeBackend:
    """Records the config it was built with; never touches docker."""
    last: dict = {}

    def __init__(self, config, on_ready=None, instance_id=""):
        _FakeBackend.last = {"config": config, "launched": False}

    def is_running(self):
        return False

    def start_distributed(self):
        _FakeBackend.last["launched"] = True
        return True


def _launch(body):
    import ainode.engine.backends as backends
    from ainode.discovery.broadcast import NodeAnnouncement, NodeStatus
    from ainode.discovery.cluster import ClusterNode, ClusterState
    from ainode.engine.sharding_routes import handle_sharding_launch

    _FakeBackend.last = {}
    config = NodeConfig(node_id="head")
    config.save = lambda *a, **k: None  # no disk writes
    cluster = ClusterState(local_announcement=NodeAnnouncement(
        node_id="head", node_name="head", gpu_name="NVIDIA GB10",
        gpu_memory_gb=128.0, unified_memory=True, model="", status="starting",
        api_port=8000, web_port=3000, distributed_mode="head"))
    cluster.add_node(ClusterNode(
        node_id="m1", node_name="host-m1", gpu_name="NVIDIA GB10",
        gpu_memory_gb=128.0, unified_memory=True, model="", status=NodeStatus.ONLINE,
        api_port=8000, web_port=3000, last_seen=0.0, distributed_mode="member",
        peer_ip="192.168.0.13", fabric_ip="10.100.0.13"))
    app = {"cluster_state": cluster, "config": config, "engine": None}

    class _Req:
        def __init__(self):
            self.app = app

        async def json(self):
            return body

    with patch.object(backends, "get_backend", _FakeBackend):
        resp = asyncio.run(handle_sharding_launch(_Req()))
    return config, resp, _FakeBackend.last.get("config")


def test_sharding_launch_applies_the_catalog_recipe():
    config, resp, launched = _launch({"model": DSPARK_REPO, "node_ids": ["head", "m1"]})
    assert resp.status == 200
    assert launched.distributed_executor == "mp"
    assert launched.engine_image == DSPARK_IMAGE
    assert launched.kv_cache_dtype == "nvfp4_ds_mla"
    assert launched.kv_cache_dtype_explicit is True
    assert launched.max_model_len == 1048576
    assert launched.trust_remote_code is True
    assert launched.gpu_memory_utilization == 0.80
    assert "--tokenizer-mode" in launched.extra_vllm_args
    assert launched.extra_env["HF_HOME"] == HF_CACHE_MOUNT
    assert launched.peer_ips == ["10.100.0.13"]
    # Persisted for the primary so a restart replays the same shape.
    assert config.distributed_executor == "mp"
    assert config.engine_image == DSPARK_IMAGE


def test_sharding_launch_body_overrides_beat_the_recipe():
    _, resp, launched = _launch({
        "model": DSPARK_REPO, "node_ids": ["head", "m1"],
        "engine_image": "ghcr.io/x/custom:1",
        "distributed_executor": "ray",
        "max_model_len": 65536,
        "gpu_memory_utilization": 0.55,
        "extra_volumes": ["/data/jit:/vllm-cache"],
    })
    assert resp.status == 200
    assert launched.engine_image == "ghcr.io/x/custom:1"
    assert launched.distributed_executor == "ray"
    assert launched.max_model_len == 65536
    assert launched.gpu_memory_utilization == 0.55
    assert launched.extra_volumes == ["/data/jit:/vllm-cache"]
    # Untouched recipe keys still apply.
    assert launched.kv_cache_dtype == "nvfp4_ds_mla"


def test_sharding_launch_leaves_an_uncurated_model_on_the_defaults():
    _, resp, launched = _launch({"model": "some/random-model",
                                 "node_ids": ["head", "m1"]})
    assert resp.status == 200
    assert launched.distributed_executor == "ray"
    assert launched.engine_image == ""
    assert launched.extra_vllm_args == []


def test_sharding_launch_rejects_a_bad_executor_name():
    _, resp, launched = _launch({"model": "m", "node_ids": ["head", "m1"],
                                 "distributed_executor": "slurm"})
    assert resp.status == 400
    assert launched is None


def test_sharding_launch_records_the_shape_on_the_instance():
    from ainode.discovery.instance import InstanceRecord

    assert InstanceRecord().distributed_executor == "ray"
    record = InstanceRecord(instance_id="h:m", model="m", distributed_executor="mp")
    assert record.to_dict()["distributed_executor"] == "mp"
    assert InstanceRecord.from_dict(record.to_dict()).distributed_executor == "mp"

    import ainode.engine.backends as backends
    from ainode.discovery.broadcast import NodeAnnouncement, NodeStatus
    from ainode.discovery.cluster import ClusterNode, ClusterState
    from ainode.engine.instance_manager import InstanceManager
    from ainode.engine.sharding_routes import handle_sharding_launch

    _FakeBackend.last = {}
    config = NodeConfig(node_id="head")
    config.save = lambda *a, **k: None
    cluster = ClusterState(local_announcement=NodeAnnouncement(
        node_id="head", node_name="head", gpu_name="NVIDIA GB10",
        gpu_memory_gb=128.0, unified_memory=True, model="", status="starting",
        api_port=8000, web_port=3000, distributed_mode="head"))
    cluster.add_node(ClusterNode(
        node_id="m1", node_name="host-m1", gpu_name="NVIDIA GB10",
        gpu_memory_gb=128.0, unified_memory=True, model="", status=NodeStatus.ONLINE,
        api_port=8000, web_port=3000, last_seen=0.0, distributed_mode="member",
        peer_ip="192.168.0.13", fabric_ip="10.100.0.13"))
    manager = InstanceManager(base_port=8000)
    app = {"cluster_state": cluster, "config": config, "engine": None,
           "instances": manager}

    class _Req:
        def __init__(self):
            self.app = app

        async def json(self):
            return {"model": DSPARK_REPO, "node_ids": ["head", "m1"]}

    with patch.object(backends, "get_backend", _FakeBackend):
        resp = asyncio.run(handle_sharding_launch(_Req()))
    assert resp.status == 200
    assert [r.distributed_executor for r in manager.records()] == ["mp"]
