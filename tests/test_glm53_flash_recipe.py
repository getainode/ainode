"""GLM-5.3-Flash on two GB10 nodes, launched by AINode itself.

The entry reproduces the community recipe that ran on 2026-11-11 outside AINode
(profile nvfp4-fp4-dcp2 of coolbho3k/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark at
ab1102a). Three things the earlier recipes did not need are covered here:

* a companion checkpoint (the DFlash2 drafter) that every rank reads from its
  own model store, so the head refuses without it and ships it to a peer;
* a page-cache drop on every rank before its engine starts, plus a watcher on
  the head while the weights load (a rank stalled 14 minutes without it);
* an engine image with the recipe's runtime files baked in, so the launch
  mounts nothing from a host path.

Fakes only: no docker, no ssh, no GPU.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import List
from unittest import mock

import pytest

from ainode.core.config import HF_CACHE_MOUNT, NodeConfig
from ainode.engine.backends.nvidia import (
    ENGINE_IMAGE_DOCKERFILES,
    PAGE_CACHE_DROP_CMD,
    PAGE_CACHE_WATCHER_PREFIX,
    NvidiaBackend,
    NvidiaBackendError,
    engine_image_missing_message,
    page_cache_watch_script,
)
from ainode.models.api_routes import RECIPE_CONFIG_KEYS, catalog_recipe
from ainode.models.registry import (
    CURATED_CLUSTER_MODELS,
    GLM53_FLASH_DRAFT_PATH,
    GLM53_FLASH_DRAFT_REPO,
    GLM53_FLASH_IMAGE,
    ModelInfo,
)
from tests.test_mp_distributed_launch import (
    PEER_CACHE,
    PEER_MODELS,
    _FakeCompleted,
    _FakePopen,
    _after,
    _backend,
    _head_cmd,
    _isolate_ainode_home,  # noqa: F401  (autouse fixture, re-exported here)
    _launch,
    _mp_config,
    _nccl_free,
    _peer_cmd,
)

GLM_ID = "glm-5.3-flash-nvfp4-dflash2"
GLM_REPO = "coolbho3k/GLM-5.3-Flash-NVFP4-Optimized"
GLM_SLUG = GLM_REPO.replace("/", "--")
DRAFT_SLUG = GLM53_FLASH_DRAFT_REPO.replace("/", "--")
MODELS_MOUNT = NvidiaBackend.MODELS_MOUNT
REPO_ROOT = Path(__file__).resolve().parent.parent


def _store(tmp_path, monkeypatch, *, target=True, draft=True) -> str:
    """A model store holding the flat dirs a download through AINode writes."""
    monkeypatch.delenv("AINODE_IN_CONTAINER", raising=False)
    monkeypatch.delenv("AINODE_HOST_HOME", raising=False)
    for present, slug in ((target, GLM_SLUG), (draft, DRAFT_SLUG)):
        if present:
            d = tmp_path / slug
            d.mkdir(parents=True)
            (d / "config.json").write_text("{}")
    return str(tmp_path)


def _glm_config(models_dir: str, **extra) -> NodeConfig:
    recipe = catalog_recipe(GLM_REPO)
    kwargs = {k: recipe[k] for k in RECIPE_CONFIG_KEYS if k in recipe}
    kwargs.update(extra)
    return _mp_config(model=GLM_REPO, models_dir=models_dir,
                      gpu_memory_utilization=recipe["gpu_memory_utilization"],
                      **kwargs)


# ---------------------------------------------------------------------------
# The catalog entry
# ---------------------------------------------------------------------------


def test_glm_entry_is_the_recipe_that_ran():
    info = CURATED_CLUSTER_MODELS[GLM_ID]
    assert info.hf_repo == GLM_REPO
    assert info.curated is True
    assert info.proven_tp == 2
    assert info.distributed_executor == "mp"
    assert info.engine_image == GLM53_FLASH_IMAGE == "glm53-flash-ainode:nvfp4-fp4-dcp2-ab1102a"
    assert info.kv_cache_dtype == "fp8_e4m3"
    assert info.max_model_len == info.context_length == 1048576
    assert info.trust_remote_code is True
    assert info.recommended_gmu == 0.87
    assert info.companion_models == [GLM53_FLASH_DRAFT_REPO]
    assert info.drop_page_cache_before_launch is True
    assert info.page_cache_ceiling_gib == 8
    # Never launched through AINode yet: no bench record, so no verified flag and
    # no load-time seed (engine and catalog provenance rules).
    assert info.verified is False and info.verified_record == ""
    assert info.typical_ready_minutes is None


def test_glm_flags_match_the_recipe_launcher():
    args = CURATED_CLUSTER_MODELS[GLM_ID].extra_vllm_args
    assert _after(args, "--decode-context-parallel-size") == "2"
    assert _after(args, "--max-num-seqs") == "6"
    assert _after(args, "--block-size") == "2304"
    assert _after(args, "--moe-backend") == "marlin"
    assert _after(args, "--max-num-batched-tokens") == "8192"
    assert _after(args, "--scheduler-cls") == \
        "glm_decode_first_scheduler.AdaptiveDecodeFirstScheduler"
    assert _after(args, "--prefill-schedule-interval") == "4"
    assert _after(args, "--long-prefill-token-threshold") == "512"
    assert _after(args, "--tool-call-parser") == "glm47"
    assert _after(args, "--reasoning-parser") == "glm45"
    assert "--enable-auto-tool-choice" in args
    assert "--enforce-eager" in args
    assert json.loads(_after(args, "--default-chat-template-kwargs")) == \
        {"enable_thinking": True, "reasoning_effort": "max"}
    assert _after(args, "--chat-template") == "/opt/glm53-ainode/chat_template_mm.jinja"
    spec = json.loads(_after(args, "--speculative-config"))
    assert spec == {
        "method": "dflash", "model": GLM53_FLASH_DRAFT_PATH,
        "num_speculative_tokens": 7, "draft_tensor_parallel_size": 2,
        "draft_sample_method": "probabilistic",
    }
    # Things the backend renders itself, or must never appear on this hardware.
    for flag in ("--tensor-parallel-size", "--kv-cache-dtype", "--max-model-len",
                 "--gpu-memory-utilization", "--trust-remote-code",
                 "--served-model-name", "--enable-expert-parallel", "--host",
                 "--port", "--nnodes", "--node-rank", "--headless"):
        assert flag not in args, flag


def test_the_draft_path_is_the_model_store_mount():
    """The speculative config names one path, valid on every rank only because
    every rank mounts its own store at MODELS_MOUNT."""
    assert GLM53_FLASH_DRAFT_PATH == f"{MODELS_MOUNT}/{DRAFT_SLUG}"


def test_glm_env_leaves_the_per_node_nccl_values_to_ainode():
    env = CURATED_CLUSTER_MODELS[GLM_ID].extra_env
    for derived in ("NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "VLLM_HOST_IP",
                    "GLOO_SOCKET_IFNAME", "MASTER_ADDR", "HF_TOKEN"):
        assert derived not in env, derived
    assert env["HF_HOME"] == HF_CACHE_MOUNT
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["VLLM_NVFP4_MLA_SCALES_FILE"] == \
        "/opt/glm53-ainode/calibration/nvfp4-gscale-v1.json"
    assert env["VLLM_NVFP4_MLA_SCALES_STRICT"] == "1"
    assert env["GLM_ADAPTIVE_PREFILL_CONFIG"] == \
        "/etc/glm53/adaptive-scheduler/adaptive.json"
    # Every cache root lives inside the HF cache mount so it persists per node
    # and the head can seed a peer (engine/AGENTS.md).
    for key in ("VLLM_CACHE_ROOT", "XDG_CACHE_HOME", "FLASHINFER_WORKSPACE_BASE",
                "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
        assert env[key].startswith(HF_CACHE_MOUNT + "/"), key
    assert all(isinstance(v, str) for v in env.values())


def test_glm_recipe_mounts_nothing_from_a_host_path():
    assert CURATED_CLUSTER_MODELS[GLM_ID].extra_volumes == []


def test_glm_recipe_reaches_the_launch_config():
    recipe = catalog_recipe(GLM_ID)
    assert recipe == catalog_recipe(GLM_REPO)
    assert recipe["companion_models"] == [GLM53_FLASH_DRAFT_REPO]
    assert recipe["drop_page_cache_before_launch"] is True
    assert recipe["page_cache_ceiling_gib"] == 8
    assert recipe["gpu_memory_utilization"] == 0.87
    assert recipe["kv_cache_dtype_explicit"] is True


def test_other_recipes_carry_none_of_the_new_keys():
    """DeepSeek and Flash-Next launch exactly as before."""
    for model in ("deepseek-v4-flash-dspark", "qwen3.8-flash-next-nvfp4"):
        recipe = catalog_recipe(model)
        for key in ("companion_models", "drop_page_cache_before_launch",
                    "page_cache_ceiling_gib"):
            assert key not in recipe, (model, key)


def test_new_fields_round_trip_through_the_catalog_cache():
    info = CURATED_CLUSTER_MODELS[GLM_ID]
    same = ModelInfo(**info.to_dict())
    assert same.companion_models == info.companion_models
    assert same.drop_page_cache_before_launch is True
    assert same.page_cache_ceiling_gib == 8
    old = {k: v for k, v in info.to_dict().items()
           if k not in ("companion_models", "drop_page_cache_before_launch",
                        "page_cache_ceiling_gib")}
    stale = ModelInfo(**old)
    assert stale.companion_models == [] and stale.drop_page_cache_before_launch is False


def test_description_has_no_em_dash():
    assert "\u2014" not in CURATED_CLUSTER_MODELS[GLM_ID].description


# ---------------------------------------------------------------------------
# The engine image
# ---------------------------------------------------------------------------


def test_a_missing_glm_image_names_its_dockerfile():
    assert ENGINE_IMAGE_DOCKERFILES["glm53-flash-ainode"] == "scripts/Dockerfile.glm53-flash"
    assert "scripts/Dockerfile.glm53-flash" in engine_image_missing_message(GLM53_FLASH_IMAGE)
    assert (REPO_ROOT / "scripts/Dockerfile.glm53-flash").is_file()


def test_the_dockerfile_bakes_every_path_the_entry_names():
    dockerfile = (REPO_ROOT / "scripts/Dockerfile.glm53-flash").read_text()
    env = CURATED_CLUSTER_MODELS[GLM_ID].extra_env
    args = CURATED_CLUSTER_MODELS[GLM_ID].extra_vllm_args
    for path in (env["VLLM_NVFP4_MLA_SCALES_FILE"], env["GLM_ADAPTIVE_PREFILL_CONFIG"],
                 _after(args, "--chat-template"),
                 "/usr/local/lib/python3.12/dist-packages/glm_decode_first_scheduler.py",
                 "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
                 "sparse_attn_indexer_kpool.py"):
        assert path in dockerfile, path
    assert "ARG BASE=glm53-v14:nvfp4-gscale-tooling" in dockerfile
    assert "ENTRYPOINT" not in dockerfile.replace("ENTRYPOINT stays", "")
    assert "\u2014" not in dockerfile


def test_the_build_script_pins_the_tag_the_entry_launches():
    script = (REPO_ROOT / "scripts/build-glm53-flash-image.sh").read_text()
    assert f"IMAGE=${{IMAGE:-{GLM53_FLASH_IMAGE}}}" in script
    assert "ab1102a8fd2c8a7102bdbfcda0f1402e37504073" in script
    assert subprocess.run(["bash", "-n", str(REPO_ROOT / "scripts/build-glm53-flash-image.sh")]
                          ).returncode == 0


def test_the_build_script_refuses_a_checkout_at_another_commit(tmp_path):
    repo = tmp_path / "recipe"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    out = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/build-glm53-flash-image.sh"), str(repo)],
        capture_output=True, text=True)
    assert out.returncode != 0
    assert "expected ab1102a8fd2c8a7102bdbfcda0f1402e37504073" in out.stderr


# ---------------------------------------------------------------------------
# Rendered commands
# ---------------------------------------------------------------------------


def test_rendered_glm_commands_serve_the_store_copy_on_both_ranks(tmp_path, monkeypatch):
    config = _glm_config(_store(tmp_path, monkeypatch))
    head, peer = _head_cmd(config), _peer_cmd(config)
    for argv, rank, store in ((head, "0", str(tmp_path)), (peer, "1", PEER_MODELS)):
        joined = " ".join(argv)
        assert GLM53_FLASH_IMAGE in argv
        assert f"-v {store}:{MODELS_MOUNT}:ro" in joined
        # The image's ENTRYPOINT is ["vllm","serve"]; the faked probe says [] so
        # the prefix is spelled out, which is the backend's job either way.
        assert f"serve {MODELS_MOUNT}/{GLM_SLUG}" in joined
        assert f"--served-model-name {GLM_REPO}" in joined
        assert "--tensor-parallel-size 2" in joined
        assert "--distributed-executor-backend mp" in joined
        assert "--kv-cache-dtype fp8_e4m3" in joined
        assert "--max-model-len 1048576" in joined
        assert "--gpu-memory-utilization 0.87" in joined
        assert "--trust-remote-code" in joined
        assert "--decode-context-parallel-size 2" in joined
        assert GLM53_FLASH_DRAFT_PATH in joined
        assert f"--node-rank {rank}" in joined
        assert joined.count("--kv-cache-dtype") == 1
        assert joined.count("--enforce-eager") == 1
        # Nothing from a host path besides the HF cache and the model store.
        mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
        assert len(mounts) == 2, mounts
        assert "--privileged" not in argv
    assert " ".join(peer).endswith("--headless")


def test_sharding_launch_carries_the_new_keys_into_the_instance_config():
    config, resp, launched = _launch({"model": GLM_REPO, "node_ids": ["head", "m1"]})
    assert resp.status == 200
    assert launched.engine_image == GLM53_FLASH_IMAGE
    assert launched.companion_models == [GLM53_FLASH_DRAFT_REPO]
    assert launched.drop_page_cache_before_launch is True
    assert launched.page_cache_ceiling_gib == 8
    assert launched.gpu_memory_utilization == 0.87
    # Persisted on the primary, so a restart relaunches the same shape.
    assert config.companion_models == [GLM53_FLASH_DRAFT_REPO]
    assert config.drop_page_cache_before_launch is True


def test_the_distributed_record_keeps_the_new_keys():
    from ainode.engine.reconcile import load_distributed_record

    _launch({"model": GLM_REPO, "node_ids": ["head", "m1"]})
    overrides = load_distributed_record()["overrides"]
    assert overrides["companion_models"] == [GLM53_FLASH_DRAFT_REPO]
    assert overrides["drop_page_cache_before_launch"] is True
    assert overrides["page_cache_ceiling_gib"] == 8


def test_config_overrides_read_the_new_keys_off_a_config():
    from ainode.engine.reconcile import _config_overrides

    out = _config_overrides(NodeConfig(companion_models=["a/b"],
                                       drop_page_cache_before_launch=True,
                                       page_cache_ceiling_gib=8))
    assert out["companion_models"] == ["a/b"]
    assert out["drop_page_cache_before_launch"] is True
    assert out["page_cache_ceiling_gib"] == 8


# ---------------------------------------------------------------------------
# Companions: refuse on the head, ship to the peer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target,draft,missing", [
    (True, False, [GLM53_FLASH_DRAFT_REPO]),
    (False, True, [GLM_REPO]),
    (False, False, [GLM_REPO, GLM53_FLASH_DRAFT_REPO]),
])
def test_launch_refuses_before_touching_a_peer_when_a_checkpoint_is_missing(
        tmp_path, monkeypatch, target, draft, missing):
    config = _glm_config(_store(tmp_path, monkeypatch, target=target, draft=draft))
    b = _backend(config)
    fabric, hca = _nccl_free()
    with fabric, hca, mock.patch.object(b, "ensure_image", return_value=True), \
            mock.patch.object(b, "_ssh_launch_mp_worker") as peer, \
            mock.patch("ainode.engine.backends.nvidia.subprocess.run") as run:
        with pytest.raises(NvidiaBackendError) as err:
            b.start_distributed()
    peer.assert_not_called()
    run.assert_not_called()   # no docker, no ssh, no page-cache drop
    for repo in missing:
        assert repo in str(err.value)
    assert b.launch_error == str(err.value)
    assert "no peer was touched" in str(err.value)


def test_a_recipe_without_companions_skips_the_check(tmp_path, monkeypatch):
    b = _backend(_mp_config(models_dir=str(tmp_path)))
    b._check_companions_on_head()  # DeepSeek shape: nothing to check, no raise


def test_peer_gets_the_draft_beside_the_target(tmp_path, monkeypatch):
    b = _backend(_glm_config(_store(tmp_path, monkeypatch)))
    shipped: List[str] = []

    def fake_dist(*, peer_ip, name, head_parent, peer_parent):
        shipped.append(f"{head_parent}/{name}->{peer_parent}/{name}")

    with mock.patch.object(b, "_distribute_dir_to_peer", side_effect=fake_dist):
        b._ensure_peer_has_model("198.51.100.5", PEER_CACHE, PEER_MODELS)
    assert shipped == [
        f"{tmp_path}/{GLM_SLUG}->{PEER_MODELS}/{GLM_SLUG}",
        f"{tmp_path}/{DRAFT_SLUG}->{PEER_MODELS}/{DRAFT_SLUG}",
    ]


# ---------------------------------------------------------------------------
# Page cache: drop on every rank, watch on the head, clean up on stop
# ---------------------------------------------------------------------------


def _full_launch(b, *, run_rc=0):
    """Run start_distributed with docker and ssh faked; return every argv run."""
    calls: List[List[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[0] == "ssh" and "test -d" in cmd[-1]:
            return _FakeCompleted(stdout="present\n")
        if cmd[:3] == ["docker", "run", "--rm"] or "drop_caches" in " ".join(cmd):
            return _FakeCompleted(returncode=run_rc, stderr="denied" if run_rc else "")
        return _FakeCompleted(returncode=0, stdout="ctr")

    fabric, hca = _nccl_free()
    with fabric, hca, mock.patch.object(b, "ensure_image", return_value=True), \
            mock.patch.object(b, "_docker_stop_and_rm_best_effort"), \
            mock.patch.object(b, "_docker_container_state", return_value="running"), \
            mock.patch.object(b, "_ensure_peer_has_jit_cache"), \
            mock.patch("ainode.engine.backends.nvidia.subprocess.run", side_effect=fake_run), \
            mock.patch("ainode.engine.backends.nvidia.subprocess.Popen",
                       return_value=_FakePopen()):
        started = b.start_distributed()
    return started, calls


def _index(calls, pred):
    for i, c in enumerate(calls):
        if pred(c):
            return i
    raise AssertionError(f"no matching call in {calls}")


def test_page_cache_is_dropped_on_the_peer_then_the_head_before_each_engine(tmp_path, monkeypatch):
    b = _backend(_glm_config(_store(tmp_path, monkeypatch)))
    started, calls = _full_launch(b)
    assert started is True

    peer_drop = _index(calls, lambda c: c[0] == "ssh" and "drop_caches" in c[-1])
    peer_run = _index(calls, lambda c: c[0] == "ssh" and "--headless" in c[-1])
    head_drop = _index(calls, lambda c: c[:3] == ["docker", "run", "--rm"])
    watcher = _index(calls, lambda c: c[:4] == ["docker", "run", "-d", "--rm"])
    head_run = _index(calls, lambda c: c[:3] == ["docker", "run", "-d"]
                      and GLM53_FLASH_IMAGE in c and "--entrypoint" not in c)
    assert peer_drop < peer_run < head_drop < watcher < head_run

    # The peer drop is the fixed command in the engine image, privileged, offline.
    remote = shlex.split(calls[peer_drop][-1])
    assert remote[:9] == ["docker", "run", "--rm", "--privileged", "--network", "none",
                          "--entrypoint", "sh", GLM53_FLASH_IMAGE]
    assert remote[-2:] == ["-c", PAGE_CACHE_DROP_CMD]
    assert calls[peer_drop][-2] == "ubuntu@198.51.100.5"

    head = calls[head_drop]
    assert head[-2:] == ["-c", PAGE_CACHE_DROP_CMD]
    w = calls[watcher]
    assert w[w.index("--name") + 1] == PAGE_CACHE_WATCHER_PREFIX
    assert "--privileged" in w and w[w.index("--network") + 1] == "none"
    assert w[-1] == page_cache_watch_script(8)


def test_a_failed_drop_warns_and_the_launch_goes_on(tmp_path, monkeypatch, caplog):
    b = _backend(_glm_config(_store(tmp_path, monkeypatch)))
    started, calls = _full_launch(b, run_rc=1)
    assert started is True
    assert "Page-cache drop on the head failed" in caplog.text
    assert "Page-cache drop on 198.51.100.5 failed" in caplog.text


def test_an_overridden_engine_image_never_runs_privileged(tmp_path, monkeypatch, caplog):
    """A caller may override engine_image on a launch; that image must not be
    what the privileged page-cache helper runs. The drop is skipped instead."""
    b = _backend(_glm_config(_store(tmp_path, monkeypatch),
                             engine_image="someone/else:latest"))
    started, calls = _full_launch(b)
    assert started is True
    flat = [" ".join(c) for c in calls]
    assert not any("--privileged" in c or "drop_caches" in c for c in flat), flat
    assert "Page-cache drop skipped" in caplog.text


def test_no_ceiling_means_a_drop_but_no_watcher(tmp_path, monkeypatch):
    b = _backend(_glm_config(_store(tmp_path, monkeypatch), page_cache_ceiling_gib=0))
    _, calls = _full_launch(b)
    assert any(c[:3] == ["docker", "run", "--rm"] for c in calls)
    assert not any(c[:4] == ["docker", "run", "-d", "--rm"] for c in calls)


def test_deepseek_launch_runs_no_privileged_container(tmp_path, monkeypatch):
    """The recipe that is serving today must launch exactly as before."""
    b = _backend(_mp_config(models_dir=str(tmp_path)))
    _, calls = _full_launch(b)
    flat = [" ".join(c) for c in calls]
    assert not any("--privileged" in c or "drop_caches" in c for c in flat), flat


def test_stop_removes_the_watcher_with_the_engine(tmp_path, monkeypatch):
    b = _backend(_glm_config(_store(tmp_path, monkeypatch)))
    calls: List[List[str]] = []
    with mock.patch.object(b, "_docker_stop_and_rm_best_effort"), \
            mock.patch.object(b, "_ssh_stop_peer_container"), \
            mock.patch("ainode.engine.backends.nvidia.subprocess.run",
                       side_effect=lambda cmd, **kw: calls.append(cmd) or _FakeCompleted()):
        b.stop()
    assert ["docker", "rm", "-f", PAGE_CACHE_WATCHER_PREFIX] in calls


def test_a_stacked_instance_gets_its_own_watcher_name(tmp_path, monkeypatch):
    b = NvidiaBackend(_glm_config(_store(tmp_path, monkeypatch), api_port=8001),
                      instance_id="8001")
    assert b._page_cache_watcher_name() == f"{PAGE_CACHE_WATCHER_PREFIX}-8001"


def test_watch_script_interpolates_only_integers():
    script = page_cache_watch_script(8, seconds=1500, interval=5)
    assert "-gt 8388608" in script            # 8 GiB in kB, as /proc/meminfo reports
    assert "+ 1500 ))" in script and "sleep 5" in script
    with pytest.raises((TypeError, ValueError)):
        page_cache_watch_script("8; rm -rf /")  # type: ignore[arg-type]


def test_watch_script_runs_under_posix_sh(tmp_path):
    """Two short iterations against this host's /proc/meminfo with the drop
    replaced by an echo: proves the loop parses and the Cached read works."""
    script = page_cache_watch_script(0, seconds=1, interval=1).replace(
        "echo 3 > /proc/sys/vm/drop_caches", "echo WOULD_DROP")
    shell = "dash" if Path("/bin/dash").exists() else "sh"
    out = subprocess.run([shell, "-c", script], capture_output=True, text=True, timeout=20)
    assert out.returncode == 0, out.stderr
    assert "WOULD_DROP" in out.stdout
