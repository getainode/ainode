"""Per-instance engine recipe: extra vLLM flags + engine image override.

Covers the 0.5.4 launch-path work that lets a model's published recipe
(spec-decode, MoE/mamba backends, reasoning + tool-call parsers) be expressed
through the normal load path instead of a hand-rolled container.
"""

import pytest

from ainode.core.config import NodeConfig
from ainode.engine.backends.nvidia import NvidiaBackend, NVIDIA_VLLM_IMAGE
from ainode.models.api_routes import catalog_recipe
from ainode.models.registry import CURATED_CLUSTER_MODELS


def args_for(**cfg_kwargs):
    cfg = NodeConfig(model="m", **cfg_kwargs)
    return NvidiaBackend(cfg)._build_vllm_serve_args(1)


# --- defaults must not move (every existing model still launches as before) ---

def test_default_launch_is_unchanged():
    a = args_for()
    assert "--enforce-eager" in a, "0.17-era GB10 workaround must stay on by default"
    assert "--kv-cache-dtype" in a
    assert NvidiaBackend(NodeConfig(model="m"))._engine_image() == NVIDIA_VLLM_IMAGE


# --- extra_vllm_args passthrough ---

def test_extra_args_are_appended_verbatim_and_in_order():
    extra = ["--moe-backend", "marlin", "--reasoning-parser", "nemotron_v3"]
    assert args_for(extra_vllm_args=extra)[-4:] == extra


def test_caller_flag_suppresses_the_builtin_rather_than_duplicating():
    # vLLM errors on duplicate flags, so the caller's value must REPLACE ours.
    a = args_for(gpu_memory_utilization=0.5,
                 extra_vllm_args=["--gpu-memory-utilization", "0.91"])
    assert a.count("--gpu-memory-utilization") == 1
    assert "0.91" in a and "0.5" not in a


def test_equals_form_also_suppresses_the_builtin():
    a = args_for(extra_vllm_args=["--kv-cache-dtype=auto"])
    assert a.count("--kv-cache-dtype") == 0
    assert "--kv-cache-dtype=auto" in a


def test_caller_may_reenable_enforce_eager_on_a_custom_image():
    a = args_for(engine_image="vllm/vllm-openai:v0.27.1",
                 extra_vllm_args=["--enforce-eager"])
    assert a.count("--enforce-eager") == 1


# --- engine image override gates the legacy workarounds ---

def test_custom_image_drops_legacy_gb10_workarounds():
    a = args_for(engine_image="vllm/vllm-openai:v0.27.1")
    assert "--enforce-eager" not in a, (
        "--enforce-eager is a 0.17 FlashInfer workaround; on 0.27.1 it only "
        "disables CUDA graphs and costs throughput"
    )


def test_custom_image_drops_nvfp4_marlin_env():
    b = NvidiaBackend(NodeConfig(model="some/model-NVFP4",
                                 engine_image="vllm/vllm-openai:v0.27.1"))
    assert b._nvfp4_serve_env() == {}


def test_pinned_default_image_keeps_nvfp4_marlin_env():
    b = NvidiaBackend(NodeConfig(model="some/model-NVFP4"))
    assert b._nvfp4_serve_env().get("VLLM_NVFP4_GEMM_BACKEND") == "marlin"


def test_pinned_default_image_keeps_the_attention_backend_pin():
    # A no-op on this build, but kept byte-identical: it is the 0.17-line hedge.
    b = NvidiaBackend(NodeConfig(model="m"))
    assert b._attention_backend_env() == {"VLLM_ATTENTION_BACKEND": "TRITON_ATTN"}
    assert "VLLM_ATTENTION_BACKEND=TRITON_ATTN" in b._build_solo_docker_cmd("c")


def test_pinned_default_image_attention_pin_stays_env_overridable(monkeypatch):
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "TRITON_ATTN_VLLM_V1")
    b = NvidiaBackend(NodeConfig(model="m"))
    assert b._attention_backend_env() == {
        "VLLM_ATTENTION_BACKEND": "TRITON_ATTN_VLLM_V1"}


@pytest.mark.parametrize("image", ["vllm/vllm-openai:v0.27.1",
                                   "vllm-dspark-runtime:dspark-nvfp4-stage-c"])
def test_custom_image_gets_no_attention_backend_override(image):
    # 0.27/0.28 log it as an unknown variable, but the 0.21-based GB10 fork
    # HONORS it, and forcing a dense attention backend onto DeepSeek V4's sparse
    # MLA path is how a serve produces confident nonsense. A recipe that wants
    # one states it in extra_env.
    b = NvidiaBackend(NodeConfig(model="m", engine_image=image))
    assert b._attention_backend_env() == {}
    assert not any(str(a).startswith("VLLM_ATTENTION_BACKEND")
                   for a in b._build_solo_docker_cmd("c"))


def test_a_recipe_can_still_set_the_attention_backend_on_a_custom_image():
    b = NvidiaBackend(NodeConfig(
        model="m", engine_image="ghcr.io/x/y:1",
        extra_env={"VLLM_ATTENTION_BACKEND": "FLASHINFER"}))
    assert b._engine_env({})["VLLM_ATTENTION_BACKEND"] == "FLASHINFER"


def test_engine_image_override_is_used_for_the_container():
    b = NvidiaBackend(NodeConfig(model="m", engine_image="ghcr.io/x/y:1"))
    assert b._engine_image() == "ghcr.io/x/y:1"
    assert "ghcr.io/x/y:1" in b._build_solo_docker_cmd("c")


def test_no_rm_flag_so_a_crashed_engine_leaves_a_corpse():
    # --rm deleted crashed containers before anyone could read their logs.
    assert "--rm" not in NvidiaBackend(NodeConfig(model="m"))._build_solo_docker_cmd("c")


# --- catalog recipes ---

@pytest.mark.parametrize("key", ["nemotron-3.5-lightning-nvfp4", "qwen3.8-27b-nvfp4"])
def test_curated_recipe_models_carry_a_complete_recipe(key):
    info = CURATED_CLUSTER_MODELS[key]
    assert info.engine_image, "recipe models need a pinned engine image"
    assert info.extra_vllm_args, "recipe models need their flag set"
    assert info.recommended_gmu > 0


def test_recipe_matches_on_both_catalog_id_and_hf_repo():
    info = CURATED_CLUSTER_MODELS["qwen3.8-27b-nvfp4"]
    assert catalog_recipe(info.id) == catalog_recipe(info.hf_repo) != {}


def test_uncurated_model_gets_no_recipe():
    assert catalog_recipe("some/random-model") == {}
    assert catalog_recipe("") == {}


def test_qwen38_recipe_uses_qwen3_coder_tool_parser():
    # hermes silently parses ZERO tool calls for this template — proven on hardware.
    a = CURATED_CLUSTER_MODELS["qwen3.8-27b-nvfp4"].extra_vllm_args
    assert a[a.index("--tool-call-parser") + 1] == "qwen3_coder"


def test_recipes_never_hardcode_enforce_eager():
    for key in ("nemotron-3.5-lightning-nvfp4", "qwen3.8-27b-nvfp4"):
        assert "--enforce-eager" not in CURATED_CLUSTER_MODELS[key].extra_vllm_args


def test_recipe_flags_survive_into_the_launch_command():
    info = CURATED_CLUSTER_MODELS["nemotron-3.5-lightning-nvfp4"]
    recipe = catalog_recipe(info.hf_repo)
    a = args_for(engine_image=recipe["engine_image"],
                 extra_vllm_args=recipe["extra_vllm_args"])
    assert "--speculative_config.model" in a
    assert "nemotron_v3" in a
    assert "--enforce-eager" not in a


# --- launch confirmation: a crashed engine must NOT report success -----------

class TestLaunchConfirmation:
    """`docker run -d` returns as soon as the CLI forks, so the old
    `poll() is None` check reported success for engines that died on startup —
    the caller then registered an instance that never existed (phantom rows,
    2026-08-14). start_solo() now confirms the container reached Running.
    """

    def _backend(self):
        return NvidiaBackend(NodeConfig(model="m"))

    def test_running_container_confirms(self):
        b = self._backend()
        b._docker_container_state = lambda name: "running"
        assert b._confirm_container_started("c", timeout=1) is True

    def test_exited_container_is_a_failed_launch(self):
        b = self._backend()
        b._docker_container_state = lambda name: "exited"
        b._docker_logs_tail = lambda name, lines=15: "ValueError: No available memory"
        assert b._confirm_container_started("c", timeout=5) is False

    def test_missing_container_is_a_failed_launch(self):
        b = self._backend()
        b._docker_container_state = lambda name: ""
        b._docker_logs_tail = lambda name, lines=15: ""
        assert b._confirm_container_started("c", timeout=1) is False

    def test_failure_surfaces_the_engine_logs(self, caplog):
        b = self._backend()
        b._docker_container_state = lambda name: "exited"
        b._docker_logs_tail = lambda name, lines=15: "ValueError: No available memory for the cache blocks"
        with caplog.at_level("ERROR"):
            b._confirm_container_started("c", timeout=1)
        assert "No available memory" in caplog.text, (
            "a failed launch must surface WHY, or every failure looks like silence"
        )


# --- entrypoint normalization across engine images ---------------------------

class TestServeArgvPrefix:
    """A per-instance image makes ENTRYPOINT differences our problem:
    vllm/vllm-openai bakes ["vllm","serve"], so emitting our own produced
    `vllm serve vllm serve <model>` and the engine exited with
    "unrecognized arguments" (caught on hardware 2026-08-15).
    """

    def _b(self, entrypoint):
        b = NvidiaBackend(NodeConfig(model="m"))
        b._image_entrypoint = lambda image: entrypoint
        return b

    def test_baked_vllm_serve_entrypoint_adds_nothing(self):
        assert self._b(["vllm", "serve"])._serve_argv_prefix("i") == []

    def test_vllm_entrypoint_adds_only_serve(self):
        assert self._b(["/usr/local/bin/vllm"])._serve_argv_prefix("i") == ["serve"]

    def test_nvidia_shim_entrypoint_gets_full_prefix(self):
        assert self._b(["/opt/nvidia/nvidia_entrypoint.sh"])._serve_argv_prefix("i") == ["vllm", "serve"]

    def test_unknown_image_falls_back_to_legacy_prefix(self):
        # A docker hiccup must never silently change how the default image launches.
        assert self._b([])._serve_argv_prefix("i") == ["vllm", "serve"]


def test_qwen38_recipe_pins_kv_cache_auto_for_vision():
    # fp8 KV corrupts VLM generation on GB10; the automatic downgrade only fires
    # for models on local disk, and this one serves from the HF cache.
    a = CURATED_CLUSTER_MODELS["qwen3.8-27b-nvfp4"].extra_vllm_args
    assert a[a.index("--kv-cache-dtype") + 1] == "auto"
    built = args_for(extra_vllm_args=a)
    assert built.count("--kv-cache-dtype") == 1 and "fp8" not in built


# --- extra_env passthrough (b12x and friends are env-selected, not flag-selected) ---

def _env_pairs(cmd):
    """Extract {NAME: value} from the `-e NAME=value` pairs of a docker cmd."""
    out = {}
    for i, tok in enumerate(cmd):
        if tok == "-e" and i + 1 < len(cmd) and "=" in cmd[i + 1]:
            k, _, v = cmd[i + 1].partition("=")
            out[k] = v
    return out


def test_extra_env_reaches_the_engine_container():
    b = NvidiaBackend(NodeConfig(model="m", extra_env={
        "VLLM_NVFP4_GEMM_BACKEND": "flashinfer-b12x",
        "VLLM_USE_FLASHINFER_MOE_FP4": "1",
    }))
    env = _env_pairs(b._build_solo_docker_cmd("c"))
    assert env["VLLM_NVFP4_GEMM_BACKEND"] == "flashinfer-b12x"
    assert env["VLLM_USE_FLASHINFER_MOE_FP4"] == "1"


def test_extra_env_overrides_a_computed_value():
    # The pinned default image forces VLLM_NVFP4_GEMM_BACKEND=marlin. A recipe
    # selecting the b12x kernel path must win, or b12x is unreachable on it.
    b = NvidiaBackend(NodeConfig(model="some/model-NVFP4",
                                 extra_env={"VLLM_NVFP4_GEMM_BACKEND": "flashinfer-b12x"}))
    assert b._nvfp4_serve_env().get("VLLM_NVFP4_GEMM_BACKEND") == "marlin"
    assert _env_pairs(b._build_solo_docker_cmd("c"))["VLLM_NVFP4_GEMM_BACKEND"] == "flashinfer-b12x"


def test_no_extra_env_leaves_the_command_untouched():
    base = NvidiaBackend(NodeConfig(model="m"))._build_solo_docker_cmd("c")
    same = NvidiaBackend(NodeConfig(model="m", extra_env={}))._build_solo_docker_cmd("c")
    assert base == same


def test_extra_env_values_are_stringified():
    b = NvidiaBackend(NodeConfig(model="m", extra_env={"FLASHINFER_DISABLE_VERSION_CHECK": 1}))
    assert _env_pairs(b._build_solo_docker_cmd("c"))["FLASHINFER_DISABLE_VERSION_CHECK"] == "1"


def test_catalog_recipe_surfaces_extra_env_when_a_model_carries_it():
    from ainode.models.registry import ModelInfo
    import ainode.models.registry as reg
    probe = ModelInfo(id="probe-b12x", name="probe", hf_repo="org/probe-b12x",
                      size_gb=1.0, description="d",
                      extra_env={"VLLM_NVFP4_GEMM_BACKEND": "flashinfer-b12x"})
    reg.CURATED_CLUSTER_MODELS["probe-b12x"] = probe
    try:
        assert catalog_recipe("org/probe-b12x")["extra_env"] == {
            "VLLM_NVFP4_GEMM_BACKEND": "flashinfer-b12x"}
    finally:
        del reg.CURATED_CLUSTER_MODELS["probe-b12x"]


# --- two entries, one checkpoint: the V100 Flash-Next lane (castor) -----------
#
# qwen3.8-flash-next-nvfp4-v100 is the first catalog entry to share an hf_repo
# with another one. The GB10 entry pins an aarch64 nightly at TP=2 in the mp
# shape; this one pins a local SM70 build on four Volta cards in one box. Neither
# recipe runs on the other's hardware, so which entry a load resolves to is the
# whole question, and it must not be an accident of dict order.

V100_FLASH_ID = "qwen3.8-flash-next-nvfp4-v100"
FLASH_REPO = "nvidia/Qwen3.8-Flash-Next-NVFP4"


def test_the_v100_flash_next_lane_resolves_by_its_catalog_id():
    recipe = catalog_recipe(V100_FLASH_ID)
    assert recipe["engine_image"] == "onecat-vllm:1.5.0-mm"
    assert recipe["kv_cache_dtype"] == "auto"
    assert recipe["kv_cache_dtype_explicit"] is True
    assert recipe["max_model_len"] == 262144
    assert recipe["trust_remote_code"] is True
    assert recipe["gpu_memory_utilization"] == 0.94
    # No distributed shape: this is four cards in ONE box, a solo launch.
    assert "distributed_executor" not in recipe
    a = recipe["extra_vllm_args"]
    assert a[a.index("--tensor-parallel-size") + 1] == "4"
    assert a[a.index("--attention-backend") + 1] == "FLASH_ATTN_V100"
    assert a[a.index("--max-num-seqs") + 1] == "4"
    assert a[a.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert a[a.index("--speculative-config") + 1] == (
        '{"method":"qwen4_exp_mtp","num_speculative_tokens":4}')


def test_the_shared_repo_id_still_answers_with_the_gb10_entry():
    """A bare repo id cannot say which hardware is asking, so it keeps answering
    with the GB10 entry it always answered with. The V100 entry's description
    says so, because a repo-id load of it on Volta would fetch an aarch64
    nightly that cannot run there."""
    assert catalog_recipe(FLASH_REPO) == catalog_recipe("qwen3.8-flash-next-nvfp4")
    assert catalog_recipe(FLASH_REPO)["engine_image"].startswith("vllm/vllm-openai:nightly-")
    assert catalog_recipe(FLASH_REPO) != catalog_recipe(V100_FLASH_ID)


def test_an_id_match_beats_a_repo_match_whatever_the_dict_order():
    """The id pass runs first, so an entry whose id is another entry's repo id
    cannot be shadowed by dict order."""
    from ainode.models.registry import ModelInfo
    import ainode.models.registry as reg
    shared = "org/probe-shared-xz1"
    reg.CURATED_CLUSTER_MODELS["probe-repo-first"] = ModelInfo(
        id="probe-repo-first", name="a", hf_repo=shared, size_gb=1.0,
        description="d", engine_image="image:repo-match")
    reg.CURATED_CLUSTER_MODELS[shared] = ModelInfo(
        id=shared, name="b", hf_repo="org/other-xz1", size_gb=1.0,
        description="d", engine_image="image:id-match")
    try:
        assert catalog_recipe(shared)["engine_image"] == "image:id-match"
    finally:
        del reg.CURATED_CLUSTER_MODELS["probe-repo-first"]
        del reg.CURATED_CLUSTER_MODELS[shared]


def test_the_v100_flash_next_recipe_renders_a_four_card_solo_command(tmp_path, monkeypatch):
    """End to end on the argv: the entry's recipe, loaded by catalog id, serves
    the on-disk weights at TP=4 on the Volta backend with MTP on, and carries
    none of the GB10 workarounds."""
    from ainode.models.api_routes import RECIPE_CONFIG_KEYS

    monkeypatch.delenv("AINODE_IN_CONTAINER", raising=False)
    monkeypatch.delenv("AINODE_HOST_HOME", raising=False)
    # A solo load derives the on-disk dir from the model string it is given, so a
    # catalog-id load looks for a directory of that name. On castor that is a
    # relative symlink beside the weights (see the entry's description).
    weights = tmp_path / V100_FLASH_ID
    weights.mkdir()
    (weights / "config.json").write_text("{}")

    recipe = catalog_recipe(V100_FLASH_ID)
    cfg_kwargs = {k: recipe[k] for k in RECIPE_CONFIG_KEYS if k in recipe}
    cfg = NodeConfig(model=V100_FLASH_ID, models_dir=str(tmp_path), api_port=8000,
                     gpu_memory_utilization=recipe["gpu_memory_utilization"],
                     **cfg_kwargs)
    b = NvidiaBackend(cfg)
    b._image_entrypoint = lambda image: ["vllm"]  # type: ignore[assignment]
    joined = " ".join(b._build_solo_docker_cmd("c"))

    assert "onecat-vllm:1.5.0-mm serve /ainode-models/" + V100_FLASH_ID in joined
    assert f"--served-model-name {V100_FLASH_ID}" in joined
    assert "--tensor-parallel-size 4" in joined
    assert "--attention-backend FLASH_ATTN_V100" in joined
    assert "--kv-cache-dtype auto" in joined
    assert "--max-model-len 262144" in joined
    assert "--gpu-memory-utilization 0.94" in joined
    assert "--max-num-seqs 4" in joined
    assert "--trust-remote-code" in joined
    assert '{"method":"qwen4_exp_mtp","num_speculative_tokens":4}' in joined
    # A local SM70 build is not the pinned GB10 default, so none of the 0.17-era
    # GB10 workarounds may ride along.
    assert "--enforce-eager" not in joined
    assert "VLLM_ATTENTION_BACKEND" not in joined
    # Volta cannot do fp8 KV, and the node default is fp8: the recipe's auto has
    # to be what reaches the engine.
    assert "--kv-cache-dtype fp8" not in joined
