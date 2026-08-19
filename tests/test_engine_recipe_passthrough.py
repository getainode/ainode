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
