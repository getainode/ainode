"""Tests for NvidiaBackend._build_vllm_serve_args — the GB10 vLLM flag defaults.

Closet #301: assert the proven distributed-MoE flags land in the serve args —
fp8 KV cache (long-context survival) + tensor-parallel sizing — and that the
fp8 default has an escape hatch.
"""

from ainode.core.config import NodeConfig
from ainode.engine.backends.nvidia import NvidiaBackend


def _args(tp, **cfg):
    c = NodeConfig(node_id="t", node_name="T", **cfg)
    return NvidiaBackend(c)._build_vllm_serve_args(tp_size=tp)


def test_fp8_kv_and_tp_land():
    a = _args(4)
    assert "--kv-cache-dtype" in a and a[a.index("--kv-cache-dtype") + 1] == "fp8"
    assert "--tensor-parallel-size" in a and a[a.index("--tensor-parallel-size") + 1] == "4"


def test_enforce_eager_always_on():
    assert "--enforce-eager" in _args(1)


def test_kv_dtype_escape_hatch():
    assert "--kv-cache-dtype" not in _args(1, kv_cache_dtype="")


def _name_args(**cfg):
    # an off-disk model -> remote path, no filesystem dependency
    c = NodeConfig(node_id="t", node_name="T", model="some-org/not-on-disk", **cfg)
    _target, args = NvidiaBackend(c)._serve_target_and_name_args()
    return args


def test_served_name_defaults_to_model():
    # no alias -> pin the repo-id so /v1/models is stable
    assert _name_args() == ["--served-model-name", "some-org/not-on-disk"]


def test_served_name_aliases_emitted():
    # custom aliases -> client can address the model by a short name; multiple supported
    args = _name_args(served_model_name=["Aegis-14B", "Aegis-7B"])
    assert args == ["--served-model-name", "Aegis-14B", "Aegis-7B"]


if __name__ == "__main__":
    test_fp8_kv_and_tp_land()
    test_enforce_eager_always_on()
    test_kv_dtype_escape_hatch()
    test_served_name_defaults_to_model()
    test_served_name_aliases_emitted()
    print("ok")
