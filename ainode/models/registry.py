"""Model registry and manager: dynamic catalog + download/delete/recommend.

The catalog is now assembled dynamically from live sources (HuggingFace Hub,
Ollama library, NVIDIA NIM) with a 24-hour on-disk cache and a small static
fallback for offline/error situations.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Bytes-per-element for the dtypes HF reports in safetensors metadata. Lets us
# compute real download size for quantized models (NVFP4 weights land as U8).
_DTYPE_BYTES = {
    "F64": 8, "I64": 8, "U64": 8,
    "F32": 4, "I32": 4, "U32": 4,
    "BF16": 2, "F16": 2, "I16": 2, "U16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
    "F4": 0.5, "FP4": 0.5,
}


def _safetensors_size_gb(safetensors) -> float:
    """Real on-disk size (decimal GB) from HF safetensors dtype breakdown."""
    params = getattr(safetensors, "parameters", None)
    if not params:
        return 0.0
    total_bytes = sum(_DTYPE_BYTES.get(dt, 2) * n for dt, n in params.items())
    return round(total_bytes / 1e9, 1)


def _download_max_workers() -> int:
    """Parallel-connection cap for model downloads (AINODE_DOWNLOAD_MAX_WORKERS,
    default 4). Keeps a fat HF pull from saturating the uplink."""
    try:
        return max(1, int(os.environ.get("AINODE_DOWNLOAD_MAX_WORKERS", "4")))
    except (TypeError, ValueError):
        return 4

from ainode.core.config import AINODE_HOME, HF_CACHE_MOUNT, MODELS_DIR  # noqa: E402


def find_model_dir(models_dir, hf_repo: str) -> Optional[Path]:
    """Return the on-disk dir holding ``hf_repo``, across every layout we support.

    A model can live as: direct ``org--name`` (our downloader), flat HF
    ``models--org--name``, HF cache ``hub/models--org--name``, or out-of-band
    ``hf-cache/hub/models--org--name`` (HF_HOME downloads). Every caller must
    detect all of them, otherwise an on-disk model reads as "not downloaded" -
    which is why the list lives here and not at the call sites (the registry's
    catalog view, the list_available scan, and ``ainode doctor`` all ask this).
    """
    models_dir = Path(models_dir)
    hf_slug = "models--" + hf_repo.replace("/", "--")
    candidates = [
        models_dir / hf_repo.replace("/", "--"),  # org--name
        models_dir / hf_slug,
        models_dir / "hub" / hf_slug,
        models_dir / "hf-cache" / "hub" / hf_slug,
    ]
    for candidate in candidates:
        try:
            if candidate.exists() and any(candidate.iterdir()):
                return candidate
        except OSError:
            continue
    return None


@dataclass
class ModelInfo:
    """Metadata for a model in the catalog."""

    id: str
    name: str
    hf_repo: str
    size_gb: float
    description: str
    quantization: Optional[str] = None
    min_memory_gb: float = 0.0
    family: str = ""
    params_b: float = 0.0
    # Active parameters per token, in billions, and which shape the model is:
    # "moe" or "dense". A MoE reads only its active experts per token, which on
    # GB10 (bandwidth-bound decode) is the number that predicts speed, so a bench
    # record carries both. None / "" mean the entry does not state it, and a
    # reader falls back to the A<n>B marker in the model id
    # (ainode/bench/fleet.py::derive_arch). Not to be confused with
    # ``architecture`` below, which is the HF class name (LlamaForCausalLM).
    active_params_b: Optional[float] = None
    arch: str = ""
    context_length: int = 0
    license: str = ""
    recommended: bool = False
    # Cluster-proven config: proven_tp = node count to launch at; verified = we've
    # actually served it on this hardware (drives the picker default + a ✓ badge).
    proven_tp: int = 0
    verified: bool = False
    # Provenance for that flag. ``verified=True`` on its own says nothing a reader
    # can check, so it comes with the date it was proven (ISO YYYY-MM-DD) and the
    # bench record that proves it (a filename under bench/results/). Both fields
    # are REQUIRED with the flag now (issue #201): six entries carried it from
    # before the bench existed with nothing behind it, and "verified" is what a
    # user reads to decide whether a recipe will come up on their hardware, so an
    # entry nobody can check is worse than an honest unverified one. The
    # provenance test asserts over the whole catalog in both directions.
    verified_on: str = ""
    verified_record: str = ""
    # Roughly how long this model takes to reach READY on this hardware, in
    # minutes, from launches we actually timed. Not a measurement of one run and
    # not a promise: the number a user needs before they click LAUNCH and wait.
    # None on an entry nobody has launched. A node's own launch-times ledger
    # (<AINODE_HOME>/launch-times.json) beats this seed wherever it has an entry.
    typical_ready_minutes: Optional[float] = None
    # True for our hand-picked CURATED_CLUSTER_MODELS. Drives the "Catalog"
    # (known-good to grab) list, separate from on-disk / HF-sweep entries.
    curated: bool = False
    created_at: str = ""
    downloads: int = 0
    likes: int = 0
    # Capabilities, inferred from HF tags or model ID, except "embedding" and
    # "speech", which are only ever stated by a curated entry: neither is a feature
    # added on top of chat but the whole of what the model does, and the interface
    # reads them to draw an Embedding or Speech chip instead of chat controls and to
    # keep the entry out of the chat model picker.
    capabilities: list = None  # ["vision", "tool_use", "reasoning", "code",
    #                             "multilingual", "embedding", "speech"]
    architecture: str = ""
    format: str = ""  # "safetensors", "gguf", "awq", etc.
    # ---- Launch recipe (proven config, applied automatically on load) --------
    # Some models only serve correctly with a specific engine build and flag set
    # (speculative decoding, MoE/mamba backends, reasoning + tool-call parsers).
    # Carrying that here is what makes them a one-click catalog load instead of a
    # hand-rolled container. A caller's explicit /api/models/load value always
    # wins over the recipe; the recipe only fills what wasn't specified.
    engine_image: str = ""          # "" = fleet default engine image
    extra_vllm_args: list = None    # verbatim `vllm serve` flags
    extra_env: dict = None          # engine-container env (e.g. b12x kernel selection)
    extra_volumes: list = None      # extra docker mounts, "host:container[:ro]"
    recommended_gmu: float = 0.0    # 0 = use node default gpu_memory_utilization
    # Distributed shape this model's engine image can actually run:
    #   "mp":  one `vllm serve` container per node (vLLM's own multi-node
    #          executor). Needs nothing but vLLM, and is the node default.
    #   "ray": a ray head + ray workers; needs the `ray` CLI inside the image.
    # Empty means the entry does not state one, in which case the node default
    # applies (DEFAULT_DISTRIBUTED_EXECUTOR). It defaulted to "ray" here, which
    # was harmless only because catalog_recipe() then dropped "ray" as if it had
    # never been stated: an entry that genuinely needed ray could not ask for it.
    distributed_executor: str = ""
    # Serve values that are part of the proven recipe rather than a user
    # preference. Empty / 0 / False mean "not stated by the recipe", in which
    # case the node default applies. A caller's explicit load value still wins.
    kv_cache_dtype: str = ""
    max_model_len: int = 0
    trust_remote_code: bool = False

    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.extra_vllm_args is None:
            self.extra_vllm_args = []
        if self.extra_env is None:
            self.extra_env = {}
        if self.extra_volumes is None:
            self.extra_volumes = []

    def to_dict(self) -> dict:
        return asdict(self)


# ---- Fallback catalog ------------------------------------------------------
#
# Used when all live sources fail (offline, rate-limited, etc.). Kept small.

FALLBACK_CATALOG: dict[str, ModelInfo] = {
    "llama-3.2-3b": ModelInfo(
        id="llama-3.2-3b",
        name="Llama 3.2 3B Instruct",
        hf_repo="meta-llama/Llama-3.2-3B-Instruct",
        size_gb=6.0,
        description="Compact, fast model for everyday tasks. Great starter model.",
        min_memory_gb=8,
        family="llama",
        params_b=3.21,
        arch="dense",
        context_length=131072,
        license="Llama 3.2",
        recommended=True,
    ),
    "qwen-2.5-7b": ModelInfo(
        id="qwen-2.5-7b",
        name="Qwen 2.5 7B Instruct",
        hf_repo="Qwen/Qwen2.5-7B-Instruct",
        size_gb=15.0,
        description="Strong 7B with excellent multilingual and reasoning capability.",
        min_memory_gb=16,
        family="qwen",
        params_b=7.62,
        arch="dense",
        context_length=131072,
        license="Qwen",
        recommended=True,
    ),
    "mistral-7b": ModelInfo(
        id="mistral-7b",
        name="Mistral 7B Instruct v0.3",
        hf_repo="mistralai/Mistral-7B-Instruct-v0.3",
        size_gb=14.0,
        description="Fast, efficient 7B with strong instruction following.",
        min_memory_gb=16,
        family="mistral",
        params_b=7.25,
        arch="dense",
        context_length=32768,
        license="Apache 2.0",
        recommended=True,
    ),
    "phi-3-mini": ModelInfo(
        id="phi-3-mini",
        name="Phi-3 Mini 4K Instruct",
        hf_repo="microsoft/Phi-3-mini-4k-instruct",
        size_gb=7.5,
        description="Microsoft's compact model. Strong reasoning for its size.",
        min_memory_gb=8,
        family="phi",
        params_b=3.82,
        arch="dense",
        context_length=4096,
        license="MIT",
        recommended=True,
    ),
    "gemma-2-9b": ModelInfo(
        id="gemma-2-9b",
        name="Gemma 2 9B IT",
        hf_repo="google/gemma-2-9b-it",
        size_gb=18.5,
        description="Google Gemma 2 9B. Strong mid-size open model.",
        min_memory_gb=20,
        family="gemma",
        params_b=9.24,
        arch="dense",
        context_length=8192,
        license="Gemma",
        recommended=True,
    ),
}


# ---- Curated cluster models (always discoverable) --------------------------
#
# The live HF sweep (top-downloads) misses the frontier/NVFP4 models this GB10
# cluster actually runs, so they were undiscoverable in the catalog and only
# appeared once already on disk. These curated entries are ALWAYS merged into
# the catalog (see ModelManager.get_catalog) so an operator can find + download
# them. NVFP4 is native on Blackwell; these run distributed (TP=N) across nodes.

# Compiled-kernel cache root for recipes whose engine image JITs kernels on first
# launch. Inside the mounted HF cache on purpose: it is the one directory AINode
# already mounts on every node, so nothing here depends on a host path.
_DSPARK_JIT_ROOT = f"{HF_CACHE_MOUNT}/.vllm-jit"

CURATED_CLUSTER_MODELS: dict[str, ModelInfo] = {
    # --- Recipe-carrying models (need a newer engine + model-specific flags) ---
    # Both were validated end-to-end on the GB10 fleet 2026-08-13/15; the flag
    # sets below are the vendor/community recipes verbatim. They require vLLM
    # 0.27.1, hence engine_image. Do NOT add --enforce-eager: it's a 0.17-era
    # workaround and only costs throughput here (see nvidia.py module header).
    "ornith-1.5-35b-a3b-nvfp4": ModelInfo(
        id="ornith-1.5-35b-a3b-nvfp4",
        name="Ornith 1.5 35B-A3B (NVFP4)",
        hf_repo="ornith-ai/Ornith-1.5-35B-A3B-NVFP4",
        size_gb=23.5,
        description=(
            "Qwen3.5-MoE coding/agentic reasoner (3B active/token) with built-in MTP "
            "speculative decoding: 40 tok/s single-stream and 269 tok/s across 16 "
            "streams on one GB10, decode holding 34 tok/s at 120K context. 19/19 on the "
            "fresh-agent rubric (tools, parallel tool calls, executed code, needle at "
            "100K). 262K context, MIT. Launched text-only: the vision tower's warmup "
            "OOM-kills the engine on a node without ~40 GB free."
        ),
        quantization="NVFP4", min_memory_gb=32, family="ornith", params_b=35.0,
        active_params_b=3.0, arch="moe",
        proven_tp=1, verified=True, curated=True,
        verified_on="2026-09-13",
        verified_record="20260913-130400-ornith-1_5-35b-a3b-nvfp4-text-only-mtp.json",
        # Stacked on Spark-1 beside the 27B, timed 2026-09-16.
        typical_ready_minutes=12.0,
        context_length=262144, license="MIT", recommended=True,
        format="safetensors",
        capabilities=["tool_use", "reasoning", "code"],
        engine_image="vllm/vllm-openai:v0.27.1",
        extra_vllm_args=[
            "--enable-prefix-caching",
            # Qwen3.5 arch carries a vision tower; the checkpoint bakes calibrated fp8
            # KV scales, which vLLM applies regardless, stated explicitly so the
            # served-from-HF-cache path never guesses.
            "--kv-cache-dtype", "auto",
            "--reasoning-parser", "qwen3",
            # Template emits <tool_call><function=..><parameter=..>, so qwen3_coder on
            # 0.27.1 (the card's qwen3_xml is the newer name for the same syntax).
            "--tool-call-parser", "qwen3_coder",
            "--enable-auto-tool-choice",
            # Text-only: skips the multimodal warmup that OOM-killed the EngineCore
            # when stacked beside Qwen3.8 on Spark-1 (2026-09-13). Drop these two
            # args to serve vision on a node with headroom.
            "--limit-mm-per-prompt", '{"image":0,"video":0}',
            "--speculative_config", '{"method":"qwen3_5_mtp","num_speculative_tokens":2}',
        ],
        # 23.5 GB weights + a 600K-token KV cache fit in 0.26 when stacked; 0.35
        # leaves room to stack it beside a second model on a 122 GB node.
        recommended_gmu=0.35,
    ),
    "nemotron-3.5-lightning-nvfp4": ModelInfo(
        id="nemotron-3.5-lightning-nvfp4",
        name="Nemotron 3.5 Lightning 30B-A3B (NVFP4)",
        hf_repo="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        size_gb=21.0,
        description=(
            "MoE hybrid Mamba-2 (3B active/token) with DSpark speculative decoding: "
            "104 tok/s single-stream and 504 tok/s across 16 streams on one GB10, the "
            "fastest model on this hardware. 1M context. The sub-agent workhorse. "
            "Text only (no vision). First launch also pulls the 1.3 GB DSpark drafter."
        ),
        quantization="NVFP4", min_memory_gb=30, family="nemotron", params_b=30.0,
        active_params_b=3.0, arch="moe",
        proven_tp=1, verified=True, curated=True,
        verified_on="2026-09-13",
        verified_record=(
            "20260913-141115-nvidia-nemotron-3_5-lightning-30b-a3b-nvfp4-smoke.json"),
        # Solo on Spark-4 (GX10), timed 2026-09-13.
        typical_ready_minutes=10.0,
        context_length=1048576, license="OpenMDW-1.1", recommended=True,
        format="safetensors", capabilities=["tool_use", "reasoning", "code"],
        engine_image="vllm/vllm-openai:v0.27.1",
        extra_vllm_args=[
            "--moe-backend", "marlin",
            "--enable-prefix-caching",
            "--speculative_config.model",
            "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark",
            "--speculative_config.num_speculative_tokens", "3",
            "--mamba-backend", "flashinfer",
            "--mamba-cache-mode", "align",
            "--reasoning-parser", "nemotron_v3",
            "--tool-call-parser", "qwen3_coder",
            "--enable-auto-tool-choice",
        ],
        recommended_gmu=0.91,
    ),
    "qwen3.8-27b-nvfp4": ModelInfo(
        id="qwen3.8-27b-nvfp4",
        name="Qwen3.8 27B (NVFP4, vision)",
        hf_repo="unsloth/Qwen3.8-27B-NVFP4",
        size_gb=23.4,
        description=(
            "Dense 27B native vision-language model (images + video) with built-in MTP "
            "speculative decoding: 19 tok/s single-stream on one GB10 (dense is "
            "bandwidth-bound; batching reaches 147 tok/s at 16 streams). 262K context, "
            "excellent instruction-following and tool use. The quality-and-eyes model. "
            "Use temperature 0 for OCR/transcription."
        ),
        quantization="NVFP4", min_memory_gb=32, family="qwen", params_b=27.0,
        arch="dense",
        proven_tp=1, verified=True, curated=True,
        verified_on="2026-08-15",
        verified_record="20260815-000000-qwen3_8-27b-nvfp4-mtp-vision.json",
        # Solo on a GB10, timed 2026-09-13: 3.5 min of weights, then torch.compile
        # and the FlashInfer fp4_gemm autotune pass, which is the rest of it.
        typical_ready_minutes=12.0,
        context_length=262144, license="Apache 2.0", recommended=True,
        format="safetensors",
        capabilities=["vision", "tool_use", "reasoning", "code", "multilingual"],
        engine_image="vllm/vllm-openai:v0.27.1",
        extra_vllm_args=[
            "--enable-prefix-caching",
            # Vision models must NOT get fp8 KV on GB10: it corrupts generation
            # (proven 2026-07-06). The automatic fp8→auto downgrade only fires
            # when the model is on local disk (it reads config.json), and this
            # one serves straight from the HF cache, so state it explicitly.
            "--kv-cache-dtype", "auto",
            "--reasoning-parser", "qwen3",
            # REQUIRED: the template emits <tool_call><function=..><parameter=..>.
            # With the hermes parser, tool calls silently never parse (0 emitted).
            "--tool-call-parser", "qwen3_coder",
            "--enable-auto-tool-choice",
            "--speculative_config", '{"method":"qwen3_5_mtp","num_speculative_tokens":2}',
        ],
        recommended_gmu=0.60,
    ),
    "spark-x2.5-4b": ModelInfo(
        id="spark-x2.5-4b",
        name="Spark-X2.5 4B (BF16)",
        hf_repo="XHToken/Spark-X2.5-4B",
        size_gb=8.2,
        description=(
            "Compact agentic model (4B, dense, BF16) with a hybrid attention stack: one "
            "full-attention layer for every three sliding-window layers, head-wise "
            "output gating, native 1M-token context. vLLM has no in-tree implementation: "
            "the engine image below is the stock vLLM 0.27.1 image plus the vendor's "
            "out-of-tree plugin (github.com/XHToken/Spark-plugin), which registers the "
            "architecture and the spark25 tool-call parser. Proven on a GB10 stacked "
            "beside Nemotron: ready in 6 min, tool calls parse, thinking switch works. "
            "BF16 is bandwidth-bound at about 18 tok/s single-stream, and thinking is "
            "verbose, so harness tasks run long (dsh 6/10 then 9/10 at 690 s mean; "
            "agentic rubric 19/25 with every tool probe passed, needle only to 8k). "
            "Small enough to stack beside any other model; a quantized build is the "
            "obvious next step."
        ),
        quantization=None, min_memory_gb=14, family="spark", params_b=4.0,
        arch="dense",
        proven_tp=1, verified=True, curated=True,
        verified_on="2026-09-18",
        verified_record=(
            "20260918-143104-spark-x2_5-4b-spark-4-stacked-beside-nemotron-first-run-harness.json"
        ),
        # Stacked on Spark-4 beside Nemotron at 0.70, timed 2026-09-18 (ledger 5.7 min).
        typical_ready_minutes=6.0,
        context_length=1048576, license="Apache-2.0", recommended=False,
        format="safetensors",
        capabilities=["tool_use", "reasoning", "code"],
        # Local build: scripts/Dockerfile.spark25 (vllm/vllm-openai:v0.27.1 + plugin).
        engine_image="ainode-spark25:v0.27.1",
        trust_remote_code=True,
        # 1M is the card's number; 256K keeps the stacked KV budget honest.
        max_model_len=262144,
        extra_vllm_args=[
            "--enable-prefix-caching",
            "--tool-call-parser", "spark25",
            "--enable-auto-tool-choice",
            # The template wraps thinking in <think> tags with an enable_thinking switch,
            # the same scheme the qwen3 parser splits into reasoning_content.
            "--reasoning-parser", "qwen3",
        ],
        # 8 GB of weights plus a long KV cache fit comfortably in 0.12 of a GB10.
        recommended_gmu=0.12,
    ),
    "qwen3.6-35b-a3b-nvfp4-v100": ModelInfo(
        id="qwen3.6-35b-a3b-nvfp4-v100",
        name="Qwen3.6 35B-A3B (NVFP4, V100)",
        hf_repo="nvidia/Qwen3.6-35B-A3B-NVFP4",
        size_gb=23.5,
        description=(
            "Qwen3.5-MoE (35B total, 3B active/token, modelopt NVFP4) on a single "
            "Tesla V100 32 GB, which is the point of this entry: Volta (SM70) was "
            "dropped from mainline vLLM in 0.20, so the engine image below is NOT a "
            "registry image. It is the 1Cat-vLLM fork built locally with SM70 "
            "kernels; Castor holds the exported tarball "
            "(/home/sem/onecat-vllm-src-full.tar.gz) and the build script "
            "(/home/sem/build-onecat-src.sh), and loading that export is the way to "
            "get it onto a node, because building it takes about a day. Its "
            "ENTRYPOINT is [\"vllm\"] with the binary at /opt/venv/bin/vllm. Served "
            "text-only: the checkpoint carries a vision tower and the two "
            "--limit-mm-per-prompt zeros keep it out of the 32 GB budget. Measured on "
            "pollux (Dell C4130, one V100 32 GB) 2026-09-19: ready in 432 s, 30.4 of "
            "32 GB used, 88.7 tok/s single-stream with thinking off at first launch "
            "and 97 tok/s in the bench record below, 344 tok/s across 16 streams, "
            "decode still 66 tok/s at 63K prompt tokens. 20/22 on the quick agentic "
            "rubric with every tool probe, every executed-code probe and all five "
            "agentic probes passed. This is the single-V100 chat lane from the "
            "Titanium Lab plan: the one recipe that puts a frontier-shaped MoE on "
            "Volta hardware at interactive speed."
        ),
        quantization="NVFP4", min_memory_gb=30, family="qwen", params_b=35.0,
        active_params_b=3.0, arch="moe",
        proven_tp=1, verified=True, recommended=False, curated=True,
        verified_on="2026-09-19",
        verified_record=(
            "20260919-040456-qwen3_6-35b-a3b-nvfp4-pollux-v100-solo-onecat-src-full.json"
        ),
        # Solo on pollux (one V100 32 GB), timed 2026-09-19: 432 s to ready.
        typical_ready_minutes=7.5,
        context_length=262144, license="Apache-2.0",
        format="safetensors",
        capabilities=["tool_use", "reasoning", "code"],
        # Local build, not on any registry: the 1Cat-vLLM fork with SM70/Volta
        # kernels (mainline vLLM dropped Volta in 0.20). Load the exported tarball
        # from Castor rather than rebuilding; the build is a day long.
        engine_image="onecat-vllm:src-full",
        # The checkpoint bakes calibrated fp8 KV scales and this is a Volta card
        # with no fp8 path, so state auto rather than inheriting an fp8 default.
        kv_cache_dtype="auto",
        # First launch's setting. 131072 is Castor's TP=4 number and is untested at
        # TP=1, so it is not what this entry promises.
        max_model_len=65536,
        trust_remote_code=True,
        recommended_gmu=0.90,
        extra_vllm_args=[
            # Volta has no FlashAttention-2/3 and no FlashInfer: the fork ships a
            # V100-specific backend and it has to be named, or the engine picks one
            # that will not build on SM70.
            "--attention-backend", "FLASH_ATTN_V100",
            # 32 GB total, so the KV budget is small; 8 concurrent sequences is what
            # fits beside 23.5 GB of weights at 64K.
            "--max-num-seqs", "8",
            "--enable-prefix-caching",
            "--reasoning-parser", "qwen3",
            # Template emits <tool_call><function=..><parameter=..>, same as the
            # Ornith and Qwen3.8 entries, so qwen3_coder is the parser that parses.
            "--tool-call-parser", "qwen3_coder",
            "--enable-auto-tool-choice",
            # Text-only: the vision tower's warmup does not fit in 32 GB beside the
            # weights. Drop these two args on a card with headroom.
            "--limit-mm-per-prompt", '{"image":0,"video":0}',
        ],
    ),
    "qwen3-embedding-0.6b": ModelInfo(
        id="qwen3-embedding-0.6b",
        name="Qwen3 Embedding 0.6B",
        hf_repo="Qwen/Qwen3-Embedding-0.6B",
        size_gb=1.2,
        description=(
            "1024-dimensional embeddings, 32k context, multilingual. Served by "
            "vLLM's pooling runner on the same engine image as everything else, so "
            "it stacks beside any chat model at 6 percent of a GB10 and needs no "
            "second runtime, no second container and no CPU library in the AINode "
            "image. POST /v1/embeddings on ANY node routes to it by model id, the "
            "same way a chat completion routes, so one instance serves the whole "
            "fleet. Measured on Spark-4 2026-09-19 stacked beside Nemotron 3.5 "
            "Lightning: ready in 87 s, 1024 dims, 71.6 ms p50 for a single short "
            "text end to end over a tailnet whose own floor is 32 ms, and 13.9, "
            "95.6 and 225.5 texts/s at batches of 1, 16 and 64 (2938 tokens/s at "
            "64). The quality check passed with room to spare: the closest "
            "unrelated pair scores 0.32 and the weakest related pair 0.82."
        ),
        quantization=None, min_memory_gb=4, family="qwen", params_b=0.6,
        arch="dense",
        proven_tp=1, verified=True, recommended=True, curated=True,
        verified_on="2026-09-19",
        verified_record=(
            "20260919-220011-qwen3-embedding-0_6b-spark-4-stacked-beside-nemotron-embed.json"
        ),
        # Stacked on Spark-4 beside Nemotron at 0.06, timed 2026-09-19 (ledger
        # 86.7 s; the same launch took 77.9 s on Spark-2).
        typical_ready_minutes=1.5,
        context_length=32768, license="Apache-2.0",
        format="safetensors",
        # The one capability that is not a chat capability: the interface reads
        # this to draw an Embedding chip instead of chat controls, and to keep the
        # entry out of the chat model picker.
        capabilities=["embedding"],
        # Fleet default image on purpose: the proven launch ran on the node's own
        # engine image, and `--runner pooling` has been in vLLM since the --task
        # flag was retired, so nothing here needs a pinned newer build.
        # 32k is the card's number; 8192 is what the proven launch served, and a
        # window this entry has not been measured at is not what it promises.
        max_model_len=8192,
        extra_vllm_args=[
            # Pooling, not generate: this checkpoint has no LM head to sample
            # from, and the default runner refuses to serve it.
            "--runner", "pooling",
            # 64 texts in one request is the batch the throughput number is taken
            # at, so the engine has to accept that many sequences at once.
            "--max-num-seqs", "64",
            "--enable-prefix-caching",
        ],
        # 1.2 GB of weights and no KV cache worth the name: 6 percent of a GB10 is
        # enough, which is what makes this a model you leave running.
        recommended_gmu=0.06,
    ),
    "whisper-large-v3-turbo": ModelInfo(
        id="whisper-large-v3-turbo",
        name="Whisper Large v3 Turbo",
        hf_repo="openai/whisper-large-v3-turbo",
        size_gb=1.6,
        description=(
            "Speech to text: 809M parameters, 99 languages, transcription only. "
            "Served by vLLM on POST /v1/audio/transcriptions, which AINode routes "
            "across the fleet the way it routes a chat completion, except that the "
            "model id arrives as a multipart form field beside the audio file "
            "instead of in a JSON body. 1.6 GB of weights and a 448-token decoder "
            "window, so at 6 percent of a GB10 it stacks beside a chat model and "
            "one instance serves every node. Turbo is a transcription model and "
            "cannot translate: vLLM's own note says so, and /v1/audio/translations "
            "is proxied for the ASR models that can. Needs an engine image with "
            "vLLM's audio extras (librosa, soundfile): no image on the fleet ships "
            "them, and vLLM imports soundfile at module scope as soon as the served "
            "model reports the transcription task, so the stock image dies at "
            "startup before it binds a port. The image below is the stock GB10 "
            "build plus those two libraries, built from scripts/Dockerfile.whisper "
            "on the node that serves it; publishing it to a registry is a "
            "follow-up."
        ),
        quantization=None, min_memory_gb=4, family="whisper", params_b=0.81,
        arch="dense",
        # Not verified, and deliberately not dressed up as it: the engine has not
        # served on this hardware yet. Both nodes with room to stack were full
        # when it was tried (2026-09-19: CUDA out of memory at context creation on
        # a node with 3 GB free), and a True flip needs a bench record to name,
        # which needs a speech section in the bench. Both are follow-ups.
        proven_tp=1, verified=False, recommended=True, curated=True,
        # 448 is Whisper's decoder window, which is what the engine reports as
        # max_model_len. The 30-second audio chunk is an encoder property and is
        # not a context length.
        context_length=448, license="MIT",
        format="safetensors",
        # Not a chat capability: this model answers the two audio paths and no
        # chat path, so the interface reads it the way it reads "embedding".
        capabilities=["speech"],
        # Local build, not published (same shape as the DSpark entry above). The
        # tag keeps the base image's version so it is obvious which build it
        # derives from.
        engine_image="ainode-whisper:0.17.0-t5",
        # A recipe dtype is explicit, so this is the recipe saying auto rather
        # than inheriting a node's fp8 default: fp8 KV buys nothing across a
        # 448-token window and is not a combination anyone has proven on an
        # encoder-decoder model here.
        kv_cache_dtype="auto",
        extra_vllm_args=[
            # Pinning a non-default engine image turns OFF the backend's
            # automatic GB10 workaround (nvidia.py::_legacy_gb10_args), and this
            # image is the 0.17 build that workaround exists for, so the recipe
            # states it: FlashInfer's prefill kernel illegal-instructions under
            # CUDA-graph capture on GB10.
            "--enforce-eager",
        ],
        # 1.6 GB of weights and a tiny KV cache: 6 percent of a GB10 is enough,
        # which is what makes this a model you leave running beside a chat model.
        recommended_gmu=0.06,
    ),
    "deepseek-v4-flash-dspark": ModelInfo(
        id="deepseek-v4-flash-dspark",
        name="DeepSeek V4 Flash (DSpark, FP8)",
        hf_repo="fraserprice/DeepSeek-V4-Flash-DSpark",
        size_gb=159.0,
        description=(
            "Frontier MoE (284B total, 13B active/token) with DSpark speculative "
            "decoding and a 1M-token context, served across TWO GB10 nodes. Needs the "
            "GB10 build of vLLM present on every node: the image below is a local "
            "build with sm121 kernels (a registry publish is a follow-up), and stock "
            "vLLM produces nonsense for this model on this hardware. Launches with "
            "vLLM's own multi-node executor (distributed_executor \"mp\"), not Ray: "
            "that image ships no ray."
        ),
        quantization="FP8", min_memory_gb=175, family="deepseek", params_b=284.0,
        active_params_b=13.0, arch="moe",
        proven_tp=2, verified=True,
        verified_on="2026-09-16",
        verified_record="20260916-033415-deepseek-v4-flash-dspark-fp8-tp2-mp.json",
        # TP=2 across two GB10s on the custom DSpark image, timed 2026-09-16.
        typical_ready_minutes=7.0,
        context_length=1048576, license="MIT", recommended=True, curated=True,
        format="safetensors",
        capabilities=["tool_use", "reasoning", "code"],
        # One vllm serve container per node. This image has no ray CLI.
        distributed_executor="mp",
        engine_image="vllm-dspark-runtime:dspark-nvfp4-stage-c",
        kv_cache_dtype="nvfp4_ds_mla",
        max_model_len=1048576,
        trust_remote_code=True,
        recommended_gmu=0.80,
        # Everything the proven two-node command carries beyond what the backend
        # emits itself (host/port, TP, the mp rendezvous flags, kv-cache dtype,
        # max-model-len, gpu-memory-utilization, trust-remote-code).
        extra_vllm_args=[
            "--block-size", "256",
            "--max-num-seqs", "6",
            "--max-num-batched-tokens", "8192",
            "--enable-prefix-caching",
            "--async-scheduling",
            "--enable-chunked-prefill",
            "--speculative-config",
            '{"method":"dspark","num_speculative_tokens":5,'
            '"draft_sample_method":"probabilistic"}',
            "--tokenizer-mode", "deepseek_v4",
            "--tool-call-parser", "deepseek_v4",
            "--enable-auto-tool-choice",
            "--reasoning-parser", "deepseek_v4",
            "--reasoning-config",
            '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>",'
            '"reasoning_end_str":"</think>"}',
            "--default-chat-template-kwargs", '{"thinking":false}',
            "--generation-config", "vllm",
            "--enable-flashinfer-autotune",
        ],
        # This image is not a vllm/vllm-openai image: its ENTRYPOINT is empty, its
        # WORKDIR and HOME are /tmp, and the vllm binary lives at /opt/env/bin/vllm,
        # so PATH and the CUDA locations have to be stated, and HF_HOME has to be
        # pointed at the mounted cache or HF would write to /tmp/.cache/huggingface
        # and re-download 159 GB. The NCCL socket/HCA/host-IP vars are deliberately
        # absent: AINode derives those per node (_build_nccl_env).
        extra_env={
            "PATH": ("/opt/env/bin:/opt/env/nvvm/bin:"
                     "/opt/env/targets/sbsa-linux/nvvm/bin:"
                     "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
            "CUDA_HOME": "/opt/env/targets/sbsa-linux",
            "CUDA_PATH": "/opt/env/targets/sbsa-linux",
            "CUDAToolkit_ROOT": "/opt/env/targets/sbsa-linux",
            "LD_LIBRARY_PATH": "/opt/env/lib:/opt/env/targets/sbsa-linux/lib",
            "HF_HOME": HF_CACHE_MOUNT,
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_XET": "1",
            # Compiled-kernel caches. The raw recipe parked these on a /vllm-cache
            # volume; a catalog entry cannot know a host path, so they live inside
            # the HF cache AINode already mounts on every node. Kernels then
            # persist per node across launches with no fleet-specific mount.
            "VLLM_CACHE_ROOT": _DSPARK_JIT_ROOT,
            "DG_JIT_CACHE_DIR": f"{_DSPARK_JIT_ROOT}/deepgemm",
            "FLASHINFER_WORKSPACE_BASE": f"{_DSPARK_JIT_ROOT}/flashinfer",
            "TILELANG_CACHE_DIR": f"{_DSPARK_JIT_ROOT}/tilelang",
            "TORCHINDUCTOR_CACHE_DIR": f"{_DSPARK_JIT_ROOT}/torchinductor",
            "TRITON_CACHE_DIR": f"{_DSPARK_JIT_ROOT}/triton",
            "TORCH_EXTENSIONS_DIR": f"{_DSPARK_JIT_ROOT}/torch_extensions",
            "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
            "DSPARK_SLOT_CLAMP": "1",
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            "VLLM_TRITON_MLA_SPARSE": "1",
            "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": "256",
            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0",
            "VLLM_SKIP_INIT_MEMORY_CHECK": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "1",
            "VLLM_USE_B12X_MOE": "1",
            "VLLM_USE_B12X_WO_PROJECTION": "1",
            "VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM": "0",
            "VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M": "16",
            "B12X_W4A16_TC_DECODE": "0",
            "VLLM_DSPARK_CONFIDENCE_THRESHOLD": "0.0",
            "VLLM_DSPARK_CONFIDENCE_SCHEDULER": "off",
            "VLLM_DSPARK_LOCAL_ARGMAX": "1",
            "VLLM_DSPARK_REPLICATE_MARKOV_W1": "1",
            "VLLM_DSPARK_FUSED_MARKOV_ARGMAX": "0",
            "VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK": "1",
            "VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT": "0",
            "VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP": "1",
            "VLLM_DSV4_B12X_COMPRESSED_MLA": "0",
            "VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE": "0",
            "VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT": "0",
            "TORCH_CUDA_ARCH_LIST": "12.1a",
            "FLASHINFER_CUDA_ARCH_LIST": "12.1a",
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
            "TILELANG_CLEANUP_TEMP_FILES": "1",
            "DG_JIT_USE_NVRTC": "0",
            "DG_JIT_NVCC_COMPILER": "/opt/env/bin/nvcc",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "NCCL_NET": "IB",
            "NCCL_IB_DISABLE": "0",
            "NCCL_IB_MERGE_NICS": "0",
            "NCCL_IB_GID_INDEX": "3",
            "NCCL_CROSS_NIC": "0",
            "NCCL_IB_ROCE_VERSION_NUM": "2",
        },
    ),
    "qwen3.8-flash-next-nvfp4": ModelInfo(
        id="qwen3.8-flash-next-nvfp4",
        name="Qwen3.8-Flash-Next (NVFP4)",
        hf_repo="nvidia/Qwen3.8-Flash-Next-NVFP4",
        size_gb=133.0,
        description=(
            "Frontier MoE (125B total, 6B active/token) plus a 51B PLE n-gram "
            "embedding and a 4B MTP module, served across TWO GB10 nodes: 133 GB of "
            "weights does not fit one 121 GB node. The strongest coding model in the "
            "Qwen3.8 line, beating the 27B and DeepSeek V4 Flash on three of the four "
            "coding rows in Qwen's own table. Needs a vLLM nightly newer than 2026-09-03 (the 0.29.0 release predates the FP8 PLE fix) on every node: "
            "the Qwen4Exp architecture and the FP8-PLE loader for mixed ModelOpt "
            "checkpoints landed there, and the 0.27/0.28 images do not know it. Mixed "
            "precision (NVFP4 routed experts, FP8 elsewhere, modelopt). MTP "
            "speculative decoding is NOT enabled here: it wants "
            "--enable-expert-parallel, which hangs on this hardware, so it stays a "
            "follow-up. Served end to end on Spark-2 + Spark-3 on 2026-09-16 from an AINode-downloaded copy (36 min to ready, 26.6 tok/s single-stream without MTP)."
        ),
        quantization="NVFP4 (mixed, FP8 PLE)", min_memory_gb=145,
        family="qwen", params_b=125.0,
        active_params_b=6.0, arch="moe",
        proven_tp=2, verified=True, recommended=True, curated=True,
        verified_on="2026-09-16",
        verified_record=(
            "20260916-204216-qwen3_8-flash-next-nvfp4-tp2-mp-nightly.json"),
        # TP=2 across Spark-2 + Spark-3 with autotune off, timed 2026-09-16. The
        # 36 min in the description above was the one launch that ran autotune.
        typical_ready_minutes=11.0,
        context_length=262144, license="Apache 2.0",
        format="safetensors",
        capabilities=["tool_use", "reasoning", "code"],
        # One vllm serve container per node: nothing in the image but vLLM is
        # needed, which is what makes a pinned upstream tag usable as the engine.
        distributed_executor="mp",
        # Pinned nightly: v0.29.0 diverged from main before the FP8 PLE loading fix
        # (vLLM d4d703c, 2026-09-03) and rank 1 dies loading ngram_embedding.weight_scale
        # on it; this nightly (2026-09-16) served the pair. Move to the first release
        # that contains the fix.
        engine_image="vllm/vllm-openai:nightly-af1c01499b289be555c475669ba50a88e96d846e",
        kv_cache_dtype="auto",
        # Qwen4Exp QSA raises "requires a BF16 main KV cache" on fp8 (vLLM 0.29.0,
        # first launch on the Spark pair 2026-09-16), so this entry overrides the
        # GB10 fp8 default with auto.
        max_model_len=262144,
        trust_remote_code=True,
        recommended_gmu=0.85,
        # Everything the model card's serve command carries beyond what the
        # backend emits itself (host/port, TP, the mp rendezvous flags,
        # kv-cache dtype, max-model-len, gpu-memory-utilization,
        # trust-remote-code). Tool calls parse with qwen3_coder on this vLLM
        # line, same as the Ornith and Qwen3.8 27B entries.
        extra_vllm_args=[
            "--quantization", "modelopt",
            "--enable-prefix-caching",
            "--reasoning-parser", "qwen3",
            "--tool-call-parser", "qwen3_coder",
            "--enable-auto-tool-choice",
            # --- Keeping the two ranks in step through kernel warmup (#134) ---
            # FlashInfer autotune is OFF for this entry. It runs per rank inside
            # kernel_warmup and tuned 856 profiles here: 35 min on the head even
            # with cache hits, and on a cold peer a first
            # trtllm::fused_moe::gemm1 profile that sat for 1800 s until gloo
            # timed out and killed the pair. The switch is real, not guessed:
            # KernelConfig.enable_flashinfer_autotune is registered with
            # argparse.BooleanOptionalAction, so --no-<name> is its off form
            # (vLLM af1c014 engine/arg_utils.py), and kernel_warmup.py skips the
            # whole phase when it is False. THE COST: the fused MoE and fp4
            # GEMMs run FlashInfer's heuristic tactic instead of a measured one,
            # so decode is slower than the 26.6 tok/s of the one launch that got
            # through autotune. A launch that finishes beats a faster one that
            # does not.
            "--no-enable-flashinfer-autotune",
            # And a floor under the collectives either way, because warmup is
            # long here with or without autotune. The CPU (gloo) group is what
            # world.barrier() and any per-profile sync use, and PyTorch's default
            # for it is 1800 s -- under the 48 min worst warmup measured on this
            # pair. The device (NCCL) group gets the same floor, where the default
            # is lower still. The tradeoff is accepted knowingly: a genuine hang
            # now takes 90 min to declare, which is the right trade for a launch
            # that was being declared dead while it was still working.
            "--cpu-distributed-timeout-seconds", "5400",
            "--distributed-timeout-seconds", "5400",
        ],
        # A stock vllm/vllm-openai image needs no PATH or HF_HOME surgery, but its
        # compiled-kernel caches default to $HOME INSIDE the container and die
        # with it, so every launch re-JITs and re-tunes from cold on every rank --
        # the condition that makes the ranks drift apart (#134). Park them in the
        # one directory AINode mounts on every node, same as the DeepSeek recipe,
        # so they persist per node and the head's copy can be shipped to a peer
        # before it launches (nvidia.py::_ensure_peer_has_jit_cache).
        extra_env={
            "VLLM_CACHE_ROOT": _DSPARK_JIT_ROOT,
            "FLASHINFER_WORKSPACE_BASE": f"{_DSPARK_JIT_ROOT}/flashinfer",
            "TORCHINDUCTOR_CACHE_DIR": f"{_DSPARK_JIT_ROOT}/torchinductor",
            "TRITON_CACHE_DIR": f"{_DSPARK_JIT_ROOT}/triton",
            "TORCH_EXTENSIONS_DIR": f"{_DSPARK_JIT_ROOT}/torch_extensions",
        },
    ),
    # --- Fast single-node quantized chat models (AWQ-4bit, awq_marlin on GB10) ---
    # The everyday "always-on" tier: fit one node, serve at interactive speed, and
    # stack several per node. proven_tp=1 (no distribution). verified=True is set
    # ONLY after a real completion was observed on the cluster, and it carries
    # verified_on + verified_record with it (root AGENTS.md). The entries below
    # carried the flag from before the bench existed with no record behind it, and
    # descriptions quoting tok/s figures no record contains; both are gone (#201).
    # They are honest unverified entries until somebody runs one and lands a
    # record.
    "qwen3.5-9b-awq": ModelInfo(
        id="qwen3.5-9b-awq",
        name="Qwen3.5 9B (AWQ-4bit)",
        hf_repo="QuantTrio/Qwen3.5-9B-AWQ",
        size_gb=12.0,
        description="Fast dense 9B, AWQ-4bit (awq_marlin). Fits one GB10 with room to stack. Nobody has benchmarked it here, so there is no throughput figure to quote.",
        quantization="AWQ", min_memory_gb=14, family="qwen", params_b=9.0,
        arch="dense",
        # No bench record: unverified until somebody runs it (#201).
        proven_tp=1, verified=False,
        context_length=262144, license="Apache 2.0", recommended=True, format="awq",
    ),
    "qwen3.5-4b-awq": ModelInfo(
        id="qwen3.5-4b-awq",
        name="Qwen3.5 4B (AWQ-4bit)",
        hf_repo="QuantTrio/Qwen3.5-4B-AWQ",
        size_gb=4.0,
        description="Tiny dense 4B, AWQ-4bit. Dense AWQ is dequant-bound on GB10 rather than size-bound, so a small MoE is usually the faster pick. Lowest memory of the tier. No bench record here yet.",
        quantization="AWQ", min_memory_gb=6, family="qwen", params_b=4.0,
        arch="dense",
        # No bench record: unverified until somebody runs it (#201).
        proven_tp=1, verified=False,
        context_length=262144, license="Apache 2.0", format="awq",
    ),
    "qwen3.5-35b-a3b-awq": ModelInfo(
        id="qwen3.5-35b-a3b-awq",
        name="Qwen3.5 35B-A3B MoE (AWQ-4bit)",
        hf_repo="QuantTrio/Qwen3.5-35B-A3B-AWQ",
        size_gb=24.0,
        description="MoE (3B active/token), AWQ-4bit. Reads 3B of weights per token, so decode stays fast at 35B of quality, and it fits one node. No bench record here yet.",
        quantization="AWQ", min_memory_gb=28, family="qwen", params_b=35.0,
        active_params_b=3.0, arch="moe",
        # No bench record: unverified until somebody runs it (#201).
        proven_tp=1, verified=False,
        context_length=262144, license="Apache 2.0", recommended=True, format="awq",
    ),
    "llama-3.1-8b-nvfp4": ModelInfo(
        id="llama-3.1-8b-nvfp4",
        name="Llama 3.1 8B Instruct (NVFP4)",
        hf_repo="nvidia/Llama-3.1-8B-Instruct-NVFP4",
        size_gb=6.0,
        description="Dense 8B, Blackwell-native NVFP4. Dense decode is bandwidth-bound on GB10. Solid general-purpose chat model, light enough to stack. No bench record here yet.",
        quantization="NVFP4", min_memory_gb=8, family="llama", params_b=8.0,
        arch="dense",
        # No bench record: unverified until somebody runs it (#201).
        proven_tp=1, verified=False,
        context_length=131072, license="Llama 3.1", recommended=True, format="nvfp4",
    ),
    # --- Community daily-driver MoE picks (DGX Spark forum + r/LocalLLaMA, 2026) ---
    "nemotron-cascade-2-30b-a3b-nvfp4": ModelInfo(
        id="nemotron-cascade-2-30b-a3b-nvfp4",
        name="Nemotron Cascade 2 30B-A3B (NVFP4)",
        hf_repo="chankhavu/Nemotron-Cascade-2-30B-A3B-NVFP4",
        size_gb=18.0,
        description="NVIDIA's distilled hybrid (mamba plus attention) MoE, 3B active, Blackwell-native NVFP4. Light enough to stack. Nobody has benchmarked this build here, so the Spark forum's figures are theirs and not ours.",
        quantization="NVFP4", min_memory_gb=22, family="nemotron", params_b=30.0,
        active_params_b=3.0, arch="moe",
        # No bench record: unverified until somebody runs it (#201).
        proven_tp=1, verified=False,
        context_length=131072, license="NVIDIA Open Model", recommended=True, format="nvfp4",
    ),
    "minimax-m2.7-awq": ModelInfo(
        id="minimax-m2.7-awq",
        name="MiniMax-M2.7 (AWQ-4bit)",
        hf_repo="demon-zombie/MiniMax-M2.7-AWQ-4bit",
        size_gb=120.0,
        description="The community's top agentic-coding pick, 'Sonnet at home'. Large MoE (A10B active), AWQ-4bit, needs two nodes. fp8 KV recommended. Never launched here, so there is no throughput figure to quote.",
        quantization="AWQ", min_memory_gb=130, family="minimax", params_b=230.0,
        active_params_b=10.0, arch="moe",
        proven_tp=2, verified=False,
        context_length=131072, license="MiniMax", recommended=True, format="awq",
    ),
    "qwen3-235b-a22b-nvfp4": ModelInfo(
        id="qwen3-235b-a22b-nvfp4",
        name="Qwen3-235B-A22B (NVFP4)",
        hf_repo="nvidia/Qwen3-235B-A22B-NVFP4",
        size_gb=250.0,
        description="Frontier MoE (A22B active). Runs distributed TP=4 on the cluster. NVFP4 for GB10.",
        quantization="NVFP4", min_memory_gb=275, family="qwen", params_b=235.0,
        active_params_b=22.0, arch="moe",
        proven_tp=4, verified=True,
        verified_on="2026-06-17",
        verified_record="20260617-000000-qwen3-235b-a22b-nvfp4-tp4-frontier-moe.json",
        context_length=262144, license="Apache 2.0", recommended=True, format="nvfp4",
    ),
    "qwen3.5-397b-a17b-nvfp4": ModelInfo(
        id="qwen3.5-397b-a17b-nvfp4",
        name="Qwen3.5-397B-A17B (NVFP4)",
        hf_repo="nvidia/Qwen3.5-397B-A17B-NVFP4",
        size_gb=468.0,
        description="Frontier MoE (A17B active), the cluster's design point. Distributed TP=4. NVFP4.",
        quantization="NVFP4", min_memory_gb=500, family="qwen", params_b=397.0,
        active_params_b=17.0, arch="moe",
        proven_tp=4, verified=False,
        context_length=262144, license="Apache 2.0", recommended=True, format="nvfp4",
    ),
    "llama-3.1-405b-nvfp4": ModelInfo(
        id="llama-3.1-405b-nvfp4",
        name="Llama 3.1 405B Instruct (NVFP4)",
        hf_repo="nvidia/Llama-3.1-405B-Instruct-NVFP4",
        size_gb=437.0,
        description="Dense 405B, NVFP4. Needs the cluster's pooled memory (TP=4).",
        quantization="NVFP4", min_memory_gb=470, family="llama", params_b=405.0,
        arch="dense",
        proven_tp=4, verified=False,
        context_length=131072, license="Llama 3.1", format="nvfp4",
    ),
    "llama-3.1-405b-awq": ModelInfo(
        id="llama-3.1-405b-awq",
        name="Llama 3.1 405B Instruct (AWQ-INT4)",
        hf_repo="hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4",
        size_gb=408.0,
        description="Dense 405B, AWQ-INT4. Distributed TP=4.",
        quantization="AWQ", min_memory_gb=440, family="llama", params_b=405.0,
        arch="dense",
        proven_tp=4, verified=False,
        context_length=131072, license="Llama 3.1", format="awq",
    ),
    "llama-3.3-70b-nvfp4": ModelInfo(
        id="llama-3.3-70b-nvfp4",
        name="Llama 3.3 70B Instruct (NVFP4)",
        hf_repo="nvidia/Llama-3.3-70B-Instruct-NVFP4",
        size_gb=80.0,
        description="Dense 70B, NVFP4. Fits TP=2. Dense decode is bandwidth-bound on GB10, and nobody has measured this one here.",
        quantization="NVFP4", min_memory_gb=88, family="llama", params_b=70.0,
        arch="dense",
        # No bench record: unverified until somebody runs it (#201).
        proven_tp=2, verified=False,
        context_length=131072, license="Llama 3.3", recommended=True, format="nvfp4",
    ),
    "glm-5.1": ModelInfo(
        id="glm-5.1",
        name="GLM-5.1",
        hf_repo="zai-org/GLM-5.1",
        size_gb=874.0,
        description="Large GLM. Needs the full cluster's pooled memory (TP=4).",
        quantization=None, min_memory_gb=900, family="glm", params_b=0.0,
        # Shape unstated on purpose: this entry has no params_b either, and
        # neither the id nor the card here says how many experts fire per token.
        proven_tp=4, verified=False,
        context_length=131072, license="GLM",
    ),
    "glm-5.2-reap-504b-nvfp4": ModelInfo(
        id="glm-5.2-reap-504b-nvfp4",
        name="GLM-5.2 NVFP4 REAP-504B",
        hf_repo="madeby561/GLM-5.2-NVFP4-REAP-504B",
        size_gb=309.0,
        description="REAP-pruned GLM-5.2 MoE, NVFP4 for GB10. ~309 GB on disk, needs the cluster's pooled memory (TP=4). DeepSeek Sparse Attention. NOT yet load-tested on GB10.",
        quantization="NVFP4", min_memory_gb=360, family="glm", params_b=504.0,
        # MoE per the card; REAP pruning moves the active count, which the id
        # does not state, so active_params_b stays unset rather than guessed.
        arch="moe",
        proven_tp=4, verified=False,
        context_length=131072, license="MIT", recommended=False, format="nvfp4",
    ),
}


# Backward-compat alias: external code may still import MODEL_CATALOG.
MODEL_CATALOG: dict[str, ModelInfo] = FALLBACK_CATALOG


# ---- Dynamic catalog aggregator --------------------------------------------


class CatalogAggregator:
    """Fetch and merge model metadata from HuggingFace, Ollama, NVIDIA NIM."""

    CACHE_TTL = 86400  # 24 hours
    CACHE_FILE = AINODE_HOME / "catalog-cache.json"

    def fetch(self, force_refresh: bool = False) -> list[ModelInfo]:
        """Fetch the merged catalog. Uses cache if fresh, else all sources."""
        if not force_refresh and self._cache_valid():
            cached = self._load_cache()
            if cached:
                return cached

        models: list[ModelInfo] = []
        models.extend(self._fetch_huggingface_popular(limit=100))
        models.extend(self._fetch_ollama_library())
        models.extend(self._fetch_nvidia_nim())

        # Dedupe by hf_repo (case-insensitive)
        seen: set[str] = set()
        unique: list[ModelInfo] = []
        for m in models:
            key = m.hf_repo.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(m)

        if unique:
            self._save_cache(unique)
        return unique

    # -- Source: HuggingFace Hub ---------------------------------------------

    def _fetch_huggingface_popular(self, limit: int = 100) -> list[ModelInfo]:
        """Top text-generation models on HF Hub by downloads."""
        try:
            from huggingface_hub import HfApi
        except ImportError:
            return []

        try:
            api = HfApi()
            queries = [
                {"filter": "text-generation", "sort": "downloads", "limit": 50},
                {"filter": "text-generation", "tags": "instruct", "sort": "downloads", "limit": 30},
                {"filter": "text-generation", "tags": "chat", "sort": "downloads", "limit": 20},
            ]
            results: list[ModelInfo] = []
            seen_ids: set[str] = set()
            for q in queries:
                try:
                    iterator = api.list_models(**q)  # `direction` dropped in hub >=1.x
                except Exception:
                    continue
                for m in iterator:
                    if m.id in seen_ids:
                        continue
                    seen_ids.add(m.id)
                    try:
                        results.append(self._hf_to_model_info(m))
                    except Exception:
                        continue
            return results
        except Exception:
            return []

    def _hf_to_model_info(self, m) -> ModelInfo:
        """Convert a HF ModelInfo-like object to our ModelInfo."""
        size_gb = self._estimate_size_gb(m)
        params_b = self._estimate_params(m)
        family = m.id.split("/")[0].lower() if "/" in m.id else "unknown"
        slug = m.id.replace("/", "--").lower()
        name = m.id.split("/")[-1].replace("-", " ")

        card_data = getattr(m, "cardData", None) or {}
        if not isinstance(card_data, dict):
            card_data = {}

        license_str = ""
        raw_license = card_data.get("license", "")
        if isinstance(raw_license, str):
            license_str = raw_license
        elif isinstance(raw_license, list) and raw_license:
            license_str = str(raw_license[0])

        context_length = 0
        for key in ("context_length", "max_position_embeddings"):
            val = card_data.get(key, 0)
            if isinstance(val, (int, float)) and val > 0:
                context_length = int(val)
                break

        downloads = getattr(m, "downloads", 0) or 0
        likes = getattr(m, "likes", 0) or 0

        # Detect capabilities from tags + ID
        tags_raw = (card_data.get("tags", []) if isinstance(card_data, dict) else []) or []
        if not isinstance(tags_raw, list):
            tags_raw = []
        tags_joined = " ".join(str(t) for t in tags_raw).lower() + " " + m.id.lower()
        capabilities = []
        if any(k in tags_joined for k in ("vision", "multimodal", "image", "vlm", "vl-", "vl ")):
            capabilities.append("vision")
        if any(k in tags_joined for k in ("tool", "function-call", "function_call")):
            capabilities.append("tool_use")
        if any(k in tags_joined for k in ("reasoning", "thinking", "r1", "o1", "cot")):
            capabilities.append("reasoning")
        if any(k in tags_joined for k in ("code", "coder", "codellama")):
            capabilities.append("code")
        if any(k in tags_joined for k in ("multilingual", "translation")):
            capabilities.append("multilingual")

        # Architecture + format
        arch = ""
        for a in ("llama", "qwen", "mistral", "mixtral", "phi", "gemma", "deepseek", "yi", "falcon", "mpt"):
            if a in m.id.lower():
                arch = a
                break
        fmt = ""
        if "gguf" in m.id.lower():
            fmt = "GGUF"
        elif "awq" in m.id.lower():
            fmt = "AWQ"
        elif "gptq" in m.id.lower():
            fmt = "GPTQ"
        else:
            fmt = "SafeTensors"

        # Extract ISO timestamp from createdAt or lastModified
        created_at = ""
        for attr in ("createdAt", "created_at", "lastModified", "last_modified"):
            val = getattr(m, attr, None)
            if val:
                # Handle datetime objects and strings
                if hasattr(val, "isoformat"):
                    created_at = val.isoformat()
                else:
                    created_at = str(val)
                break

        return ModelInfo(
            id=slug,
            name=name,
            hf_repo=m.id,
            size_gb=size_gb,
            description=self._derive_description(m),
            quantization=self._detect_quantization(m.id),
            min_memory_gb=max(size_gb * 1.2, 2.0) if size_gb > 0 else 2.0,
            family=family,
            params_b=params_b,
            context_length=context_length,
            license=license_str,
            recommended=self._is_recommended(m.id, downloads),
            created_at=created_at,
            downloads=downloads,
            likes=likes,
            capabilities=capabilities,
            architecture=arch,
            format=fmt,
        )

    # -- Source: HuggingFace trending ---------------------------------------

    def fetch_trending(self, limit: int = 30) -> list[ModelInfo]:
        """Models trending on HF (high download velocity recently)."""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            # HF's trending signal is exposed as sort="trendingScore"
            models = api.list_models(
                filter="text-generation",
                sort="trendingScore",
                limit=limit,
                direction=-1,
            )
            results: list[ModelInfo] = []
            for m in models:
                try:
                    results.append(self._hf_model_to_info(m))
                except Exception:
                    continue
            return results
        except Exception:
            return []

    # Alias matching task spec naming
    def _hf_model_to_info(self, m) -> ModelInfo:
        return self._hf_to_model_info(m)

    # -- Source: HuggingFace latest (newest releases) -----------------------

    def fetch_latest(self, limit: int = 30) -> list[ModelInfo]:
        """Most recently created text-generation models on HF."""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            models = api.list_models(
                filter="text-generation",
                sort="createdAt",
                limit=limit * 3,  # overfetch because many will lack metadata
                direction=-1,
            )
            results: list[ModelInfo] = []
            for m in models:
                try:
                    info = self._hf_to_model_info(m)
                    # Only keep models with real size/param info or high download count
                    # so we filter out abandoned uploads
                    if info.params_b > 0 or info.downloads > 100:
                        results.append(info)
                    if len(results) >= limit:
                        break
                except Exception:
                    continue
            return results
        except Exception:
            return []

    # -- Source: OpenRouter popular -----------------------------------------

    def fetch_openrouter_popular(self, limit: int = 30) -> list[ModelInfo]:
        """Models ranked by OpenRouter's actual API usage across their network."""
        try:
            import urllib.request
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/models",
                headers={"User-Agent": "AINode/0.1"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            models: list[ModelInfo] = []
            for m in data.get("data", [])[:limit]:
                hf_repo = m.get("id", "")
                # Skip proprietary ones (openai/, anthropic/, google/gemini)
                if hf_repo.startswith(("openai/", "anthropic/", "google/gemini", "cohere/", "perplexity/")):
                    continue
                context_length = m.get("context_length", 0)
                name = m.get("name", hf_repo)
                slug = hf_repo.replace("/", "--").lower()
                family = hf_repo.split("/")[0].lower() if "/" in hf_repo else ""
                params_b = self._estimate_params_from_name(name)
                size_gb = params_b * 2 if params_b else 0
                models.append(ModelInfo(
                    id=slug,
                    name=name,
                    hf_repo=hf_repo,
                    size_gb=size_gb,
                    description=m.get("description", "OpenRouter-ranked model") or "Text generation model",
                    quantization=None,
                    min_memory_gb=max(size_gb * 1.2, 2.0),
                    family=family,
                    params_b=params_b,
                    context_length=context_length,
                    license="",
                    recommended=True,
                ))
            return models
        except Exception:
            return []

    def _estimate_params_from_name(self, name: str) -> float:
        match = re.search(r'(\d+(?:\.\d+)?)\s*[Bb]', name)
        if match:
            return float(match.group(1))
        match = re.search(r'(\d+)\s*[Mm](?![a-zA-Z])', name)
        if match:
            return float(match.group(1)) / 1000
        return 0.0

    # -- Source: Ollama library (live) --------------------------------------

    def fetch_ollama_library(self, limit: int = 30) -> list[ModelInfo]:
        """Ollama's curated library -- scrape their public library page."""
        try:
            import urllib.request
            req = urllib.request.Request(
                "https://ollama.com/api/library",
                headers={"User-Agent": "AINode/0.1", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read().decode()
            try:
                data = json.loads(content)
            except Exception:
                return []
            models: list[ModelInfo] = []
            for item in (data if isinstance(data, list) else [])[:limit]:
                if not isinstance(item, dict):
                    continue
                name = item.get("name", "")
                if not name:
                    continue
                models.append(ModelInfo(
                    id=f"ollama-{name}".lower(),
                    name=name,
                    hf_repo=name,
                    size_gb=0,
                    description=item.get("description", "Ollama library model"),
                    family=name.split(":")[0].lower() if ":" in name else name.lower(),
                    params_b=0,
                    context_length=0,
                    license="",
                    recommended=True,
                ))
            return models
        except Exception:
            return []

    # -- Source: Ollama library ----------------------------------------------

    def _fetch_ollama_library(self) -> list[ModelInfo]:
        """Ollama's curated set. They don't publish a JSON catalog, so we return
        a small hand-curated list that maps Ollama tags to HF repos. The
        aggregator dedupes against HF results by hf_repo, so duplicates are OK.
        """
        try:
            known = [
                ("llama3.2:3b", "meta-llama/Llama-3.2-3B-Instruct", 3.21, 6.0, "llama"),
                ("llama3.1:8b", "meta-llama/Llama-3.1-8B-Instruct", 8.03, 16.0, "llama"),
                ("qwen2.5:7b", "Qwen/Qwen2.5-7B-Instruct", 7.62, 15.0, "qwen"),
                ("mistral:7b", "mistralai/Mistral-7B-Instruct-v0.3", 7.25, 14.0, "mistral"),
                ("gemma2:9b", "google/gemma-2-9b-it", 9.24, 18.5, "gemma"),
                ("phi3:mini", "microsoft/Phi-3-mini-4k-instruct", 3.82, 7.5, "phi"),
                ("codellama:7b", "codellama/CodeLlama-7b-Instruct-hf", 6.74, 13.5, "llama"),
                ("deepseek-r1:7b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 7.0, 14.0, "deepseek"),
            ]
            results: list[ModelInfo] = []
            for tag, repo, params_b, size_gb, family in known:
                slug = repo.replace("/", "--").lower()
                results.append(ModelInfo(
                    id=slug,
                    name=repo.split("/")[-1].replace("-", " "),
                    hf_repo=repo,
                    size_gb=size_gb,
                    description=f"Available via Ollama tag '{tag}'.",
                    quantization=self._detect_quantization(repo),
                    min_memory_gb=max(size_gb * 1.2, 2.0),
                    family=family,
                    params_b=params_b,
                    context_length=0,
                    license="",
                    recommended=True,
                ))
            return results
        except Exception:
            return []

    # -- Source: NVIDIA NIM --------------------------------------------------

    def _fetch_nvidia_nim(self) -> list[ModelInfo]:
        """NVIDIA NIM catalog. Public JSON API requires auth, so we return an
        empty list unless we can successfully hit a public endpoint.
        """
        try:
            # Placeholder: NVIDIA's build.nvidia.com catalog requires auth for
            # programmatic access. Return empty to avoid spurious failures.
            return []
        except Exception:
            return []

    # -- Parsing / heuristic helpers -----------------------------------------

    def _estimate_size_gb(self, model) -> float:
        """Estimate on-disk size in GB from safetensors metadata or model id."""
        safetensors = getattr(model, "safetensors", None)
        if safetensors and isinstance(safetensors, dict):
            total = safetensors.get("total", 0)
            if total and total > 0:
                # assume bf16 = 2 bytes/param as a rough disk size
                return round((total * 2) / (1024 ** 3), 1)

        match = re.search(r'(\d+(?:\.\d+)?)\s*[Bb](?![a-zA-Z])', model.id)
        if match:
            params_b = float(match.group(1))
            if re.search(r'awq|gptq|int4|4bit|4-bit', model.id, re.IGNORECASE):
                return round(params_b * 0.6, 1)
            if re.search(r'int8|8bit|8-bit|fp8', model.id, re.IGNORECASE):
                return round(params_b * 1.1, 1)
            return round(params_b * 2, 1)
        return 0.0

    def _estimate_params(self, model) -> float:
        match = re.search(r'(\d+(?:\.\d+)?)\s*[Bb](?![a-zA-Z])', model.id)
        if match:
            return float(match.group(1))
        match = re.search(r'(\d+)\s*[Mm](?![a-zA-Z])', model.id)
        if match:
            return float(match.group(1)) / 1000
        return 0.0

    def _detect_quantization(self, model_id: str) -> Optional[str]:
        if re.search(r'awq', model_id, re.IGNORECASE):
            return "awq"
        if re.search(r'gptq', model_id, re.IGNORECASE):
            return "gptq"
        if re.search(r'fp8', model_id, re.IGNORECASE):
            return "fp8"
        if re.search(r'int4|4bit|4-bit', model_id, re.IGNORECASE):
            return "int4"
        if re.search(r'int8|8bit|8-bit', model_id, re.IGNORECASE):
            return "int8"
        if re.search(r'gguf', model_id, re.IGNORECASE):
            return "gguf"
        return None

    def _is_recommended(self, model_id: str, downloads: int) -> bool:
        prefixes = [
            "meta-llama/Llama-3",
            "Qwen/Qwen2.5",
            "Qwen/Qwen3",
            "mistralai/Mistral",
            "google/gemma",
            "microsoft/Phi",
            "microsoft/phi",
            "deepseek-ai/DeepSeek-R1",
        ]
        if not any(model_id.startswith(p) for p in prefixes):
            return False
        if downloads and downloads < 100_000:
            return False
        lower = model_id.lower()
        return ("instruct" in lower) or ("chat" in lower) or lower.endswith("-it")

    def _derive_description(self, model) -> str:
        card = getattr(model, "cardData", None) or {}
        if not isinstance(card, dict):
            card = {}
        tags = card.get("tags", []) or []
        if isinstance(tags, str):
            tags = [tags]
        joined_tags = " ".join(str(t).lower() for t in tags)

        pieces: list[str] = []
        if "chat" in joined_tags or "conversational" in joined_tags:
            pieces.append("Conversational model")
        elif "code" in joined_tags:
            pieces.append("Code generation model")
        else:
            pieces.append("Text generation model")

        lang = card.get("language", [])
        if isinstance(lang, list) and lang and "en" not in lang:
            pieces.append(f"Languages: {', '.join(str(x) for x in lang[:3])}")
        return " · ".join(pieces)

    # -- Cache management ----------------------------------------------------

    def _cache_valid(self) -> bool:
        if not self.CACHE_FILE.exists():
            return False
        try:
            age = time.time() - self.CACHE_FILE.stat().st_mtime
            return age < self.CACHE_TTL
        except Exception:
            return False

    def _load_cache(self) -> list[ModelInfo]:
        try:
            data = json.loads(self.CACHE_FILE.read_text())
            return [ModelInfo(**m) for m in data]
        except Exception:
            return []

    def _save_cache(self, models: list[ModelInfo]) -> None:
        try:
            self.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.CACHE_FILE.write_text(
                json.dumps([asdict(m) for m in models], indent=2)
            )
        except Exception:
            pass


# ---- What is on disk, for a model id ---------------------------------------
#
# The Server view reported `size_bytes: 0` and `quantization: null` for every
# loaded model (#180) because nothing resolved a served model id back to its
# weights. These do: one snapshot directory, one measured size, one quantization
# with the source it came from. Cached, because the Server view polls and the
# snapshot of a frontier MoE is tens of thousands of files.

_DISK_CACHE_TTL = 300.0
_disk_size_cache: dict[str, tuple[float, Optional[int]]] = {}
_quantization_cache: dict[str, tuple[float, tuple[Optional[str], Optional[str]]]] = {}


def _hf_cache_roots(models_dir: Optional[Path] = None) -> list[Path]:
    """Every directory a model's weights can be under on this node."""
    base = Path(models_dir) if models_dir else MODELS_DIR
    roots = [base, base / "hub", base / "hf-cache" / "hub"]
    env_home = os.environ.get("HF_HOME")
    if env_home:
        roots.append(Path(env_home) / "hub")
    env_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env_cache:
        roots.append(Path(env_cache))
    return roots


def snapshot_dir_for(model_id: str, models_dir: Optional[Path] = None) -> Optional[Path]:
    """The directory holding a model's weight files, across every layout we write.

    HF cache (``models--org--name/snapshots/<rev>/``), flat HF cache, and the
    direct ``org--name/`` our own downloader writes. When a cache entry has
    several revisions the newest one is the answer: that is the one an engine
    launched today resolved.
    """
    repo = (model_id or "").strip()
    if not repo:
        return None
    slug = "models--" + repo.replace("/", "--")
    direct = repo.replace("/", "--")
    for root in _hf_cache_roots(models_dir):
        for name in (slug, direct, repo):
            candidate = root / name
            try:
                if not candidate.is_dir():
                    continue
            except OSError:
                continue
            snapshots = candidate / "snapshots"
            if snapshots.is_dir():
                revisions = [d for d in snapshots.iterdir() if d.is_dir()]
                if not revisions:
                    continue
                return max(revisions, key=lambda d: d.stat().st_mtime)
            return candidate
    return None


def _measure_dir_bytes(path: Path) -> int:
    """Bytes on disk under *path*, counting a shared blob once.

    An HF snapshot is a tree of symlinks into ``blobs/``, so following each link
    and adding its target's size double-counts anything linked twice. Dedupe on
    the resolved path: one stat per blob, not one per link.
    """
    seen: set = set()
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_dir():
                continue
            target = entry.resolve()
            if target in seen:
                continue
            seen.add(target)
            total += target.stat().st_size
        except OSError:
            continue
    return total


def model_disk_size_bytes(model_id: str, models_dir: Optional[Path] = None) -> Optional[int]:
    """Measured size of a model's weights on this node, or None if not found.

    None rather than 0: a model whose snapshot is not on this disk has an unknown
    size here, and 0 was read by the interface as a real, absurd measurement.
    """
    key = f"{model_id}|{models_dir or ''}"
    hit = _disk_size_cache.get(key)
    now = time.time()
    if hit is not None and (now - hit[0]) < _DISK_CACHE_TTL:
        return hit[1]
    directory = snapshot_dir_for(model_id, models_dir)
    size = None
    if directory is not None:
        try:
            measured = _measure_dir_bytes(directory)
            size = measured if measured > 0 else None
        except Exception:  # pragma: no cover - defensive
            logger.exception("failed to measure %s on disk", model_id)
            size = None
    _disk_size_cache[key] = (now, size)
    return size


def _quantization_from_config(directory: Path) -> Optional[str]:
    """Quantization the model's own ``config.json`` declares, or None."""
    config_path = directory / "config.json"
    try:
        raw = json.loads(config_path.read_text())
    except Exception:
        return None
    block = raw.get("quantization_config")
    if not isinstance(block, dict):
        return None
    # modelopt checkpoints (the NVFP4 the fleet runs) say quant_algo; the
    # transformers quantizers say quant_method.
    for key in ("quant_algo", "quant_method", "quantization"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def model_quantization(model_id: str,
                       models_dir: Optional[Path] = None) -> tuple[Optional[str], Optional[str]]:
    """``(quantization, source)`` for a model id, or ``(None, None)``.

    The catalog recipe is the first answer because it is the format AINode
    launches the model with; the model's own ``config.json`` is next, read from
    the snapshot on disk. The source travels with the value so the interface can
    say where it came from rather than presenting all three as one fact.
    """
    key = f"{model_id}|{models_dir or ''}"
    hit = _quantization_cache.get(key)
    now = time.time()
    if hit is not None and (now - hit[0]) < _DISK_CACHE_TTL:
        return hit[1]

    result: tuple[Optional[str], Optional[str]] = (None, None)
    for table in (CURATED_CLUSTER_MODELS, FALLBACK_CATALOG):
        for cid, info in (table or {}).items():
            if model_id in (cid, getattr(info, "hf_repo", "")):
                quant = getattr(info, "quantization", None)
                if quant:
                    result = (quant, "catalog")
                break
        if result[0]:
            break

    if not result[0]:
        directory = snapshot_dir_for(model_id, models_dir)
        if directory is not None:
            from_config = _quantization_from_config(directory)
            if from_config:
                result = (from_config, "config.json")

    _quantization_cache[key] = (now, result)
    return result


# ---- Model manager ---------------------------------------------------------


class ModelManager:
    """Manage model downloads, listing, and deletion against a live catalog."""

    def __init__(self, models_dir: Optional[str | Path] = None):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._active_downloads: dict[str, dict] = {}
        self._aggregator = CatalogAggregator()
        self._catalog_cache: Optional[dict[str, ModelInfo]] = None

    # -- Catalog access -------------------------------------------------------

    def get_catalog(self, refresh: bool = False) -> list[ModelInfo]:
        """Return the live merged catalog (uses memory + disk cache)."""
        if self._catalog_cache is None or refresh:
            models = self._aggregator.fetch(force_refresh=refresh)
            if not models:
                models = list(FALLBACK_CATALOG.values())
            merged = {m.id: m for m in models}
            # Always merge the curated cluster models, because the live HF sweep misses
            # them, so without this they're undiscoverable until already on disk.
            # Skip any whose hf_repo a live entry already covers (don't clobber).
            existing_repos = {m.hf_repo.lower() for m in merged.values()}
            for cid, info in CURATED_CLUSTER_MODELS.items():
                info.curated = True  # mark our hand-picked known-good set
                if cid not in merged and info.hf_repo.lower() not in existing_repos:
                    merged[cid] = info
            self._catalog_cache = merged
        return list(self._catalog_cache.values())

    def get_catalog_map(self, refresh: bool = False) -> dict[str, ModelInfo]:
        """Same as get_catalog but indexed by id."""
        self.get_catalog(refresh=refresh)
        return dict(self._catalog_cache or {})

    def _catalog_lookup(self, model_id: str) -> Optional[ModelInfo]:
        catalog = self.get_catalog_map()
        if model_id in catalog:
            return catalog[model_id]
        # Also allow lookup by hf_repo directly
        for info in catalog.values():
            if info.hf_repo == model_id or info.hf_repo.lower() == model_id.lower():
                return info
        return None

    # -- Catalog queries ------------------------------------------------------

    def list_available(self) -> list[dict]:
        """Return catalog models annotated with download status.

        Also surfaces anything present in models_dir that is NOT in the catalog
        (user-downloaded via HF search or Trending) so the UI never loses
        track of a completed download.
        """
        results = []
        catalog_repos = set()
        for info in self.get_catalog():
            entry = info.to_dict()
            entry["downloaded"] = self._is_downloaded_info(info)
            local_size = self._local_size_gb_info(info)
            if local_size is not None:
                entry["local_size_gb"] = round(local_size, 2)
            catalog_repos.add(info.hf_repo.lower())
            results.append(entry)

        # Merge in downloaded-but-not-in-catalog entries. HF's cache layout is
        # models_dir/hub/models--<org>--<name>/, skipping control dirs like
        # .locks, xet, blobs, snapshots and anything not prefixed `models--`.
        scan_roots = [
            self.models_dir,
            self.models_dir / "hub",
            self.models_dir / "hf-cache" / "hub",  # out-of-band HF_HOME=models/hf-cache downloads
        ]
        seen_slugs: set[str] = set()
        for root in scan_roots:
            if not root.exists():
                continue
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                if not child.name.startswith("models--"):
                    continue  # HF internals: .locks, xet, hub, blobs, snapshots
                if child.name in seen_slugs:
                    continue
                seen_slugs.add(child.name)
                # HF slug "models--org--name" → "org/name"
                hf_repo = child.name[len("models--"):].replace("--", "/", 1)
                if hf_repo.lower() in catalog_repos:
                    continue  # already merged
                results.append({
                    "id": child.name,
                    "slug": child.name,
                    "name": hf_repo.split("/")[-1],
                    "hf_repo": hf_repo,
                    "size_gb": round(self._dir_size_gb(child), 2),
                    "description": "User-downloaded model",
                    "quantization": None,
                    "min_memory_gb": 0,
                    "family": hf_repo.split("/")[0].lower() if "/" in hf_repo else "",
                    "params_b": 0,
                    "context_length": 0,
                    "license": "",
                    "recommended": False,
                    "created_at": "",
                    "downloads": 0,
                    "likes": 0,
                    "capabilities": [],
                    "architecture": "",
                    "format": "",
                    "downloaded": True,
                    "local_size_gb": round(self._dir_size_gb(child), 2),
                })
        return results

    def list_downloaded(self) -> list[dict]:
        """Scan models_dir and return info for every model present on disk.

        Handles three directory layouts:
          1. HF cache:   models_dir/hub/models--org--name/
          2. Flat cache: models_dir/models--org--name/
          3. Direct:     models_dir/org--name/   (written by _run_download_repo)
        """
        downloaded: list[dict] = []
        if not self.models_dir.exists():
            return downloaded

        seen: set[str] = set()

        def _add(child: "Path", hf_repo: str) -> None:
            if hf_repo in seen:
                return
            seen.add(hf_repo)
            catalog_entry = self._find_catalog_by_hf_repo(hf_repo)
            if catalog_entry:
                entry = catalog_entry.to_dict()
                entry["downloaded"] = True
                entry["local_size_gb"] = round(self._dir_size_gb(child), 2)
                downloaded.append(entry)
            else:
                downloaded.append({
                    "id": hf_repo,
                    "name": hf_repo.split("/")[-1] if "/" in hf_repo else hf_repo,
                    "hf_repo": hf_repo,
                    "size_gb": round(self._dir_size_gb(child), 2),
                    "description": "Downloaded model",
                    "quantization": None,
                    "min_memory_gb": 0,
                    "downloaded": True,
                    "local_size_gb": round(self._dir_size_gb(child), 2),
                })

        # Scan top-level models_dir
        for child in sorted(self.models_dir.iterdir()):
            if not child.is_dir():
                continue
            name = child.name
            if name.startswith("models--"):
                # HF flat cache: models--org--name
                _add(child, name[len("models--"):].replace("--", "/", 1))
            elif "--" in name and not name.startswith(".") and name != "hub":
                # Direct download: org--name
                _add(child, name.replace("--", "/", 1))

        # Also scan nested HF cache layouts: models_dir/hub and the out-of-band
        # models_dir/hf-cache/hub (HF_HOME=models/hf-cache downloads land here).
        for hub in (self.models_dir / "hub", self.models_dir / "hf-cache" / "hub"):
            if hub.is_dir():
                for child in sorted(hub.iterdir()):
                    if child.is_dir() and child.name.startswith("models--"):
                        _add(child, child.name[len("models--"):].replace("--", "/", 1))

        return downloaded

    def _find_catalog_by_hf_repo(self, hf_repo: str):
        """Find a catalog entry by HF repo ID (case-insensitive)."""
        hf_lower = hf_repo.lower()
        for info in self.get_catalog():
            if info.hf_repo.lower() == hf_lower:
                return info
        return None

    def get_model_info(self, model_id: str) -> Optional[dict]:
        """Return catalog info for a model, plus local size if downloaded."""
        info = self._catalog_lookup(model_id)
        if info is None:
            return None
        entry = info.to_dict()
        entry["downloaded"] = self._is_downloaded_info(info)
        local_size = self._local_size_gb_info(info)
        if local_size is not None:
            entry["local_size_gb"] = round(local_size, 2)
        return entry

    def recommend_for_gpu(self, gpu_memory_gb: float) -> list[dict]:
        """Return catalog models that fit within the given GPU memory."""
        results = []
        for info in self.get_catalog():
            if info.min_memory_gb <= gpu_memory_gb:
                entry = info.to_dict()
                entry["downloaded"] = self._is_downloaded_info(info)
                results.append(entry)
        results.sort(key=lambda m: m["size_gb"], reverse=True)
        return results

    # -- Download / Delete ----------------------------------------------------

    def download_model(
        self,
        model_id: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Path:
        """Download a model from HuggingFace Hub and return the local path."""
        info = self._catalog_lookup(model_id)
        if info is None:
            raise ValueError(
                f"Unknown model: {model_id}. Use an id from the catalog."
            )

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required for model downloads. "
                "Install it with: pip install huggingface_hub"
            )

        local_dir = self.models_dir / self._repo_to_dirname(info.hf_repo)

        download_path = snapshot_download(
            repo_id=info.hf_repo,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            # Cap parallel file connections so a fat model pull can't monopolise
            # the uplink (ponytail: bounds parallelism, not absolute byte-rate,
            # upgrade to a tc/trickle shaper if a single stream still saturates).
            max_workers=_download_max_workers(),
        )

        return Path(download_path)

    def delete_model(self, model_id: str) -> bool:
        """Delete a downloaded model from disk; return True if deleted."""
        info = self._catalog_lookup(model_id)
        if info is None:
            raise ValueError(f"Unknown model: {model_id}")

        model_dir = self.models_dir / self._repo_to_dirname(info.hf_repo)
        if model_dir.exists():
            shutil.rmtree(model_dir)
            return True
        return False

    # -- Internal helpers -----------------------------------------------------

    @staticmethod
    def _repo_to_dirname(hf_repo: str) -> str:
        """Convert 'org/model-name' to 'org--model-name' for filesystem safety."""
        return hf_repo.replace("/", "--")

    def _model_dir_info(self, info: ModelInfo) -> Path:
        return self.models_dir / self._repo_to_dirname(info.hf_repo)

    def _find_model_dir(self, info: ModelInfo) -> Optional[Path]:
        """Return the on-disk dir for a model across every layout we support."""
        return find_model_dir(self.models_dir, info.hf_repo)

    def _is_downloaded_info(self, info: ModelInfo) -> bool:
        return self._find_model_dir(info) is not None

    def _local_size_gb_info(self, info: ModelInfo) -> Optional[float]:
        d = self._find_model_dir(info)
        return self._dir_size_gb(d) if d else None

    @staticmethod
    def _dir_size_gb(path: Path) -> float:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return total / (1024**3)

    def search_huggingface(self, query: str, limit: int = 50) -> list[dict]:
        """Search HuggingFace Hub for text-generation models matching the query."""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            # huggingface_hub >=1.x dropped `direction`/`task`; use pipeline_tag.
            # expand=safetensors pulls the dtype breakdown so we can show real size.
            models = api.list_models(
                search=query,
                pipeline_tag="text-generation",
                limit=limit,
                sort="downloads",
                expand=["safetensors"],
            )
            catalog_repos = {info.hf_repo.lower() for info in self.get_catalog()}
            results = []
            for m in models:
                repo = m.id
                repo_l = repo.lower()
                slug = repo.replace("/", "--").lower()
                size_gb = _safetensors_size_gb(getattr(m, "safetensors", None))
                sf = getattr(m, "safetensors", None)
                total_params = getattr(sf, "total", 0) if sf else 0
                # Quant/engine from repo name, which drives the badge AND the
                # "can it run on vLLM/GB10" filter (MLX=Apple, GGUF=llama.cpp).
                quant = ""
                for tag, label in (
                    ("nvfp4", "NVFP4"), ("mxfp4", "MXFP4"), ("w4afp8", "W4AFP8"),
                    ("w4a16", "W4A16"), ("awq", "AWQ"), ("gptq", "GPTQ"),
                    ("int4", "INT4"), ("int8", "INT8"), ("fp8", "FP8"),
                    ("gguf", "GGUF"), ("mlx", "MLX"), ("bf16", "BF16"), ("fp16", "FP16"),
                ):
                    if tag in repo_l:
                        quant = label
                        break
                vllm_ok = not ("mlx" in repo_l or "gguf" in repo_l or "ggml" in repo_l)
                results.append({
                    "id": slug,
                    "name": repo.split("/")[-1],
                    "hf_repo": repo,
                    "size_gb": round(size_gb, 1),
                    "description": (m.pipeline_tag or "text-generation") + " model",
                    "family": repo.split("/")[0].lower(),
                    "params_b": round(total_params / 1e9, 1) if total_params else 0,
                    "context_length": 0,
                    "license": "",
                    "recommended": False,
                    "quant": quant,
                    "vllm_ok": vllm_ok,
                    "downloads": getattr(m, "downloads", 0),
                    "likes": getattr(m, "likes", 0),
                    "in_catalog": repo_l in catalog_repos,
                })
            # Always surface curated matches for the query, because HF's download-sorted
            # page often ranks our vetted pick past the limit, so inject it.
            ql = query.lower()
            have = {r["hf_repo"].lower() for r in results}
            for info in self.get_catalog():
                if not getattr(info, "curated", False):
                    continue
                hay = (info.hf_repo + " " + info.name + " " + (info.family or "")).lower()
                if ql in hay and info.hf_repo.lower() not in have:
                    results.append({
                        "id": info.hf_repo.replace("/", "--").lower(),
                        "name": info.name,
                        "hf_repo": info.hf_repo,
                        "size_gb": info.size_gb,
                        "description": info.description,
                        "family": info.family,
                        "params_b": info.params_b,
                        "context_length": info.context_length,
                        "license": info.license,
                        "recommended": info.recommended,
                        "quant": (info.quantization or info.format or "").upper(),
                        "vllm_ok": True,
                        "proven_tp": info.proven_tp,
                        "downloads": 0,
                        "likes": 0,
                        "in_catalog": True,
                    })
            # Vetted (in-catalog) pick first, then most-downloaded.
            results.sort(key=lambda r: (not r["in_catalog"], -(r["downloads"] or 0)))
            return results
        except Exception as e:
            logger.warning("HuggingFace search failed for %r: %s", query, e)
            return []

    def _find_catalog_by_dir(self, dirname: str) -> Optional[ModelInfo]:
        for info in self.get_catalog():
            if ModelManager._repo_to_dirname(info.hf_repo) == dirname:
                return info
        return None
