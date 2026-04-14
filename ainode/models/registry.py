"""Model registry and manager — catalog, download, delete, and recommend models."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

from ainode.core.config import MODELS_DIR


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
    context_length: int = 0
    license: str = ""
    recommended: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ---- Curated model catalog --------------------------------------------------
#
# Entries are grouped by family. Sizes are approximate: roughly params_b * 2 GB
# for bf16/fp16 weights and params_b * 0.5 GB for int4/AWQ/GPTQ variants.

MODEL_CATALOG: dict[str, ModelInfo] = {
    # ---- Llama family -------------------------------------------------------
    "llama-3.2-1b": ModelInfo(
        id="llama-3.2-1b",
        name="Llama 3.2 1B Instruct",
        hf_repo="meta-llama/Llama-3.2-1B-Instruct",
        size_gb=2.5,
        description="Tiny Llama model for edge devices and ultra-fast inference.",
        min_memory_gb=4,
        family="llama",
        params_b=1.23,
        context_length=131072,
        license="Llama 3.2",
    ),
    "llama-3.2-3b": ModelInfo(
        id="llama-3.2-3b",
        name="Llama 3.2 3B Instruct",
        hf_repo="meta-llama/Llama-3.2-3B-Instruct",
        size_gb=6.0,
        description="Compact, fast model for everyday tasks. Great starter model.",
        min_memory_gb=8,
        family="llama",
        params_b=3.21,
        context_length=131072,
        license="Llama 3.2",
        recommended=True,
    ),
    "llama-3.1-8b": ModelInfo(
        id="llama-3.1-8b",
        name="Llama 3.1 8B Instruct",
        hf_repo="meta-llama/Llama-3.1-8B-Instruct",
        size_gb=16.0,
        description="Strong general-purpose model. Good balance of quality and speed.",
        min_memory_gb=16,
        family="llama",
        params_b=8.03,
        context_length=131072,
        license="Llama 3.1",
        recommended=True,
    ),
    "llama-3.1-8b-awq": ModelInfo(
        id="llama-3.1-8b-awq",
        name="Llama 3.1 8B Instruct AWQ",
        hf_repo="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        size_gb=5.5,
        description="AWQ-quantized Llama 3.1 8B. Fits on consumer GPUs.",
        quantization="awq",
        min_memory_gb=8,
        family="llama",
        params_b=8.03,
        context_length=131072,
        license="Llama 3.1",
    ),
    "llama-3.1-70b": ModelInfo(
        id="llama-3.1-70b",
        name="Llama 3.1 70B Instruct",
        hf_repo="meta-llama/Llama-3.1-70B-Instruct",
        size_gb=141.0,
        description="Full-precision Llama 3.1 70B. Needs multi-GPU or large VRAM.",
        min_memory_gb=160,
        family="llama",
        params_b=70.6,
        context_length=131072,
        license="Llama 3.1",
    ),
    "llama-3.1-70b-awq": ModelInfo(
        id="llama-3.1-70b-awq",
        name="Llama 3.1 70B Instruct AWQ",
        hf_repo="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        size_gb=38.0,
        description="Quantized 70B for high-quality output on single-node hardware.",
        quantization="awq",
        min_memory_gb=48,
        family="llama",
        params_b=70.6,
        context_length=131072,
        license="Llama 3.1",
    ),
    "llama-3.1-70b-gptq": ModelInfo(
        id="llama-3.1-70b-gptq",
        name="Llama 3.1 70B Instruct GPTQ",
        hf_repo="hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4",
        size_gb=40.0,
        description="GPTQ-quantized 70B alternative to AWQ variant.",
        quantization="gptq",
        min_memory_gb=48,
        family="llama",
        params_b=70.6,
        context_length=131072,
        license="Llama 3.1",
    ),
    "llama-3.3-70b": ModelInfo(
        id="llama-3.3-70b",
        name="Llama 3.3 70B Instruct",
        hf_repo="meta-llama/Llama-3.3-70B-Instruct",
        size_gb=141.0,
        description="Latest Llama flagship. Near-405B quality at 70B size.",
        min_memory_gb=160,
        family="llama",
        params_b=70.6,
        context_length=131072,
        license="Llama 3.3",
    ),
    "llama-3.3-70b-awq": ModelInfo(
        id="llama-3.3-70b-awq",
        name="Llama 3.3 70B Instruct AWQ",
        hf_repo="casperhansen/llama-3.3-70b-instruct-awq",
        size_gb=39.0,
        description="AWQ-quantized Llama 3.3 70B for single-GPU deployment.",
        quantization="awq",
        min_memory_gb=48,
        family="llama",
        params_b=70.6,
        context_length=131072,
        license="Llama 3.3",
    ),

    # ---- Qwen family --------------------------------------------------------
    "qwen-2.5-0.5b": ModelInfo(
        id="qwen-2.5-0.5b",
        name="Qwen 2.5 0.5B Instruct",
        hf_repo="Qwen/Qwen2.5-0.5B-Instruct",
        size_gb=1.0,
        description="Ultra-small Qwen. Perfect for embedded and testing.",
        min_memory_gb=2,
        family="qwen",
        params_b=0.5,
        context_length=32768,
        license="Qwen",
    ),
    "qwen-2.5-1.5b": ModelInfo(
        id="qwen-2.5-1.5b",
        name="Qwen 2.5 1.5B Instruct",
        hf_repo="Qwen/Qwen2.5-1.5B-Instruct",
        size_gb=3.1,
        description="Small Qwen with strong multilingual support.",
        min_memory_gb=4,
        family="qwen",
        params_b=1.54,
        context_length=32768,
        license="Qwen",
    ),
    "qwen-2.5-3b": ModelInfo(
        id="qwen-2.5-3b",
        name="Qwen 2.5 3B Instruct",
        hf_repo="Qwen/Qwen2.5-3B-Instruct",
        size_gb=6.2,
        description="Compact Qwen balancing quality and speed.",
        min_memory_gb=8,
        family="qwen",
        params_b=3.09,
        context_length=32768,
        license="Qwen",
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
        context_length=131072,
        license="Qwen",
        recommended=True,
    ),
    "qwen-2.5-14b": ModelInfo(
        id="qwen-2.5-14b",
        name="Qwen 2.5 14B Instruct",
        hf_repo="Qwen/Qwen2.5-14B-Instruct",
        size_gb=29.0,
        description="Mid-size Qwen, competitive with much larger models.",
        min_memory_gb=32,
        family="qwen",
        params_b=14.7,
        context_length=131072,
        license="Qwen",
    ),
    "qwen-2.5-32b": ModelInfo(
        id="qwen-2.5-32b",
        name="Qwen 2.5 32B Instruct",
        hf_repo="Qwen/Qwen2.5-32B-Instruct",
        size_gb=65.0,
        description="Large Qwen for high-quality reasoning.",
        min_memory_gb=72,
        family="qwen",
        params_b=32.5,
        context_length=131072,
        license="Qwen",
    ),
    "qwen-2.5-72b": ModelInfo(
        id="qwen-2.5-72b",
        name="Qwen 2.5 72B Instruct",
        hf_repo="Qwen/Qwen2.5-72B-Instruct",
        size_gb=145.0,
        description="Flagship multilingual model. Excellent reasoning and code.",
        min_memory_gb=160,
        family="qwen",
        params_b=72.7,
        context_length=131072,
        license="Qwen",
    ),
    "qwen-2.5-coder-7b": ModelInfo(
        id="qwen-2.5-coder-7b",
        name="Qwen 2.5 Coder 7B Instruct",
        hf_repo="Qwen/Qwen2.5-Coder-7B-Instruct",
        size_gb=15.0,
        description="Code-specialized Qwen. Great for autocomplete and code gen.",
        min_memory_gb=16,
        family="qwen",
        params_b=7.62,
        context_length=131072,
        license="Apache 2.0",
        recommended=True,
    ),
    "qwen-2.5-coder-32b": ModelInfo(
        id="qwen-2.5-coder-32b",
        name="Qwen 2.5 Coder 32B Instruct",
        hf_repo="Qwen/Qwen2.5-Coder-32B-Instruct",
        size_gb=65.0,
        description="State-of-the-art open-weight code model.",
        min_memory_gb=72,
        family="qwen",
        params_b=32.5,
        context_length=131072,
        license="Apache 2.0",
    ),
    "qwen-3-8b": ModelInfo(
        id="qwen-3-8b",
        name="Qwen 3 8B",
        hf_repo="Qwen/Qwen3-8B",
        size_gb=16.4,
        description="Next-gen Qwen with hybrid thinking/non-thinking modes.",
        min_memory_gb=20,
        family="qwen",
        params_b=8.2,
        context_length=131072,
        license="Apache 2.0",
    ),
    "qwen-3-32b": ModelInfo(
        id="qwen-3-32b",
        name="Qwen 3 32B",
        hf_repo="Qwen/Qwen3-32B",
        size_gb=65.5,
        description="Qwen 3 dense 32B with thinking-mode reasoning.",
        min_memory_gb=72,
        family="qwen",
        params_b=32.8,
        context_length=131072,
        license="Apache 2.0",
    ),

    # ---- DeepSeek family ----------------------------------------------------
    "deepseek-r1-distill-qwen-1.5b": ModelInfo(
        id="deepseek-r1-distill-qwen-1.5b",
        name="DeepSeek R1 Distill Qwen 1.5B",
        hf_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        size_gb=3.5,
        description="Tiny reasoning model distilled from R1.",
        min_memory_gb=4,
        family="deepseek",
        params_b=1.78,
        context_length=131072,
        license="MIT",
    ),
    "deepseek-r1-distill-qwen-7b": ModelInfo(
        id="deepseek-r1-distill-qwen-7b",
        name="DeepSeek R1 Distill Qwen 7B",
        hf_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        size_gb=14.0,
        description="Reasoning-focused 7B distilled from DeepSeek R1.",
        min_memory_gb=16,
        family="deepseek",
        params_b=7.0,
        context_length=131072,
        license="MIT",
        recommended=True,
    ),
    # Legacy alias kept so older clients/tests keep working.
    "deepseek-r1-7b": ModelInfo(
        id="deepseek-r1-7b",
        name="DeepSeek R1 Distill Qwen 7B",
        hf_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        size_gb=14.0,
        description="Reasoning-focused model distilled from DeepSeek R1.",
        min_memory_gb=16,
        family="deepseek",
        params_b=7.0,
        context_length=131072,
        license="MIT",
    ),
    "deepseek-r1-distill-qwen-14b": ModelInfo(
        id="deepseek-r1-distill-qwen-14b",
        name="DeepSeek R1 Distill Qwen 14B",
        hf_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        size_gb=28.0,
        description="Mid-size reasoning distillation of R1.",
        min_memory_gb=32,
        family="deepseek",
        params_b=14.8,
        context_length=131072,
        license="MIT",
    ),
    "deepseek-r1-distill-qwen-32b": ModelInfo(
        id="deepseek-r1-distill-qwen-32b",
        name="DeepSeek R1 Distill Qwen 32B",
        hf_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        size_gb=65.0,
        description="Large reasoning distillation of R1, rivals GPT-4 on math.",
        min_memory_gb=72,
        family="deepseek",
        params_b=32.8,
        context_length=131072,
        license="MIT",
    ),
    "deepseek-r1-distill-llama-8b": ModelInfo(
        id="deepseek-r1-distill-llama-8b",
        name="DeepSeek R1 Distill Llama 8B",
        hf_repo="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        size_gb=16.0,
        description="R1 reasoning distilled onto Llama 3.1 8B base.",
        min_memory_gb=16,
        family="deepseek",
        params_b=8.03,
        context_length=131072,
        license="Llama 3.1",
    ),
    "deepseek-r1-distill-llama-70b": ModelInfo(
        id="deepseek-r1-distill-llama-70b",
        name="DeepSeek R1 Distill Llama 70B",
        hf_repo="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        size_gb=141.0,
        description="Top-tier reasoning distillation on Llama 3.3 70B base.",
        min_memory_gb=160,
        family="deepseek",
        params_b=70.6,
        context_length=131072,
        license="Llama 3.3",
    ),
    "deepseek-v2.5": ModelInfo(
        id="deepseek-v2.5",
        name="DeepSeek V2.5",
        hf_repo="deepseek-ai/DeepSeek-V2.5",
        size_gb=472.0,
        description="236B MoE chat model (21B active). Requires cluster deployment.",
        min_memory_gb=480,
        family="deepseek",
        params_b=236.0,
        context_length=131072,
        license="DeepSeek",
    ),
    "deepseek-coder-v2-lite": ModelInfo(
        id="deepseek-coder-v2-lite",
        name="DeepSeek Coder V2 Lite Instruct",
        hf_repo="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        size_gb=31.0,
        description="16B MoE code model (2.4B active). Fast and strong at code.",
        min_memory_gb=32,
        family="deepseek",
        params_b=15.7,
        context_length=163840,
        license="DeepSeek",
    ),

    # ---- Mistral family -----------------------------------------------------
    "mistral-7b": ModelInfo(
        id="mistral-7b",
        name="Mistral 7B Instruct v0.3",
        hf_repo="mistralai/Mistral-7B-Instruct-v0.3",
        size_gb=14.0,
        description="Fast, efficient 7B with strong instruction following.",
        min_memory_gb=16,
        family="mistral",
        params_b=7.25,
        context_length=32768,
        license="Apache 2.0",
        recommended=True,
    ),
    "mistral-small-24b": ModelInfo(
        id="mistral-small-24b",
        name="Mistral Small 24B Instruct",
        hf_repo="mistralai/Mistral-Small-24B-Instruct-2501",
        size_gb=47.0,
        description="Mistral Small 3 — Apache-licensed 24B with GPT-4o-class quality.",
        min_memory_gb=52,
        family="mistral",
        params_b=23.6,
        context_length=32768,
        license="Apache 2.0",
    ),
    "mistral-large-2411": ModelInfo(
        id="mistral-large-2411",
        name="Mistral Large 2411",
        hf_repo="mistralai/Mistral-Large-Instruct-2411",
        size_gb=246.0,
        description="Mistral flagship 123B dense model.",
        min_memory_gb=260,
        family="mistral",
        params_b=123.0,
        context_length=131072,
        license="Mistral Research",
    ),
    "mixtral-8x7b": ModelInfo(
        id="mixtral-8x7b",
        name="Mixtral 8x7B Instruct v0.1",
        hf_repo="mistralai/Mixtral-8x7B-Instruct-v0.1",
        size_gb=93.0,
        description="Sparse MoE model. 46.7B total, 12.9B active per token.",
        min_memory_gb=100,
        family="mixtral",
        params_b=46.7,
        context_length=32768,
        license="Apache 2.0",
    ),
    "mixtral-8x22b": ModelInfo(
        id="mixtral-8x22b",
        name="Mixtral 8x22B Instruct v0.1",
        hf_repo="mistralai/Mixtral-8x22B-Instruct-v0.1",
        size_gb=281.0,
        description="Large MoE. 141B total, 39B active. High throughput.",
        min_memory_gb=300,
        family="mixtral",
        params_b=141.0,
        context_length=65536,
        license="Apache 2.0",
    ),

    # ---- Phi family ---------------------------------------------------------
    "phi-3-mini": ModelInfo(
        id="phi-3-mini",
        name="Phi-3 Mini 4K Instruct",
        hf_repo="microsoft/Phi-3-mini-4k-instruct",
        size_gb=7.5,
        description="Microsoft's compact model. Strong reasoning for its size.",
        min_memory_gb=8,
        family="phi",
        params_b=3.82,
        context_length=4096,
        license="MIT",
        recommended=True,
    ),
    "phi-3-medium": ModelInfo(
        id="phi-3-medium",
        name="Phi-3 Medium 4K Instruct",
        hf_repo="microsoft/Phi-3-medium-4k-instruct",
        size_gb=28.0,
        description="14B Phi-3 with stronger reasoning at mid-size cost.",
        min_memory_gb=32,
        family="phi",
        params_b=14.0,
        context_length=4096,
        license="MIT",
    ),
    "phi-4": ModelInfo(
        id="phi-4",
        name="Phi-4",
        hf_repo="microsoft/phi-4",
        size_gb=29.0,
        description="Microsoft Phi-4 (14B). Strong reasoning, math, and coding.",
        min_memory_gb=32,
        family="phi",
        params_b=14.7,
        context_length=16384,
        license="MIT",
    ),

    # ---- Gemma family -------------------------------------------------------
    "gemma-2-2b": ModelInfo(
        id="gemma-2-2b",
        name="Gemma 2 2B IT",
        hf_repo="google/gemma-2-2b-it",
        size_gb=5.2,
        description="Tiny Gemma 2 for fast local inference.",
        min_memory_gb=6,
        family="gemma",
        params_b=2.61,
        context_length=8192,
        license="Gemma",
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
        context_length=8192,
        license="Gemma",
        recommended=True,
    ),
    "gemma-2-27b": ModelInfo(
        id="gemma-2-27b",
        name="Gemma 2 27B IT",
        hf_repo="google/gemma-2-27b-it",
        size_gb=54.0,
        description="Google Gemma 2 flagship. Competitive with Llama 3 70B.",
        min_memory_gb=60,
        family="gemma",
        params_b=27.2,
        context_length=8192,
        license="Gemma",
    ),

    # ---- CodeLlama ----------------------------------------------------------
    "codellama-7b": ModelInfo(
        id="codellama-7b",
        name="CodeLlama 7B Instruct",
        hf_repo="meta-llama/CodeLlama-7b-Instruct-hf",
        size_gb=13.5,
        description="Small code generation model based on Llama 2.",
        min_memory_gb=16,
        family="llama",
        params_b=6.74,
        context_length=16384,
        license="Llama 2",
    ),
    "codellama-13b": ModelInfo(
        id="codellama-13b",
        name="CodeLlama 13B Instruct",
        hf_repo="meta-llama/CodeLlama-13b-Instruct-hf",
        size_gb=26.0,
        description="Mid-size code generation model based on Llama 2.",
        min_memory_gb=32,
        family="llama",
        params_b=13.0,
        context_length=16384,
        license="Llama 2",
    ),
    "codellama-34b": ModelInfo(
        id="codellama-34b",
        name="CodeLlama 34B Instruct",
        hf_repo="meta-llama/CodeLlama-34b-Instruct-hf",
        size_gb=63.0,
        description="Specialized code generation and understanding model.",
        min_memory_gb=72,
        family="llama",
        params_b=33.7,
        context_length=16384,
        license="Llama 2",
    ),

    # ---- Specialty ----------------------------------------------------------
    "nemotron-super-49b": ModelInfo(
        id="nemotron-super-49b",
        name="Llama 3.3 Nemotron Super 49B",
        hf_repo="nvidia/Llama-3_3-Nemotron-Super-49B-v1",
        size_gb=99.0,
        description="NVIDIA Nemotron Super. Llama 3.3 70B pruned to 49B.",
        min_memory_gb=110,
        family="llama",
        params_b=49.0,
        context_length=131072,
        license="NVIDIA Open Model",
    ),
    "glm-4-9b": ModelInfo(
        id="glm-4-9b",
        name="GLM-4 9B Chat",
        hf_repo="THUDM/glm-4-9b-chat",
        size_gb=19.0,
        description="Zhipu GLM-4 9B chat. Strong bilingual (EN/ZH) model.",
        min_memory_gb=24,
        family="glm",
        params_b=9.4,
        context_length=131072,
        license="GLM",
    ),
    "yi-1.5-34b": ModelInfo(
        id="yi-1.5-34b",
        name="Yi 1.5 34B Chat",
        hf_repo="01-ai/Yi-1.5-34B-Chat",
        size_gb=69.0,
        description="01.AI Yi 1.5 34B. Strong English/Chinese reasoning.",
        min_memory_gb=76,
        family="yi",
        params_b=34.4,
        context_length=32768,
        license="Apache 2.0",
    ),
}


class ModelManager:
    """Manage model downloads, listing, and deletion."""

    def __init__(self, models_dir: Optional[str | Path] = None):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._active_downloads: dict[str, dict] = {}

    # -- Catalog queries -------------------------------------------------------

    def list_available(self) -> list[dict]:
        """Return catalog models annotated with download status."""
        results = []
        for model_id, info in MODEL_CATALOG.items():
            entry = info.to_dict()
            entry["downloaded"] = self._is_downloaded(model_id)
            local_size = self._local_size_gb(model_id)
            if local_size is not None:
                entry["local_size_gb"] = round(local_size, 2)
            results.append(entry)
        return results

    def list_downloaded(self) -> list[dict]:
        """Scan models_dir and return info for every downloaded model."""
        downloaded = []
        if not self.models_dir.exists():
            return downloaded

        for child in sorted(self.models_dir.iterdir()):
            if not child.is_dir():
                continue
            # Map directory name back to catalog entry
            catalog_entry = self._find_catalog_by_dir(child.name)
            if catalog_entry:
                entry = catalog_entry.to_dict()
                entry["downloaded"] = True
                entry["local_size_gb"] = round(self._dir_size_gb(child), 2)
                downloaded.append(entry)
            else:
                # Not in catalog — user-added model
                downloaded.append({
                    "id": child.name,
                    "name": child.name,
                    "hf_repo": child.name.replace("--", "/", 1),
                    "size_gb": None,
                    "description": "User-downloaded model (not in catalog)",
                    "quantization": None,
                    "min_memory_gb": 0,
                    "downloaded": True,
                    "local_size_gb": round(self._dir_size_gb(child), 2),
                })
        return downloaded

    def get_model_info(self, model_id: str) -> Optional[dict]:
        """Return catalog info for a model, plus local size if downloaded."""
        info = MODEL_CATALOG.get(model_id)
        if info is None:
            return None
        entry = info.to_dict()
        entry["downloaded"] = self._is_downloaded(model_id)
        local_size = self._local_size_gb(model_id)
        if local_size is not None:
            entry["local_size_gb"] = round(local_size, 2)
        return entry

    def recommend_for_gpu(self, gpu_memory_gb: float) -> list[dict]:
        """Return models that fit within the given GPU memory."""
        results = []
        for model_id, info in MODEL_CATALOG.items():
            if info.min_memory_gb <= gpu_memory_gb:
                entry = info.to_dict()
                entry["downloaded"] = self._is_downloaded(model_id)
                results.append(entry)
        # Sort by size descending so the most capable fitting model is first
        results.sort(key=lambda m: m["size_gb"], reverse=True)
        return results

    # -- Download / Delete -----------------------------------------------------

    def download_model(
        self,
        model_id: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Path:
        """Download a model from HuggingFace Hub and return the local path."""
        info = MODEL_CATALOG.get(model_id)
        if info is None:
            raise ValueError(f"Unknown model: {model_id}. Use a key from MODEL_CATALOG.")

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
        )

        return Path(download_path)

    def delete_model(self, model_id: str) -> bool:
        """Delete a downloaded model from disk; return True if deleted."""
        info = MODEL_CATALOG.get(model_id)
        if info is None:
            raise ValueError(f"Unknown model: {model_id}")

        model_dir = self.models_dir / self._repo_to_dirname(info.hf_repo)
        if model_dir.exists():
            shutil.rmtree(model_dir)
            return True
        return False

    # -- Internal helpers ------------------------------------------------------

    @staticmethod
    def _repo_to_dirname(hf_repo: str) -> str:
        """Convert 'org/model-name' to 'org--model-name' for filesystem safety."""
        return hf_repo.replace("/", "--")

    def _model_dir(self, model_id: str) -> Optional[Path]:
        """Return the local directory for a catalog model, or None."""
        info = MODEL_CATALOG.get(model_id)
        if info is None:
            return None
        return self.models_dir / self._repo_to_dirname(info.hf_repo)

    def _is_downloaded(self, model_id: str) -> bool:
        d = self._model_dir(model_id)
        return d is not None and d.exists() and any(d.iterdir())

    def _local_size_gb(self, model_id: str) -> Optional[float]:
        d = self._model_dir(model_id)
        if d is None or not d.exists():
            return None
        return self._dir_size_gb(d)

    @staticmethod
    def _dir_size_gb(path: Path) -> float:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return total / (1024**3)

    def _find_catalog_by_dir(self, dirname: str) -> Optional[ModelInfo]:
        for info in MODEL_CATALOG.values():
            if ModelManager._repo_to_dirname(info.hf_repo) == dirname:
                return info
        return None
