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

    def to_dict(self) -> dict:
        return asdict(self)


# ---- Curated model catalog --------------------------------------------------

MODEL_CATALOG: dict[str, ModelInfo] = {
    "llama-3.2-3b": ModelInfo(
        id="llama-3.2-3b",
        name="Llama 3.2 3B Instruct",
        hf_repo="meta-llama/Llama-3.2-3B-Instruct",
        size_gb=6.0,
        description="Compact, fast model for everyday tasks. Great starter model.",
        min_memory_gb=8,
    ),
    "llama-3.1-8b": ModelInfo(
        id="llama-3.1-8b",
        name="Llama 3.1 8B Instruct",
        hf_repo="meta-llama/Llama-3.1-8B-Instruct",
        size_gb=16.0,
        description="Strong general-purpose model. Good balance of quality and speed.",
        min_memory_gb=16,
    ),
    "llama-3.1-70b-awq": ModelInfo(
        id="llama-3.1-70b-awq",
        name="Llama 3.1 70B Instruct AWQ",
        hf_repo="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        size_gb=38.0,
        description="Quantized 70B for high-quality output on single-node hardware.",
        quantization="awq",
        min_memory_gb=48,
    ),
    "qwen-2.5-72b": ModelInfo(
        id="qwen-2.5-72b",
        name="Qwen 2.5 72B Instruct",
        hf_repo="Qwen/Qwen2.5-72B-Instruct",
        size_gb=145.0,
        description="Flagship multilingual model. Excellent reasoning and code.",
        min_memory_gb=160,
    ),
    "deepseek-r1-7b": ModelInfo(
        id="deepseek-r1-7b",
        name="DeepSeek R1 Distill Qwen 7B",
        hf_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        size_gb=14.0,
        description="Reasoning-focused model distilled from DeepSeek R1.",
        min_memory_gb=16,
    ),
    "mistral-7b": ModelInfo(
        id="mistral-7b",
        name="Mistral 7B Instruct v0.3",
        hf_repo="mistralai/Mistral-7B-Instruct-v0.3",
        size_gb=14.0,
        description="Fast, efficient 7B with strong instruction following.",
        min_memory_gb=16,
    ),
    "phi-3-mini": ModelInfo(
        id="phi-3-mini",
        name="Phi-3 Mini 4K Instruct",
        hf_repo="microsoft/Phi-3-mini-4k-instruct",
        size_gb=7.5,
        description="Microsoft's compact model. Strong reasoning for its size.",
        min_memory_gb=8,
    ),
    "codellama-34b": ModelInfo(
        id="codellama-34b",
        name="CodeLlama 34B Instruct",
        hf_repo="meta-llama/CodeLlama-34b-Instruct-hf",
        size_gb=63.0,
        description="Specialized code generation and understanding model.",
        min_memory_gb=72,
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
