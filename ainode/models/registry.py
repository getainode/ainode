"""Model registry and manager — dynamic catalog + download/delete/recommend.

The catalog is now assembled dynamically from live sources (HuggingFace Hub,
Ollama library, NVIDIA NIM) with a 24-hour on-disk cache and a small static
fallback for offline/error situations.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

from ainode.core.config import AINODE_HOME, MODELS_DIR


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
        context_length=8192,
        license="Gemma",
        recommended=True,
    ),
}


# Backward-compat alias — external code may still import MODEL_CATALOG.
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
                {"task": "text-generation", "sort": "downloads", "limit": 50},
                {"task": "text-generation", "tags": "instruct", "sort": "downloads", "limit": 30},
                {"task": "text-generation", "tags": "chat", "sort": "downloads", "limit": 20},
            ]
            results: list[ModelInfo] = []
            seen_ids: set[str] = set()
            for q in queries:
                try:
                    iterator = api.list_models(direction=-1, **q)
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
        )

    # -- Source: HuggingFace trending ---------------------------------------

    def fetch_trending(self, limit: int = 30) -> list[ModelInfo]:
        """Models trending on HF in the last day."""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            # HF sort by "trending" is via their API
            models = api.list_models(
                task="text-generation",
                sort="trending",
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
            self._catalog_cache = {m.id: m for m in models}
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
        """Return catalog models annotated with download status."""
        results = []
        for info in self.get_catalog():
            entry = info.to_dict()
            entry["downloaded"] = self._is_downloaded_info(info)
            local_size = self._local_size_gb_info(info)
            if local_size is not None:
                entry["local_size_gb"] = round(local_size, 2)
            results.append(entry)
        return results

    def list_downloaded(self) -> list[dict]:
        """Scan models_dir and return info for every downloaded model."""
        downloaded: list[dict] = []
        if not self.models_dir.exists():
            return downloaded

        for child in sorted(self.models_dir.iterdir()):
            if not child.is_dir():
                continue
            catalog_entry = self._find_catalog_by_dir(child.name)
            if catalog_entry:
                entry = catalog_entry.to_dict()
                entry["downloaded"] = True
                entry["local_size_gb"] = round(self._dir_size_gb(child), 2)
                downloaded.append(entry)
            else:
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

    def _is_downloaded_info(self, info: ModelInfo) -> bool:
        d = self._model_dir_info(info)
        return d.exists() and any(d.iterdir())

    def _local_size_gb_info(self, info: ModelInfo) -> Optional[float]:
        d = self._model_dir_info(info)
        if not d.exists():
            return None
        return self._dir_size_gb(d)

    @staticmethod
    def _dir_size_gb(path: Path) -> float:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return total / (1024**3)

    def search_huggingface(self, query: str, limit: int = 50) -> list[dict]:
        """Search HuggingFace Hub for text-generation models matching the query."""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            models = api.list_models(
                search=query,
                task="text-generation",
                limit=limit,
                sort="downloads",
                direction=-1,
            )
            catalog_ids = set(self.get_catalog_map().keys())
            results = []
            for m in models:
                repo = m.id
                slug = repo.replace("/", "--").lower()
                size_gb = 0.0
                if hasattr(m, "safetensors") and m.safetensors:
                    total_params = m.safetensors.get("total", 0)
                    size_gb = (total_params * 2) / (1024 ** 3)
                card_data = getattr(m, "cardData", None)
                license_str = ""
                if isinstance(card_data, dict):
                    lic = card_data.get("license", "")
                    if isinstance(lic, str):
                        license_str = lic
                results.append({
                    "id": slug,
                    "name": repo.split("/")[-1],
                    "hf_repo": repo,
                    "size_gb": round(size_gb, 1),
                    "description": (m.pipeline_tag or "text-generation") + " model",
                    "family": repo.split("/")[0].lower(),
                    "params_b": round((size_gb / 2), 2) if size_gb > 0 else 0,
                    "context_length": 0,
                    "license": license_str,
                    "recommended": False,
                    "downloads": getattr(m, "downloads", 0),
                    "likes": getattr(m, "likes", 0),
                    "in_catalog": slug in catalog_ids,
                })
            return results
        except Exception:
            return []

    def _find_catalog_by_dir(self, dirname: str) -> Optional[ModelInfo]:
        for info in self.get_catalog():
            if ModelManager._repo_to_dirname(info.hf_repo) == dirname:
                return info
        return None
