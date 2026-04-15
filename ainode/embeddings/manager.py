"""Embedding model manager.

Wraps ``sentence-transformers`` so embedding models can be lazy-loaded,
cached, and served from the AINode API. Designed to be safe to import
without the ``sentence-transformers`` dependency present — a friendly
RuntimeError is raised only when an actual load is attempted.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Catalog of known embedding models
# ---------------------------------------------------------------------------

KNOWN_EMBEDDING_MODELS: Dict[str, Dict[str, Any]] = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "id": "sentence-transformers/all-MiniLM-L6-v2",
        "hf_repo": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": 384,
        "max_seq_length": 256,
        "size_mb": 91,
        "description": (
            "Compact, fast general-purpose embedding model. Great starter — "
            "small enough to run on CPU, high quality for its size."
        ),
        "default": True,
    },
    "nomic-ai/nomic-embed-text-v1.5": {
        "id": "nomic-ai/nomic-embed-text-v1.5",
        "hf_repo": "nomic-ai/nomic-embed-text-v1.5",
        "dimensions": 768,
        "max_seq_length": 8192,
        "size_mb": 274,
        "description": (
            "Long-context (8K) embedding model from Nomic. Excellent for "
            "RAG over long documents."
        ),
    },
    "BAAI/bge-large-en-v1.5": {
        "id": "BAAI/bge-large-en-v1.5",
        "hf_repo": "BAAI/bge-large-en-v1.5",
        "dimensions": 1024,
        "max_seq_length": 512,
        "size_mb": 1340,
        "description": (
            "High-quality English embedding model from BAAI. Top performer "
            "on the MTEB retrieval benchmark."
        ),
    },
    "mixedbread-ai/mxbai-embed-large-v1": {
        "id": "mixedbread-ai/mxbai-embed-large-v1",
        "hf_repo": "mixedbread-ai/mxbai-embed-large-v1",
        "dimensions": 1024,
        "max_seq_length": 512,
        "size_mb": 670,
        "description": (
            "SOTA English embedding model from mixedbread. Balanced size "
            "and accuracy, strong on semantic search."
        ),
    },
}


_INSTALL_HINT = (
    "sentence-transformers is not installed. Install it with: "
    "pip install 'ainode[embeddings]'   or   pip install sentence-transformers"
)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class EmbeddingManager:
    """Tracks loaded embedding models and serves embedding requests."""

    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    # -- catalog --------------------------------------------------------------

    def list_known(self) -> List[Dict[str, Any]]:
        """Return the static catalog of known embedding models."""
        return [dict(v) for v in KNOWN_EMBEDDING_MODELS.values()]

    def list_loaded(self) -> List[Dict[str, Any]]:
        """Return metadata for every model currently in-memory."""
        with self._lock:
            return [dict(meta) for meta in self._metadata.values()]

    def is_loaded(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._models

    def dimensions_of(self, model_id: str) -> Optional[int]:
        with self._lock:
            meta = self._metadata.get(model_id)
        if meta and meta.get("dimensions") is not None:
            return int(meta["dimensions"])
        catalog = KNOWN_EMBEDDING_MODELS.get(model_id)
        if catalog:
            return int(catalog.get("dimensions", 0)) or None
        return None

    # -- load / unload --------------------------------------------------------

    def _resolve_SentenceTransformer(self):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(_INSTALL_HINT) from exc
        return SentenceTransformer

    def load(self, model_id: str, force: bool = False) -> Dict[str, Any]:
        """Load an embedding model into memory (blocking)."""
        with self._lock:
            if not force and model_id in self._models:
                return dict(self._metadata[model_id])

        SentenceTransformer = self._resolve_SentenceTransformer()

        logger.info("Loading embedding model %s", model_id)
        model = SentenceTransformer(model_id)

        catalog = KNOWN_EMBEDDING_MODELS.get(model_id, {})
        dims: Optional[int] = None
        try:
            dims = int(model.get_sentence_embedding_dimension())
        except Exception:
            dims = catalog.get("dimensions")

        max_seq: Optional[int] = None
        try:
            max_seq = int(getattr(model, "max_seq_length", 0)) or catalog.get("max_seq_length")
        except Exception:
            max_seq = catalog.get("max_seq_length")

        meta = {
            "id": model_id,
            "hf_repo": catalog.get("hf_repo", model_id),
            "dimensions": dims,
            "max_seq_length": max_seq,
            "size_mb": catalog.get("size_mb"),
            "description": catalog.get("description"),
            "loaded_at": time.time(),
        }

        with self._lock:
            self._models[model_id] = model
            self._metadata[model_id] = meta
        logger.info("Loaded embedding model %s (%s dims)", model_id, dims)
        return dict(meta)

    def unload(self, model_id: str) -> bool:
        with self._lock:
            existed = model_id in self._models
            self._models.pop(model_id, None)
            self._metadata.pop(model_id, None)
        if existed:
            logger.info("Unloaded embedding model %s", model_id)
        return existed

    # -- inference ------------------------------------------------------------

    def embed(self, model_id: str, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a batch of texts (blocking)."""
        if not isinstance(texts, list):
            raise TypeError("texts must be a list[str]")

        with self._lock:
            model = self._models.get(model_id)

        if model is None:
            self.load(model_id)
            with self._lock:
                model = self._models[model_id]

        vectors = model.encode(texts, convert_to_numpy=True)
        # Normalize to list[list[float]] regardless of numpy / torch / list return
        try:
            return [list(map(float, v)) for v in vectors.tolist()]
        except AttributeError:
            return [list(map(float, v)) for v in vectors]

    async def aembed(self, model_id: str, texts: List[str]) -> List[List[float]]:
        """Async wrapper — runs :meth:`embed` in the default executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed, model_id, texts)

    async def aload(self, model_id: str, force: bool = False) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.load, model_id, force)
