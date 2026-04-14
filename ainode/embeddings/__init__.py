"""Embedding model support for AINode.

Provides a lightweight, in-process embedding engine powered by
``sentence-transformers``. Embedding models are first-class citizens
alongside LLMs — they can be loaded, unloaded, and invoked via an
OpenAI-compatible ``/v1/embeddings`` endpoint.
"""

from ainode.embeddings.manager import EmbeddingManager, KNOWN_EMBEDDING_MODELS

__all__ = ["EmbeddingManager", "KNOWN_EMBEDDING_MODELS"]
