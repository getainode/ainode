"""NvidiaBackend — drive vLLM via NVIDIA's official nvcr.io/nvidia/vllm image.

Phase 4 implementation. This file is a skeleton in Phase 3 only to validate
the abstraction surface of :class:`ainode.engine.backends.base.EngineBackend`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Optional

from ainode.core.config import NodeConfig
from ainode.engine.backends.base import EngineBackend


_PHASE4_MSG = (
    "NvidiaBackend will be implemented in nvidia-vllm-engine Phase 4. "
    "For now, use engine_backend='eugr' in your config."
)


class NvidiaBackend(EngineBackend):
    """Placeholder backend for NVIDIA's nvcr.io/nvidia/vllm image.

    Every method raises :class:`NotImplementedError` until Phase 4 wires in
    ``run_cluster.sh`` orchestration. Provided here so :func:`get_backend`
    can dispatch on ``engine_backend='nvidia'`` without import errors.
    """

    def __init__(self, config: NodeConfig, on_ready: Optional[Callable] = None):
        self.config = config
        self.on_ready = on_ready
        # Phase 4 TODO: real state init (process handle, log paths, etc.)

    def start(self) -> bool:
        raise NotImplementedError(_PHASE4_MSG)

    def stop(self) -> None:
        raise NotImplementedError(_PHASE4_MSG)

    def wait_ready(self, timeout: float = 600.0) -> bool:
        raise NotImplementedError(_PHASE4_MSG)

    def is_running(self) -> bool:
        return False

    def health_check(self) -> dict:
        return {
            "process_alive": False,
            "api_responding": False,
            "models_loaded": [],
            "status": "not_implemented",
        }

    @property
    def api_url(self) -> str:
        return ""

    @property
    def log_path(self) -> Path:
        return Path("/tmp/nvidia-backend-not-implemented.log")

    @property
    def process(self) -> Optional[subprocess.Popen]:
        return None
