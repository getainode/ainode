"""EngineBackend abstract base class.

Defines the contract every concrete inference-engine backend must honor.
Concrete implementations (eugr, nvidia, ...) inherit from this and provide
their own lifecycle and orchestration logic.
"""

from __future__ import annotations

import abc
import subprocess
from pathlib import Path
from typing import Callable, Optional

from ainode.core.config import NodeConfig


class EngineBackend(abc.ABC):
    """Abstract contract for an AINode inference-engine backend.

    Concrete backends drive a specific engine stack (vLLM via eugr's
    launcher, vLLM via NVIDIA's nvcr.io image, etc.) but all expose the
    same lifecycle surface so ``cmd_start``/``cmd_status`` can dispatch
    polymorphically.
    """

    @abc.abstractmethod
    def __init__(
        self,
        config: NodeConfig,
        on_ready: Optional[Callable] = None,
    ) -> None:
        """Bind the backend to a node config and optional readiness callback."""

    @abc.abstractmethod
    def start(self) -> bool:
        """Kick off the engine in the mode dictated by ``config.distributed_mode``.

        Returns True if the launch sequence was initiated successfully;
        False if validation failed or the subprocess could not spawn.
        Callers should invoke :meth:`wait_ready` afterwards to block until
        the API is actually serving.
        """

    @abc.abstractmethod
    def stop(self) -> None:
        """Gracefully shut down the engine and any spawned children."""

    @abc.abstractmethod
    def wait_ready(self, timeout: float = 600.0) -> bool:
        """Block until the engine's API answers, or ``timeout`` seconds elapse.

        Returns True if the engine became ready, False on timeout or crash.
        """

    @abc.abstractmethod
    def is_running(self) -> bool:
        """Return True iff the engine's primary process is currently alive."""

    @abc.abstractmethod
    def health_check(self) -> dict:
        """Return a dict describing engine health.

        Expected shape::

            {
                "process_alive": bool,
                "api_responding": bool,
                "models_loaded": list[str],
            }
        """

    @property
    @abc.abstractmethod
    def api_url(self) -> str:
        """Return the OpenAI-compatible base URL (e.g. ``http://localhost:8000/v1``)."""

    @property
    @abc.abstractmethod
    def log_path(self) -> Path:
        """Return the path to the primary log file for this backend."""

    @property
    @abc.abstractmethod
    def process(self) -> Optional[subprocess.Popen]:
        """Return the primary subprocess handle, or None if not running."""
