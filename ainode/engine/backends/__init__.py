"""Pluggable engine backends. Each backend drives a specific inference-engine
stack (vLLM via eugr's launch-cluster.sh, vLLM via NVIDIA's run_cluster.sh,
or future options). All implement the EngineBackend ABC."""

from ainode.core.config import DEFAULT_ENGINE_BACKEND
from ainode.engine.backends.base import EngineBackend
from ainode.engine.backends.eugr import EugrBackend
from ainode.engine.backends.nvidia import NvidiaBackend


def get_backend(config, on_ready=None, instance_id="") -> EngineBackend:
    """Return the configured engine backend instance.

    Dispatches on ``config.engine_backend``, falling back to
    ``core.config.DEFAULT_ENGINE_BACKEND`` when the field is missing or empty so
    an old config.json written before the field existed lands on the same backend
    a fresh NodeConfig would. ``instance_id`` (P2-2) disambiguates per-instance
    container names for the nvidia backend.
    """
    backend = (getattr(config, "engine_backend", None) or DEFAULT_ENGINE_BACKEND).lower()
    if backend == "eugr":
        return EugrBackend(config, on_ready=on_ready)
    if backend == "nvidia":
        return NvidiaBackend(config, on_ready=on_ready, instance_id=instance_id)
    raise ValueError(
        f"Unknown engine_backend={backend!r}. Valid options: 'eugr', 'nvidia'."
    )


__all__ = ["EngineBackend", "EugrBackend", "NvidiaBackend", "get_backend"]
