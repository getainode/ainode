"""Pluggable engine backends. Each backend drives a specific inference-engine
stack (vLLM via eugr's launch-cluster.sh, vLLM via NVIDIA's run_cluster.sh,
or future options). All implement the EngineBackend ABC."""

from ainode.engine.backends.base import EngineBackend
from ainode.engine.backends.eugr import EugrBackend
from ainode.engine.backends.nvidia import NvidiaBackend


def get_backend(config, on_ready=None) -> EngineBackend:
    """Return the configured engine backend instance.

    Dispatches on ``config.engine_backend`` (new field). Defaults to ``"eugr"``
    for backward compatibility with existing installs.
    """
    backend = (getattr(config, "engine_backend", None) or "eugr").lower()
    if backend == "eugr":
        return EugrBackend(config, on_ready=on_ready)
    if backend == "nvidia":
        return NvidiaBackend(config, on_ready=on_ready)
    raise ValueError(
        f"Unknown engine_backend={backend!r}. Valid options: 'eugr', 'nvidia'."
    )


__all__ = ["EngineBackend", "EugrBackend", "NvidiaBackend", "get_backend"]
