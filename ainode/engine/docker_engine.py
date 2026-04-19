"""Deprecated shim — moved to :mod:`ainode.engine.backends.eugr`.

Will be removed in v0.6.0. Update imports to
``from ainode.engine.backends import EugrBackend, get_backend``.
"""

from __future__ import annotations

import warnings

from ainode.engine.backends.eugr import (
    EUGR_ENV_FILE,
    EUGR_LAUNCHER,
    NCCL_INIT_CONTAINER_PATH,
    NCCL_INIT_IMAGE_PATH,
    NCCL_INIT_SHARED_DIR,
    NCCL_INIT_SHARED_PATH,
    DockerEngine,
    DockerEngineError,
    EugrBackend,
    EugrBackendError,
    _local_ip_for_interface,
)


def build_engine(config, on_ready=None):
    """Deprecated — use :func:`ainode.engine.backends.get_backend` instead."""
    warnings.warn(
        "ainode.engine.docker_engine.build_engine is deprecated; "
        "use ainode.engine.backends.get_backend instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from ainode.engine.backends import get_backend as _gb
    return _gb(config, on_ready=on_ready)


__all__ = [
    "EUGR_ENV_FILE",
    "EUGR_LAUNCHER",
    "NCCL_INIT_CONTAINER_PATH",
    "NCCL_INIT_IMAGE_PATH",
    "NCCL_INIT_SHARED_DIR",
    "NCCL_INIT_SHARED_PATH",
    "DockerEngine",
    "DockerEngineError",
    "EugrBackend",
    "EugrBackendError",
    "build_engine",
]
