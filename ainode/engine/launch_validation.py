"""Validation for caller-controlled engine images and host mounts."""

import re
from pathlib import Path


_COMPONENT = r"[a-z0-9]+(?:(?:[._]|__|[-]+)[a-z0-9]+)*"
_IMAGE = re.compile(
    rf"(?:(?:[a-z0-9]+(?:[.-][a-z0-9]+)*(?::[0-9]+)?)/)?"
    rf"{_COMPONENT}(?:/{_COMPONENT})*"
    r"(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})?"
    r"(?:@sha256:[a-f0-9]{64})?"
)


def request_image(value) -> str:
    if not isinstance(value, str) or not _IMAGE.fullmatch(value):
        raise ValueError("engine_image must be a valid image reference")
    return value


def approved_mount_roots(config=None):
    from ainode.core.config import DATASETS_DIR, MODELS_DIR
    from ainode.core.storage_policy import validate_storage_root

    roots = [getattr(config, "models_dir", None) or MODELS_DIR,
             getattr(config, "datasets_dir", None) or DATASETS_DIR]
    roots.extend(getattr(config, "allowed_mount_roots", []) or [])
    return [validate_storage_root(v) for v in roots]


def safe_host_path(value: str, *, config=None) -> Path:
    """Resolve in this node's namespace and allow only ordinary data objects."""
    from ainode.core.config import AINODE_HOME
    from ainode.core.storage_policy import validate_mount_source

    try:
        return validate_mount_source(value, approved_mount_roots(config), AINODE_HOME)
    except (TypeError, ValueError, OSError, RuntimeError):
        raise ValueError("extra_volumes host path must be a file or directory inside an approved mount root") from None


def request_volume(value: str, *, config=None) -> str:
    parts = value.split(":")
    if (len(parts) not in (2, 3) or not parts[1].startswith("/")
            or (len(parts) == 3 and parts[2] not in ("ro", "rw"))):
        raise ValueError("extra_volumes must use host:container[:ro|rw]")
    import os
    from ainode.core.config import AINODE_HOME
    original = Path(parts[0]).expanduser()
    if os.environ.get("AINODE_HOST_HOME") and (
        not original.is_relative_to(AINODE_HOME)
        or any(part.is_symlink() for part in [original, *original.parents]
               if part.is_relative_to(AINODE_HOME))
    ):
        raise ValueError("Containerized request mounts require a direct path in the mapped node home")
    host = safe_host_path(parts[0], config=config)
    return f"{host}:{parts[1]}:{parts[2] if len(parts) == 3 else 'ro'}"
