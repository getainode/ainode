"""Local operator policy for storage roots and request mounts."""
import os
from pathlib import Path

from ainode.core import config as node_config

STORAGE_FIELDS = {"models_dir", "datasets_dir", "training_dir", "hf_cache_dir"}
_SYSTEM = {"/", "/etc", "/usr", "/var", "/root", "/proc", "/sys", "/dev", "/boot", "/home"}
_SECRET_PARTS = {".ssh", ".aws", ".docker", ".gnupg"}


def validate_storage_root(value) -> Path:
    """Reject broad/system roots and ancestors of node credentials."""
    try:
        original = Path(value).expanduser()
        if not original.is_absolute():
            raise ValueError
        path = original.resolve()
        if str(path) in _SYSTEM or any(
            path.is_relative_to(base) for base in ("/etc", "/usr", "/proc", "/sys", "/dev", "/boot")
        ):
            raise ValueError
        if _SECRET_PARTS.intersection(path.parts) or any(
            home.resolve().is_relative_to(path) for home in (
                node_config.AINODE_HOME, Path(os.environ.get("AINODE_HOST_HOME", str(node_config.AINODE_HOME)))
            )
        ):
            raise ValueError
        for secret in _SECRET_PARTS:
            if (path / secret).exists():
                raise ValueError
        return path
    except (TypeError, ValueError, OSError, RuntimeError):
        raise ValueError("Storage root must be a dedicated data directory without node secrets") from None


def sanitize_storage(config):
    """Bad persisted policy falls back before any manager or engine uses it."""
    import logging

    defaults = type(config)()
    for field in STORAGE_FIELDS:
        value = getattr(config, field)
        if value is not None:
            try:
                validate_storage_root(value)
            except ValueError:
                logging.getLogger(__name__).warning("Invalid %s; using the default storage root", field)
                setattr(config, field, getattr(defaults, field))
    roots = getattr(config, "allowed_mount_roots", [])
    try:
        if not isinstance(roots, list):
            raise ValueError
        config.allowed_mount_roots = [str(validate_storage_root(v)) for v in roots]
    except ValueError:
        logging.getLogger(__name__).warning("Invalid allowed_mount_roots; using no extra roots")
        config.allowed_mount_roots = []


def validate_mount_source(value, roots, secret_home):
    """Portable check, also executed on each SSH peer before docker run."""
    import stat
    from pathlib import Path

    original = Path(value).expanduser()
    if not original.is_absolute():
        raise ValueError("Mount source must be absolute")
    path = original.resolve(strict=True)
    mode = path.stat().st_mode
    if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
        raise ValueError("Mount source must be a regular file or directory")
    home = Path(secret_home).resolve()
    protected = ("/etc", "/usr", "/proc", "/sys", "/dev", "/boot")
    if (str(path) in ("/", "/var", "/root", "/home")
            or any(path.is_relative_to(base) for base in protected)
            or home.is_relative_to(path)
            or {".ssh", ".aws", ".docker", ".gnupg"}.intersection(path.parts)):
        raise ValueError("Mount source contains system files or credentials")
    if any((path / name).exists() for name in (".ssh", ".aws", ".docker", ".gnupg")):
        raise ValueError("Mount source contains credentials")
    approved = [Path(root).expanduser().resolve() for root in roots]
    if any(str(root) in ("/", "/etc", "/usr", "/var", "/root", "/proc", "/sys", "/dev", "/boot", "/home")
           or home.is_relative_to(root) for root in approved):
        raise ValueError("Invalid approved mount root")
    if not any(path.is_relative_to(root) for root in approved):
        raise ValueError("Mount source must be inside an approved mount root")
    return path
