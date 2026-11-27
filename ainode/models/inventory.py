"""Read model artifacts from the node's configured stores, without consulting the Hub."""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path

from ainode.core.storage_policy import validate_storage_root
from ainode.models.registry import model_store_roots

_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
_SHARD = re.compile(r"^(.+)-(\d{5})-of-(\d{5})(\.[^.]+)$")
_STORAGE_DIRS = {"hub", "hf-cache", "blobs", "refs", "snapshots", "xet"}
_CONFIG_NAMES = {"config.json", "params.json"}
_INDEX_SUFFIXES = (".safetensors.index.json", ".bin.index.json")


@dataclass
class ModelCopy:
    id: str
    layout: str
    status: str
    revision: str = ""
    file: str = ""
    loaded: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _confined_path(path: Path, boundary: Path) -> Path | None:
    try:
        resolved = path.resolve(strict=True)
        return resolved if resolved.is_relative_to(boundary) else None
    except (FileNotFoundError, RuntimeError):
        return None


def _entries(directory: Path, boundary: Path) -> list[Path]:
    """List an optional directory, confined to its configured store."""
    resolved = _confined_path(directory, boundary)
    return list(directory.iterdir()) if resolved is not None and resolved.is_dir() else []


def _model_file(path: Path, boundary: Path) -> tuple[Path, os.stat_result] | None:
    """HF snapshot symlinks may resolve to blobs in the same repository."""
    target = _confined_path(path, boundary)
    if target is None or target.name.endswith(".incomplete"):
        return None
    try:
        info = target.stat()
        if stat.S_ISREG(info.st_mode):
            return target, info
    except FileNotFoundError:
        pass
    return None


def _json_object(path: Path, boundary: Path) -> dict | None:
    model_file = _model_file(path, boundary)
    if model_file is None:
        return None
    target, info = model_file
    limit = 32 * 1024**2
    if not 0 < info.st_size <= limit:
        return None
    try:
        with target.open("rb") as stream:
            data = json.loads(stream.read(limit + 1))
    except (FileNotFoundError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _is_weight(path: Path) -> bool:
    return (path.suffix.lower() in _WEIGHT_SUFFIXES
            and not path.name.startswith(("adapter_", "optimizer", "training_args", "rng_state")))


def _shard_groups(paths: list[Path]) -> dict[str, tuple[set[int], int]]:
    groups: dict[str, tuple[set[int], int]] = {}
    for path in paths:
        match = _SHARD.fullmatch(path.name)
        if match:
            stem, index, count, suffix = match.groups()
            name = f"{stem}-of-{count}{suffix}"
            indices, total = groups.setdefault(name, (set(), int(count)))
            indices.add(int(index))
    return groups


def _all_shards(indices: set[int], total: int) -> bool:
    return total > 0 and len(indices) == total and min(indices) == 1 and max(indices) == total


def _nonempty_file(path: Path, boundary: Path) -> bool:
    model_file = _model_file(path, boundary)
    return model_file is not None and model_file[1].st_size > 0


def _gguf_copies(files: list[Path], boundary: Path, layout: str,
                 model_id: str = "", revision: str = "") -> list[ModelCopy]:
    present = {file for file in files if _nonempty_file(file, boundary)}
    groups = _shard_groups(list(present))
    variants = {}
    for file in files:
        name, complete = file.name, file in present
        match = _SHARD.fullmatch(name)
        if match:
            stem, _, count, suffix = match.groups()
            name = f"{stem}-of-{count}{suffix}"
            complete = name in groups and _all_shards(*groups[name])
        variants[name] = ModelCopy(model_id or name, layout,
                                   "available" if complete else "incomplete",
                                   revision, name if model_id else "")
    return list(variants.values())


def _checkpoint_copies(path: Path, boundary: Path, model_id: str, layout: str,
                       revision: str = "", *, expected: bool = True) -> list[ModelCopy]:
    entries = _entries(path, boundary)
    weights = [item for item in entries if _is_weight(item)]
    indexes = [item for item in entries if item.name.endswith(_INDEX_SUFFIXES)]
    configs = [entry for entry in entries if entry.name in _CONFIG_NAMES]
    found = (weights or indexes or configs
             or _confined_path(path / ".cache" / "huggingface" / "download", boundary))
    if not found:
        return [ModelCopy(model_id, layout, "incomplete", revision)] if expected else []

    def raise_walk_error(error: OSError) -> None:
        if not isinstance(error, FileNotFoundError):
            raise error

    # Local Hub downloads put partial files in a hidden subdirectory. Snapshot
    # links can also lose their blobs after an external cleanup.
    partial = False
    for parent, dirs, files in os.walk(path, followlinks=False, onerror=raise_walk_error):
        dirs[:] = [name for name in dirs if name != ".git" and name not in _STORAGE_DIRS]
        if any(name.endswith(".incomplete") for name in files):
            partial = True
            break
        if any(item.is_symlink() and _model_file(item, boundary) is None
               for item in (Path(parent) / name for name in chain(dirs, files))):
            partial = True
            break

    gguf = [item for item in weights if item.suffix.lower() == ".gguf"]
    copies = _gguf_copies(gguf, boundary, layout, model_id, revision)

    weights = [item for item in weights if item not in gguf]
    if not copies or weights or indexes:
        complete = (any(_json_object(config, boundary) is not None for config in configs)
                    and bool(weights or indexes) and not partial
                    and all(_nonempty_file(item, boundary) for item in weights))
        for index in indexes:
            data = _json_object(index, boundary) or {}
            weight_map = data.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                complete = False
                continue
            values = [name for name in weight_map.values() if isinstance(name, str)]
            complete = (complete and len(values) == len(weight_map)
                        and all(name and not Path(name).is_absolute()
                                and ".." not in Path(name).parts
                                and _nonempty_file(path / name, boundary)
                                for name in set(values)))
        complete = complete and all(_all_shards(*group)
                                    for group in _shard_groups(weights).values())
        copies.insert(0, ModelCopy(model_id, layout, "available" if complete else "incomplete",
                                   revision))
    return copies


def _cache_copies(repo: Path, model_id: str, layout: str) -> list[ModelCopy]:
    copies = []
    for revision in _entries(repo / "snapshots", repo):
        snapshot = _confined_path(revision, repo)
        if snapshot is not None and snapshot.is_dir():
            copies.extend(_checkpoint_copies(snapshot, repo, model_id, layout, revision.name))
    flat = _checkpoint_copies(repo, repo, model_id, layout, expected=False)
    copies.extend(flat)
    # A cache can have a completed snapshot and a separate in-flight blob.
    blobs = _entries(repo / "blobs", repo)
    if any(blob.name.endswith(".incomplete") for blob in blobs):
        copies.append(ModelCopy(model_id, layout, "incomplete"))
    elif not copies:
        status = "missing" if _entries(repo / "refs", repo) else "incomplete"
        copies.append(ModelCopy(model_id, layout, status))
    return copies


def scan_model_copies(models_dir: Path, hf_cache_dir: str | None = None) -> list[ModelCopy]:
    """Scan the stores AINode manages, including complete unregistered checkpoints.

    Only AINode's flat store and known Hugging Face cache roots are enumerated.
    A failing store scan is allowed to raise so callers cannot report it as empty.
    """
    store = validate_storage_root(models_dir).resolve(strict=True)
    if not store.is_dir():
        raise NotADirectoryError("Configured models store is not a directory")
    roots = [(path, store, layout) for path, layout in zip(
        model_store_roots(store), ("Model store", "Hub cache", "Engine cache")
    )]
    if hf_cache_dir:
        cache = validate_storage_root(hf_cache_dir)
        if cache.exists() and not cache.is_dir():
            raise NotADirectoryError("Configured engine cache is not a directory")
        roots.extend([(cache, cache, "Engine cache"), (cache / "hub", cache, "Engine cache")])

    copies = []
    seen = set()
    gguf = []
    for root, boundary, layout in roots:
        for child in _entries(root, boundary):
            name = child.name
            if name.startswith(".") or name in _STORAGE_DIRS or child in seen:
                continue
            seen.add(child)
            target = _confined_path(child, boundary)
            if target is None or target == boundary:
                continue
            if name.startswith("models--") and target.is_dir():
                model_id = name.removeprefix("models--").replace("--", "/", 1)
                copies.extend(_cache_copies(target, model_id, layout if root != store else "Flat HF cache"))
            elif root == store and target.is_dir():
                model_id = name.replace("--", "/", 1)
                rows = _checkpoint_copies(target, target, model_id, layout, expected="--" in name)
                copies.extend(rows)
                if not rows:
                    # The snapshot resolver also accepts literal org/name directories.
                    for nested in _entries(target, store):
                        subdir = _confined_path(nested, store)
                        if not nested.name.startswith(".") and subdir and subdir != store and subdir.is_dir():
                            copies.extend(_checkpoint_copies(subdir, subdir,
                                                             f"{name}/{nested.name}", "Nested store",
                                                             expected=False))
            elif root == store and child.suffix.lower() == ".gguf" and target.is_file():
                gguf.append(child)
    copies.extend(_gguf_copies(gguf, store, "Model store"))
    return sorted(copies, key=lambda row: (row.id.casefold(), row.id, row.layout, row.revision, row.file))
