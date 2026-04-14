"""Dataset manager — register, store and preview training datasets.

Datasets live under ``AINODE_HOME/datasets/`` with a small JSON registry
(``_registry.json``) at the root of that directory. Each dataset gets its own
subdirectory keyed by a short ID.

Supported sources:
  * ``upload``      — user uploaded a file (JSON, JSONL, CSV, Parquet)
  * ``huggingface`` — reference to a HuggingFace dataset repo id
  * ``local``       — path on disk the user already has (we reference it)
  * ``url``         — remote file we downloaded

This module is intentionally small and synchronous — the API layer in
``ainode.datasets.api_routes`` wraps it for the web UI. Network I/O (HF /
URL downloads) is done lazily inside each ``add_*`` method.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

from ainode.core.config import AINODE_HOME

DATASETS_DIR = AINODE_HOME / "datasets"
REGISTRY_FILE = DATASETS_DIR / "_registry.json"

# File types we know how to preview / count
_ALLOWED_EXT = {".json", ".jsonl", ".csv", ".tsv", ".parquet", ".txt"}
# Reasonable upper bound on uploads — 2 GB. Tune as needed.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024


class DatasetSource(str, Enum):
    UPLOAD = "upload"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    URL = "url"


class DatasetFormat(str, Enum):
    JSONL = "jsonl"
    JSON = "json"
    CSV = "csv"
    TSV = "tsv"
    PARQUET = "parquet"
    TEXT = "text"
    UNKNOWN = "unknown"


def _format_from_path(path: Path) -> DatasetFormat:
    ext = path.suffix.lower()
    return {
        ".jsonl": DatasetFormat.JSONL,
        ".json": DatasetFormat.JSON,
        ".csv": DatasetFormat.CSV,
        ".tsv": DatasetFormat.TSV,
        ".parquet": DatasetFormat.PARQUET,
        ".txt": DatasetFormat.TEXT,
    }.get(ext, DatasetFormat.UNKNOWN)


@dataclass
class Dataset:
    """Metadata for a single training dataset."""

    id: str
    name: str
    source: str  # DatasetSource value
    format: str  # DatasetFormat value
    path: str  # absolute path or HF repo id
    samples: int = 0
    size_bytes: int = 0
    created_at: float = field(default_factory=time.time)
    description: str = ""
    # For HF datasets: which config/split was selected (optional)
    hf_config: Optional[str] = None
    hf_split: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Dataset":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


class DatasetManager:
    """Register and look up datasets by ID. Registry is JSON on disk."""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else DATASETS_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.root / "_registry.json"
        self._datasets: dict[str, Dataset] = {}
        self._load()

    # ------------------------------------------------------------------
    # Registry I/O
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self._registry_path.exists():
            self._datasets = {}
            return
        try:
            raw = json.loads(self._registry_path.read_text())
            self._datasets = {
                d["id"]: Dataset.from_dict(d) for d in raw.get("datasets", [])
            }
        except (json.JSONDecodeError, OSError, KeyError):
            self._datasets = {}

    def _save(self) -> None:
        payload = {"datasets": [d.to_dict() for d in self._datasets.values()]}
        tmp = self._registry_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self._registry_path)

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------
    def list(self) -> list[dict]:
        return [d.to_dict() for d in sorted(
            self._datasets.values(), key=lambda d: d.created_at, reverse=True
        )]

    def get(self, dataset_id: str) -> Optional[Dataset]:
        return self._datasets.get(dataset_id)

    def delete(self, dataset_id: str) -> bool:
        ds = self._datasets.pop(dataset_id, None)
        if ds is None:
            return False
        # Clean up files we own (not user-supplied local paths)
        if ds.source in (DatasetSource.UPLOAD.value, DatasetSource.URL.value):
            ds_dir = self.root / ds.id
            if ds_dir.exists() and ds_dir.is_dir():
                shutil.rmtree(ds_dir, ignore_errors=True)
        self._save()
        return True

    def preview(self, dataset_id: str, limit: int = 3) -> dict:
        ds = self.get(dataset_id)
        if ds is None:
            raise KeyError(f"Dataset not found: {dataset_id}")
        samples: list = []
        if ds.source == DatasetSource.HUGGINGFACE.value:
            samples = [{"note": "Preview unavailable for HF datasets without the datasets lib."}]
        else:
            p = Path(ds.path)
            if p.exists() and p.is_file():
                samples = list(_read_samples(p, limit=limit))
        return {
            "id": ds.id,
            "name": ds.name,
            "format": ds.format,
            "samples": samples,
            "total_samples": ds.samples,
        }

    # ------------------------------------------------------------------
    # add_* methods
    # ------------------------------------------------------------------
    def add_upload(
        self, filename: str, content: bytes, name: Optional[str] = None,
        description: str = "",
    ) -> Dataset:
        """Register an uploaded file (bytes already in memory)."""
        safe_name = _safe_filename(filename)
        ext = Path(safe_name).suffix.lower()
        if ext not in _ALLOWED_EXT:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {', '.join(sorted(_ALLOWED_EXT))}"
            )
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"File too large ({len(content)} bytes, max {MAX_UPLOAD_BYTES})"
            )
        ds_id = _new_id()
        ds_dir = self.root / ds_id
        ds_dir.mkdir(parents=True, exist_ok=True)
        target = ds_dir / safe_name
        target.write_bytes(content)

        fmt = _format_from_path(target)
        samples = _count_samples(target, fmt)
        ds = Dataset(
            id=ds_id,
            name=name or Path(safe_name).stem,
            source=DatasetSource.UPLOAD.value,
            format=fmt.value,
            path=str(target),
            samples=samples,
            size_bytes=target.stat().st_size,
            description=description,
        )
        self._datasets[ds_id] = ds
        self._save()
        return ds

    def add_huggingface(
        self, repo_id: str, name: Optional[str] = None,
        config: Optional[str] = None, split: Optional[str] = None,
        description: str = "",
    ) -> Dataset:
        """Register a HuggingFace dataset reference (no download here)."""
        repo_id = (repo_id or "").strip()
        if not repo_id or "/" not in repo_id:
            raise ValueError("HuggingFace repo_id must look like 'org/name'")
        ds_id = _new_id()
        ds = Dataset(
            id=ds_id,
            name=name or repo_id,
            source=DatasetSource.HUGGINGFACE.value,
            format=DatasetFormat.UNKNOWN.value,
            path=repo_id,
            hf_config=config,
            hf_split=split,
            description=description,
        )
        self._datasets[ds_id] = ds
        self._save()
        return ds

    def add_local(
        self, path: str, name: Optional[str] = None, description: str = "",
    ) -> Dataset:
        """Register a path that already exists on disk (we don't copy)."""
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise ValueError(f"Path does not exist: {p}")
        if not p.is_file():
            raise ValueError(f"Path must be a regular file: {p}")
        fmt = _format_from_path(p)
        samples = _count_samples(p, fmt)
        ds_id = _new_id()
        ds = Dataset(
            id=ds_id,
            name=name or p.stem,
            source=DatasetSource.LOCAL.value,
            format=fmt.value,
            path=str(p),
            samples=samples,
            size_bytes=p.stat().st_size,
            description=description,
        )
        self._datasets[ds_id] = ds
        self._save()
        return ds

    def add_url(
        self, url: str, name: Optional[str] = None, description: str = "",
        timeout: float = 30.0,
    ) -> Dataset:
        """Download a file from a URL and register it."""
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("URL must be http(s)")
        filename = Path(parsed.path).name or "dataset.jsonl"
        safe_name = _safe_filename(filename)
        if Path(safe_name).suffix.lower() not in _ALLOWED_EXT:
            # default to .jsonl when the URL doesn't carry a usable extension
            safe_name = safe_name + ".jsonl"

        ds_id = _new_id()
        ds_dir = self.root / ds_id
        ds_dir.mkdir(parents=True, exist_ok=True)
        target = ds_dir / safe_name

        req = urllib.request.Request(url, headers={"User-Agent": "AINode/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - user-provided URL
            bytes_written = 0
            with open(target, "wb") as fh:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > MAX_UPLOAD_BYTES:
                        fh.close()
                        target.unlink(missing_ok=True)
                        raise ValueError(
                            f"Download exceeded max size ({MAX_UPLOAD_BYTES} bytes)"
                        )
                    fh.write(chunk)

        fmt = _format_from_path(target)
        samples = _count_samples(target, fmt)
        ds = Dataset(
            id=ds_id,
            name=name or Path(safe_name).stem,
            source=DatasetSource.URL.value,
            format=fmt.value,
            path=str(target),
            samples=samples,
            size_bytes=target.stat().st_size,
            description=description or url,
        )
        self._datasets[ds_id] = ds
        self._save()
        return ds


# =============================================================================
# Helpers
# =============================================================================

def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _safe_filename(name: str) -> str:
    """Reduce filename to a safe basename — strip path separators + hidden dots."""
    base = Path(name).name.strip()
    if not base or base.startswith(".."):
        base = "dataset"
    # Replace whitespace / odd chars conservatively
    allowed = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    cleaned = "".join(c if c in allowed else "_" for c in base)
    return cleaned.replace(" ", "_") or "dataset"


def _count_samples(path: Path, fmt: DatasetFormat) -> int:
    """Fast sample count; gracefully handles errors."""
    try:
        if fmt == DatasetFormat.JSONL:
            n = 0
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.strip():
                        n += 1
            return n
        if fmt == DatasetFormat.JSON:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, list):
                return len(data)
            if isinstance(data, dict):
                # Common HF pattern: {"data": [...]} or {"train": [...]}
                for v in data.values():
                    if isinstance(v, list):
                        return len(v)
                return 1
            return 0
        if fmt in (DatasetFormat.CSV, DatasetFormat.TSV):
            delim = "\t" if fmt == DatasetFormat.TSV else ","
            with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
                reader = csv.reader(fh, delimiter=delim)
                rows = sum(1 for _ in reader)
            # Subtract header row
            return max(0, rows - 1)
        if fmt == DatasetFormat.TEXT:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return sum(1 for line in fh if line.strip())
    except Exception:
        return 0
    return 0


def _read_samples(path: Path, limit: int = 3) -> Iterable[dict]:
    """Yield up to ``limit`` sample records as dicts for preview."""
    fmt = _format_from_path(path)
    try:
        if fmt == DatasetFormat.JSONL:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    if i >= limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        yield {"_raw": line[:500]}
            return
        if fmt == DatasetFormat.JSON:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            items: list = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        items = v
                        break
                else:
                    items = [data]
            for item in items[:limit]:
                yield item if isinstance(item, dict) else {"_value": item}
            return
        if fmt in (DatasetFormat.CSV, DatasetFormat.TSV):
            delim = "\t" if fmt == DatasetFormat.TSV else ","
            with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
                reader = csv.DictReader(fh, delimiter=delim)
                for i, row in enumerate(reader):
                    if i >= limit:
                        break
                    yield dict(row)
            return
        if fmt == DatasetFormat.TEXT:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    if i >= limit:
                        break
                    yield {"text": line.rstrip("\n")}
            return
    except Exception as exc:
        yield {"_error": f"preview failed: {exc}"}
        return
    yield {"_note": "Preview not available for this format."}
