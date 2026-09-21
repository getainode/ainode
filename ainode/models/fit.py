"""Will this checkpoint fit on this node? Asked BEFORE the download starts.

A pull that runs out of room does not report a disk error: it dies part way,
leaves a partial blob cache behind, and the engine that was waiting for those
weights looks like an engine that crashed (#184, point 4). The node knew enough
to refuse before it started, so this module is the place that asks.

Three rules, and they are the whole contract:

* **A size nobody could learn is unknown, never zero.** A repo the Hub will not
  describe (no network, a gated repo, an answer with no file sizes) comes back
  ``None``, the verdict is "unknown", and the download proceeds. Reporting 0
  would read as "fits in nothing", which is how a download gets waved through as
  comfortably fitting a full disk.
* **The refusal names the three numbers an operator acts on**: what the
  checkpoint needs, what is free, and WHICH path was measured. The models
  directory is usually a different filesystem from the AINode home and often a
  different one from ``/``.
* **``force`` is always available.** The operator resuming a mostly complete pull,
  or pointing at a filesystem this process cannot stat correctly, knows more than
  the check does. It refuses, it does not forbid.

The size is learned from the Hub's file metadata (one call, every file's byte
size) with the safetensors dtype breakdown as a fallback, and remembered next to
the catalog cache under ``AINODE_HOME`` so the second question about the same
repo costs nothing. Only a KNOWN size is ever cached: a transient API failure
must not freeze "unknown" in for a month.

The comparison is deliberately plain: refuse when the checkpoint is bigger than
the free space, with no invented safety margin. A margin would be a number
nobody measured standing between an operator and a download.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Where learned sizes are remembered: beside ``catalog-cache.json`` in
#: ``AINODE_HOME``, resolved per call so a test (and a node whose home moved)
#: gets its own file rather than the one this module saw at import.
SIZE_CACHE_NAME = "repo-sizes.json"

#: How long a learned size is trusted. Weights under one revision do not change,
#: but a repo can be re-quantized under the same id, so this is not forever.
SIZE_CACHE_TTL_SECONDS = 30 * 24 * 3600

#: Ceiling on one Hub metadata call. The caller is a download or a launch, both
#: of which are about to take minutes, so this is generous; it exists so a Hub
#: that hangs cannot hold a request open.
HF_METADATA_TIMEOUT_SECONDS = 15.0

#: Bytes in the decimal GB the catalog quotes (``size_gb`` comes from the Hub's
#: own dtype arithmetic, which is decimal).
GB = 1_000_000_000


def _home() -> Path:
    """``AINODE_HOME`` as the process sees it NOW, not as it was at import."""
    from ainode.core.config import AINODE_HOME

    return Path(os.environ.get("AINODE_HOME") or AINODE_HOME)


def size_cache_path() -> Path:
    """The file learned repo sizes are remembered in."""
    return _home() / SIZE_CACHE_NAME


# ---------------------------------------------------------------------------
# What we know, and what we decided
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RepoSize:
    """How big a checkpoint is, and who said so.

    ``total_bytes`` is None for "nobody could say". ``source`` names the
    evidence so a report can be read without re-deriving it: ``hub_files`` (the
    Hub's per-file metadata), ``safetensors`` (its dtype breakdown),
    ``cache`` (a previous answer on this node), ``catalog`` (the entry's own
    ``size_gb``, which is an estimate for a swept entry) or ``""`` when unknown.
    """

    total_bytes: Optional[int] = None
    source: str = ""

    @property
    def known(self) -> bool:
        return self.total_bytes is not None


@dataclass(frozen=True)
class FitVerdict:
    """The decision, with every number it was made from."""

    hf_repo: str
    path: str
    needed_bytes: Optional[int]
    free_bytes: Optional[int]
    size_source: str = ""

    @property
    def known(self) -> bool:
        """True only when BOTH sides of the comparison are real numbers."""
        return self.needed_bytes is not None and self.free_bytes is not None

    @property
    def fits(self) -> Optional[bool]:
        """True / False, or None when something could not be measured."""
        if not self.known:
            return None
        return int(self.needed_bytes) <= int(self.free_bytes)

    def message(self) -> str:
        """The one line an operator reads, in every case including unknown."""
        if self.needed_bytes is None:
            return (f"{self.hf_repo}: size unknown (the Hub would not say), "
                    f"{_gb(self.free_bytes)} free on {self.path}")
        if self.free_bytes is None:
            return (f"{self.hf_repo} needs {_gb(self.needed_bytes)} and the free "
                    f"space on {self.path} could not be read")
        if self.fits:
            return (f"{self.hf_repo} needs {_gb(self.needed_bytes)} and "
                    f"{_gb(self.free_bytes)} is free on {self.path}")
        return (f"{self.hf_repo} needs {_gb(self.needed_bytes)} and only "
                f"{_gb(self.free_bytes)} is free on {self.path}")

    def to_dict(self) -> dict:
        return {
            "hf_repo": self.hf_repo,
            "path": self.path,
            "needed_bytes": self.needed_bytes,
            "needed_gb": _round_gb(self.needed_bytes),
            "free_bytes": self.free_bytes,
            "free_gb": _round_gb(self.free_bytes),
            "fits": self.fits,
            "size_source": self.size_source or None,
            "detail": self.message(),
        }


def _gb(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "an unknown amount"
    return f"{num_bytes / GB:.1f} GB"


def _round_gb(num_bytes: Optional[int]) -> Optional[float]:
    return None if num_bytes is None else round(num_bytes / GB, 1)


# ---------------------------------------------------------------------------
# Learning the size
# ---------------------------------------------------------------------------

def _load_size_cache() -> dict:
    try:
        data = json.loads(size_cache_path().read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def cached_repo_size(hf_repo: str) -> RepoSize:
    """What this node already learned about ``hf_repo``, or unknown."""
    entry = _load_size_cache().get(hf_repo)
    if not isinstance(entry, dict):
        return RepoSize()
    total = entry.get("total_bytes")
    stamped = entry.get("at")
    if isinstance(total, bool) or not isinstance(total, (int, float)) or total <= 0:
        return RepoSize()
    if isinstance(stamped, (int, float)) and \
            time.time() - float(stamped) > SIZE_CACHE_TTL_SECONDS:
        return RepoSize()
    return RepoSize(total_bytes=int(total), source="cache")


def remember_repo_size(hf_repo: str, size: RepoSize) -> None:
    """Write a KNOWN size next to the catalog cache. Never raises, never
    caches an unknown: a Hub outage must not be remembered as a fact."""
    if not size.known or not hf_repo:
        return
    data = _load_size_cache()
    data[hf_repo] = {"total_bytes": int(size.total_bytes),
                     "source": size.source or "",
                     "at": round(time.time())}
    path = size_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(path)
    except OSError as exc:
        logger.debug("could not remember the size of %s: %s", hf_repo, exc)


def _files_total_bytes(info: Any) -> Optional[int]:
    """Sum of every file's size from a Hub ``model_info(files_metadata=True)``.

    None when not one file carried a size, which is the unknown case: a zero
    here would be published as "this repo is empty".
    """
    total = 0
    counted = 0
    for sibling in (getattr(info, "siblings", None) or []):
        size = getattr(sibling, "size", None)
        if size is None:
            lfs = getattr(sibling, "lfs", None)
            if isinstance(lfs, dict):
                size = lfs.get("size")
            else:
                size = getattr(lfs, "size", None)
        if isinstance(size, bool) or not isinstance(size, (int, float)) or size <= 0:
            continue
        total += int(size)
        counted += 1
    return total if counted else None


def _safetensors_total_bytes(info: Any) -> Optional[int]:
    """Size from the Hub's safetensors dtype breakdown, or None.

    The fallback for a repo whose file metadata carries no sizes. The dtype
    arithmetic has one home, ``registry._safetensors_size_gb``, so a new dtype
    is added there and both callers get it.
    """
    safetensors = getattr(info, "safetensors", None)
    if safetensors is None:
        return None
    try:
        from ainode.models.registry import _safetensors_size_gb

        gb = _safetensors_size_gb(safetensors)
    except Exception:
        return None
    if not gb or gb <= 0:
        return None
    return int(round(gb * GB))


def fetch_repo_size(hf_repo: str, token: Optional[str] = None) -> RepoSize:
    """Ask the Hub how big ``hf_repo`` is. Unknown on any failure.

    The module-level seam every caller goes through, so a test never needs the
    network and a node with no route to huggingface.co reports unknown instead
    of refusing a download it has no evidence against.
    """
    if not hf_repo:
        return RepoSize()
    try:
        from huggingface_hub import HfApi
    except Exception:
        return RepoSize()
    if token is None:
        token = os.environ.get("HF_TOKEN") or \
            os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    try:
        api = HfApi(token=token)
        info = api.model_info(hf_repo, files_metadata=True,
                              timeout=HF_METADATA_TIMEOUT_SECONDS)
    except Exception as exc:
        logger.info("could not learn the size of %s: %s", hf_repo, exc)
        return RepoSize()
    total = _files_total_bytes(info)
    if total is not None:
        return RepoSize(total_bytes=total, source="hub_files")
    total = _safetensors_total_bytes(info)
    if total is not None:
        return RepoSize(total_bytes=total, source="safetensors")
    return RepoSize()


def repo_size(hf_repo: str, *, catalog_size_gb: Optional[float] = None,
              learn: bool = True) -> RepoSize:
    """How big ``hf_repo`` is, from the cheapest source that knows.

    Order: this node's remembered answer, then the Hub (when ``learn``), then
    the catalog entry's own ``size_gb``. ``learn=False`` is for a caller that
    must not pay an HTTP round trip (a launch), and it still gets the cached and
    catalog answers.
    """
    cached = cached_repo_size(hf_repo)
    if cached.known:
        return cached
    if learn:
        learned = fetch_repo_size(hf_repo)
        if learned.known:
            remember_repo_size(hf_repo, learned)
            return learned
    if isinstance(catalog_size_gb, (int, float)) and \
            not isinstance(catalog_size_gb, bool) and catalog_size_gb > 0:
        return RepoSize(total_bytes=int(round(float(catalog_size_gb) * GB)),
                        source="catalog")
    return RepoSize()


# ---------------------------------------------------------------------------
# Free space, and the verdict
# ---------------------------------------------------------------------------

def free_bytes(path) -> Optional[int]:
    """Free bytes on the filesystem holding ``path``, or None.

    A models directory that does not exist yet is answered for by the nearest
    parent that does: the download would create it on that filesystem, so that
    is the filesystem to measure. None only when nothing in the chain can be
    stat'ed, which is "unknown" rather than "full".
    """
    try:
        candidate = Path(path).resolve()
    except (OSError, ValueError):
        candidate = Path(str(path))
    for parent in [candidate, *candidate.parents]:
        try:
            return int(shutil.disk_usage(str(parent)).free)
        except (OSError, ValueError):
            continue
    return None


def check_fit(hf_repo: str, models_dir, *, catalog_size_gb: Optional[float] = None,
              learn: bool = True) -> FitVerdict:
    """Decide whether ``hf_repo`` fits in ``models_dir``. Never raises."""
    size = repo_size(hf_repo, catalog_size_gb=catalog_size_gb, learn=learn)
    return FitVerdict(hf_repo=hf_repo, path=str(models_dir),
                      needed_bytes=size.total_bytes,
                      free_bytes=free_bytes(models_dir),
                      size_source=size.source)


#: What a refusal tells the caller to do instead. One sentence, because it is
#: read in a toast.
FORCE_HINT = 'send {"force": true} to download it anyway'


def refusal_payload(verdict: FitVerdict) -> dict:
    """The body of a refused download: the numbers, the path, and the way past."""
    return {
        "error": f"not enough free space: {verdict.message()}",
        "fit": verdict.to_dict(),
        "force": FORCE_HINT,
    }
