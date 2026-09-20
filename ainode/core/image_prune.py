"""Which AINode images an update may delete, and what deleting them frees.

``ainode update`` pulled a new image on every release and removed nothing, so a
node accumulated one 1.5 GB orchestrator image per release under each of the
three names the image is mirrored as. Spark-1 reached 180 images, 226 GB
reclaimable, on a filesystem at 82 percent (#184). Nothing in the product
reported it and nothing reclaimed it.

The rules, in one place so the decision can be read and tested without a docker
daemon anywhere near it:

* Only AINode's own app-image repositories are ever considered. ``ainode-base``,
  engine images (``vllm/vllm-openai`` and friends) and anything else on the host
  are not this code's business: a node's engine images are what it serves models
  with, and deleting one costs a multi-gigabyte pull.
* A tag that is not a release (``latest``, ``dev``, a sha) is never removed. It
  is somebody's pointer and its bytes are shared with a release tag anyway.
* The pinned release is kept, and so is anything NEWER than it: an operator who
  pre-pulled the next release has not asked us to undo that.
* ``keep`` generations below the pinned release are kept, counted in distinct
  RELEASES across all of those repositories together, so the default of 1 leaves
  exactly one rollback generation with every name it has. ``ainode update
  <older>`` then has something to roll back TO, which is what makes that
  command's promise true.
* Anything else is removed BY TAG, never by image id: the same image is tagged
  under three repositories, and ``docker rmi <id>`` would take all three at once,
  including a tag this node is running on.
* Nothing is decided at all when the pinned release is not in the listing. That
  means the pull did not land, or the wrong listing was handed in, and pruning
  against an unknown baseline is how a node loses the image it is running.

The freed figure is an upper bound and says so: ``docker images`` reports each
image's full size including layers it shares with others, so summing them
overcounts whatever was shared.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

# The repositories one AINode release is published or tagged under. Everything
# else on the host is left alone.
AINODE_IMAGE_REPOS = (
    "ghcr.io/getainode/ainode",
    "argentos/ainode",
    "ainode",
)

# The listing this module parses. Tab separated because a tag cannot contain a
# tab and a repository cannot either, while both can contain everything else.
DOCKER_IMAGES_FORMAT = "{{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}"

# A release tag: exactly three dotted numbers. Anything else is a pointer, a
# build or somebody's experiment, and is never removed.
RELEASE_TAG = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

# docker prints sizes in base-1000 units (kB, MB, GB); the base-1024 spellings
# are accepted too in case a future docker changes its mind.
_SIZE_UNITS = {
    "b": 1,
    "kb": 10 ** 3, "kib": 1024,
    "mb": 10 ** 6, "mib": 1024 ** 2,
    "gb": 10 ** 9, "gib": 1024 ** 3,
    "tb": 10 ** 12, "tib": 1024 ** 4,
}

# Docker's own runner signature: argv in, (returncode, stdout, stderr) out. A
# test hands in a fake; nothing else in this module knows what docker is.
DockerRunner = Callable[[Sequence[str]], Tuple[int, str, str]]


def parse_size(text: str) -> int:
    """Bytes from a ``docker images`` SIZE cell. 0 for anything unreadable."""
    raw = str(text or "").strip().replace(" ", "")
    match = re.match(r"^([0-9]*\.?[0-9]+)([a-zA-Z]*)$", raw)
    if not match:
        return 0
    number = float(match.group(1))
    unit = match.group(2).lower() or "b"
    return int(number * _SIZE_UNITS.get(unit, 1))


def release_key(tag: str) -> Optional[tuple]:
    """``(major, minor, patch)`` for a release tag, or None when it is not one."""
    match = RELEASE_TAG.match(str(tag or "").strip())
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def human_size(num_bytes: int) -> str:
    """A size to print. Base 1000, same units docker itself prints."""
    value = float(max(0, int(num_bytes)))
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1000
    return f"{value:.1f} TB"


@dataclass
class ImageRow:
    """One line of ``docker images``."""
    repository: str
    tag: str
    image_id: str
    size: str = ""

    @property
    def ref(self) -> str:
        return f"{self.repository}:{self.tag}"

    @property
    def size_bytes(self) -> int:
        return parse_size(self.size)


@dataclass
class Decision:
    """What happens to one image, and why. The why is the point: an operator
    reading a prune has to be able to tell a kept rollback generation from an
    image that was simply not ours."""
    ref: str
    image_id: str
    remove: bool
    reason: str
    size_bytes: int = 0


@dataclass
class PrunePlan:
    current: str
    keep: int
    decisions: List[Decision] = field(default_factory=list)
    # Set when nothing can be decided safely; no removals in that case.
    refused: str = ""

    @property
    def removals(self) -> List[Decision]:
        return [d for d in self.decisions if d.remove]

    @property
    def kept(self) -> List[Decision]:
        return [d for d in self.decisions if not d.remove]

    @property
    def freed_bytes_upper_bound(self) -> int:
        return sum(d.size_bytes for d in self.removals)

    @property
    def distinct_images_removed(self) -> int:
        return len({d.image_id for d in self.removals if d.image_id})


def parse_images(text: str) -> List[ImageRow]:
    """Rows from a ``docker images --format DOCKER_IMAGES_FORMAT`` listing.

    Tolerates the header of a plain ``docker images`` run and skips
    ``<none>`` tags: an untagged layer cannot be removed by reference, and
    dangling images are ``docker image prune``'s job, not an update's.
    """
    rows: List[ImageRow] = []
    for line in str(text or "").splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 3:
            continue
        repository, tag, image_id = parts[0].strip(), parts[1].strip(), parts[2].strip()
        size = parts[3].strip() if len(parts) > 3 else ""
        if repository.upper() == "REPOSITORY" or tag.upper() == "TAG":
            continue
        if tag in ("<none>", "") or repository in ("<none>", ""):
            continue
        rows.append(ImageRow(repository=repository, tag=tag, image_id=image_id, size=size))
    return rows


def plan_prune(rows: Sequence[ImageRow], current: str, keep: int = 1) -> PrunePlan:
    """Decide what an update may delete. Pure: no docker, no filesystem."""
    keep = max(0, int(keep))
    plan = PrunePlan(current=current, keep=keep)

    current_repo, _, current_tag = str(current or "").rpartition(":")
    current_version = release_key(current_tag)
    ours = [r for r in rows if r.repository in AINODE_IMAGE_REPOS]

    if not current_version:
        plan.refused = (
            f"the running image {current!r} does not name a release tag, so there "
            "is no baseline to keep generations against")
        return plan

    present = {r.ref for r in ours}
    if f"{current_repo}:{current_tag}" not in present:
        plan.refused = (
            f"{current} is not in the image listing, so the new image is not on "
            "this host: nothing is removed until it is")
        return plan

    current_ids = {r.image_id for r in ours
                   if r.tag == current_tag and r.image_id}

    # Generations are counted in RELEASES across all of AINode's repositories,
    # not per repository. The same release is tagged under each of them and the
    # three tags share one image, so keeping the mirror's copy of a release keeps
    # the whole image alive: counting per repository would remove
    # ghcr.io/...:0.5.24 while argentos/ainode:0.5.24 held every byte of it, and
    # free nothing at all. A kept generation is kept under all its names, and a
    # dropped one loses all of them.
    releases = sorted({release_key(r.tag) for r in ours if release_key(r.tag)})
    older = sorted([v for v in releases if v < current_version], reverse=True)
    generation = {version: index + 1 for index, version in enumerate(older)}

    for row in sorted(ours, key=lambda r: (r.repository, release_key(r.tag) or (0,))):
        version = release_key(row.tag)
        if version is None:
            plan.decisions.append(Decision(
                row.ref, row.image_id, False, "not a release tag", row.size_bytes))
            continue
        if row.ref == current:
            plan.decisions.append(Decision(
                row.ref, row.image_id, False, "the running release", row.size_bytes))
            continue
        if version == current_version:
            plan.decisions.append(Decision(
                row.ref, row.image_id, False,
                "the running release, under another name", row.size_bytes))
            continue
        if row.image_id and row.image_id in current_ids:
            plan.decisions.append(Decision(
                row.ref, row.image_id, False,
                "the image this node runs, under another name", row.size_bytes))
            continue
        if version > current_version:
            plan.decisions.append(Decision(
                row.ref, row.image_id, False, "newer than the running release",
                row.size_bytes))
            continue
        nth = generation[version]
        if nth <= keep:
            plan.decisions.append(Decision(
                row.ref, row.image_id, False,
                f"rollback generation {nth} of {keep}", row.size_bytes))
            continue
        plan.decisions.append(Decision(
            row.ref, row.image_id, True,
            f"{nth} releases behind the running one", row.size_bytes))

    return plan


def format_plan(plan: PrunePlan, *, verbose: bool = False) -> str:
    """The plan as an operator reads it."""
    lines: List[str] = []
    if plan.refused:
        lines.append(f"Not pruning: {plan.refused}")
        return "\n".join(lines)

    lines.append(f"Running image: {plan.current}  (keeping {plan.keep} "
                 f"rollback generation{'' if plan.keep == 1 else 's'})")
    if verbose:
        for decision in plan.kept:
            lines.append(f"  keep    {decision.ref}  ({decision.reason})")
    for decision in plan.removals:
        lines.append(f"  remove  {decision.ref}  "
                     f"({decision.reason}, {human_size(decision.size_bytes)})")
    if not plan.removals:
        lines.append("  nothing to remove")
    else:
        lines.append(
            f"  {len(plan.removals)} tag(s), {plan.distinct_images_removed} distinct "
            f"image(s), up to {human_size(plan.freed_bytes_upper_bound)} freed "
            "(an upper bound: shared layers are counted once per image)")
    return "\n".join(lines)


def _subprocess_runner(argv: Sequence[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(list(argv), capture_output=True, text=True, timeout=300)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def list_images(runner: Optional[DockerRunner] = None) -> List[ImageRow]:
    """Ask docker for its image listing."""
    run = runner or _subprocess_runner
    code, out, err = run(["docker", "images", "--format", DOCKER_IMAGES_FORMAT])
    if code != 0:
        raise RuntimeError(f"docker images failed (rc={code}): {(err or out)[-300:]}")
    return parse_images(out)


def running_image(default_version: str = "") -> str:
    """The image this node boots, as the host systemd unit sees it.

    ``image.env`` under AINODE_HOME is what the unit reads, so it is the truth
    about what a restart will start. The env var is the fallback the unit itself
    falls back to, and the running version is the last resort.
    """
    home = Path(os.environ.get("AINODE_HOME", str(Path.home() / ".ainode")))
    try:
        for line in (home / "image.env").read_text().splitlines():
            key, _, value = line.strip().partition("=")
            if key == "AINODE_IMAGE" and value:
                return value
    except OSError:
        pass
    env = os.environ.get("AINODE_IMAGE", "")
    if env:
        return env
    if default_version:
        return f"{AINODE_IMAGE_REPOS[0]}:{default_version}"
    return ""


def prune_images(
    current: str,
    keep: int = 1,
    *,
    dry_run: bool = False,
    rows: Optional[Sequence[ImageRow]] = None,
    runner: Optional[DockerRunner] = None,
) -> Tuple[PrunePlan, List[str]]:
    """Plan and, unless *dry_run*, remove. Returns the plan and the log lines.

    A removal that docker refuses (an image a container still uses) is reported
    and not fatal: the rest of the plan still applies, and the next update tries
    again.
    """
    run = runner or _subprocess_runner
    image_rows = list(rows) if rows is not None else list_images(run)
    plan = plan_prune(image_rows, current, keep=keep)
    log: List[str] = []
    if dry_run or plan.refused:
        return plan, log

    for decision in plan.removals:
        code, out, err = run(["docker", "rmi", decision.ref])
        if code == 0:
            log.append(f"removed {decision.ref} ({human_size(decision.size_bytes)})")
        else:
            decision.remove = False
            decision.reason = f"docker refused to remove it: {(err or out).strip()[:120]}"
            log.append(f"kept {decision.ref}: {decision.reason}")
    return plan, log
