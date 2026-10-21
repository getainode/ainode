"""The task set: vendored Exercism practice exercises, and how one is loaded.

A task is a directory under ``bench/harness/tasks/<slug>/`` holding four things:

  ``task.json``        slug, language, entry file, instructions file, test files,
                       test command, and where upstream it came from
  ``instructions.md``  the exercise statement, verbatim from upstream (with the
                       Python track's ``instructions.append.md`` appended, because
                       that is where "raise ValueError with this message" lives and
                       the hidden tests assert it)
  ``<entry>.py``       the stub the harness must fill in
  ``tests/``           the hidden unit tests, which the harness never sees

The split between the task directory's root and its ``tests/`` subdirectory is the
whole measurement. :mod:`ainode.bench.harness.runner` copies only the files
``task.json`` names as instructions and entry into the working directory, runs the
harness there, and copies ``tests/`` in afterwards. A task that put its tests at
the root would let a harness read the assertions and write code that satisfies
them without solving anything.

Nothing here talks to a model or a node.
"""
from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass

TASK_SET_FILE = "TASK_SET.json"

# Where the vendored set lives. The tasks are repo data rather than package data:
# they are versioned next to the results they produce, the way bench/results/ is.
ENV_TASKS_DIR = "AINODE_HARNESS_TASKS"
_REPO_TASKS = pathlib.Path(__file__).resolve().parents[3] / "bench" / "harness" / "tasks"
_HOME_TASKS = pathlib.Path.home() / ".ainode" / "bench" / "harness" / "tasks"


class TaskError(RuntimeError):
    """A task set that cannot be loaded. Never silently skipped: a bench that ran
    eight of ten tasks because two directories were malformed would report a
    pass rate against a task count nobody chose."""


def default_tasks_dir() -> pathlib.Path:
    """``$AINODE_HARNESS_TASKS``, else the repo's set, else the installed copy."""
    override = os.environ.get(ENV_TASKS_DIR)
    if override:
        return pathlib.Path(override).expanduser()
    if _REPO_TASKS.is_dir():
        return _REPO_TASKS
    return _HOME_TASKS


@dataclass(frozen=True)
class Task:
    """One exercise, loaded from its ``task.json``."""

    slug: str
    language: str
    directory: pathlib.Path
    entry: str
    instructions_file: str
    test_files: tuple[str, ...]
    test_command: tuple[str, ...]
    source: dict

    @property
    def instructions(self) -> str:
        return (self.directory / self.instructions_file).read_text()

    @property
    def stub(self) -> str:
        return (self.directory / self.entry).read_text()

    def test_names(self) -> list[str]:
        """The test files' basenames, which is how they land in the working dir.

        Flat on purpose: the hidden tests import the solution as a top-level
        module (``from isogram import is_isogram``), so they have to sit beside
        it rather than in a package.
        """
        return [pathlib.PurePosixPath(t).name for t in self.test_files]


def load_task(directory: pathlib.Path) -> Task:
    """Load one task directory. Raises :class:`TaskError` on anything missing."""
    directory = pathlib.Path(directory)
    spec_path = directory / "task.json"
    if not spec_path.is_file():
        raise TaskError(f"{directory} has no task.json")
    try:
        spec = json.loads(spec_path.read_text())
    except json.JSONDecodeError as exc:
        raise TaskError(f"{spec_path} is not valid JSON: {exc}") from exc

    for key in ("slug", "entry", "instructions", "tests", "test_command"):
        if not spec.get(key):
            raise TaskError(f"{spec_path} is missing {key!r}")

    task = Task(
        slug=spec["slug"],
        language=spec.get("language", "python"),
        directory=directory,
        entry=spec["entry"],
        instructions_file=spec["instructions"],
        test_files=tuple(spec["tests"]),
        test_command=tuple(spec["test_command"]),
        source=spec.get("source") or {},
    )
    missing = [str(p) for p in
               [directory / task.instructions_file, directory / task.entry,
                *[directory / t for t in task.test_files]]
               if not pathlib.Path(p).is_file()]
    if missing:
        raise TaskError(f"{directory} lists files that do not exist: {', '.join(missing)}")
    for name in task.test_files:
        # The isolation guarantee is structural, so it is checked at load time:
        # a test file at the task root would be copied in with the instructions.
        if pathlib.PurePosixPath(name).parent == pathlib.PurePosixPath("."):
            raise TaskError(f"{directory}: test file {name!r} sits at the task root; "
                            "hidden tests must live under tests/")
    return task


def task_set(tasks_dir: pathlib.Path | None = None) -> dict:
    """The set's own metadata (id, count, upstream source, license)."""
    tasks_dir = pathlib.Path(tasks_dir or default_tasks_dir())
    path = tasks_dir / TASK_SET_FILE
    if not path.is_file():
        raise TaskError(f"{path} not found; is {tasks_dir} a task set?")
    return json.loads(path.read_text())


def load_tasks(tasks_dir: pathlib.Path | None = None, limit: int | None = None,
               slugs: list[str] | None = None) -> list[Task]:
    """Every task in the set, slug-sorted.

    Slug order is alphabetical and the slice is the first ``limit`` of it, so
    ``--tasks 5`` names the same five exercises on every run and two runs of
    different sizes stay comparable at their overlap.
    """
    tasks_dir = pathlib.Path(tasks_dir or default_tasks_dir())
    if not tasks_dir.is_dir():
        raise TaskError(f"task set {tasks_dir} not found; set ${ENV_TASKS_DIR} or run "
                        "from a checkout that has bench/harness/tasks/")
    found = [load_task(d) for d in sorted(tasks_dir.iterdir())
             if d.is_dir() and (d / "task.json").is_file()]
    if slugs:
        by_slug = {t.slug: t for t in found}
        unknown = [s for s in slugs if s not in by_slug]
        if unknown:
            raise TaskError(f"unknown task(s) {', '.join(unknown)}; the set has "
                            f"{', '.join(sorted(by_slug))}")
        return [by_slug[s] for s in slugs]
    if limit is not None:
        if limit < 1:
            raise TaskError(f"--tasks must be at least 1, got {limit}")
        if limit > len(found):
            raise TaskError(f"asked for {limit} tasks; the set has {len(found)}")
        found = found[:limit]
    if not found:
        raise TaskError(f"no tasks in {tasks_dir}")
    return found
