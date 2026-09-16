"""The harness bench proper: drive an agent CLI at a task, then judge it on tests.

The measurement is one bit per task per attempt - did the hidden tests pass - plus
what it cost in wall clock. Everything else in this file exists to keep that bit
honest:

  * **The tests are not in the working directory while the harness runs.** Only the
    instructions and the stub are copied in. The tests go in afterwards, the
    results are read, and they come straight back out before the next attempt. The
    check is an assertion in the loop, not a comment: a harness that can read the
    assertions can satisfy them without solving anything, and the number would be
    worthless.
  * **Two attempts, the second one told what failed**, which is Aider's polyglot
    protocol. pass@1 is the model getting it right cold; pass@2 is the model
    reading a stack trace. They are different skills and the gap between them is
    the interesting part of the record.
  * **A crash is a result.** A harness that dies, times out or exits nonzero still
    gets its tests run against whatever it left on disk, because a partial edit
    that passes is a pass. Crashes and timeouts are counted separately so a low
    score can be read as "the model could not" or "the harness fell over".
  * **Nothing is loaded, unloaded or restarted**, the same rule the throughput
    bench runs under. This drives inference against an endpoint that is already
    serving, and it will add real load to it.

Token counts, when a reader is supplied, are AINode's own ``/api/metrics``
counters differenced across the window. They are a property of the node, not of
this run: other traffic on the node lands in the same delta. The record says so.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

from ainode.bench.harness.adapters import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_OUTPUT_TOKENS,
    HarnessAdapter,
    HarnessRequest,
    HarnessRun,
    tail,
)
from ainode.bench.harness.tasks import Task, task_set

SCHEMA = 1
SOURCE = "scripts/ainode-bench.py harness"

DEFAULT_TIMEOUT = 900
DEFAULT_ATTEMPTS = 2
TEST_TIMEOUT = 300
#: How much of the failing test output attempt 2 is given. Enough for the
#: assertion and a traceback, short enough not to eat a small context window.
FAILURE_CHARS = 4000

# pytest -q summary lines: "6 passed in 0.01s", "1 failed, 5 passed in 0.05s",
# "2 errors in 0.1s". Counts are read when present and the exit code decides.
COUNT_RE = re.compile(r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed)\b")


class HarnessBenchError(RuntimeError):
    """A run that cannot proceed: no adapter, no binary, no task set."""


# ---------------------------------------------------------------- the prompt

def build_prompt(task: Task, failure: str | None = None) -> str:
    """What the agent is told. The task text, then the rules, then (attempt 2) why
    the last try failed.

    The instructions are upstream's, verbatim. The two sentences after them are
    the harness contract: edit one file, do not invent tests, do not ask. Without
    them some agents write a test file of their own and report success against it.
    """
    parts = [task.instructions.strip(), "",
             f"Write your solution in {task.entry}, in the current directory. "
             f"Edit only {task.entry}.",
             "Hidden unit tests will be run against it. Do not write, create or "
             "modify any test file, and do not ask any questions: make the edit and "
             "finish."]
    if failure:
        parts += ["", "A previous attempt at this task did not pass the hidden tests. "
                      "This is what they printed:", "", "```",
                  tail(failure, FAILURE_CHARS), "```", "",
                  f"Fix {task.entry} so the tests pass."]
    return "\n".join(parts)


# ---------------------------------------------------------------- test results

@dataclass
class TestResult:
    """One pytest invocation against the task's hidden tests."""

    exit_code: int | None
    wall_s: float
    passed_count: int | None = None
    failed_count: int | None = None
    error_count: int | None = None
    timed_out: bool = False
    output_tail: str = ""
    output: str = field(default="", repr=False)

    @property
    def passed(self) -> bool:
        """Green is exit code zero. pytest exits 5 when it collected nothing, so a
        zero cannot mean "no tests ran", and a count parsed out of the summary
        line is recorded but never the verdict."""
        return self.exit_code == 0

    def as_json(self) -> dict:
        out = {"exit_code": self.exit_code, "wall_s": round(self.wall_s, 2),
               "passed": self.passed}
        for key, value in (("tests_passed", self.passed_count),
                           ("tests_failed", self.failed_count),
                           ("tests_errored", self.error_count)):
            if value is not None:
                out[key] = value
        if self.timed_out:
            out["timed_out"] = True
        if self.output_tail:
            out["output_tail"] = self.output_tail
        return out


def parse_pytest_counts(text: str) -> dict:
    """Counts off a pytest summary line. Absent when the line is not there."""
    counts = {}
    for number, word in COUNT_RE.findall(text or ""):
        key = "error" if word.startswith("error") else word
        counts[key] = counts.get(key, 0) + int(number)
    out = {}
    if "passed" in counts:
        out["passed_count"] = counts["passed"]
    if "failed" in counts:
        out["failed_count"] = counts["failed"]
    if "error" in counts:
        out["error_count"] = counts["error"]
    return out


def resolve_test_command(task: Task) -> list[str]:
    """The task's own command, with ``python`` bound to the interpreter running us.

    A task file says ``python -m pytest -q x_test.py`` because that is what it
    means; on a machine where ``python`` is not on PATH, or is the wrong one, the
    verdict would come from the wrong interpreter.
    """
    cmd = list(task.test_command)
    if cmd and cmd[0] in ("python", "python3"):
        cmd[0] = sys.executable
    return cmd


def run_tests(task: Task, workdir: pathlib.Path, timeout: float = TEST_TIMEOUT) -> TestResult:
    command = resolve_test_command(task)
    start = time.monotonic()
    try:
        proc = subprocess.run(command, cwd=str(workdir), capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return TestResult(exit_code=None, wall_s=time.monotonic() - start,
                          timed_out=True,
                          output_tail=f"test command timed out after {timeout:g}s")
    except OSError as exc:
        return TestResult(exit_code=None, wall_s=time.monotonic() - start,
                          output_tail=f"{type(exc).__name__}: {exc}")
    text = (proc.stdout or "") + (proc.stderr or "")
    return TestResult(exit_code=proc.returncode, wall_s=time.monotonic() - start,
                      output=text, output_tail=tail(text),
                      **parse_pytest_counts(text))


# ---------------------------------------------------------------- isolation

def prepare_workdir(task: Task, workdir: pathlib.Path) -> pathlib.Path:
    """A fresh directory holding the instructions and the stub. Nothing else."""
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    shutil.copyfile(task.directory / task.instructions_file,
                    workdir / task.instructions_file)
    shutil.copyfile(task.directory / task.entry, workdir / task.entry)
    return workdir


def copy_tests_in(task: Task, workdir: pathlib.Path) -> list[pathlib.Path]:
    """Flatten ``tests/`` into the working directory so the imports resolve."""
    written = []
    for rel in task.test_files:
        dest = workdir / pathlib.PurePosixPath(rel).name
        shutil.copyfile(task.directory / rel, dest)
        written.append(dest)
    return written


def remove_tests(task: Task, workdir: pathlib.Path) -> None:
    for name in task.test_names():
        (workdir / name).unlink(missing_ok=True)
    shutil.rmtree(workdir / "__pycache__", ignore_errors=True)
    shutil.rmtree(workdir / ".pytest_cache", ignore_errors=True)


def assert_tests_hidden(task: Task, workdir: pathlib.Path) -> None:
    """The invariant, checked rather than trusted, right before the harness runs."""
    present = [n for n in task.test_names() if (workdir / n).exists()]
    if present:
        raise HarnessBenchError(
            f"{workdir} still holds hidden test file(s) {', '.join(present)}; the "
            "harness must never see them")


# ---------------------------------------------------------------- one task

@dataclass
class Attempt:
    number: int
    harness: HarnessRun
    tests: TestResult

    def as_json(self) -> dict:
        return {"attempt": self.number, "harness": self.harness.as_json(),
                "tests": self.tests.as_json()}


@dataclass
class TaskResult:
    slug: str
    attempts: list = field(default_factory=list)
    tokens: dict | None = None

    @property
    def passed_at(self) -> int | None:
        for attempt in self.attempts:
            if attempt.tests.passed:
                return attempt.number
        return None

    @property
    def harness_wall_s(self) -> float:
        return sum(a.harness.wall_s for a in self.attempts)

    @property
    def crashed(self) -> bool:
        return any(a.harness.crashed for a in self.attempts)

    @property
    def timed_out(self) -> bool:
        return any(a.harness.timed_out for a in self.attempts)

    def as_json(self) -> dict:
        out = {"slug": self.slug, "passed": self.passed_at is not None,
               "passed_at_attempt": self.passed_at,
               "harness_wall_s": round(self.harness_wall_s, 2),
               "crashed": self.crashed, "timed_out": self.timed_out,
               "attempts": [a.as_json() for a in self.attempts]}
        if self.tokens:
            out["tokens"] = self.tokens
        return out


def run_task(task: Task, adapter: HarnessAdapter, endpoint: str, model: str,
             root: pathlib.Path, timeout: float = DEFAULT_TIMEOUT,
             attempts: int = DEFAULT_ATTEMPTS, api_key: str = "ainode",
             context_window: int = DEFAULT_CONTEXT_WINDOW,
             max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
             tokens_reader=None, log=print) -> TaskResult:
    """One task, up to ``attempts`` tries, one harness. Never raises on harness
    failure; raises only if the isolation invariant breaks."""
    workdir = prepare_workdir(task, root / adapter.name / task.slug)
    scratch = root / adapter.name / f"{task.slug}.scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    result = TaskResult(slug=task.slug)
    before = _read_tokens(tokens_reader)
    failure = None
    for number in range(1, attempts + 1):
        assert_tests_hidden(task, workdir)
        req = HarnessRequest(workdir=workdir, scratch=scratch,
                             prompt=build_prompt(task, failure), entry=task.entry,
                             endpoint=endpoint, model=model, api_key=api_key,
                             context_window=context_window,
                             max_output_tokens=max_output_tokens)
        run = adapter.run(req, timeout=timeout)
        copy_tests_in(task, workdir)
        tests = run_tests(task, workdir)
        remove_tests(task, workdir)
        result.attempts.append(Attempt(number, run, tests))
        state = ("pass" if tests.passed
                 else "timeout" if run.timed_out
                 else "crash" if run.crashed else "fail")
        log(f"    {adapter.name:>9} {task.slug:<20} attempt {number}: {state} "
            f"({run.wall_s:.1f}s harness, {tests.wall_s:.1f}s tests)")
        if tests.passed:
            break
        failure = tests.output or tests.output_tail
    result.tokens = _token_delta(before, _read_tokens(tokens_reader))
    return result


# ---------------------------------------------------------------- scoring

def score(results: list) -> dict:
    """pass@1, pass@2, mean wall seconds, crash count.

    ``pass_at_2`` is cumulative: a task solved on the first attempt counts in
    both. ``mean_wall_s`` is the mean over tasks of the harness's total wall clock
    for that task, every attempt included, so a harness that needs two tries pays
    for both.
    """
    total = len(results)
    if not total:
        return {"tasks": 0}
    first = sum(1 for r in results if r.passed_at == 1)
    either = sum(1 for r in results if r.passed_at is not None)
    return {
        "tasks": total,
        "passed_at_1": first,
        "passed_at_2": either,
        "pass_at_1": round(first / total, 3),
        "pass_at_2": round(either / total, 3),
        "mean_wall_s": round(statistics.mean(r.harness_wall_s for r in results), 1),
        "crashes": sum(1 for r in results if r.crashed),
        "timeouts": sum(1 for r in results if r.timed_out),
    }


# ---------------------------------------------------------------- token window

def _read_tokens(reader):
    if reader is None:
        return None
    try:
        snapshot = reader()
    except Exception:
        return None
    if not isinstance(snapshot, dict):
        return None
    requests = snapshot.get("requests") if isinstance(snapshot.get("requests"), dict) else snapshot
    total = requests.get("total")
    generated = requests.get("tokens_generated")
    if total is None and generated is None:
        return None
    return {"requests": total, "tokens_generated": generated}


def _token_delta(before, after):
    if not before or not after:
        return None
    out = {}
    for key in ("requests", "tokens_generated"):
        start, end = before.get(key), after.get(key)
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start:
            out[key] = end - start
    return out or None


def http_metrics_reader(ainode: str):
    """A reader for ``<ainode>/api/metrics``. Returns None when not configured.

    Uses the throughput bench's own tolerant GET: a control endpoint that is not
    there degrades one optional field, it never fails a run.
    """
    if not ainode:
        return None
    from ainode.bench.measure import get_json

    base = ainode.rstrip("/")

    def read():
        data = get_json(f"{base}/api/metrics")
        return None if "_error" in data else data

    return read


# ---------------------------------------------------------------- the suite

@dataclass
class HarnessResult:
    """One harness over the whole task set."""

    harness: str
    version: str | None
    tasks: list = field(default_factory=list)

    def as_json(self) -> dict:
        return {"harness": self.harness, "version": self.version,
                "scores": score(self.tasks),
                "tasks": [t.as_json() for t in self.tasks]}


def run_suite(tasks: list, adapters: list, endpoint: str, model: str,
              root: pathlib.Path | None = None, timeout: float = DEFAULT_TIMEOUT,
              attempts: int = DEFAULT_ATTEMPTS, api_key: str = "ainode",
              context_window: int = DEFAULT_CONTEXT_WINDOW,
              max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
              tokens_reader=None, log=print) -> list:
    """Every adapter over every task. Harness-major, so one harness's numbers are
    measured under conditions as close together as the fleet allows."""
    root = pathlib.Path(root) if root else pathlib.Path(tempfile.mkdtemp(prefix="ainode-harness-"))
    out = []
    for adapter in adapters:
        version = adapter.version()
        log(f"\n  {adapter.name} {version or '(version unknown)'}")
        result = HarnessResult(harness=adapter.name, version=version)
        for task in tasks:
            result.tasks.append(run_task(
                task, adapter, endpoint, model, root, timeout=timeout,
                attempts=attempts, api_key=api_key, context_window=context_window,
                max_output_tokens=max_output_tokens, tokens_reader=tokens_reader,
                log=log))
        scores = score(result.tasks)
        log(f"    {adapter.name}: pass@1 {scores['passed_at_1']}/{scores['tasks']}, "
            f"pass@2 {scores['passed_at_2']}/{scores['tasks']}, "
            f"mean {scores['mean_wall_s']}s, crashes {scores['crashes']}")
        out.append(result)
    return out


# ---------------------------------------------------------------- the record

def build_harness_block(results: list, tasks: list, endpoint: str, attempts: int,
                        timeout: float, tasks_dir=None) -> dict:
    """The record's ``harness`` block. See bench/SCHEMA.md."""
    meta = task_set(tasks_dir)
    return {
        "task_set": {"id": meta.get("id"), "count": len(tasks),
                     "available": meta.get("count"),
                     "slugs": [t.slug for t in tasks],
                     "language": meta.get("language"),
                     "source": meta.get("source")},
        "endpoint": endpoint,
        "protocol": {"attempts": attempts, "timeout_s": timeout,
                     "second_attempt_sees": "the failing test output"},
        "runs": [r.as_json() for r in results],
    }


def build_notes(results: list, tokens_used: bool, seconds: int, source: str = SOURCE) -> list:
    notes = [f"Measured by {source} in {seconds}s; nothing loaded, unloaded or restarted.",
             "The hidden tests are copied into the working directory only after the "
             "harness has exited, and removed again before the next attempt.",
             "pass@2 is cumulative and the second attempt is given the failing test "
             "output, which is Aider's polyglot protocol.",
             "A harness crash or timeout still gets its tests run against whatever it "
             "left on disk; crashes and timeouts are counted separately."]
    if tokens_used:
        notes.append("Token counts are AINode's /api/metrics counters differenced over "
                     "each task's window: they are the node's totals, so any other "
                     "traffic on the node during the run is inside them.")
    if any(r.version is None for r in results):
        missing = ", ".join(r.harness for r in results if r.version is None)
        notes.append(f"Version unknown for: {missing}.")
    return notes


def build_record(label: str, model_block: dict, placement: dict, harness_block: dict,
                 settings: dict, notes: list, stamp: str, source: str = SOURCE) -> dict:
    """A schema-1 record with a ``harness`` block and no ``results`` block.

    Deliberately no ``results``: this run measured no throughput, and a zero there
    would be a number nobody took. ``scripts/render-bench-table.py`` skips a record
    shaped like this rather than rendering it as a slow model.
    """
    return {"schema": SCHEMA, "stamp": stamp, "label": label, "model": model_block,
            "placement": placement, "settings": settings, "harness": harness_block,
            "notes": notes, "source": source}
