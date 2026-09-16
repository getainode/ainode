"""Harness adapters: one small class per coding agent CLI.

A harness bench run is two programs: the agent CLI, which we drive, and pytest,
which judges it. An adapter is everything we need to know about the first one, and
deliberately nothing else. It answers four questions:

  ``available()``   is the binary on PATH
  ``version()``     what does it say when asked
  ``config(req)``   which files must exist before it can reach an AINode endpoint
  ``command(req)``  the exact argv, and ``env(req)`` the exact environment overlay

``command``, ``env`` and ``config`` are **pure functions of the request**. That is
the rule that makes this testable without a model: the tests pin the argv and the
config bytes for all four harnesses, so a flag that moves is a failing test rather
than a silently wrong benchmark. ``run()`` is implemented once, here, in terms of
those three, so no adapter owns its own subprocess handling.

None of these ever needs a real API key. Every provider entry carries the literal
placeholder ``ainode``, because an AINode endpoint does not authenticate; a harness
that refuses to start without *something* in the key field gets that.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_API_KEY = "ainode"
VERSION_TIMEOUT = 30
TAIL_CHARS = 2000
#: Longer than this, an argv entry is the prompt, and the record elides it.
ARG_PREVIEW = 200

# Catalog defaults for the harnesses that demand a model entry before they will
# route. Conservative rather than accurate: they are declared, and recorded as
# declared, never presented as a property of the served model.
DEFAULT_CONTEXT_WINDOW = 131072
DEFAULT_MAX_OUTPUT_TOKENS = 16384

#: Provider id the adapters register. Deliberately not "ainode": a hand-made
#: provider of that name already exists in some of these tools' config files, and
#: the bench must add its own route rather than rewrite somebody's.
PROVIDER = "ainode-bench"


@dataclass(frozen=True)
class HarnessRequest:
    """Everything an adapter needs to build one invocation.

    ``workdir`` is the task's working directory and the agent's cwd: it holds the
    instructions and the stub and nothing else. ``scratch`` is a sibling directory
    for config a harness needs but the agent should not be reading, which is why
    it is not inside ``workdir``.
    """

    workdir: Path
    scratch: Path
    prompt: str
    entry: str
    endpoint: str
    model: str
    api_key: str = DEFAULT_API_KEY
    #: Two harnesses (pi, dsh) want a model catalog entry before they will route
    #: to a provider, and neither can discover these from an OpenAI-compatible
    #: endpoint. They are the run's declared budgets, not measurements: the CLI
    #: exposes them as --context-window / --max-output-tokens and the record
    #: writes both down so a reader can see what the harness was told.
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS


@dataclass
class ConfigFile:
    """One file an adapter needs on disk, and whether it merges into what is there."""

    path: Path
    content: str
    merged: bool = False


@dataclass
class HarnessRun:
    """One invocation's outcome. Timing is wall clock around the subprocess."""

    harness: str
    command: list[str]
    exit_code: int | None
    wall_s: float
    timed_out: bool
    crashed: bool
    stdout_tail: str = ""
    stderr_tail: str = ""
    turns: int | None = None
    tokens_sent: int | None = None
    tokens_received: int | None = None
    error: str | None = None

    def as_json(self) -> dict:
        out = {"command": recorded_command(self.command), "exit_code": self.exit_code,
               "wall_s": round(self.wall_s, 2), "timed_out": self.timed_out,
               "crashed": self.crashed}
        for key in ("turns", "tokens_sent", "tokens_received", "error"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.stdout_tail:
            out["stdout_tail"] = self.stdout_tail
        if self.stderr_tail:
            out["stderr_tail"] = self.stderr_tail
        return out


def tail(text: str, chars: int = TAIL_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= chars else "..." + text[-chars:]


def recorded_command(command: list[str], limit: int = ARG_PREVIEW) -> str:
    """The argv for the record, with the prompt argument elided.

    The flags are the reproducible part and they go in whole. The prompt does not:
    it is a task's instructions plus, on a second attempt, a screenful of pytest
    output, and putting all of that in the record twice per task would make a
    ten-task run's JSON mostly prompt. It is regenerated exactly by
    ``build_prompt`` from the task anyway.
    """
    return shlex.join(f"<{len(a)} chars>" if len(a) > limit else a for a in command)


class HarnessAdapter:
    """Base class. Subclasses override the pure parts, not ``run``."""

    name = ""
    binary = ""
    version_args: tuple[str, ...] = ("--version",)
    #: Some agents refuse to touch a directory that is not a git repo.
    needs_git = False

    # ------------------------------------------------------------ discovery

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def version(self) -> str | None:
        """First line of ``<binary> --version``, or None if it cannot be asked."""
        if not self.available():
            return None
        try:
            proc = subprocess.run([self.binary, *self.version_args],
                                  capture_output=True, text=True, timeout=VERSION_TIMEOUT)
        except (OSError, subprocess.SubprocessError):
            return None
        out = (proc.stdout or proc.stderr or "").strip().splitlines()
        return out[-1].strip() if out else None

    # ------------------------------------------------------------ pure parts

    def config(self, req: HarnessRequest) -> list[ConfigFile]:
        """Files that must exist before ``command`` can reach the endpoint."""
        return []

    def command(self, req: HarnessRequest) -> list[str]:
        raise NotImplementedError

    def env(self, req: HarnessRequest) -> dict[str, str]:
        """Environment overlay, applied on top of the caller's environment."""
        return {}

    def parse(self, stdout: str, stderr: str) -> dict:
        """Whatever the harness's own output makes knowable: turns, token counts."""
        return {}

    # ------------------------------------------------------------ execution

    def write_config(self, req: HarnessRequest) -> list[Path]:
        written = []
        for item in self.config(req):
            item.path.parent.mkdir(parents=True, exist_ok=True)
            item.path.write_text(item.content)
            written.append(item.path)
        return written

    def run(self, req: HarnessRequest, timeout: float) -> HarnessRun:
        """Drive the harness once, in ``req.workdir``, and never raise.

        A missing binary, a timeout and a nonzero exit are all results, not
        errors: a harness that cannot finish the task is exactly the thing the
        bench is measuring, and the tests still get run against whatever it left
        on disk.
        """
        command = self.command(req)
        env = dict(os.environ)
        env.update(self.env(req))
        try:
            self.write_config(req)
            if self.needs_git:
                _git_init(req.workdir)
        except OSError as exc:
            return HarnessRun(self.name, command, None, 0.0, False, True,
                              error=f"{type(exc).__name__}: {exc}")

        start = time.monotonic()
        try:
            proc = subprocess.run(command, cwd=str(req.workdir), env=env,
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            return HarnessRun(self.name, command, None, time.monotonic() - start,
                              True, False,
                              stdout_tail=tail(_text(exc.stdout)),
                              stderr_tail=tail(_text(exc.stderr)),
                              error=f"timed out after {timeout:g}s")
        except (OSError, ValueError) as exc:
            return HarnessRun(self.name, command, None, time.monotonic() - start,
                              False, True, error=f"{type(exc).__name__}: {exc}")

        wall = time.monotonic() - start
        run = HarnessRun(self.name, command, proc.returncode, wall, False,
                         proc.returncode != 0,
                         stdout_tail=tail(proc.stdout), stderr_tail=tail(proc.stderr))
        for key, value in self.parse(proc.stdout or "", proc.stderr or "").items():
            setattr(run, key, value)
        return run

    # ------------------------------------------------------------ dry run

    def describe(self, req: HarnessRequest) -> dict:
        """What ``--dry-run`` prints. Writes nothing."""
        return {"harness": self.name, "binary": self.binary,
                "available": self.available(), "cwd": str(req.workdir),
                "command": self.command(req),
                "env": self.env(req),
                "config": [{"path": str(c.path), "bytes": len(c.content),
                            "merged": c.merged, "content": c.content}
                           for c in self.config(req)],
                "git_init": self.needs_git}


def _text(value) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def _git_init(workdir: Path) -> None:
    """``git init`` the working directory, quietly, for harnesses that need one.

    No commit and no user config: the agents that ask for a repo ask so they can
    diff, not so they can commit, and the bench never commits anything.
    """
    if (workdir / ".git").exists() or shutil.which("git") is None:
        return
    subprocess.run(["git", "init", "-q"], cwd=str(workdir),
                   capture_output=True, text=True, timeout=60)


# ------------------------------------------------------------------ registry

@dataclass
class Registry:
    adapters: dict = field(default_factory=dict)

    def register(self, adapter: HarnessAdapter) -> None:
        self.adapters[adapter.name] = adapter

    def get(self, name: str) -> HarnessAdapter:
        try:
            return self.adapters[name]
        except KeyError:
            raise KeyError(f"unknown harness {name!r}; known: "
                           f"{', '.join(sorted(self.adapters))}") from None

    def names(self) -> list[str]:
        return sorted(self.adapters)


def registry() -> Registry:
    """Every adapter that ships. Imported late to keep the module import cheap."""
    from ainode.bench.harness.adapters.aider import AiderAdapter
    from ainode.bench.harness.adapters.dsh import DshAdapter
    from ainode.bench.harness.adapters.opencode import OpencodeAdapter
    from ainode.bench.harness.adapters.pi import PiAdapter

    reg = Registry()
    for adapter in (AiderAdapter(), DshAdapter(), PiAdapter(), OpencodeAdapter()):
        reg.register(adapter)
    return reg
