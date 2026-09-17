"""opencode - the OpenCode CLI (``opencode-ai``).

Verified end to end on opencode 1.18.31, by the runner rather than by hand: exit 0 in
26 s, the stub edited, 6/6 hidden tests green on ``isogram`` against a model on the
fleet, three consecutive times, with

    opencode run --dir <workdir> --print-logs --log-level ERROR \\
        --pure --auto --format json -m "<provider>/<model id>" "<prompt>"

in a ``git init``ed directory holding a project-local ``opencode.json``.

Four of those flags are not optional:

  * ``--dir <workdir>`` names the directory to run in. **Without it opencode does
    not use its own working directory**: it resolves the project from the
    inherited ``PWD``, which a subprocess launched with ``cwd=`` still carries
    from its parent. From the runner that is wherever the bench was started, so
    opencode built its session against that directory instead, never saw the
    run's ``opencode.json``, and died in about 1 s with ``Model not found:
    ainode-bench/<id>`` reported on stdout as "Unexpected server error. Check
    server logs for details." By hand the same command passed, because a person
    ``cd``s in first and ``PWD`` is then right. That is #118, and it is why the
    other four harnesses were unaffected. ``_launch`` now also sets ``PWD`` to
    the cwd for every harness; this flag is the explicit half of the same fix.
  * ``--auto`` approves permissions that are not explicitly denied. Without a TTY
    there is nobody to approve the file write, so the run sits until the timeout.
  * ``--format json`` makes it stream NDJSON events on stdout. Without it, two runs
    produced **no output at all** until they timed out, so this is a working
    requirement rather than a preference about parsing.
  * ``--pure`` skips external plugins.

``--print-logs --log-level ERROR`` is diagnostics, not behaviour: the logs go to
stderr, stdout stays NDJSON, and the record's ``stderr_tail`` then carries the named
cause instead of the opaque "Unexpected server error" that cost #118 a day. The
isolated log file opencode writes is empty in exactly the case worth reading.

The provider is a project-local ``opencode.json`` written into the working
directory: config, not a hint, naming the AINode endpoint as an OpenAI-compatible
provider through ``@ai-sdk/openai-compatible``. The model entry carries just a name,
which is the shape that was verified; the run's declared context window and output
cap are not passed here (they reach pi and dsh, which refuse to route without them).

OpenCode expects to be inside a git repo, so the working directory gets a bare
``git init`` with no commit and no identity.

Startup costs about 18 s of the wall clock before any token is generated: it loads
every skill under ``~/.claude/skills`` and ``~/.agents/skills`` even with ``--pure``.
That is inside ``mean_wall_s``, so this harness's wall clock is not comparable with
another harness's without subtracting it.
"""
from __future__ import annotations

import json

from ainode.bench.harness.adapters import (
    PROVIDER,
    ConfigFile,
    HarnessAdapter,
    HarnessRequest,
)

CONFIG_NAME = "opencode.json"
SCHEMA_URL = "https://opencode.ai/config.json"


def project_config(req: HarnessRequest) -> dict:
    return {
        "$schema": SCHEMA_URL,
        "provider": {
            PROVIDER: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "AINode harness bench",
                "options": {"baseURL": req.endpoint, "apiKey": req.api_key},
                "models": {req.model: {"name": req.model}},
            },
        },
    }


def parse_events(stdout: str) -> dict:
    """Turns off the NDJSON stream: one ``step_start`` event is one turn.

    Every line that is not JSON is skipped rather than guessed at. Nothing else is
    read out of the stream: the event schema is upstream's and not ours to depend
    on, so a shape change costs a missing ``turns`` field and nothing else.
    """
    steps = 0
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "step_start":
            steps += 1
    return {"turns": steps} if steps else {}


class OpencodeAdapter(HarnessAdapter):
    name = "opencode"
    binary = "opencode"
    needs_git = True

    def config(self, req: HarnessRequest) -> list[ConfigFile]:
        # In the working directory on purpose: opencode reads project config from
        # its cwd, and the file is the run's own provider wiring rather than
        # anything about the task. It is not a test file, so the isolation rule is
        # untouched.
        return [ConfigFile(req.workdir / CONFIG_NAME,
                           json.dumps(project_config(req), indent=1) + "\n")]

    def env(self, req: HarnessRequest) -> dict[str, str]:
        # OpenCode keeps a SQLite database, logs and snapshots under the XDG
        # directories and reads global config from ~/.config/opencode. Point all
        # of that at the run's own scratch dir so runs never share state and a
        # crashed run cannot poison the next one (it creates the dirs itself).
        xdg = req.scratch / "opencode-xdg"
        config = xdg / "config"
        return {
            "XDG_DATA_HOME": str(xdg / "data"),
            "XDG_CONFIG_HOME": str(config),
            "XDG_CACHE_HOME": str(xdg / "cache"),
            "XDG_STATE_HOME": str(xdg / "state"),
            # OPENCODE_CONFIG_DIR overrides the XDG search outright, so an
            # operator's exported value (Orca exports one) walks straight through
            # the isolation above and puts somebody's config, plugins and agents
            # inside the measurement. Point it at the run's own empty config dir:
            # the same isolation rule as DSH_HOME and CLAUDE_CONFIG_DIR.
            "OPENCODE_CONFIG_DIR": str(config / "opencode"),
        }

    def command(self, req: HarnessRequest) -> list[str]:
        return [
            self.binary, "run",
            # Not optional. See the module docstring and #118: without it opencode
            # takes the project directory from the inherited PWD, not its cwd.
            "--dir", str(req.workdir),
            # Diagnostics only: to stderr, leaving stdout the NDJSON stream.
            "--print-logs", "--log-level", "ERROR",
            "--pure",
            "--auto",
            "--format", "json",
            "-m", f"{PROVIDER}/{req.model}",
            req.prompt,
        ]

    def parse(self, stdout: str, stderr: str) -> dict:
        return parse_events(stdout)
