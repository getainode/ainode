"""claude - Claude Code driven against a local engine over the Messages API.

The only harness here that does not speak the OpenAI protocol. Claude Code talks
the Anthropic Messages API and nothing else, which for a long time made it the one
agent this bench could not point at an AINode fleet endpoint. Two things settled
that: vLLM serves ``/v1/messages`` natively next to its OpenAI paths, and the proxy
now forwards that path by model id the same way it forwards chat completions.

Verified end to end on claude 2.1.272 against an engine on the fleet: 6/6 hidden
tests on ``isogram``, 294 s, 6 turns, driving Qwen3.8 27B NVFP4, with

    env ANTHROPIC_BASE_URL=http://<node>:8000 ANTHROPIC_API_KEY=ainode \\
        ANTHROPIC_AUTH_TOKEN=ainode \\
        ANTHROPIC_MODEL=<model id> ANTHROPIC_SMALL_FAST_MODEL=<model id> \\
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_TELEMETRY=1 \\
      claude -p "<prompt>" --model <model id> --dangerously-skip-permissions \\
        --output-format json --max-turns 12

Four things about that invocation are load-bearing:

``ANTHROPIC_BASE_URL`` is the base **without** ``/v1``: Claude Code appends
``/v1/messages`` itself, so handing it the ``/v1`` form produces
``/v1/v1/messages`` and a 404. Every other adapter here wants the ``/v1`` form, and
the bench's ``--endpoint`` stays the ``/v1`` form for all of them; this adapter is
the one place that strips it, so nothing else has to know.

``--dangerously-skip-permissions`` is what makes the run non-interactive. There is
no TTY to approve a file write or to trust the directory, and without the flag the
run sits until the timeout rather than failing.

``--output-format json`` puts one JSON object on stdout with the final text in
``result``, the turn count in ``num_turns`` and a verdict in ``is_error``. That last
field is why the adapter parses at all: Claude Code can report a failed run and
still exit 0, so exit code alone would record a crash as a clean run.

``CLAUDE_CONFIG_DIR`` points at the run's scratch directory, so a bench run never
reads or writes the operator's own Claude Code profile: no settings, no hooks, no
MCP servers, no sessions, no credentials. That isolation is also what keeps the
measurement honest, since a personal ``settings.json`` can add hooks and MCP servers
that change what the agent does. **It needs no seed file.** Verified on 2.1.272
against a fake Messages endpoint: pointed at a directory that did not exist,
``claude -p`` with the key in the environment created ``.claude.json``, ``projects/``,
``sessions/`` and ``backups/`` itself, never prompted for a login, and exited 0.

``--effort <level>`` is appended only when the run asked for one
(``--claude-effort``). Claude Code sends reasoning effort "high" by default, and a
served chat template does not have to accept that: Qwen3.8-Flash-Next takes only
xhigh, medium and low, so every request came back
``API Error: 400 Unexpected reasoning effort high`` and the harness scored 0/10 in
0.3 s crashes on 2026-09-16, then 8/10 and 10/10 at ``--claude-effort medium``
(#127). Unset is the default because a model that accepts high keeps the
measurement already recorded for it.

Two notes on the run itself. ``stderr`` carries
``[claude-code:unrecognized_model] {...}`` for any model id that is not Anthropic's,
which is harmless and expected for every model this bench measures. And a shell
function named ``claude`` shadows the binary in an interactive shell on some
machines: irrelevant here, because the adapter is launched as an argv list with no
shell, so PATH resolves to the real file.
"""
from __future__ import annotations

import json

from ainode.bench.harness.adapters import HarnessAdapter, HarnessRequest

#: Turn cap. High enough for the verified 6-turn run to have room, low enough that
#: a model that loops does it inside one task's timeout instead of the whole suite's.
MAX_TURNS = 12

#: Where the run's isolated Claude Code profile goes, under ``req.scratch``.
CONFIG_DIR_NAME = "claude-config"


def messages_base(endpoint: str) -> str:
    """``--endpoint`` turned into what Claude Code wants: the base without ``/v1``.

    The bench's endpoint is the OpenAI-compatible base every other harness is given
    (``http://host:3000/v1``). Claude Code appends ``/v1/messages`` to whatever
    ``ANTHROPIC_BASE_URL`` holds, so the ``/v1`` has to come off or the request goes
    to ``/v1/v1/messages``.
    """
    base = (endpoint or "").rstrip("/")
    return base[:-3].rstrip("/") if base.endswith("/v1") else base


def parse_result(stdout: str) -> dict:
    """Turns and the verdict out of ``--output-format json``'s one object.

    ``is_error`` is a crash even on exit 0: Claude Code reports a run it could not
    finish (a turn cap hit mid-edit, an API error it gave up on) in the payload and
    still exits cleanly, so trusting the exit code would record that as a good run
    that simply failed the tests. Output we cannot parse costs the fields and
    nothing else: the exit code and the hidden tests still decide the task.
    """
    text = (stdout or "").strip()
    if not text.startswith("{"):
        return {}
    try:
        payload = json.loads(text)
    except ValueError:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict = {}
    turns = payload.get("num_turns")
    if isinstance(turns, int):
        out["turns"] = turns
    if payload.get("is_error") is True:
        out["crashed"] = True
    return out


class ClaudeAdapter(HarnessAdapter):
    name = "claude"
    binary = "claude"
    # Claude Code behaves better in a repo: it reads git state for context and
    # keeps its own edits diffable.
    needs_git = True

    def command(self, req: HarnessRequest) -> list[str]:
        command = [
            self.binary,
            "-p", req.prompt,
            "--model", req.model,
            "--dangerously-skip-permissions",
            "--output-format", "json",
            "--max-turns", str(MAX_TURNS),
        ]
        # Only when the run asked for one. Unset means Claude Code's own default
        # goes out, which is what every earlier measurement was taken with.
        if req.claude_effort:
            command += ["--effort", req.claude_effort]
        return command

    def env(self, req: HarnessRequest) -> dict[str, str]:
        base = messages_base(req.endpoint)
        return {
            "ANTHROPIC_BASE_URL": base,
            # Both, because which one Claude Code reads depends on the path it
            # takes to build the client, and the endpoint does not authenticate:
            # this is the literal placeholder, same as every other adapter.
            "ANTHROPIC_API_KEY": req.api_key,
            "ANTHROPIC_AUTH_TOKEN": req.api_key,
            # The served model answers both the main and the small-fast slot: there
            # is only one model on the endpoint, and leaving the small slot pointed
            # at an Anthropic default would send the run's cheap calls nowhere.
            "ANTHROPIC_MODEL": req.model,
            "ANTHROPIC_SMALL_FAST_MODEL": req.model,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_TELEMETRY": "1",
            # Never the operator's own profile. See the module docstring.
            "CLAUDE_CONFIG_DIR": str(req.scratch / CONFIG_DIR_NAME),
        }

    def parse(self, stdout: str, stderr: str) -> dict:
        return parse_result(stdout)
