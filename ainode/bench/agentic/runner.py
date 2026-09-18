"""The agentic rubric proper: the transport, the loop over the probes, the record.

The measurement is one bit per probe, and everything here exists to keep that bit
honest the way the harness bench next door does:

  * **Every verdict is mechanical.** No judge model, no reading of replies. Group C
    executes what the model wrote against asserts it never saw; group G judges the
    call trace of a real multi-turn tool loop. A rubric scored by eye is a rubric
    that drifts between runs.
  * **A probe that raises is a failed probe, not a failed run.** One broken check
    must not throw away the twenty-three that measured cleanly, so the loop catches
    and records. The same rule the throughput bench's sections run under.
  * **Nothing is loaded, unloaded or restarted.** This drives inference against an
    endpoint that is already serving, and it adds real load to it. Group E sends a
    100k-token prompt: on a busy node that is felt.
  * **A skipped probe is absent, never a zero.** ``--quick`` and ``--groups`` change
    what was asked, and the record says which groups ran rather than scoring the
    ones that did not.

Stdlib only, like the rest of ``ainode/bench``: urllib for the transport, so a bare
python3 with no pip step can run it.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from ainode.bench.agentic.probes import (
    DEFAULT_NEEDLE,
    GROUPS,
    MAX_TURNS,
    Probe,
    ProbeResult,
)

SCHEMA = 1
SOURCE = "scripts/ainode-bench.py agentic"

#: Seconds one request gets. The 100k-token needle prefill is the slow one, and on
#: a loaded cluster node it is minutes rather than seconds.
DEFAULT_TIMEOUT = 900
#: The hand-run rubric took vLLM's own default, which is 1.0. Sent explicitly so a
#: record says what the run asked for instead of implying a server default.
DEFAULT_TEMPERATURE = 1.0
#: The chat-template switch name for "no thinking". Qwen-family templates read
#: enable_thinking; DeepSeek V4 reads thinking; both are sent (see thinking_off).
DEFAULT_THINK_KW = "enable_thinking"
DEFAULT_API_KEY = "ainode"
#: How much of an error body is worth keeping in a note.
ERROR_CHARS = 300


# ---------------------------------------------------------------- transport

@dataclass
class Reply:
    """One assistant message, with what it cost and what went wrong.

    Never raises: a transport or HTTP failure comes back as ``error`` (and
    ``status`` for an HTTP code), because a model that refuses an image or a
    response_format is a probe result, not a crashed run.
    """

    content: str | None = None
    reasoning: str = ""
    message: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    wall_s: float = 0.0
    finish_reason: str | None = None
    status: int | None = None
    error: str | None = None

    @property
    def text(self) -> str:
        return (self.content or "").strip()

    @property
    def tool_calls(self) -> list:
        return list((self.message or {}).get("tool_calls") or [])

    @property
    def calls(self) -> list:
        """``(name, raw_arguments)`` per tool call, which is what the checkers read."""
        out = []
        for call in self.tool_calls:
            fn = call.get("function") or {}
            out.append((fn.get("name") or "", fn.get("arguments")))
        return out

    @property
    def completion_tokens(self):
        return (self.usage or {}).get("completion_tokens")


def chat_url(endpoint: str) -> str:
    """``<endpoint>/chat/completions``, adding the ``/v1`` if it was left off.

    ``--endpoint`` is documented in its ``/v1`` form, the same as the harness
    bench's. A base without it is still what somebody meant, and guessing wrong
    here costs a whole run.
    """
    base = (endpoint or "").rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/chat/completions"


class ChatClient:
    """Chat completions over urllib, with the run's defaults folded in.

    Every probe talks to this and nothing else, which is what lets the tests drive
    the whole rubric off a scripted fake with the same three methods.
    """

    def __init__(self, endpoint: str, model: str, api_key: str = "",
                 timeout: float = DEFAULT_TIMEOUT,
                 temperature: float = DEFAULT_TEMPERATURE,
                 think_kw: str = DEFAULT_THINK_KW):
        self.endpoint = endpoint
        self.url = chat_url(endpoint)
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.think_kw = think_kw
        self.requests = 0

    def thinking_off(self) -> dict:
        """``chat_template_kwargs`` that turns thinking off under either spelling.

        Both switch names go out together, the way ``ainode/bench/measure.py`` sends
        them: a template ignores the key it does not read, and sending only
        ``enable_thinking`` left DeepSeek V4 thinking with the switch "off".
        """
        return {"enable_thinking": False, "thinking": False, self.think_kw: False}

    def chat(self, messages, **kw) -> Reply:
        payload = {"model": self.model, "messages": messages,
                   "temperature": self.temperature}
        payload.update(kw)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                                    headers=headers)
        self.requests += 1
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "ignore")[:ERROR_CHARS]
            except Exception:
                pass
            return Reply(wall_s=time.monotonic() - start, status=exc.code,
                         error=f"HTTP {exc.code}: {body.strip()}")
        except Exception as exc:
            return Reply(wall_s=time.monotonic() - start,
                         error=f"{type(exc).__name__}: {str(exc)[:ERROR_CHARS]}")
        wall = time.monotonic() - start
        choices = data.get("choices") or []
        if not choices:
            return Reply(wall_s=wall, error="no choices in the response")
        choice = choices[0]
        message = choice.get("message") or {}
        return Reply(content=message.get("content"),
                     reasoning=(message.get("reasoning_content")
                                or message.get("reasoning") or ""),
                     message=message, usage=data.get("usage") or {}, wall_s=wall,
                     finish_reason=choice.get("finish_reason"))

    def ask(self, prompt, system=None, **kw) -> Reply:
        """One user message (a string or a content-part list), optional system."""
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kw)


# ---------------------------------------------------------------- one probe's row

@dataclass
class ProbeRun:
    """What one probe did, as it lands in the record."""

    id: str
    group: str
    passed: bool
    note: str = ""
    wall_s: float = 0.0
    completion_tokens: int | None = None
    excerpt: str = ""

    def as_json(self) -> dict:
        return {"id": self.id, "group": self.group, "pass": self.passed,
                "wall_s": round(self.wall_s, 2),
                "completion_tokens": self.completion_tokens,
                "note": self.note, "excerpt": self.excerpt}


def say(message) -> None:
    """Print a probe line and flush it.

    A run takes minutes and is usually watched, often through a redirect into a
    file. Block buffering would show nothing until the end, which is how a long
    run looks like a hung one.
    """
    print(message, flush=True)


def run_probes(probes, client, log=say) -> list:
    """Every probe in order. A probe that raises is recorded as a failure.

    The line printed per probe is the same shape the hand-run script printed, since
    that is the thing somebody watches for ten minutes.
    """
    runs = []
    for probe in probes:
        try:
            result = probe.run(client)
        except Exception as exc:                     # a broken check is one failure
            result = ProbeResult(False, f"probe raised {type(exc).__name__}: "
                                        f"{str(exc)[:200]}")
        run = ProbeRun(id=probe.id, group=probe.group, passed=bool(result.passed),
                       note=result.note, wall_s=result.wall_s,
                       completion_tokens=result.completion_tokens,
                       excerpt=result.excerpt)
        runs.append(run)
        tokens = "-" if run.completion_tokens is None else run.completion_tokens
        log(f"  [{'PASS' if run.passed else 'FAIL'}] {run.id:<20} "
            f"{run.wall_s:6.1f}s ctok={tokens:<6} {run.note}")
    return runs


# ---------------------------------------------------------------- scoring

def score(runs) -> dict:
    return {"pass": sum(1 for r in runs if r.passed), "total": len(runs)}


def group_scores(runs) -> dict:
    """Per group, in the order the groups run. A group nobody ran is absent."""
    out = {}
    for run in runs:
        block = out.setdefault(run.group, {"pass": 0, "total": 0})
        block["total"] += 1
        block["pass"] += 1 if run.passed else 0
    return {g: out[g] for g in GROUPS if g in out}


def needle_map(runs) -> dict:
    """``{"8000": true, ...}`` for the needle probes that ran."""
    out = {}
    for run in runs:
        if run.group == "E" and run.id.startswith("E_needle_"):
            out[run.id.rsplit("_", 1)[-1]] = run.passed
    return out


def supported(runs, probe_id: str):
    """True/False for a probe that ran, None for one that did not.

    ``thinking_off_supported`` and ``vision_supported`` are statements about the
    model. A run that skipped the group has nothing to say about it, and False
    would be a claim nobody measured.
    """
    for run in runs:
        if run.id == probe_id:
            return run.passed
    return None


def structured_output_mode(probes, runs):
    """Which ``response_format`` the G4 probe got through with, or None.

    Recorded because a pass under ``json_object`` is a weaker statement than a pass
    under ``json_schema``: the first only asked for JSON, the second pinned the
    schema server-side.
    """
    ran = {r.id for r in runs}
    for probe in probes:
        mode = getattr(probe, "mode", None)
        if mode and probe.id in ran:
            return mode
    return None


# ---------------------------------------------------------------- the record

def build_agentic_block(runs, probes, endpoint: str, groups, needle,
                        temperature: float = DEFAULT_TEMPERATURE,
                        timeout: float = DEFAULT_TIMEOUT,
                        think_kw: str = DEFAULT_THINK_KW) -> dict:
    """The record's ``agentic`` block. See bench/SCHEMA.md."""
    return {
        "score": score(runs),
        "groups": group_scores(runs),
        "endpoint": endpoint,
        "protocol": {"groups": list(groups), "needle_tokens": list(needle),
                     "temperature": temperature, "timeout_s": timeout,
                     "thinking_switch": think_kw, "max_turns": MAX_TURNS,
                     "verdicts": "mechanical; group C is executed, group G is a "
                                 "judged tool trace"},
        "probes": [r.as_json() for r in runs],
        "needle": needle_map(runs),
        "thinking_off_supported": supported(runs, "F1_thinking_off"),
        "vision_supported": supported(runs, "V1_vision"),
        "structured_output_mode": structured_output_mode(probes, runs),
    }


def build_notes(runs, seconds: int, groups, source: str = SOURCE) -> list:
    notes = [f"Measured by {source} in {seconds}s; nothing loaded, unloaded or "
             "restarted.",
             "Every verdict is mechanical: no judge model and no reply was read by "
             "a person to reach a score.",
             "Group C runs the model's own code against asserts it never saw, in a "
             "subprocess on the machine driving the bench.",
             "Group G drives real tool loops and judges the call trace: the order "
             "of the calls, the arguments, and whether a stated fact came back from "
             "a tool.",
             f"Groups run: {', '.join(groups)}. A group that did not run is absent "
             "from the score rather than counted as zero."]
    broken = ("HTTP ", "probe raised ", "URLError", "TimeoutError", "socket")
    errors = [r.id for r in runs if not r.passed and r.note.startswith(broken)]
    if errors:
        notes.append("Probe(s) that failed on a transport or server error rather "
                     f"than on the answer: {', '.join(errors)}.")
    return notes


def build_record(label: str, model_block: dict, placement: dict, agentic_block: dict,
                 settings: dict, notes: list, stamp: str,
                 source: str = SOURCE) -> dict:
    """A schema-1 record with an ``agentic`` block and no ``results`` block.

    Deliberately no ``results``: this run measured no throughput, and a zero in
    ``single_stream`` would be a number nobody took.
    ``scripts/render-bench-table.py`` keeps a record shaped like this out of the
    speed table and gives it a row in the agentic table instead.
    """
    return {"schema": SCHEMA, "stamp": stamp, "label": label, "model": model_block,
            "placement": placement, "settings": settings, "agentic": agentic_block,
            "notes": notes, "source": source}


__all__ = ["ChatClient", "Reply", "Probe", "ProbeRun", "DEFAULT_NEEDLE",
           "DEFAULT_TEMPERATURE", "DEFAULT_THINK_KW", "DEFAULT_TIMEOUT",
           "DEFAULT_API_KEY", "SCHEMA", "SOURCE", "build_agentic_block",
           "build_notes", "build_record", "chat_url", "group_scores", "needle_map",
           "run_probes", "score", "structured_output_mode", "supported"]
