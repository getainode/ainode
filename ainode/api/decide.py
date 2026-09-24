"""``POST /v1/decide``: typed questions in, calibrated probabilities out.

A decision endpoint rather than a chat one. The caller hands over a state, an
optional block of domain guidance and a dict of independent multiple-choice
questions; every question is asked at once and each answer comes back with the
probability the served model put on it.

The pieces below are also the core ``POST /v1/systemone`` runs on
(``api/systemone.py``, TypeSafe's Jev wire format over a local model): resolving
the model, building the prompts, asking every question at once and reading the
answers off the logprobs all happen here once, and that route translates into and
out of this shape around them. So keep them importable, and keep the pure ones
pure.

Why this is not a proxy path: the forwarded inference routes all hand ONE
upstream request the caller's own body (see ``server.py::proxy_to_vllm`` and the
invariant in ``AGENTS.md``). ``/v1/decide`` composes N chat completions of its
own making out of one request, so there is nothing to forward. It still routes
the way every other ``/v1`` path routes, by reusing the same two pieces the
proxy uses: ``_routing_candidates`` for the ordered candidate list and the
shared ``app["client_session"]`` for the transport. So a fleet head reaches the
node serving the requested model and fails over off a ghost claim, with no
second HTTP stack and no second routing rule.

How the probabilities are obtained: each question becomes one chat completion
whose output is grammar-constrained to a single option label, with
``logprobs`` / ``top_logprobs`` on. The distribution is a softmax over the first
generated token's logprobs restricted to the label tokens, renormalized. The
engine's own refusal to produce anything but a label is what makes that
restriction sound.

Calibration, when the model ships its own: a decision adapter's store directory
can carry ``temperatures.json`` beside the weights, one fitted temperature per
question kind (``choice``, ``noul``, ``score``). When the served model's
directory on this node has one, the label logprobs are divided by that kind's
temperature before the softmax, so the distribution and the confidence read the
way the adapter was fitted to be read. A request may opt out with
``"calibration": "raw"``, and every response says what was applied so a caller
can refit against the raw numbers.

vLLM field note: on the pinned engine image (``vllm/vllm-openai:v0.27.1``) the
constraint is ``structured_outputs: {"choice": [...]}``. The older
``guided_choice`` extra field is still accepted and then SILENTLY IGNORED on
0.27.1, verified by hand against a real engine where a ``guided_choice`` request
answered with free prose. So only the new spelling is sent: a dropped constraint
would be a wrong answer rather than an error, which is the worse failure.
"""

from __future__ import annotations

import asyncio
import json
import math
import logging
import time
from pathlib import Path
from typing import Any, NamedTuple, Optional

import aiohttp
from aiohttp import web

from ainode.api.chat_routes import instance_caps_index

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = ("You are a decision function. Answer with the single letter "
                 "of the best option and nothing else.")

ANSWER_INSTRUCTION = "Answer with the label of one option and nothing else."

# 255 is the ceiling on options per question: the label scheme stays inside two
# letters (A..Z, AA..IU) and a 255-way softmax is already past the point where
# the tail is measurable.
MAX_OPTIONS = 255

# How many alternatives the engine is asked for on the answer token. It is the
# ceiling on how many options can carry a probability back from one call: a label
# outside the top 20 is reported at 0 whatever the model thought of it, which is
# why `/v1/systemone` refuses a question with more criteria than this rather than
# answering one with a truncated distribution.
TOP_LOGPROBS = 20

# Room for the longest label plus the end-of-turn token the template emits.
LABEL_TOKEN_HEADROOM = 1

# A question's engine call. Long enough for a cold engine to compile the answer
# grammar for a new question shape, which measured 60 to 90 s per shape on a GB10
# and can queue behind the other questions of the same request (#277), short
# enough that a wedged node does not hold the whole request open for good.
CALL_TIMEOUT_S = 300.0

# The warm-up's engine call, per question kind. A first compile on a busy node is
# allowed to take a long time: nobody is waiting on it.
WARM_TIMEOUT_S = 600.0

# A dead or ghost node must fail the connect fast so failover moves on.
CONNECT_TIMEOUT_S = 5.0

BOOLEAN_OPTIONS = ("yes", "no")

DEFAULT_SCORE_MIN = 1
DEFAULT_SCORE_MAX = 5

# The question kinds a decision adapter is fitted per. ``/v1/systemone`` speaks
# them natively; ``/v1/decide`` maps ``type: "boolean"`` to ``noul``,
# ``type: "score"`` to ``score`` and an explicit ``options`` list to ``choice``.
CHOICE = "choice"
NOUL = "noul"
SCORE = "score"
QUESTION_KINDS = (CHOICE, NOUL, SCORE)

# What marks a decision model's store directory: the adapter's prompt contract
# and its fitted temperatures, either one. Only the second is read here.
PROMPT_CONTRACT_FILE = "prompt_contract.json"
TEMPERATURES_FILE = "temperatures.json"
DECISION_MODEL_FILES = (PROMPT_CONTRACT_FILE, TEMPERATURES_FILE)

# The one value a request's ``calibration`` field takes: the distribution exactly
# as the engine reported it, with the adapter's temperatures left off.
CALIBRATION_RAW = "raw"


class DecideError(Exception):
    """A bad request shape. Carries the message the caller gets in the 4xx.

    ``/v1/decide`` answers it as a 400 and ``/v1/systemone`` as the 422 the Jev
    format specifies, so the message says what is wrong and never which status
    somebody is about to put it in.
    """


# --------------------------------------------------------------------- labels


def option_label(index: int) -> str:
    """Zero-based option index to its letter label: A..Z, AA, AB, ... IU.

    Bijective base-26 (spreadsheet columns), so the scheme keeps going past Z
    without a separator and without ever colliding.
    """
    if index < 0:
        raise ValueError("option index cannot be negative")
    n = index + 1
    out = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out = chr(ord("A") + rem) + out
    return out


def option_labels(count: int) -> list[str]:
    """The labels for a question with `count` options, in option order."""
    return [option_label(i) for i in range(count)]


# ----------------------------------------------------------------- validation


def serialize_state(state: Any) -> str:
    """The state as the model sees it: a string verbatim, anything else compact JSON."""
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    try:
        return json.dumps(state, separators=(",", ":"), sort_keys=True,
                          ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise DecideError(f"'state' is not JSON-serializable: {exc}") from exc


def _score_options(spec: dict, key: str) -> list[str]:
    lo = spec.get("min", DEFAULT_SCORE_MIN)
    hi = spec.get("max", DEFAULT_SCORE_MAX)
    if isinstance(lo, bool) or isinstance(hi, bool) \
            or not isinstance(lo, int) or not isinstance(hi, int):
        raise DecideError(f"question '{key}': score 'min' and 'max' must be integers")
    if hi <= lo:
        raise DecideError(f"question '{key}': score 'max' must be greater than 'min'")
    if (hi - lo + 1) > MAX_OPTIONS:
        raise DecideError(
            f"question '{key}': score range spans {hi - lo + 1} values, "
            f"more than the {MAX_OPTIONS} allowed")
    return [str(v) for v in range(lo, hi + 1)]


def normalize_questions(raw: Any) -> dict[str, dict]:
    """Validate the `questions` block and expand the type sugar.

    Returns ``{key: {"question": str, "options": [str, ...]}}`` in the order the
    caller wrote them. Raises ``DecideError`` with the caller-facing message for
    every rejected shape, so the handler has one place to turn it into a 400.
    """
    if not isinstance(raw, dict) or not raw:
        raise DecideError("'questions' must be a non-empty object of "
                          "{key: {question, options|type}}")
    out: dict[str, dict] = {}
    for key, spec in raw.items():
        if not isinstance(key, str) or not key:
            raise DecideError("every question key must be a non-empty string")
        if not isinstance(spec, dict):
            raise DecideError(f"question '{key}' must be an object")
        text = spec.get("question")
        if not isinstance(text, str) or not text.strip():
            raise DecideError(f"question '{key}' needs a non-empty 'question' string")
        qtype = spec.get("type")
        kind = NOUL if qtype == "boolean" else SCORE if qtype == "score" else CHOICE
        if "options" in spec:
            options = spec["options"]
        elif qtype == "boolean":
            options = list(BOOLEAN_OPTIONS)
        elif qtype == "score":
            options = _score_options(spec, key)
        elif qtype is None:
            raise DecideError(f"question '{key}' needs 'options' or a 'type'")
        else:
            raise DecideError(
                f"question '{key}': unknown type '{qtype}' "
                f"(known: 'boolean', 'score', or pass 'options')")
        if not isinstance(options, list):
            raise DecideError(f"question '{key}': 'options' must be a list")
        for opt in options:
            if not isinstance(opt, str) or not opt.strip():
                raise DecideError(
                    f"question '{key}': every option must be a non-empty string "
                    f"(got {opt!r})")
        if len(options) < 2:
            raise DecideError(
                f"question '{key}': needs at least 2 options, got {len(options)}")
        if len(options) > MAX_OPTIONS:
            raise DecideError(
                f"question '{key}': {len(options)} options is more than the "
                f"{MAX_OPTIONS} allowed")
        if len(set(options)) != len(options):
            dupes = sorted({o for o in options if options.count(o) > 1})
            raise DecideError(
                f"question '{key}': duplicate options {dupes}. Every option must "
                f"be distinct so an answer is unambiguous")
        out[key] = {"question": text.strip(), "options": list(options), "kind": kind}
    return out


# -------------------------------------------------------------------- prompts


def build_messages(state: str, instructions: Optional[str], question: str,
                   options: list[str]) -> list[dict]:
    """The chat messages for one question. Pure, so the tests can pin the text.

    The state comes BEFORE the question on purpose: every question in a request
    then shares a byte-identical prefix (system message plus state), so the
    engine's prefix cache prefills the shared part once no matter how many
    questions are asked against it.
    """
    system = SYSTEM_PROMPT
    extra = (instructions or "").strip()
    if extra:
        system = f"{SYSTEM_PROMPT}\n\n{extra}"
    lines = ["STATE:", state, "", f"QUESTION: {question}", "", "OPTIONS:"]
    lines += [f"{option_label(i)}. {opt}" for i, opt in enumerate(options)]
    lines += ["", ANSWER_INSTRUCTION]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


def build_chat_body(model: str, messages: list[dict], labels: list[str]) -> dict:
    """The chat-completion body for one question.

    ``structured_outputs.choice`` is the vLLM 0.27.1 spelling of "emit exactly
    one of these strings"; see the module docstring for why the legacy
    ``guided_choice`` is not sent alongside it.
    """
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max(len(label) for label in labels) + LABEL_TOKEN_HEADROOM,
        "temperature": 0,
        "stream": False,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
        # Both switch names, the way ``bench/measure.py`` does it: Qwen-family
        # templates read enable_thinking, DeepSeek V4 reads thinking, and a
        # template ignores the one it does not use. A decision function must not
        # spend its two-token budget on a reasoning block.
        "chat_template_kwargs": {"enable_thinking": False, "thinking": False},
        "structured_outputs": {"choice": list(labels)},
    }


# -------------------------------------------------------------- distributions


def first_token_top_logprobs(payload: dict) -> list[dict]:
    """The ``top_logprobs`` list for the FIRST generated token, or []."""
    try:
        content = payload["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError):
        return []
    if not isinstance(content, list) or not content:
        return []
    first = content[0]
    if not isinstance(first, dict):
        return []
    tops = first.get("top_logprobs")
    if not isinstance(tops, list) or not tops:
        # No alternatives offered, but the chosen token still carries its own
        # logprob: a one-entry distribution is more honest than none.
        if isinstance(first.get("token"), str) and first.get("logprob") is not None:
            return [{"token": first["token"], "logprob": first["logprob"]}]
        return []
    return [t for t in tops if isinstance(t, dict) and isinstance(t.get("token"), str)]


def distribution_from_logprobs(labels: list[str], top_logprobs: list[dict],
                               temperature: float = 1.0) -> Optional[dict]:
    """Softmax over the first token's logprobs, restricted to the label tokens.

    ``temperature`` divides the logprobs before the softmax, which is the same as
    dividing the logits: the log-normalizer is one constant across the labels and
    drops out when the result is renormalized. 1.0 is the engine's own spread.

    A label is scored by the LONGEST token in ``top_logprobs`` that is a prefix
    of it, which is an exact match whenever the tokenizer gives the whole label
    one token (the common single-letter case, and ``AB`` on the Ornith
    tokenizer). A label no token prefixes gets 0. Then renormalize over labels.

    Known limitation, by construction: when a two-letter label is NOT a single
    token, its mass is the mass of its first letter's token, which the
    single-letter label of that same letter also claims. ``A`` and ``AA`` then
    report the same probability, and the pair is only jointly calibrated. Past 26
    options, read such a pair as "these labels together", not per label.

    Returns None when there is nothing to softmax over, which is the caller's
    signal to fall back to the constrained answer with no distribution.
    """
    if not top_logprobs:
        return None
    best: dict[str, float] = {}
    for entry in top_logprobs:
        token = entry["token"].strip()
        lp = entry.get("logprob")
        if not token or lp is None:
            continue
        try:
            lp = float(lp)
        except (TypeError, ValueError):
            continue
        # Duplicate spellings of one token (with and without a leading space)
        # collapse to the likelier of the two, not to their sum: they are the
        # same decision offered twice.
        if token not in best or lp > best[token]:
            best[token] = lp
    if not best:
        return None

    scored: dict[str, float] = {}
    for label in labels:
        prefixes = [t for t in best if label.startswith(t)]
        if not prefixes:
            continue
        scored[label] = best[max(prefixes, key=len)]
    if not scored:
        return None

    top = max(scored.values())
    weights = {label: math.exp((lp - top) / temperature)
               for label, lp in scored.items()}
    total = sum(weights.values())
    if total <= 0:
        return None
    return {label: round(weights.get(label, 0.0) / total, 6) for label in labels}


def pick_answer(labels: list[str], dist: Optional[dict],
                chosen: Optional[str]) -> Optional[str]:
    """The answer label: the distribution's argmax, ties broken toward `chosen`.

    Greedy decoding at temperature 0 makes the engine's own label maximal, so the
    tie-break only ever fires on the shared-prefix case the docstring above
    describes, where it keeps the reported answer and the engine's answer equal.
    """
    if dist:
        top = max(dist.values())
        winners = [label for label in labels if dist.get(label, 0.0) == top]
        if chosen in winners:
            return chosen
        return winners[0]
    if chosen in labels:
        return chosen
    return None


def constrained_label(payload: dict, labels: list[str]) -> Optional[str]:
    """The label the engine emitted, if its text is one of ours."""
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(content, str):
        return None
    text = content.strip()
    return text if text in labels else None


def decision_from_payload(payload: dict, options: list[str],
                          latency_ms: float, temperature: float = 1.0) -> dict:
    """One ``decisions`` entry from one engine response. Pure.

    Probabilities are reported against the OPTION strings, not the labels: the
    labels are an implementation detail of constraining the engine, and a caller
    that had to map them back would be doing our job. ``temperature`` is the
    adapter's fitted temperature for this question's kind, 1.0 for none.
    """
    labels = option_labels(len(options))
    by_label = dict(zip(labels, options))
    chosen = constrained_label(payload, labels)
    dist = distribution_from_logprobs(labels, first_token_top_logprobs(payload),
                                      temperature)
    answer_label = pick_answer(labels, dist, chosen)
    entry: dict = {
        "answer": by_label.get(answer_label),
        "confidence": None,
        "distribution": None,
        "latency_ms": round(latency_ms, 1),
    }
    if dist is None:
        # The engine still answered under the grammar, so the answer is sound;
        # only the calibration is missing. Say so rather than invent a spread.
        entry["confidence"] = 1.0 if entry["answer"] is not None else None
        entry["note"] = "no logprobs from engine"
        return entry
    entry["distribution"] = {by_label[label]: dist[label] for label in labels}
    entry["confidence"] = dist.get(answer_label)
    return entry


# ---------------------------------------------------------------- calibration


def calibration_mode(value: Any) -> Optional[str]:
    """A request's ``calibration`` field: absent for the adapter's, or ``"raw"``."""
    if value is None or value == CALIBRATION_RAW:
        return value
    raise DecideError(f"'calibration' must be \"{CALIBRATION_RAW}\" when given "
                      f"(got {value!r})")


def model_store_dir(models_dir, model: str) -> Optional[Path]:
    """The directory holding ``model``'s files in this node's store, or None.

    The store's own resolver (``models/registry.py::snapshot_dir_for``) answers
    for a repo id in every layout AINode writes; a model served straight from an
    absolute path is its own directory.
    """
    from ainode.models.registry import snapshot_dir_for
    if not model:
        return None
    try:
        direct = Path(model)
        if direct.is_absolute() and direct.is_dir():
            return direct
    except OSError:
        pass
    return snapshot_dir_for(model, Path(models_dir) if models_dir else None)


def is_decision_model_dir(directory: Optional[Path]) -> bool:
    """True when the directory carries a decision adapter's prompt contract or
    temperatures, which is how a decision model is recognised."""
    if directory is None:
        return False
    try:
        return any((Path(directory) / name).is_file() for name in DECISION_MODEL_FILES)
    except OSError:
        return False


def read_temperatures(directory: Optional[Path]) -> Optional[dict]:
    """``{kind: T}`` from the directory's ``temperatures.json``, or None.

    Only the three kinds are read, and only a finite positive number is a
    temperature: anything else leaves that kind at the engine's own spread rather
    than failing a request over an adapter's file.
    """
    if directory is None:
        return None
    try:
        raw = json.loads((Path(directory) / TEMPERATURES_FILE).read_text())
    except (OSError, ValueError):
        return None
    temps = raw.get("temperatures") if isinstance(raw, dict) else None
    if not isinstance(temps, dict):
        return None
    out: dict[str, float] = {}
    for kind in QUESTION_KINDS:
        value = temps.get(kind)
        if isinstance(value, bool):
            continue
        try:
            t = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(t) and t > 0:
            out[kind] = t
    return out or None


def calibration_for(models_dir, model: str, mode: Optional[str]) -> dict:
    """The ``calibration`` block a response carries: ``{applied, temperatures}``.

    ``temperatures`` is what the model's directory on this node carries, whether
    or not it was applied, so a caller that opted out still sees what it opted
    out of; it is null when there is none. The directory is looked up on the node
    answering the route: a model this node routes to a peer and does not hold a
    copy of answers raw, and says so with ``applied: false``.
    """
    temps = read_temperatures(model_store_dir(models_dir, model))
    return {"applied": bool(temps) and mode != CALIBRATION_RAW,
            "temperatures": temps}


# -------------------------------------------------------------------- routing


def resolve_model(request: web.Request, requested: Any) -> str:
    """The model id to decide with: the caller's, else this node's/fleet's default."""
    if requested is not None:
        if not isinstance(requested, str) or not requested.strip():
            raise DecideError("'model' must be a non-empty string when given")
        return requested.strip()
    config = request.app["config"]
    if getattr(config, "model", None):
        return config.model
    from ainode.api.server import _routing_table
    cluster = request.app.get("cluster_state")
    table = _routing_table(cluster, config.node_id, config.api_port) \
        if cluster is not None else {}
    if len(table) == 1:
        return next(iter(table))
    if not table:
        raise DecideError("no model is loaded; pass 'model' or load one first")
    raise DecideError(
        "this node has no default model and the fleet serves "
        f"{len(table)}; pass 'model' explicitly")


def candidates_for(request: web.Request, model: str) -> list:
    """The ordered (host, port) list the proxy would use for this model."""
    from ainode.api.server import _routing_candidates
    config = request.app["config"]
    cluster = request.app.get("cluster_state")
    candidates = _routing_candidates(cluster, model, config.node_id, config.api_port)
    if not candidates and (cluster is None or not cluster.members()):
        # Back-compat with the proxy: an empty fleet decides against the local
        # engine rather than reporting that nothing is serving.
        candidates = [("localhost", config.api_port)]
    return candidates


async def ask_one(session: aiohttp.ClientSession, candidates: list, body: dict,
                  timeout_s: float = CALL_TIMEOUT_S) -> tuple:
    """One question, with the proxy's failover. Returns (payload, cand, latency_ms).

    A transport failure or a 5xx moves to the next candidate: a ghost node that
    still advertises the model is indistinguishable from a live one in cluster
    state. Any other non-200 is the engine's own answer and stops the loop. On
    failure the third element is the error string instead of a latency.

    A call that connected and then ran out of time is reported as what it almost
    always is, an engine compiling the answer grammar for a question shape it has
    not seen (#277), rather than as an unreachable node.
    """
    started = time.monotonic()
    last_err = "no candidate node"
    timeout = aiohttp.ClientTimeout(total=timeout_s,
                                    sock_connect=CONNECT_TIMEOUT_S)
    for host, port in candidates:
        url = f"http://{host}:{port}/v1/chat/completions"
        try:
            async with session.post(url, json=body, timeout=timeout) as resp:
                try:
                    payload = await resp.json(content_type=None)
                except Exception:
                    payload = {}
                if resp.status == 200 and isinstance(payload, dict):
                    return payload, (host, port), (time.monotonic() - started) * 1000
                text = ""
                if isinstance(payload, dict):
                    err = payload.get("error")
                    text = (err.get("message") if isinstance(err, dict) else err) or ""
                last_err = f"{host}:{port} answered {resp.status}: {str(text)[:200]}"
                if 400 <= resp.status < 500:
                    break
        except aiohttp.ServerTimeoutError as exc:
            # The connect itself timed out (no read timeout is set): a dead node.
            last_err = f"{host}:{port} unreachable: {exc}"
            continue
        except asyncio.TimeoutError:
            last_err = (f"{host}:{port} gave no answer in {timeout_s:.0f}s: the engine "
                        "is compiling the answer grammar, retry")
            continue
        except aiohttp.ClientError as exc:
            last_err = f"{host}:{port} unreachable: {exc}"
            continue
    return None, None, last_err


def merge_usage(payloads: list[dict]) -> dict:
    """One usage block for the whole request: the engine calls added up."""
    prompt = completion = 0
    for payload in payloads:
        usage = payload.get("usage") or {}
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "calls": len(payloads)}


def node_name_for(request: web.Request, model: str, cand) -> Optional[str]:
    """The name of the node a call landed on, or None if we cannot name it."""
    if cand is None:
        return None
    entry = (instance_caps_index(request.app, model) or {}).get(tuple(cand)) or {}
    return entry.get("node_name") or None


# ------------------------------------------------------------------------ run


class DecideRun(NamedTuple):
    """What one set of questions came back as, before anybody shapes a response.

    ``decisions`` holds one entry per question the engine answered, keyed the way
    the caller keyed the question and in the caller's order; ``payloads`` are the
    raw engine responses, for the usage block; ``landed`` is the candidate the
    first answer came from, for naming the node; ``failures`` is one line per
    question that got no answer; ``calibration`` is the ``{applied,
    temperatures}`` block both routes report.
    """

    decisions: dict
    payloads: list
    landed: Optional[tuple]
    failures: list
    calibration: Optional[dict] = None


async def run_questions(request: web.Request, model: str, questions: dict[str, dict],
                        state: str, instructions: Optional[str],
                        candidates: list, calibration: Optional[str] = None
                        ) -> DecideRun:
    """Ask every question at once against `candidates`, and read the answers.

    The whole engine-facing half of a decision request, shared by ``/v1/decide``
    and ``/v1/systemone`` so there is one path to the engines and one way the
    probabilities are read. It reports what happened and decides nothing about
    the response: the status, the shape and what a failure means belong to the
    route.

    All questions are in flight together. They share a byte-identical prompt
    prefix, so once one of them has prefilled it the rest read the shared state
    out of the engine's prefix cache instead of paying for it again.

    ``calibration`` is the request's mode (``calibration_mode``). Unless it is
    ``"raw"``, each question's logprobs are divided by the temperature the
    model's directory carries for that question's ``kind`` before the softmax.
    """
    session: aiohttp.ClientSession = request.app["client_session"]
    config = request.app.get("config")
    block = await asyncio.get_running_loop().run_in_executor(
        None, calibration_for, getattr(config, "models_dir", None), model,
        calibration)
    temps = block["temperatures"] if block["applied"] else {}
    keys = list(questions)
    bodies = []
    for key in keys:
        spec = questions[key]
        labels = option_labels(len(spec["options"]))
        messages = build_messages(state, instructions, spec["question"],
                                  spec["options"])
        bodies.append(build_chat_body(model, messages, labels))

    results = await asyncio.gather(
        *(ask_one(session, candidates, b) for b in bodies))

    decisions: dict = {}
    payloads: list[dict] = []
    failures: list[str] = []
    landed = None
    for key, (payload, cand, extra) in zip(keys, results):
        if payload is None:
            failures.append(f"{key}: {extra}")
            continue
        payloads.append(payload)
        landed = landed or cand
        decisions[key] = decision_from_payload(
            payload, questions[key]["options"], float(extra),
            temps.get(questions[key].get("kind"), 1.0))
    return DecideRun(decisions, payloads, landed, failures, block)


# -------------------------------------------------------------------- warm-up
#
# The first grammar-constrained request per question shape makes vLLM compile
# the answer grammar, 60 to 90 s on a GB10, and the route's caller is the one who
# waits for it (#277). So a decision model is sent one minimal question of each
# kind as soon as its engine binds, through the same body builder and the same
# ``ask_one`` the routes use, and the instance reports ``warm`` on /api/status.
# Readiness is not held for it: the engine serves while it warms.

# Warm-ups in flight. The loop holds only a weak reference to a task.
_WARM_TASKS: set = set()


def warm_questions() -> dict[str, dict]:
    """One minimal question per kind: a two-option choice, a noul, a two-level score.

    The noul's options are ``true`` / ``false`` in the order ``/v1/systemone``
    always sends them.
    """
    return {
        CHOICE: {"question": "Which option fits the state?",
                 "options": ["first", "second"]},
        NOUL: {"question": "Is the state empty?", "options": ["true", "false"]},
        SCORE: {"question": "How complete is the state?", "options": ["1", "2"]},
    }


def served_model_id(backend) -> str:
    """The id an engine answers to: its first ``--served-model-name``, else its model."""
    cfg = getattr(backend, "config", None)
    names = getattr(cfg, "served_model_name", None) or []
    if isinstance(names, str):
        names = [names]
    return str(names[0] if names else getattr(cfg, "model", "") or "")


async def warm_decision_engine(session: aiohttp.ClientSession, port: int, model: str,
                               status: dict, timeout_s: float = WARM_TIMEOUT_S
                               ) -> bool:
    """Ask the engine on ``port`` one question per kind, in turn, and time each.

    One at a time so each compile is timed on its own and logged. ``status`` is
    the instance's entry in ``app["decision_warm"]`` and is updated in place:
    ``warm`` turns True only when every kind answered, ``compile_seconds`` holds
    each kind's time and ``error`` the first failure.
    """
    status.update(warm=False, warming=True, compile_seconds={}, error=None)
    for kind, spec in warm_questions().items():
        labels = option_labels(len(spec["options"]))
        messages = build_messages("", None, spec["question"], spec["options"])
        started = time.monotonic()
        payload, _, extra = await ask_one(session, [("localhost", port)],
                                          build_chat_body(model, messages, labels),
                                          timeout_s)
        seconds = round(time.monotonic() - started, 1)
        if payload is None:
            status.update(warming=False, error=f"{kind}: {extra}")
            logger.warning("decision warm-up of %s on :%s failed at %s after %.1fs: %s",
                           model, port, kind, seconds, extra)
            return False
        status["compile_seconds"][kind] = seconds
        logger.info("decision warm-up of %s on :%s: %s grammar ready in %.1fs",
                    model, port, kind, seconds)
    status.update(warm=True, warming=False)
    return True


async def _run_warmup(app, port: int, model: str, status: dict) -> bool:
    try:
        session = app.get("client_session")
        if session is not None:
            return await warm_decision_engine(session, port, model, status)
        async with aiohttp.ClientSession() as own:
            return await warm_decision_engine(own, port, model, status)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("decision warm-up of %s on :%s raised", model, port)
        status.update(warm=False, warming=False, error=str(exc))
        return False


def schedule_decision_warmup(app, port: int, backend) -> bool:
    """Warm ``backend`` in the background if it serves a decision model. Called on bind.

    A decision model is one whose store directory carries ``prompt_contract.json``
    or ``temperatures.json``. Anything else is left alone and its entry cleared,
    so a port that used to hold a decision model does not go on reporting it.
    Returns True when a warm-up was started.
    """
    table = app.get("decision_warm") if hasattr(app, "get") else None
    if table is None or backend is None:
        return False
    cfg = getattr(backend, "config", None)
    model = str(getattr(cfg, "model", "") or "")
    models_dir = (getattr(cfg, "models_dir", None)
                  or getattr(app.get("config"), "models_dir", None))
    if not is_decision_model_dir(model_store_dir(models_dir, model)):
        table.pop(port, None)
        return False
    status = {"model": model, "warm": False, "warming": True,
              "compile_seconds": {}, "error": None}
    table[port] = status
    task = asyncio.get_running_loop().create_task(
        _run_warmup(app, port, served_model_id(backend), status))
    _WARM_TASKS.add(task)
    task.add_done_callback(_WARM_TASKS.discard)
    return True


def instance_warm_fields(app, record) -> dict:
    """``warm`` and ``warm_compile_seconds`` for one instance on /api/status.

    ``warm`` is True once every question kind has compiled, False while a
    decision model is still warming or its warm-up failed, and null for a model
    that is not a decision model (nothing to warm).
    """
    table = app.get("decision_warm") or {}
    entry = table.get(getattr(record, "api_port", None))
    if not entry or entry.get("model") != getattr(record, "model", None):
        return {"warm": None, "warm_compile_seconds": None}
    return {"warm": bool(entry.get("warm")),
            "warm_compile_seconds": dict(entry.get("compile_seconds") or {})}


# -------------------------------------------------------------------- handler


def _bad_request(message: str) -> web.Response:
    return web.json_response({"error": {"message": message,
                                        "type": "invalid_request_error"}},
                             status=400)


def unavailable(message: str) -> web.Response:
    return web.json_response({"error": {"message": message,
                                        "type": "service_unavailable"}},
                             status=503)


async def handle_decide(request: web.Request) -> web.Response:
    """POST /v1/decide: every question asked at once, each answer with its probability."""
    started = time.monotonic()
    raw = await request.read()
    try:
        body = json.loads(raw or b"{}")
    except ValueError as exc:
        return _bad_request(f"body is not valid JSON: {exc}")
    if not isinstance(body, dict):
        return _bad_request("body must be a JSON object")

    try:
        model = resolve_model(request, body.get("model"))
        questions = normalize_questions(body.get("questions"))
        state = serialize_state(body.get("state"))
        instructions = body.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise DecideError("'instructions' must be a string when given")
        calibration = calibration_mode(body.get("calibration"))
    except DecideError as exc:
        return _bad_request(str(exc))

    # Tag the request so the server-view log middleware attributes it correctly.
    try:
        request["_log_model"] = model
    except Exception:
        pass

    candidates = candidates_for(request, model)
    if not candidates:
        return unavailable(f"no node is serving '{model}'")

    run = await run_questions(request, model, questions, state, instructions,
                              candidates, calibration)
    collector = request.app.get("metrics_collector")
    total_ms = (time.monotonic() - started) * 1000
    if run.failures:
        # A 200 always carries every question. A partial answer set would read
        # like a decision the model declined to make, and the bench treats this
        # response shape as fixed.
        if collector is not None:
            collector.record_request(model, total_ms, error=True)
        return unavailable(f"engine calls failed for '{model}': "
                           + "; ".join(run.failures[:5]))

    if collector is not None:
        collector.record_request(model, total_ms, error=False)
    return web.json_response({
        "model": model,
        "node": node_name_for(request, model, run.landed),
        "latency_ms": round(total_ms, 1),
        "decisions": run.decisions,
        "usage": merge_usage(run.payloads),
        "calibration": run.calibration,
    })
