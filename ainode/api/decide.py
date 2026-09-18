"""``POST /v1/decide``: typed questions in, calibrated probabilities out.

A decision endpoint rather than a chat one. The caller hands over a state, an
optional block of domain guidance and a dict of independent multiple-choice
questions; every question is asked at once and each answer comes back with the
probability the served model put on it.

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
import time
from typing import Any, Optional

import aiohttp
from aiohttp import web

from ainode.api.chat_routes import instance_caps_index

SYSTEM_PROMPT = ("You are a decision function. Answer with the single letter "
                 "of the best option and nothing else.")

ANSWER_INSTRUCTION = "Answer with the label of one option and nothing else."

# 255 is the ceiling on options per question: the label scheme stays inside two
# letters (A..Z, AA..IU) and a 255-way softmax is already past the point where
# the tail is measurable.
MAX_OPTIONS = 255

# Room for the longest label plus the end-of-turn token the template emits.
LABEL_TOKEN_HEADROOM = 1

# A question's engine call. Long enough for a cold-ish engine to answer two
# tokens, short enough that a wedged node does not hold the whole request open.
CALL_TIMEOUT_S = 180.0

# A dead or ghost node must fail the connect fast so failover moves on.
CONNECT_TIMEOUT_S = 5.0

BOOLEAN_OPTIONS = ("yes", "no")

DEFAULT_SCORE_MIN = 1
DEFAULT_SCORE_MAX = 5


class DecideError(Exception):
    """A bad request shape. Carries the message the caller gets in a 400."""


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


def _serialize_state(state: Any) -> str:
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
        out[key] = {"question": text.strip(), "options": list(options)}
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
        "top_logprobs": 20,
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


def distribution_from_logprobs(labels: list[str],
                               top_logprobs: list[dict]) -> Optional[dict]:
    """Softmax over the first token's logprobs, restricted to the label tokens.

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
    weights = {label: math.exp(lp - top) for label, lp in scored.items()}
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
                          latency_ms: float) -> dict:
    """One ``decisions`` entry from one engine response. Pure.

    Probabilities are reported against the OPTION strings, not the labels: the
    labels are an implementation detail of constraining the engine, and a caller
    that had to map them back would be doing our job.
    """
    labels = option_labels(len(options))
    by_label = dict(zip(labels, options))
    chosen = constrained_label(payload, labels)
    dist = distribution_from_logprobs(labels, first_token_top_logprobs(payload))
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


async def ask_one(session: aiohttp.ClientSession, candidates: list, body: dict
                  ) -> tuple:
    """One question, with the proxy's failover. Returns (payload, cand, latency_ms).

    A transport failure or a 5xx moves to the next candidate: a ghost node that
    still advertises the model is indistinguishable from a live one in cluster
    state. Any other non-200 is the engine's own answer and stops the loop. On
    failure the third element is the error string instead of a latency.
    """
    started = time.monotonic()
    last_err = "no candidate node"
    timeout = aiohttp.ClientTimeout(total=CALL_TIMEOUT_S,
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
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
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


# -------------------------------------------------------------------- handler


def _bad_request(message: str) -> web.Response:
    return web.json_response({"error": {"message": message,
                                        "type": "invalid_request_error"}},
                             status=400)


def _unavailable(message: str) -> web.Response:
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
        state = _serialize_state(body.get("state"))
        instructions = body.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise DecideError("'instructions' must be a string when given")
    except DecideError as exc:
        return _bad_request(str(exc))

    # Tag the request so the server-view log middleware attributes it correctly.
    try:
        request["_log_model"] = model
    except Exception:
        pass

    candidates = candidates_for(request, model)
    if not candidates:
        return _unavailable(f"no node is serving '{model}'")

    session: aiohttp.ClientSession = request.app["client_session"]
    keys = list(questions)
    bodies = []
    for key in keys:
        spec = questions[key]
        labels = option_labels(len(spec["options"]))
        messages = build_messages(state, instructions, spec["question"],
                                  spec["options"])
        bodies.append(build_chat_body(model, messages, labels))

    # All questions in flight at once. They share a byte-identical prompt prefix,
    # so once one of them has prefilled it the rest read the shared state out of
    # the engine's prefix cache instead of paying for it again.
    results = await asyncio.gather(
        *(ask_one(session, candidates, b) for b in bodies))

    decisions: dict = {}
    payloads: list[dict] = []
    landed = None
    failures: list[str] = []
    for key, (payload, cand, extra) in zip(keys, results):
        if payload is None:
            failures.append(f"{key}: {extra}")
            continue
        payloads.append(payload)
        landed = landed or cand
        decisions[key] = decision_from_payload(payload, questions[key]["options"],
                                               float(extra))
    collector = request.app.get("metrics_collector")
    total_ms = (time.monotonic() - started) * 1000
    if failures:
        # A 200 always carries every question. A partial answer set would read
        # like a decision the model declined to make, and the bench treats this
        # response shape as fixed.
        if collector is not None:
            collector.record_request(model, total_ms, error=True)
        return _unavailable(f"engine calls failed for '{model}': "
                            + "; ".join(failures[:5]))

    if collector is not None:
        collector.record_request(model, total_ms, error=False)
    return web.json_response({
        "model": model,
        "node": node_name_for(request, model, landed),
        "latency_ms": round(total_ms, 1),
        "decisions": decisions,
        "usage": merge_usage(payloads),
    })
