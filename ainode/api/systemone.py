"""``POST /v1/systemone``: TypeSafe's Jev wire format, answered by a local model.

Why this route exists: a growing set of clients is written against TypeSafe's
hosted System One endpoint, not against ours. Titanium's JDE asks through
``jevJudge({endpoint, model})``, browser-use's jev-ultrafast, TypeSafe's own
Python SDK and the playground all post the same body to one path. This is that
path, on a node, so pointing any of them at a model on this fleet is one string
and no fork of the client. ONE adapter, so none of them needs an AINode-shaped
branch.

What it does NOT promise: the hosted service's calibration. Its numbers are a
property of the model TypeSafe trained and of how they fit it; these are the
probabilities the served model put on the option labels, read off its logprobs
and renormalized. The one correction applied is the served model's OWN: a
decision adapter that ships ``temperatures.json`` beside its weights gets each
question's logprobs divided by the temperature fitted for that question's type,
in the decision core (``api/decide.py``), and the response's top-level
``calibration`` block says whether that happened and with which temperatures. A
request sends ``"calibration": "raw"`` to get the engine's own spread instead. A
local model's confidence is worth what that model's confidence is worth, and the
way to find out is ``scripts/ainode-bench.py decide``, which scores exactly this
against labels.

Two numbers on the way out are inferences rather than readings, and each says so
where it is computed: ``confidence`` is the hosted service's chance-corrected
formula (``normalized_confidence``), reproduced from its published examples
because no document states it, and the usage block falls back to the bench's own
estimate when an engine reports no usage at all (``usage_block``). The
distribution goes out beside them unchanged by either, so a caller who disagrees
with either has the numbers they came from.

One engine call answers one question, which is also what bounds a question: only
the top ``MAX_CRITERIA`` labels come back with a probability, so a wider option
set is refused with the cap named rather than answered with a distribution that
lost its tail.

It is a translation layer and not a second decision endpoint. Resolving the
model, composing the grammar-constrained completions, the failover and reading
the distributions are ``api/decide.py``'s, called here through
``run_questions``; this module owns the shape on the wire in both directions and
nothing else. A question type gains a meaning by being translated here, never by
a second engine path.

Latency, because a Jev client arrives with a hosted service's deadline in hand:
JDE's production ``timeout_ms`` is 750, which no chat model on this fleet will
meet. Every question is a full prefill of the state plus one constrained token,
so hundreds of milliseconds per question is the good case and a cold engine is
worse. A JDE user pointing at a node raises ``timeout_ms`` to the measured p95
of THAT node rather than trusting the hosted default, and the questions of one
ask run concurrently, so the ask costs about one question plus the spread.

Unknown fields on a question are ignored, never echoed: JDE strips its own
internal fields (``passingAnswer``, which names the answer it counts as good)
before the wire on purpose, and a field that arrives anyway is code's, not
something a model should read or something this route should hand back.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, NamedTuple, Optional

from aiohttp import web

from ainode.api.decide import (
    CHOICE,
    NOUL,
    SCORE,
    TOP_LOGPROBS,
    DecideError,
    calibration_mode,
    candidates_for,
    merge_usage,
    normalize_questions,
    resolve_model,
    run_questions,
    serialize_state,
    unavailable,
)

# The three question types the format defines, which are also the kinds a decision
# adapter's temperatures are fitted per (``decide.py``). Anything else is a 422
# rather than a guess: a client that asked for a kind of judgement this route does
# not have is better off being told which kinds it has.
QUESTION_TYPES = (CHOICE, NOUL, SCORE)

# A noul's two options, in this order, always. The answer is P(true), so `true`
# has to be an option the engine can pick whether or not the caller described it,
# and its probability has to be readable without consulting the caller's key
# order.
NOUL_OPTIONS = ("true", "false")

# A score is an ordered rubric. Two levels is the smallest thing that is still a
# degree rather than a yes or no, and ten is where the format stops.
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

# The most criteria a question can carry here, which is NOT the format's ceiling
# of 255. One engine call reports probabilities for the top `TOP_LOGPROBS` tokens
# only, so option 21 comes back at 0 whatever the model thought of it. A caller
# with a wider taxonomy is told the cap rather than handed a distribution that
# quietly lost its tail (raising it means a second pass over the remaining labels,
# which is a measurement, not a constant).
MAX_CRITERIA = TOP_LOGPROBS

# The rate the bench falls back to when it cannot measure a model's tokenizer
# (``bench/measure.py::calibrate_cpt``). Used for nothing but the usage block
# below, and only when the engine reported no usage at all.
CHARS_PER_TOKEN = 4.0


class Translated(NamedTuple):
    """One Jev question as the decision core sees it, plus the way back out.

    ``options`` is what the model reads, one line per option. ``names`` is what
    each of those options answers to on the wire, in the same order: a choice
    criteria key verbatim, ``true`` / ``false``, or a score level's position.
    Keeping the pair here is what lets the answer name the caller's own key
    rather than the letter the engine was constrained to.
    """

    kind: str
    question: str
    options: list[str]
    names: list[str]


# ----------------------------------------------------------------- translate in


def option_text(name: str, description: Any) -> str:
    """One option line: the name the answer will carry, then what it means.

    The name comes first and VERBATIM because it is the string the caller's
    client compares against, and a model that has read it beside its description
    is choosing between meanings rather than between labels. The description is
    flattened to one line, because the prompt renders one option per line and a
    description with a newline in it would read as two options.
    """
    if isinstance(description, str) and description.strip():
        return f"{name}: {' '.join(description.split())}"
    return name


def criteria_pairs(key: str, criteria: Any, kind: str) -> list[tuple[str, Any]]:
    """The ``(name, description)`` pairs of an object ``criteria``, in order.

    Insertion order is the rubric order for a score, and JSON parsing preserves
    it, so nothing here sorts. A name is validated stripped and kept as written:
    the answer has to carry the caller's own key back, byte for byte, because the
    caller's code looks that key up.
    """
    if not isinstance(criteria, dict) or not criteria:
        raise DecideError(
            f"question '{key}': a {kind} question needs a non-empty 'criteria' "
            "object of {name: description}")
    pairs: list[tuple[str, Any]] = []
    for name, description in criteria.items():
        if not isinstance(name, str) or not name.strip():
            raise DecideError(f"question '{key}': every 'criteria' name must be a "
                              f"non-empty string (got {name!r})")
        if description is not None and not isinstance(description, str):
            raise DecideError(f"question '{key}': the 'criteria' description for "
                              f"'{name}' must be a string")
        pairs.append((name.strip(), description))
    return pairs


def choice_options(key: str, criteria: Any) -> tuple[list[str], list[str]]:
    """A choice question's options and the criteria keys they answer to."""
    pairs = criteria_pairs(key, criteria, "choice")
    if len(pairs) < 2:
        raise DecideError(f"question '{key}': a choice needs at least 2 'criteria' "
                          f"options, got {len(pairs)}")
    if len(pairs) > MAX_CRITERIA:
        raise DecideError(
            f"question '{key}': {len(pairs)} 'criteria' options is more than the "
            f"{MAX_CRITERIA} this node can report a probability for. One engine "
            f"call carries back the top {TOP_LOGPROBS} labels, so a wider option "
            "set would answer with a distribution missing its tail")
    return ([option_text(name, desc) for name, desc in pairs],
            [name for name, _ in pairs])


def noul_options(key: str, criteria: Any) -> tuple[list[str], list[str]]:
    """A noul's two options, always ``true`` then ``false``.

    The criteria block is optional here, and may describe one side only: some
    clients send both, some send neither, and a yes-or-no question is still
    answerable from its instructions alone. What a caller may not do is rename
    the sides, because the answer is P(true) and nothing else can stand in for
    it.
    """
    described: dict[str, Any] = {}
    if criteria is not None:
        if not isinstance(criteria, dict):
            raise DecideError(f"question '{key}': 'criteria' must be an object of "
                              "{true: description, false: description}")
        for name, description in criteria.items():
            flat = name.strip() if isinstance(name, str) else name
            if flat not in NOUL_OPTIONS:
                raise DecideError(f"question '{key}': a noul's 'criteria' names only "
                                  f"'true' and 'false' (got {name!r})")
            if description is not None and not isinstance(description, str):
                raise DecideError(f"question '{key}': the 'criteria' description for "
                                  f"'{flat}' must be a string")
            described[flat] = description
    return ([option_text(name, described.get(name)) for name in NOUL_OPTIONS],
            list(NOUL_OPTIONS))


def score_options(key: str, criteria: Any) -> tuple[list[str], list[str]]:
    """A score's levels in rubric order: a list by position, an object by insertion.

    Both spellings are accepted because both are in the wild: the list form names
    the levels and nothing else, the object form names them and says what each
    one means. Either way position 0 is the first level the caller wrote, which
    is what the legend and the expected score are counted against.
    """
    if isinstance(criteria, list):
        names: list[str] = []
        for level in criteria:
            if not isinstance(level, str) or not level.strip():
                raise DecideError(f"question '{key}': every 'criteria' level must be "
                                  f"a non-empty string (got {level!r})")
            names.append(level.strip())
        options = list(names)
    elif isinstance(criteria, dict):
        pairs = criteria_pairs(key, criteria, "score")
        names = [name for name, _ in pairs]
        options = [option_text(name, desc) for name, desc in pairs]
    else:
        raise DecideError(
            f"question '{key}': a score question needs 'criteria', either an ordered "
            "list of levels or an object of {level: description}")
    if not MIN_SCORE_LEVELS <= len(names) <= MAX_SCORE_LEVELS:
        raise DecideError(
            f"question '{key}': a score's 'criteria' needs {MIN_SCORE_LEVELS} to "
            f"{MAX_SCORE_LEVELS} ordered levels, got {len(names)}")
    if len(set(names)) != len(names):
        raise DecideError(f"question '{key}': 'criteria' repeats a level name. Every "
                          "level must be distinct so a score names one of them")
    return options, names


def translate_one(key: str, spec: Any) -> Translated:
    """One question off the wire. Reads three fields and ignores the rest.

    ``type``, ``instructions`` and ``criteria`` are the whole question as far as
    this route is concerned, which is also exactly what JDE's
    ``questionsForWire`` sends. A field beyond them belongs to the caller's own
    code, so it is neither read nor echoed.
    """
    if not isinstance(spec, dict):
        raise DecideError(f"question '{key}' must be an object")
    kind = spec.get("type")
    if kind not in QUESTION_TYPES:
        raise DecideError(f"question '{key}': 'type' must be one of "
                          f"{', '.join(QUESTION_TYPES)} (got {kind!r})")
    instructions = spec.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise DecideError(f"question '{key}' needs a non-empty 'instructions' string")
    criteria = spec.get("criteria")
    if kind == NOUL:
        options, names = noul_options(key, criteria)
    elif kind == SCORE:
        options, names = score_options(key, criteria)
    else:
        options, names = choice_options(key, criteria)
    return Translated(kind, instructions.strip(), options, names)


def translate_questions(raw: Any) -> dict[str, Translated]:
    """Every question in the body, in the order the caller wrote them."""
    if not isinstance(raw, dict) or not raw:
        raise DecideError("'questions' must be a non-empty object of {id: question}")
    out: dict[str, Translated] = {}
    for key, spec in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise DecideError("every question id must be a non-empty string")
        out[key] = translate_one(key, spec)
    return out


def decide_questions(translated: dict[str, Translated]) -> dict[str, dict]:
    """The ``/v1/decide`` question block for a translated set.

    Goes through the core's own ``normalize_questions`` rather than around it, so
    the option ceiling, the two-option floor and distinct options are checked in
    one place for both routes. Each question keeps its Jev type as its ``kind``,
    which is what picks the adapter's temperature for it.
    """
    questions = normalize_questions({key: {"question": item.question,
                                           "options": item.options}
                                     for key, item in translated.items()})
    for key, item in translated.items():
        questions[key]["kind"] = item.kind
    return questions


# ---------------------------------------------------------------- translate out


def probability(value: Any) -> float:
    """A finite probability inside [0, 1], because a foreign parser demands one.

    JDE's ``parseAnswers`` reads a confidence outside [0, 1] as 0, and a ``noul``
    outside it as a malformed answer SET, discarding every other answer in the
    reply with it. Rounding is the only thing here that can land a hair outside,
    and a NaN from an engine that reported one is the only thing that can land
    outside the reals, but a whole judgement is too much to lose to either.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(1.0, max(0.0, number))


def normalized_confidence(top: Any, options: int) -> float:
    """``(n * p_max - 1) / (n - 1)``: the hosted service's `confidence`, chance corrected.

    This is NOT the picked option's probability, and getting that wrong would make
    every band a JDE user already tuned read too high. On the hosted endpoint a
    two-way question at 0.6 and a ten-way question at 0.6 do not report the same
    confidence: the number is how far above chance the winner is, so 1/n reports 0
    and certainty reports 1, whatever n is.

    INFERRED, not specified: it reproduces every example in TypeSafe's published
    docs and SDK types for both choice and score, and Kev's playground authors
    arrived at the same formula for choice, but no document states it. The
    distribution it came from goes out in ``probabilities`` beside it, so a
    caller who disagrees with the formula has the numbers it came from.
    """
    spread = probability(top)
    if options < 2:
        return spread
    return probability((options * spread - 1.0) / (options - 1))


def answer_from_decision(item: Translated, entry: dict) -> Optional[dict]:
    """One Jev answer from one ``/v1/decide`` decision. Pure.

    The core reports probabilities against the option strings the model read, so
    the first thing here is putting them back under the names the caller's client
    expects: its own criteria key, ``true`` / ``false``, or a level's position.

    None when the engine named no option this route can read. The handler turns
    that into the same 503 a failed call gets, because a client reading an answer
    set it cannot parse reads the whole judgement as failed anyway, and a made-up
    option would be worse than either.

    An engine that reported no logprobs answered under the grammar but offered no
    spread. The answer stands and ``probabilities`` is left OFF the answer, which
    the format allows, rather than filled with a distribution nobody measured.
    """
    picked = dict(zip(item.options, item.names)).get(entry.get("answer"))
    if picked is None:
        return None
    dist = entry.get("distribution") or {}
    by_name = {name: probability(dist.get(option))
               for option, name in zip(item.options, item.names)}
    # The core reports the picked option's own probability; the format wants that
    # corrected for how many ways the question split.
    confidence = round(normalized_confidence(entry.get("confidence"),
                                             len(item.names)), 6)

    if item.kind == NOUL:
        # P(true) is the answer. With no spread to read it is the engine's own
        # pick, which is the one thing that is known.
        return {"type": NOUL,
                "noul": by_name["true"] if dist else float(picked == "true")}

    if item.kind == SCORE:
        legend = {str(index): name for index, name in enumerate(item.names)}
        if not dist:
            return {"type": SCORE, "score": float(item.names.index(picked)),
                    "confidence": confidence, "legend": legend}
        probabilities = {str(index): by_name[name]
                         for index, name in enumerate(item.names)}
        # The expected level, not the argmax: a rubric is ordered, so a model
        # split between 3 and 4 scores 3.5 and says more than either would.
        score = sum(index * probabilities[str(index)]
                    for index in range(len(item.names)))
        return {"type": SCORE, "score": round(score, 6), "confidence": confidence,
                "legend": legend, "probabilities": probabilities}

    answer = {"type": CHOICE, "choice": picked, "confidence": confidence}
    if dist:
        answer["probabilities"] = by_name
    return answer


def estimate_tokens(text: str) -> int:
    """Tokens in ``text`` at the bench's fallback rate. An estimate, never a count."""
    if not text:
        return 0
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def usage_block(payloads: list[dict], state: str,
                translated: dict[str, Translated], answered: int) -> dict:
    """``{input_tokens, output_tokens}``: the engine's own numbers when it gives them.

    Every question is its own completion carrying the whole state, so the engine
    reports the state once per question and this adds those up unchanged. That is
    what the engines read, prefix cache or not.

    When the engine reports NO usage at all (a stub, an older build, a proxy that
    strips the block) the numbers are an ESTIMATE at the same 4 characters per
    token the bench falls back to when it cannot measure a tokenizer
    (``bench/measure.py::calibrate_cpt``), over the serialized state plus every
    question's own text, counted the same once-per-question way. The output side
    is one label token per answered question, which is the whole output the
    grammar allows. Nothing here is a tokenizer and nothing here pretends to be:
    a caller reading these as exact should ask an engine that reports them.
    """
    merged = merge_usage(payloads)
    input_tokens = merged["prompt_tokens"]
    output_tokens = merged["completion_tokens"]
    if not input_tokens:
        per_question = estimate_tokens(state)
        input_tokens = len(translated) * per_question + sum(
            estimate_tokens(item.question)
            + sum(estimate_tokens(option) for option in item.options)
            for item in translated.values())
    if not output_tokens:
        output_tokens = answered
    return {"input_tokens": int(input_tokens), "output_tokens": int(output_tokens)}


# -------------------------------------------------------------------- handler


def unprocessable(message: str) -> web.Response:
    """422, which is what the Jev format answers a request it cannot read.

    The body is the ``{"error": {"message", "type"}}`` shape every other ``/v1``
    path on this node uses. JDE keeps an error body's ``error`` field only when
    it is a string and otherwise quotes the body truncated, so the message names
    the field it refused: that name is what reaches the caller either way.
    """
    return web.json_response({"error": {"message": message,
                                        "type": "invalid_request_error"}},
                             status=422)


async def handle_systemone(request: web.Request) -> web.Response:
    """POST /v1/systemone: typed questions in the Jev shape, typed answers back.

    A 200 carries an answer for every question asked, the same rule
    ``/v1/decide`` holds: a client cannot tell a half-answered judgement from a
    model that declined, so a question nobody could answer is a 503 with the
    reason instead.
    """
    started = time.monotonic()
    raw = await request.read()
    try:
        body = json.loads(raw or b"{}")
    except ValueError as exc:
        return unprocessable(f"body is not valid JSON: {exc}")
    if not isinstance(body, dict):
        return unprocessable("body must be a JSON object")

    try:
        model = resolve_model(request, body.get("model"))
        translated = translate_questions(body.get("questions"))
        questions = decide_questions(translated)
        state = serialize_state(body.get("state"))
        calibration = calibration_mode(body.get("calibration"))
    except DecideError as exc:
        return unprocessable(str(exc))

    # Tag the request so the server-view log middleware attributes it correctly.
    try:
        request["_log_model"] = model
    except Exception:
        pass

    candidates = candidates_for(request, model)
    if not candidates:
        return unavailable(f"no node is serving '{model}'")

    # No shared instructions block: in this format a question's own instructions
    # are the whole prompt for it, and the questions of one ask still never see
    # each other's answers.
    run = await run_questions(request, model, questions, state, None, candidates,
                              calibration)

    answers: dict[str, dict] = {}
    failures = list(run.failures)
    for key, entry in run.decisions.items():
        answer = answer_from_decision(translated[key], entry)
        if answer is None:
            failures.append(f"{key}: the engine named no option this route can read")
            continue
        answers[key] = answer

    collector = request.app.get("metrics_collector")
    total_ms = (time.monotonic() - started) * 1000
    if failures:
        if collector is not None:
            collector.record_request(model, total_ms, error=True)
        return unavailable(f"engine calls failed for '{model}': "
                           + "; ".join(failures[:5]))

    if collector is not None:
        collector.record_request(model, total_ms, error=False)
    return web.json_response({
        "model": model,
        "answers": answers,
        "usage": usage_block(run.payloads, state, translated, len(answers)),
        "latency_ms": round(total_ms, 1),
        "calibration": run.calibration,
    })
