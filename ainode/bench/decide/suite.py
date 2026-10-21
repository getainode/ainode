"""The Jevals-recipe run: two transports, five repeats per question, one record.

This is the half of the decision bench that scores an AINode-served model on the same
public question sets the independent Jevals boards use, so a number here can be read
next to Jev and its clones. The recipe itself is recorded in ``bench/decide/JEVALS.md``
with the URL and the date it was read, the formulas are
:mod:`ainode.bench.decide.jevals`, and the question sets are
:mod:`ainode.bench.decide.sets`. Nothing here restates a formula or a licence: this
module is the loop, the two wire shapes and the record.

**Two transports, one flag.** ``--transport decide`` posts AINode's own
``POST /v1/decide``; ``--transport systemone`` posts ``POST /v1/systemone`` in the Jev
wire format, which is what TypeSafe's hosted Jev speaks and what a local server that
implements the same interface speaks. The second one is why a Kev, a laya.cpp or a
TypeSafe endpoint is a flag and not a fork.

**Which key goes where is decided by the HOST, not by a flag.** A request to
``api.typesafe.ai`` resolves TypeSafe's credential and a request to anything else
resolves the node's, so no ordering of flags can post a fleet key to a vendor or a
vendor's key to one of our nodes. That is the same invariant the legacy backends hold,
made structural instead of name-based.

``request()`` and ``parse()`` are pure functions of their arguments on both transports,
so ``tests/test_bench_decide_jevals.py`` pins every request shape and every response
shape from canned payloads with no server. Only ``ask()`` touches the network.

Stdlib only, like the rest of ``ainode/bench``.
"""
from __future__ import annotations

import concurrent.futures as futures
import random
import time
import urllib.parse

from ainode.bench import auth
from ainode.bench.decide import jevals, sets
from ainode.bench.decide.backends import (
    DEFAULT_API_KEY,
    DEFAULT_TIMEOUT,
    ERROR_CHARS,
    BackendError,
    Request,
    jev_api_key,
    node_api_key,
    post_json,
)

SOURCE = "scripts/ainode-bench.py decide"
#: The record's ``decide.mode``, so a reader can tell at a glance which of the two
#: measurements in this package produced the block.
MODE = "jevals-0.1.0"
RECIPE = {"source": "https://jevals.com/methodology", "read": "2026-09-21",
          "suite": "0.1.0", "doc": "bench/decide/JEVALS.md",
          "attribution": "Jevals (jevals.com), suite 0.1.0"}

#: The two transports ``--transport`` picks from.
TRANSPORTS = ("decide", "systemone")
#: The key every request names its one question under. One question per request is the
#: recipe's batch size, and the name comes back under itself in both response shapes.
QUESTION_KEY = "decision"
#: The hosted Jev endpoint, and the host that decides a request gets TypeSafe's key.
TYPESAFE_HOST = "api.typesafe.ai"
JEV_URL = "https://api.typesafe.ai/v1/systemone"

DEFAULT_CONCURRENCY = 4
#: How many probabilities a row keeps. A full 77-option vector on 1,500 decisions is
#: most of a record, so a row keeps the five it was surest about plus the labeled
#: option's own probability, which is what a reader of a wrong answer needs.
ROW_TOP_K = 5


# ------------------------------------------------------------------ option order

def order_index(repeat: int) -> int:
    """Which presented order a repeat uses: 0 and 1 share one, then one each.

    The recipe's shape, so repeats 0 and 1 are byte-identical requests (which is what
    makes the repeat flip rate a measurement of nondeterminism) and repeats 2, 3 and 4
    are the three further orders the order flip rate compares.
    """
    return 0 if int(repeat) < 2 else int(repeat) - 1


def presented_options(options, question_id: str, seed, order: int, qtype: str) -> list:
    """The options in the order this repeat shows them.

    ``choice`` is shuffled, seeded on the suite seed, the question id and the order
    index, so the permutation is deterministic and identical for every system this bench
    runs. ``noul`` and ``score`` are never reordered, which is the recipe's rule and
    necessary for ``score``, whose levels are ordinal.

    The permutation is NOT Jevals' own: they publish the properties but not the
    generator (see JEVALS.md, "What we had to author"). Ours satisfies every published
    property except being the same permutation, so an order flip rate from this bench is
    a real order flip rate and is not their number.
    """
    if qtype != jevals.CHOICE:
        return list(options)
    shuffled = list(options)
    random.Random(f"{seed}:{question_id}:{int(order)}").shuffle(shuffled)
    return shuffled


# ------------------------------------------------------------------ criteria

def option_criteria(spec: dict) -> dict:
    """``{option: description or None}`` for any of the three primitives.

    One place turns the three shapes a question states its criteria in into the one shape
    a prompt needs: a ``choice`` map keyed by option name, a ``score`` list in level
    order, and a ``noul`` map keyed ``true``/``false`` whose ``false`` belongs to the
    first option and whose ``true`` belongs to the second.
    """
    options = list(spec["options"])
    criteria = spec.get("criteria")
    if criteria is None:
        return {option: None for option in options}
    qtype = spec["type"]
    if qtype == jevals.SCORE and isinstance(criteria, list):
        return {option: (criteria[index] if index < len(criteria) else None)
                for index, option in enumerate(options)}
    if qtype == jevals.NOUL and isinstance(criteria, dict) and len(options) == 2:
        # Accept the option names themselves as keys too: a set whose noul options are
        # already spelled `no`/`yes` states its criteria under those.
        if set(criteria) & {"true", "false"}:
            return {options[0]: criteria.get("false"), options[1]: criteria.get("true")}
        return {option: criteria.get(option) for option in options}
    if isinstance(criteria, dict):
        return {option: criteria.get(option) for option in options}
    return {option: None for option in options}


def criteria_lines(spec: dict, presented) -> str:
    """``name: description`` per option, in presented order, or "" when none is described.

    Empty when every description is null, which is Banking77: its 77 options carry names
    and nothing else, and appending 77 bare names under a heading would add words to the
    prompt that the board's prompt does not have.
    """
    described = option_criteria(spec)
    if not any(described.get(option) for option in presented):
        return ""
    lines = []
    for option in presented:
        text = described.get(option)
        lines.append(f"{option}: {text}" if text else str(option))
    return "\n".join(lines)


# ------------------------------------------------------------------ the decision row

class Answer:
    """What one request came back as, before the recipe scores it.

    ``vector`` is the raw probability map as the endpoint stated it; normalization,
    malformed detection and the pick all happen in :func:`decision_for`, once, against
    the recipe's rules, so the two transports cannot drift on any of them.
    """

    def __init__(self, vector=None, stated=None, confidence=None, tokens_in=None,
                 tokens_out=None, model="", node="", error=None, excerpt="",
                 one_hot=False, malformed_reason=None):
        self.vector = vector
        self.stated = stated
        self.confidence = confidence
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.model = model
        self.node = node
        self.error = error
        self.excerpt = excerpt
        self.one_hot = one_hot
        self.malformed_reason = malformed_reason
        self.wall_ms = 0


def spec_for(doc: dict, question: dict) -> dict:
    """The typed question for one entry. One resolution point, in :mod:`sets`."""
    return sets.question_spec(doc, question)


def decision_for(doc: dict, question: dict, repeat: int, presented, answer: Answer
                 ) -> dict:
    """One decision dict, the unit :mod:`ainode.bench.decide.jevals` scores.

    The three cases the recipe distinguishes, in one place:

      * a transport failure carries the ``error`` and is never scored;
      * an answer whose probabilities cannot be read is ``malformed``, scored as the
        uniform distribution and a wrong pick;
      * an answer with no probabilities at all is ``one_hot``: it keeps its pick, is
        scored as one-hot, and is out of the calibration numbers.
    """
    spec = spec_for(doc, question)
    options = list(spec["options"])
    qtype = spec["type"]
    decision = {
        "id": question["id"], "set": doc.get("set") or doc["id"], "type": qtype,
        "repeat": int(repeat), "order": order_index(repeat),
        "options": options, "label": (doc.get("labels") or {}).get(question["id"]),
        "space": question.get("space"),
        "gold": question.get("gold"),
        "vector": None, "pick": None, "confidence": None,
        "malformed": False, "one_hot": False,
        # What the endpoint's own probabilities summed to BEFORE renormalizing, which is
        # what JevBench's two schema-validity rates are computed over: its headline
        # renormalizes inside a 2 percent band and its strict column uses 0.001.
        "sum_before_normalize": None,
        "wall_ms": answer.wall_ms, "tokens_in": answer.tokens_in,
        "tokens_out": answer.tokens_out, "error": answer.error,
        "excerpt": answer.excerpt,
    }
    if answer.error:
        return decision
    if answer.one_hot:
        pick = answer.stated if answer.stated in options else None
        if pick is None:
            decision["malformed"] = True
            decision["malformed_reason"] = (answer.malformed_reason
                                            or "no probabilities and no known answer")
            decision["vector"] = jevals.uniform(options)
            return decision
        decision["one_hot"] = True
        decision["vector"] = {option: (1.0 if option == pick else 0.0)
                             for option in options}
        decision["pick"] = pick
        decision["confidence"] = 1.0
        return decision
    if isinstance(answer.vector, dict):
        stated = [value for value in answer.vector.values()
                  if isinstance(value, (int, float)) and not isinstance(value, bool)]
        if stated:
            decision["sum_before_normalize"] = round(float(sum(stated)), 6)
    vector, why = jevals.normalize_vector(answer.vector, options)
    if vector is None:
        decision["malformed"] = True
        decision["malformed_reason"] = answer.malformed_reason or why
        decision["vector"] = jevals.uniform(options)
        return decision
    pick = jevals.pick_from_vector(vector, options, presented=presented, qtype=qtype,
                                   stated=answer.stated)
    decision["vector"] = vector
    decision["pick"] = pick
    decision["confidence"] = None if pick is None else vector.get(pick)
    if qtype == jevals.NOUL and pick is None:
        # Exactly 0.5 either way. A real answer with no pick, which counts as wrong and
        # has no confidence to calibrate: not malformed, and the recipe says so.
        decision["confidence"] = None
    return decision


def row_for(decision: dict, top_k: int = ROW_TOP_K) -> dict:
    """One record row. Carries the numbers, not the state and not the full vector.

    The state is rebuilt from the suite manifest under the same ``id`` and is verified
    against its published hash, so a record that repeated it would be mostly prompt. The
    full vector is dropped for the same reason on a wide set: ``top`` keeps the five
    options the system was surest about, and ``p_label`` keeps the labeled option's own
    probability, which together are what a reader of a wrong answer needs.
    """
    vector = decision.get("vector") or {}
    label = decision.get("label")
    top = sorted(vector.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k]
    return {
        "id": decision["id"], "set": decision["set"], "kind": decision["type"],
        "repeat": decision["repeat"], "order": decision["order"],
        "label": label, "answer": decision.get("pick"),
        "correct": (None if decision.get("error") else
                    bool(decision.get("pick") == label)),
        "p_answer": (None if decision.get("confidence") is None
                     else round(float(decision["confidence"]), 6)),
        "p_label": (None if not vector else round(float(vector.get(label, 0.0)), 6)),
        "top": {name: round(float(value), 6) for name, value in top} or None,
        "malformed": bool(decision.get("malformed")),
        "malformed_reason": decision.get("malformed_reason"),
        "one_hot": bool(decision.get("one_hot")),
        "wall_ms": decision.get("wall_ms"),
        "tokens_in": decision.get("tokens_in"),
        "tokens_out": decision.get("tokens_out"),
        "error": decision.get("error"),
    }


# ------------------------------------------------------------------ the transports

def systemone_url(endpoint: str) -> str:
    """``<endpoint>/systemone``, adding the ``/v1`` if it was left off."""
    base = (endpoint or "").rstrip("/")
    if base.endswith("/systemone"):
        return base
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/systemone"


def decide_endpoint(endpoint: str) -> str:
    """``<endpoint>/decide``, adding the ``/v1`` if it was left off."""
    base = (endpoint or "").rstrip("/")
    if base.endswith("/decide"):
        return base
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/decide"


def is_typesafe(url: str) -> bool:
    """Whether a URL points at the hosted Jev, which decides the credential."""
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host == TYPESAFE_HOST or host.endswith("." + TYPESAFE_HOST)


def wire_leaks(payload) -> list:
    """Every answer-key-shaped key in a request body, by path. Empty means clean.

    The other half of the rule :func:`ainode.bench.decide.sets.answer_key_leaks` holds
    over a question file: a label lives in the file's separate ``labels`` map, a
    transport is handed a question that has none, and this checks the body it actually
    assembled. Two ends, because "the model must not be shown the answer" is the one
    mistake in a benchmark that makes every number it produces worthless while looking
    like a very good result.

    The ``state`` value is not walked. It is the caller's own whitelisted data, a private
    set is allowed a field named whatever its dataset names it, and this guard is about
    what the bench appends beside the question rather than what the question is about.
    """
    return sets.answer_key_leaks(payload, skip=("state",))


class Transport:
    """Build a request, parse a response, and only ``ask`` touches the network."""

    name = ""
    #: True when the endpoint is one of OUR nodes, so a 401 or a 429 from it is
    #: AINode's and stops the run before anything is scored. False for a vendor's,
    #: whose refusals belong in their own row and not in an AINode sentence.
    local = True
    #: Where the probabilities came from, in JevBench's vocabulary plus the one value it
    #: has no name for. ``native`` is the model's own distribution, which is what a
    #: Jev-format server returns; ``verbalized`` is a model writing probabilities out
    #: under a schema; ``logprob`` is reading them off the decode, which JevBench states
    #: it does not do for anyone. A row is only comparable to a row with the same value
    #: here, so the record carries it rather than a footnote.
    probability_source = "native"

    def __init__(self, endpoint: str, model: str = "", api_key: str = "",
                 key_source: str = "", timeout: float = DEFAULT_TIMEOUT,
                 input_usd_per_mtok: float = 0.0, output_usd_per_mtok: float = 0.0):
        if not endpoint:
            raise BackendError(f"the {self.name} transport needs --endpoint")
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.key_source = key_source
        self.timeout = timeout
        self.input_usd_per_mtok = float(input_usd_per_mtok)
        self.output_usd_per_mtok = float(output_usd_per_mtok)
        self.requests = 0
        self.reported_model = ""
        self.reported_node = ""

    # -- pure halves ------------------------------------------------------------
    def question(self, doc: dict, presented) -> dict:
        raise NotImplementedError

    def request(self, doc: dict, question: dict, presented) -> Request:
        raise NotImplementedError

    def parse(self, doc: dict, question: dict, presented, data: dict) -> Answer:
        raise NotImplementedError

    # -- the one networked call -------------------------------------------------
    def ask(self, doc: dict, question: dict, repeat: int) -> dict:
        """One (question, repeat) end to end, as a decision dict.

        A refusal from one of our own nodes RAISES rather than becoming 4,500 wrong
        answers with a Decision Score computed over them, which is the rule the rest of
        the bench runs under. A vendor's refusal is left alone.
        """
        spec = spec_for(doc, question)
        presented = presented_options(spec["options"], question["id"], doc.get("seed"),
                                      order_index(repeat), spec["type"])
        request = self.request(doc, question, presented)
        leaks = wire_leaks(request.payload)
        if leaks:
            raise BackendError(
                f"the {self.name} request for {question['id']} carries "
                f"{', '.join(leaks)}: that is the answer on the wire. Refusing to ask, "
                "because a score measured against a prompt holding its own label is not "
                "a score")
        self.requests += 1
        data, wall, error = post_json(request, self.timeout)
        answer = Answer()
        answer.wall_ms = round(wall * 1000)
        if error:
            if self.local:
                auth.check_error(error)
            answer.error = error
        else:
            try:
                answer = self.parse(doc, question, presented, data)
            except Exception as exc:                              # noqa: BLE001
                answer = Answer(error=None, one_hot=False)
                answer.vector = None
                answer.malformed_reason = (f"unreadable response: "
                                           f"{type(exc).__name__}: "
                                           f"{str(exc)[:ERROR_CHARS]}")
            answer.wall_ms = round(wall * 1000)
            if answer.model:
                self.reported_model = answer.model
            if answer.node:
                self.reported_node = answer.node
        return decision_for(doc, question, repeat, presented, answer)

    def protocol(self) -> dict:
        return {"transport": self.name, "endpoint": self.endpoint,
                "timeout_s": self.timeout,
                "probability_source": self.probability_source,
                "cost_basis": self.cost_basis(),
                "input_usd_per_mtok": self.input_usd_per_mtok,
                "output_usd_per_mtok": self.output_usd_per_mtok}

    def cost_basis(self) -> str:
        """Why the cost column says what it says, in JevBench's vocabulary.

        A priced endpoint is ``derived_usage_times_tariff``: measured tokens times a rate
        somebody posted. An unpriced one reads $0, and this says WHY it is 0 rather than
        letting a reader take it for free: nobody bills per token for a GPU we own, the
        electricity is real, and an invented figure would be an estimate in a file of
        measurements.
        """
        if self.input_usd_per_mtok or self.output_usd_per_mtok:
            return "derived_usage_times_tariff"
        return "no_billable_account_no_price_given"


class DecideTransport(Transport):
    """AINode's ``POST /v1/decide``.

    Every primitive goes out with an EXPLICIT ``options`` list rather than the
    endpoint's ``boolean`` or ``score`` sugar, so the distribution comes back keyed by
    this set's own option strings for all three and one parser fits them. The sugar
    would be the same request (``type: boolean`` is ``["yes", "no"]`` and ``type:
    score`` is ``[str(v) for v in range(min, max + 1)]``) with a different chance of a
    key mismatch: PubMedQA's options are ``["no", "yes"]``, and the order matters to the
    presented prompt.

    The state goes out as the CANONICAL JSON STRING rather than as an object, because
    the endpoint serializes an object it is handed with sorted keys and the bytes the
    recipe hashes are in state-field order. A string is passed through verbatim, so the
    model sees exactly the bytes the published ``state_sha256`` covers.

    The criteria are appended to the question text because ``/v1/decide``'s choice
    question takes bare option names with no room for a per-option rubric. For Banking77
    that appends nothing, since its criteria are all null.
    """

    name = "decide"
    #: ``/v1/decide`` softmaxes the first generated token's logprobs, which is neither of
    #: JevBench's two categories: it states it uses token-level logprobs for nobody, and
    #: Jevals 0.1.0 states that logprob-based rows are not in that version.
    probability_source = "logprob"

    def question(self, spec: dict, presented) -> dict:
        text = spec["instructions"]
        lines = criteria_lines(spec, presented)
        if lines:
            text = f"{text}\n{lines}"
        return {"question": text, "options": list(presented)}

    def request(self, doc: dict, question: dict, presented) -> Request:
        payload = {"state": sets.state_json(question["state"]),
                   "questions": {QUESTION_KEY: self.question(
                       spec_for(doc, question), presented)}}
        if self.model:
            payload["model"] = self.model
        return Request(self.endpoint, payload, auth.bearer(self.api_key))

    def parse(self, doc: dict, question: dict, presented, data: dict) -> Answer:
        block = (data.get("decisions") or {})[QUESTION_KEY]
        usage = data.get("usage") or {}
        answer = Answer(
            vector=block.get("distribution"),
            stated=block.get("answer"),
            confidence=block.get("confidence"),
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            model=data.get("model") or self.model,
            node=data.get("node") or "",
            excerpt=str(block.get("answer"))[:ERROR_CHARS])
        if answer.vector is None:
            # The engine answered under the grammar but gave no logprobs, so there is a
            # pick and no spread. Scored as one-hot, which is the recipe's rule, and out
            # of the calibration numbers rather than credited with a confidence of 1.
            answer.one_hot = True
        return answer


class SystemOneTransport(Transport):
    """``POST /v1/systemone`` in the Jev wire format: TypeSafe's, or any server's.

    One typed question per request, the state as a JSON object (which is how Jevals
    sends it), and the criteria in the shape the question file states them: a map for
    ``choice``, a ``true``/``false`` map for ``noul``, a list in level order for
    ``score``. Choice options are presented in this repeat's order, which is what the
    order flip rate measures.

    The ``score`` response shape is the one authored piece here: jevals.com documents
    the primitive but not the field a Jev-format server answers it in, so the parser
    accepts ``score``, ``level`` or ``choice`` for the pick and ``probabilities`` or
    ``distribution`` for the vector, and a response matching none of them is one
    malformed row rather than a crash. Recorded in JEVALS.md.
    """

    name = "systemone"
    #: The model's own distribution, off the wire, which is what JevBench's ``typesafe``
    #: adapter reads from this same endpoint shape.
    probability_source = "native"

    def question(self, spec: dict, presented) -> dict:
        qtype = spec["type"]
        options = list(spec["options"])
        out = {"type": qtype, "instructions": spec["instructions"]}
        described = option_criteria(spec)
        if qtype == jevals.SCORE:
            out["criteria"] = [described.get(option) for option in options]
        elif qtype == jevals.NOUL:
            out["criteria"] = {"false": described.get(options[0]),
                               "true": described.get(options[1])}
        else:
            out["criteria"] = {option: described.get(option) for option in presented}
        if qtype == jevals.NOUL and not any(out["criteria"].values()):
            # Two of typed-decisions' twenty questions carry no criteria at all, and a
            # yes/no question needs none: its answer space is implied by the type. Only
            # this case may drop the field. For `choice` the criteria MAP KEYS ARE the
            # answer space, and for `score` its length is K, so dropping either would
            # ask a different question (Banking77's 77 criteria are all null and still
            # have to travel, because they are the option list).
            out.pop("criteria")
        return out

    def request(self, doc: dict, question: dict, presented) -> Request:
        payload = {"state": question["state"],
                   "questions": {QUESTION_KEY: self.question(
                       spec_for(doc, question), presented)}}
        if self.model:
            payload["model"] = self.model
        return Request(self.endpoint, payload, auth.bearer(self.api_key))

    def parse(self, doc: dict, question: dict, presented, data: dict) -> Answer:
        block = (data.get("answers") or {})[QUESTION_KEY]
        usage = data.get("usage") or {}
        spec = spec_for(doc, question)
        options = list(spec["options"])
        vector = None
        stated = None
        if spec["type"] == jevals.NOUL and "noul" in block:
            probability = float(block["noul"])
            # P(yes) on the wire. The second option is the positive one, which is how
            # the manifests are built (PubMedQA: ["no", "yes"]).
            vector = {options[0]: 1.0 - probability, options[1]: probability}
        else:
            raw = block.get("probabilities")
            if raw is None:
                raw = block.get("distribution")
            if isinstance(raw, dict):
                vector = {str(key): value for key, value in raw.items()}
            for key in ("choice", "score", "level"):
                if block.get(key) is not None:
                    stated = str(block[key])
                    break
        answer = Answer(
            vector=vector, stated=stated, confidence=block.get("confidence"),
            tokens_in=usage.get("input_tokens", usage.get("prompt_tokens")),
            tokens_out=usage.get("output_tokens", usage.get("completion_tokens")),
            model=data.get("model") or self.model,
            node=data.get("node") or "",
            excerpt=str(stated)[:ERROR_CHARS])
        if vector is None and stated is not None:
            answer.one_hot = True
        elif vector is None:
            answer.malformed_reason = "no probability map and no stated answer"
        return answer


def build_transport(name: str, endpoint: str = "", model: str = "", api_key: str = "",
                    timeout: float = DEFAULT_TIMEOUT, input_usd_per_mtok: float = 0.0,
                    output_usd_per_mtok: float = 0.0) -> Transport:
    """The named transport with its key resolved from the endpoint's HOST.

    ``api.typesafe.ai`` gets TypeSafe's credential (``--api-key``, then
    ``$TYPESAFE_API_KEY``, then ``~/.jev_api_key``); anything else gets the node's
    (``--api-key``, then ``$AINODE_API_KEY``, then the placeholder an open node
    accepts). Two separate resolvers, picked by the host, so no flag ordering can send
    one party's credential to the other.
    """
    if name not in TRANSPORTS:
        raise BackendError(f"unknown transport {name!r}; pick from "
                           f"{', '.join(TRANSPORTS)}")
    if name == "systemone":
        url = systemone_url(endpoint or JEV_URL)
        if is_typesafe(url):
            key, source = jev_api_key(api_key)
            if not key:
                from ainode.bench.decide.backends import missing_key_message
                raise BackendError(missing_key_message())
            transport = SystemOneTransport(
                url, model=model, api_key=key, key_source=source, timeout=timeout,
                input_usd_per_mtok=input_usd_per_mtok,
                output_usd_per_mtok=output_usd_per_mtok)
            transport.local = False
            return transport
        key, source = node_api_key(api_key)
        return SystemOneTransport(url, model=model, api_key=key, key_source=source,
                                  timeout=timeout,
                                  input_usd_per_mtok=input_usd_per_mtok,
                                  output_usd_per_mtok=output_usd_per_mtok)
    if not endpoint:
        raise BackendError("the decide transport needs --endpoint, e.g. "
                           "http://host:3000/v1")
    key, source = node_api_key(api_key)
    return DecideTransport(decide_endpoint(endpoint), model=model, api_key=key,
                           key_source=source, timeout=timeout,
                           input_usd_per_mtok=input_usd_per_mtok,
                           output_usd_per_mtok=output_usd_per_mtok)


# ------------------------------------------------------------------ the loop

def plan(docs, repeats: int, limit: int = 0) -> list:
    """``[(doc, question, repeat), ...]`` in set, question, repeat order.

    Deterministic order so two runs' rows line up one to one and a record can be read
    down the file. ``limit`` takes the first N questions of each set, which is how a
    smoke run proves a transport without spending a full suite on it.
    """
    out = []
    for doc in docs:
        questions = doc["questions"]
        if limit:
            questions = questions[:max(1, int(limit))]
        for question in questions:
            for repeat in range(max(1, int(repeats))):
                out.append((doc, question, repeat))
    return out


def run(transport: Transport, docs, repeats: int = jevals.REPEATS,
        concurrency: int = DEFAULT_CONCURRENCY, limit: int = 0, progress=None):
    """Every (question, repeat) through the transport. ``(decisions, seconds)``.

    Ordered by the plan and not by completion. The wall clock covers the whole loop,
    which is what ``questions_per_second`` is over, and it moves with ``concurrency``
    and with whatever else the node is serving, which is why the record carries both.
    """
    work = plan(docs, repeats, limit)
    decisions = [None] * len(work)
    workers = max(1, int(concurrency))
    started = time.time()
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(transport.ask, doc, question, repeat): index
                   for index, (doc, question, repeat) in enumerate(work)}
        done = 0
        for future in futures.as_completed(pending):
            index = pending[future]
            decisions[index] = future.result()
            done += 1
            if progress:
                progress(done, len(work), decisions[index])
    return decisions, time.time() - started


# ------------------------------------------------------------------ the record

def reported_model(transport: Transport) -> str:
    """The model id to put in the record: what the service said it was.

    For a hosted Jev that is the version string the API reports rather than the alias
    that was asked for, so the record names the thing that answered.
    """
    return (getattr(transport, "reported_model", "")
            or getattr(transport, "model", "") or transport.name)


def legacy_overall(decisions, input_usd_per_mtok: float = 0.0,
                   output_usd_per_mtok: float = 0.0) -> dict:
    """The fields of the legacy ``overall`` block that mean the SAME thing here.

    Deliberately partial. ``brier``, ``ece``, ``bins`` and ``thresholds`` are absent
    rather than filled in from the recipe's numbers: the legacy block's Brier is one
    term on the labeled option over five bins and the recipe's is the multiclass sum
    over ten, so putting one under the other's name would make two incomparable numbers
    look like one. The README table reads "not measured" for those cells on a run like
    this, which is the truth.
    """
    rows = jevals.scored(decisions)
    counts = jevals.tokens(rows)
    block = {"n": len(decisions), "answered": len(rows),
             "errors": len(jevals.failed(decisions)),
             "accuracy": None if not rows else round(jevals.accuracy(decisions), 4),
             "tokens": counts,
             "cost_usd": round(jevals.cost_usd(counts, input_usd_per_mtok,
                                               output_usd_per_mtok), 6)}
    block.update(jevals.latency(decisions))
    return block


def build_jevals_block(transport: Transport, docs, decisions, seconds: float,
                       repeats: int, concurrency: int, limit: int = 0) -> dict:
    """The ``decide.jevals`` sub-block: one metrics block per set, plus the mean."""
    kw = {"input_usd_per_mtok": transport.input_usd_per_mtok,
          "output_usd_per_mtok": transport.output_usd_per_mtok, "seconds": seconds}
    names = [doc.get("set") or doc["id"] for doc in docs]
    per_set = jevals.summarize_sets(decisions, names, **kw)
    block = {
        "recipe": dict(RECIPE),
        "formulas": jevals.RECIPE_JEVALS,
        "probability_source": transport.probability_source,
        "cost_basis": transport.cost_basis(),
        "repeats": int(repeats),
        "concurrency": int(concurrency),
        "batch_size": 1,
        "bins": jevals.BINS,
        "handoff_target": jevals.HANDOFF_ACCURACY,
        "handoff_min_decisions": jevals.HANDOFF_MIN_DECISIONS,
        "grid": jevals.GRID,
        "published_gates": {name: jevals.PUBLISHED_GATES.get(doc.get("type"))
                            for name, doc in zip(names, docs)},
        "recipe_of_record": {name: doc.get("recipe_of_record")
                             for name, doc in zip(names, docs)},
        "contamination": {name: (doc.get("contamination") or [])
                          for name, doc in zip(names, docs)
                          if doc.get("contamination")},
        "sets": per_set,
        "overall": jevals.mean_decision_score(per_set),
        "seconds": round(float(seconds), 1),
    }
    if limit:
        block["limit"] = int(limit)
        block["partial"] = True
    return block


def set_summary(doc: dict, limit: int = 0) -> dict:
    """What a record says about one set: which items, from where, under which licence."""
    count = len(doc["questions"])
    if limit:
        count = min(count, max(1, int(limit)))
    widths = [len(sets.question_spec(doc, q)["options"]) for q in doc["questions"]]
    return {"id": doc["id"], "type": doc.get("type"), "questions": count,
            "options": max(widths) if widths else 0, "seed": doc.get("seed"),
            "suite_version": doc.get("suite_version"),
            "recipe_of_record": doc.get("recipe_of_record"),
            "gold_distributions": any(q.get("gold") for q in doc["questions"]),
            "contamination": doc.get("contamination") or [],
            "source": doc.get("source") or {}}


def build_decide_block(transport: Transport, docs, decisions, seconds: float,
                       repeats: int, concurrency: int, limit: int = 0,
                       model_reported: str = "") -> dict:
    """The record's ``decide`` block for a Jevals-recipe run. See bench/SCHEMA.md."""
    return {
        "backend": transport.name,
        "endpoint": transport.endpoint,
        "model_reported": model_reported or None,
        "mode": MODE,
        "recipe": dict(RECIPE),
        "item_set": {"id": f"jevals-{RECIPE['suite']}",
                     "version": RECIPE["suite"],
                     "file": "bench/decide/sets/",
                     "count": sum(set_summary(d, limit)["questions"] for d in docs),
                     "sets": {(d.get("set") or d["id"]):
                              set_summary(d, limit)["questions"] for d in docs}},
        "protocol": {**transport.protocol(), "concurrency": int(concurrency),
                     "repeats": int(repeats),
                     "batch_size": 1,
                     "confidence": "the probability the endpoint put on its own pick; "
                                   "for a yes/no question the larger of P(yes) and "
                                   "P(no)",
                     "loss": "multiclass Brier for choice and noul, ranked probability "
                             "score over cumulative levels for score"},
        "sources": [set_summary(doc, limit) for doc in docs],
        "overall": legacy_overall(decisions, transport.input_usd_per_mtok,
                                  transport.output_usd_per_mtok),
        "sets": {},
        "jevals": build_jevals_block(transport, docs, decisions, seconds, repeats,
                                     concurrency, limit),
        "rows": [row_for(decision) for decision in decisions],
    }


def build_notes(transport: Transport, docs, decisions, seconds: float, repeats: int,
                limit: int = 0, source: str = SOURCE) -> list:
    """The notes a reader needs to know what these numbers are and are not."""
    names = ", ".join((doc.get("set") or doc["id"]) for doc in docs)
    errors = [d["id"] for d in jevals.failed(decisions)]
    malformed = [d["id"] for d in jevals.scored(decisions) if d.get("malformed")]
    one_hot = sum(1 for d in jevals.scored(decisions) if d.get("one_hot"))
    notes = [
        f"Measured by {source} in {round(seconds)}s: {len(decisions)} decisions, "
        f"{repeats} repeats per question, over {names}. Nothing was loaded, unloaded "
        "or restarted.",
        "Scored by the Jevals recipe, suite 0.1.0, read from "
        "https://jevals.com/methodology on 2026-09-21 and recorded in "
        "bench/decide/JEVALS.md with every deviation named. Attribution: Jevals "
        "(jevals.com), suite 0.1.0; suite files CC-BY-4.0.",
        "Decision Score is 100 * (1 - L_system / L_prior): 100 is perfect, 0 is "
        "answering with the label base rates, and below 0 is worse than that. Every "
        "set's block carries prior_accuracy, the base-rate answer's own accuracy, so "
        "the guessing floor travels with the figure.",
        "Every state was rebuilt from its upstream dataset row and checked against the "
        "manifest's state_sha256 before the run, so these are the same bytes the boards "
        "scored. The item text is not committed; only the manifests are.",
        "One question per request (batch size 1), which is the recipe's rule. A set card "
        "that measured its own reference row several questions per request is reporting a "
        "different latency and a different cost, and those two columns are not "
        "comparable to this record's.",
        "Every metrics block names the recipe its formulas follow (`recipe`), and every "
        "set names the recipe its own published third-party numbers follow "
        "(`recipe_of_record`). Where those two differ, only the figures under this "
        "record's own recipe are comparable across rows.",
    ]
    contaminated = [(doc.get("set") or doc["id"], doc.get("contamination") or [])
                    for doc in docs]
    for name, entries in contaminated:
        if not entries:
            continue
        who = "; ".join(f"{e['system']} ({e['source']})" for e in entries)
        notes.append(
            f"CONTAMINATION, {name}: this set is in the published training data of "
            f"{who}. A row for one of those systems on this set measures memorisation "
            "and not decision quality, and it must not be read as a like-for-like "
            "comparison with a system that never saw it. It says nothing about an "
            "AINode-served model that did not train on it.")
    if any(doc.get("type") == "mixed" for doc in docs):
        notes.append(
            "A set holding more than one primitive is broken down by primitive, because "
            "the two losses are different arithmetic and the label prior is per answer "
            "space. Its Decision Score is the plain mean of the per-primitive scores, "
            "which is the recipe's rule for more than one task in a tab.")
    if any(any(q.get('gold') for q in doc['questions']) for doc in docs):
        notes.append(
            "One of these sets ships a gold DISTRIBUTION rather than only a label, so "
            "its block carries a vs_gold section: soft accuracy, total variation, KL "
            "and a Brier against that distribution. Those four are AINode's own "
            "definitions, stated in the block, and are not a set card's columns of the "
            "same names.")
    notes.append(
        f"Probability source: {transport.probability_source}. JevBench labels a model's "
        "own distribution `native` and a model writing probabilities out under a schema "
        "`verbalized`, and states it uses token-level logprobs for nobody; Jevals 0.1.0 "
        "states its LLM rows are verbalized and that logprob-based rows are not in that "
        "version. A row is only comparable to a row with the same probability source.")
    if transport.name == "decide":
        notes.append(
            "Measured through AINode's /v1/decide, which constrains the engine to one "
            "option label and reads the distribution from the first token's logprobs. "
            "Jevals 0.1.0 states that logprob-based rows are not in that version and "
            "that its LLM rows are verbalized, so this is the same items, labels and "
            "formulas through a different and generally stronger instrument than the "
            "board's LLM rows.")
        if any(set_summary(doc, limit)["options"] > 26 for doc in docs):
            notes.append(
                "On a set with more than 26 options /v1/decide letters them A..Z, "
                "AA.., and a two-letter label that the tokenizer does not give its own "
                "token shares its probability with the one-letter label of the same "
                "first letter (see ainode/api/decide.py). Read the choice Decision "
                "Score as a floor rather than a point.")
    if transport.input_usd_per_mtok or transport.output_usd_per_mtok:
        notes.append(
            f"Cost is the posted rate given for this endpoint applied to the tokens it "
            f"reported: ${transport.input_usd_per_mtok:g} per million input tokens and "
            f"${transport.output_usd_per_mtok:g} per million output tokens.")
    else:
        notes.append(
            "Cost is $0: no price was given for this endpoint, so there is nothing to "
            "multiply the reported tokens by. The electricity is real and is not a "
            "number this record claims to have measured.")
    notes.append("No malformed-output retry, no discarded warm-up call, no bootstrap "
                 "interval and no rank: latency is measured at whatever --concurrency "
                 "this run used rather than the board's 4, and a Decision Score here is "
                 "a point value. Every deviation is listed in bench/decide/JEVALS.md.")
    if limit:
        notes.append(
            f"PARTIAL RUN: --limit {limit} took the first {limit} question(s) of each "
            "set, so this is a transport proof and not a suite result. A board listing "
            "needs a complete run of every task in the tab at 5 repeats.")
    if errors:
        notes.append(
            f"{len(errors)} decision(s) failed on a transport or protocol error rather "
            f"than on the answer: {', '.join(errors[:10])}"
            f"{', ...' if len(errors) > 10 else ''}. They are out of every figure, "
            "which is the recipe's rule, and a set missing any (item, repeat) is not a "
            "complete run.")
    if malformed:
        notes.append(
            f"{len(malformed)} answer(s) could not be read as a probability vector over "
            "the set's options. They are scored as the uniform distribution and a wrong "
            "pick, which puts them in the Decision Score and the accuracy and keeps "
            "them out of the ECE, the flip rates and the gate.")
    if one_hot:
        notes.append(
            f"{one_hot} answer(s) carried no probabilities at all. They are scored "
            "one-hot and are out of the calibration numbers, which is why a set's "
            "calibrated_over can be smaller than its decisions.")
    return notes


# ------------------------------------------------------------------ printing

#: Wide enough for the deepest label the breakdown produces, which is four spaces of
#: indent plus a mixed set's `<workflow>/<question>` answer-space name.
COLUMNS = (("set", 38), ("type", 7), ("n", 6), ("acc", 7), ("floor", 7), ("DS", 8),
           ("ECE pt", 8), ("hand-off", 10), ("gate", 12), ("flips", 7), ("swing", 7),
           ("p50 ms", 8), ("p95 ms", 8), ("q/s", 7), ("bad", 5), ("cost", 9))


def fmt(value, places=3):
    return "-" if value is None else f"{value:.{places}f}"


def fmt_handoff(block):
    hand = block.get("handoff_95")
    if not hand:
        return "-"
    return f"{hand['share']:.2f}@{hand['threshold']:.2f}"


def fmt_gate(block):
    gate = block.get("gate") or {}
    if gate.get("threshold") is None:
        return "none"
    coverage = gate.get("coverage")
    accuracy = gate.get("accuracy")
    return (f"{gate['threshold']:.2f}:"
            f"{'-' if coverage is None else f'{coverage:.2f}'}/"
            f"{'-' if accuracy is None else f'{accuracy:.2f}'}")


def table_rows(block) -> list:
    """``[cells, ...]``: one line per set, plus one per primitive of a mixed set.

    A mixed set's own line carries its accuracy and the mean of its primitives' Decision
    Scores; the indented lines under it are the per-primitive blocks, which is where its
    losses, its gate and its hand-off actually live.
    """
    out = []
    for name, metrics in block["sets"].items():
        out.extend(_set_lines(name, metrics))
        for qtype, sub in (metrics.get("types") or {}).items():
            out.extend(_set_lines(f"  {qtype}", sub))
        for space, sub in (metrics.get("spaces") or {}).items():
            out.extend(_set_lines(f"    {space}", sub))
    return out


def _set_lines(name, metrics) -> list:
    return [[
            name,
            str(metrics.get("type") or "-"),
            str(metrics.get("decisions") or 0),
            fmt(metrics.get("accuracy")),
            fmt(metrics.get("prior_accuracy")),
            fmt(metrics.get("decision_score"), 1),
            fmt(metrics.get("ece_points"), 1),
            fmt_handoff(metrics),
            fmt_gate(metrics),
            fmt(metrics.get("pick_flip_rate"), 2),
            fmt((metrics.get("confidence_swing") or {}).get("max"), 2),
            "-" if metrics.get("p50_ms") is None else str(metrics["p50_ms"]),
            "-" if metrics.get("p95_ms") is None else str(metrics["p95_ms"]),
            fmt(metrics.get("questions_per_second"), 2),
            str((metrics.get("malformed") or 0) + (metrics.get("failed") or 0)),
            "$0" if not metrics.get("cost_usd") else f"${metrics['cost_usd']:.4f}",
        ]]


def print_table(block, title: str, out=print) -> None:
    """The per-set table, the reliability table under it, and what the columns are."""
    out(f"\n  {title}")
    out("  " + "".join(name.ljust(width) for name, width in COLUMNS))
    for cells in table_rows(block):
        out("  " + "".join(cell.ljust(width)
                           for cell, (_h, width) in zip(cells, COLUMNS)))
    mean = (block.get("overall") or {}).get("mean_decision_score")
    out(f"\n  mean Decision Score over {len(block['sets'])} set(s): "
        f"{fmt(mean, 1)}   (100 perfect, 0 the label base rates, negative worse)")
    out("  floor    = the base-rate answer's accuracy on the same items")
    out("  hand-off = share it can take alone at 95% right @ the threshold that does it")
    out(f"  gate     = published gate : this run's coverage / accuracy at it "
        f"({jevals.PUBLISHED_GATE_SOURCE})")
    out("  flips    = share of questions whose pick changed at least once over the "
        "repeats")
    out("  swing    = the largest confidence spread one question showed over its repeats")
    out("  bad      = malformed answers plus decisions that never came back")
    out("  the board's own repeat (0 vs 1) and order (choice, 0/2/3/4) flip rates are "
        "in the record")
    for name, metrics in block["sets"].items():
        out(f"\n  reliability, {jevals.BINS} bins on the pick's own probability, {name}")
        out("  bin          count  accuracy  mean conf")
        for bucket in metrics["bins"]:
            if not bucket["count"]:
                continue
            out(f"  {bucket['lo']:.1f}-{bucket['hi']:.1f}    {bucket['count']:5d}  "
                f"{fmt(bucket['accuracy']):>8}  {fmt(bucket['confidence']):>9}")
        if metrics.get("calibrated_over") == 0:
            out("  nothing could be calibrated: no answer carried probabilities")
        gold = metrics.get("vs_gold")
        if gold:
            out(f"  against the gold distribution ({gold['over']} decisions): "
                f"soft acc {fmt(gold.get('soft_accuracy'))}  "
                f"TV {fmt(gold.get('total_variation'))}  "
                f"KL {fmt(gold.get('kl'))}  "
                f"Brier vs gold {fmt(gold.get('brier_vs_gold'))}")
    for name, entries in (block.get("contamination") or {}).items():
        who = ", ".join(entry["system"] for entry in entries)
        out(f"\n  CONTAMINATION, {name}: in the published training data of {who}. A row "
            "for one of those systems here measures memorisation.")


def print_wrong(decisions, out=print, limit: int = 20) -> None:
    """The confidently wrong list, which is this bench's useful output."""
    wrong = [d for d in jevals.scored(decisions) if not jevals.is_correct(d)]
    if not wrong:
        out("\n  no wrong picks")
        return
    out(f"\n  wrong picks ({len(wrong)} of {len(jevals.scored(decisions))}), "
        "highest confidence first")
    wrong.sort(key=lambda d: d.get("confidence") or 0.0, reverse=True)
    for decision in wrong[:limit]:
        vector = decision.get("vector") or {}
        tag = "malformed" if decision.get("malformed") else ""
        out(f"    {decision['id']:<16} r{decision['repeat']} "
            f"p={fmt(decision.get('confidence'), 2):<5} "
            f"label {str(decision['label'])[:20]:<22} "
            f"picked {str(decision.get('pick'))[:20]:<22} "
            f"p(label)={fmt(vector.get(decision['label']), 2)} {tag}")
    if len(wrong) > limit:
        out(f"    ... and {len(wrong) - limit} more, all of them in the record")


__all__ = ["COLUMNS", "DEFAULT_CONCURRENCY", "JEV_URL", "MODE", "QUESTION_KEY",
           "RECIPE", "ROW_TOP_K", "SOURCE", "TRANSPORTS", "TYPESAFE_HOST", "Answer",
           "DecideTransport", "SystemOneTransport", "Transport", "build_decide_block",
           "build_jevals_block", "build_notes", "build_transport", "criteria_lines",
           "decide_endpoint", "decision_for", "fmt", "fmt_gate", "fmt_handoff",
           "is_typesafe", "legacy_overall", "option_criteria", "order_index", "plan",
           "presented_options", "print_table", "print_wrong", "reported_model",
           "row_for", "run", "set_summary", "systemone_url", "table_rows",
           "wire_leaks", "DEFAULT_API_KEY"]
