"""Three backends, one question: what did it answer, and how sure was it.

    ``ainode``   AINode's own ``POST /v1/decide``, one typed question per call
    ``chat``     any OpenAI-compatible engine, lettered options plus logprobs
    ``jev``      TypeSafe AI's hosted System One model, ``POST /v1/systemone``

``chat`` is the fallback that makes the bench runnable against anything that serves
chat completions, including before ``/v1/decide`` ships: the options are lettered,
thinking is switched off, ``max_tokens`` is 4, and the distribution comes from the
top logprobs of the one letter token. It is a weaker instrument than a real decision
endpoint (the probabilities are over letters, not over meanings) and the record says
which backend produced a row so the two are never averaged together.

Every backend is split the same way on purpose: ``request(item)`` builds the payload
and ``parse(item, data)`` reads one back, both pure functions of their arguments, and
``decide(item)`` is the only part that touches the network. That is what lets the
tests drive every request shape and every response shape from canned payloads with
no server anywhere.

Stdlib only, like the rest of ``ainode/bench``: urllib for the transport.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from ainode.bench import auth
from ainode.bench.decide.items import CHOICE, FALSE, NOUL, TRUE, Item

#: Seconds one request gets. A decision is a short generation; a local model on a
#: busy node is still the slow case, which is why this is not five.
DEFAULT_TIMEOUT = 120
#: Bearer token AINode's proxy accepts by default, same as the other sections.
DEFAULT_API_KEY = "ainode"
#: The key every request names its one question under. The key is ours to choose and
#: comes back under the same name in every one of the three response shapes.
QUESTION_KEY = "decision"

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_KEY_ENV = "TYPESAFE_API_KEY"
JEV_KEY_FILE = "~/.jev_api_key"
#: TypeSafe's posted price: input tokens only, output free.
JEV_INPUT_USD_PER_MTOK = 0.042
JEV_OUTPUT_USD_PER_MTOK = 0.0

#: Letters a ``chat`` prompt can label options with. Four is the widest set here.
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#: How much of an error body is worth keeping in a row.
ERROR_CHARS = 300
#: Spellings of a yes and a no that a boolean answer may come back as.
TRUTHY = {"true", "yes", "y", "t", "1"}
FALSY = {"false", "no", "n", "f", "0"}

BACKENDS = ("ainode", "chat", "jev")


class BackendError(RuntimeError):
    """A backend that cannot be built: no API key, no endpoint, an item it cannot ask."""


# ---------------------------------------------------------------- transport

@dataclass
class Request:
    """One outgoing call. ``headers`` holds the credential and is never printed."""

    url: str
    payload: dict
    headers: dict = field(default_factory=dict)

    def curl_safe(self) -> str:
        """The request as a line to show a person: no header, so no key."""
        return f"POST {self.url}  {json.dumps(self.payload, sort_keys=True)}"


@dataclass
class Decision:
    """What one item came back as, before it is scored."""

    answer: object = None
    confidence: float | None = None
    distribution: dict | None = None
    wall_ms: int = 0
    server_latency_ms: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    model: str = ""
    node: str = ""
    error: str | None = None
    excerpt: str = ""


def post_json(request: Request, timeout: float):
    """``(data, wall_seconds, error)``. Never raises, so one bad item is one row.

    An HTTP error body is kept (truncated) because a 400 from a decision endpoint
    usually says which question it could not parse, and that is the finding.
    """
    req = urllib.request.Request(
        request.url, data=json.dumps(request.payload).encode(),
        headers={"Content-Type": "application/json", **request.headers})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response), time.monotonic() - started, None
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "ignore")[:ERROR_CHARS]
        except Exception:
            pass
        return None, time.monotonic() - started, f"HTTP {exc.code}: {body.strip()}"
    except Exception as exc:
        return (None, time.monotonic() - started,
                f"{type(exc).__name__}: {str(exc)[:ERROR_CHARS]}")


# ---------------------------------------------------------------- shared helpers

def coerce_boolean(value):
    """A yes/no answer as a real bool, or None when it is neither.

    Tolerant on purpose: a boolean question can come back as ``true``, ``"true"``,
    ``"yes"`` or ``1`` depending on what serves it, and a bench that scored a
    correct ``"yes"`` as wrong would be measuring the spelling.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in TRUTHY:
            return True
        if text in FALSY:
            return False
    return None


def boolean_distribution(distribution):
    """A yes/no distribution keyed ``true``/``false``, or None.

    Accepts the same spellings ``coerce_boolean`` does, and fills in the other side
    when only one was reported, so a backend that returns ``{"yes": 0.8}`` is still
    scored against the labeled option.
    """
    if not isinstance(distribution, dict) or not distribution:
        return None
    out = {}
    for key, value in distribution.items():
        side = coerce_boolean(key)
        if side is None or value is None:
            continue
        out[TRUE if side else FALSE] = float(value)
    if not out:
        return None
    if TRUE not in out:
        out[TRUE] = round(max(0.0, 1.0 - out[FALSE]), 6)
    if FALSE not in out:
        out[FALSE] = round(max(0.0, 1.0 - out[TRUE]), 6)
    return out


def normalize(item: Item, answer, confidence, distribution):
    """``(answer, confidence, distribution)`` in the item's own answer space.

    One place decides what "the same answer" means for both kinds: a noul item's
    answer is a bool and its distribution is keyed true/false, a choice item's
    answer is one of its option names. Confidence is filled in from the
    distribution when the backend did not report one.
    """
    if item.kind == NOUL:
        answer = coerce_boolean(answer)
        distribution = boolean_distribution(distribution)
    elif distribution is not None:
        distribution = {str(k): float(v) for k, v in distribution.items()
                        if v is not None}
    if distribution and answer is not None:
        if item.kind == NOUL:
            key = TRUE if answer else FALSE
        else:
            key = str(answer)
        if key in distribution:
            # The confidence a caller gates on is the probability of the answer it
            # was handed, not a vendor's own derived number.
            confidence = distribution[key]
    if confidence is None and distribution:
        confidence = max(distribution.values())
    return answer, (None if confidence is None else float(confidence)), distribution


def rubric_lines(item: Item) -> str:
    """The item's options as ``name: description`` lines, skipping empty ones."""
    lines = []
    for option in item.options:
        text = item.description(option)
        lines.append(f"{option}: {text}" if text else option)
    return "\n".join(lines)


# ---------------------------------------------------------------- the backends

class Backend:
    """Build a request, parse a response, and only ``decide`` touches the network."""

    name = ""
    endpoint = ""
    input_usd_per_mtok = 0.0
    output_usd_per_mtok = 0.0
    #: True for a backend that talks to one of OUR nodes, so a 401 or a 429 from it
    #: is AINode's auth or AINode's rate limiter and stops the run. False for the
    #: hosted one, whose refusals are somebody else's and belong in their own row.
    local = True

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout
        self.requests = 0
        #: What the service said it was, once a response has said so. A record names
        #: the thing that answered (``jev-1.13.0``) and not the alias that was asked
        #: for (``jev-latest``).
        self.reported_model = ""
        self.reported_node = ""

    def request(self, item: Item) -> Request:
        raise NotImplementedError

    def parse(self, item: Item, data: dict) -> Decision:
        raise NotImplementedError

    def decide(self, item: Item) -> Decision:
        """One item, end to end. A failure is a Decision with ``error`` set.

        The exception is a refusal from one of OUR nodes: a 401 or a 429 means no
        item was answered, so it raises rather than becoming 110 wrong answers with
        a Brier score computed over them. The hosted backend is left alone
        (``local = False``): a 401 from TypeSafe is about the TypeSafe key, and the
        AINode sentence would send the operator after the wrong credential.
        """
        request = self.request(item)
        self.requests += 1
        data, wall, error = post_json(request, self.timeout)
        wall_ms = round(wall * 1000)
        if error:
            if self.local:
                auth.check_error(error)
            return Decision(wall_ms=wall_ms, error=error)
        try:
            decision = self.parse(item, data)
        except Exception as exc:
            excerpt = json.dumps(data)[:ERROR_CHARS] if data is not None else ""
            return Decision(wall_ms=wall_ms, excerpt=excerpt,
                            error=f"unreadable response: {type(exc).__name__}: "
                                  f"{str(exc)[:ERROR_CHARS]}")
        decision.wall_ms = wall_ms
        if decision.model:
            self.reported_model = decision.model
        if decision.node:
            self.reported_node = decision.node
        return decision

    def protocol(self) -> dict:
        """What the record says about how this backend was asked."""
        return {"backend": self.name, "endpoint": self.endpoint,
                "timeout_s": self.timeout,
                "input_usd_per_mtok": self.input_usd_per_mtok,
                "output_usd_per_mtok": self.output_usd_per_mtok}


class DecideBackend(Backend):
    """AINode's ``POST /v1/decide``: one typed question, an answer with a distribution.

    The endpoint's choice question takes bare option names and has no room for a
    per-option rubric, so the option descriptions are appended to the question text
    and the options go out as names. That keeps the same words in front of the model
    as the other two backends put there, which is the only way the three rows are
    comparable.
    """

    name = "ainode"

    def __init__(self, endpoint: str, model: str = "", api_key: str = DEFAULT_API_KEY,
                 timeout: float = DEFAULT_TIMEOUT, key_source: str = ""):
        super().__init__(timeout=timeout)
        if not endpoint:
            raise BackendError("the ainode backend needs --endpoint")
        self.endpoint = decide_url(endpoint)
        self.model = model
        self.api_key = api_key
        #: Where the key came from, for the one line a run prints about it. Never
        #: the key itself, the rule the hosted backend already holds.
        self.key_source = key_source

    def question(self, item: Item) -> dict:
        if item.kind == CHOICE:
            return {"question": f"{item.question}\n{rubric_lines(item)}",
                    "options": item.options}
        return {"question": item.question, "type": "boolean"}

    def request(self, item: Item) -> Request:
        payload = {"state": item.state,
                   "questions": {QUESTION_KEY: self.question(item)}}
        if self.model:
            payload["model"] = self.model
        headers = auth.bearer(self.api_key)
        return Request(self.endpoint, payload, headers)

    def parse(self, item: Item, data: dict) -> Decision:
        block = (data.get("decisions") or {})[QUESTION_KEY]
        answer, confidence, distribution = normalize(
            item, block.get("answer"), block.get("confidence"),
            block.get("distribution"))
        usage = data.get("usage") or {}
        return Decision(answer=answer, confidence=confidence,
                        distribution=distribution,
                        server_latency_ms=block.get("latency_ms")
                        or data.get("latency_ms"),
                        tokens_in=usage.get("prompt_tokens"),
                        tokens_out=usage.get("completion_tokens"),
                        model=data.get("model") or self.model,
                        node=data.get("node") or "",
                        excerpt=str(block.get("answer")))


class ChatBackend(Backend):
    """Any OpenAI-compatible engine: lettered options, thinking off, logprobs.

    The distribution is a softmax over the top logprobs of the single letter token,
    restricted to the letters this item offered. A server that returns no logprobs
    still gives an answer, and the row carries no probability rather than a made-up
    one, which is what ``no_confidence`` in the metrics counts.
    """

    name = "chat"
    #: The prototype's wording, kept so the fallback's numbers stay comparable.
    SYSTEM = ("You are a decision function. Reply with the single letter of the best "
              "option and nothing else.")
    MAX_TOKENS = 4
    TEMPERATURE = 0.0
    TOP_LOGPROBS = 20

    def __init__(self, endpoint: str, model: str, api_key: str = DEFAULT_API_KEY,
                 timeout: float = DEFAULT_TIMEOUT, think_off: bool = True,
                 key_source: str = ""):
        super().__init__(timeout=timeout)
        if not endpoint:
            raise BackendError("the chat backend needs --endpoint")
        if not model:
            raise BackendError("the chat backend needs --model")
        self.endpoint = chat_url(endpoint)
        self.model = model
        self.api_key = api_key
        self.think_off = think_off
        #: Where the key came from, for the one line a run prints about it.
        self.key_source = key_source

    def letters(self, item: Item) -> list:
        options = item.options
        if len(options) > len(LETTERS):
            raise BackendError(f"item {item.id} offers {len(options)} options; the "
                               f"chat backend letters at most {len(LETTERS)}")
        return list(LETTERS[:len(options)])

    def prompt(self, item: Item) -> str:
        letters = self.letters(item)
        if item.kind == CHOICE:
            body = "\n".join(f"{letters[i]}. {line}"
                             for i, line in enumerate(rubric_lines(item).split("\n")))
        else:
            body = f"{letters[0]}. yes\n{letters[1]}. no"
        return (f"{item.state}\n\nQuestion: {item.question}\nOptions:\n{body}\n"
                "Answer with one letter.")

    def request(self, item: Item) -> Request:
        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": self.SYSTEM},
                                {"role": "user", "content": self.prompt(item)}],
                   "max_tokens": self.MAX_TOKENS, "temperature": self.TEMPERATURE,
                   "logprobs": True, "top_logprobs": self.TOP_LOGPROBS}
        if self.think_off:
            # Both spellings, the way ainode/bench/measure.py sends them: a template
            # ignores the key it does not read, and a reasoning trace here is pure
            # latency on a question whose answer is one letter.
            payload["chat_template_kwargs"] = {"enable_thinking": False,
                                               "thinking": False}
        headers = auth.bearer(self.api_key)
        return Request(self.endpoint, payload, headers)

    def parse(self, item: Item, data: dict) -> Decision:
        choice = (data.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        letters = self.letters(item)
        options = item.options
        probabilities = letter_probabilities(choice.get("logprobs"), letters)
        distribution = None
        if probabilities:
            distribution = {options[letters.index(k)]: v
                            for k, v in probabilities.items()}
            for option in options:
                distribution.setdefault(option, 0.0)
        picked = next((c for c in text.upper() if c in letters), None)
        answer = options[letters.index(picked)] if picked else None
        if answer is None and distribution:
            # A reply that held no letter but a distribution that does is still an
            # answer the server gave; the excerpt keeps what it actually said.
            answer = max(distribution, key=distribution.get)
        if item.kind == NOUL and answer is not None:
            answer = answer == TRUE
        usage = data.get("usage") or {}
        answer, confidence, distribution = normalize(item, answer, None, distribution)
        return Decision(answer=answer, confidence=confidence,
                        distribution=distribution,
                        tokens_in=usage.get("prompt_tokens"),
                        tokens_out=usage.get("completion_tokens"),
                        model=data.get("model") or self.model,
                        excerpt=text[:ERROR_CHARS])


class JevBackend(Backend):
    """TypeSafe AI's System One model: a typed question in, a distribution out.

    The hosted comparison. It is the thing the local backends are measured against,
    it bills per input token, and the record's placement says ``typesafe.ai hosted``
    because there is no node of ours behind it.
    """

    name = "jev"
    input_usd_per_mtok = JEV_INPUT_USD_PER_MTOK
    output_usd_per_mtok = JEV_OUTPUT_USD_PER_MTOK
    #: Not one of our nodes, so its 401s and 429s are TypeSafe's to explain.
    local = False

    def __init__(self, api_key: str, model: str = JEV_MODEL, url: str = JEV_URL,
                 timeout: float = DEFAULT_TIMEOUT, key_source: str = ""):
        super().__init__(timeout=timeout)
        if not api_key:
            raise BackendError(missing_key_message())
        self.endpoint = url
        self.model = model or JEV_MODEL
        self.api_key = api_key
        self.key_source = key_source

    def question(self, item: Item) -> dict:
        question = {"type": item.kind, "instructions": item.question}
        if item.kind == CHOICE:
            question["criteria"] = {option: item.description(option)
                                    for option in item.options}
        elif item.criteria:
            question["criteria"] = dict(item.criteria)
        return question

    def request(self, item: Item) -> Request:
        payload = {"state": item.state, "model": self.model,
                   "questions": {QUESTION_KEY: self.question(item)}}
        return Request(self.endpoint, payload, auth.bearer(self.api_key))

    def parse(self, item: Item, data: dict) -> Decision:
        answer_block = (data.get("answers") or {})[QUESTION_KEY]
        if item.kind == CHOICE:
            raw_answer = answer_block.get("choice")
            distribution = answer_block.get("probabilities")
        else:
            probability = float(answer_block["noul"])
            raw_answer = probability >= 0.5
            distribution = {TRUE: probability, FALSE: 1.0 - probability}
        answer, confidence, distribution = normalize(
            item, raw_answer, answer_block.get("confidence"), distribution)
        usage = data.get("usage") or {}
        return Decision(answer=answer, confidence=confidence,
                        distribution=distribution,
                        tokens_in=usage.get("input_tokens"),
                        tokens_out=usage.get("output_tokens"),
                        model=data.get("model") or self.model,
                        node="typesafe.ai hosted",
                        excerpt=str(raw_answer))


# ---------------------------------------------------------------- helpers

def chat_url(endpoint: str) -> str:
    """``<endpoint>/chat/completions``, adding the ``/v1`` if it was left off."""
    base = (endpoint or "").rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/chat/completions"


def decide_url(endpoint: str) -> str:
    """``<endpoint>/decide``, adding the ``/v1`` if it was left off."""
    base = (endpoint or "").rstrip("/")
    if base.endswith("/decide"):
        return base
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/decide"


def letter_probabilities(logprobs, letters) -> dict:
    """Softmax over the first generated token's top logprobs, letters only.

    The first token with any non-whitespace text is the answer token; its
    ``top_logprobs`` are the server's own distribution over what it could have said
    there. Restricting to this item's letters and renormalizing turns that into a
    distribution over the options, which is the closest a chat endpoint gets to a
    calibrated decision.
    """
    content = (logprobs or {}).get("content") or []
    found = {}
    for token in content:
        if not (token.get("token") or "").strip():
            continue
        for candidate in token.get("top_logprobs") or []:
            key = (candidate.get("token") or "").strip().upper()
            if key in letters and key not in found:
                found[key] = float(candidate["logprob"])
        break
    if not found:
        return {}
    top = max(found.values())
    weights = {k: math.exp(v - top) for k, v in found.items()}
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def missing_key_message() -> str:
    return (f"no TypeSafe API key: pass --api-key, set ${JEV_KEY_ENV}, or put the "
            f"key in {JEV_KEY_FILE}")


def jev_api_key(explicit: str = ""):
    """``(key, source)`` for the jev backend. The source names where, never what.

    Order: ``--api-key``, then ``$TYPESAFE_API_KEY``, then ``~/.jev_api_key``. The
    value is never printed, never written into a record and never put in a note; the
    source string is what a run reports so a reader knows which credential answered.
    """
    if explicit:
        return explicit.strip(), "--api-key"
    from_env = os.environ.get(JEV_KEY_ENV, "").strip()
    if from_env:
        return from_env, f"${JEV_KEY_ENV}"
    path = pathlib.Path(JEV_KEY_FILE).expanduser()
    if path.is_file():
        key = path.read_text().strip()
        if key:
            return key, JEV_KEY_FILE
    return "", ""


def node_api_key(explicit: str = ""):
    """``(key, source)`` for the two backends that talk to one of our own nodes.

    ``--api-key``, then ``$AINODE_API_KEY``, then the placeholder an open node
    accepts. Deliberately NOT the same function as :func:`jev_api_key`: the hosted
    backend's credential is TypeSafe's, and feeding this node's key to
    ``api.typesafe.ai`` (or theirs to a node of ours) would hand a credential to the
    wrong party.
    """
    return auth.key_for(explicit, DEFAULT_API_KEY)


def build_backend(name: str, endpoint: str = "", model: str = "",
                  api_key: str = "", timeout: float = DEFAULT_TIMEOUT):
    """The named backend, or ``BackendError`` saying what it still needs."""
    if name == "jev":
        key, source = jev_api_key(api_key)
        return JevBackend(key, model=model or JEV_MODEL, timeout=timeout,
                          key_source=source)
    key, source = node_api_key(api_key)
    if name == "ainode":
        return DecideBackend(endpoint, model=model, api_key=key, timeout=timeout,
                             key_source=source)
    if name == "chat":
        return ChatBackend(endpoint, model, api_key=key, timeout=timeout,
                           key_source=source)
    raise BackendError(f"unknown backend {name!r}; pick from {', '.join(BACKENDS)}")


__all__ = ["BACKENDS", "DEFAULT_API_KEY", "DEFAULT_TIMEOUT", "ERROR_CHARS",
           "JEV_INPUT_USD_PER_MTOK", "JEV_KEY_ENV", "JEV_KEY_FILE", "JEV_MODEL",
           "JEV_OUTPUT_USD_PER_MTOK", "JEV_URL", "QUESTION_KEY", "Backend",
           "BackendError", "ChatBackend", "DecideBackend", "Decision", "JevBackend",
           "Request", "boolean_distribution", "build_backend", "chat_url",
           "coerce_boolean", "decide_url", "jev_api_key", "letter_probabilities",
           "missing_key_message", "node_api_key", "normalize", "post_json",
           "rubric_lines"]
