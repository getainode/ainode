"""One call: ``POST <endpoint>/v1/embeddings``, and one reading of the answer back.

Split the way every backend in ``ainode/bench`` is split, and for the same reason:
``request_for(...)`` builds the payload and ``parse(...)`` reads one, both pure
functions of their arguments, and only ``EmbedClient.embed`` touches the network. A
test can then pin the exact request shape and drive every response shape from canned
payloads with no server anywhere.

The endpoint is deliberately the OpenAI one and nothing else. An AINode node and a
vLLM engine both serve this path with the same body, so the same measurement runs
against ``http://node:3000/v1`` (through the fleet router) and against
``http://node:8001/v1`` (the engine itself), and the record says which was used.

Stdlib only: urllib for the transport.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from ainode.bench import auth

#: Seconds one request gets. A batch of 64 short texts on a busy node is the slow
#: case here, and it is nothing like a generation, so this is not two minutes.
DEFAULT_TIMEOUT = 60
#: Bearer token AINode's proxy accepts by default, same as the other sections.
DEFAULT_API_KEY = "ainode"
#: How much of an error body is worth keeping in a row.
ERROR_CHARS = 300


class EmbedError(RuntimeError):
    """A run that cannot start: no endpoint, no model, an endpoint that never
    answered. Not used for one failed request, which is a row with an ``error``."""


@dataclass
class Request:
    """One outgoing call. ``headers`` holds the credential and is never printed."""

    url: str
    payload: dict
    headers: dict = field(default_factory=dict)

    def curl_safe(self) -> str:
        """The request as a line to show a person: no header, so no key.

        The texts are replaced by their count. A dry run is about the shape of the
        call, and printing 64 strings would bury it.
        """
        shown = dict(self.payload)
        texts = shown.get("input")
        if isinstance(texts, list):
            shown["input"] = f"<{len(texts)} text(s)>"
        return f"POST {self.url}  {json.dumps(shown, sort_keys=True)}"


@dataclass
class Reply:
    """What one call came back as, before anything is scored."""

    vectors: list = field(default_factory=list)
    model: str = ""
    tokens: int | None = None
    wall_ms: float = 0.0
    error: str | None = None

    @property
    def dimensions(self) -> int | None:
        return len(self.vectors[0]) if self.vectors else None


def request_for(endpoint: str, model: str, texts: list,
                api_key: str = DEFAULT_API_KEY) -> Request:
    """The body an embeddings call carries: the model id and the texts, nothing else.

    No ``encoding_format``, no ``dimensions``, no truncation hint. Every one of those
    would change what is being measured, and a default the engine chose is the
    default a caller gets, so it is the one worth measuring.
    """
    return Request(
        url=endpoint.rstrip("/") + "/embeddings",
        payload={"model": model, "input": list(texts)},
        headers=auth.bearer(api_key),
    )


def parse(data: dict) -> Reply:
    """Read one embeddings response: the vectors in index order, and the tokens.

    ``data`` is ordered by the server's own ``index`` rather than by arrival order in
    the list, because a batch's vectors have to line up with the texts that produced
    them and nothing in the protocol promises the order they are serialised in.
    ``usage.prompt_tokens`` is the token count; ``total_tokens`` is the fallback, and
    a response carrying neither reports ``None`` rather than a zero nobody measured.
    """
    if not isinstance(data, dict):
        return Reply(error="response was not a JSON object")
    rows = data.get("data")
    if not isinstance(rows, list) or not rows:
        return Reply(error="response carried no 'data' list")
    indexed = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            return Reply(error="an entry of 'data' was not an object")
        vector = row.get("embedding")
        if not isinstance(vector, list) or not vector:
            return Reply(error="an entry of 'data' carried no embedding")
        index = row.get("index")
        indexed.append((index if isinstance(index, int) else position,
                        [float(x) for x in vector]))
    indexed.sort(key=lambda pair: pair[0])
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    tokens = usage.get("prompt_tokens")
    if not isinstance(tokens, int):
        tokens = usage.get("total_tokens")
    return Reply(vectors=[v for _i, v in indexed],
                 model=str(data.get("model") or ""),
                 tokens=tokens if isinstance(tokens, int) else None)


def post_json(request: Request, timeout: float):
    """``(data, wall_seconds, error)``. Never raises, so one bad call is one row.

    An HTTP error body is kept (truncated) because a 400 from an embeddings endpoint
    usually names the reason (a text past the window, a model it does not serve), and
    that is the finding.
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


class EmbedClient:
    """The one thing in this package that talks to a server."""

    def __init__(self, endpoint: str, model: str, api_key: str = DEFAULT_API_KEY,
                 timeout: float = DEFAULT_TIMEOUT, key_source: str = ""):
        if not endpoint:
            raise EmbedError("--endpoint is required: the OpenAI-compatible base "
                             "with its /v1, e.g. http://100.72.9.84:8001/v1")
        if not model:
            raise EmbedError("--model is required: the model id exactly as served")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key or DEFAULT_API_KEY
        self.timeout = float(timeout)
        #: Where the key came from (``--api-key``, ``$AINODE_API_KEY`` or the
        #: default). The only half of it a run may print.
        self.key_source = key_source or "the default"
        #: What the server called the model in its last answer. Recorded instead of
        #: the requested id, so a record names the thing that replied.
        self.reported_model = ""

    def request(self, texts: list) -> Request:
        return request_for(self.endpoint, self.model, texts, self.api_key)

    def embed(self, texts: list) -> Reply:
        """One call, timed. A failure is a Reply with an ``error``, never a raise.

        The exception is a refusal: a 401 or a 429 raises
        :class:`ainode.bench.auth.EndpointRefused` rather than becoming a row, because
        every request in the run would carry the same error and the record would
        report a p50, a throughput and a pair-ordering score over nothing.
        """
        data, wall_s, error = post_json(self.request(texts), self.timeout)
        if error is not None:
            auth.check_error(error)
            return Reply(wall_ms=round(wall_s * 1000, 2), error=error)
        reply = parse(data)
        reply.wall_ms = round(wall_s * 1000, 2)
        if reply.model and not self.reported_model:
            self.reported_model = reply.model
        return reply

    def ping(self):
        """Wall ms of one ``GET <endpoint>/models``, or None if it did not answer.

        A request that embeds nothing, over the same connection the timed ones use.
        At these latencies it is not a detail: a single short text comes back in tens
        of milliseconds, and a bench driven from a laptop over a tailnet spends a
        comparable amount of that on the wire. Measuring the floor is the only way a
        p50 in the record can be read as something other than the engine's own time.
        """
        req = urllib.request.Request(
            self.endpoint + "/models",
            headers={"Accept": "application/json", **auth.bearer(self.api_key)})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                response.read()
            return round((time.monotonic() - started) * 1000, 2)
        except Exception:
            return None

    def protocol(self) -> dict:
        """What a reader needs to reproduce the calls, minus the credential."""
        return {"path": "POST /v1/embeddings",
                "endpoint": self.endpoint,
                "model_requested": self.model,
                "timeout_s": self.timeout,
                "body": "model and input only; no encoding_format, no dimensions, "
                        "no truncation hint, so every default is the engine's own"}


__all__ = ["DEFAULT_API_KEY", "DEFAULT_TIMEOUT", "ERROR_CHARS", "EmbedClient",
           "EmbedError", "Reply", "Request", "parse", "post_json", "request_for"]
