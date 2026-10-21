"""The key a bench run presents to the node it is measuring.

Every section of ``scripts/ainode-bench.py`` runs OUT of process, so every request
it makes crosses the node's auth middleware and its rate limiter. Before this
module the throughput section sent no ``Authorization`` header at all and no
section sent one on a control-plane read, so a bench pointed at a protected node
(which since #244 is what a fresh install is) answered 401 on its first call and
the model wore the zero (#245).

Three rules, all here so no section can hold a different one:

  * **One resolution order**: ``--api-key``, then ``$AINODE_API_KEY``, then
    nothing. A section that has a placeholder of its own (``ainode``, what an open
    node accepts) falls back to it after those two, so an unprotected node keeps
    working exactly as it did.
  * **The key is never printed and never written into a record.** A run reports
    the SOURCE it came from and nothing else, the rule ``bench/decide/README.md``
    already holds for the TypeSafe key.
  * **A refused request is not a measurement.** A 401 says the node wants a key
    and a 429 says the rate limiter refused this caller; neither is a statement
    about the model, so both stop the section instead of landing as failed rows.
    That is ``HiddenTestsUnavailable`` (#153) applied to the transport: a bench
    that turns one refusal into ten model failures is worse than one that stops.

Stdlib only, like the rest of ``ainode/bench``.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

#: The variable every section falls back to when no ``--api-key`` was passed. One
#: name for all six, because an operator sets it once per shell and then runs
#: whichever section they came for.
ENV_API_KEY = "AINODE_API_KEY"

#: Seconds the preflight GET gets. A control-plane read is quick or it is broken,
#: the same reasoning as ``measure.CTL_TIMEOUT``.
PREFLIGHT_TIMEOUT = 10

#: How much of a refusal body is worth quoting back. A 429 names which limit
#: refused and how long to wait, and that is the finding.
BODY_CHARS = 300

#: Statuses that refuse the RUN rather than the request. 401 is the auth
#: middleware (``ainode/auth/middleware.py``), 429 the rate limiter
#: (``ainode/ratelimit/middleware.py``). Nothing else belongs here: a 400 is about
#: the body and a 503 is about the engine, so both stay findings about the request
#: that made them, in that request's own row.
REFUSING_STATUSES = (401, 429)

#: The sentence a 401 gets. Quoted verbatim by the tests, because it is the whole
#: user-facing fix: an operator who reads it knows the next thing to type.
NEEDS_KEY = "this node wants an API key (pass --api-key or set AINODE_API_KEY)"

#: The two shapes an HTTP status reaches this module in: ``HTTP 401: <body>``, which
#: the section clients build out of an ``HTTPError``, and ``HTTPError: HTTP Error
#: 401: Unauthorized``, which is urllib's own ``str()`` as ``measure.get_json`` and
#: ``measure.stream_chat`` record it.
_STATUS_RE = re.compile(r"^(?:HTTPError: )?HTTP(?: Error)? (\d{3}):?\s*(.*)$", re.S)


class EndpointRefused(RuntimeError):
    """The node refused the run, so no verdict is a verdict.

    Raised instead of recording a failure, and carrying the status that caused it.
    A 401 scored as ten model failures is the #153 mistake in a new place: the
    measurement never happened, and a record saying it did is worse than no record.
    """

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def resolve_key(explicit: str = "", env=None):
    """``(key, source)``: ``--api-key``, then ``$AINODE_API_KEY``, then ``("", "")``.

    The source names WHERE the key came from and never what it is: it is the only
    half of this a run may print.
    """
    text = (explicit or "").strip()
    if text:
        return text, "--api-key"
    environ = os.environ if env is None else env
    from_env = (environ.get(ENV_API_KEY) or "").strip()
    if from_env:
        return from_env, f"${ENV_API_KEY}"
    return "", ""


def key_for(explicit: str = "", placeholder: str = "", env=None):
    """``(key, source)`` for a section that has a placeholder to fall back to.

    The placeholder (``ainode``) is what an open node accepts and what the harness
    adapters need in a config field that cannot be empty, so it stays the last
    resort rather than the default that would shadow ``$AINODE_API_KEY``.
    """
    key, source = resolve_key(explicit, env=env)
    if key:
        return key, source
    return placeholder, "the default" if placeholder else ""


def bearer(key: str) -> dict:
    """``{"Authorization": "Bearer <key>"}``, or ``{}`` when there is no key.

    The one place the header is spelled. Every request the bench makes merges this
    into its own headers, which is what ``tests/test_bench_auth.py`` walks the
    source of this package to insist on.
    """
    text = (key or "").strip()
    return {"Authorization": f"Bearer {text}"} if text else {}


def models_url(base: str) -> str:
    """The model-list URL under either spelling of an endpoint base.

    The sections disagree about the ``/v1``: five take ``--endpoint
    http://node:3000/v1`` and the throughput one takes ``--url http://node:8000``.
    The preflight has to work from both, so it is settled here once.
    """
    text = (base or "").rstrip("/")
    return text + "/models" if text.endswith("/v1") else text + "/v1/models"


def limit_detail(body: str) -> str:
    """The limiter's own sentence out of a 429 body, or the body, truncated.

    ``ainode/ratelimit/middleware.py`` answers with an ``error.message`` naming
    which of the two limits refused and how many seconds to wait. A bench that
    reported only "429" would leave an operator guessing whether to lower
    ``--streams``, lower ``--concurrency`` or simply wait.
    """
    text = (body or "").strip()
    if not text:
        return ""
    try:
        data = json.loads(text)
    except ValueError:
        return text[:BODY_CHARS]
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()[:BODY_CHARS]
    return text[:BODY_CHARS]


def refusal(status, body: str = ""):
    """The sentence for a status that refuses the run, or None for anything else."""
    try:
        code = int(status)
    except (TypeError, ValueError):
        return None
    if code == 401:
        return NEEDS_KEY
    if code == 429:
        detail = limit_detail(body)
        return "this node is rate limiting the bench" + (f": {detail}" if detail else "")
    return None


def split_error(error):
    """``(status, body)`` out of one of the bench's error strings, else ``(None, "")``."""
    match = _STATUS_RE.match(str(error or "").strip())
    if not match:
        return None, ""
    return int(match.group(1)), match.group(2).strip()


def status_of(error):
    """The HTTP status one of the bench's error strings carries, or None."""
    return split_error(error)[0]


def refuse(status, body: str = "") -> None:
    """Raise :class:`EndpointRefused` when ``status`` refuses the run.

    Called from the one place each section turns a failed request into a row, so a
    refusal leaves through the exception instead of through the scoring.
    """
    message = refusal(status, body)
    if message:
        raise EndpointRefused(message, status=int(status))


def check_error(error) -> None:
    """:func:`refuse` over an error string a section has already built.

    The section clients have read the status and the body by the time they format a
    row, so this reads them back rather than asking all five to change shape.
    """
    status, body = split_error(error)
    if status is not None:
        refuse(status, body)


def check_exception(exc) -> None:
    """:func:`refuse` over a urllib exception, before it is formatted into a row."""
    code = getattr(exc, "code", None)
    if code is None:
        return
    body = ""
    read = getattr(exc, "read", None)
    if callable(read):
        try:
            body = read().decode("utf-8", "ignore")[:BODY_CHARS]
        except Exception:
            body = ""
    refuse(code, body)


def explain(error) -> str:
    """One of the bench's error strings, with a refusal spelled out for a person.

    Used on a control-plane read, which degrades a placement field rather than
    stopping a run: the warning it produces should still say "this node wants an
    API key" and not ``HTTPError: HTTP Error 401: Unauthorized``.
    """
    text = str(error or "")
    status, body = split_error(text)
    if status is None:
        return text
    return refusal(status, body) or text


def preflight(base: str, api_key: str = "", timeout: float = PREFLIGHT_TIMEOUT):
    """The reason this endpoint refuses the run, or None. One GET, before scoring.

    ``harness/runner.py::preflight_test_interpreter`` for the transport, and for the
    same reason: run it once before the first measured request so no section starts
    against a node that would refuse every one of them the same way. It is the only
    protection the harness section can have, because there the requests are made by
    an agent CLI in a subprocess and a 401 reaches this process as ten tasks whose
    tests never passed.

    Only a 401 and a 429 are a reason to stop. An endpoint that 404s the model
    list, answers something else or does not answer at all gets None: none of those
    is a statement about the key, and the sections already degrade or fail on their
    own reads.
    """
    request = urllib.request.Request(
        models_url(base), headers={"Accept": "application/json", **bearer(api_key)})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read(1)
        return None
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "ignore")[:BODY_CHARS]
        except Exception:
            body = ""
        return refusal(exc.code, body)
    except Exception:
        return None


def stop(message, out=print) -> int:
    """Print a refusal and return the exit code a refused run leaves behind.

    One wording for all six sections: the operator sees the same two lines whether
    the node refused the throughput run or the speech one.
    """
    out(f"\n  REFUSED  {message}")
    out("  nothing was scored and no record was written")
    return 2


__all__ = ["BODY_CHARS", "ENV_API_KEY", "NEEDS_KEY", "PREFLIGHT_TIMEOUT",
           "REFUSING_STATUSES", "EndpointRefused", "bearer", "check_error",
           "check_exception", "explain", "key_for", "limit_detail", "models_url",
           "preflight", "refusal", "refuse", "resolve_key", "split_error",
           "status_of", "stop"]
