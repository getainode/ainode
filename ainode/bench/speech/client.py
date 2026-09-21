"""One call: ``POST <endpoint>/audio/transcriptions``, and one reading of the answer.

Split the way every backend in ``ainode/bench`` is split, and for the same reason:
``request_for(...)`` builds the body and ``parse(...)`` reads one, both pure functions
of their arguments, and only ``SpeechClient.transcribe`` touches the network. A test
can then pin the exact bytes on the wire and drive every response shape from canned
payloads with no server anywhere.

**This is the one bench section whose request body is not JSON.** OpenAI's
speech-to-text API is a ``multipart/form-data`` upload: the audio is a file part and
the model id is a form field beside it, which is also what AINode's proxy routes on
(`api/multipart.py::form_fields` reads that field out of the buffered body and
forwards the bytes unchanged). So the body is assembled here rather than handed to a
library: ``encode_multipart`` writes the parts with an explicit boundary and the
``Content-Type`` header names that same boundary, because a multipart body is only
parseable against the boundary in its own header. Stdlib only, so there is no
requests-toolbelt to reach for anyway.

The endpoint is deliberately the OpenAI one and nothing else. An AINode node and a
vLLM engine both serve this path with the same body, so the same measurement runs
against ``http://node:3000/v1`` (through the fleet router) and against
``http://node:8002/v1`` (the engine itself), and the record says which was used.
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from ainode.bench import auth

#: Seconds one request gets. A five-second clip on a busy node is the slow case, and
#: it is nothing like a generation, so this is not two minutes.
DEFAULT_TIMEOUT = 120
#: Bearer token AINode's proxy accepts by default, same as the other sections.
DEFAULT_API_KEY = "ainode"
#: How much of an error body is worth keeping in a row.
ERROR_CHARS = 300
#: The multipart boundary. Fixed rather than random so a test can pin the exact bytes;
#: it carries no data, and none of the clip texts contains it.
BOUNDARY = "----ainode-bench-speech-boundary"
#: ``response_format`` asked for. ``json`` is the default an ordinary caller gets and
#: the only one every ASR engine agrees on; ``verbose_json`` would add segments this
#: section does not score, and ``text`` would give up the error body's shape.
RESPONSE_FORMAT = "json"


class SpeechError(RuntimeError):
    """A run that cannot start: no endpoint, no model, an unreadable clip. Not used
    for one failed request, which is a row with an ``error``."""


@dataclass
class Request:
    """One outgoing call. ``headers`` holds the credential and is never printed."""

    url: str
    body: bytes
    content_type: str
    fields: dict = field(default_factory=dict)
    filename: str = ""
    headers: dict = field(default_factory=dict)

    def curl_safe(self) -> str:
        """The request as a line to show a person: no header, so no key.

        The audio is shown as its name and length rather than its bytes, which is the
        shape of the call a reader wants out of a dry run.
        """
        parts = " ".join(f"-F {k}={v}" for k, v in sorted(self.fields.items()))
        return (f"POST {self.url}  -F file=@{self.filename} "
                f"({len(self.body)} bytes of multipart)  {parts}")


@dataclass
class Reply:
    """What one call came back as, before anything is scored."""

    text: str = ""
    wall_ms: float = 0.0
    error: str | None = None
    raw: dict = field(default_factory=dict)


def encode_multipart(fields: dict, filename: str, audio: bytes,
                     boundary: str = BOUNDARY) -> bytes:
    """The exact bytes of a ``multipart/form-data`` body: text fields, then the file.

    The file part goes LAST, which is the order curl writes it in and the order the
    proxy's field reader is exercised against: it has to step over the file part to
    find a field, and a body whose fields all precede the file would never test that.
    CRLF line endings, because the multipart grammar requires them and an engine
    parsing with a strict reader rejects bare newlines.
    """
    crlf = b"\r\n"
    out = bytearray()
    for name, value in fields.items():
        out += b"--" + boundary.encode() + crlf
        out += (f'Content-Disposition: form-data; name="{name}"').encode() + crlf
        out += crlf
        out += str(value).encode() + crlf
    out += b"--" + boundary.encode() + crlf
    out += (f'Content-Disposition: form-data; name="file"; '
            f'filename="{filename}"').encode() + crlf
    out += b"Content-Type: audio/wav" + crlf
    out += crlf
    out += audio + crlf
    out += b"--" + boundary.encode() + b"--" + crlf
    return bytes(out)


def request_for(endpoint: str, model: str, path, audio: bytes,
                api_key: str = DEFAULT_API_KEY, language: str = "",
                temperature=None, path_name: str = "transcriptions") -> Request:
    """The body a transcription call carries: the model, the audio, and little else.

    No prompt, no ``verbose_json``, no ``timestamp_granularities``. Every one of those
    would change what is being measured, and a default the engine chose is the default
    a caller gets, so it is the one worth measuring. ``language`` is sent only when a
    run asks for it, because leaving it out measures the detection an ordinary caller
    gets.
    """
    fields = {"model": model, "response_format": RESPONSE_FORMAT}
    if language:
        fields["language"] = language
    if temperature is not None:
        fields["temperature"] = str(temperature)
    name = pathlib.Path(str(path)).name
    return Request(
        url=endpoint.rstrip("/") + "/audio/" + path_name,
        body=encode_multipart(fields, name, audio),
        content_type=f"multipart/form-data; boundary={BOUNDARY}",
        fields=fields,
        filename=name,
        headers=auth.bearer(api_key),
    )


def parse(data) -> Reply:
    """Read one transcription response: the ``text`` field, and nothing else.

    A body that is valid JSON but carries no ``text`` is an error rather than an empty
    transcript, because an empty transcript scores as every word deleted and would
    read as a model that heard silence. A plain-text body (an engine ignoring
    ``response_format``) is accepted as the transcript, since that is unambiguously
    what it is.
    """
    if isinstance(data, str):
        text = data.strip()
        if not text:
            return Reply(error="response body was empty")
        return Reply(text=text, raw={"text": text})
    if not isinstance(data, dict):
        return Reply(error="response was not a JSON object")
    text = data.get("text")
    if not isinstance(text, str):
        return Reply(error="response carried no 'text' field")
    return Reply(text=text.strip(), raw=data)


def post_multipart(request: Request, timeout: float):
    """``(data, wall_seconds, error)``. Never raises, so one bad call is one row.

    An HTTP error body is kept (truncated) because a 400 from an audio endpoint names
    the reason (a model it does not serve, a format libsndfile cannot read, a clip past
    the size cap), and that is the finding. A body that is not JSON is handed to
    ``parse`` as text rather than being called a failure.
    """
    req = urllib.request.Request(
        request.url, data=request.body, method="POST",
        headers={"Content-Type": request.content_type,
                 "Content-Length": str(len(request.body)),
                 **request.headers})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
        elapsed = time.monotonic() - started
        try:
            return json.loads(raw.decode("utf-8")), elapsed, None
        except Exception:
            return raw.decode("utf-8", "ignore"), elapsed, None
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


class SpeechClient:
    """The one thing in this package that talks to a server."""

    def __init__(self, endpoint: str, model: str, api_key: str = DEFAULT_API_KEY,
                 timeout: float = DEFAULT_TIMEOUT, language: str = "",
                 translate: bool = False, key_source: str = ""):
        if not endpoint:
            raise SpeechError("--endpoint is required: the OpenAI-compatible base "
                              "with its /v1, e.g. http://100.122.26.9:3000/v1")
        if not model:
            raise SpeechError("--model is required: the model id exactly as served")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key or DEFAULT_API_KEY
        self.timeout = float(timeout)
        self.language = language or ""
        #: Where the key came from (``--api-key``, ``$AINODE_API_KEY`` or the
        #: default). The only half of it a run may print.
        self.key_source = key_source or "the default"
        #: ``/v1/audio/translations`` instead of transcriptions. Whisper turbo is a
        #: transcription model and cannot translate, so this is for the ASR models
        #: that can; the record says which path was measured.
        self.path_name = "translations" if translate else "transcriptions"
        #: What the server called the model in its last answer, when it said. ASR
        #: responses usually carry only ``text``, so this is often empty and the
        #: record then names the requested id.
        self.reported_model = ""

    def request(self, clip: dict, audio: bytes) -> Request:
        return request_for(self.endpoint, self.model, clip["path"], audio,
                           api_key=self.api_key, language=self.language,
                           path_name=self.path_name)

    def read_audio(self, clip: dict) -> bytes:
        try:
            return pathlib.Path(clip["path"]).read_bytes()
        except Exception as exc:
            raise SpeechError(f"{clip['path']} could not be read: {exc}") from exc

    def transcribe(self, clip: dict, audio=None) -> Reply:
        """One call, timed. A failure is a Reply with an ``error``, never a raise.

        The exception is a refusal: a 401 or a 429 raises
        :class:`ainode.bench.auth.EndpointRefused` rather than becoming a row. A
        refused clip is already counted out of every rate (it is a row with an error
        and nulls), so ten of them would produce a record with no word error rate at
        all and a reader would have to guess why.
        """
        payload = audio if audio is not None else self.read_audio(clip)
        data, wall_s, error = post_multipart(self.request(clip, payload), self.timeout)
        if error is not None:
            auth.check_error(error)
            return Reply(wall_ms=round(wall_s * 1000, 2), error=error)
        reply = parse(data)
        reply.wall_ms = round(wall_s * 1000, 2)
        named = reply.raw.get("model") if isinstance(reply.raw, dict) else None
        if isinstance(named, str) and named and not self.reported_model:
            self.reported_model = named
        return reply

    def ping(self):
        """Wall ms of one ``GET <endpoint>/models``, or None if it did not answer.

        A request that transcribes nothing, over the same link the timed ones use. At
        these latencies it is not a detail: a bench driven from a laptop over a tailnet
        spends a measurable share of every figure on the wire, and a real-time factor
        with an unnamed wire in it cannot be read as the engine's own.
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
        return {"path": f"POST /v1/audio/{self.path_name}",
                "endpoint": self.endpoint,
                "model_requested": self.model,
                "content_type": f"multipart/form-data; boundary={BOUNDARY}",
                "response_format": RESPONSE_FORMAT,
                "language": self.language or "not sent, so the engine detects it",
                "timeout_s": self.timeout,
                "body": "the model id and response_format as form fields and the WAV "
                        "as the file part, in that order; no prompt, no "
                        "timestamp_granularities, so every default is the engine's own"}


__all__ = ["BOUNDARY", "DEFAULT_API_KEY", "DEFAULT_TIMEOUT", "ERROR_CHARS",
           "RESPONSE_FORMAT", "Reply", "Request", "SpeechClient", "SpeechError",
           "encode_multipart", "parse", "post_multipart", "request_for"]
