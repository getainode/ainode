"""The form fields of a buffered ``multipart/form-data`` body, without consuming it.

The speech-to-text paths the proxy forwards (``/v1/audio/transcriptions`` and
``/v1/audio/translations``) are the only inference paths that do not carry a JSON
body: OpenAI's audio API is a file upload, and the model id travels beside the
file as a form field. The fleet proxy routes on the model, so it has to read that
field, and it also has to forward the body BYTE FOR BYTE: a multipart body is
only parseable against the boundary in its own ``Content-Type`` header, so a
proxy that re-encoded the parts would hand the engine a body whose boundary no
longer matches the header it forwarded.

aiohttp's own ``request.multipart()`` is a one-shot reader over the request
stream, so calling it would consume the bytes the proxy still needs. This module
reads the fields out of the buffer the proxy already holds and touches nothing
else.

Only the small text fields are decoded. A part with a ``filename`` is the upload
itself (up to vLLM's 25 MB default), and no caller here wants it, so it is found
and stepped over without its bytes being copied or decoded. A value past
``MAX_FIELD_BYTES`` is treated as absent for the same reason: the fields this
reader exists for are a model id and a language code, and nothing is served by
materializing a megabyte of form data to route a request.
"""

from __future__ import annotations

import re

#: Longest field value this reader will decode. A ``model`` field is tens of
#: bytes; anything past this is either a file part with no ``filename`` or a
#: caller doing something this reader is not the place to support.
MAX_FIELD_BYTES = 64 * 1024

#: Most parts to walk in one body. A transcription request carries a handful
#: (file, model, language, response_format, temperature); the cap keeps a
#: malformed body from turning into a long scan.
MAX_PARTS = 64

_BOUNDARY_RE = re.compile(r';\s*boundary\s*=\s*(?:"([^"]*)"|([^";,\s]+))', re.I)

#: ``name=`` and ``filename=`` out of a part's Content-Disposition. Quoted is
#: what every client sends and what RFC 7578 asks for; the unquoted form is
#: accepted because a hand-rolled body is not worth a 400 over quoting.
_NAME_RE = re.compile(rb'(?:^|;)\s*name\s*=\s*(?:"([^"]*)"|([^";\r\n]*))', re.I)
_FILENAME_RE = re.compile(rb'(?:^|;)\s*filename\s*\*?\s*=', re.I)


def is_multipart(content_type: str) -> bool:
    """True when this Content-Type says the body is multipart/form-data."""
    return (content_type or "").split(";")[0].strip().lower() == "multipart/form-data"


def boundary_of(content_type: str) -> str:
    """The boundary out of a multipart Content-Type, or "" when there is none.

    A multipart body with no boundary in the header is unparseable by anyone,
    including the engine we would forward it to, so the caller of this function
    treats "" as a bad request rather than guessing one.
    """
    match = _BOUNDARY_RE.search(content_type or "")
    if not match:
        return ""
    return (match.group(1) or match.group(2) or "").strip()


def _part_name(headers: bytes) -> tuple:
    """``(name, is_file)`` for one part, from its raw header block."""
    for line in headers.split(b"\r\n"):
        if not line.lower().startswith(b"content-disposition:"):
            continue
        value = line.split(b":", 1)[1]
        match = _NAME_RE.search(value)
        if not match:
            return "", False
        raw = match.group(1) if match.group(1) is not None else (match.group(2) or b"")
        name = raw.decode("utf-8", "replace").strip()
        return name, bool(_FILENAME_RE.search(value))
    return "", False


def form_fields(body: bytes, content_type: str) -> dict:
    """The text form fields of ``body``, as ``{name: value}``.

    Never raises: a body this cannot parse comes back as the fields it managed to
    read (usually none), and the caller answers on the absence of the field it
    wanted rather than on a parse error nobody can act on. File parts are skipped,
    and a repeated field keeps the FIRST value, which is what aiohttp and vLLM's
    own form parsing do.
    """
    fields: dict = {}
    boundary = boundary_of(content_type)
    if not boundary or not body:
        return fields
    delim = b"--" + boundary.encode("utf-8", "replace")

    pos = body.find(delim)
    if pos < 0:
        return fields
    parts = 0
    while pos >= 0 and parts < MAX_PARTS:
        parts += 1
        cursor = pos + len(delim)
        if body[cursor:cursor + 2] == b"--":
            break  # closing delimiter: the body is finished
        # Past the delimiter's own line break into the part's headers.
        if body[cursor:cursor + 2] == b"\r\n":
            cursor += 2
        elif body[cursor:cursor + 1] == b"\n":
            cursor += 1
        head_end = body.find(b"\r\n\r\n", cursor)
        head_sep = 4
        if head_end < 0:
            head_end = body.find(b"\n\n", cursor)
            head_sep = 2
        next_pos = body.find(delim, cursor)
        if head_end < 0 or (next_pos >= 0 and head_end > next_pos):
            pos = next_pos  # headerless junk between delimiters
            continue
        name, is_file = _part_name(body[cursor:head_end])
        value_start = head_end + head_sep
        value_end = next_pos if next_pos >= 0 else len(body)
        # The CRLF before the next delimiter belongs to the delimiter, not the value.
        if body[value_end - 2:value_end] == b"\r\n":
            value_end -= 2
        elif body[value_end - 1:value_end] == b"\n":
            value_end -= 1
        if (name and not is_file and name not in fields
                and 0 <= value_end - value_start <= MAX_FIELD_BYTES):
            fields[name] = body[value_start:value_end].decode("utf-8", "replace")
        pos = next_pos
    return fields
