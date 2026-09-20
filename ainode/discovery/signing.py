"""HMAC signing for UDP discovery announcements.

A datagram is the whole join protocol (#169). The election prefers a node that
announces ``role: "master"``, the cluster id is readable unauthenticated over
HTTP, and an announcement also carries the instances a peer will route inference
traffic to, so any host on the broadcast domain could take over a fleet with one
UDP packet. ``cluster_secret`` existed in the config, was scrubbed from
``/api/config`` as though it protected something, and was read by nothing.

It signs the announcement now:

* A sender with a secret appends a keyed digest over the payload it is about to
  send. It reads the secret per send, not once at startup, so rotating it is a
  config edit plus one broadcast interval rather than a restart of every node.
* A receiver with a secret drops a datagram that is unsigned or badly signed.
* A receiver with NO secret keeps the old behaviour and says so once: nothing on
  this fleet sets one today, and a release that made discovery mandatory-signed
  would silently partition every existing cluster on upgrade.

Wire shape: the signature is ONE extra top-level key beside the announcement's
own fields, never a wrapper around them. ``NodeAnnouncement.from_json`` drops
keys it does not know, so a node too old to have heard of signing still parses a
signed announcement and stays visible in the cluster view. The digest covers the
raw received mapping rather than the fields this build happens to understand, so
a peer running a newer build that carries additional fields still verifies.

Not covered: replay. A recorded datagram re-sent verbatim verifies, because the
payload is what it was when it was signed. It can only re-assert what a real
node already said (an attacker cannot edit a field without the key), and the
alternative, a freshness window over ``timestamp``, drops every announcement
from a node whose clock has drifted. The cluster shares no clock, so that trade
is worse than the attack.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional

# The signature's key in the datagram. Short on purpose: the payload is capped
# by MAX_ANNOUNCEMENT_BYTES and every byte here is a byte a future field cannot
# have.
SIGNATURE_FIELD = "sig"

# Verdicts from :func:`rejection`. The empty string is "accept" so a caller can
# write ``if reason:``.
ACCEPT = ""
UNSIGNED = "unsigned"
BAD_SIGNATURE = "bad signature"


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    """The exact bytes a signature is computed over.

    Sorted keys and no whitespace, so the sender and the receiver agree without
    depending on dict order or on how either side spelled its JSON. The
    signature field itself is never part of its own digest.
    """
    body = {k: v for k, v in payload.items() if k != SIGNATURE_FIELD}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign(payload: Mapping[str, Any], secret: str) -> str:
    """Hex HMAC-SHA256 of *payload* under *secret*."""
    return hmac.new(
        secret.encode("utf-8"), canonical_bytes(payload), hashlib.sha256
    ).hexdigest()


def seal(payload: Mapping[str, Any], secret: Optional[str]) -> dict:
    """Return the dict to put on the wire: *payload*, signed when there is a key.

    With no secret this is the payload as it always was, minus any stale
    signature, which is what keeps a fleet with no secret working exactly as
    before.
    """
    sealed = {k: v for k, v in payload.items() if k != SIGNATURE_FIELD}
    if not secret:
        return sealed
    sealed[SIGNATURE_FIELD] = sign(sealed, secret)
    return sealed


def rejection(payload: Mapping[str, Any], secret: Optional[str]) -> str:
    """Why *payload* must be dropped, or ``ACCEPT`` when it may be used.

    A receiver with no secret accepts everything: see the module docstring for
    why that is the migration path and not an oversight.
    """
    if not secret:
        return ACCEPT
    provided = payload.get(SIGNATURE_FIELD)
    if not isinstance(provided, str) or not provided:
        return UNSIGNED
    if not hmac.compare_digest(provided, sign(payload, secret)):
        return BAD_SIGNATURE
    return ACCEPT


class ClusterSecret:
    """The live ``cluster_secret``, re-read from disk when the file changes.

    Callable with no arguments, which is the seam the sender and the listener
    take, so neither captures a value at startup. Rotation on a running fleet is
    then an edit of ``config.json`` on each node (or a ``PUT /api/config``, which
    writes the same file) and no restart: every node keeps signing with whatever
    the file says at send time, and verifying with whatever it says at receive
    time. Roll the secret onto the receivers first and the senders after, or the
    other way round; the cluster is only dark for the nodes that disagree, and
    only until the file is written.

    The file is stat'ed per call and parsed only when its identity changes, so a
    per-datagram call costs a stat. An unreadable or malformed file keeps the
    last value that did parse: a half-written config must not take a fleet's
    authentication down.

    ``config`` is the in-memory NodeConfig, used only when the file has nothing
    to say (a node with no config.json on disk yet).
    """

    def __init__(self, config: Any = None, path: Optional[Path] = None):
        self._config = config
        self._path = path
        self._stamp: Optional[tuple] = None
        self._value: Optional[str] = None
        self._loaded = False

    def _config_file(self) -> Path:
        if self._path is not None:
            return Path(self._path)
        # Imported per call so a test (and a container that moves AINODE_HOME)
        # sees the current value rather than the one bound at import time.
        from ainode.core.config import CONFIG_FILE

        return Path(CONFIG_FILE)

    def _fallback(self) -> Optional[str]:
        value = getattr(self._config, "cluster_secret", None)
        return str(value) if value else None

    def __call__(self) -> Optional[str]:
        path = self._config_file()
        try:
            st = os.stat(path)
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            # No file (yet): the in-memory config is all there is.
            self._stamp = None
            return self._fallback()

        if self._loaded and stamp == self._stamp:
            return self._value if self._value else self._fallback()

        try:
            data = json.loads(path.read_text())
            raw = data.get("cluster_secret") if isinstance(data, dict) else None
            self._value = str(raw) if raw else None
            self._stamp = stamp
            self._loaded = True
        except (OSError, ValueError):
            # Keep the last value that parsed; do not blank the fleet's key
            # because somebody saved a broken config.json.
            return self._value if self._value else self._fallback()

        return self._value if self._value else self._fallback()
