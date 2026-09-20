"""Joining a node to a cluster: the token a master mints, the keys a joiner writes.

Joining used to mean hand-editing ``config.json`` on the new box for
``cluster_id``, ``cluster_role`` and ``cluster_interface``, and, once discovery is
signed, for ``cluster_secret`` as well. Nothing in the product did it: the browser
wizard set a node name and a model and never touched a cluster key at all (#208).
This module is the whole join, and it is deliberately split so that neither side
has to trust the other with anything it does not need:

**The master mints a token** (``JoinTokenStore``). 32 random bytes, kept only as a
SHA-256 hash, with an expiry and a single use. The token is the joiner's one
credential: it is presented to ``POST /api/cluster/join``, which is the only
mutating-adjacent route exempt from the API-key rule, because a node that has not
joined yet cannot hold this cluster's key. So the token has to be the thing that
is short-lived, single-use and unguessable, and the route has to be rate limited
(both live here and in ``api/cluster_join.py``).

**The joiner writes exactly six keys** (``apply_join``). A ``NodeConfig.save()``
round trip writes every field of the dataclass, which on an older config.json
silently ADDS every key that release happens to default differently and DROPS
anything the dataclass does not know. Joining must not do that: it merges into the
raw JSON and leaves every other key byte-for-byte as it found it.

Two spellings worth knowing, because the config carries both:

* ``cluster_role`` is ``auto`` / ``master`` / ``worker`` (who may be elected head).
  A joiner takes ``worker``: it is joining somebody else's cluster and must never
  win the election. ``member`` is NOT a value this field accepts -- the
  ``PATCH /api/config`` validator rejects it -- so the joiner writes ``worker``.
* ``distributed_mode`` is ``solo`` / ``head`` / ``member`` (whether this node runs
  an engine of its own). A joiner takes ``member``, which is what
  ``ainode role worker`` and the installer's ``--job worker`` already write.

The pair is one decision, spelled the two ways the code spells it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# One home for the join-token shape, so the CLI, the route and the tests agree.
#: Bytes of entropy in a join token. 32 bytes is 64 hex characters.
TOKEN_BYTES = 32
#: Default lifetime of a minted token, in seconds.
DEFAULT_TTL_SECONDS = 30 * 60
#: Hard bounds on ``--ttl``: a zero-second token cannot be pasted anywhere, and a
#: token that outlives the session that minted it is a standing credential.
MIN_TTL_SECONDS = 30
MAX_TTL_SECONDS = 24 * 60 * 60
#: How long a spent or expired record is kept before it is pruned. Only the hash
#: is stored, so the record is not a credential; it is kept briefly so an
#: operator reading the file can see a token WAS used rather than only that it is
#: gone.
RETENTION_SECONDS = 24 * 60 * 60

#: The port a ``host`` with no port in it means.
DEFAULT_WEB_PORT = 3000

#: What a failed join is told, whatever the reason. A wrong token, an expired
#: token and a spent token answer with exactly this, because telling them apart
#: turns the route into an oracle for guessing.
REFUSED_MESSAGE = (
    "That join token is not valid. Tokens expire (30 minutes by default) and "
    "work once. Mint a fresh one on the master with: ainode cluster token"
)

#: What a version mismatch is told. The reason is not pedantry: a fleet running
#: two releases disagrees about the announcement wire format, which is how a
#: split fleet goes on looking healthy (#171).
VERSION_MISMATCH_TEMPLATE = (
    "This node runs AINode {joiner} and the master runs {master}. A cluster "
    "spanning two releases can disagree about the discovery wire, so the "
    "dashboard would show a split fleet with no cause visible. Update this node "
    "(ainode update) or the master so both match, or pass "
    "--allow-version-mismatch to join anyway."
)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def tokens_path() -> Path:
    """``<AINODE_HOME>/join-tokens.json``, resolved at call time.

    Resolved per call rather than at import, so a test that repoints
    ``ainode.core.config.AINODE_HOME`` at a tmp dir gets the tmp file, and so a
    process whose AINODE_HOME changes (the CLI inside the container versus on the
    host) never writes to a path it captured at import.
    """
    from ainode.core import config as core_config

    return Path(core_config.AINODE_HOME) / "join-tokens.json"


def config_path() -> Path:
    """``<AINODE_HOME>/config.json``, resolved at call time (see tokens_path)."""
    from ainode.core import config as core_config

    return Path(core_config.CONFIG_FILE)


# =============================================================================
# The token store
# =============================================================================

@dataclass
class MintedToken:
    """A freshly minted token: the only moment the plaintext exists."""

    token: str
    token_id: str
    expires_at: float

    @property
    def ttl_seconds(self) -> int:
        return max(0, int(round(self.expires_at - time.time())))


class JoinTokenStore:
    """Join tokens on disk, hashed, expiring and single use.

    The file is a list of records, each ``{"id", "token_hash", "created_at",
    "expires_at", "used_at", "used_by"}``. A malformed or unreadable file reads
    as EMPTY rather than raising: the failure mode of a corrupt token file must be
    "no token is accepted", never "the master cannot answer a join at all".
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else tokens_path()

    # -- persistence ----------------------------------------------------------

    def _load(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return []
        if not isinstance(data, dict):
            return []
        tokens = data.get("tokens")
        if not isinstance(tokens, list):
            return []
        return [t for t in tokens if isinstance(t, dict)]

    def _save(self, tokens: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"tokens": tokens}, indent=2)
        # Written 0600 through a temp file in the same directory: the records are
        # hashes, not tokens, but the file still says who may join this cluster.
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(payload)
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)

    # -- minting --------------------------------------------------------------

    def mint(self, ttl_seconds: int = DEFAULT_TTL_SECONDS,
             note: str = "") -> MintedToken:
        """Mint one token, store its hash, and return the plaintext once.

        The plaintext is never written anywhere. A caller that loses it mints
        another; there is no way to read one back.
        """
        ttl = int(ttl_seconds)
        if ttl < MIN_TTL_SECONDS or ttl > MAX_TTL_SECONDS:
            raise ValueError(
                f"ttl must be between {MIN_TTL_SECONDS} and {MAX_TTL_SECONDS} "
                f"seconds, got {ttl}"
            )
        token = secrets.token_hex(TOKEN_BYTES)
        now = time.time()
        record = {
            "id": secrets.token_hex(4),
            "token_hash": _hash_token(token),
            "created_at": now,
            "expires_at": now + ttl,
            "used_at": None,
            "used_by": None,
        }
        if note:
            record["note"] = str(note)[:120]
        tokens = self._prune(self._load(), now=now)
        tokens.append(record)
        self._save(tokens)
        return MintedToken(token=token, token_id=record["id"],
                           expires_at=record["expires_at"])

    # -- verifying ------------------------------------------------------------

    def _match(self, tokens: list[dict], token: str) -> Optional[dict]:
        """The record for this token, by constant-time hash compare."""
        if not token:
            return None
        wanted = _hash_token(token)
        for record in tokens:
            stored = record.get("token_hash") or ""
            if stored and hmac.compare_digest(wanted, stored):
                return record
        return None

    def verify(self, token: str, now: Optional[float] = None) -> Optional[dict]:
        """The record for a token that is known, unexpired and unused, else None.

        Read-only. ``consume`` is what a join calls.
        """
        moment = time.time() if now is None else now
        record = self._match(self._load(), token)
        if record is None:
            return None
        if record.get("used_at") is not None:
            return None
        if float(record.get("expires_at") or 0) <= moment:
            return None
        return record

    def consume(self, token: str, used_by: str = "",
                now: Optional[float] = None) -> Optional[dict]:
        """Spend a token: return its record and mark it used, or return None.

        The mark is persisted BEFORE the caller is told yes, so a second request
        with the same token loses even if the first one dies mid-join.
        """
        moment = time.time() if now is None else now
        tokens = self._load()
        record = self._match(tokens, token)
        if record is None:
            return None
        if record.get("used_at") is not None:
            return None
        if float(record.get("expires_at") or 0) <= moment:
            return None
        record["used_at"] = moment
        record["used_by"] = str(used_by)[:64]
        self._save(self._prune(tokens, now=moment))
        return dict(record)

    # -- housekeeping ---------------------------------------------------------

    def _prune(self, tokens: list[dict], now: float) -> list[dict]:
        """Drop records whose usefulness expired more than RETENTION ago."""
        kept = []
        for record in tokens:
            settled_at = record.get("used_at") or record.get("expires_at") or 0
            try:
                settled = float(settled_at)
            except (TypeError, ValueError):
                continue
            if now - settled <= RETENTION_SECONDS:
                kept.append(record)
        return kept

    def live(self, now: Optional[float] = None) -> list[dict]:
        """Records that would still be accepted, newest first."""
        moment = time.time() if now is None else now
        live = [
            r for r in self._load()
            if r.get("used_at") is None and float(r.get("expires_at") or 0) > moment
        ]
        return sorted(live, key=lambda r: float(r.get("created_at") or 0), reverse=True)


# =============================================================================
# The cluster secret the token hands over
# =============================================================================

def generate_cluster_secret() -> str:
    """A fresh shared secret for a cluster: 32 bytes, hex."""
    return secrets.token_hex(32)


def ensure_cluster_secret(config, path: Optional[Path] = None) -> tuple[str, bool]:
    """Return this node's ``cluster_secret``, generating one if it has none.

    Returns ``(secret, generated)``. A generated secret is written straight into
    ``config.json`` through the same raw-JSON merge ``apply_join`` uses, and set
    on the live ``config`` object, so the value the caller hands to a joiner is
    the value this node will sign with.

    Generating one is not free on an existing cluster: once discovery is signed, a
    node that HAS a secret drops announcements it cannot verify, so a master that
    mints one goes dark to every peer that does not have the same value. That is
    why this is called from ``ainode cluster token`` and from a fresh install, and
    why both of them say so out loud.
    """
    existing = (getattr(config, "cluster_secret", None) or "").strip()
    if existing:
        return existing, False
    secret = generate_cluster_secret()
    merge_config_keys({"cluster_secret": secret}, path=path)
    try:
        config.cluster_secret = secret
    except AttributeError:  # pragma: no cover - a stub config in a test
        pass
    return secret, True


# =============================================================================
# The joiner's config write
# =============================================================================

def merge_config_keys(updates: dict, path: Optional[Path] = None) -> dict:
    """Write ``updates`` into config.json and touch nothing else.

    Deliberately NOT ``NodeConfig.load(); setattr; save()``: that round trip
    rewrites the file from the dataclass, so it adds every key this release
    defaults differently from the file on disk and drops every key the dataclass
    does not declare. A join writes six keys; six keys is what must change.

    Returns the merged document.
    """
    target = Path(path) if path is not None else config_path()
    document: dict = {}
    if target.exists():
        try:
            loaded = json.loads(target.read_text())
        except ValueError as exc:
            raise ValueError(f"{target} is not valid JSON: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"{target} does not hold a JSON object")
        document = loaded
    document.update(updates)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(document, indent=2))
    tmp.replace(target)
    return document


def join_updates(payload: dict, node_name: str = "",
                 interface: str = "") -> dict:
    """The exact config keys a join writes, from a master's join answer.

    ``node_name`` and ``interface`` are written only when the caller supplied
    them: an empty ``cluster_interface`` means autodetect, and overwriting a
    pinned NIC name with the empty string is not something joining should do.
    """
    updates: dict = {
        "cluster_id": str(payload.get("cluster_id") or "default"),
        # See the module docstring: "worker" is the cluster_role spelling of
        # member, and distributed_mode is where "member" itself lives.
        "cluster_role": "worker",
        "distributed_mode": "member",
    }
    # Every value below is written only when the master actually sent one. An
    # empty answer must never CLEAR a value the node already had: joining is the
    # one operation here that is allowed to touch these keys, and it earns that by
    # only ever replacing them with something.
    address = payload.get("master_address")
    if address:
        updates["master_address"] = str(address)
    secret = payload.get("cluster_secret")
    if secret:
        updates["cluster_secret"] = str(secret)
    port = payload.get("discovery_port")
    try:
        if port:
            updates["discovery_port"] = int(port)
    except (TypeError, ValueError):
        pass
    if node_name:
        updates["node_name"] = str(node_name)[:64]
    if interface:
        updates["cluster_interface"] = str(interface)[:32]
    return updates


def apply_join(payload: dict, node_name: str = "", interface: str = "",
               path: Optional[Path] = None) -> dict:
    """Write a master's join answer into config.json. Returns the keys written."""
    updates = join_updates(payload, node_name=node_name, interface=interface)
    merge_config_keys(updates, path=path)
    return updates


# =============================================================================
# Addresses and versions
# =============================================================================

def parse_host_port(target: str, default_port: int = DEFAULT_WEB_PORT) -> tuple[str, int]:
    """Split ``host``, ``host:port`` or a full URL into ``(host, port)``.

    Accepts what an operator actually pastes: a bare IP, ``ip:3000``, a
    ``http://ip:3000`` copied out of a browser, a trailing slash, and a bracketed
    IPv6 literal.
    """
    value = (target or "").strip()
    if not value:
        raise ValueError("no master address given")
    for prefix in ("http://", "https://"):
        if value.lower().startswith(prefix):
            value = value[len(prefix):]
            break
    value = value.split("/", 1)[0].strip()
    if not value:
        raise ValueError("no master address given")
    if value.startswith("["):  # [::1]:3000
        close = value.find("]")
        if close == -1:
            raise ValueError(f"unbalanced brackets in {target!r}")
        host = value[1:close]
        rest = value[close + 1:]
        port_text = rest[1:] if rest.startswith(":") else ""
    elif value.count(":") > 1:  # a bare IPv6 literal, no port
        host, port_text = value, ""
    elif ":" in value:
        host, port_text = value.split(":", 1)
    else:
        host, port_text = value, ""
    host = host.strip()
    if not host:
        raise ValueError(f"no host in {target!r}")
    if not port_text:
        return host, int(default_port)
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"{port_text!r} is not a port number") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"port {port} is out of range")
    return host, port


def join_url(target: str, default_port: int = DEFAULT_WEB_PORT) -> str:
    """The join endpoint on a master, from whatever the operator typed."""
    host, port = parse_host_port(target, default_port=default_port)
    bracketed = f"[{host}]" if ":" in host else host
    return f"http://{bracketed}:{port}/api/cluster/join"


def version_refusal(joiner_version: str, master_version: str,
                    allow_mismatch: bool = False) -> Optional[str]:
    """Why this join must not proceed on version grounds, or None.

    A master that announces no version at all (a release older than the one that
    put it on the wire) is not a refusal: there is nothing to compare, and
    refusing would make this node unable to join the fleet it belongs to.
    """
    if allow_mismatch:
        return None
    master = (master_version or "").strip()
    joiner = (joiner_version or "").strip()
    if not master or not joiner:
        return None
    if master == joiner:
        return None
    return VERSION_MISMATCH_TEMPLATE.format(joiner=joiner, master=master)


def join_command(master_address: str, token: str) -> str:
    """The line an operator pastes on the joining node."""
    return f"ainode join {master_address} {token}"
