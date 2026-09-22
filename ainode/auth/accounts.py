"""User accounts and login sessions: the front door a PERSON walks through.

API-key auth (``auth/middleware.py``) protects the port, and it is the right
credential for a machine: the bench, the desktop app and ``curl`` all hold a key.
It is the wrong credential for a human. A key is a 32-character hex string pasted
into a browser and kept in ``localStorage``, so "who is on this node" has one
answer for everybody, revoking one person's access means rotating the key every
client holds, and the dashboard's first screen asks a person to paste a machine
secret (#261).

This module is the other half: named accounts with passwords, and sessions that
outlive a restart. It sits BESIDE the key store rather than replacing it. Both
files live under ``AINODE_HOME``, both are 0600, both are re-read when they
change, and a request may present either credential.

Three decisions are worth the words:

* **A session does not expire.** Jason's rule: once you log in, you are in until
  you log out. An idle timeout on a dashboard somebody keeps open on a second
  monitor all week is a login prompt in front of a node they never left, and a
  short expiry is what makes people paste the API key instead. Revocation is the
  control: ``ainode auth session revoke``, the dashboard's session list,
  disabling the account, or changing its password.
* **Only hashes are stored, in both directions.** A password is
  ``scrypt(password, salt)`` and a session token is stored as its SHA-256, so the
  file a backup or a support bundle picks up cannot log anybody in. A session
  token is returned exactly once, at login, the way an API key is.
* **Passwords are hashed with ``hashlib.scrypt``, from the standard library.**
  The AINode container is aiohttp plus pynvml plus Rich and nothing else
  (``CLAUDE.md``), so a login page must not be the reason the image grows an
  argon2 wheel. scrypt at n=2**14, r=8, p=1 costs about 16 MiB and tens of
  milliseconds per attempt, which is the point: it is the per-guess price an
  attacker pays for a stolen ``users.json``.

Replication is deliberately out of this module's hands. ``export_users`` and
``import_users`` move the user list between nodes as whole records, hashes
included, and the fleet endpoints in ``auth/session_routes.py`` are the only
callers. **Sessions are never replicated**: a session belongs to one node's
cookie jar, and copying one would let a revocation on the head silently not take
on a member.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from ainode.auth.fleet import FLEET_KEY_ID
from ainode.core.config import AINODE_HOME


logger = logging.getLogger(__name__)

#: Where the accounts live, beside ``auth.json``. Read through
#: :attr:`UsersStore.path` at call time, never captured in ``__init__``, so a
#: test (and ``tests/conftest.py::isolate_users_store``) can redirect it the same
#: way it redirects ``AUTH_FILE``.
USERS_FILE = AINODE_HOME / "users.json"

#: A name is lowercase, and short enough to fit a table. It ends up in
#: ``api_key_id`` as ``user:<name>``, in log lines and in the rate limiter's
#: bucket keys, so the character set is the conservative one rather than "any
#: unicode": a name that renders two ways is a name two people read as one.
NAME_PATTERN = re.compile(r"^[a-z0-9._-]{1,64}$")
NAME_RULE = ("A name is 1 to 64 characters of lowercase letters, digits, dot, "
             "underscore or hyphen.")

#: The shortest password this node will store. Low on purpose: the throttle on
#: ``POST /api/auth/login`` is what makes guessing expensive, and a rule long
#: enough to argue with is a rule an operator works around with "ainode1".
MIN_PASSWORD_LENGTH = 8
PASSWORD_RULE = f"A password is at least {MIN_PASSWORD_LENGTH} characters."

#: The two roles. ``admin`` may manage users, keys and the auth switch; ``member``
#: may use the node. Anything beyond these two is #261's explicit non-goal.
ROLE_ADMIN = "admin"
ROLE_MEMBER = "member"
ROLES = (ROLE_ADMIN, ROLE_MEMBER)

# -- scrypt parameters. They are constants rather than per-record fields because
# every hash in the file is written by this release or a later one, and a record
# that carried its own cost parameters would let a downgrade pick the cheap ones.
SCRYPT_N = 2 ** 14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
SALT_BYTES = 16

#: Bytes of entropy in a session token, before base64. 32 bytes is 256 bits, and
#: the token is the whole credential, so it is sized like one.
SESSION_TOKEN_BYTES = 32

#: Sessions kept per user before the oldest is evicted. A person accumulates one
#: per browser, per phone and per CLI, and an unbounded list is a file that grows
#: forever on a node somebody logs into from a new place every day.
MAX_SESSIONS_PER_USER = 20

#: How stale ``last_seen`` is allowed to get before a request rewrites it. Every
#: request would mean a 0600 file rewrite per request; never would make the
#: session list useless for "is this still somebody's laptop".
LAST_SEEN_REFRESH_SECONDS = 60

#: Prefix on ``request["api_key_id"]`` for a cookie-authenticated request, so
#: every reader of that field (the rate limiter, the request log) can tell a
#: person from a key without a second lookup.
USER_KEY_PREFIX = "user:"

#: The salt an unknown or disabled name is hashed against, so a login attempt for
#: a name that does not exist costs the same scrypt work as one for a name that
#: does. Without it, response time is a user-enumeration oracle in front of a
#: login page that otherwise answers one identical message for both.
_DUMMY_SALT = b"ainode-no-such-user"


def _now() -> str:
    """UTC, to the second, in the same spelling ``auth.json`` uses."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_stamp(value) -> Optional[datetime]:
    """A stored timestamp as a datetime, or None when it cannot be read.

    None is treated by every caller as "older than any window", so a record
    written by something that spelled the time differently is refreshed rather
    than trusted.
    """
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _file_stamp(path) -> Optional[tuple]:
    """``(mtime_ns, size)`` for *path*, or None when it is not there.

    The same cheap identity ``AuthConfig`` and ``discovery/signing.py`` use to
    tell "this file changed" from "this file is the one I already read".
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def normalize_name(name) -> str:
    """A name in its stored form, or "" when it is not a usable name.

    Case-folded, because "Jason" and "jason" are one person and a node that
    accepted both would have two accounts nobody can tell apart in a table.
    """
    text = str(name or "").strip().casefold()
    return text if NAME_PATTERN.match(text) else ""


def hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
    """``(hash hex, salt hex)`` for *password*, minting a salt when none is given.

    Per-password salt, so two people who pick the same password do not share a
    hash and a precomputed table buys nothing.
    """
    if salt is None:
        salt = secrets.token_bytes(SALT_BYTES)
    digest = hashlib.scrypt(str(password).encode("utf-8"), salt=salt,
                            n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=SCRYPT_DKLEN)
    return digest.hex(), salt.hex()


def verify_hash(password: str, password_hash: str, salt_hex: str) -> bool:
    """Constant-time check of *password* against a stored hash and salt."""
    try:
        salt = bytes.fromhex(str(salt_hex or ""))
    except ValueError:
        return False
    if not salt or not password_hash:
        return False
    digest, _ = hash_password(password, salt=salt)
    return hmac.compare_digest(digest, str(password_hash))


def hash_token(token: str) -> str:
    """The stored form of a session token.

    SHA-256 and not scrypt on purpose: the token is 256 random bits this node
    minted, so there is no guessing to slow down, and the check is on the hot
    path of every cookie-authenticated request.
    """
    return hashlib.sha256(str(token).encode("utf-8")).hexdigest()


def public_user(record: dict) -> dict:
    """A user record as a caller may see it: never a hash and never a salt."""
    return {
        "name": str(record.get("name") or ""),
        "role": str(record.get("role") or ROLE_MEMBER),
        "created_at": str(record.get("created_at") or ""),
        "disabled": bool(record.get("disabled", False)),
    }


def public_session(record: dict) -> dict:
    """A session as a caller may see it: never the token hash.

    The hash is not the token, but it is the only field in the file that a leak
    turns into a lookup key, and nothing outside this module needs it.
    """
    return {
        "id": str(record.get("id") or ""),
        "user": str(record.get("user") or ""),
        "client": str(record.get("client") or ""),
        "agent": str(record.get("agent") or ""),
        "created_at": str(record.get("created_at") or ""),
        "last_seen": str(record.get("last_seen") or ""),
    }


def require_admin(request) -> bool:
    """May *request* administer this node: users, keys and the auth switch?

    Three callers qualify, and the second and third are why the first can exist
    at all:

    * a session whose account is ``role: "admin"``,
    * an operator API key, which is how the CLI on the box works and how the
      FIRST admin is created on a node that has no accounts yet,
    * the fleet key, so a head can replicate its user list to a member.

    Tolerant of a request-like object with no mapping interface (several tests
    call handlers with a stub), which reads as "not an admin": the safe
    direction.
    """
    getter = getattr(request, "get", None)
    if not callable(getter):
        return False
    if str(getter("user_role", "") or "") == ROLE_ADMIN:
        return True
    if str(getter("api_key_id", "") or "") == FLEET_KEY_ID:
        return True
    # An operator key authenticates with no account attached; a cookie
    # authenticates with one. That is the whole difference between "the machine
    # credential" and "a person", and it is why a member session is refused here
    # while a key is not.
    return bool(getter("authenticated", False)) and not getter("user", "")


class UsersStore:
    """The accounts and sessions on this node, backed by ``users.json``.

    Constructed once per process and put on the app as ``app["users_store"]``,
    beside ``app["auth_config"]``. ``load()`` is a classmethod for the same
    reason ``AuthConfig.load()`` is: every caller (the server, the CLI, the
    fleet endpoints) wants "the store this node has" and not a path.

    The file is written 0600 through a temp file in the same directory, and
    re-read when its stamp changes, so ``ainode auth user add`` on the box is
    live on the running server without a restart. Same trade, same reasons, as
    ``AuthConfig``.
    """

    def __init__(self, path=None) -> None:
        # None means "whatever USERS_FILE says now", so a monkeypatch of the
        # module constant reaches a store that already exists.
        self._path: Optional[Path] = Path(path) if path is not None else None
        self.users: list[dict] = []
        self.sessions: list[dict] = []
        self._stamp: Optional[tuple] = None

    @property
    def path(self) -> Path:
        return self._path if self._path is not None else USERS_FILE

    # -- persistence ---------------------------------------------------------

    @classmethod
    def load(cls, path=None) -> "UsersStore":
        """The store on disk, or an empty one when there is no file yet.

        A malformed file raises, exactly as ``AuthConfig.load`` does: a node
        whose account list cannot be read must fail loudly at boot rather than
        come up with no accounts and an operator who thinks there are some. The
        tolerant path is :meth:`reload_if_changed`.
        """
        store = cls(path)
        if store.path.exists():
            store._adopt(json.loads(store.path.read_text()))
            store._stamp = _file_stamp(store.path)
            store.tighten_file_mode()
        return store

    def _adopt(self, data) -> None:
        if not isinstance(data, dict):
            raise ValueError(f"{self.path} does not hold a JSON object")
        self.users = [dict(u) for u in (data.get("users") or []) if isinstance(u, dict)]
        self.sessions = [dict(s) for s in (data.get("sessions") or [])
                         if isinstance(s, dict)]

    def save(self) -> None:
        """Write the store 0600, through a temp file in the same directory.

        0600 because this file decides who may log in, and the server and the
        installer both run as root, where the default umask leaves it
        world-readable. Atomic because every other process re-reads it on change
        and must never see half a document.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"users": self.users, "sessions": self.sessions},
                                  indent=2))
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)
        self._stamp = _file_stamp(self.path)

    def tighten_file_mode(self) -> bool:
        """chmod the store to 0600 when it is wider. True when it changed."""
        try:
            mode = stat.S_IMODE(os.stat(self.path).st_mode)
        except OSError:
            return False
        if not mode & 0o077:
            return False
        try:
            os.chmod(self.path, 0o600)
        except OSError as exc:  # pragma: no cover - a read-only home
            logger.warning("could not chmod %s to 0600: %s", self.path, exc)
            return False
        logger.info("tightened %s from %s to 0600", self.path, oct(mode))
        return True

    def reload_if_changed(self) -> bool:
        """Adopt what ``users.json`` says now. True when this changed something.

        Called per request by the middleware, so ``ainode auth user add`` and a
        session revoked from another process take effect on the running server
        the way an ``ainode auth key revoke`` does. Deliberately conservative in
        both directions: a file that cannot be stat'ed or parsed leaves the state
        in memory alone, because a half-written document must never be the thing
        that logs everybody out, and must never be the thing that lets somebody
        in either.
        """
        stamp = _file_stamp(self.path)
        if stamp is None or stamp == self._stamp:
            return False
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            logger.warning("could not re-read %s; keeping the accounts in memory",
                           self.path)
            return False
        if not isinstance(data, dict):
            return False
        before = (self.users, self.sessions)
        try:
            self._adopt(data)
        except ValueError:
            return False
        self._stamp = stamp
        changed = before != (self.users, self.sessions)
        if changed:
            logger.info("users.json changed on disk: %d account(s), %d session(s)",
                        len(self.users), len(self.sessions))
        return changed

    # -- users ---------------------------------------------------------------

    def find_user(self, name) -> Optional[dict]:
        """The stored record for *name*, or None. Hashes included: internal."""
        wanted = normalize_name(name)
        if not wanted:
            return None
        for record in self.users:
            if str(record.get("name") or "") == wanted:
                return record
        return None

    def has_users(self) -> bool:
        return bool(self.users)

    def admin_count(self) -> int:
        """Enabled admins. A disabled admin cannot administer anything."""
        return sum(1 for u in self.users
                   if str(u.get("role") or "") == ROLE_ADMIN
                   and not u.get("disabled", False))

    def is_last_admin(self, name) -> bool:
        """Would losing *name* leave this node with nobody who can administer it?

        The one rule behind both the 409 on ``DELETE /api/auth/users/{name}`` and
        the 409 on disabling an account: an operator must not be able to lock
        every admin out of the node they are standing in front of. A node whose
        admins are ALREADY all disabled has nothing left to protect, so this is
        False there rather than a refusal nobody can clear.
        """
        record = self.find_user(name)
        if record is None:
            return False
        if str(record.get("role") or "") != ROLE_ADMIN or record.get("disabled", False):
            return False
        return self.admin_count() <= 1

    def list_users(self) -> list[dict]:
        """Every account as a caller may see it, with its live session count."""
        out = []
        for record in self.users:
            row = public_user(record)
            row["sessions"] = len(self.sessions_for(row["name"]))
            out.append(row)
        return out

    def add_user(self, name, password, role: str = ROLE_MEMBER) -> dict:
        """Add an account and persist. Returns the record without hashes.

        Raises ``ValueError`` with the rule it broke on a bad name, a short
        password, an unknown role or a duplicate, so one message can go straight
        into a 400 and into the CLI's stderr.
        """
        wanted = normalize_name(name)
        if not wanted:
            raise ValueError(NAME_RULE)
        if len(str(password or "")) < MIN_PASSWORD_LENGTH:
            raise ValueError(PASSWORD_RULE)
        role = str(role or ROLE_MEMBER).strip().casefold()
        if role not in ROLES:
            raise ValueError(f"Role must be one of: {', '.join(ROLES)}.")
        if self.find_user(wanted) is not None:
            raise ValueError(f"There is already an account named '{wanted}'.")
        password_hash, salt = hash_password(password)
        record = {
            "name": wanted,
            "password_hash": password_hash,
            "salt": salt,
            "role": role,
            "created_at": _now(),
            "disabled": False,
        }
        self.users.append(record)
        self.save()
        logger.info("account added: %s (%s)", wanted, role)
        return public_user(record)

    def remove_user(self, name) -> bool:
        """Remove an account and every session it holds. False when unknown.

        Raises ``ValueError`` rather than stranding the node when *name* is the
        last enabled admin (see :meth:`is_last_admin`). The rule lives here and
        not in the route so the CLI cannot get around it.
        """
        record = self.find_user(name)
        if record is None:
            return False
        if self.is_last_admin(record["name"]):
            raise ValueError(
                "This is the only admin on this node. Add another admin before "
                "removing this one.")
        wanted = record["name"]
        self.users = [u for u in self.users if u is not record]
        self.sessions = [s for s in self.sessions
                         if str(s.get("user") or "") != wanted]
        self.save()
        logger.info("account removed: %s", wanted)
        return True

    def set_password(self, name, password) -> bool:
        """Set a new password and revoke every session. False when unknown.

        The revocation is the point: a password is changed because it leaked or
        because somebody is being locked out, and a change that leaves the old
        sessions logged in does neither.
        """
        record = self.find_user(name)
        if record is None:
            return False
        if len(str(password or "")) < MIN_PASSWORD_LENGTH:
            raise ValueError(PASSWORD_RULE)
        record["password_hash"], record["salt"] = hash_password(password)
        self.sessions = [s for s in self.sessions
                         if str(s.get("user") or "") != record["name"]]
        self.save()
        logger.info("password changed for %s; its sessions were revoked",
                    record["name"])
        return True

    def set_disabled(self, name, disabled: bool) -> bool:
        """Disable or enable an account. False when unknown.

        Disabling revokes its sessions too: an account that can still act through
        a cookie is not disabled. Raises ``ValueError`` when it would leave the
        node with no enabled admin.
        """
        record = self.find_user(name)
        if record is None:
            return False
        if disabled and self.is_last_admin(record["name"]):
            raise ValueError(
                "This is the only admin on this node. Add another admin before "
                "disabling this one.")
        record["disabled"] = bool(disabled)
        if disabled:
            self.sessions = [s for s in self.sessions
                             if str(s.get("user") or "") != record["name"]]
        self.save()
        logger.info("account %s: %s", record["name"],
                    "disabled" if disabled else "enabled")
        return True

    def role_of(self, name) -> str:
        """The role of an ENABLED account, or "" for unknown and disabled.

        "" is what the middleware stamps as ``request["user_role"]``, so a
        disabled account reads as no role at all rather than as a member.
        """
        record = self.find_user(name)
        if record is None or record.get("disabled", False):
            return ""
        return str(record.get("role") or ROLE_MEMBER)

    def verify_password(self, name, password) -> bool:
        """Is this the password for this account? False for unknown and disabled.

        Constant-time in two senses: the comparison is ``compare_digest``, and an
        unknown or disabled name pays the same scrypt cost as a real one, so
        response time does not say which names exist.
        """
        record = self.find_user(name)
        if record is None or record.get("disabled", False):
            hash_password(str(password or ""), salt=_DUMMY_SALT)
            return False
        return verify_hash(str(password or ""),
                           str(record.get("password_hash") or ""),
                           str(record.get("salt") or ""))

    # -- sessions ------------------------------------------------------------

    def sessions_for(self, name) -> list[dict]:
        """This account's sessions, oldest first, without the token hash."""
        wanted = normalize_name(name)
        return [public_session(s) for s in self.sessions
                if str(s.get("user") or "") == wanted]

    def create_session(self, name, client: str = "dashboard",
                       agent: str = "") -> tuple[str, dict]:
        """Mint a session for *name*. Returns ``(token, session)``.

        The token is returned HERE and nowhere else, ever: only its SHA-256 is
        stored, the same shape an API key has. Raises ``ValueError`` for an
        unknown or disabled account, so nothing can mint a session for one.
        """
        record = self.find_user(name)
        if record is None or record.get("disabled", False):
            raise ValueError("No such account on this node.")
        token = secrets.token_urlsafe(SESSION_TOKEN_BYTES)
        now = _now()
        session = {
            "id": secrets.token_hex(8),
            "token_hash": hash_token(token),
            "user": record["name"],
            "created_at": now,
            "last_seen": now,
            "client": str(client or "dashboard"),
            "agent": str(agent or "")[:80],
        }
        self.sessions.append(session)
        self._evict_over_cap(record["name"])
        self.save()
        logger.info("session %s opened for %s (%s)", session["id"],
                    session["user"], session["client"])
        return token, session

    def _evict_over_cap(self, name: str) -> None:
        """Keep at most MAX_SESSIONS_PER_USER for *name*, dropping the oldest.

        Oldest by position, which is creation order: the list is only ever
        appended to. A timestamp comparison would tie at one-second resolution,
        and twenty logins inside one second is exactly the case a cap is for.
        """
        mine = [s for s in self.sessions if str(s.get("user") or "") == name]
        if len(mine) <= MAX_SESSIONS_PER_USER:
            return
        evicted = {id(s) for s in mine[: len(mine) - MAX_SESSIONS_PER_USER]}
        self.sessions = [s for s in self.sessions if id(s) not in evicted]
        logger.info("%s is over %d sessions: dropped the oldest %d",
                    name, MAX_SESSIONS_PER_USER, len(evicted))

    def session_for_token(self, token, touch: bool = True) -> Optional[dict]:
        """The live session *token* names, or None.

        None for a token nobody holds, and also for one whose account has been
        removed or disabled since it was minted: disabling an account is the
        operator's kill switch, and a switch a cookie outlives is not one.

        Touches ``last_seen`` at most once every
        ``LAST_SEEN_REFRESH_SECONDS``, because this runs on every
        cookie-authenticated request and a rewrite per request would put a 0600
        file write on the hot path. ``touch=False`` is for the one caller that
        asks whether a session exists on a request it is about to REFUSE (the
        middleware's CSRF probe): a refused request is not activity.
        """
        if not token:
            return None
        wanted = hash_token(token)
        for session in self.sessions:
            stored = str(session.get("token_hash") or "")
            if not stored or not hmac.compare_digest(wanted, stored):
                continue
            if not self.role_of(session.get("user")):
                return None
            if touch:
                self._touch(session)
            return session
        return None

    def _touch(self, session: dict) -> None:
        seen = _parse_stamp(session.get("last_seen"))
        if seen is not None and datetime.now(timezone.utc) - seen < timedelta(
                seconds=LAST_SEEN_REFRESH_SECONDS):
            return
        session["last_seen"] = _now()
        try:
            self.save()
        except OSError as exc:  # pragma: no cover - a read-only home
            # A node that cannot write must not stop answering requests for the
            # person already logged into it. The stamp goes stale; nothing else.
            logger.warning("could not refresh last_seen in %s: %s", self.path, exc)

    def revoke_session(self, session_id, user=None) -> bool:
        """Revoke one session by id. False when there is nothing to revoke.

        *user* scopes the revocation to one account, which is how a member may
        end its own sessions and only its own; None means any session, which is
        what an admin gets.
        """
        wanted = str(session_id or "")
        if not wanted:
            return False
        owner = normalize_name(user) if user else ""
        keep, gone = [], []
        for session in self.sessions:
            if str(session.get("id") or "") == wanted and (
                    not owner or str(session.get("user") or "") == owner):
                gone.append(session)
            else:
                keep.append(session)
        if not gone:
            return False
        self.sessions = keep
        self.save()
        logger.info("session %s revoked", wanted)
        return True

    def revoke_sessions_for(self, name) -> int:
        """Revoke every session an account holds. Returns how many went."""
        wanted = normalize_name(name)
        if not wanted:
            return 0
        before = len(self.sessions)
        self.sessions = [s for s in self.sessions
                         if str(s.get("user") or "") != wanted]
        gone = before - len(self.sessions)
        if gone:
            self.save()
        return gone

    # -- fleet replication ---------------------------------------------------

    def export_users(self) -> list[dict]:
        """The accounts as another NODE needs them: hashes and salts included.

        This is the one method that hands out hash material, and it is why the
        export route is fleet-key only. A member node receiving this list can
        verify a password without the head, which is the whole point: one login
        works on every node in the cluster.

        Sorted by name so :meth:`export_stamp` over the same accounts is the
        same string on every node.
        """
        return [
            {
                "name": str(u.get("name") or ""),
                "password_hash": str(u.get("password_hash") or ""),
                "salt": str(u.get("salt") or ""),
                "role": str(u.get("role") or ROLE_MEMBER),
                "created_at": str(u.get("created_at") or ""),
                "disabled": bool(u.get("disabled", False)),
            }
            for u in sorted(self.users, key=lambda u: str(u.get("name") or ""))
        ]

    def export_stamp(self) -> str:
        """A short digest of the exported accounts.

        A content hash rather than the file's mtime: two nodes holding the same
        accounts must produce the same stamp, which is what lets a replication
        pass decide "nothing to do" without shipping the list. An mtime says
        only when a file was written, which differs on every node.
        """
        payload = json.dumps(self.export_users(), sort_keys=True,
                            separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def import_users(self, users) -> bool:
        """Replace the account list with *users*. True when anything changed.

        All or nothing: a malformed record raises ``ValueError`` and the store is
        untouched, because a partial import is a node whose account list quietly
        differs from the rest of the cluster.

        Sessions survive when their account does. A member's cookie is its own
        node's, so a replication pass must not be a fleet-wide logout; a session
        whose account the head removed goes with it.
        """
        if not isinstance(users, list):
            raise ValueError("'users' must be a list of account records.")
        adopted: list[dict] = []
        seen: set[str] = set()
        for entry in users:
            if not isinstance(entry, dict):
                raise ValueError("Every account record must be a JSON object.")
            name = normalize_name(entry.get("name"))
            if not name:
                raise ValueError(f"{NAME_RULE} Got: {entry.get('name')!r}")
            if name in seen:
                raise ValueError(f"'{name}' appears twice in the account list.")
            password_hash = str(entry.get("password_hash") or "")
            salt = str(entry.get("salt") or "")
            if not password_hash or not salt:
                raise ValueError(f"'{name}' carries no password hash to import.")
            role = str(entry.get("role") or ROLE_MEMBER).strip().casefold()
            if role not in ROLES:
                raise ValueError(f"'{name}' has an unknown role {role!r}.")
            seen.add(name)
            adopted.append({
                "name": name,
                "password_hash": password_hash,
                "salt": salt,
                "role": role,
                "created_at": str(entry.get("created_at") or _now()),
                "disabled": bool(entry.get("disabled", False)),
            })
        if adopted == self.export_users():
            return False
        if not adopted and self.users:
            # Legal, and worth a line in the log: the only way to reach it is a
            # head that really has no accounts, which on a configured fleet is a
            # mistake somebody will want to find afterwards.
            logger.warning("replication is emptying the account list on this node")
        self.users = adopted
        names = {u["name"] for u in adopted}
        self.sessions = [s for s in self.sessions
                         if str(s.get("user") or "") in names]
        self.save()
        logger.info("accounts replicated: %d on this node now", len(self.users))
        return True
