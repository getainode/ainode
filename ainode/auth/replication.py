"""Accounts come from the master. Sessions never leave the node they were made on.

The dashboard login (#261) puts a second credential on a node: a user account with
a password, in ``<AINODE_HOME>/users.json``. A cluster cannot have one of those
per node. An operator who adds an account on the master and then opens the
dashboard of whichever node a browser bookmark happens to point at would be told
their password is wrong, and the fix ("log in on a different node") is not a fix.

So there is ONE authority for accounts, the cluster's master, and this module is
the only thing that moves them:

* **The master pushes.** Every mutation the account routes make calls
  ``app["users_changed"]()``, which is registered here. It POSTs the master's
  ``export_users()`` (hashes included, which is why the route takes the fleet key
  and nothing else) to ``/api/auth/users/sync`` on every peer discovery knows
  about, in the background. A peer that did not take the push is retried on the
  next change and on a :data:`RETRY_INTERVAL_SECONDS` timer for as long as it is
  behind, so an account added while a node was rebooting lands when it comes back.
* **A worker pulls**, once at startup and every :data:`PULL_INTERVAL_SECONDS`
  after that, because a push cannot reach a node that was down when it happened
  and because a node that has just joined has no accounts at all.
* **Sessions are NOT replicated, ever.** A session is a cookie one browser holds
  against one node; copying them around would hand every node in the fleet a
  credential it never issued, and revoking a session would have to be a fan-out
  to be true. ``ainode auth session ...`` is therefore a per-node command, and
  that is the honest shape rather than a limitation.
* **A node with no ``cluster_secret``, or with nobody to talk to, does nothing.**
  The fleet key is derived from that secret (``ainode/auth/fleet.py``), so a node
  without one cannot authenticate to a peer and must not pretend to: it keeps the
  accounts it has and ``ainode doctor`` says why.

Both halves share the two questions the CLI also has to answer, so they live here
as plain functions rather than in the loop: which peers are there
(:func:`peer_targets`), and is this node the authority (:func:`local_role`).
``ainode auth user ...`` writes the store on the box and then calls
:func:`replicate_from_cli`, which is the same push over the stdlib, because the
CLI runs in a different process from the server (the installer's wrapper is
``docker exec``) and has no event loop to borrow.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from ainode.auth.fleet import cluster_secret_of, fleet_headers, fleet_key_headers

logger = logging.getLogger(__name__)

#: The two routes this module speaks, both fleet-key only (they carry hashes).
EXPORT_PATH = "/api/auth/users/export"
SYNC_PATH = "/api/auth/users/sync"
#: How the CLI asks a running node who the master is and who the peers are.
CLUSTER_INFO_PATH = "/api/cluster/info"

#: A worker re-reads the master's list this often. Five minutes because the pull
#: is the SAFETY NET under the master's push, not the mechanism: a change the
#: operator makes is on every reachable peer within a second of making it.
PULL_INTERVAL_SECONDS = 300.0
#: How often the loop wakes to retry a peer that is behind. Also the master's
#: idle tick, which costs one in-memory stamp comparison when nothing is behind.
RETRY_INTERVAL_SECONDS = 60.0
#: One node-to-node request's budget. Short: a peer that is down must not hold
#: the loop, and the next tick is 60 seconds away.
REQUEST_TIMEOUT_SECONDS = 10.0

#: What a worker remembers about its last successful pull, under AINODE_HOME. It
#: holds the MASTER's own stamp for the list it imported, so ``ainode doctor`` can
#: ask the master for its current stamp and compare two values of the same kind.
#: No hashes and no passwords go in here, so it is not a credential file.
SYNC_STATE_NAME = "users-sync.json"

#: The fallback minimum password length, used only when the account store does not
#: publish one of its own. The store is the authority and refuses a short password
#: itself; the CLI asks for the number so it can say no at the prompt instead of
#: after the operator has typed the same password twice.
DEFAULT_MIN_PASSWORD = 8


# ---------------------------------------------------------------------------
# The account store, which lands on its own branch
# ---------------------------------------------------------------------------

def users_store_class():
    """``UsersStore`` from :mod:`ainode.auth.accounts`, or ``None``.

    The import is guarded because this module, the CLI and the doctor are written
    against the store's contract and must import cleanly on a tree that does not
    carry it yet. Returning None (rather than raising) is also what lets a test
    put its own store in place by replacing this one function.
    """
    try:
        from ainode.auth.accounts import UsersStore
    except Exception:  # pragma: no cover - only on a tree without the store
        return None
    return UsersStore


def min_password_length() -> int:
    """The store's minimum password length, or :data:`DEFAULT_MIN_PASSWORD`.

    Asked of the store rather than restated here, so the CLI's refusal and the
    store's refusal cannot drift apart. The class first, then the module, because
    the contract fixes the number (8) and not where it is spelled.
    """
    cls = users_store_class()
    holders: list = [cls]
    try:
        from ainode.auth import accounts

        holders.append(accounts)
    except Exception:  # pragma: no cover - only on a tree without the store
        pass
    for holder in holders:
        value = getattr(holder, "MIN_PASSWORD_LENGTH", None)
        if isinstance(value, int) and value > 0:
            return value
    return DEFAULT_MIN_PASSWORD


def open_store(app=None):
    """The node's account store, or ``None`` when this release has none.

    Prefers the one already on the application, so the CLI-free path (the server)
    and this module cannot end up holding two objects over one file. Off an app
    there is nothing to share, so a fresh one is opened.

    ``UsersStore.load()`` is modelled on ``AuthConfig.load()``, which is a
    classmethod returning a new instance, so the result is kept when it is one and
    ignored when ``load()`` is an instance method that reads in place.
    """
    if app is not None:
        existing = None
        getter = getattr(app, "get", None)
        if callable(getter):
            existing = getter("users_store")
        if existing is not None:
            return existing
    cls = users_store_class()
    if cls is None:
        return None
    try:
        store = cls()
        loaded = store.load()
        if isinstance(loaded, cls):
            store = loaded
    except Exception:
        logger.exception("could not open the account store")
        return None
    if app is not None and callable(getattr(app, "get", None)):
        try:
            app["users_store"] = store
        except Exception:  # pragma: no cover - a mapping that refuses writes
            pass
    return store


# ---------------------------------------------------------------------------
# Pure helpers: the stamp, the addresses, the role
# ---------------------------------------------------------------------------

def users_stamp(users: Any) -> str:
    """A content id for an exported user list, so "behind" is answerable.

    The master compares this against what each peer last accepted to decide who
    still needs the push, which is what makes the retry a comparison rather than
    a queue: a peer that failed simply still disagrees. It is a hash of the list,
    so it changes on a password change (the hashes are salted) and not on a
    re-export of the same accounts.
    """
    try:
        blob = json.dumps(users, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        blob = repr(users)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def node_base_url(host: str, port: int) -> str:
    """``http://host:port`` for one node, or "" with no usable host."""
    host = str(host or "").strip()
    if not host:
        return ""
    bracketed = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{bracketed}:{int(port)}"


def peer_address(node) -> str:
    """The address to reach a peer on.

    One derivation for the whole product (``api/server_routes.peer_host``: the UDP
    source address, then the fabric IP, then the name), imported here rather than
    at module scope so this module stays importable from the CLI without dragging
    the API package in, and so a test can replace this seam with a loopback
    address the real one deliberately refuses to publish.
    """
    from ainode.api.server_routes import peer_host

    return peer_host(node)


def _local_node_ids(app, config) -> set:
    """Every id that means THIS node, so it is never pushed to as a peer.

    Both spellings, because a node whose ``config.json`` has no ``node_id`` yet
    (one that has never run ``ainode start``) announces itself as ``"unknown"``
    and would otherwise appear in its own peer list.
    """
    ids = {getattr(config, "node_id", None)}
    getter = getattr(app, "get", None) if app is not None else None
    if callable(getter):
        ids.add(getattr(getter("announcement"), "node_id", None))
    ids.discard(None)
    return ids


def peer_targets(app) -> list[tuple[str, str]]:
    """``(label, base url)`` for every other node of this cluster, from discovery.

    Offline members are left out: this is a list of nodes to push to now, not a
    roster. A member with no usable address is left out too, and named in the
    debug log rather than pushed to a guess.
    """
    cluster = None
    getter = getattr(app, "get", None)
    if callable(getter):
        cluster = getter("cluster_state")
    if cluster is None:
        return []
    config = getter("config") if callable(getter) else None
    local_ids = _local_node_ids(app, config)
    try:
        from ainode.discovery.broadcast import NodeStatus

        members = [n for n in cluster.members() if n.status != NodeStatus.OFFLINE]
    except Exception:
        logger.exception("could not read this cluster's members")
        return []

    targets: list[tuple[str, str]] = []
    for node in members:
        node_id = getattr(node, "node_id", None)
        if node_id in local_ids:
            continue
        port = int(getattr(node, "web_port", 3000) or 3000)
        base = node_base_url(peer_address(node), port)
        if not base:
            logger.debug("no usable address for %s, so accounts are not pushed to it",
                         node_id)
            continue
        targets.append((str(getattr(node, "node_name", "") or node_id or "?"), base))
    return targets


def master_target(app=None, config=None, info: Optional[dict] = None) -> str:
    """The master's base URL from this node's point of view, or "".

    Discovery first (the elected master is the live answer), then the
    ``master_address`` a join wrote into config.json, which is the whole answer on
    a node whose discovery has not heard anybody yet.
    """
    getter = getattr(app, "get", None) if app is not None else None
    if config is None and callable(getter):
        config = getter("config")
    local_ids = _local_node_ids(app, config)

    cluster = getter("cluster_state") if callable(getter) else None
    if cluster is not None:
        try:
            master = cluster.get_master()
        except Exception:  # pragma: no cover - defensive
            master = None
        if master is not None and getattr(master, "node_id", None) not in local_ids:
            base = node_base_url(peer_address(master),
                                 int(getattr(master, "web_port", 3000) or 3000))
            if base:
                return base

    if isinstance(info, dict):
        for node in info.get("members") or []:
            if not isinstance(node, dict):
                continue
            if node.get("effective_role") != "master":
                continue
            if node.get("node_id") in local_ids:
                continue
            host = str(node.get("host") or node.get("node_name") or "").strip()
            base = node_base_url(host, int(node.get("web_port") or 3000))
            if base:
                return base
        address = str(info.get("master_address") or "").strip()
        if address:
            return _base_from_address(address, config)

    address = str(getattr(config, "master_address", "") or "").strip()
    if address:
        return _base_from_address(address, config)
    return ""


def _base_from_address(address: str, config=None) -> str:
    """``http://host:port`` from whatever ``master_address`` holds."""
    from ainode.cluster.join import parse_host_port

    default_port = int(getattr(config, "web_port", 3000) or 3000)
    try:
        host, port = parse_host_port(address, default_port=default_port)
    except ValueError:
        return ""
    return node_base_url(host, port)


def local_role(config, info: Optional[dict] = None, cluster=None) -> str:
    """Is this node the account authority: ``"master"``, ``"worker"`` or ``"solo"``.

    ``"solo"`` is a node with nobody to disagree with: it is its own authority, so
    an account added there is simply the node's own and no warning is owed. The
    three sources are asked in order of how much they know: the running
    ``ClusterState`` (what the server has), a ``GET /api/cluster/info`` payload
    (what the CLI has while the service is up), then config.json alone (all a CLI
    on a stopped node has).

    A configured role WINS over an election in one direction only: an operator who
    wrote ``cluster_role: "worker"`` has said this node is not the authority, and
    a node that cannot see its master must not decide it has been promoted.
    """
    configured = str(getattr(config, "cluster_role", "") or "auto").strip().lower()
    mode = str(getattr(config, "distributed_mode", "") or "").strip().lower()
    if configured == "worker" or mode == "member":
        return "worker"

    local_id = getattr(config, "node_id", None)
    if cluster is not None:
        try:
            master = cluster.get_master()
            peers = len([n for n in cluster.members()
                         if getattr(n, "node_id", None) != local_id])
            # An election that HAS an answer is the authority, even against a
            # config that pinned this node master: two nodes both configured
            # master must not both push, and the election already picks one.
            if master is not None:
                if getattr(master, "node_id", None) == local_id:
                    return "master" if peers else "solo"
                return "worker"
        except Exception:  # pragma: no cover - defensive
            logger.exception("could not read the cluster's master")

    if isinstance(info, dict):
        members = [m for m in (info.get("members") or []) if isinstance(m, dict)]
        info_id = info.get("my_node_id") or local_id
        peers = len([m for m in members if m.get("node_id") != info_id])
        role = str(info.get("my_role") or "").strip().lower()
        if role == "master":
            return "master" if peers else "solo"
        if role:
            return role

    # Config.json alone, which is all a CLI on a stopped node has.
    peer_ips = [p for p in (getattr(config, "peer_ips", []) or []) if p]
    if configured == "master":
        return "master" if peer_ips else "solo"
    if str(getattr(config, "master_address", "") or "").strip():
        return "worker"
    if peer_ips:
        return "master"
    return "solo"


# ---------------------------------------------------------------------------
# What a worker remembers about its last pull
# ---------------------------------------------------------------------------

def ainode_home(home=None) -> Path:
    """AINODE_HOME, resolved at call time so a test's redirect is honoured."""
    if home is not None:
        return Path(home)
    from ainode.core import config as core_config

    return Path(os.environ.get("AINODE_HOME") or core_config.AINODE_HOME)


def sync_state_path(home=None) -> Path:
    return ainode_home(home) / SYNC_STATE_NAME


def read_sync_state(home=None) -> dict:
    """The last pull this node recorded, or ``{}``. Never raises."""
    path = sync_state_path(home)
    try:
        loaded = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def write_sync_state(stamp: str, master: str = "", users: int = 0, home=None) -> None:
    """Record a successful pull, temp-then-replace. Never raises."""
    path = sync_state_path(home)
    payload = {
        "stamp": str(stamp or ""),
        "master": str(master or ""),
        "users": int(users or 0),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(path)
    except OSError as exc:
        logger.warning("could not record the account sync state in %s: %s", path, exc)


# ---------------------------------------------------------------------------
# The loop: a master pushes, a worker pulls
# ---------------------------------------------------------------------------

class AccountReplicator:
    """One task per node that keeps accounts equal to the master's.

    Holds no accounts of its own: the store on the app is the only copy, and this
    reads it per push so a change the CLI wrote on the box (which the store picks
    up through ``reload_if_changed``) reaches the peers on the next tick even when
    nothing called :meth:`notify_changed`.
    """

    def __init__(self, app, pull_interval: float = PULL_INTERVAL_SECONDS,
                 retry_interval: float = RETRY_INTERVAL_SECONDS):
        self.app = app
        self.pull_interval = float(pull_interval)
        self.retry_interval = float(retry_interval)
        #: peer label -> the stamp that peer last ACCEPTED. A peer missing from
        #: here, or holding another stamp, is behind and gets the next push.
        self.accepted: dict[str, str] = {}
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._master_unreachable_logged = False
        self._next_pull = 0.0

    # -- state ------------------------------------------------------------

    @property
    def config(self):
        getter = getattr(self.app, "get", None)
        return getter("config") if callable(getter) else None

    def role(self) -> str:
        getter = getattr(self.app, "get", None)
        cluster = getter("cluster_state") if callable(getter) else None
        return local_role(self.config, cluster=cluster)

    def store(self):
        return open_store(self.app)

    def behind(self, targets: list[tuple[str, str]], stamp: str) -> list[tuple[str, str]]:
        """The peers whose last accepted stamp is not *stamp*."""
        return [(label, base) for label, base in targets
                if self.accepted.get(label) != stamp]

    # -- the two directions ------------------------------------------------

    async def broadcast(self) -> dict:
        """Push this node's accounts to every peer that is behind.

        Returns what happened, for the caller that wants to assert on it. A peer
        that refuses or cannot be reached is logged and left behind, which is
        exactly what makes the next tick retry it.
        """
        store = self.store()
        if store is None:
            return {"pushed": [], "failed": [], "reason": "no account store"}
        try:
            reload_if_changed = getattr(store, "reload_if_changed", None)
            if callable(reload_if_changed):
                reload_if_changed()
            users = store.export_users()
        except Exception:
            logger.exception("could not export this node's accounts")
            return {"pushed": [], "failed": [], "reason": "export failed"}

        stamp = users_stamp(users)
        targets = peer_targets(self.app)
        # Forget peers that are no longer in the cluster view, so a node that left
        # does not hold the loop at a 60 second tick forever.
        known = {label for label, _ in targets}
        for label in list(self.accepted):
            if label not in known:
                self.accepted.pop(label, None)
        todo = self.behind(targets, stamp)
        if not todo:
            return {"pushed": [], "failed": [], "stamp": stamp,
                    "peers": len(targets)}

        session = self._session()
        if session is None:
            return {"pushed": [], "failed": [label for label, _ in todo],
                    "reason": "no HTTP session", "stamp": stamp}

        import aiohttp

        pushed: list[str] = []
        failed: list[str] = []
        for label, base in todo:
            url = base + SYNC_PATH
            try:
                async with session.post(
                    url, json={"users": users},
                    headers=fleet_headers(self.app),
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
                ) as resp:
                    body = await resp.json(content_type=None)
                    if resp.status != 200:
                        failed.append(label)
                        logger.warning("account sync to %s answered %s", label,
                                       resp.status)
                        continue
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                failed.append(label)
                logger.warning("account sync to %s failed: %s", label, exc)
                continue
            self.accepted[label] = stamp
            pushed.append(label)
            changed = bool(isinstance(body, dict) and body.get("changed"))
            if changed:
                logger.info("accounts replicated to %s (%d account(s))", label,
                            len(users))
        return {"pushed": pushed, "failed": failed, "stamp": stamp,
                "peers": len(targets)}

    async def pull(self) -> dict:
        """Import the master's accounts onto this node.

        The master is the authority, so this REPLACES what is here. The first
        failure to reach it is a WARNING and the rest are debug lines: a master
        that is down for an hour must not write sixty identical warnings into the
        log of a node that is working perfectly well otherwise.
        """
        store = self.store()
        if store is None:
            return {"imported": False, "reason": "no account store"}
        base = master_target(self.app)
        if not base:
            return {"imported": False, "reason": "no master known"}
        session = self._session()
        if session is None:
            return {"imported": False, "reason": "no HTTP session"}

        import aiohttp

        url = base + EXPORT_PATH
        try:
            async with session.get(
                url, headers=fleet_headers(self.app),
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
            ) as resp:
                if resp.status != 200:
                    return self._master_unreachable(base, f"answered {resp.status}")
                body = await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            return self._master_unreachable(base, str(exc))

        if not isinstance(body, dict) or not isinstance(body.get("users"), list):
            return self._master_unreachable(base, "answered a body with no user list")

        users = body["users"]
        try:
            changed = bool(store.import_users(users))
        except Exception:
            logger.exception("could not import the master's accounts")
            return {"imported": False, "reason": "import failed"}

        self._master_unreachable_logged = False
        stamp = str(body.get("stamp") or "") or users_stamp(users)
        write_sync_state(stamp, master=base, users=len(users))
        if changed:
            logger.info("accounts updated from the master at %s: %d account(s)",
                        base, len(users))
        return {"imported": True, "changed": changed, "count": len(users),
                "stamp": stamp, "master": base}

    def _master_unreachable(self, base: str, reason: str) -> dict:
        if not self._master_unreachable_logged:
            self._master_unreachable_logged = True
            logger.warning(
                "cannot read accounts from the master at %s (%s); this node keeps "
                "the accounts it has and will try again every %.0fs",
                base, reason, self.pull_interval)
        else:
            logger.debug("master at %s still unreachable: %s", base, reason)
        return {"imported": False, "reason": reason, "master": base}

    def _session(self):
        getter = getattr(self.app, "get", None)
        session = getter("client_session") if callable(getter) else None
        if session is None or getattr(session, "closed", False):
            return None
        return session

    # -- one pass, and the loop around it ---------------------------------

    async def tick(self) -> dict:
        """One pass, which is the unit the tests drive.

        A master pushes what is behind (nothing on the wire when every peer
        agrees). A worker pulls, but only when its five minutes are up: the 60
        second wake exists for the master's retry, and it must not turn into a
        poll of the master twelve times more often than intended.
        """
        role = self.role()
        if role == "master":
            return await self.broadcast()
        if role == "worker":
            now = time.monotonic()
            if now < self._next_pull:
                return {"imported": False, "reason": "not due"}
            result = await self.pull()
            # A pull that FAILED waits the retry interval, not the full five
            # minutes: a worker that came up while its master was still booting
            # would otherwise sit without accounts for the rest of the interval.
            self._next_pull = now + (self.pull_interval if result.get("imported")
                                     else self.retry_interval)
            return result
        return {"reason": "solo node, nothing to replicate", "role": role}

    def notify_changed(self) -> None:
        """Registered as ``app["users_changed"]``: wake the loop, never block it.

        The account routes call this inside a request, so it does exactly one
        thing. On a worker the wake is harmless: the next tick pulls the master's
        list back over the local change, which is the behaviour the CLI warns
        about before it makes one.
        """
        self._wake.set()

    async def _run(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - a tick must never kill the loop
                logger.exception("account replication tick failed")
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._sleep_for())
            except asyncio.TimeoutError:
                pass
            self._wake.clear()

    def _sleep_for(self) -> float:
        """Sixty seconds while anything may be behind, else the pull interval."""
        if self.role() == "master":
            return self.retry_interval
        return min(self.retry_interval, self.pull_interval)

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_event_loop().create_task(self._run())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:  # pragma: no cover - a dying task must not block cleanup
            pass


def start_replication(app) -> Optional[AccountReplicator]:
    """Wire account replication onto a running app, or answer why it did not.

    Called from the server's startup beside the other background tasks. Two gates,
    both of them "this node has nobody to replicate with": no ``cluster_secret``
    means no fleet key, so every request this would make answers 401 and the
    honest thing is to make none; clustering switched off means there is no
    discovery to name a peer with. Returns None in both cases, and the node keeps
    whatever accounts it has.

    The broadcaster is registered whatever this node's role is today, because the
    role is an election and can change under a running process. It asks the role
    per push instead, so a node that becomes master starts pushing and one that
    stops being master stops.
    """
    getter = getattr(app, "get", None)
    if not callable(getter):
        return None
    config = getter("config")
    if not str(cluster_secret_of(app) or "").strip():
        logger.info("account replication off: this node has no cluster_secret, so it "
                    "cannot authenticate to a peer")
        return None
    if not bool(getattr(config, "cluster_enabled", True)) and not str(
            getattr(config, "master_address", "") or "").strip():
        logger.info("account replication off: clustering is disabled and no "
                    "master_address is set")
        return None

    replicator = AccountReplicator(app)
    app["users_replicator"] = replicator
    app["users_changed"] = replicator.notify_changed
    return replicator


async def stop_replication(app) -> None:
    """Stop the loop, if one was started. Never raises."""
    getter = getattr(app, "get", None)
    replicator = getter("users_replicator") if callable(getter) else None
    if replicator is None:
        return
    await replicator.stop()


# ---------------------------------------------------------------------------
# The same push, from the CLI, over the stdlib
# ---------------------------------------------------------------------------

def http_json(url: str, payload: Optional[dict] = None, headers: Optional[dict] = None,
              timeout: float = REQUEST_TIMEOUT_SECONDS) -> tuple[int, Any]:
    """One stdlib request: GET with no *payload*, POST with one.

    ``(status, body)``, where a status of 0 means the request never got an answer.
    urllib rather than aiohttp: this runs from ``ainode auth user add`` on the
    box, and the CLI's import cost is paid by every other subcommand too. It is a
    module-level seam, so a test replaces this one name.
    """
    import urllib.error
    import urllib.request

    data = None
    request_headers = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data,
                                 method="POST" if data is not None else "GET",
                                 headers=request_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            try:
                return resp.status, json.loads(raw)
            except ValueError:
                return resp.status, None
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, None
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        return 0, {"error": str(exc)}


def cluster_info(config) -> Optional[dict]:
    """``GET /api/cluster/info`` on THIS node, or None when it does not answer.

    The CLI runs inside the container the server runs in (the installer's wrapper
    is ``docker exec``), so this is a loopback call, and it is the only way a
    process that is not the server can know who the elected master is and which
    peers discovery has heard from. Carries the fleet key, like every other
    node-to-node read: the route needs a caller id once auth is on.
    """
    port = int(getattr(config, "web_port", 3000) or 3000)
    url = f"http://127.0.0.1:{port}{CLUSTER_INFO_PATH}"
    status, body = http_json(url, headers=fleet_key_headers(
        getattr(config, "cluster_secret", "")))
    if status != 200 or not isinstance(body, dict):
        return None
    return body


def peer_targets_from_info(info: Optional[dict], config) -> list[tuple[str, str]]:
    """``(label, base url)`` for every other node in a ``/api/cluster/info`` body.

    ``/api/cluster/info`` names members but not their addresses, so the address
    comes from the same three places it does everywhere else: the ``host`` field
    when a future release adds one, then the member's name, which on this fleet is
    a resolvable hostname. A member with neither is skipped rather than guessed at.
    """
    if not isinstance(info, dict):
        return []
    local_id = info.get("my_node_id") or getattr(config, "node_id", None)
    targets: list[tuple[str, str]] = []
    for member in info.get("members") or []:
        if not isinstance(member, dict):
            continue
        if member.get("node_id") and member.get("node_id") == local_id:
            continue
        status = str(member.get("status") or "").strip().lower()
        if status == "offline":
            continue
        host = str(member.get("host") or member.get("fabric_ip")
                   or member.get("node_name") or "").strip()
        base = node_base_url(host, int(member.get("web_port") or 3000))
        if not base:
            continue
        targets.append((str(member.get("node_name") or member.get("node_id") or "?"),
                        base))
    return targets


def peer_targets_from_config(config) -> list[tuple[str, str]]:
    """``(label, base url)`` for the peers config.json names.

    The fallback for a CLI whose own node is not answering, so there is no cluster
    view to read: a ``--job master`` install records its peers in ``peer_ips``, and
    pushing to those is better than telling an operator the change reached nobody.
    Their web port is this node's, which is true of every node the installer
    touches, and a peer that disagrees is caught by the master's own loop later.
    """
    port = int(getattr(config, "web_port", 3000) or 3000)
    targets: list[tuple[str, str]] = []
    for peer in getattr(config, "peer_ips", []) or []:
        host = str(peer or "").strip()
        base = node_base_url(host, port)
        if base:
            targets.append((host, base))
    return targets


def replicate_from_cli(config, users: list, info: Optional[dict] = None) -> dict:
    """Push *users* to every peer this node knows, with the fleet key.

    The CLI's half of the master's push: ``ainode auth user add`` writes the store
    on the box, the running server picks the file up through ``reload_if_changed``,
    and this is what puts the same accounts on the peers without waiting for the
    server's own 60 second retry. A peer that does not answer is REPORTED and not
    retried here, because the CLI exits: the master's loop pushes to it on the next
    tick, which is the mechanism that actually guarantees delivery.
    """
    targets = peer_targets_from_info(info, config) or peer_targets_from_config(config)
    headers = fleet_key_headers(getattr(config, "cluster_secret", ""))
    if not headers:
        return {"pushed": [], "failed": [], "peers": len(targets),
                "reason": "no cluster_secret on this node"}
    pushed: list[str] = []
    failed: list[tuple[str, str]] = []
    for label, base in targets:
        status, body = http_json(base + SYNC_PATH, payload={"users": users},
                                 headers=headers)
        if status == 200:
            pushed.append(label)
            continue
        detail = f"HTTP {status}" if status else "no answer"
        if isinstance(body, dict) and body.get("error"):
            detail = f"{detail}: {body['error']}"
        failed.append((label, detail))
    return {"pushed": pushed, "failed": failed, "peers": len(targets)}
