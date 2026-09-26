"""Accounts replicate from the master, sessions never do (#261).

A cluster cannot hold one account list per node: an operator who adds a login on
the master and then opens the dashboard of whichever node their bookmark points at
would be told their password is wrong. So ``ainode/auth/replication.py`` makes the
master the one authority, and this file pins the five things that has to mean:

1. **The master pushes to its peers, with the FLEET KEY.** The export carries
   password hashes, so the route takes that key and nothing else; a push with no
   ``Authorization`` header would be refused by every node in a fleet with auth on,
   which is the bug ``tests/test_fleet_auth.py`` exists to prevent in general.
2. **A peer that refused is retried.** Not queued: the master remembers the stamp
   each peer accepted, so a peer that failed simply still disagrees and the next
   tick pushes to it again. The tick is 60 seconds while anything is behind.
3. **A peer that agrees is not pushed to again**, so the idle cost of this is one
   in-memory comparison rather than a request per minute per node.
4. **A worker pulls the master's list and imports it**, records the master's own
   stamp so ``ainode doctor`` can compare like with like, and an unreachable master
   is one WARNING and a node that keeps working, not a crash and not a log line
   every tick.
5. **Sessions are never on the wire.** The push body carries ``users`` and nothing
   else, and a pull does not touch the sessions this node issued itself.

The account store lands on its own branch (``ainode/auth/accounts.py``), so these
tests drive the contract with :class:`StubStore` and replace the one guarded import
the package makes (``replication.users_store_class``).
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

from ainode.auth import replication as rep
from ainode.auth.fleet import fleet_key
from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeStatus
from ainode.discovery.cluster import ClusterNode, ClusterState

SECRET = "0123456789abcdef0123456789abcdef"
OTHER_SECRET = "fedcba9876543210fedcba9876543210"

USERS = [
    {"name": "jason", "role": "admin", "password_hash": "hash-1"},
    {"name": "ops", "role": "member", "password_hash": "hash-2"},
]


# =============================================================================
# The store's contract, in memory
# =============================================================================

class StubStore:
    """``UsersStore`` as this module uses it. The real one is another branch's."""

    MIN_PASSWORD_LENGTH = 8

    def __init__(self, users=None, sessions=None):
        self.users = [dict(u) for u in (users or [])]
        self.sessions = dict(sessions or {})
        self.reloads = 0
        self.imports: list[list] = []

    # -- what replication calls -----------------------------------------
    def load(self):
        return self

    def reload_if_changed(self) -> bool:
        self.reloads += 1
        return False

    def export_users(self) -> list:
        return [dict(u) for u in self.users]

    def import_users(self, users) -> bool:
        self.imports.append([dict(u) for u in users])
        changed = [dict(u) for u in users] != self.users
        self.users = [dict(u) for u in users]
        return changed

    def sessions_for(self, name) -> list:
        return list(self.sessions.get(name, []))


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """Keep users-sync.json out of the operator's own ~/.ainode."""
    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def loopback_peers(monkeypatch):
    """Reach a test server over loopback.

    ``api/server_routes.peer_host`` deliberately refuses to publish a 127.x
    address (it is the one answer guaranteed wrong on another machine), so the
    seam it hides behind is what a test replaces.
    """
    monkeypatch.setattr(rep, "peer_address", lambda node: "127.0.0.1")


def _node(node_id, web_port=3000, role="auto"):
    return ClusterNode(node_id=node_id, node_name=node_id, gpu_name="NVIDIA GB10",
                       gpu_memory_gb=128.0, unified_memory=True, model="",
                       status=NodeStatus.ONLINE, api_port=8000, web_port=web_port,
                       last_seen=0.0, role=role)


class FakePeer:
    """A node that answers the two replication routes and records what it got."""

    def __init__(self, users=None, stamp="master-stamp-1"):
        self.app = web.Application()
        self.app.router.add_post(rep.SYNC_PATH, self._sync)
        self.app.router.add_get(rep.EXPORT_PATH, self._export)
        self.posts: list[dict] = []
        self.gets: list[dict] = []
        self.status = 200
        self.users = [dict(u) for u in (users or USERS)]
        self.stamp = stamp

    async def _sync(self, request):
        body = await request.json()
        self.posts.append({"body": body,
                           "authorization": request.headers.get("Authorization", "")})
        if self.status != 200:
            return web.json_response({"error": "nope"}, status=self.status)
        self.users = list(body.get("users") or [])
        return web.json_response({"changed": True, "count": len(self.users)})

    async def _export(self, request):
        self.gets.append({"authorization": request.headers.get("Authorization", "")})
        if self.status != 200:
            return web.json_response({"error": "nope"}, status=self.status)
        return web.json_response({"users": self.users, "stamp": self.stamp})


@pytest_asyncio.fixture
async def peer():
    served = FakePeer()
    server = TestServer(served.app)
    await server.start_server()
    served.port = server.port
    try:
        yield served
    finally:
        await server.close()


@pytest_asyncio.fixture
async def session():
    import aiohttp

    async with aiohttp.ClientSession() as client:
        yield client


def _master_app(session, peer_port, store, secret=SECRET):
    """A master's app as replication reads it: a plain mapping is all it needs."""
    cluster = ClusterState()
    cluster.add_node(_node("spark1", role="master"))
    cluster.add_node(_node("spark2", web_port=peer_port))
    return {
        "config": NodeConfig(node_id="spark1", node_name="spark1",
                             cluster_secret=secret, cluster_role="master"),
        "cluster_state": cluster,
        "client_session": session,
        "users_store": store,
    }


def _worker_app(session, master_port, store, secret=SECRET):
    return {
        "config": NodeConfig(node_id="spark3", node_name="spark3",
                             cluster_secret=secret, cluster_role="worker",
                             distributed_mode="member",
                             master_address=f"127.0.0.1:{master_port}"),
        "client_session": session,
        "users_store": store,
    }


# =============================================================================
# 1. The master pushes, with the fleet key
# =============================================================================

@pytest.mark.asyncio
async def test_the_master_pushes_its_accounts_to_every_peer_with_the_fleet_key(
        peer, session):
    store = StubStore(USERS)
    app = _master_app(session, peer.port, store)

    result = await rep.AccountReplicator(app).broadcast()

    assert result["pushed"] == ["spark2"]
    assert result["failed"] == []
    assert len(peer.posts) == 1
    assert peer.posts[0]["authorization"] == f"Bearer {fleet_key(SECRET)}"
    assert peer.posts[0]["body"]["users"] == USERS


@pytest.mark.asyncio
async def test_the_key_on_the_push_follows_this_nodes_own_secret(peer, session):
    """Rotation is a config edit, not a fleet restart: the header is derived per
    request from whatever ``cluster_secret`` says now."""
    app = _master_app(session, peer.port, StubStore(USERS), secret=OTHER_SECRET)

    await rep.AccountReplicator(app).broadcast()

    assert peer.posts[0]["authorization"] == f"Bearer {fleet_key(OTHER_SECRET)}"
    assert peer.posts[0]["authorization"] != f"Bearer {fleet_key(SECRET)}"


@pytest.mark.asyncio
async def test_the_push_carries_users_and_nothing_else(peer, session):
    """Sessions are per node: a body that carried them would hand every node in
    the fleet a credential it never issued."""
    store = StubStore(USERS, sessions={"jason": [{"id": "sess-1"}]})
    await rep.AccountReplicator(_master_app(session, peer.port, store)).broadcast()

    assert list(peer.posts[0]["body"]) == ["users"]


@pytest.mark.asyncio
async def test_a_peer_that_already_agrees_is_not_pushed_to_again(peer, session):
    store = StubStore(USERS)
    replicator = rep.AccountReplicator(_master_app(session, peer.port, store))

    first = await replicator.broadcast()
    second = await replicator.broadcast()

    assert first["pushed"] == ["spark2"]
    assert second["pushed"] == []
    assert len(peer.posts) == 1, "an unchanged list must not be pushed every tick"


@pytest.mark.asyncio
async def test_a_changed_list_is_pushed_again(peer, session):
    store = StubStore(USERS)
    replicator = rep.AccountReplicator(_master_app(session, peer.port, store))
    await replicator.broadcast()

    store.users.append({"name": "second", "role": "member", "password_hash": "h3"})
    result = await replicator.broadcast()

    assert result["pushed"] == ["spark2"]
    assert len(peer.posts) == 2
    assert [u["name"] for u in peer.posts[1]["body"]["users"]] == \
        ["jason", "ops", "second"]


@pytest.mark.asyncio
async def test_the_master_reads_the_store_off_disk_before_it_pushes(peer, session):
    """``ainode auth user add`` writes the file on the box, so the push has to ask
    the store whether the file moved rather than trusting what it holds."""
    store = StubStore(USERS)
    await rep.AccountReplicator(_master_app(session, peer.port, store)).broadcast()

    assert store.reloads == 1


# =============================================================================
# 2. A peer that refused is retried
# =============================================================================

@pytest.mark.asyncio
async def test_a_peer_that_refuses_is_retried_on_the_next_tick(peer, session):
    store = StubStore(USERS)
    replicator = rep.AccountReplicator(_master_app(session, peer.port, store))

    peer.status = 503
    first = await replicator.broadcast()
    assert first["failed"] == ["spark2"] and first["pushed"] == []

    peer.status = 200
    second = await replicator.broadcast()
    assert second["pushed"] == ["spark2"]
    assert peer.posts[-1]["body"]["users"] == USERS


@pytest.mark.asyncio
async def test_a_peer_that_cannot_be_reached_at_all_is_a_failure_not_a_crash(session):
    store = StubStore(USERS)
    # Port 1 on loopback: nothing listens there, so this is the "node rebooting"
    # case, and it must leave the peer behind rather than raise.
    app = _master_app(session, 1, store)

    result = await rep.AccountReplicator(app).broadcast()

    assert result["failed"] == ["spark2"]
    assert result["pushed"] == []


@pytest.mark.asyncio
async def test_the_retry_timer_is_sixty_seconds_on_a_master(peer, session):
    """The loop's wake, which is what makes "retried while behind" true without a
    queue. Pinned because the number is the promise in the PR body."""
    app = _master_app(session, peer.port, StubStore(USERS))
    replicator = rep.AccountReplicator(app)

    assert rep.RETRY_INTERVAL_SECONDS == 60.0
    assert replicator._sleep_for() == 60.0


@pytest.mark.asyncio
async def test_a_peer_that_left_the_cluster_stops_being_behind(peer, session):
    store = StubStore(USERS)
    app = _master_app(session, peer.port, store)
    replicator = rep.AccountReplicator(app)
    peer.status = 503
    await replicator.broadcast()
    assert replicator.accepted == {}

    app["cluster_state"].remove_node("spark2")
    result = await replicator.broadcast()

    assert result["peers"] == 0
    assert result["failed"] == []


@pytest.mark.asyncio
async def test_a_master_with_no_peers_makes_no_requests(session):
    store = StubStore(USERS)
    cluster = ClusterState()
    cluster.add_node(_node("spark1", role="master"))
    app = {"config": NodeConfig(node_id="spark1", cluster_secret=SECRET,
                               cluster_role="master"),
           "cluster_state": cluster, "client_session": session,
           "users_store": store}

    result = await rep.AccountReplicator(app).broadcast()

    assert result["peers"] == 0 and result["pushed"] == []


# =============================================================================
# 3. A worker pulls
# =============================================================================

@pytest.mark.asyncio
async def test_a_worker_imports_the_masters_accounts(peer, session, home):
    store = StubStore([])
    app = _worker_app(session, peer.port, store)

    result = await rep.AccountReplicator(app).pull()

    assert result["imported"] is True and result["changed"] is True
    assert store.imports == [USERS]
    assert peer.gets[0]["authorization"] == f"Bearer {fleet_key(SECRET)}"
    # The MASTER's own stamp is recorded, so the doctor compares two values of the
    # same kind rather than two hashes computed by different code.
    assert rep.read_sync_state(home)["stamp"] == "master-stamp-1"
    assert rep.read_sync_state(home)["users"] == 2


@pytest.mark.asyncio
async def test_a_worker_pull_leaves_this_nodes_own_sessions_alone(peer, session):
    store = StubStore([], sessions={"jason": [{"id": "sess-1"}]})
    await rep.AccountReplicator(_worker_app(session, peer.port, store)).pull()

    assert store.sessions_for("jason") == [{"id": "sess-1"}]


@pytest.mark.asyncio
async def test_an_unchanged_pull_is_reported_as_unchanged(peer, session):
    store = StubStore(USERS)
    result = await rep.AccountReplicator(_worker_app(session, peer.port, store)).pull()

    assert result["imported"] is True and result["changed"] is False


@pytest.mark.asyncio
async def test_an_unreachable_master_warns_once_and_does_not_crash(session, caplog):
    store = StubStore(USERS)
    app = _worker_app(session, 1, store)  # nothing listens on port 1
    replicator = rep.AccountReplicator(app)

    with caplog.at_level(logging.WARNING, logger=rep.logger.name):
        first = await replicator.pull()
        second = await replicator.pull()

    assert first["imported"] is False and second["imported"] is False
    assert store.imports == [], "nothing may be imported from a master that is down"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, "a master that is down must not warn on every tick"
    assert "keeps the accounts it has" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_a_master_that_answers_rubbish_is_not_imported(session, home):
    async def handler(request):
        return web.json_response({"stamp": "s", "users": "not a list"})

    app = web.Application()
    app.router.add_get(rep.EXPORT_PATH, handler)
    server = TestServer(app)
    await server.start_server()
    try:
        store = StubStore(USERS)
        result = await rep.AccountReplicator(
            _worker_app(session, server.port, store)).pull()
    finally:
        await server.close()

    assert result["imported"] is False
    assert store.imports == []
    assert rep.read_sync_state(home) == {}


@pytest.mark.asyncio
async def test_a_worker_with_no_master_makes_no_requests(session):
    store = StubStore(USERS)
    app = {"config": NodeConfig(node_id="spark3", cluster_secret=SECRET,
                               cluster_role="worker"),
           "client_session": session, "users_store": store}

    result = await rep.AccountReplicator(app).pull()

    assert result == {"imported": False, "reason": "no master known"}


@pytest.mark.asyncio
async def test_a_worker_pulls_at_startup_and_then_on_its_own_interval(peer, session):
    """The 60 second wake is the master's retry, not a poll of the master: a
    worker that pulled on every one of them would ask twelve times too often."""
    store = StubStore([])
    replicator = rep.AccountReplicator(_worker_app(session, peer.port, store))

    first = await replicator.tick()
    second = await replicator.tick()

    assert first["imported"] is True
    assert second == {"imported": False, "reason": "not due"}
    assert len(peer.gets) == 1
    assert rep.PULL_INTERVAL_SECONDS == 300.0


# =============================================================================
# 4. Roles, and the nodes that do nothing
# =============================================================================

def test_a_node_with_no_cluster_secret_starts_nothing():
    app = {"config": NodeConfig(node_id="n1", cluster_secret=None)}
    assert rep.start_replication(app) is None
    assert "users_changed" not in app


def test_start_registers_the_broadcaster_the_routes_call():
    app = {"config": NodeConfig(node_id="n1", cluster_secret=SECRET)}
    replicator = rep.start_replication(app)

    assert replicator is not None
    assert app["users_changed"] == replicator.notify_changed
    # It must never block a request: the routes call it inside a handler.
    app["users_changed"]()
    assert replicator._wake.is_set()


def test_a_solo_node_replicates_nothing():
    cluster = ClusterState()
    cluster.add_node(_node("only", role="auto"))
    app = {"config": NodeConfig(node_id="only", cluster_secret=SECRET),
           "cluster_state": cluster}

    assert rep.AccountReplicator(app).role() == "solo"


@pytest.mark.asyncio
async def test_a_solo_tick_touches_nothing(session):
    cluster = ClusterState()
    cluster.add_node(_node("only", role="auto"))
    store = StubStore(USERS)
    app = {"config": NodeConfig(node_id="only", cluster_secret=SECRET),
           "cluster_state": cluster, "client_session": session,
           "users_store": store}

    result = await rep.AccountReplicator(app).tick()

    assert result["role"] == "solo"
    assert store.reloads == 0


def test_a_configured_worker_is_never_the_authority_whatever_discovery_says():
    """An operator who wrote cluster_role: worker has said this node is not it,
    and a node that cannot see its master must not decide it was promoted."""
    cluster = ClusterState()
    cluster.add_node(_node("me", role="worker"))
    config = NodeConfig(node_id="me", cluster_role="worker")

    assert rep.local_role(config, cluster=cluster) == "worker"
    assert rep.local_role(NodeConfig(node_id="me", distributed_mode="member")) \
        == "worker"


def test_the_election_outranks_a_config_that_pinned_two_masters():
    cluster = ClusterState()
    cluster.add_node(_node("aaa", role="master"))
    cluster.add_node(_node("zzz", role="master"))

    assert rep.local_role(NodeConfig(node_id="aaa", cluster_role="master"),
                          cluster=cluster) == "master"
    assert rep.local_role(NodeConfig(node_id="zzz", cluster_role="master"),
                          cluster=cluster) == "worker"


def test_the_role_falls_back_to_the_api_then_to_config_json():
    """A CLI has no ClusterState. With the service up it has /api/cluster/info;
    with the service down it has config.json and nothing else."""
    info = {"my_role": "master", "my_node_id": "m1",
            "members": [{"node_id": "m1"}, {"node_id": "w1"}]}
    assert rep.local_role(NodeConfig(node_id="m1"), info=info) == "master"
    assert rep.local_role(NodeConfig(node_id="w1"),
                          info={"my_role": "worker", "my_node_id": "w1",
                                "members": [{"node_id": "m1"}]}) == "worker"
    assert rep.local_role(NodeConfig(node_id="m1", cluster_role="master",
                                     peer_ips=["10.0.0.2"])) == "master"
    assert rep.local_role(NodeConfig(node_id="w1",
                                     master_address="10.0.0.1:3000")) == "worker"
    assert rep.local_role(NodeConfig(node_id="solo")) == "solo"


def test_the_master_url_comes_from_discovery_first_then_config():
    cluster = ClusterState()
    cluster.add_node(_node("me", role="worker"))
    cluster.add_node(_node("boss", web_port=3100, role="master"))
    app = {"config": NodeConfig(node_id="me", cluster_role="worker",
                               master_address="10.9.9.9:3000"),
           "cluster_state": cluster}

    assert rep.master_target(app) == "http://127.0.0.1:3100"
    assert rep.master_target(config=NodeConfig(master_address="10.9.9.9")) \
        == "http://10.9.9.9:3000"
    assert rep.master_target(config=NodeConfig()) == ""


# =============================================================================
# 5. The CLI's push, over the stdlib
# =============================================================================

def test_the_cli_push_carries_the_fleet_key_to_every_peer(monkeypatch):
    seen = []

    def fake_http_json(url, payload=None, headers=None, timeout=10.0):
        seen.append({"url": url, "payload": payload, "headers": headers})
        return 200, {"changed": True, "count": 2}

    monkeypatch.setattr(rep, "http_json", fake_http_json)
    config = NodeConfig(node_id="m1", cluster_secret=SECRET, cluster_role="master",
                        peer_ips=["10.0.0.2", "10.0.0.3"])

    result = rep.replicate_from_cli(config, USERS)

    assert result["pushed"] == ["10.0.0.2", "10.0.0.3"]
    assert [call["url"] for call in seen] == [
        "http://10.0.0.2:3000/api/auth/users/sync",
        "http://10.0.0.3:3000/api/auth/users/sync",
    ]
    for call in seen:
        assert call["headers"]["Authorization"] == f"Bearer {fleet_key(SECRET)}"
        assert list(call["payload"]) == ["users"]


def test_the_cli_push_reports_a_peer_that_refused_rather_than_retrying(monkeypatch):
    """The CLI exits; the master's own loop is what guarantees delivery."""
    monkeypatch.setattr(rep, "http_json",
                        lambda url, payload=None, headers=None, timeout=10.0:
                        (0, {"error": "connection refused"}))
    config = NodeConfig(cluster_secret=SECRET, cluster_role="master",
                        peer_ips=["10.0.0.2"])

    result = rep.replicate_from_cli(config, USERS)

    assert result["pushed"] == []
    assert result["failed"][0][0] == "10.0.0.2"
    assert "no answer" in result["failed"][0][1]


def test_the_cli_push_prefers_the_live_cluster_view_over_config(monkeypatch):
    seen = []
    monkeypatch.setattr(rep, "http_json",
                        lambda url, payload=None, headers=None, timeout=10.0:
                        (seen.append(url), (200, {}))[1])
    config = NodeConfig(node_id="m1", cluster_secret=SECRET, cluster_role="master",
                        peer_ips=["10.0.0.9"])
    info = {"my_node_id": "m1", "members": [
        {"node_id": "m1", "node_name": "m1", "web_port": 3000},
        {"node_id": "w1", "node_name": "10.0.0.2", "web_port": 3000},
        {"node_id": "w2", "node_name": "10.0.0.3", "web_port": 3000,
         "status": "offline"},
    ]}

    rep.replicate_from_cli(config, USERS, info=info)

    assert seen == ["http://10.0.0.2:3000/api/auth/users/sync"], \
        "an offline member and this node itself are not push targets"


def test_a_cli_push_with_no_cluster_secret_says_so_and_sends_nothing(monkeypatch):
    monkeypatch.setattr(rep, "http_json",
                        lambda *a, **kw: pytest.fail("no key, so no request"))
    config = NodeConfig(cluster_role="master", peer_ips=["10.0.0.2"])

    result = rep.replicate_from_cli(config, USERS)

    assert result["pushed"] == [] and "cluster_secret" in result["reason"]


# =============================================================================
# 5b. An empty list never crosses the wire (the Atlas cutover guard)
# =============================================================================
#
# import_users REPLACES the whole list, so a master promoted before its users.json
# was copied would log every dashboard user out of the fleet on its first push,
# and every worker would do the same to itself on its next pull.

@pytest.mark.asyncio
async def test_a_master_with_an_empty_store_pushes_nothing_and_warns_once(
        peer, session, caplog):
    app = _master_app(session, peer.port, StubStore([]))
    replicator = rep.AccountReplicator(app)

    with caplog.at_level(logging.WARNING, logger=rep.logger.name):
        first = await replicator.tick()
        second = await replicator.tick()

    assert peer.posts == [], "an empty list reached a worker"
    assert peer.users == USERS
    assert first["reason"] == rep.EMPTY_LIST_REASON
    assert first["pushed"] == [] and second["pushed"] == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, "the 60 second retry must not warn on every tick"
    assert "no accounts" in warnings[0].getMessage()
    assert "users.json" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_a_master_pushes_again_once_its_store_has_accounts(peer, session):
    store = StubStore([])
    replicator = rep.AccountReplicator(_master_app(session, peer.port, store))
    await replicator.broadcast()
    assert peer.posts == []

    store.users = [dict(u) for u in USERS]
    result = await replicator.broadcast()

    assert result["pushed"] == ["spark2"]
    assert peer.posts[0]["body"] == {"users": USERS}


@pytest.mark.asyncio
async def test_a_worker_refuses_an_empty_list_from_its_master(peer, session, home,
                                                              caplog):
    peer.users = []
    store = StubStore(USERS)

    with caplog.at_level(logging.WARNING, logger=rep.logger.name):
        result = await rep.AccountReplicator(
            _worker_app(session, peer.port, store)).pull()

    assert result["imported"] is False
    assert result["reason"] == rep.EMPTY_LIST_REASON
    assert store.imports == [], "the worker wiped its accounts to match an empty master"
    assert store.users == USERS
    assert rep.read_sync_state(home) == {}, "a refused pull is not a successful sync"
    assert any("empty account list" in r.getMessage() for r in caplog.records)


def test_a_cli_push_of_an_empty_list_is_refused_and_sends_nothing(monkeypatch):
    monkeypatch.setattr(rep, "http_json",
                        lambda *a, **kw: pytest.fail("an empty list went out"))
    config = NodeConfig(node_id="m1", cluster_secret=SECRET, cluster_role="master",
                        peer_ips=["10.0.0.2"])

    result = rep.replicate_from_cli(config, [])

    assert result["pushed"] == [] and result["failed"] == []
    assert rep.EMPTY_LIST_REASON in result["reason"]


# =============================================================================
# 6. The stamp, and the store seam
# =============================================================================

def test_the_stamp_follows_the_content_and_not_a_records_key_order():
    """What "behind" means. A store that writes its keys in another order must not
    read as a changed list, or every tick would push to every peer forever."""
    reordered = [{"role": u["role"], "password_hash": u["password_hash"],
                  "name": u["name"]} for u in USERS]
    assert rep.users_stamp(USERS) == rep.users_stamp(reordered)
    assert rep.users_stamp(USERS) != rep.users_stamp(USERS[:1])
    changed = [dict(USERS[0], password_hash="other"), USERS[1]]
    assert rep.users_stamp(USERS) != rep.users_stamp(changed)


def test_open_store_prefers_the_one_on_the_app(monkeypatch):
    mine = StubStore(USERS)
    assert rep.open_store({"users_store": mine}) is mine


def test_open_store_survives_a_release_with_no_account_store(monkeypatch):
    monkeypatch.setattr(rep, "users_store_class", lambda: None)
    assert rep.open_store() is None
    assert rep.min_password_length() == rep.DEFAULT_MIN_PASSWORD


def test_open_store_keeps_a_load_that_returns_a_new_instance(monkeypatch):
    """``AuthConfig.load()`` is a classmethod that returns a fresh object, and the
    account store is modelled on it, so both spellings have to work."""
    class ClassmethodStore(StubStore):
        made: list = []

        @classmethod
        def load(cls):  # type: ignore[override]
            fresh = ClassmethodStore(USERS)
            cls.made.append(fresh)
            return fresh

    monkeypatch.setattr(rep, "users_store_class", lambda: ClassmethodStore)
    store = rep.open_store()

    assert ClassmethodStore.made and store is ClassmethodStore.made[-1]
    assert store.export_users() == USERS


def test_the_sync_state_is_written_and_read_back(home):
    rep.write_sync_state("stamp-9", master="http://10.0.0.1:3000", users=3)
    state = rep.read_sync_state()

    assert state["stamp"] == "stamp-9"
    assert state["master"] == "http://10.0.0.1:3000"
    assert state["users"] == 3
    assert state["at"].endswith("Z")
    # No credential goes in this file, so nothing here needs to be 0600.
    assert "hash" not in rep.sync_state_path().read_text()


def test_an_unreadable_sync_state_is_an_empty_one(home):
    rep.sync_state_path().write_text("{not json")
    assert rep.read_sync_state() == {}


# =============================================================================
# 7. The loop, started and stopped
# =============================================================================

@pytest.mark.asyncio
async def test_the_loop_pushes_when_it_is_woken_and_stops_cleanly(peer, session):
    store = StubStore(USERS)
    app = _master_app(session, peer.port, store)
    replicator = rep.AccountReplicator(app, retry_interval=30.0)
    await replicator.start()
    try:
        for _ in range(50):
            if peer.posts:
                break
            await asyncio.sleep(0.02)
        assert peer.posts, "the first tick pushes the current list"

        store.users = [dict(USERS[0])]
        replicator.notify_changed()
        for _ in range(50):
            if len(peer.posts) > 1:
                break
            await asyncio.sleep(0.02)
        assert len(peer.posts) == 2, "a change wakes the loop rather than waiting"
    finally:
        await rep.stop_replication(app | {"users_replicator": replicator})
    assert replicator._task is None


@pytest.mark.asyncio
async def test_a_tick_that_raises_does_not_kill_the_loop(session, monkeypatch):
    app = {"config": NodeConfig(node_id="n1", cluster_secret=SECRET,
                               cluster_role="master", peer_ips=["10.0.0.2"]),
           "client_session": session, "users_store": StubStore(USERS)}
    replicator = rep.AccountReplicator(app, retry_interval=0.01)
    calls = []

    async def boom():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("the first tick explodes")
        return {}

    monkeypatch.setattr(replicator, "tick", boom)
    await replicator.start()
    try:
        for _ in range(100):
            if len(calls) > 1:
                break
            await asyncio.sleep(0.01)
    finally:
        await replicator.stop()
    assert len(calls) > 1, "the loop must survive a tick that raised"


@pytest.mark.asyncio
async def test_a_failed_pull_is_retried_on_the_retry_interval_not_the_full_five_minutes(
        peer, session):
    """A worker that came up while its master was still booting must not sit
    without accounts for five minutes."""
    store = StubStore([])
    replicator = rep.AccountReplicator(_worker_app(session, peer.port, store))
    peer.status = 503

    failed = await replicator.tick()
    assert failed["imported"] is False
    assert replicator._next_pull - time.monotonic() <= rep.RETRY_INTERVAL_SECONDS

    peer.status = 200
    replicator._next_pull = 0.0
    ok = await replicator.tick()

    assert ok["imported"] is True
    assert replicator._next_pull - time.monotonic() > rep.RETRY_INTERVAL_SECONDS, \
        "a successful pull goes back to the five minute interval"


@pytest.mark.asyncio
async def test_stop_replication_is_safe_on_a_node_that_started_none():
    await rep.stop_replication({"config": NodeConfig()})


@pytest.mark.asyncio
async def test_the_servers_startup_wires_the_broadcaster_and_its_cleanup_stops_it(
        tmp_path, monkeypatch):
    """The real ``create_app`` startup, because a loop nobody starts replicates
    nothing and the account routes would find no ``users_changed`` to call.

    Clustering is off here with a ``master_address`` set, which is a real shape (a
    member node whose discovery is disabled) and keeps the test off the UDP wire.
    """
    from aiohttp.test_utils import TestClient

    import ainode.api.server as server

    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr("ainode.auth.middleware.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.auth.middleware.AUTH_FILE", tmp_path / "auth.json")
    config = NodeConfig(node_id="n1", node_name="n1", cluster_secret=SECRET,
                        cluster_enabled=False, cluster_role="worker",
                        master_address="127.0.0.1:1", onboarded=True)
    app = server.create_app(config=config, engine=None)

    async with TestClient(TestServer(app)) as client:
        assert client is not None
        assert callable(app.get("users_changed")), \
            "the account routes call app['users_changed'] after every mutation"
        replicator = app.get("users_replicator")
        assert replicator is not None and replicator._task is not None

    assert app.get("users_replicator")._task is None, "cleanup cancels the loop"


@pytest.mark.asyncio
async def test_a_node_with_no_cluster_secret_comes_up_with_no_loop(tmp_path,
                                                                  monkeypatch):
    from aiohttp.test_utils import TestClient

    import ainode.api.server as server

    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr("ainode.auth.middleware.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.auth.middleware.AUTH_FILE", tmp_path / "auth.json")
    app = server.create_app(config=NodeConfig(node_id="n1", cluster_enabled=False,
                                              onboarded=True), engine=None)

    async with TestClient(TestServer(app)):
        assert app.get("users_replicator") is None
        assert app.get("users_changed") is None
