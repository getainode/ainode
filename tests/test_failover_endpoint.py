"""A client endpoint that survives the master.

Every node routes every fleet model, but a client holds ONE address, so an
outage on the node it holds strands it while five others could have served it.
``GET /api/cluster/endpoint`` is the fix: any node, with or without a key, can
say where the fleet answers.

What this file pins:

* the shape, for the three cases a client meets (a peer is master, this node is
  master, no master has been seen),
* the Host-header rule: the address the caller used is echoed back only when
  this node really answers on it, so the header cannot make a node advertise
  somebody else's address,
* that ``localhost`` never appears in the payload, on any node, in any field,
  because it is the one address that is guaranteed wrong on the caller's machine
  (and it is what ``/api/nodes`` reports for every row),
* that the endpoint answers with no key when auth is on, and that
  ``/api/status`` carries the same rows as ``endpoint_hint``.
"""

import json
import socket
import time

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from ainode.api import server_routes
from ainode.api.server import create_app
from ainode.api.server_routes import (
    _is_usable_host,
    endpoint_nodes,
    endpoint_payload,
    endpoint_url,
    host_header_host,
    lan_address,
    peer_host,
)
from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeAnnouncement, NodeStatus
from ainode.discovery.cluster import ClusterNode


# =============================================================================
# Fixtures and builders
# =============================================================================

#: This node's own address in every test that patches the seams, so an assert can
#: name it. Deliberately not the machine's real address: the two seams below are
#: what talks to the host, and everything above them has to be testable without.
OUR_ADDRESS = "10.0.0.7"


@pytest.fixture
def endpoint_home(tmp_path, monkeypatch):
    """Keep what create_app touches out of the operator's own ~/.ainode."""
    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr("ainode.auth.middleware.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.auth.middleware.AUTH_FILE", tmp_path / "auth.json")
    monkeypatch.setattr("ainode.secrets.manager.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.secrets.manager.SECRETS_FILE", tmp_path / "secrets.json")
    return tmp_path


@pytest.fixture(autouse=True)
def fresh_address_cache():
    """The address derivation is cached per process; start every test clean."""
    server_routes.reset_address_cache()
    yield
    server_routes.reset_address_cache()


@pytest.fixture
def fixed_addresses(monkeypatch):
    """Pin the two seams that read the real host, so the rules above are testable.

    ``own_addresses`` is the Host-header guard and ``lan_address`` is the fallback.
    Everything else in the payload is derived from these two plus the cluster
    state, and neither of them may touch a real interface inside the suite.
    """
    monkeypatch.setattr(
        server_routes, "own_addresses",
        lambda config: {OUR_ADDRESS, "spark-1", "100.122.26.9", "localhost", "127.0.0.1"},
    )
    monkeypatch.setattr(server_routes, "lan_address", lambda config: OUR_ADDRESS)


def _config(**kw) -> NodeConfig:
    defaults = dict(node_id="spark-1", node_name="Spark-1", host="0.0.0.0",
                    api_port=8000, web_port=3000, onboarded=True)
    defaults.update(kw)
    return NodeConfig(**defaults)


def _ann(**kw) -> NodeAnnouncement:
    defaults = dict(
        node_id="spark-1", node_name="Spark-1", gpu_name="GB10",
        gpu_memory_gb=128.0, unified_memory=True, model="m",
        status="serving", api_port=8000, web_port=3000,
        cluster_id="default", role="auto",
    )
    defaults.update(kw)
    return NodeAnnouncement(**defaults)


def _node(**kw) -> ClusterNode:
    defaults = dict(
        node_id="spark-2", node_name="Spark-2", gpu_name="GB10",
        gpu_memory_gb=128.0, unified_memory=True, model="m",
        status=NodeStatus.ONLINE, api_port=8000, web_port=3000,
        last_seen=time.time(), cluster_id="default", role="auto",
    )
    defaults.update(kw)
    return ClusterNode(**defaults)


class _Request:
    """The two things these helpers read off a request: headers, and the app."""

    def __init__(self, app=None, host: str = ""):
        self.app = app or {}
        self.headers = {"Host": host} if host else {}


def _app(config: NodeConfig, *peers: ClusterNode, local_role: str = "auto") -> dict:
    """An app mapping with a real ClusterState in it, no HTTP server needed."""
    from ainode.discovery.cluster import ClusterState

    cluster = ClusterState(local_announcement=_ann(
        node_id=config.node_id, node_name=config.node_name, role=local_role))
    for peer in peers:
        cluster.add_node(peer)
    return {"config": config, "cluster_state": cluster}


# =============================================================================
# The shape, for the three cases a client meets
# =============================================================================

def test_master_is_a_peer(fixed_addresses):
    """A worker answers with the master's real address, not its own."""
    config = _config(node_id="spark-2", node_name="Spark-2")
    master = _node(node_id="spark-1", node_name="Spark-1", role="master",
                   peer_ip="10.0.0.1", web_port=3000)
    payload = endpoint_payload(_app(config, master, local_role="auto"),
                               _Request(host="10.0.0.7:3000"))

    assert payload["self"]["name"] == "Spark-2"
    assert payload["self"]["role"] == "worker"
    assert payload["self"]["host"] == OUR_ADDRESS
    assert payload["self"]["port"] == 3000

    # A peer's TLS state is not on the discovery wire, so a peer block is http.
    assert payload["master"] == {
        "name": "Spark-1", "host": "10.0.0.1", "port": 3000,
        "tls": False, "tls_port": None, "url": "http://10.0.0.1:3000",
    }

    # Both nodes are offered as places to talk to, master first.
    assert [n["name"] for n in payload["nodes"]] == ["Spark-1", "Spark-2"]
    assert [n["role"] for n in payload["nodes"]] == ["master", "worker"]
    assert [n["url"] for n in payload["nodes"]] == [
        "http://10.0.0.1:3000", f"http://{OUR_ADDRESS}:3000"]
    assert payload["generated_at"] > 0


def test_the_node_answering_is_itself_the_master(fixed_addresses):
    """The master's own answer names itself, at the address the caller reached."""
    config = _config()
    payload = endpoint_payload(_app(config, _node(node_id="spark-2", role="auto",
                                                  peer_ip="10.0.0.2")),
                               _Request(host=f"{OUR_ADDRESS}:3000"))

    assert payload["self"]["role"] == "master"
    assert payload["master"]["name"] == "Spark-1"
    assert payload["master"]["host"] == OUR_ADDRESS
    assert payload["master"]["url"] == f"http://{OUR_ADDRESS}:3000"
    # The master row and the self row agree: one address, not two spellings.
    master_row = next(n for n in payload["nodes"] if n["role"] == "master")
    assert master_row["host"] == payload["self"]["host"]
    assert master_row["version"], "the local node knows its own version exactly"


def test_a_member_that_has_seen_no_master_says_null(fixed_addresses):
    """A worker-role node alone on the network reports master: null.

    Null and not itself, and not an empty dict: a client has to be able to tell
    "no coordinator right now" apart from "the coordinator is the node I asked".
    """
    config = _config(node_id="spark-3", node_name="Spark-3", cluster_role="worker")
    payload = endpoint_payload(_app(config, local_role="worker"),
                               _Request(host=f"{OUR_ADDRESS}:3000"))

    assert payload["master"] is None
    assert payload["self"]["role"] == "worker"
    # Itself is still a usable endpoint: it routes every fleet model like any
    # other node, which is the whole reason this list is worth saving.
    assert [n["name"] for n in payload["nodes"]] == ["Spark-3"]
    assert payload["nodes"][0]["url"] == f"http://{OUR_ADDRESS}:3000"


def test_a_lone_node_with_clustering_off_answers_for_itself(fixed_addresses):
    """No peers, no discovery: the payload is still a usable single-node answer."""
    payload = endpoint_payload(_app(_config()), _Request(host=f"{OUR_ADDRESS}:3000"))
    assert payload["self"]["host"] == OUR_ADDRESS
    assert payload["master"]["name"] == "Spark-1"
    assert len(payload["nodes"]) == 1


def test_offline_nodes_are_not_offered_as_fallbacks(fixed_addresses):
    """This is a list of addresses to try, not a roster of everything ever seen."""
    config = _config()
    gone = _node(node_id="spark-9", node_name="Spark-9", status=NodeStatus.OFFLINE,
                 peer_ip="10.0.0.9")
    stale = _node(node_id="spark-4", node_name="Spark-4", status=NodeStatus.STALE,
                  peer_ip="10.0.0.4")
    payload = endpoint_payload(_app(config, gone, stale), _Request())
    names = [n["name"] for n in payload["nodes"]]
    assert "Spark-9" not in names
    # A stale node has missed a heartbeat, not died: still worth trying.
    assert "Spark-4" in names


def test_a_peers_version_comes_off_the_wire_or_is_empty(fixed_addresses):
    """A peer announces its release (#171); one too old to says nothing.

    Never this node's version standing in for a peer's: that is how a split fleet
    goes on looking like a healthy one. Empty rather than null, because that is
    the spelling every other node-listing view already uses.
    """
    payload = endpoint_payload(
        _app(_config(),
             _node(node_id="spark-2", node_name="Spark-2", peer_ip="10.0.0.2",
                   ainode_version="0.5.28"),
             _node(node_id="spark-9", node_name="Spark-9", peer_ip="10.0.0.9")),
        _Request())
    rows = {n["name"]: n for n in payload["nodes"]}
    assert rows["Spark-1"]["version"] == payload["self"]["version"]
    assert rows["Spark-2"]["version"] == "0.5.28"
    assert rows["Spark-9"]["version"] == ""
    assert rows["Spark-9"]["version"] != payload["self"]["version"]


def test_nodes_of_another_cluster_are_not_offered(fixed_addresses):
    """cluster_id scoping holds here too: a stranger is not a fallback."""
    other = _node(node_id="zzz", node_name="Somebody-Else", cluster_id="other",
                  peer_ip="10.9.9.9")
    payload = endpoint_payload(_app(_config(), other), _Request())
    assert [n["name"] for n in payload["nodes"]] == ["Spark-1"]


# =============================================================================
# The Host-header rule
# =============================================================================

def test_the_host_header_is_echoed_when_this_node_answers_on_it(fixed_addresses):
    """A caller that reached us over the tailnet is told the tailnet address.

    The address the caller used is better than anything this node can derive: it
    is the one address the caller has PROVED it can reach.
    """
    payload = endpoint_payload(_app(_config()), _Request(host="100.122.26.9:3000"))
    assert payload["self"]["host"] == "100.122.26.9"
    assert payload["master"]["url"] == "http://100.122.26.9:3000"
    assert payload["nodes"][0]["url"] == "http://100.122.26.9:3000"


def test_a_host_header_this_node_does_not_answer_on_is_ignored(fixed_addresses):
    """A header naming somebody else cannot make this node advertise it.

    The Host header is caller-controlled, so a request carrying
    ``Host: attacker.example`` would otherwise have this node hand that address
    to the next client as a fleet endpoint.
    """
    payload = endpoint_payload(_app(_config()), _Request(host="attacker.example:3000"))
    assert payload["self"]["host"] == OUR_ADDRESS


def test_a_loopback_host_header_falls_back_to_the_lan_address(fixed_addresses):
    """``localhost`` is ours and still never the answer.

    A dashboard opened on the node itself sends ``Host: localhost:3000``, and
    that node's answer is read by clients on other machines.
    """
    for header in ("localhost:3000", "127.0.0.1:3000", "[::1]:3000", "0.0.0.0:3000"):
        payload = endpoint_payload(_app(_config()), _Request(host=header))
        assert payload["self"]["host"] == OUR_ADDRESS, header


def test_no_host_header_at_all_still_answers(fixed_addresses):
    payload = endpoint_payload(_app(_config()), _Request())
    assert payload["self"]["host"] == OUR_ADDRESS


def test_host_header_parsing():
    assert host_header_host(_Request(host="10.0.0.1:3000")) == "10.0.0.1"
    assert host_header_host(_Request(host="spark-1")) == "spark-1"
    assert host_header_host(_Request(host="[fd00::1]:3000")) == "[fd00::1]"
    assert host_header_host(_Request()) == ""
    assert host_header_host(object()) == ""


# =============================================================================
# localhost is never an answer
# =============================================================================

def test_localhost_appears_nowhere_in_the_payload(fixed_addresses):
    """Walk the whole serialized body: not one loopback spelling in it.

    ``/api/nodes`` reports ``"host": "localhost"`` for every row, which is why
    this is asserted over the text and not field by field.
    """
    config = _config()
    peers = [
        _node(node_id="spark-2", node_name="Spark-2", peer_ip="10.0.0.2"),
        # No source IP captured (added by hand, or an announcement seen through a
        # path that lost it), so the fabric IP answers for it.
        _node(node_id="spark-3", node_name="Spark-3", peer_ip=None,
              fabric_ip="10.100.0.3"),
        # Nothing routable at all: a null url, never a loopback one.
        _node(node_id="spark-4", node_name="unknown", peer_ip=None, fabric_ip=""),
    ]
    body = json.dumps(endpoint_payload(_app(config, *peers), _Request(host="localhost:3000")))
    for spelling in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        assert spelling not in body, f"{spelling} was handed to a client: {body}"


def test_a_node_with_no_routable_address_gets_a_null_url(fixed_addresses):
    """"Unknown" beats a wrong address: a client can skip a null, not a lie."""
    nameless = _node(node_id="spark-4", node_name="unknown", peer_ip=None, fabric_ip="")
    payload = endpoint_payload(_app(_config(), nameless), _Request())
    row = next(n for n in payload["nodes"] if n["name"] == "unknown")
    assert row["host"] == ""
    assert row["url"] is None


def test_peer_host_precedence():
    """Source IP, then fabric IP, then name. Loopback and placeholders skipped."""
    assert peer_host(_node(peer_ip="10.0.0.2", fabric_ip="10.100.0.2")) == "10.0.0.2"
    assert peer_host(_node(peer_ip=None, fabric_ip="10.100.0.2")) == "10.100.0.2"
    assert peer_host(_node(peer_ip=None, fabric_ip="", node_name="spark-2")) == "spark-2"
    assert peer_host(_node(peer_ip="127.0.0.1", fabric_ip="10.100.0.2")) == "10.100.0.2"
    assert peer_host(_node(peer_ip=None, fabric_ip="", node_name="unknown")) == ""
    assert peer_host(_node(peer_ip=None, fabric_ip="", node_name=None)) == ""


def test_usable_host_rules():
    assert _is_usable_host("192.168.1.10")
    assert _is_usable_host("spark-1.local")
    assert not _is_usable_host("localhost")
    assert not _is_usable_host("LOCALHOST")
    assert not _is_usable_host("127.0.0.1")
    assert not _is_usable_host("127.53.0.1")
    assert not _is_usable_host("0.0.0.0")
    assert not _is_usable_host("::1")
    assert not _is_usable_host("[::1]")
    assert not _is_usable_host("")
    assert not _is_usable_host("  ")
    assert not _is_usable_host(None)
    assert not _is_usable_host(3000)


def test_endpoint_url_never_builds_a_loopback_url():
    assert endpoint_url("10.0.0.1", 3000) == "http://10.0.0.1:3000"
    assert endpoint_url("10.0.0.1", 0) == "http://10.0.0.1:3000"
    assert endpoint_url("localhost", 3000) is None
    assert endpoint_url("", 3000) is None


def test_the_real_lan_address_is_never_loopback():
    """The one test that lets the host answer: whatever it says is usable or "".

    Runs the real derivation (the netdev seams are neutered by the suite's
    isolate_netdev fixture, so this exercises the outbound-socket and hostname
    fallbacks a node with no readable interface list takes).
    """
    answer = lan_address(_config())
    assert answer == "" or _is_usable_host(answer), answer


# =============================================================================
# The cost of answering
# =============================================================================

def test_the_host_is_read_once_a_minute_not_once_a_request():
    """/api/status carries these rows, and the fleet polls it constantly.

    Deriving an address reads the host: an ``ip`` subprocess and a routing
    lookup. Uncached, every dashboard poll for every node paid for that on the
    event loop, which is how this turned a 55 second test suite into a 20 minute
    one before the cache went in.
    """
    calls = {"n": 0}

    def _counted():
        calls["n"] += 1
        return ["10.0.0.7"]

    original = server_routes._local_ipv4s
    server_routes._local_ipv4s = _counted
    try:
        config = _config()
        for _ in range(25):
            lan_address(config)
            server_routes.own_addresses(config)
        assert calls["n"] == 1, f"the host was read {calls['n']} times"

        # A renamed node does not wait out the TTL: the cache is keyed on the
        # fields that feed it.
        lan_address(_config(node_name="Renamed"))
        assert calls["n"] == 2
    finally:
        server_routes._local_ipv4s = original


def test_the_answer_survives_a_host_that_says_nothing():
    """No ip(8), no route, no hostname: an empty host and a null url, not a lie."""
    original = (server_routes._local_ipv4s, server_routes._outbound_ipv4)
    server_routes._local_ipv4s = lambda: []
    server_routes._outbound_ipv4 = lambda: ""
    try:
        payload = endpoint_payload(_app(_config(node_id="n", node_name=None)),
                                   _Request(host="localhost:3000"))
        assert payload["self"]["host"] in ("", socket.gethostname())
        if payload["self"]["host"] == "":
            assert payload["nodes"][0]["url"] is None
        assert "localhost" not in json.dumps(payload)
    finally:
        server_routes._local_ipv4s, server_routes._outbound_ipv4 = original


# =============================================================================
# Over a real server: the route, the auth exemption, and /api/status
# =============================================================================

@pytest_asyncio.fixture
async def client(endpoint_home, fixed_addresses):
    app = create_app(config=_config(), engine=None)
    async with TestClient(TestServer(app)) as c:
        yield c


@pytest.mark.asyncio
async def test_the_route_answers(client):
    resp = await client.get("/api/cluster/endpoint")
    assert resp.status == 200
    body = await resp.json()
    assert set(body) == {"self", "master", "nodes", "generated_at"}
    # `tls`, `tls_port` and `url` say which scheme to prefer; `port` is still the
    # HTTP port every peer uses. See test_tls.py for the rules behind them.
    assert set(body["self"]) == {"name", "host", "port", "tls", "tls_port", "url",
                                 "version", "role"}
    assert set(body["nodes"][0]) == {"name", "host", "port", "tls", "tls_port",
                                     "version", "role", "url"}


@pytest.mark.asyncio
async def test_it_answers_with_no_key_when_auth_is_on(client):
    """The point of the exemption: the client that needs an address has no key.

    Whatever key it was given belongs to the node that just went down, so a
    keyed-only endpoint is an endpoint it cannot ask.
    """
    client.app["auth_config"].enable()
    resp = await client.get("/api/cluster/endpoint")
    assert resp.status == 200
    assert (await resp.json())["self"]["host"] == OUR_ADDRESS
    # And the thing it is exempt alongside still needs one, so the exemption is
    # this one path and not a prefix.
    assert (await client.get("/api/cluster/info")).status == 401


@pytest.mark.asyncio
async def test_it_carries_nothing_secret(client):
    """Addresses only. That is what justifies answering without a key."""
    client.app["config"].model = "some-private/model-name"
    body = await (await client.get("/api/cluster/endpoint")).text()
    for leak in ("some-private", "api_key", "auth", "token", "gpu", "disk"):
        assert leak not in body.lower(), f"{leak} leaked into an unauthenticated body"


@pytest.mark.asyncio
async def test_status_hands_a_client_the_same_fallbacks(client):
    """A client that already polls status learns its fallbacks for free."""
    status = await (await client.get("/api/status")).json()
    endpoint = await (await client.get("/api/cluster/endpoint")).json()
    assert status["endpoint_hint"] == endpoint["nodes"]
    assert status["endpoint_hint"][0]["url"] == f"http://{OUR_ADDRESS}:3000"


def test_endpoint_nodes_survives_an_app_with_no_cluster_state():
    """Never 500 a status poll over this: no cluster state means no fallbacks."""
    assert endpoint_nodes({"config": _config()}, _Request()) == []
