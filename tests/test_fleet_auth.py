"""Auth that a CLUSTER can live with, and an install that turns it on.

Auth shipped usable by the dashboard (#218) and unusable by a fleet: with
``auth.enabled`` every request a node's own peers make answered 401, so the model
card lost the peer's launch flags, a fan-out unload reached nothing, and
``update-all`` could not update anybody. The fleet therefore ran open, which is
the state this file exists to make impossible to ship again.

Six things are pinned here, in this order:

1. **The derivation.** The fleet key is ``HMAC-SHA256(cluster_secret, label)``, so
   every node with the secret computes the same key, rotation follows the secret,
   and nothing new is written to disk. The label matters: the key must not BE the
   secret, which also signs discovery datagrams.
2. **Acceptance**, through the real middleware: the fleet key gets a caller in as
   ``fleet``, a different secret's key does not, and rotating the secret in
   ``config.json`` rotates acceptance with no restart.
3. **Every outbound node-to-node call carries it.** Behaviourally for the
   load/unload forwarder, the unload fan-out, the cluster update fan-out and the
   card's peer config read, and then by WALKING THE SOURCE: every line in the
   package that builds a peer URL is checked for the header helper, because the
   next fan-out somebody writes is the one that would have quietly 401'd.
4. **The installer protects a fresh node**, prints the key once, leaves an
   existing install alone, and says why when ``AINODE_AUTH=off``.
5. **An ``ainode auth`` change is live.** The middleware re-reads ``auth.json``
   when it changes, because the CLI writes that file and the server used to hold
   whatever it said at startup: ``ainode auth enable`` did nothing at all until
   somebody restarted the service, and nothing told them to.
6. **Neither file is world-readable**, and neither ``cluster_secret`` nor
   ``hf_token`` leaves the node through ``GET /api/config``.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import hmac
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

import ainode.api.server as server
from ainode.auth.fleet import (
    FLEET_KEY_ID,
    FLEET_KEY_LABEL,
    cluster_secret_of,
    fleet_headers,
    fleet_key,
    fleet_key_headers,
    is_fleet_key,
)
from ainode.auth.middleware import AuthConfig, identify_caller
from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeStatus
from ainode.discovery.cluster import ClusterNode, ClusterState
from ainode.discovery.signing import ClusterSecret

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "ainode"

SECRET = "0123456789abcdef0123456789abcdef"
OTHER_SECRET = "fedcba9876543210fedcba9876543210"


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def auth_home(tmp_path, monkeypatch):
    """Keep every file this app writes out of the operator's own ~/.ainode.

    Same redirect ``tests/test_auth_gate.py`` uses: most paths resolve
    AINODE_HOME at call time, but AUTH_FILE, CONFIG_FILE and SECRETS_FILE are
    computed at import.
    """
    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr("ainode.auth.middleware.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.auth.middleware.AUTH_FILE", tmp_path / "auth.json")
    monkeypatch.setattr("ainode.secrets.manager.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.secrets.manager.SECRETS_FILE", tmp_path / "secrets.json")
    return tmp_path


@pytest.fixture
def config():
    return NodeConfig(node_id="fleet-node", node_name="FleetNode",
                      cluster_secret=SECRET, onboarded=True)


@pytest.fixture
def app(config, auth_home):
    return server.create_app(config=config, engine=None)


@pytest_asyncio.fixture
async def protected(app):
    """Auth on, with one operator key, on a node that holds the cluster secret."""
    entry = app["auth_config"].enable("test")
    async with TestClient(TestServer(app)) as client:
        yield client, entry["key"]


def _node(node_id, fabric="", web_port=3000, api_port=8000, model=""):
    return ClusterNode(node_id=node_id, node_name=node_id, gpu_name="NVIDIA GB10",
                       gpu_memory_gb=128.0, unified_memory=True, model=model,
                       status=NodeStatus.ONLINE, api_port=api_port,
                       web_port=web_port, last_seen=0.0, fabric_ip=fabric)


def _fleet_app(node_id="spark1", nodes=(), secret=SECRET, **config_kwargs):
    """A plain-dict app, which is all the peer-call sites read."""
    cluster = ClusterState()
    for node in nodes:
        cluster.add_node(node)
    return {
        "config": NodeConfig(node_id=node_id, api_port=8000,
                             cluster_secret=secret, **config_kwargs),
        "cluster_state": cluster,
    }


class _Req:
    """The stub request shape the cluster handlers are tested with."""

    def __init__(self, app, body, authenticated=False, query=None):
        self.app = app
        self._body = body
        self._state = {"authenticated": authenticated}
        self.query = query or {}
        self.can_read_body = body is not None

    def get(self, key, default=None):
        return self._state.get(key, default)

    async def json(self):
        return self._body


# =============================================================================
# 1. The derivation
# =============================================================================

def test_the_fleet_key_is_an_hmac_of_the_label_under_the_secret():
    expected = hmac.new(SECRET.encode(), FLEET_KEY_LABEL, hashlib.sha256).hexdigest()
    assert fleet_key(SECRET) == expected
    assert len(fleet_key(SECRET)) == 64


def test_every_node_with_the_same_secret_derives_the_same_key():
    """The whole design: the join flow hands over the secret, and that is enough."""
    assert fleet_key(SECRET) == fleet_key(SECRET)
    assert fleet_key(SECRET) != fleet_key(OTHER_SECRET)


def test_the_fleet_key_is_not_the_secret_itself():
    """Domain separation: the secret also signs discovery, so it must not travel
    as a bearer token that any proxy log would capture."""
    assert fleet_key(SECRET) != SECRET
    assert SECRET not in fleet_key(SECRET)


def test_a_node_with_no_secret_has_no_fleet_key_and_accepts_none():
    for empty in ("", None):
        assert fleet_key(empty) == ""
        assert fleet_key_headers(empty) == {}
        assert is_fleet_key("anything", empty) is False
    # An empty token is never a match either, which is what stops "no secret on
    # either side" from reading as agreement.
    assert is_fleet_key("", SECRET) is False


def test_is_fleet_key_accepts_only_the_derived_key():
    assert is_fleet_key(fleet_key(SECRET), SECRET) is True
    assert is_fleet_key(fleet_key(OTHER_SECRET), SECRET) is False
    assert is_fleet_key(fleet_key(SECRET)[:-1] + "0", SECRET) is False


def test_fleet_headers_reads_the_live_provider_before_the_config(tmp_path):
    """``app["cluster_secret"]`` is the ClusterSecret discovery shares, so a
    rotation lands on the HTTP path at the same moment it lands on the UDP one."""
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"cluster_secret": SECRET}))
    app = {"config": NodeConfig(cluster_secret=OTHER_SECRET),
           "cluster_secret": ClusterSecret(None, path=config_file)}

    assert cluster_secret_of(app) == SECRET
    assert fleet_headers(app)["Authorization"] == f"Bearer {fleet_key(SECRET)}"

    config_file.write_text(json.dumps({"cluster_secret": OTHER_SECRET}))
    assert fleet_headers(app)["Authorization"] == f"Bearer {fleet_key(OTHER_SECRET)}"


def test_fleet_headers_falls_back_to_the_config_and_survives_a_bare_app():
    app = {"config": NodeConfig(cluster_secret=SECRET)}
    assert fleet_headers(app)["Authorization"] == f"Bearer {fleet_key(SECRET)}"
    assert fleet_headers({}) == {}
    assert cluster_secret_of({}) == ""


def test_fleet_headers_is_pure_and_never_overwrites_a_callers_credential():
    app = {"config": NodeConfig(cluster_secret=SECRET)}
    mine = {"authorization": "Bearer someone-elses-key"}
    out = fleet_headers(app, mine)
    assert out == mine, "a caller's own credential must reach the engine untouched"
    assert mine == {"authorization": "Bearer someone-elses-key"}, "input mutated"

    passthrough = {"Content-Type": "application/json"}
    out = fleet_headers(app, passthrough)
    assert out["Content-Type"] == "application/json"
    assert out["Authorization"] == f"Bearer {fleet_key(SECRET)}"
    assert "Authorization" not in passthrough


# =============================================================================
# 2. Acceptance
# =============================================================================

def test_identify_caller_names_a_peer_fleet(auth_home):
    cfg = AuthConfig()
    operator = cfg.generate_key("laptop")
    app = {"config": NodeConfig(cluster_secret=SECRET)}

    assert identify_caller(app, cfg, fleet_key(SECRET)) == (True, FLEET_KEY_ID)
    assert identify_caller(app, cfg, operator["key"]) == (True, operator["id"])
    assert identify_caller(app, cfg, fleet_key(OTHER_SECRET)) == (False, "")
    assert identify_caller(app, cfg, "") == (False, "")
    # No auth.json at all is still a node a peer can reach.
    assert identify_caller(app, None, fleet_key(SECRET)) == (True, FLEET_KEY_ID)


@pytest.mark.asyncio
async def test_a_peer_with_the_fleet_key_is_let_in(protected):
    client, _operator_key = protected
    resp = await client.get("/api/status")
    assert resp.status == 401, "the node must require a key at all"

    resp = await client.get(
        "/api/status", headers={"Authorization": f"Bearer {fleet_key(SECRET)}"})
    assert resp.status == 200
    body = await resp.json()
    assert body["auth"]["enabled"] is True


@pytest.mark.asyncio
async def test_a_node_from_another_cluster_is_refused(protected):
    """A different secret is a different fleet, over HTTP exactly as over UDP."""
    client, _ = protected
    resp = await client.get(
        "/api/status",
        headers={"Authorization": f"Bearer {fleet_key(OTHER_SECRET)}"})
    assert resp.status == 401
    body = await resp.json()
    assert body["error"]["type"] == "auth_error"


@pytest.mark.asyncio
async def test_the_operators_own_key_still_works(protected):
    client, operator_key = protected
    resp = await client.get("/api/status",
                            headers={"Authorization": f"Bearer {operator_key}"})
    assert resp.status == 200


@pytest.mark.asyncio
async def test_a_mutating_route_takes_the_fleet_key(protected):
    """The fan-out targets are mutating routes, so this is the case that matters."""
    client, _ = protected
    resp = await client.post(
        "/api/models/unload", json={},
        headers={"Authorization": f"Bearer {fleet_key(SECRET)}"})
    assert resp.status != 401


@pytest.mark.asyncio
async def test_rotating_the_secret_rotates_acceptance_with_no_restart(app, tmp_path):
    """Roll cluster_secret and the fleet key rolls with it, mid-process."""
    app["auth_config"].enable("test")
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"cluster_secret": SECRET}))
    app["cluster_secret"] = ClusterSecret(app["config"], path=config_file)

    async with TestClient(TestServer(app)) as client:
        old = {"Authorization": f"Bearer {fleet_key(SECRET)}"}
        new = {"Authorization": f"Bearer {fleet_key(OTHER_SECRET)}"}
        assert (await client.get("/api/status", headers=old)).status == 200
        assert (await client.get("/api/status", headers=new)).status == 401

        config_file.write_text(json.dumps({"cluster_secret": OTHER_SECRET}))

        assert (await client.get("/api/status", headers=new)).status == 200
        assert (await client.get("/api/status", headers=old)).status == 401


@pytest.mark.asyncio
async def test_auth_off_still_stamps_a_peer_as_authenticated(app):
    """``trust_remote_code`` is gated on the stamp even with auth off, and a load
    forwarded by the cluster is an authenticated act: the head is where the
    operator presented a key."""
    async with TestClient(TestServer(app)) as client:
        resp = await client.patch(
            "/api/config", json={"trust_remote_code": True},
            headers={"Authorization": f"Bearer {fleet_key(SECRET)}"})
        body = await resp.json()
        assert body["applied"].get("trust_remote_code") is True

        resp = await client.patch("/api/config", json={"trust_remote_code": True})
        body = await resp.json()
        assert "trust_remote_code" in body["rejected"]


# =============================================================================
# 3. Every outbound node-to-node call carries the key
# =============================================================================

class _Upstream:
    status = 200
    headers = {"Content-Type": "application/json"}

    def __init__(self, payload=b"{}"):
        self._payload = payload

    async def read(self):
        return self._payload

    async def json(self):
        return json.loads(self._payload.decode())

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _RecordingSession:
    """Records every request, so a test can assert on the headers that went out."""

    def __init__(self, payload=b"{}"):
        self.calls = []
        self._payload = payload

    def _record(self, method, url, headers):
        self.calls.append({"method": method, "url": url, "headers": headers or {}})
        return _Upstream(self._payload)

    def post(self, url, json=None, timeout=None, headers=None, **kwargs):
        return self._record("POST", url, headers)

    def get(self, url, timeout=None, headers=None, **kwargs):
        return self._record("GET", url, headers)


def _authorization(call) -> str:
    return {k.lower(): v for k, v in call["headers"].items()}.get("authorization", "")


def test_the_load_forwarder_carries_the_fleet_key():
    app = _fleet_app(nodes=[_node("spark1", "10.100.0.11"),
                            _node("spark3", "10.100.0.15")])
    session = _RecordingSession(b'{"status":"launching"}')
    app["client_session"] = session

    asyncio.run(server.handle_cluster_load(
        _Req(app, {"node_id": "spark3", "model": "B"})))

    assert session.calls[0]["url"] == "http://10.100.0.15:3000/api/models/load"
    assert _authorization(session.calls[0]) == f"Bearer {fleet_key(SECRET)}"


def test_the_unload_forwarder_carries_the_fleet_key():
    app = _fleet_app(nodes=[_node("spark1", "10.100.0.11"),
                            _node("spark3", "10.100.0.15")])
    session = _RecordingSession()
    app["client_session"] = session

    asyncio.run(server.handle_cluster_unload(
        _Req(app, {"node_id": "spark3", "model": "B"})))

    assert session.calls[0]["url"] == "http://10.100.0.15:3000/api/models/unload"
    assert _authorization(session.calls[0]) == f"Bearer {fleet_key(SECRET)}"


def test_a_node_with_no_secret_forwards_exactly_as_it_did():
    """No secret, no header: a fleet that never set one behaves as before."""
    app = _fleet_app(nodes=[_node("spark1", "10.100.0.11"),
                            _node("spark3", "10.100.0.15")], secret="")
    session = _RecordingSession(b'{"status":"launching"}')
    app["client_session"] = session

    asyncio.run(server.handle_cluster_load(
        _Req(app, {"node_id": "spark3", "model": "B"})))

    assert _authorization(session.calls[0]) == ""


def test_the_unload_fanout_carries_the_fleet_key(monkeypatch, tmp_path):
    """``POST /api/models/unload {"all": true}`` reaches every peer as the fleet."""
    from ainode.models import api_routes

    app = _fleet_app(nodes=[_node("spark1", "10.100.0.11"),
                            _node("spark3", "10.100.0.15", model="B"),
                            _node("spark4", "10.100.0.16", model="B")])
    session = _RecordingSession(b'{"stopped": true, "errors": []}')
    app["client_session"] = session
    app["instances"] = None
    app["engine"] = None
    monkeypatch.setattr(api_routes, "save_instance_manifest", lambda _app: None)

    resp = asyncio.run(api_routes.handle_model_unload(
        _Req(app, {"model": "B", "all": True})))
    assert json.loads(resp.body)["scope"] == "remote-fanout"

    urls = sorted(call["url"] for call in session.calls)
    assert urls == ["http://10.100.0.15:3000/api/models/unload?fanout=0",
                    "http://10.100.0.16:3000/api/models/unload?fanout=0"]
    for call in session.calls:
        assert _authorization(call) == f"Bearer {fleet_key(SECRET)}"


def test_the_cards_peer_config_read_carries_the_fleet_key():
    from ainode.api import chat_routes

    app = _fleet_app(nodes=[_node("spark3", "10.100.0.15")])
    session = _RecordingSession(json.dumps({"model": "B"}).encode())
    entry = {"host": "10.100.0.15", "web_port": 3000, "model": "B"}

    out = asyncio.run(chat_routes._remote_node_config(app, session, entry))
    assert out == {"model": "B"}
    assert session.calls[0]["url"] == "http://10.100.0.15:3000/api/config"
    assert _authorization(session.calls[0]) == f"Bearer {fleet_key(SECRET)}"


@pytest.mark.asyncio
async def test_the_cluster_update_fanout_carries_the_fleet_key(monkeypatch, tmp_path):
    """``update-all`` POSTs /api/engine/update to each peer, which needs the key.

    The self half is pinned to the pre-swappable-unit branch on purpose: it stops
    after writing image.env, so this test never creates the task that would
    ``docker stop ainode`` on the machine running the suite. The subprocess seam
    is a recorder for the same reason.
    """
    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_fetch_latest_ghcr_tag", lambda: "9.9.9")
    monkeypatch.setattr(server, "_unit_is_swappable", lambda: False)
    monkeypatch.setattr(server, "_cluster_update_state", {})
    ran = []

    class _Done:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(argv, **kwargs):
        ran.append(list(argv))
        return _Done()

    monkeypatch.setattr(subprocess, "run", _fake_run)

    app = _fleet_app(nodes=[_node("spark1", "10.100.0.11"),
                            _node("spark3", "10.100.0.15")])
    app["config"].web_port = 3000
    session = _RecordingSession(b'{"message": "updating"}')
    app["client_session"] = session

    resp = await server.handle_cluster_update_all(_Req(app, {}))
    assert resp.status == 202

    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in pending:
        await asyncio.wait_for(task, timeout=10)

    posts = [c for c in session.calls if c["url"].endswith("/api/engine/update")]
    assert posts, f"no peer was asked to update; ran {ran}"
    for call in posts:
        assert _authorization(call) == f"Bearer {fleet_key(SECRET)}"


def test_the_bench_placement_read_carries_the_fleet_key():
    """The in-product bench reads a node's own /api/status and /api/config."""
    from ainode.bench import fleet as bench_fleet

    app = _fleet_app(nodes=[_node("spark3", "10.100.0.15", model="B")])
    session = _RecordingSession(json.dumps({"version": "9.9.9"}).encode())
    app["client_session"] = session
    target = bench_fleet.BenchTarget(model="B", host="10.100.0.15", port=8000,
                                     node_id="spark3", node_name="spark3",
                                     web_port=3000, is_local=False)

    asyncio.run(bench_fleet.busy_warnings(app, target))

    assert session.calls[0]["url"] == "http://10.100.0.15:3000/api/server/status"
    assert _authorization(session.calls[0]) == f"Bearer {fleet_key(SECRET)}"


def test_the_doctor_reads_its_own_node_and_its_peers_as_the_fleet(monkeypatch):
    """A node that requires a key would otherwise answer 401 to its own doctor,
    and a 401 is indistinguishable from a node that is down."""
    from ainode.cli import doctor

    seen = []

    def _fake_http_json(url, timeout=3.0, headers=None):
        seen.append({"url": url, "headers": headers or {}})
        if url.endswith("/api/nodes"):
            return {"nodes": [{"node_id": "spark1"},
                              {"node_id": "spark3", "node_name": "Spark-3",
                               "fabric_ip": "10.100.0.15", "web_port": 3000}]}
        return {"version": "9.9.9"}

    monkeypatch.setattr(doctor, "http_json", _fake_http_json)
    doctor.check_peers(NodeConfig(node_id="spark1", cluster_secret=SECRET), "9.9.9")

    assert [call["url"] for call in seen] == [
        "http://127.0.0.1:3000/api/nodes",
        "http://10.100.0.15:3000/api/status",
    ]
    for call in seen:
        assert _authorization(call) == f"Bearer {fleet_key(SECRET)}"


# ----------------------------------------------------------------- source walk

#: How a node-to-node URL is recognised in the source: an f-string that names a
#: peer's AINode port or node URL, or one that spells an ``/api/`` path on a host
#: this node is reaching out to. Engine calls (``/v1/...`` on an ``api_port``) are
#: deliberately NOT here: a vLLM container never sees AINode's middleware.
def _looks_like_a_peer_url(line: str) -> bool:
    if "f\"" not in line and "f'" not in line:
        return False
    if "web_port" in line and "http://" in line:
        return True
    return "/api/" in line and ("http://" in line or "node_url" in line)


#: Sites that must NOT send the fleet key, each with the reason. A joining node
#: does not have this cluster's secret yet, which is the whole point of the join
#: token (``api/cluster_join.py``), so the one route it calls is keyless.
EXEMPT_PEER_CALLS = {
    ("ainode/cluster/join.py", "join_url"):
        "a joiner holds no cluster secret yet; the join token is the credential",
    ("ainode/bench/fleet.py", "node_url"):
        "builds a base URL; its three readers each send the key",
    ("ainode/cli/main.py", "cmd_start"):
        "prints the local dashboard URL in a table; makes no request",
    ("ainode/cli/main.py", "cmd_status"):
        "prints the local dashboard URL in a table; makes no request",
}

#: Names that count as sending the key. Both come from ``ainode/auth/fleet.py``.
HEADER_HELPERS = ("fleet_headers(", "fleet_key_headers(")


def _enclosing_function(tree, lineno):
    """The innermost function whose body spans ``lineno``."""
    best = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(node, "end_lineno", node.lineno)
        if node.lineno <= lineno <= end:
            if best is None or node.lineno > best.lineno:
                best = node
    return best


def test_every_peer_call_site_in_the_package_sends_the_fleet_key():
    """Walk the source, not a list: the next fan-out is covered the day it lands.

    A list of call sites in a test is the thing that drifts. This finds them the
    way a reviewer would, then insists the function holding one names the header
    helper, so a new peer call that would silently 401 on an authenticated fleet
    fails here instead.
    """
    sites = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "/bench/harness/tasks/" in path.as_posix():
            continue  # vendored upstream task files, not our code
        text = path.read_text()
        lines = text.splitlines()
        hits = [i + 1 for i, line in enumerate(lines) if _looks_like_a_peer_url(line)]
        if not hits:
            continue
        tree = ast.parse(text)
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno in hits:
            func = _enclosing_function(tree, lineno)
            assert func is not None, f"{rel}:{lineno} builds a peer URL outside a function"
            body = ast.get_source_segment(text, func) or ""
            sites.append((rel, func.name, lineno, body))

    assert len(sites) >= 8, f"the walk found only {len(sites)} peer call sites"

    missing = []
    for rel, func_name, lineno, body in sites:
        if (rel, func_name) in EXEMPT_PEER_CALLS:
            assert not any(h in body for h in HEADER_HELPERS), (
                f"{rel}::{func_name} is listed exempt but sends the fleet key")
            continue
        if not any(helper in body for helper in HEADER_HELPERS):
            missing.append(f"{rel}:{lineno} ({func_name})")

    assert not missing, (
        "these node-to-node calls do not send the fleet key, so they answer 401 on "
        "a fleet with auth on: " + ", ".join(missing) +
        ". Pass headers=fleet_headers(app) (or fleet_key_headers(secret) off an "
        "app), or add the site to EXEMPT_PEER_CALLS with its reason.")


def test_the_walk_actually_sees_the_known_call_sites():
    """A guard on the guard: if the pattern stops matching, the test above passes
    vacuously. These five are the fan-outs the fleet depends on."""
    expected = {
        "ainode/api/server.py",         # the load/unload forwarder, update-all
        "ainode/models/api_routes.py",  # the unload fan-out
        "ainode/api/chat_routes.py",    # the card's peer config read
        "ainode/bench/fleet.py",        # the bench's node reads
        "ainode/cli/doctor.py",         # the doctor's own node and its peers
    }
    found = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        if any(_looks_like_a_peer_url(line) for line in path.read_text().splitlines()):
            found.add(path.relative_to(REPO_ROOT).as_posix())
    assert expected <= found, f"the peer-URL pattern stopped matching {expected - found}"


# =============================================================================
# 4. The installer
# =============================================================================

INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _render_install(tmp_path: Path, *args: str, env_extra: dict | None = None):
    """The real installer in --dry-run against a throwaway HOME.

    The same helper ``tests/test_fresh_install.py`` uses, kept here rather than
    imported so the two files cannot break each other; --dry-run renders
    config.json, auth.json, the unit and the wrapper and touches nothing else.
    """
    home = tmp_path / "home"
    ainode_home = home / ".ainode"
    home.mkdir(parents=True, exist_ok=True)
    sysfs = tmp_path / "sys-class-net"
    sysfs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(HOME=str(home), AINODE_HOME=str(ainode_home),
               AINODE_IMAGE="ghcr.io/getainode/ainode:9.9.9",
               SYS_CLASS_NET=str(sysfs))
    env.pop("AINODE_PEERS", None)
    env.pop("HF_TOKEN", None)
    env.pop("AINODE_AUTH", None)
    env.update(env_extra or {})
    proc = subprocess.run(["bash", str(INSTALL_SH), "--dry-run", *args],
                          capture_output=True, text=True, timeout=120, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return ainode_home, proc


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
class TestInstallerTurnsAuthOn:
    def test_a_fresh_install_requires_a_key_and_stores_only_its_hash(self, tmp_path):
        ainode_home, proc = _render_install(tmp_path)
        auth = json.loads((ainode_home / "auth.json").read_text())

        assert auth["enabled"] is True
        assert len(auth["api_keys"]) == 1
        entry = auth["api_keys"][0]
        assert entry["name"] == "installer"
        assert len(entry["key_hash"]) == 64
        assert "key" not in entry, "the installer must never store a plaintext key"

        # The key in the box is the key that was stored, and it is the only copy.
        printed = _printed_key(proc.stdout)
        assert hashlib.sha256(printed.encode()).hexdigest() == entry["key_hash"]

    def test_both_credential_files_are_written_owner_only(self, tmp_path):
        """config.json carries the cluster secret; auth.json carries the keys."""
        ainode_home, _ = _render_install(tmp_path)
        for name in ("auth.json", "config.json"):
            mode = stat.S_IMODE((ainode_home / name).stat().st_mode)
            assert mode == 0o600, f"{name} is {oct(mode)}"

    def test_the_key_is_printed_once_in_a_box_that_says_what_to_do(self, tmp_path):
        _, proc = _render_install(tmp_path)
        out = proc.stdout

        assert "YOUR API KEY. SHOWN ONCE, STORED HASHED. COPY IT NOW." in out
        assert "Config > API access" in out
        assert "ainode auth key create --name <client>" in out
        assert out.count(_printed_key(out)) == 2, (
            "the key belongs in the box and in the curl line beside it, nowhere else")

    def test_the_summary_line_says_the_api_is_protected(self, tmp_path):
        _, proc = _render_install(tmp_path)
        assert "API protected, one key" in proc.stdout
        assert "API open, no key set" not in proc.stdout

    def test_auth_off_keeps_the_old_behaviour_and_says_why(self, tmp_path):
        ainode_home, proc = _render_install(
            tmp_path, env_extra={"AINODE_AUTH": "off"})

        assert not (ainode_home / "auth.json").exists()
        assert "API open, no key set" in proc.stdout
        assert "AINODE_AUTH=off" in proc.stdout
        # The choice is stated, not just obeyed.
        assert "can load and unload" in proc.stdout
        assert "ainode auth enable" in proc.stdout

    def test_an_install_over_an_existing_node_changes_nothing(self, tmp_path):
        ainode_home, first = _render_install(tmp_path)
        before = (ainode_home / "auth.json").read_text()

        _, second = _render_install(tmp_path)

        assert (ainode_home / "auth.json").read_text() == before
        assert "YOUR API KEY" not in second.stdout
        assert "not a fresh install" in second.stdout

    def test_an_unreadable_auth_value_is_refused_before_anything_is_written(
            self, tmp_path):
        home = tmp_path / "home"
        home.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(HOME=str(home), AINODE_HOME=str(home / ".ainode"),
                   AINODE_IMAGE="ghcr.io/getainode/ainode:9.9.9",
                   AINODE_AUTH="maybe")
        proc = subprocess.run(["bash", str(INSTALL_SH), "--dry-run"],
                              capture_output=True, text=True, timeout=120, env=env)
        assert proc.returncode == 2
        assert "AINODE_AUTH must be on or off" in proc.stderr

    def test_the_installer_is_syntactically_valid(self):
        proc = subprocess.run(["bash", "-n", str(INSTALL_SH)],
                              capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr

    def test_the_update_wrapper_verifies_on_a_route_that_needs_no_key(self, tmp_path):
        """With auth on, reading /api/status here would 401 and every update
        would report that it had not applied."""
        ainode_home, _ = _render_install(tmp_path)
        wrapper = (ainode_home / "ainode-wrapper").read_text()

        verify = [ln for ln in wrapper.splitlines() if "STATUS_URL=" in ln]
        assert verify, "the wrapper no longer builds a URL to verify against"
        for line in verify:
            assert "/api/health" in line, line
            assert "/api/status" not in line, line


def _printed_key(stdout: str) -> str:
    """The 32-hex key out of the installer's box."""
    import re

    keys = re.findall(r"\b[0-9a-f]{32}\b", stdout)
    # config.json carries a 64-hex cluster_secret, never a 32-hex one, so the
    # only 32-hex token in this output is the API key.
    assert keys, f"no API key in the installer output:\n{stdout}"
    assert len(set(keys)) == 1, f"more than one key printed: {set(keys)}"
    return keys[0]


# =============================================================================
# 5. The CLI
# =============================================================================

def _run_cli(*argv):
    with patch.object(sys, "argv", ["ainode", *argv]):
        from ainode.cli.main import main

        main()


@pytest.mark.usefixtures("auth_home")
class TestAuthKeyCommands:
    def test_create_names_the_key_and_prints_it_once(self, capsys):
        _run_cli("auth", "key", "create", "--name", "laptop")
        out = capsys.readouterr().out

        assert "New API key created" in out
        assert "laptop" in out
        assert "Shown once, stored hashed" in out

        keys = AuthConfig.load().api_keys
        assert len(keys) == 1
        assert keys[0]["name"] == "laptop"
        assert keys[0]["created_at"].endswith("Z")

    def test_list_says_which_client_each_key_is_for(self, capsys):
        _run_cli("auth", "key", "create", "--name", "laptop")
        _run_cli("auth", "key", "create", "--name", "n8n")
        capsys.readouterr()

        _run_cli("auth", "key", "list")
        out = capsys.readouterr().out

        assert "laptop" in out and "n8n" in out
        for entry in AuthConfig.load().api_keys:
            assert entry["id"] in out
            assert entry["key_hash"] not in out, "a hash is not for printing"

    def test_list_on_an_empty_node_says_how_to_make_one(self, capsys):
        _run_cli("auth", "key", "list")
        out = capsys.readouterr().out
        assert "No keys on this node" in out
        assert "auth key create" in out

    def test_revoke_takes_the_id_and_reports_what_is_left(self, capsys):
        _run_cli("auth", "key", "create", "--name", "laptop")
        _run_cli("auth", "key", "create", "--name", "n8n")
        target = AuthConfig.load().api_keys[0]["id"]
        capsys.readouterr()

        _run_cli("auth", "key", "revoke", target)
        out = capsys.readouterr().out

        assert f"Revoked key {target}" in out
        assert [k["name"] for k in AuthConfig.load().api_keys] == ["n8n"]

    def test_revoking_the_last_key_on_a_protected_node_says_so(self, capsys):
        _run_cli("auth", "enable", "--name", "laptop")
        target = AuthConfig.load().api_keys[0]["id"]
        capsys.readouterr()

        _run_cli("auth", "key", "revoke", target)
        out = capsys.readouterr().out
        assert "last key" in out
        assert "auth key create" in out

    def test_revoking_something_that_is_not_there_is_not_a_crash(self, capsys):
        _run_cli("auth", "key", "revoke", "nope")
        out = capsys.readouterr().out
        assert "No key with id 'nope'" in out

    def test_a_key_created_with_no_name_still_lists(self, capsys):
        _run_cli("auth", "key", "create")
        capsys.readouterr()
        _run_cli("auth", "key", "list")
        assert "unnamed" in capsys.readouterr().out

    def test_enable_names_its_first_key(self, capsys):
        _run_cli("auth", "enable")
        out = capsys.readouterr().out
        assert "Auth enabled" in out
        cfg = AuthConfig.load()
        assert cfg.enabled is True
        assert cfg.api_keys[0]["name"] == "first key"

    def test_status_says_when_peers_cannot_authenticate(self, capsys, auth_home):
        """Auth on with no cluster_secret is the state that breaks a fleet."""
        (auth_home / "config.json").write_text(json.dumps({"node_name": "n"}))
        _run_cli("auth", "enable")
        capsys.readouterr()

        _run_cli("auth", "status")
        out = capsys.readouterr().out
        assert "No cluster_secret" in out
        assert "401" in out

    def test_status_says_peers_are_covered_when_there_is_a_secret(
            self, capsys, auth_home):
        (auth_home / "config.json").write_text(
            json.dumps({"node_name": "n", "cluster_secret": SECRET}))
        _run_cli("auth", "enable")
        capsys.readouterr()

        _run_cli("auth", "status")
        assert "derived from cluster_secret" in capsys.readouterr().out


# =============================================================================
# 6. The doctor
# =============================================================================

def _write_auth(home: Path, enabled: bool, keys: int = 1) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(json.dumps({
        "enabled": enabled,
        "api_keys": [{"id": f"k{i}", "key_hash": "0" * 64, "name": "installer"}
                     for i in range(keys)],
    }))


class TestDoctorAuthCheck:
    def _check(self, config, home, peers_seen=0):
        from ainode.cli import doctor

        checks = doctor.check_auth(config, home, peers_seen)
        assert len(checks) == 1
        assert checks[0].name == "auth.state"
        return checks[0]

    def test_auth_on_with_peers_and_no_secret_fails(self, tmp_path):
        from ainode.cli.doctor import FAIL

        _write_auth(tmp_path, True)
        check = self._check(NodeConfig(cluster_secret=""), tmp_path, peers_seen=2)

        assert check.status == FAIL
        assert "no cluster_secret" in check.detail
        assert "401" in check.detail
        assert "ainode join" in check.fix
        assert check.data == {"enabled": True, "key_count": 1, "cluster_secret": False,
                              "peers": True, "peers_seen": 2, "host": "0.0.0.0"}

    def test_configured_peers_count_even_before_discovery_sees_them(self, tmp_path):
        from ainode.cli.doctor import FAIL

        _write_auth(tmp_path, True)
        check = self._check(NodeConfig(cluster_secret="", peer_ips=["10.0.0.2"]),
                            tmp_path)
        assert check.status == FAIL

    def test_auth_on_with_peers_and_a_secret_is_ok(self, tmp_path):
        from ainode.cli.doctor import OK

        _write_auth(tmp_path, True)
        check = self._check(NodeConfig(cluster_secret=SECRET), tmp_path, peers_seen=3)

        assert check.status == OK
        assert "a key is required" in check.detail
        assert "cluster_secret" in check.detail

    def test_auth_on_alone_with_no_secret_is_ok(self, tmp_path):
        """A solo node with no peers has nothing to federate with, so a missing
        secret is not a finding: the WARN would fire on every laptop."""
        from ainode.cli.doctor import OK

        _write_auth(tmp_path, True)
        assert self._check(NodeConfig(cluster_secret=""), tmp_path).status == OK

    def test_auth_off_on_a_routable_bind_warns(self, tmp_path):
        from ainode.cli.doctor import WARN

        check = self._check(NodeConfig(host="0.0.0.0"), tmp_path)

        assert check.status == WARN
        assert "0.0.0.0" in check.detail
        assert "ainode auth enable" in check.fix

    def test_auth_off_on_loopback_is_ok(self, tmp_path):
        from ainode.cli.doctor import OK

        check = self._check(NodeConfig(host="127.0.0.1"), tmp_path)
        assert check.status == OK
        assert "nothing off this host" in check.detail

    def test_a_node_with_keys_that_does_not_require_one_still_warns(self, tmp_path):
        from ainode.cli.doctor import WARN

        _write_auth(tmp_path, False, keys=2)
        check = self._check(NodeConfig(), tmp_path)
        assert check.status == WARN
        assert check.data["key_count"] == 2

    def test_an_unreadable_auth_file_is_a_warn_not_a_crash(self, tmp_path):
        from ainode.cli.doctor import WARN

        (tmp_path / "auth.json").write_text("{not json")
        check = self._check(NodeConfig(), tmp_path)
        assert check.status == WARN
        assert "cannot read" in check.detail

    def test_the_check_is_in_the_doctors_run(self, tmp_path, monkeypatch):
        from ainode.cli import doctor

        monkeypatch.setattr(doctor, "http_json", lambda *a, **k: None)
        monkeypatch.setattr(doctor, "run_command", lambda *a, **k: (127, "no"))
        monkeypatch.setattr(doctor, "probe_gpus", lambda: [])
        monkeypatch.setattr(doctor, "latest_image_tag", lambda: None)
        _write_auth(tmp_path, True)
        (tmp_path / "config.json").write_text(json.dumps({"cluster_secret": SECRET}))

        names = [c.name for c in doctor.run_checks(home=tmp_path)]
        assert names.count("auth.state") == 1


# =============================================================================
# 7. A CLI change is live, and the files are not world-readable
# =============================================================================

@pytest.mark.asyncio
async def test_enabling_auth_from_the_cli_takes_effect_with_no_restart(app, auth_home):
    """The file is the interface between the CLI and the server.

    `ainode auth enable` runs in the same container over the same bind mount and
    writes auth.json. Before the middleware re-read it, the running server kept
    the state it loaded at create_app, so the node stayed wide open and the CLI
    said "Auth enabled."
    """
    async with TestClient(TestServer(app)) as client:
        assert (await client.get("/api/status")).status == 200, "starts open"

        # What the CLI does, in the CLI's own words.
        cli_side = AuthConfig.load()
        entry = cli_side.enable("laptop")

        resp = await client.get("/api/status")
        assert resp.status == 401, "the running node ignored the CLI's change"

        resp = await client.get(
            "/api/status", headers={"Authorization": f"Bearer {entry['key']}"})
        assert resp.status == 200

        # And back off again, live.
        AuthConfig.load().disable()
        assert (await client.get("/api/status")).status == 200


@pytest.mark.asyncio
async def test_a_key_revoked_from_the_cli_stops_working(app, auth_home):
    async with TestClient(TestServer(app)) as client:
        cli_side = AuthConfig.load()
        entry = cli_side.enable("laptop")
        keyed = {"Authorization": f"Bearer {entry['key']}"}
        assert (await client.get("/api/status", headers=keyed)).status == 200

        revoked_from = AuthConfig.load()
        assert revoked_from.revoke_key(entry["id"]) is True

        assert (await client.get("/api/status", headers=keyed)).status == 401
        # The fleet key is not stored, so revoking an operator key cannot take
        # the cluster's own access away with it.
        assert (await client.get(
            "/api/status",
            headers={"Authorization": f"Bearer {fleet_key(SECRET)}"})).status == 200


def test_a_half_written_auth_file_never_opens_the_node(auth_home):
    """The tolerant direction is "keep what I have", never "let everybody in"."""
    cfg = AuthConfig.load()
    cfg.enable("laptop")

    (auth_home / "auth.json").write_text('{"enabled": tr')
    assert cfg.reload_if_changed() is False
    assert cfg.enabled is True

    (auth_home / "auth.json").unlink()
    assert cfg.reload_if_changed() is False
    assert cfg.enabled is True


def test_reload_is_a_no_op_when_the_file_has_not_changed(auth_home, monkeypatch):
    cfg = AuthConfig.load()
    cfg.enable("laptop")

    reads = []
    real_read = Path.read_text

    def _counted(self, *args, **kwargs):
        if self.name == "auth.json":
            reads.append(self)
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _counted)
    for _ in range(5):
        assert cfg.reload_if_changed() is False
    assert reads == [], "a request that changed nothing must not parse the store"


def test_the_auth_store_is_written_0600(auth_home):
    cfg = AuthConfig.load()
    cfg.generate_key("laptop")
    mode = stat.S_IMODE((auth_home / "auth.json").stat().st_mode)
    assert mode == 0o600, oct(mode)
    assert not (auth_home / "auth.json.tmp").exists(), "the temp file was left behind"


def test_an_existing_world_readable_store_is_tightened_on_load(auth_home):
    """Every install before this release wrote it under the default umask."""
    path = auth_home / "auth.json"
    path.write_text(json.dumps({"enabled": True, "api_keys": []}))
    os.chmod(path, 0o644)

    cfg = AuthConfig.load()

    assert cfg.enabled is True
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_config_json_is_written_0600_and_atomically(auth_home, monkeypatch):
    """It carries cluster_secret, and ClusterSecret stats it per datagram."""
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", auth_home / "config.json")
    from ainode.core import config as config_module

    NodeConfig(cluster_secret=SECRET, hf_token="hf_secret").save()
    path = auth_home / "config.json"

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not (auth_home / "config.json.tmp").exists()
    assert json.loads(path.read_text())["cluster_secret"] == SECRET

    os.chmod(path, 0o644)
    assert config_module.NodeConfig.load().cluster_secret == SECRET
    assert stat.S_IMODE(path.stat().st_mode) == 0o600, "load did not tighten it"


@pytest.mark.asyncio
async def test_get_config_scrubs_every_credential(app, auth_home):
    """`hf_token` can write to the operator's own Hugging Face repositories."""
    app["config"].cluster_secret = SECRET
    app["config"].hf_token = "hf_thisisacredential"

    async with TestClient(TestServer(app)) as client:
        body = await (await client.get("/api/config")).json()

    assert "cluster_secret" not in body
    assert "hf_token" not in body
    assert "hf_thisisacredential" not in json.dumps(body)
    # The fields a peer's card read actually needs are still there.
    assert "engine_backend" in body and "gpu_memory_utilization" in body


def test_the_scrub_list_names_both_credentials():
    from ainode.api.server import SCRUBBED_CONFIG_KEYS

    assert set(SCRUBBED_CONFIG_KEYS) == {"cluster_secret", "hf_token"}
    # Anything else in the dataclass that reads like a credential belongs here.
    fields = set(NodeConfig().__dataclass_fields__)
    suspicious = {f for f in fields
                  if any(word in f for word in ("secret", "token", "password", "key"))}
    assert suspicious - set(SCRUBBED_CONFIG_KEYS) <= {"api_key_header"}, (
        f"new credential-shaped config fields are not scrubbed: {suspicious}")


@pytest.mark.usefixtures("auth_home")
def test_the_cli_says_the_change_is_live(capsys):
    from ainode.cli.main import AUTH_LIVE_NOTE

    for argv in (("auth", "enable"), ("auth", "disable"),
                 ("auth", "key", "create", "--name", "laptop")):
        _run_cli(*argv)
        out = capsys.readouterr().out
        assert "re-reads auth.json" in out, f"{argv} did not say the change is live"
    assert "no restart needed" in AUTH_LIVE_NOTE


# =============================================================================
# 8. The copy that describes the rule matches the rule
# =============================================================================

#: Keyless paths whose sentence in the dashboard's API access panel lands with the
#: FRONT-END half of #261 (``ainode/web/static/js/app.js``, a different worker's
#: file in the same wave, PR 264). Only these two are excused, and only until that
#: PR lands: once the panel names them, DELETE them from this set, because every
#: entry left here is a keyless path this test has stopped checking. The set is
#: asserted to be a subset of SKIP_PATHS, so it cannot outlive the paths it names.
COPY_PENDING_IN_THE_PANEL = {"/api/auth/login", "/api/auth/me"}


def test_the_dashboard_panel_lists_every_keyless_path():
    """One rule, one wording: the panel used to name three of the six."""
    from ainode.auth.middleware import SKIP_PATHS, SKIP_PREFIXES

    panel = (REPO_ROOT / "ainode" / "web" / "static" / "js" / "app.js").read_text()
    section = [line for line in panel.splitlines()
               if "config-section-desc" in line and "Who may call this node" in line]
    assert len(section) == 1, "the API access panel's description moved"
    text = section[0]
    assert COPY_PENDING_IN_THE_PANEL <= SKIP_PATHS, (
        "COPY_PENDING_IN_THE_PANEL names a path that is no longer keyless: "
        f"{COPY_PENDING_IN_THE_PANEL - SKIP_PATHS}. Delete those entries.")
    missing = [path for path in sorted(SKIP_PATHS - COPY_PENDING_IN_THE_PANEL)
               if f"<code>{path}</code>" not in text]
    assert not missing, (
        f"these paths answer with no key and the dashboard's API access panel does "
        f"not name them: {', '.join(missing)}. Add each as <code>the path</code> to "
        f"the 'Who may call this node' paragraph in "
        f"ainode/web/static/js/app.js, with the reason it is open.")
    for prefix in SKIP_PREFIXES:
        assert prefix.rstrip("/") in text, f"{prefix} is keyless and unmentioned"


def test_the_join_route_docstring_lists_the_same_set():
    from ainode.api import cluster_join
    from ainode.auth.middleware import SKIP_PATHS

    doc = cluster_join.__doc__ or ""
    for path in SKIP_PATHS - {"/", "/api/cluster/join"}:
        assert path in doc, f"{path} is keyless and missing from the join docstring"


def test_no_module_claims_a_config_route_that_does_not_exist():
    """``PUT /api/config`` never existed; the config routes are GET and PATCH."""
    from ainode.discovery import signing

    assert "PUT /api/config" not in (signing.ClusterSecret.__doc__ or "")
    assert "PUT /api/config" not in (signing.__doc__ or "")


# =============================================================================
# 9. The doctor's cluster.secret check
# =============================================================================

class TestDoctorClusterSecretCheck:
    def _check(self, config, peers_seen=0):
        from ainode.cli import doctor

        checks = doctor.check_cluster_secret(config, peers_seen)
        assert len(checks) == 1
        assert checks[0].name == "cluster.secret"
        return checks[0]

    def test_a_secret_is_ok_and_says_what_it_buys(self):
        from ainode.cli.doctor import OK

        check = self._check(NodeConfig(cluster_secret=SECRET))
        assert check.status == OK
        assert "signed" in check.detail
        assert check.data["set"] is True

    def test_no_secret_warns_about_both_consequences(self):
        from ainode.cli.doctor import WARN

        check = self._check(NodeConfig(cluster_secret=""), peers_seen=2)
        assert check.status == WARN
        assert "unauthenticated" in check.detail
        assert "peers cannot authenticate" in check.detail
        assert "ainode cluster token" in check.fix
        assert check.data == {"set": False, "peers_seen": 2, "configured_peers": 0,
                              "discovery_port": 5679}

    def test_the_check_is_in_the_doctors_run(self, tmp_path, monkeypatch):
        from ainode.cli import doctor

        monkeypatch.setattr(doctor, "http_json", lambda *a, **k: None)
        monkeypatch.setattr(doctor, "run_command", lambda *a, **k: (127, "no"))
        monkeypatch.setattr(doctor, "probe_gpus", lambda: [])
        monkeypatch.setattr(doctor, "latest_image_tag", lambda: None)

        names = [c.name for c in doctor.run_checks(home=tmp_path)]
        assert names.count("cluster.secret") == 1
