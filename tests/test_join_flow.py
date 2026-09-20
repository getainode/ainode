"""Joining a node to a cluster, end to end, and the dead wizard that did not (#208).

Joining a node used to mean opening ``config.json`` on the new box and typing
``cluster_id``, ``cluster_role`` and ``cluster_interface`` by hand, which is what
joining pollux took. The browser onboarding wizard looked like the thing that
should do it and could not: every deployed node has ``onboarded`` true before the
server binds, so ``/onboarding`` only ever answered a redirect, and the wizard's
complete handler never touched a cluster key anyway.

What this file pins:

* the token: 32 bytes, stored hashed, expiring, spendable once, and a corrupt
  token file that accepts nothing rather than crashing the route,
* ``POST /api/cluster/join``: the payload a joiner needs, ONE 403 body for a
  wrong, expired and spent token alike, the per-IP rate limit that stands in for
  the API key, and the exemption itself,
* ``POST /api/cluster/join-self``: behind the key, writes the config, refuses a
  version mismatch, restarts nothing,
* the config write: exactly the join's own keys change, and a file carrying keys
  this release has never heard of comes back byte-identical in every other field,
* the installer: ``AINODE_JOIN`` parses, the generated ``cluster_secret`` lands in
  config.json, and ``AINODE_PEERS`` is untouched,
* the removal: no ``/onboarding``, no ``/api/onboarding/*``, no template, and
  ``/`` serves the dashboard even on a node whose config says it is not onboarded.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import stat
import subprocess
import time
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from ainode import __version__
from ainode.api.cluster_join import (
    JOIN_RATE_LIMIT,
    JOIN_RATE_WINDOW,
    RATE_MAX_SOURCES,
    master_address_for,
    rate_limit_check,
)
from ainode.api.server import create_app
from ainode.cluster.join import (
    DEFAULT_TTL_SECONDS,
    MAX_TTL_SECONDS,
    MIN_TTL_SECONDS,
    REFUSED_MESSAGE,
    TOKEN_BYTES,
    JoinTokenStore,
    apply_join,
    ensure_cluster_secret,
    generate_cluster_secret,
    join_command,
    join_updates,
    join_url,
    merge_config_keys,
    parse_host_port,
    version_refusal,
)
from ainode.core.config import NodeConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
STATIC = REPO_ROOT / "ainode" / "web" / "static"
TEMPLATES = REPO_ROOT / "ainode" / "web" / "templates"
JOIN_JS = STATIC / "js" / "join.js"
NODE = shutil.which("node")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def home(tmp_path, monkeypatch):
    """Every file this app or store writes lands in tmp_path."""
    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr("ainode.auth.middleware.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.auth.middleware.AUTH_FILE", tmp_path / "auth.json")
    monkeypatch.setattr("ainode.secrets.manager.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.secrets.manager.SECRETS_FILE", tmp_path / "secrets.json")
    return tmp_path


@pytest.fixture
def store(home):
    return JoinTokenStore()


@pytest.fixture
def master_config():
    """A master with a cluster and a secret, on a port nothing listens on.

    ``cluster_enabled`` off and ``_skip_replay`` on: this file is about the join
    handshake, and an app that starts discovery announces on the broadcast domain
    of whatever machine runs the suite, while the startup replay reaches the real
    docker boundary for any instance record it finds.
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    config = NodeConfig(
        node_id="master01", node_name="Spark-Test-Master", onboarded=True,
        api_port=free_port, web_port=3000, discovery_port=5679,
        cluster_enabled=False,
        cluster_id="ainode-test-cluster",
        cluster_secret="f" * 64,
    )
    config._skip_replay = True
    return config


@pytest.fixture
def app(master_config, home):
    return create_app(config=master_config, engine=None)


@pytest_asyncio.fixture
async def client(app):
    async with TestClient(TestServer(app)) as c:
        yield c


@pytest_asyncio.fixture
async def keyed_client(app):
    """Auth on, plus the plaintext key, which exists only at this moment."""
    entry = app["auth_config"].enable()
    async with TestClient(TestServer(app)) as c:
        yield c, entry["key"]


# =============================================================================
# 1. The token
# =============================================================================

class TestJoinToken:
    def test_a_minted_token_is_32_bytes_of_hex(self, store):
        minted = store.mint()
        assert len(minted.token) == TOKEN_BYTES * 2
        assert re.fullmatch(r"[0-9a-f]+", minted.token)

    def test_the_plaintext_is_never_written_to_disk(self, store):
        minted = store.mint()
        raw = store.path.read_text()
        assert minted.token not in raw
        assert minted.token_id in raw

    def test_the_file_is_owner_only(self, store):
        store.mint()
        mode = stat.S_IMODE(store.path.stat().st_mode)
        assert mode == 0o600, oct(mode)

    def test_a_fresh_token_verifies(self, store):
        minted = store.mint()
        assert store.verify(minted.token) is not None

    def test_a_wrong_token_does_not(self, store):
        store.mint()
        assert store.verify("0" * 64) is None
        assert store.verify("") is None

    def test_an_expired_token_does_not(self, store):
        minted = store.mint(ttl_seconds=MIN_TTL_SECONDS)
        later = time.time() + MIN_TTL_SECONDS + 1
        assert store.verify(minted.token, now=later) is None

    def test_a_token_is_spendable_once(self, store):
        minted = store.mint()
        first = store.consume(minted.token, used_by="joiner-1")
        assert first is not None
        assert first["used_by"] == "joiner-1"
        assert store.consume(minted.token, used_by="joiner-2") is None
        assert store.verify(minted.token) is None

    def test_the_spend_is_on_disk_before_the_caller_is_told_yes(self, store):
        """A second request with the same token loses even across processes."""
        minted = store.mint()
        store.consume(minted.token)
        assert JoinTokenStore(store.path).consume(minted.token) is None

    def test_an_expired_token_cannot_be_spent(self, store):
        minted = store.mint(ttl_seconds=MIN_TTL_SECONDS)
        later = time.time() + MIN_TTL_SECONDS + 1
        assert store.consume(minted.token, now=later) is None

    def test_two_tokens_live_side_by_side(self, store):
        a = store.mint()
        b = store.mint()
        assert store.consume(a.token) is not None
        assert store.consume(b.token) is not None

    def test_the_default_ttl_is_thirty_minutes(self, store):
        minted = store.mint()
        assert DEFAULT_TTL_SECONDS == 30 * 60
        assert 29 * 60 < minted.ttl_seconds <= 30 * 60

    def test_a_ttl_outside_the_bounds_is_refused(self, store):
        with pytest.raises(ValueError, match="ttl must be between"):
            store.mint(ttl_seconds=MIN_TTL_SECONDS - 1)
        with pytest.raises(ValueError, match="ttl must be between"):
            store.mint(ttl_seconds=MAX_TTL_SECONDS + 1)

    def test_a_corrupt_token_file_accepts_nothing_and_does_not_raise(self, store):
        store.path.write_text("{ this is not json")
        assert store.verify("x" * 64) is None
        assert store.consume("x" * 64) is None
        assert store.live() == []
        # And minting over it still works, so the node is not stuck.
        minted = store.mint()
        assert store.verify(minted.token) is not None

    def test_a_missing_token_file_accepts_nothing(self, store):
        assert not store.path.exists()
        assert store.verify("x" * 64) is None

    def test_settled_records_are_pruned_and_live_ones_are_not(self, store):
        old = store.mint(ttl_seconds=MIN_TTL_SECONDS)
        store.consume(old.token)
        # A record whose expiry passed over the retention window ago is dropped
        # the next time the file is written.
        records = json.loads(store.path.read_text())["tokens"]
        records[0]["used_at"] = time.time() - (48 * 60 * 60)
        records[0]["expires_at"] = time.time() - (48 * 60 * 60)
        store.path.write_text(json.dumps({"tokens": records}))
        fresh = store.mint()
        kept = json.loads(store.path.read_text())["tokens"]
        assert [r["id"] for r in kept] == [fresh.token_id]

    def test_live_lists_only_what_would_still_be_accepted(self, store):
        spent = store.mint()
        store.consume(spent.token)
        alive = store.mint()
        assert [r["id"] for r in store.live()] == [alive.token_id]


# =============================================================================
# 2. The cluster secret
# =============================================================================

class TestClusterSecret:
    def test_a_generated_secret_is_32_bytes_of_hex(self):
        secret = generate_cluster_secret()
        assert len(secret) == 64
        assert re.fullmatch(r"[0-9a-f]+", secret)

    def test_ensure_generates_one_when_there_is_none_and_writes_it(self, home):
        config = NodeConfig(node_name="n", cluster_secret=None)
        (home / "config.json").write_text(json.dumps({"node_name": "n"}))
        secret, generated = ensure_cluster_secret(config)
        assert generated is True
        assert config.cluster_secret == secret
        on_disk = json.loads((home / "config.json").read_text())
        assert on_disk["cluster_secret"] == secret
        # And nothing else was added.
        assert set(on_disk) == {"node_name", "cluster_secret"}

    def test_ensure_leaves_an_existing_secret_alone(self, home):
        config = NodeConfig(cluster_secret="a" * 64)
        secret, generated = ensure_cluster_secret(config)
        assert (secret, generated) == ("a" * 64, False)
        assert not (home / "config.json").exists()


# =============================================================================
# 3. The joiner's config write
# =============================================================================

PAYLOAD = {
    "cluster_id": "ainode-test-cluster",
    "cluster_secret": "b" * 64,
    "discovery_port": 5679,
    "master_address": "10.0.0.1:3000",
    "ainode_version": __version__,
}


class TestConfigWrite:
    def test_join_updates_are_exactly_the_join_keys(self):
        updates = join_updates(PAYLOAD)
        assert set(updates) == {
            "cluster_id", "cluster_role", "distributed_mode",
            "master_address", "cluster_secret", "discovery_port",
        }

    def test_a_joiner_is_a_worker_that_runs_no_engine(self):
        """The two spellings of one decision, both as the code spells them.

        ``cluster_role`` is auto/master/worker and ``member`` is not a value it
        accepts -- PATCH /api/config rejects it -- while ``distributed_mode`` is
        where ``member`` lives. A joiner takes both.
        """
        updates = join_updates(PAYLOAD)
        assert updates["cluster_role"] == "worker"
        assert updates["distributed_mode"] == "member"
        from ainode.api.server import handle_patch_config  # noqa: F401
        source = (REPO_ROOT / "ainode" / "api" / "server.py").read_text()
        assert '("auto", "master", "worker")' in source

    def test_name_and_interface_are_written_only_when_given(self):
        assert "node_name" not in join_updates(PAYLOAD)
        assert "cluster_interface" not in join_updates(PAYLOAD)
        with_extras = join_updates(PAYLOAD, node_name="Spark-9", interface="enp1s0f0np0")
        assert with_extras["node_name"] == "Spark-9"
        assert with_extras["cluster_interface"] == "enp1s0f0np0"

    def test_an_empty_interface_never_clears_a_pinned_one(self, home):
        """An empty cluster_interface means autodetect, which is not a join's call."""
        target = home / "config.json"
        target.write_text(json.dumps({"cluster_interface": "enp1s0f0np0"}))
        apply_join(PAYLOAD, path=target)
        assert json.loads(target.read_text())["cluster_interface"] == "enp1s0f0np0"

    def test_the_write_touches_only_the_join_keys(self, home):
        """Every other key comes back exactly as it went in.

        Deliberately not NodeConfig.save(): that rewrites the file from the
        dataclass, so it would ADD every key this release defaults differently and
        DROP the two below that the dataclass does not declare.
        """
        target = home / "config.json"
        before = {
            "node_name": "Spark-9",
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "gpu_memory_utilization": 0.85,
            "extra_vllm_args": ["--moe-backend", "marlin"],
            "cluster_id": "old-cluster",
            "cluster_role": "master",
            "distributed_mode": "head",
            "discovery_port": 5678,
            "master_address": None,
            "cluster_secret": "old",
            "a_key_no_release_has_ever_heard_of": {"nested": [1, 2, 3]},
            "onboarded": True,
        }
        target.write_text(json.dumps(before, indent=2))

        written = apply_join(PAYLOAD, path=target)
        after = json.loads(target.read_text())

        for key in before:
            if key in written:
                continue
            assert after[key] == before[key], key
        assert after["a_key_no_release_has_ever_heard_of"] == {"nested": [1, 2, 3]}
        assert after["cluster_id"] == "ainode-test-cluster"
        assert after["cluster_secret"] == "b" * 64
        assert after["cluster_role"] == "worker"
        assert after["distributed_mode"] == "member"
        assert after["discovery_port"] == 5679
        assert after["master_address"] == "10.0.0.1:3000"
        # No key appeared that was not there and not a join key.
        assert set(after) == set(before) | set(written)

    def test_a_config_that_does_not_exist_yet_is_created(self, home):
        target = home / "config.json"
        apply_join(PAYLOAD, path=target)
        assert json.loads(target.read_text())["cluster_id"] == "ainode-test-cluster"

    def test_an_empty_master_address_never_clears_a_pinned_one(self, home):
        """The same rule as the interface: an empty answer replaces nothing."""
        target = home / "config.json"
        target.write_text(json.dumps({"master_address": "10.9.9.9:3100"}))
        written = apply_join(dict(PAYLOAD, master_address=""), path=target)
        assert "master_address" not in written
        assert json.loads(target.read_text())["master_address"] == "10.9.9.9:3100"

    def test_a_master_with_no_secret_writes_no_secret_key(self, home):
        target = home / "config.json"
        target.write_text(json.dumps({"cluster_secret": "mine"}))
        payload = dict(PAYLOAD, cluster_secret="")
        written = apply_join(payload, path=target)
        assert "cluster_secret" not in written
        assert json.loads(target.read_text())["cluster_secret"] == "mine"

    def test_a_non_json_config_is_a_refusal_not_a_silent_overwrite(self, home):
        target = home / "config.json"
        target.write_text("{ broken")
        with pytest.raises(ValueError, match="not valid JSON"):
            merge_config_keys({"cluster_id": "x"}, path=target)
        assert target.read_text() == "{ broken"

    def test_a_config_that_is_not_an_object_is_refused(self, home):
        target = home / "config.json"
        target.write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="JSON object"):
            merge_config_keys({"cluster_id": "x"}, path=target)

    def test_the_written_config_still_loads_as_a_nodeconfig(self, home):
        target = home / "config.json"
        target.write_text(json.dumps({"node_name": "Spark-9"}))
        apply_join(PAYLOAD, path=target, node_name="Spark-9")
        loaded = NodeConfig(**{
            k: v for k, v in json.loads(target.read_text()).items()
            if k in NodeConfig.__dataclass_fields__
        })
        assert loaded.cluster_id == "ainode-test-cluster"
        assert loaded.cluster_role == "worker"
        assert loaded.distributed_mode == "member"


# =============================================================================
# 4. Addresses and versions
# =============================================================================

class TestAddressParsing:
    @pytest.mark.parametrize("text,expected", [
        ("10.0.0.1", ("10.0.0.1", 3000)),
        ("10.0.0.1:3000", ("10.0.0.1", 3000)),
        ("10.0.0.1:8080", ("10.0.0.1", 8080)),
        ("http://10.0.0.1:3000", ("10.0.0.1", 3000)),
        ("https://spark1:3000/", ("spark1", 3000)),
        ("spark1.local", ("spark1.local", 3000)),
        ("[::1]:3000", ("::1", 3000)),
        ("[fe80::1]", ("fe80::1", 3000)),
        ("fe80::1", ("fe80::1", 3000)),
        ("  10.0.0.1:3000  ", ("10.0.0.1", 3000)),
    ])
    def test_what_an_operator_actually_pastes(self, text, expected):
        assert parse_host_port(text) == expected

    @pytest.mark.parametrize("bad", ["", "   ", ":3000", "10.0.0.1:abc",
                                     "10.0.0.1:0", "10.0.0.1:70000", "[::1"])
    def test_what_cannot_be_a_master(self, bad):
        with pytest.raises(ValueError):
            parse_host_port(bad)

    def test_the_url_is_the_join_route_on_that_master(self):
        assert join_url("10.0.0.1") == "http://10.0.0.1:3000/api/cluster/join"
        assert join_url("[::1]:3100") == "http://[::1]:3100/api/cluster/join"

    def test_the_printed_command_is_the_one_to_paste(self):
        assert join_command("10.0.0.1:3000", "abc") == "ainode join 10.0.0.1:3000 abc"


class TestVersionCheck:
    def test_matching_versions_pass(self):
        assert version_refusal("0.5.27", "0.5.27") is None

    def test_differing_versions_are_refused_and_named(self):
        message = version_refusal("0.5.27", "0.5.26")
        assert message is not None
        assert "0.5.27" in message and "0.5.26" in message
        assert "--allow-version-mismatch" in message

    def test_the_flag_overrides_it(self):
        assert version_refusal("0.5.27", "0.5.26", allow_mismatch=True) is None

    def test_a_master_announcing_no_version_is_not_a_refusal(self):
        """An older master has nothing to compare, and refusing would strand it."""
        assert version_refusal("0.5.27", "") is None
        assert version_refusal("0.5.27", None) is None


# =============================================================================
# 5. The rate limit that stands in for the API key
# =============================================================================

class TestRateLimit:
    def test_five_attempts_a_minute_then_no(self):
        state: dict = {}
        now = 1000.0
        for i in range(JOIN_RATE_LIMIT):
            assert rate_limit_check(state, "10.0.0.9", now + i) is True
        assert rate_limit_check(state, "10.0.0.9", now + 5) is False

    def test_the_window_rolls(self):
        state: dict = {}
        for i in range(JOIN_RATE_LIMIT):
            assert rate_limit_check(state, "10.0.0.9", 1000.0 + i) is True
        assert rate_limit_check(state, "10.0.0.9", 1000.0) is False
        assert rate_limit_check(state, "10.0.0.9", 1000.0 + JOIN_RATE_WINDOW + 1) is True

    def test_one_flooding_source_does_not_lock_out_another(self):
        state: dict = {}
        for i in range(JOIN_RATE_LIMIT + 3):
            rate_limit_check(state, "10.0.0.9", 1000.0 + i)
        assert rate_limit_check(state, "10.0.0.10", 1000.0) is True

    def test_a_forged_flood_cannot_grow_the_state_without_bound(self):
        state: dict = {}
        for i in range(RATE_MAX_SOURCES + 200):
            rate_limit_check(state, f"10.1.{i // 256}.{i % 256}", 1000.0 + i)
        assert len(state) <= RATE_MAX_SOURCES + 1


# =============================================================================
# 6. POST /api/cluster/join, on the master
# =============================================================================

@pytest.mark.asyncio
class TestJoinRoute:
    async def test_a_good_token_hands_over_the_cluster(self, client, store, master_config):
        minted = store.mint()
        resp = await client.post("/api/cluster/join",
                                 json={"token": minted.token, "node_name": "Spark-9"})
        assert resp.status == 200
        body = await resp.json()
        assert body["cluster_id"] == master_config.cluster_id
        assert body["cluster_secret"] == master_config.cluster_secret
        assert body["discovery_port"] == master_config.discovery_port
        assert body["ainode_version"] == __version__
        assert body["master_address"]

    async def test_the_token_is_spent_by_the_call(self, client, store):
        minted = store.mint()
        first = await client.post("/api/cluster/join", json={"token": minted.token})
        assert first.status == 200
        second = await client.post("/api/cluster/join", json={"token": minted.token})
        assert second.status == 403

    async def test_wrong_expired_and_spent_all_answer_the_same_403(self, client, store):
        spent = store.mint()
        await client.post("/api/cluster/join", json={"token": spent.token})
        expired = store.mint(ttl_seconds=MIN_TTL_SECONDS)
        records = json.loads(store.path.read_text())["tokens"]
        for record in records:
            if record["id"] == expired.token_id:
                record["expires_at"] = time.time() - 1
        store.path.write_text(json.dumps({"tokens": records}))

        bodies = []
        for token in ("0" * 64, spent.token, expired.token, ""):
            resp = await client.post("/api/cluster/join", json={"token": token})
            assert resp.status == 403, token
            bodies.append(await resp.json())
        assert all(b == bodies[0] for b in bodies)
        assert bodies[0]["error"]["message"] == REFUSED_MESSAGE

    async def test_the_refusal_says_how_to_get_a_working_token(self, client):
        resp = await client.post("/api/cluster/join", json={"token": "nope"})
        message = (await resp.json())["error"]["message"]
        assert "ainode cluster token" in message
        assert "once" in message

    async def test_a_bad_body_is_a_400(self, client):
        resp = await client.post("/api/cluster/join", data="not json")
        assert resp.status == 400
        resp = await client.post("/api/cluster/join", json=["a", "list"])
        assert resp.status == 400

    async def test_the_rate_limit_is_enforced_in_the_handler(self, client, store):
        for _ in range(JOIN_RATE_LIMIT):
            resp = await client.post("/api/cluster/join", json={"token": "wrong"})
            assert resp.status == 403
        resp = await client.post("/api/cluster/join", json={"token": "wrong"})
        assert resp.status == 429
        assert resp.headers["Retry-After"] == str(int(JOIN_RATE_WINDOW))
        # And a GOOD token is refused too while the window holds: the limit is on
        # the route, not on failures, so it cannot be walked around.
        minted = store.mint()
        resp = await client.post("/api/cluster/join", json={"token": minted.token})
        assert resp.status == 429

    async def test_the_route_answers_with_no_api_key(self, keyed_client, store):
        """The exemption, on the real app with auth ON."""
        client, key = keyed_client
        minted = store.mint()
        resp = await client.post("/api/cluster/join", json={"token": minted.token})
        assert resp.status == 200
        # While its neighbour on the same prefix still wants the key.
        resp = await client.get("/api/cluster/info")
        assert resp.status == 401
        resp = await client.get("/api/cluster/info",
                                headers={"Authorization": f"Bearer {key}"})
        assert resp.status == 200

    async def test_a_master_with_no_secret_says_so_by_sending_an_empty_one(
            self, master_config, home, store):
        master_config.cluster_secret = None
        app = create_app(config=master_config, engine=None)
        async with TestClient(TestServer(app)) as client:
            minted = store.mint()
            resp = await client.post("/api/cluster/join", json={"token": minted.token})
            assert (await resp.json())["cluster_secret"] == ""


class TestMasterAddress:
    def test_a_configured_master_address_wins(self):
        request = _StubRequest(config=NodeConfig(master_address="10.9.9.9:3100",
                                                 web_port=3000),
                               headers={"Host": "127.0.0.1:8080"})
        assert master_address_for(request) == "10.9.9.9:3100"

    def test_otherwise_the_host_header_the_joiner_reached_us_on(self):
        request = _StubRequest(config=NodeConfig(web_port=3000),
                               headers={"Host": "10.0.0.1:3000"})
        assert master_address_for(request) == "10.0.0.1:3000"

    def test_a_host_header_with_no_port_gets_this_nodes_web_port(self):
        request = _StubRequest(config=NodeConfig(web_port=3100),
                               headers={"Host": "spark1"})
        assert master_address_for(request) == "spark1:3100"

    def test_no_host_header_falls_back_to_a_detected_address(self):
        request = _StubRequest(config=NodeConfig(web_port=3000), headers={})
        address = master_address_for(request)
        assert address.endswith(":3000")
        assert len(address) > len(":3000")


class _StubRequest:
    """The two things master_address_for reads."""

    def __init__(self, config, headers):
        self.app = {"config": config}
        self.headers = headers
        self.transport = None
        self.remote = None


# =============================================================================
# 7. POST /api/cluster/join-self, on the joiner
# =============================================================================

@pytest.mark.asyncio
class TestJoinSelfRoute:
    async def test_it_needs_the_api_key(self, keyed_client):
        client, key = keyed_client
        resp = await client.post("/api/cluster/join-self",
                                 json={"host": "10.0.0.1", "token": "x"})
        assert resp.status == 401

    async def test_a_missing_host_or_token_is_a_400(self, client):
        for body in ({"token": "x"}, {"host": "10.0.0.1"}, {}):
            resp = await client.post("/api/cluster/join-self", json=body)
            assert resp.status == 400

    async def test_an_unparseable_host_is_a_400(self, client):
        resp = await client.post("/api/cluster/join-self",
                                 json={"host": "10.0.0.1:abc", "token": "x"})
        assert resp.status == 400

    async def test_it_joins_this_node_to_a_master_and_writes_the_config(
            self, client, home, monkeypatch):
        monkeypatch.setattr(
            "ainode.api.cluster_join.fetch_join_payload",
            _canned(dict(PAYLOAD, master_address="10.0.0.1:3000")))
        (home / "config.json").write_text(json.dumps({"node_name": "Spark-9"}))
        resp = await client.post("/api/cluster/join-self",
                                 json={"host": "10.0.0.1:3000", "token": "t"})
        assert resp.status == 200
        body = await resp.json()
        assert body["ok"] is True
        assert body["cluster_id"] == "ainode-test-cluster"
        assert body["restart_required"] is True
        assert body["restart_command"] == "sudo systemctl restart ainode"
        on_disk = json.loads((home / "config.json").read_text())
        assert on_disk["cluster_id"] == "ainode-test-cluster"
        assert on_disk["node_name"] == "Spark-9"

    async def test_it_never_returns_the_secret_to_a_browser(
            self, client, home, monkeypatch):
        monkeypatch.setattr("ainode.api.cluster_join.fetch_join_payload",
                            _canned(PAYLOAD))
        resp = await client.post("/api/cluster/join-self",
                                 json={"host": "10.0.0.1", "token": "t"})
        text = await resp.text()
        assert PAYLOAD["cluster_secret"] not in text
        assert (await resp.json())["signed_discovery"] is True

    async def test_a_version_mismatch_is_a_409_that_says_why(
            self, client, home, monkeypatch):
        monkeypatch.setattr(
            "ainode.api.cluster_join.fetch_join_payload",
            _canned(dict(PAYLOAD, ainode_version="0.0.1")))
        resp = await client.post("/api/cluster/join-self",
                                 json={"host": "10.0.0.1", "token": "t"})
        assert resp.status == 409
        body = await resp.json()
        assert "0.0.1" in body["error"]["message"]
        assert body["master_version"] == "0.0.1"
        assert not (home / "config.json").exists()

    async def test_the_mismatch_can_be_overridden(self, client, home, monkeypatch):
        monkeypatch.setattr(
            "ainode.api.cluster_join.fetch_join_payload",
            _canned(dict(PAYLOAD, ainode_version="0.0.1")))
        resp = await client.post("/api/cluster/join-self", json={
            "host": "10.0.0.1", "token": "t", "allow_version_mismatch": True})
        assert resp.status == 200

    async def test_a_refused_token_comes_back_as_a_403(self, client, monkeypatch):
        async def _refuse(*_a, **_k):
            return None, REFUSED_MESSAGE, 403
        monkeypatch.setattr("ainode.api.cluster_join.fetch_join_payload", _refuse)
        resp = await client.post("/api/cluster/join-self",
                                 json={"host": "10.0.0.1", "token": "t"})
        assert resp.status == 403
        assert REFUSED_MESSAGE in (await resp.json())["error"]["message"]

    async def test_an_unreachable_master_is_a_502(self, client, monkeypatch):
        async def _unreachable(*_a, **_k):
            return None, "could not reach it", 0
        monkeypatch.setattr("ainode.api.cluster_join.fetch_join_payload", _unreachable)
        resp = await client.post("/api/cluster/join-self",
                                 json={"host": "10.0.0.1", "token": "t"})
        assert resp.status == 502

    async def test_it_restarts_nothing(self, client, monkeypatch):
        """The restart would kill this response, so the answer is the command."""
        called = []
        monkeypatch.setattr("ainode.service.systemd.restart_service",
                            lambda **kw: called.append(kw))
        monkeypatch.setattr("ainode.api.cluster_join.fetch_join_payload",
                            _canned(PAYLOAD))
        resp = await client.post("/api/cluster/join-self",
                                 json={"host": "10.0.0.1", "token": "t"})
        assert resp.status == 200
        assert called == []


def _canned(payload):
    async def _fetch(*_args, **_kwargs):
        return dict(payload), None, 200
    return _fetch


# =============================================================================
# 8. The whole round trip: one app mints, another joins
# =============================================================================

@pytest.mark.asyncio
async def test_a_token_minted_here_joins_a_node_there(master_config, home, tmp_path):
    """The proof shape, in-process: mint on the master, join, read the config."""
    store = JoinTokenStore(tmp_path / "join-tokens.json")
    minted = store.mint()
    app = create_app(config=master_config, engine=None)
    async with TestClient(TestServer(app)) as client:
        import ainode.cluster.join as join_module

        # The route reads the master's store; point it at the one we minted into.
        original = join_module.tokens_path
        join_module.tokens_path = lambda: tmp_path / "join-tokens.json"
        try:
            resp = await client.post("/api/cluster/join",
                                     json={"token": minted.token, "node_name": "Joiner"})
            assert resp.status == 200
            payload = await resp.json()
        finally:
            join_module.tokens_path = original

    joiner_config = tmp_path / "joiner-config.json"
    joiner_config.write_text(json.dumps({
        "node_name": "Joiner", "model": None, "gpu_memory_utilization": 0.6,
        "cluster_id": "default", "onboarded": True,
    }))
    written = apply_join(payload, node_name="Joiner", path=joiner_config)
    after = json.loads(joiner_config.read_text())

    assert after["cluster_id"] == master_config.cluster_id
    assert after["cluster_secret"] == master_config.cluster_secret
    assert after["cluster_role"] == "worker"
    assert after["distributed_mode"] == "member"
    assert after["gpu_memory_utilization"] == 0.6
    assert after["model"] is None
    assert after["onboarded"] is True
    assert set(after) == {"node_name", "model", "gpu_memory_utilization",
                          "onboarded"} | set(written)


# =============================================================================
# 9. The installer
# =============================================================================

def _render_install(tmp_path: Path, *args: str, env_extra: dict | None = None,
                    expect_ok: bool = True):
    home = tmp_path / "home"
    ainode_home = home / ".ainode"
    home.mkdir(parents=True, exist_ok=True)
    sysfs = tmp_path / "sys-class-net"
    sysfs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        HOME=str(home),
        AINODE_HOME=str(ainode_home),
        AINODE_IMAGE="ghcr.io/getainode/ainode:9.9.9",
        SYS_CLASS_NET=str(sysfs),
    )
    env.pop("AINODE_PEERS", None)
    env.pop("AINODE_JOIN", None)
    env.pop("HF_TOKEN", None)
    env.update(env_extra or {})
    proc = subprocess.run(
        ["bash", str(INSTALL_SH), "--dry-run", *args],
        capture_output=True, text=True, timeout=120, env=env,
    )
    if expect_ok:
        assert proc.returncode == 0, proc.stdout + proc.stderr
    return ainode_home, proc


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
class TestInstaller:
    def test_the_script_still_parses(self):
        proc = subprocess.run(["bash", "-n", str(INSTALL_SH)],
                              capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr

    def test_a_fresh_install_generates_a_cluster_secret(self, tmp_path):
        """#169's follow-up: a node HAS a secret rather than running open."""
        ainode_home, _ = _render_install(tmp_path)
        data = json.loads((ainode_home / "config.json").read_text())
        secret = data["cluster_secret"]
        assert len(secret) == 64
        assert re.fullmatch(r"[0-9a-f]+", secret)

    def test_two_installs_do_not_get_the_same_secret(self, tmp_path):
        first, _ = _render_install(tmp_path / "a")
        second, _ = _render_install(tmp_path / "b")
        assert (json.loads((first / "config.json").read_text())["cluster_secret"]
                != json.loads((second / "config.json").read_text())["cluster_secret"])

    def test_ainode_join_is_documented_in_the_usage_header(self):
        text = INSTALL_SH.read_text()
        header = text[: text.index("set -euo pipefail")]
        assert "AINODE_JOIN" in header
        assert "ainode cluster token" in header

    def test_a_dry_run_names_the_join_and_uses_no_token(self, tmp_path):
        _, proc = _render_install(
            tmp_path, env_extra={"AINODE_JOIN": "10.0.0.1:3000:sometoken"})
        assert "would join the cluster at 10.0.0.1:3000" in proc.stdout
        assert "sometoken" not in proc.stdout

    def test_a_host_with_no_port_parses_too(self, tmp_path):
        _, proc = _render_install(
            tmp_path, env_extra={"AINODE_JOIN": "10.0.0.1:sometoken"})
        assert "would join the cluster at 10.0.0.1" in proc.stdout

    def test_a_malformed_value_warns_and_installs_anyway(self, tmp_path):
        _, proc = _render_install(tmp_path, env_extra={"AINODE_JOIN": "nocolon"})
        assert proc.returncode == 0
        assert "AINODE_JOIN must be" in proc.stdout + proc.stderr

    def test_joining_and_heading_are_different_jobs(self, tmp_path):
        _, proc = _render_install(
            tmp_path, "--job", "master",
            env_extra={"AINODE_JOIN": "10.0.0.1:3000:t"}, expect_ok=False)
        assert proc.returncode == 2
        assert "Pick one" in proc.stderr

    def test_a_pasted_secret_is_used_verbatim(self, tmp_path):
        """The escape hatch for a second node installed without joining.

        Two nodes that each generated their own secret are invisible to each other
        once discovery is signed, so an operator has to be able to paste the head's
        value at install time instead.
        """
        ainode_home, proc = _render_install(
            tmp_path, env_extra={"AINODE_CLUSTER_SECRET": "d" * 64})
        data = json.loads((ainode_home / "config.json").read_text())
        assert data["cluster_secret"] == "d" * 64
        assert "taken from AINODE_CLUSTER_SECRET" in proc.stdout

    def test_a_generated_secret_says_a_second_node_needs_the_same_value(self, tmp_path):
        _, proc = _render_install(tmp_path)
        assert "generated for this node" in proc.stdout
        assert "AINODE_CLUSTER_SECRET" in proc.stdout
        assert "ainode join" in proc.stdout

    def test_a_joining_install_does_not_lecture_about_the_secret(self, tmp_path):
        """The join overwrites it a moment later, so there is nothing to warn about."""
        _, proc = _render_install(
            tmp_path, env_extra={"AINODE_JOIN": "10.0.0.1:3000:t"})
        assert "generated for this node" not in proc.stdout

    def test_the_peers_path_is_untouched(self, tmp_path):
        """AINODE_PEERS is the eugr-era SSH bootstrap and is not what joining is."""
        ainode_home, proc = _render_install(
            tmp_path, "--job", "master",
            env_extra={"AINODE_PEERS": "10.0.0.2,10.0.0.3"})
        data = json.loads((ainode_home / "config.json").read_text())
        assert data["peer_ips"] == ["10.0.0.2", "10.0.0.3"]
        assert data["distributed_mode"] == "head"
        assert "skipping the passwordless SSH bootstrap" in proc.stdout


# =============================================================================
# 10. The dead wizard is gone
# =============================================================================

class TestOnboardingRemoved:
    def test_the_template_is_gone(self):
        assert not (TEMPLATES / "onboarding.html").exists()
        assert sorted(p.name for p in TEMPLATES.iterdir()) == ["index.html"]

    def test_the_api_routes_module_is_gone(self):
        assert not (REPO_ROOT / "ainode" / "onboarding" / "api_routes.py").exists()

    def test_the_terminal_wizard_stays(self):
        """The TTY path is reachable and is not what #208 is about."""
        from ainode.onboarding.setup import run_onboarding
        assert callable(run_onboarding)

    def test_no_onboarding_route_is_registered(self, master_config, home):
        app = create_app(config=master_config, engine=None)
        paths = set()
        for route in app.router.routes():
            info = route.resource.get_info() if route.resource else {}
            paths.add(info.get("path") or info.get("formatter") or "")
        assert not [p for p in paths if "onboarding" in p]

    def test_the_join_routes_are_registered(self, master_config, home):
        app = create_app(config=master_config, engine=None)
        found = set()
        for route in app.router.routes():
            info = route.resource.get_info() if route.resource else {}
            path = info.get("path") or info.get("formatter") or ""
            if path in ("/api/cluster/join", "/api/cluster/join-self"):
                found.add((route.method, path))
        assert found == {("POST", "/api/cluster/join"),
                         ("POST", "/api/cluster/join-self")}

    def test_the_middleware_has_no_onboarding_exemption(self):
        from ainode.auth import middleware

        assert not [p for p in middleware.SKIP_PATHS if "onboarding" in p]
        assert not [p for p in middleware.SKIP_PREFIXES if "onboarding" in p]
        assert not hasattr(middleware, "_onboarding_open")
        assert "/api/cluster/join" in middleware.SKIP_PATHS

    @pytest.mark.asyncio
    async def test_the_index_serves_the_dashboard_even_when_not_onboarded(self, home):
        """The redirect that made /onboarding unreachable is gone with it.

        The installer writes onboarded true and a non-TTY start sets it before the
        server binds, so this branch was only ever reachable in a test. It now
        serves the dashboard either way rather than bouncing to a page that
        bounced straight back.
        """
        config = NodeConfig(node_id="n", node_name="N", onboarded=False,
                            cluster_enabled=False)
        config._skip_replay = True
        app = create_app(config=config, engine=None)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/", allow_redirects=False)
            assert resp.status == 200
            assert "AINode" in await resp.text()


# =============================================================================
# 11. The browser card
# =============================================================================

class TestDocs:
    """The README describes what ships, and #208 is about a doc claim as much as code."""

    def test_the_readme_no_longer_promises_a_first_run_wizard(self):
        text = (REPO_ROOT / "README.md").read_text()
        assert "First-run onboarding walks" not in text
        assert "There is no onboarding wizard" in text

    def test_the_readme_names_both_join_commands(self):
        text = (REPO_ROOT / "README.md").read_text()
        assert "ainode cluster token" in text
        assert "ainode join HOST[:PORT] TOKEN" in text
        assert "AINODE_JOIN=" in text

    def test_the_readme_says_two_independent_installs_do_not_see_each_other(self):
        """The regression a generated per-node secret would otherwise cause."""
        text = (REPO_ROOT / "README.md").read_text()
        assert "AINODE_CLUSTER_SECRET" in text
        assert "invisible to each other" in text

    def test_the_readme_lists_the_join_route_among_the_keyless_ones(self):
        text = (REPO_ROOT / "README.md").read_text()
        assert "four deliberate exceptions" in text
        assert "POST /api/cluster/join" in text


class TestJoinCard:
    def test_the_card_is_its_own_file(self):
        assert JOIN_JS.exists()
        # Small on purpose: the 19 KB wizard is what this replaces.
        assert JOIN_JS.stat().st_size < 12 * 1024

    def test_the_shell_loads_it(self):
        html = (TEMPLATES / "index.html").read_text()
        assert "/static/js/join.js" in html
        assert html.index("/static/js/auth.js") < html.index("/static/js/join.js")

    def test_app_js_holds_exactly_one_line_about_it(self):
        """PR 219 is rewriting app.js; this card must not be in the way."""
        lines = [line for line in (STATIC / "js" / "app.js").read_text().splitlines()
                 if "AINodeJoin" in line]
        assert len(lines) == 1, lines
        assert "renderConfigCluster" in lines[0]

    def test_it_calls_the_keyed_route_through_the_auth_wrapper(self):
        text = JOIN_JS.read_text()
        assert "AINodeAuth.fetch('/api/cluster/join-self'" in text
        assert "'/api/cluster/join'" not in text  # that one is the MASTER's route

    def test_it_never_names_the_secret(self):
        assert "cluster_secret" not in JOIN_JS.read_text()

    @pytest.mark.skipif(NODE is None, reason="node is not installed")
    def test_the_decisions_behave_under_node(self, tmp_path):
        harness = tmp_path / "harness.js"
        harness.write_text(_NODE_HARNESS)
        proc = subprocess.run([NODE, str(harness), str(JOIN_JS)],
                              capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stdout + proc.stderr


_NODE_HARNESS = """
'use strict';
const assert = require('assert');
const Join = require(process.argv[2]);

// buildPayload trims, refuses an empty form, and carries the optional fields.
assert.ok(Join.buildPayload({}).error.includes('master address'));
assert.ok(Join.buildPayload({ host: '10.0.0.1' }).error.includes('ainode cluster token'));
let built = Join.buildPayload({ host: ' 10.0.0.1:3000 ', token: ' abc\\n' });
assert.deepStrictEqual(built.body, { host: '10.0.0.1:3000', token: 'abc' });
built = Join.buildPayload({ host: 'h', token: 't', name: 'Spark-9',
                            iface: 'enp1s0f0np0', allowMismatch: true });
assert.strictEqual(built.body.name, 'Spark-9');
assert.strictEqual(built.body.interface, 'enp1s0f0np0');
assert.strictEqual(built.body.allow_version_mismatch, true);
// An untouched optional field is absent, not empty.
built = Join.buildPayload({ host: 'h', token: 't' });
assert.ok(!('name' in built.body));
assert.ok(!('interface' in built.body));
assert.ok(!('allow_version_mismatch' in built.body));

// describeResult: a success names the keys and the command.
let out = Join.describeResult(200, {
  ok: true, cluster_id: 'c1', signed_discovery: true,
  written: ['cluster_id', 'cluster_role'],
});
assert.strictEqual(out.ok, true);
assert.ok(out.message.includes('c1'));
assert.ok(out.detail.includes('cluster_role'));
assert.ok(out.detail.includes('systemctl restart ainode'));
assert.ok(out.detail.includes('signed'));

// An unsigned cluster says so instead of claiming it is signed.
out = Join.describeResult(200, { ok: true, cluster_id: 'c1', signed_discovery: false, written: [] });
assert.ok(out.detail.includes('unauthenticated'));

// A version mismatch is its own tone, with the second option named.
out = Join.describeResult(409, { error: { message: 'runs 0.5.26' } });
assert.strictEqual(out.ok, false);
assert.strictEqual(out.tone, 'mismatch');
assert.ok(out.detail.includes('0.5.26'));

// A refused token, and anything else.
out = Join.describeResult(403, { error: { message: 'nope' } });
assert.ok(out.message.includes('refused the token'));
out = Join.describeResult(502, null);
assert.strictEqual(out.ok, false);
assert.ok(out.message.includes('502'));
out = Join.describeResult(0, null);
assert.ok(!out.message.includes('HTTP'));

// No DOM was touched by any of that.
assert.strictEqual(typeof document, 'undefined');
console.log('join.js decisions OK');
"""


# =============================================================================
# 12. The CLI
# =============================================================================

class TestCli:
    def test_cluster_token_mints_and_prints_the_command_to_paste(self, home, capsys):
        from ainode.cli.main import cmd_cluster

        (home / "config.json").write_text(json.dumps(
            {"node_name": "Master", "web_port": 3000}))
        cmd_cluster(_Args(cluster_action="token", ttl=None))
        out = capsys.readouterr().out
        assert "ainode join" in out
        assert "Made in Texas" in out
        # The token it printed is the one it stored, and only its hash is on disk.
        token = re.search(r"ainode join \S+ ([0-9a-f]{64})", out.replace("\n", " "))
        assert token, out
        assert JoinTokenStore().verify(token.group(1)) is not None
        assert token.group(1) not in (home / "join-tokens.json").read_text()

    def test_it_generates_a_cluster_secret_and_says_it_did(self, home, capsys):
        from ainode.cli.main import cmd_cluster

        (home / "config.json").write_text(json.dumps({"node_name": "Master"}))
        cmd_cluster(_Args(cluster_action="token", ttl=None))
        out = capsys.readouterr().out
        assert "no cluster_secret" in out
        assert "same" in out  # every node must end up with the same value
        on_disk = json.loads((home / "config.json").read_text())
        assert len(on_disk["cluster_secret"]) == 64
        assert set(on_disk) == {"node_name", "cluster_secret"}

    def test_an_existing_secret_is_not_replaced(self, home, capsys):
        from ainode.cli.main import cmd_cluster

        (home / "config.json").write_text(json.dumps(
            {"node_name": "Master", "cluster_secret": "a" * 64}))
        cmd_cluster(_Args(cluster_action="token", ttl=None))
        assert json.loads((home / "config.json").read_text())["cluster_secret"] == "a" * 64

    def test_a_bad_ttl_exits_without_minting(self, home):
        from ainode.cli.main import cmd_cluster

        with pytest.raises(SystemExit) as exc:
            cmd_cluster(_Args(cluster_action="token", ttl=1))
        assert exc.value.code == 2
        assert not (home / "join-tokens.json").exists()

    def test_tokens_lists_what_is_still_live(self, home, capsys):
        from ainode.cli.main import cmd_cluster

        minted = JoinTokenStore().mint()
        cmd_cluster(_Args(cluster_action="tokens"))
        assert minted.token_id in capsys.readouterr().out

    def test_join_refuses_an_unparseable_master(self, home):
        from ainode.cli.main import cmd_join

        with pytest.raises(SystemExit) as exc:
            cmd_join(_Args(master="10.0.0.1:abc", token="t"))
        assert exc.value.code == 2

    def test_join_refuses_a_version_mismatch_and_writes_nothing(
            self, home, monkeypatch, capsys):
        import ainode.cli.main as cli

        monkeypatch.setattr(cli, "_http_post_json",
                            lambda *a, **k: (200, dict(PAYLOAD, ainode_version="0.0.1")))
        with pytest.raises(SystemExit) as exc:
            cli.cmd_join(_Args(master="10.0.0.1:3000", token="t"))
        assert exc.value.code == 1
        assert "version mismatch" in capsys.readouterr().out
        assert not (home / "config.json").exists()

    def test_join_writes_the_config_and_names_the_restart(
            self, home, monkeypatch, capsys):
        import ainode.cli.main as cli

        monkeypatch.setattr(cli, "_http_post_json", lambda *a, **k: (200, dict(PAYLOAD)))
        monkeypatch.setattr("ainode.service.systemd.is_installed", lambda **kw: False)
        (home / "config.json").write_text(json.dumps({"node_name": "Joiner"}))
        cli.cmd_join(_Args(master="10.0.0.1:3000", token="t"))
        out = capsys.readouterr().out
        assert "Joined." in out
        assert "systemctl restart ainode" in out
        on_disk = json.loads((home / "config.json").read_text())
        assert on_disk["cluster_id"] == "ainode-test-cluster"
        assert on_disk["node_name"] == "Joiner"

    def test_join_restarts_the_service_when_one_is_installed(
            self, home, monkeypatch, capsys):
        """A fresh joiner is the one place restarting from here is right."""
        import ainode.cli.main as cli

        restarted = []
        monkeypatch.setattr(cli, "_http_post_json", lambda *a, **k: (200, dict(PAYLOAD)))
        monkeypatch.setattr("ainode.service.systemd.is_installed",
                            lambda **kw: not kw.get("user_mode"))
        monkeypatch.setattr("ainode.service.systemd.restart_service",
                            lambda **kw: restarted.append(kw))
        cli.cmd_join(_Args(master="10.0.0.1:3000", token="t"))
        assert restarted == [{"user_mode": False}]
        assert "Restarted ainode.service" in capsys.readouterr().out

    def test_a_failed_restart_prints_the_command_instead(
            self, home, monkeypatch, capsys):
        import ainode.cli.main as cli

        def _boom(**_kw):
            raise RuntimeError("no systemd bus")
        monkeypatch.setattr(cli, "_http_post_json", lambda *a, **k: (200, dict(PAYLOAD)))
        monkeypatch.setattr("ainode.service.systemd.is_installed",
                            lambda **kw: not kw.get("user_mode"))
        monkeypatch.setattr("ainode.service.systemd.restart_service", _boom)
        cli.cmd_join(_Args(master="10.0.0.1:3000", token="t"))
        out = capsys.readouterr().out
        assert "no systemd bus" in out
        assert "sudo systemctl restart ainode" in out

    def test_a_refused_token_exits_non_zero_with_the_masters_reason(
            self, home, monkeypatch, capsys):
        import ainode.cli.main as cli

        monkeypatch.setattr(cli, "_http_post_json", lambda *a, **k: (
            403, {"error": {"message": REFUSED_MESSAGE}}))
        with pytest.raises(SystemExit) as exc:
            cli.cmd_join(_Args(master="10.0.0.1:3000", token="t"))
        assert exc.value.code == 1
        assert "ainode cluster token" in capsys.readouterr().out

    def test_a_rate_limited_join_says_to_wait(self, home, monkeypatch, capsys):
        import ainode.cli.main as cli

        monkeypatch.setattr(cli, "_http_post_json", lambda *a, **k: (429, {}))
        with pytest.raises(SystemExit):
            cli.cmd_join(_Args(master="10.0.0.1:3000", token="t"))
        assert "rate limiting" in capsys.readouterr().out

    def test_an_unreachable_master_exits_non_zero(self, home, monkeypatch, capsys):
        import ainode.cli.main as cli

        monkeypatch.setattr(cli, "_http_post_json", lambda *a, **k: (
            0, {"error": {"message": "Connection refused"}}))
        with pytest.raises(SystemExit) as exc:
            cli.cmd_join(_Args(master="10.0.0.1:3000", token="t"))
        assert exc.value.code == 1
        assert "Could not reach" in capsys.readouterr().out

    def test_ainode_config_masks_the_secret(self, home, capsys):
        """`ainode config` output gets pasted into issues."""
        from ainode.cli.main import cmd_config

        (home / "config.json").write_text(json.dumps({"cluster_secret": "c" * 64}))
        cmd_config(_Args(model=None, port=None, hf_token=None, show=True))
        out = capsys.readouterr().out
        assert "c" * 64 not in out
        assert "set (hidden)" in out

    def test_both_subcommands_are_on_the_parser(self):
        import ainode.cli.main as cli

        parser_source = (REPO_ROOT / "ainode" / "cli" / "main.py").read_text()
        assert 'subparsers.add_parser(\n        "cluster"' in parser_source
        assert 'subparsers.add_parser(\n        "join"' in parser_source
        assert callable(cli.cmd_cluster)
        assert callable(cli.cmd_join)


class _Args:
    """A stand-in for argparse's Namespace."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, name):
        return None
