"""Tests for cluster master/worker election + cluster_id scoping + HTTP routes."""

import time

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from ainode.api.server import create_app
from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeAnnouncement, NodeStatus
from ainode.discovery.cluster import ClusterState, ClusterNode


def _ann(**kw):
    defaults = dict(
        node_id="node-a", node_name="a", gpu_name="GB10",
        gpu_memory_gb=128.0, unified_memory=True, model="m",
        status="serving", api_port=8000, web_port=3000,
        cluster_id="default", role="auto",
    )
    defaults.update(kw)
    return NodeAnnouncement(**defaults)


def _node(**kw):
    defaults = dict(
        node_id="n", node_name="n", gpu_name="x", gpu_memory_gb=1.0,
        unified_memory=False, model="m", status=NodeStatus.ONLINE,
        api_port=8000, web_port=3000, last_seen=time.time(),
        cluster_id="default", role="auto",
    )
    defaults.update(kw)
    return ClusterNode(**defaults)


class TestMasterElection:
    def test_auto_lowest_id_is_master(self):
        cluster = ClusterState(local_announcement=_ann(node_id="aaa"))
        cluster.add_node(_node(node_id="bbb", role="auto"))
        cluster.add_node(_node(node_id="ccc", role="auto"))
        master = cluster.get_master()
        assert master.node_id == "aaa"
        assert cluster.is_master_of_cluster() is True

    def test_worker_never_master(self):
        cluster = ClusterState(local_announcement=_ann(node_id="aaa", role="worker"))
        cluster.add_node(_node(node_id="bbb", role="auto"))
        master = cluster.get_master()
        assert master.node_id == "bbb"
        assert cluster.is_master_of_cluster() is False

    def test_explicit_master_wins_over_auto(self):
        cluster = ClusterState(local_announcement=_ann(node_id="aaa", role="auto"))
        cluster.add_node(_node(node_id="bbb", role="master"))
        cluster.add_node(_node(node_id="ccc", role="auto"))
        master = cluster.get_master()
        assert master.node_id == "bbb"

    def test_multiple_explicit_masters_lowest_id_wins(self):
        cluster = ClusterState(local_announcement=_ann(node_id="aaa", role="auto"))
        cluster.add_node(_node(node_id="mmm", role="master"))
        cluster.add_node(_node(node_id="bbb", role="master"))
        master = cluster.get_master()
        assert master.node_id == "bbb"

    def test_all_workers_no_master(self):
        cluster = ClusterState(local_announcement=_ann(node_id="aaa", role="worker"))
        cluster.add_node(_node(node_id="bbb", role="worker"))
        assert cluster.get_master() is None

    def test_cluster_id_isolates_clusters(self):
        cluster = ClusterState(local_announcement=_ann(node_id="aaa", cluster_id="prod"))
        cluster.add_node(_node(node_id="bbb", cluster_id="staging", role="auto"))
        cluster.add_node(_node(node_id="ccc", cluster_id="prod", role="auto"))
        master = cluster.get_master()
        # Only prod nodes participate; aaa has the lowest id among prod.
        assert master.node_id == "aaa"
        members = cluster.members()
        assert {m.node_id for m in members} == {"aaa", "ccc"}

    def test_get_cluster_role_for(self):
        cluster = ClusterState(local_announcement=_ann(node_id="aaa"))
        cluster.add_node(_node(node_id="bbb", role="auto"))
        assert cluster.get_cluster_role_for("aaa") == "master"
        assert cluster.get_cluster_role_for("bbb") == "worker"
        assert cluster.get_cluster_role_for("unknown") == "worker"

    def test_offline_nodes_excluded_from_election(self):
        cluster = ClusterState(local_announcement=_ann(node_id="bbb"))
        cluster.add_node(_node(node_id="aaa", status=NodeStatus.OFFLINE, role="auto"))
        master = cluster.get_master()
        assert master.node_id == "bbb"


# ---- HTTP routes ------------------------------------------------------------


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr("ainode.secrets.manager.SECRETS_FILE", tmp_path / "secrets.json")
    monkeypatch.setattr("ainode.auth.middleware.AUTH_FILE", tmp_path / "auth.json")
    config = NodeConfig(node_id="test-node-1", node_name="test", model="m")
    return create_app(config=config, engine=None)


@pytest_asyncio.fixture
async def client(app):
    async with TestClient(TestServer(app)) as c:
        yield c


@pytest.mark.asyncio
async def test_cluster_info(client):
    resp = await client.get("/api/cluster/info")
    assert resp.status == 200
    data = await resp.json()
    assert data["my_node_id"] == "test-node-1"
    assert data["cluster_id"] == "default"
    assert data["configured_role"] == "auto"
    # Alone in cluster -> I am master.
    assert data["my_role"] == "master"


@pytest.mark.asyncio
async def test_status_includes_role(client):
    resp = await client.get("/api/status")
    data = await resp.json()
    assert data["cluster_role"] == "master"
    assert data["cluster_id"] == "default"
    assert data["master_node_id"] == "test-node-1"


@pytest.mark.asyncio
async def test_set_cluster_role(client):
    resp = await client.post("/api/cluster/role", json={"role": "worker"})
    assert resp.status == 200
    data = await resp.json()
    assert data["cluster_role"] == "worker"

    resp = await client.post("/api/cluster/role", json={"role": "bad"})
    assert resp.status == 400


@pytest.mark.asyncio
async def test_set_cluster_id(client):
    resp = await client.post("/api/cluster/id", json={"cluster_id": "prod-1"})
    assert resp.status == 200
    data = await resp.json()
    assert data["cluster_id"] == "prod-1"

    resp = await client.post("/api/cluster/id", json={"cluster_id": "bad id with spaces"})
    assert resp.status == 400


@pytest.mark.asyncio
async def test_patch_config(client):
    resp = await client.patch("/api/config", json={"node_name": "RenamedNode", "cluster_role": "master"})
    assert resp.status == 200
    data = await resp.json()
    assert data["applied"]["node_name"] == "RenamedNode"
    assert data["applied"]["cluster_role"] == "master"

    # Unknown fields are rejected without error
    resp = await client.patch("/api/config", json={"sneaky": "value"})
    assert resp.status == 200
    data = await resp.json()
    assert "sneaky" in data["rejected"]


@pytest.mark.asyncio
async def test_get_config_has_no_secrets(client):
    resp = await client.get("/api/config")
    data = await resp.json()
    assert "cluster_secret" not in data
    assert data["node_id"] == "test-node-1"
