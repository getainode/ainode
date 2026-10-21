"""``/api/status`` reports free space on the two directories a launch writes to.

A full models filesystem is the one resource failure on these nodes that shows
up as "the engine died" rather than as a disk error: the pull stops part way and
vLLM never binds. The dashboard could not see it coming because nothing in the
payload said anything about disk, so the number is here, per directory, with the
warning state already decided by the server (the tooltip and ``ainode doctor``
must not be able to disagree about what "low" means).
"""

import socket

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from ainode.api import server as srv
from ainode.api.server import create_app, disk_fields
from ainode.core.config import NodeConfig


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest_asyncio.fixture
async def client(tmp_path):
    config = NodeConfig(node_id="n1", node_name="TestNode", model=None,
                        api_port=_free_port(), web_port=_free_port(),
                        models_dir=str(tmp_path / "models"))
    async with TestClient(TestServer(create_app(config=config, engine=None))) as c:
        yield c


# ------------------------------------------------------------------ disk_fields

def test_both_directories_are_reported_with_real_figures(tmp_path, monkeypatch):
    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    fields = disk_fields(NodeConfig(models_dir=str(tmp_path)))
    assert set(fields) == {"home", "models"}
    for entry in fields.values():
        assert entry["total_gb"] > 0
        assert entry["free_gb"] >= 0
        assert 0.0 <= entry["free_fraction"] <= 1.0
        assert entry["warn"] is (entry["free_fraction"] < 0.15)


def test_under_fifteen_percent_free_sets_the_warning(tmp_path, monkeypatch):
    import shutil

    class Usage:
        total = 1000 * 1024 ** 3
        used = 900 * 1024 ** 3
        free = 100 * 1024 ** 3

    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage())
    fields = disk_fields(NodeConfig(models_dir=str(tmp_path)))
    assert fields["models"] == {"path": str(tmp_path), "total_gb": 1000.0,
                                "free_gb": 100.0, "free_fraction": 0.1, "warn": True}


def test_plenty_of_space_does_not_warn(tmp_path, monkeypatch):
    import shutil

    class Usage:
        total = 1000 * 1024 ** 3
        used = 100 * 1024 ** 3
        free = 900 * 1024 ** 3

    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage())
    assert disk_fields(NodeConfig(models_dir=str(tmp_path)))["home"]["warn"] is False


def test_a_path_we_cannot_stat_is_null_and_not_zero(tmp_path, monkeypatch):
    """Unknown and full are different answers, and a dashboard must not confuse them."""
    import shutil

    monkeypatch.setenv("AINODE_HOME", str(tmp_path))

    def boom(path):
        raise OSError("no such filesystem")

    monkeypatch.setattr(shutil, "disk_usage", boom)
    entry = disk_fields(NodeConfig(models_dir="/nowhere"))["models"]
    assert entry == {"path": "/nowhere", "total_gb": None, "free_gb": None,
                     "free_fraction": None, "warn": False}


def test_the_models_dir_defaults_under_the_ainode_home(tmp_path, monkeypatch):
    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    config = NodeConfig()
    config.models_dir = ""
    fields = disk_fields(config)
    assert fields["home"]["path"] == str(tmp_path)
    assert fields["models"]["path"] == str(tmp_path / "models")


# ------------------------------------------------------------------ /api/status

@pytest.mark.asyncio
async def test_status_carries_the_disk_block(client, tmp_path):
    resp = await client.get("/api/status")
    assert resp.status == 200
    disk = (await resp.json())["disk"]
    assert set(disk) == {"home", "models"}
    assert disk["models"]["path"] == str(tmp_path / "models")
    assert set(disk["home"]) == {"path", "total_gb", "free_gb", "free_fraction", "warn"}


@pytest.mark.asyncio
async def test_a_disk_probe_that_raises_does_not_break_status(client, monkeypatch):
    """The payload has to survive a filesystem question it cannot answer."""
    monkeypatch.setattr(srv, "disk_fields", lambda config: {})
    resp = await client.get("/api/status")
    assert resp.status == 200
    assert (await resp.json())["disk"] == {}
