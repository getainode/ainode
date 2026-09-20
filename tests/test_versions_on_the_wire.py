"""Every node's release, visible everywhere the cluster is.

Issue #171: nodes did not put their version on the wire, so a cluster running two
releases looked identical to one running a single release. A roll that missed a
node left exactly that state and nothing surfaced it, while the announcement is
the cross-version contract: 0.5.25 changed how a head's instances are merged into
``instances``, so a 0.5.24 reader and a 0.5.25 sender were exchanging different
data with no indication anywhere.

Issue #182 is the same question asked about an update: ``/api/cluster/update-status``
kept its job state in a module-level dict, and the master self-stops as the LAST
step of the job it is reporting on, so the record died with the process. A poll
after the master came back got a 404 for a job that had fully succeeded.

Fakes only: no engine, no peer, no container.
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
from types import SimpleNamespace

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from ainode import __version__
from ainode.api import server
from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeStatus
from ainode.discovery.cluster import ClusterNode


@pytest.fixture(autouse=True)
def ainode_home(monkeypatch, tmp_path):
    """Never read or write the operator's own ~/.ainode while testing this."""
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def empty_update_state(monkeypatch):
    """Each test starts with no jobs in memory, as a fresh process would."""
    monkeypatch.setattr(server, "_cluster_update_state", {})
    return server._cluster_update_state


@pytest.fixture
def config():
    # A port nothing listens on: /api/status probes localhost:<api_port> for
    # /v1/models and a real vLLM on 8000 would answer instead.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    return NodeConfig(node_id="spark1", node_name="Spark-1-DGX",
                      model="", api_port=free_port)


@pytest.fixture
def app(config):
    return server.create_app(config=config, engine=None)


@pytest_asyncio.fixture
async def client(app):
    async with TestClient(TestServer(app)) as c:
        yield c


def _peer(node_id: str, version: str, **extra) -> ClusterNode:
    fields = dict(
        node_id=node_id, node_name=node_id.title(), gpu_name="NVIDIA GB10",
        gpu_memory_gb=128.0, unified_memory=True, model="",
        status=NodeStatus.ONLINE, api_port=8000, web_port=3000,
        last_seen=time.time(), engine_status="serving",
        ainode_version=version,
    )
    fields.update(extra)
    return ClusterNode(**fields)


# ------------------------------------------------------------- /api/nodes ----


class TestNodeRows:
    @pytest.mark.asyncio
    async def test_our_own_row_reports_the_running_process(self, client):
        rows = (await (await client.get("/api/nodes")).json())["nodes"]
        assert rows[0]["node_id"] == "spark1"
        assert rows[0]["ainode_version"] == __version__

    @pytest.mark.asyncio
    async def test_a_peer_reports_what_it_announced(self, client, app):
        app["cluster_state"].add_node(_peer("spark2", "0.5.24"))
        rows = (await (await client.get("/api/nodes")).json())["nodes"]
        by_id = {r["node_id"]: r for r in rows}
        assert by_id["spark2"]["ainode_version"] == "0.5.24"
        assert by_id["spark1"]["ainode_version"] == __version__

    @pytest.mark.asyncio
    async def test_a_peer_too_old_to_announce_one_reports_empty_not_ours(
            self, client, app):
        """Absence is not agreement. Filling it in with our own version is how a
        split fleet would keep looking like a healthy one."""
        app["cluster_state"].add_node(_peer("ancient", ""))
        rows = (await (await client.get("/api/nodes")).json())["nodes"]
        row = next(r for r in rows if r["node_id"] == "ancient")
        assert row["ainode_version"] == ""

    @pytest.mark.asyncio
    async def test_the_cluster_view_carries_it_too(self, client, app):
        app["cluster_state"].add_node(_peer("spark2", "0.5.24"))
        data = await (await client.get("/api/cluster/resources")).json()
        by_id = {n["node_id"]: n for n in data["nodes"]}
        assert by_id["spark2"]["ainode_version"] == "0.5.24"
        assert by_id["spark1"]["ainode_version"] == __version__


# ------------------------------------------------------ /api/version/check ---


class TestVersionCheck:
    @pytest.mark.asyncio
    async def test_a_single_release_is_not_a_split(self, client, app, monkeypatch):
        monkeypatch.setattr(server, "_fetch_latest_ghcr_tag", lambda: __version__)
        app["cluster_state"].add_node(_peer("spark2", __version__))

        data = await (await client.get("/api/version/check")).json()
        assert data["current"] == __version__
        assert data["versions"] == [__version__]
        assert data["cluster_split"] is False
        assert data["unknown_versions"] == 0
        assert {n["node_id"] for n in data["nodes"]} == {"spark1", "spark2"}

    @pytest.mark.asyncio
    async def test_the_state_a_partial_roll_leaves_is_reported_as_a_split(
            self, client, app, monkeypatch):
        """The observed fleet: four Sparks pinned to 0.5.24 while two source
        installs ran 0.5.25, and only the master's own staleness was visible."""
        monkeypatch.setattr(server, "_fetch_latest_ghcr_tag", lambda: "0.5.25")
        for node_id in ("spark2", "spark3", "spark4"):
            app["cluster_state"].add_node(_peer(node_id, "0.5.24"))
        app["cluster_state"].add_node(_peer("castor", "0.5.25"))

        data = await (await client.get("/api/version/check")).json()
        assert data["cluster_split"] is True
        assert data["versions"] == ["0.5.24", "0.5.25", __version__]
        rows = {n["node_id"]: n["ainode_version"] for n in data["nodes"]}
        assert rows["spark3"] == "0.5.24"
        assert rows["castor"] == "0.5.25"

    @pytest.mark.asyncio
    async def test_versions_are_ordered_by_release_not_by_string(
            self, client, app, monkeypatch):
        monkeypatch.setattr(server, "_fetch_latest_ghcr_tag", lambda: None)
        app["cluster_state"].add_node(_peer("spark2", "0.5.9"))
        app["cluster_state"].add_node(_peer("spark3", "0.5.10"))

        data = await (await client.get("/api/version/check")).json()
        assert data["versions"][:2] == ["0.5.9", "0.5.10"]

    @pytest.mark.asyncio
    async def test_an_unknown_peer_version_is_counted_not_guessed(
            self, client, app, monkeypatch):
        monkeypatch.setattr(server, "_fetch_latest_ghcr_tag", lambda: None)
        app["cluster_state"].add_node(_peer("ancient", ""))

        data = await (await client.get("/api/version/check")).json()
        assert data["unknown_versions"] == 1
        assert data["versions"] == [__version__]
        assert data["cluster_split"] is False, (
            "one known version plus an unknown is not a proven split")


# ----------------------------------------------- /api/cluster/update-status ---


def _job(update_id: str, target: str, nodes: dict, status: str = "complete") -> dict:
    return {
        "id": update_id,
        "status": status,
        "target": target,
        "nodes": nodes,
        "started_at": time.time() - 120,
    }


class TestUpdateStatusSurvivesTheRestartItCauses:
    @pytest.mark.asyncio
    async def test_a_job_written_before_the_master_stopped_is_still_there_after(
            self, client, app, ainode_home):
        """The #182 sequence: create the job, update the workers, stop the
        master's own container, come back on the new image with an empty dict.
        The poll has to answer for the job that caused the restart."""
        job = _job("update-1700000000", __version__, {
            "spark1": {"node_name": "Spark-1", "status": "done",
                       "log": "Updated, restarting on the new image"},
            "spark2": {"node_name": "Spark-2", "status": "done", "log": "Updated"},
        })
        server._cluster_update_state[job["id"]] = job
        server._save_cluster_updates()

        # The process dies here. Everything in memory goes with it.
        server._cluster_update_state.clear()
        assert (ainode_home / "cluster-updates.json").exists()

        resp = await client.get("/api/cluster/update-status?id=update-1700000000")
        assert resp.status == 200
        data = await resp.json()
        assert data["id"] == "update-1700000000"
        assert data["status"] == "complete"
        assert set(data["nodes"]) == {"spark1", "spark2"}

    @pytest.mark.asyncio
    async def test_the_master_row_reports_the_version_it_actually_came_back_on(
            self, client, app):
        """A job record cannot answer this: it was written by the process that
        then stopped itself. The version on the wire can."""
        app["cluster_state"].add_node(_peer("spark2", __version__))
        app["cluster_state"].add_node(_peer("spark3", "0.5.24"))
        server._cluster_update_state["update-1700000001"] = _job(
            "update-1700000001", __version__, {
                "spark1": {"node_name": "Spark-1", "status": "done", "log": ""},
                "spark2": {"node_name": "Spark-2", "status": "done", "log": ""},
                "spark3": {"node_name": "Spark-3", "status": "done", "log": ""},
            })

        data = await (await client.get("/api/cluster/update-status")).json()
        rows = data["nodes"]
        assert rows["spark1"]["ainode_version"] == __version__
        assert rows["spark1"]["on_target"] is True
        assert rows["spark2"]["on_target"] is True
        # Reported "done" by the job, actually still on the old image.
        assert rows["spark3"]["ainode_version"] == "0.5.24"
        assert rows["spark3"]["on_target"] is False
        assert data["nodes_on_target"] == 2
        assert data["nodes_total"] == 3
        assert data["verified"] is False

    @pytest.mark.asyncio
    async def test_verified_is_true_only_when_every_node_is_on_the_target(
            self, client, app):
        app["cluster_state"].add_node(_peer("spark2", __version__))
        server._cluster_update_state["update-1700000002"] = _job(
            "update-1700000002", __version__, {
                "spark1": {"node_name": "Spark-1", "status": "done", "log": ""},
                "spark2": {"node_name": "Spark-2", "status": "done", "log": ""},
            })

        data = await (await client.get("/api/cluster/update-status")).json()
        assert data["verified"] is True
        assert data["nodes_on_target"] == 2

    @pytest.mark.asyncio
    async def test_a_node_that_is_not_announcing_a_version_is_unknown_not_failed(
            self, client, app):
        app["cluster_state"].add_node(_peer("ancient", ""))
        server._cluster_update_state["update-1700000003"] = _job(
            "update-1700000003", __version__, {
                "ancient": {"node_name": "ancient", "status": "done", "log": ""},
            })

        data = await (await client.get("/api/cluster/update-status")).json()
        assert data["nodes"]["ancient"]["on_target"] is None
        assert data["verified"] is False

    @pytest.mark.asyncio
    async def test_the_newest_job_answers_when_no_id_is_given(self, client, app):
        for stamp in ("update-1700000000", "update-1700000009", "update-1700000005"):
            server._cluster_update_state[stamp] = _job(stamp, __version__, {})
        data = await (await client.get("/api/cluster/update-status")).json()
        assert data["id"] == "update-1700000009"

    @pytest.mark.asyncio
    async def test_no_job_anywhere_is_still_a_404(self, client):
        resp = await client.get("/api/cluster/update-status")
        assert resp.status == 404

    def test_only_the_most_recent_jobs_are_kept_on_disk(self, ainode_home):
        for i in range(server.MAX_PERSISTED_UPDATE_JOBS + 5):
            update_id = f"update-17000000{i:02d}"
            server._cluster_update_state[update_id] = _job(update_id, "0.5.27", {})
        server._save_cluster_updates()

        saved = json.loads((ainode_home / "cluster-updates.json").read_text())
        assert len(saved) == server.MAX_PERSISTED_UPDATE_JOBS
        assert "update-1700000014" in saved, "the newest is kept"
        assert "update-1700000000" not in saved, "the oldest is dropped"

    def test_a_corrupt_state_file_is_ignored_rather_than_fatal(self, ainode_home):
        (ainode_home / "cluster-updates.json").write_text("{not json")
        assert server._load_cluster_updates() == {}

    def test_writing_the_state_never_raises(self, monkeypatch):
        """A job must not fail over its own log file."""
        monkeypatch.setattr(server, "_ainode_home_path",
                            lambda: (_ for _ in ()).throw(OSError("no home")))
        server._cluster_update_state["update-1700000099"] = _job(
            "update-1700000099", "0.5.27", {})
        server._save_cluster_updates()  # must not raise


class TestUpdateAllWritesAsItGoes:
    @pytest.mark.asyncio
    async def test_the_job_is_on_disk_before_the_master_touches_its_container(
            self, client, app, ainode_home, monkeypatch):
        """Not "at the end": the master's own row is written moments before it
        stops itself, so the file has to be current at every step (#182)."""
        import subprocess

        monkeypatch.setattr(server, "_fetch_latest_ghcr_tag", lambda: "9.9.9")
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="", stderr=""))
        # A node whose unit predates the swappable image: pulled and pinned, never
        # restarted, and the honest outcome recorded rather than "done".
        monkeypatch.setattr(server, "_unit_is_swappable", lambda: False)

        resp = await client.post("/api/cluster/update-all")
        assert resp.status == 202
        update_id = (await resp.json())["update_id"]
        for _ in range(100):
            await asyncio.sleep(0.01)
            if server._cluster_update_state[update_id]["status"] == "complete":
                break

        saved = json.loads((ainode_home / "cluster-updates.json").read_text())
        job = saved[update_id]
        assert job["target"] == "9.9.9"
        assert job["nodes"]["spark1"]["status"] == "needs-migration"
        assert job["nodes"]["spark1"]["version_before"] == __version__

        # And the poll reports it against the version on the wire: this node is
        # still running what it was, so it is NOT on the target.
        data = await (await client.get(f"/api/cluster/update-status?id={update_id}")).json()
        assert data["nodes"]["spark1"]["on_target"] is False
        assert data["verified"] is False


# ------------------------------------------------------------- the browser ---
#
# There is no JS test harness in this repo (tests/test_load_progress.py,
# tests/test_bench.py and others all assert on the text of app.js), so this
# checks the same way.


def _app_js() -> str:
    from ainode.web.serve import STATIC_DIR

    return (STATIC_DIR / "js" / "app.js").read_text()


class TestTheSplitBanner:
    def test_the_header_renders_a_banner_when_the_fleet_disagrees(self):
        js = _app_js()
        assert "renderVersionSplitBanner()" in js
        assert "version-split-banner" in js
        # Rendered next to the existing update badge, from the same poll.
        assert js.index("renderVersionSplitBanner()") > js.index("renderVersionBadge()")

    def test_the_banner_is_driven_by_the_per_node_versions(self):
        js = _app_js()
        body = js.split("renderVersionSplitBanner() {", 1)[1].split("\n  },", 1)[0]
        assert "ainode_version" in body
        assert "nodes" in body
        # No fetch of its own: it renders what the poll already brought back.
        assert "fetch(" not in body

    def test_a_node_with_no_version_is_drawn_as_unknown(self):
        js = _app_js()
        body = js.split("renderVersionSplitBanner() {", 1)[1].split("\n  },", 1)[0]
        assert "unknown" in body
