"""Live load progress: how far into a launch a node is, right now.

0.5.21 gave the interface how long a model took LAST time (the launch-time
ledger plus the catalog's ``typical_ready_minutes`` seed). What it still could
not say was anything about the load happening at this moment, so a twelve-minute
GB10 launch showed a fixed percent per phase that never moved and read as a
hang.

``/api/status`` now carries three fields, and this is what they mean:

* ``load_started_at`` -- epoch seconds, from the engine backend's own
  ``launched_at`` wherever it has one, so the number survives a page reload and
  is the same in every browser. The mp distributed shape leaves it None until
  the head stamps it, and then the API layer stamps it once, in one place.
* ``load_elapsed_seconds`` -- the server's arithmetic, so the browser never has
  to guess when a load it did not watch began.
* ``expected_ready_minutes`` -- this node's ledger first, then the catalog seed,
  then null. Never a guess off the weight size.

All three are null whenever nothing is loading, which is the signal a client
reads as "no bar to draw".

Fakes only: no engine, no container, no node.
"""

from __future__ import annotations

import socket
import time
from types import SimpleNamespace

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from ainode.api import server
from ainode.core.config import NodeConfig
from ainode.models import api_routes

# A curated entry that states a seed, so the catalog fallback is tested against
# the real catalog rather than a mock of it.
SEEDED_MODEL = "ornith-ai/Ornith-1.5-35B-A3B-NVFP4"
SEEDED_MINUTES = 12.0


@pytest.fixture(autouse=True)
def ledger_home(monkeypatch, tmp_path):
    """Never read or write the operator's own ~/.ainode while testing this."""
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    return tmp_path


@pytest.fixture
def config():
    # A port nothing listens on: the status handler probes localhost:<api_port>
    # for /v1/models, and a real vLLM on 8000 would answer instead.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    return NodeConfig(node_id="spark1", node_name="Spark-1-DGX",
                      model=SEEDED_MODEL, api_port=free_port)


@pytest.fixture
def app(config):
    return server.create_app(config=config, engine=None)


@pytest_asyncio.fixture
async def client(app):
    async with TestClient(TestServer(app)) as c:
        yield c


def _engine(phase, launched_at=None, model=SEEDED_MODEL):
    """An engine that reports a coarse phase and (maybe) its launch stamp."""
    return SimpleNamespace(
        load_phase=phase,
        launched_at=launched_at,
        ready=False,
        config=SimpleNamespace(model=model),
        is_running=lambda: True,
    )


async def _status(client):
    resp = await client.get("/api/status")
    assert resp.status == 200
    return await resp.json()


def _ready_row(model, seconds, node_id="spark1"):
    return {"model": model, "outcome": "ready", "seconds_to_ready": seconds,
            "node_id": node_id, "node_name": "Spark-1-DGX", "api_port": 8000,
            "stacked": False, "stamp": "2026-09-16T10:00:00Z"}


# ------------------------------------------------------------------ idle node --

@pytest.mark.asyncio
async def test_an_idle_node_reports_no_load_at_all(client):
    """No engine, nothing loading: three nulls, not zeroes. A zero elapsed is a
    load that just started, which is a different claim."""
    data = await _status(client)

    assert data["load_phase"] == "idle"
    assert data["load_started_at"] is None
    assert data["load_elapsed_seconds"] is None
    assert data["expected_ready_minutes"] is None


@pytest.mark.asyncio
async def test_a_ready_engine_reports_no_load_either(client, app, monkeypatch):
    """A serving engine has no load in flight. The fields describe the launch,
    not the instance, so they go quiet the moment it is up."""
    app["engine"] = _engine("profiling", launched_at=time.time() - 300)
    app[server._LOAD_START_KEY] = time.time() - 300

    async def _probe(_app, port):
        return True

    monkeypatch.setattr(server, "_engine_port_serving", _probe)
    # engine_ready comes from the status handler's own /v1/models probe, which
    # cannot reach this fixture's dead port; drive it from the engine instead.
    app["engine"].ready = True
    app["engine"].health_check = lambda: {"api_responding": True,
                                          "models_loaded": [SEEDED_MODEL]}
    app["client_session"] = None

    data = await _status(client)

    assert data["load_phase"] == "ready"
    assert data["load_started_at"] is None
    assert data["load_elapsed_seconds"] is None
    assert data["expected_ready_minutes"] is None
    assert app[server._LOAD_START_KEY] is None, "the stamp is reset on ready"


# ----------------------------------------------------------- a load in flight --

@pytest.mark.asyncio
async def test_a_loading_engine_is_timed_from_the_backends_own_launch_stamp(client, app):
    """``launched_at`` is stamped when the container starts and cleared in
    stop(), so it is the one clock every client can agree on."""
    launched = time.time() - 195.0
    app["engine"] = _engine("loading_weights", launched_at=launched)

    data = await _status(client)

    assert data["load_phase"] == "loading_weights"
    assert data["load_started_at"] == pytest.approx(launched, abs=0.01)
    assert 195.0 <= data["load_elapsed_seconds"] <= 200.0


@pytest.mark.asyncio
async def test_a_shape_that_has_not_stamped_yet_still_gets_a_clock(client, app):
    """The mp distributed shape leaves ``launched_at`` None until the head
    stamps it. A load with no start time cannot be drawn at all, so the API
    layer stamps the first read past idle itself."""
    app["engine"] = _engine("distributed_init", launched_at=None)

    first = await _status(client)
    assert first["load_started_at"] is not None
    assert first["load_elapsed_seconds"] >= 0.0
    assert app[server._LOAD_START_KEY] == pytest.approx(first["load_started_at"], abs=0.01)

    second = await _status(client)
    assert second["load_started_at"] == first["load_started_at"], (
        "one stamp per load, not one per poll")


@pytest.mark.asyncio
async def test_the_backends_stamp_wins_over_the_fallback(client, app):
    """The fallback is late by up to one poll; the container's own stamp is not.
    Once the head has stamped, that is the number reported."""
    app[server._LOAD_START_KEY] = time.time() - 10.0
    launched = time.time() - 420.0
    app["engine"] = _engine("profiling", launched_at=launched)

    data = await _status(client)

    assert data["load_started_at"] == pytest.approx(launched, abs=0.01)
    assert data["load_elapsed_seconds"] >= 420.0


@pytest.mark.asyncio
async def test_the_fallback_stamp_is_dropped_when_the_load_settles(app):
    """Reset on ready or exit, in one place, or the next launch would be timed
    from the last one's start."""
    engine = _engine("loading_weights", launched_at=None)
    first = server.load_started_at(app, engine, "loading_weights")
    assert first is not None

    assert server.load_started_at(app, engine, "ready") is None
    assert app[server._LOAD_START_KEY] is None

    later = server.load_started_at(app, engine, "starting")
    assert later is not None and later > first


# ------------------------------------------------------- expected_ready_minutes --

@pytest.mark.asyncio
async def test_the_expected_time_comes_from_this_nodes_ledger_first(client, app):
    """What this node actually did last time beats the catalog's seed: it knows
    the node, the parallelism and the engine image."""
    api_routes.append_launch_time(_ready_row(SEEDED_MODEL, 402.0))
    app["engine"] = _engine("loading_weights", launched_at=time.time() - 30)

    data = await _status(client)

    assert data["expected_ready_minutes"] == pytest.approx(6.7, abs=0.05)
    assert data["expected_ready_minutes"] != SEEDED_MINUTES


@pytest.mark.asyncio
async def test_a_failed_launch_is_not_an_expected_time(client, app):
    """The ledger records failures too. A model that died at 40 s must not make
    the interface promise a 40-second launch."""
    api_routes.append_launch_time({"model": SEEDED_MODEL, "outcome": "failed",
                                   "seconds_to_ready": 40.0,
                                   "reason": "container exited"})
    app["engine"] = _engine("loading_weights", launched_at=time.time() - 30)

    data = await _status(client)

    assert data["expected_ready_minutes"] == SEEDED_MINUTES, "the catalog seed"


@pytest.mark.asyncio
async def test_the_catalog_seed_is_the_fallback(client, app):
    """An empty ledger (a fresh install) still knows roughly how long a curated
    model takes on this class of hardware."""
    app["engine"] = _engine("starting", launched_at=time.time() - 5)

    data = await _status(client)

    assert data["expected_ready_minutes"] == SEEDED_MINUTES


@pytest.mark.asyncio
async def test_an_unknown_model_expects_nothing(client, app):
    """No ledger row, no catalog entry: null. A number guessed off the weight
    size is worse than no bar, because the UI would draw it."""
    app["engine"] = _engine("loading_weights", launched_at=time.time() - 60,
                            model="someone/never-launched-here")

    data = await _status(client)

    assert data["load_started_at"] is not None, "it is still loading"
    assert data["expected_ready_minutes"] is None


@pytest.mark.asyncio
async def test_the_engines_model_is_what_gets_timed(client, app):
    """A load started through /api/engine/set-model runs ahead of config.model,
    so the expected time must follow the engine, not the node config."""
    api_routes.append_launch_time(_ready_row("someone/other-model", 120.0))
    app["engine"] = _engine("loading_weights", launched_at=time.time() - 10,
                            model="someone/other-model")

    data = await _status(client)

    assert data["expected_ready_minutes"] == pytest.approx(2.0, abs=0.05)


def test_expected_minutes_resolution_order_without_a_server():
    """The same order, straight off the helper: ledger, catalog, nothing."""
    assert server.expected_ready_minutes(SEEDED_MODEL) == SEEDED_MINUTES
    api_routes.append_launch_time(_ready_row(SEEDED_MODEL, 300.0))
    assert server.expected_ready_minutes(SEEDED_MODEL) == 5.0
    assert server.expected_ready_minutes("") is None
    assert server.expected_ready_minutes("nobody/nothing") is None


# --------------------------------------------------- the same fields per node --

@pytest.mark.asyncio
async def test_a_peer_loading_a_model_is_visible_on_this_dashboard(client, app):
    """The master draws progress for a load on ANOTHER node from what that node
    announced. Its own elapsed figure is passed through: the cluster shares no
    clock, so subtracting a remote start stamp from our now() would report skew
    as progress."""
    from ainode.discovery.cluster import ClusterNode
    from ainode.discovery.broadcast import NodeStatus

    app["cluster_state"].add_node(ClusterNode(
        node_id="spark2", node_name="Spark-2-DGX", gpu_name="GB10",
        gpu_memory_gb=128.0, unified_memory=True, model=SEEDED_MODEL,
        status=NodeStatus.ONLINE, api_port=8000, web_port=3000,
        last_seen=time.time(), engine_status="starting",
        load_phase="loading_weights", load_started_at=time.time() - 240.0,
        load_elapsed_seconds=240.0, expected_ready_minutes=12.0,
    ))

    resp = await client.get("/api/nodes")
    rows = (await resp.json())["nodes"]
    row = next(r for r in rows if r["node_id"] == "spark2")

    assert row["engine_ready"] is False
    assert row["load_phase"] == "loading_weights"
    assert row["load_elapsed_seconds"] == 240.0
    assert row["expected_ready_minutes"] == 12.0


@pytest.mark.asyncio
async def test_a_peer_from_an_older_build_reports_no_progress(client, app):
    """A peer that sends none of this is not loading as far as the UI is
    concerned, which is the same thing it does for an idle node."""
    from ainode.discovery.cluster import ClusterNode
    from ainode.discovery.broadcast import NodeStatus

    app["cluster_state"].add_node(ClusterNode(
        node_id="old-node", node_name="old-node", gpu_name="GB10",
        gpu_memory_gb=128.0, unified_memory=True, model="",
        status=NodeStatus.ONLINE, api_port=8000, web_port=3000,
        last_seen=time.time(), engine_status="starting",
    ))

    resp = await client.get("/api/nodes")
    rows = (await resp.json())["nodes"]
    row = next(r for r in rows if r["node_id"] == "old-node")

    assert row["load_started_at"] is None
    assert row["load_elapsed_seconds"] is None
    assert row["expected_ready_minutes"] is None


@pytest.mark.asyncio
async def test_the_local_row_is_timed_here_not_announced(client, app):
    """Our own row never reads from the broadcast: the engine is one attribute
    away."""
    app["engine"] = _engine("profiling", launched_at=time.time() - 88.0)

    resp = await client.get("/api/nodes")
    row = (await resp.json())["nodes"][0]

    assert row["node_id"] == "spark1"
    assert row["load_phase"] == "profiling"
    assert 88.0 <= row["load_elapsed_seconds"] <= 93.0
    assert row["expected_ready_minutes"] == SEEDED_MINUTES


def test_a_full_announcement_still_fits_the_listeners_buffer():
    """The size check. The announcement has grown field by field (telemetry,
    instances, now load progress) and a payload past the listener's one-datagram
    read is TRUNCATED on arrival: it fails to parse and the node disappears from
    every peer's cluster view with nothing to say why."""
    from ainode.discovery.broadcast import MAX_ANNOUNCEMENT_BYTES, NodeAnnouncement

    ann = NodeAnnouncement(
        node_id="spark1-head", node_name="Spark-1-DGX", gpu_name="NVIDIA GB10",
        gpu_memory_gb=128.0, unified_memory=True,
        model="unsloth/Qwen3.8-27B-NVFP4", status="starting",
        api_port=8000, web_port=3000, fabric_ip="10.100.0.1",
        distributed_mode="head", distributed_instance_id="spark1-head:big-moe",
        distributed_peers=["10.100.0.2", "10.100.0.3", "10.100.0.4"],
        instances=[{"instance_id": f"spark1-head:model-{i}",
                    "model": "ornith-ai/Ornith-1.5-35B-A3B-NVFP4",
                    "head_node_id": "spark1-head", "peer_ips": ["10.100.0.2"],
                    "api_port": 8000 + i, "tensor_parallel_size": 2,
                    "status": "serving", "distributed_executor": "mp"}
                   for i in range(4)],
        load_phase="loading_weights", load_started_at=time.time(),
        load_elapsed_seconds=195.3, expected_ready_minutes=12.0,
    )

    assert len(ann.to_json().encode()) < MAX_ANNOUNCEMENT_BYTES


# --------------------------------------------------------------- the browser --
#
# There is no JS test harness in this repo (tests/test_launch_times.py,
# tests/test_chat_routes.py and tests/test_bench.py all assert on the text of
# app.js), so these check the same way: the math lives in ONE function, all three
# surfaces call it, and the ticking is local rather than a second poll.


def _app_js():
    from ainode.web.serve import STATIC_DIR
    return (STATIC_DIR / "js" / "app.js").read_text()


def test_the_progress_math_is_one_pure_function():
    js = _app_js()
    assert "loadProgress(status, nowMs) {" in js
    body = js.split("loadProgress(status, nowMs) {", 1)[1].split("\n  },", 1)[0]
    # Everything it needs comes in as arguments: no fetch, no DOM, no wall clock
    # of its own beyond the nowMs it was handed.
    assert "fetch(" not in body
    assert "document." not in body
    assert "Date.now()" in body, "only as the default for a missing nowMs"
    assert body.count("Date.now()") == 1
    # The four things the surfaces draw from.
    for key in ("percent", "elapsedSeconds", "expectedSeconds", "slow"):
        assert key in body


def test_progress_is_computed_from_the_expected_time_with_a_phase_floor():
    js = _app_js()
    body = js.split("loadProgress(status, nowMs) {", 1)[1].split("\n  },", 1)[0]
    # min(95, 100 * elapsed / expected), never below the phase's floor.
    assert "Math.min(95, Math.round(100 * elapsed / out.expectedSeconds))" in body
    assert "Math.max(info[1]," in body
    # Past 1.5x expected it says so out loud.
    assert "elapsed > 1.5 * out.expectedSeconds" in body
    assert "' · taking longer than usual'" in js


def test_the_phase_floors_survive_as_the_no_timing_fallback():
    """A peer on an older build sends no timing. The phase percent is still the
    honest thing to draw, so the old table stays as the floor and the fallback."""
    js = _app_js()
    assert "LOAD_PHASES: {" in js
    for phase in ("loading_weights", "distributed_init", "profiling"):
        assert phase in js.split("LOAD_PHASES: {", 1)[1].split("},", 1)[0]


def test_all_three_surfaces_draw_the_same_line_and_bar():
    js = _app_js()
    # The chip, the catalog card and the launch panel, each through the shared
    # helpers rather than their own arithmetic.
    assert "self.loadTextHtml(inst.loadKey, prog, 'chip'" in js
    assert "self.loadBarHtml(inst.loadKey, prog, 'instance-progress')" in js
    assert "'<div class=\"download-card-loading\">' +" in js
    assert "this.loadTextHtml(key, prog, 'line') + this.loadBarHtml(key, prog, 'launch-progress')" in js
    # And the launch panel's typical line comes from the same function as the
    # live one, so they cannot disagree.
    assert "this.loadTimeLine((this.state.launchLoadTimes || {})[repo], prog)" in js
    assert "'Loading… '" in js


def test_a_loading_model_cannot_be_launched_again_from_its_card():
    js = _app_js()
    assert "id=\"md-launch\" disabled>Loading…</button>" in js
    assert "if (launchBtn && !launchBtn.disabled) {" in js


def test_the_clock_ticks_locally_and_adds_no_polling():
    js = _app_js()
    assert "self.tickLoadProgress(); }, 1000)" in js
    tick = js.split("tickLoadProgress() {", 1)[1].split("\n  },", 1)[0]
    assert "fetch(" not in tick, "the tick reads the last poll, it does not poll"
    # Elapsed advances from the server's own figure plus the time since it
    # arrived, so a page opened mid-load is right on its first paint.
    assert "_receivedAtMs = receivedAt" in js
    assert "status._receivedAtMs" in js


def test_the_progress_bar_has_a_style_and_a_slow_state():
    from ainode.web.serve import STATIC_DIR
    css = (STATIC_DIR / "css" / "style.css").read_text()
    assert ".load-progress {" in css
    assert ".load-progress-fill {" in css
    assert ".load-progress.slow .load-progress-fill {" in css
