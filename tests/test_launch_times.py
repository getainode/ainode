"""The launch-time ledger: how long a model took to come up, on this node.

Before this, the only trace of a load time was one log line ("<label> bound on
:<port> after <N>s"), which nothing reads and no browser can see, so the
interface could not tell a user that a 27B NVFP4 on a GB10 takes about 12
minutes. The bind wait now appends its verdict to
``<AINODE_HOME>/launch-times.json``.

Two things these tests pin, because both are the point of the file:

* FAILURES are recorded, with the reason. A ledger of successes only would
  describe a model that never comes up as a model that comes up fast.
* The write happens OFF the event loop. The bind wait runs on the server's loop
  during the boot replay, and every other file write on that path is already in
  the executor.

Fakes only: no engine, no container, no node.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from ainode.models import api_routes


@pytest.fixture(autouse=True)
def ledger_home(monkeypatch, tmp_path):
    """Never touch the operator's own ~/.ainode while testing a ledger."""
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    return tmp_path


def _cfg(**kw):
    base = dict(model="unsloth/Qwen3.8-27B-NVFP4", node_id="spark1",
                node_name="Spark-1-DGX", api_port=8000,
                engine_image="vllm/vllm-openai:v0.27.1")
    base.update(kw)
    return SimpleNamespace(**base)


def _app(node_cfg=None, manager=None):
    app = {"config": node_cfg if node_cfg is not None else _cfg(model="")}
    if manager is not None:
        app["instances"] = manager
    return app


def _bound_engine(alive_s=720.0, **cfg_kw):
    """An engine whose port answers: no log stamp, no activity probe, no process,
    which is the fixed-window branch of the wait, the shortest path to a verdict."""
    return SimpleNamespace(config=_cfg(**cfg_kw), launched_at=time.time() - alive_s)


def _dead_engine(alive_s=30.0, **cfg_kw):
    """An engine whose container has exited: the adaptive branch's death signal."""
    return SimpleNamespace(config=_cfg(**cfg_kw), launched_at=time.time() - alive_s,
                           process=SimpleNamespace(poll=lambda: 1))


def _serving(monkeypatch, answer):
    async def _probe(port):
        return answer
    monkeypatch.setattr(api_routes, "_port_serving", _probe)


# ------------------------------------------------------------ append on bind --

def test_a_successful_bind_is_written_down_with_everything_the_ui_needs(monkeypatch):
    _serving(monkeypatch, True)
    engine = _bound_engine(alive_s=725.0)
    manager = SimpleNamespace(instances=lambda: [
        SimpleNamespace(record=SimpleNamespace(api_port=8000, tensor_parallel_size=1))])

    bound, reason, alive = asyncio.run(api_routes._wait_for_bind(
        _app(manager=manager), 8000, engine))

    assert (bound, reason) == (True, "bound")
    assert 720 <= alive <= 730
    rows = api_routes.read_launch_times()
    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "unsloth/Qwen3.8-27B-NVFP4"
    assert row["node_id"] == "spark1"
    assert row["node_name"] == "Spark-1-DGX"
    assert row["api_port"] == 8000
    assert row["stacked"] is False
    assert row["tensor_parallel_size"] == 1
    assert row["engine_image"] == "vllm/vllm-openai:v0.27.1"
    assert 720 <= row["seconds_to_ready"] <= 730
    assert row["outcome"] == "ready"
    assert row["stamp"].endswith("Z") and row["stamp"][4] == "-"
    assert "reason" not in row


def test_an_instance_on_its_own_port_is_recorded_as_stacked(monkeypatch):
    """The primary keeps the node's api_port; every stacked load gets an allocated
    one, which is the only difference the ledger needs to report honestly: a
    stacked 12 minutes and a solo 12 minutes are not the same measurement."""
    _serving(monkeypatch, True)
    engine = _bound_engine(model="ornith-ai/Ornith-1.5-35B-A3B-NVFP4")

    asyncio.run(api_routes._wait_for_bind(_app(), 8001, engine))

    row = api_routes.read_launch_times()[-1]
    assert row["api_port"] == 8001
    assert row["stacked"] is True


def test_a_failed_launch_is_recorded_with_its_reason(monkeypatch):
    """An honest ledger. A model whose engine dies on the way up must not be
    missing from the record, or the UI would report the last time it DID work as
    if nothing had changed."""
    _serving(monkeypatch, False)
    engine = _dead_engine(alive_s=47.0)

    bound, reason, _alive = asyncio.run(api_routes._wait_for_bind(_app(), 8000, engine))

    assert bound is False and reason == "container exited"
    row = api_routes.read_launch_times()[-1]
    assert row["outcome"] == "failed"
    assert row["reason"] == "container exited"
    assert row["seconds_to_ready"] == pytest.approx(47, abs=2)


def test_a_wait_that_cannot_name_a_model_records_nothing(monkeypatch):
    """The ledger is looked up by model id, so a row keyed on "" is worse than no
    row. A backend with no config and a node config for a different port cannot
    name one."""
    _serving(monkeypatch, True)
    engine = SimpleNamespace(launched_at=time.time() - 60)

    asyncio.run(api_routes._wait_for_bind(_app(), 8123, engine))

    assert api_routes.read_launch_times() == []


def test_the_model_falls_back_to_the_node_config_on_the_primary_port(monkeypatch):
    """The boot primary's engine is launched before the web app exists; if its
    handle publishes no config, the node's own config describes that port."""
    _serving(monkeypatch, True)
    engine = SimpleNamespace(launched_at=time.time() - 60)
    app = _app(_cfg(model="nvidia/Llama-3.1-8B-Instruct-NVFP4", api_port=8000))

    asyncio.run(api_routes._wait_for_bind(app, 8000, engine))

    row = api_routes.read_launch_times()[-1]
    assert row["model"] == "nvidia/Llama-3.1-8B-Instruct-NVFP4"
    assert row["stacked"] is False


# ------------------------------------------------------------------- the cap --

def test_the_ledger_keeps_the_last_two_hundred_entries():
    for i in range(_over := api_routes._LAUNCH_TIMES_CAP + 25):
        api_routes.append_launch_time({"model": f"m/{i}", "outcome": "ready",
                                       "seconds_to_ready": float(i)})

    rows = api_routes.read_launch_times()
    assert api_routes._LAUNCH_TIMES_CAP == 200
    assert len(rows) == 200
    # The OLDEST go, not the newest: the UI wants the most recent launch.
    assert rows[0]["model"] == f"m/{_over - 200}"
    assert rows[-1]["model"] == f"m/{_over - 1}"


def test_a_junk_ledger_file_reads_as_empty_and_is_still_appendable(ledger_home):
    (ledger_home / "launch-times.json").write_text("{not json")
    assert api_routes.read_launch_times() == []
    api_routes.append_launch_time({"model": "a/b", "outcome": "ready"})
    assert [r["model"] for r in api_routes.read_launch_times()] == ["a/b"]


# ------------------------------------------------------------- off the loop --

def test_the_write_happens_off_the_event_loop(monkeypatch):
    """The bind wait runs on the server's loop during boot replay, and a read plus
    a write of a JSON file is not something to do there."""
    seen: dict = {}

    def _spy(entry):
        seen["thread"] = threading.get_ident()

    monkeypatch.setattr(api_routes, "append_launch_time", _spy)

    async def _go():
        seen["loop_thread"] = threading.get_ident()
        await api_routes.record_launch_time(
            _app(), 8000, _bound_engine(), seconds=700.0, outcome="ready")

    asyncio.run(_go())
    assert seen["thread"] != seen["loop_thread"]


def test_a_ledger_write_that_explodes_does_not_fail_the_launch(monkeypatch):
    def _boom(entry):
        raise OSError("disk full")

    monkeypatch.setattr(api_routes, "append_launch_time", _boom)
    _serving(monkeypatch, True)

    bound, reason, _alive = asyncio.run(
        api_routes._wait_for_bind(_app(), 8000, _bound_engine()))

    assert (bound, reason) == (True, "bound")


# ------------------------------------------------------------------ lookups --

_ROWS = [
    {"model": "unsloth/Qwen3.8-27B-NVFP4", "outcome": "ready",
     "seconds_to_ready": 300.0, "node_name": "Spark-1-DGX", "stacked": False,
     "stamp": "2026-09-15T04:00:00Z", "tensor_parallel_size": 1},
    {"model": "unsloth/Qwen3.8-27B-NVFP4", "outcome": "failed",
     "seconds_to_ready": 60.0, "node_name": "Spark-1-DGX", "stacked": True,
     "stamp": "2026-09-17T04:00:00Z", "reason": "container exited"},
    {"model": "unsloth/Qwen3.8-27B-NVFP4", "outcome": "ready",
     "seconds_to_ready": 726.0, "node_name": "Spark-1-DGX", "stacked": True,
     "stamp": "2026-09-16T11:02:03Z", "tensor_parallel_size": 1},
]


def test_the_summary_is_the_most_recent_SUCCESSFUL_launch():
    """Most recent by position, and a failure never becomes the reported time."""
    got = api_routes.launch_time_summary("unsloth/Qwen3.8-27B-NVFP4", entries=_ROWS)
    assert got["last_ready_minutes"] == 12.1
    assert got["last_ready_on"] == {"node_name": "Spark-1-DGX", "stacked": True,
                                   "date": "2026-09-16", "tensor_parallel_size": 1}


def test_the_summary_matches_a_catalog_id_against_the_repo_the_engine_ran():
    """The engine is launched with the HF repo; the launch dropdown may ask by
    catalog id. Either has to find the row."""
    got = api_routes.launch_time_summary("qwen3.8-27b-nvfp4",
                                         "unsloth/Qwen3.8-27B-NVFP4", entries=_ROWS)
    assert got["last_ready_minutes"] == 12.1


def test_a_model_this_node_never_brought_up_has_no_measurement():
    got = api_routes.launch_time_summary("zai-org/GLM-5.1", entries=_ROWS)
    assert got == {"last_ready_minutes": None, "last_ready_on": None}


def test_the_catalog_listing_carries_the_measured_time_beside_the_seed(monkeypatch):
    """GET /api/models is what the launch dropdown reads, so the answer to "how
    long will this take" has to be in it."""
    monkeypatch.setattr(api_routes, "read_launch_times", lambda: list(_ROWS))
    entries = api_routes.annotate_launch_times([
        {"id": "qwen3.8-27b-nvfp4", "hf_repo": "unsloth/Qwen3.8-27B-NVFP4",
         "typical_ready_minutes": 12.0},
        {"id": "glm-5.1", "hf_repo": "zai-org/GLM-5.1",
         "typical_ready_minutes": None},
    ])

    assert entries[0]["typical_ready_minutes"] == 12.0
    assert entries[0]["last_ready_minutes"] == 12.1
    assert entries[0]["last_ready_on"]["node_name"] == "Spark-1-DGX"
    assert entries[1]["last_ready_minutes"] is None
    assert entries[1]["last_ready_on"] is None


# --------------------------------------------------------------- the browser --
#
# There is no JS test harness in this repo (tests/test_chat_routes.py and
# tests/test_bench.py both assert on the text of app.js), so these check the same
# way: the markup hook exists, both phrasings exist, and the absent case returns
# nothing rather than a zero.

def test_the_launch_panel_has_a_place_for_the_load_time():
    from ainode.web.serve import get_index_html
    assert 'id="launch-load-time"' in get_index_html()


def test_the_launch_line_says_the_measured_node_and_date_when_it_has_them():
    from ainode.web.serve import STATIC_DIR
    js = (STATIC_DIR / "js" / "app.js").read_text()
    assert "loadTimeLine(m) {" in js
    assert "'Typical load: about '" in js
    assert "', measured '" in js
    assert "' stacked on '" in js
    # Nothing measured, nothing said: no estimate off the weight size. Two bare
    # returns, one for no entry at all and one for an entry with neither number,
    # and the function ends on one of them.
    body = js.split("loadTimeLine(m) {", 1)[1].split("\n  },", 1)[0]
    assert body.count("return '';") == 2
    assert body.rstrip().endswith("return '';")


def test_the_card_renders_a_measured_loaded_in_row_and_a_dated_verified_chip():
    from ainode.web.serve import STATIC_DIR
    js = (STATIC_DIR / "js" / "app.js").read_text()
    assert "card.load_time" in js
    assert "['Loaded in'," in js
    assert "'Tested on AINode '" in js
    assert "'Marked verified before the bench existed'" in js
