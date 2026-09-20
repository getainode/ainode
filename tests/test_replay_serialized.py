"""One engine launches at a time on a node, and the sweep runs before any of them.

Rolling Spark-1 from 0.5.8 to 0.5.11 (2026-09-14, issue #96) produced this from
``docker events`` (seconds relative to the update):

    +0    die    ainode-vllm-node-solo       exit=0    old primary, stopped by the update
    +7    start  ainode-vllm-node-solo                 new primary
    +21   die    ainode-vllm-node-solo-8001  exit=137  OLD stacked engine, killed 14 s
                                                       AFTER the new primary started
    +54   die    ainode-vllm-node-solo       exit=1    primary attempt 1, after 47 s
    +55   start  ainode-vllm-node-solo                 primary relaunch
    +57   start  ainode-vllm-node-solo-8001            stacked launch, 2 s later
    +652  die    ainode-vllm-node-solo-8001  exit=1    "Available KV cache memory: 1.59 GiB"

vLLM sizes its KV cache from what is FREE when the engine profiles, so the two
overlaps here (an old engine still resident, and a stacked engine profiling while
the primary loads) both under-provision whichever engine finishes second. The same
Ornith recipe at the same gpu_memory_utilization had a 600K-token KV cache when it
was launched onto a settled node.

So: sweep every engine container this node owns and WAIT for it to be gone before
anything launches, then launch one engine at a time per node, each waiting for the
one before it to bind. The AINode log line also lied about the duration ("never
bound ... after 0s" for a container that lived 47 s) because it timed the wait
rather than the container; it reports the container now.

Fakes only: no docker, no real engine, and no sleeps beyond the bind loop's own
scaled-down cadence.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from types import SimpleNamespace

import pytest

from ainode.cli import main as cli
from ainode.models import api_routes

# Captured before any test patches ``asyncio.sleep``: _settle needs a real yield
# to the loop, and several tests replace sleep with a no-op that never yields.
_real_sleep = asyncio.sleep


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_launch_state(monkeypatch):
    """Per-test module state: no sweep claimed yet, and nothing holding the slot."""
    monkeypatch.setattr(api_routes, "_BOOT_SWEEP_DONE", False)
    monkeypatch.setattr(api_routes, "_BIND_POLL_SECONDS", 0.005)
    monkeypatch.setattr(api_routes, "_GPU_RELEASE_SECONDS", 0.0)
    monkeypatch.setattr(api_routes, "_REPLAY_SETTLE_SECONDS", 0.0)
    yield
    api_routes.release_launch_slot()


class _Cfg:
    api_port = 8000
    model = ""
    engine_bind_log_silence_seconds = 120
    engine_bind_ceiling_seconds = 1800

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def save(self):
        pass


class _Req:
    """The two things a route handler uses: ``.app`` and ``await .json()``."""

    def __init__(self, app, body):
        self.app = app
        self._body = body

    async def json(self):
        return self._body


async def _settle(predicate, tries: int = 50) -> None:
    """Yield to the loop (zero-duration) until ``predicate`` holds."""
    for _ in range(tries):
        if predicate():
            return
        await _real_sleep(0)
    raise AssertionError("predicate never became true")


# ---------------------------------------------------------------------------
# 1. The sweep finishes before anything launches
# ---------------------------------------------------------------------------

def test_the_boot_sweep_removes_every_engine_this_node_owns(monkeypatch):
    """Not just the stacked ones. The old primary and a distributed head hold
    memory too, and the boot sweep runs before this process has launched anything,
    so widening the filter there is safe (and the point)."""
    assert api_routes._engine_name_filters(include_primary=False) == [
        "--filter", "name=ainode-vllm-node-solo-",
    ]
    wide = api_routes._engine_name_filters(include_primary=True)
    assert "name=ainode-vllm-node-solo" in wide, "the primary is swept at boot"
    assert "name=ainode-vllm-head" in wide, "so is a distributed head"
    assert "name=ainode-vllm-node-solo-" not in wide, \
        "the stacked engines come along with the primary prefix"
    assert not any("worker" in f for f in wide), \
        "peer worker containers belong to whichever head placed them"


def test_the_boot_sweep_waits_for_the_containers_to_be_gone(monkeypatch):
    """`docker rm -f` returns while the daemon is still deleting a --rm container,
    so the sweep polls until the filter comes back empty (#80)."""
    events = []
    monkeypatch.setattr(api_routes, "_remove_engine_containers",
                        lambda include_primary: events.append(("rm", include_primary)) or ["id1"])
    polls = iter([["id1"], ["id1"], []])
    monkeypatch.setattr(api_routes, "_engine_container_ids",
                        lambda include_primary: events.append("poll") or next(polls))
    monkeypatch.setattr(api_routes.time, "sleep", lambda _s: events.append("sleep"))

    assert api_routes.sweep_engines_before_boot() == ["id1"]
    assert events == [("rm", True), "poll", "sleep", "poll", "sleep", "poll"]


def test_a_stuck_container_is_named_and_boot_continues(monkeypatch, caplog):
    """A container the daemon will not let go of must not hold boot forever: log
    which one it is and carry on, so the launch fails loudly on its own instead."""
    monkeypatch.setattr(api_routes, "_remove_engine_containers", lambda include_primary: ["id1"])
    monkeypatch.setattr(api_routes, "_engine_container_ids", lambda include_primary: ["id1"])
    monkeypatch.setattr(api_routes, "_engine_container_names",
                        lambda include_primary: ["ainode-vllm-node-solo-8001"])
    monkeypatch.setattr(api_routes, "_ORPHAN_CLEAR_TIMEOUT_S", 0.0)
    monkeypatch.setattr(api_routes.time, "sleep", lambda _s: None)

    with caplog.at_level(logging.WARNING, logger=api_routes.logger.name):
        api_routes.sweep_engines_before_boot()

    assert "ainode-vllm-node-solo-8001" in caplog.text
    assert "launching anyway" in caplog.text


def test_the_sweep_is_claimed_once_per_process(monkeypatch):
    """The window for a wide sweep closes as soon as boot has swept: a second one
    later in the process would remove the engine this boot just launched."""
    calls = []
    monkeypatch.setattr(api_routes, "_remove_engine_containers",
                        lambda include_primary: calls.append(include_primary) or [])
    api_routes.sweep_engines_before_boot()
    api_routes.sweep_engines_before_boot()
    asyncio.run(api_routes.ensure_startup_sweep())
    assert calls == [True], "one sweep, and the replay's own is then a no-op"


@pytest.fixture
def boot(monkeypatch):
    """``ainode start`` with no disk, no GPU, no docker and no web server."""
    events = []
    cfg = _Cfg(node_id="n1", node_name="spark-1", onboarded=True, model="a/one",
               api_port=8000, web_port=3000, host="0.0.0.0", distributed_mode="solo",
               engine_backend="nvidia", engine_strategy="docker", cluster_interface="")
    monkeypatch.setattr(cli.NodeConfig, "load", classmethod(lambda cls: cfg))
    monkeypatch.setattr(cli, "ensure_dirs", lambda: None)
    monkeypatch.setattr(cli, "_write_pid", lambda: None)
    monkeypatch.setattr(cli, "_remove_pid", lambda: None)
    monkeypatch.setattr(cli, "_fabric_summary", lambda config: "eth0")
    monkeypatch.setattr("ainode.core.gpu.detect_gpu", lambda: None)
    monkeypatch.setattr("ainode.models.api_routes.consume_start_clean", lambda: False)
    monkeypatch.delenv("AINODE_IN_CONTAINER", raising=False)

    engine = SimpleNamespace(start=lambda: events.append("launch boot primary") or True,
                             stop=lambda: None)
    monkeypatch.setattr("ainode.engine.backends.get_backend",
                        lambda config, on_ready=None, instance_id="": engine)
    monkeypatch.setattr("ainode.api.server.run_server",
                        lambda config=None, engine=None: events.append("server up"))
    return events


def test_ainode_start_sweeps_before_it_launches_the_boot_primary(boot, monkeypatch):
    """The #96 ordering bug: the sweep used to run inside the replay task, 14 s
    after the new primary had already started, so the primary profiled next to the
    previous life's engine. It has to finish first."""
    events = boot
    monkeypatch.setattr(api_routes, "_remove_engine_containers",
                        lambda include_primary: events.append("sweep rm") or ["id1"])
    cleared = iter([["id1"], []])
    monkeypatch.setattr(api_routes, "_engine_container_ids",
                        lambda include_primary: next(cleared))
    monkeypatch.setattr(api_routes.time, "sleep", lambda _s: None)

    cli.cmd_start(SimpleNamespace(in_container=False))

    assert events == ["sweep rm", "launch boot primary", "server up"]


# ---------------------------------------------------------------------------
# 2. Serialized launches
# ---------------------------------------------------------------------------

def _fake_replay(monkeypatch, entries, *, primary_binds=True, launch_ok=None):
    """Wire the replay to fakes and return the ordered event list."""
    events = []

    class _Backend:
        def __init__(self, model):
            self.model = model

        def start(self):
            return True

    class _Inst:
        def __init__(self, model):
            self.backend = _Backend(model)

    class _Manager:
        def __init__(self):
            self.by = {}

        def instances(self):
            return []

        def by_model(self, m):
            return self.by.get(m)

    manager = _Manager()

    def _append(app, model, gmu, overrides=None, persist=True):
        events.append(("launch", model))
        ok = True if launch_ok is None else launch_ok(model)
        if not ok:
            return {"ok": False, "error": "engine image unavailable", "status": 500}
        manager.by[model] = _Inst(model)
        return {"ok": True, "model": model, "api_port": 8000 + len(manager.by)}

    async def _ensure(app, port, relaunch, label, timeout=300.0, backend=None):
        events.append(("bind", label))
        # Every launch in the serialized run happens with the slot held.
        assert api_routes.launch_owner() == "startup replay"
        return primary_binds if label.startswith("boot primary") else True

    async def _sweep():
        events.append("sweep")

    async def _no_sleep(*_a, **_k):
        return None

    async def _adopt(app):
        return []

    async def _replay_distributed(app):
        return {"action": "none"}

    monkeypatch.setattr(api_routes, "load_instance_manifest", lambda: entries)
    monkeypatch.setattr(api_routes, "append_solo_instance", _append)
    monkeypatch.setattr(api_routes, "_ensure_serving", _ensure)
    monkeypatch.setattr(api_routes, "ensure_startup_sweep", _sweep)
    # The reconcile step the replay now opens with (#179) asks docker what this
    # node is already running. Faked here for the same reason the sweep is: this
    # file's subject is the ORDER, and the suite runs on a machine with a real
    # docker and real engine containers.
    monkeypatch.setattr(api_routes, "adopt_running_engines", _adopt)
    monkeypatch.setattr(api_routes, "replay_distributed_if_needed", _replay_distributed)
    monkeypatch.setattr(api_routes.asyncio, "sleep", _no_sleep)
    return events, manager


@pytest.mark.asyncio
async def test_replay_sweeps_then_binds_the_primary_then_stacks_one_at_a_time(monkeypatch):
    """The whole ordering invariant in one assertion: sweep, primary, bind, then
    each stacked instance in turn."""
    entries = [{"model": "a/one"}, {"model": "b/two"}, {"model": "c/three"}]
    events, manager = _fake_replay(monkeypatch, entries)

    await api_routes.replay_instances_on_startup(
        {"config": _Cfg(model="a/one"), "engine": SimpleNamespace(start=lambda: True),
         "instances": manager})

    assert events == [
        "sweep",
        ("bind", "boot primary a/one"),
        ("launch", "b/two"), ("bind", "replay b/two"),
        ("launch", "c/three"), ("bind", "replay c/three"),
    ]


@pytest.mark.asyncio
async def test_the_sweep_runs_even_with_nothing_to_replay(monkeypatch):
    """An engine from a previous life must be freed whether or not this node has
    stacked models to bring back."""
    events, manager = _fake_replay(monkeypatch, [])
    await api_routes.replay_instances_on_startup(
        {"config": _Cfg(model="a/one"), "engine": None, "instances": manager})
    assert events == ["sweep"]


@pytest.mark.asyncio
async def test_a_primary_that_never_binds_does_not_block_the_stack(monkeypatch):
    """The primary gets its one retry inside _ensure_serving. If it still fails,
    the stacked models are still worth launching — the node coming back with one
    of three models is better than none."""
    entries = [{"model": "a/one"}, {"model": "b/two"}, {"model": "c/three"}]
    events, manager = _fake_replay(monkeypatch, entries, primary_binds=False)

    await api_routes.replay_instances_on_startup(
        {"config": _Cfg(model="a/one"), "engine": SimpleNamespace(start=lambda: True),
         "instances": manager})

    assert ("launch", "b/two") in events and ("launch", "c/three") in events


@pytest.mark.asyncio
async def test_a_stacked_launch_that_fails_logs_and_the_next_one_still_runs(
        monkeypatch, caplog):
    """A launch that cannot start at all (bad image, refused admission) must not
    swallow the rest of the manifest."""
    entries = [{"model": "a/one"}, {"model": "b/two"}, {"model": "c/three"}]
    events, manager = _fake_replay(monkeypatch, entries,
                                   launch_ok=lambda m: m != "b/two")

    with caplog.at_level(logging.ERROR, logger=api_routes.logger.name):
        await api_routes.replay_instances_on_startup(
            {"config": _Cfg(model="a/one"), "engine": SimpleNamespace(start=lambda: True),
             "instances": manager})

    assert events == [
        "sweep",
        ("bind", "boot primary a/one"),
        ("launch", "b/two"),                       # failed: no bind wait for it
        ("launch", "c/three"), ("bind", "replay c/three"),
    ]
    assert "replay load failed for b/two" in caplog.text
    assert "engine image unavailable" in caplog.text


@pytest.mark.asyncio
async def test_the_replay_queues_for_the_slot_rather_than_giving_up(monkeypatch):
    """Boot has nobody to report a refusal to, so the replay waits for the slot
    instead of returning early like an HTTP caller does."""
    entries = [{"model": "a/one"}, {"model": "b/two"}]
    events, manager = _fake_replay(monkeypatch, entries)
    monkeypatch.setattr(api_routes, "_LAUNCH_QUEUE_SECONDS", 0)

    await api_routes.acquire_launch_slot("load somebody-else")
    task = asyncio.ensure_future(api_routes.replay_instances_on_startup(
        {"config": _Cfg(model="a/one"), "engine": SimpleNamespace(start=lambda: True),
         "instances": manager}))
    await _settle(lambda: events == ["sweep"])
    assert events == ["sweep"], "queued behind the other launch, not refused"

    api_routes.release_launch_slot()
    await task
    assert ("launch", "b/two") in events


# ---------------------------------------------------------------------------
# 3. The launch slot, from the HTTP side
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_queued_launch_is_refused_with_the_owner_named():
    await api_routes.acquire_launch_slot("load a/one")
    assert api_routes.launch_owner() == "load a/one"
    with pytest.raises(api_routes.LaunchBusy) as exc:
        await api_routes.acquire_launch_slot("load b/two", wait=0)
    assert exc.value.owner == "load a/one"
    api_routes.release_launch_slot()
    assert api_routes.launch_owner() is None
    await api_routes.acquire_launch_slot("load b/two", wait=0)
    assert api_routes.launch_owner() == "load b/two"


def _loadable_app(monkeypatch, bind_gate):
    """An app whose loads launch instantly and then sit in the bind wait."""
    launched = []

    def _append(app, model, gmu, overrides=None, persist=True):
        launched.append(model)
        return {"ok": True, "model": model, "instance_id": f"n1:{model}",
                "api_port": 8000 + len(launched) - 1, "stacked": len(launched) > 1}

    async def _wait(app, port, backend, timeout=300.0):
        await bind_gate.wait()
        return True, "bound", 42.0

    monkeypatch.setattr(api_routes, "append_solo_instance", _append)
    monkeypatch.setattr(api_routes, "_wait_for_bind", _wait)
    return {"config": _Cfg(), "cluster_state": None, "ray_autostart_state": None,
            "instances": None}, launched


@pytest.mark.asyncio
async def test_two_concurrent_loads_serialize_and_the_second_is_refused(monkeypatch):
    """The other half of #96: two UI loads 2 s apart are the same race the replay
    was losing. The second one is told who is launching instead of profiling
    against half-reserved memory."""
    gate = asyncio.Event()
    app, launched = _loadable_app(monkeypatch, gate)
    monkeypatch.setattr(api_routes, "_LAUNCH_QUEUE_SECONDS", 0)

    first = await api_routes.handle_model_load(_Req(app, {"model": "a/one"}))
    assert first.status == 200
    await _settle(lambda: api_routes.launch_owner() == "load a/one")

    second = await api_routes.handle_model_load(
        _Req(app, {"model": "b/two", "gpu_memory_utilization": 0.3}))
    assert second.status == 409
    body = json.loads(second.text)["error"]
    assert "load a/one" in body, "the 409 names what is holding the slot"
    assert "KV cache" in body, "and why we refuse rather than launch"
    assert launched == ["a/one"], "the refused load never touched docker"

    # The slot is released when the first engine binds, and the next load lands.
    gate.set()
    await _settle(lambda: api_routes.launch_owner() is None)
    third = await api_routes.handle_model_load(
        _Req(app, {"model": "b/two", "gpu_memory_utilization": 0.3}))
    assert third.status == 200
    assert launched == ["a/one", "b/two"]


@pytest.mark.asyncio
async def test_a_load_that_queues_briefly_still_gets_through(monkeypatch):
    """Refusing is the fallback, not the first move: a load that arrives while a
    launch is finishing waits a few seconds and then proceeds."""
    gate = asyncio.Event()
    app, launched = _loadable_app(monkeypatch, gate)

    first = await api_routes.handle_model_load(_Req(app, {"model": "a/one"}))
    assert first.status == 200
    await _settle(lambda: api_routes.launch_owner() == "load a/one")

    queued = asyncio.ensure_future(api_routes.handle_model_load(
        _Req(app, {"model": "b/two", "gpu_memory_utilization": 0.3})))
    await _settle(lambda: launched == ["a/one"])
    assert not queued.done(), "waiting for the slot, not refused"

    gate.set()                      # the first engine binds, the slot frees
    assert (await queued).status == 200
    assert launched == ["a/one", "b/two"]


@pytest.mark.asyncio
async def test_a_failed_load_hands_the_slot_straight_back(monkeypatch):
    """A launch that never started must not leave the node refusing every other
    load for the length of a bind wait."""
    def _refuse(app, model, gmu, overrides=None, persist=True):
        return {"ok": False, "error": "no gpu_memory_utilization given", "status": 400}

    monkeypatch.setattr(api_routes, "append_solo_instance", _refuse)
    app = {"config": _Cfg(), "cluster_state": None, "ray_autostart_state": None,
           "instances": None}

    resp = await api_routes.handle_model_load(_Req(app, {"model": "b/two"}))
    assert resp.status == 400
    assert api_routes.launch_owner() is None


# ---------------------------------------------------------------------------
# 4. The bind message says how long the CONTAINER lived
# ---------------------------------------------------------------------------

class _Engine:
    """A backend handle with a launch stamp, as the real ones now publish."""

    def __init__(self, alive_for, *, exited=False, chatty=True):
        self.launched_at = time.time() - alive_for
        self.process = SimpleNamespace(poll=lambda: 1 if exited else None)
        self._activity = time.time()
        self.chatty = chatty
        self.starts = 0

    @property
    def last_log_activity(self):
        return self._activity

    def tick(self):
        if self.chatty:
            self._activity += 1.0

    def start(self):
        self.starts += 1
        return False        # no second wait to reason about in these tests


def _never_serves(monkeypatch, engine):
    async def _probe(port):
        engine.tick()
        return False
    monkeypatch.setattr(api_routes, "_port_serving", _probe)


@pytest.mark.asyncio
async def test_an_exited_container_reports_the_time_it_was_alive(monkeypatch, caplog):
    """The wording bug itself. The primary lived 47 s and the wait, which only
    started once it had died, logged 'after 0s'."""
    engine = _Engine(alive_for=47, exited=True)
    _never_serves(monkeypatch, engine)

    with caplog.at_level(logging.WARNING, logger=api_routes.logger.name):
        await api_routes._ensure_serving(
            {"config": _Cfg()}, 8000, engine.start, "boot primary a/one",
            backend=engine)

    assert "never bound on :8000 after 47s (container exited)" in caplog.text


@pytest.mark.asyncio
async def test_a_silent_engine_says_silent_and_still_reports_the_life(
        monkeypatch, caplog):
    engine = _Engine(alive_for=205, chatty=False)
    _never_serves(monkeypatch, engine)

    with caplog.at_level(logging.WARNING, logger=api_routes.logger.name):
        await api_routes._ensure_serving(
            {"config": _Cfg(engine_bind_log_silence_seconds=0.05,
                            engine_bind_ceiling_seconds=30)},
            8001, engine.start, "replay b/two", backend=engine)

    assert "never bound on :8001 after 205s (no engine activity or log for" in caplog.text


@pytest.mark.asyncio
async def test_the_ceiling_says_ceiling_and_still_reports_the_life(monkeypatch, caplog):
    engine = _Engine(alive_for=1802, chatty=True)
    _never_serves(monkeypatch, engine)

    with caplog.at_level(logging.WARNING, logger=api_routes.logger.name):
        await api_routes._ensure_serving(
            {"config": _Cfg(engine_bind_log_silence_seconds=30,
                            engine_bind_ceiling_seconds=0.05)},
            8000, engine.start, "replay chatty", backend=engine)

    assert "never bound on :8000 after 1802s (ceiling of 0s reached)" in caplog.text


@pytest.mark.asyncio
async def test_a_backend_with_no_launch_stamp_times_the_wait_as_before(monkeypatch):
    """Unchanged fallback: a handle that publishes no stamp is measured from the
    start of the wait, which is all we know about it."""
    engine = _Engine(alive_for=300, exited=True)
    del engine.launched_at
    _never_serves(monkeypatch, engine)

    _bound, reason, elapsed = await api_routes._wait_for_bind(
        {"config": _Cfg()}, 8000, engine)

    assert reason == "container exited"
    assert elapsed < 5, "the wait's own age, not a 300s container life"


def test_the_launch_stamp_reader_ignores_junk():
    """Same contract as _engine_log_mark: only a real number counts, so a fake or
    a stale attribute cannot move the clock."""
    assert api_routes._engine_launch_mark(None) is None
    assert api_routes._engine_launch_mark(object()) is None
    assert api_routes._engine_launch_mark(SimpleNamespace(launched_at=True)) is None
    assert api_routes._engine_launch_mark(SimpleNamespace(launched_at="now")) is None
    assert api_routes._engine_launch_mark(SimpleNamespace(launched_at=5.0)) == 5.0


def test_both_docker_backends_publish_a_launch_stamp():
    """The stamp is what the bind wait measures, so a backend that launches
    containers has to set it — and clear it on stop."""
    from ainode.core.config import NodeConfig
    from ainode.engine.backends.eugr import EugrBackend
    from ainode.engine.backends.nvidia import NvidiaBackend

    cfg = NodeConfig()
    for backend in (NvidiaBackend(cfg), EugrBackend(cfg)):
        assert backend.launched_at is None, "nothing launched yet"
        backend._launched_at = 123.0
        assert backend.launched_at == 123.0
