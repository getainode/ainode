"""Stacked replay: the reason survives, and the stack waits for the primary (#235).

Seen on the 0.5.28 roll (2026-09-19, Spark-4), which serves Nemotron as its
primary with Qwen3-Embedding stacked on :8001. After the restart the log said:

    replay Qwen/Qwen3-Embedding-0.6B never bound on :8001 after 79s
    (container exited); relaunching once

The 0.5.5 retry brought it back ten minutes later, so nothing was lost, but two
things were wrong. The first attempt died while the primary was still in
loading_weights/profiling, and WHY it died went with the container: the relaunch
stop/rm's the old one by name, and that corpse held the only record.

So: the engine's last lines are copied into the ainode log before the relaunch,
the stacked instances wait for the primary to BIND rather than for a clock, and
every bind window scales with how many engines are coming up at once.
"""

from __future__ import annotations

import logging

import pytest

from ainode.models import api_routes as mr


# ---------------------------------------------------------------- the scaling

def test_one_engine_changes_nothing():
    assert mr.bind_window_scale(1) == 1.0


def test_every_extra_engine_buys_more_window():
    assert mr.bind_window_scale(2) == 1.5
    assert mr.bind_window_scale(3) == 2.0


def test_the_scale_is_capped():
    assert mr.bind_window_scale(20) == mr.BIND_WINDOW_MAX_SCALE


@pytest.mark.parametrize("bad", [None, 0, -4, "two", object()])
def test_anything_unusable_reads_as_one_engine(bad):
    assert mr.bind_window_scale(bad) == 1.0


def test_the_configured_limits_are_stretched_by_it():
    from ainode.core.config import NodeConfig

    app = {"config": NodeConfig()}
    assert mr._bind_limits(app) == (300.0, 3600.0)
    assert mr._bind_limits(app, 1) == (300.0, 3600.0)
    assert mr._bind_limits(app, 3) == (600.0, 7200.0)


# --------------------------------------------------------- reading the corpse

class _Backend:
    """A fake engine handle with exactly the surface the bind wait reads."""

    def __init__(self, *, binds_after=0, alive=True, tail="", chatty=True):
        self.binds_after = binds_after
        self.tail = tail
        self.chatty = chatty
        self.polls = 0
        self.starts = 0
        self.tail_calls = []
        self._alive = alive
        self._stamp = 1_000.0
        self.process = _Proc(None if alive else 1)

    # -- the bind wait's signals
    @property
    def last_log_activity(self):
        return self._stamp

    def activity_mark(self):
        return self._stamp

    def log_tail(self, lines=40):
        self.tail_calls.append(lines)
        return self.tail

    # -- the fake engine itself
    def tick(self):
        self.polls += 1
        if self.chatty:
            self._stamp += 1.0

    def serving(self):
        return self._alive and self.polls >= self.binds_after

    def start(self):
        self.starts += 1
        self._alive = True
        self.process = _Proc(None)
        self.polls = 0
        return True


class _Proc:
    def __init__(self, rc=None):
        self._rc = rc

    def poll(self):
        return self._rc


@pytest.fixture(autouse=True)
def _fast(monkeypatch, tmp_path):
    """Milliseconds instead of minutes, and the ledger in a tmpdir."""
    monkeypatch.setattr(mr, "_BIND_POLL_SECONDS", 0.005)
    monkeypatch.setattr(mr, "_GPU_RELEASE_SECONDS", 0.0)
    monkeypatch.setattr(mr, "_launch_times_path", lambda: tmp_path / "launch-times.json")


def _app(silence=30.0, ceiling=30.0):
    class _Cfg:
        engine_bind_log_silence_seconds = silence
        engine_bind_ceiling_seconds = ceiling
    return {"config": _Cfg()}


def _wire_ports(monkeypatch, by_port):
    """Point the port probe at a {port: backend} map; each probe ticks them all."""
    async def _probe(port):
        for backend in by_port.values():
            backend.tick()
        target = by_port.get(port)
        return bool(target and target.serving())

    monkeypatch.setattr(mr, "_port_serving", _probe)


VLLM_TAIL = (
    "INFO 09-19 22:31:07 [core.py:193] init engine (profile, create kv cache)\n"
    "ERROR 09-19 22:32:26 [core.py:588] EngineCore failed to start.\n"
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.10 GiB\n"
)


@pytest.mark.asyncio
async def test_a_dead_engines_last_lines_reach_the_ainode_log(monkeypatch, caplog):
    """The reason used to die with the container. Now it is above the verdict."""
    engine = _Backend(binds_after=2, alive=False, tail=VLLM_TAIL)
    _wire_ports(monkeypatch, {8001: engine})

    with caplog.at_level(logging.WARNING, logger=mr.logger.name):
        ok = await mr._ensure_serving(_app(), 8001, engine.start,
                                      "replay Qwen/Qwen3-Embedding-0.6B",
                                      backend=engine)

    assert ok is True, "the relaunch bound, so the node comes back serving it"
    assert engine.starts == 1
    assert "container exited" in caplog.text
    assert "CUDA out of memory" in caplog.text
    assert "before the container is removed" in caplog.text


@pytest.mark.asyncio
async def test_the_tail_is_read_before_the_relaunch_removes_the_container(monkeypatch):
    """Ordering is the whole point: a launch stop/rm's the leftover by name, so a
    read after the relaunch reads nothing."""
    order = []
    engine = _Backend(binds_after=1, alive=False, tail=VLLM_TAIL)

    def _tail(lines=40):
        order.append("tail")
        return VLLM_TAIL

    def _start():
        order.append("relaunch")
        engine._alive = True
        engine.process = _Proc(None)
        engine.polls = 0
        return True

    engine.log_tail = _tail
    _wire_ports(monkeypatch, {8001: engine})

    await mr._ensure_serving(_app(), 8001, _start, "replay x", backend=engine)
    assert order[:2] == ["tail", "relaunch"]


@pytest.mark.asyncio
async def test_forty_lines_is_what_is_asked_for(monkeypatch):
    engine = _Backend(binds_after=1, alive=False, tail=VLLM_TAIL)
    _wire_ports(monkeypatch, {8001: engine})
    await mr._ensure_serving(_app(), 8001, engine.start, "replay x", backend=engine)
    assert engine.tail_calls == [mr.REPLAY_LOG_TAIL_LINES] == [40]


@pytest.mark.asyncio
async def test_a_backend_with_nothing_to_say_logs_no_empty_block(monkeypatch, caplog):
    engine = _Backend(binds_after=1, alive=False, tail="")
    _wire_ports(monkeypatch, {8001: engine})
    with caplog.at_level(logging.WARNING, logger=mr.logger.name):
        await mr._ensure_serving(_app(), 8001, engine.start, "replay x", backend=engine)
    assert "before the container is removed" not in caplog.text


@pytest.mark.asyncio
async def test_a_log_tail_that_raises_is_not_a_failed_replay(monkeypatch):
    engine = _Backend(binds_after=1, alive=False)
    engine.log_tail = lambda lines=40: 1 / 0
    _wire_ports(monkeypatch, {8001: engine})
    assert await mr._ensure_serving(_app(), 8001, engine.start, "replay x",
                                    backend=engine) is True


@pytest.mark.asyncio
async def test_a_backend_with_no_log_tail_at_all_still_works(monkeypatch):
    """An older backend, or a handle that is not a backend at all."""
    class _Bare:
        process = _Proc(1)
        last_log_activity = 1_000.0
        starts = 0

        def activity_mark(self):
            return 1_000.0

        def start(self):
            _Bare.starts += 1
            self.process = _Proc(None)
            return True

    engine = _Bare()
    assert not hasattr(engine, "log_tail")
    assert await mr._engine_log_tail(engine) == ""

    async def _serving(port):
        return engine.process.poll() is None

    monkeypatch.setattr(mr, "_port_serving", _serving)
    assert await mr._ensure_serving(_app(), 8001, engine.start, "replay x",
                                    backend=engine) is True


# ------------------------------------------------- the stacked instances wait

def _fake_replay(monkeypatch, entries, engines, primary):
    """Wire a replay whose launches produce the given fake engines."""
    events = []

    class _Inst:
        def __init__(self, backend):
            self.backend = backend

    class _Manager:
        def __init__(self):
            self.by = {}

        def instances(self):
            return []

        def by_model(self, model):
            return self.by.get(model)

    manager = _Manager()

    def _append(app, model, gmu, overrides=None, persist=True):
        events.append(("launch", model))
        backend = engines[model]
        manager.by[model] = _Inst(backend)
        return {"ok": True, "model": model, "api_port": engines[model].port}

    async def _sweep():
        events.append("sweep")

    async def _no_sleep(*_a, **_k):
        return None

    async def _adopt(app):
        return []

    async def _no_distributed(app):
        return {"action": "none"}

    monkeypatch.setattr(mr, "load_instance_manifest", lambda: entries)
    monkeypatch.setattr(mr, "append_solo_instance", _append)
    monkeypatch.setattr(mr, "ensure_startup_sweep", _sweep)
    monkeypatch.setattr(mr, "adopt_running_engines", _adopt)
    monkeypatch.setattr(mr, "replay_distributed_if_needed", _no_distributed)
    monkeypatch.setattr(mr.asyncio, "sleep", _no_sleep)

    by_port = {primary.port: primary}
    by_port.update({e.port: e for e in engines.values()})

    async def _probe(port):
        for backend in by_port.values():
            backend.tick()
        target = by_port.get(port)
        if target is not None and target.serving() and \
                ("serving", port) not in events:
            events.append(("serving", port))
        return bool(target and target.serving())

    monkeypatch.setattr(mr, "_port_serving", _probe)
    return events, manager


class _PortBackend(_Backend):
    def __init__(self, port, **kw):
        super().__init__(**kw)
        self.port = port


class _Cfg:
    api_port = 8000
    model = "a/primary"
    gpu_memory_utilization = 0.6
    # Milliseconds, so a wedged fake engine is judged inside the test rather
    # than after the real five-minute silence budget.
    engine_bind_log_silence_seconds = 0.05
    engine_bind_ceiling_seconds = 0.5


@pytest.mark.asyncio
async def test_the_stacked_launch_waits_for_the_primary_to_bind(monkeypatch):
    """Not for its start: vLLM sizes its KV cache from what is free when it
    profiles, so an engine that launches while the primary is still reserving is
    the one that dies."""
    primary = _PortBackend(8000, binds_after=6)
    stacked = _PortBackend(8001, binds_after=1)
    events, manager = _fake_replay(monkeypatch, [{"model": "b/stacked"}],
                                   {"b/stacked": stacked}, primary)
    app = {"config": _Cfg(), "engine": primary, "instances": manager}

    await mr.replay_instances_on_startup(app)

    assert ("serving", 8000) in events
    assert events.index(("serving", 8000)) < events.index(("launch", "b/stacked"))
    assert primary.starts == 0, "a slow primary must not be relaunched"


@pytest.mark.asyncio
async def test_a_stacked_engine_that_exits_during_the_load_is_retried_with_its_reason(
        monkeypatch, caplog):
    """The incident, end to end: the primary is still coming up, the stacked
    engine's container exits, the reason is captured and the retry binds."""
    primary = _PortBackend(8000, binds_after=3)
    stacked = _PortBackend(8001, binds_after=2, alive=False, tail=VLLM_TAIL)
    events, manager = _fake_replay(monkeypatch, [{"model": "b/stacked"}],
                                   {"b/stacked": stacked}, primary)
    app = {"config": _Cfg(), "engine": primary, "instances": manager}

    with caplog.at_level(logging.WARNING, logger=mr.logger.name):
        await mr.replay_instances_on_startup(app)

    assert stacked.starts == 1, "exactly one relaunch, the 0.5.5 contract"
    assert "container exited" in caplog.text
    assert "CUDA out of memory" in caplog.text, "the reason is in the ainode log"


@pytest.mark.asyncio
async def test_a_primary_that_never_binds_says_so_and_the_stack_still_runs(
        monkeypatch, caplog):
    """A node has to come back serving whatever it can, but the risk is named: a
    stacked engine that then dies has the reason written above it."""
    primary = _PortBackend(8000, binds_after=10 ** 9, chatty=False)
    stacked = _PortBackend(8001, binds_after=1)
    events, manager = _fake_replay(monkeypatch, [{"model": "b/stacked"}],
                                   {"b/stacked": stacked}, primary)
    app = {"config": _Cfg(), "engine": primary, "instances": manager}

    with caplog.at_level(logging.WARNING, logger=mr.logger.name):
        await mr.replay_instances_on_startup({**app, "config": _Cfg()})

    assert ("launch", "b/stacked") in events
    assert "the primary is not serving" in caplog.text


@pytest.mark.asyncio
async def test_the_replay_scales_every_window_by_how_many_are_loading(monkeypatch):
    """Three manifest entries plus the primary is four engines coming up, and each
    wait is told so."""
    seen = []

    async def _ensure(app, port, relaunch, label, timeout=300.0, backend=None, loading=1):
        seen.append((label, loading))
        return True

    async def _no_sleep(*_a, **_k):
        return None

    async def _sweep():
        return None

    async def _adopt(app):
        return []

    async def _no_distributed(app):
        return {"action": "none"}

    entries = [{"model": "b/two"}, {"model": "c/three"}, {"model": "d/four"}]
    monkeypatch.setattr(mr, "load_instance_manifest", lambda: entries)
    monkeypatch.setattr(mr, "_ensure_serving", _ensure)
    monkeypatch.setattr(mr, "append_solo_instance",
                        lambda app, m, g, overrides=None, persist=True: {
                            "ok": True, "api_port": 8001})
    monkeypatch.setattr(mr, "ensure_startup_sweep", _sweep)
    monkeypatch.setattr(mr, "adopt_running_engines", _adopt)
    monkeypatch.setattr(mr, "replay_distributed_if_needed", _no_distributed)
    monkeypatch.setattr(mr.asyncio, "sleep", _no_sleep)

    class _Manager:
        def instances(self):
            return []

        def by_model(self, model):
            return None

    await mr.replay_instances_on_startup(
        {"config": _Cfg(), "engine": _PortBackend(8000), "instances": _Manager()})

    assert seen, "the replay has to have waited on something"
    assert all(loading == 4 for _label, loading in seen), seen


@pytest.mark.asyncio
async def test_with_no_engine_handle_the_primary_window_is_scaled(monkeypatch):
    """The flat 300 s is what expired while Spark-4's primary was still profiling."""
    seen = {}

    async def _wait(port, timeout=300.0):
        seen["timeout"] = timeout
        return True

    monkeypatch.setattr(mr, "_wait_port_ready", _wait)
    await mr._await_primary_bind({"config": _Cfg(), "engine": None}, _Cfg(), loading=3)
    assert seen["timeout"] == mr._PRIMARY_PORT_WAIT_SECONDS * 2.0


@pytest.mark.asyncio
async def test_with_no_primary_expected_the_window_is_not_stretched(monkeypatch):
    """Nothing is coming up on that port, so waiting longer for it is only boot
    time nobody gets back."""
    seen = {}

    async def _wait(port, timeout=300.0):
        seen["timeout"] = timeout
        return False

    class _NoModel(_Cfg):
        model = ""

    monkeypatch.setattr(mr, "_wait_port_ready", _wait)
    ok = await mr._await_primary_bind({"config": _NoModel(), "engine": None},
                                      _NoModel(), loading=4)
    assert ok is False
    assert seen["timeout"] == mr._PRIMARY_PORT_WAIT_SECONDS


# ------------------------------------------------- what the backends implement

def test_the_nvidia_backend_reads_its_own_container(monkeypatch):
    from ainode.core.config import NodeConfig
    from ainode.engine.backends.nvidia import NvidiaBackend

    backend = NvidiaBackend(NodeConfig(model="a/b", api_port=8000))
    # Nothing launched yet: there is no container to read, and asking docker
    # about a name that was never used is a subprocess for nothing.
    assert backend.log_tail() == ""

    seen = {}

    def _tail(name, lines=15):
        seen["name"] = name
        seen["lines"] = lines
        return "boom"

    backend._launched_at = 1.0
    monkeypatch.setattr(backend, "_docker_logs_tail", _tail)
    assert backend.log_tail(40) == "boom"
    assert seen["lines"] == 40
    assert "ainode-vllm" in seen["name"]


def test_the_eugr_backend_reads_its_own_log_file(tmp_path, monkeypatch):
    from ainode.core.config import NodeConfig
    from ainode.engine.backends.eugr import EugrBackend

    backend = EugrBackend(NodeConfig(model="a/b"))
    log = tmp_path / "vllm.log"
    log.write_text("\n".join(f"line {i}" for i in range(100)) + "\n")
    monkeypatch.setattr(backend, "_log_file", log)
    tail = backend.log_tail(5)
    assert tail.splitlines() == ["line 95", "line 96", "line 97", "line 98", "line 99"]


def test_every_backend_answers_the_log_tail_question():
    """Part of the base contract, so the replay never has to ask whether it can."""
    from ainode.engine.backends.base import EngineBackend

    assert EngineBackend.log_tail(object(), 40) == ""
