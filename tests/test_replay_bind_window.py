"""The startup replay waits on the engine, not on a fixed clock.

Measured 2026-09-13 on Spark-1 (GB10, ainode 0.5.7, vllm/vllm-openai:v0.27.1):
`ainode update` restarted the orchestrator, the replay relaunched the primary
(27B NVFP4) and the stacked instance (35B-A3B), and both engines were healthy
but slow to bind. On that image vLLM runs a FlashInfer fp4_gemm autotune pass
plus CUDA graph capture before it listens, so time-to-bind was about 12 minutes
and 6 minutes. The fixed 300s bind window expired first, logged "never bound",
and killed two healthy starts, doubling a 14-minute boot to 28.

The wait now treats an engine as alive while its container is up and its log is
still advancing, and relaunches only on evidence: container exited, log silent
past the budget, or the absolute ceiling. The 0.5.5 guarantee (an engine that
dies on the way up still gets exactly one relaunch) is unchanged.
"""

import inspect
import logging

import pytest

from ainode.models import api_routes


# The old fixed window, in the scaled-down units these tests run in. Every
# "slow" engine here binds well past it.
OLD_FIXED_WINDOW = 0.05


class _FakeProc:
    """Stand-in for the attached ``docker run`` subprocess.

    ``docker run`` runs in the foreground, so this handle exiting is how the
    replay learns the container died.
    """

    def __init__(self, returncode=None):
        self._rc = returncode

    def poll(self):
        return self._rc

    def exit(self, code=1):
        self._rc = code


class _FakeEngine:
    """A fake engine backend: a container that prints, a port that binds late.

    Exposes exactly the surface the bind wait reads: ``process`` (the attached
    launch handle) and ``last_log_activity`` (epoch of the last line this engine
    printed). ``tick`` stands for the passage of one bind poll: a chatty engine
    prints a progress line, a wedged one does not.
    """

    def __init__(self, binds_after=0, chatty=True, tracks_log=True, alive=True):
        self.process = _FakeProc(None if alive else 1)
        self.binds_after = binds_after
        self.chatty = chatty
        self.tracks_log = tracks_log
        self.polls = 0
        self.polls_before_relaunch = None
        self.starts = 0
        self._activity = 1_000.0

    @property
    def last_log_activity(self):
        return self._activity if self.tracks_log else None

    def tick(self):
        self.polls += 1
        if self.chatty:
            self._activity += 1.0

    def serving(self) -> bool:
        return self.polls >= self.binds_after and self.process.poll() is None

    def start(self) -> bool:
        """The relaunch: a fresh container, the poll count back to zero."""
        self.starts += 1
        self.polls_before_relaunch = self.polls
        self.process = _FakeProc()
        self.polls = 0
        return True


@pytest.fixture(autouse=True)
def _fast_bind_loop(monkeypatch):
    """Shrink the wait's cadence so a whole bind wait runs in milliseconds."""
    monkeypatch.setattr(api_routes, "_BIND_POLL_SECONDS", 0.005)
    monkeypatch.setattr(api_routes, "_GPU_RELEASE_SECONDS", 0.0)


def _wire_port(monkeypatch, engine):
    """Point the port probe at the fake engine; each probe advances its clock."""
    async def _probe(port):
        engine.tick()
        return engine.serving()
    monkeypatch.setattr(api_routes, "_port_serving", _probe)


def _app(silence, ceiling):
    class _Cfg:
        engine_bind_log_silence_seconds = silence
        engine_bind_ceiling_seconds = ceiling
    return {"config": _Cfg()}


@pytest.mark.asyncio
async def test_a_slow_but_progressing_engine_is_never_relaunched(monkeypatch):
    """The regression itself: an engine that binds long after the old fixed
    window, while still printing autotune/graph-capture lines, must be left
    alone."""
    engine = _FakeEngine(binds_after=40, chatty=True)
    _wire_port(monkeypatch, engine)

    ok = await api_routes._ensure_serving(
        _app(silence=0.05, ceiling=10.0), 8000, engine.start, "replay slow-27b",
        timeout=OLD_FIXED_WINDOW, backend=engine)

    assert ok is True
    assert engine.starts == 0, "a healthy slow start must not be killed"
    assert engine.polls >= 40, "the wait has to outlast the old fixed window"


@pytest.mark.asyncio
async def test_the_old_fixed_window_would_have_killed_that_engine(monkeypatch):
    """Control for the test above: same engine, no handle to watch, so the wait
    falls back to the fixed window and relaunches a start that was fine."""
    engine = _FakeEngine(binds_after=40, chatty=True)
    _wire_port(monkeypatch, engine)

    await api_routes._ensure_serving(
        _app(silence=0.05, ceiling=10.0), 8000, engine.start, "replay slow-27b",
        timeout=OLD_FIXED_WINDOW, backend=None)

    assert engine.starts == 1


@pytest.mark.asyncio
async def test_b_a_silent_engine_is_relaunched_once(monkeypatch, caplog):
    """Container up but the log has gone quiet past the budget: that is the
    wedged case the 0.5.5 relaunch exists for."""
    engine = _FakeEngine(binds_after=10**9, chatty=False)
    _wire_port(monkeypatch, engine)

    with caplog.at_level(logging.WARNING, logger=api_routes.logger.name):
        ok = await api_routes._ensure_serving(
            _app(silence=0.05, ceiling=30.0), 8001, engine.start, "replay wedged",
            timeout=300.0, backend=engine)

    assert ok is False
    assert engine.starts == 1, "exactly one relaunch, then give up"
    assert "log silent for" in caplog.text, "the log must name the cause"
    assert "never bound on :8001" in caplog.text


@pytest.mark.asyncio
async def test_c_an_exited_container_is_relaunched_once(monkeypatch, caplog):
    """The 0.5.5 behaviour, unchanged: an engine that dies on the way up gets one
    relaunch, and the relaunch is given the same adaptive wait."""
    engine = _FakeEngine(binds_after=3, chatty=True, alive=False)
    _wire_port(monkeypatch, engine)

    with caplog.at_level(logging.WARNING, logger=api_routes.logger.name):
        ok = await api_routes._ensure_serving(
            _app(silence=30.0, ceiling=30.0), 8000, engine.start, "boot primary x",
            timeout=300.0, backend=engine)

    assert ok is True, "the relaunched engine bound, so the node comes back serving"
    assert engine.starts == 1
    assert "container exited" in caplog.text


@pytest.mark.asyncio
async def test_c2_an_exited_container_is_not_waited_out(monkeypatch):
    """A dead container is detected on the first poll, not after the ceiling:
    the relaunch should start while the GPU is still warm, not 30 minutes on."""
    engine = _FakeEngine(binds_after=10**9, chatty=True, alive=False)
    _wire_port(monkeypatch, engine)

    await api_routes._ensure_serving(
        _app(silence=30.0, ceiling=0.05), 8000, engine.start, "boot primary x",
        timeout=300.0, backend=engine)

    assert engine.polls_before_relaunch <= 2, \
        "no reason to keep polling a container that exited"


@pytest.mark.asyncio
async def test_d_the_ceiling_stops_a_wedged_but_chatty_engine(monkeypatch, caplog):
    """An engine that keeps printing forever without ever binding cannot hold
    boot open: the absolute ceiling fires."""
    engine = _FakeEngine(binds_after=10**9, chatty=True)
    _wire_port(monkeypatch, engine)

    with caplog.at_level(logging.WARNING, logger=api_routes.logger.name):
        ok = await api_routes._ensure_serving(
            _app(silence=30.0, ceiling=0.05), 8000, engine.start, "replay chatty",
            timeout=300.0, backend=engine)

    assert ok is False
    assert engine.starts == 1
    assert "ceiling" in caplog.text


@pytest.mark.asyncio
async def test_an_engine_with_no_progress_signal_relies_on_the_ceiling(monkeypatch):
    """A backend that publishes no log stamp (the legacy host-venv engine) must
    not be relaunched for saying nothing. Silence is only judged on engines that
    actually report."""
    engine = _FakeEngine(binds_after=10**9, chatty=False, tracks_log=False)
    _wire_port(monkeypatch, engine)

    await api_routes._ensure_serving(
        _app(silence=0.005, ceiling=0.1), 8000, engine.start, "replay quiet-backend",
        timeout=300.0, backend=engine)

    # Silence would have fired within a couple of polls; the ceiling is what got
    # there, so the wait lasted far longer than the silence budget.
    assert engine.polls > 5


@pytest.mark.asyncio
async def test_a_handle_that_reports_nothing_falls_back_to_the_fixed_window(monkeypatch):
    """A handle with no launch process AND no log stamp gives the wait nothing to
    watch. Staying adaptive there would hold the port open for the whole ceiling
    on an engine we know nothing about, so that case takes the fixed window."""
    engine = _FakeEngine(binds_after=10**9, chatty=False, tracks_log=False)
    engine.process = None
    _wire_port(monkeypatch, engine)

    ok, reason, _ = await api_routes._wait_for_bind(
        _app(silence=30.0, ceiling=30.0), 8000, engine, timeout=0.02)

    assert ok is False
    assert "window expired" in reason


@pytest.mark.asyncio
async def test_a_bound_engine_logs_how_long_it_waited(monkeypatch, caplog):
    """Requirement 3: the cause and the duration belong in the ainode log, so
    the next person does not go reading dmesg."""
    engine = _FakeEngine(binds_after=2, chatty=True)
    _wire_port(monkeypatch, engine)

    with caplog.at_level(logging.INFO, logger=api_routes.logger.name):
        assert await api_routes._ensure_serving(
            _app(silence=30.0, ceiling=30.0), 8000, engine.start, "boot primary x",
            timeout=300.0, backend=engine) is True

    assert "bound on :8000 after" in caplog.text


@pytest.mark.asyncio
async def test_the_knobs_come_from_nodeconfig(monkeypatch):
    """The two limits are NodeConfig fields, with the module defaults as the
    fallback for a config written by an older release."""
    from ainode.core.config import NodeConfig

    cfg = NodeConfig()
    assert cfg.engine_bind_log_silence_seconds == 120
    assert cfg.engine_bind_ceiling_seconds == 1800
    assert api_routes._bind_limits({"config": cfg}) == (120.0, 1800.0)
    # No config, or a config from before these fields existed.
    assert api_routes._bind_limits({}) == (
        api_routes._DEFAULT_BIND_LOG_SILENCE_SECONDS,
        api_routes._DEFAULT_BIND_CEILING_SECONDS,
    )
    assert api_routes._bind_limits({"config": object()}) == (
        api_routes._DEFAULT_BIND_LOG_SILENCE_SECONDS,
        api_routes._DEFAULT_BIND_CEILING_SECONDS,
    )


@pytest.mark.asyncio
async def test_e_replay_still_loads_one_model_at_a_time_in_manifest_order(monkeypatch):
    """The serialization is the other half of the 0.5.5 contract: concurrent
    vLLM loads on a unified-memory node race for memory and one gets OOM-killed.
    Each model must bind before the next one launches, in manifest order, and
    each wait must get that instance's own engine handle."""
    events = []

    class _Backend:
        def __init__(self, model):
            self.model = model

        def start(self):
            return True

    class _Manager:
        def __init__(self):
            self.by = {}

        def instances(self):
            return []

        def by_model(self, m):
            return self.by.get(m)

    manager = _Manager()

    class _Inst:
        def __init__(self, model):
            self.backend = _Backend(model)

    def _fake_append(app, model, gmu, overrides=None, persist=True):
        events.append(("launch", model))
        manager.by[model] = _Inst(model)
        return {"ok": True, "model": model, "api_port": 8000 + len(events)}

    async def _fake_ensure(app, port, relaunch, label, timeout=300.0, backend=None):
        events.append(("bind", label, getattr(backend, "model", None)))
        return True

    async def _no_sleep(*_a, **_k):
        return None

    async def _bound(port, timeout=300.0):
        return True

    monkeypatch.setattr(api_routes, "load_instance_manifest",
                        lambda: [{"model": "a/one"}, {"model": "b/two"}, {"model": "c/three"}])
    monkeypatch.setattr(api_routes, "append_solo_instance", _fake_append)
    monkeypatch.setattr(api_routes, "_ensure_serving", _fake_ensure)
    monkeypatch.setattr(api_routes, "_wait_port_ready", _bound)
    monkeypatch.setattr(api_routes.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr("subprocess.run", lambda *a, **k: None)

    class _Cfg:
        api_port = 8000
        model = ""

    app = {"config": _Cfg(), "engine": None, "instances": manager}
    await api_routes.replay_instances_on_startup(app)

    assert events == [
        ("launch", "a/one"), ("bind", "replay a/one", "a/one"),
        ("launch", "b/two"), ("bind", "replay b/two", "b/two"),
        ("launch", "c/three"), ("bind", "replay c/three", "c/three"),
    ]


@pytest.mark.asyncio
async def test_the_boot_primary_wait_watches_the_boot_engine(monkeypatch):
    """The primary is the instance the incident relaunched first, so its wait
    has to get the boot engine's handle too."""
    seen = {}

    async def _fake_ensure(app, port, relaunch, label, timeout=300.0, backend=None):
        seen["label"] = label
        seen["backend"] = backend
        return True

    async def _no_sleep(*_a, **_k):
        return None

    boot = _FakeEngine()
    monkeypatch.setattr(api_routes, "load_instance_manifest", lambda: [{"model": "a/one"}])
    monkeypatch.setattr(api_routes, "_ensure_serving", _fake_ensure)
    monkeypatch.setattr(api_routes, "append_solo_instance",
                        lambda *a, **k: {"ok": True, "api_port": 8001})
    monkeypatch.setattr(api_routes.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr("subprocess.run", lambda *a, **k: None)

    class _Cfg:
        api_port = 8000
        model = "a/one"

    await api_routes.replay_instances_on_startup(
        {"config": _Cfg(), "engine": boot, "instances": None})

    assert seen["label"] == "boot primary a/one"
    assert seen["backend"] is boot


def test_the_docker_backends_publish_a_log_stamp():
    """The wait's proof of life has to come from the engine's own stream, not a
    log file: stacked instances share one log file, so file mtime would let a
    busy primary vouch for a wedged neighbour."""
    from ainode.core.config import NodeConfig
    from ainode.engine.backends.eugr import EugrBackend
    from ainode.engine.backends.nvidia import NvidiaBackend

    cfg = NodeConfig()
    for backend in (NvidiaBackend(cfg), EugrBackend(cfg)):
        assert backend.last_log_activity is None, "nothing printed yet"
        # _stream_logs stamps this per line; simulate one line landing.
        backend._last_log_activity = 123.0
        assert backend.last_log_activity == 123.0


def test_exit_evidence_needs_a_handle():
    """No handle is not evidence of death. An engine that outlived a previous
    orchestrator has no subprocess here, and must not be relaunched for it."""
    assert api_routes._engine_exited(None) is False
    assert api_routes._engine_exited(object()) is False          # no process attr
    assert api_routes._engine_exited(_FakeEngine(alive=True)) is False
    assert api_routes._engine_exited(_FakeEngine(alive=False)) is True


def test_a_backend_that_raises_on_poll_is_treated_as_alive():
    class _Bad:
        class process:
            @staticmethod
            def poll():
                raise OSError("no such process")

    assert api_routes._engine_exited(_Bad()) is False


def test_log_stamp_reader_ignores_junk():
    class _Junk:
        last_log_activity = "recently"

    class _Bool:
        last_log_activity = True

    assert api_routes._engine_log_mark(_Junk()) is None
    assert api_routes._engine_log_mark(_Bool()) is None
    assert api_routes._engine_log_mark(_FakeEngine()) == 1_000.0


def test_module_defaults_match_nodeconfig():
    """Two homes for the same number is a drift bug waiting to happen; assert
    they agree instead."""
    from ainode.core.config import NodeConfig

    cfg = NodeConfig()
    assert api_routes._DEFAULT_BIND_LOG_SILENCE_SECONDS == float(
        cfg.engine_bind_log_silence_seconds)
    assert api_routes._DEFAULT_BIND_CEILING_SECONDS == float(
        cfg.engine_bind_ceiling_seconds)


def test_the_bind_loop_stays_off_the_event_loop_thread():
    """Guard against someone reintroducing a blocking sleep or a blocking probe
    in the bind loop: this runs during server startup."""
    assert inspect.iscoroutinefunction(api_routes._wait_for_bind)
    assert inspect.iscoroutinefunction(api_routes._port_serving)


# ---------------------------------------------------------------------------
# Container evidence beats the launch subprocess (0.5.12 regression: solo
# ``docker run -d`` returns in a second, which read as "container exited")
# ---------------------------------------------------------------------------

def _nvidia_backend_with_state(monkeypatch, state, launched):
    from ainode.core.config import NodeConfig
    from ainode.engine.backends.nvidia import NvidiaBackend
    b = NvidiaBackend(NodeConfig(engine_backend="nvidia", distributed_mode="solo"))
    monkeypatch.setattr(b, "_docker_container_state", lambda name: state)
    b._launched_at = 1.0 if launched else None
    b._process = None
    return b


def test_running_container_is_not_exited_even_when_the_launch_client_returned(monkeypatch):
    from ainode.models.api_routes import _engine_exited
    b = _nvidia_backend_with_state(monkeypatch, "running", launched=True)
    b._process = _FakeProc(0)  # docker run -d already returned
    assert b.engine_exited() is False
    assert _engine_exited(b) is False
    assert b.is_running() is True


def test_exited_container_is_exited(monkeypatch):
    from ainode.models.api_routes import _engine_exited
    b = _nvidia_backend_with_state(monkeypatch, "exited", launched=True)
    assert b.engine_exited() is True
    assert _engine_exited(b) is True
    assert b.is_running() is False


def test_absent_container_counts_as_exited_only_after_a_launch(monkeypatch):
    b = _nvidia_backend_with_state(monkeypatch, "", launched=False)
    assert b.engine_exited() is False
    b._launched_at = 1.0
    assert b.engine_exited() is True


def test_backend_without_container_view_falls_back_to_the_subprocess():
    from ainode.models.api_routes import _engine_exited
    class NoView:
        process = _FakeProc(1)
    assert _engine_exited(NoView()) is True
    class Alive:
        process = _FakeProc(None)
    assert _engine_exited(Alive()) is False


def test_solo_launch_follows_its_container_logs_into_the_solo_log(monkeypatch, tmp_path):
    """The detached run's client prints one id and exits; the follower is what
    keeps last_log_activity moving so the bind wait does not misread silence."""
    from ainode.core.config import NodeConfig
    from ainode.engine.backends import nvidia as nv
    b = nv.NvidiaBackend(NodeConfig(engine_backend="nvidia", distributed_mode="solo", model="m/x"))
    b._log_file = tmp_path / "solo.log"
    calls = []
    class P:
        stdout = None
        def poll(self): return 0
    monkeypatch.setattr(nv.subprocess, "Popen", lambda cmd, **kw: calls.append(cmd) or P())
    monkeypatch.setattr(b, "_docker_stop_and_rm_best_effort", lambda name: None)
    monkeypatch.setattr(b, "ensure_image", lambda image: True)
    monkeypatch.setattr(b, "_image_entrypoint", lambda image: [])
    monkeypatch.setattr(b, "_confirm_container_started", lambda name, timeout=25.0: True)
    targets = []
    monkeypatch.setattr(b, "_stream_logs", lambda proc, target: targets.append(target))
    assert b.start_solo() is True
    assert [c[:2] for c in calls] == [["docker", "run"], ["docker", "logs"]]
    assert calls[1][2:4] == ["-f", "ainode-vllm-node-solo"]
    b._log_thread.join(timeout=2)
    assert tmp_path / "solo.log" in targets, "the follower must write to the SOLO log"


def test_solo_launch_does_not_follow_when_the_container_never_started(monkeypatch):
    from ainode.core.config import NodeConfig
    from ainode.engine.backends import nvidia as nv
    b = nv.NvidiaBackend(NodeConfig(engine_backend="nvidia", distributed_mode="solo", model="m/x"))
    calls = []
    class P:
        stdout = None
        def poll(self): return 0
    monkeypatch.setattr(nv.subprocess, "Popen", lambda cmd, **kw: calls.append(cmd) or P())
    monkeypatch.setattr(b, "_docker_stop_and_rm_best_effort", lambda name: None)
    monkeypatch.setattr(b, "ensure_image", lambda image: True)
    monkeypatch.setattr(b, "_image_entrypoint", lambda image: [])
    monkeypatch.setattr(b, "_confirm_container_started", lambda name, timeout=25.0: False)
    monkeypatch.setattr(b, "_stream_logs", lambda proc, target: None)
    assert b.start_solo() is False
    assert [c[:2] for c in calls] == [["docker", "run"]]
