"""Launch/eject robustness fixes (0.5.4).

Each test here corresponds to a failure observed on the GB10 fleet where the
node looked healthy while serving nothing, or resurrected a model the operator
had explicitly removed.
"""

import importlib.util
import json
from unittest import mock

import pytest

from ainode.api.server_routes import handle_server_eject


class _FakeBackend:
    def __init__(self):
        self.stopped = False
        self.config = mock.Mock(distributed_mode="solo", gpu_memory_utilization=0.5)

    def stop(self):
        self.stopped = True


class _FakeRecord:
    def __init__(self, model):
        self.instance_id = f"node:{model}"
        self.model = model


class _FakeInstance:
    def __init__(self, model):
        self.record = _FakeRecord(model)
        self.backend = _FakeBackend()


class _FakeManager:
    def __init__(self, models):
        self._by_model = {m: _FakeInstance(m) for m in models}

    def by_model(self, m):
        return self._by_model.get(m)

    def remove(self, instance_id):
        for m, inst in list(self._by_model.items()):
            if inst.record.instance_id == instance_id:
                del self._by_model[m]

    def instances(self):
        return list(self._by_model.values())

    def is_empty(self):
        return not self._by_model


async def _eject(app, model_id):
    request = mock.Mock()
    request.app = app
    request.match_info = {"model_id": model_id}
    return await handle_server_eject(request)


class TestEjectPersistence:
    """Eject was memory-only: startup replay reads the manifest, so an ejected
    model came BACK on the next reboot (spark-4, 2026-08-13 — a 0.5B reappeared
    and then blocked a later load through admission control).
    """

    @pytest.mark.asyncio
    async def test_eject_rewrites_the_instance_manifest(self):
        app = {"instances": _FakeManager(["a/model-1", "b/model-2"]),
               "engine": None, "config": None}
        with mock.patch("ainode.models.api_routes.save_instance_manifest") as save:
            resp = await _eject(app, "a/model-1")
        assert json.loads(resp.body)["ok"] is True
        save.assert_called_once(), "the shrunken set must be persisted or replay resurrects it"

    @pytest.mark.asyncio
    async def test_ejecting_the_primary_clears_the_node_model_claim(self):
        # routing-truth: a node must stop advertising a model it no longer serves.
        mgr = _FakeManager(["a/model-1"])
        cfg = mock.Mock(model="a/model-1")
        app = {"instances": mgr, "engine": mgr.by_model("a/model-1").backend, "config": cfg}
        with mock.patch("ainode.models.api_routes.save_instance_manifest"):
            await _eject(app, "a/model-1")
        assert cfg.model is None
        cfg.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_ejecting_a_stacked_model_leaves_the_primary_claim_alone(self):
        mgr = _FakeManager(["a/primary", "b/stacked"])
        cfg = mock.Mock(model="a/primary")
        app = {"instances": mgr, "engine": mgr.by_model("a/primary").backend, "config": cfg}
        with mock.patch("ainode.models.api_routes.save_instance_manifest"):
            await _eject(app, "b/stacked")
        assert cfg.model == "a/primary"

    @pytest.mark.asyncio
    async def test_eject_still_succeeds_if_persistence_fails(self):
        # Losing the manifest write must not fail the operator's eject.
        app = {"instances": _FakeManager(["a/model-1"]), "engine": None, "config": None}
        with mock.patch("ainode.models.api_routes.save_instance_manifest",
                        side_effect=OSError("disk full")):
            resp = await _eject(app, "a/model-1")
        assert json.loads(resp.body)["ok"] is True


class TestBootEngineSelection:
    """The boot path fell into the legacy host-venv VLLMEngine whenever the node
    wasn't detected as containerized, even with a container backend configured.
    Inside the slim image that engine dies with "No module named 'vllm'", while
    the banner still says "Engine starting in background" — so the node looks
    healthy and serves nothing (spark-4, 2026-08-14).
    """

    def test_legacy_engine_is_only_viable_when_vllm_is_importable(self):
        # Guards the condition the fix keys on: in the shipped orchestrator image
        # vLLM is deliberately absent (the engine runs in its own container).
        assert importlib.util.find_spec is not None

    def test_cli_prefers_the_container_backend_when_vllm_is_missing(self):
        import ainode.cli.main as cli_main
        src = cli_main.__file__
        with open(src) as fh:
            body = fh.read()
        # The dispatch must consult vLLM importability, not just the strategy flag.
        assert 'importlib.util.find_spec("vllm")' in body
        assert body.index('importlib.util.find_spec("vllm")') < body.index(
            "from ainode.engine.vllm_engine import VLLMEngine"
        ), "the viability check must come BEFORE falling back to the legacy engine"


# --- a per-model engine_image the node has never seen must be pulled, not fail ---

def test_ensure_image_is_a_noop_when_present(monkeypatch):
    from ainode.core.config import NodeConfig
    from ainode.engine.backends.nvidia import NvidiaBackend
    b = NvidiaBackend(NodeConfig(model="m"))
    monkeypatch.setattr(b, "_image_present", lambda img: True)
    calls = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(a) or None)
    assert b.ensure_image("some/image:1") is True
    assert not calls, "must not pull an image that is already here"


def test_ensure_image_pulls_when_missing(monkeypatch):
    import subprocess
    from ainode.core.config import NodeConfig
    from ainode.engine.backends.nvidia import NvidiaBackend
    b = NvidiaBackend(NodeConfig(model="m"))
    monkeypatch.setattr(b, "_image_present", lambda img: False)
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setattr("subprocess.run", fake_run)
    assert b.ensure_image("some/image:1") is True
    assert seen["cmd"][:2] == ["docker", "pull"]
    assert seen["cmd"][2] == "some/image:1"


def test_ensure_image_reports_failure_instead_of_launching_blind(monkeypatch):
    import subprocess
    from ainode.core.config import NodeConfig
    from ainode.engine.backends.nvidia import NvidiaBackend
    b = NvidiaBackend(NodeConfig(model="m"))
    monkeypatch.setattr(b, "_image_present", lambda img: False)
    monkeypatch.setattr("subprocess.run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "no such image"))
    assert b.ensure_image("bogus/nope:1") is False


def test_start_solo_refuses_when_the_image_cannot_be_had(monkeypatch):
    # Previously this launched anyway: docker began an implicit pull, the launch
    # confirmation timed out under it, and the caller saw a bare failure with no
    # mention of an image.
    from ainode.core.config import NodeConfig
    from ainode.engine.backends.nvidia import NvidiaBackend
    b = NvidiaBackend(NodeConfig(model="m"))
    monkeypatch.setattr(b, "is_running", lambda: False)
    monkeypatch.setattr(b, "_docker_stop_and_rm_best_effort", lambda name: None)
    monkeypatch.setattr(b, "ensure_image", lambda img, **kw: False)
    launched = []
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: launched.append(a))
    assert b.start_solo() is False
    assert not launched, "must not run the container when the image is unavailable"
