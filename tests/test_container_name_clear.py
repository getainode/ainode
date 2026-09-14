"""A relaunch must not reuse a container name the daemon is still removing.

Engines run with ``--rm``: after ``docker stop`` the daemon deletes the
container asynchronously and ``docker rm -f`` returns first. On the 0.5.8 roll
every engine's first launch died with "Conflict. The container name ... is
already in use" and replay burned a second launch each (issue #80). Both the
backend's pre-launch cleanup and the replay's orphan sweep now wait for the
name to disappear.
"""
import subprocess
import types

import pytest

from ainode.engine.backends import nvidia as nvidia_mod
from ainode.models import api_routes


ID = "a848351b1569"


class _Ps:
    """Fake docker: ``check_output`` (the poll) reports an id N times; ``run`` records."""

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def run(self, cmd, **kw):
        self.calls.append(list(cmd))
        return types.SimpleNamespace(stdout="ctr_abc\n", stderr="", returncode=0)

    def check_output(self, cmd, **kw):
        self.calls.append(list(cmd))
        out = ID + "\n" if self.hits > 0 else ""
        self.hits -= 1
        return out


def _backend(monkeypatch):
    from ainode.core.config import NodeConfig
    b = nvidia_mod.NvidiaBackend(NodeConfig())
    monkeypatch.setattr(nvidia_mod.time, "sleep", lambda _s: None)
    return b


def test_cleanup_waits_until_the_name_is_gone(monkeypatch):
    fake = _Ps(hits=3)
    monkeypatch.setattr(nvidia_mod.subprocess, "run", fake.run)
    monkeypatch.setattr(nvidia_mod.subprocess, "check_output", fake.check_output)
    b = _backend(monkeypatch)
    assert b._wait_for_container_name_to_clear("ainode-vllm-node-solo") is True
    ps_calls = [c for c in fake.calls if c[:2] == ["docker", "ps"]]
    assert len(ps_calls) == 4, "three hits then one clear read"
    assert ps_calls[0][-1] == "name=^/ainode-vllm-node-solo$", "exact-name filter, not a prefix"


def test_stop_and_rm_runs_stop_rm_then_waits(monkeypatch):
    fake = _Ps(hits=1)
    monkeypatch.setattr(nvidia_mod.subprocess, "run", fake.run)
    monkeypatch.setattr(nvidia_mod.subprocess, "check_output", fake.check_output)
    b = _backend(monkeypatch)
    b._docker_stop_and_rm_best_effort("ainode-vllm-node-solo-8001")
    heads = [c[:2] for c in fake.calls]
    assert heads[:2] == [["docker", "stop"], ["docker", "rm"]]
    assert heads[2:] == [["docker", "ps"], ["docker", "ps"]], "polled after rm until clear"


def test_a_generic_fake_answer_does_not_read_as_in_use(monkeypatch):
    """A line that is not a container id (e.g. a test fake's 'ok') must not spin."""
    monkeypatch.setattr(nvidia_mod.subprocess, "check_output", lambda *a, **k: "ctr_abc\n")
    b = _backend(monkeypatch)
    assert b._container_name_in_use("ainode-vllm-node-solo") is False


def test_cleanup_gives_up_at_the_ceiling_and_says_so(monkeypatch, caplog):
    fake = _Ps(hits=10_000)
    monkeypatch.setattr(nvidia_mod.subprocess, "run", fake.run)
    monkeypatch.setattr(nvidia_mod.subprocess, "check_output", fake.check_output)
    b = _backend(monkeypatch)
    monkeypatch.setattr(nvidia_mod.NvidiaBackend, "NAME_CLEAR_TIMEOUT_S", 0.0)
    with caplog.at_level("WARNING"):
        assert b._wait_for_container_name_to_clear("ainode-vllm-node-solo") is False
    assert "still in use" in caplog.text


@pytest.mark.asyncio
async def test_orphan_sweep_waits_for_removal(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        if cmd[:2] == ["docker", "ps"]:
            return types.SimpleNamespace(stdout="id1 id2\n", stderr="", returncode=0)
        return types.SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    polls = iter([[ID], [ID], []])   # removal drains over three polls
    monkeypatch.setattr(api_routes, "_orphan_engine_ids", lambda: next(polls))
    slept = []

    async def _fast(s):
        slept.append(s)
    monkeypatch.setattr(api_routes.asyncio, "sleep", _fast)
    await api_routes._sweep_orphan_engine_containers()
    heads = [c[:2] for c in calls]
    assert heads[0] == ["docker", "ps"]
    assert heads[1] == ["docker", "rm"] and calls[1][2:] == ["-f", "id1", "id2"]
    assert heads[2:] == [], "polling goes through its own seam, not subprocess.run"
    assert len(slept) == 2, "slept between polls until the filter came back empty"


@pytest.mark.asyncio
async def test_orphan_sweep_is_a_no_op_with_nothing_to_remove(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        return types.SimpleNamespace(stdout="", stderr="", returncode=0)
    monkeypatch.setattr(subprocess, "run", fake_run)
    await api_routes._sweep_orphan_engine_containers()
    assert [c[:2] for c in calls] == [["docker", "ps"]]
