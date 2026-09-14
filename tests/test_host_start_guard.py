"""Issue #61: `ainode start` on a host with no vLLM must fail cleanly.

A user pip-installed ainode on a GX10 and ran `ainode start` outside the
container. With the default engine_backend="eugr" that reaches
``subprocess.Popen(["vllm", ...])`` and died with a raw
``FileNotFoundError: [Errno 2] No such file or directory: 'vllm'``.

Two layers are covered here: a pre-flight guard in ``cmd_start`` that exits 1
with guidance, and ``EugrBackend.start_solo`` translating the Popen failure
into ``EugrBackendError`` carrying the same text for any other caller.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ainode.cli import main as cli
from ainode.core.config import NodeConfig
from ainode.engine.backends.eugr import (
    NO_VLLM_MESSAGE,
    EugrBackend,
    EugrBackendError,
)


def _config(**kw):
    defaults = dict(
        node_id="abc123",
        onboarded=True,
        model="Qwen/Qwen3-8B",
        engine_backend="eugr",
        engine_strategy="pip",
    )
    defaults.update(kw)
    return NodeConfig(**defaults)


@pytest.fixture
def start_harness(monkeypatch):
    """Stub cmd_start's environment: no disk, no PID file, no GPU, no engine."""
    state = {"get_backend_calls": 0, "removed_pid": 0}

    monkeypatch.setattr(cli, "ensure_dirs", lambda: None)
    monkeypatch.setattr(cli, "_write_pid", lambda: None)
    monkeypatch.setattr(
        cli, "_remove_pid",
        lambda: state.__setitem__("removed_pid", state["removed_pid"] + 1),
    )
    monkeypatch.setattr("ainode.core.gpu.detect_gpu", lambda: None)
    monkeypatch.setattr("ainode.models.api_routes.consume_start_clean", lambda: False)
    monkeypatch.delenv("AINODE_IN_CONTAINER", raising=False)

    def fake_get_backend(config, on_ready=None, instance_id=""):
        state["get_backend_calls"] += 1
        return state["engine"]

    monkeypatch.setattr("ainode.engine.backends.get_backend", fake_get_backend)
    monkeypatch.setattr("ainode.api.server.run_server",
                        lambda config=None, engine=None: None)
    state["engine"] = SimpleNamespace(start=lambda: True, stop=lambda: None)
    return state


def _run(monkeypatch, config, *, vllm_on_path=False, in_container=False):
    monkeypatch.setattr(cli.NodeConfig, "load", classmethod(lambda cls: config))
    monkeypatch.setattr(
        cli.shutil, "which",
        lambda name: "/usr/local/bin/vllm" if vllm_on_path else None,
    )
    args = SimpleNamespace(model=None, port=None, in_container=in_container)
    return cli.cmd_start(args)


# ---------------------------------------------------------------------------
# The pre-flight guard
# ---------------------------------------------------------------------------

def test_host_start_without_vllm_exits_one_with_guidance(
    start_harness, monkeypatch, capsys
):
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, _config(), vllm_on_path=False)

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "Cannot start the engine on this host." in out
    assert "AINode runs as a container image" in out
    assert "curl -fsSL https://ainode.dev/install | bash" in out
    assert "engine_backend" in out
    assert "nvidia" in out
    # A clean exit, not a crash, and the engine was never constructed.
    assert "Traceback" not in out
    assert "FileNotFoundError" not in out
    assert start_harness["get_backend_calls"] == 0
    assert start_harness["removed_pid"] == 1


def test_guard_does_not_fire_when_vllm_is_on_path(start_harness, monkeypatch):
    _run(monkeypatch, _config(), vllm_on_path=True)
    assert start_harness["get_backend_calls"] == 1


def test_guard_does_not_fire_in_the_container(start_harness, monkeypatch):
    _run(monkeypatch, _config(), vllm_on_path=False, in_container=True)
    assert start_harness["get_backend_calls"] == 1


def test_guard_does_not_fire_for_the_nvidia_backend(start_harness, monkeypatch):
    """engine_backend=nvidia runs the engine in Docker; no host vLLM needed."""
    _run(monkeypatch, _config(engine_backend="nvidia"), vllm_on_path=False)
    assert start_harness["get_backend_calls"] == 1


def test_backend_error_from_start_is_printed_cleanly(
    start_harness, monkeypatch, capsys
):
    def boom():
        raise EugrBackendError(NO_VLLM_MESSAGE)

    start_harness["engine"] = SimpleNamespace(start=boom, stop=lambda: None)

    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, _config(), vllm_on_path=True)

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "Cannot start the engine." in out
    assert "AINode runs as a container image" in out
    assert "Traceback" not in out


# ---------------------------------------------------------------------------
# The backend-level translation
# ---------------------------------------------------------------------------

def test_start_solo_translates_missing_vllm_into_a_backend_error(monkeypatch):
    backend = EugrBackend(_config())
    monkeypatch.setattr(backend, "is_running", lambda: False)
    monkeypatch.setattr("ainode.core.gpu.detect_gpu", lambda: None)

    def missing_binary(*a, **kw):
        raise FileNotFoundError(2, "No such file or directory: 'vllm'")

    monkeypatch.setattr("subprocess.Popen", missing_binary)

    with pytest.raises(EugrBackendError) as exc:
        backend.start_solo()

    assert str(exc.value) == NO_VLLM_MESSAGE
    assert isinstance(exc.value.__cause__, FileNotFoundError)


def test_no_vllm_message_text():
    """The exact guidance. Pinned so CLI and backend cannot drift apart."""
    assert NO_VLLM_MESSAGE == (
        "AINode runs as a container image, and this host has no vLLM install.\n"
        "  `ainode start` outside the container has nothing to launch the "
        "engine with.\n"
        "\n"
        "  Install the container-native way:\n"
        "      curl -fsSL https://ainode.dev/install | bash\n"
        "\n"
        "  Or, to run the engine in Docker from this host checkout, set\n"
        '  "engine_backend": "nvidia" in ~/.ainode/config.json and start again.'
    )
