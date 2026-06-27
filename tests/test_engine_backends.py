"""Tests for the pluggable engine backends."""

import pytest

from ainode.core.config import NodeConfig
from ainode.engine.backends import (
    EngineBackend,
    EugrBackend,
    NvidiaBackend,
    get_backend,
)


class TestEngineBackendFactory:
    def test_default_backend_is_eugr(self):
        config = NodeConfig()
        assert config.engine_backend == "eugr"
        backend = get_backend(config)
        assert isinstance(backend, EugrBackend)

    def test_explicit_eugr(self):
        config = NodeConfig(engine_backend="eugr")
        backend = get_backend(config)
        assert isinstance(backend, EugrBackend)

    def test_nvidia_backend_instance(self):
        """Phase 4: NvidiaBackend is a real implementation, not a stub.

        Pre-Phase-4 this asserted ``NotImplementedError`` on ``start()``.
        The full lifecycle is now exercised in ``test_nvidia_backend.py``;
        here we only verify the factory wires it up.
        """
        config = NodeConfig(engine_backend="nvidia")
        backend = get_backend(config)
        assert isinstance(backend, NvidiaBackend)
        # Unknown distributed_mode is the only remaining way ``start`` can
        # raise synchronously — useful as a smoke test for the dispatch.
        config.distributed_mode = "bogus"
        with pytest.raises(Exception):
            backend.start()

    def test_invalid_backend_raises(self):
        config = NodeConfig(engine_backend="bogus")
        with pytest.raises(ValueError, match="Unknown engine_backend"):
            get_backend(config)

    def test_both_backends_implement_abc(self):
        """Ensure concrete backends implement all abstract methods."""
        eugr = EugrBackend(NodeConfig())
        nvidia = NvidiaBackend(NodeConfig())
        assert isinstance(eugr, EngineBackend)
        assert isinstance(nvidia, EngineBackend)
        for method in ["start", "stop", "wait_ready", "is_running", "health_check"]:
            assert hasattr(eugr, method) and callable(getattr(eugr, method))
            assert hasattr(nvidia, method) and callable(getattr(nvidia, method))
        for prop in ["api_url", "log_path", "process"]:
            assert hasattr(eugr, prop)
            assert hasattr(nvidia, prop)


class TestBackwardCompatShim:
    def test_docker_engine_import_still_works(self):
        """Old code doing ``from ainode.engine.docker_engine import DockerEngine`` must keep working."""
        from ainode.engine.docker_engine import DockerEngine, DockerEngineError
        assert DockerEngine is not None
        assert issubclass(DockerEngineError, RuntimeError)

    def test_docker_engine_build_emits_deprecation(self, recwarn):
        from ainode.engine.docker_engine import build_engine
        config = NodeConfig()
        build_engine(config)
        assert any(issubclass(w.category, DeprecationWarning) for w in recwarn)

    def test_docker_engine_alias_points_to_eugr_backend(self):
        """The shim's DockerEngine alias must resolve to the real EugrBackend class."""
        from ainode.engine.docker_engine import DockerEngine
        assert DockerEngine is EugrBackend

    def test_docker_engine_error_alias_points_to_eugr_error(self):
        from ainode.engine.backends.eugr import EugrBackendError
        from ainode.engine.docker_engine import DockerEngineError
        assert DockerEngineError is EugrBackendError
