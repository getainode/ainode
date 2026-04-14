"""Docker-based vLLM engine wrapper.

For GB10 / CUDA 13 nodes where pip vLLM wheels are unavailable, AINode runs
vLLM inside the pre-built ``scitrera/dgx-spark-vllm`` container. This class
drives ``docker compose`` and exposes the same interface as
:class:`ainode.engine.vllm_engine.VLLMEngine` so ``cmd_start`` can dispatch
transparently based on ``config.engine_strategy``.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from ainode.core.config import AINODE_HOME, LOGS_DIR, NodeConfig

logger = logging.getLogger(__name__)

CONTAINER_NAME = "ainode-vllm"
COMPOSE_FILE = AINODE_HOME / "docker-compose.yml"
ENV_FILE = AINODE_HOME / ".env"


class DockerEngineError(RuntimeError):
    """Raised when the docker CLI is unavailable or a compose command fails."""


class DockerEngine:
    """Drive the vLLM container via ``docker compose``.

    Mirrors :class:`VLLMEngine` so callers don't need to care which backend
    is serving inference. All state lives in the container + on disk — this
    class is stateless aside from a cached ``_log_file`` path.
    """

    def __init__(self, config: NodeConfig, on_ready=None):
        self.config = config
        self.on_ready = on_ready
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._log_file: Path = LOGS_DIR / "vllm.log"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_docker(self) -> None:
        if shutil.which("docker") is None:
            raise DockerEngineError(
                "docker CLI not found. Install Docker + compose plugin first."
            )

    def _compose_cmd(self, *args: str) -> list[str]:
        """Build a ``docker compose ... <args>`` command list."""
        base = ["docker", "compose", "-f", str(COMPOSE_FILE)]
        if ENV_FILE.exists():
            base.extend(["--env-file", str(ENV_FILE)])
        base.extend(args)
        return base

    def _run(self, cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        """Run a subprocess, capturing output by default."""
        kwargs.setdefault("capture_output", True)
        kwargs.setdefault("text", True)
        return subprocess.run(cmd, **kwargs)

    def _read_env(self) -> dict:
        """Parse the KEY=VALUE .env file. Missing file → empty dict."""
        env: dict[str, str] = {}
        if not ENV_FILE.exists():
            return env
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
        return env

    def _write_env(self, env: dict) -> None:
        ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{k}={v}" for k, v in env.items()]
        ENV_FILE.write_text("\n".join(lines) + "\n")

    def _update_env(self, **updates: str) -> None:
        env = self._read_env()
        for key, value in updates.items():
            env[key] = str(value)
        self._write_env(env)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Bring up the compose stack in detached mode."""
        self._require_docker()
        if not COMPOSE_FILE.exists():
            raise DockerEngineError(
                f"compose file missing at {COMPOSE_FILE}. Run the installer."
            )

        result = self._run(self._compose_cmd("up", "-d"))
        if result.returncode != 0:
            logger.error("docker compose up failed: %s", result.stderr)
            return False
        logger.info("docker compose up -d: %s", result.stdout.strip())
        return True

    def stop(self) -> None:
        """Bring down the compose stack."""
        if shutil.which("docker") is None or not COMPOSE_FILE.exists():
            return
        self._run(self._compose_cmd("down"))

    def is_running(self) -> bool:
        """Return True iff the ainode-vllm container is listed by ``docker ps``."""
        if shutil.which("docker") is None:
            return False
        result = self._run([
            "docker", "ps",
            "--filter", f"name={CONTAINER_NAME}",
            "--format", "{{.Names}}",
        ])
        if result.returncode != 0:
            return False
        names = [n.strip() for n in result.stdout.splitlines() if n.strip()]
        return CONTAINER_NAME in names

    def wait_ready(self, timeout: int = 300) -> bool:
        """Poll the vLLM ``/v1/models`` endpoint until it returns 200 or we time out."""
        url = f"http://127.0.0.1:{self.config.api_port}/v1/models"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if 200 <= resp.status < 300:
                        if self.on_ready:
                            try:
                                self.on_ready()
                            except Exception:  # pragma: no cover - user callback
                                logger.exception("on_ready callback failed")
                        return True
            except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, OSError):
                pass
            time.sleep(2)
        return False

    def launch(self, model_id: str) -> bool:
        """Swap the model by writing AINODE_MODEL into .env and recreating the container."""
        self._require_docker()
        self._update_env(AINODE_MODEL=model_id)
        self.config.model = model_id
        try:
            self.config.save()
        except Exception:  # pragma: no cover - config persistence best-effort
            logger.exception("Failed to persist config.model")
        result = self._run(self._compose_cmd("up", "-d"))
        return result.returncode == 0

    def launch_distributed(self, sharding_config) -> bool:
        """Set TP size / Ray head address and recreate the container."""
        self._require_docker()

        updates: dict[str, str] = {}
        tp = getattr(sharding_config, "tensor_parallel_size", 1) or 1
        ray_addr = getattr(sharding_config, "ray_head_address", None)
        model = getattr(sharding_config, "model", None)

        updates["AINODE_TP_SIZE"] = str(tp)
        updates["AINODE_RAY_ADDRESS"] = ray_addr or ""
        if model:
            updates["AINODE_MODEL"] = model

        self._update_env(**updates)
        result = self._run(self._compose_cmd("up", "-d"))
        return result.returncode == 0

    def logs(self, n: int = 100) -> str:
        """Return the last ``n`` lines from the container's logs."""
        if shutil.which("docker") is None:
            return ""
        result = self._run(["docker", "logs", "--tail", str(n), CONTAINER_NAME])
        if result.returncode != 0:
            return result.stderr or ""
        return result.stdout

    def health_check(self) -> dict:
        """Parity with VLLMEngine.health_check — used by ``ainode status``."""
        result = {
            "process_alive": self.is_running(),
            "api_responding": False,
            "models_loaded": [],
        }
        try:
            url = f"http://127.0.0.1:{self.config.api_port}/v1/models"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                result["api_responding"] = True
                result["models_loaded"] = [m["id"] for m in data.get("data", [])]
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------
    # Properties for VLLMEngine parity
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self.is_running()

    @property
    def api_url(self) -> str:
        return f"http://localhost:{self.config.api_port}/v1"

    @property
    def log_path(self) -> Optional[Path]:
        return self._log_file

    @property
    def process(self):
        """VLLMEngine exposes .process — return a lightweight shim so ``engine.process.wait()`` works."""
        return _ContainerProcessShim(self)


class _ContainerProcessShim:
    """Minimal duck-typed stand-in for ``subprocess.Popen`` used by cmd_start.

    ``cmd_start`` calls ``engine.process.wait()`` to block until the engine
    exits. For the Docker backend we instead block until the container stops
    showing up in ``docker ps`` or the caller sends SIGINT.
    """

    def __init__(self, engine: DockerEngine):
        self._engine = engine

    def wait(self, timeout: Optional[float] = None):
        deadline = None if timeout is None else time.time() + timeout
        while self._engine.is_running():
            if deadline is not None and time.time() > deadline:
                raise subprocess.TimeoutExpired(cmd="docker compose", timeout=timeout)
            time.sleep(5)
        return 0

    def poll(self):
        return None if self._engine.is_running() else 0
