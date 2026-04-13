"""vLLM engine wrapper — manages the vLLM inference server lifecycle."""

import logging
import subprocess
import signal
import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional, Callable
from ainode.core.config import NodeConfig, LOGS_DIR
from ainode.core.gpu import detect_gpu

logger = logging.getLogger(__name__)


class VLLMEngine:
    """Start, stop, manage, and health-check a vLLM inference server."""

    def __init__(self, config: NodeConfig, on_ready: Optional[Callable] = None):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.on_ready = on_ready
        self._log_thread: Optional[threading.Thread] = None
        self._ready = False
        self._log_file: Optional[Path] = None

    def build_cmd(self) -> list[str]:
        gpu = detect_gpu()

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.model,
            "--host", "127.0.0.1",
            "--port", str(self.config.api_port),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            ]

        if self.config.trust_remote_code:
            cmd.append("--trust-remote-code")

        if self.config.max_model_len:
            cmd.extend(["--max-model-len", str(self.config.max_model_len)])

        if self.config.quantization:
            cmd.extend(["--quantization", self.config.quantization])

        if self.config.models_dir:
            cmd.extend(["--download-dir", self.config.models_dir])

        # DGX Spark / GB10 unified memory needs dtype override
        if gpu and gpu.unified_memory:
            cmd.extend(["--dtype", "bfloat16"])

        return cmd

    def start(self) -> bool:
        """Start the vLLM server with log streaming and readiness detection."""
        if self.process and self.process.poll() is None:
            return True

        cmd = self.build_cmd()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

        # Auto-detect CUDA runtime for GB10/Spark unified memory systems
        # vLLM wheels may be built for CUDA 12 but the system has CUDA 13
        if not env.get("LD_PRELOAD"):
            import glob
            for pattern in [
                os.path.join(sys.prefix, "lib/python*/site-packages/nvidia/cu*/lib/libcudart.so.*"),
                "/usr/local/cuda/targets/*/lib/libcudart.so.*",
                "/usr/local/cuda-*/targets/*/lib/libcudart.so.*",
            ]:
                matches = sorted(glob.glob(pattern))
                if matches:
                    env["LD_PRELOAD"] = matches[-1]
                    break

        # Log to file
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._log_file = LOGS_DIR / "vllm.log"

        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )

        # Start log streaming thread
        self._log_thread = threading.Thread(target=self._stream_logs, daemon=True)
        self._log_thread.start()

        return self.process.poll() is None

    def _stream_logs(self):
        """Read vLLM stdout, write to log file, detect readiness."""
        if not self.process or not self.process.stdout:
            return

        with open(self._log_file, "a") as log_f:
            for line in self.process.stdout:
                log_f.write(line)
                log_f.flush()

                # Detect vLLM readiness
                if not self._ready and ("Uvicorn running on" in line or "Application startup complete" in line):
                    self._ready = True
                    if self.on_ready:
                        self.on_ready()

    def wait_ready(self, timeout: int = 300) -> bool:
        """Block until vLLM is ready to serve requests, or timeout."""
        start = time.time()
        while time.time() - start < timeout:
            if self._ready:
                return True
            if self.process and self.process.poll() is not None:
                return False  # Process died
            time.sleep(1)
        return False

    def health_check(self) -> dict:
        """Check if vLLM is healthy and responding.

        Works both when this instance owns the process and when checking
        an externally-running vLLM server (e.g. from ``ainode status``).
        """
        import urllib.request
        import json

        result = {
            "process_alive": self.is_running(),
            "api_responding": False,
            "models_loaded": [],
        }

        try:
            req = urllib.request.Request(f"{self.api_url}/models")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                result["api_responding"] = True
                result["models_loaded"] = [m["id"] for m in data.get("data", [])]
        except Exception:
            pass

        return result

    def stop(self):
        """Graceful shutdown of vLLM server."""
        if self.process and self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self.process = None
            self._ready = False

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def api_url(self) -> str:
        return f"http://localhost:{self.config.api_port}/v1"

    @property
    def log_path(self) -> Optional[Path]:
        return self._log_file

    # --- Distributed / Sharded Inference ---

    def build_distributed_cmd(self, sharding_config) -> list[str]:
        """Generate vLLM launch args for distributed inference."""
        from ainode.engine.sharding import ShardingConfig

        cmd = self.build_cmd()

        if not isinstance(sharding_config, ShardingConfig):
            return cmd

        tp = sharding_config.tensor_parallel_size
        pp = sharding_config.pipeline_parallel_size

        if tp > 1:
            cmd.extend(["--tensor-parallel-size", str(tp)])
        if pp > 1:
            cmd.extend(["--pipeline-parallel-size", str(pp)])

        if sharding_config.model and sharding_config.model != self.config.model:
            try:
                model_idx = cmd.index("--model")
                cmd[model_idx + 1] = sharding_config.model
            except (ValueError, IndexError):
                pass

        return cmd

    def launch_distributed(self, sharding_config) -> bool:
        """Launch vLLM with distributed sharding via Ray."""
        from ainode.engine.sharding import ShardingConfig

        if self.process and self.process.poll() is None:
            logger.warning("Engine already running, stop it first")
            return False

        if not isinstance(sharding_config, ShardingConfig):
            logger.error("Invalid sharding config")
            return False

        cmd = self.build_distributed_cmd(sharding_config)
        env = os.environ.copy()

        if sharding_config.is_distributed:
            if sharding_config.ray_head_address:
                env["RAY_ADDRESS"] = sharding_config.ray_head_address
            logger.info(
                "Launching distributed vLLM: TP=%d PP=%d across %d nodes",
                sharding_config.tensor_parallel_size,
                sharding_config.pipeline_parallel_size,
                len(sharding_config.nodes),
            )
        else:
            env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            logger.info("Launching single-node vLLM with TP=%d", sharding_config.tensor_parallel_size)

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._log_file = LOGS_DIR / "vllm.log"

        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )

        self._log_thread = threading.Thread(target=self._stream_logs, daemon=True)
        self._log_thread.start()

        return self.process.poll() is None
