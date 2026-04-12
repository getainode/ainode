"""vLLM engine wrapper — manages the vLLM inference server."""

import subprocess
import signal
import os
from typing import Optional
from ainode.core.config import NodeConfig
from ainode.core.gpu import detect_gpu


class VLLMEngine:
    """Start, stop, and manage a vLLM inference server."""

    def __init__(self, config: NodeConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None

    def start(self) -> bool:
        """Start the vLLM server."""
        if self.process and self.process.poll() is None:
            return True  # Already running

        gpu = detect_gpu()

        cmd = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.model,
            "--host", self.config.host,
            "--port", str(self.config.api_port),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--trust-remote-code",
        ]

        if self.config.max_model_len:
            cmd.extend(["--max-model-len", str(self.config.max_model_len)])

        if self.config.quantization:
            cmd.extend(["--quantization", self.config.quantization])

        # DGX Spark / GB10 unified memory needs dtype override
        if gpu and gpu.unified_memory:
            cmd.extend(["--dtype", "bfloat16"])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        return self.process.poll() is None

    def stop(self):
        """Stop the vLLM server."""
        if self.process and self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def is_running(self) -> bool:
        """Check if the vLLM server is running."""
        return self.process is not None and self.process.poll() is None

    @property
    def api_url(self) -> str:
        return f"http://localhost:{self.config.api_port}/v1"
