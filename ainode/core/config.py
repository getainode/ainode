"""AINode configuration management."""

import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

AINODE_HOME = Path(os.environ.get("AINODE_HOME", Path.home() / ".ainode"))
CONFIG_FILE = AINODE_HOME / "config.json"
MODELS_DIR = AINODE_HOME / "models"
LOGS_DIR = AINODE_HOME / "logs"


@dataclass
class NodeConfig:
    """Configuration for this AINode instance."""

    # Identity
    node_id: Optional[str] = None
    node_name: Optional[str] = None

    # Network
    host: str = "0.0.0.0"
    api_port: int = 8000
    web_port: int = 3000
    discovery_port: int = 5678

    # Engine
    model: str = "meta-llama/Llama-3.2-3B-Instruct"
    models_dir: str = str(MODELS_DIR)
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    quantization: Optional[str] = None  # awq, gptq, fp8, None

    # Cluster
    cluster_enabled: bool = True
    cluster_secret: Optional[str] = None

    # Onboarding
    email: Optional[str] = None
    onboarded: bool = False

    # Telemetry (opt-in)
    telemetry: bool = False

    def save(self):
        """Save config to disk."""
        AINODE_HOME.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls) -> "NodeConfig":
        """Load config from disk, or return defaults."""
        if CONFIG_FILE.exists():
            data = json.loads(CONFIG_FILE.read_text())
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        return cls()


def ensure_dirs():
    """Create AINode directories if they don't exist."""
    for d in [AINODE_HOME, MODELS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
