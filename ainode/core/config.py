"""AINode configuration management."""

import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional

AINODE_HOME = Path(os.environ.get("AINODE_HOME", Path.home() / ".ainode"))
CONFIG_FILE = AINODE_HOME / "config.json"
MODELS_DIR = AINODE_HOME / "models"
LOGS_DIR = AINODE_HOME / "logs"
DATASETS_DIR = AINODE_HOME / "datasets"
TRAINING_DIR = AINODE_HOME / "training"


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
    engine_strategy: str = "pip"  # "pip" | "docker"
    model: str = "meta-llama/Llama-3.2-3B-Instruct"
    models_dir: str = str(MODELS_DIR)
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    quantization: Optional[str] = None  # awq, gptq, fp8, None
    trust_remote_code: bool = False

    # Cluster
    cluster_enabled: bool = True
    cluster_secret: Optional[str] = None
    # Role this node should take in the cluster:
    #   "auto"   -> elected dynamically (lowest node_id among online auto nodes)
    #   "master" -> explicitly the cluster head
    #   "worker" -> never becomes master; follows whichever node is master
    cluster_role: str = "auto"
    # Shared identifier -- only nodes with the same cluster_id see each other.
    cluster_id: str = "default"
    # Optional explicit master address for workers (e.g. "10.0.0.1:3000").
    master_address: Optional[str] = None

    # Distributed inference mode:
    #   "solo"    — run a single vLLM locally. No Ray, no peers.
    #   "head"    — run vLLM sharded (TP/PP) across this node + peer_ips via
    #               eugr's launch-cluster.sh. The head additionally launches
    #               the Ray worker containers on each peer over SSH/docker
    #               socket. UI + API served here.
    #   "member"  — run AINode discovery + aiohttp API + UI on this node, but
    #               do NOT start any vLLM. Serves as a cluster member so the
    #               head's eugr launcher can place a Ray worker container
    #               directly on this box. Expected to announce itself via
    #               UDP discovery so the head's UI sees it.
    distributed_mode: str = "solo"  # "solo" | "head" | "member"
    # IPs of peer workers (on the cluster_interface subnet) when distributed.
    # Used only when distributed_mode="head".
    peer_ips: List[str] = field(default_factory=list)
    # SSH user for head-to-worker passwordless login (eugr launcher uses it).
    ssh_user: str = "ubuntu"
    # Interface NCCL/Ray/Gloo bind to (e.g. "enp1s0f0np0" for DGX Spark direct
    # connect, or the dedicated cluster-switch NIC).
    cluster_interface: str = "eno1"

    # Storage paths (override defaults)
    datasets_dir: Optional[str] = None
    training_dir: Optional[str] = None
    hf_cache_dir: Optional[str] = None

    # CORS
    cors_origins: Optional[str] = None  # comma-separated list

    # Training defaults
    training_default_method: str = "lora"       # lora | full | qlora
    training_default_epochs: int = 3
    training_default_batch_size: int = 4
    training_default_learning_rate: float = 2e-4

    # Onboarding
    email: Optional[str] = None
    onboarded: bool = False

    # Hugging Face credentials — needed for gated repos (e.g. Llama variants).
    # Stored in config.json; not exported to the environment by default.
    # Set via onboarding or manually: ainode config --hf-token <token>
    hf_token: Optional[str] = None

    # Telemetry (opt-in)
    telemetry: bool = False

    def save(self):
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
    for d in [AINODE_HOME, MODELS_DIR, LOGS_DIR, DATASETS_DIR, TRAINING_DIR]:
        d.mkdir(parents=True, exist_ok=True)
