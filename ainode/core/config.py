"""AINode configuration management."""

import os
import json
import stat
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

AINODE_HOME = Path(os.environ.get("AINODE_HOME", Path.home() / ".ainode"))
CONFIG_FILE = AINODE_HOME / "config.json"
MODELS_DIR = AINODE_HOME / "models"
LOGS_DIR = AINODE_HOME / "logs"
DATASETS_DIR = AINODE_HOME / "datasets"
TRAINING_DIR = AINODE_HOME / "training"

# Container path the engine's Hugging Face cache is mounted at. One home for the
# value: every engine backend mounts ``hf_cache_dir`` here, and a catalog recipe
# that has to point a tool at that cache (HF_HOME, a JIT/kernel cache dir) reads
# it from here instead of spelling the path a second time.
HF_CACHE_MOUNT = "/root/.cache/huggingface"

# The UDP port discovery broadcasts and listens on. One home for the value:
# NodeConfig below, both discovery classes and the installer all read THIS, so a
# node that never wrote a config.json agrees with the fleet and with the docs.
# It was 5678 in this dataclass and in discovery/broadcast.py while the installer
# wrote 5679, the fleet ran 5679 and the public docs documented 5679 (#181): a
# source install then broadcast into a port nobody listened on, came up healthy,
# served its own model, and never appeared in any peer's cluster view, with no log
# line anywhere saying the port disagreed.
DEFAULT_DISCOVERY_PORT = 5679

# Which multi-node executor a distributed (head) launch uses when nothing says
# otherwise. One home for the value: every caller that reads
# ``config.distributed_executor`` defensively falls back to THIS.
#   "mp":  one `vllm serve` container per node (rank 0 here, `--headless` rank k
#          on each peer) rendezvousing on --master-addr/--master-port with vLLM's
#          own multi-node executor. Needs nothing but vLLM.
#   "ray": a `ray start --head` container here plus `ray start` workers on each
#          peer over SSH. REQUIRES the `ray` CLI inside the engine image.
# This defaulted to "ray" through 0.5.26, and no image AINode ships or launches
# has ray in it: not the orchestrator image (python:3.12-slim + this package) and
# not vllm/vllm-openai. So every distributed launch that did not carry a catalog
# recipe naming "mp" died inside the container (#172, and #84 before it). "mp" is
# also the shape every proven distributed launch on the fleet actually used.
DEFAULT_DISTRIBUTED_EXECUTOR = "mp"

# The engine backend a node uses when nothing says otherwise. One home for the
# value: every caller that reads ``config.engine_backend`` defensively falls back
# to THIS, so a config.json with the key missing or empty behaves exactly like a
# fresh NodeConfig instead of picking a different backend per call site.
DEFAULT_ENGINE_BACKEND = "nvidia"


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
    discovery_port: int = DEFAULT_DISCOVERY_PORT  # see the constant (#181)

    # Engine
    engine_strategy: str = "pip"  # "pip" | "docker"
    # Which Docker-engine backend to use when engine_strategy == "docker".
    #   "nvidia": one vLLM engine CONTAINER per instance, from
    #             $NVIDIA_VLLM_IMAGE or the catalog recipe's engine_image
    #             (engine/backends/nvidia.py). THE DEFAULT, because it is the
    #             only backend a node installed the documented way can run: the
    #             shipped image is python:3.12-slim plus this package
    #             (scripts/Dockerfile.ainode), with no vllm binary, no ray and
    #             no eugr launcher in it.
    #   "eugr":   eugr/spark-vllm-docker's launch-cluster.sh plus a `vllm` on
    #             PATH (the v0.4.x default). Still supported, but OPT-IN: it
    #             only works where something actually ships vLLM, so a config
    #             has to ask for it by name.
    # This defaulted to "eugr" through 0.5.25, which is why a fresh install could
    # not load any model until somebody hand-edited config.json (issue #164).
    engine_backend: str = DEFAULT_ENGINE_BACKEND
    model: str = "meta-llama/Llama-3.2-3B-Instruct"
    models_dir: str = str(MODELS_DIR)
    # Optional API aliases for /v1/models — emitted as ``--served-model-name a b c``.
    # Lets a client/router address the model by a short name (e.g. "Aegis-14B")
    # instead of the repo-id/slug. Falls back to ``model`` when unset.
    served_model_name: Optional[List[str]] = None
    max_model_len: Optional[int] = None
    # vLLM sizes the KV cache to this fraction of the GPU regardless of model
    # size, so 0.9 made a tiny model reserve ~110 GB on a 122 GB unified-memory
    # node — starving the OS and blocking model stacking. 0.5 is a safer default
    # for GB10 (still fits a 70B / per-node MoE share); push it higher per-load
    # (gpu_memory_utilization in the load body) for big-MoE long-context runs.
    gpu_memory_utilization: float = 0.5
    # KV-cache precision. fp8 is the GB10 design default — required for long
    # context (32k+) or vLLM OOMs sizing the cache at bf16 (see engine/AGENTS.md).
    # Set "" / "auto" to let vLLM choose if a model/quant ever rejects fp8.
    kv_cache_dtype: str = "fp8"
    # Provenance of kv_cache_dtype: True only when a caller EXPLICITLY supplied it
    # (per-load body or config). The multimodal fp8→auto safety downgrade
    # (engine/backends/nvidia.py) fires only on the DEFAULT fp8 — an explicit
    # fp8 request on a VLM is honored, giving the user a way to opt back in.
    kv_cache_dtype_explicit: bool = False
    quantization: Optional[str] = None  # awq, gptq, fp8, None
    trust_remote_code: bool = False
    # Extra `vllm serve` flags appended verbatim to the engine command line, e.g.
    # ["--moe-backend", "marlin", "--reasoning-parser", "qwen3"]. Models whose
    # published recipe needs flags AINode doesn't model (speculative decoding,
    # mamba/MoE backends, reasoning + tool-call parsers) launch through the
    # normal path instead of a hand-rolled container. Deliberately NOT validated
    # here — vLLM is the authority and rejects unknown flags at startup. A flag
    # supplied here WINS over the same built-in flag (see _build_vllm_serve_args).
    extra_vllm_args: List[str] = field(default_factory=list)
    # Per-instance engine container image. Empty = the backend default
    # ($NVIDIA_VLLM_IMAGE). Required when a model needs a newer vLLM than the
    # fleet default — e.g. Nemotron 3.5 Lightning and Qwen3.8 need
    # `vllm/vllm-openai:v0.27.1`, while the fleet default is a 0.17 build.
    # Setting this also disables the 0.17-era GB10 workarounds that would
    # otherwise be forced on (see NvidiaBackend._is_pinned_default_image).
    engine_image: str = ""
    # Per-instance environment for the engine container. Some engine features
    # are selected by env var, not by a `vllm serve` flag — the b12x FP4 kernel
    # path is VLLM_NVFP4_GEMM_BACKEND + friends, with no CLI equivalent. Merged
    # OVER the computed NCCL env at launch, so a recipe can also correct an
    # autodetected NCCL value when a model needs it. Deliberately unvalidated,
    # same as extra_vllm_args: the engine is the authority on what it accepts.
    extra_env: Dict[str, str] = field(default_factory=dict)
    # Extra docker volume mounts for the engine container, each
    # "host:container" or "host:container:ro". Applied to the solo AND the
    # distributed launch. A recipe needs this when the engine wants a writable
    # directory that is not the HF cache: a JIT/kernel cache an image compiles
    # into on first launch, for instance. Host paths are fleet-specific, so a
    # catalog entry should prefer a path under the HF cache mount (which AINode
    # already mounts on every node) and leave this for an operator override.
    extra_volumes: List[str] = field(default_factory=list)
    # Which multi-node executor a distributed (head) launch uses:
    #   "ray": a `ray start --head` container here, `ray start` worker
    #           containers on each peer over SSH, then `vllm serve
    #           --distributed-executor-backend ray` via docker exec in the head.
    #           REQUIRES the `ray` CLI inside the engine image.
    #   "mp":  one `vllm serve` container per node (rank 0 here, `--headless`
    #           rank k on each peer) rendezvousing on --master-addr/--master-port
    #           with vLLM's own multi-node executor. Needs nothing but vLLM, so
    #           it is the shape for a custom engine image: the GB10 build that
    #           serves DeepSeek V4 Flash ships no ray, and neither does stock
    #           vllm/vllm-openai.
    # Per-model, so a catalog recipe can pin the shape its image supports.
    # Defaults to "mp" (see DEFAULT_DISTRIBUTED_EXECUTOR): "ray" was the default
    # through 0.5.26 and no shipped image can run it (#172).
    distributed_executor: str = DEFAULT_DISTRIBUTED_EXECUTOR  # "mp" | "ray"
    # Max inbound request body for the API server, in MB. aiohttp defaults to
    # 1 MB, which silently caps a 262k-context model at roughly 190k tokens of
    # prompt: the proxy 413s the request before the engine ever sees it, and the
    # caller gets "Request Entity Too Large" with nothing pointing at us. Sized
    # for a 1M-token context (~5 MB of text) plus base64 image/video parts on
    # the multimodal models, with headroom.
    max_request_mb: int = 64
    # Startup-replay bind wait (see models/api_routes.py::_ensure_serving).
    # Time-to-bind is a property of the model and the engine image, not of
    # AINode: on vllm/vllm-openai:v0.27.1 a 27B NVFP4 model spends minutes in
    # FlashInfer fp4_gemm autotune and CUDA graph capture before it listens on
    # its port. So the replay does not wait a fixed span; it waits as long as
    # the engine is visibly making progress and gives up only on evidence of
    # death.
    #
    # engine_bind_log_silence_seconds: how long the engine may show NO SIGN OF
    # WORK while still counting as alive. Two signals feed it and either one
    # resets the clock (#112): the engine container's CPU time advancing
    # (EngineBackend.activity_mark, the primary signal) and a new log line
    # (secondary). The log alone was never enough -- vLLM prints nothing through
    # weight load, torch.compile and FlashInfer autotune, measured quiet for
    # 206 s (Qwen3.8 27B NVFP4), 363 s (Nemotron 3.5) and once 48 minutes
    # (autotune), which is what pushed this knob from 120 to 900 s while those
    # engines were healthy and busy the whole time. A wedged engine does no work
    # at all, so with activity watched the budget is back to five minutes. Past
    # this gap (container still up, port still closed, nothing happening) the
    # replay calls the start dead and relaunches once. The name is kept for
    # config compatibility.
    engine_bind_log_silence_seconds: int = 300
    # engine_bind_ceiling_seconds: absolute cap on one bind wait, so an engine
    # that is wedged but still chatty cannot hold boot open forever. Generous on
    # purpose: the slowest bind measured on a GB10 (27B NVFP4, autotune + graph
    # capture) was about 12 minutes, and a multi-node mp launch is far slower
    # still -- Qwen3.8-Flash-Next on the Spark pair took 36 minutes to ready on
    # the one launch that survived, with 48 minutes spent in kernel warmup on the
    # ones that did not (#134). At 1800 the ceiling was killing a launch that was
    # still making progress, which is the failure this knob exists to avoid. This
    # is the last resort, not the working limit: an engine doing no work at all is
    # already dead in five minutes by the knob above.
    engine_bind_ceiling_seconds: int = 3600

    # Cluster
    cluster_enabled: bool = True
    # Shared key every node in the cluster signs its UDP announcements with
    # (discovery/signing.py). Set it to the SAME value on every node: a node that
    # has it drops any announcement it cannot verify, so a host on the broadcast
    # domain can no longer announce `role: "master"`, win the election and pull
    # inference traffic to an address of its choosing (#169). Left unset, discovery
    # stays unauthenticated exactly as it was, and the listener says so once per
    # process. Rotating it needs no restart: sender and listener both re-read this
    # file when it changes. Never leaves the node: scrubbed from GET /api/config.
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
    # Empty means autodetect: ainode.cluster.netdev.resolve_cluster_interface
    # ranks this host's real interfaces (RDMA-capable first, then the
    # default route) instead of guessing a hardware-specific name that may
    # not exist here. Set a name to pin it.
    cluster_interface: str = ""

    # Storage paths (override defaults)
    datasets_dir: Optional[str] = None
    training_dir: Optional[str] = None
    hf_cache_dir: Optional[str] = None

    # CORS
    cors_origins: Optional[str] = None  # comma-separated list

    # TLS for the API port. A nested block rather than four flat keys because it
    # is one decision: either this node serves HTTPS on its own port or it does
    # not. Shape and defaults live in ainode/tls/config.py::TLSConfig; written by
    # `ainode tls enable` (which writes the block surgically, so nothing else in
    # config.json is rewritten) and read at boot by api/server.py::listener_plan.
    #   {"enabled": true, "port": 3443,
    #    "cert_file": "/root/.ainode/tls/cert.pem",
    #    "key_file": "/root/.ainode/tls/key.pem"}
    # web_port stays plain HTTP whatever this says: every client in the fleet
    # talks to :3000, so TLS is an ADDITIONAL listener and never a replacement.
    tls: Dict = field(default_factory=dict)

    # Per-client limits on /v1. Off by default, because a home node behind one
    # user needs nothing and the limiter would only be one more thing to explain.
    # Shape and defaults live in ainode/ratelimit/middleware.py::RateLimitConfig.
    #   {"enabled": true, "requests_per_minute": 600, "burst": 60,
    #    "max_inflight": 8}
    # max_inflight is the one that matters on a GPU node: a single client opening
    # 200 concurrent completions occupies every engine in the cluster, and no
    # request-per-minute figure stops it, because it is one burst.
    rate_limit: Dict = field(default_factory=dict)

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

    # Metrics retention. Keeps the figures /api/metrics reports in a small SQLite
    # file under AINODE_HOME so a restart does not reset the node's history to
    # nothing (ainode/metrics/store.py). Keys, all optional:
    #   enabled          bool,  default True. The file is small and the write is
    #                    a handful of rows every interval; off is for a node with
    #                    a read-only or precious data directory.
    #   retention_hours  int,   default 48. How long the raw per-tick samples live.
    #   retention_days   int,   default 30. How long the 1-minute roll-ups live.
    #   interval_seconds float, default 15. The sampling cadence.
    # A plain dict and not a nested dataclass because ``load()`` feeds
    # config.json's values straight back into the field, so a dataclass would
    # come back as a dict on the second boot and nothing downstream could tell
    # which shape it had. MetricsSettings.from_config validates and clamps.
    metrics: Dict[str, object] = field(default_factory=dict)

    def save(self):
        """Write config.json 0600, through a temp file in the same directory.

        0600 because this file carries ``cluster_secret`` (the discovery signing
        key, and the key every node-to-node call derives from) and may carry
        ``hf_token``; it was written under the default umask, which is 0644 for
        the root the container runs as. Atomic because several readers stat and
        parse it live: ``ClusterSecret`` per discovery datagram and the auth
        middleware per request, and a half-written document would blank the
        cluster's key for as long as the write took.
        """
        AINODE_HOME.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_FILE.with_name(CONFIG_FILE.name + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        os.chmod(tmp, 0o600)
        tmp.replace(CONFIG_FILE)

    @classmethod
    def load(cls) -> "NodeConfig":
        """Load config from disk, or return defaults."""
        if CONFIG_FILE.exists():
            data = json.loads(CONFIG_FILE.read_text())
            tighten_config_mode()
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        return cls()


def tighten_config_mode() -> bool:
    """chmod config.json to 0600 when it is wider. True when it changed.

    Called once per load, for the nodes installed before :meth:`NodeConfig.save`
    started writing it that way: the file names this cluster's shared secret, and
    on a multi-user host every account could read it.
    """
    try:
        mode = stat.S_IMODE(os.stat(CONFIG_FILE).st_mode)
    except OSError:
        return False
    if not mode & 0o077:
        return False
    try:
        os.chmod(CONFIG_FILE, 0o600)
    except OSError:
        return False
    return True


def ensure_dirs():
    for d in [AINODE_HOME, MODELS_DIR, LOGS_DIR, DATASETS_DIR, TRAINING_DIR]:
        d.mkdir(parents=True, exist_ok=True)
