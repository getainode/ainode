# CLAUDE.md, AINode Product

> **DOX:** Before editing, walk this repo's `AGENTS.md` chain (root to target folder) and obey the nearest one as the local edit contract. This file holds **Claude-specific** config only and must not restate the `AGENTS.md` edit rules or the Vault's state/"why". Boundary: Obsidian Vault, `Systems/Claude Code Harness/DOX, Ownership Charter & Pilot`.

## Project Overview

AINode turns any NVIDIA GPU into a local AI platform. Inference and fine-tuning in your browser, one command to install, and nodes that find each other.

- **Product repo:** https://github.com/getainode/ainode
- **Marketing site repo:** https://github.com/getainode/ainode.dev
- **Live site:** https://ainode.dev
- **Public docs:** https://docs.ainode.dev (repo: github.com/getainode/ainode-docs)
- **License:** Apache 2.0

## Tech Stack

The AINode container: Python 3.10+, aiohttp (API server and web UI), pynvml plus
psutil (GPU detection), Rich (terminal UI). Nothing else: no vLLM, no torch, no ray,
no CUDA.

The engine container, one per loaded model: vLLM, from the catalog recipe's
`engine_image` or `NVIDIA_VLLM_IMAGE` in `engine/backends/nvidia.py`. Whatever NCCL
that image ships, configured by env AINode computes from the host's fabric interface.

## Distribution

AINode publishes ONE orchestrator image, `ghcr.io/getainode/ainode:<version>`, to GHCR
and nowhere else. There is no Docker Hub mirror: `argentaios/ainode` there stops at
0.4.7 and the CI mirror step points at a namespace that 404s. CI builds on a
self-hosted aarch64 runner (a Spark) via `.github/workflows/publish-image.yml`.

Two job/engine images are published beside it, each on its own trigger and its own tag
scheme, because neither tracks an AINode release: `ainode-train:<ainode version>`
(`publish-train-image.yml`, a `train-v*` tag) and
`ainode-whisper:<base engine tag>` (`publish-whisper-image.yml`, a `whisper-v*` tag),
which is the engine image the speech-to-text catalog entry pins.

A node runs two images at minimum: that orchestrator plus an engine container per
loaded model, pulled on first launch (the installer pre-pulls the default one).

## Key Commands

```bash
# End-user install (one node):
curl -fsSL https://ainode.dev/install | bash

# Distributed head with SSH bootstrap to the peers. --job master is required:
# without it the node installs as solo and the peer list is used for ssh-copy-id only.
AINODE_PEERS="10.0.0.2,10.0.0.3" curl -fsSL https://ainode.dev/install | bash -s -- --job master

# Peers:
curl -fsSL https://ainode.dev/install | bash -s -- --job worker

# Dev (inside repo):
pip install -e ".[dev]"              # tests + ruff
docker build -f scripts/Dockerfile.ainode -t ainode:dev .
scripts/install.sh --dry-run         # render config/unit/wrapper, touch nothing else
systemctl status ainode              # after install
pytest tests/                        # unit tests
```

## Architecture

One AINode container per node, which orchestrates. The engine is a SEPARATE
container per loaded model, so an engine version is a property of the model's recipe
rather than of the AINode release.

```
ainode/
├── core/          # Config, GPU detection. DEFAULT_ENGINE_BACKEND lives here.
├── engine/
│   ├── backends/nvidia.py # THE engine path: one vLLM container per instance,
│   │                      #   solo and distributed (see engine/AGENTS.md)
│   ├── backends/eugr.py   # Opt-in: needs a `vllm` on PATH + eugr's launcher
│   ├── sharding_routes.py # /api/sharding/{plan,launch,status}: the node picker
│   ├── docker_engine.py   # Deprecation shim, removed in v0.6.0
│   └── vllm_engine.py     # Legacy host-venv path; retained for dev only.
├── api/           # Federated /v1 proxy + routes + /v1/decide + server console
├── web/           # Embedded chat UI (served by aiohttp)
├── models/        # Catalog (registry.py), load/unload, download manager
├── discovery/     # UDP node discovery + cluster state
├── cluster/       # Fabric interface detection (netdev)
├── embeddings/    # /v1/embeddings, routed to a pooling engine on the fleet
├── bench/         # The measurement suite + /api/bench
├── metrics/       # Prometheus + JSON metrics
├── auth/          # API-key auth
├── secrets/       # Local secrets store (HF read/write, NGC, W&B, OpenAI)
├── datasets/      # Training datasets
├── cli/           # `ainode start`, `ainode service ...`, etc.
├── service/       # systemd unit renderer (ExecStart = docker run ...)
├── onboarding/    # First-run setup (installer presets onboarded=true)
└── training/      # Fine-tuning (LoRA / QLoRA / full, single node)

scripts/
├── Dockerfile.ainode         # FROM python:3.12-slim + pip install /src
├── Dockerfile.quant          # The training/quantization job image (~22 GB), published
│                             #   by publish-train-image.yml as ainode-train:<version>
├── Dockerfile.whisper        # The speech engine image: the default engine build plus
│                             #   vLLM's audio extras, published by
│                             #   publish-whisper-image.yml as ainode-whisper:<base tag>
├── docker-entrypoint.sh      # exec ainode start --in-container
├── install.sh                # End-user installer (--dry-run renders and stops)
└── build-base-image.sh       # eugr base; not an input to Dockerfile.ainode
```

Multi-node: `POST /api/sharding/launch` (the launch panel's node toggles) or
`distributed_mode: "head"` plus `peer_ips`. The nvidia backend SSHes the peers
itself and starts one engine container per node. vLLM's own multi-node executor
(`distributed_executor: "mp"`) is the shape every proven launch used and the shape
every multi-node catalog recipe pins; the `"ray"` shape needs a `ray` CLI inside the
engine image and no image we ship or pin has one. eugr's `launch-cluster.sh` is not
in the image and is not called. Only tensor parallel exists: a `strategy` naming
pipeline parallel is accepted and ignored.

## Working Conventions

- Follow ops-approved workflow (see ops/)
- All work on `fable/*` branches (was `codex/*` until 2026-08-15)
- PRs required, and never push directly to main
- Handoffs use the threadmaster-handoff runbook
- Test on real GPU hardware when possible

## Target Hardware

Proven, with bench records: NVIDIA DGX Spark (GB10, 128 GB unified memory), ASUS GX10,
and one Dell C4130 with a Tesla V100 32 GB through a catalog recipe that pins a
Volta-capable vLLM build.

Design target, not yet run: other GB10 variants (Dell Pro Max, HP ZGX Nano) and any
Linux system with an NVIDIA GPU and CUDA. The default engine image is a GB10 aarch64
build, so a new architecture may need its own `engine_image` in its recipe, the way
Volta did.

## Performance Design Point (GB10 / DGX Spark)

GB10 decode is **memory-bandwidth bound** (273 GB/s LPDDR5x per node), not
compute bound. Single-stream decode reads the active weights once per token,
so the ceiling is `bandwidth ÷ bytes-read-per-token`. Every number below is a
measurement with a record under `bench/results/`; nothing here is an estimate, and a
figure that cannot cite a record does not belong in this file.

- **Dense is bandwidth-limited, MoE is not.** Measured on one GB10: dense Qwen3.8 27B
  NVFP4 at 19.0 tok/s single-stream, against MoE Nemotron 3.5 Lightning 30B-A3B
  (3B active per token) at 104.5 on a GX10 and Ornith 1.5 35B-A3B at 40.0 stacked.
  Same class of total parameters, five times the decode rate, because an MoE reads
  only its active params.
- **Multi-node does NOT improve single-stream latency.** It adds two all-reduces per
  layer over the fabric, and the comms tax eats the per-node bandwidth gain. A model
  that fits on one node should run TP=1. What the cluster buys is capacity: a model
  whose weights do not fit one node at all.
- **The cluster's measured multi-node points** are TP=2 on two GB10s: DeepSeek V4
  Flash at 34.5 tok/s single-stream and 94.0 across 16 streams, and
  Qwen3.8-Flash-Next at 26.2 and 207.6. At TP=4, Qwen3-235B-A22B NVFP4 at 16.5 tok/s
  single-stream, measured June 2026 on 0.4.x.
- **The cluster pays off on:** frontier MoE that cannot fit one node, batched
  multi-user throughput, and fine-tuning. Not single-stream chat.
- **No dense 70B or 405B number exists.** Neither has been served here. The catalog
  entries say so rather than carrying an estimate, and so should we.

**Distributed vLLM flags**, emitted by the engine because users never hand-edit vLLM
commands (the invariants live in `ainode/engine/AGENTS.md`):
`--tensor-parallel-size <N>`, `--kv-cache-dtype fp8` (required for long context, and
downgraded to `auto` on a vision model unless a caller asked for fp8 explicitly), the
`mp` executor's `--master-addr` / `--master-port` / `--node-rank` / `--headless`
rendezvous, and RoCE/RDMA NCCL env computed per node from sysfs. `--enforce-eager` is
NOT universal: AINode forces it only on the pinned 0.17 image, where CUDA-graph
capture crashed on GB10, and it costs throughput on 0.27.1. `gpu_memory_utilization`
defaults to 0.5 in code and 0.6 from the installer, and it is a per-load knob, not a
constant.

## Brand

- "Made in Texas" (with the small Texas mark) in all CLI output and the web UI footer. "Powered by argentos.ai" was retired 2026-09-13 at Jason's direction, and so is any argentos Discord link.
- No em dashes, in code, comments, docs or UI copy.
- Product name: AINode (capital A, capital I, capital N)
