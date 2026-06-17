# CLAUDE.md — AINode Product

> **DOX:** Before editing, walk this repo's `AGENTS.md` chain (root → target folder) and obey the nearest one as the local edit contract. This file holds **Claude-specific** config only and must not restate the `AGENTS.md` edit rules or the Vault's state/"why". Boundary: Obsidian Vault → `Systems/Claude Code Harness/DOX — Ownership Charter & Pilot`.

## Project Overview

AINode — Turn any NVIDIA GPU into a local AI platform. Inference + fine-tuning in your browser. One command to start, automatic clustering.

- **Product repo:** https://github.com/getainode/ainode
- **Marketing site repo:** https://github.com/getainode/ainode.dev
- **Live site:** https://ainode.dev
- **Powered by:** argentos.ai
- **Public docs:** https://docs.ainode.dev (repo: /tmp/ainode-docs → github.com/getainode/ainode-docs)
- **License:** Apache 2.0

## Tech Stack

- Python 3.10+ (shipped inside the container image)
- vLLM (inference engine) via eugr/spark-vllm-docker base
- Ray (cross-node orchestration, via eugr's launch-cluster.sh)
- NCCL (patched `dgxspark-3node-ring`) for cross-node all-reduce
- aiohttp (API server + web UI serving)
- pynvml + psutil (GPU detection)
- Rich (terminal UI)

## Distribution

AINode ships as a single container image: `ghcr.io/getainode/ainode:<version>`
(mirrored on Docker Hub as `argentos/ainode`). End users only ever
`docker pull` — no host venv, no vLLM source build. Our CI builds on a
self-hosted aarch64 runner (a Spark) via `.github/workflows/publish-image.yml`.

## Key Commands

```bash
# End-user install (one node):
curl -fsSL https://ainode.dev/install | bash

# Distributed (head + peers, SSH bootstrap):
AINODE_PEERS="10.0.0.2,10.0.0.3" curl -fsSL https://ainode.dev/install | bash

# Dev (inside repo):
pip install -e ".[dev]"              # tests + ruff
scripts/build-base-image.sh          # build eugr base locally
docker build -f scripts/Dockerfile.ainode -t ainode:dev .
systemctl status ainode              # after install
pytest tests/                        # unit tests
```

## Architecture

One container per node — web UI, API, vLLM engine, and cross-node
orchestrator are version-locked in a single image.

```
ainode/
├── core/          # Config, GPU detection
├── engine/
│   ├── docker_engine.py   # Solo: direct vllm serve. Distributed: shell-out to eugr's launch-cluster.sh.
│   └── vllm_engine.py     # Legacy host-venv path; retained for dev only.
├── api/           # API proxy (OpenAI-compatible) + aiohttp routes
├── web/           # Embedded chat UI (served by aiohttp)
├── discovery/     # UDP node discovery (port 5679)
├── cli/           # `ainode start`, `ainode service ...`, etc.
├── service/       # systemd unit renderer (ExecStart = docker run ...)
├── onboarding/    # First-run setup
└── training/      # Fine-tuning (LoRA / DDP)

scripts/
├── Dockerfile.ainode         # FROM ainode-base + pip install ainode
├── build-base-image.sh       # Clone eugr @ pinned SHA, build base image
├── docker-entrypoint.sh      # exec ainode start --in-container
├── install.sh                # End-user installer (~80 lines)
└── _eugr/                    # Shallow checkout of eugr/spark-vllm-docker at pinned SHA
```

Distributed mode (`config.distributed_mode == "head"`) calls
`/opt/spark-vllm-docker/launch-cluster.sh` inside the container — eugr's
launcher handles SSH to peers, Ray head/worker formation, and NCCL.

## Working Conventions

- Follow ops-approved workflow (see ops/)
- All work on `codex/*` branches
- PRs required — never push directly to main
- Handoffs use the threadmaster-handoff runbook
- Test on real GPU hardware when possible

## Target Hardware

- NVIDIA DGX Spark (GB10, 128 GB unified memory)
- ASUS/Dell/HP GB10 variants
- Any Linux system with NVIDIA GPU + CUDA

## Performance Design Point (GB10 / DGX Spark)

GB10 decode is **memory-bandwidth bound** (273 GB/s LPDDR5x per node), not
compute bound. Single-stream decode reads the active weights once per token,
so the ceiling is `bandwidth ÷ bytes-read-per-token`. Implications that should
shape engine defaults and user expectations:

- **Dense models are bandwidth-limited.** Dense 70B (NVFP4 ~35 GB) caps near
  ~7-8 t/s single-stream on one node; dense 405B is ~1 t/s. Multi-node tensor
  parallelism does **not** improve single-stream latency — it adds 2
  all-reduces/layer over the fabric, and the comms tax eats the per-node
  bandwidth gain. A model that fits on one node should run TP=1.
- **MoE is the design point for distributed serving.** A Mixture-of-Experts
  model (e.g. Qwen3.5-397B-**A17B**) needs the cluster's pooled memory for its
  total params but only reads its *active* params per token, so decode stays
  fast (~16-17 t/s single-stream observed at TP=4 on a 4-node RoCE cluster).
  MoE decouples capacity from decode cost — the right match for this hardware.
- **The cluster pays off on:** frontier MoE that can't fit one node, batched
  multi-user throughput, and fine-tuning/training. Not single-stream chat.

**Proven distributed vLLM flags** for frontier MoE on this hardware (the engine
should emit these automatically — users never hand-edit vLLM commands):
`--tensor-parallel-size <N>`, `--kv-cache-dtype fp8` (required for long context),
`--enforce-eager` (Blackwell/ARM stability), `--gpu-memory-utilization 0.85`,
RoCE/RDMA NCCL env per-node (see `scripts/nccl-env-init.sh`).

## Brand

- "Powered by argentos.ai" in all CLI output and web UI footer
- Product name: AINode (capital A, capital I, capital N)
