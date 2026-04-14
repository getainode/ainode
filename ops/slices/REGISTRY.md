# Slice Registry — AINode Product

## Active Slices

| Slice | Owner | Branch | Status | Linear |
|-------|-------|--------|--------|--------|
| engine-vllm | Threadmaster | codex/engine-vllm | MERGED TO DEV | — |
| api-proxy | Agent (pane 1) | codex/api-proxy | IN PROGRESS | — |
| cli-polish | Agent (pane 2) | codex/cli-polish | IN PROGRESS | — |
| discovery-cluster | Agent (pane 3) | codex/discovery-cluster | IN PROGRESS | — |

### docker-engine — scope
Full scripted Docker-based vLLM for GB10 nodes. Runbook: `ops/runbooks/docker-engine-deploy.md`.
- `ainode/engine/docker_engine.py` (new) — same interface as VLLMEngine, drives `docker compose`
- `scripts/install.sh` — writes compose file, .env, systemd user unit, auto-enables
- `ainode/cli/main.py` — cmd_start picks engine by `config.engine_strategy`
- `ainode/core/config.py` — add `engine_strategy: "docker" | "pip"` field
- systemd user unit at `~/.config/systemd/user/ainode.service`, linger enabled
- Acceptance: one-liner install on Spark 2 brings it to cluster with zero manual steps

## Completed Slices

| Slice | Owner | Merged | Date |
|-------|-------|--------|------|
| initial-scaffold | Threadmaster | main | 2026-04-12 |
| docker-engine | Claude (Opus 4.6) | dev | 2026-04-14 |

## Available Slices (Priority Order)

### P0 — MVP (must ship for v0.1)
- `engine-vllm` — vLLM engine: health checks, readiness wait, log streaming, graceful shutdown
- `api-proxy` — API proxy: sits in front of vLLM, adds /status, /nodes, model catalog
- `web-ui` — Embedded chat UI: simple HTML/JS, served by ainode, no external deps

### P1 — Cluster + Polish
- `cli-polish` — Rich terminal output, progress bars, spinners, colored status
- `discovery-cluster` — Multi-node: form cluster from discovered nodes, route requests, shard models
- `onboarding-web` — Browser-based onboarding instead of terminal prompts

### P2 — Differentiators
- `training-ui` — Fine-tuning from browser: dataset upload, config, launch, monitor
- `training-engine` — Backend: vLLM/PyTorch training pipeline, LoRA, progress streaming
- `model-manager` — Download, delete, organize models from UI
- `metrics` — Prometheus endpoint for GPU, memory, request stats

### P3 — Production
- `installer-test` — Test install script on real DGX Spark hardware
- `systemd-service` — Auto-start on boot, service management
- `auth` — Optional API key auth, user accounts
