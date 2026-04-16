# Slice Registry — AINode Product

## Active Slices

| Slice | Owner | Branch | Status | Linear |
|-------|-------|--------|--------|--------|
| training-artifacts | Threadmaster | codex/training-artifacts | MERGED TO MAIN | — |

## Completed Slices

| Slice | Owner | Merged | Date |
|-------|-------|--------|------|
| initial-scaffold | Threadmaster | main | 2026-04-12 |
| docker-engine | Claude (Opus 4.6) | main | 2026-04-14 |
| engine-vllm | Threadmaster | main | 2026-04-14 |
| api-proxy | Threadmaster | main | 2026-04-14 |
| cli-polish | Threadmaster | main | 2026-04-14 |
| discovery-cluster | Threadmaster | main | 2026-04-14 |
| cluster-member-mode | Threadmaster | main | 2026-04-14 |
| docker-image-distribution | Threadmaster | main | 2026-04-15 |
| v0.4.0-release | Threadmaster | main | 2026-04-15 |
| v0.4.1-release | Threadmaster | main | 2026-04-15 |
| v0.4.2-release | Threadmaster | main | 2026-04-15 |

### training-artifacts — scope (active)
Close training module gaps: artifact retrieval, LoRA merge, checkpoint resume,
HF token propagation, OOM error handling, DDP validation, eval loop, W&B support,
custom template persistence.

**Files changed:**
- `ainode/training/api_routes.py` — 6 new endpoints
- `ainode/training/engine.py` — new config fields (hf_token, eval_split, eval_steps, wandb_project)
- `ainode/training/_run_training.py` — HF token, DDP validation, OOM handling, eval loop, W&B, checkpoint resume
- `tests/test_training.py` — 10 new test cases (artifacts, HF token propagation)
- `docs/mintlify-docs/ainode/training.mdx` — new public doc page

**Acceptance criteria:**
- `pytest tests/test_training.py` — all pass ✅
- Training job output accessible via API ✅
- HF token flows from NodeConfig → job config ✅
- OOM error produces actionable message ✅
- eval_split creates validation split ✅

## Available Slices (Priority Order)

### P1 — Training smoke tests
- `training-smoke-gpu` — Run all 4 training methods on real GB10 hardware and verify outputs
- `training-benchmarks-ui` — Replace benchmark tab stub with real NCCL/RDMA/storage benchmarks

### P2 — Polish
- `model-manager-enhancements` — Model tagging, sorting by disk size, bulk delete
- `auth-api-keys` — Per-user API keys for multi-tenant deployments
