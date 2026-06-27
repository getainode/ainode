# AGENTS.md — AINode

Turn any NVIDIA GPU into a local AI platform (inference + fine-tuning in the browser); ships as one container image per node — web UI, API, vLLM engine, and cross-node orchestrator are version-locked together.
State / architecture / decisions / "why": Obsidian Vault → `AINode` (cluster ops: `Titanium Lab`). Claude-specific config: `CLAUDE.md`.

## DOX — Read Before Editing

> Boundary (canonical): Obsidian Vault → `Systems/Claude Code Harness/DOX — Ownership Charter & Pilot`.

- Before editing any path, walk this repo's `AGENTS.md` chain from root to the target folder and obey the **nearest** one as the local edit contract. Re-read in-session — don't trust memory.
- After a change that alters an **edit contract** (local rules, commands, invariants, structure), update the affected `AGENTS.md`. Do **not** touch docs for ordinary code changes.
- One home per fact: **edit-rule/command → `AGENTS.md` · Claude-only config → `CLAUDE.md` · state/decisions/architecture/why → Obsidian Vault.** Link across layers, never copy.
- This file is imperative + operational only. Anything describing *why/history/state* belongs in the Vault.

## Operational source of truth

- All work on `codex/*` branches; PRs required — **never push directly to `main`**.
- Build/test: `pip install -e ".[dev]"` → `pytest tests/` · lint `ruff check`. Base image: `scripts/build-base-image.sh`; app image: `docker build -f scripts/Dockerfile.ainode`.
- Handoffs use the threadmaster-handoff runbook; ops state lives in `ops/` (runbooks under `ops/runbooks/`).
- Distribution is `docker pull` only — end users never hand-edit vLLM commands; the engine emits flags (see `engine/AGENTS.md`).

## Child DOX Index

Read the nearest child before editing in its subtree. Add a child only at folders with non-obvious/dangerous constraints — not one per folder (see the charter).

| Path | Owns |
|------|------|
| `ainode/engine/AGENTS.md` | Distributed launch path + vLLM flag invariants (GB10/Blackwell footguns) |
