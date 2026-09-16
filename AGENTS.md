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

- All work on `fable/*` branches (renamed from `codex/*` 2026-08-15 — the old prefix came from OpenAI Codex; no CI keys on either, so existing `codex/*` branches are fine to leave). PRs required — **never push directly to `main`**.
- Build/test: `pip install -e ".[dev]"` → `pytest tests/` · lint `ruff check`. Base image: `scripts/build-base-image.sh`; app image: `docker build -f scripts/Dockerfile.ainode`.
- **Do not delete `tests/conftest.py`'s `isolate_netdev` fixture.** `ainode.cluster.netdev` reads the real host and caches per process, so without it any test pinning `cluster_interface=` to a Spark NIC name passes on a Mac and fails on the Linux CI runner (autodetect resolves it to `eth0`). A test that wants detection to fire monkeypatches `netdev.SYS_CLASS_NET` and `netdev._run_command` itself.
- Measured numbers live **only** in `bench/results/*.json` (format: `bench/SCHEMA.md`; never fill a missing measurement with an estimate). The README's "Models tested on AINode" table is generated: add a result file, then `python3 scripts/render-bench-table.py`. Never hand-edit between the `bench-table` markers; `tests/test_bench_table.py` fails on drift.
- The benchmark itself is `ainode/bench/` (stdlib-only measurement + the `/api/bench` routes + the report renderer). `scripts/ainode-bench.py` and `bench/report.py` are thin CLI shims over it: change the package, not the shims. In-product runs land in `~/.ainode/bench/results/`; copy one into `bench/results/` by hand to publish it.
- **A multimodal chat request consults the capability cache before routing** (`api/server.py::proxy_to_vllm`). A body carrying an `image_url` / `input_audio` / `video_url` / `file` part is never routed on the model id alone: order candidates accepting-first (cached `vision: true`), then never-probed, and drop instances cached `vision: false`; vLLM's `may be provided in one prompt` 400 is a routing miss, so record `vision: false` and fail over, while every other 4xx goes back to the caller untouched. A request with no media keeps the plain order (local hop first, then peers). There is ONE capability cache: `app["chat_caps_cache"]`, filled by `/api/models/caps` in `api/chat_routes.py`, which probes remote instances directly on their engine port. Never add a second.
- Handoffs use the threadmaster-handoff runbook; ops state lives in `ops/` (runbooks under `ops/runbooks/`).
- Distribution is `docker pull` only — end users never hand-edit vLLM commands; the engine emits flags (see `engine/AGENTS.md`).

## Child DOX Index

Read the nearest child before editing in its subtree. Add a child only at folders with non-obvious/dangerous constraints — not one per folder (see the charter).

| Path | Owns |
|------|------|
| `ainode/engine/AGENTS.md` | Distributed launch path + vLLM flag invariants (GB10/Blackwell footguns) |
