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
- **The harness bench (`ainode/bench/harness/`, `scripts/ainode-bench.py harness`) must never let a harness see the hidden tests.** `bench/harness/tasks/<slug>/tests/` is copied into the working directory only after the agent CLI has exited and removed again before the next attempt; a task with a `*_test.py` at its root is a load error. Those vendored files are also excluded from pytest collection (`norecursedirs` in `pyproject.toml`) and from ruff (`extend-exclude`) because they import a module that only exists inside a run, and because they are upstream's text kept verbatim. Adapter `command()` / `env()` / `config()` stay pure functions of the request so `tests/test_bench_harness.py` can pin every harness's exact argv; `run()` lives once in the base class. **An adapter for an agent that keeps state under `$HOME` points it at the run's own directory** (`DSH_HOME`, the XDG vars, `CLAUDE_CONFIG_DIR`): a bench run never reads or writes the operator's own agent profile, both so runs cannot poison each other and so a personal `settings.json` full of hooks is not inside the measurement. Adapters and flags: `bench/harness/README.md`.
- **One proxy handler serves every forwarded inference path**: `proxy_to_vllm` is registered for `POST /v1/chat/completions`, `/v1/completions`, `/v1/messages` and `/v1/messages/count_tokens` (`GET /v1/models` is the federated union and does not forward). Add a path by registering it on that handler, never by writing a second proxy: routing on the body's `model`, transport failover, the multimodal ordering below, SSE passthrough and header passthrough are all protocol-agnostic and already there. Forward `request.path_qs`, not `request.path`: Claude Code posts to `/v1/messages?beta=true`. This is a route table and not a catch-all: an unregistered path under `/v1` stays a 404.
- **A multimodal chat request consults the capability cache before routing** (`api/server.py::proxy_to_vllm`). A body carrying an `image_url` / `input_audio` / `video_url` / `file` part, or the Anthropic Messages spelling (an `image` / `document` block, including one nested in a `tool_result`), is never routed on the model id alone: order candidates accepting-first (cached `vision: true`), then never-probed, and drop instances cached `vision: false`; vLLM's `may be provided in one prompt` 400 is a routing miss, so record `vision: false` and fail over, while every other 4xx goes back to the caller untouched. A request with no media keeps the plain order (local hop first, then peers). There is ONE capability cache: `app["chat_caps_cache"]`, filled by `/api/models/caps` in `api/chat_routes.py`, which probes remote instances directly on their engine port. Never add a second.
- Handoffs use the threadmaster-handoff runbook; ops state lives in `ops/` (runbooks under `ops/runbooks/`).
- Distribution is `docker pull` only — end users never hand-edit vLLM commands; the engine emits flags (see `engine/AGENTS.md`).

## Child DOX Index

Read the nearest child before editing in its subtree. Add a child only at folders with non-obvious/dangerous constraints — not one per folder (see the charter).

| Path | Owns |
|------|------|
| `ainode/engine/AGENTS.md` | Distributed launch path + vLLM flag invariants (GB10/Blackwell footguns) |
