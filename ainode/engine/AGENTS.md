# engine/ — AGENTS.md (edit contract)

Parent: `../../AGENTS.md` · State / "why" / history: Obsidian Vault → `Titanium Lab`. Working-state runbook: `ops/runbooks/2026-06-17-235b-moe-tp4-working-state.md`.

## Edit contract — DISTRIBUTED LAUNCH (dangerous, read before any change)

- **Launch distributed serves via the systemd path, NOT the dashboard LAUNCH button / `POST /api/sharding/launch`** (`sharding_routes.py`). That path auto-discovers peers from mgmt-LAN UDP source IPs (`192.168.0.x`), lands a Ray worker on a non-GPU address, and dies with `RuntimeError: current platform does not support ray` (and pollutes Ray with mgmt-IP nodes). Use `config.json` `distributed_mode="head"` + explicit fabric `peer_ips` + `systemctl restart ainode`.
- **`peer_ips` are the fabric (`10.100.0.x`)** — never mgmt LAN or Tailscale addresses.
- **Never read `config.cluster_interface` directly. Call `ainode.cluster.netdev.resolve_cluster_interface(config)`.** The configured name is hardware-specific (`enP2p1s0f1np1` on a Spark, `enp1s0f0np0` on a GX10) and the default is now empty, meaning autodetect; a direct read binds NCCL/Ray/Gloo/UCX to a device that may not exist on this host and fails opaquely (#34, #61). Same for any new code that needs the fabric netdev. When an interface has no IPv4, surface `netdev.interface_candidates_hint()` in the error so the user can fix `config.json` without a second tool.
- **vLLM flags are emitted by the backend, not hand-edited per run.** Change them in `backends/`, not by asking a user to edit a command.

## vLLM flag invariants (GB10 / Blackwell ARM)

- **Keep `--enforce-eager` — this is the GB10/sm120 fix, not a perf knob.** FlashInfer (vLLM's auto-pick on Blackwell) crashes its prefill kernel (`BatchPrefillWithPagedKVCache`, `illegal instruction`) **under CUDA-graph capture** on GB10 (sm120), killing EngineCore on the **first real prefill**. The engine loads + reports READY, then suicides (vLLM SIGTERMs its own Ray workers) — `/v1/models` 200 is NOT proof of a working engine; verify with a **long-prompt generation**, not the readiness endpoint. `--enforce-eager` disables graph capture and the same kernel runs clean. Re-enabling graphs (for throughput) needs a working non-FlashInfer backend first.
- **`VLLM_ATTENTION_BACKEND=TRITON_ATTN` is set but currently a NO-OP** — this scitrera/vLLM 0.17.1 build ignores it (ranks still log `Using FLASHINFER`). Kept as an env-overridable hedge (correct value may be `TRITON_ATTN_VLLM_V1`); do not rely on it — `--enforce-eager` is what's actually preventing the crash.
- **Never add `--enable-expert-parallel`** — it hangs on this MoE/hardware.
- **`--gpu-memory-utilization` target `0.85`** (config default is `0.9`).
- `--kv-cache-dtype fp8` is required for long context (32k+) or it OOMs.

## Don't kill a slow launch

A multi-minute MoE profiling forward-pass with GPUs at 0% and quiet logs is **not** a hang — do not SIGTERM it. (A premature "hung" call cost a whole session; see the runbook.) Time-to-bind belongs to the model and the engine image, not to us: on `vllm/vllm-openai:v0.27.1` a 27B NVFP4 model measured ~12 min to bind on a GB10 (FlashInfer fp4_gemm autotune plus CUDA graph capture), a 35B-A3B ~6 min.

- **No fixed bind timeout in code.** The startup replay (`models/api_routes.py::_wait_for_bind`) waits while the container is up and the engine's log is advancing, and gives up only on evidence: container exited, log silent past `NodeConfig.engine_bind_log_silence_seconds`, or `NodeConfig.engine_bind_ceiling_seconds` reached. Do not reintroduce a constant window; tune the knobs.
- **A backend's `last_log_activity` must come from that engine's own stdout stream** (`_stream_logs`), never from the log file's mtime or size: stacked instances on a node share one log file, so file-based freshness lets a busy primary vouch for a wedged neighbour. A backend that cannot report returns `None`, and the wait then leans on container exit plus the ceiling.

## Verification

- After engine/flag changes: `pytest tests/` and confirm the resolved vLLM command in the launch logs matches the invariants above. Don't claim a serve works unless you saw it reach READY and generate tokens.
