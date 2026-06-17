# engine/ — AGENTS.md (edit contract)

Parent: `../../AGENTS.md` · State / "why" / history: Obsidian Vault → `Titanium Lab`. Working-state runbook: `ops/runbooks/2026-06-17-235b-moe-tp4-working-state.md`.

## Edit contract — DISTRIBUTED LAUNCH (dangerous, read before any change)

- **Launch distributed serves via the systemd path, NOT the dashboard LAUNCH button / `POST /api/sharding/launch`** (`sharding_routes.py`). That path auto-discovers peers from mgmt-LAN UDP source IPs (`192.168.0.x`), lands a Ray worker on a non-GPU address, and dies with `RuntimeError: current platform does not support ray` (and pollutes Ray with mgmt-IP nodes). Use `config.json` `distributed_mode="head"` + explicit fabric `peer_ips` + `systemctl restart ainode`.
- **`peer_ips` are the fabric (`10.100.0.x`)** — never mgmt LAN or Tailscale addresses.
- **vLLM flags are emitted by the backend, not hand-edited per run.** Change them in `backends/`, not by asking a user to edit a command.

## vLLM flag invariants (GB10 / Blackwell ARM)

- **Pin `VLLM_ATTENTION_BACKEND=TRITON_ATTN`.** FlashInfer (vLLM's auto-pick on Blackwell) ships a prebuilt attention kernel that emits an `illegal instruction` on GB10 (sm120) and kills EngineCore on the **first prefill** (`BatchPrefillWithPagedKVCache`). The engine loads + reports READY, then suicides on the first real request — `/v1/models` 200 is NOT proof of a working engine. `FLASH_ATTN` is unavailable (no `flash_attn` in the image); Triton JIT-compiles to the live arch. NvidiaBackend sets this in `_build_nccl_env`; vLLM forwards `VLLM_*` to Ray workers. Env-overridable for a future image that fixes FlashInfer.
- **Keep `--enforce-eager`.** It's the Blackwell/ARM stability flag. Removing it (to capture CUDA graphs) is a deliberate, revertible perf experiment — never a silent default.
- **Never add `--enable-expert-parallel`** — it hangs on this MoE/hardware.
- **`--gpu-memory-utilization` target `0.85`** (config default is `0.9`).
- `--kv-cache-dtype fp8` is required for long context (32k+) or it OOMs.

## Don't kill a slow launch

A multi-minute MoE profiling forward-pass with GPUs at 0% and quiet logs is **not** a hang — do not SIGTERM it. (A premature "hung" call cost a whole session; see the runbook.) Wait 3–5 min for `:8000` to bind.

## Verification

- After engine/flag changes: `pytest tests/` and confirm the resolved vLLM command in the launch logs matches the invariants above. Don't claim a serve works unless you saw it reach READY and generate tokens.
