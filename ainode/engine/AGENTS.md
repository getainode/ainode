# engine/ — AGENTS.md (edit contract)

Parent: `../../AGENTS.md` · State / "why" / history: Obsidian Vault → `Titanium Lab`. Working-state runbook: `ops/runbooks/2026-06-17-235b-moe-tp4-working-state.md`.

## Edit contract — DISTRIBUTED LAUNCH (dangerous, read before any change)

- **Launch distributed serves via the systemd path, NOT the dashboard LAUNCH button / `POST /api/sharding/launch`** (`sharding_routes.py`). That path auto-discovers peers from mgmt-LAN UDP source IPs (`192.168.0.x`), lands a Ray worker on a non-GPU address, and dies with `RuntimeError: current platform does not support ray` (and pollutes Ray with mgmt-IP nodes). Use `config.json` `distributed_mode="head"` + explicit fabric `peer_ips` + `systemctl restart ainode`.
- **`peer_ips` are the fabric (`10.100.0.x`)** — never mgmt LAN or Tailscale addresses.
- **Never read `config.cluster_interface` directly. Call `ainode.cluster.netdev.resolve_cluster_interface(config)`.** The configured name is hardware-specific (`enP2p1s0f1np1` on a Spark, `enp1s0f0np0` on a GX10) and the default is now empty, meaning autodetect; a direct read binds NCCL/Ray/Gloo/UCX to a device that may not exist on this host and fails opaquely (#34, #61). Same for any new code that needs the fabric netdev. When an interface has no IPv4, surface `netdev.interface_candidates_hint()` in the error so the user can fix `config.json` without a second tool.
- **vLLM flags are emitted by the backend, not hand-edited per run.** Change them in `backends/`, not by asking a user to edit a command.

## Two distributed shapes: pick by what the engine image ships

`NodeConfig.distributed_executor` selects the shape per instance (a catalog
recipe or a `/api/sharding/launch` body can set it). Both live in
`backends/nvidia.py::start_distributed`; both use the same fabric-IP detection,
the same `_build_vllm_serve_args`, the same peer container names and the same
`stop()`.

- **`"ray"` (default).** `ray start --head` container here, an SSH-launched
  `ray start` worker container per peer, then `docker exec` into the head to run
  `vllm serve --distributed-executor-backend ray`. **Only usable when the engine
  image ships the `ray` CLI.** Stock `vllm/vllm-openai` does not: the head
  container exits 127. The head must reach Running before the peers are
  SSH-launched, or workers race an unbound `:6379`.
- **`"mp"`.** One `vllm serve` container per node, rank 0 here and
  `--node-rank k --headless` on each peer, all rendezvousing on
  `--master-addr`/`--master-port` via vLLM's own multi-node executor. Needs
  nothing in the image beyond vLLM, so **this is the shape for a custom engine
  image** (pin it in the catalog entry next to `engine_image`).
  - **Launch peers BEFORE the head, and do not wait on the head reaching
    Running first.** The rendezvous does the waiting; the Ray ordering rule does
    not apply here.
  - The head container **is** the server: there is no `docker exec`'d process.
    `is_running` falls back to the head container's docker state, and the engine
    log comes from `docker logs -f` of that container (same
    `nvidia-distributed.log`, so `last_log_activity` still feeds the adaptive
    bind wait). Never swap that for the log file's mtime.
  - Container flags are `--network host --ipc host --shm-size 64g --ulimit
    memlock=-1 --ulimit stack=67108864 --gpus all`, plus
    `--device /dev/infiniband:/dev/infiniband` **only when the host has that
    path** (`_infiniband_present`). A host without it must still launch.
  - Every rank serves the model by REPO-ID from the mounted HF cache, never a
    local mount path: the identifier has to resolve identically on all ranks,
    and a peer only gets its HF cache (filled by `_ensure_peer_has_model`).
  - A recipe that states `--nnodes`, `--node-rank`, `--master-addr`,
    `--master-port` or `--headless` itself suppresses ours, same rule as every
    other serve flag.

An engine image that is not a `vllm/vllm-openai` image may need `PATH`, the CUDA
paths and `HF_HOME` set through `extra_env`: HF defaults its cache to `$HOME`,
which is not where AINode mounts it. Point a recipe's writable caches inside the
HF cache mount (`ainode.core.config.HF_CACHE_MOUNT`) rather than inventing a
host path; `NodeConfig.extra_volumes` exists for an operator who needs one.

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
