# engine/ — AGENTS.md (edit contract)

Parent: `../../AGENTS.md` · State / "why" / history: Obsidian Vault → `Titanium Lab`. Working-state runbook: `ops/runbooks/2026-06-17-235b-moe-tp4-working-state.md`.

## Edit contract — DISTRIBUTED LAUNCH (dangerous, read before any change)

- **Launch distributed serves via the systemd path, NOT the dashboard LAUNCH button / `POST /api/sharding/launch`** (`sharding_routes.py`). That path auto-discovers peers from mgmt-LAN UDP source IPs (`192.168.0.x`), lands a Ray worker on a non-GPU address, and dies with `RuntimeError: current platform does not support ray` (and pollutes Ray with mgmt-IP nodes). Use `config.json` `distributed_mode="head"` + explicit fabric `peer_ips` + `systemctl restart ainode`.
- **`peer_ips` are the fabric (`10.100.0.x`)** — never mgmt LAN or Tailscale addresses.
- **Never read `config.cluster_interface` directly. Call `ainode.cluster.netdev.resolve_cluster_interface(config)`.** The configured name is hardware-specific (`enP2p1s0f1np1` on a Spark, `enp1s0f0np0` on a GX10) and the default is now empty, meaning autodetect; a direct read binds NCCL/Ray/Gloo/UCX to a device that may not exist on this host and fails opaquely (#34, #61). Same for any new code that needs the fabric netdev. When an interface has no IPv4, surface `netdev.interface_candidates_hint()` in the error so the user can fix `config.json` without a second tool.
- **vLLM flags are emitted by the backend, not hand-edited per run.** Change them in `backends/`, not by asking a user to edit a command.

## Two distributed shapes: pick by what the engine image ships

`NodeConfig.distributed_executor` selects the shape per instance (a catalog
recipe or a `/api/sharding/launch` body can set it). It defaults to `"mp"`, and
every defensive fallback for a missing or empty value reads
`DEFAULT_DISTRIBUTED_EXECUTOR` rather than spelling a shape again: the default was
`"ray"` through 0.5.26 and no image AINode ships or launches has ray in it, so an
uncurated distributed launch died inside the container (#172). A catalog entry
that needs ray states `distributed_executor="ray"` and gets it. Both shapes live
in
`backends/nvidia.py::start_distributed`; both use the same fabric-IP detection,
the same `_build_vllm_serve_args`, the same peer container names and the same
`stop()`.

- **`"ray"`.** `ray start --head` container here, an SSH-launched
  `ray start` worker container per peer, then `docker exec` into the head to run
  `vllm serve --distributed-executor-backend ray`. **Only usable when the engine
  image ships the `ray` CLI.** Stock `vllm/vllm-openai` does not: the head
  container exits 127. The head must reach Running before the peers are
  SSH-launched, or workers race an unbound `:6379`.
- **`"mp"` (the default: `core/config.py::DEFAULT_DISTRIBUTED_EXECUTOR`).** One
  `vllm serve` container per node, rank 0 here and
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
  - **Every rank must resolve the SAME serve-target string**, and both shapes get
    it from one resolver (`_distributed_serve_target_and_name_args`). Two cases:
    - The model was downloaded THROUGH AINode, i.e. the flat
      `<models_dir>/<owner--name>` dir `POST /api/models/download-repo` writes
      (`_servable_local_model_dir`, same test solo uses, including the
      AINODE_HOST_HOME mount-trust gate). Then every rank serves
      `MODELS_MOUNT/<slug>` with `--served-model-name <repo id>`, and every rank
      mounts its OWN store there: the head's `config.models_dir`, a peer's
      `_peer_models_dir()` (`/home/<ssh_user>/ainode-nvidia-models`, chosen like
      `_peer_hf_cache`). `_ensure_peer_has_model` ships that flat dir over the
      fabric instead of a hub entry. Without this a UI-downloaded model made
      every node re-download it (133 GB per node on the Flash-Next launch).
    - No such copy: the REPO-ID, unchanged, and only the HF cache is mounted
      (vLLM pulls into each node's cache). No `--served-model-name` either.
    Derive the mount from the target actually chosen, never from a second probe,
    so a mount-path target can never be rendered without its mount.
  - **Weights are not the only thing the head ships.** Kernel warmup runs per
    rank, so a peer with a cold compiled-kernel cache JITs from source while the
    head (warm from an earlier launch) is already waiting for it inside a
    collective, and the CPU group's default 1800 s timeout declares the pair dead
    (#134). `_ensure_peer_has_jit_cache` therefore sends the head's
    `VLLM_CACHE_ROOT` subtree before the peer's container starts: one transfer per
    child of that root, each skipped when the peer already has it, and every
    failure a warning rather than a refused launch. It only fires for a recipe
    whose `VLLM_CACHE_ROOT` is INSIDE `HF_CACHE_MOUNT` (`registry._DSPARK_JIT_ROOT`),
    because that is the only way one container path maps to a per-node host dir.
    A recipe that leaves the cache roots unset does not just lose the seeding: its
    caches live in the container and die with it, so every launch re-JITs on every
    rank. State them.
  - **Do not expect a copied FlashInfer autotune cache to be read on a peer.** On
    this vLLM line only rank 0 reads
    `$VLLM_CACHE_ROOT/flashinfer_autotune_cache/<flashinfer workspace parent>/<flashinfer workspace>/<sha256 of vllm_config.compute_hash()>/autotune_configs.json`
    and broadcasts the bytes to every other rank
    (`model_executor/warmup/kernel_warmup.py::flashinfer_autotune`), which then
    write them to their own resolved path. The key is the whole `VllmConfig` hash
    (so any flag change orphans it) and is NOT rank-specific. Seeding a peer is
    belt-and-braces; the head's cache persisting is what actually matters, and the
    per-rank win is in the JIT'd kernels around it. When warmup is the problem,
    reach for `--no-enable-flashinfer-autotune` (the off form of
    `KernelConfig.enable_flashinfer_autotune`) and
    `--cpu-distributed-timeout-seconds` / `--distributed-timeout-seconds`, all
    three recipe-level, as the Flash-Next entry does.
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
- **`VLLM_ATTENTION_BACKEND=TRITON_ATTN` is emitted for the PINNED DEFAULT IMAGE ONLY** (`_attention_backend_env`, same gate as `_legacy_gb10_args`). On that 0.17.1 build it is a NO-OP (ranks still log `Using FLASHINFER`), kept as an env-overridable hedge (correct value may be `TRITON_ATTN_VLLM_V1`); do not rely on it, `--enforce-eager` is what's actually preventing the crash. **Never re-widen it to every image.** 0.27/0.28 log it as an unknown variable, but the 0.21-based GB10 fork honors it, and a dense attention backend over DeepSeek V4's sparse MLA path makes the serve talk nonsense. A custom image that wants one states it in the recipe's `extra_env`.
- **Never add `--enable-expert-parallel`** — it hangs on this MoE/hardware.
- **`--gpu-memory-utilization` target `0.85`** (config default is `0.9`).
- `--kv-cache-dtype fp8` is required for long context (32k+) or it OOMs.

## Don't kill a slow launch

A multi-minute MoE profiling forward-pass with GPUs at 0% and quiet logs is **not** a hang — do not SIGTERM it. (A premature "hung" call cost a whole session; see the runbook.) Time-to-bind belongs to the model and the engine image, not to us: on `vllm/vllm-openai:v0.27.1` a 27B NVFP4 model measured ~12 min to bind on a GB10 (FlashInfer fp4_gemm autotune plus CUDA graph capture), a 35B-A3B ~6 min.

- **No fixed bind timeout in code.** The startup replay (`models/api_routes.py::_wait_for_bind`) waits while the container is up and the engine is visibly doing something, and gives up only on evidence: container exited, BOTH liveness signals quiet past `NodeConfig.engine_bind_log_silence_seconds`, or `NodeConfig.engine_bind_ceiling_seconds` reached. Do not reintroduce a constant window; tune the knobs.
- **The engine's ACTIVITY is the primary liveness signal; the log is secondary** (#112). Either one advancing resets the silence clock, and the verdict needs both quiet, because vLLM prints nothing at all through weight load, torch.compile and FlashInfer autotune (measured quiet for 206 s, 363 s and once 48 minutes on healthy engines), which is what had pushed the silence budget to 900 s. With activity watched it is back to 300 s. Never widen that knob to cover a quiet phase again: fix the probe instead.
- **`EngineBackend.activity_mark()` reads the ENGINE'S OWN CONTAINER, never the host and never the whole GPU.** `NvidiaBackend` takes cumulative CPU time from that container's cgroup when the tree is visible and falls back to `docker stats --no-stream` (about a second, so the backend caches one answer per poll); the wait calls it off the event loop thread. Do not substitute pynvml utilization: `metrics/collector.py` reads device 0 as a whole, so it cannot say which engine is busy (a stacked neighbour would vouch for a wedged one) and it reads 0% during exactly the phases this probe exists for. A backend that cannot see its engine's container returns `None` and the wait falls back to the log alone.
- **A backend's `last_log_activity` must come from that engine's own stdout stream** (`_stream_logs`), never from the log file's mtime or size: stacked instances on a node share one log file, so file-based freshness lets a busy primary vouch for a wedged neighbour. A backend that cannot report returns `None`, and the wait then leans on activity, container exit and the ceiling.

## Reconcile before you launch: adopt, record, replay (`reconcile.py`)

Engine containers are siblings spawned through docker.sock, so they outlive the
orchestrator. `reconcile.py` is the only home for what a boot does about that, and
its three steps run in this order, ahead of the launch order below.

- **ADOPT FIRST, and adopt rather than relaunch.** `adopt_running_engines` runs
  before the settle wait, before the sweep and before anything launches: it asks
  `docker inspect` about the containers this node would have created, and puts the
  RUNNING ones back in the `InstanceManager`. **The container's own argv is the
  authority on the shape** (model from `--served-model-name`, width from
  `--tensor-parallel-size`, port from `--port`, `mp` from `--nnodes`), because
  config.json can have been edited since the launch and a stacked instance whose
  manifest write never happened has no other record at all. An adopted record
  carries `adopted=True`, its backend is built on a per-instance config snapshot
  (never the shared `app["config"]`) with the same `instance_id` token the launch
  used so `stop()` reaches the right container, and `_launched_at` is stamped from
  the container's `StartedAt` so `is_running()` asks docker instead of answering
  from a launch subprocess this process never had. An adopted backend has no log
  follower, so a bind wait on it leans on activity and container state.
- **The sweep may not remove an adopted container.** `_remove_engine_containers`
  and `_orphan_engine_ids` both filter on `reconcile.adopted_container_ids()`.
  Nothing is adopted before the boot sweep, so that one is unaffected.
- **A distributed launch is PERSISTED, to `distributed.json`, not to
  `instances.json`.** The solo manifest is replayed entry by entry through
  `append_solo_instance`, so a distributed shape in that list would come back as a
  single-node load. The record is written when the launch succeeds (and refreshed
  by adoption, because the mp head is a config state with no launch call of its
  own) and removed by the unload path via `forget_distributed_instance`, matched on
  model AND port so unloading a neighbour never clears it.
- **Replay only on evidence, once, and never in a loop.**
  `replay_distributed_if_needed` relaunches ONLY when the record exists, the
  container is gone, and every peer answers `probe_peer` (the same
  `ssh -o BatchMode=yes -o ConnectTimeout=10` the launch places a peer container
  with, running `docker version`). Anything else stamps the record `degraded` with
  which peer answered what, and that is surfaced in three places:
  `ainode doctor` (`cluster.distributed`), `/api/status`'s `degraded_instances`,
  and the dashboard banner. One attempt per process: a launch that needs a human
  is not improved by retrying it every ten seconds.
- **A solo load may not replace a MULTI-NODE instance.** `append_solo_instance`
  refuses with a 409 when the model is already up here with peers or a width above
  1 (`_is_distributed_record`). Re-loading a model that is up replaces that
  instance, and doing that to a distributed engine stops it and brings a frontier
  MoE back on one node. Before adoption the manager was empty after a restart, so
  the load path could not see the head and the question never came up.
- **Tests fake the seams, not the checks.** `inspect_container`,
  `list_engine_containers`, `probe_peer` and `port_serving` are module-level
  functions for that reason, and `adopt_running_engines` /
  `replay_distributed_if_needed` are module-level names in `models/api_routes.py`
  so a replay test can neuter them. **A test that calls
  `replay_instances_on_startup` must fake both**, or it reaches the real docker on
  the machine running the suite (the CI runner is a Spark with live engines).

## Launch ORDER on a node: sweep, primary, bind, then stacked one at a time

vLLM sizes its KV cache from what is FREE when the engine profiles, so two engines
profiling on one node at once under-provision whichever finishes second. On the
0.5.11 roll a stacked engine that launched 2 s behind the primary got
`Available KV cache memory: 1.59 GiB` and failed engine init, at the same
`gpu_memory_utilization` that gave it a 600K-token cache on a settled node (#96).
The order below is a contract, not a nicety.

1. **Sweep before anything launches, and wait for it.** `ainode start` calls
   `models/api_routes.py::sweep_engines_before_boot()` BEFORE it starts the boot
   engine: `docker rm -f` every engine container this node owns (the primary
   `ainode-vllm-node-solo`, the stacked `-<port>` ones, a distributed
   `ainode-vllm-head`; never a peer's `ainode-vllm-worker-*`, which belongs to
   whichever head placed it) and poll until the names are gone, because `--rm`
   removal is asynchronous. A container that will not go is logged BY NAME and boot
   continues. The sweep is claimed once per process, so the replay's
   `ensure_startup_sweep()` is then a no-op. **Never widen the replay's own sweep
   past the stacked prefix**: it runs ~10 s into boot, and a wide sweep there would
   remove the primary this boot just launched.
2. **Boot primary next**, and let it BIND before anything stacks on it
   (`_ensure_serving`, the adaptive wait, one retry).
3. **Then the stacked instances, one at a time**, each waiting for the previous one
   to bind. A launch that fails after its one retry logs and the replay moves to the
   next model; it never blocks the rest of the manifest.
4. **Every launch path takes the per-node launch slot** and holds it until the
   engine binds: the startup replay (`launch_slot(..., wait=WAIT_FOREVER)`, boot has
   nobody to refuse to), `POST /api/models/load`, `POST /api/sharding/launch`,
   `POST /api/engine/set-model`. Release the slot when the engine BINDS, not when
   the launch returns, or the next load profiles against memory this one has not
   finished reserving. An HTTP caller that cannot get the slot within
   `_LAUNCH_QUEUE_SECONDS` gets a 409 naming the holder; do not make it queue
   forever.
5. **A bind wait reports the CONTAINER's life**, from `EngineBackend.launched_at`,
   not the age of the wait (which can start minutes after the launch). A backend
   that launches containers stamps it on launch and clears it in `stop()`.

## Verification

- After engine/flag changes: `pytest tests/` and confirm the resolved vLLM command in the launch logs matches the invariants above. Don't claim a serve works unless you saw it reach READY and generate tokens.
