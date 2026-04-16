# Changelog

All notable changes to AINode are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).  
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

_Next release — changes accumulate here until tagged._

---

## [0.4.6] — 2026-04-16

### Fixed
- **Download button hidden for already-downloaded models** — all catalog views (trending, openrouter, latest, HF search, main catalog) now check disk presence via `/api/models/downloaded`. Badge shows "Downloaded" instead of "Available". Re-downloading is blocked with a toast directing users to Launch instead. Fixes [#35](https://github.com/getainode/ainode/issues/35).

---

## [0.4.5] — 2026-04-16

### Fixed
- **Master node shows real identity while engine loads** — previously the master showed a plain "Loading." placeholder even when the node was already discovered, hiding the node name and GPU. Now the real node (name, GB10, crown) is always visible once discovered; a dim veil + spinning arc + "starting..." text overlays it while the engine warms up, fading out when ready.
- **"Update all" button hidden when cluster is current** — the button now only appears in the CLUSTER pill when `GET /api/version/check` confirms a newer version is available on GHCR. Shows the target version: `⬆ Update all  v0.4.6`. Hides immediately after a successful update.

---

## [0.4.4] — 2026-04-16

### Fixed
- **AWQ models crash on GB10 (sm_12.1)** — vLLM auto-upgrades AWQ → `awq_marlin`, but the Marlin CUDA kernels aren't compiled for sm_12.1 in the eugr base image. Engine now pins `--quantization awq` whenever the model name contains `awq`, preventing the upgrade. Fixes [#34](https://github.com/getainode/ainode/issues/34) reported by Chennu@riai360.

### Added
- **Cluster-wide update from the master UI** — `⬆ Update all` button in the CLUSTER pill. Master SSHes into each worker in parallel, runs `docker pull + systemctl restart`, updates itself last. Per-node progress panel shows pending → updating → done/failed status live.
- **`POST /api/cluster/update-all`** — trigger cluster-wide update via REST.
- **`GET /api/cluster/update-status`** — poll per-node update progress.
- **Topology loading animation** — before the engine is ready, the cluster view shows a pulsating dashed circle at center with a breathing `Loading...` label and a spinning arc. When the engine comes online, it cross-fades into the real master node over 0.8 seconds. Worker nodes fade in individually (~1.2s each) as they are discovered — not all at once.

---

## [0.4.3] — 2026-04-15

### Added

**Training — Phase 1: Artifact retrieval & robustness**
- `GET /api/training/jobs/{id}/output` — list all artifact files after training completes (name, size, download URL)
- `GET /api/training/jobs/{id}/output/{filename}` — stream download any artifact file; path traversal blocked
- HF token propagation — `NodeConfig.hf_token` (set via `ainode config --hf-token`) automatically flows to every training job; runners inject `HUGGING_FACE_HUB_TOKEN` + `HF_TOKEN` enabling gated models in training without per-job config
- DDP validation — `torchrun` launch now fails fast with an actionable message if `MASTER_ADDR` is unset, instead of a cryptic NCCL timeout
- OOM error detection — `RuntimeError: CUDA out of memory` is caught and re-emitted as `AINODE_ERROR:CUDA_OOM` with suggestions (lower batch_size, enable gradient checkpointing, switch to QLoRA)
- `TrainingConfig.hf_token` field

**Training — Phase 2: Merge & resume**
- `POST /api/training/jobs/{id}/merge` — merge a completed LoRA/QLoRA adapter into the base model using `PEFT.merge_and_unload()`; runs async, returns a `merge_job_id` to poll
- `POST /api/training/jobs/{id}/resume` — resume training from the latest (or specified) checkpoint; discovers `checkpoint-N/` dirs in the output folder and creates a new job wired to `resume_from_checkpoint`
- `TrainingConfig._resume_from_checkpoint` field

**Training — Phase 3: Custom templates**
- `POST /api/training/templates` — save a custom training template; persisted to `~/.ainode/training/custom_templates.json`
- `GET /api/training/templates` — now returns built-in templates + persisted custom templates

**Training — Phase 4: Evaluation loop**
- `TrainingConfig.eval_split` (default 0.1) — hold out a fraction of the dataset for validation; set to 0 to disable
- `TrainingConfig.eval_steps` — run eval every N steps (default: once per epoch)
- `eval_loss` + `eval_samples_per_second` included in `AINODE_PROGRESS` events when eval is active
- `load_best_model_at_end=True` when eval is enabled — saves the checkpoint with lowest eval_loss

**Training — Phase 5: W&B integration**
- `TrainingConfig.wandb_project` — set a W&B project name to enable Weights & Biases logging; injects `WANDB_PROJECT` + `WANDB_NAME` env vars automatically

**Public docs**
- `docs.argentos.ai/ainode/training` — new guide covering full training workflow, config reference, gated model setup, DDP, all API endpoints, and troubleshooting

### Fixed
- Training datasets now correctly split into train/eval — `dataset.train_test_split()` with fixed seed 42
- `MASTER_PORT` defaults to 29500 if unset when `MASTER_ADDR` is configured

---

## [0.4.2] — 2026-04-15

### Added
- **Cancel download button** — red `✕` button on every in-progress download. `POST /api/models/download-cancel` signals the thread to stop and cleans up partial files.
- **"Downloaded" filter now shows disk-present models** — previously the Downloads tab "Downloaded" filter only showed models loaded in vLLM. Now correctly scans the filesystem.
- **`/api/models/downloaded` endpoint** — returns all models present on disk (HF cache, flat cache, and direct-download layouts).
- **"Launch Model" button** — downloaded-but-not-loaded models show "◉ Downloaded — click Launch to run" and a Launch button instead of Download. Sets the model in config and restarts the engine.
- **`/api/engine/set-model`** — switch the active model and restart the engine without touching the terminal.
- **Version update polling** — UI checks `GET /api/version/check` every 30 minutes. When a newer version is on GHCR, a pulsing green `⬆ Update available: vX.Y.Z` badge appears in the top bar. Click to update in place.
- **`/api/engine/update`** — triggers `docker pull + systemctl restart` from the browser.
- **`list_downloaded()` rewrite** — scans all three HF layout conventions: `hub/models--org--name/`, `models--org--name/`, and `org--name/` (direct download).

### Fixed
- `pynvml FutureWarning` suppressed at import time — no longer floods logs on every start. (Reported by Chennu@riai360, getainode/ainode#33)
- Downloaded model not appearing in chat model selector after download (Chennu@riai360 report).
- "Downloaded" filter in model catalog showed empty for disk-present models not loaded in vLLM.

---

## [0.4.1] — 2026-04-15

### Added
- **`ainode role master|worker|solo`** — set or show this node's cluster role from the CLI. Persistent, saved to `config.json`, applies on next restart.
- **`--job master|worker|solo` install flag** — `curl -fsSL https://ainode.dev/install | bash -s -- --job worker` installs with the correct role from the first boot.
- **Worker nodes start instantly** — `distributed_mode=member` and no model configured skips the vLLM engine entirely. Web server is up in seconds.
- **Web portal starts immediately** — engine now launches in the background. Browser is accessible the moment the container starts, not after 2-10 minutes of model warmup.
- **4-node cluster verified** — 3× DGX Spark + 1× ASUS GX10, 487 GB aggregated VRAM, all four discovered automatically via UDP broadcast.
- Initial config written at install time based on `--job` flag — no manual `config.json` editing required.

### Fixed
- `systemctl daemon-reload` running inside the container during install (no systemd bus available). Unit file now written directly by `install.sh` on the host.
- Banner `\033[` escape codes printed literally. Switched from `cat <<HEREDOC` to `printf`.
- `EOFError` when onboarding called `input()` as a systemd service (no TTY). Non-interactive starts now skip onboarding and mark `onboarded=true` immediately.
- Host wrapper `ainode update` pinned to `:0.4.0` forever. Wrapper now defaults to `:latest`.
- All `docker run` fallback paths in the wrapper double-prefixed `ainode` (entrypoint collision). Fixed with `--entrypoint ainode` on every fallback.
- `__version__` in `ainode/__init__.py` was not updated alongside `pyproject.toml`.

---

## [0.4.1-pre] — 2026-04-15

### Fixed
- **Install entrypoint collision** — `install.sh` was calling `docker run $IMAGE ainode service install`, which passed through `docker-entrypoint.sh` (which prepends `ainode start --in-container`), resulting in `ainode: error: unrecognized arguments: ainode service install`. Fixed by adding `--entrypoint ainode` to the `docker run` call so the CLI is invoked directly. ([#32](https://github.com/getainode/ainode/issues/32) — reported by @Chennu)
- **Gated model 401 on first install** — onboarding defaulted to `meta-llama/Llama-3.1-8B-Instruct` (HF-gated). Users without a token got an OSError and the engine timed out. Defaults are now **Qwen 2.5** (1.5B / 7B / 72B-AWQ) — fully open-access, no token required.
- **Host wrapper double-prefix** — `/usr/local/bin/ainode` fallback `docker run` was passing `ainode "$@"` when the entrypoint already provides `ainode`, causing `ainode ainode <cmd>`.
- **`test_version` hardcoded `"0.1.0"`** — test now reads `ainode.__version__` dynamically.

### Added
- **`ainode config --hf-token <TOKEN>`** — set or clear a Hugging Face token post-install. Token is stored in `~/.ainode/config.json` (never baked into the image). Engine injects `HUGGING_FACE_HUB_TOKEN` + `HF_TOKEN` env vars automatically when present.
- **`scripts/uninstall.sh`** — proper uninstaller: stops/disables system + user service, removes unit files, removes all AINode image tags across GHCR and Docker Hub (no hardcoded version), removes the host wrapper. Data at `~/.ainode` is kept by default — `--purge` required to delete it.
- **`https://ainode.dev/uninstall` redirect** — `curl -fsSL https://ainode.dev/uninstall | bash` works.
- **`NodeConfig.hf_token`** field (optional, default `None`).

---

## [0.4.0] — 2026-04-15

### Added
- **Container-native distribution** — AINode ships as a single unified Docker image (`ghcr.io/getainode/ainode:0.4.0`, mirrored at `argentaios/ainode:0.4.0`). No host Python venv, no vLLM source build. Upgrade is `ainode update`.
- **`ainode update` command** — `docker pull ghcr.io/getainode/ainode:latest` + `systemctl restart ainode` in one command. Installed as `/usr/local/bin/ainode` by `install.sh`. Forwards all other `ainode <cmd>` into the running container via `docker exec`.
- **Cluster member mode** (`distributed_mode = "member"`) — member nodes skip the engine and only run the API + discovery. Head node can now correctly report TP=N topology to the UI.
- **Real distributed tensor-parallel inference** — `docker_engine.py` shells out to eugr's `launch-cluster.sh` for distributed mode. Ray head + worker formation, NCCL over RoCE on ConnectX-7 at 200 Gbps. Verified TP=2 across two DGX Sparks with 244 GB aggregated VRAM at ~35 tok/s.
- **Prometheus `/metrics` endpoint** — standard text-format Prometheus exposition at `http://localhost:8000/metrics`. No `prometheus_client` dependency. Exports: uptime, request counters (total/errors/by-model), token rate, latency P50/P95/P99, GPU util/memory/temp, and `ainode_build_info{version=...}`. JSON endpoints (`/api/metrics`, `/api/metrics/gpu`, `/api/metrics/requests`) retained alongside.
- **Real QLoRA + Full fine-tune + DDP training runners** — `_run_training.py` dispatches per method: QLoRA (bitsandbytes NF4 4-bit + `paged_adamw_8bit`), LoRA (bf16 base + PEFT), Full (no PEFT). All three are DDP-aware (rank-0-only logging/save). `_build_command` picks `torchrun` only when genuinely multi-GPU or multi-node.
- **Speed: dropped `--enforce-eager`** — 2-3× inference speedup. NCCL env vars (`NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, `NCCL_NET_GDR_LEVEL=5`, `NCCL_IB_DISABLE=0`) wired automatically from host interface config.
- **`/v1/embeddings` endpoint** — OpenAI-compatible embeddings passthrough.
- **UI: distributed instance badges** — "DISTRIBUTED · TP=N" badge when a sharded model is running. Launch hint turns amber when peer count is insufficient.
- **`scripts/install.sh`** rewrite — pure `docker pull` + systemd unit install. No host Python. Optional `--setup-ssh` / `AINODE_PEERS` for distributed bootstrap.
- **`scripts/docker-entrypoint.sh`** — copies host SSH keys from `/host-ssh` to `/root/.ssh` with correct ownership. Injects `User <ssh_user>` for peer IPs in ssh_config.
- **`scripts/Dockerfile.ainode`** — `FROM ainode-base` + openssh-client, sshpass, iproute2, curl, docker-ce-cli, docker-compose-plugin.
- **`.github/workflows/publish-image.yml`** — `workflow_dispatch` build + push to GHCR and Docker Hub on self-hosted aarch64 runner.
- **`ops/runbooks/release-flow.md`** — full runbook covering the three distribution surfaces (marketing site / install.sh / container image), decision tree, aarch64 build on Spark 1, GHCR push, rollback.

### Changed
- **systemd unit** — `ExecStart` is now `docker run --gpus all ... ainode:0.4.0` (not a host Python process). Dropped `ProtectSystem`/`ProtectHome` (conflict with docker socket mount).
- **`AINODE_IMAGE_TAG = "0.4.0"`** in `systemd.py` keeps unit file and pyproject.toml version in lockstep.
- **`ainode/engine/docker_engine.py`** rewritten — `start_solo()` → `vllm serve` Popen; `start_distributed()` → `launch-cluster.sh` subprocess.

### Fixed
- **NCCL placement group hang** — caused by multi-NIC routing ambiguity on 192.168.0.0/24. Fixed by switching cluster fabric to direct-connect 10.0.0.0/24 (`enp1s0f0np0`).
- **ext4 bitmap corruption on `/mnt/rosa-models`** — silent zero-writes from bad block bitmap. Switched NFS export to `/mnt/rosa-storage` (healthy).
- **SSH key ownership in container** — host uid ≠ root, OpenSSH refused keys. Fixed by mounting at `/host-ssh:ro` and copying to `/root/.ssh` in entrypoint.
- **eugr launcher uses bare `ssh <host>`** — defaults to root@host when run as root. Fixed: entrypoint injects `User <ssh_user>` for peer IPs.
- **`docker: command not found` in container** — needed docker CLI for eugr's `docker cp/run`. Fixed: `apt install docker-ce-cli`.
- **`ip` command not found** — autodiscovery failed. Fixed: `iproute2` added to Dockerfile.

---

## [0.3.0] — 2026-04-10 _(pre-container era)_

### Added
- **Server view** — LM Studio-style API console with live request logs, loaded model, endpoint catalog.
- **Orbital topology UI** — master node at center with pulsing rings, workers on circumference, data pulses inward.
- **Cluster config panel** — minimum nodes, TP/PP selection, cluster interface picker.
- **Training experience overhaul** — context-switching sidebar, job wizard, dataset manager, loss charts.
- **Downloads UI** — real-time download progress (percent, speed, ETA), model detail modal, capability badges.
- **Chat enhancements** — stop generation, TTFT/TPS metrics, conversation history persistence, code blocks, image drag-drop.
- **Phase 1 distributed inference** — Ray autostart, VRAM aggregation across nodes, sharded launch prototype.
- **`/v1/embeddings`** — OpenAI-compatible embeddings endpoint.
- **`list_available`** merges disk-downloaded models not in catalog.
- **Delete model UI** — remove models from disk via the downloads view.
- **Config panel** — cluster master/worker role, secrets management.

### Changed
- **Default model** switched to `Qwen/Qwen2.5-1.5B-Instruct` (Llama requires HF token).
- **Install script** adapted for Docker-based vLLM on GB10/CUDA 13, pip-based vLLM elsewhere.
- **Topology graph** rewritten — static workers on circumference (not force-directed).
- **Nav** renamed for clarity; chat promoted to primary view; launch dropdown fixed.

### Fixed
- Download tracking survives page refresh (no DOM wipe, server reconciliation).
- Chat bar hidden behind footer — correct 80px reservation.
- HF cache scan path corrected; scroll position preserved on refresh.
- `node_name` auto-falls back to `socket.gethostname()`.
- Ray autostart no longer blocks the event loop; skips bogus master addresses.

---

## [0.2.0] — 2026-04 _(packaging + UI iteration)_

### Added
- **Training UI** — job dashboard, new job form, detail view with live loss chart.
- **Dashboard real-time widgets** — GPU utilization gauge, memory ring, temperature, request metrics.
- **Interactive topology graph** — force-directed, live node connections.
- **Models page** — full model management (catalog browse, download, delete, recommend).
- **Optional API key auth** — enable/disable via `ainode auth enable/disable`.
- **Browser-based onboarding wizard** — first-run setup via the web UI.

### Changed
- Packaging fixed to include `static/` and `templates/`. Version bumped to 0.2.0.

### Fixed
- AI slop cleanup — removed dead code, restating docstrings, unused imports.

---

## [0.1.0] — 2026-04 _(initial build)_

### Added
- CLI skeleton: `ainode start`, `stop`, `status`, `models`, `config`, `logs`, `service`, `auth`.
- GPU detection via pynvml/psutil.
- vLLM engine wrapper (host Python path).
- OpenAI-compatible API proxy (`/v1/chat/completions`, `/v1/completions`, `/v1/models`).
- Multi-node UDP cluster discovery (port 5679).
- Metrics/monitoring: GPU stats, request counters, latency percentiles.
- Model manager: catalog, download, delete, GPU-fit recommendations.
- Training engine: LoRA + full fine-tune, job queue, progress streaming.
- systemd service management (`ainode service install/uninstall/status`).
- Rich terminal output (banners, tables, spinners).
- Browser-based dashboard: chat, topology, training, downloads.
- 238-test suite across all modules.
- Ops structure: rules, runbooks, slices registry, agent conventions.

---

[Unreleased]: https://github.com/getainode/ainode/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/getainode/ainode/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/getainode/ainode/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/getainode/ainode/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/getainode/ainode/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/getainode/ainode/releases/tag/v0.1.0
