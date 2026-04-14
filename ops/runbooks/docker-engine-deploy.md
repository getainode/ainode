# Docker Engine Deploy — AINode on GB10 / CUDA 13

**Status:** Ready for deploy (2026-04-14, updated with NFS shared models layout). Target: demo-ready week of 2026-04-20.
**Author:** Claude (Opus 4.6, 1M context) working with Jason Brashear.

This runbook is the source of truth for how AINode is installed and started
on any GB10 / CUDA 13 node (ASUS GX10, DGX Spark 1, DGX Spark 2, any future
Blackwell unified-memory box). It must remain accurate — if the install
script changes, update this file in the same commit.

## 1. Why Docker on GB10

- vLLM pip wheels target CUDA 12. GB10 ships with CUDA 13. No wheel matches.
- Source-building vLLM on aarch64 for sm_121 compiles ~952 CUDA kernels and takes 15+ hours on a DGX Spark. Not viable for an end-user install.
- Community image `scitrera/dgx-spark-vllm:0.17.0-t5` is pre-built for sm_121 and works out of the box.
- Conclusion: on GB10, vLLM runs in Docker. AINode's Python stack (API, discovery, web UI, training) runs in a native venv on the host and talks to the container on localhost:8000.

## 2. Architecture

```
┌──────────────────────────────────────────────────────────┐
│ Host (asus-ainode-gx10 / spark-dgx-1 / spark-dgx-2)      │
│                                                          │
│  ~/.ainode/venv/ ── ainode CLI ── API server :3000       │
│                         │                                │
│                         ▼                                │
│                    DockerEngine ──► docker compose       │
│                                         │                │
│                                         ▼                │
│  ┌────────────────────────────────────────────────┐      │
│  │ container: ainode-vllm                         │      │
│  │ image: scitrera/dgx-spark-vllm:0.17.0-t5       │      │
│  │ listens on :8000 (bound to host 127.0.0.1:8000)│      │
│  │ --gpus all, --ipc=host, --shm-size=16g         │      │
│  │ -v ~/.ainode/models:/models                    │      │
│  └────────────────────────────────────────────────┘      │
│                                                          │
│  systemd user unit: ainode.service ──► ainode start      │
└──────────────────────────────────────────────────────────┘
```

## 3. Files the install creates

| Path | Purpose |
|------|---------|
| `~/.ainode/venv/` | Python venv with AINode + deps |
| `~/.ainode/config.json` | NodeConfig (includes `engine_strategy: "docker"`) |
| `~/.ainode/docker-compose.yml` | vLLM service definition |
| `~/.ainode/.env` | Env overrides (AINODE_MODEL, AINODE_GPU_MEMORY_UTIL, AINODE_TP_SIZE, AINODE_RAY_ADDRESS) |
| `~/.ainode/models/` | HF cache, mounted into container as `/models` |
| `~/.config/systemd/user/ainode.service` | systemd user service (auto-start on login) |

## 4. Install flow (`scripts/install.sh`)

1. Detect GPU + CUDA. If GB10 / sm_121 / CUDA 13 → `ENGINE_STRATEGY=docker`. Else → `pip`.
2. Create venv at `~/.ainode/venv/`, install AINode + `sentence-transformers` + `ray[default]`.
3. (Docker path only)
   a. `docker pull scitrera/dgx-spark-vllm:0.17.0-t5`
   b. Write `~/.ainode/docker-compose.yml` (see §5)
   c. Write `~/.ainode/.env` with defaults
   d. Seed `~/.ainode/config.json` with `engine_strategy: "docker"`, `discovery_port: 5679`, `cluster_id: "default"`, `onboarded: true`
   e. Write + enable systemd user unit → `systemctl --user enable --now ainode.service`
   f. `loginctl enable-linger $USER` so the service survives logout
4. Health-check: wait for `http://localhost:3000/api/health` to return 200 (timeout 5m).

## 5. docker-compose.yml template

```yaml
services:
  vllm:
    image: scitrera/dgx-spark-vllm:0.17.0-t5
    container_name: ainode-vllm
    restart: unless-stopped
    ipc: host
    shm_size: "16gb"
    ports:
      - "127.0.0.1:8000:8000"
    volumes:
      - ${HOME}/.ainode/models:/models
    environment:
      HF_HOME: /models
      AINODE_MODEL: ${AINODE_MODEL:-meta-llama/Llama-3.2-3B-Instruct}
      AINODE_GPU_MEMORY_UTIL: ${AINODE_GPU_MEMORY_UTIL:-0.9}
      AINODE_TP_SIZE: ${AINODE_TP_SIZE:-1}
      AINODE_RAY_ADDRESS: ${AINODE_RAY_ADDRESS:-}
    command:
      - --model
      - ${AINODE_MODEL:-meta-llama/Llama-3.2-3B-Instruct}
      - --host
      - 0.0.0.0
      - --port
      - "8000"
      - --gpu-memory-utilization
      - ${AINODE_GPU_MEMORY_UTIL:-0.9}
      - --tensor-parallel-size
      - ${AINODE_TP_SIZE:-1}
      - --dtype
      - bfloat16
      - --download-dir
      - /models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## 6. DockerEngine class (`ainode/engine/docker_engine.py`)

Same interface as `VLLMEngine`:
- `start()` → `docker compose up -d`
- `stop()` → `docker compose down`
- `is_running()` → parses `docker ps --filter name=ainode-vllm --format json`
- `wait_ready(timeout)` → polls `http://127.0.0.1:8000/v1/models` for 200
- `launch(model_id)` → writes `AINODE_MODEL` into `.env`, `docker compose up -d` (recreates container)
- `launch_distributed(config)` → sets `AINODE_TP_SIZE`, `AINODE_RAY_ADDRESS`, recreates
- `logs(n)` → `docker logs --tail n ainode-vllm`

`cmd_start` picks the engine by `config.engine_strategy`:
```python
if config.engine_strategy == "docker":
    engine = DockerEngine(config)
else:
    engine = VLLMEngine(config)
```

## 7. systemd user unit

```ini
[Unit]
Description=AINode
After=network-online.target docker.service

[Service]
Type=simple
ExecStart=%h/.ainode/venv/bin/ainode start
ExecStop=%h/.ainode/venv/bin/ainode stop
Restart=on-failure
RestartSec=5
Environment=PATH=%h/.ainode/venv/bin:/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=default.target
```

Enable: `systemctl --user enable --now ainode.service`
Linger: `sudo loginctl enable-linger $USER` (so it starts at boot, not login)

## 8. One-command install target

From a fresh GB10 box:
```bash
curl -fsSL https://raw.githubusercontent.com/webdevtodayjason/ainode/main/scripts/install.sh | bash
```

End state (zero other commands):
- venv ready
- image pulled
- compose file + .env + config.json written
- systemd service enabled and running
- Container up, vLLM serving at 127.0.0.1:8000
- AINode API at :3000
- Node discoverable on UDP 5679, cluster_id `default`

## 9. Validation — run after deploy on each node

```bash
# Service is up
systemctl --user status ainode.service

# Container is healthy
docker ps --filter name=ainode-vllm
curl -s http://localhost:8000/v1/models | jq

# AINode API is healthy
curl -s http://localhost:3000/api/health

# Cluster sees this node (from any node)
curl -s http://localhost:3000/api/nodes | jq

# Cluster resources aggregate (should show total VRAM = 122 GB × N)
curl -s http://localhost:3000/api/cluster/resources | jq
```

## 10. Deploy order (2026-04-14)

1. Build DockerEngine class + update install.sh + systemd unit. Commit to `codex/docker-engine` branch.
2. Open PR → merge to `dev`.
3. Push to `origin/dev`, `jason/dev`, `jason dev:main`.
4. **Spark 2** (100.81.184.19, sem) — fresh install target. Run one-liner. Verify.
5. **Spark 1** (100.122.26.9, sem) — re-run one-liner. Should be idempotent.
6. **ASUS GX10** (100.72.9.84, SSH user TBD from Jason) — re-run one-liner. Stop the old manual container first.
7. Verify cluster forms with 3 nodes, ~384 GB total VRAM.
8. Demo the Launch Instance → sharded across 3 nodes flow.

## 10.5. Shared model storage (NFS from master)

AINode is designed to **download a model once and use it everywhere** on the cluster.
NVMe-over-TCP exported from our MikroTik RDS2216 ROSA array presents each host
with its own block namespace — ext4 on those namespaces is **not multi-writer
safe**. So we put NFS on top for file-level sharing; NVMe-TCP still provides the
backing speed on the master.

### Topology

```
    ROSA / RDS2216
    ├─ namespace-A  ────(NVMe-TCP)───►  Spark 1 : /mnt/rosa-models (ext4)
    │                                       │
    │                                       ├── NFS server exports /mnt/rosa-models
    │                                       │
    │                                       ├── Spark 2 mounts as /mnt/rosa-shared
    │                                       └── GX10   mounts as /mnt/rosa-shared
```

Spark 1 is the **explicit master**: serves NFS, runs AINode as `cluster_role=master`.
Spark 2 and GX10 run as workers; their existing local `/mnt/rosa-models` ext4 (where
present) is **untouched** — we add a new mountpoint `/mnt/rosa-shared` that points
at Spark 1's export.

### Spark 1 (NFS server)

```bash
sudo apt-get install -y nfs-kernel-server
sudo chown -R sem:sem /mnt/rosa-models
cat /etc/exports   # must contain:
# /mnt/rosa-models  <gx10-ip>(rw,sync,no_subtree_check,no_root_squash) \
#                   <spark2-ip>(rw,sync,no_subtree_check,no_root_squash)
sudo exportfs -ra
sudo systemctl enable --now nfs-kernel-server
```

### Spark 2 + GX10 (NFS clients)

```bash
sudo apt-get install -y nfs-common
sudo mkdir -p /mnt/rosa-shared
# Add to /etc/fstab (survives reboot):
# <spark1-ip>:/mnt/rosa-models /mnt/rosa-shared nfs \
#   rw,noatime,nodiratime,rsize=1048576,wsize=1048576,nconnect=16,hard,vers=4.2,_netdev 0 0
sudo mount /mnt/rosa-shared
```

### AINode wiring

- Master (Spark 1): compose volume `/mnt/rosa-models:/models`
- Workers: compose volume `/mnt/rosa-shared:/models`
- All nodes: `config.json` → `"models_dir": "/models-shared"` (the in-container
  path already served by the volume above — HF_HOME points there).
- Explicit roles: Spark 1 `cluster_role=master`, Spark 2 + GX10 `worker`.

### Why this layout

- **Block-level NVMe-TCP** can't be multi-mounted ext4 safely (no DLM on generic
  Linux). File-level NFS on top adds the needed coordination.
- **NFS over the 100G fabric** delivers ~3–8 GB/s sequential reads — vLLM model
  loading is a one-shot sequential read, so the penalty vs local-NVMe is
  noticeable only for 100 GB+ models and can be further optimised later with a
  hybrid rsync-to-local staging step.
- **Single writer = single source of truth.** Downloads initiated on any node
  land on the master's NVMe; all nodes read the same file.

### Future: hybrid staging for huge models

For 100 GB+ models where load latency matters:

1. Download once into `/mnt/rosa-shared/models--org--name/`
2. On first use, worker rsyncs the snapshot into a local scratch (e.g.
   `/var/lib/ainode/hot/`) and points vLLM there
3. Subsequent loads hit local Gen5 NVMe (~10–14 GB/s)

Not implemented yet; file this as a follow-up slice when it becomes a
bottleneck.

## 11. Known issues / gotchas

- EXO grabs UDP 5678 on DGX Sparks. AINode discovery pinned to 5679 in the install config.
- `docker compose` plugin required (not `docker-compose` v1). Install.sh checks and installs via `get.docker.com` if missing.
- User must be in `docker` group OR install uses `sudo docker` — install.sh adds user to group and warns to re-login.
- `loginctl enable-linger` requires sudo. If it fails, service starts on login instead of boot.
- HF_TOKEN is needed for gated models (Llama 3). Install.sh prompts or reads from existing `~/.ainode/secrets.json`.

## 12. If this breaks — diagnostic checklist

| Symptom | Check |
|---------|-------|
| `ainode start` exits with ModuleNotFoundError: vllm | `config.engine_strategy` is still `pip` — fix config |
| Container restarts endlessly | `docker logs ainode-vllm` — usually HF auth, disk full, or CUDA driver mismatch |
| Cluster doesn't see peer | `ss -ulnp \| grep 5679` — confirm both nodes bound; check `cluster_id` matches |
| Web UI 500s | `journalctl --user -u ainode.service -f` |
| Launch Instance returns "Engine not initialized" | `DockerEngine` not instantiated — check cmd_start branch |
