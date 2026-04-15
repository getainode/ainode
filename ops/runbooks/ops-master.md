# AINode Operations Master Runbook

**Owner:** Jason Brashear / Argentos AI  
**Last updated:** 2026-04-15  
**Status:** Active — update this file when infra or procedures change.

This is the single source of truth for day-to-day AINode operations.
Developers and agents should read this before touching anything in production.

---

## Table of Contents

1. [Infrastructure inventory](#1-infrastructure-inventory)
2. [Access and credentials](#2-access-and-credentials)
3. [Day-to-day operations](#3-day-to-day-operations)
4. [Installing AINode on a new node](#4-installing-ainode-on-a-new-node)
5. [Cluster management](#5-cluster-management)
6. [Releasing a new version](#6-releasing-a-new-version)
7. [Troubleshooting](#7-troubleshooting)
8. [Emergency procedures](#8-emergency-procedures)
9. [Monitoring and metrics](#9-monitoring-and-metrics)
10. [Network topology](#10-network-topology)
11. [Storage](#11-storage)
12. [Security](#12-security)

---

## 1. Infrastructure Inventory

### AI Compute Nodes

| Node | Tailscale IP | Local fabric IP | SSH user | Role | GPU | Status |
|------|-------------|-----------------|----------|------|-----|--------|
| DGX Spark 1 | 100.122.26.9 | 10.0.0.1 | sem | cluster head | GB10 128 GB | ✅ Active |
| DGX Spark 2 | 100.81.184.19 | 10.0.0.2 | sem | cluster worker | GB10 128 GB | ✅ Active |
| DGX Spark 3 | TBD | TBD | TBD | cluster worker | GB10 128 GB | 🔄 Onboarding |
| ASUS GX10 | 100.72.9.84 | TBD | sem | standalone | GB10 128 GB | ✅ Active |

> **Update Spark 3 row** when it comes up. Local fabric IP should be `10.0.0.3`.

### Support Infrastructure

| System | Location | Role |
|--------|----------|------|
| Dell R750 | On-prem | 2TB RAM, 72-core host for self-hosted E2B sandboxes + CI runners |
| M3 Studio Ultra | On-prem | 256 GB, primary dev workstation |
| M2 Mac Studio | On-prem | 64 GB, secondary dev |
| MikroTik ROSA RDS2216 | On-prem | NVMe-over-TCP block storage, 400 Gbps backbone |

### Cloud / SaaS

| Service | Purpose | Account |
|---------|---------|---------|
| GitHub | Source — `getainode` org | webdevtodayjason |
| GHCR | Container registry — `ghcr.io/getainode/ainode` | getainode org |
| Docker Hub | Mirror — `argentaios/ainode` | semfreak / argentaios org |
| Railway | Hosts `ainode.dev` marketing site | linked to `getainode/ainode.dev` |
| Tailscale | Overlay VPN for management SSH to all nodes | — |

---

## 2. Access and Credentials

**SSH — all nodes:**
```bash
# From your workstation (Tailscale must be up)
ssh sem@100.122.26.9   # Spark 1
ssh sem@100.81.184.19  # Spark 2
ssh sem@100.72.9.84    # GX10
```

Password and all sensitive values live in `/Users/sem/code/ainode/.env`  
(gitignored — never commit). File layout:
```
AINODE_SSH_USER=sem
AINODE_SSH_PASSWORD=<redacted>
AINODE_SPARK1_HOST=100.122.26.9
AINODE_SPARK2_HOST=100.81.184.19
AINODE_GX10_HOST=100.72.9.84
DOCKERHUB_USER=semfreak
DOCKERHUB_TOKEN=dckr_pat_<redacted>
```

**GHCR push** requires `write:packages` scope on the GitHub token:
```bash
gh auth refresh -s write:packages,read:packages
```

**Docker Hub push:**
```bash
echo "$DOCKERHUB_TOKEN" | docker login docker.io -u "$DOCKERHUB_USER" --password-stdin
```

---

## 3. Day-to-Day Operations

### Check cluster status

From any node:
```bash
ainode status
```

From your workstation (hitting the API directly):
```bash
curl -s http://100.122.26.9:8000/api/nodes | python3 -m json.tool
curl -s http://100.122.26.9:8000/api/cluster/resources | python3 -m json.tool
```

### Check service health

```bash
systemctl status ainode          # is it running?
journalctl -u ainode -f          # live logs
ainode logs -f                   # vLLM engine logs (inside container)
```

### Restart the service

```bash
sudo systemctl restart ainode
# or using the wrapper:
ainode update   # pull latest image + restart (use for upgrades)
```

### Check Prometheus metrics

```bash
curl http://localhost:8000/metrics | grep -E "^ainode_"
```

---

## 4. Installing AINode on a New Node

### Prerequisites (on the target node)

```bash
# 1. Docker Engine
curl -fsSL https://get.docker.com | bash
sudo usermod -aG docker $USER
newgrp docker

# 2. NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Install AINode (single node)

```bash
curl -fsSL https://ainode.dev/install | bash
```

The installer:
1. Sanity-checks Linux + Docker + NVIDIA runtime
2. Pulls `ghcr.io/getainode/ainode:0.4.1` (~18 GB, one-time)
3. Writes and enables `/etc/systemd/system/ainode.service`
4. Installs `/usr/local/bin/ainode` host wrapper
5. Starts the service

Open `http://<node-ip>:3000` — onboarding wizard will run on first visit.

### Install for distributed cluster (head node)

```bash
AINODE_PEERS="10.0.0.2,10.0.0.3" curl -fsSL https://ainode.dev/install | bash
```

This also bootstraps passwordless SSH to peer IPs.

### Post-install checklist

- [ ] `systemctl status ainode` → active (running)
- [ ] `curl http://localhost:8000/v1/models` → returns model list
- [ ] `http://<ip>:3000` → web UI loads
- [ ] `ainode status` → shows node info + GPU stats
- [ ] Add node IP to `.env` on your workstation
- [ ] Update memory in `reference_ainode_infrastructure.md`

---

## 5. Cluster Management

### Current cluster topology

```
Spark 1 (10.0.0.1) — HEAD
  └── Spark 2 (10.0.0.2) — WORKER   ← TP=2 verified
  └── Spark 3 (10.0.0.3) — WORKER   ← onboarding
```

NCCL transport: **RoCE on ConnectX-7** (`enp1s0f0np0`) at 200 Gbps.  
Aggregated VRAM: 256 GB (2-node) → 384 GB (3-node once Spark 3 is up).

### Add a new node to the cluster

1. Install AINode on the new node (see §4)
2. On the head node, update `~/.ainode/config.json`:
   ```json
   {
     "distributed_mode": "head",
     "peer_ips": ["10.0.0.2", "10.0.0.3"],
     "cluster_interface": "enp1s0f0np0",
     "ssh_user": "sem"
   }
   ```
3. `sudo systemctl restart ainode`
4. Open `http://10.0.0.1:3000` → Config → Cluster → verify new node appears in topology

### Trigger distributed launch

Via UI: Config → Cluster → set Minimum Nodes → click Launch (tensor-parallel mode).

Via API:
```bash
curl -s -X POST http://localhost:8000/api/engine/start \
  -H "Content-Type: application/json" \
  -d '{"distributed": true}'
```

### Verify NCCL is using RoCE (not socket fallback)

After distributed launch, check vLLM worker logs on the head node:
```bash
ainode logs -f | grep -E "IB|RoCE|network"
# expect: "Using network IB" + "mlx5_0:1/RoCE ... speed=200000"
```

If you see "Using network Socket" instead, RDMA is not working — check:
```bash
# Confirm exactly one interface has an IP on the cluster subnet
ip addr | grep "10.0.0"
# Should show exactly one: enp1s0f0np0

# Confirm mlx5 driver is loaded
lsmod | grep mlx5
```

### Single-NIC rule (critical)

Each node must have **exactly one NIC with an IP on the cluster subnet**.  
Multiple NICs on the same subnet → Ray placement group hangs indefinitely.  
If you see multiple, remove the extra IP before launching distributed.

---

## 6. Releasing a New Version

> Full detail in `ops/runbooks/release-flow.md`. This is the quick reference.

### Decision tree

```
Changed install.sh / README only  →  git push main, done
Changed web UI or Python code     →  rebuild image (§6a)
Changed vLLM / Ray / base OS      →  rebuild base image first (ops/runbooks/release-flow.md §6)
```

### §6a. Release a Python or UI change

```bash
# 1. Bump versions (two files)
# pyproject.toml:  version = "0.4.2"
# ainode/service/systemd.py:  AINODE_IMAGE_TAG = "0.4.2"

# 2. Commit + push to main
git commit -am "chore: bump to 0.4.2" && git push origin main

# 3. Build on Spark 1 (must be aarch64 — no Mac builds)
ssh sem@100.122.26.9
cd ~/ainode && git pull
docker build \
  -f scripts/Dockerfile.ainode \
  --build-arg BASE_IMAGE=ghcr.io/getainode/ainode-base:c026c92 \
  -t ghcr.io/getainode/ainode:0.4.2 \
  -t ghcr.io/getainode/ainode:latest \
  .

# 4. Smoke test
docker run --rm ghcr.io/getainode/ainode:0.4.2 ainode --version

# 5. Push (token refresh if needed: gh auth refresh -s write:packages,read:packages)
gh auth token | docker login ghcr.io -u webdevtodayjason --password-stdin
docker push ghcr.io/getainode/ainode:0.4.2
docker push ghcr.io/getainode/ainode:latest

# 6. Mirror to Docker Hub
echo "$DOCKERHUB_TOKEN" | docker login docker.io -u "$DOCKERHUB_USER" --password-stdin
docker tag ghcr.io/getainode/ainode:0.4.2 argentaios/ainode:0.4.2
docker tag ghcr.io/getainode/ainode:latest argentaios/ainode:latest
docker push argentaios/ainode:0.4.2
docker push argentaios/ainode:latest

# 7. Tag + GitHub release
git tag v0.4.2 && git push origin v0.4.2
gh release create v0.4.2 --generate-notes

# 8. Upgrade all nodes
for host in $AINODE_SPARK1_HOST $AINODE_SPARK2_HOST $AINODE_GX10_HOST; do
  ssh sem@$host "ainode update"
done
```

### Update CHANGELOG

Add an entry under `[Unreleased]` as work happens.  
When tagging, rename `[Unreleased]` to `[0.4.2] — YYYY-MM-DD` and add a fresh `[Unreleased]` section at the top.

---

## 7. Troubleshooting

### Engine fails to become ready within 5 minutes

**Check logs first:**
```bash
ainode logs -f
# or
journalctl -u ainode -n 100
```

Common causes:

| Symptom in logs | Cause | Fix |
|----------------|-------|-----|
| `OSError: You are trying to access a gated repo` | Model requires HF token | `ainode config --hf-token hf_xxx && sudo systemctl restart ainode` |
| `CUDA out of memory` | Model too large for single GPU | Use smaller model or enable distributed mode |
| `docker: command not found` | Docker not installed | Run Docker install (§4 prerequisites) |
| `permission denied /var/run/docker.sock` | User not in docker group | `sudo usermod -aG docker $USER && newgrp docker` |
| `vllm serve` hangs at startup | Port 8000 already in use | `sudo ss -tlnp | grep 8000` then kill the process |

### Ray placement group creation hangs (distributed mode)

```bash
# 1. Check for multi-NIC ambiguity
ip addr | grep "10.0.0"   # should show exactly ONE address

# 2. Check Ray head is running
docker exec ainode ray status 2>/dev/null

# 3. Kill and restart
sudo systemctl restart ainode
```

### Discovery not seeing peers

```bash
# Confirm UDP 5679 is reachable from peer
# On Spark 1:
nc -ulp 5679 &
# On Spark 2:
echo "test" | nc -u 10.0.0.1 5679

# Check firewall
sudo ufw status
sudo iptables -L INPUT | grep 5679
```

### Container exits immediately after start

```bash
docker logs ainode 2>&1 | tail -30
# Common: GPU not accessible
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
# If that fails: restart docker + nvidia-container-runtime
sudo systemctl restart docker
```

### `ainode update` pulls but service doesn't restart

```bash
# Verify the unit file uses :latest or the new tag
cat /etc/systemd/system/ainode.service | grep Image
# If it pins an old tag, update it:
sudo sed -i 's|ainode:0.4.0|ainode:latest|' /etc/systemd/system/ainode.service
sudo systemctl daemon-reload && sudo systemctl restart ainode
```

### Marketing site not updating after push

Railway's GitHub App sometimes misses the webhook. Touch the README:
```bash
cd /Users/sem/code/ainode.dev
date >> README.md && git add README.md && git commit -m "chore: nudge deploy" && git push
```

Or log in to the Railway dashboard and click **Redeploy**.

---

## 8. Emergency Procedures

### Roll back to previous version

```bash
# On affected node:
AINODE_IMAGE=ghcr.io/getainode/ainode:0.4.0 ainode update

# To make it stick across reboots:
sudo sed -i 's|ainode:latest|ainode:0.4.0|' /etc/systemd/system/ainode.service
sudo systemctl daemon-reload
```

### Completely remove AINode from a node

```bash
curl -fsSL https://ainode.dev/uninstall | bash          # keeps ~/.ainode data
curl -fsSL https://ainode.dev/uninstall | bash --purge  # also wipes data
```

### Node is down / unreachable

1. Check Tailscale: `tailscale status` — is the node online?
2. Try KVM over IP (physical console) if Tailscale is down
3. Check if Docker daemon is running: `ssh sem@<ip> systemctl is-active docker`
4. Check disk space — vLLM logs can grow large: `df -h /`
5. Reboot if needed: `ssh sem@<ip> sudo reboot`

### Nuclear option — wipe and reinstall

```bash
ssh sem@<node-ip>
curl -fsSL https://ainode.dev/uninstall | bash --purge
# Wait 30s for systemd to settle
curl -fsSL https://ainode.dev/install | bash
```

---

## 9. Monitoring and Metrics

### Prometheus scrape config

Add to your `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: ainode-spark1
    static_configs:
      - targets: ["100.122.26.9:8000"]
  - job_name: ainode-spark2
    static_configs:
      - targets: ["100.81.184.19:8000"]
  - job_name: ainode-gx10
    static_configs:
      - targets: ["100.72.9.84:8000"]
```

### Key metrics to alert on

| Metric | Alert threshold | Meaning |
|--------|----------------|---------|
| `ainode_gpu_memory_used_bytes / ainode_gpu_memory_total_bytes` | > 0.95 | GPU memory nearly full |
| `ainode_gpu_temperature_celsius` | > 85 | Thermal throttling risk |
| `ainode_request_errors_total` rate | > 5/min | Engine returning errors |
| `ainode_gpu_available` | = 0 | pynvml can't reach GPU |
| `up` (Prometheus built-in) | = 0 | Node down |

### Quick one-liner metrics check

```bash
# All nodes:
for h in 100.122.26.9 100.81.184.19; do
  echo "=== $h ==="; curl -s http://$h:8000/api/metrics | python3 -m json.tool | grep -E "uptime|total|tokens_per"
done
```

---

## 10. Network Topology

```
Internet
    │
    ▼
MikroTik CHR (400 Gbps backbone)
    │
    ├── 10GbE management network (192.168.0.0/24)
    │       ├── Spark 1:  192.168.0.110
    │       ├── Spark 2:  192.168.0.111
    │       ├── Spark 3:  192.168.0.112 (TBD)
    │       └── GX10:     192.168.0.xxx
    │
    └── Direct-connect cluster fabric (10.0.0.0/24)  ← NCCL uses this
            ├── Spark 1:  10.0.0.1  (enp1s0f0np0, ConnectX-7)
            ├── Spark 2:  10.0.0.2  (enp1s0f0np0, ConnectX-7)
            └── Spark 3:  10.0.0.3  (TBD)

Tailscale (100.x.x.x) — management SSH only, NOT used for NCCL
```

**Critical rules:**
- NCCL must go over the direct-connect fabric (10.0.0.0/24), never Tailscale
- Each node must have exactly one NIC with an IP on the cluster subnet
- Discovery UDP broadcast (port 5679) works on 192.168.0.0/24
- SSH for eugr's cluster launcher must be passwordless on 10.0.0.x

### Ports used

| Port | Protocol | Service |
|------|----------|---------|
| 3000 | TCP | AINode web UI |
| 8000 | TCP | AINode API (OpenAI-compatible + /metrics) |
| 5679 | UDP | AINode discovery broadcast |
| 6379 | TCP | Ray head |
| 10001–10009 | TCP | Ray workers |
| 29500 | TCP | NCCL rendezvous |

---

## 11. Storage

### Model storage

Models live at `~/.ainode/models/` on each node (default, backed by local NVMe).

For shared models across the cluster (download once, use everywhere):

```bash
# On the NFS server (Spark 1 or dedicated host):
sudo mkdir -p /mnt/rosa-storage/ainode-models
sudo exportfs -av  # verify /etc/exports includes this path

# On each worker node:
sudo mount -t nfs 10.0.0.1:/mnt/rosa-storage/ainode-models /mnt/models

# Set in config.json on each node:
# "models_dir": "/mnt/models"
```

**Warning:** `/mnt/rosa-models` on Spark 1 has ext4 bitmap corruption.  
**Use `/mnt/rosa-storage` instead** (14 TB, healthy). See `project_rosa_models_fsck_todo.md`.

### Training artifacts

```
~/.ainode/
├── models/       # downloaded model weights
├── logs/         # vLLM + distributed engine logs
├── datasets/     # uploaded training datasets
└── training/     # fine-tune job outputs (LoRA adapters, full checkpoints)
    └── <job-id>/
        ├── config.json
        ├── adapter_model.bin  (LoRA)
        └── ...
```

Training artifacts are **never touched by `ainode update`** — they're on the host, outside the container.

---

## 12. Security

### API authentication

Disabled by default (trusted LAN). Enable for internet-exposed instances:
```bash
ainode auth enable         # generates an API key
ainode auth new-key        # rotate key
```

Then clients pass `Authorization: Bearer <key>` on all `/v1/*` requests.

### What should never be committed

- `/Users/sem/code/ainode/.env` — SSH credentials, Docker Hub token (gitignored)
- `~/.ainode/config.json` — contains `hf_token` if set
- Any private SSH keys

### SSH key management

Passwordless SSH between cluster nodes is required for distributed mode.  
Keys are generated during `install.sh --setup-ssh` or `AINODE_PEERS=... install.sh`.  
The container mounts the host's `~/.ssh` at `/host-ssh:ro` and copies it to `/root/.ssh` at startup.

### Image supply chain

- Base image: `ghcr.io/getainode/ainode-base:c026c92` (pinned SHA, built from eugr/spark-vllm-docker)
- AINode image: built on Spark 1 (self-hosted aarch64 runner), tagged by version
- Both are public on GHCR and Docker Hub — no auth required to pull
- **Never push** unreviewed code to `main` — all changes via PR

---

## See Also

- `ops/runbooks/release-flow.md` — detailed build + push + rollback procedures
- `ops/runbooks/dev-workflow.md` — branch conventions, slice lifecycle
- `ops/runbooks/2026-04-14-sharding-lessons-learned.md` — distributed inference lessons
- `CHANGELOG.md` — full version history
- `README.md` — user-facing docs + networking requirements
- Memory files in `.claude/projects/.../memory/` — session context
