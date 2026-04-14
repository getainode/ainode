# AINode

### Turn any NVIDIA GPU into a local AI platform.

**Inference + fine-tuning in your browser. One command to start. Add a second GPU, they find each other automatically.**

```bash
curl -fsSL https://ainode.dev/install | bash
```

### Architecture

AINode ships as **one container per node**: web UI, API, vLLM engine, and
cross-node orchestrator are version-locked in a single image pulled from
`ghcr.io/getainode/ainode`. No host Python venv to maintain. The systemd
unit runs `docker run ... ainode` once per host.

Distributed mode (tensor/pipeline-parallel across nodes) is production on
2-node setups and experimental on 3+ nodes until a QSFP switch or mesh
topology is in place.

**Image footprint:** ~18 GB (includes CUDA 13 + patched vLLM + Ray + NCCL
tuned for NVIDIA GB10). One-time pull per node.

**Security note:** the systemd unit bind-mounts `/var/run/docker.sock` into
the container so the head node can launch peer workers over SSH. That
grants the container root-equivalent control of the host Docker daemon,
appropriate for a dedicated AI-appliance node. Don't run AINode alongside
untrusted workloads on the same box.

```
  ┌──────────────────────────────────────┐
  │         Welcome to AINode            │
  │   Your local AI platform for NVIDIA  │
  ├──────────────────────────────────────┤
  │                                      │
  │   Detected: NVIDIA GB10 · 128 GB     │
  │                                      │
  │   ✓ vLLM engine ready                │
  │   ✓ API server on :8000              │
  │   ✓ Web UI on :3000                  │
  │                                      │
  │   Open: http://localhost:3000         │
  │                                      │
  └──────────────────────────────────────┘
```

---

## What Is This?

AINode is a free, open-source tool that turns NVIDIA GPU systems into a complete local AI platform — chat, inference API, and model fine-tuning — all from your browser.

**Built for:**
- NVIDIA DGX Spark
- ASUS AI Server (GB10)
- Dell AI Factory
- HP AI Workstations
- Any system with an NVIDIA GPU and Linux

**No Linux expertise required.** Install, open your browser, start using AI.

---

## Features

| Feature | Status |
|---|---|
| One-command install | ✅ |
| Auto-detect GPU and memory | ✅ |
| Chat UI in your browser | ✅ |
| OpenAI-compatible API | ✅ |
| Model fine-tuning from browser | 🔜 |
| Multi-node auto-discovery | 🔜 |
| Automatic model sharding across nodes | 🔜 |

---

## Getting Started — Step by Step

### Single node (solo mode)

1. **Install Docker** + NVIDIA container toolkit on your Linux box.
2. **Pull the image** and wire up the systemd unit:
   ```bash
   curl -fsSL https://ainode.dev/install | bash
   ```
3. **Open the UI** at `http://<your-ip>:3000`. First-run onboarding walks you
   through picking a model. Click a model card → click **Launch** → chat.

That's it. The systemd service auto-starts on boot. To update:
```bash
sudo docker pull ghcr.io/getainode/ainode:latest
sudo systemctl restart ainode
```

### Two nodes (distributed mode)

For models that don't fit on a single GPU — e.g. a 70 B+ model sharded
across two DGX Sparks — you need:

1. **A clean high-speed link between the two nodes.** Simplest is a
   direct QSFP cable with both ports on their own subnet (e.g.
   `10.0.0.1/24` and `10.0.0.2/24`). If you have a switch, dedicate a
   VLAN/subnet to the cluster and plug only one NIC per node into it.
   See [Networking requirements](#networking-requirements) below — this
   **matters a lot**.
2. **Install AINode on both nodes** (step 1 above).
3. **On the peer node**, set member mode in `~/.ainode/config.json`:
   ```json
   {
     "distributed_mode": "member",
     "cluster_interface": "enp1s0f0np0",
     "ssh_user": "sem"
   }
   ```
   Restart: `sudo systemctl restart ainode`. The peer now announces
   itself on UDP 5679 and waits for work.
4. **On the head node**, set head mode:
   ```json
   {
     "distributed_mode": "head",
     "peer_ips": ["10.0.0.2"],
     "cluster_interface": "enp1s0f0np0",
     "ssh_user": "sem"
   }
   ```
   Add passwordless SSH from head → peer: `ssh-copy-id sem@10.0.0.2`.
   Restart: `sudo systemctl restart ainode`.
5. **Open the head UI.** You should see both nodes, aggregated VRAM
   ("2 nodes · 244 GB · 2 GPUs"), and a model instance badged as
   **DISTRIBUTED · TP=2**.

If the UI shows only one node, see [Troubleshooting](#troubleshooting).

---

## How It Works

```
┌─────────────────────────────────────────────────────┐
│  AINode                                              │
│  ┌─────────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Web UI      │  │  REST API │  │  CLI          │  │
│  │  Chat +      │  │  OpenAI-  │  │  ainode start │  │
│  │  Fine-tune   │  │  compat   │  │  ainode train │  │
│  └──────┬──────┘  └────┬─────┘  └───────┬───────┘  │
│         │              │                │            │
│  ┌──────▼──────────────▼────────────────▼───────┐   │
│  │  Orchestrator                                 │   │
│  │  Node discovery · Model routing · Sharding    │   │
│  └──────────────────────┬───────────────────────┘   │
│                         │                            │
│  ┌──────────────────────▼───────────────────────┐   │
│  │  vLLM Engine                                  │   │
│  │  PagedAttention · Continuous batching · CUDA  │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## State of Distributed Inference (April 2026)

Being honest about what's real and what's not — because we spent the first
few weeks of this project being quietly wrong about it and would like to
save you the same pain.

### What works today (verified on real hardware)

- **Single-node inference** on any NVIDIA GB10 box (DGX Spark, ASUS GX10).
- **Two-node tensor-parallel** (TP=2) with one GPU per node, split over a
  direct-connect QSFP cable on its own `/24`. Weight shards are visible on
  both GPUs simultaneously (`nvidia-smi` on each host shows ~61 GB of
  `ray::RayWorkerWrapper` memory for a small model).
- **One-container-per-node install** — `docker pull ghcr.io/getainode/ainode`
  → systemd → done. No host Python venv, no vLLM source build.
- **Shared model storage over NFS** exported from a master node backed by
  NVMe-over-TCP. Download a model once, all nodes load it from the same
  files.
- **Cluster discovery over UDP** on port 5679 — the head sees peers, the UI
  aggregates VRAM, and the "DISTRIBUTED · TP=N across K nodes" badge isn't
  a cosmetic lie: it reflects the real placement group.

### What doesn't work today

- **Three-node TP=3 on our current network topology.** See [Why 3 nodes is
  harder than 2 (and why 4 nodes is probably easier)](#why-3-nodes-is-harder-than-2-and-why-4-nodes-is-probably-easier)
  below.
- **Automatic topology detection.** You still have to tell AINode which
  interface is the cluster fabric (`cluster_interface`) and which IPs the
  peers have (`peer_ips`). The install script is wired for it; there's just
  no auto-discovery of the *right* subnet when a host has multiple NICs in
  the same IP range.
- **Cross-node Ray over Tailscale.** The tunnel works fine for SSH and
  small coordination traffic, but NCCL's default transport doesn't
  negotiate well over a VPN. Use a dedicated physical link.
- **One-click "add me to the cluster" UX.** You currently hand-edit
  `config.json` to set `distributed_mode: "member"`. Coming soon — see
  the UI launch-flow wiring.

### What we learned the hard way

1. **Single NIC per cluster subnet.** If you have three interfaces all
   holding IPs on `192.168.0.0/24` (onboard Ethernet + two ConnectX-7
   ports), Linux kernel routing picks one non-deterministically for
   outbound traffic. NCCL listens on the interface you told it to via
   `NCCL_SOCKET_IFNAME`, but the peer's SYN-ACK may arrive from a
   different interface on the same subnet. The TCP handshake completes;
   the NCCL ring handshake silently hangs forever. This was our single
   biggest time sink before we understood it.

2. **Ray placement group reservations survive SIGKILL.** If you kill a
   hung vLLM, Ray's GCS still considers the GPUs reserved for the dead
   placement group. Subsequent launches fail with "no GPU available" even
   though `nvidia-smi` shows nothing running. Full teardown is
   `docker rm -f` on every ainode + ray container across every node.

3. **Block-level shared storage is unsafe for multi-writer.** A NVMe-oF
   LUN exported to two hosts and mounted with ext4 on both will silently
   corrupt under concurrent writes. Use NFS on top of a single-host
   mount. (We now do.)

4. **The NVIDIA Spark image ships stock NCCL**; the eugr community image
   ships a patched NCCL (`dgxspark-3node-ring` branch) that actually
   handles GB10 unified-memory topologies. Our base image inherits that
   patched NCCL — don't use stock.

5. **SSH from a root-ID container into a host-user account** fails
   silently when the keys are mounted read-only from the host. OpenSSH
   refuses to use keys it didn't create with the right ownership. Our
   entrypoint copies the mount at `/host-ssh` into `/root/.ssh` with
   correct perms + injects `User <ssh_user>` into `ssh_config` for the
   peer IPs.

### Current challenges

- **Three nodes on all-same-subnet topology** — see next section.
- **Throughput on cross-node TP** is still gated by NCCL falling back to
  Socket transport (~1.2 GB/s) instead of RoCE RDMA (~25 GB/s on
  ConnectX-7). RoCE works on the link — `ibstat mlx5_0` shows "Active"
  and "Rate: 200" — but vLLM's init path hasn't always picked it up.
  Tuning `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, and `NCCL_NET_GDR_LEVEL=5`
  is on the next slice.
- **UI "Launch Instance" button** currently triggers a single-node load.
  Wiring `Minimum Nodes > 1` + `Tensor/Pipeline` selector to the
  distributed path is coming in the next UI slice.

### Why 3 nodes is harder than 2 (and why 4 nodes is probably easier)

**Two nodes**: the natural topology is a single direct-connect cable on a
`/24` between exactly two hosts. One cable, one subnet, one candidate
interface per host. NCCL cannot get confused about which NIC to use. TP=2
means each node holds half the weights. This is a solved problem.

**Three nodes**: there's no simple physical topology. You need one of:

- A **triangle mesh** (A↔B, B↔C, A↔C) of three direct-connect cables
  with each link on its own `/30`. That means three separate subnets,
  which the community tooling (eugr's `launch-cluster.sh`, NCCL's IB
  plugin) assumes is present but nobody autoconfigures for you.
- A **dedicated cluster switch** with one NIC per node plugged in on an
  isolated subnet. This is clean — it's actually easier than the 3-node
  mesh — but it's a hardware purchase.
- A **star topology** where one of the nodes acts as a hub. Adds a hop
  and asymmetric latency; not recommended.

If your three nodes share a regular LAN subnet (`192.168.x.x` through a
switch), you hit the multi-NIC routing ambiguity described in Lessons
Learned #1. We did, hard. NCCL ring setup succeeded on paper; actual
data never flowed.

**Four nodes**: paradoxically simpler once you commit to a dedicated
switch — which is the only practical topology for 4+ nodes anyway. You
buy one QSFP switch (or partition a MikroTik RDS), plug one NIC per node
into it on a fresh `/24`, and you're back to the "single NIC per cluster
subnet" rule that works at 2 nodes. TP=4 is also a power-of-two, which
lines up nicely with vLLM's default sharding heuristics. Most of the
community's 4× Spark recipes (eugr, PFN, NVIDIA samples) assume this
topology.

**Our hypothesis:** the difficulty is not *N* nodes — it's *how you wire
N nodes*. Two nodes are easy because one cable makes the decision for
you. Three nodes force you to make a topology decision (mesh vs switch
vs star) and the mesh path is poorly documented. Four-or-more nodes are
easy *again* because a switch is the only rational answer and the
community has tooling for it.

When our fourth Spark arrives and the dedicated switch is wired, we
expect TP=4 on a clean `/24` to light up in about 30 minutes of config
— based on eugr's `recipes/4x-spark-cluster/` reports and Shibata-san's
published runs.

### Troubleshooting

**The UI shows only 1 node on the head.** Your peer isn't broadcasting.
Check on the peer:
```bash
sudo docker exec ainode cat /root/.ainode/config.json | grep distributed_mode
sudo docker logs ainode | tail -20
```
Expect `"distributed_mode": "member"` and a "Member mode — awaiting
work from the cluster head" line.

**Head starts but `docker logs ainode` says "SSH to \<peer\> failed".**
Passwordless SSH from head's host user → peer isn't working yet. On
the head: `ssh-copy-id sem@<peer-ip>` and test `ssh sem@<peer-ip> true`.

**NCCL hangs indefinitely in `ncclAllReduce`.** 99% chance it's the
multi-NIC routing issue. On each node run
`ip -br addr | grep <cluster-subnet>` — if more than one interface shows
an IP in that subnet, pick one and put the others on their own subnet
or take their IP off.

**I see the cluster but chat is slow.** NCCL is using Socket transport.
Run `docker logs ainode 2>&1 | grep "Using network"` inside an inference
request — should say `Using network IB`, not `Using network Socket`.
Open an issue with your `ibstat mlx5_0` output.

---

## Networking requirements

AINode relies on NCCL for cross-node tensor-parallel communication, and
NCCL works best when it owns a clean link. Our minimum requirements:

- **Passwordless SSH** from the head node's host user account into every
  peer node, on the cluster subnet (not through a jump host).
- **Single active NIC per cluster subnet** on every node. If you have
  multiple ConnectX-7 ports plugged in and sitting on the same `/24`,
  you will hit the NCCL ring-hang described in Lessons Learned.
- **No VPN between nodes for cluster traffic.** Tailscale is fine for
  `ssh` from your laptop; not fine as the NCCL transport. Use physical
  cables or a dedicated switch port.
- **Consistent MTU.** If one node has `MTU 9000` (jumbo) and the other
  has `1500`, large NCCL messages silently truncate on the way through
  an intermediary switch.

### Recommended topologies

| Cluster size | Topology | Notes |
|---|---|---|
| 2 nodes | Direct QSFP cable, each end on its own IP in a fresh `/24` | Simplest, verified working |
| 3 nodes | **Triangle direct-connect** (3 cables, each link on its own `/30`) or a dedicated switch | Mesh is finicky; switch is easier |
| 4+ nodes | Dedicated QSFP switch on its own `/24`, one NIC per node | The only practical option |

### Diagnostic commands

```bash
# Confirm RDMA is live on your ConnectX-7
ibstat mlx5_0 | grep -E "State|Rate"
# Expect: "State: Active" + "Rate: 200" (Gb/s)

# Confirm exactly one interface has an IP in your cluster subnet
ip -br -4 addr | grep 10.0.0
# Expect exactly one line on each node (e.g. "enp1s0f0np0 ... 10.0.0.1/24")

# Confirm the peer is reachable on the cluster subnet (not via Tailscale!)
ping -c 2 10.0.0.2
traceroute 10.0.0.2    # 1 hop means you're on the right link

# Confirm passwordless SSH works as your ainode ssh_user
ssh sem@10.0.0.2 true && echo OK
```

---

## Shared Model Storage Across a Cluster

Downloading a 70 GB model three times on a three-node cluster is wasteful. AINode
supports a shared models directory so every node in the cluster pulls from the
same cache.

**Why this isn't automatic:** block-level shared storage (NVMe-over-TCP, iSCSI,
Fibre Channel) is great for speed but **unsafe for multiple Linux kernels to
write at the same time** — ext4/XFS have no distributed lock manager. Use a file
protocol on top:

```
  Storage array (NVMe-oF, SAN, local NVMe, …)
          │
          ▼
   MASTER NODE  ← ext4/XFS mounted here, owns the disk
     │   │
     │   └── NFS server exports /mnt/ai-models
     ▼
  WORKERS       ← mount the NFS share at /mnt/ai-shared
```

### Set it up (two commands per side)

**On the master:**
```bash
sudo apt-get install -y nfs-kernel-server
echo "/mnt/ai-models <worker1-ip>(rw,sync,no_subtree_check,no_root_squash) \
                     <worker2-ip>(rw,sync,no_subtree_check,no_root_squash)" \
  | sudo tee -a /etc/exports
sudo exportfs -ra && sudo systemctl enable --now nfs-kernel-server
```

**On each worker:**
```bash
sudo apt-get install -y nfs-common
sudo mkdir -p /mnt/ai-shared
echo "<master-ip>:/mnt/ai-models /mnt/ai-shared nfs \
  rw,noatime,nodiratime,rsize=1048576,wsize=1048576,nconnect=16,hard,vers=4.2,_netdev 0 0" \
  | sudo tee -a /etc/fstab
sudo mount /mnt/ai-shared
```

Then point AINode at the shared path: edit `~/.ainode/docker-compose.yml` volume
line from `- ${HOME}/.ainode/models:/models` to `- /mnt/ai-models:/models`
(master) or `- /mnt/ai-shared:/models` (workers). Restart the service.

**Performance note:** NFS over a 100G fabric typically reads at 3–8 GB/s — vLLM
model loading is a one-shot sequential read, so you won't notice for most
models. For 100 GB+ models where load time hurts, add an rsync-to-local staging
step on the worker.

---

## Supported Models

AINode works with any model vLLM supports. Some popular choices:

| Model | Memory Needed | Nodes |
|---|---|---|
| Llama 3.2 3B | ~6 GB | 1 |
| Llama 3.1 8B | ~16 GB | 1 |
| Llama 3.1 70B (4-bit) | ~35 GB | 1 |
| Qwen 2.5 72B | ~40 GB | 1 |
| Llama 3.1 405B (4-bit) | ~200 GB | 2 |
| DeepSeek V3 (671B) | ~350 GB | 2+ |

---

## CLI Reference

```bash
ainode start              # Start AINode (inference + web UI)
ainode stop               # Stop AINode
ainode status             # Show cluster status
ainode models             # List available models
ainode train              # Launch fine-tuning UI
ainode join               # Join an existing cluster
ainode config             # Edit configuration
```

---

## API

AINode exposes an OpenAI-compatible API. Use it with any tool that speaks OpenAI:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="llama-3.1-8b",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

Works with Open WebUI, LiteLLM, LangChain, and anything else that uses the OpenAI SDK.

---

## Requirements

- **OS:** Ubuntu 22.04+ (including DGX Spark OS)
- **GPU:** Any NVIDIA GPU with CUDA support
- **Memory:** 8 GB+ GPU memory (128 GB recommended for large models)
- **Disk:** 20 GB+ for models
- **Python:** 3.10+

---

## Why AINode?

| | Cloud AI (OpenAI, etc.) | AINode |
|---|---|---|
| Monthly cost | $100-10,000+ | $0 (you own the hardware) |
| Data privacy | Your data on their servers | Your data stays local |
| Rate limits | Yes | No |
| Latency | 200-2000ms | 10-50ms |
| Fine-tuning | Limited, expensive | Unlimited, free |
| Internet required | Yes | No |
| Models available | Their choice | Your choice |

---

## Roadmap

- [x] Core CLI and installer
- [x] vLLM integration
- [x] Web UI (chat)
- [ ] Browser-based fine-tuning
- [ ] Multi-node auto-discovery
- [ ] Automatic model sharding
- [ ] Training job dashboard
- [ ] Model management (download, delete, organize)
- [ ] Prometheus metrics endpoint
- [ ] Mobile-friendly UI

---

## Contributing

AINode is open source and welcomes contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

Apache 2.0 — use it however you want.

---

<p align="center">
  <sub>Powered by <a href="https://argentos.ai">argentos.ai</a></sub>
</p>
