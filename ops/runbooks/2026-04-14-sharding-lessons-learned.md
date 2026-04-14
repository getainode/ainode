# AINode Distributed Sharding — Lessons Learned

**Date:** 2026-04-14
**Duration:** ~6 hours of focused work
**Outcome:** ✅ First real cross-node tensor-parallel inference on this hardware
**Authors:** Claude (Opus 4.6, 1M context) + Jason Brashear

This document captures the full journey: what worked, what didn't, why, and
what to do differently next time. Written immediately after the first
successful `{"content":"Hello! How can I assist you today?"}` came out of a
model sharded across two DGX Sparks.

---

## 1. Scope & goal

Build AINode into a local-AI cluster platform that can:

1. Install with one command (`curl | bash`)
2. Auto-discover cluster peers
3. Serve inference from browser UI
4. **Distribute one large model across multiple GB10 GPUs** so combined VRAM
   unlocks models that don't fit on a single node

The last item is what this session focused on. Items 1–3 were already working
from earlier days.

## 2. Hardware

| Node | Tailscale | Local fabric | GPU | OS |
|---|---|---|---|---|
| ASUS GX10 | 100.72.9.84 | enP7s7 on 192.168.0.112 | 1× NVIDIA GB10, 122 GB unified | Ubuntu 24.04, CUDA 13.0 |
| DGX Spark 1 | 100.122.26.9 | enP7s7 on 192.168.0.110, **enp1s0f0np0 on 10.0.0.1** | 1× GB10, 122 GB | Ubuntu 24.04, CUDA 13.0 |
| DGX Spark 2 | 100.81.184.19 | enP7s7 on 192.168.0.111, **enp1s0f0np0 on 10.0.0.2** | 1× GB10, 122 GB | Ubuntu 24.04, CUDA 13.0 |

Direct-connect QSFP cable exists only between Spark 1 ↔ Spark 2
(`enp1s0f0np0`, 10.0.0.0/24). GX10 has CX7 ports but they're unwired; it's
reachable only on 192.168.0.0/24 via switched Ethernet.

Storage:
- MikroTik RDS2216 ROSA array exports NVMe-over-TCP namespaces (block-level)
- Spark 1's namespace mounts at `/mnt/rosa-storage` (14 TB, healthy)
- Spark 2's namespace mounts at `/mnt/rosa-storage` (14 TB)
- `/mnt/rosa-models` on Spark 1 is **corrupt** (ext4 bitmap checksum errors —
  see `project_rosa_models_fsck_todo.md` in memory)

## 3. What worked (AINode platform, ignoring sharding)

All of this shipped and is stable:

- Install one-liner → venv + image pull + compose + systemd + linger (v0.3.0)
- Docker-based vLLM engine on GB10 (scitrera/dgx-spark-vllm image)
- UDP broadcast discovery on port 5679 (EXO-safe; 5678 was the original
  collision with EXO)
- UI showing all 3 nodes with hostnames + 365 GB aggregated VRAM
- Shared NFS model storage: Spark 1 exports `/mnt/rosa-storage`,
  Spark 2 + GX10 mount as `/mnt/rosa-shared`. Download once, use everywhere.
- `/v1/embeddings` endpoint (sentence-transformers, OpenAI-compatible)
- Hot model swap via `POST /api/models/load` (recreates container with new
  `AINODE_MODEL` env)
- Server view with per-node model tracking, eject, swap, embed types
- Config panel, training wizard, chat view, download manager

## 4. The sharding journey — every attempt chronologically

### 4.1. Attempt: stock scitrera image, direct `docker run` + ray start

**Setup:** Manually started ray head on Spark 1, ray workers on Spark 2 + GX10.
Then `docker exec vllm serve ... --tensor-parallel-size 3
--distributed-executor-backend ray`.

**Result:** Hang. Log stops at:
```
WARNING ray_utils.py:269 tensor_parallel_size=3 is bigger than a reserved
number of GPUs (1 GPUs) in a node ... Tensor parallel workers can be spread
out to 2+ nodes...
```

Never progresses past placement-group creation. GPUs show nothing allocated.
No NCCL output at all.

**Why:** We didn't yet know — thought it was an NCCL init issue. It was
actually downstream of NCCL — the worker spawn itself.

### 4.2. Attempt: eugr/spark-vllm-docker image, same direct invocation

**Setup:** Built eugr's community image (12 min, 17.5 GB, ships custom NCCL
patch `dgxspark-3node-ring`). Same TP=3 over 3 nodes.

**Result:** Identical hang at identical line.

**Conclusion:** The eugr NCCL patches fix NCCL behavior *after* workers start.
We weren't reaching NCCL at all. Changing the image didn't help because the
stall was in vLLM's own Ray placement-group logic before NCCL init.

### 4.3. Attempt: PP=3 TP=1 instead of TP=3

**Rationale:** Pipeline-parallel splits layers, not tensors. With 1 GPU per
node, 1 worker per node, should avoid the "TP > GPUs per node" warning.

**Result:** Same hang. Different warning, same spot.

### 4.4. Attempt: `VLLM_DISTRIBUTED_EXECUTOR_CONFIG='{"placement_group_options":{"strategy":"SPREAD"}}'`

**Rationale:** Expert-recommended: force SPREAD instead of PACK so bundles
explicitly distribute to different nodes.

**Result:** Got **further** — log reached 141 lines before dying. Clear error
surfaced:
```
ValueError: Current node has no GPU available. current_node_resource={...
  'accelerator_type:GB10': 1.0, 'bundle_group_...': 999.999}
```

**Cause:** Previous hung run had reserved all GPUs in a placement group that
wasn't released when I `pkill`d vLLM. Ray held the reservation even though
the owning process was gone.

**Fix:** Full teardown (`docker rm` all ray containers on all nodes) before
re-launching.

### 4.5. Attempt: SPREAD on clean cluster

**Result:** Stall again at the 28-line mark. SPREAD wasn't the fix.

### 4.6. Attempt: eugr's `launch-cluster.sh` with 3 nodes on 192.168.0.x

**Setup:** Their SSH-orchestrated multi-node launcher. Needed to:
- Set up passwordless SSH between nodes (done)
- Handle subnet-collision check (3 NICs per host all on 192.168.0.0/24)
- Fix SSH host-key algorithm mismatch (`sk-*` FIDO2 keys) → pinned
  `HostKeyAlgorithms ssh-ed25519,ssh-rsa,rsa-sha2-*,ecdsa-*` in
  `~/.ssh/config`

**Result:** Identical stall. Even their production-tested orchestrator hit
the same wall on our topology.

**What this proved:** The problem was NOT AINode, NOT raw docker+ray
invocation, NOT env vars. The problem was intrinsic to the networking
topology or vLLM's worker-spawn logic.

### 4.7. Attempt: 2-node TP=2 on 192.168.0.x

**Rationale:** Power-of-2 is better supported. 2 nodes reduces complexity.

**Result:** **First meaningful progress.** Log reached "Copying env vars to
workers". RayWorkerWrappers spawned on both nodes. GPU memory allocated
(~289 MB and 375 MB — CUDA context only, not weights). Workers stuck in
`ncclAllReduce` per py-spy stack dump.

### 4.8. Diagnosis: py-spy the stuck worker

```
Thread 287 (active): "MainThread"
    ncclAllReduce (pynccl_wrapper.py:440)
    all_reduce (pynccl.py:177)
    __init__ (pynccl.py:144)
    __init__ (cuda_communicator.py:75)
    __init__ (parallel_state.py:376)
    ...
    init_worker_distributed_environment (gpu_worker.py:1048)
    init_device (gpu_worker.py:263)
```

NCCL collective was hanging. Added `CONTAINER_NCCL_DEBUG=INFO` to the eugr
`.env` to see why.

### 4.9. Attempt: with NCCL_DEBUG; discovered VLLM_HOST_IP collision

**Result:** Clean error:
```
RuntimeError: Every node should have a unique IP address. Got 2 nodes with
node ids [...] and 1 unique IP addresses {'192.168.0.110'}.
```

**Cause:** I had set `CONTAINER_VLLM_HOST_IP=192.168.0.110` in the `.env`
which applied that same IP to every node's container. vLLM detected
the collision and refused to start.

**Fix:** Let eugr auto-detect VLLM_HOST_IP per-node from ETH_IF.

### 4.10. Attempt: without HOST_IP override, NCCL_DEBUG on

**Result:** Got **much further**. Workers fully initialized. NCCL topology
came out:
```
NCCL INFO NCCL version 2.29.7+cuda13.2
NCCL INFO NCCL git version dgxspark-3node-ring fab1850
NCCL INFO NET/Socket : Using [0]enP7s7:192.168.0.111<0>
NCCL INFO comm 0x43803750 rank 1 nRanks 2 nNodes 2
NCCL INFO Ring 00 : 0 -> 1 -> 0
```

But the first `ncclAllReduce` still hung.

**Diagnosis (the real one):** Checked `ss -tn state established` on both
nodes. Found:
- Spark 1 → Spark 2 outgoing used **`192.168.0.188`** (source IP on
  `enp1s0f1np1` — ConnectX-7 port, NOT the configured `enP7s7`)
- Spark 2 → Spark 1 used `192.168.0.219` similarly

**Root cause:** Linux kernel routing picks one of the three interfaces on
`192.168.0.0/24` as the egress. NCCL told workers to use `enP7s7` for
listening, but kernel decided to send outbound via `enp1s0f1np1`. NCCL's ring
handshake expects source IP to match destination IP family, saw mismatch,
hangs silently. No timeout by default.

### 4.11. Attempted fix: `ip link set dev enp1s0f1np1 down` on both Sparks

**Result:** Both Sparks lost Tailscale connectivity. SSH access via
`100.x.x.x` gone for ~3 minutes. Had to jump through GX10 (still reachable)
and `sshpass`-chain into the Sparks via `192.168.0.110/111` to `ip link
set ... up` and recover.

**Lesson:** Taking NICs down remotely without a local console is
**dangerous**. Tailscale's direct-wireguard connections depend on specific
interface routes; kicking out the interface it was using can orphan the
node until wireguard rediscovers a new path. Never script this as part of
`curl | bash install.sh`.

### 4.12. The actual fix: use the direct-connect 10.0.0.x fabric

**Setup:**
```
# ~/spark-vllm-docker/.env
CLUSTER_NODES=10.0.0.1,10.0.0.2
ETH_IF=enp1s0f0np0
IB_IF=enp1s0f0np0
MASTER_PORT=29501
CONTAINER_NCCL_DEBUG=INFO
CONTAINER_NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
CONTAINER_NCCL_SOCKET_IFNAME=enp1s0f0np0
CONTAINER_NCCL_IB_HCA=mlx5_0
CONTAINER_NCCL_IB_DISABLE=0
CONTAINER_NCCL_IGNORE_CPU_AFFINITY=1
CONTAINER_UCX_NET_DEVICES=enp1s0f0np0
```

The `enp1s0f0np0` interface is the **direct QSFP cable** between Spark 1
and Spark 2 — no other interface has an IP on `10.0.0.0/24`, so Linux
routing has exactly one choice.

**Result:** ✅ **IT WORKED.**

```
(EngineCore) INFO init engine (profile, create kv cache, warmup model) took 10.57 seconds
(APIServer) INFO api_server.py:600 Supported tasks: ['generate']
```

GPU memory on both nodes: 61,311 MB (61 GB each) — model weights actually
sharded.

```
curl http://10.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct",
       "messages":[{"role":"user","content":"hi"}],"max_tokens":10}'

→ {"choices":[{"message":{"content":"Hello! How can I assist you today?"}}]}
```

29 seconds for 10 tokens — slow, but real.

## 5. Lessons learned (in priority order)

### 5.1. **Single NIC per cluster subnet.** Non-negotiable.
Multi-NIC on same subnet breaks NCCL silently because Linux routing is
non-deterministic for egress selection. Either use direct-connect cables
on their own `/30` or use a dedicated cluster switch on its own `/24`.

### 5.2. **Tailscale depends on underlying NICs — never `ip link down` them remotely.**
If Tailscale is running direct-wireguard (not relayed), it's using specific
interface routes. Taking those down orphans the node. Only do NIC changes
with local console access OR script them to auto-restore after a timeout.

### 5.3. **Each node needs a unique `VLLM_HOST_IP`.**
eugr's `.env` pattern applies the same `CONTAINER_VAR` to every node. Don't
put `VLLM_HOST_IP` in there. Let eugr auto-detect per node from `ETH_IF`.

### 5.4. **Ray placement-group cleanup is fragile.**
If a vLLM process is killed with SIGKILL, its Ray placement group can remain
reserved forever. Subsequent attempts fail with "no GPU available" even
though `nvidia-smi` shows no processes. Always `docker rm` all ray
containers for a full teardown.

### 5.5. **vLLM's `ray_utils` warning "TP > GPUs per node" is misleading.**
It warns but then *does* create a cross-node placement group. It doesn't
block — but the workers that get spawned then fail silently at actual
NCCL collective ops if the network is mis-configured.

### 5.6. **`NCCL_DEBUG=INFO` is only helpful if it reaches the workers.**
eugr's launch script copies a whitelist of env vars via `ray_env.py`.
`NCCL_DEBUG` is in the "NCCL_" prefix so it propagates — but only if set
in the eugr `.env` file with the `CONTAINER_` prefix, not just in the host
shell.

### 5.7. **ssh host-key algorithms matter when the client has FIDO2 keys.**
Spark 1's default OpenSSH config offered `sk-ecdsa-sha2-nistp256` as a host
key algorithm. GX10's sshd doesn't have that key type. Result: "Connection
closed by <host> port 22" with no useful error. Fix: pin
`HostKeyAlgorithms ssh-ed25519,ssh-rsa,rsa-sha2-256,rsa-sha2-512,ecdsa-sha2-nistp256`
in `~/.ssh/config` for cluster peer IPs.

### 5.8. **Build once, pull everywhere.**
eugr's image took 12 minutes to compile (NCCL + vLLM). Future end users
should never compile this — build it once in our GitHub Actions (self-hosted
aarch64 runner on a Spark), push to GHCR + Docker Hub, and install.sh
pulls the prebuilt tag.

### 5.9. **ext4 on NVMe-oF can silently corrupt.**
Spark 1's `/mnt/rosa-models` had bitmap checksum errors. Large writes
returned all-zero on read. `dmesg` showed the corruption. AINode had to
be moved to the healthy `/mnt/rosa-storage` (14 TB, same ROSA array,
different namespace). Track in memory: needs `e2fsck` with the volume
unmounted. See `project_rosa_models_fsck_todo.md`.

### 5.10. **NFS from master beats block-level shared storage.**
NVMe-oF + ext4 is not safe for concurrent multi-writer. We layered NFS
from Spark 1 (the master) to the workers. Each AINode instance sees the
same models directory via `/mnt/rosa-shared`. Download once, use everywhere.

## 6. What's still broken / not yet done

### 6.1. Cross-node inference is slow (29 s for 10 tokens)
NCCL reported socket transport at 1.2 GB/s. ConnectX-7 should be ~200 Gbps
via RoCE/RDMA. `NCCL_IB_HCA=mlx5_0` was set but we didn't confirm RDMA
was actually used. Likely need:
- `NCCL_NET_PLUGIN=none` (disable socket fallback)
- Enable RoCE on `enp1s0f0np0` (`ibstat mlx5_0` should show "Active")
- Possibly `NCCL_IB_SUBNET_AWARE_ROUTING=1`
Investigate next session. Target: sub-second first-token for 1.5 B model.

### 6.2. 3-node TP never worked
GX10 has no direct-connect fabric — its `enp1s0f0np0` is DOWN. Without
wiring the triangle (A↔B, B↔C, C↔A QSFP cables) or installing a dedicated
switch, 3-node will keep hitting the multi-NIC routing ambiguity.

### 6.3. 4-node TP is the next planned target
When Spark #4 arrives + QSFP switch installed: put all 4 Sparks on a
dedicated /24 on the switch (e.g. `10.100.0.0/24`), try TP=4 with the same
eugr config pattern. Power-of-2 is well-supported by the community.

### 6.4. Sharding not yet wired into AINode UI
Current state: you launch distributed inference manually via `launch-cluster.sh`
on Spark 1. The AINode "Launch Instance → Minimum Nodes > 1" button
doesn't know how to drive that flow yet. Phase 2 work.

### 6.5. Ray autostart in AINode is disabled
Commit `287bafc` added a guard that skips Ray worker-join when the master's
hostname is bogus. That guard is too aggressive — it blocks all cross-node
Ray. Remove once peer-IP discovery (Task #10) is in and Ray orchestration
is driven by eugr rather than our own code.

### 6.6. AINode's own discovery broadcasts over `enP7s7` (192.168.0.x)
That's the regular Ethernet subnet. For the *user experience* (cluster view,
model swap), this is fine — it just needs to reach peers.
The **distributed inference fabric** is separate (direct-connect 10.0.0.x
or future switch subnet). Two different networks serving two different
purposes. Don't conflate them.

### 6.7. vLLM V1 engine forces async scheduling off when using Ray
Async scheduling not supported with `ray` distributed backend — only `mp`,
`uni`, `external_launcher`. We accept this as a known trade-off for
cross-node TP.

## 7. The working config (canonical reference)

Captured in memory: `~/.claude/projects/-Users-sem-code-ainode/memory/project_sharding_working_config.md`

In the repo: see §10.7 of `ops/runbooks/docker-engine-deploy.md` for the
smoke test; a separate §11 will add the sharded-launch runbook once
Phase 2 wires it into AINode.

## 8. Demo-ready state (as of 2026-04-14 end of day)

| Capability | Status |
|---|---|
| 1-command install on any GB10 Spark/GX10 | ✅ |
| 3-node cluster discovery + UI | ✅ |
| Hot model swap from UI | ✅ |
| Embeddings endpoint | ✅ |
| NFS shared model cache | ✅ |
| Independent-node inference per box | ✅ |
| **2-node distributed inference (TP=2) — manual invoke** | ✅ (today) |
| 3/4-node distributed inference | ❌ (needs switch or triangle mesh) |
| AINode UI drives distributed launch | ❌ (Phase 2) |

For the demo: show the UI, show independent inference, show the just-proven
distributed TP=2 via CLI. Mention 4-node with the switch as the next
milestone. This is an honest and impressive story.

## 9. Post-mortem on my own approach

Things I did poorly:
- Declared victory twice on sharding before verifying GPU memory allocation
- Added `CONTAINER_VLLM_HOST_IP` without thinking that every node would get
  the same value — wasted ~20 min on a self-inflicted error
- Ran `ip link set dev ... down` over SSH without appreciating the Tailscale
  dependency — locked myself out of two nodes for 3 minutes

Things that went right:
- Caught the stale placement-group issue by reading the resource dict
- Used py-spy to identify the exact stuck frame
- Read `ss -tn` output to spot the routing-mismatch symptom
- Escalated to GX10 as a bastion when Tailscale broke instead of panicking

Things to do differently next time:
- **Verify GPU memory allocation before claiming "it works"** — CUDA context
  (~300 MB) is not the same as model weights (~60 GB).
- **Before any network config change**: confirm the change is reversible
  without local console access, or pre-arrange a bastion.
- **Always test single-node first.** We should have confirmed `tp=1` on one
  node + SSH-based launch path end-to-end before adding Ray + multi-node.

---

*"The problem was not AINode, NCCL, Ray, or vLLM. It was Linux kernel
routing selecting a different interface than NCCL was configured for. The
fix was a dedicated single-NIC subnet."*
