<!--
AINode: local AI platform for NVIDIA GB10 and any NVIDIA GPU server.
Keywords: NVIDIA DGX Spark, ASUS GX10, vLLM, tensor parallel,
OpenAI-compatible API, local LLM, self-hosted AI, LoRA fine-tuning,
cluster inference, GB10, CUDA 13, NCCL, RoCE, RDMA, container AI
platform, open source ChatGPT alternative.
-->

<p align="center">
  <img src="docs/images/ainode-logo.png" alt="AINode" width="220"/>
</p>

<h1 align="center">AINode</h1>

<p align="center">
  <strong>Turn any NVIDIA GPU into a local AI platform.</strong><br/>
  <em>Inference + fine-tuning in your browser. One command to install. Add nodes, they find each other.</em>
</p>

<p align="center">
  <a href="https://github.com/getainode/ainode/releases/latest"><img alt="release" src="https://img.shields.io/github/v/release/getainode/ainode?display_name=tag&style=flat-square&color=76B900&label=release"></a>
  <a href="https://github.com/getainode/ainode/blob/main/LICENSE"><img alt="license" src="https://img.shields.io/badge/license-Apache%202.0-76B900?style=flat-square"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <a href="https://github.com/orgs/getainode/packages/container/package/ainode"><img alt="ghcr" src="https://img.shields.io/badge/ghcr-getainode%2Fainode-24292e?style=flat-square&logo=github"></a>
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-13-76B900?style=flat-square&logo=nvidia&logoColor=white">
  <img alt="vLLM" src="https://img.shields.io/badge/vLLM-per%20recipe-7C3AED?style=flat-square">
  <a href="https://github.com/getainode/ainode/stargazers"><img alt="stars" src="https://img.shields.io/github/stars/getainode/ainode?style=flat-square&color=FFD700"></a>
  <a href="https://releasebot.io/updates/getainode/ainode"><img alt="Release Bot" src="https://releasebot.io/Full.svg" height="20"></a>
</p>

<p align="center">
  <a href="https://ainode.dev">ainode.dev</a>
  &nbsp;·&nbsp;
  <a href="https://docs.ainode.dev">docs</a>
  &nbsp;·&nbsp;
  <a href="#getting-started-step-by-step">Getting Started</a>
  &nbsp;·&nbsp;
  <a href="#screenshots">Screenshots</a>
  &nbsp;·&nbsp;
  <a href="#models-tested-on-ainode">Models tested</a>
  &nbsp;·&nbsp;
  <a href="#state-of-distributed-inference">What Works / What Doesn't</a>
</p>

---

## What AINode is

AINode is a self-hosted AI appliance for **NVIDIA GB10** (DGX Spark, ASUS
GX10) and any NVIDIA GPU box. One command installs an orchestrator container
and a systemd unit per box, and that orchestrator gives you:

- A modern web UI (chat, cluster topology, server console, downloads, training)
- An OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`)
  and the Anthropic Messages API (`/v1/messages`), both routed fleet-wide by model id
- A decision endpoint (`/v1/decide`): typed questions in, calibrated
  probabilities out, every question answered in one request
- vLLM engines started for you, one container per loaded model, with the engine
  image and flags each model was proven with
- Cross-node tensor parallel over RoCE with NCCL configured from the host's fabric
  interface, launched by picking the nodes in the browser
- UDP node discovery for automatic clustering
- NFS-shared model storage so you download once and use everywhere
- Scripted fine-tuning on a node's own GPU (LoRA, QLoRA, full fine-tune)

The orchestrator image is `python:3.12-slim` plus this package, about 364 MB. It
holds no vLLM and no CUDA: the engine runs as its own container, so a model's
engine version is a property of the model and not of your AINode install. No host
Python venv, no source-built vLLM, no fragile runtime wiring.

```bash
curl -fsSL https://ainode.dev/install | bash
```

---

## Screenshots

### Cluster view: 4 nodes, 487 GB aggregated VRAM

![Cluster view](docs/images/cluster-4node.gif)

The "MASTER" node (head) runs the API and orchestrates. The smaller orbiting node
(member) runs a headless engine container for the rank the head assigned it. The
instance card shows **DISTRIBUTED · TP=2**, meaning the model is sharded across
both GPUs.

### Chat

![Chat](docs/images/chat.png)

Full-featured chat with streaming tokens, prompt history, code
highlighting, per-message metrics (TTFT, tokens/sec, total tokens).
Works against whatever model the cluster has loaded, solo or sharded.

### Server: API console (LM Studio style)

![Server view, API console](docs/images/server-api-console.png)

Live developer console: which models are loaded on which node,
OpenAI-/LM-Studio-/Anthropic-compatible endpoints, per-request logs
with status codes and latency, eject-model buttons, copyable cURL
snippets.

### Downloads: live HF catalog

![Model downloads](docs/images/downloads.png)

Browse trending HuggingFace models, with **AVAILABLE** / **FITS GPU**
badges computed from your cluster's aggregate VRAM. Queue downloads to
the shared NFS cache; any node can load them instantly.

### Training: overview

![Training overview](docs/images/training-overview.png)

Three quick-start paths: **LoRA** (lightweight, most users), **QLoRA**
(4-bit base, for the biggest models), **Full fine-tune** (single
large-memory node). Every run trains on the node's own GPU. Track active +
completed runs, GPU-hours, and jump into dataset management.

### Training: templates

![Training templates](docs/images/training-templates.png)

Starter recipes for instruction tuning (Alpaca), chat fine-tuning
(ShareGPT) and classification heads. Each template ships a working dataset
schema so you can start training in minutes.

### Config: cluster

![Config, cluster](docs/images/config-cluster.png)

Pin the node's role (`auto` / `master` / `worker`), set a shared
`cluster_id` so only matching nodes see each other, and inspect the
current member list with per-node role, address, and last-seen.

---

## Getting Started, step by step

### Single node (solo mode)

1. **Install Docker** + NVIDIA container toolkit on your Linux box.
2. **Pull the image** and wire up the systemd unit:
   ```bash
   curl -fsSL https://ainode.dev/install | bash
   ```
   That one-liner:
   ```
   # resolves the highest numeric GHCR tag (never a floating :latest)
   docker pull ghcr.io/getainode/ainode:<latest-release>
   # pins it to ~/.ainode/image.env and installs a swappable systemd unit
   systemctl enable --now ainode.service
   ```
   The unit reads the pinned image from `~/.ainode/image.env`
   (`EnvironmentFile`, `Restart=always`), so it survives cold power
   cycles and **replays the solo and stacked models you had loaded** on boot. A
   distributed head is deliberately not replayed: it needs its peers, so you
   relaunch it from the browser.
   The installer also pre-pulls the vLLM engine image the engine will run (about
   8.5 GB to download, about 22 GB on disk) so your first launch is not waiting on
   it. `AINODE_NVIDIA_IMAGE=skip` skips that.
3. **Open the UI** at `http://<your-ip>:3000` and pick a model from the catalog.
   Click a model card, then **Launch**, then chat. There is no onboarding wizard:
   the installer writes `onboarded: true` and leaves `model` null, so the node
   comes up with nothing loaded and waits for you.
4. **To add a node**, mint a token on this one with `ainode cluster token` and run
   the `ainode join` line it prints on the new box. That writes the cluster id,
   the shared secret and the master address for you, so joining is never a
   hand-edited `config.json`.

Upgrade is `ainode update`, which resolves and pulls the newest release and
restarts the unit. Pass a version to pin one (`ainode update 0.5.24`). Run it as the
user that installed AINode, and read [Updating AINode](#updating-ainode) before you
reach for `sudo`.

**Prefer to pull the image yourself?** GHCR is the only registry AINode publishes
to:

```bash
docker pull ghcr.io/getainode/ainode:latest      # newest release
# or pin a release, which is what the installer does. Tags are at
# https://github.com/getainode/ainode/releases
docker pull ghcr.io/getainode/ainode:X.Y.Z
```

There is no Docker Hub mirror. An `argentaios/ainode` repository exists there and
stops at 0.4.7 from April 2026, which predates the engine backend that runs today,
the catalog, stacking, the bench and embeddings. Do not pull it.

### Two nodes (distributed mode)

For a model that does not fit on one GPU, such as a frontier MoE sharded across two
DGX Sparks. The browser path is the supported one:

1. **Wire a clean high-speed link** between the two nodes (direct QSFP cable on its
   own `/24`, or a dedicated switch port). See
   [Networking requirements](#networking-requirements): this matters.
2. **Install the head first**, naming the peers so the installer copies your SSH
   key to them:
   ```bash
   AINODE_PEERS="10.0.0.2" curl -fsSL https://ainode.dev/install | bash -s -- --job master
   ```
   `--job master` is not optional. Without it the peer list is used for
   `ssh-copy-id` and nothing else, and the node installs as solo.
3. **Join the peer to it.** On the head, mint a token; on the peer, install and
   join in one command:
   ```bash
   ainode cluster token                      # on the head, prints the line below
   AINODE_JOIN="10.0.0.1:3000:<token>" curl -fsSL https://ainode.dev/install | bash
   ```
   That writes the head's `cluster_id`, its `cluster_secret`, the discovery port and
   the master address on the peer, sets it to `distributed_mode: "member"`, and
   touches no other key. Two nodes installed independently each generate their OWN
   `cluster_secret` and are then invisible to each other, so either join, or install
   the peer with `AINODE_CLUSTER_SECRET` set to the head's value. A peer installed
   with `--job worker` and no join is a member of nothing.
4. **Check passwordless SSH** from the head's install user to each peer. The head
   starts the peers' engine containers over SSH, so a password prompt is a failed
   launch.
5. **Open the head UI.** You should see both nodes and the aggregated memory
   ("2 nodes · 244 GB · 2 GPUs"). Each node detects and announces its own GPU
   count, so a multi-GPU box contributes all of them to that total.
6. **Launch it.** In the Launch Instance panel, pick the model, toggle on the nodes
   to span, and launch. That posts one `POST /api/sharding/launch` with the node ids;
   the head is always the node you are on and the rest become peers, resolved to
   their fabric IPs. A peer with no known fabric IP is refused rather than launched
   over the management LAN. The instance shows as **DISTRIBUTED · TP=2**.

A multi-node launch starts one engine container per node: rank 0 on the head and
`--headless` rank k on each peer, all rendezvousing on `--master-addr` and
`--master-port` through vLLM's own multi-node executor. That is the shape every
multi-node model in the catalog is pinned to and the shape every proven launch used,
because it needs nothing in the engine image beyond vLLM itself. `distributed_mode:
"head"` plus explicit `peer_ips` in `config.json` and a service restart is the other
way in, and it is what an unattended node should use.

Only tensor parallel exists. The launch body accepts a `strategy` field, and any
node count above one is tensor parallel regardless of what it says. There is no
pipeline parallelism.

---

## Quantize a model (AWQ / NVFP4)

AINode can compress a full-precision model to 4-bit **in the browser**, on your
own GPU, with no external service. Open **Training, Quantize a Model**:

1. **Base model**: a Hugging Face repo id (`Qwen/Qwen3.5-4B`) or an installed model.
2. **Scheme**: **AWQ** (W4A16, proven on GB10 via `awq_marlin`) or **NVFP4**
   (Blackwell-native 4-bit float).
3. **Calibration samples**: default 256 (from `HuggingFaceH4/ultrachat_200k`).
4. *(optional)* **Push result to Hugging Face**, which requires a **write** token
   and pushes a private repo under your namespace.

The target node must be **idle**. Quantization needs the full unified memory, so
AINode refuses to start a quant job while a model is loaded (unload first, or pass
`force=true`). The output lands in **Installed** as `<org--name>-<scheme>`, ready
to serve.

**The node needs the job image.** Quantization, training and adapter merge all run in
a spawned GPU container, and it is about 22 GB, so nothing pulls it implicitly. The
node resolves it from an explicit `AINODE_QUANT_IMAGE` / `AINODE_TRAIN_IMAGE`, then
`ghcr.io/getainode/ainode-train:<ainode version>`, then a locally built
`ainode-quant:0.17.0-t5`. A job on a node with none of them is rejected up front with
all three named, rather than dying with `docker` exit 125 in a log. See
[The job image](#quantize-a-model-awq--nvfp4) under training for how to publish or
point at one.

> AWQ is the proven path on GB10. NVFP4 quantization is newer, and **NVFP4 on
> multimodal models (Qwen3.5 for instance) is experimental and not yet verified**, so
> prefer AWQ for the Qwen3.5 family today.

**Hugging Face tokens (read vs write).** AINode keeps credentials in a local
Secrets store (`~/.ainode/secrets.json`, mode 0600, obfuscated at rest) with two
HF slots: a **read** token (download gated models) and a **write** token (push to
the Hub, where read-only tokens are rejected before any multi-GB transfer). Set them
in **Config, Secrets** (each has a **Test** button showing the detected scope), or
set the read token with `ainode config --hf-token hf_xxx`.

---

## Fine-tune a model (LoRA / QLoRA / full)

Open **Training → New Run**, pick a base model and a dataset, and submit. The run
goes through a spawned GPU container (the orchestrator image has no torch), one
job at a time per node, and the queue starts the next job by itself when one
exits.

**The job image.** Training, quantize and merge all need the training image. A
node resolves it in this order, and a job is rejected up front with all three
named if the node has none of them (it used to die with `docker` exit 125):

1. `AINODE_TRAIN_IMAGE` / `AINODE_QUANT_IMAGE`, an explicit override.
2. `ghcr.io/getainode/ainode-train:<ainode version>`, the release image.
3. `ainode-quant:0.17.0-t5`, a locally built tag.

The release image is published by `.github/workflows/publish-train-image.yml`,
which builds `scripts/Dockerfile.quant` on the self-hosted Spark runner. It is
about **22 GB**, so nothing pulls or builds it implicitly and an ordinary release
does not rebuild it. Publish one by pushing a `train-v<version>` tag (the version
must match `pyproject.toml`, because that is the tag the engine looks for), or run
the workflow by hand:

```bash
gh workflow run publish-train-image.yml -f push=true -f version=0.5.27
```

One build can serve several releases through that workflow's `alias_versions`
input (an extra tag on the same digest uploads nothing). On a node that already
has an image, point it there instead:

```bash
sudo systemctl set-environment AINODE_TRAIN_IMAGE=ghcr.io/getainode/ainode-train:0.5.27
```

**Dataset formats.** A `.jsonl` file (under `~/.ainode/datasets/`) or a Hugging
Face dataset id. Rows may be any of:

| Shape | Trained as |
|---|---|
| `{"text": "..."}` | the text itself |
| `{"instruction": "...", "output": "..."}` | Alpaca-style prompt + response |
| `{"prompt": "...", "completion": "..."}` | the two joined |
| `{"conversations": [{"from": "human", "value": "..."}, ...]}` | rendered with the model's own chat template |

The last one is what **AutoData** writes and what the `sharegpt-chat` template
advertises; `role`/`content` turns and a `messages` column work the same way. The
whole rendered conversation is supervised (assistant-only masking is not
implemented, so this does not claim it). Pad positions are never labels.

**Runs survive a restart.** Each job writes a `status.json` into its own job dir at
every transition, and the registry is rebuilt from `~/.ainode/training/jobs/` at
startup, so the Runs table, the stats tiles, logs, artifact download, merge and
resume all still work for jobs from before the restart. Two consequences worth
knowing:

- A job that was RUNNING when AINode stopped comes back **failed**: its process
  went with the restart.
- A job dir written by AINode 0.5.26 or earlier has no status file, so its status
  is reconstructed from what is on disk (completed if it left weights, failed
  otherwise). Those rows are marked `restored` and carry a `note` saying exactly
  that, and they report no duration, because none was ever recorded.

**Resume.** `POST /api/training/jobs/{id}/resume` (or the button on the job) starts
a NEW job from the latest checkpoint: its own output dir, with the source job's dir
mounted read-only at `/src` inside the container and the checkpoint path rewritten
to match. A resume never writes into the run it resumed.

---

## Features

| Feature | Status |
|---|---|
| One-command install | ✅ |
| Slim orchestrator image, engine as its own container per model | ✅ v0.5.0 |
| Auto-detect GPU and memory | ✅ |
| Chat UI in your browser | ✅ |
| OpenAI-compatible API | ✅ |
| Embeddings endpoint (`/v1/embeddings`), routed to a pooling engine on the fleet | ✅ v0.5.25 |
| Live HF model catalog with trending + download manager | ✅ |
| NFS-shared model storage across cluster | ✅ |
| Multi-node auto-discovery (UDP broadcast) | ✅ |
| Distributed tensor-parallel inference across nodes | ✅ (TP=2 on 0.5.x, TP=4 in June 2026 on 0.4.x) |
| Cluster topology UI (members, VRAM aggregate, instance badges) | ✅ |
| Browser-based fine-tuning (LoRA / QLoRA / Full, single node) | ✅ |
| Training artifact retrieval + download via API | ✅ |
| LoRA adapter merge into base model | ✅ |
| Checkpoint resume (own output dir; source run mounted read-only) | ✅ v0.5.27 |
| Training jobs survive a restart (`status.json` per job, registry rebuilt from disk) | ✅ v0.5.27 |
| Training image published to GHCR (`ainode-train:<version>`), with a named preflight | ✅ v0.5.27 |
| Chat / ShareGPT / AutoData `conversations` datasets train (model's own chat template) | ✅ v0.5.27 |
| Evaluation loop (configurable train/eval split) | ✅ |
| W&B logging integration | ✅ |
| Custom training template persistence | ✅ |
| Prometheus metrics endpoint (`/metrics`) | ✅ |
| `ainode role master\|worker\|solo` CLI | ✅ |
| Worker nodes start with no model required | ✅ |
| Web portal available immediately on start | ✅ |
| Cluster-wide update from master UI (`⬆ Update all` button) | ✅ |
| Topology loading animation + per-node fade-in | ✅ |
| AWQ models on GB10 (sm_12.1), `awq_marlin` kernel fix | ✅ |
| In-browser quantization (AWQ W4A16 / NVFP4), then serve or push to HF | ✅ v0.4.44 |
| Push quantized / fine-tuned models to Hugging Face (write-token) | ✅ |
| Secrets store (HF read + HF write + NGC + W&B + OpenAI), masked + testable | ✅ |
| Federated master router: `/v1/*` routed by model name across the cluster | ✅ |
| Load / unload any model on any node from the master UI | ✅ |
| Model stacking: N concurrent models per node, persisted and replayed on boot | ✅ |
| Serve models from on-disk weights (`~/.ainode/models/<slug>`) | ✅ |
| fp8 KV-cache default on GB10 (long-context headroom) | ✅ |
| Per-load overrides (`served_model_name` / `max_model_len` / `kv_cache_dtype` / `quantization` / `trust_remote_code`), persisted across restarts | ✅ v0.5.0 |
| Node-targeted model load (`POST /api/cluster/load {node_id}`) | ✅ v0.5.1 |
| Stacked-load admission guard: explicit `gpu_memory_utilization` required, reject > 0.9 projected total (409) | ✅ v0.5.1 |
| VLM (vision) support: fp8 KV auto-skipped on GB10, `kv_cache_dtype=auto` per-load override | ✅ v0.5.1 |
| LoRA / QLoRA training **and** adapter merge run in a spawned GPU container (slim orchestrator has no torch) | ✅ v0.5.0 |
| Deploy pipeline: a release tag, CI on a self-hosted Spark runner, GHCR, then `ainode update` or cluster update-all (genuine pull and swap) | ✅ v0.5.0 |
| Cancellable, commit-pinned, parallel model downloads | ✅ v0.5.2 |
| Delete a downloaded model from disk (`delete-repo`, frees GB) | ✅ |
| AutoData: Δ-filtered synthetic-data generation (v2.2 val-set lift objective) | ✅ v0.5.0 |

---

## Relation to the Community

AINode builds on excellent open-source work in the DGX Spark ecosystem.
**[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)** and its
patched NCCL (`dgxspark-3node-ring` branch) is where AINode's multi-node work
started: the 0.4.x releases ran eugr's `launch-cluster.sh` and its base image, and
the flags and topology lessons in this repo came out of that.

What runs today is different, and it is worth being exact about it. AINode 0.5.x
launches the engine itself, one vLLM container per instance from the image a model's
recipe pins, and multi-node runs through vLLM's own multi-node executor rather than
Ray. The `ainode-base` image built from eugr's tree is no longer an input to the
shipped product. The debt is real all the same, and eugr's project remains the
go-to for raw vLLM clustering on Spark hardware.

Huge thanks to eugr and the contributors making multi-node Spark setups
practical.

---

## Models tested on AINode

Every row below is a model we launched through AINode on our own hardware and
then measured. One JSON record per run lives in [`bench/results/`](bench/results/),
and this table is generated from those files by
[`scripts/render-bench-table.py`](scripts/render-bench-table.py) so the prose
cannot drift from the data. If a model has no record there, we have not measured
it, and we would rather say so than publish a number we did not take.

<!-- BEGIN bench-table (generated by scripts/render-bench-table.py) -->

| Model | Params / active | Quant | Placement | Single-stream tok/s | 16 streams tok/s | Rubric | Date | Run |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | 125B / 6B active | NVFP4 (mixed, FP8 PLE) | castor, 4× Tesla PG500-216, TP=4 | 49.6 | 104.2 | not run | 2026-09-20 | [AINode 0.5.29, castor TP=4 four V100 32 GB, onecat-vllm 1.5.0-mm, MTP 4, via the fleet endpoint](https://github.com/getainode/ainode/blob/main/bench/results/20260920-122006-qwen3_8-flash-next-nvfp4-v100-ainode-0_5_29-castor-tp-4-four-v100-32-gb-onecat-vllm-1_5_0-mm-mtp-4-via-the-fleet-endpoint.json) |
| Qwen3.6 35B-A3B | 35B / 3B active | NVFP4 | pollux, 1× Tesla PG500-216, TP=1 | 97.4 | 343.9 | not run | 2026-09-19 | [Pollux V100 solo, onecat src-full](https://github.com/getainode/ainode/blob/main/bench/results/20260919-040456-qwen3_6-35b-a3b-nvfp4-pollux-v100-solo-onecat-src-full.json) |
| Spark-X2.5 4B | 4B dense | BF16 | Spark-4-GX10, 1× GB10, TP=1 (stacked) | 18.9 | 283.8 | not run | 2026-09-18 | [Spark-4 stacked beside Nemotron, BF16](https://github.com/getainode/ainode/blob/main/bench/results/20260918-155217-spark-x2_5-4b-spark-4-stacked-beside-nemotron-bf16.json) |
| Qwen3.8-Flash-Next (NVFP4) | 125B / 6B active | NVFP4 (mixed, FP8 PLE) | Spark-2-DGX, 2× GB10, TP=2 | 26.2 | 207.6 | not run | 2026-09-16 | [AINode 0.5.18, TP=2 Spark-2+3, vLLM nightly, no MTP](https://github.com/getainode/ainode/blob/main/bench/results/20260916-204216-qwen3_8-flash-next-nvfp4-tp2-mp-nightly.json) |
| DeepSeek V4 Flash (DSpark, FP8) | 284B / 13B active | FP8 (FP4 experts) | Spark-2-DGX, 2× GB10, TP=2 | 34.5 | 94.0 | not run | 2026-09-16 | [AINode 0.5.11, TP=2 Spark-2+Spark-3, mp shape](https://github.com/getainode/ainode/blob/main/bench/results/20260916-033415-deepseek-v4-flash-dspark-fp8-tp2-mp.json) |
| Nemotron 3.5 Lightning 30B-A3B | 30B / 3B active | NVFP4 | Spark-4-GX10, 1× GB10, TP=1 (stacked) | 52.6 | not measured | not run | 2026-09-13 | [smoke](https://github.com/getainode/ainode/blob/main/bench/results/20260913-141115-nvidia-nemotron-3_5-lightning-30b-a3b-nvfp4-smoke.json) |
| Ornith 1.5 35B-A3B | 35B / 3B active | NVFP4 | Spark-1-DGX, 1× GB10, TP=1 (stacked) | 40.0 | 269.2 | 19/19 | 2026-09-13 | [text-only-mtp](https://github.com/getainode/ainode/blob/main/bench/results/20260913-130400-ornith-1_5-35b-a3b-nvfp4-text-only-mtp.json) |
| Qwen3.8 27B | 27B dense | NVFP4 | Spark-3-DGX, 1× GB10, TP=1 | 19.0 | 147.0 | 23/23 | 2026-08-15 | [mtp-vision](https://github.com/getainode/ainode/blob/main/bench/results/20260815-000000-qwen3_8-27b-nvfp4-mtp-vision.json) |
| Nemotron 3.5 Lightning 30B-A3B | 30B / 3B active | NVFP4 | Spark-4-GX10, 1× GB10, TP=1 | 104.5 | 504.3 | 19/19 | 2026-08-13 | [dspark-recipe](https://github.com/getainode/ainode/blob/main/bench/results/20260813-000000-nvidia-nemotron-3_5-lightning-30b-a3b-nvfp4-dspark-recipe.json) |
| Qwen3 235B-A22B | 235B / 22B active | NVFP4 | Spark-1..4, 4× GB10, TP=4 | 16.5 | not measured | not run | 2026-06-17 | [tp4-frontier-moe](https://github.com/getainode/ainode/blob/main/bench/results/20260617-000000-qwen3-235b-a22b-nvfp4-tp4-frontier-moe.json) |

<!-- END bench-table -->

## Coding harness runs

The same models, driven by their own coding agents. Each row is one model on one
harness (aider, dsh, ..., claude), generated from the same bench records as the
table above by
[`scripts/render-bench-table.py`](scripts/render-bench-table.py), so it cannot
drift from the data either. The latest run for a model-harness pair is the one
shown, and a harness a run flags "not measured" renders that phrase rather than a
number.

<!-- BEGIN harness-bench-table (generated by scripts/render-bench-table.py) -->

| Model | Harness | pass@1 | pass@2 | Mean s | Placement | Date | Run |
|---|---|---|---|---|---|---|---|
| DeepSeek V4 Flash | aider aider 0.86.2 | 9/10 | 10/10 | 47.3 | Spark-1-DGX, TP=2 | 2026-09-16 | [DeepSeek V4 Flash, TP=2 Spark-2+Spark-3, four harnesses, 10 Exercism tasks](https://github.com/getainode/ainode/blob/main/bench/results/20260916-112526-deepseek-v4-flash-dspark-four-harnesses-10-exercism-harness.json) |
| DeepSeek V4 Flash | dsh 0.1.5-rc.1 | 10/10 | 10/10 | 16.4 | Spark-1-DGX, TP=2 | 2026-09-16 | [DeepSeek V4 Flash, TP=2 Spark-2+Spark-3, four harnesses, 10 Exercism tasks](https://github.com/getainode/ainode/blob/main/bench/results/20260916-112526-deepseek-v4-flash-dspark-four-harnesses-10-exercism-harness.json) |
| DeepSeek V4 Flash | opencode 1.18.31 | 10/10 | 10/10 | 23.4 | Spark-2-DGX, TP=2 | 2026-09-17 | [DeepSeek V4 Flash TP=2, opencode after the PWD fix, 10 Exercism tasks](https://github.com/getainode/ainode/blob/main/bench/results/20260917-110113-deepseek-v4-flash-dspark-deepseek-v4-flash-tp-2-opencode-after-the-pwd-fix-10-exercism-tasks-harness.json) |
| DeepSeek V4 Flash | pi 0.73.1 | 9/10 | 10/10 | 13.8 | Spark-1-DGX, TP=2 | 2026-09-16 | [DeepSeek V4 Flash, TP=2 Spark-2+Spark-3, four harnesses, 10 Exercism tasks](https://github.com/getainode/ainode/blob/main/bench/results/20260916-112526-deepseek-v4-flash-dspark-four-harnesses-10-exercism-harness.json) |
| Nemotron 3.5 Lightning 30B-A3B | aider aider 0.86.2 | 8/10 | 10/10 | 389.5 | Spark-4-GX10, TP=1 | 2026-09-18 | [Spark-4 solo, thinking default](https://github.com/getainode/ainode/blob/main/bench/results/20260918-042104-nvidia-nemotron-3_5-lightning-30b-a3b-nvfp4-spark-4-solo-thinking-default-harness.json) |
| Nemotron 3.5 Lightning 30B-A3B | claude 2.1.274 (Claude Code) | 8/10 | 9/10 | 239.5 | Spark-4-GX10, TP=1 | 2026-09-18 | [Spark-4 solo, thinking default](https://github.com/getainode/ainode/blob/main/bench/results/20260918-042104-nvidia-nemotron-3_5-lightning-30b-a3b-nvfp4-spark-4-solo-thinking-default-harness.json) |
| Nemotron 3.5 Lightning 30B-A3B | dsh 0.1.5-rc.1 | 7/10 | 10/10 | 108.9 | Spark-4-GX10, TP=1 | 2026-09-18 | [Spark-4 solo, thinking default, dsh rerun after pytest preflight](https://github.com/getainode/ainode/blob/main/bench/results/20260918-053315-nvidia-nemotron-3_5-lightning-30b-a3b-nvfp4-spark-4-solo-thinking-default-dsh-rerun-after-pytest-preflight-harness.json) |
| Nemotron 3.5 Lightning 30B-A3B | pi 0.73.1 | 7/10 | 10/10 | 119.2 | Spark-4-GX10, TP=1 | 2026-09-18 | [Spark-4 solo, thinking default](https://github.com/getainode/ainode/blob/main/bench/results/20260918-042104-nvidia-nemotron-3_5-lightning-30b-a3b-nvfp4-spark-4-solo-thinking-default-harness.json) |
| Ornith 1.5 35B-A3B | aider | 9/10 | 10/10 | 116.3 | Spark-1-DGX, TP=1 | 2026-09-16 | [Ornith 1.5 35B-A3B on Spark-1, three harnesses, 10 Exercism tasks](https://github.com/getainode/ainode/blob/main/bench/results/20260916-165609-ornith-1_5-35b-a3b-nvfp4-ornith-1_5-35b-a3b-on-spark-1-three-harnesses-10-exercism-tasks-harness.json) |
| Ornith 1.5 35B-A3B | claude 2.1.272 (Claude Code) | 8/10 | 10/10 | 50.8 | Spark-1-DGX, TP=1 | 2026-09-16 | [Ornith 1.5 on Spark-1, Claude Code harness, 10 Exercism tasks](https://github.com/getainode/ainode/blob/main/bench/results/20260916-174134-ornith-1_5-35b-a3b-nvfp4-ornith-1_5-on-spark-1-claude-code-harness-10-exercism-tasks-harness.json) |
| Ornith 1.5 35B-A3B | dsh 0.1.5-rc.1 | 7/10 | 9/10 | 51.2 | Spark-1-DGX, TP=1 | 2026-09-17 | [Ornith 1.5 on Spark-1, thinking off by default, dsh, 10 Exercism tasks](https://github.com/getainode/ainode/blob/main/bench/results/20260917-022907-ornith-1_5-35b-a3b-nvfp4-ornith-1_5-on-spark-1-thinking-off-by-default-dsh-10-exercism-tasks-harness.json) |
| Ornith 1.5 35B-A3B | pi 0.73.1 | 5/10 | 10/10 | 43.2 | Spark-1-DGX, TP=1 | 2026-09-16 | [Ornith 1.5 35B-A3B on Spark-1, three harnesses, 10 Exercism tasks](https://github.com/getainode/ainode/blob/main/bench/results/20260916-165609-ornith-1_5-35b-a3b-nvfp4-ornith-1_5-35b-a3b-on-spark-1-three-harnesses-10-exercism-tasks-harness.json) |
| Qwen3.8 27B | aider aider 0.86.2 | 7/10 | 9/10 | 455.6 | Spark-1-DGX, TP=1 | 2026-09-16 | [Qwen3.8 27B NVFP4 on Spark-1, three harnesses, 10 Exercism tasks](https://github.com/getainode/ainode/blob/main/bench/results/20260916-144500-qwen3_8-27b-nvfp4-three-harnesses-10-exercism-harness.json) |
| Qwen3.8 27B | dsh 0.1.5-rc.1 | 10/10 | 10/10 | 269.9 | Spark-1-DGX, TP=1 | 2026-09-16 | [Qwen3.8 27B NVFP4 on Spark-1, three harnesses, 10 Exercism tasks](https://github.com/getainode/ainode/blob/main/bench/results/20260916-144500-qwen3_8-27b-nvfp4-three-harnesses-10-exercism-harness.json) |
| Qwen3.8 27B | pi 0.73.1 | 9/10 | 10/10 | 205.6 | Spark-1-DGX, TP=1 | 2026-09-16 | [Qwen3.8 27B NVFP4 on Spark-1, three harnesses, 10 Exercism tasks](https://github.com/getainode/ainode/blob/main/bench/results/20260916-144500-qwen3_8-27b-nvfp4-three-harnesses-10-exercism-harness.json) |
| Qwen3.8-Flash-Next | aider | 9/10 | 10/10 | 319.8 | Spark-1-DGX, TP=2 | 2026-09-16 | [Qwen3.8-Flash-Next TP=2 Spark-2+Spark-3, four harnesses, 10 Exercism tasks](https://github.com/getainode/ainode/blob/main/bench/results/20260916-201023-qwen3_8-flash-next-nvfp4-qwen3_8-flash-next-tp-2-spark-2-spark-3-four-harnesses-10-exercism-tasks-harness.json) |
| Qwen3.8-Flash-Next | claude 2.1.272 (Claude Code) | 8/10 | 10/10 | 46.1 | Spark-1-DGX, TP=2 | 2026-09-16 | [Flash-Next TP=2, Claude Code harness effort medium, 10 Exercism tasks](https://github.com/getainode/ainode/blob/main/bench/results/20260916-205159-qwen3_8-flash-next-nvfp4-flash-next-tp-2-claude-code-harness-effort-medium-10-exercism-tasks-harness.json) |
| Qwen3.8-Flash-Next | dsh 0.1.5-rc.1 | 10/10 | 10/10 | 96.2 | Spark-1-DGX, TP=2 | 2026-09-16 | [Qwen3.8-Flash-Next TP=2 Spark-2+Spark-3, four harnesses, 10 Exercism tasks](https://github.com/getainode/ainode/blob/main/bench/results/20260916-201023-qwen3_8-flash-next-nvfp4-qwen3_8-flash-next-tp-2-spark-2-spark-3-four-harnesses-10-exercism-tasks-harness.json) |
| Qwen3.8-Flash-Next | pi | 10/10 | 10/10 | 139.6 | Spark-1-DGX, TP=2 | 2026-09-16 | [Qwen3.8-Flash-Next TP=2 Spark-2+Spark-3, four harnesses, 10 Exercism tasks](https://github.com/getainode/ainode/blob/main/bench/results/20260916-201023-qwen3_8-flash-next-nvfp4-qwen3_8-flash-next-tp-2-spark-2-spark-3-four-harnesses-10-exercism-tasks-harness.json) |
| Spark-X2.5 4B | aider aider 0.86.2 | 3/10 | 6/10 | 923.4 | Spark-4-GX10, TP=1 | 2026-09-18 | [Spark-4 stacked beside Nemotron, first run](https://github.com/getainode/ainode/blob/main/bench/results/20260918-143104-spark-x2_5-4b-spark-4-stacked-beside-nemotron-first-run-harness.json) |
| Spark-X2.5 4B | claude 2.1.274 (Claude Code) | 4/10 | 7/10 | 636.7 | Spark-4-GX10, TP=1 | 2026-09-18 | [Spark-4 stacked beside Nemotron, first run](https://github.com/getainode/ainode/blob/main/bench/results/20260918-143104-spark-x2_5-4b-spark-4-stacked-beside-nemotron-first-run-harness.json) |
| Spark-X2.5 4B | dsh 0.1.5-rc.1 | 6/10 | 9/10 | 689.6 | Spark-4-GX10, TP=1 | 2026-09-18 | [Spark-4 stacked beside Nemotron, first run](https://github.com/getainode/ainode/blob/main/bench/results/20260918-143104-spark-x2_5-4b-spark-4-stacked-beside-nemotron-first-run-harness.json) |
| Spark-X2.5 4B | pi 0.73.1 | 3/10 | 7/10 | 574.8 | Spark-4-GX10, TP=1 | 2026-09-18 | [Spark-4 stacked beside Nemotron, first run](https://github.com/getainode/ainode/blob/main/bench/results/20260918-143104-spark-x2_5-4b-spark-4-stacked-beside-nemotron-first-run-harness.json) |

<!-- END harness-bench-table -->

### What is tested

- **Single-node MoE through the AINode launch path.** Nemotron 3.5 Lightning
  30B-A3B on a GX10 (104.5 tok/s single-stream, 504.3 tok/s across 16 streams,
  DSpark speculative decode accepting 55-61%) and Ornith 1.5 35B-A3B on a DGX
  Spark (40.0 tok/s, 269.2 tok/s across 16 streams) as a *second* stacked
  instance at `gpu_memory_utilization=0.26`.
- **Single-node dense.** Qwen3.8 27B NVFP4, 19.0 tok/s with Qwen3.5 MTP
  speculative decode against 11.8 tok/s without it. Dense decode on GB10 is
  bandwidth-bound, and the number shows it.
- **Model stacking.** Two engines on one node, both routed by model id through
  the AINode API. The Ornith run is the stacked instance, not a solo one.
- **Decode over context depth.** Ornith measured at 4K, 16K, 32K, 64K and 120K
  prompt tokens: 40.0 down to 34.4 tok/s decode, TTFT 194 ms up to 136 s.
- **Concurrency sweeps, not just peak.** 1/2/4/8/16 streams per run where it was
  measured, because the aggregate under load is the number an agent workload
  actually gets.
- **Four-node TP=4 on frontier MoE.** Qwen3-235B-A22B NVFP4 across 4× GB10 at
  16.5 tok/s single-stream, surviving a 3,513-token prefill. Measured June 2026
  on AINode 0.4.x with the `scitrera/dgx-spark-vllm` image.
- **A non-GB10 GPU through the launch path.** Qwen3.6 35B-A3B NVFP4 on pollux, a
  Dell C4130 with one Tesla V100 32 GB, on 2026-09-19: 97.4 tok/s single-stream and
  343.9 across 16 streams, on a Volta-capable vLLM fork pinned by its catalog entry.
  Volta needs its own engine image because mainline vLLM dropped SM70, which is
  exactly what a catalog recipe is for.
- **Embeddings.** Qwen3 Embedding 0.6B stacked beside Nemotron on a GX10: 1024
  dimensions, p50 71.6 ms against a measured 32.1 ms transport floor, 13.9 texts/s at
  batch 1 rising to 225.5 at batch 64.
- **Correctness, separately from speed.** Each rubric run is a fresh agent
  against the served endpoint: executed code, parallel tool calls, needle
  retrieval at three context depths, thinking on and off.

### What is not tested yet

- **Vision on Ornith 1.5.** It is launched text-only
  (`--limit-mm-per-prompt {"image":0,"video":0}`) because the multimodal warmup
  OOM-killed the engine on a node with roughly 35 GB free. No vision numbers and
  no vision rubric for that model.
- **TP=4 through the current launch path.** The 235B row predates 0.5.x and has not
  been re-run on a current release, and that run has no concurrency sweep, no
  sustained number and no rubric.
- **The Dell C4130's other three GPUs.** One model has been launched on a C4130
  through AINode (the pollux row above), on one of its GPUs. The node announces its
  real GPU count now, so the cluster sees all four, but nothing has been measured
  across them: no multi-GPU launch on that box has been run or benchmarked.
- **GLM-5.3-Flash at TP=2.** It ran raw on the Spark-2/Spark-3 pair, outside
  AINode, and was stopped on 2026-09-14 when the pair moved to DeepSeek V4 Flash
  through AINode. No AINode-launched measurement exists for it.
- **DeepSeek V4.1 Flash.** Not launched through AINode: no vLLM build serves its
  architecture on GB10 yet. V4 Flash is in the table above, launched by AINode
  across two nodes with the mp shape; its 16-stream point has not been run.
- **Anything else.** Training throughput and quantization jobs have no bench
  records yet. Embeddings do, and it is in the Embedding runs table below.

### How these numbers were taken

All of it was measured on the hardware described in each row, through AINode's
own OpenAI-compatible endpoint. Generations are streamed and timed from the
first token, and the token counts are the server's `usage` block rather than our
own tokenizer estimate, so a bad chars-per-token guess cannot move a result.
Depth runs carry a unique nonce at the front of every prompt so that prefix
caching cannot serve a cached prefill and make long context look free. The
single-stream column is one request at a time; the 16-stream column is the
aggregate across 16 concurrent requests. The rubric is a separate pass with a
fresh agent driving the endpoint, and it only counts a task as passed when the
code it wrote actually executed or the needle came back verbatim. Sections
nobody measured are simply absent from the JSON and render here as "not
measured"; we do not backfill them with estimates. The record format is
documented in [`bench/SCHEMA.md`](bench/SCHEMA.md).

## Agentic rubric runs

The same models, scored on the mechanical parts an agent loop is made of: hold a
format, call one tool, call three at once, recover when a tool returns an error
instead of inventing the answer, keep a system rule alive over four turns, find one
sentence in a 100k-token prompt. Every verdict is mechanical, so no row here rests
on somebody's reading of a reply: group C runs the model's own code against asserts
it never saw, and group G judges the call trace of a real tool loop. Generated from
the same bench records by
[`scripts/render-bench-table.py`](scripts/render-bench-table.py); the flags and the
probe list are in [`bench/agentic/README.md`](bench/agentic/README.md). The latest
run for a model is the one shown, and the needle column is the largest prompt size
whose needle came back verbatim.

<!-- BEGIN agentic-bench-table (generated by scripts/render-bench-table.py) -->

| Model | Placement | Score | Tools (B+G) | Coding (C) | Reasoning (D) | Needle | Date | Run |
|---|---|---|---|---|---|---|---|---|
| DeepSeek V4 Flash | Spark-2-DGX, TP=2 | 22/25 | 9/9 | 3/3 | 2/4 | 100k | 2026-09-18 | [DeepSeek TP=2 Spark-2+3, full](https://github.com/getainode/ainode/blob/main/bench/results/20260918-013839-deepseek-v4-flash-dspark-deepseek-tp-2-spark-2-3-full-agentic.json) |
| Ornith 1.5 35B-A3B | Spark-1-DGX, TP=1 | 24/25 | 9/9 | 3/3 | 4/4 | 100k | 2026-09-18 | [Ornith stacked Spark-1, thinking on](https://github.com/getainode/ainode/blob/main/bench/results/20260918-013703-ornith-1_5-35b-a3b-nvfp4-ornith-stacked-spark-1-thinking-on-agentic.json) |
| Qwen3.6 35B-A3B | pollux, TP=1 | 20/22 | 9/9 | 3/3 | 3/4 | 8k | 2026-09-19 | [Pollux V100 solo, quick](https://github.com/getainode/ainode/blob/main/bench/results/20260919-040711-qwen3_6-35b-a3b-nvfp4-pollux-v100-solo-quick-agentic.json) |
| Qwen3.8 27B | Spark-1-DGX, TP=1 | 24/25 | 9/9 | 2/3 | 4/4 | 100k | 2026-09-18 | [Qwen3.8 27B solo Spark-1, thinking on](https://github.com/getainode/ainode/blob/main/bench/results/20260918-020831-qwen3_8-27b-nvfp4-qwen3_8-27b-solo-spark-1-thinking-on-agentic.json) |
| Qwen3.8-Flash-Next | castor, TP=4 | 20/22 | 9/9 | 3/3 | 3/4 | 8k | 2026-09-20 | [castor TP=4 four V100, onecat-vllm 1.5.0-mm, MTP 4, quick rubric via the fleet endpoint](https://github.com/getainode/ainode/blob/main/bench/results/20260920-122230-qwen3_8-flash-next-nvfp4-v100-castor-tp-4-four-v100-onecat-vllm-1_5_0-mm-mtp-4-quick-rubric-via-the-fleet-endpoint-agentic.json) |
| Spark-X2.5 4B | Spark-4-GX10, TP=1 | 19/25 | 9/9 | 1/3 | 4/4 | 48k | 2026-09-18 | [Spark-4 stacked beside Nemotron, first run](https://github.com/getainode/ainode/blob/main/bench/results/20260918-151444-spark-x2_5-4b-spark-4-stacked-beside-nemotron-first-run-agentic.json) |

<!-- END agentic-bench-table -->

## Decision runs

A different question again: not how fast a model generates or whether it can drive
an agent, but whether a typed decision it makes can be trusted by code that acts on
the answer. Each row is one backend answering the same 110 labeled items in
[`bench/decide/items.json`](bench/decide/items.json) (route a request, triage a
ticket, is it urgent, is this diff safe to merge, is this statement true), scored on
accuracy and on something that matters more for automation: whether the confidence
it reports can gate anything.

Read the calibration columns before the accuracy one. A wrong answer at 0.95 gets
acted on and does damage; a wrong answer at 0.45 is an abstention a person looks at.
The Brier score is the probability put on the right answer, ECE is the gap between
claimed confidence and observed accuracy over five bins, and "Wrong at 0.9" is the
count that survives a 0.9 gate, over the answers that gate lets through. Generated
from the same bench records by
[`scripts/render-bench-table.py`](scripts/render-bench-table.py); the flags, the
sets and the protocol are in [`bench/decide/README.md`](bench/decide/README.md).

<!-- BEGIN decide-bench-table (generated by scripts/render-bench-table.py) -->

| Backend/Model | Placement | Items | Accuracy | Brier | ECE | Wrong at 0.9 | p50 ms | Cost | Date | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| chat / Ornith 1.5 35B-A3B | Spark-1-DGX, TP=1 | 110 | 0.964 | 0.027 | 0.023 | 2 of 106 | 659 | $0 | 2026-09-18 | [Ornith stacked Spark-1, chat fallback](https://github.com/getainode/ainode/blob/main/bench/results/20260918-030249-ornith-1_5-35b-a3b-nvfp4-ornith-stacked-spark-1-chat-fallback-decide.json) |
| jev / jev-1.13.0 | typesafe.ai hosted | 110 | 0.964 | 0.024 | 0.049 | 0 of 80 | 306 | $0.0016 | 2026-09-18 | [jev-latest, 110 items](https://github.com/getainode/ainode/blob/main/bench/results/20260918-030240-jev-1_13_0-jev-latest-110-items-decide.json) |

<!-- END decide-bench-table -->

## Embedding runs

The model a retrieval pipeline calls all day writes no tokens at all, so none of
the tables above says anything about it. An embedding model is served here the same
way every chat model is, by vLLM on the same engine image with `--runner pooling`,
which means it stacks beside a chat model on one GPU and `POST /v1/embeddings` on
any node routes to it by model id.

Four numbers decide whether one is usable. **Dims** is the width of the vector,
read off the response, because every index downstream has to be built for it.
**p50 ms** is one short text embedded on its own, end to end from wherever the bench
ran; the record carries the measured transport floor beside it, which on a tailnet
is a real share of that figure. **Texts/s at 64** and **Tokens/s** are the same
sweep's batch-64 row, and the gap between them and the batch-1 row in the record is
what an indexer gains by batching. **Pairs ordered** is a sanity check and not a
leaderboard: six hand-written pairs, three related and three not, with one question
asked of them, does every related pair beat every unrelated one, and the margin in
brackets. That catches an engine returning well-formed vectors that mean nothing (a
pooling runner on the wrong checkpoint, a truncated window, a normalisation that
never ran); MTEB is where retrieval quality belongs. Generated from the same bench
records by [`scripts/render-bench-table.py`](scripts/render-bench-table.py); the
block is documented in [`bench/SCHEMA.md`](bench/SCHEMA.md).

<!-- BEGIN embed-bench-table (generated by scripts/render-bench-table.py) -->

| Model | Placement | Dims | p50 ms | Texts/s at 64 | Tokens/s | Pairs ordered | Date | Run |
|---|---|---|---|---|---|---|---|---|
| Qwen3 Embedding 0.6B | Spark-4-GX10, TP=1 | 1024 | 72 | 225 | 2938 | yes (+0.50) | 2026-09-19 | [Spark-4 stacked beside Nemotron](https://github.com/getainode/ainode/blob/main/bench/results/20260919-220011-qwen3-embedding-0_6b-spark-4-stacked-beside-nemotron-embed.json) |

<!-- END embed-bench-table -->

---

## State of Distributed Inference

We owe readers the honest picture, not a checkmark-soup. Here's what's
really running on our hardware. Every throughput figure in this section is a row in
the tables above with a record behind it; there are no estimates here.

### What works today (verified)

- **Single-node inference** on any NVIDIA GB10 box (DGX Spark, ASUS GX10), and on one
  Tesla V100 box through a catalog recipe that pins a Volta-capable vLLM build.
- **Two-node tensor-parallel** (TP=2) with one GPU per node on a direct-connect QSFP
  `/24`, through vLLM's own multi-node executor: rank 0 on the head, `--headless`
  rank 1 on the peer, rendezvousing on `--master-addr` and `--master-port`. NCCL
  chose `NET/IB RoCE @ 200 Gb/s`. Two models are in the tables at TP=2: DeepSeek V4
  Flash at 34.5 tok/s single-stream and Qwen3.8-Flash-Next at 26.2.
- **Four-node cluster** (3× DGX Spark + 1× ASUS GX10), 487 GB aggregated VRAM,
  all four discovered automatically over UDP, topology visible in the browser UI.
  Each node detects its own GPU count and announces it, so the cluster total is a
  sum over what the nodes report rather than a count of nodes.
- **One command per node.** `curl -fsSL https://ainode.dev/install | bash -s -- --job worker`
  installs a worker that needs no model. It still pulls the orchestrator image and
  pre-pulls the engine image, so budget the download.
- **`ainode role`** CLI sets master/worker/solo instantly.
- **Worker nodes start with no model** and no engine warmup. The web portal is up
  within seconds of `systemctl start ainode`.
- **Shared model storage over NFS** from an NVMe-oF-backed master.
- **UDP cluster discovery** with real peer-IP capture. The installer writes port 5679
  and `cluster_id: "ainode-cluster"` on every node; the code defaults are 5678 and
  `"default"`, so a hand-written config has to match its neighbours to see them.
- **Four-node TP=4** on frontier MoE: `nvidia/Qwen3-235B-A22B-NVFP4` at TP=4 across
  4× GB10, 16.5 tok/s single-stream, surviving a 3,513-token prefill. Measured
  June 2026 on 0.4.x. The GB10 sm120 fix there was `--enforce-eager`, because vLLM's
  FlashInfer prefill kernel emitted an `illegal instruction` under CUDA-graph capture
  on that engine build. It is not a flag to carry forward blindly: AINode forces it
  only on the pinned 0.17 engine image, and it costs throughput on 0.27.1.
- **Federated serving.** A master routes `/v1/*` to the node holding each model, and
  models load and unload per node from the browser.
- **Model stacking.** Multiple models per node, persisted and replayed on boot.
- **In-browser quantization.** AWQ and NVFP4 jobs run on an idle node and land the
  result in Installed, optionally pushed to Hugging Face, on a node where the job
  container image has been built.

### What still needs care

- **No fabric over Tailscale.** Use physical cables or a dedicated switch: peers are
  addressed by their fabric IP, and a tunnel is not one.
- **Single NIC per cluster subnet.** Multi-NIC ambiguity still breaks the NCCL ring.
- **The engine image decides the multi-node shape.** The `mp` executor needs nothing
  but vLLM. The `ray` executor needs a `ray` CLI inside the engine image, and no
  image AINode ships or pins has one, so a catalog recipe that spans nodes pins `mp`.

### Lessons learned the hard way

0. **Role clarity eliminates half the problems.** The single biggest
   UX improvement was `ainode role master|worker|solo`. Workers don't
   need a model, don't need to think, don't need config editing. They
   start in 3 seconds and announce themselves. The master is the only
   node that needs a model. Everything else follows from that.

1. **Single NIC per cluster subnet.** Multi-NIC ambiguity breaks NCCL
   ring setup silently; `NCCL_SOCKET_IFNAME` only tells NCCL which
   address to *listen* on, not which source the kernel picks for
   outbound traffic.
2. **A hung engine does not release the GPU.** Always `docker rm -f` the full
   chain of engine containers, on the head and on every peer, before retrying a
   multi-node launch. On the 0.4.x Ray shape the reservation outlived SIGKILL
   because Ray's GCS still thought the GPU was busy; on the `mp` shape the
   containers themselves are what you have to clear.
3. **Block-level shared storage is unsafe for multi-writer.** NVMe-oF +
   ext4 mounted on two hosts corrupts under concurrent writes. Put NFS
   on top of a single-host mount.
4. **The patched NCCL in `eugr/spark-vllm-docker`** (`dgxspark-3node-ring`
   branch) was the only variant we saw reliably handle GB10 unified-memory
   topologies in the 0.4.x era, when AINode's engine came from that base image.
   0.5.x runs the engine image a model's recipe names, so the NCCL in play is
   whichever one that image ships. What AINode contributes is the env:
   `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `UCX_NET_DEVICES` and
   `NCCL_IB_HCA`, computed from the host's sysfs and set on the engine container.
5. **SSH from a root container into a host user** fails silently when
   keys are mounted read-only from the host. Our entrypoint copies
   `/host-ssh` → `/root/.ssh` with correct perms and injects
   `User <ssh_user>` for peer IPs.

### Why 3 nodes is harder than 2 (and why 4 is probably easier)

**Two nodes**: a single direct-connect cable on one `/24`. One cable,
one subnet, one candidate interface per host. NCCL can't get confused.
TP=2 splits the weights evenly. Solved problem.

**Three nodes**: no simple physical topology. Options:

- **Triangle mesh** (A↔B, B↔C, A↔C) with each link on its own `/30`. Community
  tooling assumes this, and nobody autoconfigures it.
- **Dedicated cluster switch** with one NIC per node on an isolated subnet. Easier,
  but a hardware purchase.
- **Star topology**: asymmetric latency, not recommended.

If your three nodes just share a regular LAN, you hit multi-NIC routing
ambiguity (lesson #1). We did. NCCL ring setup succeeded; data never
flowed.

**Four nodes**: paradoxically simpler once you commit to a switch,
which is the only practical option for 4+. One NIC per node on a fresh
`/24`, TP=4 lines up with vLLM's defaults, and the community has
published recipes (eugr's `recipes/4x-spark-cluster/`, NVIDIA's internal
4× Spark reference setups).

**Our hypothesis:** the difficulty is not *N* nodes, it is *how you
wire N nodes*. Two is forced (one cable). Three forces a topology
decision. Four+ forces a switch, which is what the community tools
expect. Stick to 2 now, buy the switch, jump straight to 4.

---

## Networking requirements

AINode relies on NCCL for cross-node tensor-parallel, and NCCL works
best when it owns a clean link.

- **Passwordless SSH** from the head's host user to every peer.
- **Single active NIC per cluster subnet** on every node. Multiple
  interfaces on the same `/24` breaks the NCCL ring.
- **No VPN between nodes for cluster traffic.** Tailscale is fine for
  laptop-to-cluster SSH, and not fine as the NCCL transport.
- **Consistent MTU** across the cluster subnet.
- **Open the rendezvous port** between nodes on the fabric subnet. A multi-node
  launch through vLLM's own executor rendezvouses on `--master-port`, which AINode
  sets to 29501 plus a per-instance offset. Ray's 6379 is only needed if you point a
  recipe at the `ray` executor with an engine image that ships it.

### Recommended topologies

| Cluster size | Topology | Notes |
|---|---|---|
| 2 nodes | Direct QSFP cable, each end on its own IP in a fresh `/24` | Simplest, verified |
| 3 nodes | Triangle direct-connect (3 cables, each on a `/30`) **or** dedicated switch | Mesh is finicky; switch is easier |
| 4+ nodes | Dedicated QSFP switch on its own `/24`, one NIC per node | The only practical option |

### Diagnostic commands

```bash
# Confirm RDMA is live on your ConnectX-7
ibstat mlx5_0 | grep -E "State|Rate"         # expect "Active" + "Rate: 200"

# Confirm exactly one interface has an IP on the cluster subnet
ip -br -4 addr | grep 10.0.0                 # expect one line per node

# Confirm cross-node reach on the cluster subnet (not Tailscale)
ping -c 2 10.0.0.2
traceroute 10.0.0.2                          # 1 hop = right link

# Passwordless SSH works
ssh sem@10.0.0.2 true && echo OK

# After launching distributed: confirm NCCL uses RoCE, not Socket.
# `ainode logs` resolves the log file the configured backend writes and says which
# one it is following, so this works on the head and on a peer.
ainode logs | grep -E "Using network|NET/IB.*RoCE" | head -5
# Expect: "Using network IB" + "NET/IB ... mlx5_0:1/RoCE ... speed=200000"

# Or read the engine container directly. The head is ainode-vllm-head (plus the
# port token for a stacked instance) and each peer runs ainode-vllm-worker-*:
docker ps --format '{{.Names}}' | grep ainode-vllm
docker logs ainode-vllm-head 2>&1 | grep -E "Using network|NET/IB.*RoCE" | head -5
```

### Optional: GPU Direct RDMA (GDR)

Without GDR, traffic goes GPU → CPU → NIC → NIC → CPU → GPU. With GDR
it bypasses the CPU hop. On GB10 the unified-memory CPU hop is cheap,
so the win is smaller than on discrete GPUs but still measurable.

```bash
# Load the peermem module on each host (not the container)
sudo modprobe nvidia_peermem
echo nvidia_peermem | sudo tee -a /etc/modules-load.d/nvidia-peermem.conf

# Verify NCCL picks it up next launch
ainode logs | grep "GPU Direct RDMA"
# Expect: "GPU Direct RDMA Enabled"
```

---

## Shared model storage across a cluster

Downloading a 70 GB model three times on a three-node cluster is
wasteful. AINode supports a shared `models_dir` so every node pulls
from the same cache.

Block-level shared storage (NVMe-oF, iSCSI, Fibre Channel) is fast but
**unsafe for multiple Linux kernels writing simultaneously**, because ext4
and XFS have no distributed lock manager. Layer NFS on top:

```
  Storage array (NVMe-oF, SAN, local NVMe)
         │
         ▼
  MASTER NODE  ← ext4/XFS mounted here, owns the disk
    │   │
    │   └── NFS server exports /mnt/ai-models
    ▼
  WORKERS      ← mount the NFS share at /mnt/ai-shared
```

NFS over a 100G fabric gives 3 to 8 GB/s, and vLLM model loading is a
one-shot sequential read, so you won't notice. For 100 GB+ models
where load time hurts, add an rsync-to-local staging step.

---

## CLI reference

The installer puts a thin `ainode` wrapper at `/usr/local/bin/ainode`.
Host-side commands (`update`) run directly, and everything else is forwarded
into the running container with `docker exec`. Day to day you never need to type
`docker` yourself, though a job container for training or quantization has to be
built by hand once (see [Quantize a model](#quantize-a-model-awq--nvfp4)).

```bash
ainode update [version]      # pull, pin, restart, verify the node came back on it,
                             #   then prune the images it replaced
ainode start                 # Start AINode (inference + web UI)
ainode stop                  # Stop AINode
ainode status                # Show cluster status
ainode models                # List available models
ainode role master|worker|solo  # Set or show this node's cluster role
ainode cluster token         # On the master: mint a single-use join token and print
                             #   the `ainode join` line to run on the new node
ainode cluster tokens        # List the join tokens that are still valid
ainode join HOST[:PORT] TOKEN   # On the new node: join the cluster that token
                             #   came from, then restart to apply
ainode service install       # Install the systemd unit
ainode service status        # Show systemd state + recent journal
ainode config                # Show current configuration
ainode auth enable|disable|status|new-key   # API key auth
ainode doctor                # 21 checks over config, docker, GPUs, disk, ports, peers
ainode prune-images          # Reclaim older AINode images (what update does for you)
ainode logs -f               # Tail the engine log the configured backend writes
```

`ainode doctor` is where to start on a node that is behaving oddly. One line per
check with OK, WARN or FAIL and a one-line fix: the engine backend against what is
actually on the node, `gpu_memory_utilization` against the stacked-load guard, the
discovery port and `cluster_id` against what the installer writes, docker and the
engine image, GPUs and whether the memory is unified, free space on the AINode home
and the models dir, the pin in `image.env` against the running container and the
newest published tag, the unit, ports 3000 / 8000 / 5679, peers and whether the fleet
agrees on a release, the fabric interface, the secrets store's mode, and the sudo
trap below. `--json` for machines, `--peer HOST` for the same report over SSH, a
non-zero exit on any FAIL so it can gate a script, and `--fix` applies only the
changes that cannot lose anything (create a directory, chmod the secrets store to
0600, write the fleet discovery port) and then re-runs the checks.

`ainode cluster token` and `ainode join` are the whole join. The token is 32 random
bytes, stored on the master as a SHA-256 hash with an expiry (30 minutes, `--ttl`) and
one use, and it is the joiner's only credential: `POST /api/cluster/join` is the one
route that answers without an API key, because a node that has not joined cannot hold
this cluster's key yet. A wrong, an expired and a spent token all get the same 403,
and the handler allows five attempts a minute per source address. The joining side
writes `cluster_id`, `cluster_secret`, `cluster_role`, `distributed_mode`,
`master_address` and `discovery_port` into `config.json` and touches nothing else,
refuses to join a master on a different AINode release unless you pass
`--allow-version-mismatch`, and then restarts the service or prints the command.
The browser can do the same thing from **Config, Cluster**.

`ainode logs` resolves the log path from the backend and prints which file, and
which backend, it is following: `nvidia-vllm.log` for a solo or stacked engine and
`nvidia-distributed.log` on a distributed head.

### Updating AINode

Releases ship through a tag-triggered pipeline: a release tag starts CI on a
self-hosted Spark runner, which builds and pushes
`ghcr.io/getainode/ainode:X.Y.Z`. GHCR is the only registry. To upgrade a node in
place:

```bash
ainode update
```

That resolves the **highest numeric GHCR tag** (never a floating `:latest`), pulls
it, pins it to `~/.ainode/image.env`, restarts the systemd service, and then waits for
this node's `/api/status` to report the version it just installed. **An update that
did not apply exits non-zero** instead of reporting success, and nothing is removed in
that case, so a failed update still has something to fall back to. Once the new
version is proven serving, the images it replaced are pruned, keeping one rollback
generation by default so `ainode update <older>` can go back. `--keep-images N` sets
that; `0` removes every older AINode image. Engine images and `ainode-base` are never
touched. Your config (`~/.ainode/config.json`), models (`~/.ainode/models/`) and
fine-tune outputs are on the host, and the container is stateless, so upgrades never
touch your data.

To pin a specific version:

```bash
ainode update 0.5.24
```

**Run it as the user that installed AINode.** The `image.env` the systemd unit reads
is the one in that user's `~/.ainode`, and the path is baked into the unit at install
time. Under `sudo`, `$HOME` is root's, so through 0.5.25 `sudo ainode update` pulled
the new image, pinned it into `/root/.ainode/image.env` where nothing reads it,
restarted the unit and relaunched the OLD image while reporting success. It bit this
fleet repeatedly. From 0.5.26 the wrapper resolves the `AINODE_HOME` the unit
actually reads (an explicit `AINODE_HOME` first, then the unit file's
`Environment=AINODE_HOME=`, then `$SUDO_USER`'s home), prints the `image.env` it
wrote, and where it cannot tell it refuses rather than writing a file nothing reads:

```bash
sudo AINODE_HOME=/home/<user>/.ainode ainode update
```

To roll every node in a cluster from the master, use the **Update all**
button or:

```bash
curl -X POST http://<master>:3000/api/cluster/update-all   # genuine pull + swap on every node
```

---

## API

AINode exposes an OpenAI-compatible API. Drop it into any tool that
speaks OpenAI:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3000/v1",
    api_key="not-needed",
)

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

Works with Open WebUI, LiteLLM, LangChain, llama.cpp clients, and
anything else that speaks OpenAI.

### Two ports, and which one you want

Point your tools at **3000**. That is the only port AINode itself listens on, and
everything it offers is there: the web UI, the whole `/api` surface, the `ainode_*`
Prometheus metrics on `/metrics`, and every forwarded inference path, each routed
fleet-wide on the body's `model` with transport failover. Forwarded today:
`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/responses`,
`/v1/rerank`, `/v1/score`, the Anthropic `/v1/messages` and
`/v1/messages/count_tokens`, and `/tokenize` and `/detokenize` (those two are not
under `/v1`, because vLLM does not serve them there). The speech-to-text pair,
`/v1/audio/transcriptions` and `/v1/audio/translations`, is forwarded too, and is
the one pair that is not JSON: they are multipart uploads, so the model id rides
beside the audio file as a form field, and that field is what picks the node.
`GET /v1/models` is the
federated union and is answered here rather than forwarded, and `POST /v1/decide` is
composed here out of grammar-constrained completions of its own.

**8000** is the primary vLLM engine container, talking to you directly. It serves one
model with no routing and no failover, it is closed until a model is loaded, and its
`/metrics` is vLLM's own `vllm:*` set rather than AINode's. `:8000/api/...` is a 404.
It is useful for looking straight at an engine and for nothing else. A stacked model
gets 8001, 8002 and so on the same way.

**Neither port asks for a credential until you turn one on.** That is what a private
network wants and it is what this fleet runs, so the dashboard says so in the header
("API open, no key set") and the installer prints the same words rather than letting
you assume a password exists. Turn it on in **Config, API access** (the browser that
flips the switch stores the key it mints, so enabling auth cannot lock you out) or
with `ainode auth enable`. With auth on, every path under `/api` and `/v1` wants the
key, with four deliberate exceptions: `/api/health` because a probe has no key,
`/api/auth/status` so the UI can say a key is wanted instead of rendering blank, the
static shell, and `POST /api/cluster/join` because a node joining this cluster cannot
hold this cluster's key yet. That last one takes a single-use expiring join token
instead, and the handler rate limits it to five attempts a minute per source address.

### Pinning a request to one instance

By default a forwarded request is routed on its `model`, cheapest hop first, so
with the same model served on two nodes either one may answer. To address ONE
instance, send its node id and engine port:

```bash
curl http://localhost:3000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-AINode-Node: spark3' \
  -H 'X-AINode-Port: 8001' \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"hi"}]}'
```

A client that cannot set headers can pin in the body instead, with
`"ainode_target": {"node_id": "spark3", "port": 8001}` or the short
`"ainode_target": "spark3:8001"`; AINode strips the field before forwarding. A
port on its own pins a stacked instance on the node you asked. A pin is honoured
exactly: that instance answers or the request fails, with no failover to another
copy, and an unknown node id is a 404 rather than a quiet fall back to the local
engine. Every forwarded response carries `X-AINode-Served-By: <host>:<port>`,
which is where the request actually went. The node ids are in `/api/nodes`, and
the web UI's chat and bench pickers send this pair for you.

### Decisions: `POST /v1/decide`

A decision endpoint rather than a chat one. You hand over a state, an optional
block of domain guidance and a dict of independent multiple-choice questions.
Every question is asked at once against whatever model the node or fleet has
loaded, and each answer comes back with the probability the model put on it plus
the full distribution over that question's options.

Each question becomes one chat completion whose output is grammar-constrained
(vLLM's `structured_outputs: {"choice": [...]}`) to a single lettered label, with
`logprobs` on and thinking off. The probability is a softmax over the first
generated token's logprobs restricted to those labels, renormalized. The shared
state goes in front of the question so every call in a request has the same
prompt prefix and the engine's prefix cache prefills it once.

Request:

```json
{
  "model": "ornith-ai/Ornith-1.5-35B-A3B-NVFP4",
  "state": "any string, or any JSON object (serialized compactly for you)",
  "instructions": "optional domain guidance, appended to the system prompt",
  "questions": {
    "category":    {"question": "Which queue?", "options": ["billing", "bug", "spam"]},
    "needs_human": {"question": "Does this need a human?", "type": "boolean"},
    "urgency":     {"question": "How urgent?", "type": "score", "min": 1, "max": 5}
  }
}
```

`type: "boolean"` is sugar for `["yes", "no"]` and `type: "score"` for
`["1", ... "5"]` (or whatever `min`/`max` you give). A question takes 2 to 255
options, all distinct strings. `model` is optional and defaults to the node's
loaded model, or to the fleet's only model if this node has none.

Response:

```json
{
  "model": "ornith-ai/Ornith-1.5-35B-A3B-NVFP4",
  "node": "Spark-Ornith",
  "latency_ms": 541.3,
  "decisions": {
    "category": {
      "answer": "bug",
      "confidence": 0.999331,
      "distribution": {"billing": 8.7e-05, "bug": 0.999331, "spam": 0.00014},
      "latency_ms": 540.7
    }
  },
  "usage": {"prompt_tokens": 567, "completion_tokens": 6, "calls": 3}
}
```

Probabilities are keyed by your option strings, not by the letters used to
constrain the engine. A `200` always carries every question: a bad shape is a
`400` naming what is wrong, and an engine that cannot answer is a `503`. If the
engine returns no logprobs, `distribution` is `null`, `confidence` is `1.0` and
the entry carries `"note": "no logprobs from engine"`.

```bash
curl -s http://localhost:3000/v1/decide -H 'Content-Type: application/json' -d '{
  "state": {"subject": "Invoice PDF download returns 500 since Tuesday", "plan": "pro"},
  "instructions": "B2B SaaS support queue. Revenue-blocking regressions outrank cosmetic bugs.",
  "questions": {
    "category":    {"question": "Which queue should this go to?",
                    "options": ["billing", "bug", "feature request", "account access", "spam"]},
    "urgency":     {"question": "How urgent is this, 1 lowest and 5 highest?",
                    "type": "score", "min": 1, "max": 5},
    "needs_human": {"question": "Does this need a human to reply?", "type": "boolean"}
  }
}'
```

Caveat past 26 options: labels continue `AA`, `AB`, and so on. When the
tokenizer does not give a two-letter label its own token, that label's mass is
its first letter's token mass, which the single-letter label of the same letter
also claims. Read such a pair as jointly calibrated.

### Metrics: `/metrics` (Prometheus) and `/api/metrics` (JSON)

AINode exposes its own metrics on port 3000, the same port as its API. The engine's
own `vllm:*` metrics are a separate set on the engine port and are not these:

```bash
curl http://localhost:3000/metrics           # Prometheus text exposition, ainode_*
curl http://localhost:3000/api/metrics       # JSON snapshot
curl http://localhost:3000/api/metrics/gpu   # GPU subset
curl http://localhost:8000/metrics           # the primary engine's own vllm:* set
```

Key series:

- `ainode_uptime_seconds`, `ainode_build_info{version=...}`
- `ainode_requests_total`, `ainode_request_errors_total`
- `ainode_tokens_generated_total`, `ainode_tokens_per_second`
- `ainode_request_latency_milliseconds{quantile="0.5|0.95|0.99"}`
- `ainode_requests_by_model_total{model=...}`
- `ainode_gpu_utilization_percent`, `ainode_gpu_memory_used_bytes`,
  `ainode_gpu_memory_total_bytes`, `ainode_gpu_temperature_celsius`,
  `ainode_gpu_available`

Scrape config for Prometheus:

```yaml
scrape_configs:
  - job_name: ainode
    static_configs:
      - targets: ["ainode-host:3000"]
```

---

## Requirements

- **OS**: Ubuntu 22.04+ (DGX Spark OS works out of the box)
- **GPU**: NVIDIA GB10 (DGX Spark, ASUS GX10), which is where every measurement in
  this README was taken, plus one Tesla V100 lane proven through a catalog recipe.
  Any NVIDIA GPU with CUDA 13 drivers is the design target, and a GPU outside those
  two families has not been run: the default engine image is a GB10 build, so a new
  architecture may need its own `engine_image` in the recipe, the way Volta did.
- **Memory**: 8 GB+ GPU for small models, 128 GB recommended for the
  big ones, 240 GB+ aggregated for real sharded work
- **Disk**: about 400 MB for the AINode image, plus the engine image (roughly 22 GB
  on disk for the default one, and a second one if a recipe pins its own), plus the
  model weights
- **Docker**: 24.0+ with the NVIDIA Container Toolkit

---

## Why AINode?

| | Cloud AI | AINode |
|---|---|---|
| Monthly cost | $100 to $10,000+ | $0 (you own the hardware) |
| Data privacy | Your data on their servers | Your data stays local |
| Rate limits | Yes | None |
| Network hop | Out to a provider and back | Your own LAN |
| Fine-tuning | Limited, expensive | LoRA, QLoRA and full, single node, free |
| Internet required | Yes | Only to pull images and weights |
| Models available | Their choice | Your choice |

Measured latency belongs in the tables above, not here. The fastest time to first
token in any of our records is 194 ms, on a 4K prompt.

---

## Roadmap

- [x] Core CLI + installer
- [x] vLLM engines launched for you, one container per model, per-recipe image
- [x] Web UI (chat, server, downloads, training, config)
- [x] Multi-node auto-discovery + cluster topology view
- [x] Model sharding across nodes from the browser node picker (TP=2 verified)
- [x] NFS-shared model storage
- [x] Slim orchestrator image + systemd install
- [x] Browser-driven fine-tuning (LoRA / QLoRA / Full, single node)
- [x] Training artifact retrieval, LoRA merge, checkpoint resume
- [x] Training jobs, artifacts and resume survive a restart
- [x] Evaluation loop + W&B integration
- [x] Prometheus metrics endpoint (`/metrics`)
- [x] 4-node TP=4 sharded inference (235B-A22B-NVFP4, June 2026 on 0.4.x)
- [x] In-browser quantization (AWQ / NVFP4) + push to Hugging Face
- [x] Federated multi-model serving (master routes `/v1/*` by model name)
- [x] Model stacking (N models per node, persisted + replayed)
- [x] Training + adapter-merge in a spawned GPU container (slim orchestrator)
- [x] Deploy pipeline (release tag, CI, GHCR, then `ainode update` or update-all)
- [x] VLM (vision) serving with fp8-KV auto-skip on GB10
- [x] AutoData, Δ-filtered synthetic-data generation (val-set lift objective)
- [x] A non-GB10 GPU served through a catalog recipe (Tesla V100)
- [x] Training image published from CI, with a preflight that names it
- [ ] Publish the quantization job image from CI
- [ ] Model marketplace (custom registries)
- [ ] Mobile-friendly UI

---

## Contributing

AINode is Apache-2.0 and welcomes contributions. Work on a branch, open a PR, and
run the test suite and the linter before you do:

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
ruff check ainode tests
```

The repo's edit rules live in [`AGENTS.md`](AGENTS.md) and in the `AGENTS.md` nearest
the folder you are changing. To report a security issue, read
[SECURITY.md](SECURITY.md) and mail it rather than opening an issue.

---

## License

Apache 2.0. Use it however you want.

---

<p align="center">
  <sub>crafted with <span style="color:#e74c3c">♥</span> by Jason Brashear · Made in Texas</sub>
</p>
