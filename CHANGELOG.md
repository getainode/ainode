# Changelog

All notable changes to AINode are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

_Nothing yet._

---

## [0.5.33] - 2026-09-26

- A decision is answered by the node that owns the model, wherever the request enters.
- When the receiving node does have to call the engines itself, it tempers with the serving node's table.
- A slow resolver can no longer freeze a node.
- A decision model kept across a restart on a stacked port warms its answer grammar again.

---

## [0.5.32] - 2026-09-26

- `/v1/decide` and `/v1/systemone` apply a decision adapter's own temperatures.
- A decision model warms its answer grammar as soon as its engine binds.
- The decision bench scores models on the public question sets the independent Jevals boards use, with the boards' own formulas.
- A decision request that meets a cold grammar compile gets a 503 saying so, and has longer to finish.
- `ainode doctor` passes on a node that loads no model.
- A request with no `model` on a node that serves no model of its own gets a 400 `missing_model_field` naming the field.
- Unloading or ejecting a node's primary no longer promotes another instance.
- An empty account list is never replicated.

---

## [0.5.31] - 2026-09-21

- The bench can present an API key, and a node that refuses a run no longer scores the model at zero.
- A login for people, beside the API key for machines.
- Roles, and a node that cannot be locked away from its operator.
- Session and account routes.
- `ainode auth user add|list|remove|passwd|disable|enable` and `ainode auth session list|revoke|clear`.
- `ainode auth status` prints the user, admin and session counts beside the key count, and names the fix when a protected node has no account.
- Account replication (`ainode/auth/replication.py`).
- `ainode doctor` gains a Login check.
- The installer prints `ainode auth enable` and `ainode auth user add <name> --admin` after an install that requires a credential, and nothing when `AINODE_AUTH=off`.
- Add `/v1/systemone` compatibility for TypeSafe decision clients.
- The README describes 0.5.30.
- A 401 now names both doors.
- A cookie-authenticated write must carry `X-AINode-Client: dashboard`.
- Failed logins are throttled per source address.
- Sign in to the dashboard.
- The docs describe how a person gets in, not only how a program does.
- The host wrapper the installer writes asks for a TTY only when it has one.

---

## [0.5.30] - 2026-09-19

- Qwen3.8-Flash-Next NVFP4 runs on four Tesla V100s through the launch path.
- A catalog id beats an hf repo when two entries share a checkpoint.
- A bench record states the width of a single-node tensor-parallel launch.
- A speech section for the bench: `scripts/ainode-bench.py speech` measures word error rate, latency per clip and real-time factor against audio the repo ships.
- `openai/whisper-large-v3-turbo` is verified, with a record behind it.
- A cluster can run with auth on.
- `ainode auth key create --name <client>`, `ainode auth key list`, and `ainode auth key revoke <id>`.
- Two doctor checks.
- `/api/health` reports the running version.
- The dashboard draws the metrics a node has been keeping.
- Any node's charts, from any node.
- The Prometheus latency summary tells the truth about an idle node.
- `ainode tls enable --tailscale` works, and it runs on the host.
- The certificate renews itself.
- `ainode tls renew [--check]`.
- A download that cannot fit is refused before it starts.
- `ainode doctor` warns when persistence mode is off on a node with discrete GPUs.
- The README describes the product that ships.
- A speech run driven at a master now records the right node's neighbours.
- `bench/README.md` lists every bench again.
- A fresh install requires an API key.
- An `ainode auth` change takes effect immediately.
- `auth.json` and `config.json` are written 0600.
- The API access panel, the README and the join route's docstring state the real set of keyless paths.
- A graceful shutdown leaves a still-serving engine container running.
- A node advertises the scheme it really serves.
- `ainode doctor`'s TLS check names this node.
- `hf_token` no longer leaves the node in `GET /api/config`.
- `ainode update` no longer reports a failed update on a protected node.
- `POST /api/cluster/update-all` no longer crashes on a peer that announced no source address.
- A restart no longer reloads a model the node is already serving.
- Every view that lists instances now says which ones were adopted.
- `ainode doctor` inside the container no longer carries a WARN nobody can clear.
- A slow GPU driver can no longer make a node look dead.
- A replayed engine that dies on the way up says why.
- `ainode doctor --fix --peer HOST` no longer ignores `--fix` silently.
- `ainode update` exits non-zero when it cannot resolve a version to verify against.
- `ainode prune-images` reclaims the Docker Hub mirror tag that nodes actually carry.
- The unused metrics ring buffer in the dashboard.

---

## [0.5.29] - 2026-09-19

- A head writes down what it is serving, and says so when it cannot serve it.
- A node joins a cluster with one command instead of a hand-edited `config.json`.
- A fresh install generates a `cluster_secret`, so a new node is not running unauthenticated discovery by default.
- `POST /api/cluster/join`, the fourth and last route that answers without an API key.
- A node now remembers what it measured.
- `GET /api/metrics/history` serves any series over any window.
- The Prometheus endpoint can be scraped for the first time.
- The dashboard's metrics buffer is seeded from disk at page load.
- A client endpoint that survives the master.
- `/api/status` now carries `endpoint_hint`.
- TLS on its own port, with HTTP left exactly where it was.
- Per-client rate limits on the inference API.
- Speech to text is a fleet feature: `POST /v1/audio/transcriptions` and `/v1/audio/translations` now route across the cluster like a chat completion.
- `openai/whisper-large-v3-turbo` is in the curated catalog.
- That engine image is published: `ghcr.io/getainode/ainode-whisper:0.17.0-t5`.
- A restart on a distributed head no longer loses the multi-node model.
- A solo load can no longer replace a multi-node instance by accident.
- The dashboard no longer claims there is no TLS.
- A launch refused over a missing engine image now says which image, and how to get it.
- A speech instance is its own kind across the interface.
- Remove the unreachable browser onboarding wizard; use token-based cluster joining.

---

## [0.5.28] - 2026-09-19

- `ainode prune-images`.
- The repo's own docs describe the system that ships.
- A multi-GPU node counts as all of its GPUs.
- On a unified-memory node, host RAM is no longer reported as VRAM.
- GPU utilisation says n/a where it cannot be measured, instead of 0.
- `available_vram_gb` is what is actually free.
- The topology crowns the node the cluster elected.
- `/api/nodes` reports an address that is reachable.
- The Server view fills in size and quantization.
- A bench record no longer reports a GPU figure nobody measured.
- Training jobs survive restarts, resume, and accept AutoData output.
- The training image can be published, and a node says which images it tried.
- A forged UDP announcement can no longer claim master.
- A cluster split across two releases is visible.
- The discovery port is 5679 in code too.
- `ainode update` no longer reports success for an update that did not apply.
- `ainode update` removes the images it replaced.
- Cluster update job state survives the restart the job itself causes.
- The default distributed executor is `"mp"`.
- `pytest tests/` no longer starts a vLLM container.
- The speed bench slugs its `--label` into the record filename.
- The installer gives the real reason `/mnt/shared-models` must exist.
- Every launch and unload button does what it says.

---

## [0.5.27] - 2026-09-19

- A real `ainode doctor`.
- `ainode doctor --fix` applies only what cannot lose anything.
- `/v1/responses`, `/tokenize`, `/detokenize`, `/v1/rerank` and `/v1/score` through the fleet endpoint.
- Disk figures in `/api/status` and the cluster graphic.
- Authentication the dashboard can use.
- The header says how the port is protected.
- `ainode auth enable` on a node that already has a key stops crashing.
- Fine-tuning produces finite weights.
- A run that goes non-finite FAILS instead of reporting success.
- A job the engine cannot launch answers 400 and leaves the queue working.
- A missing training image is named before the job runs.
- The Hugging Face token stops travelling through job files and job logs.
- An unauthenticated port 3000 is no longer root-equivalent.
- `trust_remote_code` can no longer be set by an unauthenticated caller.
- Two training templates that promised runs the product cannot do.

---

## [0.5.26] - 2026-09-19

- A fresh install can load a model.
- The installer no longer writes a config that refuses every stacked load.
- The installer pre-pulls the engine image the engine actually runs.
- A fresh node no longer boots into a launch of a model nobody picked.
- `ainode logs` tails the log the configured backend writes.
- `sudo ainode update` no longer keeps the old image and calls it a success.
- `scripts/install.sh --dry-run`.

---

## [0.5.25] - 2026-09-19

- Embeddings served by the fleet.
- Catalog entry: Qwen3 Embedding 0.6B, verified.
- `ainode-bench embed`: the fifth bench section, for a model that writes no tokens.
- A second model loaded on a node that HEADS a distributed launch is a stacked load, not a new primary.
- The head of a distributed launch stays in the node's announcement when a model is stacked beside it.

---

## [0.5.24] - 2026-09-19

- Qwen3.6 35B-A3B on a V100, through the catalog.

---

## [0.5.23] - 2026-09-18

- `POST /v1/decide`: typed questions in, probabilities out.
- `ainode-bench decide`: a decision bench scored on calibration, not accuracy.

---

## [0.5.22] - 2026-09-18

- The interface shows a loading model as loading, with elapsed and expected time.
- `ainode-bench agentic`: a fresh-agent rubric with mechanical verdicts.

---

## [0.5.21] - 2026-09-17

- The interface says how long a model takes to load.
- A `verified=True` catalog entry now says when it was proven and which bench record proves it.
- Qwen3.8-Flash-Next is marked verified and recommended.

---

## [0.5.20] - 2026-09-17

- `ainode-bench harness --claude-effort <level>` sets the reasoning effort Claude Code runs with.
- The harness bench record's `placement.node` now names the node that serves the model, not the master the CLI talked to.
- The harness bench's OpenCode adapter now runs in the task's directory instead of the one the bench was started from, which is the whole of [#118](https://github.com/getainode/ainode/issues/118).
- An OpenCode bench run no longer reads the operator's global config, and a failed one now says what failed.
- The bind wait watches what the engine is DOING; log silence is only the second opinion.
- `engine_ready` in `/api/nodes` means the engine answers `/v1/models`, not that the node is reachable.
- Keep distributed ranks in step through FlashInfer autotune.
- Catalog entries state their active parameter count and architecture, so a bench record stops reporting a MoE as dense.

---

## [0.5.19] - 2026-09-16

- The fleet endpoint forwards the Anthropic Messages API.
- Harness bench adapter: `claude`, Claude Code driven against a served model.

---

## [0.5.18] - 2026-09-16

- A distributed launch now serves a model that was downloaded through AINode instead of making every node re-download it.
- Catalog entry: Qwen3.8-Flash-Next (NVFP4), `nvidia/Qwen3.8-Flash-Next-NVFP4`, as a two-node `mp` launch.
- A harness bench: `scripts/ainode-bench.py harness` measures whether a served model can drive a coding agent to passing tests.

---

## [0.5.17] - 2026-09-16

- The bind wait's log-silence window defaults to 900 s, up from 360 s.

---

## [0.5.16] - 2026-09-16

- The chat Thinking toggle and the bench's reasoning section now reach DeepSeek V4.
- The chat MODEL CARD stays inside the left rail at any panel width.

---

## [0.5.15] - 2026-09-16

- The bind wait's log-silence window defaults to 360 s, up from 120 s.

---

## [0.5.14] - 2026-09-16

- The solo engine's logs are followed after the detached launch.

---

## [0.5.13] - 2026-09-15

- A solo engine is no longer declared dead the moment its `docker run -d` client returns.

---

## [0.5.12] - 2026-09-15

- A multimodal request is routed to an instance that accepts images.
- A distributed instance reports its real parallelism to the master.
- Startup replay sweeps first, then launches one engine at a time per node.
- DeepSeek V4 Flash (DSpark, FP8) is marked verified and recommended.

---

## [0.5.11] - 2026-09-14

- `NCCL_IB_HCA` names only the RoCE port behind the cluster interface.

---

## [0.5.10] - 2026-09-14

- The mp distributed launch maps `/dev/infiniband` into the engine containers on a real node.

---

## [0.5.9] - 2026-09-14

- A distributed launch that needs no Ray in the engine image: the vLLM `mp` multi-node shape.
- The distributed launch is recipe-aware.
- Catalog entry: DeepSeek V4 Flash (DSpark, FP8).
- `VLLM_ATTENTION_BACKEND=TRITON_ATTN` is no longer forced onto a custom or newer engine image.
- The cluster interface is auto-detected instead of guessed.
- `ainode start` on a host with no vLLM fails with an explanation, not a traceback.
- Replay no longer launches every engine twice.

---

## [0.5.8]  -  2026-09-14

- Static assets can no longer go stale across an update.
- The model card only renders on the Chat view.
- Startup replay no longer kills slow-but-healthy engines.
- Use consistent product branding.

---

## [0.5.7]  -  2026-09-13

- Fleet-aware chat.
- Bench view.
- Ornith 1.5 35B-A3B (NVFP4).
- Download job progress race.

---

## [0.5.5]  -  2026-08-19

- Startup replay retries an engine that dies on the way up.

---

## [0.5.4]  -  2026-08-19

- Models launch with their own engine image and flags.
- Catalog recipes.

---

## [0.5.3]  -  2026-07-07

- Report instance readiness from live engine probes.

---

## [0.5.2]  -  2026-07-06

- Download Cancel is real.
- Launch form respects user node picks.

---

## [0.5.1]  -  2026-07-06

- Per-load overrides survive restarts.
- fp8 KV cache off by default for multimodal models.
- Stacked-load admission guard.
- Training accepts on-disk models.
- Training/merge no longer require live PyPI.
- Node-targeted launches.
- Stacked instances visible.
- GUI lifecycle polish.
- AutoData uses a held-out validation objective.

---

## [0.5.0]  -  2026-07-06

- AutoData (training utility).
- Training in a spawned GPU container.
- Training view completed.
- Launch path: served-model-name aliases + per-load overrides.

---

## 0.4.x

Introduced container-based installation and updates, federated inference routing,
model stacking, shared model storage, catalog recipes, quantization and Hugging Face
uploads. Improved chat, downloads, training artifacts, metrics, node selection and
distributed launch configuration.

## 0.3.x

Added the API console, topology view, cluster configuration, training and dataset
workflows, download progress, chat history and model deletion. Improved discovery,
cache scanning and nonblocking startup.

## 0.2.x

Added the training dashboard, live resource widgets, model management and optional
API-key authentication. Fixed packaging of dashboard templates and static assets.

## 0.1.x

Introduced the CLI, GPU detection, inference proxy, UDP discovery, metrics, model
catalog, training queue, systemd service management and browser dashboard.
