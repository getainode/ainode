# Changelog

All notable changes to AINode are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).  
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

_Nothing yet._

---

## [0.5.30] - 2026-09-19

AINode grows up: a fresh install is protected by default and a fleet can run with auth on everywhere, a tailnet certificate is one command with renewal, a restart or an update keeps the engines that are serving, Whisper runs on the fleet with a speech bench behind its verified flag, the dashboard draws the metrics it keeps, a download is refused when it will not fit, the doctor stops warning about what it cannot see, and Castor's Flash-Next lane is a catalog entry.

### Added
- **Qwen3.8-Flash-Next NVFP4 runs on four Tesla V100s through the launch path.**
  A new curated entry, `qwen3.8-flash-next-nvfp4-v100`, carries the TP=4 Volta
  recipe that castor had been running as a hand-rolled container: the SM70
  attention backend, a BF16 KV cache, 262144 context, the qwen3 reasoning and
  qwen3_coder tool parsers, and the checkpoint's own 4-token MTP module, on the
  local 1Cat-vLLM build that is the only engine able to run Qwen4Exp on Volta.
  Verified on castor 2026-09-20: ready in 855 s, 49.6 tok/s single-stream,
  65.3 sustained, 104.2 across 16 streams, and 20/22 on the quick agentic rubric.
- **A catalog id beats an hf repo when two entries share a checkpoint.**
  `catalog_recipe` resolves ids in their own pass, so a hardware-specific lane
  can be loaded by name without depending on which entry the catalog reaches
  first. A bare repo id keeps answering exactly as it did.
- **A bench record states the width of a single-node tensor-parallel launch.**
  A solo instance record counts nodes, so a launch across four cards in one box
  recorded `tp: 1` beside its own `--tensor-parallel-size 4`. The width is read
  back from the launch flags now, and only ever upward, so an unstated width
  still records as 1 rather than a guess.
- **A speech section for the bench: `scripts/ainode-bench.py speech` measures word error
  rate, latency per clip and real-time factor against audio the repo ships.** Ten clips
  in six voices across six English locales are committed under `bench/speech/clips/`,
  because a word error rate is only comparable over the same bytes, and the reference is
  the exact text they were made from rather than anything transcribed by hand. Two rates
  are reported, one folding number words to digits and one orthographic, so a transcript
  that heard every word and wrote "nine" for "9" is not scored wrong for its spelling and
  a reader can still see how much of the error was spelling. The one warm-up request is
  recorded rather than hidden: on `--enforce-eager` the first transcription after a
  launch compiles the kernels and took 89 seconds against 0.7 for every one after it.
- **`openai/whisper-large-v3-turbo` is verified, with a record behind it.** It serves on
  a GB10 stacked at 6 percent beside a 30B MoE chat model and an embedder, ready in 110
  seconds, at 2.3 percent word error rate with 8 of 10 clips word for word, 739 ms median
  per clip through the fleet endpoint and a real-time factor of 0.18, so it transcribes
  about five times faster than the audio plays. The entry names its bench record and
  carries the launch time as `typical_ready_minutes`.
- **A cluster can run with auth on.** Every node-to-node request AINode makes now
  presents a fleet key derived from the cluster's shared secret,
  `HMAC-SHA256(cluster_secret, "ainode-fleet-key-v1")`, and the middleware accepts it
  as the caller id `fleet`. Every node holding the secret computes the same key, so
  `ainode join` is all a new node needs; rotation follows the secret with no restart;
  and nothing is written to disk for it. The load and unload fan-outs, `update-all`,
  the model card's read of a peer's config, the bench's node reads and the doctor's
  peer probes all carry it. Engine ports do not, because a vLLM container never sees
  AINode's auth.
- **`ainode auth key create --name <client>`, `ainode auth key list`, and `ainode
  auth key revoke <id>`.** A key is minted for one client with a name on it, shown
  once, and `list` says which client holds which key and when it was created.
- **Two doctor checks.** `auth.state` fails a node that has auth on and peers but no
  `cluster_secret` (its own cluster cannot call it), warns when auth is off on a node
  bound to something other than loopback, and is OK otherwise. `cluster.secret` warns
  when there is no secret at all, because discovery is then unauthenticated and no
  peer can authenticate either.
- **`/api/health` reports the running version.** It is the one route that answers
  without a key, which is what `ainode update` needs to verify a release on a node
  that requires one.
- **The dashboard draws the metrics a node has been keeping.** A new Metrics view charts GPU memory against the node's total, GPU utilization, temperature, request rate with errors, the latency percentiles and process uptime, over 1 h, 6 h, 24 h or 7 d of the node's own retained history. Hand drawn on canvas with no library, and theme-aware through the existing design tokens. A gap in the data is a gap in the line, a counter's rate across a restart is absent rather than negative, and a series the node cannot measure is said in words: on a DGX Spark the driver exposes no GPU utilization counter, so that panel says so instead of drawing a flat zero that would read as an idle GPU.
- **Any node's charts, from any node.** `GET /api/metrics/history?node=<id or name>` fetches that node's own history and passes it through, so the dashboard can show a peer's measurements without a head inventing them. A node that cannot be reached is an error naming it rather than an empty chart.
- **The Prometheus latency summary tells the truth about an idle node.** `ainode_request_latency_milliseconds` is absent until a request has actually been timed, where it used to publish three zeros that read as instant answers, and it no longer carries a `_sum` that was always 0 and made every average read as zero latency. `ainode_gpu_available` now answers 1 on a healthy node as its help text always promised, and `ainode_model_loaded` carries a `tp` label so a multi-node instance is not indistinguishable from a solo one.
- **`ainode tls enable --tailscale` works, and it runs on the host.** The
  installer's wrapper resolves this node's MagicDNS name, runs `tailscale cert`
  into `<AINODE_HOME>/tls/<name>.crt` and `.key` (plain first, then `sudo -n`, so
  `tailscale set --operator=$USER` is honoured), and hands off to the container,
  which adopts the pair and writes the `tls` block. The CLI is inside a container
  with no tailscale binary and no daemon socket, so this was the one shape that
  could not work before.
- **The certificate renews itself.** `ainode-tls-renew.timer`, rendered by the
  installer for every node whatever its TLS state, runs daily, asks the node for
  the decision, and inside the last 14 days re-runs `tailscale cert` and restarts
  so the new pair is served. It renews into the files the config points at,
  restarts only when the expiry actually moved, and refuses to replace a
  self-signed pair or a certificate issued outside the tailnet.
  `AINODE_TLS_RENEW_RESTART=0` leaves the restart to a human.
- **`ainode tls renew [--check]`**, the decision half: `key=value` lines and exit
  10 for "nothing to do", which is what lets a daily timer stay silent for 75
  days.
- **A download that cannot fit is refused before it starts.** AINode learns a
  checkpoint's size from the Hugging Face API (cached under `AINODE_HOME`) and
  compares it with free space on the models directory: a pull with nowhere to go
  is a 507 naming what is needed, what is free and which path was measured,
  instead of a transfer that dies part way and leaves a partial blob cache behind.
  `{"force": true}` downloads it anyway, a size nobody can learn is reported as
  unknown and downloads as before, and the model card shows the size with whether
  it fits on that node.
- **`ainode doctor` warns when persistence mode is off on a node with discrete
  GPUs.** With it off, every process that opens a GPU pays a full initialisation,
  which is how a node with no engine loaded can stop answering its own API. One
  command fixes it, and the check names it.

### Changed
- **The README describes the product that ships.** The CLI reference gained `doctor`, `update` (verify and prune), `prune-images`, `auth`, `tls`, `cluster token` and `join`; the feature rows for fine-tuning, TLS, rate limiting, the failover endpoint and speech match 0.5.27 to 0.5.29; four training rows that were dated 0.5.27 now say 0.5.28, where they shipped.
- **A speech run driven at a master now records the right node's neighbours.**
  `describe_via_http` keyed a record's `stacked_with` on the node it was asking rather
  than the node serving the model, so a run pointed at the master named the master's own
  stack for an instance on a peer. It takes the resolved serving node now, which is the
  normal shape for a speech or embedding instance: those stack on a peer by design.
- **`bench/README.md` lists every bench again.** It said four and had never been updated
  for the embedding section; it says six and describes both the embedding and the speech
  ones, with the command to run each.
- **A fresh install requires an API key.** The installer mints one, stores its
  SHA-256 in `~/.ainode/auth.json`, prints it once in a box at the end, and the
  summary line reads "API protected, one key". `AINODE_AUTH=off` keeps the previous
  open node and prints what that choice means. An install over an existing
  `config.json` changes neither the auth state nor the keys, so no update turns auth
  on under a running fleet.
- **An `ainode auth` change takes effect immediately.** The server re-reads
  `auth.json` when it changes instead of holding what it read at startup, so
  `enable`, `disable`, `key create` and `key revoke` no longer need a service
  restart, and each command says so.
- **`auth.json` and `config.json` are written 0600** and atomically, and an existing
  wider file is tightened on load. `config.json` carries `cluster_secret` and may
  carry `hf_token`.
- **The API access panel, the README and the join route's docstring state the real
  set of keyless paths** (`/`, `/static/*`, `/api/health`, `/api/auth/status`,
  `/api/cluster/endpoint`, `POST /api/cluster/join`), each with its reason. They
  named three of the six.
- **A graceful shutdown leaves a still-serving engine container running.** Stopping
  it was what left the next boot with nothing to adopt, and every STACKED engine
  already outlived the orchestrator, so the primary and the head now behave the same
  way. A container that is not running is still stopped through the backend (which
  reaps a corpse and a head's peer containers), the `docker logs -f` follower is
  always closed, and the line on the way out names `docker rm -f <container>` for
  freeing the GPU. `touch ~/.ainode/.start-clean` still starts a node idle: it adopts
  nothing by design. The FIRST restart after this release still reloads on every
  node, because the process being stopped predates the fix.
- **A node advertises the scheme it really serves.** `url` on every
  `/api/cluster/endpoint` row, in `endpoint_hint` and in the `self` and `master`
  blocks is `https://host:<tls port>` when that node opens a TLS listener, with
  `tls` and `tls_port` beside it. `port` stays the HTTP port in every row, always:
  the peer proxy and every older client build `http://host:port` from it. Peers
  keep talking HTTP on the LAN, and only a node's own rows can claim TLS.
- **`ainode doctor`'s TLS check names this node.** With TLS off it prints the
  node's MagicDNS name and `ainode tls enable --tailscale (a real certificate for
  <name>)`; inside the renewal window on a tailnet certificate it points at
  `ainode tls renew` and says the timer already runs it. The name is found from
  the wrapper's environment variable, `tailscale status`, or a reverse MagicDNS
  lookup of this node's own tailnet address, bounded at two seconds.

### Fixed
- **`hf_token` no longer leaves the node in `GET /api/config`.** Only
  `cluster_secret` was scrubbed, so a Hugging Face credential that can write to the
  operator's own repositories came back in clear to any caller, and on an open node
  to anyone who could reach the port. `ainode config` masks it as "set (hidden)".
- **`ainode update` no longer reports a failed update on a protected node.** The host
  wrapper verified the running release on `/api/status`, which needs the key, so
  every update would have pulled, pinned, restarted and then exited non-zero saying
  it had not applied. It reads `/api/health`.
- **`POST /api/cluster/update-all` no longer crashes on a peer that announced no
  source address.** It read `n.host`, which `ClusterNode` does not have, so one such
  peer took the whole handler down with an `AttributeError` instead of updating the
  fleet. It uses the one derivation of a peer's reachable address.
🤖 Generated with [Claude Code](https://claude.com/claude-code)
- **A restart no longer reloads a model the node is already serving.** Engine
  containers outlive the orchestrator, but `ainode start` freed every one of them
  before the boot engine relaunched `config.model`, and the orchestrator stopped its
  own engine on the way out, so a `systemctl restart ainode` (and therefore every
  `ainode update`) cost a full model load: 420 s on pollux, 834 s on castor, measured.
  A boot now inspects the container a launch would have created BEFORE it sweeps
  anything, and keeps it when it is running, matches the configured recipe (model,
  port, engine image, TP width, executor) and is answering (`/health` plus
  `/v1/models` naming that model). A kept engine is skipped by the sweep, not
  relaunched, and recorded in the `InstanceManager` with `adopted: true` and
  `serving`; anything that fails one of those checks reloads exactly as before, with
  one line saying which. Same treatment for a distributed head and for every stacked
  instance the manifest names. Measured on pollux: a restart that cost 420 s now has
  the model answering through the master in 1 s, on the same engine container, with
  nothing reloaded and no phantom row in the launch-time ledger.
- **Every view that lists instances now says which ones were adopted.**
  `/api/nodes` projects each announced instance down to the keys it names and was
  dropping the flag, so nothing could tell an engine that survived a restart from one
  the reporting process launched; `/api/status` gained `engine_adopted` for the node's
  own port.
- **`ainode doctor` inside the container no longer carries a WARN nobody can
  clear.** The systemd, docker and image checks are about the host, and the
  documented deployment runs the CLI in a container: they print INFO naming the
  host command now, and the host wrapper hands the real unit state in so the
  service line is a real answer.
- **A slow GPU driver can no longer make a node look dead.** NVML is sampled on a
  worker thread with a timeout and the last good sample is served while a read is
  slow, so no request waits on the driver. A stale sample is never recorded in the
  retained metrics history.
- **A replayed engine that dies on the way up says why.** Its last 40 log lines go
  into the AINode log before its container is removed, stacked instances wait for
  the primary to finish binding rather than for a fixed window, and the bind window
  grows with the number of engines coming up at once.
- **`ainode doctor --fix --peer HOST` no longer ignores `--fix` silently.** It
  refuses the combination and names the command to run on that node.
- **`ainode update` exits non-zero when it cannot resolve a version to verify
  against.** It pulled a floating tag, restarted and returned success without ever
  confirming what came back.
- **`ainode prune-images` reclaims the Docker Hub mirror tag that nodes actually
  carry.** The repository list had the mirror name misspelled, so those tags held
  the bytes of images the prune reported as removed.

### Removed
- **The unused metrics ring buffer in the dashboard.** It was seeded from the same history route the charts now read, and nothing ever drew it; keeping a 20 minute buffer beside a 48 hour store would be two sources for one line.

---

## [0.5.29] - 2026-09-19

Wave 2 of the 19 September audit: a head keeps its multi-node model across a restart or says why, one command joins a node, metrics survive a restart and /metrics finally scrapes, a client endpoint that survives the master, TLS on a second port and per-client limits, and speech routed through the fleet.

### Added
- **A head writes down what it is serving, and says so when it cannot serve it.**
  The distributed shape is persisted to `<AINODE_HOME>/distributed.json` on a
  successful launch and removed on unload. On the next start, if the engine
  container is gone, the shape is relaunched only when every peer answers the
  same ssh probe the launch depends on; otherwise the record is marked degraded
  with which peer answered what, and that shows up as a WARN in `ainode doctor`,
  as `degraded_instances` in `/api/status`, and as a banner in the dashboard top
  bar. One attempt per start, never a retry loop.
- **A node joins a cluster with one command instead of a hand-edited `config.json`.** `ainode cluster token` on the master mints 32 random bytes, keeps only their SHA-256 hash in `<AINODE_HOME>/join-tokens.json` with a 30-minute expiry (`--ttl`) and a single use, and prints the `ainode join <master> <token>` line to paste. On the new node that command writes the master's `cluster_id`, the cluster's `cluster_secret`, the master's address, the discovery port and the member role, leaves every other key in the file exactly as it found it, and restarts the service or prints the command. `AINODE_JOIN="<host>[:<port>]:<token>"` does it as part of the install, and the dashboard does it from **Config, Cluster**. Joining a node used to mean typing `cluster_id`, `cluster_role` and `cluster_interface` into `config.json` on the new box, which is what joining pollux took, while "automatic clustering" was the headline claim ([#208](https://github.com/getainode/ainode/issues/208)).
- **A fresh install generates a `cluster_secret`, so a new node is not running unauthenticated discovery by default** ([#169](https://github.com/getainode/ainode/issues/169)'s follow-up). Two nodes installed independently each generate their own and are invisible to each other, so joining is what copies one value onto the other, and `AINODE_CLUSTER_SECRET` pastes the master's value at install time for a node that will not join. The installer prints which it did and what the second node needs. Nothing already installed is touched: the value is written only when there is no `config.json` yet.
- **`POST /api/cluster/join`, the fourth and last route that answers without an API key.** A node that has not joined cannot hold this cluster's key, so the join token is the credential instead: a wrong, an expired and a spent token all get one identical 403 (telling them apart would make the route an oracle for guessing), and the handler allows five attempts a minute per source address because there is no key in front of it. `POST /api/cluster/join-self` is the keyed twin the dashboard card calls, and it restarts nothing.
- **A node now remembers what it measured.** Metrics are kept in a small SQLite file at `<AINODE_HOME>/metrics.db`, 48 hours of raw samples and 30 days of one-minute roll-ups, so a restart no longer resets the GPU series, the request counters and the latency percentiles to nothing. A figure the node could not measure is stored as null and read back as null, never as zero. Retention is configurable in a `metrics` block of `config.json` (`enabled`, `retention_hours`, `retention_days`, `interval_seconds`) and costs about 70 MB per node at steady state, measured. A read-only or full disk logs one warning and keeps serving.
- **`GET /api/metrics/history` serves any series over any window.** Raw samples or one-minute roll-ups depending on how far back the window reaches, every series on the same evenly spaced grid, `null` in every slot nothing measured. `since` and `until` take a timestamp or a relative offset such as `-6h`. The existing `/api/metrics`, `/api/metrics/gpu` and `/api/metrics/requests` shapes are unchanged.
- **The Prometheus endpoint can be scraped for the first time.** `GET /metrics` answered 500 on every request since 0.4, because the handler built a response with both a Content-Type header and aiohttp's `content_type` argument. Every series now also carries `node` and `node_id` labels, so a fleet scraped into one Prometheus is told apart by node name rather than by address, and there are new `ainode_model_loaded` and `ainode_metrics_retention_*` gauges. `docs/prometheus-scrape.yml` is a working scrape config.
- **The dashboard's metrics buffer is seeded from disk at page load**, so a chart drawn over it opens on the node's real recent history instead of one point from the moment the page loaded.
- **A client endpoint that survives the master.** `GET /api/cluster/endpoint` answers on every node with the fleet's addresses: this node, the elected master, and every online member with a URL a client can use. It needs no API key, like `/api/health`, because a client whose node has gone away has nothing else to ask and no key to ask with, and it carries nothing but names, addresses, ports, roles and versions. Clients no longer hold one address as a single point of failure for a cluster that does not have one.
- **`/api/status` now carries `endpoint_hint`**, the same node list, so anything already polling status learns its fallbacks for free. `localhost` never appears in either payload: the local node reports the address the caller reached it on (when the `Host` header names one of its own), a peer the address its announcement arrived from, and a node with nothing routable reports an empty host and a null URL rather than an address that is wrong on every machine but its own.
- **TLS on its own port, with HTTP left exactly where it was.** `ainode tls
  enable` generates a self-signed certificate (hostname, LAN IP and tailnet IP as
  SANs, 825 days, key mode 0600), installs a pair you already have, or runs
  `tailscale cert` for a real Let's Encrypt certificate on the tailnet name, which
  is what a Mac client with App Transport Security needs. The pair lives in
  `~/.ainode/tls/` so the CLI and the containerised server see the same files, and
  the node serves HTTPS on port 3443 (configurable) alongside the plain HTTP port
  every client in a fleet already uses. `ainode tls status` and a `tls` block on
  `/api/status` say what is served and when the certificate expires; `ainode
  doctor` warns two weeks out and fails on an expired one.
- **Per-client rate limits on the inference API.** A token bucket
  (`requests_per_minute`, `burst`) and a concurrency cap (`max_inflight`) keyed by
  API key id, or by address for an unkeyed caller, so one client can no longer
  open two hundred concurrent completions and occupy every engine in the cluster.
  Off by default, `/v1` only: health, the dashboard's own `/api` routes and static
  files are never limited. A refusal is a 429 with `Retry-After` and a body naming
  the limit, and a streaming answer holds its slot until the stream ends.
- **Speech to text is a fleet feature: `POST /v1/audio/transcriptions` and
  `/v1/audio/translations` now route across the cluster like a chat completion.**
  They are multipart uploads rather than JSON, so the proxy reads the `model` form
  field out of the body it buffered and forwards the identical bytes under the
  caller's own boundary, with the same candidate ordering, failover, header and
  query passthrough every other forwarded path gets. A body that names no model
  gets a 400 naming the field instead of being sent to whichever model the node
  happens to serve.
- **`openai/whisper-large-v3-turbo` is in the curated catalog**, capability
  `speech`, 1.6 GB of weights at 6 percent of a GB10 so it stacks beside a chat
  model, with the launch recipe it needs (`--enforce-eager`, `kv-cache-dtype auto`)
  and an engine image that carries vLLM's audio extras: no image on the fleet has
  librosa or soundfile, and vLLM imports soundfile as soon as the model reports the
  transcription task, so a Whisper serve on the stock image dies before it binds a
  port.
- **That engine image is published: `ghcr.io/getainode/ainode-whisper:0.17.0-t5`**,
  built from `scripts/Dockerfile.whisper` by the new
  `.github/workflows/publish-whisper-image.yml` on a `whisper-v*` tag or a manual
  dispatch. Its tag is the base engine image's tag rather than an AINode version,
  because the image tracks the engine it derives from and a release does not rebuild
  it. A catalog entry pinning a hand-built tag would be a LAUNCH button that fails
  on every node but the one that built it.

### Fixed
- **A restart on a distributed head no longer loses the multi-node model.** The
  engine containers outlive the orchestrator, so a head came back with an empty
  `InstanceManager` while its container kept serving: the node's own view of what
  it ran was reconstructed from `config.json` by three separate fallbacks, an
  UNLOAD naming the head's port matched nothing, and once the container did stop
  the model was gone with nothing to bring it back. A node now adopts the engine
  containers it is still running at startup, with the shape read from each
  container's own argv, so `/api/status`, `/api/nodes`, `/v1/models` and the
  cluster graphic describe a real instance from the first poll. A live stacked
  container that no manifest entry knows about, which is what a crash between a
  launch and the manifest write leaves behind, is adopted the same way and
  written back into the manifest.
- **A solo load can no longer replace a multi-node instance by accident.**
  Re-loading a model already serving here across several nodes is refused with a
  409 naming the width and the peers, instead of stopping the distributed engine
  and bringing the model back on one node.
- **The dashboard no longer claims there is no TLS.** Config > API access read out
  a fixed sentence saying a key travels in plain text; it now reads the node's real
  TLS state, including the certificate's kind and expiry, and says so plainly when
  the browser has no key to read `/api/status` with.
- **A launch refused over a missing engine image now says which image, and how to
  get it.** The launch returned a bool, so the reason stayed in the node's log while
  the API answered "Failed to launch engine": the one failure an operator can always
  fix, reported as the one thing they cannot act on. Backends now carry
  `launch_error`, `ensure_image` fills it with the image, what docker said, and the
  fixes (pull the tag, build it from the Dockerfile this repo ships, or point the
  instance at an image the node has), and both the solo and the distributed load
  routes report it verbatim.
- **A speech instance is its own kind across the interface.** It stays out of the
  chat and bench pickers the way an embedding instance does, the capability probe
  reports `speech` off the catalog instead of asking a chat question of an engine
  with no chat path, and the model card, the capability badges, the endpoint list
  and the Server view's copy-curl button all know what it serves.

### Removed
- **The browser onboarding wizard, which no deployed node could reach and which never joined anything.** `handle_index` redirected to `/onboarding` only while `config.onboarded` was false, and it never is: the installer writes `"onboarded": true` and every non-TTY start sets it before the server binds, so `/onboarding` answered a redirect to a page that redirected straight back. Its complete handler set a node name, a model and an email and touched no cluster key, downloaded nothing and launched nothing. The 19 KB template, the three `/api/onboarding/*` routes and the auth exemption they needed are gone; joining is `ainode join` and a card in **Config, Cluster**, and the terminal wizard on a TTY `ainode start` is unchanged ([#208](https://github.com/getainode/ainode/issues/208)).

---

## [0.5.28] - 2026-09-19

Wave 1 of the 19 September audit: telemetry that stops inventing numbers, signed discovery with versions on the wire and an update that verifies and prunes, training jobs that survive a restart, launch and unload buttons that do what they say, and docs that describe the system that ships.

### Added
- **`ainode prune-images`.** The reclaim step as its own command, so it can be run by hand on a node that has been updated many times, and so the update path has one place to call. `--keep-images N` sets the rollback generations to keep, `--current IMAGE` names the baseline (default: `image.env`, which is what the systemd unit actually reads, then `$AINODE_IMAGE`, then this build's tag), `--dry-run` prints the decision and removes nothing, `--verbose` also prints every image it keeps and why, and `--images-from FILE` decides against a `docker images` listing copied from another node and implies a dry run, which is how a prune can be reviewed for a node you are not on.

### Changed
- **The repo's own docs describe the system that ships.** README, CLAUDE.md, AGENTS.md, SECURITY.md and the Dockerfile header stopped describing the retired Ray launcher, a one-container install that never existed, a Docker Hub mirror frozen at 0.4.7 and a `nvcr.io` engine image nothing uses. Every throughput figure without a bench record is gone; the six catalog entries that claimed `verified` with no record now say `verified: False`, and the rule that a `verified` flag needs a record is written next to the flag. The 158 audit rows across the product repo, the docs site and ainode.dev are closed by file and line in the PRs (#216, docs PR 5, site PR 29). Closes #200, #201, #211.

### Fixed
- **A multi-GPU node counts as all of its GPUs** ([#163](https://github.com/getainode/ainode/issues/163)). The collector read device 0 and the cluster counted one GPU per node, so castor's four Tesla V100s announced one 32 GB GPU and the fleet's nine GPUs read as six. Every number built on that was short: the cluster's total VRAM by 96 GB on that node alone, the topology's GPU count, and the per-node figure the browser sizes a model against. `detect_gpus()` enumerates every NVIDIA device and sums their memory; the announcement carries a `gpu_count`; `/api/cluster/resources` sums that field instead of counting rows; and `/api/metrics/gpu` carries the per-device list. Proven read-only on castor from this branch: `gpu_count=4`, `memory_total_mb=131072`, four named devices, against the 32 GB the shipped code reported from the same host.
- **On a unified-memory node, host RAM is no longer reported as VRAM** ([#175](https://github.com/getainode/ainode/issues/175)). NVML answers `NVMLError_NotSupported` for memory on a GB10, so the collector fell through to `psutil.virtual_memory().used` and published it as GPU memory used: page cache and every non-GPU process included, which pinned every Spark at 84 to 100 percent whatever the user did, and made the gauge convey nothing. Memory on such a node is now reported as `memory_kind: "unified"`, and usage is the one figure AINode actually knows: the sum of `gpu_memory_utilization` across the engines it launched, which is also what the stacked-load admission gate already reasons about. The host reading is still reported, as `system_memory_used_mb`, labelled as system memory and never as VRAM. Proven on Spark-3: 108728 MB of host RAM in use, reported as 74767 MB reserved by engines (0.6 of 124611) with `memory_used_source: "engine_reservations"`.
- **GPU utilisation says n/a where it cannot be measured, instead of 0** ([#176](https://github.com/getainode/ainode/issues/176)). `nvidia-smi` on a GB10 prints `0 %` next to `[N/A]` memory: the driver does not populate the counter, and the 0 is the struct's default. Published as a percentage it read as an idle node, on every node in the fleet, while they were serving requests. A device whose memory NVML will not report does not have its utilisation trusted either, so the figure travels as `null` from the collector through the announcement to the row, and the interface draws "n/a (not exposed here)" rather than a number. A discrete GPU is unaffected: castor's V100s still report a real utilisation, so a 0 there still means idle, which is the distinction that was missing. Temperature comes from the same NVML handle and still comes through on both.
- **`available_vram_gb` is what is actually free** ([#174](https://github.com/getainode/ainode/issues/174)). It was a copy of `total_vram_gb`, so the cluster view showed an empty fleet with six models loaded across five nodes, and anyone sizing a model against it was reading a number that is only true on an idle cluster. It is now summed from real per-node usage, and a node with no usable figure is not counted as free: it is named in `vram_unknown_nodes`, with `available_vram_is_floor` saying the total is a floor. When not one node can say, the field is `null` rather than the total twice. Each node's row carries `vram_used_gb` and `vram_free_gb` on the same terms. The browser follows: a node that does not report memory use is no longer treated as wholly free by the launch hint or the auto-placement recommender.
- **The topology crowns the node the cluster elected** ([#203](https://github.com/getainode/ainode/issues/203)). `topology.js` looked for `effective_role` or `is_leader` on each row, `/api/nodes` carried neither, and the fallback crowned `incoming[0]`, which is always the local node because `ClusterState` inserts itself first. So Spark-3's dashboard crowned Spark-3 and Spark-4's crowned Spark-4, while the Config view's members table showed the real master from the same server's `get_master()`: two answers in one product, and the graphic had the wrong one. The rows now carry `effective_role` and `is_leader` from that same election, and the row-order fallback is gone: with no role information no node wears a crown. The node at the centre of the graphic and the node wearing the crown are now separate ideas in the renderer, which is what let one stand in for the other.
- **`/api/nodes` reports an address that is reachable** ([#178](https://github.com/getainode/ainode/issues/178)). Every row said `host: "localhost"`, including remote ones, so anything built from it pointed at the viewer's own machine. Routing was never affected because the proxy computes its own host; the damage was to every other consumer. A row now carries `localhost` for exactly one node and, for a peer, the address its announcement arrived from (the listener already captures it with `recvfrom`, which is the management-LAN address a browser is on), falling back to the fabric IP. No new announcement field was needed for this.
- **The Server view fills in size and quantization** ([#180](https://github.com/getainode/ainode/issues/180)). Every row reported `size_bytes: 0` and `quantization: null`, including for models whose repo names say NVFP4, because nothing resolved a served model id back to the weights on disk. Size is now measured from the snapshot directory under the HF cache, following blob symlinks once rather than per link (an HF snapshot is a tree of links into `blobs/`, so a naive walk double-counts), cached for five minutes so the view can poll. Quantization comes from the catalog recipe first and the model's own `config.json` `quantization_config` second, with a `quantization_source` saying which. A model whose weights are not on this disk reports `null`, not 0. Proven against the real cache on Spark-3: Ornith 1.5 35B-A3B NVFP4 at 23.45 GB from the catalog, Qwen2.5-7B-Instruct at 15.23 GB with no quantization claimed, Qwen3.5-0.8B-awq reading `compressed-tensors` out of its own config.
- **A bench record no longer reports a GPU figure nobody measured.** `bench/fleet.py`'s in-process telemetry reader had the same `float(x or 0)` coercion, and a record is where it costs the most: `bench/SCHEMA.md` says a missing measurement is never filled with an estimate, and every GB10 run was writing a peak utilisation of 0 percent and a memory figure derived from host RAM. The reader passes the null through now, and `Telemetry.result()` already drops a series that was never measured, so those keys are simply absent from the block instead of present and wrong. The coercion rule has one home (`metrics/collector.py::optional_float`) that the collector, the announcement sender, `/api/nodes` and the bench reader all call, rather than four copies to drift apart.
The announcement grew one field (`gpu_count`) and made four existing telemetry fields nullable; a fully populated payload measures 1715 bytes against the 4096-byte listener ceiling on a live node.
- **Training jobs survive restarts, resume, and accept AutoData output.** Four defects that kept the training screen a demo. (1) `TrainingManager._jobs` was memory-only, so a restart emptied the Runs table, zeroed the stats tiles and 404'd merge, resume, logs and artifact download for every earlier job, while ten real job dirs sat on disk. Each job now writes `status.json` into its own job dir at every transition (the status setter does the write, because eight places move a job's status and one of them forgetting is how the registry drifts from disk again) and the registry is rebuilt from `JOBS_DIR` at construction. A job that was RUNNING when the process stopped comes back failed, since its process went with the restart and a phantom RUNNING job blocks the queue. A job dir written before this release has no status file, so it gets a best-effort verdict from what is on disk (completed if it left weights, failed otherwise, quantize jobs judged by their checkpoint in the models store), always with a `note` saying so, marked `restored`, and with no `start_time`, because `stats()` counts GPU hours from it and a duration nobody recorded must not be invented. A directory with neither a status file nor a config is not a job, which keeps the 6,000 empty dirs the old suite left out of the history. (2) Resume could not work in container mode: the orchestrator's checkpoint path went to the container unchanged with the source job dir unmounted, so HF answered "Can't find a valid checkpoint". The source job dir is mounted read-only at `/src` and the path rewritten into it, and the resumed job gets its OWN `output_dir` rather than writing a second run's checkpoints into the first run's directory. (3) AutoData writes `{"conversations": [...]}`, which `tokenize_fn` had no branch for, so the documented handoff died with an IndexError and the `sharegpt-chat` template advertised the same unusable shape. Conversation rows now render through `tokenizer.apply_chat_template` with ShareGPT `from`/`value` and OpenAI `role`/`content` both mapped onto chat roles, and a plain-text fallback for a tokenizer with no template; the whole rendered conversation is supervised, the same as the other branches, and assistant-only masking is not implemented so nothing claims it. (4) The queue only advanced from a submit or a resume, so a second queued job waited for a human to submit a third; the job's monitor now tells the manager when its process exits, and a next job that cannot start is reported on the finished job's log rather than raised out of the monitor task.
- **The training image can be published, and a node says which images it tried.** `ainode-quant:0.17.0-t5` was hand-built on the four Sparks and published nowhere, so a fresh node answered every training, quantize and merge job with `docker` exit 125. `.github/workflows/publish-train-image.yml` builds `scripts/Dockerfile.quant` on the self-hosted aarch64 runner and pushes `ghcr.io/getainode/ainode-train:<version>` on a `train-v*` tag or a manual dispatch, with an `alias_versions` input so one 22 GB build can answer for several releases (an alias tag re-uploads nothing). The engine resolves a job's image as `AINODE_TRAIN_IMAGE` / `AINODE_QUANT_IMAGE`, then that release tag for the running version, then the local hand-built tag, and the preflight error names all three with the pull, the build and the override. A docker that is missing or not answering is still reported as a broken docker, not as a missing image.
- **A forged UDP announcement can no longer claim master** ([#169](https://github.com/getainode/ainode/issues/169)). `cluster_secret` was declared in `NodeConfig`, scrubbed from `GET /api/config` as though it protected something, and read by no code path at all: a datagram was the entire join protocol. The cluster id is readable unauthenticated over HTTP, the election prefers any node announcing `role: "master"` with a low `node_id`, and an announcement also advertises the instances every peer will route `/v1/chat/completions` traffic to, so one host on the broadcast domain could take a fleet over and pull real inference to an address of its choosing. Announcements are signed now: HMAC-SHA256 over the payload, keyed by `cluster_secret`, and a node that HAS a secret drops anything it cannot verify, naming the source and the reason once per source so a forged flood cannot fill the log. The signature is ONE extra key beside the payload's own fields and never a wrapper around them, so a peer that has never heard of it drops the unknown key in `from_json` and stays in the cluster view, which is what keeps a fleet visible to itself mid-roll. Both directions read the secret per datagram rather than capturing it at startup, so rotating the key is a `config.json` edit plus one broadcast interval instead of a restart of every node, and a half-written config keeps the last value that parsed rather than blanking a fleet's authentication. Measured on the live fleet (Spark-2 as a TP=2 head): the announcement it sends today is 1016 bytes, 1097 signed, against the 4096-byte listener ceiling, so 2999 bytes of headroom are left. Replay is deliberately not covered: a recorded datagram re-sent verbatim still verifies, because an attacker cannot edit a field without the key, and the alternative (a freshness window over `timestamp`) would drop every announcement from a node whose clock has drifted, which on a cluster that shares no clock is a worse trade.
- **A cluster split across two releases is visible** ([#171](https://github.com/getainode/ainode/issues/171)). Nodes did not put their version on the wire, so a fleet running two releases looked exactly like one running a single release, which is the state every partial roll leaves. The announcement is also the cross-version contract (0.5.25 changed how a head's instances are merged into `instances`), so a 0.5.24 reader and a 0.5.25 sender were exchanging different data with nothing anywhere to say so. `ainode_version` now travels on the announcement and through `ClusterNode` into every `/api/nodes` row, the `/api/cluster/resources` node list, `/api/version/check` (per-node rows, the distinct releases ordered by release rather than by string, a count of nodes not announcing one, and a `cluster_split` flag) and `/api/cluster/update-status`. A peer too old to announce one reports `""` and is counted as unknown rather than folded into agreement: filling it in with the local version is how a split fleet goes on looking healthy. The dashboard header draws an amber banner next to the update badge naming each release and the nodes on it. All six nodes the fleet can see today announce no version, which is the blindness this closes.
- **The discovery port is 5679 in code too** ([#181](https://github.com/getainode/ainode/issues/181)). The `NodeConfig` default was 5678 and both discovery classes repeated it, while the installer wrote 5679, every node in the fleet runs 5679 and the public docs document 5679. A source install, or any node whose `config.json` predates the key, therefore broadcast and listened where nobody else spoke: it came up healthy, served its own model, and never appeared in a single peer's cluster view, with no log line anywhere about the mismatch. The value has one home now (`core.config.DEFAULT_DISCOVERY_PORT`), read by the config and by both discovery classes, and the port, the cluster id and whether the wire is signed are logged together at startup so a mismatch is greppable at both ends.
- **`ainode update` no longer reports success for an update that did not apply.** The wrapper printed "Update complete" and exited 0 whatever happened, because nothing ever asked the running node what version it was: the sudo-home bug fixed in 0.5.26 was invisible for exactly that reason, and so was every other way an old container can come back. After the restart the wrapper now polls this node's own `/api/status` until it reports the version just installed, and exits non-zero when it does not, printing what the node actually reports and what to check (`AINODE_UPDATE_VERIFY_TIMEOUT`, default 180 seconds). A node whose service was not running is pinned and left alone, and says so instead of claiming an update.
- **`ainode update` removes the images it replaced** ([#184](https://github.com/getainode/ainode/issues/184)). It pulled a release every time and never removed the one it replaced, so a node carried every release it had ever run under each of the three names the image is mirrored as: 180 images and 226 GB reclaimable on Spark-1, on a root filesystem at 82 percent, on a box whose whole job is holding 35 to 400 GB of model weights. The prune runs only after the new version is confirmed serving, so a failed update still has something to fall back to, and it keeps one rollback generation by default (`--keep-images N`, `AINODE_KEEP_IMAGES`) so `ainode update <older>` has a release to go back to. Decisions live in `core/image_prune.py` as a pure function of a `docker images` listing: only AINode's own app repositories are considered (`ainode-base` and engine images are never touched), a tag that is not a release is never removed, anything newer than the running release is left alone, removals go by `repo:tag` and never by image id because three mirrored tags share one image, and nothing is decided at all when the running release is not in the listing. Against the real listing from Spark-2 with one rollback generation kept, the decision removes four tags for 19.6 GB, and names the freed figure as the upper bound it is (shared layers are counted once per image by `docker images`).
- **Cluster update job state survives the restart the job itself causes** ([#182](https://github.com/getainode/ainode/issues/182)). `/api/cluster/update-status` read a module-level dict, and the master updates the workers first and self-stops as the last step, so the record died with the process that was reporting on it and a poll after the master came back got a 404 for a job that had fully succeeded. The state is written to `<AINODE_HOME>/cluster-updates.json` on every mutation, the same bind-mounted directory `instances.json` lives in, and job ids are wall-clock stamps so "the most recent job" still means that across a restart. The poll additionally reports the version each node is announcing right now against the target, with `on_target` per node and a `verified` flag for the whole job: whether a node came BACK on the new image is a live question, and not one a record written moments before a container stopped gets to answer. `null` where it cannot be told (no target, or a node not announcing a version) rather than `false`, which would read as failure.
- **The default distributed executor is `"mp"`** ([#172](https://github.com/getainode/ainode/issues/172)). It was `"ray"`, and no image AINode ships or launches has ray in it: not the orchestrator image (`python:3.12-slim` plus this package) and not `vllm/vllm-openai`. A distributed launch that did not carry a catalog recipe naming `"mp"` therefore died inside the container, which meant the default only ever bit uncurated models, which is the worst place for it to bite. `"mp"` needs nothing beyond vLLM and is the shape every proven distributed launch on this fleet used. The value has one home (`core.config.DEFAULT_DISTRIBUTED_EXECUTOR`) and the defensive `or "ray"` fallbacks at three call sites now read it, so a config with the key empty behaves like a fresh one. A catalog entry that needs ray says so and gets it: recipe merging dropped `"ray"` as if it had never been stated, which was invisible only while ray was also the default.
- **`pytest tests/` no longer starts a vLLM container.** `test_failed_load_clears_model_claim`
  stubbed the engine at `app["engine"]`, which the solo load path stopped reading when it
  moved to appending instances through `get_backend`, so the test drove a real `docker run`
  and left an `ainode-vllm-node-solo` container in `Created` state on every machine with
  docker. The fake now goes in at `get_backend`, and a session-scoped guard in
  `tests/conftest.py` fails the run if any test creates an `ainode-vllm*` container, so the
  next one to forget says so instead of leaving evidence only on the developer's machine.
  That file also dropped from 26 seconds to under one. (#221)
- **The speed bench slugs its `--label` into the record filename.** It interpolated the
  label verbatim while the harness, agentic, decide and embed sections all ran theirs
  through `slug()`, so a label with spaces or a comma produced a filename no shell could
  name without quoting, and a label with a slash wrote into a directory that did not exist
  and lost the record. Naming moved into one `record_path()` helper reusing the same
  `slug()`. (#158)
- **The installer gives the real reason `/mnt/shared-models` must exist.** It cited "the
  per-node NCCL init shim", which was the retired eugr launcher's use of the path, not the
  default backend's. The requirement itself stands: the systemd unit bind-mounts the path
  with `--mount type=bind`, so the service fails at container start when it is missing, and
  the path is where downloaded weights are staged. The same message also printed literal
  `\n` characters instead of line breaks, because `die()` prints through `printf`'s `%s`.
🤖 Generated with [Claude Code](https://claude.com/claude-code)
- **Every launch and unload button does what it says.** A round of UI-truth fixes, all of them the same failure: a control that reported success for something it had not done.
  - **The Models view no longer offers a second, broken way to launch across the cluster.** "Shard Across Cluster" drew a sharding plan and its "Launch Sharded" button then posted the model id alone, so `min_nodes` stayed 1 and `/api/sharding/launch` fell straight through to a plain solo load on the local node, of a model the card only offered while it was NOT downloaded, and toasted "Sharded model launching" ([#188](https://github.com/getainode/ainode/issues/188)). The plan behind it was invented as well: `ShardingPlanner` sizes a model as params x 2 bytes from a Llama-3.x-era name table (an NVFP4 27B came out as 54 GB, a name it cannot parse as a flat 14 GB) and derived the "Layers 0-39" range as `size_gb / 2`, and none of the four CSS classes the preview emitted had a rule, so it rendered unstyled. Both are gone, along with the `GET /api/sharding/plan` route that served the preview and the `active_sharding` field that was declared, read and never assigned. A card for a model that needs more than one node now points at the right-panel LAUNCH INSTANCE, which posts the picked `node_ids` and genuinely launches TP = node count.
  - **The model-detail modal's "Launch Model" launches.** It posted `/api/engine/set-model`, which stops whatever `app["engine"]` is serving (including stacked instances it has no accounting of) and, with no engine object at all, saves the config and starts nothing while still answering `{"status": "restarting"}`. The UI toasted "engine restarting" for all three outcomes ([#189](https://github.com/getainode/ainode/issues/189)). It now goes through the same one call the right-hand LAUNCH panel uses (`POST /api/cluster/load`, InstanceManager-aware, stacks instead of replacing), on a node the modal lets you pick, and the toast says what the handler actually started. `/api/engine/set-model` is documented as the destructive boot-config route it is, and nothing in the UI calls it.
  - **UNLOAD stops one copy, not every copy in the fleet.** The button posted `{model}` with no node, and `/api/models/unload` fanned that out to EVERY peer serving the id whenever this node was not serving it, so pressing UNLOAD on one instance card took down the model everywhere ([#207](https://github.com/getainode/ainode/issues/207)). The card knows its node and its engine port (it renders the node pill from them) and now sends both through `/api/cluster/unload`. The port is what tells two stacked instances of the same model apart, a named port that matches nothing refuses instead of falling through to stop the node's primary, and the fleet-wide fan-out is opt-in behind `all: true`.
  - **The chat and bench instance pickers are honoured.** Both let you pick "model @ node:port" and then sent the model id alone, so the proxy re-routed on the id with the local hop first: with the same model on two nodes the answer, the per-turn stats and the "routing to node:port" line could all belong to a different engine, and a bench record, which is kept and compared, could be filed against the wrong node ([#197](https://github.com/getainode/ainode/issues/197)). A forwarded request can now pin its target with `X-AINode-Node` / `X-AINode-Port` headers or an `ainode_target` body field (stripped before forwarding), which `proxy_to_vllm` honours exactly: one candidate, no failover, and an unknown node id is a 404 rather than a quiet fall back to the local engine. Every forwarded response carries `X-AINode-Served-By: host:port`, which the chat records per turn instead of echoing the pick. The bench form sends the same pair in its run body, and its picker's options are instances rather than model ids, which is what let two copies collapse into one option in the first place. Documented in the README.
  - **One image attachment no longer breaks chat persistence for good.** `saveConversations()` had no try/catch and stored image attachments as data URLs up to 10 MB each, against a 5 to 10 MB per-origin quota, so the first `QuotaExceededError` escaped `sendMessage()` and nothing in that browser persisted again until the user deleted conversations by hand ([#202](https://github.com/getainode/ainode/issues/202)). The image bytes are no longer written to localStorage at all (the stored turn keeps a count, and the live session keeps the bytes so they are still re-sent), and the write is wrapped: a full or blocked store evicts the oldest conversation, retries, and says so in a toast rather than throwing.
  - **The four Config fields that saved and did nothing now drive the code.** `cors_origins` was the one with teeth: `cors_middleware` hardcoded localhost and never read the setting, so a user who allowed an origin had not. It is parsed now, with localhost still implicit and `*` allowed. `datasets_dir` and `training_dir` are read by the training engine through one resolver each (the `AINODE_HOME` subpaths stay the fallback), and the `DatasetManager` takes the same root. The New Run wizard seeds its method, epochs, batch size and learning rate from Training Defaults instead of hardcoding lora / 3 / 4 / 2e-4 ([#204](https://github.com/getainode/ainode/issues/204)).
  - **Changing the models directory no longer splits the manager from the engine.** `ModelManager` was built with no argument and used the module constant, while the engine backends mount `config.models_dir`, so after a change the model you downloaded was not the model the engine looked for, Installed listed models no launch could see, and Delete freed space in a directory nothing used ([#205](https://github.com/getainode/ainode/issues/205)). The manager takes `config.models_dir`.
  - **The Server view stops advertising what it cannot do.** The Stop toggle, the mcp.json export and the eye Preview were `toast('not yet implemented')` stubs and are gone until something implements them. The MODEL INFO Load and Inference tabs were disabled inputs hardcoded to 4096 / -1 / 0.7 / 0.95 / 40, none of which was this engine's `max_model_len` or a sampling default anything sends ("GPU layers" is a llama.cpp idea vLLM has no equivalent for); both tabs are gone and the one real number in them, the launch width off the instance record, is a row on the Info tab. `GET /api/v1/models` was listed in the LM Studio endpoints tab with no `planned` marker and no route registered anywhere, so copying that curl gave a 404; the row is gone. And `/api/server/status` reported the literal `"running"`, true by tautology, so the status dot stayed green with every engine dead: it now reports `running` / `loading` / `idle` off the instance records it already had ([#206](https://github.com/getainode/ainode/issues/206)).
  - **Dead UI code and unstyled markup.** `renderLiveCatalog` (a whole second catalog view over `/api/models/trending|openrouter|latest` that nothing could reach), `updateDownloadProgress`, `skeletonCards`, `gaugeColor` and `tempColor` are deleted, as are the CSS rules whose only markup went with them. `.quantize-panel`, emitted twice with no rule at all, gets one. The streaming rate is called "deltas/s" in both places it appears instead of "tok/s" in one of them. `/api/sharding/status` is no longer polled every three seconds for a field nothing read ([#210](https://github.com/getainode/ainode/issues/210)).
  - **`GET /api/models/downloaded` is registered before `/api/models/{model_id}`.** The dynamic route came first and the literal worked only because aiohttp 3.9+ resolves a plain resource before a dynamic one. Under registration-order resolution it would have answered `404 Model 'downloaded' not found in catalog`, and quietly: that route backs the Installed list, the LAUNCH INSTANCE dropdown and the on-disk cards, so the UI would have shown an empty model list with no error ([#209](https://github.com/getainode/ainode/issues/209)).

### Tests
- `tests/test_training_persistence.py` (new): the registry rebuild over job-dir fixtures with and without status files (including a legacy quantize job, a legacy merge job and an empty dir), the status file written at every transition with no token in it, the queue advancing from the monitor, the resume mount and path rewrite, and a resume of a job rebuilt from disk. `tests/test_training_command.py` gained the image resolution order and the preflight message; a test that pins an image in an argv patches `_image_present`, because resolution asks the docker daemon and a Spark running the suite really does have the hand-built train image.

---

## [0.5.27] - 2026-09-19

Fine-tuning produces finite weights (for real this time), a real `ainode doctor`, auth the dashboard can use.

### Corrected
- **0.5.26's changelog overstated its contents.** It said fine-tuning produced finite weights. That fix (#213) had failed to merge behind a changelog conflict, so 0.5.26 shipped without it; it lands here, in 0.5.27. The fresh-install fix (#212) was in 0.5.26 as stated.

### Added
- **A real `ainode doctor`.** It printed "coming in v0.5.0" on a 0.5.26 build and accepted `--peer`, `--json` and `--fix` only to drop them on the floor, so `ainode doctor --json` exited 0 with prose ([#177](https://github.com/getainode/ainode/issues/177)). It now runs the 21 checks a human ran by hand during the 2026-09-19 fleet audit, which is the point: nearly every gotcha that audit turned up was one command away from being visible. Config sanity (an `engine_backend` whose binary or image is actually on the node, `gpu_memory_utilization` in range and not so high the stacked-load guard refuses every second model, the discovery port against the one the installer writes, a `cluster_id` still `"default"` on a node that expects peers, a model pinned but not downloaded), docker reachable and the engine image pulled, GPUs enumerated with their count and whether the memory is unified, free space on the AINode home and the models dir with a warning under 15 percent, the pin in `image.env` against the running container and against the newest published tag, the systemd unit installed and active, ports 3000, 8000 and 5679 listening or free as expected, peers seen on discovery with each one's version and whether the fleet agrees on a release, the fabric interface and its address, the secrets store's mode, whether a Hugging Face token exists anywhere (presence only, never a value), and the sudo trap that moved `$HOME` under root and made `sudo ainode update` keep the old image. One line per check with OK / WARN / FAIL and a one-line fix, `--json` for machines, `--peer HOST` for the same report over SSH, and exit non-zero on any FAIL so it can gate a script. A WARN is deliberately not fatal: a doctor that fails on every warning is a doctor nobody runs.
- **`ainode doctor --fix` applies only what cannot lose anything**: create a missing directory, chmod the secrets store to 0600, write the fleet discovery port into `config.json` (one key, every other key untouched). Then it re-runs the checks so the report describes the node as it is now, and lists everything still left for a human rather than pulling an image or restarting a service on its own.
- **`/v1/responses`, `/tokenize`, `/detokenize`, `/v1/rerank` and `/v1/score` through the fleet endpoint.** The engines serve all five and port 3000 answered 404 for all five, so a caller who wanted a token count, a Responses call or a rerank had to abandon fleet routing and address one engine's port directly, losing the model-id lookup and the failover with it. They are registered on `proxy_to_vllm` like every other forwarded path, so they get routing on the body's `model`, ordering by capability, transport failover, the query string and the SSE passthrough for a streamed response with no code of their own. Two of them are not under `/v1` because vLLM does not serve them there. The list is what `curl http://<node>:<port>/openapi.json` returns on the fleet, checked against vLLM 0.27.1 (chat) and the 0.17.0 pooling engine stacked beside it on Spark-4, so nothing is registered that an engine would 404 anyway.
- **Disk figures in `/api/status` and the cluster graphic.** Free and total for the AINode home and the models dir, each with its own warning state under 15 percent, and the models figure on the node's tooltip. A full models filesystem is the one resource failure on these nodes that reads as "the engine died": the pull stops part way and vLLM never binds. The warning state is decided server-side so the tooltip and `ainode doctor` cannot disagree about what low means, and a path we cannot stat reports null rather than zero, because unknown and full are different answers.
- **Authentication the dashboard can use.** API key auth has shipped since 0.4 and nothing in the fleet used it, because turning it on broke the product: the web UI never sent an `Authorization` header anywhere, so `ainode auth enable` left the shell rendering and every panel behind it 401ing, and the only way back was `ainode auth disable` on the box ([#167](https://github.com/getainode/ainode/issues/167)). The UI now has one wrapper (`web/static/js/auth.js`) that every request in `app.js`, `bench.js` and the onboarding page goes through; it attaches the key when one is stored, keeps it in `localStorage` under `ainode.apiKey` (wrapped, because a browser with site data blocked throws on the read), and reports a 401 exactly once per poll cycle to the one handler that opens **Config > API access**. That panel is the whole story in one place: whether this port wants a key, the switch that turns it on, the key shown once when it is minted, the keys that exist with a Revoke on each, and a box to paste a key into on a fresh browser. The browser that flips the switch stores the new key itself, so the click that enables auth can no longer lock the operator out. The bench view keeps working because the two things a header cannot travel on, the report iframe and the JSON download links, are fetched through the wrapper and handed to the browser as a blob instead of an `href`.
- **The header says how the port is protected.** `GET /api/status` gained an `auth` block whose `label` is the sentence the dashboard chip shows, and the installer's summary line prints the same words: **API open, no key set**. The default is unchanged (open, which is what a private network wants and what the fleet runs), but a dashboard that never mentions it is a dashboard that lets you assume a password exists. The chip is one click from the panel that changes it, and it reads its state from `/api/auth/status`, the one API route that still answers without a key.

### Changed
- **`ainode auth enable` on a node that already has a key stops crashing.** `AuthConfig.enable()` returned the stored entry, the caller read `entry["key"]`, and only the hash is stored: a `KeyError` that 500'd `POST /api/auth/enable` and traced out of the CLI. It returns `{"id", "key"}` with `key` set only for a key it just minted, and both callers say so instead ("using the keys this node already has, stored hashed, so not shown again").

### Fixed
- **Fine-tuning produces finite weights.** AINode had completed exactly one LoRA run in its life (Spark-4, 2026-07-06, job `681b658ad647`, Qwen2.5-0.5B-Instruct, 60 samples) and that run's adapter was 100 percent NaN: 2,162,688 of 2,162,688 values, with NaN in every q/k/v/o projection of the model it was merged into. Its `trainer_state.json` says what happened and nobody read it: loss 12.61 at step 10, which is the uniform-random baseline for this vocabulary (ln 151936 = 11.93), grad_norm 3.3 million, NaN from step 20, and `COMPLETED` at the end. Two independent faults, both reproduced on Spark-3 against the current runner before anything was changed. First, the training image's memory-efficient attention kernels are built for sm80-sm100, so on a GB10 (sm121) they refuse to launch: torch prints `FATAL: kernel fmha_cutlassF/cutlassB..._sm80 is for sm80-sm100, but was built for sm121` a quarter of a million times per run, the forward then returns zeros and the backward returns NaN from step one. Training now loads the model with `attn_implementation="eager"` by default, a per-job config field for the operator who wants something else. Second, tokenization padded every sequence to `max_seq_length` and copied the padded `input_ids` into `labels`, so roughly 90 percent of the positions asked the model to predict the pad token: that is where the 12.61 loss and the six-orders-too-large gradient came from. Padding is dynamic per batch now (`DataCollatorForSeq2Seq`, `pad_to_multiple_of=8`), labels are `-100` wherever attention is masked, and the tokenizer reaches the Trainer as `processing_class`. Same experiment, same node, same 20 steps: eager and masked labels give loss 5.11 falling to 2.13 with grad_norm 8.2 at step 1, against the unchanged runner's 11.93 and NaN.
- **A run that goes non-finite FAILS instead of reporting success.** The old runner could not tell: HuggingFace's `logging_nan_inf_filter` silently replaces a NaN loss with the running mean, which is why the dead July run logged a tidy `loss: 0.0` for its last 20 steps. That filter is off now, and an `on_log` callback aborts the run with `AINODE_ERROR:NAN_LOSS` naming the metric and the step the moment loss or grad_norm is NaN or inf, so nothing is saved. After a successful save the runner reads its own `*.safetensors` back and fails with `AINODE_ERROR:NAN_WEIGHTS` if any tensor holds a non-finite value, because the artifact is the only thing that speaks for the run. The first 20 steps are logged one by one (then the configured cadence), so the loss curve the browser draws is real where it matters.
- **A job the engine cannot launch answers 400 and leaves the queue working.** `TrainingJob.start()` set the job RUNNING, and `start_next()` claimed the active slot, before the launch command was even built. A submit the engine refuses (multi-node DDP in container-spawn mode is the reproducible case) therefore returned a 500 to the browser and left a phantom RUNNING job with no process behind it, which blocked every later job until someone deleted it by hand. RUNNING, `start_time` and the active slot are now set only after `Popen` returns a process; a refusal marks the job failed and `POST /api/training/jobs` answers 400 with the reason.
- **A missing training image is named before the job runs.** Training, quantize and merge all spawn `ainode-quant:0.17.0-t5`, which CI does not publish, so on a node that never built it every job died with `docker` exit 125 in a log. The image is checked with `docker image inspect` before launch, and the error names the image, the `scripts/Dockerfile.quant` build and the `AINODE_TRAIN_IMAGE` / `AINODE_QUANT_IMAGE` overrides.
- **The Hugging Face token stops travelling through job files and job logs.** It was serialized into each job's `config.json` and `config.container.json` (mode 644, in a directory the job API reads) and spelled out as `-e HF_TOKEN=<token>` in the launch line appended to the job log that `GET /api/training/jobs/{id}/logs` serves. Training, quantize and merge now pass it through a `--env-file` written 0600 and deleted when the container exits, the token is stripped from the config written to disk, and every launch line is logged through one scrubber that masks secret env values. A job's status is served by `GET /api/training/jobs` with auth off by default, so the token is masked to `***` there too: the key stays, so a caller can still see that a job has one.
- **An unauthenticated port 3000 is no longer root-equivalent.** With auth enabled the middleware now requires the key on every path under `/api` and `/v1`, with three exceptions and a reason for each: `/api/health` (a probe has no key), `/api/auth/status` (so the UI can say a key is wanted instead of rendering blank), and the static shell. First-run onboarding stays open only while the node is NOT onboarded: `POST /api/onboarding/complete` writes the node's identity, so on a configured node it was a mutating route with no key on it. Every mutating route named in [#168](https://github.com/getainode/ainode/issues/168) is covered by that one rule rather than by a list that can drift, and `tests/test_auth_gate.py` walks the real route table and fails on any route reachable without a key that is not one of the three.
- **`trust_remote_code` can no longer be set by an unauthenticated caller.** It was in `PATCHABLE_CONFIG_FIELDS`, so `PATCH /api/config {"trust_remote_code": true}` followed by a load of an attacker-named repo executed that repository's `modeling_*.py` as root inside the engine container, which mounts the host HF cache and the host SSH directory. Setting it now requires a request that presents an API key (which is true whether or not auth is enabled: a node running open can still have a key, and presenting it is what separates the operator from anyone who can reach the port), and clearing it is always allowed so a client can put the node back. Per-load it has one other route in: a curated catalog entry whose recipe already declares it, which is how the models that genuinely need it stay one click. Everything else is a 400 that prints the rule.

### Removed
- **Two training templates that promised runs the product cannot do.** `dpo-preference` offered chosen/rejected preference pairs, but there is no DPO trainer here: the runner would have space-joined prompt, chosen and rejected into one supervised string and trained the model ON the rejected answer. `distributed-ddp` offered multi-node data parallelism, which has no launch path at all (the container-spawn build refuses it and the host `torchrun` path has no rendezvous), so the tile only ever produced a failed run. Both are gone from the template list, along with the Distributed Training quick-start tile, the wizard's multi-node toggle and node count, and the "Distributed Training" guide page. Either comes back together with an implementation and a proven run.

### Tests
- The training fixtures no longer write into the developer's real `~/.ainode`: every `TrainingJob` mkdirs its job dir on construction, and 6,000 empty directories had accumulated under `~/.ainode/training/jobs`. An autouse fixture redirects `AINODE_HOME` and `JOBS_DIR` at a tmp path.
- First tests that execute the training runner: `tests/test_training_runner.py` pins the tokenization and NaN contracts with no torch needed, and `tests/test_training_gpu.py` (marked `gpu`, skipped without CUDA) runs a real LoRA job and asserts a finite loss at every step and zero non-finite values in the adapter. Both pass in the train image on a GB10.

---

## [0.5.26] - 2026-09-19

A fresh install can load a model; fine-tuning produces finite weights.

A fresh install can load a model.

### Fixed
- **A fresh install can load a model.** `curl -fsSL https://ainode.dev/install | bash` produced a node that could not serve anything, and had since 0.5.0. `NodeConfig.engine_backend` defaulted to `"eugr"`, a backend that `Popen`s a `vllm` binary, and the shipped image is `python:3.12-slim` plus this package: no vLLM, no ray, no eugr launcher. Nothing in the code or the installer ever wrote `"nvidia"`, so the first click on Launch answered `500 Launch failed: AINode runs as a container image, and this host has no vLLM install`, and the only way past it was to hand-edit `~/.ainode/config.json`. Every node in the fleet had been edited by hand at some point; both C4130s hit it again on 2026-09-19 ([#164](https://github.com/getainode/ainode/issues/164)). The default is `"nvidia"` now, which is the backend that runs the engine as its own container and is therefore the only one a node installed the documented way can use, and it lives in one place (`core.config.DEFAULT_ENGINE_BACKEND`) rather than being re-guessed as `or "eugr"` at three call sites that could drift apart. The eugr backend is unchanged and still selected by a config that names it; its no-vLLM error now also prints the path of the config file it wants edited, because on a container install that file is on the host and the message used to guess at the path. The installer writes `"engine_backend": "nvidia"` into config.json explicitly as well, so the node says what it is doing instead of inheriting it.
- **The installer no longer writes a config that refuses every stacked load.** It wrote `gpu_memory_utilization: 0.9`; the stacked-load admission guard refuses a load whose total would pass 0.90. On a brand-new node the arithmetic was therefore `0.9 + anything > 0.9` and the second model was always a 409, so the one thing you cannot discover from the error message was that the PRIMARY's number was the problem. It writes 0.6 now: about 73 GB of KV cache for a single model on a 122 GB GB10, with 0.30 left for one stacked neighbour.
- **The installer pre-pulls the engine image the engine actually runs.** It spent 5 to 10 minutes and ~15 GB pulling `nvcr.io/nvidia/vllm:26.02-py3`, behind an NGC login it also walked the user through, for an image no code path has ever launched. The engine image is `NVIDIA_VLLM_IMAGE` in `engine/backends/nvidia.py` (`scitrera/dgx-spark-vllm:0.17.0-t5`, on Docker Hub, no login), and the user paid for that one on their first Launch click instead. The installer now reads the value out of the AINode image it just pulled, so there is no second copy of the tag to drift, pre-pulls that, and says what it costs (~8.5 GB to download, ~22 GB on disk). `AINODE_NVIDIA_IMAGE=skip` still skips it and a tag still overrides it.
- **A fresh node no longer boots into a launch of a model nobody picked.** The installer's config.json left `model` out, and an absent key inherits `NodeConfig`'s default of `meta-llama/Llama-3.2-3B-Instruct`, so every new node started an engine for a gated Llama repo at boot: a 401 the user cannot place where there is no HF access, and 6 GB of download plus 0.6 of the GPU reserved where there is. Found on Spark-3 while proving this release. It writes `"model": null` now, which is the state the CLI already handles and the installer's own comment already claimed.
- **`ainode logs` tails the log the configured backend writes.** The path was hardcoded to `~/.ainode/logs/vllm.log`, which is the eugr backend's file; the nvidia backend writes `nvidia-vllm.log`, or `nvidia-distributed.log` on a distributed head. On every node in the fleet that made `ainode logs -f` a command that tailed a file last written in June, which also made every `ainode logs -f | grep ...` step in the docs read nothing. It resolves the path from the backend now and prints which file, and which backend, it is following.
- **`sudo ainode update` no longer keeps the old image and calls it a success.** The host wrapper resolved `AINODE_HOME` from `$HOME`, which under sudo is root's, so the pull succeeded, the new tag was pinned in `/root/.ainode/image.env`, and systemd restarted the unit, which reads the install user's `image.env` (the path is baked into the unit at install time) and relaunched the OLD image. Nothing said so, and this bit the fleet repeatedly. The wrapper now resolves the `AINODE_HOME` the UNIT reads: an explicit `AINODE_HOME` first, then the `Environment=AINODE_HOME=` line in the unit file, then `$SUDO_USER`'s home. It prints the `image.env` it wrote, and where it cannot tell it refuses with the command to say which one rather than writing a file nothing reads.

### Added
- **`scripts/install.sh --dry-run`.** Renders config.json, the systemd unit and the host wrapper into `$AINODE_HOME` and stops: no pulls, no systemd, no sudo, no docker and no GPU needed. It exists so the installer can be tested rather than transcribed into a test (`tests/test_fresh_install.py` runs the real script and asserts on what it wrote), and it answers "what would this do to my machine" without doing it.

---

## [0.5.25] - 2026-09-19

Embeddings served by the fleet; a stacked load can no longer clobber a distributed head.

### Added
- **Embeddings served by the fleet.** `POST /v1/embeddings` has been in the API since the Server view shipped, and on our hardware it has never worked: it ran `sentence-transformers` in-process on the CPU of the AINode container from a static catalog of its own, the library is not in that image, so every request came back `dependency_missing` and nothing in the fleet served embeddings at all. The fix is not a second runtime. An embedding model is an ordinary stacked vLLM engine with `--runner pooling`, on the same image as every chat model, so the fleet already knew how to launch it and the model id is already enough to say where the vectors are. `/v1/embeddings` now asks the fleet first, through the same `_routing_candidates` and the same failover loop `proxy_to_vllm` uses (local hop first, then peers, a stacked instance found on its own engine port), forwards the caller's body verbatim, and hands back the engine's status and content type untouched. A candidate that will not connect fails over to the next; every candidate dead is a 502 rather than a quiet fall back to a different model's vectors under the id the caller asked for; and a model no node serves still reaches the in-process manager exactly as before. The master and a worker behave identically, because the candidate list is built from cluster state and every node has it.
- **Catalog entry: Qwen3 Embedding 0.6B, verified.** 1024 dimensions, 32k context, multilingual, `--runner pooling --max-num-seqs 64 --enable-prefix-caching` at gmu 0.06 with an 8192-token window, on the node's own engine image. At 6 percent of a GB10 it is a model you leave running: it stacked beside Nemotron 3.5 Lightning on Spark-4 and came up in 87 s. `capabilities` gains `embedding`, which is the first capability in the catalog that is not a chat capability, so the model card draws an Embedding chip instead of a "Use in New Chat" button, `/api/server/status` reports the instance as `type: "embed"` with `capabilities: ["embeddings"]` (derived from the catalog, because vLLM's `/v1/models` says nothing about which runner is behind an id), and the chat model picker's existing `type !== 'embed'` filter therefore stops offering a model that would 400 every message.
- **`ainode-bench embed`: the fifth bench section, for a model that writes no tokens.** None of the other four says anything about an embedding model, because throughput, harness pass rates, an agent rubric and typed decisions are all about generation. This one measures the four things that decide whether one is usable: the vector width read off the response, single-request p50/p95 over 50 short texts in six languages, texts and tokens per second at batches of 1, 16 and 64 over the same 64 texts each time, and a quality sanity check of six hand-written pairs asking only whether every related pair beats every unrelated one. The latency block also carries a measured `transport_floor_ms` (the median of five `GET /v1/models` calls over the same link), because a p50 of 71 ms means two different things depending on whether 32 ms of it was the wire, and nothing in a record should need the reader to guess which. First record, Qwen3 Embedding 0.6B on Spark-4 stacked beside Nemotron: 1024 dims, p50 71.6 ms against a 32.1 ms floor, 13.9 texts/s at batch 1 rising to 225.5 at batch 64 (2938 tokens/s), pairs ordered with the weakest related pair at 0.82 against the closest unrelated at 0.32. README gains an "Embedding runs" table under the same drift guard as the other four, and the `embed` block is documented in `bench/SCHEMA.md`.

### Fixed
- **A second model loaded on a node that HEADS a distributed launch is a stacked load, not a new primary.** Found launching the embedding model on Spark-2 while it headed DeepSeek V4 Flash at TP=2 with Spark-3 as its peer. A distributed launch is deliberately never written to `instances.json` (it needs peer coordination and is not auto-replayed), so after a `systemctl restart` the head keeps serving on `:8000` while the InstanceManager starts empty, and `is_primary` was `len(others) == 0` over that empty manager. The next `POST /api/models/load` therefore called itself the primary: it answered `{"api_port": 8001, "stacked": false}`, which is the contradiction in one line, since `allocate_port` had already skipped the busy 8000. It then overwrote `config.model` with the new model, flipped `distributed_mode` back to `"solo"` and dropped `peer_ips`, so the node stopped advertising the model its head container was still serving and the master's federated `/v1/models` and its routing lost a live distributed engine while requests sent straight at that head kept working. The question is now asked about the port instead, which is the definition of "stacked" the node's own `/api/models` listing already used (`port != node_port`): a load is the primary only when the primary port is free for it to take, and a reload of the instance holding that port is still a reload of the primary. The stacked-load admission gate consequently applies to this case too, which it did not before, so a second engine can no longer inherit the node default 0.5 beside a TP=2 head and push a GB10 past its unified memory.
- **The head of a distributed launch stays in the node's announcement when a model is stacked beside it** ([#162](https://github.com/getainode/ainode/issues/162)). The announcement's `instances` list was either the InstanceManager's live records or a head synthesised from config, never both, so the first stacked model loaded on a head node made the head instance vanish from the broadcast: its peer was drawn as an empty node, the head itself as SINGLE rather than DISTRIBUTED TP=2, and the master's fleet view lost an engine it could otherwise have routed to. The two sources are merged now instead of chosen between, keyed on the port a node can only have one engine on, with a manager record winning where both describe the same port because that one carries the launch's real peer set and executor rather than the shape reconstructed from config. A head whose engine does not answer is still left out, same liveness rule as everything else on the wire, and a worst case of a head plus two stacked models is 1634 bytes against the announcement's 4096-byte ceiling.

---

## [0.5.24] - 2026-09-19

A V100 serves through the catalog.

### Added
- **Qwen3.6 35B-A3B on a V100, through the catalog.** The first model launched through AINode's own launch path on Volta hardware: `nvidia/Qwen3.6-35B-A3B-NVFP4` (Qwen3.5-MoE, 35B total, 3B active per token, modelopt NVFP4) on pollux, a Dell C4130 with one Tesla V100 32 GB. It is a catalog entry (`qwen3.6-35b-a3b-nvfp4-v100`) and not a note in a runbook because the recipe is the hard part and none of it is guessable: mainline vLLM dropped SM70 in 0.20, so the engine image is `onecat-vllm:src-full`, a local build of the 1Cat-vLLM fork with Volta kernels (Castor holds the exported tarball and the build script; the build itself runs about a day), and it serves only with `--attention-backend FLASH_ATTN_V100`, `--max-num-seqs 8`, gmu 0.90, a 65536-token window and the vision tower switched off, because the tower's warmup does not fit in 32 GB beside 23.5 GB of weights. Carrying all of that in the catalog is what makes it one click on any V100 node instead of a hand-rolled container, which is the single-V100 chat lane the Titanium Lab plan asks for. Measured on pollux 2026-09-19: ready in 432 s, 30.4 of 32 GB used, 97.4 tok/s single-stream, 343.9 tok/s across 16 streams, 3428 tok/s of prefill at 4K prompt tokens and decode still 66.2 tok/s at 63K. It scores 20/22 on the quick agentic rubric, with all four tool probes, all three executed-code probes and all five agentic probes passed. Records: `bench/results/20260919-040456-qwen3_6-35b-a3b-nvfp4-pollux-v100-solo-onecat-src-full.json` and `bench/results/20260919-040711-qwen3_6-35b-a3b-nvfp4-pollux-v100-solo-quick-agentic.json`. The 120K prefill depth is absent from the speed record rather than zero: it is past the window this recipe serves, and the engine answered 400.

---

## [0.5.23] - 2026-09-18

Typed decisions from the fleet, and a bench that scores them on calibration.

### Added
- **`POST /v1/decide`: typed questions in, probabilities out.** Post a state (text or JSON), an optional instruction, and a map of questions (a list of options, a boolean, or a score range) and get every answer back at once with a probability per option, a confidence, and per-question latency. Each question is one constrained chat completion (vLLM's `structured_outputs` choice grammar, thinking off, temperature 0) run concurrently against the same state prefix, with the distribution read from the first token's logprobs. Routes by model across the fleet like every other `/v1` path. This is the local counterpart of the hosted "System One" interface TypeSafe AI introduced with Jev: the shape is worth having on private hardware even when the calibration is not theirs yet. (#150)
- **`ainode-bench decide`: a decision bench scored on calibration, not accuracy.** 110 labeled items in five sets (fleet routing, support triage, urgency, PR safety, facts), three backends (`ainode` via `/v1/decide`, `chat` via lettered choices and logprobs on any OpenAI-compatible engine, `jev` via TypeSafe's hosted API), and metrics that matter for automation: accuracy, Brier, expected calibration error with a reliability table, and how many wrong answers survive a 0.8 or 0.9 confidence gate. First records: Jev and Ornith both score 96.4 percent on the same items; every Jev miss sits below 0.66 confidence and none survive a 0.9 gate, while two of Ornith's misses sail through at 0.96 and 0.98. README gains a "Decision runs" table under the same drift guard. (#151)

---

## [0.5.22] - 2026-09-18

A loading model looks like it is loading; an agentic rubric joins the bench.

### Added
- **The interface shows a loading model as loading, with elapsed and expected time.** While an instance comes up, the instance chip, the catalog card and the launch panel show the phase, the elapsed time and the expected time ("loading weights, 3:12 of about 12 min"), a progress bar, and "taking longer than usual" once a load runs past one and a half times its expectation. The expectation comes from the launch-time ledger first (this model on this node) and the catalog's `typical_ready_minutes` second. `/api/status` and every `/api/nodes` row now carry `load_started_at`, `load_elapsed_seconds` and `expected_ready_minutes`; the discovery announcement carries them too, so a model loading on a peer draws the same bar on the master. The announcement has a stated byte ceiling now (`MAX_ANNOUNCEMENT_BYTES`), checked by a test, because the listener reads one datagram and an oversized payload used to make a node vanish silently. (#146)
- **`ainode-bench agentic`: a fresh-agent rubric with mechanical verdicts.** Twenty-four probes in eight groups: instruction precision, tool calling (single, parallel, not needed, round trip), executed coding, reasoning traps, needle in 8k/48k/100k prompts, thinking off, vision, and a new agentic group: a list-then-read tool loop with a decoy file, tool error recovery that fails on a fabricated answer, argument schema fidelity, structured output, and a system rule held over four turns. Records carry an `agentic` block (documented in `bench/SCHEMA.md`) and the README gains an "Agentic rubric runs" table with the same drift guard as the other two. First record: DeepSeek V4 Flash, 20/22, all five agentic probes passed. (#147)

---

## [0.5.21] - 2026-09-17

Load times in the interface; verification provenance in the catalog; Flash-Next verified.

### Added
- **The interface says how long a model takes to load.** Until now the only trace of a load time was one log line (`<label> bound on :<port> after <N>s`), so nothing could tell a user that a 27B NVFP4 on a GB10 takes about 12 minutes, and a slow launch looked like a broken one. Every bind wait now appends its verdict to `<AINODE_HOME>/launch-times.json` (model, node, port, stacked or not, TP, engine image, seconds to ready and a stamp), and failures go in too, with the reason, because a ledger of successes only would describe a model that never comes up as one that comes up fast. It keeps the last 200 entries and is written off the event loop, like every other file write on that path. Catalog entries also carry a `typical_ready_minutes` seed from the launches measured this week (Qwen3.8 27B solo on a GB10 about 12 min, Ornith 1.5 stacked about 12 min, Nemotron 3.5 Lightning on the GX10 about 10 min, DeepSeek V4 Flash at TP=2 about 7 min, Qwen3.8-Flash-Next at TP=2 with autotune off about 11 min), left unset on an entry nobody has launched rather than estimated from the weight size. `GET /api/models` and `GET /api/models/card` report both numbers and prefer the ledger, so the launch panel shows a quiet "Typical load: about 12 min on Spark-1, measured 2026-09-16" beside the model choice and the model card gains a "Loaded in" row, and both say nothing at all when neither number exists.
- **A `verified=True` catalog entry now says when it was proven and which bench record proves it.** The flag was a claim a reader could not check: the entries proved this week had records nobody could find from the entry, and a set carried it from earlier eras with no record at all. `ModelInfo` gains `verified_on` (ISO date) and `verified_record` (a filename under `bench/results/`), filled in for the six entries that have records, and the entries that predate the bench keep both empty with a comment saying so instead of being quietly demoted. The model card's "catalog verified" chip is unchanged to look at but now carries a title: "Tested on AINode 2026-08-15, see bench record" where there is one, "Marked verified before the bench existed" where there is not. A test fails on a `verified_record` naming a file that is not there, and on a curated entry that has a bench record but is still marked unverified.

### Changed
- **Qwen3.8-Flash-Next is marked verified and recommended.** With #136 it launches from the catalog on the Spark pair in 11 minutes, repeatably; speed and harness records are in the set. Autotune off costs about 6 percent of single-stream decode (24.9 vs 26.6 tok/s).

---

## [0.5.20] - 2026-09-17

Engine activity as the liveness signal, ranks kept in step through autotune, OpenCode in the harness bench, catalog shapes in every record.

### Added
- **`ainode-bench harness --claude-effort <level>` sets the reasoning effort Claude Code runs with** ([#127](https://github.com/getainode/ainode/issues/127)). Claude Code sends effort "high" by default, and a served chat template does not have to accept that value: Qwen3.8-Flash-Next takes only xhigh, medium and low, so every request came back `API Error: 400 Unexpected reasoning effort high`, the harness crashed in about 0.3 s per attempt and the model scored 0/10 on a suite the other three harnesses were passing. The same ten tasks then scored 8/10 and 10/10 at medium. The level is appended to the claude argv as `--effort <level>` and nothing else about the invocation moves; unset stays the default and sends nothing, so every number already recorded (Ornith, Qwen3.8 27B) keeps its meaning. It is written down where a reader will find it: the run's `settings` as `claude_effort` and the claude block's `options` as `effort`, both absent when the flag was not used, because a run with the agent's own default is a different statement from a null. Docs: the `claude` section of `bench/harness/README.md`.

### Fixed
- **The harness bench record's `placement.node` now names the node that serves the model, not the master the CLI talked to.** `ainode-bench harness` built the placement block by calling `describe_via_http` with the `--ainode` web base as both the control-plane and engine URL, so when DeepSeek V4 Flash was served on Spark-2 while the CLI pointed at Spark-1, the record's `placement.node` read "Spark-1". The CLI now resolves the serving node through the master's `/api/server/status` `loaded_models` (the same fleet view the speed bench and the in-product bench use), calls `describe_via_http` against that node's own web base, and records the engine port in placement. A model served on the master is unchanged; an unloaded model produces the same "does not report serving" warning.
- **The harness bench's OpenCode adapter now runs in the task's directory instead of the one the bench was started from, which is the whole of [#118](https://github.com/getainode/ainode/issues/118).** Every `--harness opencode` invocation from the runner exited 1 in about 1 s with `{"type":"error",...,"message":"Unexpected server error. Check server logs for details."}` on stdout and an empty log, while the identical command run by hand passed, so the adapter was carried as unverified and the DeepSeek harness record's OpenCode rows are marked not measured. `--print-logs --log-level DEBUG` off a runner-shaped invocation named it in one run: opencode creates its instance in the working directory, reads the run's project-local `opencode.json` there, and then creates a *second* instance and builds the session against the directory named by `PWD` - which a subprocess started with `cwd=` still inherits from its parent, so it was the shell the bench was launched in. That instance has no `ainode-bench` provider, so the session died with `ProviderModelNotFoundError: Model not found: ainode-bench/<model id>`, reported on stdout as the generic server error. By hand it passed because a person `cd`s in first and `PWD` is then correct; that asymmetry, not flakiness, is why it looked intermittent, and it is why the other four harnesses were never affected (none of them reads `PWD` to find its project). Two fixes, one cause: the argv now states `--dir <workdir>`, and `_launch` sets `PWD` to the launch cwd and drops a stale `OLDPWD` for **every** harness, since the wrong value also reaches whatever shell an agent spawns for its own tools. Verified by the runner itself, `--only-tasks isogram` against Ornith 1.5 35B A3B NVFP4 on the fleet: three consecutive runs, pass@1 1/1 at 25.5 s, 26.7 s and 26.5 s, zero crashes, where the same command on the previous commit crashed in 0.9 s twice out of two.
- **An OpenCode bench run no longer reads the operator's global config, and a failed one now says what failed.** `OPENCODE_CONFIG_DIR` overrides the XDG config search outright, so the four isolated `XDG_*` variables the adapter already set were bypassed by any exported value (Orca exports one) and somebody's global config, plugins and agents were inside the measurement; the overlay now points that variable at the run's own empty config directory too, the same rule `DSH_HOME` and `CLAUDE_CONFIG_DIR` follow. The argv also carries `--print-logs --log-level ERROR`, which is diagnostics rather than behaviour - the logs go to stderr, stdout stays the NDJSON event stream the adapter parses - so a future failure reaches the record's `stderr_tail` with a named cause instead of the opaque "Unexpected server error" that #118 had to work from; verified against the old broken shape, where it prints the `ProviderModelNotFoundError` in 1388 bytes.
- **The bind wait watches what the engine is DOING; log silence is only the second opinion** ([#112](https://github.com/getainode/ainode/issues/112)). vLLM prints nothing at all while it loads weight shards, runs torch.compile and autotunes FlashInfer, so "no log line for N seconds" could never tell a wedged engine from a working one: three healthy starts in two days were declared silent (Qwen3.8 27B NVFP4 quiet for 206 s between shard updates, Nemotron 3.5 Lightning for 363 s inside autotune and graph capture, then a 48-minute autotune), and the window had to grow 120 to 360 to 900 s to keep up. Since 0.5.13 that verdict broke nothing (the relaunch is a no-op while the container is alive) but it spent the engine's single retry and wrote a wrong reason into the log. Backends now publish `activity_mark()`: for `NvidiaBackend` the last time THAT engine's container was seen burning CPU, read from the container's cgroup `cpu.stat` when the tree is visible and otherwise from `docker stats --no-stream`, cached so one bind poll costs one probe and run off the event loop thread. A log line OR an advancing activity mark resets the silence clock, and the start is called dead only when both have been quiet for the whole budget, which is why the budget comes back down to 300 s (`engine_bind_log_silence_seconds`): a truly wedged engine does no work at all and is caught in five minutes again instead of fifteen. The hard death signal (container exited) and the 1800 s ceiling are unchanged, and a backend that cannot see its engine's container reports `None` and keeps the old log-driven behaviour. GPU utilization is deliberately not part of this: the pynvml path reads device 0 as a whole, so it cannot say which stacked engine is busy, and it reads 0% during exactly the phases that need covering.
- **`engine_ready` in `/api/nodes` means the engine answers `/v1/models`, not that the node is reachable** (the second half of #112). The field was derived from `status`, which is heartbeat health (online/stale/offline) and true of every node whose orchestrator is up, because `ClusterNode.from_discovered` replaces the announced engine state with that health: Spark-4 was published `engine_ready: true` while its engine was still loading and its `loaded_models` was empty. `ClusterNode` now carries the node's own word on its engine (`engine_status`, straight from the announcement, which that node's sync loop already sets from a live `/v1/models` probe), the local row answers from a fresh localhost probe rather than the `ready` latch, and a member node (which runs no local vLLM) still reads ready once discovered. `status` is untouched, so the topology's online count keeps working.
- **The two ranks of an `mp` launch stay in step through FlashInfer autotune, instead of gloo killing the pair 48 minutes in** ([#134](https://github.com/getainode/ainode/issues/134)). Kernel warmup runs per rank. On Qwen3.8-Flash-Next the head, warm from an earlier launch, finished autotune in tens of minutes while the peer tuned 856 profiles from cold, sat 1800 s on its first `trtllm::fused_moe::gemm1` profile, and the barrier after autotune died with `Connection closed by peer` on one side and `Application timeout caused pair closure` on the other. 1800 s is not a kernel hang: it is PyTorch's default timeout for the CPU (gloo) group, which is the group the barrier and the autotune sync both use. Three things change. The head now ships its compiled-kernel cache to a peer before that peer's container starts (`_ensure_peer_has_jit_cache`): one transfer per child of the recipe's `VLLM_CACHE_ROOT`, skipped per child when the peer already has it, and a transfer that fails is a warning rather than a refused launch, since a missing cache costs minutes and a refused launch costs the model. The Flash-Next entry turns autotune off with `--no-enable-flashinfer-autotune` (the off form of `KernelConfig.enable_flashinfer_autotune`, which `kernel_warmup.py` reads to skip the whole phase) and raises both collective floors to 5400 s, above the worst warmup measured on the pair; the stated cost is that its fused MoE and fp4 GEMMs now run FlashInfer's heuristic tactic rather than a measured one, so decode is slower than the 26.6 tok/s of the launch that got through. It also states the cache roots it was missing, so its JIT output persists per node inside the mounted HF cache instead of dying with the container and re-tuning from cold every launch. The bind wait's ceiling goes from 1800 s to 3600 s, because a launch that took 36 minutes to ready was being killed by the ceiling as well; the ceiling is the last resort and the activity signal from #112 still calls a genuinely wedged engine dead inside five minutes, so nothing waits an hour on a dead start. Worth writing down for the next person: seeding a peer's autotune file cannot help on its own. Only rank 0 reads that file and broadcasts its bytes to the other ranks, and the path holds a sha256 of the whole `VllmConfig` hash, so a hand-copied directory is both unread and easy to orphan. DeepSeek V4 Flash keeps its own autotune and timeouts exactly as proven.
- **Catalog entries state their active parameter count and architecture, so a bench record stops reporting a MoE as dense.** `ModelInfo` had no field for either, and the record's `model` block was filled by reading the `A<n>B` marker off the model id, which works for `Ornith-1.5-35B-A3B` and not at all for an id that does not carry one: DeepSeek V4 Flash (284B total, 13B active) and Qwen3.8-Flash-Next (125B, 6B) were both written down as `params_b 125/284, active_b 125/284, arch dense`, which on GB10 is the one number that predicts decode speed, said wrong. `ModelInfo` now carries `active_params_b` and `arch` (`"moe"` / `"dense"`), every curated and fallback entry states its shape, and the model block takes both from the catalog when it has them and falls back to the id only for a model the catalog does not describe. An entry that says MoE without an active count (GLM-5.2 REAP, where pruning moves it) leaves `active_b` out and both renderers print `504B MoE` rather than claiming dense. The catalog cache round-trips the new fields and a cache file written before they existed still loads. Applies to the throughput bench, the in-product `/api/bench` runs and the harness bench alike: all three build the block through `ainode/bench/fleet.py`.

---

## [0.5.19] - 2026-09-16

The fleet endpoint forwards the Anthropic Messages API; Claude Code is a harness bench adapter; Qwen3.8-Flash-Next entry pinned and proven.

### Fixed
- **The fleet endpoint forwards the Anthropic Messages API.** vLLM serves `POST /v1/messages` natively alongside its OpenAI paths, but AINode's port 3000 registered only `/v1/chat/completions` and `/v1/completions`, so `/v1/messages` was a 404 on every node: a client that speaks Messages and nothing else (Claude Code, the Anthropic SDKs) had to be aimed at one engine's `:8000` and lost the model-id lookup, capability-aware ordering and failover that every other protocol gets for free. `/v1/messages` and `/v1/messages/count_tokens` now go through `proxy_to_vllm`, the same handler as chat completions, so routing on the body's `model`, transport failover across every node serving it, the 404 for a model nobody serves, SSE streaming and header passthrough (`x-api-key`, `anthropic-version`, `anthropic-beta` reach the engine untouched) are the code that already existed rather than a second implementation. `is_multimodal_request` now also reads the Messages API's own spelling of media (an `{"type": "image"}` or `{"type": "document"}` block, including one nested in a `tool_result`), so an image sent over `/v1/messages` is ordered onto an instance that accepts images instead of being routed on the model id alone. The proxy also stops dropping the caller's query string (`path_qs`, not `path`): Claude Code posts to `/v1/messages?beta=true`. The Server view's endpoint list no longer calls `/v1/messages` planned.

### Added
- **Harness bench adapter: `claude`, Claude Code driven against a served model.** The first harness here that does not speak the OpenAI protocol, and the reason the proxy fix above exists. `--harness claude` runs `claude -p "<prompt>" --model <id> --dangerously-skip-permissions --output-format json --max-turns 12` with `ANTHROPIC_BASE_URL` set to `--endpoint` minus its trailing `/v1` (Claude Code appends `/v1/messages` itself, so the `/v1` form would produce `/v1/v1/messages`); `--endpoint` stays the `/v1` form on the command line for every harness. The placeholder key goes in both `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN`, the served model fills both the main and small-fast model slots, and telemetry and non-essential traffic are off. `CLAUDE_CONFIG_DIR` points at the run's own scratch directory so a bench run never reads or writes the operator's Claude Code profile (settings, hooks, MCP servers, sessions, credentials), which is both hygiene and honesty, since a personal `settings.json` changes what the agent does; it needs no seed file (verified: claude 2.1.272 creates the whole tree itself from a directory that never existed and never prompts for a login). `is_error: true` in the result JSON is recorded as a crash even when the exit code is 0, because Claude Code reports a run it could not finish and still exits cleanly. Verified end to end on claude 2.1.272: 6/6 hidden tests on `isogram` driving Qwen3.8 27B NVFP4 on the fleet, 294 s over 6 turns. Known gap, documented rather than fixed: a `CLAUDE.md` in a parent of the working directory is still discovered, so a `--work-dir` inside a checkout that has one puts that file in the run.

---

## [0.5.18] - 2026-09-16

The distributed launch serves a model downloaded through AINode; harness bench; Qwen3.8-Flash-Next entry.

### Fixed
- **A distributed launch now serves a model that was downloaded through AINode instead of making every node re-download it.** `POST /api/models/download-repo` writes a flat `<models_dir>/<owner--name>` directory, not the Hugging Face cache layout, and the solo launch has served that directory for releases. The distributed shapes did not: both served the repo id, the head and peer containers mounted only the HF cache, and `_ensure_peer_has_model` distributed only a `hub/models--<owner>--<name>` entry, so a 133 GB model already sitting on the head was pulled again on every rank, over the WAN, for every launch. Both shapes now go through one resolver: when that flat directory exists on the head and the mount is trustworthy (the same `AINODE_HOST_HOME` test the solo path uses), every rank serves `/ainode-models/<owner--name>` with `--served-model-name <repo id>` so `/v1/models` is unchanged, and every rank mounts its own copy of the store there: the head's `models_dir`, a peer's `/home/<ssh_user>/ainode-nvidia-models` (chosen like the peer HF cache, because we ssh in as that user). `_ensure_peer_has_model` ships the flat directory over the fabric (rsync when present, else tar over ssh, skipped when the peer already has it) instead of the hub entry, and the peer's `mkdir -p` covers the new path so docker can never invent a root-owned empty bind source. With no such local copy, nothing changes: the repo id is served and only the HF cache is mounted. A recipe that states `--served-model-name` itself still wins, in solo too.

### Added
- **Catalog entry: Qwen3.8-Flash-Next (NVFP4), `nvidia/Qwen3.8-Flash-Next-NVFP4`, as a two-node `mp` launch.** Frontier MoE, 125B total and 6B active per token, plus a 51B PLE n-gram embedding and a 4B MTP module; 133 GB of mixed-precision weights (NVFP4 routed experts, FP8 elsewhere, modelopt), so it does not fit one 121 GB GB10 node. The entry pins `vllm/vllm-openai:v0.29.0`: the architecture and the FP8-PLE loader for mixed ModelOpt checkpoints landed in 0.29, and the fleet's 0.27.1 default and the local 0.28.0 images do not know it. Carries the card's recipe minus what the backend emits itself (`--quantization modelopt`, `--enable-prefix-caching`, `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`, `--enable-auto-tool-choice`), fp8 KV cache, 262K context, `--trust-remote-code`, gmu 0.85. MTP speculative decoding is deliberately not enabled: it wants `--enable-expert-parallel`, which hangs on this MoE and hardware, so it is a follow-up. `verified=false` and `recommended=false` until it is served end to end on the pair. It is the strongest coding model in the Qwen3.8 line, ahead of the 27B and DeepSeek V4 Flash on three of the four coding rows in Qwen's own table.
- **A harness bench: `scripts/ainode-bench.py harness` measures whether a served model can drive a coding agent to passing tests**, as opposed to how fast it generates. Ten Python practice exercises vendored from [exercism/python](https://github.com/exercism/python) (MIT, the same source Aider's polyglot benchmark draws on) are handed to a real agent CLI, and the score is what pytest says afterwards: pass@1, pass@2, mean wall seconds and crash count per harness. Adapters ship for `aider`, `dsh`, `pi` and `opencode`; every one is pointed at an AINode endpoint (`http://<node>:3000/v1`) and needs no API key beyond a placeholder. The hidden tests are copied into the working directory only after the harness has exited and removed again before the next attempt, so a harness can never read the assertions it is being judged on, and the second attempt is given the real test output the way Aider's protocol does it. `--dry-run` prints the exact argv, environment and config for every task and harness without running anything, and is the only part that works in a sandboxed shell: the agents themselves need normal network and home-directory access, and a sandboxed run hangs rather than failing loudly. dsh gets its own `DSH_HOME` because it validates every configured provider route at boot, so one stale entry in a person's `~/.dsh/settings.yaml` would end every run with a transport error whatever provider the run selected. Records are schema 1 with a new top-level `harness` block and no `results` block, so `scripts/render-bench-table.py` skips them rather than rendering a coding run as a very slow model. Docs: `bench/harness/README.md`.

---

## [0.5.17] - 2026-09-16

The bind wait tolerates a long autotune pause.

### Changed
- **The bind wait's log-silence window defaults to 900 s, up from 360 s.** Nemotron 3.5 Lightning on the GX10 goes quiet for 363 s during FlashInfer autotune and graph capture, so the 360 s window still declared a healthy start silent on the 0.5.16 roll of Spark-4 (`never bound on :8000 after 870s (log silent for 363s); relaunching once`). The relaunch is a no-op since 0.5.13, but the verdict spends the engine's single retry and the line misleads. Fifteen minutes covers every startup pause measured so far; the 1800 s ceiling still bounds a wedged engine.

---

## [0.5.16] - 2026-09-16

The chat model card fits its panel; the Thinking toggle reaches DeepSeek V4.

### Fixed
- **The chat Thinking toggle and the bench's reasoning section now reach DeepSeek V4.** Both sent only Qwen's `enable_thinking` switch, and the toggle sent nothing at all when ON, so a DeepSeek engine launched with thinking off by default could never be turned on from the UI, and the bench's DeepSeek "thinking on" measurement was thinking off. Both states are now sent explicitly under both template switch names (`enable_thinking` and `thinking`); a template ignores the name it does not read.
- **The chat MODEL CARD stays inside the left rail at any panel width.** An old MODELS LIST rule still declared `.model-card`, and the chat rail's card is the only element left carrying that class, so it picked up `align-items: center` and `padding: 20px`. On a column flex container the cross axis is horizontal, so centring sized the card's head and body to their own content instead of the rail's width: with DeepSeek V4 Flash selected the body measured 368 px inside a 233 px card and `overflow: hidden` cut roughly 67 px off each side, clipping the leading characters of the model id, the section labels and the chips on the left and the values on the right. The dead rule is gone and the rail card now stretches its children, and the pieces that can hold an unbreakable token wrap instead of widening the box: the label and value grid uses `minmax(0, ...)` tracks, the id, HF link, engine image, capability note and chips get `overflow-wrap: anywhere`, the catalog paragraph wraps as prose, and the card, groups and flex rows carry a zero min-width. Measured at 260, 300 and 360 px: `scrollWidth` equals `clientWidth` and nothing sits past an edge.

---

## [0.5.15] - 2026-09-16

The bind wait tolerates a real weight-loading pause.

### Changed
- **The bind wait's log-silence window defaults to 360 s, up from 120 s.** vLLM prints its weight-loading progress bar once per shard; a two-shard checkpoint (Qwen3.8 27B NVFP4 on Spark-1) is quiet for 206 s between updates, so the old window declared a healthy load silent, logged `never bound ... (log silent for 121s); relaunching once`, and spent the engine's single retry on a no-op. Six minutes covers the observed gaps with margin and the 1800 s ceiling still bounds a truly wedged engine. `engine_bind_log_silence_seconds` in config.json overrides it.

---

## [0.5.14] - 2026-09-16

The solo engine's logs are followed after the detached launch.

### Fixed
- **The solo engine's logs are followed after the detached launch**, so the bind wait sees the engine's own output instead of one container id and then silence. On the 0.5.13 roll of Spark-1 the wait declared Qwen3.8 silent after 121 s, moved on to Ornith while Qwen was still loading, and Ornith died with "No available memory for the cache blocks". `start_solo` now attaches `docker logs -f` to the solo log the moment the container is confirmed running, the same follower the mp head uses, so serialized replay actually waits for the primary to bind.

---

## [0.5.13] - 2026-09-15

Engine liveness comes from the container, not the launch client.

### Fixed
- **A solo engine is no longer declared dead the moment its `docker run -d` client returns.** 0.5.12 made the solo launch detached (#85), but the bind wait still judged death from the launch subprocess, so every solo launch read as `container exited` after 1 to 11 s and got a relaunch that failed on the container-name conflict with its own live engine (seen on every 0.5.12 roll of Spark-1). The backend now answers `engine_exited()` from `docker inspect`, `is_running()` asks docker for every launch shape, and the bind wait trusts that before the subprocess.

---

## [0.5.12] - 2026-09-15

Serialized startup replay, capability-aware routing, real parallelism in the fleet view.

### Fixed
- **A multimodal request is routed to an instance that accepts images** (#83). Ornith 1.5 is served twice under one model id, text-only on Spark-1 (stacked, `--limit-mm-per-prompt '{"image":0,"video":0}'`) and with vision on Spark-3, and the fleet proxy ordered candidates local-first on the model id alone, so a chat completion carrying an `image_url` landed on the text-only instance and came back as vLLM's `At most 0 image(s) may be provided in one prompt` even though another node could serve it. A request carrying an `image_url`, `input_audio`, `video_url` or `file` part now consults the capability cache `/api/models/caps` already fills: instances cached as accepting images go first, never-probed ones next, cached refusers are dropped, and that specific 400 is treated as a routing miss (record `vision: false`, fail over) instead of an answer. Every other 4xx still goes straight back to the caller, un-retried. When no instance of a loaded model will take an image, the 400 says so and names the nodes tried. One cache, not two: the master probes a peer's engine port directly, so its own cache already describes remote instances.
- **A distributed instance reports its real parallelism to the master** (#92). `loaded_models[].parallel` in `/api/server/status` was a hard-coded 1 on every row, so DeepSeek V4 Flash launched across Spark-2 + Spark-3 (mp shape, tensor_parallel_size 2) read as single-GPU in the Server view, in the fleet dashboard and in a bench record's `placement.tp`, even though the launch response and the `InstanceRecord` both carried 2. Every row now reads the width off the record that launched it: local instances from the InstanceManager, a peer's from the instance list on its announcement (which already carried `tensor_parallel_size`), with the peer's legacy `distributed_peers` list as the fallback. `/api/nodes` carries it per instance too. The wire format is unchanged: a node that sends no field still reads as 1.
- **Startup replay sweeps first, then launches one engine at a time per node** (#96). Rolling Spark-1 to 0.5.11 landed two engines profiling on one node at once, and vLLM sizes its KV cache from what is free when it profiles: the stacked model that launched 2 s behind the primary reported "Available KV cache memory: 1.59 GiB" and failed engine init, at the same GPU fraction that had given it a 600K-token cache on a settled node. The old engines from the previous life were also only reaped 14 s after the new primary had started, so the primary profiled next to a model that was still resident. `ainode start` now removes every engine container this node owns and waits for the daemon to finish before it launches anything, a per-node launch slot serializes the replay, `/api/models/load`, `/api/sharding/launch` and `/api/engine/set-model` (a load that cannot get the slot comes back 409 naming the launch that holds it), and the bind log reports how long the container was actually alive instead of how long the wait had been running, which is what made a 47-second life read as "never bound after 0s".

### Changed
- **DeepSeek V4 Flash (DSpark, FP8) is marked verified and recommended** (#84). Launched from the catalog on AINode 0.5.11 across Spark-2 + Spark-3 (mp shape, TP=2): ready in 10 minutes, greedy answers correct, 36.7 tok/s single-stream with DSpark speculative decoding, 78.5 tok/s aggregate at 4 streams, tool calls parsed, routed fleet-wide by the master.

---

## [0.5.11] - 2026-09-14

NCCL_IB_HCA names only the port behind the cluster interface.

### Fixed
- **`NCCL_IB_HCA` names only the RoCE port behind the cluster interface** (#84). The whitelist accepted every HCA with an IPv4 GID, so on a node with more than one RoCE NIC it also listed a live direct-connect port (Spark-2's `rocep1s0f0` on 10.0.0.x) and a link-local autoconf port. NCCL pairs HCAs by list position across ranks, tried that port against the peer, and died in `ibv_modify_qp` with a connection timeout. The list is now filtered to the ACTIVE port whose GID address is the node's fabric IP, read straight from sysfs with no extra shell call, which is the port the proven recipes pin by hand.

---

## [0.5.10] - 2026-09-14

The mp distributed launch maps RDMA devices on a real node.

### Fixed
- **The mp distributed launch maps `/dev/infiniband` into the engine containers on a real node** (#84). The presence check looked for `/dev/infiniband` from inside the AINode container, which sees sysfs but has no device node, so the head and worker were rendered with no `--device` and NCCL's IB plugin failed to initialise (`NCCL_NET=IB`, "Failed to initialize any NET plugin"). Presence is now judged from `/sys/class/infiniband` having entries, with the device path as the host-side fallback.

---

## [0.5.9] - 2026-09-14

Distributed launch without Ray, interface autodetect, and the DeepSeek V4 Flash recipe.

### Added
- **A distributed launch that needs no Ray in the engine image: the vLLM `mp`
  multi-node shape** (#84). The distributed path could only do one shape, a
  `ray start --head` container plus SSH-launched `ray start` workers plus a
  `docker exec` of `vllm serve --distributed-executor-backend ray` inside the
  head, so it required the `ray` CLI in the engine image. Two engine images we
  actually need do not have it: stock `vllm/vllm-openai:v0.27.1` (the head
  container exits 127, "ray: command not found") and the custom GB10 build that
  is the only thing serving DeepSeek V4 Flash correctly on sm121. `NodeConfig`
  gains `distributed_executor` (`ray` by default, or `mp`), and the `mp` shape
  runs one `vllm serve` container per node: rank 0 on the head, `--node-rank k
  --headless` on each peer over SSH, all rendezvousing on
  `--master-addr`/`--master-port` with vLLM's own multi-node executor. Peers go
  out before the head because the rendezvous, not a Ray port, is what waits.
  Container shape is the proven one: `--network host --ipc host --shm-size 64g
  --ulimit memlock=-1 --ulimit stack=67108864 --gpus all`, plus
  `--device /dev/infiniband` when the host has it. Serve args come from the same
  builder as every other launch, so a recipe flag still suppresses the built-in
  rather than duplicating it. Readiness, `last_log_activity`, the adaptive bind
  wait and `stop()` all work with no `docker exec`'d process: the head container
  is the server, its `docker logs -f` is the engine log, and teardown removes the
  head plus every peer container. `NodeConfig.extra_volumes` is new alongside it
  (extra `host:container[:ro]` mounts, applied to the solo and distributed
  commands) for an engine image that wants a writable cache outside the HF cache.
- **The distributed launch is recipe-aware** (#84). `POST /api/sharding/launch`
  built its per-instance config from the shared `NodeConfig` and ignored the
  catalog, so a curated model launched across nodes on the fleet default image
  with none of its proven flags, while the same model loaded on one node got its
  whole recipe. It now resolves the catalog recipe (engine image, extra vLLM
  args, env, volumes, distributed shape, KV-cache dtype, context length,
  trust-remote-code, recommended GPU fraction) and applies it as defaults, with
  the same per-launch body keys the solo load accepts overriding it, plus
  `distributed_executor` for the shape. Body parsing is now one shared
  validator, so both paths reject a malformed recipe with the same 400 instead of
  silently dropping it. The instance record carries `distributed_executor` and a
  primary launch persists the shape to `config.json`, so a restart replays the
  same shape rather than falling back to Ray on the default image.
- **Catalog entry: DeepSeek V4 Flash (DSpark, FP8)** (#84), id
  `deepseek-v4-flash-dspark`, `fraserprice/DeepSeek-V4-Flash-DSpark`. Frontier
  MoE, 284B total with 13B active per token, 1M context, MIT, `proven_tp=2`. It
  carries the full two-node recipe: `distributed_executor="mp"`, the GB10 engine
  image, `nvfp4_ds_mla` KV cache, DSpark speculative decoding, the deepseek_v4
  tokenizer/tool-call/reasoning parsers, and the engine env the image needs
  (its ENTRYPOINT is empty, `HOME` is `/tmp` and vllm lives at
  `/opt/env/bin/vllm`, so `PATH`, the CUDA paths, `HF_HOME` and the JIT cache
  dirs are all stated, the last pointed inside the HF cache mount so compiled
  kernels persist per node without a fleet-specific path). `verified=False`
  until the two-node serve is proven live, and it needs that GB10 vLLM build
  present on every node: it is a local image today, a registry publish is a
  follow-up.

### Fixed
- **`VLLM_ATTENTION_BACKEND=TRITON_ATTN` is no longer forced onto a custom or
  newer engine image** (#84). It was injected into every engine container. On the
  pinned 0.17 default it is a documented no-op hedge, and vLLM 0.27/0.28 merely
  log it as an unknown variable, but the 0.21-based GB10 fork that serves
  DeepSeek V4 Flash does honor it, and pinning a dense attention backend over
  that model's sparse MLA path is exactly the kind of override that makes a serve
  produce confident nonsense. The pin now sits behind the same gate as the other
  0.17-era workarounds (`_is_pinned_default_image`), so behaviour on the default
  image is byte-identical, including the systemd env override, and every other
  image gets no attention override unless its recipe's `extra_env` states one.
- **The cluster interface is auto-detected instead of guessed** (#34, #61). The
  installer wrote the DGX Spark NIC name `enP2p1s0f1np1` into every new
  `config.json` and `NodeConfig.cluster_interface` defaulted to `eno1`, so on an
  ASUS GX10 or any other box neither name existed. NCCL, Ray, Gloo and UCX were
  pointed at a device that was not there: fabric-IP detection returned nothing
  and the engine either bound `127.0.0.1` silently or failed EngineCore init
  with no usable reason. The reporter on #34 had to find the real name
  (`enp1s0f0np0`) with `ip -br addr` and hand-edit `config.json`. AINode now
  ranks this host's real interfaces, preferring an up RDMA-capable port with an
  IPv4, then the default-route device, then any up non-virtual device, and
  skipping loopback, bridges, veth pairs and VPN or overlay tunnels. The
  configured name still wins whenever it exists, so a pinned interface is never
  overruled; when it does not exist AINode logs one warning naming both the
  configured and the chosen device. `cluster_interface` now defaults to empty,
  meaning auto-detect, the installer detects a name at install time with the
  same ranking, `ainode start` prints the chosen interface and its address on a
  `Fabric` line, and the "could not detect fabric IP" error now lists the
  interfaces that do have an address.
- **`ainode start` on a host with no vLLM fails with an explanation, not a
  traceback** (#61). A pip-installed AINode outside the container runs the eugr
  backend by default, which shells out to `vllm serve`, so the start died with a
  raw `FileNotFoundError: [Errno 2] No such file or directory: 'vllm'`. The
  start now checks for vLLM first and exits 1 with the install command and the
  `engine_backend: nvidia` alternative, the backend turns that `Popen` failure
  into a clear error carrying the same guidance, and the old "using the eugr
  container backend" line is gone: eugr is the in-container path, not a way to
  run a container from the host.
- **Replay no longer launches every engine twice** (#80). Engines run with
  `--rm`, so after `docker stop` the daemon removes them asynchronously and
  `docker rm -f` returns first; a `docker run --name` in that gap failed with
  "Conflict. The container name ... is already in use". On the 0.5.8 roll every
  engine's first launch died that way at 0 s and the adaptive wait relaunched
  it. The pre-launch cleanup and the replay's orphan sweep now poll until the
  name is actually gone (bounded at 90 s, logged if it never clears).

---

## [0.5.8] — 2026-09-14

### Fixed
- **Static assets can no longer go stale across an update** (#77) — every
  `/static` URL in the served HTML carries `?v=<version>` and `/static`
  responses are `Cache-Control: no-cache`. After the 0.5.7 roll a browser kept
  the 0.5.6 stylesheet on heuristic freshness and rendered the new chat
  unstyled.
- **The model card only renders on the Chat view** (#77); it was showing on
  Cluster, Server, Models and Bench.
- **Startup replay no longer kills slow-but-healthy engines.** The bind wait had
  a fixed 300s window. On `vllm/vllm-openai:v0.27.1` a GB10 node spends minutes
  in FlashInfer fp4_gemm autotune and CUDA graph capture before the server
  listens, so on Spark-1 (2026-09-13) the window expired at 5 minutes on a 27B
  NVFP4 primary that bound at ~12 and a 35B-A3B stacked instance that bound at
  ~6. Both were relaunched from scratch, turning a 14-minute boot into 28. The
  wait now treats an engine as alive while its container is up and its log is
  still advancing, and relaunches only on evidence: container exited, log silent
  past `engine_bind_log_silence_seconds` (default 120), or
  `engine_bind_ceiling_seconds` reached (default 1800). An engine that dies on
  the way up still gets exactly one relaunch (0.5.5), and the ainode log now
  carries the reason and how long the wait lasted.

### Changed
- **"Made in Texas"** replaces "Powered by argentos.ai" in the web and
  onboarding footers, the CLI banner and status output, the bench report and
  the status API (`powered_by: ainode.dev`) (#77).

---

## [0.5.7] — 2026-09-13

### Added
- **Fleet-aware chat** (#73) — the picker lists every ready instance across the
  cluster; a model card shows node, GPU, VRAM, TP, quant, engine image,
  speculative decoding and the engine's live context length; capabilities are
  probed with a real request (16 px image, dummy tool) rather than guessed;
  every turn carries measured TTFT, decode tok/s from server usage, reasoning
  tokens and the serving instance, with averages kept per instance. System
  prompt, temperature, max tokens, thinking toggle and stop.
- **Bench view** (#75) — run the benchmark suite from the browser against any
  loaded instance: single stream, prefill scaling, sustained generation,
  concurrency, reasoning tax. Live progress and cancel, results table with
  JSON download in the public `bench/SCHEMA.md` shape, and the report
  rendered in-app. One run per node; refuses a not-ready instance; warns on
  a busy node. `scripts/ainode-bench.py` and `bench/report.py` are now thin
  shims over `ainode/bench/`.
- **Ornith 1.5 35B-A3B (NVFP4)** in the curated catalog (#68), and
  `bench/results/` with the measured runs behind the README table (#69, #70, #72).

### Fixed
- **Download job progress race** (#74) — the progress poller's last disk read
  could land after completion and overwrite 100% with a stale partial.

---

## [0.5.5] — 2026-08-19

### Fixed
- **Startup replay retries an engine that dies on the way up** (#63) — an engine
  can pass the launch check (its container reaches Running) and then die minutes
  later during weight load. Seen while rolling 0.5.4 onto a node: the startup
  sweep killed the running engine and replay relaunched immediately, while the
  driver was still releasing the GPU, so the nvidia hook handed the replacement
  no device (`Can't initialize NVML`, `0 active driver(s) found`) and it exited
  during load. Nothing retried, so the node came back advertising nothing and the
  model stayed missing until someone re-loaded it by hand. `_ensure_serving()`
  now waits for each engine to bind and, if it never does, waits for the GPU to
  finish releasing and relaunches once — for both the boot primary and each
  replayed stacked instance. The retry is deliberately single: a model that fails
  twice has a real problem, and a loop would hide it.

---

## [0.5.4] — 2026-08-19

### Added
- **Models launch with their own engine image and flags** (#62) — per-instance
  `extra_vllm_args` and `engine_image` on the node config, threaded through the
  per-load override set so they persist and survive restart-replay. A model whose
  published recipe needs flags AINode doesn't model (speculative decoding,
  MoE/mamba backends, reasoning + tool-call parsers) now launches through the
  normal load path instead of a hand-rolled container. A flag supplied by the
  caller suppresses the matching built-in rather than duplicating it.
- **Catalog recipes** for NVIDIA Nemotron 3.5 Lightning 30B-A3B (NVFP4) and
  Qwen3.8-27B (NVFP4, native vision), each carrying its proven engine image,
  flag set, and recommended memory fraction — applied as defaults so a bare
  `{"model": ...}` load (what the dashboard sends) launches correctly.

### Fixed
- The 0.17-era GB10 workarounds (`--enforce-eager`, the NVFP4 MARLIN env) now
  apply only to the pinned default engine image. They are bugs in that build;
  on newer engines they only disable CUDA graphs and cost throughput.
- `vllm serve` argv is normalized across engine images. `vllm/vllm-openai` bakes
  `ENTRYPOINT ["vllm","serve"]` while the default image uses NVIDIA's passthrough
  shim, so emitting our own prefix produced `vllm serve vllm serve <model>` and
  the engine exited with "unrecognized arguments". The entrypoint is deliberately
  not overridden — that would bypass CUDA setup.
- Engine containers no longer run with `--rm`, so an engine that dies during
  startup leaves a readable corpse instead of erasing itself. `start_solo()` now
  confirms the container reached Running and logs the engine's last output on
  failure, rather than reporting success as soon as the docker CLI forked.
- Eject survives reboot. It was memory-only, so startup replay resurrected
  ejected models; it now rewrites the instance manifest and clears the node's
  model claim when the ejected instance was the primary.
- The boot path no longer launches the legacy host-venv engine when vLLM isn't
  importable — it uses the configured container backend instead of starting a
  guaranteed "No module named 'vllm'" failure behind an "Engine starting" banner.
- Vision models served from the HF cache now get `--kv-cache-dtype auto`
  explicitly; the automatic fp8 downgrade only fires for models on local disk,
  and fp8 KV corrupts VLM generation on GB10.

### Changed
- Working branches use the `fable/*` prefix (was `codex/*`).
- The ruff rule set is pinned explicitly so lint is deterministic across ruff
  upgrades; an unpinned `ruff>=0.1.0` had started failing every PR on rules the
  repo never opted into.

---

## [0.5.3] — 2026-07-07

### Fixed
- **Truthful instances everywhere** (#59) — every instance card shows its node
  NAME (not a hex id); the Server view lists stacked instances with node + port
  and its count matches reality (Eject only where it can actually target);
  per-instance status probes both directions (`starting → serving`, and back to
  `failed` when an engine dies — no more stale STARTING bars or false READY);
  the top-bar update banner now updates the whole fleet
  (`/api/cluster/update-all`) with an honest confirm, not just the head node.

---

## [0.5.2] — 2026-07-06

> The two majors from the 0.5.1 live lifecycle verification. Image published to GHCR;
> deploy via `ainode update` at the operator's discretion.

### Fixed
- **Download Cancel is real** (#56) — cancel was a backend no-op (the flag was never
  wired into `snapshot_download`; a cancelled 15 GB download ran to completion).
  Downloads are now per-file, cancel-checked between files, **commit-pinned** (single
  sha for the whole snapshot — no mixed-commit directories if the repo is pushed
  mid-download) and parallel (`AINODE_DOWNLOAD_MAX_WORKERS`, cancellable futures).
- **Launch form respects user node picks** (#56) — auto-recommend fired on model
  select and silently reset the node dots to the head, landing node-targeted loads
  on the wrong machine. Manual node toggles now set an intent flag auto-recommend
  honors.

---

## [0.5.1] — 2026-07-06

> Same-day follow-up to 0.5.0: every defect found while live-proving the 0.5.0 fleet
> (GUI sweep, browser-driven training, vision serving), fixed and review-gated.

### Fixed
- **Per-load overrides survive restarts** (#51) — solo loads persisted `kv_cache_dtype`
  / `max_model_len` / aliases only to the stacked-instance manifest; the primary model
  boots from `config.json` and silently lost them (a VLM loaded with `kv_cache_dtype=auto`
  came back on fp8 → corrupted output). All override fields now persist and reset
  correctly, with no cross-model inheritance on the launch config.
- **fp8 KV cache off by default for multimodal models** (#51) — fp8 KV corrupts VLM
  generation on GB10 (text models unaffected); vision models now default to `auto`,
  explicit operator fp8 still honored via a provenance flag.
- **Stacked-load admission guard** (#53) — stacked loads require explicit
  `gpu_memory_utilization` (400) and reject when the node's projected total exceeds
  0.9 (409) — the missing check let a default-sized second model hard-crash a node
  (unified memory). Gate runs before any existing instance is touched.
- **Training accepts on-disk models** (#53) — wizard cards submitted disk slugs
  (`org--name`) that HF rejects; wizard now sends canonical repo ids and the container
  builder maps any on-disk reference to its mount across all four disk layouts.
- **Training/merge no longer require live PyPI** (#53) — peft wheel vendored per node
  with a tolerant, import-guarded install chain (a review-caught event-loop freeze in
  the wheel fetch was fixed with executor offloading).
- **Node-targeted launches** (#53) — the launch form loaded on the head regardless of
  node selection; now routes through `/api/cluster/load` with the chosen `node_id`,
  plus a GMU input (validated on distributed launches too).
- **Stacked instances visible** (#53) — `/api/nodes` surfaces per-node instances;
  dashboard renders STACKED sub-cards (ports 8001+) with their own unload controls.
- **GUI lifecycle polish** (#54) — Server-view LLM "Load" button actually loads
  (was a "coming soon" toast); cancel-download failures surface a toast and recover
  (true event delegation on the queue); training wizard shows an honest empty state
  instead of hardcoded not-on-disk models; dead `renderModels()` view purged.

### Added
- **AutoData v2.2 live results** (#52) — first live val-set run on the fleet:
  weak-solver lift **+0.458** (0.500 → 0.958 on 24 held-out GT probes),
  exact-McNemar p=0.0005, 44% yield, early stop at target in round 1.

---

## [0.5.0] — 2026-07-06

> Consolidates the fleet-internal 0.4.45 / 0.4.46 builds (never published to GHCR)
> plus everything since. First release cut by the tag-triggered CI pipeline.

### Added
- **AutoData (training utility)** — agentic Δ-filtered synthetic-data generation
  (Meta Autodata). `ainode/training/autodata/`: a Challenger generates tasks, weak +
  strong solvers attempt each, a Judge grades both, and only `Δ = I_strong - I_weak == 1`
  examples (strong solves, weak fails — the "zone of proximal development") are kept and
  emitted as ShareGPT JSONL with a yield report. Pure HTTP over AInode-served
  OpenAI-compatible endpoints (no torch — runs in the slim orchestrator). CLI:
  `python -m ainode.training.autodata.run --config cfg.json`. Includes the v2.1
  history-aware meta-optimizer (`run --meta`), a dashboard panel, and a verify-mode
  judge (#40, #44).
- **Training in a spawned GPU container** (#46) — LoRA/QLoRA/full jobs and adapter
  **merge** now run in a GPU container (quant-image based) when the orchestrator is
  the slim shipped image: models store mounted RW, datasets RO, runner + container
  config staged into the job dir, `peft` pip-shimmed at launch. Job cancel tears down
  the actual container. Distributed (DDP) training remains host-mode.
- **Training view completed** (#45) — job detail gains state-gated **Merge Adapter →
  Full Model** and **Resume from Checkpoint** actions plus an **Artifacts** panel with
  download links; dead legacy form + placeholder Benchmarks tab removed.
- **Deploy pipeline** (#47) — `git tag vX.Y.Z` → CI (self-hosted Spark runner) builds
  and pushes `ghcr.io/getainode/ainode:X.Y.Z`; `ainode update [version]`, `POST
  /api/engine/update`, and `POST /api/cluster/update-all` now genuinely swap the
  running image via an `image.env` EnvironmentFile read by the systemd unit
  (`Restart=always`, self-stop relaunch), gated so un-migrated units pin but never
  self-kill. `install.sh` resolves and pins the latest numeric tag instead of
  trusting `:latest`.
- **PR CI** (#45) — `tests.yml` runs ruff + pytest on every PR and main push.
- **Launch path: served-model-name aliases + per-load overrides** (#41) —
  `served_model_name: [aliases]`, `max_model_len`, `kv_cache_dtype`, `quantization`,
  `trust_remote_code` accepted by `POST /api/models/load`, emitted on both local-mount
  and remote paths, persisted in the replay manifest.

### Fixed
- Merge endpoint rejected its own sentinel config (`method="__merge__"` failed
  validation) — merges silently 500'd (#45).
- `__version__` is now single-sourced from package metadata (pyproject) — the banner
  can no longer drift from the built image (#47).
- Models page "Installed" list shows disk-only models, not just catalog entries (#42).
- Dashboard no longer re-renders the active view on the 5s poll while the user is
  interacting (#43).
- Repo-wide lint debt cleared (64 ruff errors → 0) and kept at zero by CI (#45).

---

## [0.4.44] — 2026-06-27

### Added
- **In-browser quantization (Quantize feature).** New **Training → Quantize a
  Model** panel and `POST /api/training/jobs` with `method:"quantize"`. Runs
  llm-compressor one-shot PTQ inside a GPU container, producing a
  compressed-tensors checkpoint vLLM serves natively. Schemes: **AWQ** (W4A16,
  serves as `awq_marlin` on GB10) and **NVFP4** (Blackwell-native). Output lands
  in the model store at `~/.ainode/models/<org--name>-<scheme>`, discoverable in
  Installed.
- **Push quantized models to Hugging Face.** Optional `push_to_hf` uploads a
  private repo under the write-token owner's namespace after the job exits.
- **Hugging Face _write_ token + Secrets slot.** `huggingface_write_token`
  preferred over the read token for pushes; read-only tokens are rejected up
  front (no wasted multi-GB transfer). Secrets store also holds NGC / W&B /
  OpenAI keys, masked with a per-key Test.
- **Idle-node guardrail.** A quant job is refused with `409` if the node is
  serving a model (quantization needs the full unified memory); override with
  `force=true`.

### Fixed
- **Qwen3.5 family quantizes to a servable checkpoint.** These are
  `*ForConditionalGeneration` (bundled vision tower + text sub-config) and hybrid
  (Gated-DeltaNet linear attention). The runner now loads the full model class so
  `save_pretrained` emits the complete config, ignores the vision tower /
  embeddings / `lm_head` and the tiny `linear_attn` projections (Marlin can't
  tile them), and saves the image processor. AWQ verified servable;
  NVFP4-on-Qwen3.5 remains experimental.
- Quant job bodies parse without a dataset; output always written to the
  RW-mounted model store (never the throwaway `--rm` layer); tokenizer passed as
  processor to avoid the `mistral_common` import conflict.
- `service/systemd.py` image tag now follows `__version__` (was pinned to an old
  tag, so a fresh install could render a unit pulling the wrong image).

---

## [0.4.38] – [0.4.42] — 2026-06-25..26

### Added
- **Models page redesign** — two lists (Installed vs Browse), smarter live HF
  browse, and the instance control relabeled **DELETE → UNLOAD** (a real unload,
  including force-clear of dead/phantom instances).
- **fp8 KV-cache default on GB10 vLLM serve** — long-context headroom by default.
- **`start-clean` knob** — skip model replay on boot.

### Fixed
- **BUG A (discovery cache).** The cluster re-broadcasts the live model each sync
  cycle, and discovery probes engine _liveness_ (`/v1/models`) instead of a
  latched "ready" flag, so a node that died no longer advertises a stale model.
- Dashboard displays the truthful state (no phantom READY).

---

## [0.4.32] – [0.4.37] — 2026-06-23..24

### Added
- **Model stacking — N models per node.** Solo loads append into an
  `InstanceManager` instead of fighting over the primary slot; the boot engine
  seeds the manager so it doesn't collide on `:8000`.
- **AWQ catalog + always-on instance persistence** — running instances persist
  and replay on restart (orphan sweep first; loads serialized to avoid
  concurrent-load OOM).
- **Serve models from on-disk weights** (`~/.ainode/models/<slug>`) + per-node
  download cap. Community MoE picks added (Nemotron Cascade 2, MiniMax-M2.7);
  4B-AWQ verified (~15 t/s — dense AWQ is dequant-bound on GB10).

---

## [0.4.24] – [0.4.31] — 2026-06-21

### Added
- **Federated master router (F1).** The master routes `/v1/*` requests to the
  node serving the requested model.
- **Load / unload any model on any node from the master (F2).**
- **Federation usable from the browser** — per-node memory-utilization knob,
  fleet UI, routing failover, and GB10 unified-memory telemetry.

### Fixed
- Federation proxy: strip forwarded charset/Content-Length, drop the stale 503
  guard; `--include-dashboard` is head-only (restores worker join); `start_solo`
  pre-cleans its container name for idempotent launch.

---

## [0.4.23] — 2026-06-21

### Added
- Auto-distribute weights at launch (rsync-preferred) groundwork toward
  federation.

---

## [0.4.22] — 2026-06-20

### Added
- **Concurrent multi-instance serving (P2-2).** A head node can now run several
  distributed instances at once: an `InstanceManager` tracks them, each gets its own
  api port (8000, 8001, …), Ray port (6379, 6380, …), MASTER_PORT, container names, and
  config snapshot. `/api/sharding/launch` now **appends** an instance instead of tearing
  down the running one; eject/unload stops a single instance by model. The announcement
  advertises all instances a head runs. Route-by-model on :3000 is P2-3 (a 2nd instance
  is reachable on its own port for now).

---

## [0.4.21] — 2026-06-20

### Changed
- **Weight distribution prefers rsync** (resumable + incremental) over tar-over-ssh; the
  image now ships rsync. A dropped transfer no longer re-sends the whole model on relaunch.
  Falls back to tar if rsync is absent.

---

## [0.4.20] — 2026-06-20

### Added
- **Phase 3a — auto-distribute weights at launch.** A distributed launch now ensures
  each selected peer has the model in its cache; if missing, the head streams the
  weights over the fabric (tar-over-ssh; the image has no rsync) before starting that
  peer's worker. No more manual pre-placement — pick any nodes and launch. Reports a
  `distributing` load phase. Skips peers that already have it.

---

## [0.4.19] — 2026-06-20

### Fixed
- **Server MODEL INFO "Size on disk" now shows the real size** (completes B2). The
  server view loads a raw `/api/models` catalog into `_serverState` and reads
  `local_size_gb`/`size_gb` — the loaded-model object had no size and
  `this.state.catalog` was never populated in this view (and remapped the field).

---

## [0.4.18] — 2026-06-20

### Fixed
- **Server view "Reachable at" no longer renders blank** — falls back through
  `reachable_at[1] → [0] → "—"` (was blank when the preferred entry was empty).
- **Server MODEL INFO shows real values** — Quantization (e.g. NVFP4) and Arch
  (family) and Size on disk are now derived from the catalog/model-id instead of
  showing `none` / the org name / `—`. (Real arch like `LlamaForCausalLM` needs the
  per-model config.json — tracked as a follow-up.)
- **`fabric_ip` now live in `/api/nodes`** (carried in this roll; was added to
  `/api/cluster/resources` in 0.4.17).

---

## [0.4.17] — 2026-06-20

### Fixed
- **Node-picker pills no longer wrap.** `.node-dot` was a fixed 40px circle (sized for the
  old single-digit count picker); node-name labels like "Spark-4" wrapped to two lines. Now an
  auto-width pill (`nowrap`), so "Spark-1 ★" / "Spark-4" render on one line.

### Added
- **`fabric_ip` in `/api/nodes`** — each node's cluster-fabric IP is exposed for UI/debugging
  (the launch already resolves it server-side; this just surfaces it).

---

## [0.4.16] — 2026-06-20

### Added
- **Proven-config catalog.** Curated models carry `proven_tp` + `verified`; the launch
  dropdown marks verified models with ✓ and picking a model pre-selects its proven node
  count in the picker (70B → 2, 235B → 4).

### Changed
- **`NVIDIA_VLLM_IMAGE` is env-resolved** with the proven GB10 default
  (`scitrera/dgx-spark-vllm:0.17.0-t5`); no deployment hand-seds the source anymore.

---

## [0.4.15] — 2026-06-20

### Fixed
- **Dashboard showed a distributed instance as `SINGLE`.** A running TP=N model
  rendered as single-node because (1) `distributed_instance.model` came back empty
  when the head had started idle then launched (stale announcement model) — the UI
  gates its DISTRIBUTED card on that field; and (2) membership matched `peer_ips`
  (now fabric IPs) against `node_id`. The cluster/resources builder now reads the
  model from the instance_id and resolves fabric-IP peers back to member node_ids
  (`peer_node_ids` / `member_names`); the UI keys on those and no longer over-counts
  idle `member`-mode nodes. Verified in a live browser.

---

## [0.4.14] — 2026-06-20

### Added
- **Node selection for distributed launch.** The launch panel is now a per-node
  picker (head pinned ★, others toggle) instead of a count selector; the launch
  POSTs explicit `node_ids`, so you choose *which* nodes span a model (head +
  selected peers), not just how many.
- **Single stable endpoint, visible loading.** `:3000` returns a clear
  `503 {load_phase}` during a model swap instead of proxying into a hang.

### Fixed
- **BUG D — distributed launch now uses FABRIC IPs.** Nodes broadcast their
  fabric IP (`NodeAnnouncement.fabric_ip`); the head resolves participating peers
  to fabric IPs and refuses to launch on a node with no known fabric IP. The old
  path used the mgmt-LAN UDP source IP, landing a Ray worker on a non-GPU address.

---

## [0.4.13] — 2026-06-18

### Added (Phase 3 — model lifecycle, in-app)
- **Curated cluster models in the catalog (3b)** — the frontier/NVFP4 models this GB10 cluster runs (235B/397B/405B-NVFP4, 70B, GLM) are now always merged into the catalog (`CURATED_CLUSTER_MODELS`), so they're discoverable + downloadable instead of only appearing once on disk. `_find_model_dir` now resolves downloaded-state across all on-disk layouts (`org--name`, `hub/`, `hf-cache/hub/`), so a catalog model present anywhere reads as downloaded.
- **Live load-phase card during spin-up (3c)** — `NvidiaBackend` tracks a coarse load phase (`starting→loading_weights→distributed_init→profiling→ready`) from engine log markers, exposed via `health_check` + `/api/status` `load_phase`. The instances panel shows a LAUNCHING card with the phase + a progress bar through the multi-minute launch; a stall is visible as the phase that stops advancing.

### Added (Phase 3 — telemetry fan-out, shipped in the 0.4.12 lab build)
- **Per-peer GPU telemetry fan-out** — every node now stamps live VRAM/util/temp onto its 5s discovery broadcast (`NodeAnnouncement` gains `gpu_memory_used_mb`/`gpu_memory_total_mb`/`gpu_utilization`/`gpu_temp`, default-valued so older nodes stay compatible). `BroadcastSender` refreshes them each tick from the node's `MetricsCollector`; `ClusterNode` carries them; `/api/nodes` exposes `gpu_memory_used_pct`/`gpu_utilization`/`gpu_temp` per node (local node read fresh from its own collector). The cluster graphic now shows real VRAM on **all** nodes, not just the head — closing the Phase-1 worker-VRAM gap.

---

## [0.4.11] — 2026-06-17

### Fixed (dashboard now reports the truth — Phase 1+2)
- **Phantom READY eliminated** — `/api/status` `engine_ready` is now a live `/v1/models` probe each call (latched `engine.ready` no longer trusted); the instances panel + master node card derive status from it (READY vs STARTING) instead of a hardcoded `READY`.
- **Per-node VRAM no longer stuck at 0%** — `metrics/collector.py` falls back to `psutil` for GB10 unified memory (nvidia-smi reports N/A), and the UI merges live GPU metrics into the (local) node so the memory ring shows real %. *(Per-peer VRAM still pending a metrics fan-out — Phase 3.)*
- **Cluster graphic shows distributed membership** — participating nodes are stamped with the model + `TP=N` from the authoritative `/api/cluster/resources` `distributed_instance` (the old path keyed off `active_sharding`, which is `null` while serving, so workers showed nothing).
- **Ray status no longer falsely "not installed"** — `/api/sharding/status` derives Ray health from the head engine when a distributed instance is serving (the orchestrator container has no `ray` binary to probe).
- **DELETE works on a dead/phantom instance** — `unload` force-clears (`stopped: true`) when the engine is unreachable instead of blocking on a `SIGTERM` that can't confirm.

### Changed
- **AUTO sharding now defaults to Tensor parallel** (was Pipeline) — matches the distributed launch path and this hardware. Launch UI defaults the Sharding pill to **Tensor** and relabels the node selector "Tensor Parallel Size (nodes)".
- **Launch dropdown marks model state** — loaded (●) / on-disk (○) so you can tell what's downloaded.

---

## [0.4.10] — 2026-06-17

### Fixed
- **GB10 (sm120) frontier-MoE inference no longer crashes on the first request** — `NvidiaBackend` now passes **`--enforce-eager`**. vLLM auto-selects the FlashInfer attention backend on Blackwell, whose prefill kernel (`BatchPrefillWithPagedKVCache`) emits an `illegal instruction` **under CUDA-graph capture** on GB10/sm120 and kills EngineCore on the first real prefill (the engine loads, reports READY, then suicides — vLLM SIGTERMs its own Ray workers; `/v1/models` 200 is not proof the engine works). `--enforce-eager` disables graph capture and the same FlashInfer kernel runs clean. Verified live: `nvidia/Qwen3-235B-A22B-NVFP4` at TP=4 across 4× GB10 survived a 3,513-token prefill and stayed alive. (`ainode/engine/backends/nvidia.py`)
- **`models/hf-cache/hub/` now scanned for downloaded models** — `registry.py` `list_available` + `list_downloaded` previously walked only `models/` and `models/hub/`, so HF-transfer downloads (e.g. the 235B) were invisible in the launch dropdown despite being on disk. +2 tests.

### Changed
- **`NvidiaBackend` sets `VLLM_ATTENTION_BACKEND` (env-overridable, default `TRITON_ATTN`)** — **currently a no-op**: the scitrera/vLLM 0.17.1 serving image does not honor it (ranks still log `Using FLASHINFER`). Retained as a hedge for a build that does (correct value may be `TRITON_ATTN_VLLM_V1`). `--enforce-eager` above is what actually prevents the crash; re-enabling CUDA graphs for throughput will require a working non-FlashInfer backend first.
- **Adopted the DOX `AGENTS.md` edit-contract convention** — root `AGENTS.md` + a child at `ainode/engine/AGENTS.md` (distributed-launch + GB10 vLLM-flag invariants), plus a chain-walk pointer atop `CLAUDE.md`. References only; no duplication.

### Known limitations
- **The lab's serving image (`scitrera/dgx-spark-vllm:0.17.0-t5`) is still injected by a build-time patch, not the repo.** `NVIDIA_VLLM_IMAGE` is hardcoded to `nvcr.io/nvidia/vllm:26.02-py3` (vLLM 0.15.1, can't serve MoE); the deployed `0.4.10` image is built `FROM` the patched orchestrator with the scitrera repoint re-applied. Follow-up: make `NVIDIA_VLLM_IMAGE` env/config-driven so a clean `Dockerfile.ainode` build is the canonical path. Until then, a from-scratch repo build regresses the serving image.

---

## [0.4.9] — 2026-04-18

### Fixed
- **4-node distributed inference: `NCCL_IB_HCA` now auto-detected per-node** — four bugs in the NCCL environment generator prevented TP=4 ring formation on any cluster with heterogeneous HCA naming (MOFED `mlx5_*` on some nodes, stock Ubuntu `rocep*`/`roceP*` on others), or with vestigial direct-connect ports on dual-homed nodes:
  1. `_detect_ib_hca()` filtered `ibdev2netdev` output by the `mlx5_` prefix and silently returned empty on non-MOFED nodes, triggering a hardcoded `mlx5_0` fallback that pointed at a direct-connect port. Now accepts both naming schemes.
  2. `_detect_ib_hca()` did not filter by port `(Up)` state. Down ports could poison the output. Fixed to match eugr's autodiscover contract.
  3. Detection ran on the head only and broadcast one value cluster-wide. New startup shim (`scripts/nccl-env-init.sh`) runs inside each vllm_node container via `docker run --entrypoint` and exports the correct node-local `NCCL_IB_HCA`.
  4. `_write_eugr_env()` conflated `IB_IF` (HCA device name in eugr's model) with `cluster_interface` (netdev). Fixed so eugr's `launch-cluster.sh` receives a real HCA list for `NCCL_IB_HCA`.
- **Direct-connect HCAs now excluded from NCCL ring automatically** — detection now filters HCAs to those whose netdev IP is on the cluster subnet (derived from `cluster_interface`). On Sparks 1 & 2 in our reference cluster, this keeps `mlx5_1` / `mlx5_3` (switch-facing) and drops `mlx5_0` (10.0.0.x direct-connect), eliminating one source of TP=4 hangs.

### Changed
- **ainode systemd unit now mounts `/mnt/shared-models`** via `docker --mount type=bind`. Required by the new per-node NCCL shim publish path. Uses `--mount` (not `-v`) so the service **fails loudly** if the path is missing rather than silently degrading to partial fix.

### Upgrade notes
- **Create `/mnt/shared-models` on every node before upgrading** — `sudo mkdir -p /mnt/shared-models`. NFS mount recommended for clusters (so the master's `docker run -v` mount on each peer resolves to shared model files and the NCCL init shim). On the master, bind-mount or NFS-export the path from your model storage location.
- The v0.4.9 fix is fully effective only when every node has `/mnt/shared-models` populated. Without it, the service fails to start until the path exists. Create the dir, or downgrade if you're not ready.

### Known limitations
- The per-node NCCL shim is distributed via `/mnt/shared-models`, not baked into the `ainode-base` image. Clusters without shared storage get a startup failure by design. `TODO(v0.4.10)` — move the shim into `ainode-base` so the mount becomes optional and `ainode-base` containers have the shim natively.
- Performance numbers for TP=2 on direct-connect 10.0.0.0/24 (advertised in v0.4.0's README as "NET/IB RoCE @ 200 Gb/s") have not been re-measured on the switched fabric post-fix. v0.4.9 smoke test captures live `NCCL_DEBUG=INFO`, `ib_write_bw`, and pre/post TP=2 throughput to `ops/runbooks/2026-04-18-v0.4.9-verification.md`. If results differ from v0.4.0's claim, a README footnote will acknowledge the earlier measurement was on direct-connect and may have been partially socket-fallback.

---

## [0.4.8] — 2026-04-17

### Fixed
- **"Launch Instance" silently failed in distributed mode** — `DockerEngine` was missing a `launch_distributed()` method that `/api/models/load` expects. The UI Launch button kicked off an 8-way inference request that quietly returned without starting anything. Added a compatibility shim that accepts a sharding config, writes it through to `NodeConfig`, and delegates to the existing `start_distributed()` path.
- **Cluster-wide "Update all" no longer requires SSH** — previously the master SSHed into each worker to run `docker pull && systemctl restart`, which broke whenever a peer's `~/.ssh` was owned by root or the keys weren't pushed. Updates now use each worker's HTTP API (`POST /api/engine/update`). The master updates itself last. SSH remains required only for distributed inference bootstrap via eugr's launcher.

---

## [0.4.7] — 2026-04-16

### Fixed
- **Image drop overlay stuck** — dragging a file across the chat window showed a "DROP IMAGES TO ATTACH" overlay that couldn't be dismissed. Now dismisses on Escape key, clicking outside, or dragging away from the window.
- **Master node stuck on "starting..." forever** — when the master had no model configured, the topology showed a permanent "starting..." overlay. Now correctly shows "online" when the web server is up with no engine to wait for.
- **Minimum Nodes selector capped at 3** — was hardcoded in the HTML. Now dynamically populated from the actual discovered cluster size. With 4 nodes online, shows 1-2-3-4.
- **Launch Instance model dropdown incomplete** — only showed models loaded in vLLM. Now merges the catalog with `/api/models/downloaded` so all disk-present models appear, even if not currently running.
- **Download button shown for already-downloaded models** — live catalog tabs (Trending, Most Used, Latest, HF Search) didn't check disk presence. Fixed across all views. Badge shows "Downloaded" instead of "Available". Re-downloading blocked with a toast. (Fixes [#35](https://github.com/getainode/ainode/issues/35))

### Infrastructure note
AINode supports NFS-shared model storage across a cluster. Download a model once to a central NVMe-over-TCP volume (e.g. a NAS, MikroTik ROSA, or TrueNAS), export it via NFS, and mount it on every node. Set `models_dir` in each node's `~/.ainode/config.json` to the shared mount path — all nodes serve from the same weights with zero duplicate downloads. See the [Cluster Setup docs](https://docs.ainode.dev/cluster-setup) for details.

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
