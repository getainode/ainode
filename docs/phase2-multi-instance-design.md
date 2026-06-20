# Phase 2 — Concurrent Multi-Instance Serving (Design)

**Goal:** run **N models at once** on **disjoint node sets** (e.g. 70B on spark-1+2 *and*
a different model on spark-3+4), route requests by model name behind the single `:3000`
endpoint, and allow an instance whose **head is not the dashboard node** (e.g. "3+4 only").

**Bottom line from the architecture review (4 read-only agents, code-grounded):** this is
**additive, not a rewrite.** Today there is *no first-class instance object* — "the running
instance" is projected each tick from one head's flat config and hard-capped to one by a
single `break`. Phase 2 introduces a real instance record and lets the existing projections
become lists. The single-instance path keeps working throughout (back-compat for one release).

---

## The four single-instance chokepoints (current code)

### 1. Engine launch/lifecycle — `nvidia.py`, `sharding_routes.py`, `server.py`
- One engine slot: `request.app["engine"]` (server.py:85), overwritten on every launch.
- `handle_sharding_launch` mutates the **one** global `config` (sharding_routes.py:184-186),
  **tears down the prior engine** (`:196-200`), then overwrites the slot (`:212`). Launch = swap.
- **Container names collide:** head is the constant `HEAD_CONTAINER_NAME="ainode-vllm-head"`
  (nvidia.py:82); `_launch_head_container` does `docker rm -f` on that exact name first
  (nvidia.py:664) → a 2nd launch destroys the 1st's head. Workers key on peer IP, not instance.
- **Port collides:** every vLLM binds `config.api_port` (8000) under `--network host`
  (nvidia.py:742, 563) → two heads on one node both try to bind 8000; the 2nd fails.
- **Non-local head:** the head is hardcoded as "this node" (sharding_routes.py:148-150, 185);
  `_head_fabric_ip` reads the *local* NIC (nvidia.py:1018); the head container is launched with
  **local** `docker run` (nvidia.py:643) and served via **local** `docker exec` (nvidia.py:798);
  readiness probes `127.0.0.1:{api_port}` (nvidia.py:342,374). An instance excluding the
  dashboard node would run on the wrong machine.

### 2. Routing — `server.py`
- `proxy_to_vllm` forwards to a single hardcoded upstream `http://localhost:{config.api_port}`
  (server.py:556). The request `model` is parsed only for metrics (server.py:570-585), never for
  routing. `/v1/models` transparently proxies the one engine.

### 3. Data model — `config.py`, `broadcast.py`, `cluster.py`, `server.py`
- `NodeConfig.distributed_mode / peer_ips / model` are single-valued (config.py:37,68,71).
- The broadcast advertises one `distributed_instance_id` + `distributed_peers` (broadcast.py:52-53).
- `handle_cluster_resources` scans nodes, takes the **first** with an instance id, emits **one**
  `distributed_instance` dict, capped by `break` (server.py:717). `tp_size = 1+len(peers)` (712).

### 4. Dashboard — `app.js`
- `renderInstances` reads the **singular** `cr.distributed_instance` (app.js:665). Topology
  membership stamps one `di` (app.js:604-620). The node picker toggles all nodes freely
  (app.js:830) with no notion of "busy".
- **Already N-capable:** the per-card template + N-card render loop exist (app.js:733-754) and
  DELETE-by-model already works (`/api/models/unload`, app.js:765). The only gaps are the singular
  field and the swap-style launch.

---

## Target design

### Data model (the spine)
A first-class **InstanceRecord** becomes the unit of truth:
```
InstanceRecord = { instance_id, model, head_node_id, member_node_ids[], api_port,
                   tensor_parallel_size, status }   # status: starting|distributing|serving|failed
```
- **Authoritative:** an in-memory `InstanceManager` (`Dict[instance_id → record + backend]`) on
  each head — a head can own several.
- **Persisted:** `instances: List[dict]` in `config.json` (survive restart). `NodeConfig`'s
  `model`/`peer_ips`/`distributed_mode` become per-instance; node-level `distributed_mode` is
  **derived** ("head" if it owns ≥1 instance, "member" if it's in someone's `member_node_ids`).
- **Broadcast:** replace the two scalar fields with `instances: List` on `NodeAnnouncement`
  (each head advertises the instances it heads); `from_json` already drops unknown keys
  (broadcast.py:74) → older peers degrade gracefully. Keep the old fields = `instances[0]` for one
  release.
- **Exposure:** `handle_cluster_resources` drops the `break` and returns
  `distributed_instances: List` (keep `distributed_instance = list[0]` one release).

### Engine concurrency
- **Per-instance container names:** `ainode-vllm-head-<instId>`, `ainode-vllm-worker-<instId>-<ip>`.
- **Per-instance port:** allocate at launch (e.g. 8000, 8001, …); thread through serve args,
  readiness, `api_url`, and the proxy upstream. (`--network host` makes the host port the bind
  port, so distinct ports are mandatory for co-resident heads.)
- **Per-instance config snapshot:** each record carries its own config, NOT the shared mutable
  `app["config"]` (which `handle_sharding_launch` rewrites in place) — `stop()`/`start_distributed`
  must read the record, not the global.
- **Append, don't swap:** `handle_sharding_launch` adds an instance; never tears down others. `stop`
  targets one instance by id.

### Routing
- `app["instance_registry"]: {model → (host, port)}`, kept current by the InstanceManager at
  load/unload. `proxy_to_vllm` picks the upstream by request `model`; **404 model_not_found** when
  absent, **503 loading** when known-but-not-ready. `/v1/models` becomes a dedicated handler that
  unions the registry (and later, cluster members).

### Non-local head
- A **remote-head launch path**: SSH/RPC to run `docker run` (head) + `docker exec` (vllm serve) on
  the chosen head; pass the head's fabric IP in (don't locally detect); stage weights from the
  remote head's cache; probe readiness at `<head_fabric_ip>:<port>`. This is the hardest piece and
  is **separable** (Phase 2b) — concurrent-with-local-heads delivers most of the value first.

### Dashboard
- Iterate `distributed_instances` → N cards (template already supports it). Build a **busy-node
  set** (union of all instances' nodes); mark those dots disabled in `_renderNodeDots` (and fold the
  busy set into the `nodekey` re-render cache); exclude busy nodes from `_selectNodes`/hint; guard
  `launchInstance` so launching on a free node set is **additive** (toast if it overlaps a busy
  instance).

---

## Fan-out map — implementation contracts

Dependency spine: **P2-1 first (everyone depends on it)**, then P2-2 ∥ P2-3, then P2-4; P2-5 last/separable.

| # | Contract | Touches | Depends on | Verify |
|---|---|---|---|---|
| **P2-1** | **InstanceRecord + InstanceManager + instances data model** — record type, in-memory manager, `instances` list in config.json, broadcast `instances`, `ClusterNode.instances`, `/api/cluster/resources` → `distributed_instances` list (drop the `break`); keep singular back-compat. | config.py, broadcast.py, cluster.py, server.py | — | unit tests; `/api/cluster/resources` returns a list; one instance still works |
| **P2-2** | **Engine concurrency (local heads)** — per-instance container names + ports + own config snapshot; `handle_sharding_launch` appends (no teardown of others); `stop` by id. | nvidia.py, sharding_routes.py, server.py | P2-1 | unit tests; **live smoke: 2 models on disjoint node sets, both serving** |
| **P2-3** | **Route-by-model** — `instance_registry {model→host:port}`; `proxy_to_vllm` picks by model (404/503); aggregated `/v1/models`. | server.py | P2-1 (test w/ P2-2) | unit + live: model A → A's port, B → B's; `/v1/models` unions |
| **P2-4** | **Multi-instance dashboard** — N cards from the list; busy-node picker (+ nodekey cache fix); additive launch guard. | web/app.js, index.html | P2-1, P2-2 | Playwright: N cards; busy nodes disabled; 2nd launch doesn't kill 1st |
| **P2-5** | **Non-local head (Phase 2b)** — remote-head launch via SSH/RPC; head fabric IP passed in; readiness at head IP; remote weight staging. | nvidia.py, sharding_routes.py | P2-1, P2-2 | live: instance on nodes excluding the dashboard node serves + routes |

**Suggested orchestration:** P2-1 solo (the spine) → fan out P2-2 + P2-3 in parallel → P2-4 →
P2-5 if/when "skip spark-1" is needed. Each is its own locked contract with a live smoke.

## Risks / decisions
- **Port allocation** must be deterministic + recorded per instance (collisions are silent under
  `--network host`).
- **Shared global config** is the subtle trap: every record needs its own config snapshot; leaving
  `stop()`/`start_distributed` reading `app["config"]` will cross-wire instances (flagged by the
  engine map).
- **Non-local head (P2-5)** is the only piece that's a genuinely new code path (remote docker
  orchestration) rather than a generalization — hence separated as Phase 2b.
- **Back-compat:** keep the singular `distributed_instance` field + old broadcast fields for one
  release so nothing breaks mid-migration.
