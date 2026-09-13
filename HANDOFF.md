# HANDOFF — AINode (2026-07-08, session end: Fable final push + polish day)

> Cold-start handoff. **Headline:** AInode is a shipped product. Fleet runs **0.5.3** on all 4 Sparks (deployed with ONE `update-all` call — the pipeline is routine now). Every surface is synced: GHCR + GitHub Releases (0.5.0–0.5.3), docs.ainode.dev (23 truth-synced pages), ainode.dev (self-updating version badge + live-fleet screenshots), README/CHANGELOG. Both locked contracts closed: the product push (7/7 acceptance) and the docs rebuild (4/4, Opus-only).

---

## ✅ Shipped this session (2026-07-06 → 07-08, PRs #45–#60 + docs#2/#3 + site#1)

| Release | What | PRs |
|---|---|---|
| 0.5.0 | Training view complete; LoRA+merge in spawned GPU container; deploy pipeline (tag→CI on spark-1 runner→GHCR→real `ainode update` via image.env swappable units); PR CI; lint 64→0; drop-ins retired fleet-wide | #45–#49 |
| 0.5.1 | Per-load overrides persist across restarts; **fp8-KV auto-skip for multimodal** (fp8 corrupts VLMs on GB10); **stacked-load admission guard** (400 explicit-gmu / 409 >0.9 total — the missing check that hard-crashed spark-2); node-targeted launches + GMU field; stacked instances visible; slug→mount mapping; offline peft vendoring; GUI lifecycle polish | #50–#55 |
| 0.5.2 | Real cancellable downloads (commit-pinned + parallel); launch form respects user node picks | #56–#57 |
| 0.5.3 | Truthful instances everywhere (node NAMES on cards, stacked in Server view + honest counts, bidirectional status probes — no stale STARTING/READY); **update banner is fleet-wide** (`/api/cluster/update-all`) | #59–#60 |

**Also:** AutoData v2.2 (val-set lift objective, exact McNemar gate) with a LIVE WIN — weak 0.5B lift **+0.458** (0.50→0.96 on held-out GT), p=0.0005, results in `V2_DESIGN.md` (#50/#52). Vision/OCR proven (Qwen2.5-VL → perfect medical-note JSON; HoLaCe driver). Training proven from the browser (real LoRA → Merge → 942MB model → artifact downloads). Full GUI sweep + model-lifecycle parity matrix verified in-browser on the live fleet.

## 🖥️ Fleet right now
```
Spark-1 (100.122.26.9) → AInode 0.5.3 head (no AInode model) + Aegis-14B RAW :8001 (never blipped all session)
Spark-2 (10.100.0.12)  → OpenThinker3-7B          (0.5.3; survived a hard crash + power cycle with zero-touch replay)
Spark-3 (10.100.0.14)  → Nemotron-Cascade-2-30B   (0.5.3)
Spark-4 (10.100.0.16)  → Qwen2.5-VL-7B kv=auto    (0.5.3) + datalab-to/lift weights staged (untested)
```
- **Deploy = the pipeline, period:** `git tag vX.Y.Z` → Actions (runner `spark-1`, a systemd service) → GHCR → dashboard update banner (fleet-wide as of 0.5.3) or `ainode update [version]` / `POST /api/cluster/update-all`. Rollback = `ainode update <older>`.
- Units are image.env-swappable, `Restart=always`, survive cold power cycles with automatic model replay.
- Quant image `ainode-quant:0.17.0-t5` on spark-1 + spark-4 only (training vehicle; hand-built, outside CI).

## ▶ START HERE (next session) — top candidates, all filed in the Closet with owners
1. **P1 product bugs from the residual-gap audit** (fresh-eyes sweep, none GUI-visible yet):
   - **DDP training submission WEDGES the node's training queue** (GUI offers the tile; in-container it 500s AND jams the queue until restart — worst known bug).
   - **Embeddings tab is a mirage** (sentence-transformers not in the shipped image → every load 503s).
   - **Auth enable self-locks the dashboard** (SPA never sends bearer tokens).
2. **First-run killers for new users:** installer dies without `mkdir /mnt/shared-models`; default model is gated Llama-3.2-3B with no token.
3. **Aegis → AInode migration** (post-hackathon, unblocked): load body in the Closet entry; retires the last raw container.
4. **AutoData v2.2 multi-round** on a harder domain (round 1 hit target instantly — the prompt-optimizer loop is still live-unexercised).
5. **lift (datalab-to, 9.65B, schema-JSON OCR)** staged on spark-4, untested — the HoLaCe production play. Load with `kv_cache_dtype:"auto"`.

## ⚠️ Standing rules & gotchas (see vault `06 - Known Gotchas` for the full list)
- Serve through AInode's launch path — never raw docker (the standing dogfood rule).
- **VLMs on GB10: never fp8 KV** (auto-skip shipped, but explicit fp8 still honored — don't).
- Stacked loads need explicit `gpu_memory_utilization`; the admission guard is the only thing between you and a node crash.
- Small reasoning models (OpenThinker3) spiral at default chat sampling — per-model presets filed.
- spark-4's default route + DNS fix is **runtime-only** (netplan durability filed; watch for recurrence).
- Docker Hub mirror (`argentaios/ainode`) is stale at 0.4.7 — CI has no Hub push step; README/docs reference it.

## 📔 Where everything lives
- **Vault:** `~/Obsidian Vault/AINode/` — `01 - Current State`, `05 - Decisions Log`, `06 - Known Gotchas`, `Daily Updates/2026-07-06 + 07-07`, `Orchestration Handoffs/` (this session's full record).
- **Closet:** `~/Obsidian Vault/Follow-Ups & To-Dos.md` — three new AINode sections (2026-07-06/07), every item with Owner/Next/Proof.
- **Memory:** `ainode-050x-shipped-state`, `ainode-vlm-fp8-kv-corruption-gb10` + priors.
- **Evidence:** session scratchpad (`a4/`, `a6/`, `a7/`, `lifecycle/` screenshots, OCR JSON, adapters, run logs) — ephemeral; durable copies referenced from the vault handoff.
- **Access:** Mac → `spark1-remote` (Tailscale, `-i ~/.ssh/id_ed25519_server`); workers via fabric `ssh sem@10.100.0.{12,14,16}`.

## Operator loose ends (Jason)
- **Proctor sign-off** for the docs contract: `proctor signoff e29cb523-04b4-4370-9646-38be2728f725` (checklist = the six doc areas, all merged + live).
- Seven **marketing-copy judgment calls** on ainode.dev, listed in site PR #1's body (2-node claim vs 4-node reality, Docker Hub namespace, `:latest` examples, missing AutoData/vision in features, dated model examples, one caption, "50+ models").
