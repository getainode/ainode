# AutoData v2 — design (Open Thoughts–grounded)

> v1 (shipped) = the Δ-filter loop: Challenger → weak/strong solvers → Judge → keep Δ=1 → ShareGPT JSONL.
> v2 closes the loop and stops reinventing curation: it adopts the **Open Thoughts** recipe
> ([arXiv 2506.04178], Bespoke Labs + DataComp) and **Evalchemy** (their open eval harness),
> so AutoData *productizes a proven curation pipeline on the fleet* instead of guessing.
> v1's 8% yield on a fixed prompt is exactly the gap v2 closes.

## Why Open Thoughts
Open Thoughts ran **1000+ ablations** on what makes good reasoning-SFT data and shipped it
fully open: datasets (OpenThoughts3-1.2M, OpenThoughts-Agent), models (OpenThinker3-7B, SOTA
open-data 7B), the generation/curation code, and **Evalchemy**. Their loop —
*generate traces from a strong teacher → **verify correctness** → curate* — is exactly
AutoData's Challenger→Judge, validated at 1.2M scale. We take their findings as the priors.

## v2 components (over the v1 MVP)

### 1. Orchestrator meta-optimization (raises yield — the headline)
Implements the AutoData reward objective `P* = argmax_P E[Reward(M(x; D(G,P)), y)]`.
- Loop: run a batch → measure yield (Δ=1 rate) **and** a val-set score → if yield/score is
  below threshold, an LLM optimizer rewrites `P_t → P_{t+1}` (and recipe knobs) toward the ZPD.
- This automates the manual ablation search Open Thoughts did by hand. Fixes the v1 8%.
- Loop-until-target or loop-until-budget (mirror the workflow patterns).

### 2. Evalchemy as the Judge + the objective
Replace v1's hand-rolled judge with Evalchemy's **proven verifiers** for benchmarkable domains:
- **Math-Verify** (symbolic exact), **code execution** (test-passing), **LLM-judge-with-GT**.
- The meta-optimizer's **reward** = Evalchemy score on a held-out val set (not just Δ-yield),
  giving the objective a real, comparable signal. Evalchemy already evaluates **vLLM
  OpenAI endpoints** → drops onto AInode-served models with no new infra.
- v1's rubric/exact judge stays as the lightweight default for open-ended domains.

### 3. Recipe knobs (from the OpenThoughts paper, as config)
Expose what they ablated, with their best-performers as defaults; the meta-opt searches them:
- **Teacher (strong solver)** choice — they moved R1 → QwQ-32B; we point it at a fleet model.
- **Question-generation methodology** — they ablated 26 and sampled the top; v2 ships a small
  menu (persona, difficulty-laddered, seed-perturbation, multi-hop) the meta-opt selects from.
- **Verification + dedup** — Evalchemy verify + content-hash dedup (their CuratedThoughts step).
- **Domain mix** — math/code/science/agent weights (their OpenThoughts3 = 850k/250k/100k).

### 4. Seeds from Open Thoughts datasets (+ lift)
- Bootstrap the Challenger with exemplars from **OpenThoughts3-1.2M** / **OpenThoughts-Agent**
  (few-shot the generator toward proven-good distributions), and benchmark against their splits.
- **lift** (datalab-to) ingests PDFs/scans → structured seeds → AutoData extends them
  synthetically (your small-dataset → big-dataset insight). Real-world seed → synthetic scale.

## Build slices (each its own /goal, smallest-first)
- **v2.1 — meta-optimizer** (the yield fix): the `P_t→P_{t+1}` loop + Δ=1-yield proxy. SHIPPED (`meta.py`).
- **v2.2 — Evalchemy-style val-set objective** (the *real* yield fix): swap the noisy keep-rate proxy
  for a held-out, GT-graded LIFT objective. **IMPLEMENTED** (`valset.py` + `meta.py objective="valset"`). See below.
- **v2.3 — recipe knobs + OpenThoughts seeds**: the methodology menu + dataset bootstrapping.
- **v2.4 — lift ingestion**: documents → structured seeds → synthetic extension.

## v2.2 — the val-set objective (IMPLEMENTED)

**Why (the honest finding, 2026-06-28):** live yield was noisy/low (best ~25%). Root cause is
the **eval signal**, not the optimizer. v2.1's reward is the per-batch Δ=1 keep-rate — a
self-graded proxy that (a) re-samples fresh tasks every round (the number wobbles for reasons
unrelated to P) and (b) rode exact-match arithmetic, whose thin ZPD mislabels correct-but-
reformatted answers. Optimizing that proxy chases noise.

**What v2.2 optimizes instead — measured lift on a held-out val set:**

    lift(P) = acc(weak | few-shot of D(P))  −  acc(weak alone)      # on a FIXED, GT-labeled val set

- `D(P)` = the round's kept Δ=1 "teacher" traces (strong-solves / weak-fails).
- The weak solver is scored on a fixed, ground-truth-labeled val set (`val_set`), **cold**
  (baseline, computed once) and **primed** with a few-shot sample (`val_shots`) of `D(P)`.
- Both are graded by the **same trusted verifier** the Δ-filter uses (`core.judge_correct` →
  `verify.py` value-verify + rubric fallback) — the Evalchemy "proven verifier" role.
- This is **in-context learning as a torch-free proxy for fine-tuning lift**: if the kept
  traces teach, showing a few raises verified val accuracy; if the batch is junk, lift ≈ 0.
  Because the val set is fixed and GT-graded, the reward is **stable and comparable
  round-over-round** — the property raw keep-rate lacks. Evalchemy grades served vLLM OpenAI
  endpoints, so this drops onto AInode-served models with no new infra (pure HTTP, stdlib-only).

**Where it lives:** `valset.py` (`evaluate`, `valset_lift`, self-check `demo()`); the meta-loop
(`meta.py meta_optimize(..., objective="valset")`) tracks `best_lift`/`best_score`, feeds the
lift trajectory to the prompt-optimizer, and stops at `val_target`. The v2.1 `objective="yield"`
path is the untouched default — nothing regresses. Config keys: `objective`, `val_set`,
`val_shots`, `val_target` (see `core.AutoDataConfig`). CLI: `--meta --objective valset`.

**Offline-testable:** `valset.demo()`, `meta.demo_valset()`, and pytest cases in
`tests/test_autodata.py` run the whole objective with injected fake clients — no network. Live
endpoints are exercised by the operator via the recipe below.

## RESULTS (live fleet run — 2026-07-06)

First live `--objective valset` run on AInode-served models (4-node GB10 fleet, ainode 0.5.0),
executed from the spark-1 host against fleet endpoints:

| date | domain | weak / strong+judge+challenger | val_set size | baseline acc | best lift | p (exact McNemar) | rounds | notes |
|------|--------|--------------------------------|--------------|--------------|-----------|-------------------|--------|-------|
| 2026-07-06 | math word problems | Qwen2.5-0.5B-Instruct / Nemotron-Cascade-2-30B-A3B-NVFP4 | 24 (GT-verified) | 0.500 (12/24) | **+0.458** (0.958 primed) | **0.0005** (significant) | 1 (early stop at target) | yield 44% (7/16 kept); config `n_tasks=16`, `val_shots=3`, `val_target=0.15` |

**Read-out.** The weak solver alone scored 12/24; primed with a 3-shot sample of the round's
7 kept Δ=1 teacher traces it scored 23/24 — lift +0.458 at p=0.0005 on the paired exact test,
clearing the 0.15 target with significance in the first round (the loop stopped early by
design). Keep-rate was 44% vs. v2.1's noisy ~25% best — and unlike v2.1, the number now
measures *teaching value on held-out probes*, not the keep-rate proxy. The verify-mode judge
+ val-set objective resolved the thin-ZPD problem that motivated v2.2.

**Caveats (honest limits).** Single round — the prompt optimizer never had to iterate, so the
history-aware rewriting path is exercised only by the offline demos so far; lift is
in-context-learning lift (the torch-free proxy), not fine-tuning lift; val set is 24 probes
(small but paired + exact-tested); judge and strong solver share one model (Nemotron). A
multi-round run on a harder domain (or lower initial-P quality) is the natural next probe.

**Ops notes.** Challenger `max_tokens` must accommodate reasoning models — the run initially
crashed with truncated JSON at the 512-token endpoint default (fixed via config:
challenger 6000, n_tasks 16). Filed: chunked task generation in `generate_tasks` would remove
the monolithic-JSON truncation class entirely.

## Non-goals (still)
- Re-deriving curation research (we adopt Open Thoughts' findings).
- A new eval harness (use Evalchemy).
- Re-generating OpenThoughts data (use theirs as seeds/benchmarks; generate only domain-specific deltas).

## References
- OpenThoughts: Data Recipes for Reasoning Models — arXiv 2506.04178
- Datasets: open-thoughts/OpenThoughts3-1.2M, OpenThoughts-Agent · Model: open-thoughts/OpenThinker3-7B (Qwen2.5-7B base, Apache-2.0)
- Evalchemy: github.com/mlfoundations/evalchemy
