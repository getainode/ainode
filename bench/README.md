# AINode bench

What an AINode-served model actually does on the hardware in front of us, as
opposed to what a model card says.

Four benches live here, and they answer different questions:

- **Throughput** (this file): TTFT, decode rate, prefill scaling, concurrency,
  reasoning tax. How fast the model generates.
- **Harness** (`bench/harness/README.md`): pass@1, pass@2, mean wall clock and
  crash count for a model driving a real coding agent CLI (aider, dsh, pi,
  opencode) at ten vendored Exercism exercises with hidden unit tests. Whether
  what it generates works. Run it with
  `python3 scripts/ainode-bench.py harness --endpoint http://<node>:3000/v1
  --model <id> --harness aider --tasks 10 --label <label>`, and always with
  `--dry-run` first.
- **Agentic rubric** (`bench/agentic/README.md`): 25 probes with mechanical
  verdicts over the parts an agent loop is made of. Formats, tool calls (one,
  three at once, none when none is needed), executed code, reasoning traps, a
  needle at three prompt sizes, the thinking switch, vision, and a group G of
  multi-turn agentic work: a dependent tool loop, recovery from a tool error,
  argument schema fidelity, structured output, and a system rule over four turns.
  Run it with `python3 scripts/ainode-bench.py agentic --endpoint
  http://<node>:3000/v1 --ainode http://<node>:3000 --model <id> --label <label>
  --quick`.
- **Decision** (`bench/decide/README.md`): accuracy, Brier score, calibration error
  with its reliability table, wrong answers surviving a 0.8 and a 0.9 confidence
  gate, latency and cost, over 110 labeled typed decisions (route a request, triage
  a ticket, is it urgent, is this diff safe to merge, is this statement true).
  Whether a decision it makes can be trusted by code that acts on the answer, which
  is a question about its confidence more than about its accuracy. Three backends:
  AINode's `POST /v1/decide`, any OpenAI-compatible engine through the lettered
  chat fallback, and TypeSafe AI's hosted Jev for comparison. Run it with
  `python3 scripts/ainode-bench.py decide --backend chat --endpoint
  http://<node>:3000/v1 --ainode http://<node>:3000 --model <id> --label <label>`.

All four write one schema-1 JSON into `bench/results/`. A harness record carries
a `harness` block, an agentic record an `agentic` block and a decision record a
`decide` block instead of `results`, and all three are skipped by the README's tok/s
table in favour of their own.

The rest of this file is the throughput bench. Two ways to run it, one measurement:

| Path | What it is |
|------|-----------|
| `ainode/bench/` | The benchmark, as a package. The measurement lives here |
| `scripts/ainode-bench.py` | CLI shim over the package. Writes one JSON per run |
| `bench/results/*.json` | The runs, schema 1, one file per model/placement/day |
| `bench/SCHEMA.md` | The record format. Authoritative |
| `bench/report.py` | CLI shim over `ainode/bench/report.py`; writes `bench/report.html` |
| `bench/report.html` | Generated. One self-contained page, no CDN, no JS |

The product runs the same code from the browser: the **Bench** view posts to
`/api/bench/runs`, which points a run at one already-loaded instance, writes
schema-1 JSON into `~/.ainode/bench/results/`, and serves the rendered report at
`/api/bench/report`. A run started from the browser and a run started from the
terminal are the same measurement with the same honesty rules, because they are
the same module (`ainode/bench/measure.py`); only where the file lands differs.
`bench/results/` in the repo is the curated set that the site and the README
table are generated from, so a run worth keeping gets copied there by hand.

The package is still stdlib-only and still runs on a bare `python3` with no pip
step, which is why the measurement is blocking urllib rather than aiohttp; the
in-product runner drives it through `asyncio.to_thread` so it never blocks the
API server's event loop.

## Run it

```bash
python3 scripts/ainode-bench.py \
  --url http://100.72.9.84:8000 \
  --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --ainode http://100.72.9.84:3000 \
  --label dspark-recipe --no-think

python3 bench/report.py          # -> bench/report.html
```

`--url` is the engine or the AINode proxy (anything OpenAI-compatible).
`--ainode` is optional and only ever read from: it supplies GPU telemetry and
the placement block, so a result file records the node, GPU, engine image, vLLM
flags, KV dtype and what else was stacked on the node instead of relying on
someone's memory. `--label` is required and says what made the run distinct.

Useful flags:

- `--only prefill,concurrency` - sections are `single`, `prefill`, `sustained`,
  `concurrency`, `reasoning`; all five run by default.
- `--depths 4000,16000,32000,64000,120000` - prefill sweep, prompt tokens.
- `--streams 1,2,4,8,16` - concurrency sweep.
- `--no-think` - sends `chat_template_kwargs.enable_thinking=false` for every
  section **except** `reasoning`, which always measures both states because the
  comparison is the entire point of that section.
- `--max-tokens` / `--sustained-tokens` / `--reasoning-tokens` - generation
  budgets. Lower them for a slow model; a dense 405B at ~1 tok/s will sit on the
  default 1500-token sustained run for 25 minutes.
- `--show bench/results/<file>.json` - pretty-print a saved run.

The script is **inference only**. It never loads, unloads, restarts or deletes
anything, so it is safe to point at a node someone else is using. It will add
load, so do not run the wide sweeps against a node serving live traffic.

## What each section measures

- **single** - TTFT and decode on a short prompt. What one user feels.
- **prefill** - TTFT and decode against prompt length. Decode gets quoted at 4k
  and used at 120k; those are different numbers.
- **sustained** - one long unbroken generation. Does the rate hold as the KV
  cache grows and the node heats.
- **concurrency** - aggregate and per-stream throughput at each stream count.
  On GB10 this is several times the single-stream number and it is the number
  that matters for agents and multi-user serving.
- **reasoning** - the same prompt with thinking on and off, in wall clock.

## Rules the numbers depend on

These are why the results are worth keeping, so do not relax them casually:

1. **Nothing is loaded or unloaded.** Pure inference against whatever is already
   serving.
2. **Token counts come from the server.** Prompt sizes are the engine's
   `usage.prompt_tokens` via `stream_options.include_usage`, never a
   chars-per-token estimate. Generated counts are `usage.completion_tokens`,
   never a count of SSE chunks: under speculative decoding (DSpark, MTP) one
   chunk can carry several accepted tokens and chunk-counting halves the rate.
3. **A unique nonce leads every prompt**, so `--enable-prefix-caching` cannot
   serve a cached prefill and make a deep prompt look free.
4. **Decode excludes prefill.** The clock starts at the first content delta.
   Reasoning-parser output counts as generated tokens, because it costs decode
   time like any other token.
5. **`prefill_tok_s` is a floor.** It is `prompt_tokens / TTFT`, and TTFT
   includes queueing; the OpenAI-compatible API exposes no internal prefill
   timing.
6. **Missing is missing.** A section that did not run is absent from the JSON
   and renders as "Not measured". Never fill a gap with an estimate or a number
   carried over from a similar model.

## Reading the result files

The format is `bench/SCHEMA.md`; that file wins over this one. Notes on the
fields the script fills automatically:

- `placement.flags_source` says where the flags came from: `live node config
  (/api/config)` when AINode's config still describes the benched model, else
  the curated catalog recipe, which is the intended launch command rather than a
  read of the running container.
- `model.arch` / `model.active_b` are derived from the model id: the `A3B` in
  `30B-A3B` is the vendor's own active-parameter count. No `A<n>B` marker means
  the row is recorded as dense.
- `results.telemetry` holds **peaks** observed over the run, sampled from
  `/api/nodes`. On GB10, AINode reports `0` for GPU utilisation (pynvml cannot
  read a unified-memory GPU), so the script writes a note saying to read that as
  unread rather than idle. Memory and temperature are real.
- `rubric` is never written by the script. It is the hand-scored quality pass,
  added by whoever ran it.
- A model that is not in the AINode catalog gets a warning and an incomplete
  `model` block; fill `params_b`, `license` and `context` in by hand.

## How the site consumes them

`bench/results/*.json` is the source of truth, and everything downstream is
generated from it:

- `python3 bench/report.py` writes `bench/report.html` - leaderboard on top,
  then one block per run with the prefill curve, concurrency bars, reasoning
  tax, telemetry and notes. Self-contained: inline CSS tokens and inline SVG, no
  CDN, no JavaScript, dark and light via `prefers-color-scheme`. Open it from
  disk, publish it as an artifact, or serve it from the marketing site as is.
- The page is regenerated, never edited. Fix `ainode/bench/report.py` or the JSON.
- `--results` and `--out` point the renderer somewhere else, which is how you
  preview a single run without touching the committed page.
- `/api/bench/report` renders the same page from `~/.ainode/bench/results/` on
  every request, which is what the Bench view shows in its iframe. Same renderer,
  different directory.

Adding a run measured by hand is fine: write a schema-1 JSON into
`bench/results/` with `"source": "manual: ..."` saying where the numbers came
from, and re-run the renderer.
