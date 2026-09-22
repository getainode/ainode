# AINode bench results

One JSON file per run in `bench/results/`, named `<stamp>-<model-slug>-<label>.json`.
Every number is as the engine reported it; nothing is extrapolated. A run is one
model on one placement (node, engine image, flags) on one day.

```json
{
  "schema": 1,
  "stamp": "20260913-130400",            // UTC, YYYYMMDD-HHMMSS
  "label": "text-only-mtp",              // free text, what made this run distinct
  "model": {
    "id": "ornith-ai/Ornith-1.5-35B-A3B-NVFP4",   // HF repo id as served
    "name": "Ornith 1.5 35B-A3B",
    "params_b": 35, "active_b": 3, "arch": "moe|dense", "quant": "NVFP4",
    "context": 262144, "license": "MIT", "vision": false
  },
  "placement": {
    "node": "Spark-1-DGX", "gpu": "NVIDIA GB10", "gpus": 1, "tp": 1,
    "engine": "vllm", "engine_image": "vllm/vllm-openai:v0.27.1",
    "ainode": "0.5.6", "flags": ["--reasoning-parser","qwen3", "..."],
    "stacked_with": ["unsloth/Qwen3.8-27B-NVFP4"]
  },
  "settings": { "thinking": false, "max_tokens": 200, "temperature": null },
  "results": {
    "single_stream": { "decode_tok_s": 40.0, "ttft_ms": 194, "prompt_tokens": 4013, "gen_tokens": 200 },
    "sustained":     { "decode_tok_s": 39.7, "gen_tokens": 1500 },          // one long unbroken generation
    "prefill":       [ { "prompt_tokens": 4013, "ttft_ms": 1752, "prefill_tok_s": 2290, "decode_tok_s": 40.0 } ],
    "concurrency":   [ { "streams": 16, "aggregate_tok_s": 269.2, "per_stream_tok_s": 18.7, "median_ttft_ms": 679 } ],
    "reasoning_tax": { "thinking_off_wall_s": 5.0, "thinking_on_wall_s": 12.1, "thinking_on_tokens": 598 },
    "telemetry":     { "gpu_mem_used_gb": 88.5, "gpu_mem_total_gb": 121.7, "gpu_util_pct": 71, "temp_c": 41 }
  },
  "rubric": { "pass": 19, "total": 19, "notes": "one miss was a harness artifact, re-verified 2/2" },
  "notes": ["prefill capped by spec-decode max_num_batched_tokens=2048"],
  "source": "scripts/ainode-bench.py"       // or "manual" with a link to the session/notes
}
```

Sections may be omitted when not measured. Never fill a missing measurement with an
estimate; the page renders "not measured" for a missing key.

That holds key by key inside a section, not just section by section. The `telemetry`
block is routinely partial on a GB10: the driver exposes no GPU utilisation counter
there and NVML reports no memory usage, so `gpu_util_pct` and `gpu_mem_used_gb` are
absent from those records rather than recorded as 0 (a peak of 0 percent during a run
is a claim nobody measured). `temp_c` and `gpu_mem_total_gb` are real on the same
node and stay.

`active_b` and `arch` come from the catalog entry when it states them
(`active_params_b` / `arch` in `ainode/models/registry.py`) and otherwise off the
`A<n>B` marker in the model id, which is the vendor stating the active count in the
name. Neither is a measurement, and an entry whose shape nobody stated leaves them
out rather than reporting a MoE as dense.

## The `harness` block

A harness-bench run (`scripts/ainode-bench.py harness`, docs in
`bench/harness/README.md`) writes the same record with a top-level `harness` block
and **no `results` block**: it measured a model driving a coding agent to passing
tests, not throughput, and a zero in `single_stream` would be a number nobody took.
`scripts/render-bench-table.py` skips a record shaped like that, so it never shows up
in the README's tok/s table as a very slow model. A run that measured both puts both
blocks in one file and does get a row.

```json
{
  "schema": 1, "stamp": "20260916-142200", "label": "fleet-flash",
  "model": { "...": "as above" },
  "placement": { "...": "as above" },
  "settings": {
    "attempts": 2, "timeout_s": 900, "tasks_requested": 10,
    "harnesses": ["aider", "dsh"],
    "context_window_declared": 131072, "max_output_tokens_declared": 16384
  },
  "harness": {
    "task_set": {
      "id": "exercism-python-10", "count": 10, "available": 10, "language": "python",
      "slugs": ["binary-search", "bob", "..."],
      "source": { "repo": "https://github.com/exercism/python", "commit": "1f6aab86...",
                  "license": "MIT", "license_file": "LICENSE-exercism" }
    },
    "endpoint": "http://100.122.26.9:3000/v1",
    "protocol": { "attempts": 2, "timeout_s": 900,
                  "second_attempt_sees": "the failing test output" },
    "runs": [
      {
        "harness": "aider", "version": "aider 0.86.2",
        "scores": { "tasks": 10, "passed_at_1": 8, "passed_at_2": 9,
                    "pass_at_1": 0.8, "pass_at_2": 0.9,
                    "mean_wall_s": 21.4, "crashes": 0, "timeouts": 0 },
        "tasks": [
          {
            "slug": "isogram", "passed": true, "passed_at_attempt": 1,
            "harness_wall_s": 10.2, "crashed": false, "timed_out": false,
            "tokens": { "requests": 2, "tokens_generated": 143 },
            "attempts": [
              {
                "attempt": 1,
                "harness": { "command": "aider --model openai/... --message '<602 chars>' isogram.py",
                             "exit_code": 0, "wall_s": 10.2, "timed_out": false,
                             "crashed": false, "turns": 1,
                             "tokens_sent": 709, "tokens_received": 87,
                             "stdout_tail": "..." },
                "tests":   { "exit_code": 0, "wall_s": 0.4, "passed": true,
                             "tests_passed": 6, "output_tail": "6 passed in 0.02s" }
              }
            ]
          }
        ]
      }
    ]
  },
  "notes": ["..."],
  "source": "scripts/ainode-bench.py harness"
}
```

Rules specific to this block, all load-bearing:

- `passed` is the test command's **exit code**, never a count scraped from its
  output. `tests_passed` / `tests_failed` / `tests_errored` are recorded when the
  summary line carried them and absent when it did not.
- `pass_at_2` is cumulative: a task solved on attempt 1 counts in both.
- `harness_wall_s` includes every attempt, so a task that needed two tries pays for
  both, and `mean_wall_s` is the mean of that over tasks.
- `crashes` and `timeouts` count tasks, not attempts, and are separate: a low score
  has to be readable as "the model could not" or "the harness fell over".
- `tokens` under a task is AINode's `/api/metrics` counters differenced over that
  task's window. It is the **node's** total, so other traffic during the run is
  inside it; the key is absent when the endpoint was unreachable. `tokens_sent` /
  `tokens_received` under an attempt are the harness's own report of what it sent,
  which is a different thing and labelled separately.
- `version: null` means the harness could not be asked, and the notes say so.
- A run block carries `options` only when that harness was given one: `claude` run
  with `--claude-effort medium` gets `"options": {"effort": "medium"}` and the level
  also lands in the record's `settings` as `claude_effort`. A record without either
  was run with the agent's own default, which is not the same statement as a null.
- `command` keeps every flag verbatim and elides any argument over 200 characters as
  `<N chars>`. That argument is the prompt, which is the task's instructions plus (on
  a second attempt) a screenful of pytest output; it is regenerated exactly from the
  task by `build_prompt`, and recording it twice per task would make the file mostly
  prompt.
- `context_window_declared` / `max_output_tokens_declared` are what two of the
  harnesses had to be told before they would route; they are declared inputs, not
  properties of the served model.

## The `agentic` block

An agentic-rubric run (`scripts/ainode-bench.py agentic`, docs in
`bench/agentic/README.md`) writes the same record with a top-level `agentic` block
and **no `results` block**, for the same reason the harness block has none: it scored
a capability rubric and took no throughput. `scripts/render-bench-table.py` keeps a
record shaped like that out of the README's tok/s table and gives it a row in the
"Agentic rubric runs" table instead. A run that measured both puts both blocks in one
file and gets a row in both.

This block replaces the hand-typed `rubric` block above it. That one carried a score
somebody typed from a scratch script and a reader could not check; this one carries
every probe, its verdict, its note and an excerpt of the reply. The three older
records keep their `rubric` block as a historical claim, and the README's Rubric
column still renders it.

```json
{
  "schema": 1, "stamp": "20260917-214000",
  "label": "DeepSeek TP=2 Spark-2+3, quick",
  "model": { "...": "as above" },
  "placement": { "...": "as above" },
  "settings": {
    "groups": ["A", "B", "C", "D", "E", "F", "G"],
    "needle_tokens": [8000], "quick": true, "temperature": 1.0,
    "timeout_s": 900, "thinking_switch": "enable_thinking", "probes_requested": 21
  },
  "agentic": {
    "score": { "pass": 19, "total": 21 },
    "groups": { "A": {"pass": 4, "total": 4}, "B": {"pass": 4, "total": 4},
                "C": {"pass": 3, "total": 3}, "D": {"pass": 4, "total": 4},
                "E": {"pass": 1, "total": 1}, "F": {"pass": 1, "total": 1},
                "G": {"pass": 2, "total": 5} },
    "endpoint": "http://100.122.26.9:3000/v1",
    "protocol": { "groups": ["A", "..."], "needle_tokens": [8000],
                  "temperature": 1.0, "timeout_s": 900,
                  "thinking_switch": "enable_thinking", "max_turns": 4,
                  "verdicts": "mechanical; group C is executed, group G is a judged tool trace" },
    "probes": [
      { "id": "G1_tool_chain", "group": "G", "pass": true, "wall_s": 12.4,
        "completion_tokens": 310,
        "note": "list_files -> read_file -> read_file; answered 137",
        "excerpt": "137" }
    ],
    "needle": { "8000": true },
    "thinking_off_supported": true,
    "vision_supported": null,
    "structured_output_mode": "json_schema"
  },
  "notes": ["..."],
  "source": "scripts/ainode-bench.py agentic"
}
```

Rules specific to this block, all load-bearing:

- **Every verdict is mechanical.** No judge model and no human reading. Group C's
  `pass` is a subprocess exit: the reply's code block is run against asserts the
  model never saw, and `PASS` on stdout is the verdict. Group G's `pass` is the call
  trace: the order of the calls, the types of the arguments, and whether a number in
  the final answer came back from a tool. G2 in particular fails a reply that states a
  temperature no tool returned, whatever else the model did.
- `score` counts only probes that **ran**. A group left out by `--groups` or
  `--quick` is absent from `groups` rather than scored zero, which is why `settings`
  records what was asked for and `agentic.protocol.groups` records what ran.
- `probes` is in run order, one entry per probe, and `note` is the checker's own
  reason. A probe that failed on a transport or server error says so in `note`
  (`HTTP 400: ...`), and the record's notes list those ids separately: "the server
  refused" and "the model got it wrong" are different findings.
- `excerpt` is the reply flattened to one line and capped at 300 characters. It is
  there so a surprising verdict can be read, not so the record holds the generation.
- `needle` maps each requested prompt size to whether the password came back
  verbatim. The size is the **ask** (0.75 words per token); the probe's note carries
  the `prompt_tokens` the server actually counted, which is the fact.
- `thinking_off_supported` / `vision_supported` are `true`/`false` for a probe that
  ran and `null` for one that did not. `false` means it ran and the model did not
  comply (still reasoned with the switch off, or refused the image); `null` is not a
  claim about the model at all.
- `structured_output_mode` is `json_schema` when the server accepted a named schema,
  `json_object` when it answered 400 to that and the weaker mode was used instead,
  `unsupported` when it refused both, and absent when G4 did not run. A pass under
  `json_object` is a weaker statement than a pass under `json_schema`, so the mode is
  part of the result rather than a footnote.
- `completion_tokens` is the server's `usage` for that probe, summed over turns for
  the multi-turn probes (B4, G1, G2, G5), and `null` when the server reported none.

## The `decide` block

A decision-bench run (`scripts/ainode-bench.py decide`, docs in
`bench/decide/README.md`) writes the same record with a top-level `decide` block and
**no `results` block**, for the same reason the two blocks above have none: it scored
typed decisions against labels and took no throughput. `scripts/render-bench-table.py`
keeps a record shaped like that out of the README's tok/s table and gives it a row in
the "Decision runs" table instead.

The `model` block is whatever answered: for the hosted backend that is the version the
API reported (`jev-1.13.0`), not the alias that was asked for (`jev-latest`, which is
in `settings.model_requested`), and its placement is the one honest string there is
for a service with no node of ours behind it.

```json
{
  "schema": 1, "stamp": "20260918-041500",
  "label": "jev-latest, 110 items",
  "model": { "id": "jev-1.13.0", "name": "jev-1.13.0", "vendor": "typesafe.ai" },
  "placement": { "node": "typesafe.ai hosted" },
  "settings": {
    "backend": "jev", "endpoint": "https://api.typesafe.ai/v1/systemone",
    "model_requested": "jev-latest",
    "items_file": "bench/decide/items.json", "item_set": "decide-110",
    "items": 110, "sets": ["route", "triage", "urgency", "pr_safe", "fact"],
    "concurrency": 8, "timeout_s": 120
  },
  "decide": {
    "backend": "jev",
    "endpoint": "https://api.typesafe.ai/v1/systemone",
    "model_reported": "jev-1.13.0",
    "item_set": { "id": "decide-110", "version": 1, "file": "items.json",
                  "count": 110,
                  "sets": {"route": 30, "triage": 20, "urgency": 20,
                           "pr_safe": 20, "fact": 20} },
    "protocol": { "backend": "jev", "endpoint": "...", "timeout_s": 120,
                  "input_usd_per_mtok": 0.042, "output_usd_per_mtok": 0.0,
                  "concurrency": 8, "bins": 5, "thresholds": ["0.8", "0.9"],
                  "confidence": "the probability the backend put on the answer it gave; ...",
                  "brier": "one term, on the labeled option's probability" },
    "overall": {
      "n": 110, "answered": 110, "errors": 0,
      "accuracy": 0.964, "brier": 0.024, "ece": 0.059,
      "bins": [ { "lo": 0.8, "hi": 1.0, "count": 97, "accuracy": 1.0,
                  "confidence": 0.993 } ],
      "thresholds": { "0.9": { "kept": 104, "wrong": 2, "abstained": 6,
                               "no_confidence": 0 } },
      "tokens": { "in": 38214, "out": 4620 }, "cost_usd": 0.001605,
      "p50_ms": 290, "p95_ms": 412
    },
    "sets": { "route": { "...": "the same block, over that set's items" } },
    "rows": [
      { "id": "route-01", "set": "route", "kind": "choice", "label": "code",
        "answer": "code", "correct": true, "p_answer": 1.0, "p_label": 1.0,
        "distribution": {"code": 1.0, "chat": 0.0, "vision": 0.0,
                         "long_document": 0.0},
        "wall_ms": 310, "server_latency_ms": null,
        "tokens_in": 385, "tokens_out": 46, "error": null }
    ]
  },
  "notes": ["..."],
  "source": "scripts/ainode-bench.py decide"
}
```

Rules specific to this block, all load-bearing:

- **Accuracy is the weakest number in it.** The failure mode automation cares about is
  a wrong answer at high confidence, which is what `brier`, `ece` with its `bins`, and
  `thresholds` measure. A wrong answer at 0.95 gets acted on; a wrong answer at 0.45
  is an abstention.
- `p_answer` is the probability the backend put on **its own answer** and `p_label` the
  probability it put on the **labeled** one. Both come from `distribution` when the
  backend returned one and from its reported confidence when it did not; both are
  `null` when it reported neither, and such a row is in the accuracy, out of the
  calibration numbers, and counted in `thresholds.*.no_confidence`.
- `brier` is one term, `(1 - p_label)^2`, averaged over the rows that carried a
  probability. It is not the multiclass sum.
- `ece` is the weighted gap between `accuracy` and `confidence` over `bins`, five bins
  on `p_answer`. `bins` keeps its five entries whatever ran, with `count: 0` and null
  accuracy for an empty one.
- `thresholds` counts answers, not items: `kept` survived the gate, `wrong` is how many
  of those disagree with the label, `abstained` fell below it, and `no_confidence` is
  the answers that could not be gated either way. `kept + abstained + no_confidence`
  is `answered`, and `wrong` is a subset of `kept`.
- `errors` counts items that failed on transport or an unreadable response. They are in
  `n`, out of `answered`, and listed by id in the record's notes, because "the server
  refused" and "the model was wrong" are different findings.
- `cost_usd` is the backend's posted rate over the tokens it reported: `0` for a local
  backend, because nobody bills per token for our own hardware and the electricity is
  not a number this record measured.
- `rows` is in item order, one per item, and carries no state text: the state is in
  `bench/decide/items.json` under the same `id`, and `item_set.id` plus
  `item_set.version` say which version of that file produced these numbers. Two
  records are only comparable over the same `item_set.id`.
- A set left out by `--sets` is absent from `decide.sets` rather than scored zero, and
  `settings.sets` records what was asked for.
- No API key is ever in the record. Not in `settings`, not in `protocol`, not in a
  note.

### The `decide.jevals` sub-block

A run of the **Jevals recipe** (`scripts/ainode-bench.py decide --suite ... --transport
...`, the recipe recorded in `bench/decide/JEVALS.md` with the URL and the date it was
read) writes the same `decide` block with three additions and two deliberate absences.
It exists so an AINode-served model can be read next to Jev and its clones on the same
public question sets with the same formulas. **Old records stay valid**: everything above
still describes a record with no `jevals` key.

The additions: `decide.mode` is `"jevals-0.1.0"`, `decide.recipe` names the source page
and the date it was read, `decide.sources` describes each question set (its dataset,
split, pinned revision, licence, seed and whether its state hashes are upstream's or
ours), and `decide.jevals` holds the measurement.

The absences, both load-bearing: `decide.sets` is `{}` and `decide.overall` carries **no
`brier`, `ece`, `bins` or `thresholds`**. Those four names mean the legacy definitions
above (a one-term Brier on the labeled option, five bins, gates at 0.8 and 0.9), and the
recipe's own arithmetic is a multiclass Brier over ten bins with a per-primitive gate.
Putting one under the other's name would make two incomparable numbers look like one, so
`scripts/render-bench-table.py` renders those cells as "not measured" for such a record
and reads the recipe's figures from `jevals` instead. `decide.overall` keeps only what
means the same thing either way: `n`, `answered`, `errors`, `accuracy`, `tokens`,
`cost_usd`, `p50_ms`, `p95_ms`.

```json
"decide": {
  "backend": "decide", "mode": "jevals-0.1.0",
  "endpoint": "https://spark-1-dgx...:3443/v1/decide",
  "recipe": { "source": "https://jevals.com/methodology", "read": "2026-09-21",
              "suite": "0.1.0", "doc": "bench/decide/JEVALS.md",
              "attribution": "Jevals (jevals.com), suite 0.1.0" },
  "item_set": { "id": "jevals-0.1.0", "version": "0.1.0",
                "file": "bench/decide/sets/", "count": 900,
                "sets": {"pubmedqa": 300, "banking77": 300, "helpsteer2": 300} },
  "protocol": { "transport": "decide", "concurrency": 4, "repeats": 5,
                "batch_size": 1, "timeout_s": 120,
                "input_usd_per_mtok": 0.0, "output_usd_per_mtok": 0.0,
                "confidence": "...", "loss": "..." },
  "sources": [ { "id": "pubmedqa", "type": "noul", "questions": 300, "options": 2,
                 "seed": 20260918, "recipe_of_record": "jevals-0.1.0",
                 "gold_distributions": false,
                 "contamination": [ { "system": "...", "evidence": "...",
                                      "source": "https://..." } ],
                 "source": { "dataset": "qiaojin/PubMedQA", "split": "train",
                             "hf_revision": "9001f285...", "license": "MIT",
                             "state_hash_source": "jevals" } } ],
  "overall": { "n": 4500, "answered": 4500, "errors": 0, "accuracy": 0.712,
               "tokens": {"in": 1, "out": 1}, "cost_usd": 0.0,
               "p50_ms": 1420, "p95_ms": 7028 },
  "sets": {},
  "jevals": {
    "recipe": { "...": "the same block as decide.recipe" },
    "formulas": "jevals-0.1.0",
    "repeats": 5, "concurrency": 4, "batch_size": 1, "bins": 10,
    "handoff_target": 0.95, "handoff_min_decisions": 100, "grid": 0.01,
    "published_gates": { "pubmedqa": 0.91, "banking77": 0.96, "helpsteer2": null },
    "recipe_of_record": { "pubmedqa": "jevals-0.1.0" },
    "contamination": { "banking77": [ { "system": "...", "source": "https://..." } ] },
    "seconds": 812.4,
    "sets": {
      "pubmedqa": {
        "recipe": "jevals-0.1.0", "type": "noul",
        "items": 300, "repeats": 5, "decisions": 1500,
        "failed": 0, "malformed": 0, "one_hot": 0, "calibrated_over": 1483,
        "accuracy": 0.712, "prior_accuracy": 0.62,
        "decision_score": 41.2, "loss": 0.188, "loss_prior": 0.32,
        "ece_points": 5.8,
        "bins": [ { "lo": 0.9, "hi": 1.0, "count": 900, "accuracy": 0.96,
                    "confidence": 0.991 } ],
        "handoff_95": { "threshold": 0.93, "share": 0.62, "n": 930,
                        "accuracy": 0.951 },
        "gate": { "threshold": 0.91, "source": "jevals.com/methodology, suite 0.1.0, frozen",
                  "coverage": 0.6, "n": 900, "accuracy": 0.833 },
        "gate_local": 0.94,
        "pick_flip_rate": 0.04, "pick_flip_over": 300,
        "confidence_swing": { "max": 0.44, "mean": 0.02,
                              "question": "pubmedqa-37", "over": 300 },
        "repeat_flip_rate": 0.01, "repeat_flip_over": 300,
        "order_flip_rate": null, "order_flip_over": 0,
        "questions_per_second": 1.85,
        "tokens": {"in": 748500, "out": 3000}, "cost_usd": 0.0,
        "usd_per_1k_decisions": 0.0,
        "p50_ms": 1420, "p95_ms": 7028
      }
    },
    "overall": { "sets": ["pubmedqa"], "scored": ["pubmedqa"],
                 "mean_decision_score": 41.2 }
  },
  "rows": [
    { "id": "pubmedqa-0", "set": "pubmedqa", "kind": "noul", "repeat": 0, "order": 0,
      "label": "yes", "answer": "yes", "correct": true,
      "p_answer": 0.999769, "p_label": 0.999769,
      "top": {"yes": 0.999769, "no": 0.000231},
      "malformed": false, "malformed_reason": null, "one_hot": false,
      "wall_ms": 7028, "tokens_in": 499, "tokens_out": 2, "error": null }
  ]
}
```

Rules specific to this sub-block:

- **The unit is a decision, not an item.** `decisions` is items times repeats, and
  `accuracy` is over all of them. `items` and `repeats` say which is which, and
  `decide.rows` is one row per decision with its `repeat` and its `order`.
- **`decision_score` is `100 * (1 - loss / loss_prior)`**, with both losses in the same
  block so a reader can recompute it. 100 is perfect, 0 is answering with the label base
  rates, and **a negative score is written as it is, never clamped**. `null` when the
  prior's loss is 0, which happens only on a set where every item carries one label.
- **`prior_accuracy` is the guessing floor and travels with every block.** It is the
  base-rate answer's own accuracy on the same items, so an accuracy of 0.62 next to a
  floor of 0.62 reads as what it is. A block never states an accuracy without it.
- `loss` is the mean over items of the mean over that item's repeats: a multiclass Brier
  for `choice` and `noul`, a ranked probability score over cumulative levels for `score`.
  One item answered five times weighs the same as one answered once.
- **`ece_points` is in POINTS** (5.8 means 0.058), on the top label, over ten equal-width
  bins by `min(9, floor(round(100*c)/10))`, and `bins` is the reliability table behind it:
  ten rows always, `count: 0` and nulls for an empty one, which is the per-set
  confidence-versus-accuracy data a model card draws a calibration curve from. `null`
  when nothing could be calibrated, which is the board's dash.
- **`calibrated_over` can be smaller than `decisions`.** Malformed answers and answers
  that carried no probabilities at all (`one_hot`) are in the accuracy and the Decision
  Score and out of the ECE, the flip rates and the gate, which is the recipe's rule.
- `handoff_95` is the system's own threshold: the lowest confidence on the 0.01 grid
  where at least 100 decisions clear it and at least 95 percent of those are right.
  `share` is those decisions over **all** decisions, malformed included. `null` when no
  threshold qualifies.
- `gate` is the **published frozen** gate for that primitive with this run's coverage and
  accuracy at it, and `source` says where the threshold came from. `threshold: null` for
  `score`, which has no gate in suite 0.1.0. `gate_local` is the same rule computed over
  this one run and is **not** a board number.
- **Three flip figures, because they answer three questions.** `pick_flip_rate` is the
  share of questions whose pick changed at least once across the repeats, which is the
  one a caller who has to trust a single answer wants; `repeat_flip_rate` compares only
  repeats 0 and 1 (byte-identical requests, so nondeterminism) and `order_flip_rate`
  only the four distinct option orders of a `choice` set. Each carries its own `*_over`
  count, and a rate nothing could be compared for is `null` with `over: 0`, never 0.0.
- `confidence_swing` is the largest spread one question's confidence showed across its
  repeats, with the question named, plus the mean over questions.
- **A set holding more than one ANSWER SPACE is broken down by answer space, not by
  primitive.** An answer space is one `(type, options)` pair, and it is the unit the
  Decision Score is defined over: the label prior is the base rates of the labels in that
  option list, so pooling two questions with different option lists builds a baseline over
  an answer space neither of them has, and the thing that is meant to define 0 stops
  defining it. Such a block carries `spaces` (one full metrics block each, with its
  `options` beside it), a `types` roll-up per primitive the way a board shows one, a
  `decision_score` that is the plain mean over the spaces, and `loss` / `loss_prior` of
  `null` because a loss over two answer spaces is not a number. Its accuracy, its latency
  and its ECE are over everything, because those do carry across. A space is named by the
  question file (`<group>/<question>` for a mixed manifest) and `type#n` in first-seen
  order otherwise.
- `vs_gold` is present only for a set that ships gold DISTRIBUTIONS rather than only
  labels: `soft_accuracy` (the gold probability of the pick), `total_variation`, `kl`
  (with its `kl_floor`) and `brier_vs_gold`. Its `definition` string states the
  arithmetic, because these are **AINode's definitions and not a set card's columns of
  the same names**.
- `within_one_level` is present only on a `score` block: the share of picks within one
  level of the labeled one.
- **Every block names its recipe** (`recipe`), and every source names the recipe its own
  published third-party numbers follow (`recipe_of_record`). Where those differ, only the
  figures under this record's recipe are comparable across rows.
- **`contamination` is a finding, not a footnote.** A set in a listed system's published
  training data carries that system, the evidence and the primary source URL, and the
  record's notes repeat it in words. It says nothing about an AINode-served model that
  did not train on the set; it says a row for THAT system on THAT set measures
  memorisation.
- `batch_size` is always 1: one question per request, which is the recipe's rule and
  removes the cross-question order effect. A set card whose own reference row was
  measured several questions per request is reporting a different latency and a different
  cost, and those columns are not comparable to this record's.
- **A row carries no state and no full vector.** The state is rebuilt from
  `bench/decide/sets/<id>.json` under the same `id` and verified against its
  `state_sha256`; `top` keeps the five options the system was surest about and `p_label`
  the labeled option's own probability, which is what a reader of a wrong answer needs.
- `partial: true` with a `limit` means `--limit` took only the first N questions of each
  set. It is a transport proof, not a suite result, and the notes say so. A board listing
  needs a complete run of every task in the tab at 5 repeats.

## The `embed` block

An embedding-bench run (`scripts/ainode-bench.py embed`) writes the same record with a
top-level `embed` block and **no `results` block**, for the reason the three above have
none: the model it measured generates no tokens at all, so a tok/s figure would be
meaningless even as a zero. `scripts/render-bench-table.py` keeps a record shaped like
that out of the README's throughput table and gives it a row in the "Embedding runs"
table instead.

The `endpoint` is part of the measurement and not a footnote. The same instance
measured straight at its engine port and through an AINode node's `:3000/v1` are two
different numbers, because the second one includes the fleet routing hop, which is why
the table is keyed on the pair.

```json
{
  "schema": 1, "stamp": "20260919-220011",
  "label": "Spark-4 stacked beside Nemotron",
  "model": { "id": "Qwen/Qwen3-Embedding-0.6B", "name": "Qwen3 Embedding 0.6B",
             "params_b": 0.6, "arch": "dense", "vision": false },
  "placement": { "node": "Spark-4-GX10", "gpu": "NVIDIA GB10", "gpus": 1, "tp": 1,
                 "port": 8001, "engine": "vllm", "ainode": "0.5.24",
                 "stacked_with": ["nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"] },
  "settings": {
    "endpoint": "http://100.72.9.84:8001/v1",
    "model_requested": "Qwen/Qwen3-Embedding-0.6B",
    "corpus": "embed-50", "corpus_version": 1, "latency_texts": 50,
    "batches": [1, 16, 64], "texts_per_batch": 64, "timeout_s": 60
  },
  "embed": {
    "endpoint": "http://100.72.9.84:8001/v1",
    "model_reported": "Qwen/Qwen3-Embedding-0.6B",
    "dimensions": 1024,
    "corpus": { "id": "embed-50", "version": 1, "latency_texts": 50,
                "pairs_id": "pairs-6", "pairs": 6, "related_pairs": 3,
                "source": "ainode/bench/embed/corpus.py" },
    "protocol": { "path": "POST /v1/embeddings", "endpoint": "...",
                  "model_requested": "...", "timeout_s": 60,
                  "body": "model and input only; no encoding_format, ..." },
    "latency": { "n": 50, "answered": 50, "errors": 0,
                 "p50_ms": 71.59, "p95_ms": 76.2, "min_ms": 68.94,
                 "max_ms": 236.71, "mean_ms": 75.93, "transport_floor_ms": 32.07 },
    "throughput": [
      { "batch": 64, "requests": 1, "texts": 64, "errors": 0, "seconds": 0.284,
        "texts_per_s": 225.48, "tokens": 834, "tokens_per_s": 2938.4 }
    ],
    "quality": {
      "pairs": [ { "id": "rel-1", "related": true, "a": "...", "b": "...",
                   "cosine": 0.938286 } ],
      "related_min": 0.82146, "unrelated_max": 0.320991,
      "margin": 0.500469, "ordered": true
    },
    "errors": [], "seconds": 8.6
  },
  "notes": ["..."],
  "source": "scripts/ainode-bench.py embed"
}
```

Rules specific to this block, all load-bearing:

- `dimensions` is read off the first answered response, never off a model card. It is
  what every index downstream has to be built for, and a checkpoint served through the
  wrong pooling mode has been known to return a different width than its card says.
- **`latency` is end to end from wherever the bench ran, and `transport_floor_ms` says
  how much of that was the wire.** The floor is the median of five `GET /v1/models`
  calls over the same link, a request that embeds nothing. At these latencies it is not
  a detail: a p50 of 71 ms with a 32 ms floor is a very different engine from a p50 of
  71 ms with a 2 ms floor, and the number alone cannot tell them apart. It is a
  measurement, not a correction: nothing is subtracted anywhere in the record.
- `latency` is one text per request, sent one at a time, so the percentiles describe a
  request that had the engine to itself rather than a queue this bench created. `n`
  counts the requests made and `answered` the ones that came back, so a percentile is
  never quietly taken over a smaller set than the header implies. Percentiles are
  interpolated, not nearest-rank.
- `throughput` is one row per batch size, each row over the **same** number of texts
  (`settings.texts_per_batch`), so the rows compare directly: 64 requests at batch 1,
  4 at batch 16, 1 at batch 64. The texts wrap around the corpus rather than repeating
  one string, which would measure the prefix cache instead of the engine. `seconds` is
  the wall time of the whole sweep at that size, gaps between requests included,
  because that is the time a caller waits.
- `tokens` and `tokens_per_s` come from `usage.prompt_tokens` as the engine reported
  it, never from a tokenizer run by the bench. A sweep where no response carried a
  usage block reports both as `null` rather than zero.
- **`quality` is a sanity check and the record says so in its notes.** `ordered` is
  true only when the LOWEST related pair's cosine is above the HIGHEST unrelated one's,
  which is a stronger statement than any per-pair threshold and encodes no number about
  the checkpoint; `margin` is the gap, negative when the check fails. Cosine is computed
  here rather than trusting the engine to have normalised. Six hand-written pairs say
  nothing about recall on a real corpus and this is not presented as a retrieval
  benchmark: it catches an engine returning well-formed vectors that mean nothing.
- A run that could not score every pair reports `ordered: null`, not `false`: nothing
  was measured, so nothing failed.
- `corpus.id` plus `corpus.version` say which texts produced these numbers. Two records
  are only comparable over the same pair, the same rule `decide.item_set` lives by.
- `errors` lists the requests that failed on transport or an unreadable response, and
  they are counted out of every percentile and rate in the block and named in the notes,
  because "the server refused" and "the vectors were bad" are different findings.
- No API key is ever in the record. Not in `settings`, not in `protocol`, not in a note.

## The `speech` block

A speech-bench run (`scripts/ainode-bench.py speech`) writes the same record with a
top-level `speech` block and **no `results` block**, for the reason the four above have
none: the model it measured takes audio in and is scored against words nobody typed at
it, so a tok/s figure would be meaningless even as a zero.
`scripts/render-bench-table.py` keeps a record shaped like that out of the README's
throughput table and gives it a row in the "Speech runs" table instead.

The `endpoint` is part of the measurement and not a footnote. The same instance
measured straight at its engine port and through an AINode node's `:3000/v1` are two
different numbers, because the second one includes the fleet routing hop, which for
this path also includes reading the model id out of a multipart body and forwarding the
bytes unchanged.

```json
{
  "schema": 1, "stamp": "20260921-013000",
  "label": "Spark-4 stacked beside Nemotron, via the fleet endpoint",
  "model": { "id": "openai/whisper-large-v3-turbo", "name": "Whisper Large v3 Turbo",
             "params_b": 0.81, "arch": "dense", "vision": false },
  "placement": { "node": "Spark-4-GX10", "gpu": "NVIDIA GB10", "gpus": 1, "tp": 1,
                 "port": 8002, "engine": "vllm", "ainode": "0.5.29",
                 "stacked_with": ["nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"] },
  "settings": {
    "endpoint": "http://100.122.26.9:3000/v1",
    "model_requested": "openai/whisper-large-v3-turbo",
    "path": "POST /v1/audio/transcriptions",
    "clips": "say-10", "clips_version": 1,
    "clips_directory": "bench/speech/clips", "clip_count": 10,
    "audio_seconds": 42.348, "language": "", "timeout_s": 120
  },
  "speech": {
    "endpoint": "http://100.122.26.9:3000/v1",
    "path": "POST /v1/audio/transcriptions",
    "model_reported": null,
    "warmup": { "clip": "clip-01", "wall_ms": 1404.58, "error": null,
                "note": "one untimed request, its transcript discarded: ..." },
    "clips": { "id": "say-10", "version": 1, "clips": 10,
               "voices": ["Daniel", "Karen", "Moira", "Rishi", "Samantha", "Tessa"],
               "locales": ["en_AU", "en_GB", "en_IE", "en_IN", "en_US", "en_ZA"],
               "reference_words": 129, "audio_seconds": 43.567,
               "sample_rates": [16000], "directory": "bench/speech/clips",
               "source": "ainode/bench/speech/clips.py" },
    "normalizer": { "id": "case-punct-numbers-1", "version": 1,
                    "folds": ["case", "unicode NFKC", "punctuation", "whitespace",
                              "ordinal suffixes", "number words to digits"],
                    "orthographic_folds": ["case", "unicode NFKC", "punctuation",
                                           "whitespace"],
                    "source": "ainode/bench/speech/metrics.py" },
    "protocol": { "path": "POST /v1/audio/transcriptions", "endpoint": "...",
                  "model_requested": "...",
                  "content_type": "multipart/form-data; boundary=...",
                  "response_format": "json",
                  "language": "not sent, so the engine detects it",
                  "timeout_s": 120, "body": "the model id and response_format ..." },
    "accuracy": { "clips": 10, "scored": 10, "reference_words": 129, "edits": 3,
                  "substitutions": 2, "deletions": 0, "insertions": 1,
                  "wer": 0.023256, "wer_orthographic": 0.045757,
                  "wer_per_clip_mean": 0.022424, "wer_max": 0.133333,
                  "clips_exact": 8 },
    "latency": { "n": 10, "answered": 10, "errors": 0,
                 "p50_ms": 739.33, "p95_ms": 961.2, "min_ms": 663.27,
                 "max_ms": 965.73, "mean_ms": 790.52, "transport_floor_ms": 28.78 },
    "rtf": { "scored": 10, "audio_seconds": 43.567, "wall_seconds": 7.905,
             "pooled": 0.1815, "p50": 0.1827, "min": 0.1265, "max": 0.2616 },
    "rows": [
      { "id": "clip-01", "voice": "Samantha", "locale": "en_US",
        "audio_seconds": 3.583, "reference": "The train from Austin ...",
        "transcript": "The train from Austin ...", "wall_ms": 655.1, "error": null,
        "reference_words": 12, "hypothesis_words": 12, "substitutions": 0,
        "deletions": 0, "insertions": 0, "edits": 0,
        "wer": 0.0, "wer_orthographic": 0.0, "rtf": 0.1829 }
    ],
    "errors": [], "seconds": 8.1
  },
  "notes": ["..."],
  "source": "scripts/ainode-bench.py speech"
}
```

Rules specific to this block, all load-bearing:

- **The reference is the text the clip was made from, fixed before the run.** It is the
  string handed to macOS `say` in `ainode/bench/speech/clips.py`, so it is exactly what
  was spoken, and nothing adjusts it after a transcript is seen. A reference edited to
  match what a model said would make the error rate a statement about the editor.
- **The audio is committed, not synthesised per run** (`bench/speech/clips/`, 1.3 MB for
  the ten). A word error rate is only comparable over the same bytes, so `clips.id` plus
  `clips.version` say which set produced these numbers and `CLIPS_VERSION` is bumped on
  any edit to a text, a voice or a file. `--generate-clips` rebuilds the set on a Mac and
  is a maintenance step, never part of a run.
- **`wer` is pooled over words, not averaged over clips.** Total edits over total
  reference words, which is the standard definition and the honest one: a mean of
  per-clip rates weights a four-word clip like a twenty-word one. The per-clip mean is
  carried beside it as `wer_per_clip_mean` for readers who want it, and `wer_max` is the
  worst single clip.
- **Two rates are reported and neither replaces the other.** `wer` uses the full
  normaliser (case, punctuation, whitespace, ordinal suffixes, number words folded to
  digits), because a transcript that heard every word and wrote "9" where the reference
  says "nine" is not a hearing error. `wer_orthographic` folds case, punctuation and
  whitespace only. The gap between them is how much of the error was spelling rather
  than hearing. `normalizer.id` and `normalizer.version` are in the record because the
  rate depends on them: a number taken under a different normaliser is a different
  number under the same name.
- **Nothing is normalised away that changes a word.** No stopword list, no stemming, no
  synonym map, and no per-clip exception.
- `substitutions`, `deletions` and `insertions` are kept apart because they are
  different findings: a model that drops the end of every clip and one that
  hallucinates a trailing sentence both score badly, and only the breakdown tells them
  apart. `edits` is their sum and the numerator of `wer`.
- **A clip that failed is one row with an `error` and nulls for every number**, counted
  out of every rate, percentile and factor, and named in the notes. It is never folded
  in as a 100 percent error rate: a transport failure inside a figure a reader takes as
  the model's is the one mistake this section can make. `accuracy.clips` counts the
  clips sent and `accuracy.scored` the ones that came back.
- **`latency` is one upload per request, sent one at a time**, so the percentiles
  describe a request that had the engine to itself rather than a queue this bench
  created. Percentiles are interpolated, not nearest-rank, the same choice the embedding
  block documents.
- **`warmup` is the one untimed request that came first, recorded rather than hidden.**
  A vLLM speech engine on `--enforce-eager` compiles its kernels on the FIRST real
  transcription: 89 seconds measured on a GB10 against 0.7 for every one after it.
  Leaving that inside the timed set would put a one-time compile in a p50 and in a
  real-time factor a reader takes as steady state; dropping it silently would hide a
  cost a user meets once per launch. So it is sent, its transcript discarded, and its
  wall time written here. A `warmup` carrying an `error` means the first timed clip may
  still hold that compile, and the notes say so.
- **`latency` and `rtf` are end to end from wherever the bench ran, and
  `transport_floor_ms` says how much of that was the wire.** The floor is the median of
  five `GET /v1/models` calls over the same link, a request that transcribes nothing. It
  is a measurement, not a correction: nothing is subtracted anywhere in the record.
- `rtf.pooled` is total wall over total audio, which is what a batch of clips costs;
  `p50` and `max` are the per-clip spread, which is what one caller waits. Below 1 means
  the engine transcribes faster than the clip plays. `audio_seconds` comes off each
  WAV's own header (frames over frame rate), never from a duration somebody typed.
- `rows` is in manifest order, one per clip, and carries both the reference and the
  transcript verbatim, because the transcript is the evidence behind the rate and a
  reader has to be able to see what differed.
- `path` says which of the two audio paths was measured. Whisper turbo is a
  transcription model and cannot translate, so a `translations` run belongs to an ASR
  model that can, and the two are never compared.
- No API key is ever in the record. Not in `settings`, not in `protocol`, not in a note.
