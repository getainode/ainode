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
