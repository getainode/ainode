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
