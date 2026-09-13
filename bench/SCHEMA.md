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
