"""AutoData CLI: `python -m ainode.training.autodata.run --config cfg.json`.

Config JSON keys: task_spec, gen_prompt, challenger/weak/strong/judge (each
{url, model, max_tokens?, temperature?, api_key?}), n_tasks?, system_prompt?,
judge_mode? (rubric|exact), concurrency?, out?.
"""
import argparse
import json
import sys

from .core import run


def main() -> None:
    ap = argparse.ArgumentParser(description="AutoData — Δ-filtered synthetic data generation")
    ap.add_argument("--config", required=True, help="Path to AutoData config JSON")
    args = ap.parse_args()

    cfg = json.loads(open(args.config).read())
    result = run(cfg, on_progress=lambda r: print(
        f"\r  kept={r['kept']} too_easy={r['too_easy']} too_hard={r['too_hard']} "
        f"strong_worse={r['strong_worse']} err={r['errors']}", end="", file=sys.stderr))
    print(file=sys.stderr)
    rep = result["report"]
    print(json.dumps({"report": rep, "out": result["out"]}, indent=2))
    print(f"\nKept {rep['kept']}/{rep['total']} ({rep['yield_pct']}% yield) -> {result['out'] or '(not written)'}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
