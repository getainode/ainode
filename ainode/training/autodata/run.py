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
    ap.add_argument("--meta", action="store_true", help="v2.1 meta-optimizer: rewrite P each round to raise yield")
    ap.add_argument("--target-yield", type=float, default=30, help="meta: stop when yield%% reaches this")
    ap.add_argument("--max-rounds", type=int, default=4, help="meta: max optimization rounds")
    args = ap.parse_args()

    cfg = json.loads(open(args.config).read())

    if args.meta:
        from .meta import meta_optimize
        out = meta_optimize(cfg, target_yield=args.target_yield, max_rounds=args.max_rounds,
                            on_round=lambda e: print(
                                f"  round {e['round']}: yield={e['yield_pct']}% "
                                f"(kept={e['kept']} too_easy={e['too_easy']} too_hard={e['too_hard']})",
                                file=sys.stderr))
        print(json.dumps({"best_yield": out["best_yield"], "best_prompt": out["best_prompt"],
                          "rounds": [{k: r[k] for k in ("round", "yield_pct", "kept")} for r in out["rounds"]],
                          "dataset_size": len(out["dataset"]), "out": out["out"]}, indent=2))
        print(f"\nBest yield {out['best_yield']}% over {len(out['rounds'])} rounds; "
              f"{len(out['dataset'])} examples -> {out['out'] or '(not written)'}", file=sys.stderr)
        return
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
