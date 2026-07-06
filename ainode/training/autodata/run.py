"""AutoData CLI: `python -m ainode.training.autodata.run --config cfg.json`.

Config JSON keys: task_spec, gen_prompt, challenger/weak/strong/judge (each
{url, model, max_tokens?, temperature?, api_key?}), n_tasks?, system_prompt?,
judge_mode? (rubric|verify|exact), concurrency?, out?. For the v2.2 meta objective
also: objective? (yield|valset), val_set? ([{input, reference}]), val_shots?, val_target?.
"""
import argparse
import json
import sys

from .core import run


def main() -> None:
    ap = argparse.ArgumentParser(description="AutoData — Δ-filtered synthetic data generation")
    ap.add_argument("--config", required=True, help="Path to AutoData config JSON")
    ap.add_argument("--meta", action="store_true", help="v2.1/v2.2 meta-optimizer: rewrite P each round")
    ap.add_argument("--objective", choices=("yield", "valset"), default=None,
                    help="meta reward: yield (v2.1 Δ=1 proxy) or valset (v2.2 held-out lift); default from config")
    ap.add_argument("--target-yield", type=float, default=30, help="meta/yield: stop when yield%% reaches this")
    ap.add_argument("--target", type=float, default=None,
                    help="meta: objective stop-threshold (yield%% or absolute lift); overrides --target-yield / val_target")
    ap.add_argument("--max-rounds", type=int, default=4, help="meta: max optimization rounds")
    args = ap.parse_args()

    cfg = json.loads(open(args.config).read())

    if args.meta:
        from .meta import meta_optimize
        objective = args.objective or cfg.get("objective", "yield")
        is_valset = objective == "valset"

        def _on_round(e):
            if is_valset:
                print(f"  round {e['round']}: lift={e['lift']} "
                      f"(val_acc {e['val_acc_baseline']}→{e['val_acc_primed']} "
                      f"kept={e['kept']} yield={e['yield_pct']}%)", file=sys.stderr)
            else:
                print(f"  round {e['round']}: yield={e['yield_pct']}% "
                      f"(kept={e['kept']} too_easy={e['too_easy']} too_hard={e['too_hard']})",
                      file=sys.stderr)

        out = meta_optimize(cfg, target_yield=args.target_yield, max_rounds=args.max_rounds,
                            objective=objective, target=args.target, on_round=_on_round)
        keys = ("round", "yield_pct", "kept") + (("lift",) if is_valset else ())
        print(json.dumps({"objective": out["objective"], "best_score": out["best_score"],
                          "best_yield": out["best_yield"], "best_lift": out["best_lift"],
                          "best_prompt": out["best_prompt"],
                          "rounds": [{k: r.get(k) for k in keys} for r in out["rounds"]],
                          "dataset_size": len(out["dataset"]), "out": out["out"]}, indent=2))
        summary = (f"Best lift {out['best_lift']}" if is_valset
                   else f"Best yield {out['best_yield']}%")
        print(f"\n{summary} over {len(out['rounds'])} rounds; "
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
