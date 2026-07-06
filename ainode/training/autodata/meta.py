"""AutoData v2.1/v2.2 — meta-optimizer.

Close the loop: run v1's Δ-filter batch, read the bucket stats, and have an LLM rewrite
the generation prompt P_t → P_t+1 toward the zone of proximal development (too_easy ⇒
harder, too_hard ⇒ easier). Repeat until the target objective or max_rounds.

Two reward objectives (`cfg.objective`, or the `objective=` arg):
- "yield"  (v2.1, default): the per-batch Δ=1 keep-rate PROXY. Cheap, but noisy/self-graded.
- "valset" (v2.2): the Evalchemy-style held-out-lift objective (see valset.py). The reward is
  the weak solver's verified-accuracy LIFT on a fixed labeled val set when primed with a
  few-shot sample of the round's kept traces — a stable, GT-graded signal that fixes the noisy
  proxy that motivated v2.2. The v2.1 path is untouched, so nothing regresses.

Reuses v1 (run / call_model / _retry). `core.call_model` is referenced dynamically so the
self-check's fake injection is seen here too. See demo() / demo_valset() at the bottom.
"""
from __future__ import annotations

import copy
import json
import re

from . import core
from .core import AutoDataConfig, run
from .valset import valset_items, valset_lift, evaluate


def _optimize_prompt(ep, rounds: list, task_spec: str, retries: int = 2,
                     objective: str = "yield") -> str:
    """Propose the NEXT generator prompt from the FULL round trajectory, so the optimizer
    learns from an overshoot (too_hard) instead of repeating it. too_easy = both solved
    (push harder); too_hard = both failed (dial back). The trajectory is framed around the
    active objective (yield proxy vs. held-out val-set lift)."""
    if objective == "valset":
        traj = "\n".join(
            f"Round {e['round']}: lift={e.get('lift')} "
            f"(val_acc {e.get('val_acc_baseline')}→{e.get('val_acc_primed')}) "
            f"kept={e['kept']}  | prompt: {e['prompt']}"
            for e in rounds)
        goal = (
            "We keep only tasks a STRONG model solves but a WEAK model FAILS (the zone of "
            "proximal development), then measure LIFT = the weak model's verified-accuracy "
            "gain on a held-out validation set when it is shown a few of your generated "
            "examples. Higher lift ⇒ your tasks teach. Trajectory so far:\n"
            f"{traj}\n\n"
            f"Task domain: {task_spec}\n"
            "Propose the NEXT generator prompt to MAXIMIZE lift — LEARN from the trajectory: "
            "keep what raised lift; if lift stalled or fell, shift the difficulty/coverage so "
            "the kept traces better cover what the weak model still gets wrong on the val set. ")
    else:
        traj = "\n".join(
            f"Round {e['round']}: yield={e['yield_pct']}% too_easy={e['too_easy']} "
            f"too_hard={e['too_hard']}  | prompt: {e['prompt']}"
            for e in rounds)
        goal = (
            "We keep only tasks a STRONG model solves but a WEAK model FAILS (the zone of "
            "proximal development). too_easy = both solved (too easy); too_hard = both failed "
            "(too hard). Trajectory so far:\n"
            f"{traj}\n\n"
            f"Task domain: {task_spec}\n"
            "Propose the NEXT generator prompt to MAXIMIZE yield — LEARN from the trajectory: "
            "if a prompt overshot to too_hard, dial difficulty back; if too_easy, push harder; "
            "converge toward the middle. ")
    msg = (
        "You tune the SYSTEM PROMPT of a task-generator in a synthetic-data pipeline. "
        + goal +
        'Return STRICT JSON only: {"prompt": "<the next generator system prompt>"}.'
    )

    def _opt():
        out = core.call_model(ep, [{"role": "user", "content": msg}], True)
        p = (json.loads(out).get("prompt") or "").strip()
        if not p:
            raise ValueError("empty rewritten prompt")
        return p
    return core._retry(_opt, retries)


def _key(ex: dict) -> str:
    for m in ex["conversations"]:
        if m["from"] == "human":
            return m["value"]
    return json.dumps(ex, sort_keys=True)


def meta_optimize(config, target_yield: float = 30, max_rounds: int = 4,
                  optimizer=None, on_round=None, objective=None, target=None) -> dict:
    """Iteratively rewrite P to raise the reward objective.

    objective: "yield" (v2.1 Δ=1 keep-rate proxy) or "valset" (v2.2 Evalchemy held-out lift).
    Defaults to cfg.objective. `target` overrides the stop-threshold (yield%% for "yield",
    absolute lift for "valset"; falls back to target_yield / cfg.val_target).

    Returns {best_prompt, best_yield, best_score, best_lift, dataset (merged+deduped Δ=1),
    rounds[], objective, out}. best_yield stays the best round's Δ=1 yield%% for both
    objectives (so existing callers keep working); best_score is the active-objective value.
    """
    cfg = config if isinstance(config, AutoDataConfig) else AutoDataConfig.from_dict(config)
    objective = objective or getattr(cfg, "objective", "yield") or "yield"
    is_valset = objective == "valset"
    opt_ep = optimizer or cfg.challenger          # reuse the challenger endpoint for the rewrite
    out_path = cfg.out
    cfg = copy.copy(cfg)
    cfg.out = ""                                    # we write the merged dataset ourselves

    val_items, val_baseline = [], None
    if is_valset:
        val_items = valset_items(cfg)
        if not val_items:
            raise ValueError("objective='valset' requires a non-empty val_set in the config")
        # weak-alone accuracy on the fixed val set — computed once, reused as the lift baseline
        val_baseline = evaluate(cfg, cfg.weak, val_items)["acc"]

    target_score = target if target is not None else (
        cfg.val_target if is_valset else target_yield)

    rounds, dataset, seen = [], [], set()
    best = {"score": float("-inf"), "yield": -1.0, "lift": None, "prompt": cfg.gen_prompt}
    for r in range(1, max_rounds + 1):
        result = run(cfg)
        rep = result["report"]
        for ex in result["kept"]:
            k = _key(ex)
            if k not in seen:
                seen.add(k)
                dataset.append(ex)
        entry = {"round": r, "yield_pct": rep["yield_pct"], "kept": rep["kept"],
                 "too_easy": rep["too_easy"], "too_hard": rep["too_hard"],
                 "total": rep["total"], "prompt": cfg.gen_prompt}
        if is_valset:
            # attribute the lift to THIS round's kept traces (few-shot from result["kept"])
            lr = valset_lift(cfg, result["kept"], valset=val_items, baseline=val_baseline)
            entry.update(lift=lr["lift"], val_acc_primed=lr["acc_primed"],
                         val_acc_baseline=lr["acc_baseline"], score=lr["lift"])
            score = lr["lift"]
        else:
            entry["score"] = rep["yield_pct"]
            score = rep["yield_pct"]
        rounds.append(entry)
        if score > best["score"]:
            best = {"score": score, "yield": rep["yield_pct"],
                    "lift": entry.get("lift"), "prompt": cfg.gen_prompt}
        if on_round:
            on_round(entry)
        if score >= target_score or r == max_rounds:
            break
        try:
            # feed the full trajectory so the optimizer corrects an overshoot, not repeats it
            cfg.gen_prompt = _optimize_prompt(opt_ep, rounds, cfg.task_spec, cfg.retries, objective)
        except Exception:
            break                                   # can't improve P → stop

    if out_path:
        with open(out_path, "w") as f:
            for ex in dataset:
                f.write(json.dumps(ex) + "\n")
    return {"best_prompt": best["prompt"], "best_yield": best["yield"],
            "best_score": best["score"], "best_lift": best["lift"],
            "dataset": dataset, "rounds": rounds, "objective": objective, "out": out_path}


def demo() -> None:
    """Self-check: a fake where harder prompts (DIFF=k) yield more Δ=1; the loop must
    raise yield round-over-round and converge to the harder prompt."""
    def fake(ep, messages, json_mode):
        sysmsg = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        last = messages[-1]["content"]
        if json_mode and "task-generator" in last:             # optimizer: bump difficulty
            ds = [int(x) for x in re.findall(r"DIFF=(\d+)", last)]
            return json.dumps({"prompt": f"DIFF={(max(ds) if ds else 0) + 1}"})
        if json_mode and "Generate" in last:                   # challenger: emit tasks at DIFF
            d = int((re.search(r"DIFF=(\d+)", sysmsg) or [0, "0"])[1])
            return json.dumps({"tasks": [{"input": f"L{d}_{i}", "reference": None} for i in range(6)]})
        if json_mode and "Candidate answer" in last:           # judge: CORRECT in the answer?
            return json.dumps({"correct": "CORRECT" in last})
        level = int((re.search(r"L(\d+)_", last) or [0, "0"])[1])  # solver
        ability = 0 if ep.model == "weak" else 2               # weak solves L0; strong solves ≤L2
        return "CORRECT" if level <= ability else "WRONG"

    core.call_model = fake
    try:
        cfg = AutoDataConfig(
            task_spec="x", gen_prompt="DIFF=0", n_tasks=6, concurrency=1,
            challenger=core.Endpoint("u", "challenger"), weak=core.Endpoint("u", "weak"),
            strong=core.Endpoint("u", "strong"), judge=core.Endpoint("u", "judge"))
        out = meta_optimize(cfg, target_yield=30, max_rounds=4)
        rounds = out["rounds"]
        assert len(rounds) >= 2, rounds
        assert rounds[0]["yield_pct"] == 0, rounds            # DIFF=0 → all too_easy
        assert rounds[-1]["yield_pct"] > rounds[0]["yield_pct"], rounds  # yield rose
        assert out["best_yield"] >= 30, out
        assert "DIFF=1" in out["best_prompt"], out            # converged to the harder prompt
        assert len(out["dataset"]) == out["rounds"][-1]["kept"], out
        print("autodata meta demo OK:", [(r["round"], r["yield_pct"]) for r in rounds])
    finally:
        core.call_model = core._http_chat


def demo_valset() -> None:
    """Self-check for the v2.2 val-set objective: a fake world where a higher TEACH level in
    the generator prompt makes the kept strong-traces carry a stronger HINT, which in turn
    lets the primed weak solver clear more held-out val probes. The meta-loop must optimize
    the held-out LIFT (not raw yield) and converge to the higher-TEACH prompt."""
    def fake(ep, messages, json_mode):
        sysmsg = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        last = messages[-1]["content"]
        if json_mode and "task-generator" in last:              # optimizer: bump TEACH
            ts = [int(x) for x in re.findall(r"TEACH=(\d+)", last)]
            return json.dumps({"prompt": f"TEACH={(max(ts) if ts else 0) + 1}"})
        if json_mode and "Generate" in last:                    # challenger: emit tasks at TEACH
            t = int((re.search(r"TEACH=(\d+)", sysmsg) or [0, "0"])[1])
            return json.dumps({"tasks": [{"input": f"K{t}_{i}", "reference": "1"} for i in range(4)]})
        # solver on a GENERATED task "K{t}_i": strong emits a good trace carrying HINT=t
        # (verifies to 1); weak fails (verifies to 0) -> Δ=1 kept.
        m = re.match(r"K(\d+)_", last)
        if m:
            t = int(m.group(1))
            return f"HINT={t}, answer = 1" if ep.model == "strong" else "answer = 0"
        # val probe "L{req}": the primed weak solver clears it iff a HINT>=req is in context
        if last and last[0] == "L" and last[1:].isdigit():
            req = int(last[1:])
            hints = [int(tok.split("=", 1)[1].rstrip(","))
                     for mm in messages for tok in str(mm.get("content", "")).split()
                     if tok.startswith("HINT=") and tok.split("=", 1)[1].rstrip(",").isdigit()]
            return "answer = 1" if (hints and max(hints) >= req) else "answer = 0"
        return "answer = 0"

    core.call_model = fake
    try:
        cfg = AutoDataConfig(
            task_spec="x", gen_prompt="TEACH=0", n_tasks=4, concurrency=1, judge_mode="verify",
            objective="valset", val_shots=4, val_target=0.9,
            challenger=core.Endpoint("u", "challenger"), weak=core.Endpoint("u", "weak"),
            strong=core.Endpoint("u", "strong"), judge=core.Endpoint("u", "judge"),
            val_set=[{"input": "L1", "reference": "1"}, {"input": "L2", "reference": "1"},
                     {"input": "L3", "reference": "1"}])
        out = meta_optimize(cfg, max_rounds=5)
        rounds = out["rounds"]
        assert out["objective"] == "valset", out
        assert rounds[0]["lift"] == 0.0, rounds            # TEACH=0 -> HINT=0 clears no probe
        assert rounds[-1]["lift"] > rounds[0]["lift"], rounds   # lift rose as TEACH climbed
        assert out["best_lift"] > 0 and out["best_score"] == out["best_lift"], out
        assert "TEACH=" in out["best_prompt"], out
        print("autodata meta valset demo OK:", [(r["round"], r["lift"]) for r in rounds])
    finally:
        core.call_model = core._http_chat


if __name__ == "__main__":
    demo()
    demo_valset()
