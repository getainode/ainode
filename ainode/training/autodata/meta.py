"""AutoData v2.1 — meta-optimizer.

Close the loop: run v1's Δ-filter batch, read the bucket stats, and have an LLM rewrite
the generation prompt P_t → P_t+1 toward the zone of proximal development (too_easy ⇒
harder, too_hard ⇒ easier). Repeat until target yield or max_rounds. Optimizes on the
Δ=1-yield PROXY only — the val-set objective (Evalchemy) is v2.2.

Reuses v1 (run / call_model / _retry). `core.call_model` is referenced dynamically so the
self-check's fake injection is seen here too. See demo() at the bottom.
"""
from __future__ import annotations

import copy
import json
import re

from . import core
from .core import AutoDataConfig, run


def _optimize_prompt(ep, current_prompt: str, task_spec: str, report: dict, retries: int = 2) -> str:
    """Ask an LLM to rewrite the generator prompt toward the ZPD, given last round's stats."""
    stats = (f"yield={report.get('yield_pct', 0)}% kept={report['kept']} "
             f"too_easy={report['too_easy']} (both strong+weak solved) "
             f"too_hard={report['too_hard']} (both failed) total={report['total']}")
    msg = (
        "You tune the SYSTEM PROMPT of a task-generator in a synthetic-data pipeline. "
        "We keep only tasks a STRONG model solves but a WEAK model fails (the zone of "
        "proximal development). Current generator prompt:\n"
        f"---\n{current_prompt}\n---\n"
        f"Task domain: {task_spec}\n"
        f"Last round: {stats}\n"
        "If too_easy dominates, make generated tasks HARDER / more multi-step so the weak "
        "model fails. If too_hard dominates, make them EASIER / clearer so the strong model "
        "still solves. Maximize the strong-solves-weak-fails zone. "
        'Return STRICT JSON only: {"prompt": "<the rewritten generator system prompt>"}.'
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
                  optimizer=None, on_round=None) -> dict:
    """Iteratively rewrite P to raise Δ=1 yield.

    Returns {best_prompt, best_yield, dataset (merged+deduped Δ=1), rounds[], out}.
    """
    cfg = config if isinstance(config, AutoDataConfig) else AutoDataConfig.from_dict(config)
    opt_ep = optimizer or cfg.challenger          # reuse the challenger endpoint for the rewrite
    out_path = cfg.out
    cfg = copy.copy(cfg)
    cfg.out = ""                                    # we write the merged dataset ourselves

    rounds, dataset, seen = [], [], set()
    best = {"yield": -1.0, "prompt": cfg.gen_prompt}
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
        rounds.append(entry)
        if rep["yield_pct"] > best["yield"]:
            best = {"yield": rep["yield_pct"], "prompt": cfg.gen_prompt}
        if on_round:
            on_round(entry)
        if rep["yield_pct"] >= target_yield or r == max_rounds:
            break
        try:
            cfg.gen_prompt = _optimize_prompt(opt_ep, cfg.gen_prompt, cfg.task_spec, rep, cfg.retries)
        except Exception:
            break                                   # can't improve P → stop

    if out_path:
        with open(out_path, "w") as f:
            for ex in dataset:
                f.write(json.dumps(ex) + "\n")
    return {"best_prompt": best["prompt"], "best_yield": best["yield"],
            "dataset": dataset, "rounds": rounds, "out": out_path}


def demo() -> None:
    """Self-check: a fake where harder prompts (DIFF=k) yield more Δ=1; the loop must
    raise yield round-over-round and converge to the harder prompt."""
    def fake(ep, messages, json_mode):
        sysmsg = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        last = messages[-1]["content"]
        if json_mode and "rewritten generator" in last:        # optimizer: bump difficulty
            d = int((re.search(r"DIFF=(\d+)", last) or [0, "0"])[1])
            return json.dumps({"prompt": f"DIFF={d + 1}"})
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


if __name__ == "__main__":
    demo()
