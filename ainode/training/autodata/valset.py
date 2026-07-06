"""AutoData v2.2 — Evalchemy-style val-set objective for the meta-loop.

THE HONEST FINDING (2026-06-28 handoff): live yield was noisy/low (best ~25%) and the root
cause is the EVAL SIGNAL, not the optimizer. The v2.1 reward is the per-batch Δ=1 keep-rate
— a self-graded proxy that (a) re-samples fresh tasks every round (so the number wobbles for
reasons unrelated to P) and (b) rides exact-match arithmetic, whose thin zone-of-proximal-
development mislabels correct-but-reformatted answers. Optimizing that proxy chases noise.

v2.2 replaces the proxy with a **held-out validation objective**, the Open-Thoughts/Evalchemy
move: score P against a FIXED, ground-truth-labeled val set with a trusted verifier, so the
reward is comparable round-over-round. The objective is *measured lift*:

    lift(P) = acc(weak | few-shot of D(P))  −  acc(weak alone)

on the val set, where D(P) is the batch of Δ=1 "teacher" traces the loop just kept. This is
in-context learning used as a **torch-free proxy for fine-tuning lift** — if the kept traces
actually teach, showing a few to the weak model raises its verified val accuracy; if the batch
is junk, lift is ~0. Because the val set is fixed and GT-labeled and graded by the same trusted
verifier the Δ-filter uses (`core.judge_correct` → verify.py value-verify + rubric fallback),
the signal is stable and directly optimizable, unlike raw keep-rate. Evalchemy grades served
vLLM OpenAI endpoints, so this drops onto AInode-served models with no new infra.

SIGNIFICANCE GUARD: a raw lift is a difference of two proportions on a finite val set, so on a
small set it is dominated by sampling noise — one probe flipping right on n=3 already moves lift
by 0.33, trivially clearing any modest target. Because baseline and primed accuracy are measured
on the SAME fixed items, the two runs are *paired*, so we grade the lift with an exact one-sided
McNemar (sign) test on the discordant probes (`mcnemar_pvalue`): b = probes that went wrong→right
when primed, c = right→wrong; under H0 b ~ Binomial(b+c, 0.5). `valset_lift` reports that p-value
and a `significant` flag (p ≤ cfg.alpha AND n ≥ cfg.val_min_n AND lift > 0). The meta-loop only
*accepts* a round (early-stops) on a `significant` lift, so it can no longer declare victory on
noise — the objective is stable AND its wins are statistically real, not just the max of an
order-statistic over rounds.

Pure HTTP + stdlib, like the rest of the package: `core.call_model` is referenced dynamically
so the self-check's fake injection is seen here too. See demo() at the bottom.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

from . import core
from .core import AutoDataConfig


def valset_items(cfg: AutoDataConfig) -> list:
    """The held-out probes: {"input","reference"?} dicts with a non-empty input."""
    return [v for v in (cfg.val_set or []) if isinstance(v, dict) and v.get("input")]


def _shots_to_messages(shots) -> list:
    """Flatten kept ShareGPT examples into alternating user/assistant few-shot turns.
    Each kept example is {"conversations":[{"from":"human"...},{"from":"gpt"...}]}."""
    msgs = []
    for ex in shots or []:
        human = gpt = None
        for m in ex.get("conversations", []):
            if m.get("from") == "human":
                human = m.get("value")
            elif m.get("from") == "gpt":
                gpt = m.get("value")
        if human is not None and gpt is not None:
            msgs.append({"role": "user", "content": human})
            msgs.append({"role": "assistant", "content": gpt})
    return msgs


def _solve_primed(cfg: AutoDataConfig, ep, x: str, shot_msgs: list) -> str:
    """Solve `x` with `ep`, optionally primed by pre-built few-shot turns."""
    msgs = ([{"role": "system", "content": cfg.system_prompt}] if cfg.system_prompt else [])
    msgs += shot_msgs
    msgs += [{"role": "user", "content": x}]
    return core._retry(lambda: core.call_model(ep, msgs, False), cfg.retries)


def evaluate(cfg: AutoDataConfig, ep, valset=None, shots=None) -> dict:
    """Verified accuracy of `ep` on the val set, optionally primed with `shots` (kept
    ShareGPT examples as few-shot context). Grading reuses `core.judge_correct`, so the
    val objective and the Δ-filter share the exact same trusted verifier.

    Returns {acc, correct, n, per_item} where `per_item` is the 0/1 correctness of each
    probe in val-set order — the paired vector the significance test consumes."""
    items = valset if valset is not None else valset_items(cfg)
    n = len(items)
    if n == 0:
        return {"acc": 0.0, "correct": 0, "n": 0, "per_item": []}
    shot_msgs = _shots_to_messages(shots)

    def _one(item):
        x, ref = item["input"], item.get("reference")
        try:
            out = _solve_primed(cfg, ep, x, shot_msgs)
        except Exception:
            return 0
        return int(bool(core.judge_correct(cfg, x, out, ref)))

    workers = max(1, min(cfg.concurrency, n))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        per_item = list(ex.map(_one, items))   # ex.map preserves item order -> paired vector
    correct = sum(per_item)
    return {"acc": round(correct / n, 4), "correct": int(correct), "n": n, "per_item": per_item}


def mcnemar_pvalue(baseline: list, primed: list) -> tuple:
    """Exact one-sided McNemar (sign) test that `primed` beats `baseline` on the SAME paired
    probes. `baseline`/`primed` are 0/1 vectors aligned by probe. b = probes that improved
    (wrong→right when primed), c = probes that regressed (right→wrong). Under H0 (priming has
    no effect) the discordant outcomes split 50/50, so b ~ Binomial(b+c, 0.5); the one-sided
    p-value is P(X ≥ b). No discordant pairs ⇒ no evidence of change ⇒ p=1.0. Stdlib only.
    Returns (p_value, b, c)."""
    b = sum(1 for base, prm in zip(baseline, primed) if not base and prm)
    c = sum(1 for base, prm in zip(baseline, primed) if base and not prm)
    disc = b + c
    if disc == 0:
        return 1.0, b, c
    tail = sum(math.comb(disc, i) for i in range(b, disc + 1)) / (2 ** disc)
    return round(min(1.0, tail), 4), b, c


def valset_lift(cfg: AutoDataConfig, kept, valset=None, baseline=None, shots=None) -> dict:
    """Evalchemy-style objective: the weak solver's verified-accuracy LIFT on the held-out
    val set when primed with a few-shot sample of `kept` (the round's Δ=1 teacher traces),
    vs the weak baseline. Higher lift ⇒ the batch teaches ⇒ a better generator prompt P.

    `baseline` may be a full `evaluate()` dict (carries `per_item`, so the paired significance
    test can run — the meta-loop passes this and computes it once, reused across rounds), a bare
    weak-alone acc float (back-compat), or None (evaluated fresh here).

    Returns {lift, acc_primed, acc_baseline, n, shots, p_value, improved, regressed, significant}.
    `p_value` is the exact one-sided McNemar p for "primed beats baseline" (None when no paired
    baseline vector is available); `significant` is the guard the meta-loop early-stops on —
    a real, non-noise win: p ≤ cfg.alpha AND n ≥ cfg.val_min_n AND lift > 0.
    """
    items = valset if valset is not None else valset_items(cfg)
    if not items:
        return {"lift": 0.0, "acc_primed": 0.0, "acc_baseline": 0.0, "n": 0, "shots": 0,
                "p_value": None, "improved": 0, "regressed": 0, "significant": False}
    k = cfg.val_shots if shots is None else shots
    sample = list(kept or [])[:k]
    if isinstance(baseline, dict):                 # full evaluate() dict -> enables McNemar
        base_eval = baseline
    elif baseline is None:
        base_eval = evaluate(cfg, cfg.weak, items)
    else:                                          # bare acc float (back-compat) -> no per_item
        base_eval = {"acc": float(baseline), "per_item": None}
    base_acc = base_eval["acc"]
    primed_eval = evaluate(cfg, cfg.weak, items, shots=sample)
    primed = primed_eval["acc"]
    lift = round(primed - base_acc, 4)

    base_items, primed_items = base_eval.get("per_item"), primed_eval.get("per_item")
    if base_items and primed_items and len(base_items) == len(primed_items):
        p_value, improved, regressed = mcnemar_pvalue(base_items, primed_items)
    else:                                          # no paired baseline -> can't confirm significance
        p_value, improved, regressed = None, 0, 0
    n = len(items)
    significant = (p_value is not None and p_value <= cfg.alpha
                   and n >= cfg.val_min_n and lift > 0)
    return {"lift": lift, "acc_primed": primed, "acc_baseline": round(base_acc, 4),
            "n": n, "shots": len(sample), "p_value": p_value,
            "improved": improved, "regressed": regressed, "significant": significant}


def demo() -> None:
    """Self-check with a fake model: the kept traces carry a hint that lets the weak solver
    answer the held-out val probes it fails cold — so priming lifts verified val accuracy."""
    import json  # noqa: F401 — parity with sibling demos; kept explicit

    def fake(ep, messages, json_mode):
        last = messages[-1]["content"]
        # val probe "L{req}" needs a hint of level >= req to be solved by the weak model.
        # scan the few-shot context (prior assistant turns) for the strongest HINT=k available.
        req = None
        if last and last[0] == "L" and last[1:].isdigit():
            req = int(last[1:])
        if req is not None:
            hints = []
            for m in messages:
                for tok in str(m.get("content", "")).split():
                    if tok.startswith("HINT="):
                        try:
                            hints.append(int(tok.split("=", 1)[1].rstrip(",")))
                        except ValueError:
                            pass
            return "answer = 1" if (hints and max(hints) >= req) else "answer = 0"
        return "answer = 0"

    core.call_model = fake
    try:
        cfg = AutoDataConfig(
            task_spec="x", gen_prompt="x", n_tasks=1, concurrency=1, judge_mode="verify",
            challenger=core.Endpoint("u", "challenger"), weak=core.Endpoint("u", "weak"),
            strong=core.Endpoint("u", "strong"), judge=core.Endpoint("u", "judge"),
            val_set=[{"input": "L1", "reference": "1"}, {"input": "L2", "reference": "1"}],
            val_shots=2)
        # cold baseline: weak has no hint -> both probes wrong
        base = evaluate(cfg, cfg.weak, valset_items(cfg))
        assert base["acc"] == 0.0, base
        # a teaching batch: two kept traces carrying HINT=2 -> both probes solved when primed
        kept = [{"conversations": [{"from": "human", "value": "K"},
                                   {"from": "gpt", "value": "HINT=2, answer = 1"}]}] * 2
        res = valset_lift(cfg, kept, baseline=base["acc"])
        assert res["acc_baseline"] == 0.0 and res["acc_primed"] == 1.0, res
        assert res["lift"] == 1.0 and res["shots"] == 2, res
        # a junk batch (no usable hint) teaches nothing -> zero lift
        junk = [{"conversations": [{"from": "human", "value": "K"},
                                   {"from": "gpt", "value": "HINT=0, answer = 1"}]}] * 2
        res0 = valset_lift(cfg, junk, baseline=base["acc"])
        assert res0["lift"] == 0.0, res0
        print("autodata valset demo OK: teaching lift", res["lift"], "vs junk", res0["lift"])
    finally:
        core.call_model = core._http_chat


if __name__ == "__main__":
    demo()
