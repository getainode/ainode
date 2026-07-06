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

Pure HTTP + stdlib, like the rest of the package: `core.call_model` is referenced dynamically
so the self-check's fake injection is seen here too. See demo() at the bottom.
"""
from __future__ import annotations

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
    val objective and the Δ-filter share the exact same trusted verifier."""
    items = valset if valset is not None else valset_items(cfg)
    n = len(items)
    if n == 0:
        return {"acc": 0.0, "correct": 0, "n": 0}
    shot_msgs = _shots_to_messages(shots)

    def _one(item):
        x, ref = item["input"], item.get("reference")
        try:
            out = _solve_primed(cfg, ep, x, shot_msgs)
        except Exception:
            return 0
        return core.judge_correct(cfg, x, out, ref)

    workers = max(1, min(cfg.concurrency, n))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        correct = sum(ex.map(_one, items))
    return {"acc": round(correct / n, 4), "correct": int(correct), "n": n}


def valset_lift(cfg: AutoDataConfig, kept, valset=None, baseline=None, shots=None) -> dict:
    """Evalchemy-style objective: the weak solver's verified-accuracy LIFT on the held-out
    val set when primed with a few-shot sample of `kept` (the round's Δ=1 teacher traces),
    vs the weak baseline. Higher lift ⇒ the batch teaches ⇒ a better generator prompt P.

    Returns {lift, acc_primed, acc_baseline, n, shots}. `baseline` (weak-alone acc) can be
    passed in to compute it once and reuse it across meta rounds.
    """
    items = valset if valset is not None else valset_items(cfg)
    if not items:
        return {"lift": 0.0, "acc_primed": 0.0, "acc_baseline": 0.0, "n": 0, "shots": 0}
    k = cfg.val_shots if shots is None else shots
    sample = list(kept or [])[:k]
    base = baseline if baseline is not None else evaluate(cfg, cfg.weak, items)["acc"]
    primed = evaluate(cfg, cfg.weak, items, shots=sample)["acc"]
    return {"lift": round(primed - base, 4), "acc_primed": primed,
            "acc_baseline": round(base, 4), "n": len(items), "shots": len(sample)}


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
