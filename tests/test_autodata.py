"""Tests for ainode.training.autodata — the self-check demos + the loop-closing route.

The demos assert internally (they raise on failure), so collecting them here puts the
Δ-filter and meta-optimizer logic under CI. The route test proves the whole loop:
run -> registered dataset -> dataset_id accepted by a training job.
"""

import asyncio
import json

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from ainode.training.autodata import core, meta, valset
from ainode.training.autodata.core import AutoDataConfig, Endpoint, run
from ainode.training.autodata.valset import evaluate, valset_items, valset_lift
from ainode.training.autodata.verify import is_correct
from ainode.training.engine import TrainingManager
from ainode.training.api_routes import setup_training_routes
from ainode.datasets.manager import DatasetManager
from ainode.datasets.api_routes import setup_dataset_routes


def test_core_demo():
    core.demo()  # asserts internally: only the strong-solves/weak-fails task survives


def test_verify_demo():
    from ainode.training.autodata import verify
    verify.demo()  # asserts internally: equivalence the old substring check failed


# Answers that are CORRECT but reformatted — the verifier must grade every one right.
@pytest.mark.parametrize("output, reference", [
    ("1/2", "0.5"),
    ("0.5", "1/2"),
    ("x = 2", "2"),
    ("  2  ", "2"),
    (r"\frac{3}{4}", "0.75"),
    ("$0.5$", "1/2"),
    ("50%", "0.5"),
    ("1,000", "1000"),
])
def test_verify_grades_reformatted_answers(output, reference):
    assert is_correct(output, reference) is True


# The subset where the reference digits are NOT a literal substring of the output — exact
# substring-match genuinely MISgrades these (the thin-ZPD root cause); the verifier fixes them.
@pytest.mark.parametrize("output, reference", [
    ("1/2", "0.5"),
    (r"\frac{3}{4}", "0.75"),
    ("$0.5$", "1/2"),
    ("50%", "0.5"),
    ("1,000", "1000"),
])
def test_verify_beats_exact_on_true_conversions(output, reference):
    def _norm(s):
        return " ".join(str(s).lower().split())
    assert _norm(reference) not in _norm(output)   # exact-match would mark this WRONG
    assert is_correct(output, reference) is True    # ...the verifier marks it RIGHT


def test_verify_marks_genuinely_wrong_and_defers_open_ended():
    assert is_correct("7", "8") is False           # different numbers -> wrong
    assert is_correct("paris", None) is None       # no reference -> rubric
    assert is_correct(r"\sqrt{4}", "2") is None    # symbolic -> rubric (ceiling, no false-neg)


def test_signal_shift_exact_vs_verify(capsys):
    """Re-measurement: SAME tasks, exact vs verify. Strong returns the right value in a
    different format, weak is wrong -> exact buries them in too_hard; verify recovers Δ=1."""
    def fake(ep, messages, json_mode):
        last = messages[-1]["content"]
        if json_mode and "Generate" in last:
            return json.dumps({"tasks": [
                {"input": "Q1", "reference": "0.5"},
                {"input": "Q2", "reference": "0.75"},
                {"input": "Q3", "reference": "0.25"}]})
        if json_mode and "Candidate answer" in last:       # rubric (unused for clean numerics)
            return json.dumps({"correct": False})
        if ep.model != "strong":
            return "0"                                      # weak: wrong
        # right value, different format — reference digits are NOT a substring of these
        return {"Q1": "1/2", "Q2": r"\frac{3}{4}", "Q3": "1/4"}.get(last, "0")

    core.call_model = fake
    try:
        base = dict(task_spec="x", gen_prompt="x", n_tasks=3, concurrency=1,
                    challenger=Endpoint("u", "challenger"), weak=Endpoint("u", "weak"),
                    strong=Endpoint("u", "strong"), judge=Endpoint("u", "judge"))
        exact = run(AutoDataConfig(judge_mode="exact", **base))["report"]
        ver = run(AutoDataConfig(judge_mode="verify", **base))["report"]
    finally:
        core.call_model = core._http_chat

    print(f"\nsignal shift  exact: kept={exact['kept']} too_hard={exact['too_hard']}"
          f"  ->  verify: kept={ver['kept']} too_hard={ver['too_hard']}")
    assert exact["kept"] == 0 and exact["too_hard"] == 3   # format variants buried as too-hard
    assert ver["kept"] == 3 and ver["too_hard"] == 0       # verifier recovers all three
    assert ver["kept"] > exact["kept"] and ver["too_hard"] < exact["too_hard"]


def test_meta_demo():
    meta.demo()  # asserts internally: yield rises round-over-round to the harder prompt


# --- v2.2: Evalchemy-style val-set objective -------------------------------------------

def test_valset_demo():
    valset.demo()  # asserts internally: teaching traces lift primed val-accuracy, junk doesn't


def test_meta_valset_demo():
    meta.demo_valset()  # asserts internally: the meta-loop optimizes held-out lift, not yield


def _valset_cfg(**over):
    base = dict(task_spec="x", gen_prompt="x", n_tasks=1, concurrency=1, judge_mode="verify",
                challenger=Endpoint("u", "challenger"), weak=Endpoint("u", "weak"),
                strong=Endpoint("u", "strong"), judge=Endpoint("u", "judge"),
                val_set=[{"input": "L1", "reference": "1"}, {"input": "L2", "reference": "1"}],
                val_shots=2)
    base.update(over)
    return AutoDataConfig(**base)


def _fake_hint(ep, messages, json_mode):
    """weak val solver: clears probe L{req} iff a few-shot HINT>=req is in context."""
    last = messages[-1]["content"]
    if last and last[0] == "L" and last[1:].isdigit():
        req = int(last[1:])
        hints = [int(t.split("=", 1)[1].rstrip(","))
                 for m in messages for t in str(m.get("content", "")).split()
                 if t.startswith("HINT=") and t.split("=", 1)[1].rstrip(",").isdigit()]
        return "answer = 1" if (hints and max(hints) >= req) else "answer = 0"
    return "answer = 0"


def test_valset_evaluate_is_gt_graded():
    """Verified accuracy uses the trusted verifier (verify-mode), not substring — the weak
    solver clears both probes only when the teaching hint is in the few-shot context."""
    core.call_model = _fake_hint
    try:
        cfg = _valset_cfg()
        items = valset_items(cfg)
        assert evaluate(cfg, cfg.weak, items)["acc"] == 0.0            # cold: no hint -> 0/2
        shots = [{"conversations": [{"from": "human", "value": "K"},
                                    {"from": "gpt", "value": "HINT=2, answer = 1"}]}] * 2
        assert evaluate(cfg, cfg.weak, items, shots=shots)["acc"] == 1.0  # primed -> 2/2
    finally:
        core.call_model = core._http_chat


def test_valset_lift_teaching_vs_junk():
    """The objective separates a teaching batch (positive lift) from a junk batch (zero lift)."""
    core.call_model = _fake_hint
    try:
        cfg = _valset_cfg()
        teach = [{"conversations": [{"from": "human", "value": "K"},
                                    {"from": "gpt", "value": "HINT=2, answer = 1"}]}] * 2
        junk = [{"conversations": [{"from": "human", "value": "K"},
                                   {"from": "gpt", "value": "HINT=0, answer = 1"}]}] * 2
        assert valset_lift(cfg, teach)["lift"] == 1.0
        assert valset_lift(cfg, junk)["lift"] == 0.0
        # empty val_set -> objective degrades to a zero, never raises
        assert valset_lift(_valset_cfg(val_set=[]), teach)["lift"] == 0.0
    finally:
        core.call_model = core._http_chat


def test_meta_valset_requires_val_set():
    """objective='valset' with no probes is a config error, surfaced clearly (not silent)."""
    with pytest.raises(ValueError):
        meta.meta_optimize(_valset_cfg(val_set=[], objective="valset"), max_rounds=1)


# --- v2.2 significance guard: a raw lift on a small val set is sampling noise -----------

def test_mcnemar_pvalue_exact():
    """Exact one-sided McNemar (sign) test on paired 0/1 probe vectors, stdlib-only."""
    from ainode.training.autodata.valset import mcnemar_pvalue
    assert mcnemar_pvalue([0, 0, 0, 0, 0], [1, 1, 1, 1, 1]) == (0.0312, 5, 0)  # 0.5**5
    assert mcnemar_pvalue([1, 0, 1], [1, 0, 1]) == (1.0, 0, 0)                 # no discordant pairs
    assert mcnemar_pvalue([0, 1], [1, 0]) == (0.75, 1, 1)                      # 1 up, 1 down
    assert mcnemar_pvalue([0, 0, 0], [1, 1, 1]) == (0.125, 3, 0)              # n=3 all-flip != sig


def test_valset_lift_reports_significance_but_small_n_is_never_significant():
    """A full teaching lift on a 2-probe val set is real-looking (lift=1.0) but the paired test
    on n < val_min_n can't call it significant — the guard the meta-loop reads. A bare-float
    baseline carries no paired vector, so significance is conservatively withheld (None)."""
    core.call_model = _fake_hint
    try:
        cfg = _valset_cfg()  # 2-probe val set, val_shots=2
        teach = [{"conversations": [{"from": "human", "value": "K"},
                                    {"from": "gpt", "value": "HINT=2, answer = 1"}]}] * 2
        base = evaluate(cfg, cfg.weak, valset_items(cfg))          # full dict -> paired per_item
        lr = valset_lift(cfg, teach, baseline=base)
        assert lr["lift"] == 1.0 and lr["improved"] == 2 and lr["regressed"] == 0
        assert lr["p_value"] is not None and lr["significant"] is False   # n=2 < val_min_n
        lr_float = valset_lift(cfg, teach, baseline=base["acc"])   # bare float -> no per_item
        assert lr_float["p_value"] is None and lr_float["significant"] is False
    finally:
        core.call_model = core._http_chat


def _fake_teach_all(ep, messages, json_mode):
    """Every round teaches maximally: the strong solver emits HINT=9 (clears any L-probe) and
    the weak solver fails cold, so baseline acc=0 and primed acc=1.0 — every val probe is a
    wrong→right improvement. Lets a test dial n up/down and watch the significance guard flip."""
    last = messages[-1]["content"]
    if json_mode and "task-generator" in last:                # optimizer: no-op rewrite
        return json.dumps({"prompt": "x"})
    if json_mode and "Generate" in last:                      # challenger: 4 Δ=1-able tasks
        return json.dumps({"tasks": [{"input": f"K{i}", "reference": "1"} for i in range(4)]})
    if json_mode and "Candidate answer" in last:              # rubric judge (unused for numerics)
        return json.dumps({"correct": True})
    if last and last[0] == "K":                               # solver on a generated task
        return "HINT=9, answer = 1" if ep.model == "strong" else "answer = 0"
    if last and last[0] == "L" and last[1:].isdigit():        # weak val solver, primed via HINT
        req = int(last[1:])
        hints = [int(t.split("=", 1)[1].rstrip(","))
                 for m in messages for t in str(m.get("content", "")).split()
                 if t.startswith("HINT=") and t.split("=", 1)[1].rstrip(",").isdigit()]
        return "answer = 1" if (hints and max(hints) >= req) else "answer = 0"
    return "answer = 0"


def test_meta_valset_significance_blocks_noise():
    """The finding's repro: a 3-probe val set with a trivially-cleared target (0.1). Round 1's
    lift hits 1.0, but n < val_min_n and the paired McNemar p (0.125) can't clear alpha, so the
    loop must NOT early-stop — it keeps running instead of declaring victory on sampling noise."""
    core.call_model = _fake_teach_all
    try:
        cfg = _valset_cfg(objective="valset", n_tasks=4, val_shots=4, val_target=0.1,
                          val_set=[{"input": "L1", "reference": "1"}] * 3)
        out = meta.meta_optimize(cfg, max_rounds=3)
        assert len(out["rounds"]) == 3, out                     # ran every round; no early victory
        r0 = out["rounds"][0]
        assert r0["lift"] >= cfg.val_target                     # target trivially cleared...
        assert r0["p_value"] is not None and r0["significant"] is False   # ...but not significant
        assert out["best_significant"] is False
    finally:
        core.call_model = core._http_chat


def test_meta_valset_significant_lift_stops():
    """The mirror: with a 20-probe val set, the same all-probes improvement clears McNemar
    (p≈0, n≥val_min_n), so the loop accepts the round and early-stops — the objective still
    WORKS when the signal is genuinely real, it just no longer trusts noise."""
    core.call_model = _fake_teach_all
    try:
        cfg = _valset_cfg(objective="valset", n_tasks=4, val_shots=4, val_target=0.1,
                          val_set=[{"input": "L1", "reference": "1"}] * 20)
        out = meta.meta_optimize(cfg, max_rounds=3)
        assert len(out["rounds"]) == 1, out                     # accepted -> stopped after round 1
        r0 = out["rounds"][0]
        assert r0["significant"] is True and r0["p_value"] <= cfg.alpha
        assert out["best_significant"] is True
    finally:
        core.call_model = core._http_chat


def test_meta_yield_path_unchanged_by_v22():
    """The v2.1 yield objective still returns its keys and optimizes yield — no regression."""
    def fake(ep, messages, json_mode):
        sysmsg = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        last = messages[-1]["content"]
        if json_mode and "task-generator" in last:
            ds = [int(x) for x in __import__("re").findall(r"DIFF=(\d+)", last)]
            return json.dumps({"prompt": f"DIFF={(max(ds) if ds else 0) + 1}"})
        if json_mode and "Generate" in last:
            d = int((__import__("re").search(r"DIFF=(\d+)", sysmsg) or [0, "0"])[1])
            return json.dumps({"tasks": [{"input": f"L{d}_{i}", "reference": None} for i in range(6)]})
        if json_mode and "Candidate answer" in last:
            return json.dumps({"correct": "CORRECT" in last})
        level = int((__import__("re").search(r"L(\d+)_", last) or [0, "0"])[1])
        ability = 0 if ep.model == "weak" else 2
        return "CORRECT" if level <= ability else "WRONG"

    core.call_model = fake
    try:
        cfg = AutoDataConfig(
            task_spec="x", gen_prompt="DIFF=0", n_tasks=6, concurrency=1,
            challenger=Endpoint("u", "challenger"), weak=Endpoint("u", "weak"),
            strong=Endpoint("u", "strong"), judge=Endpoint("u", "judge"))
        out = meta.meta_optimize(cfg, target_yield=30, max_rounds=4)
        assert out["objective"] == "yield"
        assert out["best_score"] == out["best_yield"] and out["best_lift"] is None
        assert out["rounds"][-1]["yield_pct"] > out["rounds"][0]["yield_pct"]
    finally:
        core.call_model = core._http_chat


def _fake(ep, messages, json_mode):
    """3 tasks: A (only strong solves -> kept), B (both solve), C (neither)."""
    last = messages[-1]["content"]
    if json_mode and "Generate" in last:                       # challenger
        return json.dumps({"tasks": [
            {"input": "TASK_A", "reference": "A"},
            {"input": "TASK_B", "reference": "B"},
            {"input": "TASK_C", "reference": "C"}]})
    if json_mode and "Candidate answer" in last:               # judge
        return json.dumps({"correct": "WRONG" not in last})
    strong = ep.model == "strong"                              # solver
    if last == "TASK_A":
        return "A" if strong else "WRONG"
    if last == "TASK_B":
        return "B"
    return "WRONG"                                             # TASK_C: both wrong


@pytest_asyncio.fixture
async def client(tmp_path):
    app = web.Application()
    dsm = DatasetManager(root=tmp_path / "datasets")
    app["dataset_manager"] = dsm
    setup_training_routes(app, TrainingManager(dataset_manager=dsm))
    setup_dataset_routes(app, dsm)
    async with TestClient(TestServer(app)) as c:
        yield c, dsm


@pytest.mark.asyncio
async def test_autodata_route_closes_the_loop(client, monkeypatch):
    c, dsm = client
    monkeypatch.setattr(core, "call_model", _fake)  # seen by the background run thread too

    cfg = {
        "task_spec": "x", "gen_prompt": "x", "n_tasks": 3, "concurrency": 1,
        "challenger": {"url": "u", "model": "challenger"},
        "weak": {"url": "u", "model": "weak"},
        "strong": {"url": "u", "model": "strong"},
        "judge": {"url": "u", "model": "judge"},
    }
    resp = await c.post("/api/training/autodata", json={"config": cfg})
    assert resp.status == 202
    run_id = (await resp.json())["run_id"]

    # poll to completion (background run; fake model is instant)
    data = None
    for _ in range(200):
        data = await (await c.get(f"/api/training/autodata/{run_id}")).json()
        if data["status"] in ("completed", "failed"):
            break
        await asyncio.sleep(0.02)
    assert data["status"] == "completed", data

    ds_id = data["dataset_id"]
    assert ds_id, data
    assert data["report"]["kept"] == 1            # only TASK_A survives the Δ-filter

    # registered, resolvable, and listed via the real dataset API
    assert dsm.get(ds_id) is not None
    assert dsm.get(ds_id).samples == 1            # samples == kept count
    listing = await (await c.get("/api/datasets")).json()
    assert any(d["id"] == ds_id for d in listing["datasets"])

    # and the dataset_id is directly usable as a training job's input
    job_resp = await c.post("/api/training/jobs", json={
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "dataset_id": ds_id, "method": "lora",
    })
    assert job_resp.status == 201, await job_resp.text()


@pytest.mark.asyncio
async def test_autodata_route_rejects_bad_body(client):
    c, _ = client
    resp = await c.post("/api/training/autodata", json={"meta": True})  # no config
    assert resp.status == 400
    assert (await c.get("/api/training/autodata/nope")).status == 404   # unknown run, not a crash
