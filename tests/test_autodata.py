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

from ainode.training.autodata import core, meta
from ainode.training.autodata.core import AutoDataConfig, Endpoint, run
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
