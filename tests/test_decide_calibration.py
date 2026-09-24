"""The adapter's own temperatures on /v1/decide and /v1/systemone (#276).

A decision model's store directory can carry ``temperatures.json`` beside the
weights, one fitted temperature per question kind. When the served model's
directory has one, each question's label logprobs are divided by its kind's
temperature before the softmax; ``"calibration": "raw"`` opts out; and both
routes say what was applied. The route tests reuse the fake vLLM the decide tests
drive, with this node's model store pointed at a temp directory.
"""

import json
import math

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from ainode.api.decide import (
    CHOICE,
    NOUL,
    SCORE,
    DecideError,
    calibration_for,
    calibration_mode,
    decision_from_payload,
    distribution_from_logprobs,
    is_decision_model_dir,
    model_store_dir,
    normalize_questions,
    read_temperatures,
)
from tests.test_decide import MODEL, QUESTIONS, TICKET, FakeEngine, _app, _payload
from tests.test_systemone import QUESTIONS as JEV_QUESTIONS

TEMPS = {"choice": 2.0, "noul": 0.5, "score": 1.5}


def _store(models_dir, temps=TEMPS, name="temperatures.json"):
    """A model directory the way our downloader writes one: ``org--name``."""
    directory = models_dir / MODEL.replace("/", "--")
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    if temps is not None:
        (directory / name).write_text(json.dumps({"temperatures": temps,
                                                  "fitted_on": "held-out"}))
    return directory


def _softmax(logprobs, temperature=1.0):
    top = max(logprobs)
    weights = [math.exp((lp - top) / temperature) for lp in logprobs]
    return [w / sum(weights) for w in weights]


def _fake_logprobs(count):
    """What ``FakeEngine`` puts on the labels when it picks the first one."""
    return [-0.0625] + [-4.0 - i for i in range(1, count)]


# ------------------------------------------------------------------- the math


def test_a_temperature_divides_the_logprobs_before_the_softmax():
    tops = [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.1}]
    raw = distribution_from_logprobs(["A", "B"], tops)
    hot = distribution_from_logprobs(["A", "B"], tops, temperature=2.0)
    cold = distribution_from_logprobs(["A", "B"], tops, temperature=0.5)
    assert raw["A"] == pytest.approx(1 / (1 + math.exp(-2.0)), abs=1e-6)
    assert hot["A"] == pytest.approx(1 / (1 + math.exp(-1.0)), abs=1e-6)
    assert cold["A"] == pytest.approx(1 / (1 + math.exp(-4.0)), abs=1e-6)
    # Tempering flattens or sharpens, never reorders.
    assert cold["A"] > raw["A"] > hot["A"] > 0.5


def test_a_decision_reads_its_confidence_off_the_tempered_spread():
    payload = _payload("A", [("A", -0.1), ("B", -2.1)])
    raw = decision_from_payload(payload, ["yes", "no"], 10.0)
    hot = decision_from_payload(payload, ["yes", "no"], 10.0, temperature=2.0)
    assert raw["answer"] == hot["answer"] == "yes"
    assert hot["confidence"] == pytest.approx(1 / (1 + math.exp(-1.0)), abs=1e-6)
    assert hot["confidence"] < raw["confidence"]
    assert sum(hot["distribution"].values()) == pytest.approx(1.0, abs=1e-5)


def test_each_decide_question_is_tagged_with_the_kind_it_is_fitted_as():
    out = normalize_questions(QUESTIONS)
    assert out["category"]["kind"] == CHOICE
    assert out["urgency"]["kind"] == SCORE
    assert out["needs_human"]["kind"] == NOUL


# ------------------------------------------------------------ reading the file


def test_the_temperatures_are_read_from_the_model_s_store_directory(tmp_path):
    directory = _store(tmp_path)
    assert model_store_dir(tmp_path, MODEL) == directory
    assert is_decision_model_dir(directory)
    assert read_temperatures(directory) == TEMPS


def test_a_prompt_contract_alone_marks_a_decision_model(tmp_path):
    directory = _store(tmp_path, temps=None)
    assert not is_decision_model_dir(directory)
    (directory / "prompt_contract.json").write_text("{}")
    assert is_decision_model_dir(directory)
    assert read_temperatures(directory) is None


def test_a_bad_entry_leaves_that_kind_at_the_engine_s_own_spread(tmp_path):
    directory = _store(tmp_path, temps={"choice": 0, "noul": "hot", "score": 1.25,
                                        "other": 3.0, "extra": True})
    assert read_temperatures(directory) == {"score": 1.25}
    (directory / "temperatures.json").write_text("not json")
    assert read_temperatures(directory) is None
    (directory / "temperatures.json").write_text(json.dumps({"temperatures": []}))
    assert read_temperatures(directory) is None


def test_the_calibration_block_says_what_was_applied(tmp_path):
    assert calibration_for(tmp_path, MODEL, None) == {"applied": False,
                                                      "temperatures": None}
    _store(tmp_path)
    assert calibration_for(tmp_path, MODEL, None) == {"applied": True,
                                                      "temperatures": TEMPS}
    # Opting out still shows what the caller opted out of.
    assert calibration_for(tmp_path, MODEL, "raw") == {"applied": False,
                                                       "temperatures": TEMPS}


def test_raw_is_the_only_calibration_a_request_can_ask_for():
    assert calibration_mode(None) is None
    assert calibration_mode("raw") == "raw"
    with pytest.raises(DecideError, match="'calibration' must be \"raw\""):
        calibration_mode("tempered")


# ------------------------------------------------------------------ the routes


def _calibrated_app(port, models_dir):
    app = _app(port)
    app["config"].models_dir = str(models_dir)
    return app


@pytest_asyncio.fixture
async def engine_server():
    fake = FakeEngine()
    server = TestServer(fake.app())
    await server.start_server()
    try:
        yield server
    finally:
        await server.close()


@pytest_asyncio.fixture
async def calibrated(engine_server, tmp_path):
    _store(tmp_path)
    async with TestClient(TestServer(_calibrated_app(engine_server.port,
                                                     tmp_path))) as c:
        yield c


@pytest_asyncio.fixture
async def uncalibrated(engine_server, tmp_path):
    _store(tmp_path, temps=None)
    async with TestClient(TestServer(_calibrated_app(engine_server.port,
                                                     tmp_path))) as c:
        yield c


def _decide_body(**over):
    body = {"model": MODEL, "state": TICKET, "questions": QUESTIONS}
    body.update(over)
    return body


@pytest.mark.asyncio
async def test_decide_applies_each_question_kind_s_temperature(calibrated):
    resp = await calibrated.post("/v1/decide", json=_decide_body())
    assert resp.status == 200
    data = await resp.json()
    assert data["calibration"] == {"applied": True, "temperatures": TEMPS}
    decisions = data["decisions"]
    cases = (("category", 5, TEMPS["choice"]), ("urgency", 5, TEMPS["score"]),
             ("needs_human", 2, TEMPS["noul"]))
    for key, count, temperature in cases:
        expected = _softmax(_fake_logprobs(count), temperature)
        assert decisions[key]["confidence"] == pytest.approx(expected[0], abs=1e-5), key
        got = list(decisions[key]["distribution"].values())
        assert got == pytest.approx(expected, abs=1e-5), key


@pytest.mark.asyncio
async def test_decide_raw_opt_out_is_the_engine_s_own_spread(calibrated):
    resp = await calibrated.post("/v1/decide", json=_decide_body(calibration="raw"))
    assert resp.status == 200
    data = await resp.json()
    assert data["calibration"] == {"applied": False, "temperatures": TEMPS}
    expected = _softmax(_fake_logprobs(5))
    assert data["decisions"]["category"]["confidence"] == pytest.approx(expected[0],
                                                                        abs=1e-5)


@pytest.mark.asyncio
async def test_decide_without_a_temperatures_file_is_unchanged(uncalibrated):
    resp = await uncalibrated.post("/v1/decide", json=_decide_body())
    assert resp.status == 200
    data = await resp.json()
    assert data["calibration"] == {"applied": False, "temperatures": None}
    expected = _softmax(_fake_logprobs(2))
    assert data["decisions"]["needs_human"]["confidence"] == pytest.approx(
        expected[0], abs=1e-5)


@pytest.mark.asyncio
async def test_decide_refuses_an_unknown_calibration_with_a_400(calibrated):
    resp = await calibrated.post("/v1/decide", json=_decide_body(calibration="warm"))
    assert resp.status == 400
    assert "'calibration'" in (await resp.json())["error"]["message"]


def _jev_body(**over):
    body = {"model": MODEL, "state": TICKET, "questions": JEV_QUESTIONS}
    body.update(over)
    return body


@pytest.mark.asyncio
async def test_systemone_applies_the_temperatures_and_reports_them(calibrated):
    resp = await calibrated.post("/v1/systemone", json=_jev_body())
    assert resp.status == 200
    data = await resp.json()
    assert data["calibration"] == {"applied": True, "temperatures": TEMPS}
    answers = data["answers"]
    queue = _softmax(_fake_logprobs(3), TEMPS["choice"])
    assert answers["queue"]["probabilities"]["billing"] == pytest.approx(queue[0],
                                                                        abs=1e-5)
    # Chance-corrected off the TEMPERED top probability.
    assert answers["queue"]["confidence"] == pytest.approx((3 * queue[0] - 1) / 2,
                                                           abs=1e-5)
    noul = _softmax(_fake_logprobs(2), TEMPS["noul"])
    assert answers["needs_human"]["noul"] == pytest.approx(noul[0], abs=1e-5)


@pytest.mark.asyncio
async def test_systemone_raw_opt_out_reports_temperatures_it_did_not_apply(calibrated):
    resp = await calibrated.post("/v1/systemone", json=_jev_body(calibration="raw"))
    assert resp.status == 200
    data = await resp.json()
    assert data["calibration"] == {"applied": False, "temperatures": TEMPS}
    noul = _softmax(_fake_logprobs(2))
    assert data["answers"]["needs_human"]["noul"] == pytest.approx(noul[0], abs=1e-5)


@pytest.mark.asyncio
async def test_systemone_refuses_an_unknown_calibration_with_a_422(calibrated):
    resp = await calibrated.post("/v1/systemone", json=_jev_body(calibration=1))
    assert resp.status == 422
    assert "'calibration'" in (await resp.json())["error"]["message"]
