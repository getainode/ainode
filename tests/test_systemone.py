"""POST /v1/systemone: TypeSafe's Jev wire format over this node's decision core.

What is pinned here:

* **The translation, both ways.** A ``choice`` answers with the caller's own
  criteria key, a ``noul`` answers with P(true), and a ``score`` answers with the
  expected level plus a legend from position to level name. Every probability
  block is keyed by exactly the names the caller wrote, because a foreign client
  looks its own keys up in it.
* **That a real client can read the answer.** ``parse_answers_like_jde`` below is
  JDE's own ``parseAnswers`` (``src/judge/jev.ts``) rewritten line for line: it
  discards the WHOLE answer set over one malformed answer, so "our 200 parses" is
  a property worth asserting rather than assuming, and it is asserted on every
  live response in this file.
* **A malformed request is a 422 that names the field**, which is what the format
  specifies and what a caller needs to fix it.
* **The same engine the decide tests drive.** ``FakeEngine`` is imported from
  ``tests/test_decide.py`` rather than copied: the two routes share one core, so
  they are proven against one engine's behaviour, and an engine quirk fixed for
  one is fixed for both.

The engine-facing tests run a REAL AINode app whose cluster state points at that
fake on a real port, so routing, the body the engine receives and the failure
paths are exercised rather than mocked out from underneath.
"""

import json
import math
import os
import socket
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from ainode.api.decide import DecideError
from ainode.api.server import create_app
from ainode.api.systemone import (
    CHARS_PER_TOKEN,
    MAX_CRITERIA,
    Translated,
    answer_from_decision,
    decide_questions,
    estimate_tokens,
    normalized_confidence,
    translate_questions,
    usage_block,
)
from ainode.core.config import NodeConfig
from tests.test_decide import MODEL, FakeEngine, _app, _free_port

TICKET = ("Customer writes: the invoice PDF download 500s on every browser since "
          "your Tuesday release. We bill 400 clients on Friday.")

# One of each type, in the shape JDE's `questionsForWire` puts on the wire: the
# three fields a judge reasons from, and nothing else.
QUESTIONS = {
    "queue": {
        "type": "choice",
        "instructions": "Which queue should this ticket go to?",
        "criteria": {
            "billing": "an invoice, a charge or a refund",
            "bug": "the product did something it should not",
            "other": "none of these fit",
        },
    },
    "needs_human": {
        "type": "noul",
        "instructions": "Does this ticket need a human today?",
        "criteria": {"true": "a person has to act on it today",
                     "false": "it can wait or answer itself"},
    },
    "severity": {
        "type": "score",
        "instructions": "How severe is it?",
        "criteria": {"none": "cosmetic", "some": "a workaround exists",
                     "bad": "money or data is at risk"},
    },
}


# =============================================================================
# JDE's own reader, so "it parses" is checked and not assumed
# =============================================================================

def parse_answers_like_jde(value):
    """JDE's ``parseAnswers`` (``src/judge/jev.ts``), rule for rule.

    A shape check and nothing more, and ONE bad answer discards the set: that is
    the behaviour our 200 has to survive, so it is reproduced here rather than
    approximated. Returns None for a reply JDE would call malformed.
    """
    def finite01(number):
        return (isinstance(number, (int, float)) and not isinstance(number, bool)
                and math.isfinite(number) and 0.0 <= number <= 1.0)

    def probabilities(block):
        if not isinstance(block, dict):
            return None
        return {key: raw for key, raw in block.items()
                if isinstance(raw, (int, float)) and not isinstance(raw, bool)
                and math.isfinite(raw)}

    if not isinstance(value, dict):
        return None
    out = {}
    for key, answer in value.items():
        if not isinstance(answer, dict):
            return None
        kind = answer.get("type")
        if kind == "noul":
            if not finite01(answer.get("noul")):
                return None
            out[key] = {"type": "noul", "noul": answer["noul"]}
        elif kind == "choice":
            if not isinstance(answer.get("choice"), str):
                return None
            out[key] = {"type": "choice", "choice": answer["choice"],
                        "confidence": (answer["confidence"]
                                       if finite01(answer.get("confidence")) else 0),
                        "probabilities": probabilities(answer.get("probabilities"))}
        elif kind == "score":
            score = answer.get("score")
            if not (isinstance(score, (int, float)) and not isinstance(score, bool)
                    and math.isfinite(score)):
                return None
            out[key] = {"type": "score", "score": score,
                        "confidence": (answer["confidence"]
                                       if finite01(answer.get("confidence")) else 0),
                        "legend": answer.get("legend"),
                        "probabilities": probabilities(answer.get("probabilities"))}
        else:
            return None
    return out


def test_the_jde_reader_here_rejects_what_jde_rejects():
    """The guard above is only worth something if it refuses bad answers."""
    assert parse_answers_like_jde({"k": {"type": "noul", "noul": 0.5}}) is not None
    assert parse_answers_like_jde({"k": {"type": "noul", "noul": 1.5}}) is None
    assert parse_answers_like_jde({"k": {"type": "noul", "noul": "0.5"}}) is None
    assert parse_answers_like_jde({"k": {"type": "choice"}}) is None
    assert parse_answers_like_jde({"k": {"type": "guess", "choice": "a"}}) is None
    # One bad answer discards the set, which is why a partial 200 is a 503 here.
    assert parse_answers_like_jde({"good": {"type": "noul", "noul": 0.5},
                                   "bad": {"type": "score", "score": None}}) is None


# =============================================================================
# Translating a question in
# =============================================================================

def _err(raw):
    with pytest.raises(DecideError) as exc:
        translate_questions(raw)
    return str(exc.value)


def test_a_choice_keeps_its_criteria_keys_as_the_answer_names():
    item = translate_questions(QUESTIONS)["queue"]
    assert item.kind == "choice"
    assert item.names == ["billing", "bug", "other"]
    # The model reads the key AND what it means, in that order.
    assert item.options[0] == "billing: an invoice, a charge or a refund"
    assert item.question == "Which queue should this ticket go to?"


def test_a_noul_is_two_options_with_true_first():
    item = translate_questions(QUESTIONS)["needs_human"]
    assert item.kind == "noul"
    assert item.names == ["true", "false"]
    assert item.options[0].startswith("true: a person has to act")


def test_a_noul_takes_a_missing_or_partial_criteria_block():
    """Some clients send one side, some send none. The question still answers."""
    bare = translate_questions({"k": {"type": "noul", "instructions": "Done?"}})["k"]
    assert bare.options == ["true", "false"] and bare.names == ["true", "false"]
    half = translate_questions({"k": {"type": "noul", "instructions": "Done?",
                                      "criteria": {"true": "it was done"}}})["k"]
    assert half.options == ["true: it was done", "false"]


def test_a_noul_may_not_rename_its_sides():
    msg = _err({"k": {"type": "noul", "instructions": "Done?",
                      "criteria": {"yes": "it was", "no": "it was not"}}})
    assert "'criteria'" in msg and "'true' and 'false'" in msg


def test_a_score_takes_the_object_form_in_insertion_order():
    item = translate_questions(QUESTIONS)["severity"]
    assert item.kind == "score"
    assert item.names == ["none", "some", "bad"]
    assert item.options[2] == "bad: money or data is at risk"


def test_a_score_takes_the_list_form_in_position_order():
    item = translate_questions({"k": {"type": "score", "instructions": "How bad?",
                                      "criteria": ["fine", "poor", "awful"]}})["k"]
    assert item.names == ["fine", "poor", "awful"]
    assert item.options == ["fine", "poor", "awful"]


def test_a_rubric_outside_two_to_ten_levels_is_refused():
    assert "2 to 10 ordered levels" in _err(
        {"k": {"type": "score", "instructions": "q", "criteria": ["only"]}})
    assert "2 to 10 ordered levels" in _err(
        {"k": {"type": "score", "instructions": "q",
               "criteria": [f"level{i}" for i in range(11)]}})


def test_a_repeated_level_name_is_refused():
    msg = _err({"k": {"type": "score", "instructions": "q",
                      "criteria": ["same", "same"]}})
    assert "'criteria'" in msg and "repeats a level" in msg


def test_an_unknown_type_names_the_three_that_exist():
    msg = _err({"k": {"type": "vibe", "instructions": "q", "criteria": {"a": "b"}}})
    assert "'type'" in msg and "choice, noul, score" in msg
    assert "'vibe'" in msg


def test_a_question_with_no_instructions_is_refused():
    assert "'instructions'" in _err({"k": {"type": "noul"}})
    assert "'instructions'" in _err({"k": {"type": "noul", "instructions": "  "}})


def test_a_choice_with_no_criteria_is_refused():
    msg = _err({"k": {"type": "choice", "instructions": "which?"}})
    assert "'criteria'" in msg
    assert "at least 2 'criteria' options" in _err(
        {"k": {"type": "choice", "instructions": "which?", "criteria": {"a": "only"}}})


def test_a_criteria_set_wider_than_the_engine_can_report_is_refused():
    """20 labels come back with a probability, so a 21st option is not answerable.

    The format allows 255. This node cannot report them, and a distribution that
    silently lost its tail would be worse than a refusal that names the cap.
    """
    at_cap = {f"opt{i}": f"option {i}" for i in range(MAX_CRITERIA)}
    assert len(translate_questions(
        {"k": {"type": "choice", "instructions": "which?",
               "criteria": at_cap}})["k"].names) == MAX_CRITERIA
    over = dict(at_cap, one_too_many="the tail")
    msg = _err({"k": {"type": "choice", "instructions": "which?", "criteria": over}})
    assert f"more than the {MAX_CRITERIA}" in msg
    assert "'criteria'" in msg and "distribution missing its tail" in msg


def test_a_score_with_no_criteria_is_refused():
    assert "'criteria'" in _err({"k": {"type": "score", "instructions": "how bad?"}})


def test_an_empty_or_missing_question_set_is_refused():
    assert "non-empty object" in _err(None)
    assert "non-empty object" in _err({})
    assert "non-empty object" in _err([QUESTIONS])


def test_a_field_the_format_does_not_define_is_ignored_and_never_echoed():
    """JDE strips `passingAnswer` on purpose; one that arrives anyway is code's."""
    item = translate_questions({"k": {"type": "noul", "instructions": "Done?",
                                      "criteria": {"true": "yes", "false": "no"},
                                      "passingAnswer": "false"}})["k"]
    assert item == Translated("noul", "Done?", ["true: yes", "false: no"],
                              ["true", "false"])


def test_the_translated_set_goes_through_the_shared_normalizer():
    """One guard for both routes: the ceiling, the floor and distinct options."""
    questions = decide_questions(translate_questions(QUESTIONS))
    assert list(questions) == ["queue", "needs_human", "severity"]
    assert questions["queue"]["question"] == "Which queue should this ticket go to?"
    assert questions["needs_human"]["options"][0].startswith("true: ")


def test_question_order_is_the_order_the_caller_wrote():
    assert list(translate_questions(QUESTIONS)) == ["queue", "needs_human",
                                                    "severity"]


# =============================================================================
# Translating an answer out
# =============================================================================

def _entry(answer, distribution=None, confidence=None):
    """A ``/v1/decide`` decision entry, the shape the core hands over."""
    return {"answer": answer, "confidence": confidence,
            "distribution": distribution, "latency_ms": 12.3}


CHOICE_ITEM = Translated("choice", "Which queue?",
                         ["billing: money", "bug: broken", "other: neither"],
                         ["billing", "bug", "other"])
NOUL_ITEM = Translated("noul", "Human?", ["true: yes", "false: no"],
                       ["true", "false"])
SCORE_ITEM = Translated("score", "How bad?", ["none: cosmetic", "some: workaround",
                                              "bad: money at risk"],
                        ["none", "some", "bad"])


def test_a_choice_answers_with_the_caller_s_key_and_its_own_probabilities():
    entry = _entry("bug: broken", {"billing: money": 0.1, "bug: broken": 0.7,
                                   "other: neither": 0.2}, 0.7)
    answer = answer_from_decision(CHOICE_ITEM, entry)
    # Confidence is chance corrected over three options: (3 * 0.7 - 1) / 2.
    assert answer == {"type": "choice", "choice": "bug", "confidence": 0.55,
                      "probabilities": {"billing": 0.1, "bug": 0.7, "other": 0.2}}
    # Exactly the criteria keys, nothing added and nothing left out.
    assert set(answer["probabilities"]) == set(CHOICE_ITEM.names)
    assert math.isclose(sum(answer["probabilities"].values()), 1.0, abs_tol=1e-6)


def test_a_noul_is_the_probability_of_the_true_option():
    entry = _entry("true: yes", {"true: yes": 0.82, "false: no": 0.18}, 0.82)
    assert answer_from_decision(NOUL_ITEM, entry) == {"type": "noul", "noul": 0.82}
    # A confident no is a low noul, not a high confidence: the number is P(true)
    # whichever way the model went.
    entry = _entry("false: no", {"true: yes": 0.04, "false: no": 0.96}, 0.96)
    assert answer_from_decision(NOUL_ITEM, entry) == {"type": "noul", "noul": 0.04}


def test_a_score_is_the_expected_level_with_a_legend_by_position():
    entry = _entry("some: workaround", {"none: cosmetic": 0.0,
                                        "some: workaround": 0.5,
                                        "bad: money at risk": 0.5}, 0.5)
    answer = answer_from_decision(SCORE_ITEM, entry)
    assert answer["type"] == "score"
    # Split between level 1 and level 2, so 1.5: the expected level, not the pick.
    assert answer["score"] == 1.5
    # Two levels at 0.5 over three levels: (3 * 0.5 - 1) / 2, not 0.5.
    assert answer["confidence"] == 0.25
    assert answer["legend"] == {"0": "none", "1": "some", "2": "bad"}
    assert answer["probabilities"] == {"0": 0.0, "1": 0.5, "2": 0.5}
    assert math.isclose(sum(answer["probabilities"].values()), 1.0, abs_tol=1e-6)


def test_a_score_on_one_certain_level_is_that_level():
    entry = _entry("bad: money at risk", {"none: cosmetic": 0.0,
                                          "some: workaround": 0.0,
                                          "bad: money at risk": 1.0}, 1.0)
    assert answer_from_decision(SCORE_ITEM, entry)["score"] == 2.0


def test_confidence_is_chance_corrected_and_not_the_picked_probability():
    """The hosted service's own formula: 1/n reports 0 and certainty reports 1.

    A two-way question at 0.6 and a ten-way question at 0.6 are not the same
    judgement, and a JDE user's bands are tuned against the hosted numbers.
    """
    assert normalized_confidence(1.0, 4) == 1.0
    assert normalized_confidence(0.25, 4) == 0.0  # uniform over four: no signal
    assert normalized_confidence(0.7, 4) == 0.6  # (4 * 0.7 - 1) / 3
    assert normalized_confidence(0.6, 2) == pytest.approx(0.2)
    assert normalized_confidence(0.6, 10) == pytest.approx(0.5555555, abs=1e-6)
    # Below chance cannot happen off an argmax, and is floored rather than negative.
    assert normalized_confidence(0.1, 10) == 0.0


def test_an_engine_with_no_logprobs_answers_without_inventing_a_spread():
    """The core's no-distribution fallback: an answer, and no probabilities."""
    choice = answer_from_decision(CHOICE_ITEM, _entry("other: neither",
                                                      None, 1.0))
    assert choice == {"type": "choice", "choice": "other", "confidence": 1.0}
    assert "probabilities" not in choice
    score = answer_from_decision(SCORE_ITEM, _entry("some: workaround", None, 1.0))
    assert score["score"] == 1.0 and "probabilities" not in score
    assert score["legend"] == {"0": "none", "1": "some", "2": "bad"}
    assert answer_from_decision(NOUL_ITEM, _entry("true: yes", None, 1.0)) == {
        "type": "noul", "noul": 1.0}
    assert answer_from_decision(NOUL_ITEM, _entry("false: no", None, 1.0)) == {
        "type": "noul", "noul": 0.0}


def test_an_answer_the_route_cannot_read_is_no_answer():
    """None, so the handler can refuse the request instead of guessing a key."""
    assert answer_from_decision(CHOICE_ITEM, _entry(None)) is None
    assert answer_from_decision(NOUL_ITEM, _entry("maybe")) is None


def test_a_probability_outside_the_reals_cannot_poison_the_answer_set():
    """One NaN would make JDE discard every answer in the reply, not just this one."""
    entry = _entry("true: yes", {"true: yes": float("nan"), "false: no": 0.5}, 1.5)
    answer = answer_from_decision(NOUL_ITEM, entry)
    assert answer["noul"] == 0.0
    assert parse_answers_like_jde({"k": answer}) is not None


# =============================================================================
# Usage
# =============================================================================

def test_usage_is_the_engine_s_own_count_when_it_reports_one():
    payloads = [{"usage": {"prompt_tokens": 120, "completion_tokens": 2}},
                {"usage": {"prompt_tokens": 118, "completion_tokens": 2}}]
    assert usage_block(payloads, TICKET, translate_questions(QUESTIONS), 2) == {
        "input_tokens": 238, "output_tokens": 4}


def test_usage_falls_back_to_the_bench_s_own_estimate():
    """No usage block at all: an estimate at 4 chars per token, documented as one."""
    translated = translate_questions(QUESTIONS)
    usage = usage_block([{}, {}, {}], TICKET, translated, 3)
    state_tokens = estimate_tokens(TICKET)
    assert state_tokens == math.ceil(len(TICKET) / CHARS_PER_TOKEN)
    # The state is read once per question, which is what the engines really do.
    assert usage["input_tokens"] > len(translated) * state_tokens
    # One constrained label per question is the whole output the grammar allows.
    assert usage["output_tokens"] == 3
    assert isinstance(usage["input_tokens"], int)


# =============================================================================
# The live route
# =============================================================================

@pytest.fixture
def engine_fake():
    return FakeEngine()


@pytest_asyncio.fixture
async def engine(engine_fake):
    server = TestServer(engine_fake.app())
    await server.start_server()
    try:
        yield server
    finally:
        await server.close()


@pytest_asyncio.fixture
async def client(engine):
    """A real AINode app that believes a peer serves MODEL on the fake's port."""
    async with TestClient(TestServer(_app(engine.port))) as c:
        yield c


def _body(**over):
    body = {"model": MODEL, "state": TICKET, "questions": QUESTIONS}
    body.update(over)
    return body


@pytest.mark.asyncio
async def test_every_question_type_round_trips(client, engine_fake):
    resp = await client.post("/v1/systemone", json=_body())
    assert resp.status == 200
    data = await resp.json()
    assert data["model"] == MODEL
    assert data["latency_ms"] > 0
    assert set(data) == {"model", "answers", "usage", "latency_ms", "calibration"}
    # No temperatures.json in this node's store: the engine's own spread.
    assert data["calibration"] == {"applied": False, "temperatures": None}
    assert list(data["answers"]) == ["queue", "needs_human", "severity"]
    assert parse_answers_like_jde(data["answers"]) is not None

    # The fake always picks the first option, which is the first criteria key.
    queue = data["answers"]["queue"]
    assert queue == {"type": "choice", "choice": "billing",
                     "confidence": queue["confidence"],
                     "probabilities": queue["probabilities"]}
    assert set(queue["probabilities"]) == {"billing", "bug", "other"}
    assert queue["confidence"] > 0.9

    needs_human = data["answers"]["needs_human"]
    assert needs_human["type"] == "noul" and needs_human["noul"] > 0.9

    severity = data["answers"]["severity"]
    assert severity["legend"] == {"0": "none", "1": "some", "2": "bad"}
    assert set(severity["probabilities"]) == {"0", "1", "2"}
    assert 0.0 <= severity["score"] < 0.1  # nearly all the mass on level 0

    # A noul's whole answer is P(true), so it carries no probabilities block; the
    # other two spread over their own names and the spread is a distribution.
    assert "probabilities" not in needs_human
    for answer in (queue, severity):
        block = answer["probabilities"]
        assert math.isclose(sum(block.values()), 1.0, abs_tol=1e-4)
        assert all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in block.values())


@pytest.mark.asyncio
async def test_the_answers_come_from_one_constrained_call_per_question(client,
                                                                      engine_fake):
    await client.post("/v1/systemone", json=_body())
    assert len(engine_fake.seen) == 3
    # In flight together, the way /v1/decide asks: one ask is about one question
    # plus the spread, not three questions end to end.
    assert engine_fake.max_in_flight == 3
    prefixes = set()
    for body in engine_fake.seen:
        assert body["model"] == MODEL
        assert body["structured_outputs"]["choice"][0] == "A"
        assert body["logprobs"] is True
        prefixes.add(body["messages"][1]["content"].split("QUESTION:")[0])
    # Every question shares the state's prefix, so the engine prefills it once.
    assert len(prefixes) == 1
    # A question's options reach the model as the caller's keys plus their meaning.
    options = "\n".join(body["messages"][1]["content"] for body in engine_fake.seen)
    assert "A. billing: an invoice, a charge or a refund" in options
    assert "A. true: a person has to act on it today" in options
    assert "C. bad: money or data is at risk" in options


@pytest.mark.asyncio
async def test_usage_and_a_json_state_come_back_in_the_format_s_own_keys(client,
                                                                        engine_fake):
    resp = await client.post("/v1/systemone", json=_body(
        state={"subject": "invoice 500s", "plan": "pro"},
        questions={"k": QUESTIONS["needs_human"]}))
    data = await resp.json()
    assert data["usage"] == {"input_tokens": 100, "output_tokens": 2}
    user = engine_fake.seen[0]["messages"][1]["content"]
    assert '{"plan":"pro","subject":"invoice 500s"}' in user


@pytest.mark.asyncio
async def test_the_model_defaults_the_way_decide_defaults_it(client):
    resp = await client.post("/v1/systemone", json={
        "state": TICKET, "questions": {"k": QUESTIONS["needs_human"]}})
    assert resp.status == 200
    assert (await resp.json())["model"] == MODEL


@pytest.mark.asyncio
@pytest.mark.parametrize("body,fragment", [
    ({"state": "s"}, "'questions' must be a non-empty object"),
    ({"questions": {}}, "'questions' must be a non-empty object"),
    ({"questions": {"k": {"instructions": "q", "criteria": {"a": "b", "c": "d"}}}},
     "'type' must be one of"),
    ({"questions": {"k": {"type": "guess", "instructions": "q"}}},
     "'type' must be one of"),
    ({"questions": {"k": {"type": "choice", "criteria": {"a": "b", "c": "d"}}}},
     "needs a non-empty 'instructions' string"),
    ({"questions": {"k": {"type": "choice", "instructions": "q"}}},
     "'criteria'"),
    ({"questions": {"k": {"type": "choice", "instructions": "q",
                          "criteria": {"only": "one"}}}},
     "at least 2 'criteria' options"),
    ({"questions": {"k": {"type": "choice", "instructions": "q",
                          "criteria": {f"o{i}": "d" for i in range(21)}}}},
     "more than the 20 this node can report a probability for"),
    ({"questions": {"k": {"type": "score", "instructions": "q"}}}, "'criteria'"),
    ({"questions": {"k": {"type": "score", "instructions": "q",
                          "criteria": ["one"]}}},
     "2 to 10 ordered levels"),
    ({"questions": {"k": {"type": "noul", "instructions": "q",
                          "criteria": {"yes": "a", "no": "b"}}}},
     "'true' and 'false'"),
    ({"questions": {"k": {"type": "noul", "instructions": "q", "criteria": 7}}},
     "'criteria' must be an object"),
    ({"model": "", "questions": {"k": {"type": "noul", "instructions": "q"}}},
     "'model' must be a non-empty string"),
])
async def test_a_malformed_request_is_a_422_that_names_the_field(client, engine_fake,
                                                                body, fragment):
    resp = await client.post("/v1/systemone", json=body)
    assert resp.status == 422
    payload = await resp.json()
    assert fragment in payload["error"]["message"]
    assert payload["error"]["type"] == "invalid_request_error"
    assert engine_fake.seen == []  # nothing reached the engine


@pytest.mark.asyncio
async def test_a_body_that_is_not_json_is_a_422(client):
    resp = await client.post("/v1/systemone", data=b"{nope",
                             headers={"Content-Type": "application/json"})
    assert resp.status == 422
    assert "not valid JSON" in (await resp.json())["error"]["message"]
    resp = await client.post("/v1/systemone", json=["a", "list"])
    assert resp.status == 422
    assert "must be a JSON object" in (await resp.json())["error"]["message"]


@pytest.mark.asyncio
async def test_a_model_no_node_serves_is_a_503(client, engine_fake):
    """A client that left the hosted default in place is told which id failed."""
    resp = await client.post("/v1/systemone", json=_body(model="jev-latest"))
    assert resp.status == 503
    body = await resp.json()
    assert "no node is serving 'jev-latest'" in body["error"]["message"]
    assert body["error"]["type"] == "service_unavailable"
    assert engine_fake.seen == []


@pytest.mark.asyncio
async def test_an_unreachable_engine_is_a_503_and_never_half_an_answer_set():
    dead = _free_port()
    async with TestClient(TestServer(_app(dead))) as c:
        resp = await c.post("/v1/systemone", json=_body())
        assert resp.status == 503
        msg = (await resp.json())["error"]["message"]
        assert "engine calls failed" in msg and "queue" in msg


@pytest.mark.asyncio
async def test_an_engine_with_no_logprobs_still_answers_every_question():
    quiet = FakeEngine(with_logprobs=False)
    server = TestServer(quiet.app())
    await server.start_server()
    try:
        async with TestClient(TestServer(_app(server.port))) as c:
            resp = await c.post("/v1/systemone", json=_body())
            assert resp.status == 200
            answers = (await resp.json())["answers"]
            assert parse_answers_like_jde(answers) is not None
            assert answers["needs_human"] == {"type": "noul", "noul": 1.0}
            assert answers["queue"]["choice"] == "billing"
            assert "probabilities" not in answers["queue"]
            assert answers["severity"]["score"] == 0.0
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_systemone_is_in_the_endpoint_catalog(client):
    catalog = await (await client.get("/api/server/endpoints")).json()
    paths = [ep["path"] for group in catalog.values() for ep in group]
    assert "/v1/systemone" in paths


# =============================================================================
# Auth: it is a /v1 path, so it wants a key
# =============================================================================

@pytest.fixture
def auth_home(tmp_path, monkeypatch):
    """Keep every file this app writes out of the operator's own ~/.ainode.

    Same redirect ``tests/test_auth_gate.py`` uses: most paths resolve
    AINODE_HOME at call time, but AUTH_FILE, CONFIG_FILE and SECRETS_FILE are
    computed at import.
    """
    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr("ainode.auth.middleware.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.auth.middleware.AUTH_FILE", tmp_path / "auth.json")
    monkeypatch.setattr("ainode.secrets.manager.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.secrets.manager.SECRETS_FILE",
                        tmp_path / "secrets.json")
    return tmp_path


@pytest_asyncio.fixture
async def keyed(auth_home):
    """Auth on, plus the plaintext key, which exists only at this moment."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    app = create_app(config=NodeConfig(node_id="auth-node", node_name="AuthNode",
                                       api_port=free_port, onboarded=True),
                     engine=None)
    entry = app["auth_config"].enable()
    async with TestClient(TestServer(app)) as c:
        yield c, entry["key"]


@pytest.mark.asyncio
async def test_the_route_is_refused_without_a_key_when_auth_is_on(keyed):
    client, key = keyed
    body = {"state": TICKET, "questions": {"k": QUESTIONS["needs_human"]}}
    resp = await client.post("/v1/systemone", json=body)
    assert resp.status == 401
    assert (await resp.json())["error"]["type"] == "auth_error"

    resp = await client.post("/v1/systemone", json=body,
                             headers={"Authorization": "Bearer not-the-key"})
    assert resp.status == 401

    # With the key it reaches the handler, which on this node has no engine to
    # ask: past the gate is the whole assertion, and a 401 is not.
    resp = await client.post("/v1/systemone", json=body,
                             headers={"Authorization": f"Bearer {key}"})
    assert resp.status != 401


# =============================================================================
# One real JDE case, replayed
# =============================================================================

# JDE's blind case set, read where it lives and never copied into this repo: the
# cases are someone else's measurement data, and a copy here would be a second
# version of it. Absent (CI, another machine) means this test skips; the shapes
# it checks are covered above without it.
JDE_CASES = Path(os.environ.get(
    "JDE_COMPLETION_CASES",
    "/Users/sem/code/jde/cases/completion-check-blind.json"))

# The wording is JDE's, from src/decisions/completion-check.ts, copied because
# that file says it is the artefact: rewording a clause invalidates the
# measurement behind it. `passingAnswer` is deliberately NOT here, because
# questionsForWire strips it before the wire and a judge must never see it.
RESULT_IS_ECHO_QUESTION = {
    "type": "noul",
    "instructions": "Is `claimed_result` a restatement of `task` rather than a report of an outcome?",
    "criteria": {
        "true": "`claimed_result` repeats the task's own words or its instructions back, with no outcome of its own",
        "false": "`claimed_result` reports what happened, what was produced, or what was found",
    },
}


def _part_question(part):
    """JDE's ``partQuestion``: a reply part is judged on the claim, an action on receipts."""
    if part["kind"] == "reply":
        return {
            "type": "noul",
            "instructions": f"Does `claimed_result` contain {part['text']}, stated as an outcome rather than a plan?",
            "criteria": {
                "true": "`claimed_result` carries that content itself, written as something already produced or found",
                "false": "`claimed_result` does not carry it, or only says it will be produced",
            },
        }
    return {
        "type": "noul",
        "instructions": f"Do `receipts` show that this part was carried out: {part['text']}? Count a file written, a search run, a page fetched, or a tool call that produces it; do not count `claimed_result` saying so.",
        "criteria": {
            "true": "`receipts` carry a file, a search, a fetched page or a tool call that carries out this part",
            "false": "nothing in `receipts` carries out this part, whatever `claimed_result` says about it",
        },
    }


def _completion_questions(parts):
    """JDE's ``completionQuestions``: the echo question, then the parts in order.

    A file part is code's own fact (the path was written, and not empty), so it
    never becomes a question. That is why the ids below skip an index.
    """
    questions = {"result_is_echo": RESULT_IS_ECHO_QUESTION}
    for index, part in enumerate(parts):
        if part["kind"] == "file":
            continue
        questions[f"part_{index}_done"] = _part_question(part)
    return questions


@pytest.mark.asyncio
@pytest.mark.skipif(not JDE_CASES.is_file(),
                    reason=f"JDE's case set is not on this machine ({JDE_CASES})")
async def test_a_real_jde_completion_check_case_round_trips(client):
    """The question set a JDE completion check really asks, answered by this route.

    Built the way ``completionCheck`` builds it and read the way ``parseAnswers``
    reads it, so what is proven is that an unmodified JDE pointed at a node gets
    an answer set its own reader accepts. What the model SAYS is the fake
    engine's, so nothing here asserts a verdict: that is the decision bench's
    job, against labels.
    """
    cases = json.loads(JDE_CASES.read_text())
    case = next(c for c in cases if c["id"] == "heldout5-001")
    state = case["state"]
    questions = _completion_questions(state["task_parts"])
    # Three parts, one of them a file: two part questions plus the echo question.
    assert list(questions) == ["result_is_echo", "part_0_done", "part_2_done"]

    resp = await client.post("/v1/systemone",
                             json={"model": MODEL, "state": state,
                                   "questions": questions})
    assert resp.status == 200
    data = await resp.json()

    parsed = parse_answers_like_jde(data["answers"])
    assert parsed is not None, "JDE would call this answer set malformed"
    assert list(parsed) == list(questions)
    for answer in parsed.values():
        # completionCheck reads every one of these as a noul and floors it at 0.7.
        assert answer["type"] == "noul"
        assert math.isfinite(answer["noul"]) and 0.0 <= answer["noul"] <= 1.0
    assert data["usage"]["input_tokens"] > 0
    assert isinstance(data["usage"]["output_tokens"], int)
    assert data["model"] == MODEL
