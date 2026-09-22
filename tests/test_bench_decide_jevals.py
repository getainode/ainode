"""Tests for the Jevals-recipe half of the decision bench.

No model, no node, no network. Four kinds of fake stand in for the real thing:

  * hand-computed decisions handed straight to the metric functions, because a metric is
    a pure function of the decisions and the numbers are the whole product here;
  * canned response payloads handed to each transport's parser, one per response shape
    the two wire formats return, so a parser is pinned without a server;
  * small question-file fixtures for the loaders, plus the four committed manifests;
  * a real HTTP server on a loopback port answering both wire formats, so the loop, the
    record and the console table are exercised end to end with no fleet anywhere.

The verdicts that matter most are the calibration ones and the three anchors the recipe
defines: a perfectly calibrated toy, a constant guesser scoring 0 on the Decision Score,
and a wrong-and-confident system scoring negative. If those three move, the board number
this bench claims to be comparable to has stopped meaning the same thing.
"""

import importlib.util
import json
import pathlib
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ainode.bench.decide import jevals as jv
from ainode.bench.decide import cli as decide_cli
from ainode.bench.decide import sets as st
from ainode.bench.decide import suite as su

REPO = pathlib.Path(__file__).resolve().parent.parent
RENDERER = REPO / "scripts" / "render-bench-table.py"
MANIFESTS = REPO / "bench" / "decide" / "sets"


# ---------------------------------------------------------------- decision fixtures

def decision(set_name="s", item="q1", repeat=0, options=("a", "b"), label="a",
             vector=None, pick=None, qtype=jv.CHOICE, wall_ms=100, tokens_in=10,
             tokens_out=2, error=None, malformed=False, one_hot=False, gold=None):
    """One decision dict, with the pick and the confidence derived from the vector."""
    options = list(options)
    if vector is None and not error and not malformed:
        vector = {option: (1.0 if option == label else 0.0) for option in options}
    if pick is None and vector:
        pick = jv.pick_from_vector(vector, options, presented=options, qtype=qtype)
    row = {"id": item, "set": set_name, "type": qtype, "repeat": repeat,
           "order": su.order_index(repeat), "options": options, "label": label,
           "vector": vector, "pick": pick,
           "confidence": None if (pick is None or not vector) else vector.get(pick),
           "malformed": malformed, "one_hot": one_hot, "wall_ms": wall_ms,
           "tokens_in": tokens_in, "tokens_out": tokens_out, "error": error,
           "gold": gold}
    if one_hot:
        row["confidence"] = 1.0
    return row


def spread(vectors, label="a", options=("a", "b"), qtype=jv.CHOICE, set_name="s"):
    """One decision per vector, each its own question, one repeat each."""
    return [decision(set_name=set_name, item=f"q{index}", options=options, label=label,
                     vector=v, qtype=qtype)
            for index, v in enumerate(vectors)]


# ---------------------------------------------------------------- the two losses

def test_the_multiclass_brier_is_the_full_sum_and_not_one_term():
    """Two options, all the mass on the wrong one, is 2.0 and not 1.0."""
    assert jv.brier_loss({"a": 0.0, "b": 1.0}, ["a", "b"], "a") == pytest.approx(2.0)
    assert jv.brier_loss({"a": 1.0, "b": 0.0}, ["a", "b"], "a") == 0.0
    # A uniform answer over K options scores 1 - 1/K.
    assert jv.brier_loss({o: 0.25 for o in "abcd"}, list("abcd"), "a") == \
        pytest.approx(0.75)


def test_the_ranked_probability_score_punishes_a_far_level_more_than_a_near_one():
    """The ordinal loss, which a Brier score would flatten."""
    options = ["0", "1", "2", "3"]
    near = {"0": 0.0, "1": 1.0, "2": 0.0, "3": 0.0}
    far = {"0": 0.0, "1": 0.0, "2": 0.0, "3": 1.0}
    assert jv.rps_loss(near, options, "0") < jv.rps_loss(far, options, "0")
    # Hand-computed: one level away puts one cut point wrong out of three.
    assert jv.rps_loss(near, options, "0") == pytest.approx(1 / 3)
    # Three levels away puts all three cut points wrong.
    assert jv.rps_loss(far, options, "0") == pytest.approx(1.0)
    assert jv.rps_loss({"0": 1.0, "1": 0.0, "2": 0.0, "3": 0.0}, options, "0") == 0.0


def test_a_score_question_uses_the_ranked_score_and_the_others_use_brier():
    options = ["0", "1", "2"]
    vector = {"0": 0.5, "1": 0.5, "2": 0.0}
    assert jv.loss_for(jv.SCORE, vector, options, "0") == \
        pytest.approx(jv.rps_loss(vector, options, "0"))
    assert jv.loss_for(jv.CHOICE, vector, options, "0") == \
        pytest.approx(jv.brier_loss(vector, options, "0"))
    assert jv.loss_for(jv.NOUL, vector, options, "0") == \
        pytest.approx(jv.brier_loss(vector, options, "0"))


def test_an_item_loss_is_the_mean_over_its_repeats_and_L_is_the_mean_over_items():
    """One item answered five times weighs the same as one answered once."""
    perfect = [decision(item="q1", repeat=r) for r in range(5)]
    # q2 is answered perfectly four times and disastrously once.
    mixed = [decision(item="q2", repeat=r) for r in range(4)]
    mixed.append(decision(item="q2", repeat=4, vector={"a": 0.0, "b": 1.0}))
    loss = jv.mean_item_loss(perfect + mixed)
    # q1 costs 0; q2 costs 2.0 on one of five repeats.
    assert loss == pytest.approx(((0.0) + (2.0 / 5)) / 2)


# ---------------------------------------------------------------- the Decision Score

def test_a_perfectly_calibrated_toy_scores_zero_ece_and_zero_decision_score():
    """Ten questions, always 0.8 on "a", and "a" is the label on exactly eight.

    Stated confidence 0.8 against an observed hit rate of 0.8 is the definition of a
    zero calibration gap. It is ALSO exactly the label prior on these items, so its
    Decision Score is 0: perfect calibration while knowing nothing about the state. This
    pair is the reason a calibration gap is never read on its own, and the reason every
    block here carries the guessing floor beside the accuracy.
    """
    rows = [decision(item=f"q{index}", vector={"a": 0.8, "b": 0.2},
                     label="a" if index < 8 else "b")
            for index in range(10)]
    assert jv.ece(rows) == pytest.approx(0.0, abs=1e-9)
    assert jv.ece_points(rows) == pytest.approx(0.0, abs=1e-7)
    assert jv.accuracy(rows) == pytest.approx(0.8)
    block = jv.summarize(rows)
    assert block["decision_score"] == pytest.approx(0.0)
    assert block["prior_accuracy"] == pytest.approx(0.8)
    assert block["recipe"] == jv.RECIPE_JEVALS


def test_a_system_that_reads_the_state_scores_well_above_zero():
    """A balanced set the system gets right at 0.9: the prior can only ever guess.

    Hand-computed: L = (0.9-1)^2 + (0.1-0)^2 = 0.02 on every item, and the 0.5/0.5 prior
    pays 0.25 + 0.25 = 0.5 on every item, so the Decision Score is 100*(1 - 0.04) = 96.
    """
    rows = [decision(item=f"q{index}",
                     vector=({"a": 0.9, "b": 0.1} if index < 5
                             else {"a": 0.1, "b": 0.9}),
                     label="a" if index < 5 else "b")
            for index in range(10)]
    block = jv.summarize(rows)
    assert block["accuracy"] == pytest.approx(1.0)
    assert block["prior_accuracy"] == pytest.approx(0.5)
    assert jv.mean_item_loss(rows) == pytest.approx(0.02)
    assert jv.prior_loss(rows) == pytest.approx(0.5)
    assert block["decision_score"] == pytest.approx(96.0)


def test_a_constant_guesser_that_answers_the_base_rates_scores_exactly_zero():
    """The definition of 0 on the scale, checked against the definition of the prior."""
    # Eight of ten labeled "a", so the base rates are 0.8 / 0.2.
    labels = ["a"] * 8 + ["b"] * 2
    rows = [decision(item=f"q{i}", label=label, vector={"a": 0.8, "b": 0.2})
            for i, label in enumerate(labels)]
    prior, options = jv.label_prior(rows)
    assert prior == {"a": pytest.approx(0.8), "b": pytest.approx(0.2)}
    assert options == ["a", "b"]
    assert jv.mean_item_loss(rows) == pytest.approx(jv.prior_loss(rows))
    assert jv.summarize(rows)["decision_score"] == pytest.approx(0.0)


def test_a_wrong_and_confident_system_scores_a_negative_decision_score():
    """Negative is shown, not clamped: that is the whole point of the scale."""
    labels = ["a"] * 8 + ["b"] * 2
    rows = [decision(item=f"q{i}", label=label, vector={"a": 0.02, "b": 0.98})
            for i, label in enumerate(labels)]
    score = jv.summarize(rows)["decision_score"]
    assert score is not None and score < 0
    # Hand-computed. An item labeled "a" costs (0.02-1)^2 + (0.98-0)^2 = 1.9208 and one
    # labeled "b" costs (0.02-0)^2 + (0.98-1)^2 = 0.0008, so
    # L = (8*1.9208 + 2*0.0008) / 10 = 1.5368.
    # The 0.8/0.2 prior costs 0.04+0.04 = 0.08 on an "a" and 0.64+0.64 = 1.28 on a "b",
    # so L_prior = (8*0.08 + 2*1.28) / 10 = 0.32.
    assert jv.mean_item_loss(rows) == pytest.approx(1.5368, abs=1e-9)
    assert jv.prior_loss(rows) == pytest.approx(0.32, abs=1e-9)
    assert score == pytest.approx(100.0 * (1 - 1.5368 / 0.32), abs=0.01)


def test_a_perfect_system_scores_one_hundred():
    labels = ["a"] * 7 + ["b"] * 3
    rows = [decision(item=f"q{i}", label=label) for i, label in enumerate(labels)]
    assert jv.summarize(rows)["decision_score"] == pytest.approx(100.0)


def test_the_guessing_floor_travels_with_every_block():
    """prior_accuracy is the base-rate answer's own accuracy on the same items."""
    labels = ["a"] * 7 + ["b"] * 3
    rows = [decision(item=f"q{i}", label=label) for i, label in enumerate(labels)]
    block = jv.summarize(rows)
    assert block["prior_accuracy"] == pytest.approx(0.7)
    assert block["loss_prior"] is not None
    assert block["accuracy"] == pytest.approx(1.0)


def test_a_prior_with_no_majority_on_a_yes_no_question_is_right_never():
    """A 50/50 prior has no pick under the recipe's yes/no rule, so its floor is 0."""
    rows = [decision(item=f"q{i}", qtype=jv.NOUL, options=("no", "yes"),
                     label="yes" if i % 2 else "no",
                     vector={"no": 0.5, "yes": 0.5})
            for i in range(10)]
    assert jv.prior_accuracy(rows) == 0.0


def test_the_decision_score_is_none_when_the_prior_cannot_lose():
    """Every item one label: the prior is perfect and the ratio has no denominator."""
    rows = [decision(item=f"q{i}", label="a") for i in range(4)]
    assert jv.prior_loss(rows) == 0.0
    assert jv.summarize(rows)["decision_score"] is None


# ---------------------------------------------------------------- ECE and its bins

def test_the_bin_index_is_the_published_formula_with_javascript_rounding():
    """min(9, floor(round(100*c)/10)), and 1.00 lands in the last bin.

    The rounding is the outer operation, so 0.099 rounds to 10 percent and lands in bin
    1 rather than in bin 0: the formula bins a ROUNDED percentage, not a truncated one.
    """
    assert jv.bin_index(0.0) == 0
    assert jv.bin_index(0.094) == 0
    assert jv.bin_index(0.099) == 1
    assert jv.bin_index(0.1) == 1
    assert jv.bin_index(0.95) == 9
    assert jv.bin_index(1.0) == 9
    # Python's round() breaks a tie to even, which would put these one bin low.
    assert jv.round_half_up(0.5) == 1
    assert jv.round_half_up(9.5) == 10
    assert jv.bin_index(0.005) == 0
    assert jv.bin_index(0.095) == 1


def test_the_reliability_table_keeps_ten_rows_whatever_ran():
    rows = spread([{"a": 0.95, "b": 0.05}])
    table = jv.reliability(rows)
    assert len(table) == jv.BINS
    assert [b["count"] for b in table] == [0] * 9 + [1]
    assert table[0]["accuracy"] is None and table[0]["confidence"] is None
    assert table[9]["lo"] == 0.9 and table[9]["hi"] == 1.0


def test_an_overconfident_system_has_a_calibration_gap_in_points():
    """Says 1.00, right half the time: ECE is 0.5, printed as 50 points."""
    rows = [decision(item=f"q{i}", vector={"a": 1.0, "b": 0.0},
                     label="a" if i % 2 == 0 else "b") for i in range(10)]
    assert jv.ece(rows) == pytest.approx(0.5)
    assert jv.ece_points(rows) == pytest.approx(50.0)


def test_a_system_with_no_probabilities_shows_no_calibration_gap_at_all():
    """A one-hot row has a confidence of 1.00 by construction; the board shows a dash."""
    rows = [decision(item=f"q{i}", one_hot=True, vector={"a": 1.0, "b": 0.0})
            for i in range(10)]
    assert jv.calibratable(rows) == []
    assert jv.ece(rows) is None
    block = jv.summarize(rows)
    assert block["ece_points"] is None
    assert block["one_hot"] == 10
    assert block["calibrated_over"] == 0
    # Still in the Decision Score and the accuracy, scored as one-hot.
    assert block["accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------- malformed, refused

def test_a_malformed_vector_is_refused_for_each_published_reason():
    options = ["a", "b"]
    assert jv.normalize_vector(None, options)[0] is None
    assert jv.normalize_vector({}, options)[0] is None
    assert "unknown" in jv.normalize_vector({"c": 1.0}, options)[1]
    assert "sum to 0" in jv.normalize_vector({"a": 0.0, "b": 0.0}, options)[1]
    assert jv.normalize_vector({"a": "x"}, options)[0] is None
    assert jv.normalize_vector({"a": float("nan")}, options)[0] is None
    assert jv.normalize_vector({"a": 2.0}, options)[0] is None
    assert jv.normalize_vector({"a": True, "b": 0.5}, options)[0] is None


def test_a_vector_that_sums_to_nearly_one_is_renormalized_both_directions():
    """Jev rounds to two decimals, so 0.99 happens; so does 1.02."""
    low, _ = jv.normalize_vector({"a": 0.5, "b": 0.49}, ["a", "b"])
    assert sum(low.values()) == pytest.approx(1.0)
    high, _ = jv.normalize_vector({"a": 0.6, "b": 0.42}, ["a", "b"])
    assert sum(high.values()) == pytest.approx(1.0)
    # An option the answer did not name is 0, not absent.
    partial, _ = jv.normalize_vector({"a": 0.5}, ["a", "b", "c"])
    assert partial == {"a": 1.0, "b": 0.0, "c": 0.0}


def test_a_malformed_answer_is_uniform_and_wrong_and_out_of_the_calibration():
    rows = [decision(item="q1", malformed=True, vector=None)]
    made = su.decision_for(
        {"id": "s", "set": "s", "type": jv.CHOICE, "instructions": "i",
         "options": ["a", "b"], "labels": {"q1": "a"}},
        {"id": "q1", "state": "x"}, 0, ["a", "b"], su.Answer(vector={"c": 1.0}))
    assert made["malformed"] and made["pick"] is None
    assert made["vector"] == {"a": 0.5, "b": 0.5}
    assert jv.is_correct(made) is False
    assert jv.calibratable([made]) == []
    # It is still scored, at the uniform distribution's loss.
    assert jv.decision_loss(made) == pytest.approx(0.5)
    assert rows  # the hand-built row above is only here for symmetry


def test_a_transport_failure_is_never_scored():
    rows = [decision(item="q1"), decision(item="q2", error="HTTP 502: nope")]
    assert len(jv.scored(rows)) == 1
    assert len(jv.failed(rows)) == 1
    block = jv.summarize(rows)
    assert block["decisions"] == 1 and block["failed"] == 1
    assert block["accuracy"] == pytest.approx(1.0)


def test_a_yes_no_answer_of_exactly_half_has_no_pick_and_counts_as_wrong():
    made = su.decision_for(
        {"id": "s", "set": "s", "type": jv.NOUL, "instructions": "i",
         "options": ["no", "yes"], "labels": {"q1": "yes"}},
        {"id": "q1", "state": "x"}, 0, ["no", "yes"],
        su.Answer(vector={"no": 0.5, "yes": 0.5}))
    assert made["pick"] is None
    assert made["confidence"] is None
    assert not made["malformed"]
    assert jv.is_correct(made) is False


# ---------------------------------------------------------------- ties

def test_a_choice_tie_goes_to_what_the_system_said_then_to_the_presented_order():
    vector = {"a": 0.5, "b": 0.5}
    assert jv.pick_from_vector(vector, ["a", "b"], presented=["b", "a"],
                               qtype=jv.CHOICE, stated="b") == "b"
    assert jv.pick_from_vector(vector, ["a", "b"], presented=["b", "a"],
                               qtype=jv.CHOICE) == "b"
    assert jv.pick_from_vector(vector, ["a", "b"], presented=["a", "b"],
                               qtype=jv.CHOICE) == "a"


def test_a_score_tie_goes_to_the_lower_level():
    options = ["0", "1", "2"]
    vector = {"0": 0.4, "1": 0.4, "2": 0.2}
    assert jv.pick_from_vector(vector, options, presented=options,
                               qtype=jv.SCORE) == "0"


# ---------------------------------------------------------------- gate and hand-off

def test_the_published_gates_are_the_frozen_ones_jevals_states():
    assert jv.PUBLISHED_GATES[jv.CHOICE] == 0.96
    assert jv.PUBLISHED_GATES[jv.NOUL] == 0.91
    assert jv.PUBLISHED_GATES[jv.SCORE] is None


def test_hand_off_at_95_needs_a_hundred_decisions_and_finds_the_lowest_threshold():
    # 120 decisions at 0.97 with six wrong (95% exactly), and 30 at 0.60 all wrong.
    rows = []
    for index in range(120):
        right = index >= 6
        rows.append(decision(item=f"hi{index}", vector={"a": 0.97, "b": 0.03},
                             label="a" if right else "b"))
    for index in range(30):
        rows.append(decision(item=f"lo{index}", vector={"a": 0.6, "b": 0.4},
                             label="b"))
    hand = jv.handoff(rows)
    assert hand is not None
    assert hand["n"] == 120
    assert hand["accuracy"] == pytest.approx(114 / 120)
    # The share is over ALL decisions, refused and malformed included.
    assert hand["share"] == pytest.approx(120 / 150)
    assert hand["threshold"] <= 0.97


def test_hand_off_is_none_when_accuracy_never_reaches_95_percent():
    rows = [decision(item=f"q{i}", vector={"a": 0.9, "b": 0.1},
                     label="a" if i % 2 else "b") for i in range(200)]
    assert jv.handoff(rows) is None
    assert jv.summarize(rows)["handoff_95"] is None


def test_hand_off_is_none_when_fewer_than_a_hundred_decisions_clear_any_threshold():
    rows = [decision(item=f"q{i}") for i in range(99)]
    assert jv.handoff(rows) is None


def test_the_gate_block_reports_the_published_threshold_and_this_runs_coverage():
    rows = [decision(item=f"q{i}", vector={"a": 0.97, "b": 0.03}) for i in range(60)]
    rows += [decision(item=f"r{i}", vector={"a": 0.5, "b": 0.5}, pick="a")
             for i in range(40)]
    block = jv.summarize(rows)
    assert block["gate"]["threshold"] == 0.96
    assert block["gate"]["source"] == jv.PUBLISHED_GATE_SOURCE
    assert block["gate"]["n"] == 60
    assert block["gate"]["coverage"] == pytest.approx(0.6)


def test_a_score_set_records_that_its_primitive_has_no_published_gate():
    rows = [decision(item=f"q{i}", qtype=jv.SCORE, options=("0", "1", "2"),
                     label="0", vector={"0": 0.9, "1": 0.1, "2": 0.0})
            for i in range(10)]
    block = jv.summarize(rows)
    assert block["gate"] == {"threshold": None,
                             "source": jv.PUBLISHED_GATE_SOURCE,
                             "coverage": None, "n": 0, "accuracy": None}


def test_the_local_gate_is_the_same_rule_over_one_system():
    rows = [decision(item=f"q{i}", vector={"a": 0.99, "b": 0.01}) for i in range(150)]
    assert jv.gate_local(rows) == 0.0        # nothing is wrong, so the lowest t wins
    rows += [decision(item=f"w{i}", vector={"a": 0.2, "b": 0.8}, label="a")
             for i in range(50)]
    assert jv.gate_local(rows) is not None


# ---------------------------------------------------------------- flips and swing

def test_the_pick_flip_rate_is_any_change_over_all_the_repeats():
    stable = [decision(item="q1", repeat=r) for r in range(5)]
    flipped = [decision(item="q2", repeat=r) for r in range(4)]
    flipped.append(decision(item="q2", repeat=4, vector={"a": 0.1, "b": 0.9}))
    rate, over = jv.pick_flips(stable + flipped)
    assert over == 2 and rate == pytest.approx(0.5)


def test_a_question_with_fewer_than_two_readable_repeats_is_out_of_the_flip_rate():
    rows = [decision(item="q1", repeat=0),
            decision(item="q1", repeat=1, error="boom"),
            decision(item="q1", repeat=2, malformed=True, vector=None)]
    rate, over = jv.pick_flips(rows)
    assert (rate, over) == (None, 0)


def test_the_boards_repeat_flip_rate_compares_only_repeats_zero_and_one():
    rows = [decision(item="q1", repeat=0), decision(item="q1", repeat=1),
            decision(item="q1", repeat=2, vector={"a": 0.1, "b": 0.9})]
    assert jv.repeat_flip_rate(rows) == (0.0, 1)
    assert jv.pick_flips(rows)[0] == pytest.approx(1.0)


def test_the_order_flip_rate_is_choice_only_and_uses_the_four_distinct_orders():
    choice = [decision(item="q1", repeat=r) for r in (0, 2, 3)]
    choice.append(decision(item="q1", repeat=4, vector={"a": 0.1, "b": 0.9}))
    rate, over = jv.order_flip_rate(choice)
    assert over == 1 and rate == pytest.approx(1.0)
    scores = [decision(item="q1", repeat=r, qtype=jv.SCORE, options=("0", "1"),
                       label="0") for r in (0, 2, 3, 4)]
    assert jv.order_flip_rate(scores) == (None, 0)


def test_the_confidence_swing_reports_the_worst_question_and_the_mean():
    rows = [decision(item="q1", repeat=0, vector={"a": 0.55, "b": 0.45}),
            decision(item="q1", repeat=1, vector={"a": 0.99, "b": 0.01}),
            decision(item="q2", repeat=0, vector={"a": 0.90, "b": 0.10}),
            decision(item="q2", repeat=1, vector={"a": 0.92, "b": 0.08})]
    swing = jv.confidence_swing(rows)
    assert swing["question"] == "q1"
    assert swing["max"] == pytest.approx(0.44)
    assert swing["mean"] == pytest.approx((0.44 + 0.02) / 2)
    assert swing["over"] == 2
    assert jv.summarize(rows)["confidence_swing"]["question"] == "q1"


def test_a_block_carries_both_the_first_class_flip_rate_and_the_boards_two():
    rows = [decision(item="q1", repeat=r) for r in range(5)]
    block = jv.summarize(rows)
    for key in ("pick_flip_rate", "pick_flip_over", "repeat_flip_rate",
                "order_flip_rate", "confidence_swing"):
        assert key in block


# ---------------------------------------------------------------- latency and cost

def test_the_percentile_is_nearest_rank_so_a_p95_really_happened():
    assert jv.percentile([10, 20, 30, 40], 0.5) == 20
    assert jv.percentile([10, 20, 30, 40], 0.95) == 40
    assert jv.percentile([], 0.5) is None


def test_questions_per_second_is_over_the_runs_own_clock_and_never_invented():
    assert jv.questions_per_second(100, 50.0) == pytest.approx(2.0)
    assert jv.questions_per_second(100, 0) is None
    assert jv.questions_per_second(0, 10.0) is None


def test_cost_is_a_posted_rate_over_reported_tokens_or_zero():
    rows = [decision(item="q1", tokens_in=1_000_000, tokens_out=500_000)]
    assert jv.cost_usd(jv.tokens(rows), 0.042, 0.0) == pytest.approx(0.042)
    assert jv.cost_usd(jv.tokens(rows)) == 0.0
    block = jv.summarize(rows, input_usd_per_mtok=0.042)
    assert block["cost_usd"] == pytest.approx(0.042)
    assert block["usd_per_1k_decisions"] == pytest.approx(42.0)


# ---------------------------------------------------------------- gold distributions

def test_the_gold_block_is_absent_when_a_set_ships_no_gold_distribution():
    assert jv.gold_block([decision(item="q1")]) == {}
    assert "vs_gold" not in jv.summarize([decision(item="q1")])


def test_soft_accuracy_is_the_gold_probability_of_the_pick():
    rows = [decision(item="q1", options=("a", "b"), label="a",
                     vector={"a": 0.9, "b": 0.1}, gold={"a": 0.6, "b": 0.4}),
            decision(item="q2", options=("a", "b"), label="a",
                     vector={"a": 0.1, "b": 0.9}, gold={"a": 0.6, "b": 0.4})]
    assert jv.soft_accuracy(rows) == pytest.approx((0.6 + 0.4) / 2)
    assert jv.accuracy(rows) == pytest.approx(0.5)


def test_total_variation_and_kl_reward_matching_the_teachers_spread():
    matched = [decision(item="q1", vector={"a": 0.6, "b": 0.4},
                        gold={"a": 0.6, "b": 0.4})]
    committed = [decision(item="q1", vector={"a": 1.0, "b": 0.0},
                          gold={"a": 0.6, "b": 0.4})]
    assert jv.total_variation(matched) == pytest.approx(0.0)
    assert jv.total_variation(committed) == pytest.approx(0.4)
    assert jv.kl_from_gold(matched) == pytest.approx(0.0, abs=1e-12)
    # A confident miss is large and finite, because the prediction is floored.
    assert jv.kl_from_gold(committed) > 1.0
    assert jv.brier_from_gold(matched) == pytest.approx(0.0)
    block = jv.summarize(matched)
    assert block["vs_gold"]["over"] == 1
    assert block["vs_gold"]["kl_floor"] == jv.KL_FLOOR
    assert "AINode's definitions" in block["vs_gold"]["definition"]


# ---------------------------------------------------------------- JevBench's names

def test_the_guessing_floor_carries_both_boards_names_for_it():
    """Jevals calls it the label prior, JevBench calls it majority_class_accuracy."""
    labels = ["a"] * 7 + ["b"] * 3
    rows = [decision(item=f"q{i}", label=label) for i, label in enumerate(labels)]
    block = jv.summarize(rows)
    assert block["prior_accuracy"] == pytest.approx(0.7)
    assert block["majority_class_accuracy"] == block["prior_accuracy"]


def test_ordinal_mae_is_the_probability_weighted_level_against_the_labeled_one():
    """JevBench's name and rule, reported beside argmax accuracy and not instead of it."""
    options = ("0", "1", "2", "3")
    # All the mass on level 2 with the label at 0: the expected level is 2.0.
    far = decision(item="q1", qtype=jv.SCORE, options=options, label="0",
                   vector={"0": 0.0, "1": 0.0, "2": 1.0, "3": 0.0})
    assert jv.ordinal_mae([far]) == pytest.approx(2.0)
    # Split either side of the label: argmax is wrong, the expectation is close.
    split = decision(item="q2", qtype=jv.SCORE, options=options, label="1",
                     vector={"0": 0.5, "1": 0.0, "2": 0.5, "3": 0.0})
    assert jv.ordinal_mae([split]) == pytest.approx(0.0)
    assert jv.accuracy([split]) == 0.0
    assert jv.ordinal_mae([decision(item="q3")]) is None
    block = jv.summarize([far, split])
    assert block["ordinal_mae"] is not None
    assert block["within_one_level"] is not None


def test_ordinal_mae_and_within_one_level_are_absent_from_a_non_score_block():
    block = jv.summarize([decision(item="q1")])
    assert "ordinal_mae" not in block and "within_one_level" not in block


def test_schema_validity_uses_jevbenchs_two_sum_tolerances():
    """Its headline renormalizes inside 2 percent; its strict column uses 0.001."""
    doc = {"id": "s", "set": "s", "type": jv.CHOICE, "instructions": "i",
           "options": ["a", "b"], "labels": {"q1": "a"}}
    # 0.99 is three-decimal rounding: inside the loose band, outside the strict one.
    rounded = su.decision_for(doc, {"id": "q1", "state": "x"}, 0, ["a", "b"],
                              su.Answer(vector={"a": 0.79, "b": 0.20}))
    assert rounded["sum_before_normalize"] == pytest.approx(0.99)
    assert jv.schema_validity([rounded]) == pytest.approx(1.0)
    assert jv.schema_validity([rounded], jv.SUM_TOLERANCE_STRICT) == 0.0
    block = jv.summarize([rounded])
    assert block["schema_validity"] == pytest.approx(1.0)
    assert block["schema_validity_strict"] == pytest.approx(0.0)

    # Well outside the band, and a malformed answer, are invalid under both.
    wild = su.decision_for(doc, {"id": "q2", "state": "x"}, 0, ["a", "b"],
                           su.Answer(vector={"a": 0.4, "b": 0.2}))
    assert jv.schema_validity([wild]) == 0.0
    bad = su.decision_for(doc, {"id": "q3", "state": "x"}, 0, ["a", "b"],
                          su.Answer(vector={"z": 1.0}))
    assert jv.schema_validity([bad]) == 0.0


def test_a_one_hot_answer_counts_as_schema_valid_because_it_stated_no_sum():
    doc = {"id": "s", "set": "s", "type": jv.CHOICE, "instructions": "i",
           "options": ["a", "b"], "labels": {"q1": "a"}}
    made = su.decision_for(doc, {"id": "q1", "state": "x"}, 0, ["a", "b"],
                           su.Answer(stated="a", one_hot=True))
    assert made["one_hot"] and made["sum_before_normalize"] is None
    assert jv.schema_validity([made]) == pytest.approx(1.0)


def test_each_transport_declares_where_its_probabilities_came_from():
    """JevBench's `native` / `verbalized` vocabulary, plus the value it has no name for."""
    assert su.SystemOneTransport("http://k/v1/systemone").probability_source == "native"
    assert su.DecideTransport("http://n/v1/decide").probability_source == "logprob"
    protocol = su.DecideTransport("http://n/v1/decide").protocol()
    assert protocol["probability_source"] == "logprob"
    assert protocol["cost_basis"] == "no_billable_account_no_price_given"
    priced = su.SystemOneTransport("http://k/v1/systemone", input_usd_per_mtok=0.042)
    assert priced.cost_basis() == "derived_usage_times_tariff"


def test_the_notes_say_which_probability_source_a_row_can_be_compared_to():
    docs = toy_docs()
    notes = su.build_notes(su.DecideTransport("http://n/v1/decide"), docs, [], 0.0, 5)
    text = " ".join(notes)
    assert "Probability source: logprob" in text
    assert "native" in text and "verbalized" in text


def test_within_one_level_is_a_score_only_correction():
    options = ("0", "1", "2", "3")
    near = decision(item="q1", qtype=jv.SCORE, options=options, label="0",
                    vector={"0": 0.2, "1": 0.8, "2": 0.0, "3": 0.0})
    far = decision(item="q2", qtype=jv.SCORE, options=options, label="0",
                   vector={"0": 0.0, "1": 0.0, "2": 0.0, "3": 1.0})
    assert jv.within_one_level([near, far]) == pytest.approx(0.5)
    assert jv.accuracy([near, far]) == 0.0
    assert jv.within_one_level([decision(item="q1")]) is None


# ---------------------------------------------------------------- mixed sets

def test_a_mixed_set_is_broken_down_by_primitive_and_averages_their_scores():
    rows = []
    for index in range(4):
        rows.append(decision(item=f"c{index}", qtype=jv.CHOICE,
                             label="a" if index else "b"))
        rows.append(decision(item=f"n{index}", qtype=jv.NOUL,
                             options=("no", "yes"),
                             label="yes" if index else "no",
                             vector={"no": 0.1, "yes": 0.9} if index
                             else {"no": 0.9, "yes": 0.1}))
    block = jv.summarize(rows)
    assert block["type"] == "mixed"
    assert set(block["types"]) == {jv.NOUL, jv.CHOICE}
    scores = [block["types"][t]["decision_score"] for t in block["types"]]
    assert block["decision_score"] == pytest.approx(
        round(sum(scores) / len(scores), 2))
    assert "plain mean" in block["decision_score_is"]
    # Accuracy and latency are over everything, because those do carry across.
    assert block["accuracy"] == pytest.approx(1.0)
    assert block["decisions"] == 8


def test_two_questions_with_different_option_lists_are_two_answer_spaces():
    """The Decision Score's baseline is per answer space, so they must not be pooled.

    Pooling them builds a label prior over an option list one of the two does not have,
    which is how a baseline that is supposed to define 0 ends up defining something else.
    """
    four = [decision(item=f"a{i}", options=("a", "b", "c", "d"),
                     label="a" if i % 2 else "b",
                     vector={"a": 0.5, "b": 0.5, "c": 0.0, "d": 0.0})
            for i in range(4)]
    two = [decision(item=f"t{i}", options=("x", "y"), label="x" if i % 2 else "y",
                    vector={"x": 0.5, "y": 0.5}) for i in range(4)]
    spaces = jv.answer_spaces(four + two)
    assert len(spaces) == 2
    block = jv.summarize(four + two)
    assert set(block["spaces"]) == set(spaces)
    # Both halves answer their own base rates exactly, so every space scores 0 and so
    # does the set. Pooled into one prior they would not.
    assert all(b["decision_score"] == pytest.approx(0.0)
               for b in block["spaces"].values())
    assert block["decision_score"] == pytest.approx(0.0)
    for name, sub in block["spaces"].items():
        assert sub["options"], name


def test_an_answer_space_is_named_by_the_question_file_when_it_says_so():
    rows = [decision(item="q1", options=("a", "b"), label="a"),
            decision(item="q2", options=("x", "y"), label="x")]
    rows[0]["space"] = "customer_service/action"
    rows[1]["space"] = "invoice_processing/duplicate"
    assert set(jv.answer_spaces(rows)) == {"customer_service/action",
                                           "invoice_processing/duplicate"}
    # With no name, first-seen order per primitive.
    for row in rows:
        row["space"] = None
    assert list(jv.answer_spaces(rows)) == ["choice#1", "choice#2"]


def test_a_single_answer_space_set_gets_no_breakdown_at_all():
    rows = [decision(item=f"q{i}", label="a" if i % 2 else "b") for i in range(4)]
    block = jv.summarize(rows)
    assert "spaces" not in block and "types" not in block
    assert block["type"] == jv.CHOICE


@pytest.mark.parametrize("name", st.SUITES)
def test_the_label_prior_scores_exactly_zero_on_every_committed_set(name, monkeypatch,
                                                                   tmp_path):
    """The strongest single check on the scale, over the real manifests.

    The baseline that DEFINES 0 has to land on 0 for each of the four sets and each of
    their primitives, not only on a toy. Built from the manifests alone, so it needs no
    download: a manifest carries the target per item, which is all a label prior is.
    """
    doc = st.load_manifest(name)
    per_space = {}
    for item in doc["items"]:
        spec = st.manifest_spec(doc, item)
        key = (spec["type"], tuple(spec["options"]))
        per_space.setdefault(key, []).append(
            spec["options"][int(item["target"])])
    decisions = []
    for (qtype, options), labels in per_space.items():
        prior = {option: labels.count(option) / len(labels) for option in options}
        space = f"{qtype}/{len(options)}/{sorted(options)[0]}"
        for index, label in enumerate(labels):
            row = decision(item=f"{space}-{index}", options=options, label=label,
                           vector=dict(prior), qtype=qtype)
            row["space"] = space
            decisions.append(row)
    block = jv.summarize(decisions)
    scores = ([b["decision_score"] for b in block["spaces"].values()]
              if "spaces" in block else [block["decision_score"]])
    for score in scores:
        assert score == pytest.approx(0.0), (name, scores)
    assert block["decision_score"] == pytest.approx(0.0)
    # And the floor equals what the prior actually scores, which is the point of it.
    assert block["prior_accuracy"] is not None


def test_a_mixed_block_leaves_the_pooled_losses_null_because_they_are_not_comparable():
    """The losses live in the answer-space blocks; the `types` map is a roll-up."""
    rows = [decision(item="c1", qtype=jv.CHOICE),
            decision(item="s1", qtype=jv.SCORE, options=("0", "1", "2"), label="0",
                     vector={"0": 1.0, "1": 0.0, "2": 0.0})]
    block = jv.summarize(rows)
    assert block["loss"] is None and block["loss_prior"] is None
    score_space = next(b for b in block["spaces"].values()
                       if b["type"] == jv.SCORE)
    assert score_space["loss"] is not None
    assert block["types"][jv.SCORE]["spaces"] == 1
    assert "loss" not in block["types"][jv.SCORE]


def test_summarize_sets_skips_a_set_with_no_decisions_and_keeps_the_asked_order():
    rows = spread([{"a": 0.9, "b": 0.1}], set_name="second")
    out = jv.summarize_sets(rows, ["first", "second"])
    assert list(out) == ["second"]


def test_the_mean_over_sets_needs_every_set_to_have_a_score():
    blocks = {"a": {"decision_score": 10.0}, "b": {"decision_score": 20.0}}
    assert jv.mean_decision_score(blocks)["mean_decision_score"] == pytest.approx(15.0)
    blocks["c"] = {"decision_score": None}
    out = jv.mean_decision_score(blocks)
    assert out["mean_decision_score"] is None
    assert out["sets"] == ["a", "b", "c"] and out["scored"] == ["a", "b"]


# ---------------------------------------------------------------- option order

def test_repeats_zero_and_one_share_one_order_and_two_three_four_get_their_own():
    assert [su.order_index(r) for r in range(5)] == [0, 0, 1, 2, 3]


def test_choice_options_are_shuffled_deterministically_and_the_others_are_not():
    options = list("abcdefgh")
    first = su.presented_options(options, "q1", 42, 0, jv.CHOICE)
    assert first == su.presented_options(options, "q1", 42, 0, jv.CHOICE)
    assert sorted(first) == options
    assert first != su.presented_options(options, "q1", 42, 1, jv.CHOICE)
    assert first != su.presented_options(options, "q2", 42, 0, jv.CHOICE)
    for qtype in (jv.NOUL, jv.SCORE):
        assert su.presented_options(options, "q1", 42, 3, qtype) == options


# ---------------------------------------------------------------- the manifests

def test_every_committed_manifest_loads_and_declares_what_a_reader_needs():
    for suite_id in st.SUITES:
        doc = st.load_manifest(suite_id)
        assert doc["id"] == suite_id
        assert doc["license"]
        assert doc["hf_revision"]
        assert len(doc["items"]) == int(doc["n_items"])
        summary = st.manifest_summary(suite_id)
        assert summary["recipe_of_record"]


def test_the_three_jevals_manifests_are_the_published_sets_at_three_hundred_items():
    for suite_id, primitive, options in (("pubmedqa", "noul", 2),
                                         ("banking77", "choice", 77),
                                         ("helpsteer2", "score", 5)):
        doc = st.load_manifest(suite_id)
        assert doc["version"] == "0.1.0"
        assert doc["primitive"] == primitive
        assert len(doc["options"]) == options
        assert doc["n_items"] == 300
        assert doc["seed"] == 20260918
        assert doc["length_cap"] == 6000


def test_the_fourth_set_is_the_mixed_one_already_in_the_wire_shape():
    doc = st.load_manifest("typed-decisions")
    assert doc["primitive"] == "mixed"
    assert doc["license"] == "Apache-2.0"
    assert doc["n_items"] == 2000 and doc["n_cases"] == 400
    assert doc["gold"] == "distribution"
    assert set(doc["question_schemas"]) == {"agent_trace_observability",
                                            "customer_service",
                                            "invoice_processing",
                                            "security_incidents"}
    assert all(item.get("gold") for item in doc["items"])
    # Its state hashes are ours, not upstream's, and it says so.
    assert doc["state_hash_source"] == "ainode"


def test_a_manifest_item_resolves_its_question_through_its_own_group():
    doc = st.load_manifest("typed-decisions")
    action = next(i for i in doc["items"]
                  if i["item_id"].endswith(":action")
                  and i["group"] == "customer_service")
    spec = st.manifest_spec(doc, action)
    assert spec["type"] == "choice"
    assert len(spec["options"]) == 5           # four in the other workflow
    other = next(i for i in doc["items"]
                 if i["item_id"].endswith(":action")
                 and i["group"] == "agent_trace_observability")
    assert len(st.manifest_spec(doc, other)["options"]) == 4


def test_a_manifest_naming_an_undeclared_group_is_a_load_error():
    doc = st.load_manifest("typed-decisions")
    with pytest.raises(st.SetError) as exc:
        st.manifest_spec(doc, {"item_id": "x:action", "group": "nope"})
    assert "nope" in str(exc.value)


def test_the_contamination_table_names_a_system_and_its_primary_source():
    assert "banking77" in st.CONTAMINATION
    for entries in st.CONTAMINATION.values():
        for entry in entries:
            assert entry["system"] and entry["evidence"]
            assert entry["source"].startswith("https://")
    assert any("Kev" in e["system"] for e in st.CONTAMINATION["banking77"])


# ---------------------------------------------------------------- the state bytes

def test_the_state_is_hashed_as_compact_insertion_ordered_real_utf8():
    state = {"b": "é", "a": 1}
    assert st.state_json(state) == '{"b":"é","a":1}'
    assert st.state_sha256(state) == st.state_sha256({"b": "é", "a": 1})
    # Key ORDER is part of the bytes, so a reordered state is a different state.
    assert st.state_sha256(state) != st.state_sha256({"a": 1, "b": "é"})
    # A string state is passed through verbatim.
    assert st.state_json("already text") == "already text"


def test_the_published_state_hash_of_one_known_item_is_reproduced():
    """The banking77 state object is {"message": <row.text>}, recovered from the hash.

    If this ever fails, the state construction in ``sets.py`` has drifted from what the
    boards scored and no number from this bench is comparable to a board number.
    """
    doc = st.load_manifest("banking77")
    first = next(i for i in doc["items"] if i["item_id"] == "banking77-2")
    text = "I ordered a card but it has not arrived. Help please!"
    assert st.state_sha256({"message": text}) == first["state_sha256"]
    assert first["state_sha256"] == \
        "e74cb88e43831226c2d2e746f4487999b105bc98abf54ccd2cf3734e03c781ba"


# ---------------------------------------------------------------- question files

def question_file(tmp_path, name="set.json", **over):
    doc = {"id": "toy", "set": "toy", "type": "choice",
           "instructions": "Which one?", "criteria": {"a": "the a", "b": "the b"},
           "options": ["a", "b"], "seed": 7,
           "questions": [{"id": "t1", "state": {"text": "one"}},
                         {"id": "t2", "state": {"text": "two"}}],
           "labels": {"t1": "a", "t2": "b"}}
    doc.update(over)
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return path, doc


def test_a_question_file_loads_and_its_labels_live_outside_the_questions(tmp_path):
    path, _ = question_file(tmp_path)
    doc = st.load_questions(path)
    assert doc["labels"] == {"t1": "a", "t2": "b"}
    for question in doc["questions"]:
        assert "label" not in question


def test_a_question_carrying_an_answer_key_is_a_load_error(tmp_path):
    for bad in ("label", "expected", "passingAnswer", "gold_label", "solution"):
        path, doc = question_file(tmp_path, name=f"{bad}.json")
        doc["questions"][0][bad] = "a"
        path.write_text(json.dumps(doc))
        with pytest.raises(st.SetError) as exc:
            st.load_questions(path)
        assert bad in str(exc.value)


def test_a_state_may_hold_a_field_whose_name_looks_like_an_answer_key(tmp_path):
    """The state is the caller's own data; the guard is about what we append to it."""
    path, doc = question_file(tmp_path)
    doc["questions"][0]["state"] = {"label": "a sticker on a parcel"}
    path.write_text(json.dumps(doc))
    assert st.load_questions(path)["questions"][0]["state"]["label"]


def test_a_gold_distribution_is_allowed_and_is_not_an_answer_key(tmp_path):
    path, doc = question_file(tmp_path)
    doc["questions"][0]["gold"] = {"a": 0.7, "b": 0.3}
    path.write_text(json.dumps(doc))
    assert st.load_questions(path)["questions"][0]["gold"] == {"a": 0.7, "b": 0.3}


def test_a_gold_for_an_option_that_does_not_exist_is_a_load_error(tmp_path):
    path, doc = question_file(tmp_path)
    doc["questions"][0]["gold"] = {"z": 1.0}
    path.write_text(json.dumps(doc))
    with pytest.raises(st.SetError):
        st.load_questions(path)


@pytest.mark.parametrize("mutate,message", [
    (lambda d: d.pop("labels"), "labels"),
    (lambda d: d.pop("questions"), "questions"),
    (lambda d: d.update(type="guess"), "type"),
    (lambda d: d.update(options=["a"]), "two options"),
    (lambda d: d.update(options=["a", "a"]), "repeats an option"),
    (lambda d: d.update(labels={"t1": "zzz", "t2": "b"}), "not one of its options"),
    (lambda d: d["questions"].append({"id": "t1", "state": "x"}), "appears twice"),
    (lambda d: d["questions"].append({"id": "t3", "state": ""}), "no state"),
])
def test_a_malformed_question_file_is_a_load_error_naming_it(tmp_path, mutate, message):
    path, doc = question_file(tmp_path)
    mutate(doc)
    path.write_text(json.dumps(doc))
    with pytest.raises(st.SetError) as exc:
        st.load_questions(path)
    assert message in str(exc.value)


def test_a_question_file_whose_state_does_not_match_its_own_hash_is_refused(tmp_path):
    path, doc = question_file(tmp_path)
    doc["questions"][0]["state_sha256"] = "0" * 64
    path.write_text(json.dumps(doc))
    with pytest.raises(st.SetError) as exc:
        st.load_questions(path)
    assert "state hash" in str(exc.value)


def test_a_mixed_question_file_takes_its_type_per_question(tmp_path):
    path, doc = question_file(tmp_path, type="mixed")
    doc.pop("options")
    doc.pop("instructions")
    doc.pop("criteria")
    doc["questions"] = [
        {"id": "t1", "state": "x", "type": "noul", "instructions": "yes?",
         "options": ["no", "yes"], "criteria": {"false": "n", "true": "y"}},
        {"id": "t2", "state": "y", "type": "score", "instructions": "how much?",
         "options": ["0", "1", "2"], "criteria": ["low", "mid", "high"]},
    ]
    doc["labels"] = {"t1": "yes", "t2": "1"}
    path.write_text(json.dumps(doc))
    loaded = st.load_questions(path)
    assert st.question_spec(loaded, loaded["questions"][0])["type"] == "noul"
    assert st.question_spec(loaded, loaded["questions"][1])["options"] == ["0", "1", "2"]


def test_a_missing_download_says_which_command_fetches_it(tmp_path, monkeypatch):
    monkeypatch.setenv(st.ENV_CACHE, str(tmp_path))
    with pytest.raises(st.SetError) as exc:
        st.load_suite_questions(["pubmedqa"])
    assert "decide download pubmedqa" in str(exc.value)


def test_an_unknown_suite_is_refused_by_name():
    with pytest.raises(st.SetError) as exc:
        st.load_suite_questions(["nope"])
    assert "nope" in str(exc.value)


# ---------------------------------------------------------------- build_questions

def test_building_a_question_file_verifies_every_state_hash():
    doc = {"id": "banking77", "version": "0.1.0", "primitive": "choice",
           "dataset": "mteb/banking77", "config": "default", "split": "test",
           "hf_revision": "abc", "license": "CC-BY-4.0",
           "instructions": "Which intent?", "criteria": {"a": None, "b": None},
           "options": ["a", "b"], "state_fields": ["text"], "seed": 1, "n_items": 1,
           "items": [{"item_id": "banking77-0", "row_idx": 0, "target": 1,
                      "state_sha256": st.state_sha256({"message": "hello"})}]}
    built = st.build_questions(doc, {0: {"text": "hello"}})
    assert built["labels"] == {"banking77-0": "b"}
    assert built["questions"][0]["state"] == {"message": "hello"}
    assert "label" not in built["questions"][0]

    doc["items"][0]["state_sha256"] = "0" * 64
    with pytest.raises(st.SetError) as exc:
        st.build_questions(doc, {0: {"text": "hello"}})
    assert "hashes to" in str(exc.value)


def test_a_row_missing_from_the_download_is_an_error_naming_the_item():
    doc = {"id": "banking77", "version": "0.1.0", "primitive": "choice",
           "dataset": "d", "config": "c", "split": "s", "hf_revision": "r",
           "license": "l", "instructions": "i", "options": ["a", "b"],
           "state_fields": ["text"], "seed": 1, "n_items": 1,
           "items": [{"item_id": "banking77-9", "row_idx": 9, "target": 0,
                      "state_sha256": "0" * 64}]}
    with pytest.raises(st.SetError) as exc:
        st.build_questions(doc, {})
    assert "banking77-9" in str(exc.value)


def test_the_rate_limited_rows_api_is_retried_and_then_explained(monkeypatch):
    import urllib.error

    calls = {"n": 0}

    def always_429(url, timeout=0):
        calls["n"] += 1
        raise urllib.error.HTTPError(url, 429, "slow down", {}, None)

    monkeypatch.setattr(st.urllib.request, "urlopen", always_429)
    with pytest.raises(st.SetError) as exc:
        st._get("https://example.invalid/rows", sleep=lambda _s: None)
    assert calls["n"] == st.HTTP_RETRIES
    assert "429" in str(exc.value) and "rate limiting" in str(exc.value)


def test_a_404_is_not_retried():
    import urllib.error

    calls = {"n": 0}

    def gone(url, timeout=0):
        calls["n"] += 1
        raise urllib.error.HTTPError(url, 404, "no", {}, None)

    original = st.urllib.request.urlopen
    st.urllib.request.urlopen = gone
    try:
        with pytest.raises(st.SetError):
            st._get("https://example.invalid/x", sleep=lambda _s: None)
    finally:
        st.urllib.request.urlopen = original
    assert calls["n"] == 1


# ---------------------------------------------------------------- the two transports

def toy_doc(qtype="choice", options=("a", "b"), criteria=None, seed=7):
    return {"id": "toy", "set": "toy", "type": qtype, "instructions": "Which one?",
            "criteria": criteria if criteria is not None else {"a": "the a",
                                                               "b": "the b"},
            "options": list(options), "seed": seed,
            "questions": [{"id": "t1", "state": {"text": "one"}}],
            "labels": {"t1": "a"}}


def test_the_decide_request_sends_the_state_as_canonical_text_and_explicit_options():
    doc = toy_doc()
    transport = su.DecideTransport("http://n/v1/decide", model="m", api_key="k")
    request = transport.request(doc, doc["questions"][0], ["b", "a"])
    assert request.url == "http://n/v1/decide"
    assert request.payload["state"] == '{"text":"one"}'
    block = request.payload["questions"]["decision"]
    assert block["options"] == ["b", "a"]
    assert block["question"].startswith("Which one?")
    assert "b: the b" in block["question"] and "a: the a" in block["question"]
    assert request.payload["model"] == "m"
    assert request.headers["Authorization"] == "Bearer k"
    assert su.wire_leaks(request.payload) == []


def test_the_decide_request_appends_nothing_when_no_option_is_described():
    doc = toy_doc(criteria={"a": None, "b": None})
    transport = su.DecideTransport("http://n/v1/decide")
    block = transport.request(doc, doc["questions"][0], ["a", "b"])\
        .payload["questions"]["decision"]
    assert block["question"] == "Which one?"


def test_the_decide_response_becomes_a_scored_decision():
    doc = toy_doc()
    transport = su.DecideTransport("http://n/v1/decide")
    payload = {"model": "served/model", "node": "Spark-1", "latency_ms": 12,
               "decisions": {"decision": {"answer": "a", "confidence": 0.8,
                                          "distribution": {"a": 0.8, "b": 0.2},
                                          "latency_ms": 11}},
               "usage": {"prompt_tokens": 40, "completion_tokens": 1, "calls": 1}}
    answer = transport.parse(doc, doc["questions"][0], ["a", "b"], payload)
    made = su.decision_for(doc, doc["questions"][0], 0, ["a", "b"], answer)
    assert made["pick"] == "a" and made["confidence"] == pytest.approx(0.8)
    assert made["vector"] == {"a": 0.8, "b": 0.2}
    assert made["label"] == "a" and made["tokens_in"] == 40
    assert answer.model == "served/model" and answer.node == "Spark-1"


def test_a_decide_response_with_no_distribution_is_one_hot_and_says_so():
    doc = toy_doc()
    transport = su.DecideTransport("http://n/v1/decide")
    payload = {"decisions": {"decision": {"answer": "b", "confidence": 1.0,
                                          "distribution": None,
                                          "note": "no logprobs from engine"}},
               "usage": {}}
    answer = transport.parse(doc, doc["questions"][0], ["a", "b"], payload)
    made = su.decision_for(doc, doc["questions"][0], 0, ["a", "b"], answer)
    assert made["one_hot"] and made["pick"] == "b"
    assert made["vector"] == {"a": 0.0, "b": 1.0}
    assert jv.calibratable([made]) == []


def test_the_systemone_request_is_the_jev_wire_format_for_all_three_primitives():
    transport = su.SystemOneTransport("http://kev/v1/systemone")

    choice = toy_doc()
    block = transport.request(choice, choice["questions"][0], ["b", "a"])\
        .payload["questions"]["decision"]
    assert block == {"type": "choice", "instructions": "Which one?",
                     "criteria": {"b": "the b", "a": "the a"}}

    noul = toy_doc(qtype="noul", options=("no", "yes"),
                   criteria={"false": "nope", "true": "yep"})
    block = transport.request(noul, noul["questions"][0], ["no", "yes"])\
        .payload["questions"]["decision"]
    assert block["criteria"] == {"false": "nope", "true": "yep"}

    score = toy_doc(qtype="score", options=("0", "1", "2"),
                    criteria=["low", "mid", "high"])
    block = transport.request(score, score["questions"][0], ["0", "1", "2"])\
        .payload["questions"]["decision"]
    assert block["criteria"] == ["low", "mid", "high"]


def test_the_systemone_request_sends_the_state_as_an_object():
    doc = toy_doc()
    transport = su.SystemOneTransport("http://kev/v1/systemone")
    payload = transport.request(doc, doc["questions"][0], ["a", "b"]).payload
    assert payload["state"] == {"text": "one"}
    assert su.wire_leaks(payload) == []


def test_a_yes_no_question_with_no_criteria_sends_none_rather_than_a_map_of_nulls():
    doc = toy_doc(qtype="noul", options=("no", "yes"), criteria=None)
    transport = su.SystemOneTransport("http://kev/v1/systemone")
    block = transport.request(doc, doc["questions"][0], ["no", "yes"])\
        .payload["questions"]["decision"]
    assert "criteria" not in block


def test_a_choice_question_always_carries_its_criteria_because_they_are_the_options():
    """Banking77's 77 criteria are all null and still have to travel."""
    doc = toy_doc(criteria={"a": None, "b": None})
    transport = su.SystemOneTransport("http://kev/v1/systemone")
    block = transport.request(doc, doc["questions"][0], ["a", "b"])\
        .payload["questions"]["decision"]
    assert block["criteria"] == {"a": None, "b": None}


def test_the_systemone_noul_answer_is_p_yes_over_the_two_options():
    doc = toy_doc(qtype="noul", options=("no", "yes"),
                  criteria={"false": "n", "true": "y"})
    doc["labels"] = {"t1": "yes"}
    transport = su.SystemOneTransport("http://kev/v1/systemone")
    payload = {"model": "jev-1.13.0",
               "answers": {"decision": {"noul": 0.93}},
               "usage": {"input_tokens": 90, "output_tokens": 0}}
    answer = transport.parse(doc, doc["questions"][0], ["no", "yes"], payload)
    made = su.decision_for(doc, doc["questions"][0], 0, ["no", "yes"], answer)
    assert made["vector"]["yes"] == pytest.approx(0.93)
    assert made["pick"] == "yes" and jv.is_correct(made)
    assert made["tokens_in"] == 90


def test_the_systemone_choice_answer_reads_its_probabilities_and_its_own_pick():
    doc = toy_doc()
    transport = su.SystemOneTransport("http://kev/v1/systemone")
    payload = {"answers": {"decision": {"choice": "a",
                                        "probabilities": {"a": 0.5, "b": 0.5}}},
               "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
    answer = transport.parse(doc, doc["questions"][0], ["b", "a"], payload)
    made = su.decision_for(doc, doc["questions"][0], 0, ["b", "a"], answer)
    # A tie goes to the option the system named, not to the presented order.
    assert made["pick"] == "a"
    assert made["tokens_in"] == 5


def test_the_systemone_score_answer_accepts_the_spellings_we_had_to_author():
    doc = toy_doc(qtype="score", options=("0", "1", "2"),
                  criteria=["low", "mid", "high"])
    doc["labels"] = {"t1": "1"}
    transport = su.SystemOneTransport("http://kev/v1/systemone")
    for key in ("score", "level", "choice"):
        payload = {"answers": {"decision": {
            key: "1", "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1}}}, "usage": {}}
        answer = transport.parse(doc, doc["questions"][0], ["0", "1", "2"], payload)
        made = su.decision_for(doc, doc["questions"][0], 0, ["0", "1", "2"], answer)
        assert made["pick"] == "1" and jv.is_correct(made)


def test_a_systemone_answer_with_neither_a_map_nor_a_pick_is_one_malformed_row():
    doc = toy_doc()
    transport = su.SystemOneTransport("http://kev/v1/systemone")
    payload = {"answers": {"decision": {"note": "nothing useful"}}, "usage": {}}
    answer = transport.parse(doc, doc["questions"][0], ["a", "b"], payload)
    made = su.decision_for(doc, doc["questions"][0], 0, ["a", "b"], answer)
    assert made["malformed"] and made["pick"] is None
    assert made["vector"] == {"a": 0.5, "b": 0.5}


def test_the_urls_get_their_v1_and_are_not_doubled():
    assert su.decide_endpoint("http://n:3000") == "http://n:3000/v1/decide"
    assert su.decide_endpoint("http://n:3000/v1") == "http://n:3000/v1/decide"
    assert su.decide_endpoint("http://n:3000/v1/decide") == "http://n:3000/v1/decide"
    assert su.systemone_url("http://k:8080") == "http://k:8080/v1/systemone"
    assert su.systemone_url("http://k:8080/v1/systemone") == \
        "http://k:8080/v1/systemone"


def test_the_credential_is_chosen_by_the_endpoints_host_and_not_by_a_flag(monkeypatch):
    """A fleet key can never reach the vendor, whatever order the flags came in."""
    assert su.is_typesafe("https://api.typesafe.ai/v1/systemone")
    assert su.is_typesafe("https://eu.api.typesafe.ai/v1/systemone")
    assert not su.is_typesafe("http://kev-host:8080/v1/systemone")

    monkeypatch.setenv("TYPESAFE_API_KEY", "vendor-key")
    monkeypatch.setenv("AINODE_API_KEY", "fleet-key")
    hosted = su.build_transport("systemone", endpoint="https://api.typesafe.ai/v1")
    assert hosted.api_key == "vendor-key"
    assert hosted.key_source == "$TYPESAFE_API_KEY"
    assert hosted.local is False
    local = su.build_transport("systemone", endpoint="http://kev-host:8080/v1")
    assert local.api_key == "fleet-key"
    assert local.local is True
    node = su.build_transport("decide", endpoint="http://node:3000/v1")
    assert node.api_key == "fleet-key"


def test_a_transport_needs_an_endpoint_and_an_unknown_name_is_refused():
    from ainode.bench.decide.backends import BackendError

    with pytest.raises(BackendError):
        su.build_transport("decide")
    with pytest.raises(BackendError):
        su.build_transport("nope", endpoint="http://x/v1")


def test_a_request_that_somehow_carried_a_label_is_refused_before_it_is_sent():
    """The structural guard, from the other end: no score is worth a leaked answer."""
    doc = toy_doc()

    class Leaky(su.DecideTransport):
        def question(self, spec, presented):
            block = super().question(spec, presented)
            block["expected"] = "a"
            return block

    transport = Leaky("http://n/v1/decide")
    from ainode.bench.decide.backends import BackendError

    with pytest.raises(BackendError) as exc:
        transport.ask(doc, doc["questions"][0], 0)
    assert "expected" in str(exc.value)


def test_the_wire_guard_walks_nested_objects_but_leaves_the_state_alone():
    assert su.wire_leaks({"questions": {"d": {"criteria": {"label": "x"}}}}) == \
        ["questions.d.criteria.label"]
    assert su.wire_leaks({"state": {"label": "fine"}}) == []
    # A gold distribution is the answer key's soft form. A question FILE may carry one;
    # a request body may not, at any depth and inside a list.
    assert su.wire_leaks({"a": [{"gold": {"x": 1.0}}]}) == ["a[0].gold"]
    assert su.wire_leaks({"questions": {"d": {"type": "choice"}}}) == []


# ---------------------------------------------------------------- a fake endpoint

class FakeHandler(BaseHTTPRequestHandler):
    """Answers both wire formats, always confidently picking the FIRST option."""

    def log_message(self, *args):          # keep the suite's output clean
        pass

    def do_POST(self):                     # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.seen.append((self.path, body))
        block = body["questions"]["decision"]
        if self.path.endswith("/decide"):
            options = block["options"]
            reply = {"model": "fake/model", "node": "fake-node", "latency_ms": 5,
                     "decisions": {"decision": {
                         "answer": options[0], "confidence": 0.9,
                         "distribution": {o: (0.9 if o == options[0]
                                              else 0.1 / (len(options) - 1))
                                          for o in options},
                         "latency_ms": 4}},
                     "usage": {"prompt_tokens": 20, "completion_tokens": 1,
                               "calls": 1}}
        else:
            if block["type"] == "noul":
                # P(yes) low, so the pick is the FIRST option, which is what every
                # other branch here does too.
                answer = {"noul": 0.1}
            elif block["type"] == "score":
                levels = [str(i) for i in range(len(block["criteria"]))]
                answer = {"score": levels[0],
                          "probabilities": {level: (0.9 if level == levels[0]
                                                    else 0.1 / (len(levels) - 1))
                                            for level in levels}}
            else:
                options = list(block["criteria"])
                answer = {"choice": options[0],
                          "probabilities": {o: (0.9 if o == options[0]
                                                else 0.1 / (len(options) - 1))
                                            for o in options}}
            reply = {"model": "fake-jev-1.0", "answers": {"decision": answer},
                     "usage": {"input_tokens": 30, "output_tokens": 0}}
        payload = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def fake_endpoint():
    server = HTTPServer(("127.0.0.1", 0), FakeHandler)
    server.seen = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def toy_docs():
    """One set per primitive, small enough to run five repeats of in a test."""
    choice = toy_doc()
    choice["questions"] = [{"id": f"c{i}", "state": {"text": str(i)}}
                           for i in range(4)]
    choice["labels"] = {f"c{i}": ("a" if i % 2 == 0 else "b") for i in range(4)}
    noul = toy_doc(qtype="noul", options=("no", "yes"),
                   criteria={"false": "n", "true": "y"})
    noul["id"] = noul["set"] = "noulset"
    noul["questions"] = [{"id": f"n{i}", "state": {"text": str(i)}} for i in range(4)]
    noul["labels"] = {f"n{i}": "no" for i in range(4)}
    score = toy_doc(qtype="score", options=("0", "1", "2"),
                    criteria=["low", "mid", "high"])
    score["id"] = score["set"] = "scoreset"
    score["questions"] = [{"id": f"s{i}", "state": {"text": str(i)},
                           "gold": {"0": 0.7, "1": 0.2, "2": 0.1}}
                          for i in range(4)]
    score["labels"] = {f"s{i}": "0" for i in range(4)}
    return [choice, noul, score]


@pytest.mark.parametrize("name", ["decide", "systemone"])
def test_a_round_trip_against_a_fake_endpoint_scores_every_primitive(fake_endpoint,
                                                                    name):
    host, port = fake_endpoint.server_address
    transport = su.build_transport(name, endpoint=f"http://{host}:{port}/v1",
                                   model="m", api_key="k")
    docs = toy_docs()
    decisions, seconds = su.run(transport, docs, repeats=5, concurrency=4)
    assert len(decisions) == 3 * 4 * 5
    assert not jv.failed(decisions)
    assert all(d["vector"] for d in decisions)

    # Nothing on the wire ever carried an answer key.
    for _path, body in fake_endpoint.seen:
        assert su.wire_leaks(body) == []
        assert "labels" not in body

    block = su.build_decide_block(transport, docs, decisions, seconds, 5, 4,
                                  model_reported=su.reported_model(transport))
    assert block["backend"] == name
    assert block["mode"] == su.MODE
    assert set(block["jevals"]["sets"]) == {"toy", "noulset", "scoreset"}
    for name_, metrics in block["jevals"]["sets"].items():
        assert metrics["decisions"] == 20
        assert metrics["recipe"] == jv.RECIPE_JEVALS
        assert metrics["accuracy"] is not None
        assert metrics["prior_accuracy"] is not None
        assert len(metrics["bins"]) == jv.BINS
        assert metrics["items"] == 4
        assert name_
    # The fake always picks the first option, and every noul label here is the first.
    assert block["jevals"]["sets"]["noulset"]["accuracy"] == pytest.approx(1.0)
    # The choice set's options ARE reordered per repeat, so a fake that always takes the
    # first presented option flips its pick across the orders. That is the proof the
    # shuffle reaches the model rather than only the record.
    toy = block["jevals"]["sets"]["toy"]
    assert 0.0 < toy["accuracy"] < 1.0
    assert toy["order_flip_rate"] is not None and toy["order_flip_rate"] > 0
    assert toy["repeat_flip_rate"] == pytest.approx(0.0)
    # The one set shipping gold gets its distribution block.
    assert block["jevals"]["sets"]["scoreset"]["vs_gold"]["over"] == 20
    assert block["overall"]["n"] == len(decisions)
    assert "brier" not in block["overall"] and "ece" not in block["overall"]
    assert len(block["rows"]) == len(decisions)
    row = block["rows"][0]
    assert set(row) >= {"id", "set", "kind", "repeat", "order", "label", "answer",
                        "correct", "p_answer", "p_label", "malformed", "one_hot",
                        "wall_ms", "error"}
    assert "state" not in row


def test_the_run_keeps_the_plan_order_whatever_order_the_answers_arrive_in(
        fake_endpoint):
    host, port = fake_endpoint.server_address
    transport = su.build_transport("decide", endpoint=f"http://{host}:{port}/v1")
    docs = toy_docs()
    decisions, _ = su.run(transport, docs, repeats=2, concurrency=8)
    plan = su.plan(docs, 2)
    assert [(d["id"], d["repeat"]) for d in decisions] == \
        [(q["id"], r) for _doc, q, r in plan]


def test_limit_takes_the_first_questions_of_each_set_and_marks_the_block_partial(
        fake_endpoint):
    host, port = fake_endpoint.server_address
    transport = su.build_transport("decide", endpoint=f"http://{host}:{port}/v1")
    docs = toy_docs()
    decisions, seconds = su.run(transport, docs, repeats=2, concurrency=2, limit=1)
    assert len(decisions) == 3 * 1 * 2
    block = su.build_decide_block(transport, docs, decisions, seconds, 2, 2, limit=1)
    assert block["jevals"]["partial"] is True and block["jevals"]["limit"] == 1
    notes = su.build_notes(transport, docs, decisions, seconds, 2, limit=1)
    assert any("PARTIAL RUN" in note for note in notes)


def test_a_transport_failure_is_one_row_with_an_error_and_is_never_scored():
    """No server at all: every decision fails, and the set says so instead of scoring 0.

    The set is PRESENT with a failure count and nulls rather than absent, because the
    run really did ask: absence would read as a set nobody selected, and a zero would
    read as a model that got everything wrong.
    """
    transport = su.build_transport("decide", endpoint="http://127.0.0.1:1/v1",
                                   timeout=1)
    docs = [toy_doc()]
    decisions, seconds = su.run(transport, docs, repeats=1, concurrency=1)
    assert len(decisions) == 1
    assert decisions[0]["error"]
    assert jv.scored(decisions) == []
    block = su.build_decide_block(transport, docs, decisions, seconds, 1, 1)
    toy = block["jevals"]["sets"]["toy"]
    assert toy["failed"] == 1 and toy["decisions"] == 0
    assert toy["accuracy"] is None and toy["decision_score"] is None
    assert toy["ece_points"] is None and toy["handoff_95"] is None
    assert block["jevals"]["overall"]["mean_decision_score"] is None
    notes = su.build_notes(transport, docs, decisions, seconds, 1)
    assert any("failed on a transport" in note for note in notes)


def test_the_notes_name_the_recipe_the_contamination_and_the_deviations():
    docs = toy_docs()
    docs[0]["contamination"] = st.CONTAMINATION["banking77"]
    transport = su.SystemOneTransport("http://k/v1/systemone")
    notes = su.build_notes(transport, docs, [], 0.0, 5)
    text = " ".join(notes)
    assert "jevals.com/methodology" in text and "2026-09-21" in text
    assert "CONTAMINATION" in text and "Kev" in text
    assert "batch size 1" in text
    assert "recipe_of_record" in text
    assert "$0" in text


def test_the_console_table_carries_every_set_and_breaks_a_mixed_one_down(
        fake_endpoint):
    host, port = fake_endpoint.server_address
    transport = su.build_transport("systemone", endpoint=f"http://{host}:{port}/v1")
    docs = toy_docs()
    decisions, seconds = su.run(transport, docs, repeats=2, concurrency=4)
    block = su.build_decide_block(transport, docs, decisions, seconds, 2, 4)["jevals"]
    lines = []
    su.print_table(block, "fake", out=lines.append)
    text = "\n".join(lines)
    for name in ("toy", "noulset", "scoreset"):
        assert name in text
    assert "mean Decision Score" in text
    assert "floor" in text and "hand-off" in text and "swing" in text
    su.print_wrong(decisions, out=lines.append)


# ---------------------------------------------------------------- the CLI

def test_the_dry_run_prints_a_request_per_primitive_and_writes_nothing(tmp_path,
                                                                      capsys):
    path, _ = question_file(tmp_path)
    code = decide_cli.main(["--questions", str(path), "--transport", "systemone",
                            "--endpoint", "http://kev/v1", "--dry-run"],
                           out_dir=tmp_path / "out")
    assert code == 0
    printed = capsys.readouterr().out
    assert "dry run: nothing was requested" in printed
    assert "example (choice)" in printed
    assert "wire    : no answer key" in printed
    assert not (tmp_path / "out").exists()


def test_a_dry_run_never_prints_the_key(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("AINODE_API_KEY", "sk-do-not-print-me")
    path, _ = question_file(tmp_path)
    decide_cli.main(["--questions", str(path), "--transport", "decide",
                     "--endpoint", "http://node:3000/v1", "--dry-run"],
                    out_dir=tmp_path / "out")
    printed = capsys.readouterr().out
    assert "sk-do-not-print-me" not in printed
    assert "from $AINODE_API_KEY (never printed)" in printed


def test_the_transport_is_required_with_a_suite(capsys, tmp_path):
    path, _ = question_file(tmp_path)
    with pytest.raises(SystemExit):
        decide_cli.main(["--questions", str(path), "--label", "x"],
                        out_dir=tmp_path)
    assert "--transport is required" in capsys.readouterr().err


def test_the_two_modes_cannot_be_mixed(capsys, tmp_path):
    path, _ = question_file(tmp_path)
    with pytest.raises(SystemExit):
        decide_cli.main(["--questions", str(path), "--transport", "decide",
                         "--backend", "ainode", "--endpoint", "http://n/v1",
                         "--label", "x"], out_dir=tmp_path)
    assert "legacy 110-item path" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        decide_cli.main(["--backend", "jev", "--transport", "decide", "--label", "x"],
                        out_dir=tmp_path)
    assert "--transport belongs to" in capsys.readouterr().err


def test_the_label_is_required_for_a_real_suite_run(capsys, tmp_path):
    path, _ = question_file(tmp_path)
    with pytest.raises(SystemExit):
        decide_cli.main(["--questions", str(path), "--transport", "decide",
                         "--endpoint", "http://n/v1"], out_dir=tmp_path)
    assert "--label is required" in capsys.readouterr().err


def test_two_question_files_with_one_name_are_refused(capsys, tmp_path):
    first, _ = question_file(tmp_path, name="a.json")
    second, _ = question_file(tmp_path, name="b.json")
    with pytest.raises(SystemExit):
        decide_cli.main(["--questions", f"{first},{second}", "--transport", "decide",
                         "--endpoint", "http://n/v1", "--label", "x"],
                        out_dir=tmp_path)
    assert "a set is one measurement" in capsys.readouterr().err


def test_a_full_run_writes_one_record_with_a_jevals_block(tmp_path, fake_endpoint,
                                                          capsys):
    host, port = fake_endpoint.server_address
    path, _ = question_file(tmp_path)
    out_dir = tmp_path / "results"
    code = decide_cli.main(["--questions", str(path), "--transport", "decide",
                            "--endpoint", f"http://{host}:{port}/v1",
                            "--model", "fake/model", "--repeats", "2",
                            "--concurrency", "2", "--price-in", "0.042",
                            "--label", "fake run"], out_dir=out_dir)
    assert code == 0
    written = list(out_dir.glob("*-decide.json"))
    assert len(written) == 1
    record = json.loads(written[0].read_text())
    assert record["schema"] == 1
    assert record["source"] == "scripts/ainode-bench.py decide"
    assert record["decide"]["mode"] == su.MODE
    assert record["decide"]["recipe"]["suite"] == "0.1.0"
    assert record["decide"]["jevals"]["repeats"] == 2
    assert record["decide"]["jevals"]["batch_size"] == 1
    assert record["settings"]["mode"] == su.MODE
    assert record["settings"]["transport"] == "decide"
    assert "results" not in record
    # A priced endpoint bills the tokens it reported.
    assert record["decide"]["overall"]["cost_usd"] > 0
    # No key anywhere in the record.
    assert "Bearer" not in json.dumps(record)


def test_the_download_subcommand_lists_the_sets_and_reports_a_failure(tmp_path,
                                                                     monkeypatch,
                                                                     capsys):
    monkeypatch.setenv(st.ENV_CACHE, str(tmp_path))
    monkeypatch.setattr(st, "download",
                        lambda *a, **k: (_ for _ in ()).throw(
                            st.SetError("no network here")))
    assert decide_cli.main(["download", "pubmedqa"], out_dir=tmp_path) == 1
    printed = capsys.readouterr().out
    assert "pubmedqa" in printed and "no network here" in printed


def test_the_decide_subcommand_reaches_the_new_mode_from_the_shim(tmp_path, capsys):
    from ainode.bench.cli import main as bench_main

    path, _ = question_file(tmp_path)
    code = bench_main(["decide", "--questions", str(path), "--transport", "systemone",
                       "--endpoint", "http://kev/v1", "--dry-run"],
                      out_dir=tmp_path)
    assert code == 0
    assert "the Jevals recipe" in capsys.readouterr().out


# ---------------------------------------------------------------- the README table

def _renderer():
    spec = importlib.util.spec_from_file_location("render_bench_table", RENDERER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _jevals_record(stamp="20260101-000001", score=41.2, ece=5.8, share=0.62,
                   threshold=0.93):
    return {
        "schema": 1, "stamp": stamp, "label": "jevals run",
        "model": {"id": "a/b", "name": "A Model"},
        "placement": {"node": "Spark-1", "gpus": 1, "tp": 1},
        "decide": {
            "backend": "decide", "mode": "jevals-0.1.0",
            "recipe": {"suite": "0.1.0"},
            "overall": {"n": 4500, "answered": 4500, "errors": 0, "accuracy": 0.71,
                        "tokens": {"in": 1, "out": 1}, "cost_usd": 0.0,
                        "p50_ms": 300, "p95_ms": 900},
            "sets": {},
            "jevals": {
                "sets": {
                    "pubmedqa": {"type": "noul", "decision_score": score,
                                 "ece_points": ece, "accuracy": 0.71,
                                 "prior_accuracy": 0.62,
                                 "handoff_95": {"threshold": threshold,
                                                "share": share, "n": 900,
                                                "accuracy": 0.96}},
                },
                "overall": {"mean_decision_score": score, "sets": ["pubmedqa"],
                            "scored": ["pubmedqa"]},
            },
            "rows": [],
        },
        "source": "scripts/ainode-bench.py decide",
    }


def test_a_jevals_record_renders_its_decision_score_ece_and_handoff(tmp_path):
    m = _renderer()
    (tmp_path / "a.json").write_text(json.dumps(_jevals_record()))
    table = m.render_decide_table(m.load_runs(tmp_path))
    assert "Decision Score" in table and "Hand-off at 95%" in table
    assert "41.2" in table
    assert "5.8 pt" in table
    assert "0.62 @ 0.93" in table


def test_a_legacy_decide_record_keeps_its_own_columns_and_says_not_measured(tmp_path):
    m = _renderer()
    (tmp_path / "a.json").write_text(json.dumps({
        "schema": 1, "stamp": "20260101-000001", "label": "legacy",
        "model": {"id": "a/b", "name": "Legacy"},
        "placement": {"node": "typesafe.ai hosted"},
        "decide": {"backend": "jev",
                   "overall": {"n": 110, "answered": 110, "errors": 0,
                               "accuracy": 0.964, "brier": 0.024, "ece": 0.059,
                               "thresholds": {"0.9": {"kept": 104, "wrong": 2,
                                                      "abstained": 6,
                                                      "no_confidence": 0}},
                               "p50_ms": 290, "p95_ms": 412,
                               "tokens": {"in": 1, "out": 1}, "cost_usd": 0.0016},
                   "sets": {}, "rows": []},
        "source": "scripts/ainode-bench.py decide"}))
    row = m.render_decide_table(m.load_runs(tmp_path)).splitlines()[-1]
    assert "| 0.059 |" in row                      # the legacy ratio, no unit suffix
    assert row.count(m.NOT_MEASURED) == 2          # Decision Score and hand-off
    assert "| 0.024 |" in row


def test_a_missing_jevals_metric_renders_not_measured():
    m = _renderer()
    assert m.fmt_decision_score({}) == m.NOT_MEASURED
    assert m.fmt_handoff_share({}) == m.NOT_MEASURED
    assert m.fmt_decide_ece({}) == m.NOT_MEASURED
    assert m.fmt_decide_ece({"jevals": {"sets": {"a": {"ece_points": None}}}}) == \
        m.NOT_MEASURED


def test_the_mean_over_several_sets_is_what_the_table_shows(tmp_path):
    m = _renderer()
    record = _jevals_record()
    record["decide"]["jevals"]["sets"]["banking77"] = {
        "type": "choice", "decision_score": 20.0, "ece_points": 10.0,
        "accuracy": 0.5, "prior_accuracy": 0.04, "handoff_95": None}
    record["decide"]["jevals"]["overall"] = {"mean_decision_score": 30.6,
                                            "sets": ["pubmedqa", "banking77"],
                                            "scored": ["pubmedqa", "banking77"]}
    (tmp_path / "a.json").write_text(json.dumps(record))
    table = m.render_decide_table(m.load_runs(tmp_path))
    assert "30.6" in table
    # A set with no hand-off pulls the averaged cell to not measured.
    assert m.NOT_MEASURED in table


def test_the_committed_readme_still_matches_the_committed_records():
    """The drift guard over every table, against what is in the repo."""
    proc = subprocess.run([sys.executable, str(RENDERER), "--check"],
                          capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_the_speed_table_still_ignores_a_jevals_record(tmp_path):
    m = _renderer()
    (tmp_path / "a.json").write_text(json.dumps(_jevals_record()))
    runs = m.load_runs(tmp_path)
    assert m.throughput_runs(runs) == []
