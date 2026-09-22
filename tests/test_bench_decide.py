"""Tests for ainode.bench.decide - the decision bench.

No model, no node, no network. Three kinds of fake stand in for the real thing:

  * canned rows handed straight to the metric functions, because a metric is a pure
    function of the rows and the numbers are the whole product here;
  * canned response payloads handed to each backend's parser, one per response shape
    the three services return, so a parser is pinned without a server;
  * a scripted backend for the loop, so ordering, concurrency and the record shape
    are exercised with no transport at all.

The verdicts that matter most are the calibration ones: a backend that is confidently
wrong has to score worse than one that hedges, because that is the failure mode the
bench exists to find.
"""

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

from ainode.bench.cli import main as bench_main
from ainode.bench.decide import backends as be
from ainode.bench.decide import cli as decide_cli
from ainode.bench.decide import metrics as mt
from ainode.bench.decide import runner as rn
from ainode.bench.decide.items import (
    Item,
    ItemError,
    ItemSet,
    default_items_path,
    load_items,
    validate_document,
)

REPO = pathlib.Path(__file__).resolve().parent.parent
RENDERER = REPO / "scripts" / "render-bench-table.py"
ITEMS = REPO / "bench" / "decide" / "items.json"


# ---------------------------------------------------------------- fixtures, fakes

def choice_item(item_id="route-01", label="code"):
    return Item(id=item_id, set="route", kind="choice",
                state="User request: fix my Dockerfile", question="Which model?",
                label=label,
                criteria={"code": "Programming", "chat": "Everything else"})


def noul_item(item_id="fact-01", label=True):
    return Item(id=item_id, set="fact", kind="noul",
                state="Statement: water boils at 100C at sea level",
                question="Is this statement true?", label=label)


def row(set_name="route", label="code", answer="code", p_answer=0.9, p_label=None,
        wall_ms=100, tokens_in=10, tokens_out=2, error=None):
    """A canned row, with p_label defaulting to the consistent value."""
    if p_label is None and p_answer is not None:
        p_label = p_answer if answer == label else round(1.0 - p_answer, 6)
    return {"set": set_name, "label": label, "answer": answer, "p_answer": p_answer,
            "p_label": p_label, "wall_ms": wall_ms, "tokens_in": tokens_in,
            "tokens_out": tokens_out, "error": error}


class FakeBackend(be.Backend):
    """A backend with the transport replaced by a dict of canned decisions."""

    name = "fake"
    endpoint = "http://fake/v1/decide"

    def __init__(self, answers, input_usd_per_mtok=0.0):
        super().__init__(timeout=1)
        self.answers = answers
        self.input_usd_per_mtok = input_usd_per_mtok
        self.asked = []

    def decide(self, item):
        self.asked.append(item.id)
        self.requests += 1
        return self.answers[item.id]


# ---------------------------------------------------------------- metrics

def test_accuracy_counts_only_answered_rows():
    rows = [row(answer="code"), row(answer="chat"), row(answer=None, error="HTTP 500")]
    assert mt.accuracy(rows) == 0.5
    assert len(mt.answered(rows)) == 2
    assert mt.accuracy([]) is None
    assert mt.accuracy([row(answer=None, error="boom")]) is None


def test_brier_is_one_term_on_the_labeled_option():
    # Right at 0.9 and wrong at 0.9: (1-0.9)^2 and (1-0.1)^2.
    rows = [row(p_answer=0.9), row(answer="chat", p_answer=0.9)]
    assert mt.brier(rows) == pytest.approx((0.01 + 0.81) / 2)
    # A perfect, perfectly confident backend scores 0; a coin flip scores 0.25.
    assert mt.brier([row(p_answer=1.0)]) == 0.0
    assert mt.brier([row(p_answer=0.5)]) == 0.25
    assert mt.brier([row(p_answer=None, p_label=None)]) is None


def test_a_confidently_wrong_backend_scores_worse_than_one_that_hedges():
    """The whole point of the bench, as one assertion."""
    confident = [row(answer="chat", p_answer=0.97)]
    hedging = [row(answer="chat", p_answer=0.45)]
    assert mt.brier(confident) > mt.brier(hedging)
    assert mt.threshold_counts(confident)["0.9"]["wrong"] == 1
    assert mt.threshold_counts(hedging)["0.9"]["wrong"] == 0
    assert mt.threshold_counts(hedging)["0.9"]["abstained"] == 1


def test_bins_hold_their_shape_and_place_rows_by_confidence():
    rows = [row(p_answer=0.05), row(p_answer=0.5), row(p_answer=0.95),
            row(p_answer=1.0)]
    table = mt.reliability(rows)
    assert len(table) == mt.BINS
    assert [b["count"] for b in table] == [1, 0, 1, 0, 2]
    assert table[1] == {"lo": 0.2, "hi": 0.4, "count": 0, "accuracy": None,
                        "confidence": None}
    assert mt.bin_index(1.0) == mt.BINS - 1
    assert mt.bin_index(0.0) == 0


def test_ece_is_zero_when_the_confidence_matches_the_hit_rate():
    # Ten rows at 0.9, nine right: claimed 0.9, observed 0.9.
    rows = [row(p_answer=0.9) for _ in range(9)] + [row(answer="chat", p_answer=0.9)]
    assert mt.ece(rows) == pytest.approx(0.0)
    # The same hit rate claimed at 1.0 is a 0.1 gap.
    rows = [row(p_answer=1.0) for _ in range(9)] + [row(answer="chat", p_answer=1.0)]
    assert mt.ece(rows) == pytest.approx(0.1)
    assert mt.ece([row(p_answer=None, p_label=None)]) is None


def test_threshold_counts_split_kept_wrong_abstained_and_unscored():
    rows = [row(p_answer=0.95), row(answer="chat", p_answer=0.92),
            row(p_answer=0.85), row(p_answer=0.4),
            row(p_answer=None, p_label=None)]
    counts = mt.threshold_counts(rows)
    assert counts["0.9"] == {"kept": 2, "wrong": 1, "abstained": 2,
                             "no_confidence": 1}
    assert counts["0.8"] == {"kept": 3, "wrong": 1, "abstained": 1,
                             "no_confidence": 1}
    assert set(counts) == {"0.8", "0.9"}


def test_latency_percentiles_are_observed_values():
    rows = [row(wall_ms=ms) for ms in (10, 20, 30, 40, 100)]
    assert mt.latency(rows) == {"p50_ms": 30, "p95_ms": 100}
    assert mt.latency([row(wall_ms=7)]) == {"p50_ms": 7, "p95_ms": 7}
    assert mt.latency([]) == {"p50_ms": None, "p95_ms": None}


def test_cost_is_the_posted_rate_over_reported_tokens():
    rows = [row(tokens_in=500_000, tokens_out=1_000_000)]
    assert mt.tokens(rows) == {"in": 500_000, "out": 1_000_000}
    assert mt.cost_usd(mt.tokens(rows), be.JEV_INPUT_USD_PER_MTOK) == \
        pytest.approx(0.021)
    # A local backend passes zero rates and the column is $0, never an estimate.
    assert mt.cost_usd(mt.tokens(rows), 0.0, 0.0) == 0.0


def test_summarize_is_the_block_that_lands_in_the_record():
    rows = [row(), row(answer="chat", p_answer=0.95), row(answer=None, error="boom")]
    block = mt.summarize(rows, input_usd_per_mtok=be.JEV_INPUT_USD_PER_MTOK)
    assert block["n"] == 3 and block["answered"] == 2 and block["errors"] == 1
    assert block["accuracy"] == 0.5
    assert set(block) == {"n", "answered", "errors", "accuracy", "brier", "ece",
                          "bins", "thresholds", "tokens", "cost_usd", "p50_ms",
                          "p95_ms"}
    assert len(block["bins"]) == mt.BINS
    assert block["cost_usd"] > 0


def test_summarize_sets_skips_a_set_with_no_rows():
    rows = [row(set_name="route"), row(set_name="fact", label=True, answer=True)]
    blocks = mt.summarize_sets(rows, ["route", "triage", "fact"])
    assert list(blocks) == ["route", "fact"]
    assert blocks["route"]["n"] == 1


# ---------------------------------------------------------------- the item file

def test_the_committed_item_file_is_the_documented_set():
    item_set = load_items(ITEMS)
    assert item_set.id == "decide-110"
    assert len(item_set.items) == 110
    counts = {name: sum(1 for i in item_set.items if i.set == name)
              for name in item_set.set_names}
    assert counts == {"route": 30, "triage": 20, "urgency": 20, "pr_safe": 20,
                      "fact": 20}
    assert item_set.as_json()["count"] == 110


def test_every_item_carries_a_set_a_state_a_kind_and_a_label():
    for item in load_items(ITEMS).items:
        assert item.set and item.state and item.question
        assert item.kind in ("choice", "noul")
        if item.kind == "choice":
            assert item.criteria and len(item.criteria) >= 2
            assert item.label in item.criteria
            assert item.label_option == item.label
        else:
            assert isinstance(item.label, bool)
            assert item.options == ["true", "false"]


def test_the_default_items_path_is_the_repo_copy():
    assert default_items_path() == ITEMS


def test_sets_narrows_the_selection_and_a_typo_is_refused():
    item_set = load_items(ITEMS, ["fact", "urgency"])
    assert {i.set for i in item_set.items} == {"fact", "urgency"}
    assert item_set.set_names == ["urgency", "fact"]        # file order, not argv
    with pytest.raises(ItemError) as exc:
        load_items(ITEMS, ["facts"])
    assert "unknown set" in str(exc.value)


def _doc(items, sets=None):
    return {"id": "t", "version": 1,
            "sets": sets or {"s": {"kind": items[0]["kind"]}}, "items": items}


def _item(**kw):
    base = {"id": "s-01", "set": "s", "kind": "choice", "state": "st",
            "question": "q?", "criteria": {"a": None, "b": None}, "label": "a"}
    base.update(kw)
    return base


@pytest.mark.parametrize("doc, message", [
    ({"items": [_item()]}, "declares no `sets` header"),
    ({"sets": {"s": {}}, "items": []}, "holds no `items`"),
    (_doc([_item(state=None)]), "is missing state"),
    (_doc([_item(set="other")]), "which the header does not declare"),
    (_doc([_item(kind="vibes")]), "has kind"),
    (_doc([_item(criteria=None)]), "needs `criteria`"),
    (_doc([_item(criteria={"a": None})]), "at least two options"),
    (_doc([_item(label="z")]), "not one of its options"),
    (_doc([_item(kind="noul", criteria=None, label="yes")],
          {"s": {"kind": "noul"}}), "a yes/no item's label is true or false"),
    (_doc([_item(), _item()]), "appears twice"),
    (_doc([_item(), _item(id="s-02", question="other?")]),
     "asks two different questions"),
    (_doc([_item(), _item(id="s-02", criteria={"a": None, "c": None})]),
     "different options"),
    (_doc([_item()], {"s": {"kind": "choice", "count": 3}}), "declares 3 items"),
    (_doc([_item()], {"s": {"kind": "noul"}}), "declares kind"),
    (_doc([_item(criteria={"a": 7, "b": None})]), "neither a string nor null"),
])
def test_a_malformed_item_file_is_a_load_error_that_names_the_item(doc, message):
    with pytest.raises(ItemError) as exc:
        validate_document(doc)
    assert message in str(exc.value)


def test_a_false_label_is_a_label_and_not_a_missing_key():
    doc = _doc([_item(kind="noul", criteria=None, label=False)],
               {"s": {"kind": "noul"}})
    items = validate_document(doc)
    assert items[0].label is False
    assert items[0].label_option == "false"


def test_a_missing_item_file_says_how_to_point_at_one(tmp_path):
    with pytest.raises(ItemError) as exc:
        load_items(tmp_path / "nope.json")
    assert "AINODE_DECIDE_ITEMS" in str(exc.value)


def test_invalid_json_is_a_load_error(tmp_path):
    path = tmp_path / "items.json"
    path.write_text("{not json")
    with pytest.raises(ItemError) as exc:
        load_items(path)
    assert "not valid JSON" in str(exc.value)


# ---------------------------------------------------------------- the ainode backend

def decide_backend():
    return be.DecideBackend("http://node:3000/v1", model="org/model")


def test_the_decide_request_sends_one_typed_question_with_the_rubric():
    item = choice_item()
    request = decide_backend().request(item)
    assert request.url == "http://node:3000/v1/decide"
    question = request.payload["questions"]["decision"]
    assert request.payload["state"] == item.state
    assert request.payload["model"] == "org/model"
    assert question["options"] == ["code", "chat"]
    # The endpoint's choice question has no room for a per-option rubric, so the
    # descriptions ride along in the question text.
    assert question["question"].startswith("Which model?")
    assert "code: Programming" in question["question"]
    assert "type" not in question


def test_a_yes_no_item_is_a_boolean_question():
    question = decide_backend().request(noul_item()).payload["questions"]["decision"]
    assert question == {"question": "Is this statement true?", "type": "boolean"}


def test_the_endpoint_gets_its_v1_and_the_bearer_token():
    assert be.decide_url("http://node:3000") == "http://node:3000/v1/decide"
    assert be.decide_url("http://node:3000/v1/decide") == "http://node:3000/v1/decide"
    request = decide_backend().request(choice_item())
    assert request.headers["Authorization"] == "Bearer ainode"
    # The line a dry run prints carries no header, so it cannot carry a key.
    assert "Authorization" not in request.curl_safe()


def test_the_decide_response_is_parsed_into_an_answer_and_a_distribution():
    payload = {"model": "org/model", "node": "Spark-1-DGX", "latency_ms": 210,
               "decisions": {"decision": {"answer": "code", "confidence": 0.71,
                                          "distribution": {"code": 0.8, "chat": 0.2},
                                          "latency_ms": 180}},
               "usage": {"prompt_tokens": 120, "completion_tokens": 3, "calls": 1}}
    decision = decide_backend().parse(choice_item(), payload)
    assert decision.answer == "code"
    # The confidence a caller gates on is the answer's own probability, not the
    # service's derived number.
    assert decision.confidence == 0.8
    assert decision.distribution == {"code": 0.8, "chat": 0.2}
    assert decision.server_latency_ms == 180
    assert (decision.tokens_in, decision.tokens_out) == (120, 3)
    assert decision.node == "Spark-1-DGX"


def test_a_boolean_answer_is_read_whatever_it_is_spelled_as():
    payload = {"decisions": {"decision": {"answer": "yes",
                                          "distribution": {"yes": 0.7, "no": 0.3}}}}
    decision = decide_backend().parse(noul_item(), payload)
    assert decision.answer is True
    assert decision.distribution == {"true": 0.7, "false": 0.3}
    assert decision.confidence == 0.7
    assert be.coerce_boolean(False) is False
    assert be.coerce_boolean("TRUE") is True
    assert be.coerce_boolean("nah") is None
    # One side reported is enough: the other is what is left.
    assert be.boolean_distribution({"true": 0.8}) == {"true": 0.8, "false": 0.2}
    assert be.boolean_distribution({}) is None


def test_a_response_with_no_distribution_keeps_the_reported_confidence():
    payload = {"decisions": {"decision": {"answer": "chat", "confidence": 0.6,
                                          "distribution": None}}}
    decision = decide_backend().parse(choice_item(), payload)
    assert (decision.answer, decision.confidence) == ("chat", 0.6)
    assert decision.distribution is None


def test_an_unreadable_decide_response_is_one_row_and_not_a_crash(monkeypatch):
    backend = decide_backend()
    monkeypatch.setattr(be, "post_json", lambda request, timeout: ({"nope": 1}, 0.2,
                                                                  None))
    decision = backend.decide(choice_item())
    assert decision.answer is None
    assert "unreadable response" in decision.error
    assert decision.wall_ms == 200


def test_a_transport_error_is_one_row_with_the_body_kept(monkeypatch):
    monkeypatch.setattr(be, "post_json",
                        lambda request, timeout: (None, 0.05, "HTTP 404: no route"))
    decision = decide_backend().decide(choice_item())
    assert decision.answer is None and decision.error == "HTTP 404: no route"


# ---------------------------------------------------------------- the chat backend

def chat_backend():
    return be.ChatBackend("http://node:3000/v1", "org/model")


def logprob_payload(letters_to_logprob, text="A", tokens=(133, 2)):
    return {"model": "org/model",
            "choices": [{"message": {"content": text},
                         "logprobs": {"content": [
                             {"token": text[:1],
                              "top_logprobs": [{"token": k, "logprob": v}
                                               for k, v in
                                               letters_to_logprob.items()]}]}}],
            "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1]}}


def test_the_chat_request_letters_the_options_and_switches_thinking_off():
    payload = chat_backend().request(choice_item()).payload
    assert payload["max_tokens"] == 4 and payload["temperature"] == 0.0
    assert payload["logprobs"] is True and payload["top_logprobs"] == 20
    assert payload["chat_template_kwargs"] == {"enable_thinking": False,
                                               "thinking": False}
    prompt = payload["messages"][1]["content"]
    assert "A. code: Programming" in prompt and "B. chat: Everything else" in prompt
    assert prompt.endswith("Answer with one letter.")
    assert payload["messages"][0]["content"].startswith("You are a decision function")


def test_a_yes_no_item_is_lettered_yes_and_no():
    prompt = chat_backend().request(noul_item()).payload["messages"][1]["content"]
    assert "A. yes\nB. no" in prompt


def test_the_chat_distribution_is_a_softmax_over_the_letter_logprobs():
    decision = chat_backend().parse(choice_item(),
                                    logprob_payload({"A": -0.01, "B": -4.0}))
    assert decision.answer == "code"
    assert decision.distribution["code"] > 0.97
    assert sum(decision.distribution.values()) == pytest.approx(1.0)
    assert decision.confidence == decision.distribution["code"]
    assert (decision.tokens_in, decision.tokens_out) == (133, 2)


def test_an_option_no_logprob_mentioned_is_zero_and_not_absent():
    item = Item(id="x", set="s", kind="choice", state="s", question="q",
                label="a", criteria={"a": None, "b": None, "c": None})
    decision = chat_backend().parse(item, logprob_payload({"A": -0.1, "B": -2.0}))
    assert decision.distribution["c"] == 0.0


def test_a_chat_reply_with_no_logprobs_answers_without_a_confidence():
    payload = {"choices": [{"message": {"content": "B"}}], "usage": {}}
    decision = chat_backend().parse(choice_item(), payload)
    assert decision.answer == "chat"
    assert decision.confidence is None and decision.distribution is None


def test_a_chat_yes_no_answer_comes_back_as_a_bool():
    decision = chat_backend().parse(noul_item(),
                                    logprob_payload({"A": -0.2, "B": -1.6}))
    assert decision.answer is True
    assert set(decision.distribution) == {"true", "false"}


def test_a_reply_that_held_no_letter_falls_back_to_the_distribution():
    payload = logprob_payload({"A": -3.0, "B": -0.05}, text="hmm")
    decision = chat_backend().parse(choice_item(), payload)
    assert decision.answer == "chat"
    assert decision.excerpt == "hmm"


def test_letter_probabilities_reads_the_first_non_blank_token():
    logprobs = {"content": [{"token": " ", "top_logprobs": [{"token": "B",
                                                             "logprob": -0.1}]},
                            {"token": "A", "top_logprobs": [{"token": "A",
                                                             "logprob": -0.2}]}]}
    assert list(be.letter_probabilities(logprobs, ["A", "B"])) == ["A"]
    assert be.letter_probabilities(None, ["A"]) == {}


def test_the_chat_backend_needs_an_endpoint_and_a_model():
    with pytest.raises(be.BackendError):
        be.ChatBackend("", "org/model")
    with pytest.raises(be.BackendError):
        be.ChatBackend("http://n/v1", "")
    assert be.chat_url("http://n") == "http://n/v1/chat/completions"


# ---------------------------------------------------------------- the jev backend

def jev_backend():
    return be.JevBackend("secret-key-never-printed")


def test_the_jev_request_is_a_typed_question_with_its_criteria():
    request = jev_backend().request(choice_item())
    assert request.url == be.JEV_URL
    assert request.payload["model"] == be.JEV_MODEL
    question = request.payload["questions"]["decision"]
    assert question == {"type": "choice", "instructions": "Which model?",
                        "criteria": {"code": "Programming",
                                     "chat": "Everything else"}}
    assert request.headers["Authorization"].startswith("Bearer ")


def test_a_yes_no_item_is_a_noul_question_with_no_criteria():
    question = jev_backend().request(noul_item()).payload["questions"]["decision"]
    assert question == {"type": "noul", "instructions": "Is this statement true?"}


def test_the_jev_choice_answer_is_parsed_with_its_probabilities():
    payload = {"model": "jev-1.13.0",
               "answers": {"decision": {"type": "choice", "choice": "code",
                                        "confidence": 0.82,
                                        "probabilities": {"code": 0.85,
                                                          "chat": 0.15}}},
               "usage": {"input_tokens": 385, "output_tokens": 46}}
    decision = jev_backend().parse(choice_item(), payload)
    assert decision.answer == "code"
    assert decision.confidence == 0.85          # the answer's own probability
    assert decision.distribution == {"code": 0.85, "chat": 0.15}
    assert (decision.tokens_in, decision.tokens_out) == (385, 46)
    assert decision.model == "jev-1.13.0"
    assert decision.node == "typesafe.ai hosted"


def test_a_noul_answer_becomes_a_bool_and_a_two_sided_distribution():
    payload = {"model": "jev-1.13.0",
               "answers": {"decision": {"type": "noul", "noul": 0.03}},
               "usage": {"input_tokens": 100, "output_tokens": 10}}
    decision = jev_backend().parse(noul_item(), payload)
    assert decision.answer is False
    assert decision.distribution == {"true": 0.03, "false": 0.97}
    assert decision.confidence == 0.97
    # And the labeled side is what Brier is taken on, not the answered side.
    assert rn.row_for(noul_item(), decision)["p_label"] == 0.03


def test_the_jev_backend_prices_input_tokens_and_the_others_do_not():
    assert jev_backend().input_usd_per_mtok == 0.042
    assert jev_backend().output_usd_per_mtok == 0.0
    assert chat_backend().input_usd_per_mtok == 0.0
    assert decide_backend().input_usd_per_mtok == 0.0


def test_the_key_comes_from_the_flag_then_the_env_then_the_file(tmp_path,
                                                               monkeypatch):
    monkeypatch.delenv(be.JEV_KEY_ENV, raising=False)
    monkeypatch.setattr(be.pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(be, "JEV_KEY_FILE", str(tmp_path / ".jev_api_key"))
    assert be.jev_api_key("flag-key") == ("flag-key", "--api-key")
    monkeypatch.setenv(be.JEV_KEY_ENV, "env-key")
    assert be.jev_api_key() == ("env-key", f"${be.JEV_KEY_ENV}")
    monkeypatch.delenv(be.JEV_KEY_ENV)
    assert be.jev_api_key() == ("", "")
    pathlib.Path(be.JEV_KEY_FILE).write_text("file-key\n")
    key, source = be.jev_api_key()
    assert (key, source) == ("file-key", be.JEV_KEY_FILE)


def test_no_key_is_a_message_naming_where_to_put_one(monkeypatch):
    monkeypatch.delenv(be.JEV_KEY_ENV, raising=False)
    monkeypatch.setattr(be, "JEV_KEY_FILE", "/nonexistent/.jev_api_key")
    with pytest.raises(be.BackendError) as exc:
        be.build_backend("jev")
    assert be.JEV_KEY_ENV in str(exc.value)


def test_build_backend_refuses_an_unknown_name():
    with pytest.raises(be.BackendError):
        be.build_backend("oracle")


# ---------------------------------------------------------------- the loop and record

def canned(answer, confidence=None, distribution=None, wall_ms=100, tokens=(10, 2),
           error=None, model="jev-1.13.0"):
    return be.Decision(answer=answer, confidence=confidence,
                       distribution=distribution, wall_ms=wall_ms,
                       tokens_in=tokens[0], tokens_out=tokens[1], error=error,
                       model=model)


def test_rows_come_back_in_item_order_whatever_order_they_finished_in():
    items = [choice_item("route-01"), choice_item("route-02", label="chat"),
             noul_item("fact-01")]
    answers = {"route-01": canned("code", distribution={"code": 0.9, "chat": 0.1}),
               "route-02": canned("chat", distribution={"code": 0.3, "chat": 0.7}),
               "fact-01": canned(True, distribution={"true": 0.8, "false": 0.2})}
    backend = FakeBackend(answers)
    rows = rn.run_items(backend, items, concurrency=3)
    assert [r["id"] for r in rows] == ["route-01", "route-02", "fact-01"]
    assert [r["correct"] for r in rows] == [True, True, True]
    assert sorted(backend.asked) == ["fact-01", "route-01", "route-02"]


def test_a_row_records_the_answer_its_probability_and_the_labels():
    item = choice_item()
    decision = canned("chat", distribution={"code": 0.25, "chat": 0.75})
    built = rn.row_for(item, decision)
    assert built["answer"] == "chat" and built["correct"] is False
    assert built["p_answer"] == 0.75 and built["p_label"] == 0.25
    assert built["kind"] == "choice" and built["set"] == "route"
    assert "state" not in built            # the state lives in items.json


def test_a_row_with_only_a_confidence_splits_it_over_the_two_sides():
    item = noul_item(label=True)
    built = rn.row_for(item, canned(False, confidence=0.7))
    assert built["p_answer"] == 0.7
    assert built["p_label"] == pytest.approx(0.3)
    assert built["correct"] is False


def test_an_errored_row_is_neither_right_nor_wrong():
    built = rn.row_for(choice_item(), canned(None, error="HTTP 500: boom"))
    assert built["answer"] is None and built["correct"] is None
    assert built["error"] == "HTTP 500: boom"


def test_the_decide_block_holds_the_protocol_the_sets_and_every_row():
    items = [choice_item("route-01"), noul_item("fact-01")]
    item_set = ItemSet(id="t", version=1, description="", sets={"route": {},
                                                               "fact": {}},
                       items=tuple(items), path=pathlib.Path("items.json"))
    backend = FakeBackend({}, input_usd_per_mtok=be.JEV_INPUT_USD_PER_MTOK)
    rows = [rn.row_for(items[0], canned("code",
                                        distribution={"code": 0.9, "chat": 0.1})),
            rn.row_for(items[1], canned(False,
                                        distribution={"true": 0.2, "false": 0.8}))]
    block = rn.build_decide_block(backend, item_set, rows, ["route", "fact"],
                                  concurrency=4, model_reported="jev-1.13.0")
    assert block["backend"] == "fake"
    assert block["model_reported"] == "jev-1.13.0"
    assert block["protocol"]["concurrency"] == 4
    assert block["protocol"]["bins"] == mt.BINS
    assert block["protocol"]["thresholds"] == ["0.8", "0.9"]
    assert block["item_set"]["count"] == 2
    assert list(block["sets"]) == ["route", "fact"]
    assert block["overall"]["n"] == 2 and block["overall"]["accuracy"] == 0.5
    assert len(block["rows"]) == 2
    assert block["overall"]["cost_usd"] > 0


def test_the_record_has_a_decide_block_and_no_results_block():
    record = rn.build_record("my-label", {"id": "jev-1.13.0"},
                             {"node": "typesafe.ai hosted"}, {"backend": "jev"},
                             {"backend": "jev"}, ["a note"], "20260918-120000")
    assert record["schema"] == 1
    assert "results" not in record
    assert record["source"] == "scripts/ainode-bench.py decide"
    assert record["placement"] == {"node": "typesafe.ai hosted"}


def test_the_notes_separate_a_failed_request_from_a_wrong_answer():
    items = [choice_item("route-01"), choice_item("route-02")]
    item_set = ItemSet(id="t", version=1, description="", sets={"route": {}},
                       items=tuple(items), path=pathlib.Path("items.json"))
    rows = [rn.row_for(items[0], canned(None, error="HTTP 500: boom")),
            rn.row_for(items[1], canned("chat"))]
    notes = rn.build_notes(FakeBackend({}), item_set, rows, 12)
    joined = " ".join(notes)
    assert "route-01" in joined and "transport or protocol error" in joined
    assert "no probability" in joined
    assert "$0 for the fake backend" in joined


def test_a_priced_backend_says_what_the_rate_was():
    item_set = ItemSet(id="t", version=1, description="", sets={}, items=(),
                       path=pathlib.Path("items.json"))
    backend = FakeBackend({}, input_usd_per_mtok=be.JEV_INPUT_USD_PER_MTOK)
    assert any("0.042 per million input tokens" in n
               for n in rn.build_notes(backend, item_set, [], 3))


def test_the_printed_table_carries_every_set_and_an_overall_line():
    rows = [row(), row(set_name="fact", label=True, answer=True, p_answer=0.8)]
    block = {"sets": mt.summarize_sets(rows, ["route", "fact"]),
             "overall": mt.summarize(rows)}
    lines = []
    rn.print_table(block, "fake model", out=lines.append)
    text = "\n".join(lines)
    assert "fake model" in text
    assert [name for name, _cells in rn.table_rows(block)] == ["route", "fact", "ALL"]
    assert "reliability" in text and "0.8-1.0" in text


def test_the_side_by_side_table_lines_the_two_runs_up():
    rows_a = [row(p_answer=0.95)]
    rows_b = [row(answer="chat", p_answer=0.95)]
    blocks = [{"sets": mt.summarize_sets(rows_a, ["route"]),
               "overall": mt.summarize(rows_a)},
              {"sets": mt.summarize_sets(rows_b, ["route"]),
               "overall": mt.summarize(rows_b)}]
    lines = []
    rn.print_compare(blocks, ["jev-1.13.0", "local"], out=lines.append)
    text = "\n".join(lines)
    assert "jev-1.13.0" in text and "local" in text
    assert "wrong at 0.9" in text and "accuracy per set" in text
    assert rn.dig(blocks[1]["overall"], "thresholds/0.9/wrong") == 1


def test_the_wrong_list_is_ordered_by_confidence():
    items = [choice_item("route-01"), choice_item("route-02"), noul_item("fact-01")]
    rows = [rn.row_for(items[0], canned("chat",
                                        distribution={"code": 0.1, "chat": 0.9})),
            rn.row_for(items[1], canned("code",
                                        distribution={"code": 0.99, "chat": 0.01})),
            rn.row_for(items[2], canned(False,
                                        distribution={"true": 0.4, "false": 0.6}))]
    lines = []
    rn.print_wrong(rows, items, out=lines.append)
    text = "\n".join(lines)
    assert "wrong answers (2)" in text
    assert text.index("route-01") < text.index("fact-01")
    rn.print_wrong([rows[1]], items, out=lines.append)
    assert "no wrong answers" in lines[-1]


# ---------------------------------------------------------------- the CLI

def test_the_dry_run_prints_both_request_shapes_and_writes_nothing(tmp_path, capsys):
    code = decide_cli.main(["--backend", "ainode", "--endpoint",
                            "http://node:3000/v1", "--dry-run"], out_dir=tmp_path)
    assert code == 0
    out = capsys.readouterr().out
    assert "POST http://node:3000/v1/decide" in out
    assert '"type": "boolean"' in out
    assert "110 items" in out
    assert "no file was written" in out
    assert not list(tmp_path.glob("*"))


def test_a_dry_run_never_prints_the_key(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(be.JEV_KEY_ENV, "super-secret-value")
    assert decide_cli.main(["--backend", "jev", "--dry-run"], out_dir=tmp_path) == 0
    out = capsys.readouterr().out
    assert "super-secret-value" not in out
    assert f"from ${be.JEV_KEY_ENV} (never printed)" in out


def test_the_backend_is_required_and_checked(capsys):
    with pytest.raises(SystemExit):
        decide_cli.main(["--label", "l"])
    assert "--backend is required" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        decide_cli.main(["--backend", "psychic", "--label", "l"])


def test_the_label_is_required_for_a_real_run(capsys):
    with pytest.raises(SystemExit):
        decide_cli.main(["--backend", "jev"])
    assert "--label is required" in capsys.readouterr().err


def test_compare_refuses_the_backend_it_already_has(capsys):
    with pytest.raises(SystemExit):
        decide_cli.main(["--backend", "jev", "--compare", "jev", "--label", "l"])
    assert "--compare names a second backend" in capsys.readouterr().err


def test_an_unknown_set_is_refused_before_a_request(capsys):
    with pytest.raises(SystemExit):
        decide_cli.main(["--backend", "jev", "--label", "l", "--sets", "nope"])
    assert "unknown set" in capsys.readouterr().err


def test_the_hosted_backend_keeps_its_own_model_when_compared_to_a_local_one():
    args = decide_cli.build_parser().parse_args(
        ["--backend", "chat", "--compare", "jev", "--model", "org/local",
         "--label", "l"])
    assert decide_cli.model_for(args, "chat") == "org/local"
    assert decide_cli.model_for(args, "jev") == be.JEV_MODEL
    alone = decide_cli.build_parser().parse_args(
        ["--backend", "jev", "--model", "jev-1.13.0", "--label", "l"])
    assert decide_cli.model_for(alone, "jev") == "jev-1.13.0"


def test_a_hosted_run_records_the_model_the_api_reported_and_no_node_of_ours():
    args = decide_cli.build_parser().parse_args(["--backend", "jev", "--label", "l"])
    backend = be.JevBackend("k")
    backend.reported_model = "jev-1.13.0"
    model_block, placement, warnings = decide_cli.describe(args, backend)
    assert model_block["id"] == "jev-1.13.0" and model_block["name"] == "jev-1.13.0"
    assert placement == {"node": "typesafe.ai hosted"}
    assert warnings == []


def test_a_local_run_with_no_ainode_flag_records_no_placement():
    args = decide_cli.build_parser().parse_args(
        ["--backend", "chat", "--endpoint", "http://n/v1", "--model", "org/m",
         "--label", "l"])
    model_block, placement, _warn = decide_cli.describe(args, chat_backend())
    assert model_block == {"id": "org/model"}
    assert placement == {}


def test_two_backends_of_one_run_cannot_overwrite_each_others_record(tmp_path):
    first = decide_cli.record_path(tmp_path, "20260918-120000", "org/model", "run",
                                  "ainode")
    first.write_text("{}")
    second = decide_cli.record_path(tmp_path, "20260918-120000", "org/model", "run",
                                    "chat")
    assert first.name == "20260918-120000-model-run-decide.json"
    assert second.name == "20260918-120000-model-run-chat-decide.json"


def test_the_decide_subcommand_is_reachable_from_the_shim(capsys, tmp_path):
    assert bench_main(["decide", "--backend", "jev", "--dry-run"],
                      out_dir=tmp_path) == 0
    assert "ainode-bench decide" in capsys.readouterr().out


def test_a_run_writes_one_record_per_backend(tmp_path, monkeypatch, capsys):
    """The whole CLI path with the transport faked, down to the file on disk."""
    def fake_post(request, timeout):
        if "systemone" in request.url:
            return ({"model": "jev-1.13.0",
                     "answers": {"decision": {"type": "noul", "noul": 0.91}},
                     "usage": {"input_tokens": 200, "output_tokens": 20}}, 0.3, None)
        return (logprob_payload({"A": -0.05, "B": -3.0}), 0.8, None)

    monkeypatch.setattr(be, "post_json", fake_post)
    code = decide_cli.main(["--backend", "jev", "--compare", "chat",
                            "--endpoint", "http://node:3000/v1",
                            "--model", "org/model", "--api-key", "k",
                            "--sets", "fact", "--label", "two backends",
                            "--concurrency", "4"], out_dir=tmp_path)
    assert code == 0
    written = sorted(p.name for p in tmp_path.glob("*.json"))
    assert len(written) == 2
    assert written[0].endswith("-decide.json")
    jev_record = json.loads(next(p for p in tmp_path.glob("*jev-1_13_0*")).read_text())
    assert jev_record["model"]["id"] == "jev-1.13.0"
    assert jev_record["settings"]["model_requested"] == be.JEV_MODEL
    assert jev_record["decide"]["backend"] == "jev"
    assert jev_record["decide"]["overall"]["n"] == 20
    assert jev_record["placement"] == {"node": "typesafe.ai hosted"}
    out = capsys.readouterr().out
    assert "side by side" in out and "accuracy per set" in out


# ---------------------------------------------------------------- the README table

def _renderer():
    spec = importlib.util.spec_from_file_location("render_bench_table", RENDERER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _decide_record(stamp="20260101-000001", backend="jev", name="Jev 1.13.0",
                   placement=None, accuracy=0.964, wrong=2, kept=106, cost=0.0017,
                   p50=290):
    return {
        "schema": 1, "stamp": stamp, "label": "my-run",
        "model": {"id": "jev-1.13.0", "name": name},
        "placement": placement if placement is not None else {
            "node": "typesafe.ai hosted"},
        "decide": {
            "backend": backend,
            "overall": {"n": 110, "answered": 110, "errors": 0,
                        "accuracy": accuracy, "brier": 0.024, "ece": 0.059,
                        "thresholds": {"0.8": {"kept": 108, "wrong": 3,
                                               "abstained": 2, "no_confidence": 0},
                                       "0.9": {"kept": kept, "wrong": wrong,
                                               "abstained": 110 - kept,
                                               "no_confidence": 0}},
                        "p50_ms": p50, "p95_ms": 412, "tokens": {"in": 41000,
                                                                 "out": 5000},
                        "cost_usd": cost},
            "sets": {}, "rows": [],
        },
        "source": "scripts/ainode-bench.py decide",
    }


def _write(directory, name, obj):
    (directory / name).write_text(json.dumps(obj))


def test_a_decide_record_renders_a_row(tmp_path):
    """A legacy 110-item record fills its own columns and says "not measured" in the
    two the Jevals recipe added (Decision Score and the hand-off share), because those
    are a different measurement and not a number this record took."""
    m = _renderer()
    _write(tmp_path, "20260101-000001-decide.json", _decide_record())
    table = m.render_decide_table(m.load_runs(tmp_path))
    lines = table.splitlines()
    assert len(lines) == 3                     # header + rule + one row
    assert (f"| jev / Jev 1.13.0 | typesafe.ai hosted | 110 | 0.964 "
            f"| {m.NOT_MEASURED} | 0.024 | 0.059 | {m.NOT_MEASURED} "
            f"| 2 of 106 | 290 | $0.0017 |") in table
    assert "[my-run](https://github.com/getainode/ainode/blob/main/bench/results/" \
        in table


def test_a_local_backend_row_reads_zero_and_names_its_node(tmp_path):
    m = _renderer()
    _write(tmp_path, "a.json", _decide_record(
        backend="chat", name="Ornith 1.5 35B-A3B",
        placement={"node": "Spark-1-DGX", "gpus": 1, "tp": 1}, cost=0.0))
    row_text = m.render_decide_table(m.load_runs(tmp_path)).splitlines()[-1]
    assert "| chat / Ornith 1.5 35B-A3B | Spark-1-DGX, TP=1 |" in row_text
    assert "| $0 |" in row_text


def test_a_failed_item_is_visible_in_the_items_column(tmp_path):
    m = _renderer()
    record = _decide_record()
    record["decide"]["overall"]["errors"] = 4
    _write(tmp_path, "a.json", record)
    assert "110 (4 failed)" in m.render_decide_table(m.load_runs(tmp_path))


def test_a_missing_metric_renders_not_measured():
    m = _renderer()
    assert m.fmt_ratio(None) == m.NOT_MEASURED
    assert m.fmt_wrong_at({}) == m.NOT_MEASURED
    assert m.fmt_usd({}) == m.NOT_MEASURED
    assert m.fmt_items({}) == m.NOT_MEASURED
    assert m.fmt_latency_ms({}) == m.NOT_MEASURED


def test_the_speed_table_ignores_decide_records(tmp_path):
    """A decision record took no tok/s, so it is not a very slow model."""
    m = _renderer()
    _write(tmp_path, "20260101-000001-decide.json", _decide_record())
    _write(tmp_path, "20260101-000002-speed.json", {
        "schema": 1, "stamp": "20260101-000002", "label": "thr",
        "model": {"id": "a/b", "name": "Thr", "params_b": 7, "arch": "dense"},
        "placement": {"node": "N", "gpus": 1, "tp": 1},
        "results": {"single_stream": {"decode_tok_s": 10.0}},
    })
    runs = m.load_runs(tmp_path)
    assert [r["_file"] for r in m.throughput_runs(runs)] == \
        ["20260101-000002-speed.json"]
    assert "20260101-000001-decide.json" not in m.render_table(runs)


def test_one_model_measured_through_two_backends_gets_two_rows(tmp_path):
    m = _renderer()
    _write(tmp_path, "a.json", _decide_record(backend="chat", name="Ornith"))
    _write(tmp_path, "b.json", _decide_record(backend="ainode", name="Ornith",
                                              stamp="20260101-000002"))
    table = m.render_decide_table(m.load_runs(tmp_path))
    assert len(table.splitlines()) == 4
    assert "ainode / Ornith" in table and "chat / Ornith" in table


def test_the_later_decide_record_wins_for_one_backend_and_model(tmp_path):
    m = _renderer()
    _write(tmp_path, "early.json", _decide_record(stamp="20260101-000001",
                                                  accuracy=0.5))
    _write(tmp_path, "late.json", _decide_record(stamp="20260201-000001",
                                                 accuracy=0.964))
    table = m.render_decide_table(m.load_runs(tmp_path))
    assert len(table.splitlines()) == 3
    assert "0.964" in table and "0.500" not in table


def test_check_detects_a_stale_decide_table(tmp_path):
    """--check returns 1 when only the decision table of the README has drifted."""
    m = _renderer()
    results = tmp_path / "results"
    results.mkdir()
    _write(results, "throughput.json", {
        "schema": 1, "stamp": "20260101-000001", "label": "thr",
        "model": {"id": "x/y", "name": "Thr", "params_b": 7, "active_b": 7,
                  "arch": "dense"},
        "placement": {"node": "N", "gpus": 1, "tp": 1},
        "results": {"single_stream": {"decode_tok_s": 10.0},
                    "concurrency": [{"streams": 16, "aggregate_tok_s": 5.0}]},
    })
    _write(results, "decide.json", _decide_record(stamp="20260101-000002"))

    readme = tmp_path / "README.md"
    runs = m.load_runs(results)
    readme.write_text(
        "# Bench\n\n"
        f"{m.BEGIN}\n\n{m.render_table(runs)}\n\n{m.END}\n\n"
        f"{m.DECIDE_BEGIN}\n\n{m.render_decide_table(runs)}\n\n{m.DECIDE_END}\n")

    def check():
        proc = subprocess.run(
            [sys.executable, str(RENDERER), "--check", "--results", str(results),
             "--readme", str(readme)],
            capture_output=True, text=True, cwd=REPO)
        return proc.returncode

    assert check() == 0
    readme.write_text(readme.read_text().replace("0.964", "0.999"))
    assert check() == 1


def test_the_committed_readme_matches_the_committed_records():
    """The drift guard, over all four tables, against what is in the repo."""
    proc = subprocess.run([sys.executable, str(RENDERER), "--check"],
                          capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "decide rows" in proc.stdout


def test_every_decide_record_in_the_repo_has_the_documented_shape():
    """bench/SCHEMA.md's `decide` block, checked against the committed records.

    The legacy assertions apply to a legacy record, which is one with no `jevals` block:
    a Jevals-recipe record's `overall` deliberately carries no `brier`, `ece`, `bins` or
    `thresholds`, because those names mean the legacy definitions and filling them from
    the recipe's own arithmetic would put two incomparable numbers under one name. Its
    own shape is pinned in `tests/test_bench_decide_jevals.py`.
    """
    from ainode.bench.decide import suite as su

    m = _renderer()
    runs = [r for r in m.load_runs(REPO / "bench" / "results") if r.get("decide")]
    item_ids = {i.id for i in load_items(ITEMS).items}
    for run in runs:
        block = run["decide"]
        assert block["backend"] in tuple(be.BACKENDS) + tuple(su.TRANSPORTS)
        assert set(block) >= {"backend", "endpoint", "item_set", "protocol",
                              "overall", "sets", "rows"}
        overall = block["overall"]
        assert overall["n"] == len(block["rows"])
        if block.get("jevals"):
            assert block["mode"] == su.MODE
            assert block["jevals"]["sets"]
            continue
        assert overall["n"] == sum(s["n"] for s in block["sets"].values())
        assert len(overall["bins"]) == mt.BINS
        assert set(overall["thresholds"]) == {"0.8", "0.9"}
        for key in ("accuracy", "brier", "ece", "cost_usd", "p50_ms", "p95_ms"):
            assert key in overall
        for record_row in block["rows"]:
            assert set(record_row) >= {"id", "set", "kind", "label", "answer",
                                       "correct", "p_answer", "p_label", "wall_ms",
                                       "error"}
            assert record_row["id"] in item_ids
        assert "results" not in run or run["results"]
        assert run["source"] == "scripts/ainode-bench.py decide"
        # A hosted run names the service; a local one names a node of ours or nothing.
        if block["backend"] == "jev":
            assert run["placement"] == {"node": "typesafe.ai hosted"}
            assert run["model"]["id"].startswith("jev-")
