"""POST /v1/decide: typed questions in, calibrated probabilities out.

The response shape here is a contract the bench reads, so these tests pin it:
every question answered in one request, each with an option string, a
probability, a full distribution over the options and its own latency, plus one
merged usage block for the whole request.

The engine-facing tests run a REAL AINode app whose cluster state points at a
fake vLLM on a real port, the same arrangement ``test_messages_proxy.py`` uses,
so routing, the request body the engine actually receives and the failure paths
are all exercised rather than mocked out from underneath.
"""

import asyncio
import math
import socket

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from ainode.api.decide import (
    DecideError,
    SYSTEM_PROMPT,
    build_chat_body,
    build_messages,
    decision_from_payload,
    distribution_from_logprobs,
    first_token_top_logprobs,
    merge_usage,
    normalize_questions,
    option_label,
    option_labels,
)
from ainode.api.server import create_app
from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeStatus
from ainode.discovery.cluster import ClusterNode

MODEL = "ornith-ai/Ornith-1.5-35B-A3B-NVFP4"

TICKET = ("Customer writes: the invoice PDF download 500s on every browser since "
          "your Tuesday release. We bill 400 clients on Friday.")

QUESTIONS = {
    "category": {"question": "Which queue should this ticket go to?",
                 "options": ["billing", "bug", "feature request",
                             "account access", "spam"]},
    "urgency": {"question": "How urgent is it?", "type": "score",
                "min": 1, "max": 5},
    "needs_human": {"question": "Does this need a human?", "type": "boolean"},
}


# ------------------------------------------------------------------- the label scheme


def test_labels_are_bijective_base_26_past_z():
    assert option_label(0) == "A"
    assert option_label(25) == "Z"
    # The letter after Z is AA, not [ or A0: the scheme has to keep going for a
    # question with more than 26 options and never collide with a shorter label.
    assert option_label(26) == "AA"
    assert option_label(27) == "AB"
    assert option_label(51) == "AZ"
    assert option_label(52) == "BA"
    assert option_label(254) == "IU"  # the 255th option, the ceiling


def test_every_label_up_to_the_ceiling_is_distinct():
    labels = option_labels(255)
    assert len(set(labels)) == 255
    assert labels[0] == "A" and labels[-1] == "IU"
    assert all(1 <= len(label) <= 2 for label in labels)


def test_a_negative_index_is_a_programming_error_not_a_label():
    with pytest.raises(ValueError):
        option_label(-1)


# --------------------------------------------------------------------- validation


def _err(raw):
    with pytest.raises(DecideError) as exc:
        normalize_questions(raw)
    return str(exc.value)


def test_questions_must_be_a_non_empty_object():
    assert "non-empty object" in _err(None)
    assert "non-empty object" in _err({})
    assert "non-empty object" in _err([{"question": "x", "options": ["a", "b"]}])


def test_a_question_needs_text_and_a_way_to_answer_it():
    assert "must be an object" in _err({"k": "just a string"})
    assert "non-empty 'question' string" in _err({"k": {"options": ["a", "b"]}})
    assert "non-empty 'question' string" in _err({"k": {"question": "  ",
                                                       "options": ["a", "b"]}})
    assert "needs 'options' or a 'type'" in _err({"k": {"question": "why?"}})
    assert "unknown type 'colour'" in _err({"k": {"question": "why?",
                                                  "type": "colour"}})


def test_fewer_than_two_options_is_not_a_question():
    msg = _err({"k": {"question": "why?", "options": ["only"]}})
    assert "at least 2 options" in msg
    assert "at least 2 options" in _err({"k": {"question": "why?", "options": []}})


def test_more_than_255_options_is_refused():
    opts = [f"opt{i}" for i in range(256)]
    assert "more than the 255 allowed" in _err({"k": {"question": "which?",
                                                      "options": opts}})
    # 255 exactly is fine.
    ok = normalize_questions({"k": {"question": "which?", "options": opts[:255]}})
    assert len(ok["k"]["options"]) == 255


def test_duplicate_options_are_refused():
    msg = _err({"k": {"question": "why?", "options": ["yes", "no", "yes"]}})
    assert "duplicate options" in msg and "yes" in msg


def test_a_non_string_option_is_refused():
    assert "non-empty string" in _err({"k": {"question": "why?",
                                             "options": ["a", 7]}})
    assert "non-empty string" in _err({"k": {"question": "why?",
                                             "options": ["a", None]}})
    assert "non-empty string" in _err({"k": {"question": "why?",
                                             "options": ["a", "  "]}})
    assert "'options' must be a list" in _err({"k": {"question": "why?",
                                                     "options": "a,b"}})


def test_the_type_sugar_expands_to_options():
    out = normalize_questions({
        "b": {"question": "human?", "type": "boolean"},
        "s": {"question": "how bad?", "type": "score"},
        "r": {"question": "how bad?", "type": "score", "min": 0, "max": 10},
    })
    assert out["b"]["options"] == ["yes", "no"]
    assert out["s"]["options"] == ["1", "2", "3", "4", "5"]
    assert out["r"]["options"] == [str(v) for v in range(0, 11)]


def test_a_broken_score_range_is_refused():
    assert "greater than 'min'" in _err({"k": {"question": "q", "type": "score",
                                                "min": 5, "max": 5}})
    assert "must be integers" in _err({"k": {"question": "q", "type": "score",
                                              "min": "1", "max": 5}})


def test_question_order_is_preserved():
    out = normalize_questions(QUESTIONS)
    assert list(out) == ["category", "urgency", "needs_human"]


# ----------------------------------------------------------------- prompt builder


def test_the_prompt_puts_the_state_before_the_question():
    msgs = build_messages("ticket text", None, "Which queue?", ["billing", "bug"])
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    user = msgs[1]["content"]
    assert user.index("ticket text") < user.index("Which queue?")
    assert "A. billing" in user and "B. bug" in user
    assert user.rstrip().endswith("nothing else.")


def test_two_questions_over_one_state_share_a_byte_identical_prefix():
    """Prefix caching is the whole reason the state comes first."""
    a = build_messages("the same long state", "guidance", "Q one?", ["x", "y"])
    b = build_messages("the same long state", "guidance", "Q two?", ["x", "y"])
    assert a[0] == b[0]
    prefix = "\n".join(["STATE:", "the same long state", ""])
    assert a[1]["content"].startswith(prefix)
    assert b[1]["content"].startswith(prefix)


def test_instructions_are_appended_to_the_system_prompt():
    msgs = build_messages("s", "  Escalate anything touching payments.  ",
                          "Q?", ["a", "b"])
    assert msgs[0]["content"].startswith(SYSTEM_PROMPT)
    assert msgs[0]["content"].endswith("Escalate anything touching payments.")
    # An empty instructions block leaves the system prompt alone.
    assert build_messages("s", "   ", "Q?", ["a", "b"])[0]["content"] == SYSTEM_PROMPT


def test_the_chat_body_constrains_the_answer_and_asks_for_logprobs():
    body = build_chat_body(MODEL, [{"role": "user", "content": "x"}], ["A", "B"])
    # vLLM 0.27.1 spelling. `guided_choice` is accepted and then ignored there,
    # so it is deliberately absent.
    assert body["structured_outputs"] == {"choice": ["A", "B"]}
    assert "guided_choice" not in body
    assert body["logprobs"] is True and body["top_logprobs"] == 20
    assert body["temperature"] == 0 and body["stream"] is False
    assert body["max_tokens"] == 2  # one letter plus the end-of-turn token
    assert body["chat_template_kwargs"] == {"enable_thinking": False,
                                            "thinking": False}


def test_a_two_letter_label_gets_room_to_be_emitted():
    labels = option_labels(30)
    body = build_chat_body(MODEL, [], labels)
    assert labels[-1] == "AD"
    assert body["max_tokens"] == 3


# ------------------------------------------------------- softmax and calibration


def _payload(content, tops, usage=None):
    """A chat completion shaped the way vLLM 0.27.1 answers one."""
    return {
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "logprobs": {"content": [{"token": content, "logprob": tops[0][1],
                                      "top_logprobs": [{"token": t, "logprob": lp}
                                                       for t, lp in tops]}]},
            "finish_reason": "stop",
        }],
        "usage": usage or {"prompt_tokens": 73, "completion_tokens": 2},
    }


def test_softmax_over_the_label_tokens_renormalizes_to_one():
    dist = distribution_from_logprobs(
        ["A", "B", "C"], [{"token": "A", "logprob": -0.1},
                          {"token": "B", "logprob": -2.3},
                          {"token": "C", "logprob": -4.6}])
    assert set(dist) == {"A", "B", "C"}
    assert math.isclose(sum(dist.values()), 1.0, abs_tol=1e-5)
    assert dist["A"] > dist["B"] > dist["C"]
    # Worked by hand off the same three logprobs.
    weights = [math.exp(-0.1), math.exp(-2.3), math.exp(-4.6)]
    total = sum(weights)
    assert math.isclose(dist["A"], weights[0] / total, abs_tol=1e-6)
    assert math.isclose(dist["C"], weights[2] / total, abs_tol=1e-6)


def test_a_label_absent_from_top_logprobs_gets_zero_and_the_rest_renormalize():
    dist = distribution_from_logprobs(
        ["A", "B", "C", "D"], [{"token": "A", "logprob": -0.05},
                               {"token": "C", "logprob": -3.0}])
    assert dist["B"] == 0.0 and dist["D"] == 0.0
    assert math.isclose(dist["A"] + dist["C"], 1.0, abs_tol=1e-5)


def test_the_mask_sentinel_vllm_sends_for_a_grammar_blocked_token_is_harmless():
    """Under a choice grammar vLLM reports every blocked token at -9999.0."""
    dist = distribution_from_logprobs(
        ["A", "B", "C"], [{"token": "A", "logprob": -9.5e-06},
                          {"token": "C", "logprob": -12.1875},
                          {"token": "B", "logprob": -12.3437},
                          {"token": "1", "logprob": -9999.0},
                          {"token": "!", "logprob": -9999.0}])
    assert math.isclose(sum(dist.values()), 1.0, abs_tol=1e-5)
    assert dist["A"] > 0.99


def test_a_two_letter_label_is_scored_by_its_own_token_when_it_has_one():
    """The Ornith tokenizer gives AB one token, so AB is exact-matched."""
    labels = option_labels(28)  # A..Z, AA, AB
    dist = distribution_from_logprobs(
        labels, [{"token": "AB", "logprob": -0.294},
                 {"token": "A", "logprob": -1.544},
                 {"token": "B", "logprob": -3.232}])
    assert dist["AB"] > dist["A"] > dist["B"]
    assert dist["AA"] == dist["A"]  # AA has no token of its own: documented tie
    assert math.isclose(sum(dist.values()), 1.0, abs_tol=1e-5)


def test_no_logprobs_falls_back_to_the_constrained_answer():
    payload = {"choices": [{"message": {"content": "B"}}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
    assert first_token_top_logprobs(payload) == []
    entry = decision_from_payload(payload, ["billing", "bug"], 12.34)
    assert entry["answer"] == "bug"
    assert entry["confidence"] == 1.0
    assert entry["distribution"] is None
    assert entry["note"] == "no logprobs from engine"
    assert entry["latency_ms"] == 12.3


def test_a_decision_reports_probabilities_against_the_option_strings():
    payload = _payload("A", [("A", -0.05), ("B", -3.0)])
    entry = decision_from_payload(payload, ["billing", "bug"], 40.0)
    assert entry["answer"] == "billing"
    assert set(entry["distribution"]) == {"billing", "bug"}
    assert entry["confidence"] == entry["distribution"]["billing"]
    assert "note" not in entry


def test_a_lone_chosen_token_with_no_alternatives_still_yields_a_distribution():
    payload = {"choices": [{"message": {"content": "A"},
                            "logprobs": {"content": [{"token": "A",
                                                      "logprob": -0.2}]}}]}
    entry = decision_from_payload(payload, ["yes", "no"], 5.0)
    assert entry["distribution"] == {"yes": 1.0, "no": 0.0}
    assert entry["confidence"] == 1.0


def test_usage_merges_into_one_block_that_counts_the_calls():
    usage = merge_usage([{"usage": {"prompt_tokens": 73, "completion_tokens": 2}},
                         {"usage": {"prompt_tokens": 80, "completion_tokens": 2}},
                         {"usage": {}}])
    assert usage == {"prompt_tokens": 153, "completion_tokens": 4, "calls": 3}


# ----------------------------------------------------------------- the live route


class FakeEngine:
    """A vLLM that answers a choice-constrained chat completion with logprobs."""

    def __init__(self, answer_index=0, with_logprobs=True, status=200):
        self.seen: list = []
        self.answer_index = answer_index
        self.with_logprobs = with_logprobs
        self.status = status
        self.in_flight = 0
        self.max_in_flight = 0

    def app(self):
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.completions)
        return app

    async def completions(self, request):
        body = await request.json()
        self.seen.append(body)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.05)  # long enough for overlap to be observable
        finally:
            self.in_flight -= 1
        if self.status != 200:
            return web.json_response({"error": {"message": "engine says no"}},
                                     status=self.status)
        labels = body["structured_outputs"]["choice"]
        chosen = labels[self.answer_index]
        message = {"role": "assistant", "content": chosen}
        choice = {"index": 0, "message": message, "finish_reason": "stop"}
        if self.with_logprobs:
            tops = [{"token": chosen, "logprob": -0.0625}]
            tops += [{"token": label, "logprob": -4.0 - i}
                     for i, label in enumerate(labels) if label != chosen]
            choice["logprobs"] = {"content": [{"token": chosen,
                                               "logprob": -0.0625,
                                               "top_logprobs": tops}]}
        return web.json_response({
            "id": "chatcmpl-1", "object": "chat.completion", "model": body["model"],
            "choices": [choice],
            "usage": {"prompt_tokens": 100, "completion_tokens": 2,
                      "total_tokens": 102},
        })


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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


def _app(port):
    config = NodeConfig(node_id="local-node", node_name="LocalNode", model=None,
                        api_port=_free_port(), web_port=_free_port(),
                        cluster_enabled=False)
    app = create_app(config=config, engine=None)
    app["cluster_state"].add_node(ClusterNode(
        node_id="spark-1", node_name="Spark-1-DGX", gpu_name="NVIDIA GB10",
        gpu_memory_gb=121.7, unified_memory=True, model=MODEL,
        status=NodeStatus.ONLINE, api_port=port, web_port=port,
        last_seen=0.0, fabric_ip="127.0.0.1"))
    return app


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
async def test_every_question_is_answered_in_one_request(client, engine_fake):
    resp = await client.post("/v1/decide", json=_body())
    assert resp.status == 200
    data = await resp.json()
    assert data["model"] == MODEL
    assert data["node"] == "Spark-1-DGX"
    assert data["latency_ms"] > 0
    assert list(data["decisions"]) == ["category", "urgency", "needs_human"]
    cat = data["decisions"]["category"]
    assert cat["answer"] == "billing"  # the fake always picks the first option
    assert 0.0 < cat["confidence"] <= 1.0
    assert math.isclose(sum(cat["distribution"].values()), 1.0, abs_tol=1e-4)
    assert set(cat["distribution"]) == set(QUESTIONS["category"]["options"])
    assert set(data["decisions"]["urgency"]["distribution"]) == {"1", "2", "3",
                                                                 "4", "5"}
    assert set(data["decisions"]["needs_human"]["distribution"]) == {"yes", "no"}


@pytest.mark.asyncio
async def test_n_questions_are_n_concurrent_engine_calls_and_one_usage(
        client, engine_fake):
    resp = await client.post("/v1/decide", json=_body())
    data = await resp.json()
    assert len(engine_fake.seen) == 3
    # In flight together, not one after another: that is the point of the endpoint.
    assert engine_fake.max_in_flight == 3
    assert data["usage"] == {"prompt_tokens": 300, "completion_tokens": 6,
                             "calls": 3}
    # The whole request is faster than the three calls end to end would be.
    assert data["latency_ms"] < 3 * 50


@pytest.mark.asyncio
async def test_the_engine_sees_a_constrained_body_with_a_shared_prefix(
        client, engine_fake):
    await client.post("/v1/decide", json=_body(instructions="Bias to escalate."))
    assert len(engine_fake.seen) == 3
    prefixes = set()
    for body in engine_fake.seen:
        assert body["model"] == MODEL
        assert body["structured_outputs"]["choice"][0] == "A"
        assert body["logprobs"] is True
        assert body["chat_template_kwargs"] == {"enable_thinking": False,
                                                "thinking": False}
        assert body["messages"][0]["content"].endswith("Bias to escalate.")
        prefixes.add(body["messages"][1]["content"].split("QUESTION:")[0])
    assert len(prefixes) == 1  # identical prefix, so the engine caches it once


@pytest.mark.asyncio
async def test_a_json_object_state_is_serialized_compactly(client, engine_fake):
    await client.post("/v1/decide", json=_body(
        state={"subject": "invoice 500s", "plan": "pro"},
        questions={"k": {"question": "urgent?", "type": "boolean"}}))
    user = engine_fake.seen[0]["messages"][1]["content"]
    assert '{"plan":"pro","subject":"invoice 500s"}' in user


@pytest.mark.asyncio
async def test_the_model_defaults_to_the_only_one_the_fleet_serves(client,
                                                                   engine_fake):
    resp = await client.post("/v1/decide", json={
        "state": TICKET, "questions": {"k": {"question": "urgent?",
                                             "type": "boolean"}}})
    assert resp.status == 200
    assert (await resp.json())["model"] == MODEL


@pytest.mark.asyncio
@pytest.mark.parametrize("body,fragment", [
    ({"state": "s"}, "'questions' must be a non-empty object"),
    ({"state": "s", "questions": {}}, "'questions' must be a non-empty object"),
    ({"questions": {"k": {"question": "q", "options": ["only"]}}},
     "at least 2 options"),
    ({"questions": {"k": {"question": "q",
                          "options": [f"o{i}" for i in range(256)]}}},
     "more than the 255 allowed"),
    ({"questions": {"k": {"question": "q", "options": ["a", "b", "a"]}}},
     "duplicate options"),
    ({"questions": {"k": {"question": "q", "options": ["a", 2]}}},
     "must be a non-empty string"),
    ({"questions": {"k": {"question": "q"}}}, "needs 'options' or a 'type'"),
    ({"questions": {"k": {"question": "q", "options": ["a", "b"]}},
      "instructions": 7}, "'instructions' must be a string"),
    ({"model": "", "questions": {"k": {"question": "q", "options": ["a", "b"]}}},
     "'model' must be a non-empty string"),
])
async def test_a_bad_shape_is_a_400_that_says_what_is_wrong(client, engine_fake,
                                                            body, fragment):
    resp = await client.post("/v1/decide", json=body)
    assert resp.status == 400
    assert fragment in (await resp.json())["error"]["message"]
    assert engine_fake.seen == []  # nothing reached the engine


@pytest.mark.asyncio
async def test_a_body_that_is_not_json_is_a_400(client):
    resp = await client.post("/v1/decide", data=b"{nope",
                             headers={"Content-Type": "application/json"})
    assert resp.status == 400
    assert "not valid JSON" in (await resp.json())["error"]["message"]
    resp = await client.post("/v1/decide", json=["a", "list"])
    assert resp.status == 400
    assert "must be a JSON object" in (await resp.json())["error"]["message"]


@pytest.mark.asyncio
async def test_a_model_no_node_serves_is_a_503(client, engine_fake):
    resp = await client.post("/v1/decide", json=_body(model="who/Knows-3B"))
    assert resp.status == 503
    assert "no node is serving 'who/Knows-3B'" in (await resp.json())["error"]["message"]
    assert engine_fake.seen == []


@pytest.mark.asyncio
async def test_an_unreachable_engine_is_a_503_that_names_the_question():
    """A node that advertises the model but answers nothing. No partial 200."""
    dead = _free_port()
    async with TestClient(TestServer(_app(dead))) as c:
        resp = await c.post("/v1/decide", json=_body())
        assert resp.status == 503
        msg = (await resp.json())["error"]["message"]
        assert "engine calls failed" in msg and "category" in msg


@pytest.mark.asyncio
async def test_an_engine_error_is_a_503_rather_than_half_an_answer():
    broken = FakeEngine(status=400)
    server = TestServer(broken.app())
    await server.start_server()
    try:
        async with TestClient(TestServer(_app(server.port))) as c:
            resp = await c.post("/v1/decide", json=_body())
            assert resp.status == 503
            assert "engine says no" in (await resp.json())["error"]["message"]
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_an_engine_without_logprobs_still_answers_every_question():
    quiet = FakeEngine(with_logprobs=False)
    server = TestServer(quiet.app())
    await server.start_server()
    try:
        async with TestClient(TestServer(_app(server.port))) as c:
            resp = await c.post("/v1/decide", json=_body())
            assert resp.status == 200
            for entry in (await resp.json())["decisions"].values():
                assert entry["distribution"] is None
                assert entry["confidence"] == 1.0
                assert entry["note"] == "no logprobs from engine"
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_decide_is_in_the_endpoint_catalog(client):
    catalog = await (await client.get("/api/server/endpoints")).json()
    paths = [ep["path"] for group in catalog.values() for ep in group]
    assert "/v1/decide" in paths
