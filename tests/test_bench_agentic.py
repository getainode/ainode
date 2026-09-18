"""Tests for ainode.bench.agentic - the agentic capability rubric.

No model, no node, no network. Two kinds of fake stand in for the real thing:

  * canned replies handed straight to the checkers, because a checker is the thing
    that decides a probe and every one of them is a pure function of the reply;
  * a scripted client for the multi-turn probes, so G1's turn driving and G2's error
    path are exercised with real tool-call plumbing and no server.

The verdicts that matter most are G2's: a model that states a temperature the tool
never returned has to fail, and one that says it could not find the city has to pass.
That is the whole reason the group exists.
"""

import copy
import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

from ainode.bench.agentic import cli as agentic_cli
from ainode.bench.agentic import probes as p
from ainode.bench.agentic.probes import (
    ArgumentFidelityProbe,
    CodeProbe,
    DependentToolLoopProbe,
    InstructionPersistenceProbe,
    NeedleProbe,
    StructuredOutputProbe,
    ThinkingOffProbe,
    ToolErrorRecoveryProbe,
    ToolRoundTripProbe,
    VisionProbe,
    all_probes,
    check_a1_format,
    check_a2_json_only,
    check_a3_constraints,
    check_a4_persona,
    check_b1_single,
    check_b2_parallel,
    check_b3_not_needed,
    check_b4_roundtrip,
    check_expected,
    check_g1_trace,
    check_g2_recovery,
    check_g3_arguments,
    check_g4_keys,
    check_g5_replies,
    check_needle,
    check_thinking_off,
    check_vision,
    code_of,
    execute,
    loads_json,
    needle_prompt,
    normalize_city,
    temperatures_in,
)
from ainode.bench.agentic.runner import (
    Reply,
    build_agentic_block,
    build_notes,
    build_record,
    chat_url,
    group_scores,
    needle_map,
    run_probes,
    score,
    structured_output_mode,
    supported,
)
from ainode.bench.cli import main as bench_main

REPO = pathlib.Path(__file__).resolve().parent.parent
RENDERER = REPO / "scripts" / "render-bench-table.py"


# ---------------------------------------------------------------- fakes

def tool_call(name, args, call_id="call-1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def assistant(content=None, tool_calls=None, tokens=12, reasoning=""):
    """A Reply shaped the way ChatClient builds one from a real response."""
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = list(tool_calls)
    return Reply(content=content, reasoning=reasoning, message=message,
                 usage={"completion_tokens": tokens}, wall_s=0.01)


class FakeClient:
    """Scripted replies, with every outgoing payload kept for inspection.

    A script entry is a Reply, or a callable taking (messages, kwargs) so a test can
    answer differently depending on what the probe sent.
    """

    def __init__(self, script=()):
        self.script = list(script)
        self.sent = []

    def thinking_off(self):
        return {"enable_thinking": False, "thinking": False}

    def chat(self, messages, **kw):
        self.sent.append({"messages": copy.deepcopy(messages), **kw})
        if not self.script:
            return Reply(content="", message={"role": "assistant", "content": ""})
        item = self.script.pop(0)
        return item(messages, kw) if callable(item) else item

    def ask(self, prompt, system=None, **kw):
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kw)


def tool_messages(sent):
    """Every tool-role message the probe fed back, in order."""
    out = []
    for payload in sent:
        for message in payload["messages"]:
            if message.get("role") == "tool":
                out.append(message["content"])
    return sorted(set(out), key=out.index)


# ---------------------------------------------------------------- A checkers

def test_a1_accepts_the_asked_for_shape_and_rejects_prose():
    assert check_a1_format("1. JUPITER\n2. SATURN\n3. URANUS\n4. NEPTUNE\n5. EARTH")[0]
    assert check_a1_format("1. JUPITER\n2. SATURN\n3. URANUS\n4. NEPTUNE\n5. EARTH\n")[0]
    # Six lines, four lines, a preamble: all wrong.
    assert not check_a1_format("Sure! Here you go:\n1. JUPITER\n2. SATURN")[0]
    assert not check_a1_format("1. JUPITER\n2. SATURN\n3. URANUS\n4. NEPTUNE")[0]
    assert not check_a1_format("1. Jupiter\n2. Saturn\n3. Uranus\n4. Neptune\n5. Earth")[0]


def test_a1_never_raises_on_a_line_without_a_number():
    """The hand-run script indexed [1] after a split and could blow up on prose."""
    passed, note = check_a1_format("JUPITER\nSATURN\nURANUS\nNEPTUNE\nEARTH")
    assert not passed
    assert "prefix" in note


def test_a2_wants_three_keys_raw_and_landlocked_true():
    good = '{"city": "Vienna", "population_millions": 2.0, "landlocked": true}'
    assert check_a2_json_only(good)[0]
    assert not check_a2_json_only("```json\n" + good + "\n```")[0]
    assert not check_a2_json_only('{"city": "Vienna", "landlocked": true}')[0]
    extra = '{"city": "Vienna", "population_millions": 2, "landlocked": true, "x": 1}'
    assert not check_a2_json_only(extra)[0]
    false = '{"city": "Vienna", "population_millions": 2, "landlocked": false}'
    assert not check_a2_json_only(false)[0]


def test_a3_fails_a_z_or_a_the():
    good = "A rider pushes pedals. That turns a chain. A wheel spins forward."
    assert check_a3_constraints(good)[0]
    assert not check_a3_constraints("A rider pushes the pedals. It spins. Done.")[0]
    assert not check_a3_constraints("A rider zips along. It spins. Done.")[0]
    # One sentence is not three.
    assert not check_a3_constraints("A rider pushes pedals and a wheel spins")[0]


def test_a4_wants_the_persona_alive_after_the_injection():
    assert check_a4_persona("Arr, a mutex be a lock, matey. Arr!")[0]
    assert not check_a4_persona("A mutex is a mutual exclusion primitive.")[0]


# ---------------------------------------------------------------- B checkers

def test_b1_wants_exactly_one_weather_call_naming_the_city():
    assert check_b1_single([("get_weather", '{"city": "Tokyo", "unit": "c"}')])[0]
    assert not check_b1_single([])[0]
    assert not check_b1_single([("get_weather", '{"city": "Osaka"}')])[0]
    assert not check_b1_single([("convert_currency", '{"amount": 1}')])[0]
    twice = [("get_weather", '{"city": "Tokyo"}')] * 2
    assert not check_b1_single(twice)[0]


def test_b2_wants_three_calls_for_three_things():
    calls = [("get_weather", '{"city": "Paris"}'), ("get_weather", '{"city": "Cairo"}'),
             ("convert_currency", '{"amount": 250}')]
    assert check_b2_parallel(calls)[0]
    assert not check_b2_parallel(calls[:2])[0]


def test_b3_wants_no_call_but_still_wants_a_haiku():
    assert check_b3_not_needed([], "Leaves fall in silence")[0]
    assert not check_b3_not_needed([("get_weather", "{}")], "Leaves fall")[0]
    assert not check_b3_not_needed([], "")[0]


def test_b4_wants_the_tool_result_used_and_no_second_call():
    assert check_b4_roundtrip([], "It is 31 C with thunderstorms in Tokyo.")[0]
    assert not check_b4_roundtrip([], "I could not determine the weather.")[0]
    assert not check_b4_roundtrip([("get_weather", "{}")], "It is 31 C.")[0]


def test_b4_makes_its_own_first_call_and_feeds_the_result_back():
    client = FakeClient([
        assistant(tool_calls=[tool_call("get_weather", {"city": "Tokyo"})]),
        assistant("It is 31 C and stormy in Tokyo."),
    ])
    result = ToolRoundTripProbe().run(client)
    assert result.passed, result.note
    assert json.loads(tool_messages(client.sent)[0])["temp_c"] == 31
    assert result.completion_tokens == 24     # summed over both turns


def test_b4_says_so_when_there_was_no_call_to_answer_from():
    result = ToolRoundTripProbe().run(FakeClient([assistant("It is warm in Tokyo.")]))
    assert not result.passed
    assert "no tool call" in result.note


# ---------------------------------------------------------------- C, executed

def test_code_of_takes_the_defining_blocks_not_the_usage_example():
    text = ("Here:\n```python\ndef f():\n    return 1\n```\n"
            "and then\n```python\nprint(f())\n```\n")
    assert code_of(text) == "def f():\n    return 1\n"
    # No fences at all: the reply is taken whole, because some models answer bare.
    assert code_of("def f():\n    return 1") == "def f():\n    return 1"


def test_execute_is_the_verdict_not_the_reply():
    good, note = execute("def top(x):\n    return x", "assert top(1)==1\nprint('PASS')")
    assert good and "asserts passed" in note
    bad, note = execute("def top(x):\n    return 0", "assert top(1)==1\nprint('PASS')")
    assert not bad and note


def test_execute_times_out_rather_than_hanging_the_run():
    passed, note = execute("import time\ntime.sleep(5)", "print('PASS')", timeout=0.5)
    assert not passed
    assert "did not finish" in note


def test_a_code_probe_runs_what_the_model_wrote():
    solution = ("```python\n"
                "def merge_intervals(intervals):\n"
                "    out = []\n"
                "    for start, end in sorted(intervals):\n"
                "        if out and start <= out[-1][1]:\n"
                "            out[-1][1] = max(out[-1][1], end)\n"
                "        else:\n"
                "            out.append([start, end])\n"
                "    return out\n```")
    probe = CodeProbe("C2_intervals", "write it", "assert merge_intervals([[1,3],[2,6]])"
                                                 "==[[1,6]]\nprint('PASS')")
    assert probe.run(FakeClient([assistant(solution)])).passed


# ---------------------------------------------------------------- D checkers

def test_d_answers_ignore_spacing_and_case():
    assert check_expected("9.9", "9.9")[0]
    assert check_expected("The answer is 9.9.", "9.9")[0]
    assert not check_expected("9.11", "9.9")[0]
    assert check_expected("a=knight, b=knave", "A=Knight, B=Knave")[0]
    assert not check_expected("A=Knave, B=Knight", "A=Knight, B=Knave")[0]


# ---------------------------------------------------------------- E needle

def test_the_needle_sits_inside_the_haystack_and_is_asked_for():
    prompt = needle_prompt(8000)
    assert p.NEEDLE_SECRET in prompt
    assert prompt.endswith(p.NEEDLE_QUESTION)
    before = prompt.index(p.NEEDLE_SECRET)
    assert 0.3 < before / len(prompt) < 0.6          # not at either edge
    assert 4000 < len(prompt.split()) < 8000


def test_every_needle_size_is_reproducible_on_its_own():
    """One size run alone gets the haystack it would get in a full sweep."""
    assert needle_prompt(8000) == needle_prompt(8000)
    assert needle_prompt(8000) != needle_prompt(48000)


def test_check_needle_wants_the_password_verbatim():
    assert check_needle(f"{p.NEEDLE_SECRET}")[0]
    assert not check_needle("violet otter 2931")[0]
    assert not check_needle("I could not find a password.")[0]


def test_the_needle_note_carries_the_prompt_tokens_the_server_counted():
    reply = Reply(content=p.NEEDLE_SECRET, message={"role": "assistant"},
                  usage={"prompt_tokens": 8123, "completion_tokens": 9})
    result = NeedleProbe(8000).run(FakeClient([reply]))
    assert result.passed
    assert "prompt_tokens=8123" in result.note


# ---------------------------------------------------------------- F and V

def test_thinking_off_wants_the_answer_and_an_empty_reasoning_channel():
    assert check_thinking_off("391", "")[0]
    assert not check_thinking_off("391", "let me think...")[0]
    assert not check_thinking_off("17 times 23 is 392", "")[0]


def test_the_thinking_probe_sends_both_switch_names():
    client = FakeClient([assistant("391")])
    assert ThinkingOffProbe().run(client).passed
    kwargs = client.sent[0]["chat_template_kwargs"]
    assert kwargs["enable_thinking"] is False and kwargs["thinking"] is False


def test_vision_accepts_the_reds_and_reports_a_refusal_as_one():
    assert check_vision("Red")[0]
    assert check_vision("It is crimson.")[0]
    assert not check_vision("blue")[0]
    refused = Reply(status=400, error="HTTP 400: image input is not supported")
    result = VisionProbe().run(FakeClient([refused]))
    assert not result.passed
    assert "refused the image" in result.note


def test_the_vision_probe_sends_one_data_uri_image():
    client = FakeClient([assistant("red")])
    assert VisionProbe().run(client).passed
    parts = client.sent[0]["messages"][0]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------- G1

def test_g1_wants_list_then_the_right_file_then_the_value():
    assert check_g1_trace(["list_files", "read_file"], ["config.json"], "137")[0]
    # Read the decoy: wrong file and, because the decoy says 4, a wrong answer too.
    assert not check_g1_trace(["list_files", "read_file"], ["config.json.bak"], "4")[0]
    # Answered from memory with no calls at all.
    assert not check_g1_trace([], [], "137")[0]
    # Guessed a filename without listing first.
    assert not check_g1_trace(["read_file"], ["config.json"], "137")[0]
    # Right trace, no final answer within the turn budget.
    assert not check_g1_trace(["list_files", "read_file"], ["config.json"], "")[0]
    # Right trace, wrong number.
    assert not check_g1_trace(["list_files", "read_file"], ["config.json"], "42")[0]


def test_g1_drives_the_turns_and_hands_back_the_listing_then_the_file():
    client = FakeClient([
        assistant(tool_calls=[tool_call("list_files", {}, "c1")]),
        assistant(tool_calls=[tool_call("read_file", {"path": "config.json"}, "c2")]),
        assistant("137"),
    ])
    probe = DependentToolLoopProbe()
    result = probe.run(client)
    assert result.passed, result.note
    assert probe.names == ["list_files", "read_file"]
    assert probe.turns == 3
    listing, config = tool_messages(client.sent)
    assert "config.json.bak" in listing            # the decoy is offered
    assert "137" in config
    assert result.completion_tokens == 36
    # The tool results went back under the ids the calls carried.
    tool_turn = client.sent[1]["messages"]
    assert tool_turn[-1]["tool_call_id"] == "c1"


def test_g1_fails_a_model_that_reads_the_stale_copy():
    client = FakeClient([
        assistant(tool_calls=[tool_call("list_files", {})]),
        assistant(tool_calls=[tool_call("read_file", {"path": "config.json.bak"})]),
        assistant("4"),
    ])
    result = DependentToolLoopProbe().run(client)
    assert not result.passed
    assert "config.json" in result.note


def test_g1_stops_at_the_turn_budget():
    """A model that only ever calls tools does not run forever."""
    client = FakeClient([assistant(tool_calls=[tool_call("list_files", {})])
                         for _ in range(10)])
    probe = DependentToolLoopProbe(max_turns=4)
    result = probe.run(client)
    assert not result.passed
    assert probe.turns == 4
    assert len(client.sent) == 4


def test_g1_reports_a_missing_file_instead_of_pretending():
    client = FakeClient([
        assistant(tool_calls=[tool_call("list_files", {})]),
        assistant(tool_calls=[tool_call("read_file", {"path": "nope.txt"})]),
        assistant("137"),
    ])
    result = DependentToolLoopProbe().run(client)
    assert not result.passed
    assert "no such file" in tool_messages(client.sent)[1]


# ---------------------------------------------------------------- G2

def test_temperatures_in_reads_a_stated_reading_not_every_number():
    assert temperatures_in("It is 18 C and foggy") == [18.0]
    assert temperatures_in("22°C right now") == [22.0]
    assert temperatures_in("the temperature is 15 degrees") == [15.0]
    assert temperatures_in("I called the tool 2 times and failed") == []


def test_normalize_city_folds_spacing_and_case_but_not_a_typo():
    assert normalize_city("San Francisco") == normalize_city("san francisco")
    assert normalize_city("Sanfrancsico") != normalize_city("San Francisco")


def test_g2_fails_a_fabricated_temperature():
    passed, note = check_g2_recovery([p.G2_BAD_CITY], [], "It is 22 C in the city.")
    assert not passed
    assert "22" in note and "never received" in note


def test_g2_passes_a_corrected_retry():
    passed, note = check_g2_recovery([p.G2_BAD_CITY, "San Francisco"], [18.0],
                                     "It is 18 C and foggy in San Francisco.")
    assert passed
    assert "San Francisco" in note and "after the error" in note


def test_g2_passes_a_model_that_normalized_before_it_had_to():
    """The first real run failed on this: the model fixed the spelling on the first
    call, so it never "retried" and never fabricated either."""
    passed, note = check_g2_recovery(["San Francisco"], [18.0],
                                     "The temperature in San Francisco is 18 C.")
    assert passed
    assert "on the first call" in note


def test_g2_passes_an_honest_failure():
    passed, note = check_g2_recovery([p.G2_BAD_CITY], [],
                                     "I could not find that city. Check the spelling?")
    assert passed
    assert "could not find" in note


def test_g2_fails_a_silent_give_up():
    passed, note = check_g2_recovery([p.G2_BAD_CITY], [], "Let me know if I can help.")
    assert not passed
    assert "did not retry" in note


def test_g2_fails_a_model_that_never_called_the_tool():
    passed, note = check_g2_recovery([], [], "I am not sure.")
    assert not passed
    assert "never called" in note


def test_g2_returns_the_error_first_then_the_reading():
    client = FakeClient([
        assistant(tool_calls=[tool_call("get_weather", {"city": p.G2_BAD_CITY})]),
        assistant(tool_calls=[tool_call("get_weather", {"city": "San Francisco"}, "c2")]),
        assistant("It is 18 C and foggy in San Francisco."),
    ])
    probe = ToolErrorRecoveryProbe()
    result = probe.run(client)
    assert result.passed, result.note
    first, second = tool_messages(client.sent)
    assert json.loads(first) == {"error": "city not found"}
    assert json.loads(second)["temp_c"] == p.G2_TEMP_C
    assert probe.delivered == [float(p.G2_TEMP_C)]


def test_g2_catches_the_model_that_invents_a_reading_after_the_error():
    client = FakeClient([
        assistant(tool_calls=[tool_call("get_weather", {"city": p.G2_BAD_CITY})]),
        assistant("It is currently 22 C and sunny in Sanfrancsico."),
    ])
    result = ToolErrorRecoveryProbe().run(client)
    assert not result.passed
    assert "22" in result.note


def test_g2_errors_again_on_an_identical_retry():
    """Retrying the same misspelling is not recovery, so the tool keeps failing."""
    client = FakeClient([
        assistant(tool_calls=[tool_call("get_weather", {"city": p.G2_BAD_CITY})]),
        assistant(tool_calls=[tool_call("get_weather", {"city": "sanfrancsico"}, "c2")]),
        assistant("I could not find that city."),
    ])
    probe = ToolErrorRecoveryProbe()
    result = probe.run(client)
    assert probe.delivered == []
    assert result.passed          # it admitted the failure, which is the other half
    assert "could not find" in result.note


# ---------------------------------------------------------------- G3

def test_g3_wants_the_declared_types_and_nothing_extra():
    good = [("schedule_job", '{"mode": "daily", "retries": 3, "notify": true}')]
    passed, note = check_g3_arguments(good)
    assert passed and "daily" in note
    # Already-parsed arguments are accepted: some servers hand back an object.
    assert check_g3_arguments([("schedule_job",
                               {"mode": "once", "retries": 0, "notify": False})])[0]


@pytest.mark.parametrize("args,reason", [
    ('{"mode": "nightly", "retries": 3, "notify": true}', "enum"),
    ('{"mode": "daily", "retries": "3", "notify": true}', "integer"),
    ('{"mode": "daily", "retries": 3, "notify": "yes"}', "boolean"),
    ('{"mode": "daily", "retries": 3}', "missing"),
    ('{"mode": "daily", "retries": 3, "notify": true, "job": "backup"}', "invented"),
    ('{"mode": "daily", "retries": 3.5, "notify": true}', "integer"),
    ('{"mode": "daily", "retries": true, "notify": true}', "integer"),
    ("mode=daily", "JSON"),
])
def test_g3_rejects_each_way_the_arguments_can_be_wrong(args, reason):
    passed, note = check_g3_arguments([("schedule_job", args)])
    assert not passed
    assert reason.lower() in note.lower()


def test_g3_wants_one_call_to_the_right_tool():
    assert not check_g3_arguments([])[0]
    assert not check_g3_arguments([("get_weather", '{"city": "Tokyo"}')])[0]


def test_the_g3_probe_offers_one_tool_with_the_three_types():
    client = FakeClient([assistant(tool_calls=[tool_call(
        "schedule_job", {"mode": "daily", "retries": 3, "notify": True})])])
    assert ArgumentFidelityProbe().run(client).passed
    schema = client.sent[0]["tools"][0]["function"]["parameters"]
    assert schema["properties"]["mode"]["enum"] == list(p.G3_MODES)
    assert schema["properties"]["retries"]["type"] == "integer"
    assert sorted(schema["required"]) == ["mode", "notify", "retries"]


# ---------------------------------------------------------------- G4

def test_loads_json_prefers_raw_but_reads_a_fenced_object():
    assert loads_json('{"a": 1}')[0] == {"a": 1}
    assert loads_json('```json\n{"a": 1}\n```')[0] == {"a": 1}
    assert loads_json('Here you go: {"a": 1} hope that helps')[0] == {"a": 1}
    assert loads_json("no json here")[0] is None


def test_g4_wants_the_required_keys_with_the_right_types():
    good = ('{"title": "Login 500s", "severity": 2, "resolved": true, '
            '"tags": ["login", "deploy"]}')
    assert check_g4_keys(good)[0]
    assert not check_g4_keys('{"title": "x", "severity": 2, "resolved": true}')[0]
    assert not check_g4_keys('{"title": "x", "severity": "2", "resolved": true, '
                             '"tags": []}')[0]
    assert not check_g4_keys('{"title": "x", "severity": true, "resolved": true, '
                             '"tags": []}')[0]
    assert not check_g4_keys('{"title": "x", "severity": 2, "resolved": "yes", '
                             '"tags": []}')[0]
    assert not check_g4_keys('{"title": "x", "severity": 2, "resolved": true, '
                             '"tags": [1]}')[0]


def test_g4_records_json_schema_when_the_server_takes_it():
    body = ('{"title": "Login 500s", "severity": 2, "resolved": true, '
            '"tags": ["login"]}')
    client = FakeClient([assistant(body)])
    probe = StructuredOutputProbe()
    result = probe.run(client)
    assert result.passed
    assert probe.mode == "json_schema"
    assert client.sent[0]["response_format"]["type"] == "json_schema"
    assert "[json_schema]" in result.note


def test_g4_falls_back_to_json_object_on_a_400_and_says_which_ran():
    body = ('{"title": "Login 500s", "severity": 2, "resolved": true, '
            '"tags": ["login"]}')
    client = FakeClient([Reply(status=400, error="HTTP 400: unsupported"),
                         assistant(body)])
    probe = StructuredOutputProbe()
    result = probe.run(client)
    assert result.passed
    assert probe.mode == "json_object"
    assert client.sent[1]["response_format"] == {"type": "json_object"}
    assert "[json_object]" in result.note


def test_g4_records_unsupported_when_the_server_refuses_both():
    client = FakeClient([Reply(status=400, error="HTTP 400: no"),
                         Reply(status=400, error="HTTP 400: no")])
    probe = StructuredOutputProbe()
    result = probe.run(client)
    assert not result.passed
    assert probe.mode == "unsupported"


# ---------------------------------------------------------------- G5

def test_g5_wants_the_rule_on_every_turn():
    assert check_g5_replies(["Paris\nDONE"] * 4)[0]
    passed, note = check_g5_replies(["Paris\nDONE", "Seine, Loire\nDONE",
                                     "The Seine is long.", "Paris\nDONE"])
    assert not passed
    assert "3" in note                  # the turn that dropped it is named
    assert not check_g5_replies([])[0]
    # DONE has to be the last line on its own, not a word in a sentence.
    assert not check_g5_replies(["I am DONE with that"] * 4)[0]


def test_g5_drives_four_turns_and_keeps_the_system_rule_in_front():
    client = FakeClient([assistant(f"answer {n}\nDONE") for n in range(1, 5)])
    probe = InstructionPersistenceProbe()
    result = probe.run(client)
    assert result.passed, result.note
    assert len(client.sent) == 4
    assert client.sent[0]["messages"][0] == {"role": "system", "content": p.G5_SYSTEM}
    assert client.sent[-1]["messages"][0]["role"] == "system"
    # The fourth turn carries the whole conversation, not just the last question.
    assert len(client.sent[-1]["messages"]) == 1 + 4 + 3


def test_g5_fails_the_turn_where_the_rule_slipped():
    client = FakeClient([assistant("a\nDONE"), assistant("b\nDONE"),
                         assistant("c"), assistant("d\nDONE")])
    result = InstructionPersistenceProbe().run(client)
    assert not result.passed
    assert "3" in result.note


# ---------------------------------------------------------------- the loop

def test_a_probe_that_raises_is_one_failure_not_a_dead_run():
    class Exploding(p.Probe):
        def run(self, client):
            raise RuntimeError("boom")

    runs = run_probes([Exploding("A9_boom"), p.Ask("A8_ok", "hi", lambda t: (True, "ok"))],
                      FakeClient([assistant("hello")]), log=lambda *a: None)
    assert [r.passed for r in runs] == [False, True]
    assert "probe raised RuntimeError" in runs[0].note


def test_a_transport_error_is_a_failed_probe_with_the_reason():
    runs = run_probes([p.Ask("A1_format", "x", check_a1_format)],
                      FakeClient([Reply(error="URLError: connection refused")]),
                      log=lambda *a: None)
    assert not runs[0].passed
    assert "connection refused" in runs[0].note


def test_scores_are_counts_of_what_ran():
    runs = run_probes(
        [p.Ask("A1_format", "x", lambda t: (True, "")),
         p.Ask("A2_json_only", "x", lambda t: (False, "no")),
         p.Ask("D1_decimal", "x", lambda t: (True, ""))],
        FakeClient([assistant("a"), assistant("b"), assistant("c")]),
        log=lambda *a: None)
    assert score(runs) == {"pass": 2, "total": 3}
    assert group_scores(runs) == {"A": {"pass": 1, "total": 2},
                                  "D": {"pass": 1, "total": 1}}


def test_chat_url_takes_a_base_with_or_without_its_v1():
    assert chat_url("http://host:3000/v1") == "http://host:3000/v1/chat/completions"
    assert chat_url("http://host:3000/v1/") == "http://host:3000/v1/chat/completions"
    assert chat_url("http://host:3000") == "http://host:3000/v1/chat/completions"


def test_a_group_nobody_ran_is_absent_rather_than_zero():
    runs = run_probes([p.Ask("A1_format", "x", lambda t: (True, ""))],
                      FakeClient([assistant("a")]), log=lambda *a: None)
    block = build_agentic_block(runs, [], "http://h/v1", ["A"], [])
    assert list(block["groups"]) == ["A"]
    assert block["needle"] == {}
    assert block["thinking_off_supported"] is None
    assert block["vision_supported"] is None
    assert block["structured_output_mode"] is None


def test_supported_is_none_for_a_probe_that_did_not_run():
    runs = run_probes([ThinkingOffProbe()], FakeClient([assistant("391")]),
                      log=lambda *a: None)
    assert supported(runs, "F1_thinking_off") is True
    assert supported(runs, "V1_vision") is None
    failing = run_probes([ThinkingOffProbe()],
                         FakeClient([assistant("391", reasoning="hmm")]),
                         log=lambda *a: None)
    assert supported(failing, "F1_thinking_off") is False


def test_needle_map_keys_are_the_requested_sizes():
    runs = run_probes([NeedleProbe(8000), NeedleProbe(48000)],
                      FakeClient([assistant(p.NEEDLE_SECRET), assistant("no idea")]),
                      log=lambda *a: None)
    assert needle_map(runs) == {"8000": True, "48000": False}


def test_structured_output_mode_comes_off_the_probe_that_ran():
    probe = StructuredOutputProbe()
    body = '{"title": "t", "severity": 1, "resolved": false, "tags": []}'
    runs = run_probes([probe], FakeClient([Reply(status=400, error="HTTP 400"),
                                           assistant(body)]), log=lambda *a: None)
    assert structured_output_mode([probe], runs) == "json_object"
    assert structured_output_mode([StructuredOutputProbe()], runs) is None


def test_the_notes_separate_a_server_error_from_a_wrong_answer():
    runs = run_probes([p.Ask("A1_format", "x", check_a1_format),
                       p.Ask("A2_json_only", "x", check_a2_json_only)],
                      FakeClient([Reply(status=400, error="HTTP 400: nope"),
                                  assistant("not json")]),
                      log=lambda *a: None)
    notes = build_notes(runs, 12, ["A"])
    flagged = [n for n in notes if "transport or server error" in n]
    assert flagged and "A1_format" in flagged[0] and "A2_json_only" not in flagged[0]


# ---------------------------------------------------------------- the set

def test_the_default_set_is_every_group_and_ids_carry_their_group():
    probes = all_probes()
    assert len(probes) == 25
    assert {pr.group for pr in probes} == set(p.GROUPS)
    for probe in probes:
        assert probe.group == probe.id[0]
    assert len({pr.id for pr in probes}) == len(probes)


def test_groups_and_needle_select():
    probes = all_probes(needle=[8000], groups=["B", "G"])
    assert [pr.id for pr in probes][:1] == ["B1_tool_single"]
    assert all(pr.group in ("B", "G") for pr in probes)
    assert len(probes) == 9
    assert [pr.id for pr in all_probes(needle=[8000, 200000], groups=["E"])] == \
        ["E_needle_8000", "E_needle_200000"]


# ---------------------------------------------------------------- the record

def scripted_cli_client(monkeypatch, script):
    """Point the CLI at a scripted client instead of the network."""
    client = FakeClient(script)

    def factory(*args, **kwargs):
        return client

    monkeypatch.setattr(agentic_cli, "ChatClient", factory)
    return client


def test_the_record_has_an_agentic_block_and_no_invented_throughput(tmp_path,
                                                                   monkeypatch,
                                                                   capsys):
    """The shape bench/SCHEMA.md documents, written by the CLI."""
    scripted_cli_client(monkeypatch, [
        assistant("9.9"), assistant("3"), assistant("2"),
        assistant("A=Knight, B=Knave"), assistant("391"),
    ])
    code = agentic_cli.main(["--endpoint", "http://node:3000/v1", "--model", "org/M-1",
                             "--label", "fake run, D and F", "--groups", "D,F"],
                            out_dir=tmp_path)
    assert code == 0
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert files[0].name.endswith("-m-1-fake-run-d-and-f-agentic.json")

    record = json.loads(files[0].read_text())
    assert record["schema"] == 1
    assert record["source"] == "scripts/ainode-bench.py agentic"
    assert record["label"] == "fake run, D and F"
    assert "results" not in record            # this run took no throughput
    assert record["model"] == {"id": "org/M-1"}
    assert record["placement"] == {}          # no --ainode, so no guessed placement
    assert record["settings"]["groups"] == ["D", "F"]
    assert record["settings"]["temperature"] == 1.0

    block = record["agentic"]
    assert block["score"] == {"pass": 5, "total": 5}
    assert block["groups"] == {"D": {"pass": 4, "total": 4},
                              "F": {"pass": 1, "total": 1}}
    assert block["endpoint"] == "http://node:3000/v1"
    assert block["thinking_off_supported"] is True
    assert block["vision_supported"] is None
    assert block["structured_output_mode"] is None
    assert block["needle"] == {}
    assert [pr["id"] for pr in block["probes"]] == [
        "D1_decimal", "D2_strawberry", "D3_sisters", "D4_knights", "F1_thinking_off"]
    first = block["probes"][0]
    assert set(first) == {"id", "group", "pass", "wall_s", "completion_tokens",
                          "note", "excerpt"}
    assert first["group"] == "D" and first["pass"] is True
    assert any("mechanical" in note for note in record["notes"])
    assert "SCORE 5/5" in capsys.readouterr().out


def test_the_record_is_valid_json_with_a_trailing_newline(tmp_path, monkeypatch):
    scripted_cli_client(monkeypatch, [assistant("9.9"), assistant("3"),
                                      assistant("2"), assistant("A=Knight, B=Knave")])
    agentic_cli.main(["--endpoint", "http://node:3000/v1", "--model", "org/M",
                      "--label", "x", "--groups", "D"], out_dir=tmp_path)
    text = list(tmp_path.glob("*.json"))[0].read_text()
    assert text.endswith("\n")
    json.loads(text)


def test_build_record_keeps_the_top_level_shape_the_harness_record_has():
    record = build_record("lab", {"id": "x/y"}, {"node": "Spark-1"},
                          {"score": {"pass": 1, "total": 1}}, {"groups": ["A"]},
                          ["note"], "20260917-000000")
    assert list(record) == ["schema", "stamp", "label", "model", "placement",
                            "settings", "agentic", "notes", "source"]


def test_quick_drops_vision_and_pins_the_cheap_needle(capsys):
    code = agentic_cli.main(["--endpoint", "http://node:3000/v1", "--model", "m",
                             "--label", "l", "--quick", "--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    assert "V1_vision" not in out
    assert "E_needle_8000" in out and "E_needle_100000" not in out


def test_dry_run_lists_the_probes_and_writes_nothing(tmp_path, capsys):
    code = agentic_cli.main(["--endpoint", "http://node:3000/v1", "--model", "m",
                             "--label", "l", "--dry-run"], out_dir=tmp_path)
    assert code == 0
    out = capsys.readouterr().out
    assert "POST http://node:3000/v1/chat/completions" in out
    assert "G2_tool_error" in out
    assert "no file was written" in out
    assert not list(tmp_path.glob("*"))


def test_an_unknown_group_is_refused_before_a_request(capsys):
    with pytest.raises(SystemExit):
        agentic_cli.main(["--endpoint", "http://n/v1", "--model", "m", "--label", "l",
                          "--groups", "Z"])
    assert "unknown group" in capsys.readouterr().err


def test_the_label_is_required():
    with pytest.raises(SystemExit):
        agentic_cli.main(["--endpoint", "http://n/v1", "--model", "m"])


def test_the_agentic_subcommand_is_reachable_from_the_shim(capsys, tmp_path):
    assert bench_main(["agentic", "--endpoint", "http://n/v1", "--model", "m",
                       "--label", "l", "--dry-run"], out_dir=tmp_path) == 0
    assert "probe(s)" in capsys.readouterr().out


def test_the_flat_bench_cli_still_owns_every_other_flag(tmp_path, capsys):
    with pytest.raises(SystemExit):
        bench_main(["--only", "nonsense", "--url", "http://x", "--model", "m",
                    "--label", "l"], out_dir=tmp_path)
    assert "unknown section" in capsys.readouterr().err


# ---------------------------------------------------------------- the README table

def _renderer():
    spec = importlib.util.spec_from_file_location("render_bench_table", RENDERER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _agentic_record(stamp="20260101-000001", name="Xeno", node="Spark-5", tp=2,
                    groups=None, needle=None, score_pair=(20, 25)):
    return {
        "schema": 1, "stamp": stamp, "label": "my-run",
        "model": {"id": "x/y", "name": name, "params_b": 7, "arch": "moe"},
        "placement": {"node": node, "gpu": "NVIDIA GB10", "gpus": tp, "tp": tp},
        "agentic": {
            "score": {"pass": score_pair[0], "total": score_pair[1]},
            "groups": groups if groups is not None else {
                "A": {"pass": 4, "total": 4}, "B": {"pass": 3, "total": 4},
                "C": {"pass": 2, "total": 3}, "D": {"pass": 4, "total": 4},
                "E": {"pass": 2, "total": 3}, "F": {"pass": 1, "total": 1},
                "G": {"pass": 4, "total": 5}},
            "needle": needle if needle is not None else {"8000": True, "48000": True,
                                                         "100000": False},
        },
    }


def _write(directory, name, obj):
    (directory / name).write_text(json.dumps(obj))


def test_an_agentic_record_renders_a_row(tmp_path):
    m = _renderer()
    _write(tmp_path, "20260101-000001-agentic.json", _agentic_record())
    table = m.render_agentic_table(m.load_runs(tmp_path))
    lines = table.splitlines()
    assert len(lines) == 3                     # header + rule + one row
    assert "| Xeno | Spark-5, TP=2 | 20/25 | 7/9 | 2/3 | 4/4 | 48k |" in table
    assert ("[my-run](https://github.com/getainode/ainode/blob/main/bench/results/"
            "20260101-000001-agentic.json)") in table


def test_the_tools_column_adds_b_and_g_because_they_are_one_skill(tmp_path):
    m = _renderer()
    block = {"groups": {"B": {"pass": 4, "total": 4}, "G": {"pass": 1, "total": 5}}}
    assert m.fmt_group(block, m.TOOL_GROUPS) == "5/9"
    assert m.fmt_group({"groups": {}}, m.TOOL_GROUPS) == m.NOT_MEASURED
    # A run that only did B still reports what it measured, not a padded total.
    assert m.fmt_group({"groups": {"B": {"pass": 2, "total": 4}}}, m.TOOL_GROUPS) == "2/4"


def test_the_needle_column_is_the_largest_size_that_passed():
    m = _renderer()
    assert m.fmt_needle({"needle": {"8000": True, "48000": True, "100000": False}}) \
        == "48k"
    assert m.fmt_needle({"needle": {"8000": True}}) == "8k"
    assert m.fmt_needle({"needle": {"8000": False}}) == "none"
    assert m.fmt_needle({}) == m.NOT_MEASURED


def test_a_skipped_group_renders_not_measured_rather_than_zero(tmp_path):
    m = _renderer()
    _write(tmp_path, "20260101-000001-agentic.json", _agentic_record(
        groups={"A": {"pass": 4, "total": 4}}, needle={}, score_pair=(4, 4)))
    row = m.render_agentic_table(m.load_runs(tmp_path)).splitlines()[-1]
    assert "| 4/4 | not measured | not measured | not measured | not measured |" in row


def test_the_speed_table_ignores_agentic_records(tmp_path):
    """An agentic record took no tok/s, so it is not a very slow model."""
    m = _renderer()
    _write(tmp_path, "20260101-000001-agentic.json", _agentic_record())
    _write(tmp_path, "20260101-000002-speed.json", {
        "schema": 1, "stamp": "20260101-000002", "label": "thr",
        "model": {"id": "a/b", "name": "Thr", "params_b": 7, "arch": "dense"},
        "placement": {"node": "N", "gpus": 1, "tp": 1},
        "results": {"single_stream": {"decode_tok_s": 10.0}},
    })
    runs = m.load_runs(tmp_path)
    assert [r["_file"] for r in m.throughput_runs(runs)] == ["20260101-000002-speed.json"]
    assert "20260101-000001-agentic.json" not in m.render_table(runs)


def test_a_record_with_both_blocks_gets_a_row_in_both_tables(tmp_path):
    m = _renderer()
    record = _agentic_record()
    record["results"] = {"single_stream": {"decode_tok_s": 26.2}}
    _write(tmp_path, "20260101-000001-both.json", record)
    runs = m.load_runs(tmp_path)
    assert not m.is_agentic_run(record)
    assert "20260101-000001-both.json" in m.render_table(runs)
    assert "20260101-000001-both.json" in m.render_agentic_table(runs)


def test_the_later_agentic_record_wins_for_one_model(tmp_path):
    m = _renderer()
    _write(tmp_path, "early.json", _agentic_record(stamp="20260101-000001",
                                                   score_pair=(10, 25)))
    _write(tmp_path, "late.json", _agentic_record(stamp="20260201-000001",
                                                  node="Spark-6",
                                                  score_pair=(24, 25)))
    table = m.render_agentic_table(m.load_runs(tmp_path))
    assert len(table.splitlines()) == 3
    assert "| 24/25 |" in table and "10/25" not in table
    assert "Spark-6" in table


def test_check_detects_a_stale_agentic_table(tmp_path):
    """--check returns 1 when only the agentic table of the README has drifted."""
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
    _write(results, "agentic.json", _agentic_record(stamp="20260101-000002"))

    readme = tmp_path / "README.md"
    runs = m.load_runs(results)
    readme.write_text(
        "# Bench\n\n"
        f"{m.BEGIN}\n\n{m.render_table(runs)}\n\n{m.END}\n\n"
        f"{m.AGENTIC_BEGIN}\n\n{m.render_agentic_table(runs)}\n\n{m.AGENTIC_END}\n")

    def check():
        proc = subprocess.run(
            [sys.executable, str(RENDERER), "--check", "--results", str(results),
             "--readme", str(readme)],
            capture_output=True, text=True, cwd=REPO)
        return proc.returncode

    assert check() == 0
    readme.write_text(readme.read_text().replace("20/25", "25/25"))
    assert check() == 1


def test_the_committed_readme_matches_the_committed_records():
    """The drift guard, over all three tables, against what is in the repo."""
    proc = subprocess.run([sys.executable, str(RENDERER), "--check"],
                          capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "agentic rows" in proc.stdout


def test_every_agentic_record_in_the_repo_has_the_documented_shape():
    """bench/SCHEMA.md's `agentic` block, checked against the committed records."""
    m = _renderer()
    runs = [r for r in m.load_runs(REPO / "bench" / "results") if r.get("agentic")]
    for run in runs:
        block = run["agentic"]
        assert set(block["score"]) == {"pass", "total"}
        assert block["score"]["total"] == len(block["probes"])
        assert block["score"]["pass"] == sum(1 for pr in block["probes"] if pr["pass"])
        assert sum(g["total"] for g in block["groups"].values()) \
            == block["score"]["total"]
        for probe in block["probes"]:
            assert probe["group"] == probe["id"][0]
            assert set(probe) >= {"id", "group", "pass", "wall_s", "note", "excerpt"}
        assert block["structured_output_mode"] in (None, "json_schema", "json_object",
                                                   "unsupported")
        for key in ("thinking_off_supported", "vision_supported"):
            assert block[key] in (None, True, False)
        assert "results" not in run or run["results"]
        assert run["source"] == "scripts/ainode-bench.py agentic"
