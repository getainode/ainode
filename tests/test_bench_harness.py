"""Tests for ainode.bench.harness - the coding-harness bench.

No model, no node, no network. Two kinds of fake stand in for the real thing:

  * an adapter whose "agent" is a python one-liner, so ``run()``'s real subprocess,
    timeout and environment handling are exercised while the behaviour under test
    (solve it, get it wrong, hang, be missing) is chosen by the test;
  * the vendored task set itself, which is real, because the isolation guarantee is
    a property of those directories and faking them would test nothing.

The two tests that matter most are the isolation pair: the hidden tests must not be
in the working directory while the harness runs, and must be there when pytest
runs. Everything else is arithmetic and argv.
"""

import json
import pathlib
import sys

import pytest

from ainode.bench.harness import adapters as adapters_mod
from ainode.bench.harness.adapters import (
    PROVIDER,
    HarnessAdapter,
    HarnessRequest,
)
from ainode.bench.harness.adapters.aider import AiderAdapter, parse_tokens
from ainode.bench.harness.adapters.claude import (
    ClaudeAdapter,
    messages_base,
    parse_result,
)
from ainode.bench.harness.adapters.dsh import (
    API_KEY_ENV,
    DshAdapter,
    api_key_envs,
    patch_overlay,
    settings_yaml,
    thinking_format,
)
from ainode.bench.harness.adapters.opencode import (
    OpencodeAdapter,
    parse_events,
    project_config,
)
from ainode.bench.harness.adapters.pi import PiAdapter, merge_models_json
from ainode.bench.harness.cli import ainode_base
from ainode.bench.harness.cli import main as harness_main
from ainode.bench.harness.runner import (
    HarnessBenchError,
    TaskResult,
    build_harness_block,
    build_notes,
    build_prompt,
    build_record,
    parse_pytest_counts,
    resolve_test_command,
    run_task,
    run_suite,
    score,
)
from ainode.bench.harness.tasks import (
    TaskError,
    default_tasks_dir,
    load_task,
    load_tasks,
    task_set,
)

ENDPOINT = "http://fake-node.invalid:3000/v1"
MODEL = "fakeorg/Fake-Coder-30B-A3B-NVFP4"

# two-fer is the task the fake agents "solve": one function, three assertions.
SOLUTION = 'def two_fer(name="you"):\n    return f"One for {name}, one for me."\n'
WRONG = 'def two_fer(name="you"):\n    return "nope"\n'

#: The fake agent. Records what it could see, then writes whichever solution the
#: test asked for - keyed on whether the prompt already carries a failure report,
#: which is how the attempt-2 behaviour gets tested without a model.
SCRIPT = """
import json, os, pathlib, sys
prompt = sys.argv[1]
seen = sorted(p for p in os.listdir(".") if not p.startswith("."))
with open("seen.jsonl", "a") as fh:
    fh.write(json.dumps({"files": seen, "prompt": prompt}) + "\\n")
mode = os.environ["FAKE_MODE"]
second = "did not pass the hidden tests" in prompt
if mode == "solve" or (mode == "second-try" and second):
    pathlib.Path("two_fer.py").write_text(SOLUTION)
elif mode == "wrong":
    pathlib.Path("two_fer.py").write_text(WRONG)
elif mode == "hang":
    import time; time.sleep(30)
elif mode == "boom":
    sys.exit(3)
""".replace("SOLUTION", repr(SOLUTION)).replace("WRONG", repr(WRONG))


class ScriptAdapter(HarnessAdapter):
    """A harness that is a python one-liner instead of a coding agent."""

    name = "fake"
    binary = sys.executable

    def __init__(self, mode="solve"):
        self.mode = mode

    def version(self):
        return "fake 1.0"

    def command(self, req):
        return [self.binary, "-c", SCRIPT, req.prompt]

    def env(self, req):
        return {"FAKE_MODE": self.mode}


class MissingAdapter(HarnessAdapter):
    name = "missing"
    binary = "ainode-harness-no-such-binary"

    def version(self):
        return None

    def command(self, req):
        return [self.binary, req.prompt]


@pytest.fixture
def tasks_dir():
    return default_tasks_dir()


@pytest.fixture
def two_fer(tasks_dir):
    return load_tasks(tasks_dir, slugs=["two-fer"])[0]


def _request(tmp_path, entry="two_fer.py", prompt="do the thing", **kwargs):
    return HarnessRequest(workdir=tmp_path / "work", scratch=tmp_path / "scratch",
                          prompt=prompt, entry=entry, endpoint=ENDPOINT, model=MODEL,
                          **kwargs)


# ---------------------------------------------------------------- the task set

def test_the_vendored_set_loads_and_is_complete(tasks_dir):
    meta = task_set(tasks_dir)
    tasks = load_tasks(tasks_dir)
    assert meta["id"] == "exercism-python-10"
    assert meta["count"] == len(tasks) == 10
    assert meta["source"]["license"] == "MIT"
    for task in tasks:
        assert task.instructions.strip(), f"{task.slug} has empty instructions"
        assert task.test_files, f"{task.slug} has no hidden tests"
        assert task.stub != "", f"{task.slug} has an empty stub"
        assert task.source.get("commit"), f"{task.slug} does not say where it came from"


def test_hidden_tests_live_under_tests_not_at_the_task_root(tasks_dir):
    """The isolation guarantee is structural, so it is checked on load."""
    for task in load_tasks(tasks_dir):
        for rel in task.test_files:
            assert rel.startswith("tests/"), f"{task.slug}: {rel} is not hidden"
        root_files = {p.name for p in task.directory.iterdir() if p.is_file()}
        assert not any(n.endswith("_test.py") for n in root_files), task.slug


def test_a_task_with_a_test_file_at_its_root_is_rejected(tmp_path):
    (tmp_path / "instructions.md").write_text("do it")
    (tmp_path / "thing.py").write_text("pass\n")
    (tmp_path / "thing_test.py").write_text("def test_x(): pass\n")
    (tmp_path / "task.json").write_text(json.dumps({
        "slug": "thing", "entry": "thing.py", "instructions": "instructions.md",
        "tests": ["thing_test.py"],
        "test_command": ["python", "-m", "pytest", "-q", "thing_test.py"]}))
    with pytest.raises(TaskError, match="hidden tests must live under tests/"):
        load_task(tmp_path)


def test_task_slice_is_deterministic(tasks_dir):
    assert [t.slug for t in load_tasks(tasks_dir, limit=3)] == \
           [t.slug for t in load_tasks(tasks_dir, limit=3)]
    assert [t.slug for t in load_tasks(tasks_dir, limit=3)] == \
           [t.slug for t in load_tasks(tasks_dir)][:3]
    with pytest.raises(TaskError, match="the set has 10"):
        load_tasks(tasks_dir, limit=99)
    with pytest.raises(TaskError, match="unknown task"):
        load_tasks(tasks_dir, slugs=["not-an-exercise"])


# ---------------------------------------------------------------- the prompt

def test_the_prompt_carries_the_instructions_and_the_one_file_rule(two_fer):
    prompt = build_prompt(two_fer)
    assert two_fer.instructions.strip()[:40] in prompt
    assert "Edit only two_fer.py" in prompt
    assert "do not write, create or modify any test file" in prompt.lower()
    assert "did not pass" not in prompt


def test_the_second_prompt_carries_the_failure_output(two_fer):
    prompt = build_prompt(two_fer, failure="E   AssertionError: 'nope' != 'One for you'")
    assert "did not pass the hidden tests" in prompt
    assert "AssertionError: 'nope' != 'One for you'" in prompt
    assert "Fix two_fer.py so the tests pass." in prompt


# ---------------------------------------------------------------- adapters: argv

def test_aider_command_is_pinned(tmp_path):
    req = _request(tmp_path, prompt="solve isogram")
    assert AiderAdapter().command(req) == [
        "aider",
        "--model", f"openai/{MODEL}",
        "--openai-api-base", ENDPOINT,
        "--yes-always", "--no-git", "--no-auto-commits", "--no-show-model-warnings",
        "--no-check-update", "--no-analytics", "--no-pretty",
        "--message", "solve isogram",
        "two_fer.py",
    ]
    assert AiderAdapter().env(req) == {"OPENAI_API_KEY": "ainode"}


def test_aider_reads_its_own_token_lines():
    one = parse_tokens("Applied edit to isogram.py\nTokens: 709 sent, 87 received.")
    assert one == {"tokens_sent": 709, "tokens_received": 87, "turns": 1}
    # k suffixes and one line per exchange: both summed, the count is the turns.
    many = parse_tokens("Tokens: 12k sent, 1.3k received. Cost: $0.00\n"
                        "Tokens: 500 sent, 40 received.")
    assert many == {"tokens_sent": 12500, "tokens_received": 1340, "turns": 2}
    assert parse_tokens("nothing to see") == {}


def test_pi_command_is_pinned(tmp_path):
    req = _request(tmp_path, prompt="solve bob")
    assert PiAdapter().command(req) == [
        "pi", "--provider", PROVIDER, "--model", MODEL,
        "--tools", "read,grep,find,ls,edit,write,bash",
        "--no-session", "--no-context-files", "-p", "solve bob",
    ]


def test_pi_provider_merges_and_never_clobbers(tmp_path, monkeypatch):
    monkeypatch.setenv("AINODE_HARNESS_PI_HOME", str(tmp_path))
    existing = json.dumps({"providers": {"someones-own": {"baseUrl": "http://kept"}},
                           "defaults": {"model": "kept"}})
    merged = json.loads(merge_models_json(existing, _request(tmp_path)))
    assert merged["providers"]["someones-own"] == {"baseUrl": "http://kept"}
    assert merged["defaults"] == {"model": "kept"}
    entry = merged["providers"][PROVIDER]
    assert entry["baseUrl"] == ENDPOINT
    assert entry["api"] == "openai-completions"
    assert entry["apiKey"] == "ainode"
    assert entry["models"][0]["id"] == MODEL

    # And the adapter writes exactly that file, under the redirected home.
    written = PiAdapter().write_config(_request(tmp_path))
    assert written == [tmp_path / ".pi" / "agent" / "models.json"]
    assert PROVIDER in json.loads(written[0].read_text())["providers"]


def test_pi_only_moves_home_when_the_config_was_redirected(tmp_path, monkeypatch):
    monkeypatch.delenv("AINODE_HARNESS_PI_HOME", raising=False)
    assert PiAdapter().env(_request(tmp_path)) == {}
    monkeypatch.setenv("AINODE_HARNESS_PI_HOME", str(tmp_path))
    assert PiAdapter().env(_request(tmp_path)) == {"HOME": str(tmp_path)}


def test_opencode_command_and_project_config(tmp_path):
    req = _request(tmp_path)
    # --auto because there is no TTY to approve the write, and --format json because
    # without it two verified runs produced no output at all until the timeout.
    assert OpencodeAdapter().command(req) == [
        "opencode", "run", "--pure", "--auto", "--format", "json",
        "-m", f"{PROVIDER}/{MODEL}", req.prompt,
    ]
    assert OpencodeAdapter().needs_git is True
    provider = project_config(req)["provider"][PROVIDER]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"] == {"baseURL": ENDPOINT, "apiKey": "ainode"}
    # The verified model entry carries a name and nothing else.
    assert provider["models"] == {MODEL: {"name": MODEL}}
    cfg = OpencodeAdapter().config(req)[0]
    assert cfg.path == req.workdir / "opencode.json"
    assert cfg.merged is False


def test_opencode_counts_turns_off_its_event_stream():
    stream = ('{"type":"step_start"}\n'
              'not json at all\n'
              '{"type":"tool","part":{"tool":"edit","state":{"status":"completed"}}}\n'
              '{"type":"step_start"}\n'
              '{"type":"text","part":{"text":"done"}}\n'
              '{"type":"step_finish"}\n')
    assert parse_events(stream) == {"turns": 2}
    # A stream we cannot read costs the turns field and nothing else.
    assert parse_events("plain human output\n") == {}
    assert parse_events("") == {}


def test_claude_command_is_pinned(tmp_path):
    req = _request(tmp_path, prompt="solve isogram")
    assert ClaudeAdapter().command(req) == [
        "claude",
        "-p", "solve isogram",
        "--model", MODEL,
        # No TTY to approve a write or trust the directory: without this the run
        # sits until the timeout instead of failing.
        "--dangerously-skip-permissions",
        "--output-format", "json",
        "--max-turns", "12",
    ]
    assert ClaudeAdapter().needs_git is True


def test_claude_effort_is_appended_only_when_the_run_asked_for_one(tmp_path):
    """#127: Claude Code sends effort "high" by default and Qwen3.8-Flash-Next's
    template rejects it with a 400, so the level has to be settable - and unset has
    to keep sending nothing, or every number already recorded moves."""
    plain = _request(tmp_path)
    assert plain.claude_effort is None
    assert "--effort" not in ClaudeAdapter().command(plain)

    asked = _request(tmp_path, claude_effort="medium")
    command = ClaudeAdapter().command(asked)
    assert command[-2:] == ["--effort", "medium"]
    # Nothing else about the invocation moves.
    assert command[:-2] == ClaudeAdapter().command(plain)
    # And it is the argv, not the environment: the env stays what it was.
    assert ClaudeAdapter().env(asked) == ClaudeAdapter().env(plain)


def test_the_effort_level_is_recorded_on_the_harness_that_got_it(tmp_path):
    """It reaches one argv, so it is recorded on one harness's block."""
    from ainode.bench.harness.runner import HarnessResult, harness_options

    assert harness_options(ClaudeAdapter(), "medium") == {"effort": "medium"}
    # Not on a harness that never saw it, and not at all when unset.
    assert harness_options(AiderAdapter(), "medium") == {}
    assert harness_options(ClaudeAdapter(), None) == {}
    assert "options" not in HarnessResult("claude", "2.1.272").as_json()
    assert HarnessResult("claude", "2.1.272",
                         options={"effort": "medium"}).as_json()["options"] == \
        {"effort": "medium"}


def test_claude_is_given_the_endpoint_without_its_v1(tmp_path):
    """Claude Code appends /v1/messages itself, so the /v1 has to come off."""
    assert messages_base("http://node:3000/v1") == "http://node:3000"
    assert messages_base("http://node:3000/v1/") == "http://node:3000"
    # Anything else is passed through: an endpoint already in base form, and a
    # path that merely ends in something v1-ish, are both left alone.
    assert messages_base("http://node:8000") == "http://node:8000"
    assert messages_base("http://node/openai/v1") == "http://node/openai"
    assert messages_base("") == ""


def test_claude_env_isolates_the_operators_own_profile(tmp_path):
    req = _request(tmp_path)
    env = ClaudeAdapter().env(req)
    assert env["ANTHROPIC_BASE_URL"] == "http://fake-node.invalid:3000"
    # The placeholder in both, because which one it reads depends on how the
    # client got built, and the endpoint authenticates nothing either way.
    assert env["ANTHROPIC_API_KEY"] == "ainode"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "ainode"
    assert env["ANTHROPIC_MODEL"] == MODEL
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == MODEL
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert env["DISABLE_TELEMETRY"] == "1"
    # The run's own profile, under the scratch dir: never ~/.claude, so a bench
    # run cannot read or write the operator's settings, hooks, sessions or keys.
    assert env["CLAUDE_CONFIG_DIR"] == str(req.scratch / "claude-config")
    assert str(pathlib.Path.home()) not in env["CLAUDE_CONFIG_DIR"]
    # And it needs no seed file: verified on 2.1.272, claude creates the whole
    # tree itself from a directory that does not exist.
    assert ClaudeAdapter().config(req) == []


def test_claude_reads_turns_out_of_its_result_json():
    payload = {"type": "result", "subtype": "success", "is_error": False,
               "num_turns": 6, "duration_ms": 294118, "total_cost_usd": 0.42,
               "result": "isogram.py now returns False for repeated letters",
               "session_id": "831de781-4bb9-4ff3-869c-c621c84d7fd5",
               "usage": {"input_tokens": 9001, "output_tokens": 311}}
    assert parse_result(json.dumps(payload)) == {"turns": 6}


def test_claude_is_error_is_a_crash_even_on_a_clean_exit():
    """Claude Code reports a run it could not finish in the payload and still
    exits 0, so the exit code alone would record that as a good run."""
    payload = {"type": "result", "subtype": "error_max_turns", "is_error": True,
               "num_turns": 12, "result": "", "duration_ms": 900000}
    assert parse_result(json.dumps(payload)) == {"turns": 12, "crashed": True}
    # Output we cannot read costs the fields and nothing else.
    assert parse_result("[claude-code:unrecognized_model] {...}") == {}
    assert parse_result("{not json") == {}
    assert parse_result("") == {}


def test_dsh_command_and_overlay(tmp_path, monkeypatch):
    monkeypatch.setenv("AINODE_HARNESS_DSH_HOME", str(tmp_path / "dsh-home"))
    req = _request(tmp_path)
    adapter = DshAdapter()
    patch = tmp_path / "scratch" / "dsh-harness.patch.yml"
    assert adapter.command(req) == [
        "dsh", "--profile", "headless", "--patch", str(patch), req.prompt,
    ]
    # The overlay both defines the provider and selects it; verified against
    # `dsh --profile headless --patch <file> --dump-config` on dsh 0.1.5-rc.1.
    overlay = patch_overlay(req)
    assert "- id: llm-pi-ai" in overlay
    assert "- id: agent-default-model" in overlay
    assert f"      {PROVIDER}:" in overlay
    # baseURL, not baseUrl: the key the llm-pi-ai plugin documents.
    assert f"baseURL: {json.dumps(ENDPOINT)}" in overlay
    assert f"apiKeyEnv: {API_KEY_ENV}" in overlay
    assert "api: openai-completions" in overlay
    # A model id has slashes and dots in it; it has to survive as one scalar.
    assert f'model: "{MODEL}"' in overlay
    assert [c.path for c in adapter.config(req)] == \
           [tmp_path / "dsh-home" / "settings.yaml", patch]


def test_dsh_declares_deepseek_thinking_only_for_deepseek(tmp_path):
    assert thinking_format("fraserprice/DeepSeek-V4-Flash-DSpark") == "deepseek"
    assert thinking_format("unsloth/Qwen3.8-27B-NVFP4") is None
    plain = patch_overlay(_request(tmp_path))
    assert "thinkingFormat" not in plain

    deep = HarnessRequest(workdir=tmp_path, scratch=tmp_path, prompt="p", entry="x.py",
                          endpoint=ENDPOINT, model="fraserprice/DeepSeek-V4-Flash")
    assert "thinkingFormat: deepseek" in patch_overlay(deep)
    assert "thinkingFormat: deepseek" in settings_yaml(deep)


def test_dsh_runs_in_its_own_home_and_never_the_real_one(tmp_path, monkeypatch):
    """One stale route in somebody's ~/.dsh/settings.yaml fails every dsh run,
    whichever provider the run selected, so the bench brings its own home."""
    home = tmp_path / "dsh-home"
    monkeypatch.setenv("AINODE_HARNESS_DSH_HOME", str(home))
    monkeypatch.setenv("DSH_HOME", "/somebody/elses/.dsh")
    adapter = DshAdapter()
    req = _request(tmp_path)

    assert adapter.env(req)["DSH_HOME"] == str(home)
    settings = [c for c in adapter.config(req) if c.path.name == "settings.yaml"][0]
    assert settings.path == home / "settings.yaml"
    # Exactly one route, so the boot-time check has one thing to validate.
    assert settings.content.count("baseURL:") == 1
    assert f"    {PROVIDER}:" in settings.content
    assert f"  provider: {PROVIDER}" in settings.content

    # A home somebody curated is left completely alone; the overlay still carries
    # the provider, so the run works without rewriting their file.
    adapter.write_config(req)
    home.joinpath("settings.yaml").write_text("llm-pi-ai:\n  providers:\n    theirs:\n"
                                              "      apiKeyEnv: THEIR_KEY\n")
    assert [c.path.name for c in adapter.config(req)] == ["dsh-harness.patch.yml"]
    assert PROVIDER in patch_overlay(req)


def test_dsh_sets_every_api_key_env_the_settings_file_names(tmp_path, monkeypatch):
    text = ("llm-pi-ai:\n  providers:\n    spark4:\n      apiKeyEnv: SPARK4_API_KEY\n"
            "      api: openai-completions\n    other:\n      apiKeyEnv: 'OTHER_KEY'\n"
            "      apiKeyEnv: SPARK4_API_KEY\n")
    assert api_key_envs(text) == ["SPARK4_API_KEY", "OTHER_KEY"]

    home = tmp_path / "curated"
    home.mkdir()
    (home / "settings.yaml").write_text(text)
    monkeypatch.setenv("AINODE_HARNESS_DSH_HOME", str(home))
    monkeypatch.delenv("SPARK4_API_KEY", raising=False)
    monkeypatch.setenv("OTHER_KEY", "a-real-key-already-set")
    env = DshAdapter().env(_request(tmp_path))
    # dsh validates every provider route at boot, so an unset one has to be filled.
    assert env[API_KEY_ENV] == "ainode"
    assert env["SPARK4_API_KEY"] == "ainode"
    # ... and a key somebody actually set is left alone rather than overwritten.
    assert "OTHER_KEY" not in env


def test_every_shipped_adapter_is_registered_and_needs_no_real_key(tmp_path, monkeypatch):
    monkeypatch.setenv("AINODE_HARNESS_PI_HOME", str(tmp_path / "pi-home"))
    monkeypatch.setenv("AINODE_HARNESS_DSH_HOME", str(tmp_path / "dsh-home"))
    registry = adapters_mod.registry()
    assert registry.names() == ["aider", "claude", "dsh", "opencode", "pi"]
    for name in registry.names():
        adapter = registry.get(name)
        req = _request(tmp_path)
        argv = adapter.command(req)
        assert argv[0] == adapter.binary
        blob = " ".join(argv) + json.dumps(adapter.env(req)) + \
            "".join(c.content for c in adapter.config(req))
        # Every harness is told where the endpoint is, somewhere, and the model.
        # claude is the one exception to the exact string: it speaks the Messages
        # API and appends /v1/messages itself, so it gets the base without the /v1.
        assert (ENDPOINT[:-3] if name == "claude" else ENDPOINT) in blob, name
        assert MODEL in blob, name
        # And none of them is handed anything that looks like a real key.
        assert "sk-" not in blob, name
        assert "ainode" in blob, name
    with pytest.raises(KeyError, match="unknown harness"):
        registry.get("claude-code")


# ---------------------------------------------------------------- isolation

def test_the_harness_never_sees_the_tests_but_pytest_does(two_fer, tmp_path):
    result = run_task(two_fer, ScriptAdapter("solve"), ENDPOINT, MODEL, tmp_path,
                      timeout=60, log=lambda *_: None)
    assert result.passed_at == 1

    workdir = tmp_path / "fake" / "two-fer"
    seen = [json.loads(line) for line in (workdir / "seen.jsonl").read_text().splitlines()]
    assert len(seen) == 1
    # What the agent could list while it was running.
    assert "two_fer_test.py" not in seen[0]["files"]
    assert set(seen[0]["files"]) >= {"instructions.md", "two_fer.py"}
    # The tests did run - and were taken back out afterwards.
    assert result.attempts[0].tests.passed_count == 3
    assert not (workdir / "two_fer_test.py").exists()


def test_a_test_file_left_in_the_working_dir_stops_the_run(two_fer, tmp_path):
    """The invariant is asserted in the loop, not assumed."""
    from ainode.bench.harness.runner import assert_tests_hidden, prepare_workdir

    workdir = prepare_workdir(two_fer, tmp_path / "w")
    (workdir / "two_fer_test.py").write_text("def test_x(): pass\n")
    with pytest.raises(HarnessBenchError, match="must never see them"):
        assert_tests_hidden(two_fer, workdir)


def test_attempt_two_is_told_what_failed(two_fer, tmp_path):
    result = run_task(two_fer, ScriptAdapter("second-try"), ENDPOINT, MODEL, tmp_path,
                      timeout=60, log=lambda *_: None)
    assert result.passed_at == 2
    assert [a.tests.passed for a in result.attempts] == [False, True]

    seen = [json.loads(line) for line in
            (tmp_path / "fake" / "two-fer" / "seen.jsonl").read_text().splitlines()]
    assert len(seen) == 2
    assert "did not pass the hidden tests" not in seen[0]["prompt"]
    assert "did not pass the hidden tests" in seen[1]["prompt"]
    # pytest's real output, not a summary of it: attempt 1 left the stub alone, so
    # the failure the model is shown is the TypeError the stub raises.
    assert "two_fer.py" in seen[1]["prompt"]
    assert "TypeError" in seen[1]["prompt"]
    assert "3 failed" in seen[1]["prompt"]
    # Attempt 2 also started clean of the tests.
    assert "two_fer_test.py" not in seen[1]["files"]


def test_a_task_that_is_never_solved_records_both_attempts(two_fer, tmp_path):
    result = run_task(two_fer, ScriptAdapter("wrong"), ENDPOINT, MODEL, tmp_path,
                      timeout=60, log=lambda *_: None)
    assert result.passed_at is None
    assert len(result.attempts) == 2
    assert result.attempts[0].tests.failed_count == 3
    assert result.crashed is False


# ---------------------------------------------------------------- failure paths

def test_a_timeout_is_recorded_and_the_tests_still_run(two_fer, tmp_path):
    result = run_task(two_fer, ScriptAdapter("hang"), ENDPOINT, MODEL, tmp_path,
                      timeout=1.0, attempts=1, log=lambda *_: None)
    run = result.attempts[0].harness
    assert run.timed_out is True
    assert run.crashed is False
    assert run.exit_code is None
    assert result.timed_out is True
    # The stub is still there, so pytest ran and failed rather than being skipped.
    assert result.attempts[0].tests.exit_code not in (None, 0)


def test_a_nonzero_exit_is_a_crash_not_an_exception(two_fer, tmp_path):
    result = run_task(two_fer, ScriptAdapter("boom"), ENDPOINT, MODEL, tmp_path,
                      timeout=60, attempts=1, log=lambda *_: None)
    run = result.attempts[0].harness
    assert (run.exit_code, run.crashed, run.timed_out) == (3, True, False)
    assert result.crashed is True


def test_a_missing_binary_is_a_crash_with_the_reason(two_fer, tmp_path):
    adapter = MissingAdapter()
    assert adapter.available() is False
    result = run_task(two_fer, adapter, ENDPOINT, MODEL, tmp_path, timeout=60,
                      attempts=1, log=lambda *_: None)
    run = result.attempts[0].harness
    assert run.crashed is True
    assert run.exit_code is None
    assert "FileNotFoundError" in (run.error or "")


# ---------------------------------------------------------------- scoring

def _fake_results(passed_at, crashed=(), timed_out=()):
    from ainode.bench.harness.adapters import HarnessRun
    from ainode.bench.harness.runner import Attempt, TestResult

    out = []
    for index, at in enumerate(passed_at):
        result = TaskResult(slug=f"task-{index}")
        for number in (1, 2):
            run = HarnessRun("fake", ["x"], 0, 10.0, index in timed_out,
                             index in crashed)
            tests = TestResult(exit_code=0 if at == number else 1, wall_s=0.1)
            result.attempts.append(Attempt(number, run, tests))
            if at == number:
                break
        out.append(result)
    return out


def test_scores_are_cumulative_and_wall_clock_counts_every_attempt():
    results = _fake_results([1, 2, None, None], crashed={3}, timed_out={2})
    assert score(results) == {
        "tasks": 4, "passed_at_1": 1, "passed_at_2": 2,
        "pass_at_1": 0.25, "pass_at_2": 0.5,
        # task 0 solved first try (10s), the other three took two attempts (20s).
        "mean_wall_s": 17.5, "crashes": 1, "timeouts": 1,
    }
    assert score([]) == {"tasks": 0}


def test_pytest_counts_are_read_but_the_exit_code_decides():
    assert parse_pytest_counts("3 passed in 0.01s") == {"passed_count": 3}
    assert parse_pytest_counts("1 failed, 5 passed in 0.05s") == \
        {"passed_count": 5, "failed_count": 1}
    assert parse_pytest_counts("2 errors in 0.1s") == {"error_count": 2}
    assert parse_pytest_counts("no tests ran in 0.01s") == {}

    from ainode.bench.harness.runner import TestResult
    # Counts present, exit code nonzero: not a pass. The bit comes from pytest.
    assert TestResult(exit_code=1, wall_s=0.1, passed_count=3).passed is False
    assert TestResult(exit_code=0, wall_s=0.1).passed is True


def test_the_test_command_runs_under_this_interpreter(two_fer):
    assert resolve_test_command(two_fer)[0] == sys.executable
    assert resolve_test_command(two_fer)[1:] == ["-m", "pytest", "-q", "two_fer_test.py"]


# ---------------------------------------------------------------- token window

def test_token_counts_are_a_delta_over_the_window(two_fer, tmp_path):
    counters = iter([{"requests": {"total": 10, "tokens_generated": 1000}},
                     {"requests": {"total": 13, "tokens_generated": 1600}}])
    result = run_task(two_fer, ScriptAdapter("solve"), ENDPOINT, MODEL, tmp_path,
                      timeout=60, tokens_reader=lambda: next(counters),
                      log=lambda *_: None)
    assert result.tokens == {"requests": 3, "tokens_generated": 600}


def test_an_unreachable_metrics_endpoint_costs_one_optional_field(two_fer, tmp_path):
    def boom():
        raise OSError("connection refused")

    result = run_task(two_fer, ScriptAdapter("solve"), ENDPOINT, MODEL, tmp_path,
                      timeout=60, tokens_reader=boom, log=lambda *_: None)
    assert result.passed_at == 1
    assert result.tokens is None
    assert "tokens" not in result.as_json()


# ---------------------------------------------------------------- the record

def test_the_record_has_a_harness_block_and_no_invented_throughput(two_fer, tmp_path,
                                                                   tasks_dir):
    results = run_suite([two_fer], [ScriptAdapter("solve")], ENDPOINT, MODEL,
                        root=tmp_path, timeout=60, log=lambda *_: None)
    block = build_harness_block(results, [two_fer], ENDPOINT, 2, 900, tasks_dir)
    record = build_record("unit", {"id": MODEL}, {"node": "Fake-Spark"}, block,
                          {"attempts": 2}, build_notes(results, True, 12),
                          "20260916-120000")

    assert record["schema"] == 1
    assert record["stamp"] == "20260916-120000"
    assert set(record) == {"schema", "stamp", "label", "model", "placement",
                           "settings", "harness", "notes", "source"}
    # No results block at all: this run measured no throughput and a zero would
    # be a number nobody took.
    assert "results" not in record

    harness = record["harness"]
    assert harness["task_set"]["id"] == "exercism-python-10"
    assert harness["task_set"]["count"] == 1
    assert harness["task_set"]["source"]["license"] == "MIT"
    assert harness["endpoint"] == ENDPOINT
    assert harness["protocol"] == {"attempts": 2, "timeout_s": 900,
                                   "second_attempt_sees": "the failing test output"}
    run = harness["runs"][0]
    assert run["harness"] == "fake"
    assert run["scores"]["pass_at_1"] == 1.0
    assert run["tasks"][0]["slug"] == "two-fer"
    assert run["tasks"][0]["attempts"][0]["tests"]["tests_passed"] == 3
    # The record is JSON, all the way down.
    assert json.loads(json.dumps(record))

    notes = " ".join(record["notes"])
    assert "only after the harness has exited" in notes
    assert "/api/metrics" in notes


def test_the_recorded_command_keeps_the_flags_and_elides_the_prompt():
    from ainode.bench.harness.adapters import recorded_command

    recorded = recorded_command(["aider", "--model", "openai/x", "--message",
                                 "instructions " * 200, "x.py"])
    assert "--model openai/x" in recorded
    assert "instructions instructions" not in recorded
    assert "<2600 chars>" in recorded
    assert recorded.endswith("x.py")


def test_the_ainode_base_is_the_endpoint_without_its_v1():
    assert ainode_base("http://node:3000/v1") == "http://node:3000"
    assert ainode_base("http://node:3000/v1/") == "http://node:3000"
    assert ainode_base("http://node:8000") == "http://node:8000"
    assert ainode_base("http://node:3000/v1", "http://other:3000/") == "http://other:3000"


# ---------------------------------------------------------------- the CLI

def test_dry_run_prints_the_commands_and_writes_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("AINODE_HARNESS_PI_HOME", str(tmp_path / "pi-home"))
    monkeypatch.setenv("AINODE_HARNESS_DSH_HOME", str(tmp_path / "dsh-home"))
    out_dir = tmp_path / "results"
    code = harness_main(["--endpoint", ENDPOINT, "--model", MODEL, "--label", "unit",
                         "--harness", "aider,dsh", "--only-tasks", "two-fer",
                         "--work-dir", str(tmp_path / "work"), "--dry-run"],
                        out_dir=out_dir)
    printed = capsys.readouterr().out
    assert code == 0
    assert "aider --model openai/" + MODEL in printed
    assert "--openai-api-base " + ENDPOINT in printed
    assert "dsh --profile headless --patch" in printed
    assert "two_fer_test.py (after the harness exits)" in printed
    assert "nothing was executed and no file was written" in printed
    # Not one byte on disk: no results file, no working copies, no config.
    assert not out_dir.exists()
    assert not (tmp_path / "work").exists()
    assert not (tmp_path / "pi-home").exists()
    assert not (tmp_path / "dsh-home").exists()


def test_dry_run_shows_claude_pointed_at_the_base_and_its_own_config_dir(tmp_path,
                                                                        capsys):
    work = tmp_path / "work"
    code = harness_main(["--endpoint", ENDPOINT, "--model", MODEL, "--label", "unit",
                         "--harness", "claude", "--only-tasks", "two-fer",
                         "--work-dir", str(work), "--dry-run"],
                        out_dir=tmp_path / "results")
    printed = capsys.readouterr().out
    assert code == 0
    assert "claude -p" in printed
    assert f"--model {MODEL} --dangerously-skip-permissions" in printed
    assert "--output-format json --max-turns 12" in printed
    # The base without the /v1, and a config dir under this run's scratch.
    assert f"'ANTHROPIC_BASE_URL': '{ENDPOINT[:-3]}'" in printed
    assert f"'CLAUDE_CONFIG_DIR': '{work / 'claude' / 'two-fer.scratch' / 'claude-config'}'" \
        in printed
    assert f"git init : {work / 'claude' / 'two-fer'}" in printed
    assert not work.exists()


def test_the_cli_parses_the_effort_level_and_dry_run_shows_it(tmp_path, capsys):
    from ainode.bench.harness.cli import build_parser, effort

    # Default is unset, which means "pass nothing", not "pass a default".
    assert build_parser().parse_args([]).claude_effort is None
    assert effort(build_parser().parse_args([])) is None
    assert effort(build_parser().parse_args(["--claude-effort", "xhigh"])) == "xhigh"
    # A shell variable that expanded to nothing is not an empty --effort argument.
    assert effort(build_parser().parse_args(["--claude-effort", "  "])) is None

    code = harness_main(["--endpoint", ENDPOINT, "--model", MODEL, "--label", "unit",
                         "--harness", "claude", "--only-tasks", "two-fer",
                         "--claude-effort", "medium",
                         "--work-dir", str(tmp_path / "work"), "--dry-run"],
                        out_dir=tmp_path / "results")
    printed = capsys.readouterr().out
    assert code == 0
    assert "effort  : claude --effort medium" in printed
    assert "--max-turns 12 --effort medium" in printed


def test_the_effort_level_is_written_into_the_records_settings(tmp_path, monkeypatch):
    """The flag is only honest if the record says it was used."""
    import shutil

    from ainode.bench import fleet as fleet_mod
    from ainode.bench.harness import cli as cli_mod

    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/claude")
    monkeypatch.setattr(fleet_mod, "describe_via_http",
                        lambda *_a, **_k: ({"id": MODEL}, {"node": "Fake-Spark"},
                                           "fake-node", []))
    seen = {}

    def fake_suite(tasks, adapters, endpoint, model, **kwargs):
        seen.update(kwargs)
        return [_harness_result(a, kwargs.get("claude_effort")) for a in adapters]

    monkeypatch.setattr(cli_mod, "run_suite", fake_suite)

    def record_for(extra):
        out_dir = tmp_path / f"results{len(list(tmp_path.iterdir()))}"
        assert harness_main(["--endpoint", ENDPOINT, "--model", MODEL,
                            "--label", "unit", "--harness", "claude",
                            "--only-tasks", "two-fer", "--no-metrics", *extra],
                           out_dir=out_dir) == 0
        written = list(out_dir.glob("*-harness.json"))
        assert len(written) == 1
        return json.loads(written[0].read_text())

    record = record_for(["--claude-effort", "medium"])
    assert seen["claude_effort"] == "medium"
    assert record["settings"]["claude_effort"] == "medium"
    # And on the block of the harness that was actually given it.
    assert record["harness"]["runs"][0]["options"] == {"effort": "medium"}

    record = record_for([])
    assert seen["claude_effort"] is None
    # Absent, not null: a run that sent no --effort says nothing about the level.
    assert "claude_effort" not in record["settings"]
    assert "options" not in record["harness"]["runs"][0]


def _harness_result(adapter, claude_effort):
    from ainode.bench.harness.runner import HarnessResult, harness_options

    return HarnessResult(harness=adapter.name, version="2.1.272",
                         tasks=[TaskResult(slug="two-fer")],
                         options=harness_options(adapter, claude_effort))


def test_the_env_line_prints_paths_and_masks_a_key_it_did_not_set(tmp_path, capsys,
                                                                 monkeypatch):
    """Paths are the useful half of that line, so they print.

    Nothing the adapters build currently trips the mask: they only ever put the
    placeholder in a key variable, and skip one that already holds a real value. The
    mask is there so that stays true by construction rather than by memory, which is
    why it is tested directly as well as through the dry run.
    """
    from ainode.bench.harness.cli import _mask

    assert _mask({"DSH_HOME": "/home/x/.dsh", "AINODE_BENCH_API_KEY": "ainode",
                  "THEIR_REAL_KEY": "sk-do-not-print-me"}, "ainode") == {
        "AINODE_BENCH_API_KEY": "ainode",
        "DSH_HOME": "/home/x/.dsh",
        "THEIR_REAL_KEY": "***",
    }

    home = tmp_path / "dsh-home"
    monkeypatch.setenv("AINODE_HARNESS_DSH_HOME", str(home))
    harness_main(["--endpoint", ENDPOINT, "--model", MODEL, "--label", "unit",
                  "--harness", "dsh", "--only-tasks", "two-fer", "--dry-run"],
                 out_dir=tmp_path / "results")
    printed = capsys.readouterr().out
    assert f"'DSH_HOME': '{home}'" in printed
    assert "'AINODE_BENCH_API_KEY': 'ainode'" in printed


def test_dry_run_never_prints_a_merged_config_it_did_not_write(tmp_path, capsys,
                                                              monkeypatch):
    """pi's config file is somebody's own; the bench merges into it and must not
    echo what is already in there."""
    home = tmp_path / "pi-home"
    (home / ".pi" / "agent").mkdir(parents=True)
    (home / ".pi" / "agent" / "models.json").write_text(json.dumps(
        {"providers": {"anthropic": {"apiKey": "sk-ant-do-not-print-me"}}}))
    monkeypatch.setenv("AINODE_HARNESS_PI_HOME", str(home))
    harness_main(["--endpoint", ENDPOINT, "--model", MODEL, "--label", "unit",
                  "--harness", "pi", "--only-tasks", "two-fer", "--dry-run"],
                 out_dir=tmp_path / "results")
    printed = capsys.readouterr().out
    assert "merged into the existing file" in printed
    assert "sk-ant-do-not-print-me" not in printed


def test_the_flat_bench_cli_still_owns_every_other_flag(tmp_path, capsys):
    """The subcommand is dispatched ahead of argparse, so nothing else moved."""
    from ainode.bench.cli import main as bench_main

    saved = tmp_path / "run.json"
    saved.write_text(json.dumps({"schema": 1, "label": "kept"}))
    assert bench_main(["--show", str(saved)]) == 0
    assert '"label": "kept"' in capsys.readouterr().out


def test_the_harness_subcommand_is_reachable_from_the_shim(tmp_path, capsys,
                                                           monkeypatch):
    from ainode.bench.cli import main as bench_main

    monkeypatch.setenv("AINODE_HARNESS_PI_HOME", str(tmp_path / "pi-home"))
    code = bench_main(["harness", "--endpoint", ENDPOINT, "--model", MODEL,
                       "--label", "unit", "--only-tasks", "two-fer", "--dry-run"],
                      out_dir=tmp_path / "results")
    assert code == 0
    assert "ainode-bench harness" in capsys.readouterr().out


def test_an_empty_task_set_is_refused(tmp_path, capsys):
    with pytest.raises(SystemExit):
        harness_main(["--endpoint", ENDPOINT, "--model", MODEL, "--label", "unit",
                      "--harness", "aider", "--tasks-dir", str(tmp_path)],
                     out_dir=tmp_path / "r")
    assert "no tasks in" in capsys.readouterr().err


def test_a_missing_binary_is_refused_before_a_single_request(tmp_path, capsys,
                                                            monkeypatch):
    """Nothing is asked of the node until every named harness is installed."""
    import shutil

    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: None)
    with pytest.raises(SystemExit):
        harness_main(["--endpoint", ENDPOINT, "--model", MODEL, "--label", "unit",
                      "--harness", "aider,dsh", "--only-tasks", "two-fer"],
                     out_dir=tmp_path / "r")
    err = capsys.readouterr().err
    assert "not on PATH: aider, dsh" in err
    assert not (tmp_path / "r").exists()


def test_list_harnesses_says_what_is_installed(capsys):
    assert harness_main(["--list-harnesses"]) == 0
    printed = capsys.readouterr().out
    for name in ("aider", "claude", "dsh", "opencode", "pi"):
        assert name in printed


# ---------------------------------------------------------------- the renderer

def _renderer():
    import importlib.util

    script = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "render-bench-table.py"
    spec = importlib.util.spec_from_file_location("render_bench_table", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_throughput_table_ignores_harness_records(tmp_path):
    module = _renderer()
    (tmp_path / "20260916-000000-fake-harness.json").write_text(json.dumps({
        "schema": 1, "stamp": "20260916-000000", "label": "harness",
        "model": {"id": MODEL, "name": "Fake Coder"},
        "placement": {"node": "Fake-Spark"},
        "harness": {"task_set": {"id": "exercism-python-10", "count": 10},
                    "runs": [{"harness": "aider", "scores": {"pass_at_1": 0.8}}]},
    }))
    (tmp_path / "20260915-000000-fake-speed.json").write_text(json.dumps({
        "schema": 1, "stamp": "20260915-000000", "label": "speed",
        "model": {"id": MODEL, "name": "Fake Coder", "params_b": 30, "active_b": 3,
                  "quant": "NVFP4"},
        "placement": {"node": "Fake-Spark", "gpu": "NVIDIA GB10", "gpus": 1, "tp": 1},
        "results": {"single_stream": {"decode_tok_s": 40.0}},
    }))
    runs = module.load_runs(tmp_path)
    assert len(runs) == 2
    assert [module.is_harness_run(r) for r in runs] == [True, False]

    table = module.render_table(runs)
    assert "fake-speed.json" in table
    assert "fake-harness.json" not in table
    assert len(table.splitlines()) == 3  # header, rule, the one throughput run


def test_a_harness_record_with_throughput_in_it_still_gets_a_row(tmp_path):
    """The filter is "no single_stream", not "has a harness block": a run that
    measured both belongs in the table."""
    module = _renderer()
    (tmp_path / "20260916-000000-both.json").write_text(json.dumps({
        "schema": 1, "stamp": "20260916-000000", "label": "both",
        "model": {"id": MODEL, "name": "Fake Coder"}, "placement": {},
        "harness": {"runs": []},
        "results": {"single_stream": {"decode_tok_s": 12.0}},
    }))
    runs = module.load_runs(tmp_path)
    assert module.is_harness_run(runs[0]) is False
    assert "both.json" in module.render_table(runs)


# ------------------------------------------------------------ process group

def test_timeout_kills_the_whole_process_group(tmp_path):
    """The agent forks a child that outlives it; after the timeout neither may
    remain (OpenCode left a server behind and the next run hung)."""
    import os as _os
    import signal as _signal
    import subprocess as _sp
    import time as _time
    from ainode.bench.harness.adapters import _launch
    pidfile = tmp_path / "child.pid"
    script = (
        "import os, sys, time, subprocess\n"
        f"c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(c.pid))\n"
        "time.sleep(60)\n"
    )
    with pytest.raises(_sp.TimeoutExpired):
        _launch([sys.executable, "-c", script], cwd=str(tmp_path), env=dict(_os.environ), timeout=1.5)
    child = int(pidfile.read_text())
    for _ in range(20):
        try:
            _os.kill(child, 0)
        except ProcessLookupError:
            break
        _time.sleep(0.1)
    else:
        _os.kill(child, _signal.SIGKILL)
        pytest.fail("the grandchild survived the timeout")


def test_normal_completion_returns_output_and_reaps_the_group(tmp_path):
    import os as _os
    import sys as _sys
    from ainode.bench.harness.adapters import _launch
    proc = _launch([_sys.executable, "-c", "print('hi'); import sys; sys.stderr.write('err')"],
                   cwd=str(tmp_path), env=dict(_os.environ), timeout=10)
    assert proc.returncode == 0 and proc.stdout.strip() == "hi" and proc.stderr == "err"


def test_opencode_isolates_its_state_per_run(tmp_path):
    from ainode.bench.harness.adapters.opencode import OpencodeAdapter
    from ainode.bench.harness.adapters import HarnessRequest
    req = HarnessRequest(workdir=tmp_path / "w", scratch=tmp_path / "s", prompt="p", entry="x.py",
                         endpoint="http://e/v1", model="m")
    env = OpencodeAdapter().env(req)
    assert set(env) == {"XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"}
    assert all(v.startswith(str(tmp_path / "s")) for v in env.values())
