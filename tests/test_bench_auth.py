"""The key a bench run presents, and the refusal that stops it instead of scoring it.

Every section of ``scripts/ainode-bench.py`` runs out of process, so a node with
``auth.enabled`` refuses all of it. Since #244 that is what a fresh install is, and
the fleet is next, so #245 gave the bench one ``--api-key`` flag, one
``$AINODE_API_KEY`` fallback and one helper that spells the header.

Five things are pinned here, in this order:

1. **The resolution order**, once for all six sections: ``--api-key``, then
   ``$AINODE_API_KEY``, then the placeholder an open node accepts. A run reports the
   SOURCE and never the key.
2. **The header goes on every request the bench makes.** Behaviourally for each of
   the transports, and then by WALKING THE SOURCE: the string ``Authorization``
   appears in exactly one file of the package, and every place that builds a urllib
   request is checked for the helper, because the next call somebody adds is the one
   that would have quietly 401'd.
3. **The key reaches nothing that a reader can see**: not a record, not a note, not
   a printed line, not the harness dry run's environment overlay.
4. **A 401 stops the section before anything is scored**, in all six, and says what
   to pass. This is the ``HiddenTestsUnavailable`` lesson (#153) applied to the
   transport: a refusal recorded as ten model failures is a lie about the model.
5. **A 429 does the same and names the limit**, fed the rate limiter's own body so
   the two cannot drift; and a 400 or a 503 is still one failed row, because those
   are findings about the request that made them.

No node, no network: ``urlopen`` is faked where a request is built, and the
conftest's ``no_bench_preflight`` fixture keeps the opening GET inside the suite.
"""

import ast
import io
import json
import pathlib
import shutil
import sys
import urllib.error

import pytest

from ainode.bench import auth
# Bound at import time on purpose: conftest's autouse fixture replaces the module
# attribute with a stub, and the tests below still need the real function.
from ainode.bench.auth import preflight as real_preflight

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "ainode" / "bench"

MODEL = "org/Model-1"
ENDPOINT = "http://fake-node.invalid:3000/v1"
URL = "http://fake-node.invalid:8000"
KEY = "ak-live-do-not-print-me"


# ----------------------------------------------------------------- fakes ---

class FakeResponse:
    """Enough of an HTTP response for urllib's two use shapes: read, and iterate."""

    def __init__(self, payload=b"{}", lines=()):
        self._payload = payload
        self._lines = list(lines)

    def read(self, *args):
        return self._payload

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code, body=b"", url="http://fake-node.invalid:3000/v1/models"):
    """A real ``HTTPError``, with a readable body the way urllib hands one over."""
    return urllib.error.HTTPError(url, code, "refused", {}, io.BytesIO(body))


def capture(monkeypatch, module, response=None, raises=None):
    """Patch ``urlopen`` in one module and return the list of requests it saw."""
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append(req)
        if raises is not None:
            raise raises
        return response if response is not None else FakeResponse()

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    return seen


def refuse_preflight(monkeypatch, message=auth.NEEDS_KEY):
    """Make the opening GET report a refusal. Patches later than the conftest stub."""
    monkeypatch.setattr(auth, "preflight", lambda *a, **k: message)


# =============================================================================
# 1. The resolution order
# =============================================================================

def test_the_flag_wins_then_the_environment_then_nothing(monkeypatch):
    assert auth.resolve_key("from-flag", env={}) == ("from-flag", "--api-key")
    assert auth.resolve_key("", env={auth.ENV_API_KEY: "from-env"}) == (
        "from-env", "$AINODE_API_KEY")
    assert auth.resolve_key("", env={}) == ("", "")
    # The flag still wins when both are set: an operator overriding their own shell
    # for one run is the reason the flag exists.
    assert auth.resolve_key("from-flag", env={auth.ENV_API_KEY: "from-env"})[0] == (
        "from-flag")


def test_a_blank_flag_or_a_blank_variable_is_not_a_key():
    """A shell variable that expanded to nothing must not become an empty Bearer."""
    assert auth.resolve_key("   ", env={}) == ("", "")
    assert auth.resolve_key("", env={auth.ENV_API_KEY: "  "}) == ("", "")
    assert auth.bearer("") == {}
    assert auth.bearer("   ") == {}


def test_the_placeholder_is_the_last_resort_and_never_shadows_the_variable():
    """``ainode`` is what an open node accepts, so it stays behind both real sources.

    Before #245 four sections defaulted the flag TO the placeholder, which meant the
    environment variable could never be reached: the default was always truthy.
    """
    assert auth.key_for("", "ainode", env={}) == ("ainode", "the default")
    assert auth.key_for("", "ainode", env={auth.ENV_API_KEY: KEY}) == (
        KEY, "$AINODE_API_KEY")
    assert auth.key_for(KEY, "ainode", env={}) == (KEY, "--api-key")


def test_the_environment_variable_is_read_live_when_no_env_is_handed_in(monkeypatch):
    monkeypatch.setenv(auth.ENV_API_KEY, KEY)
    assert auth.resolve_key() == (KEY, "$AINODE_API_KEY")
    monkeypatch.delenv(auth.ENV_API_KEY)
    assert auth.resolve_key() == ("", "")


def test_the_source_names_where_and_never_what():
    """The only half of a credential a run may print."""
    sources = [auth.resolve_key(KEY, env={})[1],
               auth.resolve_key("", env={auth.ENV_API_KEY: KEY})[1],
               auth.key_for("", "ainode", env={})[1]]
    assert sources == ["--api-key", "$AINODE_API_KEY", "the default"]
    assert all(KEY not in source for source in sources)


# =============================================================================
# 2. The header, on every request, spelled once
# =============================================================================

def test_the_authorization_header_is_spelled_in_exactly_one_file():
    """The one-place rule, walked rather than remembered.

    Any other file that writes the header itself is a place a key can be forgotten,
    or sent as ``Bearer `` with nothing after it. ``auth.bearer`` is the only
    spelling, so a call site can only pass a key or pass nothing.
    """
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "/harness/tasks/" in path.as_posix():
            continue  # vendored upstream exercises, not our code
        if path.name == "auth.py":
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if "Authorization" in line:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}")
    assert not offenders, (
        "these lines spell the Authorization header themselves instead of calling "
        "ainode.bench.auth.bearer(): " + ", ".join(offenders))


#: Functions that build a urllib request WITHOUT naming the header helper, each with
#: the reason. All three forward the headers of a ``Request`` the section's own
#: ``request()`` built with ``auth.bearer``, so the key is already on it.
EXEMPT_REQUEST_BUILDERS = {
    ("ainode/bench/decide/backends.py", "post_json"):
        "forwards the headers of a Request built by a backend's request()",
    ("ainode/bench/embed/client.py", "post_json"):
        "forwards the headers of a Request built by request_for()",
    ("ainode/bench/speech/client.py", "post_multipart"):
        "forwards the headers of a Request built by request_for()",
}


def _enclosing_function(tree, lineno):
    """The innermost function whose body spans ``lineno``."""
    best = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(node, "end_lineno", node.lineno)
        if node.lineno <= lineno <= end and (best is None or node.lineno > best.lineno):
            best = node
    return best


def test_every_outgoing_request_in_the_package_carries_the_key():
    """Walk the source, not a list: the next transport is covered the day it lands.

    ``tests/test_fleet_auth.py`` does this for the peer calls and for the same
    reason. A bench request with no header is not a small omission: it is a 401 that
    lands in a record as a model that could not answer.
    """
    sites = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "/harness/tasks/" in path.as_posix():
            continue
        text = path.read_text()
        hits = [i + 1 for i, line in enumerate(text.splitlines())
                if "urllib.request.Request(" in line]
        if not hits:
            continue
        tree = ast.parse(text)
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno in hits:
            func = _enclosing_function(tree, lineno)
            assert func is not None, f"{rel}:{lineno} builds a request outside a function"
            sites.append((rel, func.name, lineno, ast.get_source_segment(text, func) or ""))

    assert len(sites) >= 8, f"the walk found only {len(sites)} request sites"

    missing = []
    for rel, name, lineno, body in sites:
        if (rel, name) in EXEMPT_REQUEST_BUILDERS:
            assert "bearer(" not in body, (
                f"{rel}::{name} is listed exempt but builds the header itself")
            continue
        if "bearer(" not in body:
            missing.append(f"{rel}:{lineno} ({name})")

    assert not missing, (
        "these bench requests do not carry the node's API key, so they answer 401 on "
        "a protected node: " + ", ".join(missing) + ". Merge auth.bearer(key) into "
        "the headers, or add the site to EXEMPT_REQUEST_BUILDERS with its reason.")


def test_the_walk_actually_sees_the_known_transports():
    """A guard on the guard: if the pattern stops matching, the walk passes empty."""
    expected = {
        "ainode/bench/measure.py",            # the control reads and the speed section
        "ainode/bench/auth.py",               # the preflight
        "ainode/bench/agentic/runner.py",     # the rubric's chat client
        "ainode/bench/decide/backends.py",    # the three decision backends
        "ainode/bench/embed/client.py",       # embeddings and its transport floor
        "ainode/bench/speech/client.py",      # transcriptions and its floor
    }
    found = {path.relative_to(REPO_ROOT).as_posix()
             for path in PACKAGE.rglob("*.py")
             if "urllib.request.Request(" in path.read_text()}
    assert expected <= found, f"the request pattern stopped matching {expected - found}"


# =============================================================================
# 3. Each transport, behaviourally
# =============================================================================

def test_the_control_plane_read_sends_the_key(monkeypatch):
    from ainode.bench import measure

    seen = capture(monkeypatch, measure)
    measure.get_json("http://node:3000/api/status", api_key=KEY)
    assert seen[0].get_header("Authorization") == f"Bearer {KEY}"

    seen = capture(monkeypatch, measure)
    measure.get_json("http://node:3000/api/status")
    assert seen[0].get_header("Authorization") is None


def test_the_speed_sections_chat_request_sends_the_key(monkeypatch):
    from ainode.bench import measure

    seen = capture(monkeypatch, measure, response=FakeResponse(lines=[b"data: [DONE]"]))
    measure.stream_chat(URL, MODEL, "hello", 8, api_key=KEY)
    assert seen[0].get_header("Authorization") == f"Bearer {KEY}"


def test_every_placement_read_passes_the_key_down(monkeypatch):
    """``describe_via_http`` and ``resolve_serving_node`` are five reads and two."""
    from ainode.bench import fleet as fleet_mod

    keys = []

    def fake_get_json(url, timeout=None, api_key=""):
        keys.append(api_key)
        if url.endswith("/api/server/status"):
            return {"loaded_models": [{"id": MODEL, "node_hostname": "Spark-2",
                                       "port": 8001, "parallel": 1}]}
        return {}

    monkeypatch.setattr(fleet_mod, "get_json", fake_get_json)
    fleet_mod.resolve_serving_node("http://node:3000", MODEL, api_key=KEY)
    fleet_mod.describe_via_http("http://node:3000", "http://node:3000", MODEL,
                                api_key=KEY)
    assert len(keys) >= 6 and set(keys) == {KEY}


def test_the_telemetry_and_token_readers_send_the_key(monkeypatch):
    from ainode.bench import measure
    from ainode.bench.harness import runner as harness_runner

    seen = {}

    def fake_get_json(url, timeout=None, api_key=""):
        seen[url] = api_key
        return {"nodes": [{"node_id": "n1", "gpu_memory_gb": 128}]}

    monkeypatch.setattr(measure, "get_json", fake_get_json)
    measure.http_nodes_reader("http://node:3000", "n1", api_key=KEY)()
    harness_runner.http_metrics_reader("http://node:3000", KEY)()
    assert set(seen.values()) == {KEY}
    assert "http://node:3000/api/metrics" in seen


def test_the_agentic_rubric_sends_the_key(monkeypatch):
    from ainode.bench.agentic import runner as agentic_runner

    seen = capture(monkeypatch, agentic_runner,
                   response=FakeResponse(payload=json.dumps(
                       {"choices": [{"message": {"content": "hi"}}]}).encode()))
    agentic_runner.ChatClient(ENDPOINT, MODEL, api_key=KEY).ask("hello")
    assert seen[0].get_header("Authorization") == f"Bearer {KEY}"


def test_the_three_decision_backends_send_their_own_key():
    from ainode.bench.decide import backends as be
    from ainode.bench.decide.items import load_items

    item = load_items(None, ["fact"]).items[0]
    for backend in (be.DecideBackend(ENDPOINT, model=MODEL, api_key=KEY),
                    be.ChatBackend(ENDPOINT, MODEL, api_key=KEY),
                    be.JevBackend(KEY)):
        assert backend.request(item).headers == {"Authorization": f"Bearer {KEY}"}


def test_the_embedding_and_speech_requests_send_the_key(tmp_path):
    from ainode.bench.embed.client import EmbedClient
    from ainode.bench.speech.client import SpeechClient

    embed = EmbedClient(ENDPOINT, MODEL, api_key=KEY)
    assert embed.request(["a"]).headers == {"Authorization": f"Bearer {KEY}"}

    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFF")
    speech = SpeechClient(ENDPOINT, MODEL, api_key=KEY)
    request = speech.request({"id": "c1", "path": str(wav)}, b"RIFF")
    assert request.headers == {"Authorization": f"Bearer {KEY}"}


# =============================================================================
# 4. The environment fallback, per section
# =============================================================================

def test_the_speed_section_takes_the_key_from_the_environment(monkeypatch, tmp_path):
    from ainode.bench import cli as bench_cli

    monkeypatch.setenv(auth.ENV_API_KEY, KEY)
    seen = {}

    def fake_measure(opts, rep, cpt=None):
        seen["key"] = opts.api_key
        return {}, 1, 4.0

    monkeypatch.setattr(bench_cli, "measure", fake_measure)
    monkeypatch.setattr(bench_cli, "describe_via_http",
                        lambda *a, **k: ({"id": MODEL}, {}, "n1", []))
    monkeypatch.setattr(bench_cli, "http_nodes_reader", lambda *a, **k: None)
    code = bench_cli.main(["--url", URL, "--model", MODEL, "--label", "env"],
                          out_dir=tmp_path)
    assert code == 0 and seen["key"] == KEY


def test_the_harness_takes_the_key_from_the_environment(monkeypatch, tmp_path):
    """And hands it to the adapters, which put it where their provider config names."""
    from ainode.bench import fleet as fleet_mod
    from ainode.bench.harness import cli as harness_cli

    monkeypatch.setenv(auth.ENV_API_KEY, KEY)
    monkeypatch.setattr(shutil, "which", lambda *a, **k: sys.executable)
    monkeypatch.setattr(fleet_mod, "get_json", lambda *a, **k: {})
    seen = {}

    def fake_suite(tasks, adapters, endpoint, model, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(harness_cli, "run_suite", fake_suite)
    code = harness_cli.main(["--endpoint", ENDPOINT, "--model", MODEL,
                             "--label", "env", "--harness", "aider",
                             "--only-tasks", "two-fer", "--no-metrics"],
                            out_dir=tmp_path)
    assert code == 0 and seen["api_key"] == KEY


def test_the_other_four_sections_take_the_key_from_the_environment(monkeypatch,
                                                                  tmp_path):
    from ainode.bench.agentic import cli as agentic_cli
    from ainode.bench.decide import backends as be
    from ainode.bench.embed.cli import build_parser as embed_parser
    from ainode.bench.embed.client import EmbedClient
    from ainode.bench.harness.adapters import DEFAULT_API_KEY
    from ainode.bench.speech.client import SpeechClient

    monkeypatch.setenv(auth.ENV_API_KEY, KEY)

    # agentic and the two clients resolve through the same helper the CLIs call
    assert auth.key_for("", DEFAULT_API_KEY)[0] == KEY
    assert be.node_api_key()[0] == KEY
    assert be.build_backend("chat", endpoint=ENDPOINT, model=MODEL).api_key == KEY
    assert be.build_backend("ainode", endpoint=ENDPOINT, model=MODEL).api_key == KEY

    args = embed_parser().parse_args(["--endpoint", ENDPOINT, "--model", MODEL,
                                     "--label", "env"])
    key, source = auth.key_for(args.api_key, "ainode")
    assert (key, source) == (KEY, "$AINODE_API_KEY")
    assert EmbedClient(ENDPOINT, MODEL, api_key=key).api_key == KEY
    assert SpeechClient(ENDPOINT, MODEL, api_key=key).api_key == KEY
    assert agentic_cli.build_parser().parse_args(
        ["--endpoint", ENDPOINT, "--model", MODEL, "--label", "l"]).api_key == ""


def test_an_in_process_run_sends_no_key_at_all(monkeypatch):
    """The browser's own bench talks to an engine container, which has no middleware.

    ``/api/bench/runs`` builds its options from the request body, so a key wired in
    there would be a credential posted to a vLLM container by whoever opened the
    page. The field stays empty, and no request body can fill it.
    """
    from ainode.bench.api_routes import _options_from_body

    monkeypatch.setenv(auth.ENV_API_KEY, KEY)
    opts = _options_from_body({"model": MODEL, "label": "web", "api_key": KEY,
                               "sections": ["single"]})
    assert opts.api_key == ""


def test_the_hosted_decision_backend_never_gets_this_nodes_key(monkeypatch):
    """A credential goes to the party it belongs to, and no further.

    ``$AINODE_API_KEY`` is a key for one of our nodes. Feeding it to
    ``api.typesafe.ai`` would hand a fleet credential to a third party, so the two
    resolution orders are deliberately separate functions.
    """
    from ainode.bench.decide import backends as be

    monkeypatch.setenv(auth.ENV_API_KEY, KEY)
    monkeypatch.delenv(be.JEV_KEY_ENV, raising=False)
    monkeypatch.setattr(be, "JEV_KEY_FILE", "/nonexistent/.jev_api_key")
    assert be.jev_api_key() == ("", "")
    with pytest.raises(be.BackendError):
        be.build_backend("jev")


# =============================================================================
# 5. The key reaches nothing a reader can see
# =============================================================================

def test_a_record_and_the_console_never_carry_the_key(monkeypatch, tmp_path, capsys):
    """The whole speed path with a key set, down to the file on disk."""
    from ainode.bench import cli as bench_cli

    monkeypatch.setattr(bench_cli, "describe_via_http",
                        lambda *a, **k: ({"id": MODEL}, {"node": "Spark-1"}, "n1", []))
    monkeypatch.setattr(bench_cli, "http_nodes_reader", lambda *a, **k: None)
    monkeypatch.setattr(bench_cli, "measure",
                        lambda opts, rep, cpt=None: ({"single_stream":
                                                      {"decode_tok_s": 19.0}}, 7, 3.9))
    code = bench_cli.main(["--url", URL, "--model", MODEL, "--label", "keyless",
                           "--api-key", KEY], out_dir=tmp_path)
    assert code == 0
    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    text = written[0].read_text()
    assert KEY not in text
    assert "api_key" not in json.loads(text)["settings"]
    printed = capsys.readouterr().out
    assert KEY not in printed
    assert "key     : from --api-key (never printed)" in printed


def test_the_harness_dry_run_masks_a_real_key(monkeypatch, tmp_path, capsys):
    """The env line prints paths, and a key-shaped variable only when it holds the
    placeholder. It used to compare against ``--api-key``, which printed the key the
    moment that flag could carry a real one."""
    from ainode.bench.harness.cli import _mask, main as harness_main

    assert _mask({"OPENAI_API_KEY": KEY, "HOME": "/home/x"}) == {
        "HOME": "/home/x", "OPENAI_API_KEY": "***"}

    monkeypatch.setattr(shutil, "which", lambda *a, **k: sys.executable)
    harness_main(["--endpoint", ENDPOINT, "--model", MODEL, "--label", "unit",
                  "--harness", "aider", "--only-tasks", "two-fer", "--dry-run",
                  "--api-key", KEY], out_dir=tmp_path)
    printed = capsys.readouterr().out
    assert KEY not in printed
    assert "'OPENAI_API_KEY': '***'" in printed


def test_a_dry_run_request_line_still_carries_no_header():
    """``curl_safe`` is the line a person is shown, so it has no headers at all."""
    from ainode.bench.decide import backends as be
    from ainode.bench.decide.items import load_items
    from ainode.bench.embed.client import EmbedClient

    item = load_items(None, ["fact"]).items[0]
    shown = be.ChatBackend(ENDPOINT, MODEL, api_key=KEY).request(item).curl_safe()
    assert KEY not in shown and "Authorization" not in shown
    shown = EmbedClient(ENDPOINT, MODEL, api_key=KEY).request(["a"]).curl_safe()
    assert KEY not in shown and "Authorization" not in shown


# =============================================================================
# 6. A 401 stops the section before anything is scored
# =============================================================================

def test_the_preflight_reads_a_401_as_a_missing_key(monkeypatch):
    from ainode.bench import auth as auth_mod

    body = json.dumps({"error": {"message": "This node requires an API key.",
                                 "type": "auth_error"}}).encode()
    capture(monkeypatch, auth_mod, raises=http_error(401, body))
    assert real_preflight(ENDPOINT, "") == auth.NEEDS_KEY
    assert "--api-key" in auth.NEEDS_KEY and auth.ENV_API_KEY in auth.NEEDS_KEY


def test_the_preflight_asks_the_model_list_under_either_spelling(monkeypatch):
    from ainode.bench import auth as auth_mod

    seen = capture(monkeypatch, auth_mod)
    real_preflight(ENDPOINT, KEY)
    real_preflight(URL, KEY)
    assert [r.full_url for r in seen] == [
        "http://fake-node.invalid:3000/v1/models",
        "http://fake-node.invalid:8000/v1/models"]
    assert all(r.get_header("Authorization") == f"Bearer {KEY}" for r in seen)


def test_the_preflight_is_silent_about_everything_that_is_not_a_refusal(monkeypatch):
    """An endpoint that 404s the list or does not answer is not a key problem.

    A bench pointed straight at an engine port is the normal case there, and a
    preflight that stopped on it would make the flag mandatory for a node that never
    wanted one.
    """
    from ainode.bench import auth as auth_mod

    for outcome in (http_error(404), http_error(503), OSError("connection refused")):
        capture(monkeypatch, auth_mod, raises=outcome)
        assert real_preflight(ENDPOINT, KEY) is None
    capture(monkeypatch, auth_mod)
    assert real_preflight(ENDPOINT, KEY) is None


@pytest.mark.parametrize("section", ["speed", "harness", "agentic", "decide",
                                     "embed", "speech"])
def test_a_refusing_node_stops_every_section_with_nothing_written(section, tmp_path,
                                                                 monkeypatch, capsys):
    """Six sections, one answer: say what to pass, score nothing, write nothing."""
    from ainode.bench import fleet as fleet_mod
    from ainode.bench.cli import main as bench_main

    refuse_preflight(monkeypatch)
    monkeypatch.setattr(shutil, "which", lambda *a, **k: sys.executable)
    # Nothing below should reach these, and a call is a failure of the stop itself.
    def unreachable(*a, **k):
        raise AssertionError("the section measured something after a refusal")

    monkeypatch.setattr(fleet_mod, "get_json", unreachable)

    argv = {
        "speed": ["--url", URL, "--model", MODEL, "--label", "refused"],
        "harness": ["harness", "--endpoint", ENDPOINT, "--model", MODEL,
                    "--label", "refused", "--harness", "aider",
                    "--only-tasks", "two-fer"],
        "agentic": ["agentic", "--endpoint", ENDPOINT, "--model", MODEL,
                    "--label", "refused", "--quick"],
        "decide": ["decide", "--backend", "chat", "--endpoint", ENDPOINT,
                   "--model", MODEL, "--label", "refused", "--sets", "fact"],
        "embed": ["embed", "--endpoint", ENDPOINT, "--model", MODEL,
                  "--label", "refused"],
        "speech": ["speech", "--endpoint", ENDPOINT, "--model", MODEL,
                   "--label", "refused"],
    }[section]

    code = bench_main(argv, out_dir=tmp_path)
    printed = capsys.readouterr().out
    assert code == 2, f"{section} did not stop"
    assert auth.NEEDS_KEY in printed, f"{section} did not say what to pass"
    assert "nothing was scored and no record was written" in printed
    assert list(tmp_path.glob("*.json")) == [], f"{section} wrote a record"


def test_a_401_mid_run_raises_out_of_every_transport(monkeypatch, tmp_path):
    """A refusal after the preflight passed, which is what a rotated key looks like.

    Each of these is the one place its section turns a failed request into a row, so
    this is the difference between "no measurement" and a row that reads as one.
    """
    from ainode.bench import measure
    from ainode.bench.agentic import runner as agentic_runner
    from ainode.bench.decide import backends as be
    from ainode.bench.decide.items import load_items
    from ainode.bench.embed import client as embed_client
    from ainode.bench.speech import client as speech_client

    refusal = http_error(401, b'{"error": {"message": "Invalid API key"}}')

    capture(monkeypatch, measure, raises=refusal)
    with pytest.raises(auth.EndpointRefused):
        measure.stream_chat(URL, MODEL, "hi", 8, api_key="stale")

    capture(monkeypatch, agentic_runner, raises=refusal)
    with pytest.raises(auth.EndpointRefused):
        agentic_runner.ChatClient(ENDPOINT, MODEL, api_key="stale").ask("hi")

    capture(monkeypatch, embed_client, raises=refusal)
    with pytest.raises(auth.EndpointRefused):
        embed_client.EmbedClient(ENDPOINT, MODEL, api_key="stale").embed(["a"])

    wav = tmp_path / "c.wav"
    wav.write_bytes(b"RIFF")
    capture(monkeypatch, speech_client, raises=refusal)
    with pytest.raises(auth.EndpointRefused):
        speech_client.SpeechClient(ENDPOINT, MODEL, api_key="stale").transcribe(
            {"id": "c1", "path": str(wav)})

    item = load_items(None, ["fact"]).items[0]
    capture(monkeypatch, be, raises=refusal)
    with pytest.raises(auth.EndpointRefused):
        be.ChatBackend(ENDPOINT, MODEL, api_key="stale").decide(item)


def test_a_401_from_the_hosted_backend_is_its_own_row(monkeypatch):
    """TypeSafe's 401 is about TypeSafe's key, so it must not raise our sentence."""
    from ainode.bench.decide import backends as be
    from ainode.bench.decide.items import load_items

    item = load_items(None, ["fact"]).items[0]
    capture(monkeypatch, be, raises=http_error(401, b"bad token"))
    decision = be.JevBackend("their-key").decide(item)
    assert decision.error.startswith("HTTP 401")
    assert auth.NEEDS_KEY not in decision.error


def test_a_refusal_leaves_the_sections_rather_than_being_logged_as_one(monkeypatch):
    """``measure`` swallows a broken section and ``run_probes`` a broken probe. A
    refusal is neither, so both let it through."""
    from ainode.bench import measure
    from ainode.bench.agentic import runner as agentic_runner
    from ainode.bench.agentic.probes import all_probes

    def refusing_section(a, cpt, rep):
        raise auth.EndpointRefused(auth.NEEDS_KEY, status=401)

    monkeypatch.setitem(measure.SECTIONS, "single", ("single_stream", refusing_section))
    opts = measure.BenchOptions(url=URL, model=MODEL, sections=["single"])
    with pytest.raises(auth.EndpointRefused):
        measure.measure(opts, measure.Reporter(), cpt=4.0)

    class RefusingClient:
        def ask(self, *a, **k):
            raise auth.EndpointRefused(auth.NEEDS_KEY, status=401)

        chat = ask

    with pytest.raises(auth.EndpointRefused):
        agentic_runner.run_probes(all_probes(groups=["A"]), RefusingClient(),
                                  log=lambda _m: None)


def test_a_control_plane_401_is_explained_rather_than_quoted(monkeypatch):
    """A placement read the node refused is a warning, and the warning has to be
    readable: ``HTTPError: HTTP Error 401: Unauthorized`` names no fix."""
    from ainode.bench import fleet as fleet_mod

    monkeypatch.setattr(fleet_mod, "get_json", lambda *a, **k: {
        "_error": "HTTPError: HTTP Error 401: Unauthorized"})
    _mb, _pl, _node, warnings = fleet_mod.describe_via_http(
        "http://node:3000", "http://node:3000", MODEL)
    assert any(auth.NEEDS_KEY in w for w in warnings)
    _n, _p, _g, warn = fleet_mod.resolve_serving_node("http://node:3000", MODEL)
    assert auth.NEEDS_KEY in warn


# =============================================================================
# 7. A 429 stops the same way, and names the limit
# =============================================================================

def real_limit_body(limit="max_inflight", limit_value=8, retry_after=3):
    """The rate limiter's own 429 body, so this test and the node cannot drift."""
    from ainode.ratelimit.middleware import (
        Decision as RateDecision,
        RateLimitConfig,
        too_many_requests,
    )

    decision = RateDecision(allowed=False, limit=limit, limit_value=limit_value,
                            retry_after=retry_after)
    return too_many_requests(decision, RateLimitConfig(enabled=True)).body


def test_a_429_reports_the_limit_that_refused_it():
    for limit, expected in (("max_inflight", "too many requests in flight"),
                            ("requests_per_minute", "rate limit exceeded")):
        message = auth.refusal(429, real_limit_body(limit=limit).decode())
        assert message.startswith("this node is rate limiting the bench")
        assert expected in message
        assert "Retry in 3s." in message


def test_a_429_with_nothing_to_quote_still_stops():
    assert auth.refusal(429, "") == "this node is rate limiting the bench"
    assert auth.refusal(429, "Too Many Requests").endswith("Too Many Requests")


def test_a_429_mid_run_stops_the_section_without_scoring(monkeypatch, tmp_path):
    from ainode.bench.embed import client as embed_client

    capture(monkeypatch, embed_client,
            raises=http_error(429, real_limit_body(limit="requests_per_minute",
                                                   limit_value=600)))
    with pytest.raises(auth.EndpointRefused) as caught:
        embed_client.EmbedClient(ENDPOINT, MODEL, api_key=KEY).embed(["a"])
    assert caught.value.status == 429
    assert "rate limit exceeded" in str(caught.value)


def test_a_429_stops_a_section_before_it_scores(tmp_path, monkeypatch, capsys):
    from ainode.bench.cli import main as bench_main

    limit = auth.refusal(429, real_limit_body().decode())
    refuse_preflight(monkeypatch, limit)
    code = bench_main(["embed", "--endpoint", ENDPOINT, "--model", MODEL,
                       "--label", "limited"], out_dir=tmp_path)
    printed = capsys.readouterr().out
    assert code == 2 and limit in printed
    assert list(tmp_path.glob("*.json")) == []


# =============================================================================
# 8. Everything else is still one failed row
# =============================================================================

@pytest.mark.parametrize("status", [400, 404, 500, 503])
def test_an_ordinary_failure_is_a_row_and_not_a_stop(status, monkeypatch):
    """The stop must not swallow the failures a bench exists to report.

    A 400 names a body the engine could not read and a 503 an engine that is not
    up. Both belong in the row that made the request, which is where they were
    before and where every reader looks for them.
    """
    from ainode.bench.embed import client as embed_client

    assert auth.refusal(status) is None
    capture(monkeypatch, embed_client, raises=http_error(status, b"nope"))
    reply = embed_client.EmbedClient(ENDPOINT, MODEL, api_key=KEY).embed(["a"])
    assert reply.error.startswith(f"HTTP {status}")


def test_the_status_reader_understands_both_error_spellings():
    """The section clients write ``HTTP 401: ...``; urllib's own str is different."""
    assert auth.split_error("HTTP 429: {\"a\": 1}") == (429, '{"a": 1}')
    assert auth.split_error("HTTPError: HTTP Error 401: Unauthorized") == (
        401, "Unauthorized")
    assert auth.status_of("URLError: <urlopen error timed out>") is None
    assert auth.status_of("") is None
    # A status inside a body must not be read as the status OF the answer.
    assert auth.status_of("HTTP 400: upstream said HTTP 429") == 400
