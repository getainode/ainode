# AINode harness bench

Can a model served by AINode drive a coding agent to passing tests. Not how fast it
generates: whether the thing it generates works.

The throughput bench next door (`bench/README.md`) answers "what does one user
feel" and "what does the cluster do at 16 streams". Those numbers say nothing about
whether the model can be given a real task in a real agent CLI and come back with
green tests, and a model can be excellent at one and useless at the other. This is
the gate that says which model and which harness are worth putting subagents on.

| Path | What it is |
|------|-----------|
| `ainode/bench/harness/` | The bench, as a package |
| `ainode/bench/harness/tasks.py` | How a task is loaded, and the isolation rule |
| `ainode/bench/harness/adapters/` | One class per agent CLI: aider, dsh, pi, opencode, claude |
| `ainode/bench/harness/runner.py` | The measurement: attempts, tests, scoring |
| `bench/harness/tasks/` | The task set, vendored (see below) |
| `bench/results/*.json` | Where a run lands, schema 1 with a `harness` block |
| `bench/SCHEMA.md` | The record format. Authoritative |

## Run it

```bash
python3 scripts/ainode-bench.py harness \
    --endpoint http://100.122.26.9:3000/v1 \
    --model fraserprice/DeepSeek-V4-Flash-DSpark \
    --harness aider,dsh,pi --tasks 10 --label fleet-flash
```

`--endpoint` is the base every harness is pointed at in its `/v1` form, normally an
AINode node's port 3000 (the fleet endpoint, so the run goes wherever the model is
actually loaded). `--model` is the id exactly as served. `--label` is required and
says what made the run distinct.

One endpoint, whatever protocol the harness speaks. Four of the five talk the
OpenAI protocol and get `--endpoint` verbatim; `claude` talks the Anthropic Messages
API, and its adapter strips the `/v1` because Claude Code appends `/v1/messages`
itself. The fleet endpoint forwards `/v1/messages` (and `/v1/messages/count_tokens`)
by model id, the same way it forwards chat completions, so
`--endpoint http://<master>:3000/v1` is the right answer for every harness here.
Until that landed, `:3000` answered 404 on that path and a Messages client had to be
aimed at one engine's `:8000`, which threw away fleet routing, capability-aware
ordering and failover.

`--dry-run` prints the exact argv, environment overlay and config file for every
task and harness and touches nothing: no subprocess, no config write, no request.
Run it first, every time, especially after a harness upgrade:

```
  aider (aider, on PATH)

    task     : isogram
    cwd      : /tmp/ainode-harness-DRYRUN/aider/isogram
    copied in: instructions.md, isogram.py
    hidden   : isogram_test.py (after the harness exits)
    command  : aider --model openai/fraserprice/DeepSeek-V4-Flash-DSpark
               --openai-api-base http://100.122.26.9:3000/v1 --yes-always --no-git
               --no-auto-commits --no-show-model-warnings --no-check-update
               --no-analytics --no-pretty --message '# Instructions ... [602 chars]'
               isogram.py
    env      : {'OPENAI_API_KEY': 'ainode'}
    tests    : python -m pytest -q isogram_test.py
```

Useful flags:

- `--harness aider,dsh,pi,opencode,claude` - which harnesses. `--list-harnesses`
  prints them with whether each binary is on PATH.
- `--tasks N` - the first N of the set in slug order, so two runs of different
  sizes still agree on their overlap. `--only-tasks isogram,bob` names them instead.
- `--attempts 1` - drop the second try (see the protocol below).
- `--timeout 900` - seconds per harness invocation. A first `dsh` run installs its
  profile before it does anything, so give that one room.
- `--work-dir DIR` - put the working copies somewhere you chose. Without it they go
  to a fresh temp dir, which the run prints as `workdir :` and leaves in place when
  it finishes: reading what the agent actually wrote is the first thing worth doing
  after a surprising score.
- `--context-window` / `--max-output-tokens` - what pi and dsh are told about the
  model, because neither can discover it from an OpenAI-compatible endpoint. They
  are declared, and the record says they were declared.
- `--claude-effort medium` - the reasoning effort `claude` is run with, passed to it
  as `--effort`. Unset sends nothing, which is what every measurement before this
  option was taken with; set it when the served chat template rejects Claude Code's
  default (see the `claude` section). Recorded in the run's `settings` and on the
  claude block's `options`, and ignored by every other harness.
- `--no-metrics` - skip the `/api/metrics` token window.

It is inference only. It never loads, unloads, restarts or deletes anything, so it
is safe to point at a node somebody else is using; it will add real load, and ten
tasks across three harnesses is a lot of tokens.

## Where it has to run

**An unsandboxed shell with normal network and home-directory access.** This is not
a preference. Every one of these agents writes state under `$HOME` (`~/.pi`,
`~/.local/share/opencode`, `~/.dsh`) and opens its own connections, and pi, opencode
and dsh were all seen to hang or error when launched from a sandboxed shell (Claude
Code's Bash sandbox) on a machine where a plain Node `fetch` to the same endpoint
succeeded. pi passed 6/6 the moment it ran unsandboxed. A sandboxed run does not
fail loudly, it just times out, so a whole sweep of zeroes with long wall clocks is
the signature to look for.

Everything except the harness subprocesses is sandbox-safe: `--dry-run`, the task
loader, the scoring and the record all work anywhere, which is why the test suite
does.

## What it measures

Four numbers per harness:

| Score | Meaning |
|-------|---------|
| `pass_at_1` | fraction of tasks whose hidden tests passed on the first attempt |
| `pass_at_2` | ... on either attempt. Cumulative: a first-try pass counts in both |
| `mean_wall_s` | mean over tasks of the harness's total wall clock, every attempt |
| `crashes` | tasks where the harness exited nonzero or could not run at all |

`timeouts` is counted separately from `crashes`, and every task row keeps its own
exit code, wall clock, test counts and output tails, so a bad score can be read as
"the model could not do it" rather than "the harness fell over".

## The rules the numbers depend on

1. **The harness never sees the tests.** Only `instructions.md` and the stub are
   copied into the working directory. The hidden tests go in after the harness has
   exited, are run, and come straight back out before the next attempt. This is
   asserted in the loop, not assumed, and again when a task is loaded: a test file
   at a task's root is a load error. A harness that can read the assertions can
   satisfy them without solving anything.
2. **Two attempts, and the second one is told what failed.** Attempt 2 gets the
   same instructions with the real pytest output appended, in the same working
   directory, which is Aider's polyglot protocol. pass@1 is the model getting it
   right cold and pass@2 is the model reading a stack trace; the gap between them
   is the interesting part of the record.
3. **A crash is a result.** A harness that dies, hangs to its timeout, or exits
   nonzero still gets its tests run against whatever it left on disk, because a
   partial edit that passes is a pass. Nothing is retried to make a number look
   better.
4. **pytest decides.** The verdict is the test command's exit code, never a count
   scraped from its output; counts are recorded alongside because they are useful,
   but a zero exit with three passes and a one exit with three passes are different
   results and only the exit code is consulted. The task's own `test_command` runs
   under the interpreter running the bench, so the verdict cannot come from some
   other `python` on PATH.
5. **Nothing is loaded, unloaded or restarted**, the same rule the throughput bench
   runs under.
6. **Missing is missing.** A harness whose version cannot be read records `null` and
   a note saying so. Token counts are absent unless they were actually read.
7. **Token counts are the node's, not the run's.** When `/api/metrics` is reachable
   the record carries AINode's request and generated-token counters differenced over
   each task's window. Any other traffic on that node during the run is inside that
   delta, and the record's notes say so. Aider additionally reports its own
   `Tokens: N sent, M received` line per exchange, which is what it sent rather than
   what the engine counted; those land in the task row too, labelled per attempt.

## The task set: why Exercism

`bench/harness/tasks/` is ten Python practice exercises vendored verbatim from
[exercism/python](https://github.com/exercism/python) at commit
`1f6aab8667bf653b10cc3799f94352fcdb749db6`, **MIT licensed** (the upstream license
is kept alongside them as `bench/harness/tasks/LICENSE-exercism`).

That source was chosen because it is the same one Aider's polyglot benchmark draws
on, so a number measured here can be read against published numbers for other
models instead of existing only in this repo. The exercises are small, they have
real hidden unit tests rather than a rubric, and each one has exactly one obvious
entry point, so a fail is a fail about the code rather than about file layout.

The ten, in slug order (which is also `--tasks N` order): `binary-search`, `bob`,
`hamming`, `isogram`, `matrix`, `raindrops`, `robot-name`, `run-length-encoding`,
`two-fer`, `word-count`.

They are deliberately easy. This measures whether a model can be trusted to drive
an agent at all, not how clever it is; a model that cannot get `two-fer` green
through an agent is not going to hold up on a real codebase, and finding that out
costs ten minutes here instead of an afternoon.

### What is in a task, and how to add one

```
bench/harness/tasks/<slug>/
  task.json          slug, language, entry, instructions, tests, test_command, source
  instructions.md    the exercise statement
  <entry>.py         the stub the agent must fill in
  tests/<entry>_test.py   the hidden tests. Never copied in until the agent is done
```

`instructions.md` is upstream's `.docs/instructions.md` with
`.docs/instructions.append.md` appended when it exists. The append file is where
the Python track puts "raise `ValueError` with this message", and the hidden tests
assert it, so a task that dropped it would be unsolvable.

To add one: make the directory, write the four things, and keep the test file under
`tests/`. Then bump `count` and `slugs` in `bench/harness/tasks/TASK_SET.json`, and
check the task is real by confirming the hidden tests fail against the stub and pass
against a correct solution. The loader validates the rest (every listed file exists,
no test file at the root) and `tests/test_bench_harness.py` walks the whole set.

`$AINODE_HARNESS_TASKS` points the bench at a different set entirely, which is how
you try a set you have not committed yet.

## The harnesses, and exactly what each one is told

Every adapter passes the model id and the AINode endpoint, and none of them needs
an API key beyond the literal placeholder `ainode`: an AINode endpoint does not
authenticate, and the placeholder exists only because some of these tools refuse to
start with an empty key field.

The provider id the adapters register is `ainode-bench`, not `ainode`, so the bench
adds its own route rather than rewriting a provider somebody made by hand. `claude`
registers no provider at all: Claude Code has exactly one, and the adapter re-points
it at the endpoint through the environment.

All five are verified end to end on one task (`isogram`) against a model on the
fleet, hidden tests 6/6 for every one of them.

| Harness | Version | Verified against | First attempt |
|---------|---------|------------------|---------------|
| `aider` | 0.86.2 (PyPI `aider-chat`) | DeepSeek V4 Flash | 10 s |
| `opencode` | 1.18.31 (`opencode-ai`) | Ornith 1.5 35B A3B | 26 s |
| `pi` | 0.73.1 (`@mariozechner/pi-coding-agent`) | DeepSeek V4 Flash | 20 s |
| `dsh` | 0.1.5-rc.1 (`@deepseek-ai/dsh`) | DeepSeek V4 Flash | 34 s |
| `claude` | 2.1.272 (Claude Code) | Qwen3.8 27B NVFP4 | 294 s, 6 turns |

Every row except `opencode`'s was taken by hand; `opencode`'s was taken by the runner
itself, three consecutive times (25.5 s, 26.7 s, 26.5 s, no crashes) on 2026-09-17
after #118, because a by-hand opencode run was exactly the thing that did not
generalise to the runner.

One task on one model is a working adapter, not a score. Read those columns as "this
adapter reaches the endpoint and the model can do the task" and nothing more: these
are single observations, the harnesses were not run under identical conditions, and
opencode's startup alone was separately timed at about 18 s, which is most of its
26 s above. `claude`'s number is not comparable with the other four at all: a
different model answered it, on a different day. Startup overhead is real and it is
inside `mean_wall_s` (opencode loads every installed skill; dsh installs its profile
on first use), so a comparison worth making comes out of a full run with the caveats
in the per-harness notes below.

### aider

```
OPENAI_API_KEY=ainode aider \
    --model openai/<model id> --openai-api-base <endpoint> \
    --yes-always --no-git --no-auto-commits --no-show-model-warnings \
    --no-check-update --no-analytics --no-pretty \
    --message "<prompt>" <stub>
```

`openai/` is a litellm provider prefix, not part of the model id: it says "speak the
OpenAI protocol to `--openai-api-base`", which is what AINode proxies on port 3000.
The first four `--no-*` flags plus `--yes-always` are the set that was run end to
end. The last three were added on top and are about side effects only: no release
check, no analytics, and no ANSI in the captured output so the `Tokens:` line parses.
`aider --version` gives the version; `Tokens: N sent, M received.` gives one turn
and its token counts, summed over the exchanges in a run.

This is the reference harness. If a task fails under aider it is probably the model;
if it fails only under one of the others, suspect that adapter first.

### dsh

```
dsh --profile headless --patch <generated overlay> "<prompt>"
```

`dsh --profile headless "<job>"` answers one task in the invoking directory, prints
the final message and exits. It takes no endpoint or model flags: a dsh profile is
an ordered stack of plugin config layers, and the last layer is `--patch <file>`. So
the adapter generates a two-entry overlay per run, in the run's scratch directory:

```yaml
- id: llm-pi-ai
  config:
    providers:
      ainode-bench:
        api: openai-completions
        baseURL: "<endpoint>"
        apiKeyEnv: AINODE_BENCH_API_KEY
        models:
          - id: "<model id>"
            name: "<model id>"
            contextWindow: 131072
            maxTokens: 16384
- id: agent-default-model
  config:
    provider: ainode-bench
    model: "<model id>"
```

`baseURL` is the spelling the plugin documents, not `baseUrl`. `compat.thinkingFormat:
deepseek` is added when the model id contains "deepseek", because those models return
reasoning in DeepSeek's own shape and the plugin has to be told or it arrives as
content. The composition is checkable offline, with no endpoint involved, by running
`dsh --profile headless --patch <overlay> --dump-config` and reading both overrides
back out of the composed tree.

**The bench gives dsh its own `DSH_HOME`, and that is the load-bearing part.** dsh
validates **every** configured provider route at boot, so one stale entry in a
person's `~/.dsh/settings.yaml` (a provider pointing at a port that stopped serving)
ends every run with `dsh: TRANSPORT: Connection error.` after about 17 s, no matter
which provider the run selected. That is what the failure looks like, and it looks
nothing like its cause. So the adapter points `DSH_HOME` at
`~/.ainode/bench/harness/dsh-home`, writes a one-route `settings.yaml` there, and
never reads or writes the real one. That directory persists between runs on purpose:
the first use of a profile installs it, which takes minutes. **Pre-warm it** with
`dsh --profile headless --dump-config` before a real run. Not only so the first task
does not hit its timeout: an install inside the first invocation lands in that task's
wall clock and drags `mean_wall_s` with it, and that would be a package manager in a
number that is supposed to be about a model.

`$AINODE_HARNESS_DSH_HOME` moves that home. Point it at a home you have curated and
the adapter leaves its `settings.yaml` completely alone and relies on the overlay
instead; every route in that file then has to resolve, for the reason above, and the
adapter fills in a placeholder for each `apiKeyEnv:` it names that is unset while
leaving real values alone.

dsh prints the final assistant message on stdout and streams reasoning to stderr;
both tails land in the record.

### pi

```
pi --provider ainode-bench --model <model id> \
   --tools read,grep,find,ls,edit,write,bash --no-session --no-context-files \
   -p "<prompt>"
```

Verified end to end: exit 0 in 20 s, the final answer on stdout, the stub edited,
6/6 hidden tests green. `-p` is a boolean that makes the run non-interactive; the
prompt is positional, so it goes last. pi has no permission prompts at all, so nothing else is needed to let
it edit. There is no base-URL flag: an OpenAI-compatible endpoint is a provider
entry in `~/.pi/agent/models.json`, and the adapter **merges** one key under
`providers` into whatever is already in that file, leaving the rest byte for byte
alone. `$AINODE_HARNESS_PI_HOME` moves the file, which is how the tests write into a
tmp dir rather than a real one; when it is set, `HOME` follows it for the subprocess.

`--no-session` and `--no-context-files` are the bench's own hygiene: nothing carries
over between tasks, and no `AGENTS.md` or `CLAUDE.md` from an enclosing directory
gets to change what the model was told.

### opencode

```
opencode run --dir <workdir> --print-logs --log-level ERROR \
    --pure --auto --format json -m ainode-bench/<model id> "<prompt>"
```

Four of those flags are working requirements, not preferences:

- `--dir <workdir>` names the directory to run in. **Without it opencode does not
  use its own working directory**: it resolves the project from the inherited `PWD`,
  and a subprocess started with `cwd=` still carries its parent's. From the runner
  that is wherever the bench was started, so opencode built its session there, never
  read the `opencode.json` sitting beside the task, and died in about 1 s with
  `ProviderModelNotFoundError: Model not found: ainode-bench/<id>`, printed on stdout
  as the unhelpful `{"type":"error",...,"message":"Unexpected server error. Check
  server logs for details."}`. By hand the same command passed, because a person
  `cd`s in first and `PWD` is then correct, which is what made
  [#118](https://github.com/getainode/ainode/issues/118) look intermittent for a day.
  `_launch` now also sets `PWD` to the cwd for every harness; this flag is the
  explicit half of the same fix.
- `--auto` approves permissions that are not explicitly denied. Without a TTY there
  is nobody to approve the file write, so the run sits until the timeout.
- `--format json` makes it stream NDJSON events on stdout. Without it, two verified
  runs produced **no output at all** until they timed out.
- `--pure` skips external plugins.

`--print-logs --log-level ERROR` is diagnostics and not behaviour: opencode's logs go
to stderr, stdout stays the NDJSON stream, and a failure then reaches the record's
`stderr_tail` with a name on it. The isolated log file opencode writes is empty in
exactly the case worth reading, and "Unexpected server error" on its own is what #118
had to work from.

The adapter reads one thing out of that stream: `step_start` events, counted as
turns. The event schema is upstream's, so a line that does not parse is skipped and
a shape change costs the `turns` field and nothing else.

The provider is a project-local `opencode.json` written into the working directory
(config, not a hint), naming the endpoint through `@ai-sdk/openai-compatible`. The
model entry carries just a name, which is the shape that was verified, so the run's
declared context window and output cap do not reach opencode; they exist for pi and
dsh, which will not route without them. OpenCode expects to be inside a git repo, so
the working directory gets a bare `git init` with no commit and no identity.

**Its state is the run's own, and `OPENCODE_CONFIG_DIR` is the load-bearing part of
that.** The four XDG variables point at `<scratch>/opencode-xdg/`, which is where
opencode keeps its SQLite database, logs and snapshots, so runs cannot poison each
other and the operator's own profile is never read or written. But
`OPENCODE_CONFIG_DIR` overrides that search outright, and it is a variable people and
tools export (Orca sets one), so the overlay names it too and points it at the run's
own empty config directory. Without that line somebody's global config, plugins and
agents are inside the measurement while every XDG variable says otherwise.

Two things to know before reading its wall clock: startup costs about 18 s because it
loads every skill under `~/.claude/skills` and `~/.agents/skills` even with `--pure`,
and that is inside `mean_wall_s`. And `-m ainode-bench/<model id>` relies on the
provider/model split taking the first slash so a model id with slashes in it survives;
verified for one such id, and an "unknown model" error is the first place to look.

### claude

The only harness here that does not speak the OpenAI protocol. Claude Code talks the
Anthropic Messages API and nothing else, which is why it could not be benched against
AINode until the proxy forwarded that path.

```
env ANTHROPIC_BASE_URL=http://<node>:3000 ANTHROPIC_API_KEY=ainode \
    ANTHROPIC_AUTH_TOKEN=ainode \
    ANTHROPIC_MODEL=<model id> ANTHROPIC_SMALL_FAST_MODEL=<model id> \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_TELEMETRY=1 \
    CLAUDE_CONFIG_DIR=<scratch>/claude-config \
  claude -p "<prompt>" --model <model id> --dangerously-skip-permissions \
    --output-format json --max-turns 12
```

Verified end to end on claude 2.1.272: 6/6 hidden tests on `isogram` driving Qwen3.8
27B NVFP4 on the fleet, 294 s over 6 turns, exit 0.

**`ANTHROPIC_BASE_URL` is the base without the `/v1`.** Claude Code appends
`/v1/messages` itself, so the `/v1` form every other harness wants would produce
`/v1/v1/messages` and a 404. `--endpoint` stays the `/v1` form on the command line and
this adapter is the one place that strips it, so nothing else has to know.

`--dangerously-skip-permissions` is what makes the run non-interactive: there is no
TTY to approve a file write or to trust the directory, and without it the run sits
until the timeout rather than failing. `--output-format json` puts one JSON object on
stdout, and the adapter reads two fields out of it: `num_turns` becomes the task's
`turns`, and **`is_error: true` is recorded as a crash even on exit 0**, because
Claude Code reports a run it could not finish (a turn cap hit mid-edit, an API error
it gave up on) in the payload and still exits cleanly. `stderr` carries
`[claude-code:unrecognized_model] {...}` for any non-Anthropic model id, which is
expected for every model this bench measures and is harmless.

**The effort level is a real failure mode, and `--claude-effort` is the way out of
it.** Claude Code sends reasoning effort "high" by default, and a served chat
template does not have to accept that value. Qwen3.8-Flash-Next takes only xhigh,
medium and low: on 2026-09-16 every request came back `API Error: 400 Unexpected
reasoning effort high`, so the harness crashed in about 0.3 s per attempt and scored
0/10 on a model the other three harnesses were passing. The same suite scored 8/10
and then 10/10 once the level was lowered:

```
scripts/ainode-bench.py harness --endpoint http://<node>:3000/v1 \
    --model nvidia/Qwen3.8-Flash-Next-NVFP4 --harness claude \
    --claude-effort medium --label flash-next-claude
```

which appends `--effort medium` to the argv above. Leave it unset for a model that
accepts high, so its number stays comparable with the runs already recorded. The
level goes into the run's `settings` as `claude_effort` and onto the claude block's
`options` as `effort`, so a record always says which one was measured, and a run
without the flag says nothing rather than claiming a default.

**`CLAUDE_CONFIG_DIR` points at the run's scratch directory, and that is the
load-bearing part.** A bench run must never read or write the operator's own Claude
Code profile: no settings, no hooks, no MCP servers, no sessions, no credentials.
That is also what keeps the measurement honest, since a personal `settings.json` can
add hooks and MCP servers that change what the agent does. **It needs no seed file.**
Verified on 2.1.272 against a fake Messages endpoint: pointed at a directory that had
never existed, `claude -p` with the key in the environment created `.claude.json`,
`projects/`, `sessions/` and `backups/` itself, never prompted for a login, and
exited 0. So `config()` writes nothing.

One gap that isolation does not close: a `CLAUDE.md` in a **parent** of the working
directory is still discovered and still reaches the model (measured: a marker string
in a parent `CLAUDE.md` arrived in the request body). The default work dir is a fresh
temp dir, so a normal run is clean; if you pass `--work-dir` into a checkout that has
a `CLAUDE.md` above it, that file is in the run. There is no per-run flag for this
short of `--bare`, which also turns off hooks, skills and auto-memory and has not
been verified end to end against a model, so it is not wired in.

Claude Code is given a `git init`ed working directory: it reads git state for context
and keeps its own edits diffable.

## How to add a harness

Four small methods in `ainode/bench/harness/adapters/<name>.py`, subclassing
`HarnessAdapter`:

- `name` and `binary`, plus `version_args` if `--version` is not how it answers.
- `command(req) -> list[str]` - the argv. **A pure function of the request.**
- `env(req) -> dict` - the environment overlay. Also pure.
- `config(req) -> list[ConfigFile]` - files that must exist before the command can
  reach the endpoint. Set `merged=True` on a file the adapter merges into rather
  than owns, and the dry run will not echo its contents (it is somebody else's file
  and may hold their keys).
- `parse(stdout, stderr) -> dict` - optional, for `turns` / `tokens_sent` /
  `tokens_received` when the harness says so itself.
- `needs_git = True` if it insists on a repo.

Then register it in `registry()` and pin all of the above in
`tests/test_bench_harness.py`. Purity is the rule that makes this testable with no
model in the loop: the tests assert the exact argv and the exact config bytes for
every harness, so a flag that moves upstream is a failing test rather than a
silently wrong benchmark. `run()` is implemented once in the base class and no
adapter should override it.

Two things the environment will not tell an agent correctly, learned from #118: say
the working directory in the argv if the tool has a flag for it rather than trusting
it to read its own cwd (opencode reads `PWD`, which a `cwd=` subprocess inherits from
its parent), and neutralise any variable that overrides the config search you just
isolated (`OPENCODE_CONFIG_DIR` beats every `XDG_*` you set).

## The record

One JSON per run in `bench/results/`, schema 1, with a top-level `harness` block and
**no `results` block**: a harness run measures no throughput and a zero there would
be a number nobody took. `bench/SCHEMA.md` has the shape.
`scripts/render-bench-table.py` skips records shaped like that, so a harness run
never appears in the README's tok/s table as a very slow model. A run that measured
both throughput and harness scores (same file, both blocks) does get a row.

There is no published harness table yet, and there will not be one until there are
real runs to put in it: the first numbers have to come off the fleet, not out of
this document.
