# AINode agentic rubric

Can a served model be trusted with the parts an agent loop is actually made of. Not
how fast it generates, and not whether it can solve an Exercism task: whether it
follows a format, calls one tool with the right arguments, calls three at once when
three were asked for, notices a tool came back with an error instead of inventing the
answer, keeps a system rule alive over four turns, and finds one sentence in a
100k-token prompt.

Those are the failures that make a local model unusable in a harness long before its
coding score does. A model can pass 10/10 on the harness bench and still fabricate a
temperature the tool never returned.

| Path | What it is |
|------|-----------|
| `ainode/bench/agentic/` | The bench, as a package |
| `ainode/bench/agentic/probes.py` | The probes and their checkers, one mechanical verdict each |
| `ainode/bench/agentic/runner.py` | The transport, the loop, the scoring, the record |
| `ainode/bench/agentic/cli.py` | `scripts/ainode-bench.py agentic ...` |
| `bench/results/*.json` | Where a run lands, schema 1 with an `agentic` block |
| `bench/SCHEMA.md` | The record format. Authoritative |

## Run it

```bash
python3 scripts/ainode-bench.py agentic \
    --endpoint http://100.122.26.9:3000/v1 \
    --ainode http://100.122.26.9:3000 \
    --model fraserprice/DeepSeek-V4-Flash-DSpark \
    --label "DeepSeek TP=2 Spark-2+3, quick" --quick
```

`--endpoint` is the OpenAI-compatible base in its `/v1` form, normally an AINode
node's port 3000 so the run goes wherever the model is actually loaded. `--model` is
the id exactly as served. `--label` is required and says what made the run distinct.
`--ainode` is the web base used for placement; without it the record carries no
placement rather than a guessed one.

`--dry-run` prints the probe list and the request shape and touches nothing: no
request, no file. Run it first.

`--quick` is the shape to run on a busy node: the 8000-token needle only and no
vision probe, which is a few minutes instead of a long wait on a 100k prefill.

Useful flags:

- `--groups A,B,C,D,E,F,V,G` - which groups to run. Default all. A group that did not
  run is absent from the score rather than counted as zero.
- `--needle 8000,48000,100000` - prompt sizes for group E, in tokens.
- `--temperature 1.0` - applied to every probe except B2, which pins 0.2 the way the
  hand-run script did.
- `--no-think-kw NAME` - the `chat_template_kwargs` switch name for the thinking-off
  probe. Both `enable_thinking` and `thinking` always go out together (Qwen-family
  templates read the first, DeepSeek V4 reads the second), and `NAME` adds a third.
- `--timeout 900` - seconds per request. The 100k needle prefill is the slow one.
- `--api-key` - bearer token for the endpoint. Defaults to `$AINODE_API_KEY`, then to
  the placeholder `ainode` that an open node accepts. Never printed and never written
  into a record: a run reports only which of the three it came from. A node that wants
  a key and did not get one stops the run before any probe is scored, rather than
  failing all 25 the same way.

## The probes

24 or 25 probes depending on `--needle`. Every verdict is mechanical: nothing here is
scored by reading a reply.

**A, instruction precision**

- `A1_format` - five planets, exactly five lines of `N. NAME` in capitals, nothing else.
- `A2_json_only` - raw JSON for Vienna with exactly three keys and `landlocked: true`, no fences.
- `A3_constraints` - three sentences about a bicycle with no letter z and no word "the".
- `A4_persona` - a pirate system prompt against a user turn that says to ignore it; the reply still ends in `Arr!`.

**B, tool calling**

- `B1_tool_single` - one `get_weather` call for Tokyo, and only one.
- `B2_tool_parallel` - three things asked in one turn, three or more calls back.
- `B3_tool_not_needed` - a haiku with tools offered: no call, and actual text.
- `B4_tool_roundtrip` - the tool answers 31 C; the reply uses it and does not call again.

**C, coding, executed**

- `C1_ttl_cache` - an LRU-plus-TTL cache with an injectable clock, run against hidden asserts.
- `C2_intervals` - `merge_intervals` and `free_slots`, run against hidden asserts.
- `C3_bugfix` - a broken `top_k_words` (case, punctuation, tie order) fixed, run against hidden asserts.

**D, reasoning traps**

- `D1_decimal` - 9.9 is larger than 9.11.
- `D2_strawberry` - three r's in "strawberry".
- `D3_sisters` - Alice's brother has two sisters.
- `D4_knights` - A is a knight, B is a knave.

**E, needle in a haystack**

- `E_needle_<size>` - one password sentence 43% into a haystack of filler at roughly `<size>` prompt tokens; the reply has to quote it verbatim. The probe's note carries the `prompt_tokens` the server counted.

**F, the thinking switch**

- `F1_thinking_off` - 17*23 with thinking switched off in `chat_template_kwargs`: right answer and an empty reasoning channel.

**V, vision**

- `V1_vision` - a 64x64 solid red PNG as a data URI, named in one word. A 400 here is a text-only model, and the record says `vision_supported: false` rather than pretending the probe was not run.

**G, agentic work**

- `G1_tool_chain` - find a setting with `list_files` and `read_file`: the verdict is the trace, so it has to list before it reads, read the live config rather than the `.bak` decoy, and answer with the value that was in it.
- `G2_tool_error` - the city in the question is misspelled, so `get_weather` answers `{"error": "city not found"}` for it; any corrected spelling works. Pass is calling the tool with a city it accepts (whether the model normalized the spelling itself or only after the error) or telling the user it could not find the place. Stating a temperature the tool never returned fails the probe whatever else it did, and that is the failure the probe exists to catch.
- `G3_arg_schema` - one tool with an enum string, an integer and a required boolean: the arguments have to parse as JSON, match those types, name a real enum member, and invent no extra keys.
- `G4_structured` - `response_format` with a named JSON schema, falling back to `json_object` if the server answers 400. The required keys have to be there with the right types, and the record says which mode got through.
- `G5_persistence` - a system rule ("end every reply with DONE on its own line") checked on all four turns of a conversation, not just the first.

## Honesty rules

The same ones the rest of `bench/` runs under, plus two of its own.

- **Nothing is loaded, unloaded or restarted.** The run drives inference against an
  endpoint that is already serving and adds real load to it. Group E sends a
  100k-token prompt; on a busy node that is felt.
- **Every verdict is mechanical.** No judge model, no reading of replies to reach a
  score. Group C is the strongest form of that: the model's code is executed against
  asserts it never saw, and `PASS` on stdout is the verdict.
- **Group C runs model-written code on the machine driving the bench**, in a
  subprocess with a 60-second timeout, in a temporary directory. That is the same
  trade the harness bench makes when it runs an agent's edit, and it is the only way
  to know whether the code works. Run the bench from a machine where that is
  acceptable.
- **A probe that raises is one failed probe, not a failed run.** One broken check
  must not throw away the twenty-three that measured cleanly.
- **A skipped probe is absent, never a zero.** `--groups` and `--quick` change what
  was asked; the record keeps both the request (`settings`) and what ran
  (`agentic.protocol.groups`).
- **A server error is labelled as one.** A probe that failed on an HTTP 400 says so
  in its note and is listed in the record's notes separately from a probe the model
  got wrong.

## Where the numbers go

One JSON per run in `bench/results/`, named
`<stamp>-<model-slug>-<label-slug>-agentic.json`, schema 1 with an `agentic` block
and no `results` block. The format is `bench/SCHEMA.md`. The README's "Agentic rubric
runs" table is generated from those files:

```bash
python3 scripts/render-bench-table.py          # rewrite the table
python3 scripts/render-bench-table.py --check  # exit 1 if it drifted
```

## History

This started as a scratch script run by hand against one endpoint at a time, and two
records in `bench/results/` still carry the score it produced as a hand-typed
`rubric` block. That is exactly the shape of number the repo's own rules say not to
trust: a total nobody can check, with no per-probe detail behind it. The block stays
in those records as a historical claim, the README still renders it in the Rubric
column of the speed table, and every new run writes the `agentic` block instead.

The port kept every probe and every check semantic from that script, with two
deliberate differences so a probe can be run on its own: each needle size seeds its
own filler stream instead of sharing one across sizes, and `B4_tool_roundtrip` makes
its own first tool call rather than reusing B1's. Group G and the vision probe are
new; the script's vision probe was present but switched off, and its embedded PNG was
truncated, so this one carries a valid image and a comment saying how to regenerate
it.
