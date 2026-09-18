# AINode decision bench

Can a backend's typed decisions be trusted by code that acts on them. Not how fast
it generates, not whether it can solve an Exercism task, not whether it can hold an
agent loop together: given a state and a fixed set of options, does it pick the right
one, and is the number it reports next to that pick worth anything.

That second half is the measurement. A router, a triage step, a guardrail or an
auto-merge check all do the same thing with a decision: gate on the confidence and
escalate what falls below it. A backend with 96% accuracy whose confidence means
nothing is worse to automate than a slightly less accurate one that knows when it is
guessing, because **a wrong answer at 0.95 gets acted on and a wrong answer at 0.45
is an abstention a person looks at**. So read the calibration columns before the
accuracy column.

| Path | What it is |
|------|-----------|
| `ainode/bench/decide/` | The bench, as a package |
| `ainode/bench/decide/items.py` | The labeled set, loaded and validated strictly |
| `ainode/bench/decide/metrics.py` | Accuracy, Brier, calibration, thresholds, latency, cost |
| `ainode/bench/decide/backends.py` | The three backends, each a request builder plus a parser |
| `ainode/bench/decide/runner.py` | The loop, the tables, the record |
| `ainode/bench/decide/cli.py` | `scripts/ainode-bench.py decide ...` |
| `bench/decide/items.json` | The 110 labeled items. Repo data, versioned next to the results |
| `bench/results/*.json` | Where a run lands, schema 1 with a `decide` block |
| `bench/SCHEMA.md` | The record format. Authoritative |

## Run it

```bash
# the hosted comparison (well under a cent for 110 items)
python3 scripts/ainode-bench.py decide --backend jev --label "jev-latest, 110 items"

# any OpenAI-compatible engine, through AINode so the run goes where the model is
python3 scripts/ainode-bench.py decide --backend chat \
    --endpoint http://100.122.26.9:3000/v1 \
    --ainode http://100.122.26.9:3000 \
    --model ornith-ai/Ornith-1.5-35B-A3B-NVFP4 \
    --label "Ornith stacked Spark-1, chat fallback"

# AINode's own decision endpoint
python3 scripts/ainode-bench.py decide --backend ainode \
    --endpoint http://100.122.26.9:3000/v1 --ainode http://100.122.26.9:3000 \
    --model ornith-ai/Ornith-1.5-35B-A3B-NVFP4 --label "Ornith via /v1/decide"

# two backends over the same items, printed side by side
python3 scripts/ainode-bench.py decide --backend chat --compare jev ... --label ...
```

`--dry-run` prints the item counts, the backend and one example request per question
shape, and touches nothing: no request, no file. Run it first.

Flags:

- `--backend ainode|chat|jev` (required) - which backend answers. See below.
- `--compare BACKEND` - run a second backend over the same items and print the two
  side by side. Each backend still writes its own record, because a record is one
  model on one placement.
- `--endpoint http://host:3000/v1` - the OpenAI-compatible base for the `ainode` and
  `chat` backends, normally an AINode node's port 3000 so the run goes wherever the
  model is loaded.
- `--model` - the id exactly as served. It belongs to the local backends; the `jev`
  backend uses `jev-latest` unless it is the only backend and `--model` names one of
  its own aliases, so a `--compare jev` never posts a local model id to TypeSafe.
- `--ainode http://host:3000` - web base used for placement (the same
  `resolve_serving_node` path the other sections use). Without it the record carries
  no placement rather than a guessed one.
- `--label` (required) - free text saying what made this run distinct.
- `--items PATH` - item file (default `bench/decide/items.json`, overridable with
  `$AINODE_DECIDE_ITEMS`).
- `--sets route,fact` - which sets to run. Default all. A typo is refused rather than
  quietly measuring nothing.
- `--concurrency 8` - items in flight at once.
- `--timeout 120` - seconds per item.
- `--api-key` - bearer token. The endpoint's for `ainode`/`chat` (default `ainode`),
  TypeSafe's for `jev`, which otherwise reads `$TYPESAFE_API_KEY` and then
  `~/.jev_api_key`. **The key is never printed, never written into a record and never
  put in a note**; a run reports only which of the three places it came from.

## The item sets

110 items, each one state, one typed question and one label a person can check. They
are five different shapes of the same job, not five difficulty levels.

| Set | n | Kind | What it measures |
|-----|---|------|------------------|
| `route` | 30 | choice, 4 options | Which engine class should serve a request: code, chat, vision, or a long document. The decision a model router makes on every request, and the one where a wrong high-confidence answer sends a 300-page contract to a 4k-context model |
| `triage` | 20 | choice, 4 options | Which support team owns a ticket: billing, technical, sales, account. Boundaries are deliberately close, so a confident miss is a ticket a human never sees |
| `urgency` | 20 | yes/no | Whether a ticket has to be handled within the hour. Written so the yes/no split is judgement and not keyword matching: a double charge is urgent, a wrong VAT number is not |
| `pr_safe` | 20 | yes/no | Whether a proposed diff is safe to merge without review. Half of them are ordinary (a typo fix, a type hint, a dependency patch) and half are the ones that must never be auto-approved: md5 hashing, a deleted test, a hardcoded AWS key, `rm -rf` in an installer, a bypassed PR requirement |
| `fact` | 20 | yes/no | Whether a statement is true. Plain world and infrastructure facts, including four popular wrong ones (the Great Wall from the Moon, leap years every four years without exception, Everest in the Andes, JSON comments) |

The labels are the point of the file: an item with a label somebody could argue with
is a bad item, and the two support sets deliberately reuse the same 20 tickets so the
`triage` and `urgency` answers are two decisions about one state.

## The metrics

Per set and overall, in every record:

- **n / answered / errors** - how many items, how many came back with an answer, and
  how many failed on the transport or an unreadable response. A failed item is never
  a wrong answer and never a silent drop.
- **accuracy** - correct over answered. The weakest number here.
- **Brier score** - mean `(1 - p)^2` where `p` is the probability the backend put on
  the **labeled** option. One term rather than the full multiclass sum, because that
  is the quantity a caller gates on. Lower is better; a coin flip scores 0.25.
- **ECE, with its reliability table** - five bins over the probability the backend
  gave its own answer. Per bin: how many items landed there, how often they were
  right, and the mean confidence claimed. The ECE is the weighted gap between those
  last two. A backend that says 0.9 and is right 90% of the time scores 0; one that
  says 0.99 and is right 80% of the time scores about 0.19, and that number is the
  answer to "can I automate this".
- **wrong at 0.8 and at 0.9** - of the answers that survive the gate, how many
  disagree with the label, plus how many items the gate abstained on. This is the
  same information as the ECE said as a count, and it is the one to quote: "at 0.9 it
  answered 106 of 110 and got 2 of those wrong".
- **p50 / p95 latency** - the measured round trip per item, as observed values.
- **tokens in / out** - what the backend reported, summed. Absent counters add
  nothing rather than being estimated.
- **cost** - the vendor's posted rate over those tokens. Jev bills $0.042 per million
  input tokens and nothing for output; the local backends are $0, because nobody
  bills per token for our own hardware. The electricity is real and is not a number
  these records claim to have measured.

Confidence, throughout, is **the probability the backend put on the answer it gave**,
taken from its distribution when it returns one and from its own reported confidence
when it does not. An answered item with neither is counted in `no_confidence` and left
out of the calibration numbers instead of being given a probability nobody reported.

## The three backends

**`ainode`** posts one typed question per item to AINode's own
`POST /v1/decide`:

```json
{"state": "<the item's state>",
 "questions": {"decision": {"question": "<the question, with the option rubric>",
                            "options": ["code", "chat", "vision", "long_document"]}}}
```

A yes/no item goes out as `{"question": "...", "type": "boolean"}`. The endpoint also
takes a `score` question (`{"question": "...", "type": "score", "min": 0, "max": 2}`);
no item in the set is a score yet, so there is no mapping for it here rather than an
untested one. The endpoint's
choice question takes bare option names and has no room for a per-option rubric, so
the option descriptions are appended to the question text: that keeps the same words
in front of the model that the other two backends put there, which is the only way
the rows are comparable. The answer is read tolerantly on purpose (`true`, `"true"`,
`"yes"` and `1` are all a yes, and a one-sided distribution is completed), so a
correct answer is never scored wrong over a spelling.

**`chat`** is the fallback, and the reason the bench works against any
OpenAI-compatible engine including before `/v1/decide` ships. The options are
lettered, thinking is switched off under both spellings, `max_tokens` is 4,
`temperature` is 0, and the distribution is a softmax over the top logprobs of the
single letter token, restricted to this item's letters. It is a weaker instrument
than a decision endpoint: those probabilities are over **letters**, not over
meanings, and a server that returns no logprobs gives an answer with no confidence at
all. The record says which backend produced every row, and the README table keys on
(backend, model), so two backends' numbers for one model are two rows and never an
average.

**`jev`** is TypeSafe AI's hosted System One model, `POST /v1/systemone`, a choice
question with its `criteria` map or a `noul` question. It is the outside comparison:
a model trained for calibrated typed decisions, priced per input token, with no node
of ours behind it, which is why its records carry the placement
`{"node": "typesafe.ai hosted"}` and a model id of whatever version the API reported
(`jev-1.13.0`) rather than the alias that was asked for (`jev-latest`).

## Honesty rules

The same ones the rest of `bench/` runs under, plus three of its own.

- **Nothing is loaded, unloaded or restarted.** The run drives inference against an
  endpoint that is already serving and adds real load to it.
- **An item that failed says why.** A transport error or an unreadable response is one
  row with an `error`, never a wrong answer, and the record's notes count those items
  separately from the ones the backend got wrong: "the server refused" and "the model
  was wrong" are different findings.
- **A probability nobody reported is absent, never assumed.** No backfilled 0.5.
- **A skipped set is absent, never a zero.** `--sets` changes what was asked; the
  record keeps both the request (`settings.sets`) and what ran (`decide.sets`).
- **The API key never leaves the process.** Not in the console, not in the record, not
  in a note, not in a dry run. A run reports the source (`--api-key`,
  `$TYPESAFE_API_KEY`, `~/.jev_api_key`) and nothing else.
- **Cost is a posted rate over reported tokens, or zero.** Never an estimate.

## Adding items

Edit `bench/decide/items.json`. An item is self-contained:

```json
{
  "id": "pr_safe-21",
  "set": "pr_safe",
  "kind": "noul",
  "state": "Proposed change in a pull request: <the diff, in one sentence>",
  "question": "Is this change safe to merge without a human review? ...",
  "label": false
}
```

A choice item carries its options as well:

```json
{
  "id": "route-31", "set": "route", "kind": "choice",
  "state": "User request: ...",
  "question": "Which kind of model should serve this request?",
  "criteria": {"code": "Programming, debugging, writing tests, shell commands",
               "chat": "...", "vision": "...", "long_document": "..."},
  "label": "vision"
}
```

Then bump that set's `count` in the file's `sets` header. Loading is strict and a bad
item is a load error rather than a skipped item, because a bench that quietly ran 104
of 110 items would publish an accuracy against a count nobody chose. The rules,
checked by `tests/test_bench_decide.py`:

- every item has `id`, `set`, `kind`, `state`, `question` and `label`, and ids are
  unique;
- a `choice` item has `criteria` with at least two options, each described by a string
  or `null`, and its label is one of them;
- a `noul` item's label is a real boolean, and its optional `criteria` may only say
  what `true` and `false` mean;
- every item in a set shares that set's kind, question and options, because a set is
  one measurement;
- a set's declared `count` matches the items it holds.

Adding items changes what a number means, so the sets are versioned: the file carries
an `id` (`decide-110`) and a `version`, every record copies both into its
`decide.item_set` block, and a record's accuracy is only comparable to another record
over the same item-set id.

## Where the numbers go

One JSON per run in `bench/results/`, named
`<stamp>-<model-slug>-<label-slug>-decide.json`, schema 1 with a `decide` block and
no `results` block (a `--compare` run's second record gets its backend name in the
filename so the two cannot collide). The format is `bench/SCHEMA.md`. The README's
"Decision runs" table is generated from those files:

```bash
python3 scripts/render-bench-table.py          # rewrite the table
python3 scripts/render-bench-table.py --check  # exit 1 if it drifted
```

## History

This started as a scratch script comparing Jev against Ornith 1.5 on a hand-built
list of items, run once by hand. Every item and every label was kept in the port,
along with the fallback's exact prompt wording and the five-bin calibration, so the
numbers stayed comparable; what the port added is the strict item file, the record,
the per-set blocks, the threshold counts, the wrong-answer list and the
`/v1/decide` backend.
