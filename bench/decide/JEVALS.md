# The Jevals recipe, suite 0.1.0

Read from <https://jevals.com/methodology> on **2026-09-21**, plus
<https://jevals.com/policy/> (listing rules) and <https://jevals.com/> (the boards) the
same day. The three suite files were downloaded the same day from
`https://jevals.com/data/suites/0.1.0/<id>.json` and are committed verbatim under
`bench/decide/sets/`.

Two further sections below record, from their own primary sources read the same day, the
maintained multi-system board this field actually ranks on (**JevBench**, whose metric
names this bench adopts) and the **fourth question set** (`LocalLLaMA/typed-decisions`,
the one already in the `/v1/systemone` wire shape). A record says which recipe every
figure follows, because two boards measuring the same word differently is how a
comparison becomes a lie.

This file exists so the bench can be read against the recipe it claims to follow
rather than against somebody's memory of it. Everything under "The recipe" is a
statement Jevals publishes. Everything under "What we had to author" is ours, and a
number this bench produces is only comparable to a board number to the extent those
authored parts do not matter. Both lists are meant to be short and complete.

Attribution, per `https://jevals.com/policy/#license`: board data, run logs and suite
files are CC-BY-4.0, cited as "Jevals (jevals.com), release <release>". Jevals is an
independent project and is not affiliated with TypeSafe AI, and neither is AINode.

## The recipe

### What a decision is

A system reads a **state** (any text or JSON) and answers one **typed question** with a
probability distribution over the allowed answers. Three question types, the three
primitives of Jev's interface:

| Type | What it is | What comes back |
|------|-----------|-----------------|
| `noul` | Yes or no (short for Bernoulli) | P(yes) |
| `choice` | Pick one of up to 255 options | a probability per option |
| `score` | Place the state on an ordered rubric of 2 to 10 levels | a probability per level |

The system's **pick** is its most likely answer. **Criteria** are the descriptions of
the options or levels that come with a question. **Confidence** is the probability of
the pick. Each primitive has its own board and its own ranking, and there is no overall
index across primitives. Every label is ground truth from a public human-labelled
dataset; no model grades another model.

### The three tasks

One task per primitive. Each is a fixed sample of **300 items** drawn by proportional
allocation (largest remainder) over the split's natural label distribution, after
dropping items whose state is longer than **6,000 Unicode code points**, with a fixed
seed (`20260918` in all three suite files). **Every item is answered 5 times.**

| Board | Dataset | Config / split | Revision | Licence | Items | K | State fields | Question |
|-------|---------|----------------|----------|---------|-------|---|--------------|----------|
| `noul` | `qiaojin/PubMedQA` | `pqa_labeled` / `train` | `9001f285` | MIT | 300 | 2 | `question`, `context.contexts` | Given the context passages from a biomedical abstract, is the answer to the research question yes? |
| `choice` | `mteb/banking77` | `default` / `test` | `18072d26` | CC-BY-4.0 (Banking77, PolyAI; mirror tagged MIT) | 300 | 77 | `text` | Which intent does this banking customer's message express? |
| `score` | `nvidia/HelpSteer2` | `default` / `validation` | `990b2711` | CC-BY-4.0 | 300 | 5 | `prompt`, `response` | How helpful is the response to the prompt? |

The state is a JSON object built **only** from those whitelisted fields, so no field
that reveals the label reaches any system. Item text is not republished by Jevals; each
item links to its upstream row. `state_sha256` is the SHA-256 of the UTF-8 bytes of
`JSON.stringify(state)` and is checked before every paid run.

### The question wording, verbatim from the suite files

`noul`, PubMedQA. `instructions` is the question in the table above. `criteria`:

```json
{"true":  "Yes: the passages support answering the question yes.",
 "false": "No: the passages support answering the question no."}
```

`choice`, Banking77. `instructions` is the question in the table above. `criteria` is a
map of all 77 intent names to `null`: **the options carry no descriptions**, only their
names (`activate_my_card`, `age_limit`, ... `wrong_exchange_rate_for_cash_withdrawal`),
in Banking77's own label-id order, which is not alphabetical (`Refund_not_showing_up`
and `reverted_card_payment?` sit where their label ids put them).

`score`, HelpSteer2 helpfulness. `instructions` is the question in the table above.
`criteria` is a list in level order, levels `"0"` through `"4"`:

```
0  Not helpful at all: the response misses the essence of what the user wanted.
1  Borderline unhelpful: mostly misses what the user wanted, but is useful in a small way.
2  Partially helpful: misses the overall goal of the request in some way.
3  Mostly helpful: aligned with the request, with some room for improvement.
4  Extremely helpful: completely aligned with what the prompt asked for.
```

So a `score` question's levels are defined by that ordered list of strings, one per
level, and the level names are the stringified indices `"0"`..`"4"`. The label is
HelpSteer2's own `helpfulness` column, which is already 0..4.

### Probability vectors

Every answer becomes a probability vector over the task's options before scoring.

- Jev's values are rounded to 2 decimals on the wire; vectors that sum to 0.99 are
  renormalized.
- An answer is **malformed** if it is not a JSON object with a probability map, names an
  unknown or duplicate option (names must match exactly, including case and
  punctuation), has a value that is not a finite number in [0, 1], or sums to 0.
- If the listed values sum to more than 1 they are divided by their sum. If they sum to
  less than 1, the remainder is spread evenly over unlisted options (top-5 mode) or the
  vector is divided by its sum (full mode).
- The pick is the most probable option. For `choice`, ties go to the option the system
  listed first (for Jev, its own `choice` field); for `score`, ties go to the lower
  level. A yes/no answer of exactly 0.5 has **no pick and counts as wrong**.

### Decision Score

```
Decision Score = 100 * (1 - L_system / L_prior)
```

`L` is the mean per-item loss. Each item's loss is the average over its 5 repeats of

- the multiclass Brier score, `sum_k (p_k - y_k)^2`, for `choice` and `noul`;
- the ranked probability score over cumulative level probabilities,
  `sum_{k<K} (P_k - Y_k)^2 / (K - 1)`, for `score`.

`L_prior` is the same loss for the **label prior**, the baseline that answers every item
with the base rates of the evaluated items. It defines 0 on the scale.

**100 = perfect. 0 = no better than answering with the label base rates. Below 0 = worse
than that.** Negative scores are shown, not clamped; the chart floors at −10 and marks
rows below it. Both losses are proper scoring rules, so they reward being right and
being honest about uncertainty together and cannot be gamed by overconfidence. A system
with no probabilities (one-hot answers) is scored as one-hot. With more than one task in
a tab the tab score is the plain mean of the task scores.

### Accuracy

The share of decisions whose pick equals the label, **over all items and repeats**.
`score` uses the most probable level (exact match). Refused and malformed answers count
as wrong.

### Calibration gap (ECE)

Expected calibration error on the **top label**: confidence is the probability of the
pick, and for yes/no the larger of P(yes) and P(no). Decisions go into **10 equal-width
bins** by

```
min(9, floor(round(100 * c) / 10))
```

so a confidence of exactly 1.00 lands in the last bin. Then

```
ECE = sum_b (n_b / N) * |accuracy_b - mean confidence_b|
```

shown **in points** (5.8 means 0.058). Lower is better. Rows without probabilities show
a dash. ECE never sets the Decision Score rank, because calibration comparisons only
mean something at similar accuracy.

### Confidence gate and hand-off at 95 percent

The **gate** `t` for a primitive is the smallest confidence on the **0.01 grid** at
which the pooled error of all non-baseline decisions with confidence >= `t` is at most
5%, with **at least 100** such decisions. If no such `t` exists the gate is empty. It is
computed once per suite version from the first release and then **frozen**, pooled
across every listed system, and each row shows its coverage (share of its decisions at
or above `t`) and its accuracy on those decisions. Rows without probabilities (one-hot
answers) have no coverage.

**Current frozen gates, suite 0.1.0: `choice` 0.96 · `score` none · `noul` 0.91.**

**Hand-off at 95 percent** gives each system its own threshold instead: the **lowest**
confidence `t` on the 0.01 grid at which its decisions with confidence >= `t` (at least
100 of them) are **at least 95% correct**. Its hand-off share is those decisions over
**all** its decisions, refused and malformed ones included. The threshold is chosen on
the same items it is measured on, so the share is optimistic in the same way for every
system. A row whose accuracy never reaches 95% at any threshold shows a dash.

### Flip rates

**Repeat flip rate**: share of items whose pick differs between repeats 0 and 1, which
are identical requests. It measures nondeterminism. **Order flip rate** (`choice` only):
share of items whose pick is not the same across the four distinct option orders
(repeats 0, 2, 3, 4). Refused and malformed answers are excluded from both.

### Option order across the repeats

Every system gets one question per request (batch size 1), zero-shot, with identical
instructions and criteria. **Choice options are presented in a seeded random order that
is identical for every system: repeats 0 and 1 share one order, repeats 2, 3 and 4 each
get a new one. Score levels and yes/no are never reordered.**

### Cost, latency, failures

- **Cost**: `$ per 1k decisions` is the total cost of all calls, malformed-output
  retries and reasoning tokens included, divided by the number of decisions, times
  1,000. Always logged token usage times the list price snapshot in the run header. The
  one discarded warm-up call at the start of a run is charged to it. Jev's posted rate
  is $0.042 per million input tokens, output free.
- **Latency**: p95 is the 95th percentile, **nearest-rank**, of end-to-end time from
  sending the request to having a parsed, validated answer, over all answered decisions.
  It includes malformed-output retries but not transport-error backoff. Requests run at
  **concurrency 4** after one discarded warm-up call, all from one machine on a
  residential connection.
- Malformed output is retried up to **2 times**, and the retries count in cost and
  latency. A refusal (a provider refusal field or a content-filter stop) is not retried.
  After retries, a refused or malformed answer is **scored as the uniform distribution
  and a wrong pick**: it counts in the Decision Score and accuracy and is excluded from
  ECE, flip rates and the gate.
- **Transport failures** (HTTP errors, provider errors, a 60 s timeout, a truncated
  response) are retried with backoff and **never scored**. A decision that still fails
  is not written and the run resumes it later; a board cannot be built while any
  (item, repeat) is missing.

### 95 percent ranges and ranks

Intervals are 95% percentile intervals from an item-cluster bootstrap: 2,000 seeded
resamples of items with replacement, all repeats of an item moving together, the prior's
loss recomputed on each resample, the same resamples shared by every row in a tab so
differences are paired. `rank = 1 + the number of rows that are significantly better`,
where row j beats row i when the 95% interval of `DS_j - DS_i` over the shared resamples
lies wholly above 0. Rows that cannot be told apart share a rank, and the pairwise tests
are not adjusted for multiple comparisons.

**Board order**: a model wins a column when it is in that column's top two, ties
included (the two highest Decision Scores, with none when the label prior ties for
first; the two highest accuracies; the two lowest calibration gaps; the two lowest
prices; the two lowest p95 times). Boards list models by number of wins, then by
Decision Score, and models with as many wins share a place. The label prior has no wins
and no place.

### Versions

Every board is a release with a permanent URL under `/r/<release>/`. A patch changes no
score. A minor version regrades stored logs with no new calls. Adding, removing or
re-sampling a task is a **new suite version whose scores are not comparable with the
previous one**. Launch suite is 0.1.0; 1.0.0 is reserved for the full suite.

### The LLM adapter (not what this bench uses)

Jevals scores LLMs through one adapter for every model family: prompt-only JSON (no
structured-output mode), one pinned host per model with fallbacks disabled, provider
default temperature, the lowest reasoning setting the host allows, and a **verbalized**
probability distribution. Prompt hash `0383a0e3e592`; the published template is

```
You are answering one typed decision question about a state.

STATE (JSON):
{state}

QUESTION: {instructions}

{options_heading}
{options}

Give a probability for {what}
Reply with only this JSON object and nothing else:
{"probabilities": {"<option>": <probability between 0 and 1>}}
```

For questions with more than 10 options the adapter asks for the 5 most likely options
with probabilities and spreads the remaining mass evenly over the unlisted ones; for 10
or fewer it asks for every option. In adapter prompt v1 the yes/no options read
`- yes: Yes: the passages support...`.

### Listing rules (jevals.com/policy)

- Every row is run by the maintainer with the published harness. No vendor submissions,
  no private variants: **one configuration per model version**.
- Each lab's newest models are preferred. **A system is listed on a tab only when it has
  a complete run of every task in that tab at 5 repeats.**
- Every run that is started is disclosed, stopped runs included, with the reason in the
  changelog.
- The model string each provider served is recorded per row; if an alias starts pointing
  at a different model the row is rerun in a new release.
- Baselines that any system can lose to are always listed.
- Rows that cannot be told apart share a rank; "first place" is only claimed when the
  95% interval of the difference excludes zero.
- The maintainer pays list price for every call. No sponsorship, credits, discounts or
  pre-release access.

### Known limits Jevals states about its own suite

- Every item comes from a public dataset that is probably in LLM pretraining data.
  There is no private held-out set in 0.1.0.
- One dataset per primitive, so each tab is one task; results may not transfer, and
  calibration in particular is known to be task-dependent.
- LLM probabilities are **verbalized** and cluster on round values. **Logprob-based rows
  are not in this version.**
- Latency is measured from one residential location; Jev is served from one US region.
- Labels are taken as published, and public datasets have label noise.

## JevBench, the maintained multi-system board

Read from <https://raw.githubusercontent.com/fstandhartinger/jevbench/main/README.md> on
**2026-09-21**. Harness and public items: <https://github.com/fstandhartinger/jevbench>
(MIT for the harness and the original public decisions). Leaderboard:
<https://benchmarkheaven.com/jev-models>, JevBench v1.2.10, 534 frozen decisions per
system, 231 of them public, 38 systems listed. Maintainer: Florian Standhartinger, who
adds entrants per release.

jevals.com and JevBench are two different recipes, so this bench names which one every
figure follows (`recipe` on each metrics block, `recipe_of_record` on each set) and uses
JevBench's own names wherever the two boards measure the same thing.

**Its score**, verbatim from its README: JevBench Score = chance-corrected Intelligence,
Calibration, Speed and Cost at 25 percent each, geometric mean; below 50 Intelligence the
score is multiplied by `(Intelligence / 50)^2`. Intelligence is
`(accuracy - chance) / (1 - chance)` per tier, clipped at 0, weighted hard 30 percent,
easy 14, standard 28, judge 28. Calibration is the hard tier only: ECE plus fidelity to
the exact gold distributions, and a label-only system scores 0 on it.

**Its metric names, which this bench adopts** (all from the "What gets measured" section):

| JevBench's name | What it is there | Where it is here |
|-----------------|------------------|------------------|
| `accuracy` | argmax over the exact label set | `accuracy` |
| `majority_class_accuracy` | the score of always answering the commonest label, "because some cohorts are skewed and an accuracy has to be read against its floor" | `majority_class_accuracy`, the same number Jevals' label prior gives on a set like these, and also carried as `prior_accuracy` |
| Brier | "the multi-class sum `sum_k (p_k - y_k)^2` over the exact label set" | exactly the recipe's `choice`/`noul` loss (`brier_loss`) |
| ECE | top-label confidence, 10 equal-width bins | `ece_points`, same definition. One difference: JevBench drops an empty bin, this keeps it with `count: 0` so the reliability table always has ten rows and a calibration curve can be drawn from the record |
| Ordinal MAE | for score questions, the probability-weighted level against the reference level, "reported beside argmax accuracy rather than instead of it" | `ordinal_mae`, same rule |
| `schema_validity` / `schema_validity_strict` | a distribution must cover the label set, sit in [0,1] and sum to 1; the headline renormalizes inside a 2 percent band and the strict column uses the frozen 0.001 tolerance | both, under those names |
| `native` / `verbalized` | the model's own distribution versus a model writing probabilities out under a schema, "labelled everywhere" | `probability_source`, with a third value `logprob` for `/v1/decide` (see below) |
| `derived_usage_times_tariff` | measured token usage times the provider's published tariff | `cost_basis` |

**Two of its rules this bench follows because they are better than a bare zero:**

- *"An unknown price is `null`, not `0`. An empty metric is `null` with `n = 0`, not a
  flattering `0.0`."* That is this repo's own rule already. A cost of $0 here carries
  `cost_basis: "no_billable_account_no_price_given"` so nobody reads it as free.
- *"A 401, 403 or 429 ends the run. The rest stays unattempted and is reported as
  unattempted, never as answers the model got wrong."* Same rule, same reason, already in
  `ainode/bench/auth.py`.

**Token-level logprobs.** JevBench states plainly: *"Token-level logprobs are not used
anywhere, for anyone."* Jevals 0.1.0 states *"Logprob-based rows are not in this
version."* AINode's `/v1/decide` reads the first generated token's logprobs, so it is
neither board's category, and its rows carry `probability_source: "logprob"`. That is
not a small footnote: it is generally a stronger instrument than a verbalized
distribution, and a row measured that way should be compared with other logprob rows and
with native ones, not treated as interchangeable with a board's LLM rows.

**Its `typesafe` adapter is our `systemone` transport.** JevBench's adapter table says
`typesafe` serves *"TypeSafe's `/v1/systemone`, and the open rebuilds that implement the
same wire format"*, and its own CLI drives it with `--adapter typesafe --endpoint <url>
--key-env ''`. So a server that answers `POST /v1/systemone` is directly runnable by
JevBench's harness as well as by this one, which is the practical reason the transport is
a flag here rather than a fork.

**Training on its public split is allowed and must be declared.** Its listing rule says
so, which is the same class of fact as the contamination flags below.

## The fourth set: LocalLLaMA/typed-decisions

Card read from
<https://huggingface.co/datasets/LocalLLaMA/typed-decisions/raw/main/README.md> on
**2026-09-21**; rows fetched at revision `ea9306458d6e9563628369a3d1e72e362fb381d2`.
Apache-2.0. It is here because it is the one public set already in the `/v1/systemone`
request shape, with a documented train/test split and gold DISTRIBUTIONS rather than only
labels: the card says *"`state` and `questions` together are exactly the body of a
`POST /v1/systemone` request. You can replay a row without reshaping it."*

- **Shape**: four workflows (`agent_trace_observability`, `customer_service`,
  `invoice_processing`, `security_incidents`), 100 test cases each, **five typed questions
  over one shared state** per case, so 400 cases and 2,000 decisions. Mixed primitive: 600
  `choice`, 600 `noul`, 800 `score`. Two of its twenty questions carry no criteria at all.
- **Gold** is the mean of three teacher samples at temperature 0.7, so *"a score measures
  agreement with that teacher. It does not measure correctness."* The card's reference
  points on the 1,600-case set: **majority baseline 0.520, perfect scenario understanding
  0.704, teacher self-agreement 0.735**, and it says to read 0.52 as the floor, 0.70 as
  strong and 0.75 as saturation.
- **The card's own test-split rows**: Uniform 0.308, Prior 0.470, MiniLM-L6 (22M,
  specialist) 0.587, ModernBERT-base (149M, specialist) 0.646, **TypeSafe Jev 1.13.0
  (general) 0.727** with ECE 0.144, KL 1.442, TV 0.251, Brier 0.148, 710 ms per case and
  $0.016 total, measured 2026-09-18 through `POST /v1/systemone`.
- **Specialist and generalist are not comparable**, and the card requires a row to say
  which it is: a specialist is fitted per workflow with the label spaces fixed at training
  time; a generalist answers arbitrary schemas zero-shot. **A run from this bench is a
  generalist run**, and the record says so. The card's two learned rows are specialists
  fitted on the `train` split of the very workflows they are scored on.
- **Laya**'s `laya-typed-decisions` checkpoint reports **0.766** on this test split
  against Jev's **0.727**
  (<https://raw.githubusercontent.com/NandhaKishorM/laya/main/BENCHMARKS.md>), and its own
  README states the two qualifications that go with that number: *"The 0.766 figure comes
  from the checkpoint fine-tuned on that benchmark's own training split"*, and the Jev
  figure is *"third-party published, never measured here"*. Its base checkpoints score
  0.362 and 0.342, below the 0.461 majority baseline it quotes. So 0.766 is a specialist
  number on the set's own training distribution and 0.727 is a generalist one; they are
  the card's two modes, not a ranking.
- **What this bench computes on it**: the Jevals formulas (Decision Score against the
  label prior, ECE, hand-off, the flips), broken down **per answer space** (one
  `(type, options)` pair, named `<workflow>/<question>`, twenty of them here) with a
  per-primitive roll-up, PLUS a `vs_gold` block against the gold distributions, which
  is what the card asks for: *"Score against the full distributions, not just the argmax.
  Calibration is the point."* The breakdown is not presentation: `action` is a four-option
  question in one workflow and a five-option one in another, so a single pooled label prior
  would be a baseline over an answer space neither of them has. The card independently says
  the same thing: *"Per-question ceilings vary a lot... Read every score against its own
  question, not against the mean."*
- **Our label prior is not the card's Prior row.** Ours is fitted on the evaluated items,
  which is the Jevals rule ("the base rates of the evaluated items"); the card's Prior row
  fits each question's label frequencies on the `train` split. Scored on the test split the
  two are close but not equal: the card reports Prior at 0.470 accuracy, and our own prior
  measured over the committed manifest scores 0.435 pooled over decisions with a
  space-averaged floor of 0.474. Neither is wrong; they are two baselines and a record
  should not be read as quoting the card's.
- **`vs_gold` is ours, not theirs.** `soft_accuracy` is the gold probability of the pick,
  `total_variation` is `0.5*sum|p-g|`, `kl` is `sum g*log(g/p)` with `p` floored, and
  `brier_vs_gold` is `sum (p-g)^2`. The card prints columns called Soft acc, TV, KL and
  Brier without publishing the arithmetic behind each, so our numbers are not its numbers
  and the block says so in its own `definition` field.
- **One question per request.** The card's Jev row was measured per CASE (five questions
  in one call, 710 ms per case), and this bench sends one question per request because
  that is the Jevals recipe's rule and because packing questions into one prompt is
  documented elsewhere in this field to change answers. So our latency and cost on this
  set are not comparable to the card's.
- **State hashes on this set are ours**, computed at the pinned revision on 2026-09-21 and
  recorded as `state_hash_source: "ainode"`, because the dataset publishes none. They pin
  the bytes exactly as the Jevals ones do.

## Contamination, and why it is in the record

A set inside a listed system's published training data measures memorisation for that
system. `ainode/bench/decide/sets.py::CONTAMINATION` carries the finding, the evidence and
the primary source, and a run's notes repeat it in words. Only training exposure a
project's own card or README states goes in there.

- **Banking77 is in Kev's training data.** The `kev-9b` model card's front matter lists
  `legacy-datasets/banking77` under `datasets:`
  (<https://raw.githubusercontent.com/jaredpalmer/kev/main/docs/model-cards/kev-9b.md>),
  and its README describes `decision-v7` as *"10,000 examples from ten public datasets"*
  used for both training and development. A Kev row on the `choice` board is therefore
  contaminated.
- **Banking77 is also in Laya's training mix**, whose write-up lists "banking intents"
  among the training groups while Banking77 is simultaneously in its published eval list.
- **PubMedQA sits inside decider's development loop**: it is named among the held-out
  datasets of that project's 94-task regression set
  (<https://raw.githubusercontent.com/Mapika/decider/main/README.md>), which means it was
  read repeatedly during development even though the card calls it held out.
- None of this says anything about an AINode-served model that did not train on these
  sets. It says which OTHER rows on the same board are not like-for-like.

## What we verified rather than assumed

The suite files publish `state_sha256` per item, which pins the exact bytes each system
was shown. The loader in `ainode/bench/decide/sets.py` rebuilds every state from the
canonical Hugging Face source and **checks that hash**, and a mismatch is a load error.
Reproducing the hash is also how the state construction below was pinned, because the
suite files name the upstream field but not the key the state object uses:

| Set | State object, `json.dumps(..., separators=(",",":"), ensure_ascii=False)` | Verified |
|-----|--------------------------------------------------------------------------|----------|
| `banking77` | `{"message": <row.text>}` | 300 of 300 |
| `helpsteer2` | `{"prompt": <row.prompt>, "response": <row.response>}` | 300 of 300 |
| `pubmedqa` | `{"question": <row.question>, "context": <row.context.contexts>}` | 300 of 300 |

Note that the Banking77 state key is `message`, not the upstream field name `text`, and
the PubMedQA key is `context` holding the **list** of passages, not `context.contexts`.
Neither is stated on jevals.com; both were recovered by matching the published hashes,
and all 900 match, so the states this bench scores are byte-identical to the ones the
boards scored.

The labels were checked the same way: `banking77.target` equals the row's `label` and
`options[target]` equals its `label_text` for all 300; `helpsteer2.target` equals the
row's `helpfulness` for all 300; `pubmedqa.target` equals
`{"no": 0, "yes": 1}[row.final_decision]` for all 300.

One trap worth recording: `mteb/banking77` carries both a `test.jsonl` (3,080 lines) and
the parquet behind its `default/test` config (3,076 rows). They agree at the start and
diverge later, and the suite's `row_idx` is the **parquet** index. Loading from
`test.jsonl` puts 258 of the 300 states on the wrong row, which the hash check catches.

## What we had to author

Marked so a reader knows which parts of a number here are not the board's.

1. **The seeded option order for `choice`.** Jevals says the order is seeded, identical
   for every system, shared by repeats 0 and 1, and new for each of repeats 2, 3 and 4.
   It does not publish the generator, so the orders cannot be reproduced. Ours is
   `random.Random(f"{seed}:{item_id}:{order_index}").shuffle(options)` over the suite's
   own seed, which satisfies every published property except being the same permutation
   as theirs. It is deterministic and identical across our own systems, so our order
   flip rate is a real order flip rate; it is not their order flip rate.
2. **The `score` response shape.** jevals.com documents the `score` primitive but not
   the field a Jev-format server answers it in. The `systemone` parser accepts `score`,
   `level` or `choice` for the pick and `probabilities` or `distribution` for the
   vector, and a response matching none of those is one malformed row.
3. **The adapter-prompt fills.** The template above is published with
   `{options_heading}`, `{options}` and `{what}` unfilled, and the hash is over the
   filled version, so it cannot be reconstructed. **This bench does not implement the
   adapter.** Its two transports are `/v1/decide` and `/v1/systemone`, both of which
   take a typed question natively, so there is no prompt of Jevals' to match.
4. **Which question a `/v1/decide` request carries the criteria in.** `/v1/decide`'s
   `choice` question takes bare option names with no per-option rubric, so the criteria
   are appended to the question text, the same thing the legacy 110-item path already
   does. For Banking77 that appends nothing at all, because its criteria are all null.
5. **The run-local gate.** The board's gate is pooled across every listed system and
   frozen from the first release, which one run cannot recompute. This bench reports the
   published frozen gate for the primitive, this run's coverage and accuracy at it, and
   separately a `gate_local` computed by the published rule over this run's own
   decisions alone. `gate_local` is not a board number.

## Deviations from the recipe, on purpose

1. **No bootstrap intervals or ranks.** A rank is a property of a board with several
   rows on shared resamples, and this bench measures one system at a time. The Decision
   Score is reported as a point value with no interval, which means two AINode numbers
   that are close should not be read as one beating the other.
2. **`/v1/decide` is logprob-based, which the suite excludes.** AINode's decision path
   constrains the engine to a single option label and reads the distribution from the
   first token's logprobs. Jevals 0.1.0 states that logprob-based rows are not in that
   version and that its LLM rows are verbalized. So an AINode row is measured on the
   same items, the same labels and the same formulas, through a **different and
   generally stronger instrument** than the board's LLM rows. Compare it with Jev and
   with other typed-decision servers freely; compare it with the board's LLM rows
   knowing that difference.
3. **`/v1/decide` letters its options, and past 26 options a two-letter label can share
   its mass with a one-letter one** (see `ainode/api/decide.py`). Banking77 has 77
   options, so on the `choice` set a pair like `A` and `AA` may report the same
   probability on a tokenizer that does not give `AA` its own token. That is a property
   of the instrument on that one set, it is recorded in the run's notes, and it is a
   reason to read the `choice` Decision Score as a floor rather than a point.
4. **No malformed retries.** Jevals retries malformed output up to twice. This bench
   scores the first answer, counts malformed answers, and records the count next to
   every other number; retrying would make the malformed count a statement about the
   retry policy instead of about the model. A transport failure is still one row with an
   `error` and is never scored, which is the recipe's rule and the repo's.
5. **Concurrency is a flag, not 4.** Latency here is measured at whatever `--concurrency`
   the run used, and the record says which. A p95 from a run at concurrency 8 on a busy
   GPU node is not comparable to a board p95, and the run's own number is the one a
   reader of ours needs.
6. **No warm-up call is discarded.** Every request is in the numbers, which on a cold
   engine makes the first one visible in the p95. The record says how many decisions it
   covers, so nothing is hidden.

## Citation

> Jevals (jevals.com), suite 0.1.0. Methodology read 2026-09-21. Suite files CC-BY-4.0.

Upstream datasets keep their own licences: PubMedQA MIT, Banking77 CC-BY-4.0 (PolyAI),
HelpSteer2 CC-BY-4.0 (NVIDIA).
