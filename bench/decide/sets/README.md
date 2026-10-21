# The public question sets

Four manifests. Each one says which upstream dataset, which split, which pinned revision,
which licence, and for every question an id, an upstream row index, the label and a
`state_sha256`. **None of them carries any dataset item text.**

| File | Primitive | Questions | Upstream | Licence | Hashes |
|------|-----------|-----------|----------|---------|--------|
| `pubmedqa.json` | `noul` | 300 | `qiaojin/PubMedQA` `pqa_labeled/train` @ `9001f285` | MIT | Jevals' |
| `banking77.json` | `choice`, K=77 | 300 | `mteb/banking77` `default/test` @ `18072d26` | CC-BY-4.0 (PolyAI) | Jevals' |
| `helpsteer2.json` | `score`, K=5 | 300 | `nvidia/HelpSteer2` `default/validation` @ `990b2711` | CC-BY-4.0 (NVIDIA) | Jevals' |
| `typed-decisions.json` | mixed, 5 per case | 2000 (400 cases) | `LocalLLaMA/typed-decisions` `all/test` @ `ea930645` | Apache-2.0 | ours |

The first three are Jevals' own suite files for suite 0.1.0, committed **verbatim** as
downloaded from `https://jevals.com/data/suites/0.1.0/<id>.json` on 2026-09-21. They are
CC-BY-4.0: cite *Jevals (jevals.com), suite 0.1.0*. The fourth is built by AINode from the
dataset at the pinned revision, because that dataset publishes no suite file of its own; it
carries the labels, the gold distributions and our own state hashes, and says so in its
`state_hash_source` and `notes`.

## Why the item text is not here

Jevals does not republish item text either: every item links to its upstream row, and the
upstream licences are the item text's licence and not ours to relicense. Following that
also keeps 300 biomedical abstracts, 300 chat transcripts and 400 synthetic workflow
states out of the repo. Fetch them instead, once:

```bash
python3 scripts/ainode-bench.py decide download              # all four
python3 scripts/ainode-bench.py decide download banking77    # one
```

That writes `bench/decide/cache/<id>.questions.json`, which is gitignored, and **verifies
every state against the hash in the manifest before writing anything**. A mismatch is a
load error naming the question: it means the upstream row moved, and a number measured
against moved bytes is not comparable to the board it sits next to.

## The labels are not in the questions

A question file's entries hold an id, a state and a hash. The labels live in one separate
`labels` map keyed by question id, and a question object carrying anything
answer-key-shaped (`label`, `expected`, `passingAnswer`, `gold_label`, ...) is a load
error. The transport is handed a question, so a label it never sees is a label it cannot
put in the prompt, and the rule is checked from both ends:
`sets.answer_key_leaks` on the file and `suite.wire_leaks` on the assembled request body,
which refuses to send rather than scoring a leak.

A gold DISTRIBUTION is allowed in a question and is not an answer key, but it may never
reach the wire either, and the guard enforces that at any depth.

## Contamination

`banking77` is in the published training data of Kev and of Laya, and `pubmedqa` sits
inside decider's development loop. Every source is named in `bench/decide/JEVALS.md` and in
`ainode/bench/decide/sets.py::CONTAMINATION`, and a run's record and notes repeat it. It
says nothing about an AINode-served model that did not train on them; it says which other
rows on the same board are not like-for-like.

## Adding a set

A set is one measurement, so a new manifest is a new file here plus an entry in
`SOURCES` (how to fetch the split, how to build one state from a row) and in
`RECIPE_OF_RECORD` (which published recipe that set's own third-party numbers follow). A
single-primitive set states one `instructions`, one `options` list and one `criteria`; a
mixed one states `question_schemas` per group and each item names its `group`. Loading is
strict in both shapes: a missing row, a moved hash or a target outside the option list is a
load error, never a skipped question.

The recipe, every deviation from it and every fact quoted above are in
`bench/decide/JEVALS.md` with the URL and the date read.
