"""The three public Jevals sets: the committed manifests, the download, the questions.

``bench/decide/sets/<id>.json`` is Jevals' own suite file, committed verbatim
(CC-BY-4.0, cited in ``bench/decide/JEVALS.md``). It names the dataset, the config, the
split, the pinned revision, the licence, the seed, the instructions, the criteria, the
option list, and for each of the 300 items an id, an upstream row index, the label and a
``state_sha256``. What it does NOT carry is any dataset item text: Jevals does not
republish item text, every item links to its upstream row, and the three upstream
licences are the item text's licence and not ours to relicense. This bench follows that
exactly, which also keeps three sets of 300 long biomedical abstracts and chat
transcripts out of the repo.

So the item text is **downloaded, never committed**:

    python3 scripts/ainode-bench.py decide download            # all three
    python3 scripts/ainode-bench.py decide download pubmedqa   # one

writes ``bench/decide/cache/<id>.questions.json`` (gitignored) and nothing else. That
file is a question file: the Jev question shape (``type``, ``instructions``,
``criteria``) plus one entry per question holding an id, a state and the published hash,
which is the same shape ``--questions <file>`` takes, so a private blind set can be
scored by the same code without ever being committed.

**A label is never inside a question.** The file's labels live in one separate
``labels`` map, keyed by question id, and a question object carrying anything
answer-key-shaped is a load error. That is not tidiness: the transport is handed a
question and builds the wire body out of it, so a label it never sees is a label it
cannot leak into the prompt, and the rule is checked from both ends
(:func:`answer_key_leaks` here, ``suite.wire_leaks`` on the assembled body).

**Every state is checked against the published ``state_sha256`` before it is written**,
and a mismatch is a load error naming the item. That check is what makes a number here
comparable to a board number: it proves the bytes this bench put in front of the model
are the bytes Jevals put in front of theirs, which is a stronger guarantee than pinning
a dataset revision, and it is how the state construction for all three sets was
recovered in the first place (see JEVALS.md, "What we verified rather than assumed").

Stdlib only: ``urllib`` for the transport, ``gzip`` for the one set whose upstream file
is compressed, ``hashlib`` for the check.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request

#: The sets this bench knows, in the order a run lists them. The first three are the
#: Jevals boards' one-task-per-primitive sets; the fourth is the one public set already
#: in the ``/v1/systemone`` request shape that ships gold DISTRIBUTIONS rather than only
#: labels, and it is mixed-primitive (five questions over one shared state).
SUITES = ("pubmedqa", "banking77", "helpsteer2", "typed-decisions")

#: Which published recipe a set's own third-party numbers follow, so a record can say
#: which recipe each figure is comparable to. This bench computes the Jevals formulas on
#: all four sets; a set's own card may print differently named columns, and naming the
#: recipe of record is how those two are kept apart.
RECIPE_OF_RECORD = {"pubmedqa": "jevals-0.1.0", "banking77": "jevals-0.1.0",
                    "helpsteer2": "jevals-0.1.0",
                    "typed-decisions": "typed-decisions-card"}

#: Sets a listed system is known to have TRAINED on, with the primary source that says
#: so, so a record flags its own contamination rather than leaving a reader to find out.
#: A row for one of these systems on that set measures memorisation and not decision
#: quality. Only training exposure a project's own card or README states goes in here; a
#: guess from a name does not.
CONTAMINATION = {
    "banking77": [
        {"system": "Kev (jaredpalmer/kev)",
         "evidence": "the kev-9b model card front matter lists "
                     "legacy-datasets/banking77 under `datasets:`, and the README's "
                     "decision-v7 recipe is 10,000 examples from ten public datasets",
         "source": "https://raw.githubusercontent.com/jaredpalmer/kev/main/"
                   "docs/model-cards/kev-9b.md"},
        {"system": "Laya (convaiinnovations/laya)",
         "evidence": "its write-up lists banking intents among the training groups, and "
                     "Banking77 is also in its own published eval list",
         "source": "https://raw.githubusercontent.com/NandhaKishorM/laya/main/"
                   "BENCHMARKS.md"},
    ],
    "pubmedqa": [
        {"system": "decider (Mapika/decider)",
         "evidence": "PubMedQA is named among the held-out datasets of its 94-task "
                     "regression set, so it sits inside that project's development loop "
                     "even though the card calls it held out",
         "source": "https://raw.githubusercontent.com/Mapika/decider/main/README.md"},
    ],
}

ENV_SETS = "AINODE_DECIDE_SETS"
ENV_CACHE = "AINODE_DECIDE_CACHE"

_REPO = pathlib.Path(__file__).resolve().parents[3]
_REPO_SETS = _REPO / "bench" / "decide" / "sets"
_REPO_CACHE = _REPO / "bench" / "decide" / "cache"
_HOME_SETS = pathlib.Path.home() / ".ainode" / "bench" / "decide" / "sets"
_HOME_CACHE = pathlib.Path.home() / ".ainode" / "bench" / "decide" / "cache"

#: Hugging Face's rows API, which answers JSON and needs no pip install. It does not
#: take a revision, which is exactly why every row is hash-checked: a drifted row fails
#: the check and is a load error, where a revision pin would only have been a promise.
ROWS_API = "https://datasets-server.huggingface.co/rows"
#: Its page ceiling.
ROWS_PAGE = 100
#: ``https://huggingface.co/datasets/<ds>/resolve/<revision>/<path>``, which DOES pin a
#: revision and is used for the one set whose upstream file is stdlib-readable.
RESOLVE = "https://huggingface.co/datasets/{dataset}/resolve/{revision}/{path}"

HTTP_TIMEOUT = 120
HTTP_RETRIES = 6
HTTP_BACKOFF = 3.0
#: Ceiling on one backoff wait, including one the server asked for.
HTTP_BACKOFF_MAX = 60.0
#: Courtesy pause between pages of a public API nobody is paying us to hammer. The rows
#: API rate limits an unauthenticated caller partway through a 31-page fetch at a quarter
#: of a second, so this is deliberately slower than it needs to be.
PAGE_PAUSE = 1.0


class SetError(RuntimeError):
    """A manifest, a download or a question file that cannot be used as asked."""


# ------------------------------------------------------------------ the state bytes

def state_json(state) -> str:
    """The state as the bytes its hash is over: compact, insertion order, real UTF-8.

    ``JSON.stringify`` with no arguments, which is what the recipe says the hash is
    taken of: no spaces, no key sorting (insertion order is the order the state fields
    are listed in), and non-ASCII written as itself rather than as an escape. All three
    differ from ``json.dumps``' defaults, and getting any one of them wrong moves every
    hash.
    """
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def state_sha256(state) -> str:
    return hashlib.sha256(state_json(state).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ paths

def sets_dir() -> pathlib.Path:
    """``$AINODE_DECIDE_SETS``, else the repo's manifests, else the installed copy."""
    override = os.environ.get(ENV_SETS)
    if override:
        return pathlib.Path(override).expanduser()
    if _REPO_SETS.is_dir():
        return _REPO_SETS
    return _HOME_SETS


def cache_dir() -> pathlib.Path:
    """``$AINODE_DECIDE_CACHE``, else ``bench/decide/cache`` beside the manifests."""
    override = os.environ.get(ENV_CACHE)
    if override:
        return pathlib.Path(override).expanduser()
    if _REPO_SETS.is_dir():
        return _REPO_CACHE
    return _HOME_CACHE


def manifest_path(suite_id: str) -> pathlib.Path:
    return sets_dir() / f"{suite_id}.json"


def questions_path(suite_id: str) -> pathlib.Path:
    return cache_dir() / f"{suite_id}.questions.json"


# ------------------------------------------------------------------ the manifests

def load_manifest(suite_id: str) -> dict:
    """One committed suite file, validated down to the fields the loader relies on.

    Two manifest shapes, one loader. A **single-primitive** manifest (the three Jevals
    sets) states one ``instructions``, one ``options`` list and one ``criteria`` for the
    whole set. A **mixed** one (``typed-decisions``) states ``question_schemas`` per
    workflow instead, because it asks five differently typed questions over one state and
    a set-level option list would be meaningless.
    """
    path = manifest_path(suite_id)
    if not path.is_file():
        raise SetError(f"no suite manifest at {path}; known suites: "
                       f"{', '.join(SUITES)}")
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SetError(f"{path} is not valid JSON: {exc}") from exc
    for key in ("id", "version", "primitive", "dataset", "config", "split",
                "hf_revision", "license", "items", "state_fields", "n_items"):
        if key not in doc:
            raise SetError(f"{path} is missing '{key}'")
    if doc["id"] != suite_id:
        raise SetError(f"{path} calls itself {doc['id']!r}, not {suite_id!r}")
    schemas = doc.get("question_schemas")
    if schemas is None:
        for key in ("instructions", "options"):
            if key not in doc:
                raise SetError(f"{path} states no 'question_schemas' and is missing "
                               f"'{key}'")
        options = doc["options"]
        if not isinstance(options, list) or len(options) < 2:
            raise SetError(f"{path} declares fewer than two options")
    else:
        if not isinstance(schemas, dict) or not schemas:
            raise SetError(f"{path} has a 'question_schemas' that is not a map")
        for group, block in schemas.items():
            if not isinstance(block, dict) or not block:
                raise SetError(f"{path}: question_schemas[{group!r}] holds no questions")
            for qkey, spec in block.items():
                for key in ("type", "instructions", "options"):
                    if key not in spec:
                        raise SetError(f"{path}: {group}/{qkey} is missing '{key}'")
                if len(spec["options"]) < 2:
                    raise SetError(f"{path}: {group}/{qkey} declares fewer than two "
                                   "options")
    items = doc["items"]
    if not isinstance(items, list) or len(items) != int(doc["n_items"]):
        raise SetError(f"{path} says n_items={doc['n_items']} and holds "
                       f"{len(items) if isinstance(items, list) else 'none'}")
    for item in items:
        for key in ("item_id", "row_idx", "state_sha256", "target"):
            if key not in item:
                raise SetError(f"{path}: an item is missing '{key}'")
        width = len(manifest_spec(doc, item)["options"])
        if not 0 <= int(item["target"]) < width:
            raise SetError(f"{path}: item {item['item_id']} has target "
                           f"{item['target']}, outside its {width} options")
    return doc


def manifest_spec(doc: dict, item: dict) -> dict:
    """The typed question for one manifest item: the set's own, or its group's.

    A mixed manifest states its schemas per ``group`` (typed-decisions: per workflow) and
    each item names its own group and carries the question key after the colon in its
    ``item_id``, so one upstream row becomes as many items as it asks questions. The
    group is IN THE ITEM rather than derived from the row, because the same question key
    means different things in two groups (``action`` has four options in one workflow and
    five in another) and a manifest has to be readable without a download.
    """
    schemas = doc.get("question_schemas")
    if not schemas:
        return {"type": doc["primitive"], "instructions": doc["instructions"],
                "criteria": doc.get("criteria"), "options": list(doc["options"])}
    qkey = str(item["item_id"]).rsplit(":", 1)[-1]
    group = item.get("group")
    block = schemas.get(group)
    if block is None:
        raise SetError(f"{doc['id']}: item {item['item_id']} is in group {group!r}, "
                       f"which this manifest does not declare (declared: "
                       f"{', '.join(sorted(schemas))})")
    if qkey not in block:
        raise SetError(f"{doc['id']}: item {item['item_id']} names question {qkey!r}, "
                       f"which group {group!r} does not declare")
    return dict(block[qkey])


def criteria_for(doc: dict):
    """The criteria as the manifest states them: a map for choice and noul, a list for
    score. Passed through unchanged, because the wording IS the question."""
    return doc.get("criteria")


# ------------------------------------------------------------------ the transport

def _get(url: str, timeout: int = HTTP_TIMEOUT, sleep=time.sleep) -> bytes:
    """One GET with backoff. A public API's 429 or hiccup is not a load error yet.

    A 429 is the expected failure here, not an exception: the rows API rate limits an
    unauthenticated caller partway through a multi-page fetch, so it backs off (honouring
    ``Retry-After`` when the server sends one) and keeps going. Only a 4xx that says the
    request itself is wrong stops early, because retrying a 404 is just slower.
    """
    last = ""
    for attempt in range(HTTP_RETRIES):
        wait = HTTP_BACKOFF * (attempt + 1)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
            if exc.code in (400, 401, 403, 404):
                break
            if exc.code == 429:
                try:
                    asked = float(exc.headers.get("Retry-After") or 0)
                except (TypeError, ValueError):
                    asked = 0.0
                wait = max(wait, min(asked, HTTP_BACKOFF_MAX), HTTP_BACKOFF * 4)
        except Exception as exc:                                  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        if attempt + 1 < HTTP_RETRIES:
            sleep(min(wait, HTTP_BACKOFF_MAX))
    raise SetError(f"could not fetch {url}: {last}. A 429 here is the rows API rate "
                   "limiting an unauthenticated caller; wait a minute and run the "
                   "download again, it starts from the beginning of that set")


def fetch_rows(dataset: str, config: str, split: str, up_to: int,
               progress=None) -> dict:
    """``{row_idx: row}`` from the rows API, paged to cover every index up to ``up_to``.

    The whole prefix rather than only the wanted indices, because the API pages by
    offset and a page is one request either way: 31 requests for Banking77's 3,076 rows,
    10 for PubMedQA's 1,000.
    """
    out = {}
    offset = 0
    while offset <= up_to:
        query = urllib.parse.urlencode({"dataset": dataset, "config": config,
                                        "split": split, "offset": offset,
                                        "length": ROWS_PAGE})
        page = json.loads(_get(f"{ROWS_API}?{query}").decode("utf-8"))
        rows = page.get("rows") or []
        if not rows:
            break
        for row in rows:
            out[int(row["row_idx"])] = row["row"]
        offset += ROWS_PAGE
        if progress:
            progress(min(offset, up_to + 1), up_to + 1)
        if offset <= up_to:
            time.sleep(PAGE_PAUSE)
    return out


def fetch_jsonl(dataset: str, revision: str, path: str, progress=None) -> dict:
    """``{row_idx: row}`` from a JSONL (or gzipped JSONL) file at a PINNED revision.

    The row index is the line number, which is what a split's own file ordering means
    and what the manifest's ``row_idx`` counts.
    """
    raw = _get(RESOLVE.format(dataset=urllib.parse.quote(dataset),
                              revision=revision, path=path))
    if path.endswith(".gz"):
        raw = gzip.decompress(raw)
    out = {}
    for index, line in enumerate(raw.decode("utf-8").splitlines()):
        if line.strip():
            out[index] = json.loads(line)
    if progress:
        progress(len(out), len(out))
    return out


# ------------------------------------------------------------------ the three sets

def _pubmedqa_state(row) -> dict:
    """``{"question": ..., "context": [passages]}``.

    The manifest's state fields are ``question`` and ``context.contexts``, but the state
    object's second key is ``context`` holding the list itself. That is not published;
    it was recovered from the hashes and matches all 300.
    """
    return {"question": row["question"], "context": list(row["context"]["contexts"])}


def _banking77_state(row) -> dict:
    """``{"message": ...}``. The upstream field is ``text``; the state key is
    ``message``. Also recovered from the hashes, and it matches all 300."""
    return {"message": row["text"]}


def _helpsteer2_state(row) -> dict:
    return {"prompt": row["prompt"], "response": row["response"]}


def _typed_decisions_state(row):
    """The state ships as a JSON STRING and is already the request body's own state.

    Parsed to the object every transport here works with; the hash is over our canonical
    re-serialization of that object, so it is consistent with the other three sets. The
    manifest says the hashes are ours and not upstream's, because this dataset publishes
    none of its own.
    """
    state = row["state"]
    return json.loads(state) if isinstance(state, str) else state


#: Per suite: how to fetch the split, and how to build one state out of a row.
SOURCES = {
    "pubmedqa": {"fetch": "rows", "state": _pubmedqa_state},
    "banking77": {"fetch": "rows", "state": _banking77_state},
    "helpsteer2": {"fetch": "jsonl", "path": "validation.jsonl.gz",
                   "state": _helpsteer2_state},
    "typed-decisions": {"fetch": "rows", "state": _typed_decisions_state},
}


def build_questions(doc: dict, rows: dict) -> dict:
    """The question file for one suite: the Jev shape plus the label and the hash.

    Strict, the way the rest of this bench loads data: a missing row, a state whose hash
    does not match the published one, or a label that does not line up is an error
    naming the item, never a skipped item. A set that quietly ran 297 of 300 would
    publish an accuracy against a count nobody chose, and a set whose states drifted
    would publish a Decision Score that is not comparable to the board it is next to.
    """
    suite_id = doc["id"]
    build_state = SOURCES[suite_id]["state"]
    mixed = bool(doc.get("question_schemas"))
    questions = []
    labels = {}
    states = {}
    for item in doc["items"]:
        index = int(item["row_idx"])
        row = rows.get(index)
        if row is None:
            raise SetError(f"{suite_id}: row {index} for item {item['item_id']} is "
                           "not in the download")
        if index not in states:
            try:
                states[index] = build_state(row)
            except (KeyError, TypeError, ValueError) as exc:
                raise SetError(
                    f"{suite_id}: row {index} has no {exc} field; the upstream split is "
                    "not the one this manifest was built from") from exc
        state = states[index]
        digest = state_sha256(state)
        if digest != item["state_sha256"]:
            raise SetError(
                f"{suite_id}: item {item['item_id']} (row {index}) hashes to "
                f"{digest[:16]} and the manifest says {item['state_sha256'][:16]}. The "
                "upstream row has changed, or this is the wrong split file")
        spec = manifest_spec(doc, item)
        options = list(spec["options"])
        entry = {"id": item["item_id"], "row_idx": index, "state": state,
                 "state_sha256": digest}
        if mixed:
            # A mixed set's questions differ per entry, so each one carries its own
            # typed question. A single-primitive set says it once at the top instead.
            entry.update({"type": spec["type"], "instructions": spec["instructions"],
                          "criteria": spec.get("criteria"), "options": options})
            # Which ANSWER SPACE this question belongs to, which is the unit the
            # Decision Score is defined over: its label prior is the base rates of the
            # labels in THIS option list. `action` is a four-option question in one
            # workflow and a five-option one in another, so the group has to be in the
            # name or two different spaces would be pooled into one meaningless prior.
            qkey = str(item["item_id"]).rsplit(":", 1)[-1]
            entry["space"] = f"{item.get('group', doc['id'])}/{qkey}"
        if isinstance(item.get("gold"), dict) and item["gold"]:
            # A gold DISTRIBUTION, not a second label: it is what the answer's spread is
            # compared against, and it is never sent on the wire (see wire_leaks).
            entry["gold"] = {option: float(item["gold"].get(option, 0.0))
                             for option in options}
        questions.append(entry)
        labels[item["item_id"]] = options[int(item["target"])]
    built = {
        "id": suite_id,
        "suite_version": doc["version"],
        "set": suite_id,
        "recipe_of_record": doc.get("recipe_of_record")
                            or RECIPE_OF_RECORD.get(suite_id),
        "type": doc["primitive"],
        "instructions": doc.get("instructions"),
        "criteria": criteria_for(doc),
        "options": list(doc["options"]) if doc.get("options") else None,
        "seed": doc.get("seed"),
        "source": {"dataset": doc["dataset"], "config": doc["config"],
                   "split": doc["split"], "hf_revision": doc["hf_revision"],
                   "license": doc["license"], "url": doc.get("source_url", ""),
                   "state_fields": doc["state_fields"],
                   "length_cap": doc.get("length_cap"),
                   "state_hash_source": doc.get("state_hash_source", "jevals")},
        "attribution": doc.get("attribution") or (
            "Jevals (jevals.com), suite " + str(doc["version"])
            + ". Suite files CC-BY-4.0. Item text is not committed; it is rebuilt from "
              "the upstream dataset and hash-checked."),
        "contamination": CONTAMINATION.get(suite_id, []),
        "notes": doc.get("notes", []),
        "questions": questions,
        "labels": labels,
    }
    if mixed:
        built["instructions"] = (f"per question; {len(doc['question_schemas'])} question "
                                "schemas in the manifest")
        built["options"] = None
    return built


def download(suite_ids=None, out_dir=None, progress=None) -> list:
    """Fetch, verify and write one question file per suite. The only networked call.

    Returns the paths written. Nothing is written for a suite whose states did not all
    verify, because half a set on disk is a set somebody runs by accident.
    """
    wanted = list(suite_ids or SUITES)
    unknown = [s for s in wanted if s not in SUITES]
    if unknown:
        raise SetError(f"unknown suite(s) {', '.join(unknown)}; pick from "
                       f"{', '.join(SUITES)}")
    directory = pathlib.Path(out_dir) if out_dir else cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for suite_id in wanted:
        doc = load_manifest(suite_id)
        source = SOURCES[suite_id]
        if progress:
            progress(f"{suite_id}: {doc['dataset']} {doc['config']}/{doc['split']}")
        if source["fetch"] == "jsonl":
            rows = fetch_jsonl(doc["dataset"], doc["hf_revision"], source["path"])
        else:
            up_to = max(int(i["row_idx"]) for i in doc["items"])
            rows = fetch_rows(doc["dataset"], doc["config"], doc["split"], up_to)
        built = build_questions(doc, rows)
        path = directory / f"{suite_id}.questions.json"
        path.write_text(json.dumps(built, ensure_ascii=False, indent=1) + "\n")
        written.append(path)
        if progress:
            progress(f"{suite_id}: {len(built['questions'])} questions verified "
                     f"against their published hashes -> {path}")
    return written


# ------------------------------------------------------------------ question files

#: Keys an answer key could hide behind. A question object carrying one of these would
#: put the answer next to the state on the wire, so the label lives in the file's own
#: separate ``labels`` map instead and a question that names one of these is a load
#: error. :func:`ainode.bench.decide.suite.wire_leaks` holds the other half of the rule.
ANSWER_KEYS = ("label", "labels", "target", "expected", "expected_answer",
               "answer", "answer_key", "answerkey", "passinganswer",
               "passing_answer", "correct", "correct_answer", "ground_truth",
               "groundtruth", "gold", "gold_label", "solution", "truth")


def answer_key_leaks(obj, skip=()) -> list:
    """Every answer-key-shaped key anywhere in ``obj``, by path. Empty means clean.

    Walks dicts and lists, so a key nested three levels down inside a question block is
    found. ``skip`` names top-level keys not to descend into, which is how the state is
    left alone: the state is the caller's own whitelisted data and a private set is
    allowed a field called whatever its dataset calls it.
    """
    found = []

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                here = f"{path}.{key}" if path else str(key)
                if str(key).lower() in ANSWER_KEYS:
                    found.append(here)
                walk(value, here)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in skip:
                continue
            here = str(key)
            if str(key).lower() in ANSWER_KEYS:
                found.append(here)
            walk(value, here)
    else:
        walk(obj, "")
    return found


def validate_questions(doc, path=None) -> dict:
    """A question file, checked down to what the runner and the metrics assume.

    The same strictness the manifests get, and for the same reason. A question file is
    also what a private blind set arrives as, so this is the one gate between "somebody
    handed us a file" and "we published a Decision Score from it".

    **The labels live in the file's own ``labels`` map, never inside a question.** A
    question object holds an id, a state and nothing that reveals the answer, so the
    transport has no label to leak even by accident: it never sees one. A question
    carrying an answer-key-shaped field is a load error naming the field.
    """
    where = f" in {path}" if path else ""
    if not isinstance(doc, dict):
        raise SetError(f"the question file{where} is not a JSON object")
    for key in ("id", "type", "questions", "labels"):
        if not doc.get(key):
            raise SetError(f"the question file{where} is missing '{key}'")
    if not isinstance(doc["labels"], dict):
        raise SetError(f"the question file{where} has a 'labels' that is not a map of "
                       "question id to label")
    from ainode.bench.decide.jevals import TYPES

    mixed = doc["type"] == "mixed"
    if not mixed and doc["type"] not in TYPES:
        raise SetError(f"the question file{where} has type {doc['type']!r}; pick from "
                       f"{', '.join(TYPES)} or 'mixed' with a type per question")
    if not mixed:
        for key in ("instructions", "options"):
            if not doc.get(key):
                raise SetError(f"the question file{where} is missing '{key}'")
    questions = doc["questions"]
    if not isinstance(questions, list) or not questions:
        raise SetError(f"the question file{where} holds no questions")
    seen = set()
    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            raise SetError(f"question {index}{where} is not an object")
        qid = question.get("id")
        if not qid:
            raise SetError(f"question {index}{where} has no id")
        if qid in seen:
            raise SetError(f"question id {qid} appears twice{where}")
        seen.add(qid)
        if question.get("state") in (None, ""):
            raise SetError(f"question {qid}{where} has no state")
        # `gold` is a distribution, not an answer key, and the wire guard skips it the
        # way it skips the state; everything else in a question is walked.
        leaks = answer_key_leaks(question, skip=("state", "gold", "space"))
        if leaks:
            raise SetError(
                f"question {qid}{where} carries {', '.join(leaks)}, which is an answer "
                "key beside the state. Labels belong in the file's own 'labels' map, "
                "which the transport never reads")
        spec = question_spec(doc, question)
        if spec["type"] not in TYPES:
            raise SetError(f"question {qid}{where} has type {spec['type']!r}; pick from "
                           f"{', '.join(TYPES)}")
        if not spec.get("instructions"):
            raise SetError(f"question {qid}{where} has no instructions")
        options = spec["options"]
        if not isinstance(options, list) or len(options) < 2:
            raise SetError(f"question {qid}{where} declares fewer than two options")
        if len(set(options)) != len(options):
            raise SetError(f"question {qid}{where} repeats an option")
        if doc["labels"].get(qid) not in options:
            raise SetError(f"question {qid}{where} is labeled "
                           f"{doc['labels'].get(qid)!r} in 'labels', which is not one "
                           "of its options")
        gold = question.get("gold")
        if gold is not None:
            if not isinstance(gold, dict) or not gold:
                raise SetError(f"question {qid}{where} has a 'gold' that is not a "
                               "probability map")
            unknown = [k for k in gold if k not in options]
            if unknown:
                raise SetError(f"question {qid}{where} has gold for "
                               f"{', '.join(map(str, unknown))}, which are not its "
                               "options")
    return doc


def question_spec(doc: dict, question: dict) -> dict:
    """The typed question for one entry: its own fields, else the file's.

    One resolution point for both question-file shapes, so the transports, the metrics
    and the validator can never disagree about what was asked. A mixed file states the
    type, the instructions, the criteria and the options per question; a
    single-primitive file states them once at the top and every question inherits them.
    """
    return {
        "type": question.get("type") or doc.get("type"),
        "instructions": question.get("instructions") or doc.get("instructions"),
        "criteria": (question["criteria"] if "criteria" in question
                     else doc.get("criteria")),
        "options": list(question.get("options") or doc.get("options") or []),
    }


def load_questions(path) -> dict:
    """Read and validate a question file. Hash-checks whatever carries a hash.

    A file written by ``download`` carries a ``state_sha256`` per question, so loading
    it re-verifies every state rather than trusting the disk; a hand-authored private
    set carries none and is loaded as it is.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        raise SetError(f"no question file at {path}")
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SetError(f"{path} is not valid JSON: {exc}") from exc
    validate_questions(doc, path)
    for question in doc["questions"]:
        expected = question.get("state_sha256")
        if expected and state_sha256(question["state"]) != expected:
            raise SetError(f"{path}: question {question['id']} does not match its own "
                           "state hash; delete the cache and download it again")
    doc.setdefault("set", doc["id"])
    return doc


def load_suite_questions(suite_ids=None) -> list:
    """The downloaded question files for those suites, or an error saying how to get them."""
    wanted = list(suite_ids or SUITES)
    unknown = [s for s in wanted if s not in SUITES]
    if unknown:
        raise SetError(f"unknown suite(s) {', '.join(unknown)}; pick from "
                       f"{', '.join(SUITES)}")
    out = []
    for suite_id in wanted:
        path = questions_path(suite_id)
        if not path.is_file():
            raise SetError(
                f"{suite_id} has not been downloaded: no {path}. Run "
                f"`python3 scripts/ainode-bench.py decide download {suite_id}` first. "
                "The item text is deliberately not committed (see "
                "bench/decide/JEVALS.md)")
        out.append(load_questions(path))
    return out


def manifest_summary(suite_id: str) -> dict:
    """The one-line description of a suite, for a dry run and for a record's settings."""
    doc = load_manifest(suite_id)
    schemas = doc.get("question_schemas") or {}
    widths = [len(spec["options"]) for block in schemas.values()
              for spec in block.values()] or [len(doc.get("options") or [])]
    return {"id": doc["id"], "suite_version": doc["version"],
            "type": doc["primitive"], "title": doc.get("title", doc["id"]),
            "items": int(doc["n_items"]),
            "options": max(widths) if widths else 0,
            "dataset": doc["dataset"],
            "split": f"{doc['config']}/{doc['split']}",
            "hf_revision": doc["hf_revision"], "license": doc["license"],
            "seed": doc.get("seed"),
            "recipe_of_record": doc.get("recipe_of_record")
                                or RECIPE_OF_RECORD.get(suite_id),
            "contaminated_for": [c["system"] for c in CONTAMINATION.get(suite_id, [])],
            "downloaded": questions_path(suite_id).is_file()}


__all__ = ["ANSWER_KEYS", "CONTAMINATION", "ENV_CACHE", "ENV_SETS", "HTTP_RETRIES",
           "HTTP_TIMEOUT", "RECIPE_OF_RECORD", "RESOLVE", "ROWS_API", "ROWS_PAGE",
           "SOURCES", "SUITES", "SetError", "answer_key_leaks", "build_questions",
           "cache_dir", "criteria_for", "download", "fetch_jsonl", "fetch_rows",
           "load_manifest", "load_questions", "load_suite_questions", "manifest_path",
           "manifest_spec", "manifest_summary", "question_spec", "questions_path",
           "sets_dir", "state_json", "state_sha256", "validate_questions"]
