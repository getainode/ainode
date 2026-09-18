"""The decision bench: can a backend's typed decisions be trusted by code.

The other three sections measure generation. The throughput bench answers how fast a
model generates, the harness bench whether it can drive a coding agent to green
tests, the agentic rubric whether it can hold an agent loop together. None of them
answers the question a router, a triage step or a guardrail actually asks: given this
state, pick one of these options, and tell me how sure you are in a number I can gate
on.

That last part is the measurement. A backend with 96% accuracy whose confidence means
nothing is worse to automate than a slightly less accurate one that knows when it is
guessing: a wrong answer at 0.95 gets acted on, and a wrong answer at 0.45 is an
abstention a person looks at. So every block here carries a Brier score, an expected
calibration error with the reliability table behind it, and a count of the wrong
answers that survive a 0.8 and a 0.9 gate.

    ``items.py``      the labeled set, loaded strictly from bench/decide/items.json
    ``metrics.py``    accuracy, Brier, calibration, thresholds, latency, cost
    ``backends.py``   ainode (POST /v1/decide), chat (lettered options), jev (hosted)
    ``runner.py``     the loop, the tables, the record
    ``cli.py``        ``scripts/ainode-bench.py decide ...``

Stdlib only, like the rest of ``ainode/bench``. See ``bench/decide/README.md``.
"""

from ainode.bench.decide.backends import (
    BACKENDS,
    JEV_INPUT_USD_PER_MTOK,
    JEV_MODEL,
    JEV_URL,
    Backend,
    BackendError,
    ChatBackend,
    DecideBackend,
    Decision,
    JevBackend,
    Request,
    build_backend,
)
from ainode.bench.decide.items import (
    CHOICE,
    KINDS,
    NOUL,
    Item,
    ItemError,
    ItemSet,
    default_items_path,
    load_items,
    validate_document,
)
from ainode.bench.decide.metrics import (
    BINS,
    THRESHOLDS,
    accuracy,
    brier,
    ece,
    reliability,
    summarize,
    summarize_sets,
    threshold_counts,
)
from ainode.bench.decide.runner import (
    DEFAULT_CONCURRENCY,
    SOURCE,
    build_decide_block,
    build_notes,
    build_record,
    row_for,
    run_items,
)

__all__ = ["BACKENDS", "BINS", "CHOICE", "DEFAULT_CONCURRENCY",
           "JEV_INPUT_USD_PER_MTOK", "JEV_MODEL", "JEV_URL", "KINDS", "NOUL",
           "SOURCE", "THRESHOLDS", "Backend", "BackendError", "ChatBackend",
           "DecideBackend", "Decision", "Item", "ItemError", "ItemSet",
           "JevBackend", "Request", "accuracy", "brier", "build_backend",
           "build_decide_block", "build_notes", "build_record",
           "default_items_path", "ece", "load_items", "reliability", "row_for",
           "run_items", "summarize", "summarize_sets", "threshold_counts",
           "validate_document"]
