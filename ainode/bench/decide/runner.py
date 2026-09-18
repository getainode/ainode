"""The loop over the items, the tables it prints, and the record it writes.

One row per item, one metrics block per set and one overall, and a record with a
``decide`` block and no ``results`` block: this run measured decisions, not
throughput, and a zero in ``single_stream`` would be a number nobody took.

The honesty rules are the ones the rest of ``bench/`` runs under:

  * **Nothing is loaded, unloaded or restarted.** The run drives inference against
    whatever is already serving and adds real load to it.
  * **An item that failed says why.** A transport error or an unreadable response is
    one row with an ``error``, never a wrong answer and never a silent drop, and the
    record's notes count them separately from answers the backend got wrong.
  * **A probability nobody reported is absent, never assumed.** A backend that gives
    no distribution and no confidence is scored on accuracy alone, and the
    calibration numbers say how many rows they were taken over.
  * **Cost is the vendor's posted rate applied to reported tokens, or zero.** A local
    backend costs $0 in this table because nobody bills per token for it, and an
    invented electricity figure would be an estimate in a file of measurements.

Stdlib only: a thread pool over blocking urllib, which is what keeps the package
importable on a bare python3.
"""
from __future__ import annotations

import concurrent.futures as futures

from ainode.bench.decide import metrics
from ainode.bench.decide.items import NOUL

SCHEMA = 1
SOURCE = "scripts/ainode-bench.py decide"
DEFAULT_CONCURRENCY = 8


def row_for(item, decision) -> dict:
    """One item's row: what was asked, what came back, and how sure it was.

    The state text is not repeated here. It is in ``bench/decide/items.json`` under
    the same ``id``, and a record that carried both would be mostly prompt.
    """
    label_key = item.label_option
    distribution = decision.distribution or None
    p_answer = decision.confidence
    p_label = None
    if distribution and decision.answer is not None:
        # The answer's own probability is the confidence a caller gates on, whether
        # or not the backend also reported a number of its own.
        p_answer = distribution.get(item.option_for(decision.answer), p_answer)
    if distribution:
        p_label = distribution.get(label_key)
    elif decision.answer is not None and decision.confidence is not None:
        # No distribution, one confidence: it belongs to the answer given, so the
        # labeled option only inherits it when the answer was right.
        p_label = (decision.confidence if decision.answer == item.label
                   else 1.0 - decision.confidence)
    return {
        "id": item.id,
        "set": item.set,
        "kind": item.kind,
        "label": item.label,
        "answer": decision.answer,
        "correct": (None if decision.answer is None
                    else bool(decision.answer == item.label)),
        "p_answer": None if p_answer is None else round(float(p_answer), 6),
        "p_label": None if p_label is None else round(float(p_label), 6),
        "distribution": (None if not distribution else
                         {k: round(float(v), 6) for k, v in distribution.items()}),
        "wall_ms": decision.wall_ms,
        "server_latency_ms": decision.server_latency_ms,
        "tokens_in": decision.tokens_in,
        "tokens_out": decision.tokens_out,
        "error": decision.error,
    }


def run_items(backend, items, concurrency: int = DEFAULT_CONCURRENCY,
              progress=None) -> list:
    """Every item through ``backend``, in item order, ``concurrency`` at a time.

    Ordered by item and not by completion, so two backends' rows line up one to one
    and a comparison can be read down the file.
    """
    rows = [None] * len(items)
    workers = max(1, int(concurrency))
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(backend.decide, item): index
                   for index, item in enumerate(items)}
        done = 0
        for future in futures.as_completed(pending):
            index = pending[future]
            rows[index] = row_for(items[index], future.result())
            done += 1
            if progress:
                progress(done, len(items), rows[index])
    return rows


def reported_model(backend) -> str:
    """The model id to put in the record: what the service said it was.

    For the hosted backend that is the version string the API reports
    (``jev-1.13.0``) rather than the alias that was asked for (``jev-latest``), so a
    record names the thing that answered. Falls back to the requested id.
    """
    return (getattr(backend, "reported_model", "")
            or getattr(backend, "model", "") or backend.name)


def build_decide_block(backend, item_set, rows, set_names, concurrency: int,
                       model_reported: str = "") -> dict:
    """The record's ``decide`` block. See bench/SCHEMA.md."""
    kw = {"input_usd_per_mtok": backend.input_usd_per_mtok,
          "output_usd_per_mtok": backend.output_usd_per_mtok}
    overall = metrics.summarize(rows, **kw)
    return {
        "backend": backend.name,
        "endpoint": backend.endpoint,
        "model_reported": model_reported or None,
        "item_set": item_set.as_json(),
        "protocol": {**backend.protocol(), "concurrency": concurrency,
                     "bins": metrics.BINS,
                     "thresholds": [f"{t:g}" for t in metrics.THRESHOLDS],
                     "confidence": "the probability the backend put on the answer it "
                                   "gave; its reported confidence when it gave no "
                                   "distribution",
                     "brier": "one term, on the labeled option's probability"},
        "overall": overall,
        "sets": metrics.summarize_sets(rows, set_names, **kw),
        "rows": rows,
    }


def build_notes(backend, item_set, rows, seconds: int, source: str = SOURCE) -> list:
    """The notes a reader needs to know what the numbers are and are not."""
    errors = [r["id"] for r in rows if r.get("error")]
    unscored = [r["id"] for r in metrics.answered(rows) if r.get("p_answer") is None]
    notes = [
        f"Measured by {source} in {seconds}s over {len(rows)} labeled items from "
        f"{item_set.path.name} ({item_set.id}); nothing was loaded, unloaded or "
        "restarted.",
        "Accuracy is the weakest number here: the failure mode automation cares "
        "about is a wrong answer at high confidence, which is what the Brier score, "
        "the calibration error and the wrong-at-threshold counts measure.",
        "Confidence is the probability the backend put on the answer it gave. Brier "
        "is a single term on the labeled option's probability, and the calibration "
        "error uses 5 bins over the reliability table in the same block.",
    ]
    if backend.input_usd_per_mtok or backend.output_usd_per_mtok:
        notes.append(f"Cost is the posted rate for {backend.name} applied to the "
                     f"tokens the API reported: "
                     f"${backend.input_usd_per_mtok:g} per million input tokens and "
                     f"${backend.output_usd_per_mtok:g} per million output tokens.")
    else:
        notes.append(f"Cost is $0 for the {backend.name} backend: it runs on our own "
                     "hardware and nobody bills per token for it. The electricity is "
                     "real and is not a number this record claims to have measured.")
    if backend.name == "chat":
        notes.append("The chat backend is the fallback instrument: lettered options "
                     "with thinking off, and a distribution softmaxed from the top "
                     "logprobs of the one letter token. Those probabilities are over "
                     "letters rather than over meanings, so they are a weaker "
                     "statement than a decision endpoint's.")
    if errors:
        notes.append(f"{len(errors)} item(s) failed on a transport or protocol error "
                     f"rather than on the answer: {', '.join(errors[:10])}"
                     f"{', ...' if len(errors) > 10 else ''}.")
    if unscored:
        notes.append(f"{len(unscored)} answered item(s) carried no probability, so "
                     "they are in the accuracy and out of the calibration numbers.")
    return notes


def build_record(label: str, model_block: dict, placement: dict, decide_block: dict,
                 settings: dict, notes: list, stamp: str,
                 source: str = SOURCE) -> dict:
    """A schema-1 record with a ``decide`` block and no ``results`` block."""
    return {"schema": SCHEMA, "stamp": stamp, "label": label, "model": model_block,
            "placement": placement, "settings": settings, "decide": decide_block,
            "notes": notes, "source": source}


# ---------------------------------------------------------------- printing

#: The per-set table's columns, and how wide each one prints.
COLUMNS = (("set", 9), ("n", 4), ("acc", 6), ("brier", 6), ("ece", 6),
           ("wrong@0.8", 10), ("wrong@0.9", 10), ("abstain@0.9", 12),
           ("p50 ms", 7), ("p95 ms", 7), ("tok in", 8), ("tok out", 8), ("cost", 9))


def fmt_number(value, places=3):
    return "-" if value is None else f"{value:.{places}f}"


def fmt_cost(value):
    if not value:
        return "$0"
    return f"${value:.4f}" if value >= 0.0001 else f"${value:.6f}"


def table_rows(block) -> list:
    """``[(name, cells), ...]``: one line per set, then ALL. Cells are strings."""
    out = []
    for name, metrics_block in list(block["sets"].items()) + [("ALL", block["overall"])]:
        thresholds = metrics_block.get("thresholds") or {}
        at_08 = thresholds.get("0.8") or {}
        at_09 = thresholds.get("0.9") or {}
        out.append((name, [
            name,
            str(metrics_block["n"]),
            fmt_number(metrics_block["accuracy"]),
            fmt_number(metrics_block["brier"]),
            fmt_number(metrics_block["ece"]),
            str(at_08.get("wrong", "-")),
            str(at_09.get("wrong", "-")),
            str(at_09.get("abstained", "-")),
            "-" if metrics_block["p50_ms"] is None else str(metrics_block["p50_ms"]),
            "-" if metrics_block["p95_ms"] is None else str(metrics_block["p95_ms"]),
            str((metrics_block["tokens"] or {}).get("in", 0)),
            str((metrics_block["tokens"] or {}).get("out", 0)),
            fmt_cost(metrics_block["cost_usd"]),
        ]))
    return out


def print_table(block, title: str, out=print) -> None:
    """The per-set table and the reliability table under it."""
    out(f"\n  {title}")
    out("  " + "".join(name.ljust(width) for name, width in COLUMNS))
    for _name, cells in table_rows(block):
        out("  " + "".join(cell.ljust(width)
                           for cell, (_h, width) in zip(cells, COLUMNS)))
    out(f"\n  reliability, {metrics.BINS} bins on the answer's own probability "
        "(overall)")
    out("  bin        count  accuracy  mean conf")
    for bucket in block["overall"]["bins"]:
        out(f"  {bucket['lo']:.1f}-{bucket['hi']:.1f}  {bucket['count']:5d}  "
            f"{fmt_number(bucket['accuracy']):>8}  "
            f"{fmt_number(bucket['confidence']):>9}")
    errors = block["overall"]["n"] - block["overall"]["answered"]
    if errors:
        out(f"  {errors} item(s) did not come back; they are in n and out of the "
            "accuracy")


#: The rows of the side-by-side table, as (label, key path into a metrics block).
#: The path separator is a slash and not a dot because two of the keys are threshold
#: names ("0.8") that carry a dot of their own.
COMPARE_ROWS = (("items", "n"), ("accuracy", "accuracy"), ("brier", "brier"),
                ("ece", "ece"), ("wrong at 0.8", "thresholds/0.8/wrong"),
                ("wrong at 0.9", "thresholds/0.9/wrong"),
                ("abstained at 0.9", "thresholds/0.9/abstained"),
                ("p50 ms", "p50_ms"), ("p95 ms", "p95_ms"),
                ("tokens in", "tokens/in"), ("tokens out", "tokens/out"),
                ("cost", "cost_usd"))


def dig(block, path: str):
    """One value out of a nested block, addressed as ``a/b/c``."""
    node = block
    for part in path.split("/"):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def print_compare(blocks, titles, out=print) -> None:
    """The overall numbers of two runs in one table, and the per-set accuracies."""
    width = max(16, *(len(t) for t in titles))
    out("\n  side by side, overall")
    out("  " + "metric".ljust(18) + "".join(t.ljust(width + 2) for t in titles))
    for name, path in COMPARE_ROWS:
        cells = []
        for block in blocks:
            value = dig(block["overall"], path)
            if name == "cost":
                cells.append(fmt_cost(value))
            elif isinstance(value, float):
                cells.append(fmt_number(value))
            else:
                cells.append("-" if value is None else str(value))
        out("  " + name.ljust(18) + "".join(c.ljust(width + 2) for c in cells))
    names = []
    for block in blocks:
        for name in block["sets"]:
            if name not in names:
                names.append(name)
    out("\n  accuracy per set")
    out("  " + "set".ljust(18) + "".join(t.ljust(width + 2) for t in titles))
    for name in names:
        cells = [fmt_number(dig(block["sets"].get(name) or {}, "accuracy"))
                 for block in blocks]
        out("  " + name.ljust(18) + "".join(c.ljust(width + 2) for c in cells))


def wrong_rows(rows, items_by_id) -> list:
    """``(row, item)`` for every answered item the backend got wrong.

    Printed after a run because a decision bench's useful output is the disagreement
    list: which item, what the label says, what came back, and at what confidence.
    """
    out = []
    for row in rows:
        if row.get("error") or row.get("answer") is None:
            continue
        if not row.get("correct"):
            out.append((row, items_by_id.get(row["id"])))
    return out


def print_wrong(rows, items, out=print, limit: int = 20) -> None:
    by_id = {item.id: item for item in items}
    wrong = wrong_rows(rows, by_id)
    if not wrong:
        out("\n  no wrong answers")
        return
    out(f"\n  wrong answers ({len(wrong)}), highest confidence first")
    wrong.sort(key=lambda pair: pair[0].get("p_answer") or 0.0, reverse=True)
    for row, item in wrong[:limit]:
        state = (item.state if item else "")[:70]
        label = fmt_label(item.kind if item else "", row["label"])
        answer = fmt_label(item.kind if item else "", row["answer"])
        out(f"    {row['id']:<12} p={fmt_number(row['p_answer'], 2):<5} "
            f"label {label:<14} answered {answer:<14} {state}")
    if len(wrong) > limit:
        out(f"    ... and {len(wrong) - limit} more, all of them in the record")


def fmt_label(kind: str, value) -> str:
    if kind == NOUL or isinstance(value, bool):
        return "yes" if value else "no"
    return "-" if value is None else str(value)


__all__ = ["COLUMNS", "COMPARE_ROWS", "DEFAULT_CONCURRENCY", "SCHEMA", "SOURCE",
           "build_decide_block", "build_notes", "build_record", "dig", "fmt_cost",
           "fmt_label", "fmt_number", "print_compare", "print_table", "print_wrong",
           "reported_model", "row_for", "run_items", "table_rows", "wrong_rows"]
