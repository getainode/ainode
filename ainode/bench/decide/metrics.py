"""The metrics: accuracy, Brier, calibration, threshold survival, latency, cost.

Accuracy is the least interesting number here. A decision backend is used by code
that acts on the answer, so what matters is whether the confidence it reports can be
trusted as a gate: a wrong answer at 0.95 gets automated and does damage, and a
wrong answer at 0.45 is an abstention a human looks at. That is why every block
carries a Brier score, an expected calibration error with the bins behind it, and a
count of wrong answers that survive a 0.8 and a 0.9 threshold.

Every function takes plain rows (dicts) and returns plain values, so the tests score
canned rows with no backend and no network. A row is what
:mod:`ainode.bench.decide.runner` writes into the record:

    ``set``          which item set the row belongs to
    ``label``        the labeled answer
    ``answer``       what the backend answered, or None
    ``p_answer``     probability the backend gave its own answer, or None
    ``p_label``      probability it gave the labeled answer, or None
    ``wall_ms``      measured round trip for this item
    ``tokens_in``    prompt tokens the backend reported, or None
    ``tokens_out``   completion tokens the backend reported, or None
    ``error``        transport or protocol failure, or None

``p_answer`` and ``p_label`` come from the backend's distribution when it returns
one, and from its reported confidence when it does not (see the runner). A row with
neither is counted in ``no_confidence`` and left out of the calibration numbers
rather than given a probability nobody reported.
"""
from __future__ import annotations

import math

#: Bins for the reliability table and the ECE. Five over [0, 1], as the prototype
#: this bench was ported from used, which keeps the numbers comparable.
BINS = 5
#: The confidence gates a caller would actually automate behind.
THRESHOLDS = (0.8, 0.9)


def answered(rows) -> list:
    """Rows the backend answered: no error and an answer that is not None."""
    return [r for r in rows if not r.get("error") and r.get("answer") is not None]


def scored(rows) -> list:
    """Answered rows that also carry a probability for the answer given."""
    return [r for r in answered(rows) if r.get("p_answer") is not None]


def is_correct(row) -> bool:
    return row.get("answer") == row.get("label")


def accuracy(rows):
    """Correct over answered. ``None`` when nothing was answered."""
    rows = answered(rows)
    if not rows:
        return None
    return sum(1 for r in rows if is_correct(r)) / len(rows)


def brier(rows):
    """Mean ``(1 - p_label) ** 2`` over rows that reported a probability.

    One term, on the labeled option, rather than the full multiclass sum: it is the
    probability the backend put on the right answer, which is the quantity a caller
    gates on. Lower is better and 0.25 is what a coin flip scores.
    """
    rows = [r for r in scored(rows) if r.get("p_label") is not None]
    if not rows:
        return None
    return sum((1.0 - float(r["p_label"])) ** 2 for r in rows) / len(rows)


def bin_index(probability: float, bins: int = BINS) -> int:
    return min(bins - 1, max(0, int(float(probability) * bins)))


def reliability(rows, bins: int = BINS) -> list:
    """The table behind the ECE: per bin, how many rows and how often they were right.

    An empty bin keeps its row with ``count: 0`` and null accuracy so the shape of
    the table does not change between runs.
    """
    buckets = [[] for _ in range(bins)]
    for row in scored(rows):
        buckets[bin_index(row["p_answer"], bins)].append(row)
    table = []
    for index, bucket in enumerate(buckets):
        block = {"lo": round(index / bins, 3), "hi": round((index + 1) / bins, 3),
                 "count": len(bucket), "accuracy": None, "confidence": None}
        if bucket:
            block["accuracy"] = sum(1 for r in bucket if is_correct(r)) / len(bucket)
            block["confidence"] = sum(float(r["p_answer"]) for r in bucket) / len(bucket)
        table.append(block)
    return table


def ece(rows, bins: int = BINS):
    """Expected calibration error: the bins' |accuracy - confidence| gap, weighted.

    ``None`` when no row reported a probability. A backend that says 0.9 and is
    right 90% of the time scores 0; one that says 0.99 and is right 80% of the time
    scores about 0.19, which is the number that decides whether its confidence can
    gate anything.
    """
    rows = scored(rows)
    if not rows:
        return None
    total = 0.0
    for block in reliability(rows, bins):
        if block["count"]:
            total += abs(block["accuracy"] - block["confidence"]) * block["count"]
    return total / len(rows)


def threshold_counts(rows, thresholds=THRESHOLDS) -> dict:
    """Per threshold: how many answers survive it, how many of those are wrong.

    ``kept`` is the answers at or above the gate, ``wrong`` the ones among them that
    disagree with the label (the failure mode this bench exists to find), and
    ``abstained`` the answers below the gate, which a caller would hand to a person.
    ``no_confidence`` counts answered rows the backend gave no probability for: they
    cannot be gated either way, and calling them abstentions would flatter a backend
    that reports nothing.
    """
    out = {}
    rows_answered = answered(rows)
    unscored = [r for r in rows_answered if r.get("p_answer") is None]
    for threshold in thresholds:
        kept = [r for r in rows_answered
                if r.get("p_answer") is not None
                and float(r["p_answer"]) >= threshold]
        below = [r for r in rows_answered
                 if r.get("p_answer") is not None
                 and float(r["p_answer"]) < threshold]
        out[f"{threshold:g}"] = {"kept": len(kept),
                                 "wrong": sum(1 for r in kept if not is_correct(r)),
                                 "abstained": len(below),
                                 "no_confidence": len(unscored)}
    return out


def percentile(values, fraction: float):
    """Nearest-rank percentile: sort, take the ``ceil(fraction * n)``th, no interpolation.

    An observed round trip rather than a number between two of them, so a p95 is
    always a request that really took that long.
    """
    ordered = sorted(v for v in values if v is not None)
    if not ordered:
        return None
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[min(max(index, 0), len(ordered) - 1)]


def latency(rows) -> dict:
    """p50/p95 of the measured round trip, over every row that came back at all."""
    walls = [r.get("wall_ms") for r in rows if r.get("wall_ms") is not None]
    return {"p50_ms": percentile(walls, 0.5), "p95_ms": percentile(walls, 0.95)}


def tokens(rows) -> dict:
    """Summed reported usage. A row the backend reported nothing for adds nothing."""
    return {"in": sum(int(r.get("tokens_in") or 0) for r in rows),
            "out": sum(int(r.get("tokens_out") or 0) for r in rows)}


def cost_usd(counts: dict, input_usd_per_mtok: float,
             output_usd_per_mtok: float = 0.0) -> float:
    """Dollars for one block's reported tokens at the backend's posted rate.

    A local backend passes 0 for both rates, and the cost column reads $0 for it:
    the electricity is real but nobody is billing per token, and an invented figure
    would be exactly the kind of number this repo does not publish.
    """
    return (counts.get("in", 0) * input_usd_per_mtok
            + counts.get("out", 0) * output_usd_per_mtok) / 1e6


def _round(value, places=3):
    return None if value is None else round(value, places)


def summarize(rows, input_usd_per_mtok: float = 0.0,
              output_usd_per_mtok: float = 0.0, bins: int = BINS,
              thresholds=THRESHOLDS) -> dict:
    """One metrics block over ``rows``: the shape that lands in the record."""
    token_counts = tokens(rows)
    block = {
        "n": len(rows),
        "answered": len(answered(rows)),
        "errors": sum(1 for r in rows if r.get("error")),
        "accuracy": _round(accuracy(rows)),
        "brier": _round(brier(rows)),
        "ece": _round(ece(rows, bins)),
        "bins": [{**b, "accuracy": _round(b["accuracy"]),
                  "confidence": _round(b["confidence"])}
                 for b in reliability(rows, bins)],
        "thresholds": threshold_counts(rows, thresholds),
        "tokens": token_counts,
        "cost_usd": round(cost_usd(token_counts, input_usd_per_mtok,
                                   output_usd_per_mtok), 6),
    }
    block.update(latency(rows))
    return block


def summarize_sets(rows, set_names, **kw) -> dict:
    """One block per set, in the order ``set_names`` gives, plus nothing else.

    A set with no rows is absent rather than a block of nulls, the same rule the
    agentic rubric's groups run under: a measurement nobody took is not a zero.
    """
    out = {}
    for name in set_names:
        mine = [r for r in rows if r.get("set") == name]
        if mine:
            out[name] = summarize(mine, **kw)
    return out


__all__ = ["BINS", "THRESHOLDS", "accuracy", "answered", "bin_index", "brier",
           "cost_usd", "ece", "is_correct", "latency", "percentile", "reliability",
           "scored", "summarize", "summarize_sets", "threshold_counts", "tokens"]
