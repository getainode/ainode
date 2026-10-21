"""Percentiles, throughput and cosine similarity, over plain numbers.

Every function here takes plain values and returns plain values, so the tests drive
all of them from canned payloads and the runner is the only thing that has to know
what a request is. Stdlib only, and no numpy: a dot product over 1024 floats is not
where a bench spends its time.

The honesty rules are the ones the rest of ``bench/`` runs under. A measurement that
was not taken is ``None`` and never a zero: a sweep where every request failed
reports null throughput rather than 0.0 texts per second, and a run where no response
carried a token count reports null tokens per second rather than pretending the
tokens were free.
"""
from __future__ import annotations

import math


def percentile(values, q: float):
    """The ``q``-th percentile (0..1) by linear interpolation, or None if empty.

    Interpolating rather than picking the nearest sample, so a p95 over 50 values is
    not silently the 48th value: with 50 samples the exact index is 46.55, and
    rounding it either way is a different number reported under the same name.
    """
    xs = sorted(float(v) for v in values if v is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    position = (len(xs) - 1) * max(0.0, min(1.0, float(q)))
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return xs[low]
    return xs[low] + (xs[high] - xs[low]) * (position - low)


def latency_block(wall_ms_values, errors: int = 0) -> dict:
    """p50/p95 and the shape around them, over single-text requests.

    ``n`` counts the requests made and ``answered`` the ones that came back, so a
    percentile is never quietly taken over a smaller set than the header implies.
    """
    taken = [v for v in wall_ms_values if v is not None]

    def r(value):
        return None if value is None else round(value, 2)

    return {
        "n": len(wall_ms_values) + errors,
        "answered": len(taken),
        "errors": errors,
        "p50_ms": r(percentile(taken, 0.50)),
        "p95_ms": r(percentile(taken, 0.95)),
        "min_ms": r(min(taken)) if taken else None,
        "max_ms": r(max(taken)) if taken else None,
        "mean_ms": r(sum(taken) / len(taken)) if taken else None,
    }


def throughput_row(batch: int, requests: int, texts: int, seconds: float,
                   tokens, errors: int = 0) -> dict:
    """One row of the batch sweep: how much work went through, and how fast.

    ``tokens`` is what the responses reported, or None when none of them did. Texts
    per second is the rate a caller cares about and tokens per second is the rate the
    hardware sets; both are over the wall time of the whole sweep at that batch size,
    including the gaps between requests, because that is the time a caller waits.
    """
    ok = seconds > 0 and texts > 0
    return {
        "batch": int(batch),
        "requests": int(requests),
        "texts": int(texts),
        "errors": int(errors),
        "seconds": round(float(seconds), 3),
        "texts_per_s": round(texts / seconds, 2) if ok else None,
        "tokens": None if tokens is None else int(tokens),
        "tokens_per_s": (round(tokens / seconds, 1)
                         if (ok and tokens is not None) else None),
    }


def cosine(a, b):
    """Cosine similarity of two vectors, or None when either has no length.

    Computed here rather than trusting the engine to have normalised: a checkpoint
    served through the wrong pooling mode returns vectors whose norms are all over
    the place, and dividing by them is what makes the ordering check meaningful
    instead of an accident of scale.
    """
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na == 0.0 or nb == 0.0:
        return None
    return dot / (na * nb)


def quality_block(pairs, vectors_by_text: dict) -> dict:
    """The six pairs scored, and the one verdict taken off them.

    ``ordered`` is True only when the LOWEST related score is above the HIGHEST
    unrelated one, which is a stronger statement than any per-pair threshold and does
    not encode a number about this checkpoint. ``margin`` is the gap between those
    two, negative when the check fails, so a reader sees how close it was. A pair
    missing a vector makes the verdict None rather than False: nothing was measured,
    so nothing failed.
    """
    rows = []
    for pair in pairs:
        score = cosine(vectors_by_text.get(pair["a"]), vectors_by_text.get(pair["b"]))
        rows.append({"id": pair["id"], "related": bool(pair["related"]),
                     "a": pair["a"], "b": pair["b"],
                     "cosine": None if score is None else round(float(score), 6)})
    related = [r["cosine"] for r in rows if r["related"]]
    unrelated = [r["cosine"] for r in rows if not r["related"]]
    if any(v is None for v in related + unrelated) or not related or not unrelated:
        return {"pairs": rows, "related_min": None, "unrelated_max": None,
                "margin": None, "ordered": None}
    related_min, unrelated_max = min(related), max(unrelated)
    return {
        "pairs": rows,
        "related_min": round(related_min, 6),
        "unrelated_max": round(unrelated_max, 6),
        "margin": round(related_min - unrelated_max, 6),
        "ordered": bool(related_min > unrelated_max),
    }


__all__ = ["cosine", "latency_block", "percentile", "quality_block",
           "throughput_row"]
