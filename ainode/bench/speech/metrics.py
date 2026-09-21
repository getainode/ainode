"""Word error rate, the normaliser behind it, latency percentiles, real-time factor.

Every function here takes plain values and returns plain values, so the tests drive
all of them from canned transcripts and the runner is the only thing that has to know
what a request is. Stdlib only: a Levenshtein table over fifteen words is not where a
bench spends its time.

**The normaliser is part of the measurement, so it is documented and versioned.** An
error rate over raw strings is mostly a statement about orthography: a transcript that
heard every word and wrote "9" where the reference says "nine", or "19th" for "19",
would be scored wrong for a reason that has nothing to do with the audio. So two rates
are recorded and neither replaces the other:

  * ``wer`` uses the full normaliser: case folded, punctuation dropped, whitespace
    collapsed, ordinal suffixes removed, and number words folded to digits. This is
    the number the README column reports, because it is the one that answers "did the
    model hear the words".
  * ``wer_orthographic`` uses case, punctuation and whitespace only. It is the
    stricter number, and the gap between the two is exactly how much of the error was
    spelling rather than hearing.

The honesty rules are the ones the rest of ``bench/`` runs under. Nothing is
normalised away that changes a word: no stopword list, no stemming, no synonym map,
and no per-clip exception. A measurement nobody took is ``None`` and never a zero: a
clip that failed on transport has no rate at all rather than a 0.0 that would read as
a perfect transcript, or a 1.0 that would read as a model that heard nothing.
"""
from __future__ import annotations

import math
import re
import unicodedata

#: Bumped when the normaliser changes what it folds. A record carries it, because a
#: rate taken under a different normaliser is a different number under the same name.
NORMALIZER_VERSION = 1
NORMALIZER_ID = "case-punct-numbers-1"

#: Number words folded to digits so "nine" and "9" are one token. Units, teens and
#: tens only as literals; "hundred" and "thousand" compose the run around them.
_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000}

#: Ordinal spellings that are not the cardinal plus a suffix.
_ORDINAL_WORDS = {"first": "one", "second": "two", "third": "three",
                  "fifth": "five", "eighth": "eight", "ninth": "nine",
                  "twelfth": "twelve"}
_ORDINAL_SUFFIX = re.compile(r"^(\d+)(st|nd|rd|th)$")
_ORDINAL_TAIL = re.compile(r"^(twent|thirt|fort|fift|sixt|sevent|eight|ninet)ieth$")

#: Kept inside a word: an apostrophe makes "don't" one token, and a hyphen makes
#: "twenty-six" one token the number pass can then split.
_STRIP = re.compile(r"[^\w\s'\-]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def _fold_ordinal(word: str) -> str:
    """"19th" to "19", "third" to "three", "fortieth" to "forty". Else unchanged."""
    match = _ORDINAL_SUFFIX.match(word)
    if match:
        return match.group(1)
    if word in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[word]
    tail = _ORDINAL_TAIL.match(word)
    if tail:
        stem = tail.group(1)
        return {"twent": "twenty", "thirt": "thirty", "fort": "forty",
                "fift": "fifty", "sixt": "sixty", "sevent": "seventy",
                "eight": "eighty", "ninet": "ninety"}[stem]
    if word.endswith("ieth") or word.endswith("th"):
        base = word[:-4] + "y" if word.endswith("ieth") else word[:-2]
        if base in _UNITS or base in _TENS:
            return base
    return word


def _fold_numbers(words: list) -> list:
    """Fold runs of number words into the digits they spell.

    "four hundred and eighty seven" becomes "487" and "one hundred twenty eight"
    becomes "128", so a transcript that wrote the digits and a reference that wrote the
    words are the same tokens.

    **Magnitude decides where one number ends and the next begins**, which is the whole
    difficulty: "twenty one" is 21 but "one two" is two separate digits, and summing
    every adjacent number word would turn "one two three" into a single 6. So a slot is
    filled at most once per number: a second unit, or a second tens, closes the number
    being built and opens the next. "hundred" and "thousand" scale what is in hand and
    reopen the slots below them, which is what lets "one hundred twenty eight" keep
    accumulating.

    "and" is only swallowed INSIDE a run and only when what follows continues the
    number, so "salt and 3 eggs" keeps its conjunction.
    """
    out: list = []
    total = 0
    current = 0
    have_unit = False
    have_tens = False
    started = False

    def flush():
        nonlocal total, current, have_unit, have_tens, started
        if started:
            out.append(str(total + current))
        total = current = 0
        have_unit = have_tens = False
        started = False

    index = 0
    while index < len(words):
        word = words[index]
        if word in _UNITS:
            if have_unit:
                flush()
            current += _UNITS[word]
            have_unit = True
            started = True
        elif word in _TENS:
            if have_tens or have_unit:
                flush()
            current += _TENS[word]
            have_tens = True
            started = True
        elif word in _SCALES and started:
            scale = _SCALES[word]
            if scale == 100:
                current = (current or 1) * 100
            else:
                total += (current or 1) * scale
                current = 0
            have_unit = have_tens = False
        elif (word == "and" and started and index + 1 < len(words)
              and (words[index + 1] in _UNITS or words[index + 1] in _TENS
                   or words[index + 1] in _SCALES)):
            pass
        else:
            flush()
            out.append(word)
        index += 1
    flush()
    return out


def normalize(text: str, numbers: bool = True) -> list:
    """The words a rate is taken over. Returns a list of tokens.

    Case folded, unicode normalised, punctuation dropped (an apostrophe and a hyphen
    survive inside a word), whitespace collapsed. With ``numbers`` on, hyphenated
    numbers are split, ordinals folded to cardinals and number words folded to digits.
    With it off this is the orthographic normaliser, which is the stricter rate.
    """
    folded = unicodedata.normalize("NFKC", str(text or "")).lower()
    folded = folded.replace("’", "'").replace("‘", "'")
    folded = _STRIP.sub(" ", folded)
    folded = folded.replace(",", " ")
    words = [w.strip("-'") for w in _SPACES.sub(" ", folded).strip().split(" ")]
    words = [w for w in words if w]
    if not numbers:
        return words
    split: list = []
    for word in words:
        split.extend(part for part in word.split("-") if part)
    return _fold_numbers([_fold_ordinal(w) for w in split])


def edit_counts(reference: list, hypothesis: list) -> dict:
    """Levenshtein over words, with the three edit kinds kept apart.

    Substitutions, deletions and insertions are reported separately because they are
    different findings: a model that drops the end of every clip and one that
    hallucinates a trailing sentence both score badly, and only the breakdown tells
    them apart.
    """
    ref, hyp = list(reference), list(hypothesis)
    rows, cols = len(ref) + 1, len(hyp) + 1
    # (cost, subs, dels, ins) per cell; the backtrace is carried forward rather than
    # reconstructed, which keeps this one pass over a table of fifteen by fifteen.
    previous = [(j, 0, 0, j) for j in range(cols)]
    for i in range(1, rows):
        current = [(i, 0, i, 0)] + [(0, 0, 0, 0)] * (cols - 1)
        for j in range(1, cols):
            if ref[i - 1] == hyp[j - 1]:
                cost, subs, dels, ins = previous[j - 1]
                best = (cost, subs, dels, ins)
            else:
                pc, ps, pd, pi = previous[j - 1]
                sub = (pc + 1, ps + 1, pd, pi)
                dc, ds, dd, di = previous[j]
                delete = (dc + 1, ds, dd + 1, di)
                ic, isub, idel, iins = current[j - 1]
                insert = (ic + 1, isub, idel, iins + 1)
                best = min(sub, delete, insert, key=lambda t: t[0])
            current[j] = best
        previous = current
    cost, subs, dels, ins = previous[cols - 1]
    return {"reference_words": len(ref), "hypothesis_words": len(hyp),
            "substitutions": subs, "deletions": dels, "insertions": ins,
            "edits": cost}


def rate(edits: int, reference_words: int):
    """``edits / reference_words``, or None when there were no reference words."""
    if not reference_words:
        return None
    return edits / float(reference_words)


def clip_row(clip: dict, transcript, wall_ms, error=None) -> dict:
    """One clip's row: the transcript, both rates, the edit breakdown, the timings.

    A clip that failed carries its ``error`` and nulls for every number: no rate, no
    real-time factor, no edit counts. Folding a failed request in as a 100 percent
    error rate would put a transport failure in a figure a reader takes as the model's.
    """
    seconds = clip.get("seconds")
    row = {
        "id": clip["id"],
        "voice": clip.get("voice"),
        "locale": clip.get("locale"),
        "audio_seconds": None if seconds is None else round(float(seconds), 3),
        "reference": clip["text"],
        "transcript": None if error else str(transcript or ""),
        "wall_ms": None if wall_ms is None else round(float(wall_ms), 2),
        "error": error,
    }
    if error:
        row.update({"reference_words": len(normalize(clip["text"])),
                    "hypothesis_words": None, "substitutions": None,
                    "deletions": None, "insertions": None, "edits": None,
                    "wer": None, "wer_orthographic": None, "rtf": None})
        return row
    counts = edit_counts(normalize(clip["text"]), normalize(transcript))
    strict = edit_counts(normalize(clip["text"], numbers=False),
                         normalize(transcript, numbers=False))
    row.update(counts)
    wer = rate(counts["edits"], counts["reference_words"])
    strict_wer = rate(strict["edits"], strict["reference_words"])
    row["wer"] = None if wer is None else round(wer, 6)
    row["wer_orthographic"] = None if strict_wer is None else round(strict_wer, 6)
    row["rtf"] = real_time_factor(row["wall_ms"], seconds)
    return row


def real_time_factor(wall_ms, audio_seconds):
    """Seconds of wall per second of audio, or None when either is missing.

    Below 1 means the engine transcribes faster than the clip plays, which is the
    number that decides whether a stream can be kept up with. It is end to end from
    wherever the bench ran, so the transport floor beside it is part of reading it.
    """
    if wall_ms is None or not audio_seconds:
        return None
    return round((float(wall_ms) / 1000.0) / float(audio_seconds), 4)


def percentile(values, q: float):
    """The ``q``-th percentile (0..1) by linear interpolation, or None if empty.

    Interpolating rather than picking the nearest sample, the same choice the
    embedding bench documents: rounding an index either way is a different number
    reported under the same name.
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


def latency_block(rows) -> dict:
    """p50/p95 of the per-clip wall time, and the shape around it.

    ``n`` counts the clips sent and ``answered`` the ones that came back, so a
    percentile is never quietly taken over a smaller set than the header implies.
    """
    taken = [r.get("wall_ms") for r in rows if not r.get("error")
             and r.get("wall_ms") is not None]

    def r(value):
        return None if value is None else round(value, 2)

    return {
        "n": len(rows),
        "answered": len(taken),
        "errors": sum(1 for row in rows if row.get("error")),
        "p50_ms": r(percentile(taken, 0.50)),
        "p95_ms": r(percentile(taken, 0.95)),
        "min_ms": r(min(taken)) if taken else None,
        "max_ms": r(max(taken)) if taken else None,
        "mean_ms": r(sum(taken) / len(taken)) if taken else None,
    }


def accuracy_block(rows) -> dict:
    """The aggregate error rate, pooled over words rather than averaged over clips.

    Pooling is the standard definition and the honest one: a mean of ten per-clip
    rates weights a four-word clip like a twenty-word one. ``clips_exact`` counts the
    clips transcribed with no edits at all, which is the number a user feels.
    """
    scored = [r for r in rows if not r.get("error") and r.get("edits") is not None]
    ref_words = sum(r.get("reference_words") or 0 for r in scored)
    edits = sum(r.get("edits") or 0 for r in scored)
    pooled = rate(edits, ref_words)
    strict_edits = sum((r.get("substitutions") or 0) for r in scored)
    per_clip = [r["wer"] for r in scored if r.get("wer") is not None]
    orthographic = [r["wer_orthographic"] for r in scored
                    if r.get("wer_orthographic") is not None]
    return {
        "clips": len(rows),
        "scored": len(scored),
        "reference_words": ref_words,
        "edits": edits,
        "substitutions": strict_edits,
        "deletions": sum((r.get("deletions") or 0) for r in scored),
        "insertions": sum((r.get("insertions") or 0) for r in scored),
        "wer": None if pooled is None else round(pooled, 6),
        "wer_orthographic": (round(sum(orthographic) / len(orthographic), 6)
                             if orthographic else None),
        "wer_per_clip_mean": (round(sum(per_clip) / len(per_clip), 6)
                              if per_clip else None),
        "wer_max": round(max(per_clip), 6) if per_clip else None,
        "clips_exact": sum(1 for r in scored if r.get("edits") == 0),
    }


def rtf_block(rows) -> dict:
    """Real-time factor over the run: the pooled figure and the per-clip spread.

    ``pooled`` is total wall over total audio, which is what a batch of clips costs;
    ``p50`` and ``max`` are the per-clip spread, which is what one caller waits.
    """
    scored = [r for r in rows if not r.get("error") and r.get("rtf") is not None]
    wall = sum((r.get("wall_ms") or 0) for r in scored) / 1000.0
    audio = sum((r.get("audio_seconds") or 0) for r in scored)
    values = [r["rtf"] for r in scored]
    return {
        "scored": len(scored),
        "audio_seconds": round(audio, 3) if audio else None,
        "wall_seconds": round(wall, 3) if scored else None,
        "pooled": round(wall / audio, 4) if audio else None,
        "p50": (None if not values else round(percentile(values, 0.50), 4)),
        "min": round(min(values), 4) if values else None,
        "max": round(max(values), 4) if values else None,
    }


def normalizer_block() -> dict:
    """What the record says about the folding, because the rate depends on it."""
    return {
        "id": NORMALIZER_ID,
        "version": NORMALIZER_VERSION,
        "folds": ["case", "unicode NFKC", "punctuation", "whitespace",
                  "ordinal suffixes", "number words to digits"],
        "orthographic_folds": ["case", "unicode NFKC", "punctuation", "whitespace"],
        "source": "ainode/bench/speech/metrics.py",
    }


__all__ = ["NORMALIZER_ID", "NORMALIZER_VERSION", "accuracy_block", "clip_row",
           "edit_counts", "latency_block", "normalize", "normalizer_block",
           "percentile", "rate", "real_time_factor", "rtf_block"]
