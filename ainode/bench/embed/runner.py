"""The three measurements, the table they print, and the record they land in.

A run writes an ``embed`` block and no ``results`` block, for the reason the harness,
agentic and decision runs write theirs: it measured an embedding model and took no
tok/s, so a zero in ``single_stream`` would be a number nobody took.
``scripts/render-bench-table.py`` keeps a record shaped like that out of the README's
throughput table and gives it a row in "Embedding runs" instead.

The honesty rules are the ones the rest of ``bench/`` runs under:

  * **Nothing is loaded, unloaded or restarted.** The run drives whatever is already
    serving, and adds real load to it.
  * **A request that failed says why.** It is counted as an error and named in the
    notes, never folded into a latency percentile and never a silent drop.
  * **A measurement nobody took is absent.** No tokens reported means null tokens per
    second, not a zero, and a quality check that could not score every pair reports
    ``ordered: null`` rather than a failure.
  * **The quality check is a sanity check, and the record says so.** Six pairs is not
    a retrieval benchmark, and nothing here is presented as one.

Stdlib only: blocking urllib calls in sequence. Sequential on purpose, because the
thing being measured at each batch size is how much work the engine does per request,
not how many requests can be in flight at once.
"""
from __future__ import annotations

import time

from ainode.bench.embed import metrics
from ainode.bench.embed.corpus import (
    BATCH_SIZES,
    LATENCY_TEXTS,
    PAIR_TEXTS,
    PAIRS,
    TEXTS_PER_BATCH_SIZE,
    corpus_block,
    cycle_texts,
)

SCHEMA = 1
SOURCE = "scripts/ainode-bench.py embed"


#: GETs taken for the transport floor. Five is enough for a median and costs
#: nothing; the floor is a property of the link, not something that needs a sweep.
FLOOR_PROBES = 5


def run_transport_floor(client, probes: int = FLOOR_PROBES):
    """Median wall ms of a request that embeds nothing, or None if none answered.

    ``GET /v1/models`` over the same link as the timed calls. Recorded beside the
    percentiles because a p50 of 70 ms means two different things depending on
    whether 5 ms or 35 ms of it was the wire, and a reader cannot tell from the
    number alone which machine the bench was driven from.
    """
    taken = [ms for ms in (client.ping() for _ in range(max(1, int(probes))))
             if ms is not None]
    if not taken:
        return None
    return round(metrics.percentile(taken, 0.5), 2)


def run_latency(client, texts=LATENCY_TEXTS, progress=None) -> tuple:
    """One request per text, in order, timed. Returns ``(block, dimensions, errors)``.

    One text per request is the interactive shape: a lookup embeds the query it was
    just handed and waits. Sequential, so the percentiles describe a request that had
    the engine to itself rather than a queue this bench created.
    """
    wall = []
    failures = []
    dimensions = None
    for index, text in enumerate(texts, start=1):
        reply = client.embed([text])
        if reply.error:
            failures.append({"index": index, "error": reply.error})
        else:
            wall.append(reply.wall_ms)
            if dimensions is None:
                dimensions = reply.dimensions
        if progress:
            progress("latency", index, len(texts), reply)
    block = metrics.latency_block(wall, errors=len(failures))
    block["transport_floor_ms"] = run_transport_floor(client)
    return block, dimensions, failures


def run_throughput(client, sizes=BATCH_SIZES, per_size: int = TEXTS_PER_BATCH_SIZE,
                   progress=None) -> tuple:
    """The batch sweep: ``per_size`` texts through each batch size. ``(rows, errors)``.

    Every size does the same amount of work (64 texts), so the rows compare directly:
    64 requests at batch 1, 4 at batch 16, 1 at batch 64. The texts wrap around the
    corpus rather than repeating one string, which would measure the prefix cache.
    """
    rows = []
    failures = []
    for batch in sizes:
        batch = max(1, int(batch))
        requests = max(1, (per_size + batch - 1) // batch)
        tokens_total = 0
        tokens_seen = False
        texts_done = 0
        errors_here = 0
        started = time.monotonic()
        for number in range(1, requests + 1):
            payload = cycle_texts(batch)
            reply = client.embed(payload)
            if reply.error:
                errors_here += 1
                failures.append({"batch": batch, "request": number,
                                 "error": reply.error})
            else:
                texts_done += len(reply.vectors)
                if reply.tokens is not None:
                    tokens_total += reply.tokens
                    tokens_seen = True
            if progress:
                progress(f"batch {batch}", number, requests, reply)
        seconds = time.monotonic() - started
        rows.append(metrics.throughput_row(
            batch, requests, texts_done, seconds,
            tokens_total if tokens_seen else None, errors=errors_here))
    return rows, failures


def run_quality(client, pairs=PAIRS, texts=PAIR_TEXTS) -> tuple:
    """Every side of every pair in ONE request, then the cosines. ``(block, errors)``.

    One request on purpose: a pair whose two sides were embedded in different calls
    would be comparing across whatever the engine's batch did to them, and the point
    of the check is the vectors, not the batching.
    """
    reply = client.embed(list(texts))
    if reply.error:
        return ({"pairs": [], "related_min": None, "unrelated_max": None,
                 "margin": None, "ordered": None},
                [{"stage": "quality", "error": reply.error}])
    by_text = dict(zip(texts, reply.vectors))
    return metrics.quality_block(pairs, by_text), []


def build_embed_block(client, latency: dict, throughput: list, quality: dict,
                      dimensions, errors: list, seconds: float) -> dict:
    """The record's ``embed`` block. See bench/SCHEMA.md."""
    return {
        "endpoint": client.endpoint,
        "model_reported": client.reported_model or None,
        "dimensions": dimensions,
        "corpus": corpus_block(),
        "protocol": client.protocol(),
        "latency": latency,
        "throughput": throughput,
        "quality": quality,
        "errors": errors,
        "seconds": round(float(seconds), 1),
    }


def row_at(throughput: list, batch: int):
    """The sweep row for one batch size, or None when that size was not run."""
    for row in throughput or []:
        if row.get("batch") == batch:
            return row
    return None


def build_notes(client, block: dict, seconds: float, source: str = SOURCE) -> list:
    """The notes a reader needs to know what the numbers are and are not."""
    latency = block.get("latency") or {}
    quality = block.get("quality") or {}
    errors = block.get("errors") or []
    top = max((row.get("batch") or 0) for row in (block.get("throughput") or [{}])) or 0
    notes = [
        f"Measured by {source} in {round(seconds)}s against {client.endpoint}; "
        "nothing was loaded, unloaded or restarted, and the run adds real load to "
        "whatever else that node is serving.",
        f"Latency is {latency.get('n', 0)} single-text requests sent one at a time, "
        "so p50 and p95 describe a request with the engine to itself rather than a "
        "queue this bench created.",
        ("Those figures are end to end from wherever the bench ran, and "
         f"transport_floor_ms is the measured floor: "
         f"{latency.get('transport_floor_ms')} ms for a GET /v1/models over the same "
         "link, which embeds nothing. Subtract it to read the engine's own time; a "
         "run driven from the node itself would report a floor near zero."
         if latency.get("transport_floor_ms") is not None else
         "The transport floor could not be measured, so the latency figures are end "
         "to end with an unknown share of the wire in them."),
        "Throughput is texts and tokens over the wall time of the whole sweep at "
        f"each batch size, {block.get('corpus', {}).get('latency_texts', 0)} texts "
        "wrapped to fill every batch rather than one text repeated, which would "
        "measure the prefix cache instead of the engine.",
        "Tokens per second comes from usage.prompt_tokens as the engine reported it, "
        "not from a tokenizer run here; a response that reported no usage leaves the "
        "figure absent rather than zero.",
    ]
    if quality.get("ordered") is True:
        notes.append(
            f"The quality check passed: the lowest of the 3 related pairs "
            f"({quality.get('related_min')}) scores above the highest of the 3 "
            f"unrelated ones ({quality.get('unrelated_max')}), a margin of "
            f"{quality.get('margin')}. It is a sanity check on whether the vectors "
            "mean anything, not a retrieval benchmark: six hand-written pairs say "
            "nothing about recall on a real corpus, and MTEB is where that question "
            "belongs.")
    elif quality.get("ordered") is False:
        notes.append(
            f"The quality check FAILED: the lowest related pair "
            f"({quality.get('related_min')}) does not beat the highest unrelated one "
            f"({quality.get('unrelated_max')}). Vectors of the right width that do "
            "not separate meaning are the failure this check exists for: suspect the "
            "pooling mode, the served checkpoint or a truncated window before "
            "suspecting the model.")
    else:
        notes.append("The quality check did not score every pair, so it reports no "
                     "verdict rather than a failure.")
    if top:
        row = row_at(block.get("throughput") or [], top) or {}
        notes.append(f"At batch {top} the engine returned {row.get('texts_per_s')} "
                     f"texts/s and {row.get('tokens_per_s')} tokens/s; batch 1 is in "
                     "the same table, and the gap between them is what an indexer "
                     "gains by batching.")
    if errors:
        shown = ", ".join(str(e.get("error"))[:80] for e in errors[:3])
        notes.append(f"{len(errors)} request(s) failed on transport or an unreadable "
                     f"response rather than on the vectors: {shown}"
                     f"{', ...' if len(errors) > 3 else ''}. They are counted out of "
                     "every percentile and rate above.")
    return notes


def build_record(label: str, model_block: dict, placement: dict, embed_block: dict,
                 settings: dict, notes: list, stamp: str,
                 source: str = SOURCE) -> dict:
    """A schema-1 record with an ``embed`` block and no ``results`` block."""
    return {"schema": SCHEMA, "stamp": stamp, "label": label, "model": model_block,
            "placement": placement, "settings": settings, "embed": embed_block,
            "notes": notes, "source": source}


# ---------------------------------------------------------------- printing

def fmt(value, places=2):
    return "-" if value is None else f"{value:.{places}f}"


def print_table(block: dict, title: str, out=print) -> None:
    """The three measurements, as three short tables."""
    latency = block.get("latency") or {}
    out(f"\n  {title}")
    out(f"  dimensions : {block.get('dimensions') or 'not measured'}")
    out(f"  single request over {latency.get('answered', 0)} of "
        f"{latency.get('n', 0)} texts")
    out("  p50 ms   p95 ms   min ms   max ms   mean ms  floor ms")
    out(f"  {fmt(latency.get('p50_ms')):<9}{fmt(latency.get('p95_ms')):<9}"
        f"{fmt(latency.get('min_ms')):<9}{fmt(latency.get('max_ms')):<9}"
        f"{fmt(latency.get('mean_ms')):<9}{fmt(latency.get('transport_floor_ms'))}")

    out("\n  batch   requests   texts   seconds   texts/s   tokens   tokens/s")
    for row in block.get("throughput") or []:
        out(f"  {row.get('batch'):<8}{row.get('requests'):<11}{row.get('texts'):<8}"
            f"{fmt(row.get('seconds'), 2):<10}{fmt(row.get('texts_per_s')):<10}"
            f"{str(row.get('tokens') if row.get('tokens') is not None else '-'):<9}"
            f"{fmt(row.get('tokens_per_s'), 1)}")

    quality = block.get("quality") or {}
    verdict = {True: "PASS", False: "FAIL", None: "no verdict"}[quality.get("ordered")]
    out(f"\n  quality: 6 pairs, related above unrelated  {verdict}")
    for pair in quality.get("pairs") or []:
        kind = "related " if pair.get("related") else "unrelated"
        out(f"    {pair['id']:<9} {kind}  cos {fmt(pair.get('cosine'), 4)}   "
            f"{pair['a'][:44]}")
    if quality.get("ordered") is not None:
        out(f"    lowest related {fmt(quality.get('related_min'), 4)} vs highest "
            f"unrelated {fmt(quality.get('unrelated_max'), 4)}, margin "
            f"{fmt(quality.get('margin'), 4)}")


__all__ = ["FLOOR_PROBES", "SCHEMA", "SOURCE", "build_embed_block", "build_notes",
           "build_record", "fmt", "print_table", "row_at", "run_latency",
           "run_quality", "run_throughput", "run_transport_floor"]
