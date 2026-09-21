"""The measurement, the table it prints, and the record it lands in.

A run writes a ``speech`` block and no ``results`` block, for the reason the harness,
agentic, decision and embedding runs write theirs: it measured a transcription model
and took no tok/s, so a zero in ``single_stream`` would be a number nobody took.
``scripts/render-bench-table.py`` keeps a record shaped like that out of the README's
throughput table and gives it a row in "Speech runs" instead.

The honesty rules are the ones the rest of ``bench/`` runs under:

  * **Nothing is loaded, unloaded or restarted.** The run drives whatever is already
    serving, and adds real load to it.
  * **A request that failed says why.** It is one row with an ``error``, named in the
    notes, and it is counted out of every rate, percentile and factor. A failed clip
    is never folded in as a 100 percent error rate: a transport failure in a figure a
    reader takes as the model's is the one mistake this section can make.
  * **A measurement nobody took is absent.** No clip answered means a null error rate,
    not a 1.0, and a clip whose duration could not be read has no real-time factor
    rather than one computed from a guess.
  * **The reference is the text that was spoken, fixed before the run.** It is the
    string ``say`` was given, committed in ``clips.py``, and nothing adjusts it after
    a transcript is seen.
  * **Both error rates are reported.** The normaliser is part of the measurement, so
    the record carries the number with number words folded to digits and the stricter
    orthographic one beside it, plus the normaliser's own id and version.
  * **The one warm-up request is recorded, not hidden.** A vLLM speech engine running
    ``--enforce-eager`` compiles its kernels on the FIRST real transcription, which on a
    GB10 measured 89 seconds against 0.7 for every one after it. Leaving that inside the
    timed set would put a one-time compile in a p50 and in a real-time factor a reader
    takes as steady state; dropping it silently would hide a cost a user meets once per
    launch. So one untimed request is sent first, its transcript discarded, and its wall
    time recorded as ``warmup`` beside the figures it is deliberately out of.

Stdlib only: blocking urllib calls in sequence. Sequential on purpose, because what
is being measured is what one clip costs the engine, not how many uploads can be in
flight at once.
"""
from __future__ import annotations

from ainode.bench.speech import metrics
from ainode.bench.speech.clips import clips_block

SCHEMA = 1
SOURCE = "scripts/ainode-bench.py speech"

#: GETs taken for the transport floor. Five is enough for a median and costs nothing;
#: the floor is a property of the link, not something that needs a sweep.
FLOOR_PROBES = 5


def run_transport_floor(client, probes: int = FLOOR_PROBES):
    """Median wall ms of a request that transcribes nothing, or None if none answered.

    ``GET /v1/models`` over the same link as the timed calls. Recorded beside the
    latency and the real-time factor because a 900 ms clip means two different things
    depending on whether 30 ms or 300 ms of it was the wire, and a reader cannot tell
    from the number alone which machine the bench was driven from.
    """
    taken = [ms for ms in (client.ping() for _ in range(max(1, int(probes))))
             if ms is not None]
    if not taken:
        return None
    return round(metrics.percentile(taken, 0.5), 2)


def run_warmup(client, clip: dict) -> dict:
    """One untimed transcription whose answer is thrown away. Returns what it cost.

    Not an optimisation and not a courtesy to the engine: it is the only way the timed
    figures can be read as steady state. An eager-mode speech engine compiles on its
    first real transcription, so without this the first clip carries a one-time cost
    that lands in the p50, the maximum and the pooled real-time factor. The cost is
    recorded rather than dropped, because a user meets it once per launch and a record
    that hid it would be describing an engine nobody starts.
    """
    reply = client.transcribe(clip, audio=client.read_audio(clip))
    return {"clip": clip["id"], "wall_ms": reply.wall_ms,
            "error": reply.error,
            "note": "one untimed request, its transcript discarded: an eager-mode "
                    "engine compiles on its first transcription, and that cost belongs "
                    "beside the figures rather than inside them"}


def run_clips(client, clips: list, progress=None) -> list:
    """One upload per clip, in manifest order, timed. Returns the rows.

    Sequential, so the per-clip latency and real-time factor describe a request that
    had the engine to itself rather than a queue this bench created. The audio is read
    once per clip and handed to the client, so the file read is outside the timing.
    """
    rows = []
    for index, clip in enumerate(clips, start=1):
        audio = client.read_audio(clip)
        reply = client.transcribe(clip, audio=audio)
        rows.append(metrics.clip_row(clip, reply.text, reply.wall_ms,
                                     error=reply.error))
        if progress:
            progress(clip, index, len(clips), rows[-1])
    return rows


def build_speech_block(client, clips: list, rows: list, seconds: float,
                       floor_ms=None, warmup=None) -> dict:
    """The record's ``speech`` block. See bench/SCHEMA.md."""
    latency = metrics.latency_block(rows)
    latency["transport_floor_ms"] = floor_ms
    return {
        "endpoint": client.endpoint,
        "path": f"POST /v1/audio/{client.path_name}",
        "model_reported": client.reported_model or None,
        "warmup": warmup,
        "clips": clips_block(clips),
        "normalizer": metrics.normalizer_block(),
        "protocol": client.protocol(),
        "accuracy": metrics.accuracy_block(rows),
        "latency": latency,
        "rtf": metrics.rtf_block(rows),
        "rows": rows,
        "errors": [{"id": r["id"], "error": r["error"]} for r in rows if r.get("error")],
        "seconds": round(float(seconds), 1),
    }


def pct(value, places=1):
    """A rate as a percentage string, or "-" when nobody measured it."""
    return "-" if value is None else f"{value * 100:.{places}f}%"


def build_notes(client, block: dict, seconds: float, source: str = SOURCE) -> list:
    """The notes a reader needs to know what the numbers are and are not."""
    accuracy = block.get("accuracy") or {}
    latency = block.get("latency") or {}
    rtf = block.get("rtf") or {}
    clips = block.get("clips") or {}
    errors = block.get("errors") or []
    notes = [
        f"Measured by {source} in {round(seconds)}s against {client.endpoint}; "
        "nothing was loaded, unloaded or restarted, and the run adds real load to "
        "whatever else that node is serving.",
        f"{clips.get('clips', 0)} clips, {clips.get('audio_seconds')}s of audio in "
        f"{len(clips.get('voices') or [])} voices across "
        f"{len(clips.get('locales') or [])} English locales "
        f"({', '.join(clips.get('locales') or [])}), mono 16-bit PCM at 16 kHz. They "
        f"are committed audio ({clips.get('directory')}, set "
        f"{clips.get('id')} v{clips.get('version')}), not synthesised per run, because "
        "a word error rate is only comparable over the same bytes.",
        "The reference is the exact text handed to macOS `say` to make each clip, "
        "fixed before the run and never adjusted after a transcript was seen: a "
        "reference edited to match what a model said would make the rate a statement "
        "about the editor.",
        (f"Word error rate is {pct(accuracy.get('wer'))} pooled over "
         f"{accuracy.get('reference_words')} reference words "
         f"({accuracy.get('edits')} edits: {accuracy.get('substitutions')} "
         f"substitutions, {accuracy.get('deletions')} deletions, "
         f"{accuracy.get('insertions')} insertions), and "
         f"{accuracy.get('clips_exact')} of {accuracy.get('scored')} clips came back "
         "word for word. Pooled over words rather than averaged over clips, which is "
         "the standard definition and the honest one: a mean of per-clip rates weights "
         "a short clip like a long one."),
        (f"That rate folds case, punctuation, ordinal suffixes and number words to "
         f"digits (normaliser {(block.get('normalizer') or {}).get('id')} "
         f"v{(block.get('normalizer') or {}).get('version')}), so a transcript that "
         f"wrote \"9\" where the reference says \"nine\" is not scored wrong for its "
         f"spelling. The stricter orthographic rate, case and punctuation only, is "
         f"{pct(accuracy.get('wer_orthographic'))}, and the gap between the two is how "
         "much of the error was spelling rather than hearing."),
        (f"Latency is one clip per request, sent one at a time: p50 "
         f"{latency.get('p50_ms')} ms, p95 {latency.get('p95_ms')} ms over "
         f"{latency.get('answered')} of {latency.get('n')} clips, end to end from "
         "wherever the bench ran."),
        ("Those figures are end to end, and transport_floor_ms is the measured floor: "
         f"{latency.get('transport_floor_ms')} ms for a GET /v1/models over the same "
         "link, which transcribes nothing. Subtract it to read the engine's own time; "
         "a run driven from the node itself would report a floor near zero."
         if latency.get("transport_floor_ms") is not None else
         "The transport floor could not be measured, so the latency and real-time "
         "figures are end to end with an unknown share of the wire in them."),
        (f"Real-time factor is {rtf.get('pooled')} pooled ({rtf.get('wall_seconds')}s "
         f"of wall over {rtf.get('audio_seconds')}s of audio), p50 {rtf.get('p50')} "
         f"and worst {rtf.get('max')} per clip. Below 1 means the engine transcribes "
         "faster than the clip plays, which is what decides whether a live stream can "
         "be kept up with; the wire is in it, so read it against the floor above."),
    ]
    warmup = block.get("warmup") or {}
    if warmup.get("wall_ms") is not None:
        notes.append(
            f"One untimed warm-up request came first ({warmup.get('clip')}, "
            f"{warmup.get('wall_ms')} ms) and its transcript was discarded. An "
            "eager-mode speech engine compiles its kernels on the first real "
            "transcription, so that cost is recorded here rather than left inside a p50 "
            "or a real-time factor a reader takes as steady state. A user meets it once "
            "per launch.")
    elif warmup.get("error"):
        notes.append(f"The warm-up request failed ({warmup.get('error')}), so the first "
                     "timed clip may carry the engine's one-time compile.")
    if errors:
        shown = ", ".join(f"{e['id']}: {str(e.get('error'))[:70]}" for e in errors[:3])
        notes.append(f"{len(errors)} clip(s) failed on transport or an unreadable "
                     f"response rather than on the words: {shown}"
                     f"{', ...' if len(errors) > 3 else ''}. They are counted out of "
                     "every rate, percentile and factor above, never folded in as a "
                     "100 percent error rate.")
    return notes


def build_record(label: str, model_block: dict, placement: dict, speech_block: dict,
                 settings: dict, notes: list, stamp: str,
                 source: str = SOURCE) -> dict:
    """A schema-1 record with a ``speech`` block and no ``results`` block."""
    return {"schema": SCHEMA, "stamp": stamp, "label": label, "model": model_block,
            "placement": placement, "settings": settings, "speech": speech_block,
            "notes": notes, "source": source}


# ---------------------------------------------------------------- printing

def fmt(value, places=2):
    return "-" if value is None else f"{value:.{places}f}"


def print_table(block: dict, title: str, out=print) -> None:
    """The per-clip rows, then the three aggregates."""
    accuracy = block.get("accuracy") or {}
    latency = block.get("latency") or {}
    rtf = block.get("rtf") or {}
    out(f"\n  {title}")
    out(f"  path       : {block.get('path')}")
    out(f"  clips      : {(block.get('clips') or {}).get('clips')} "
        f"({(block.get('clips') or {}).get('audio_seconds')}s of audio)")
    warmup = block.get("warmup") or {}
    if warmup:
        out(f"  warmup     : {warmup.get('clip')} "
            f"{fmt(warmup.get('wall_ms'), 0)} ms, discarded"
            + (f" (error: {warmup['error']})" if warmup.get("error") else ""))

    out("\n  clip      voice       words  WER      ms       RTF     transcript")
    for row in block.get("rows") or []:
        if row.get("error"):
            out(f"  {row['id']:<10}{str(row.get('voice') or '-'):<12}"
                f"{str(row.get('reference_words') or '-'):<7}"
                f"{'err':<9}{fmt(row.get('wall_ms'), 0):<9}{'-':<8}"
                f"{str(row.get('error'))[:44]}")
            continue
        out(f"  {row['id']:<10}{str(row.get('voice') or '-'):<12}"
            f"{row.get('reference_words'):<7}{pct(row.get('wer')):<9}"
            f"{fmt(row.get('wall_ms'), 0):<9}{fmt(row.get('rtf'), 3):<8}"
            f"{(row.get('transcript') or '')[:44]}")

    out(f"\n  word error rate  {pct(accuracy.get('wer'))} pooled over "
        f"{accuracy.get('reference_words')} words "
        f"({accuracy.get('edits')} edits = {accuracy.get('substitutions')}S + "
        f"{accuracy.get('deletions')}D + {accuracy.get('insertions')}I)")
    out(f"  orthographic     {pct(accuracy.get('wer_orthographic'))} "
        f"(case and punctuation only, no number folding)")
    out(f"  exact clips      {accuracy.get('clips_exact')} of "
        f"{accuracy.get('scored')}")
    out(f"  latency ms       p50 {fmt(latency.get('p50_ms'), 0)}  p95 "
        f"{fmt(latency.get('p95_ms'), 0)}  min {fmt(latency.get('min_ms'), 0)}  max "
        f"{fmt(latency.get('max_ms'), 0)}  floor "
        f"{fmt(latency.get('transport_floor_ms'), 0)}")
    out(f"  real-time factor pooled {fmt(rtf.get('pooled'), 3)}  p50 "
        f"{fmt(rtf.get('p50'), 3)}  worst {fmt(rtf.get('max'), 3)}")


__all__ = ["FLOOR_PROBES", "SCHEMA", "SOURCE", "build_notes", "build_record",
           "build_speech_block", "fmt", "pct", "print_table", "run_clips",
           "run_transport_floor", "run_warmup"]
