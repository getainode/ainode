"""``scripts/ainode-bench.py speech ...`` - the command line for the speech bench.

A sixth subcommand next to ``harness``, ``agentic``, ``decide`` and ``embed``, asking
the question none of them do: can this model hear. How many words does it get wrong,
how long does one clip take, and does it transcribe faster than the clip plays.

    scripts/ainode-bench.py speech \\
        --endpoint http://100.122.26.9:3000/v1 \\
        --ainode http://100.122.26.9:3000 \\
        --model openai/whisper-large-v3-turbo \\
        --label "Spark-4 stacked beside Nemotron, via the fleet endpoint"

The endpoint can be an engine's own port or an AINode node's ``:3000/v1``, which routes
to whichever node serves the model; the record says which was measured, because the
fleet hop is part of the figure. ``--dry-run`` prints the plan and one example request
and touches nothing: no request, no file. ``--generate-clips`` rebuilds the committed
audio from the manifest with macOS ``say`` and writes nothing else; it is a maintenance
step, never part of a run. The API key is never printed, only where it came from.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from ainode.bench.speech.clips import (
    CLIPS,
    ClipError,
    clips_block,
    clips_dir,
    generate_clips,
    load_clips,
)
from ainode.bench.speech.client import (
    DEFAULT_API_KEY,
    DEFAULT_TIMEOUT,
    SpeechClient,
    SpeechError,
)
from ainode.bench.speech.runner import (
    SOURCE,
    build_notes,
    build_record,
    build_speech_block,
    pct,
    print_table,
    run_clips,
    run_transport_floor,
    run_warmup,
)


def say(message) -> None:
    """Print a line and flush it: a run is watched, often through a redirect."""
    print(message, flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ainode-bench speech",
        description="measure a speech-to-text model: word error rate against known "
                    "text, latency per clip, and real-time factor")
    p.add_argument("--endpoint", default="",
                   help="OpenAI-compatible base with its /v1, e.g. "
                        "http://100.122.26.9:3000/v1 to go through the fleet router or "
                        "http://100.72.9.84:8002/v1 for the engine itself")
    p.add_argument("--model", default="", help="model id exactly as served")
    p.add_argument("--ainode", default="",
                   help="AINode web base for placement, e.g. http://host:3000. "
                        "Omitted means the record carries no placement rather than a "
                        "guessed one")
    p.add_argument("--label", default="",
                   help="free text: what makes this run distinct")
    p.add_argument("--clips", default="",
                   help="directory holding the committed WAVs (default: the repo's "
                        "bench/speech/clips, or $AINODE_SPEECH_CLIPS)")
    p.add_argument("--language", default="",
                   help="language hint sent as a form field. Omitted by default, so "
                        "the run measures the detection an ordinary caller gets")
    p.add_argument("--translate", action="store_true",
                   help="measure /v1/audio/translations instead of transcriptions. "
                        "Whisper turbo is a transcription model and cannot translate; "
                        "this is for the ASR models that can")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                   help=f"seconds per request (default {DEFAULT_TIMEOUT})")
    p.add_argument("--api-key", default="",
                   help=f"bearer token for the endpoint (default {DEFAULT_API_KEY}). "
                        "Never printed and never written into a record")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and one example request, write nothing")
    p.add_argument("--generate-clips", action="store_true",
                   help="rebuild the committed WAVs from the manifest with macOS `say` "
                        "and `afconvert`, then exit. A maintenance step: regenerating "
                        "audio changes what every rate was taken over, so bump "
                        "CLIPS_VERSION when you do")
    return p


def describe(args, client):
    """``(model_block, placement, warnings)`` for the record.

    The other sections' rule, for the same reason: the serving node is a fact about the
    cluster, so a run that was not told where the control plane is records no placement
    rather than stamping the endpoint's host as the node. The fleet view names the node
    actually serving the model, which for a stacked speech instance is never the master
    by default.
    """
    model_id = client.reported_model or client.model
    if not args.ainode:
        return {"id": model_id}, {}, []
    from ainode.bench.fleet import describe_via_http, resolve_serving_node

    base = args.ainode.rstrip("/")
    node_name, engine_port, gpu_name, resolve_warn = resolve_serving_node(base,
                                                                         client.model)
    # The resolved name is passed through so `stacked_with` names what shares the GPU
    # with THIS instance. A speech model is a stacked instance on a peer by design, so
    # a run driven at the master would otherwise record the master's own neighbours.
    model_block, placement, _node_id, warnings = describe_via_http(
        base, base, client.model, serving_node_name=node_name)
    if node_name:
        placement["node"] = node_name
        placement["port"] = engine_port
        if gpu_name:
            placement["gpu"] = gpu_name
    warnings = list(warnings)
    if resolve_warn:
        warnings.insert(0, resolve_warn)
    return model_block, placement, warnings


def settings_for(args, client, clips: list) -> dict:
    block = clips_block(clips)
    return {"endpoint": client.endpoint,
            "model_requested": client.model,
            "path": f"POST /v1/audio/{client.path_name}",
            "clips": block["id"],
            "clips_version": block["version"],
            "clips_directory": block["directory"],
            "clip_count": block["clips"],
            "audio_seconds": block["audio_seconds"],
            "language": args.language or "",
            "timeout_s": args.timeout}


def record_path(out_dir, stamp: str, model_id: str, label: str) -> pathlib.Path:
    """``<stamp>-<model-slug>-<label-slug>-speech.json``, the naming the other
    sections use, with their slug helper."""
    from ainode.bench.measure import slug

    return out_dir / f"{stamp}-{slug(model_id)}-{slug(label)}-speech.json"


def dry_run(args, client, clips: list, out=say) -> int:
    block = clips_block(clips)
    out(f"\n  ainode-bench speech  {client.model}")
    out(f"  endpoint : {client.endpoint}")
    out(f"  path     : POST /v1/audio/{client.path_name}")
    out(f"  key      : {'--api-key' if args.api_key else 'the default'} "
        "(never printed)")
    out(f"  clips    : {block['clips']} from {block['directory']} "
        f"({block['id']} v{block['version']}, {block['audio_seconds']}s of audio, "
        f"{block['reference_words']} reference words)")
    for clip in clips:
        out(f"    {clip['id']:<10}{clip['voice']:<10}{clip['locale']:<8}"
            f"{clip['seconds']:5.2f}s  {clip['bytes']:7d} B  {clip['text'][:52]}")
    first = clips[0]
    out(f"  example  : {client.request(first, client.read_audio(first)).curl_safe()}")
    out("\n  dry run: nothing was requested and no file was written")
    return 0


def progress(clip, done, count, row) -> None:
    """One line per clip: it is ten of them, and each one is a sentence worth seeing."""
    if row.get("error"):
        say(f"    {clip['id']:<10}{done:2d}/{count}  err  "
            f"{str(row['error'])[:60]}")
        return
    say(f"    {clip['id']:<10}{done:2d}/{count}  {row.get('wall_ms', 0):6.0f} ms  "
        f"WER {pct(row.get('wer')):<7} {(row.get('transcript') or '')[:46]}")


def main(argv=None, out_dir=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    out_dir = pathlib.Path(out_dir) if out_dir else pathlib.Path.cwd() / "bench" / "results"

    if args.generate_clips:
        target = pathlib.Path(args.clips) if args.clips else clips_dir()
        say(f"\n  rebuilding {len(CLIPS)} clips into {target}")
        try:
            written = generate_clips(target)
        except Exception as exc:
            p.error(f"--generate-clips needs macOS `say` and `afconvert`: {exc}")
        total = sum(path.stat().st_size for path in written)
        say(f"  wrote {len(written)} WAVs, {total} bytes total")
        say("  bump CLIPS_VERSION in ainode/bench/speech/clips.py: a rate taken over "
            "different audio is a different number under the same name")
        return 0

    if not args.dry_run and not args.label:
        p.error("--label is required")
    if args.timeout <= 0:
        p.error("--timeout must be positive")

    try:
        clips = load_clips(args.clips or None)
    except ClipError as exc:
        return p.error(str(exc))
    try:
        client = SpeechClient(args.endpoint, args.model,
                              api_key=args.api_key or DEFAULT_API_KEY,
                              timeout=args.timeout, language=args.language,
                              translate=args.translate)
    except SpeechError as exc:
        return p.error(str(exc))

    if args.dry_run:
        return dry_run(args, client, clips)

    block = clips_block(clips)
    say(f"\n  ainode-bench speech  {client.model}")
    say(f"  label    : {args.label}")
    say(f"  endpoint : {client.endpoint}")
    say(f"  clips    : {block['clips']} ({block['audio_seconds']}s of audio, "
        f"{len(block['voices'])} voices, {len(block['locales'])} locales)")
    say("  measures : word error rate against the text the clips were made from, "
        "latency per clip, and real-time factor")

    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    floor_ms = run_transport_floor(client)
    say("    warmup     one untimed request, transcript discarded")
    warmup = run_warmup(client, clips[0])
    say(f"    warmup     {warmup['clip']} {warmup['wall_ms']:.0f} ms"
        + (f"  ERROR {warmup['error']}" if warmup.get("error") else ""))
    started = time.time()
    rows = run_clips(client, clips, progress=progress)
    seconds = time.time() - started

    speech_block = build_speech_block(client, clips, rows, seconds, floor_ms=floor_ms,
                                      warmup=warmup)
    model_block, placement, warnings = describe(args, client)
    for warning in warnings:
        say(f"  warn     : {warning}")
    notes = build_notes(client, speech_block, seconds) + list(warnings)
    record = build_record(args.label, model_block, placement, speech_block,
                          settings_for(args, client, clips), notes, stamp)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = record_path(out_dir, stamp, model_block.get("id") or client.model,
                       args.label)
    path.write_text(json.dumps(record, indent=1) + "\n")
    title = (f"{model_block.get('name') or model_block.get('id')}  "
             f"({placement.get('node') or 'placement not read'})")
    print_table(speech_block, title, out=say)
    say(f"\n  saved {path}")
    say("  render the README table with: python3 scripts/render-bench-table.py")
    return 0


def run() -> int:
    try:
        return main()
    except KeyboardInterrupt:
        print("\n  interrupted; nothing was written")
        return 130


__all__ = ["SOURCE", "build_parser", "describe", "dry_run", "main", "progress",
           "record_path", "run", "settings_for"]
