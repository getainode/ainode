"""``scripts/ainode-bench.py embed ...`` - the command line for the embedding bench.

A fifth subcommand next to ``harness``, ``agentic`` and ``decide``, asking the
question none of them do: what is this embedding model like to build a retrieval
pipeline on. How wide is the vector, how long does one lookup take, how much does
batching buy, and do the vectors separate meaning at all.

    scripts/ainode-bench.py embed \\
        --endpoint http://100.72.9.84:8001/v1 \\
        --ainode http://100.72.9.84:3000 \\
        --model Qwen/Qwen3-Embedding-0.6B \\
        --label "Spark-4 stacked beside Nemotron"

The endpoint can be an engine's own port or an AINode node's ``:3000/v1``, which
routes to whichever node serves the model; the record says which was measured.
``--dry-run`` prints the plan and one example request and touches nothing: no
request, no file. The API key is never printed, only where it came from.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from ainode.bench import auth
from ainode.bench.embed.client import (
    DEFAULT_API_KEY,
    DEFAULT_TIMEOUT,
    EmbedClient,
    EmbedError,
)
from ainode.bench.embed.corpus import (
    BATCH_SIZES,
    LATENCY_TEXTS,
    PAIR_TEXTS,
    TEXTS_PER_BATCH_SIZE,
    corpus_block,
)
from ainode.bench.embed.runner import (
    SOURCE,
    build_embed_block,
    build_notes,
    build_record,
    print_table,
    run_latency,
    run_quality,
    run_throughput,
)


def say(message) -> None:
    """Print a line and flush it: a run is watched, often through a redirect."""
    print(message, flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ainode-bench embed",
        description="measure an embedding model: dimensions, single-request latency, "
                    "throughput by batch size, and whether the vectors separate "
                    "meaning")
    p.add_argument("--endpoint", default="",
                   help="OpenAI-compatible base with its /v1, e.g. "
                        "http://100.72.9.84:8001/v1 for the engine itself or "
                        "http://100.72.9.84:3000/v1 to go through the fleet router")
    p.add_argument("--model", default="", help="model id exactly as served")
    p.add_argument("--ainode", default="",
                   help="AINode web base for placement, e.g. http://host:3000. "
                        "Omitted means the record carries no placement rather than a "
                        "guessed one")
    p.add_argument("--label", default="",
                   help="free text: what makes this run distinct")
    p.add_argument("--batches", default=",".join(str(b) for b in BATCH_SIZES),
                   help="comma list of batch sizes for the throughput sweep "
                        f"(default {','.join(str(b) for b in BATCH_SIZES)})")
    p.add_argument("--texts-per-batch", type=int, default=TEXTS_PER_BATCH_SIZE,
                   help="texts pushed through EACH batch size, so the rows compare "
                        f"(default {TEXTS_PER_BATCH_SIZE})")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                   help=f"seconds per request (default {DEFAULT_TIMEOUT})")
    p.add_argument("--api-key", default="",
                   help=f"bearer token for the endpoint (default ${auth.ENV_API_KEY}, "
                        f"else the placeholder {DEFAULT_API_KEY} that an open node "
                        "accepts). Never printed and never written into a record")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and one example request, write nothing")
    return p


def int_list(text: str) -> list:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def describe(args, client):
    """``(model_block, placement, warnings)`` for the record.

    The agentic and decision benches' rule, for the same reason: the serving node is
    a fact about the cluster, so a run that was not told where the control plane is
    records no placement rather than stamping the endpoint's host as the node. The
    fleet view names the node actually serving the model, which for a stacked
    embedding instance is never the master by default.
    """
    model_id = client.reported_model or client.model
    if not args.ainode:
        return {"id": model_id}, {}, []
    from ainode.bench.fleet import describe_via_http, resolve_serving_node

    base = args.ainode.rstrip("/")
    node_name, engine_port, gpu_name, resolve_warn = resolve_serving_node(
        base, client.model, api_key=client.api_key)
    model_block, placement, _node_id, warnings = describe_via_http(
        base, base, client.model, api_key=client.api_key)
    if node_name:
        placement["node"] = node_name
        placement["port"] = engine_port
        if gpu_name:
            placement["gpu"] = gpu_name
    warnings = list(warnings)
    if resolve_warn:
        warnings.insert(0, resolve_warn)
    return model_block, placement, warnings


def settings_for(args, client, batches: list) -> dict:
    return {"endpoint": client.endpoint,
            "model_requested": client.model,
            "corpus": corpus_block()["id"],
            "corpus_version": corpus_block()["version"],
            "latency_texts": len(LATENCY_TEXTS),
            "batches": list(batches),
            "texts_per_batch": int(args.texts_per_batch),
            "timeout_s": args.timeout}


def record_path(out_dir, stamp: str, model_id: str, label: str) -> pathlib.Path:
    """``<stamp>-<model-slug>-<label-slug>-embed.json``, the naming the other
    sections use, with their slug helper."""
    from ainode.bench.measure import slug

    return out_dir / f"{stamp}-{slug(model_id)}-{slug(label)}-embed.json"


def dry_run(args, client, batches: list, out=say) -> int:
    out(f"\n  ainode-bench embed  {client.model}")
    out(f"  endpoint : {client.endpoint}")
    out(f"  key      : from {client.key_source} (never printed)")
    out(f"  latency  : {len(LATENCY_TEXTS)} single-text requests")
    for batch in batches:
        requests = max(1, (args.texts_per_batch + batch - 1) // batch)
        out(f"  batch {batch:<4}: {requests} request(s) of {batch} text(s)")
    out(f"  quality  : {len(PAIR_TEXTS)} texts in one request, 6 pairs scored")
    out(f"  example  : {client.request(list(LATENCY_TEXTS[:1])).curl_safe()}")
    out("\n  dry run: nothing was requested and no file was written")
    return 0


def progress(stage, done, count, reply) -> None:
    """One line every ten requests, and always the last one of a stage."""
    if done != count and done % 10:
        return
    mark = "err" if reply.error else f"{reply.wall_ms:.0f} ms"
    say(f"    {stage:<10} {done:3d}/{count}  {mark}")


def main(argv=None, out_dir=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    out_dir = pathlib.Path(out_dir) if out_dir else pathlib.Path.cwd() / "bench" / "results"

    if not args.dry_run and not args.label:
        p.error("--label is required")
    if args.timeout <= 0:
        p.error("--timeout must be positive")
    if args.texts_per_batch < 1:
        p.error("--texts-per-batch must be at least 1")
    try:
        batches = int_list(args.batches)
    except ValueError:
        return p.error("--batches is a comma list of integers, e.g. 1,16,64")
    if not batches or any(b < 1 for b in batches):
        p.error("--batches must be positive integers, e.g. 1,16,64")

    key, key_source = auth.key_for(args.api_key, DEFAULT_API_KEY)
    try:
        client = EmbedClient(args.endpoint, args.model, api_key=key,
                             timeout=args.timeout, key_source=key_source)
    except EmbedError as exc:
        return p.error(str(exc))

    if args.dry_run:
        return dry_run(args, client, batches)

    refused = auth.preflight(client.endpoint, client.api_key)
    if refused:
        return auth.stop(refused, out=say)

    say(f"\n  ainode-bench embed  {client.model}")
    say(f"  label    : {args.label}")
    say(f"  endpoint : {client.endpoint}")
    say(f"  key      : from {client.key_source} (never printed)")
    say("  measures : dimensions, single-request p50/p95, texts and tokens per "
        "second by batch size, and 6 pairs checked for related above unrelated")

    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    started = time.time()
    try:
        latency, dimensions, latency_errors = run_latency(client, progress=progress)
        throughput, throughput_errors = run_throughput(
            client, sizes=batches, per_size=args.texts_per_batch, progress=progress)
        quality, quality_errors = run_quality(client)
    except auth.EndpointRefused as exc:
        # Mid-run, which the preflight cannot rule out: the limiter can start
        # refusing at the batch-of-64 sweep. Nothing is written.
        return auth.stop(str(exc), out=say)
    seconds = time.time() - started

    errors = list(latency_errors) + list(throughput_errors) + list(quality_errors)
    block = build_embed_block(client, latency, throughput, quality, dimensions,
                              errors, seconds)
    model_block, placement, warnings = describe(args, client)
    for warning in warnings:
        say(f"  warn     : {warning}")
    notes = build_notes(client, block, seconds) + list(warnings)
    record = build_record(args.label, model_block, placement, block,
                          settings_for(args, client, batches), notes, stamp)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = record_path(out_dir, stamp, model_block.get("id") or client.model,
                       args.label)
    path.write_text(json.dumps(record, indent=1) + "\n")
    title = (f"{model_block.get('name') or model_block.get('id')}  "
             f"({placement.get('node') or 'placement not read'})")
    print_table(block, title, out=say)
    say(f"\n  saved {path}")
    say("  render the README table with: python3 scripts/render-bench-table.py")
    return 0


def run() -> int:
    try:
        return main()
    except KeyboardInterrupt:
        print("\n  interrupted; nothing was written")
        return 130


__all__ = ["SOURCE", "build_parser", "describe", "dry_run", "int_list", "main",
           "progress", "record_path", "run", "settings_for"]
