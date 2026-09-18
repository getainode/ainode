"""``scripts/ainode-bench.py decide ...`` - the command line for the decision bench.

A fourth subcommand next to ``harness`` and ``agentic``, asking the fourth question:
not how fast the model generates, not whether it can drive a coding agent, not
whether it can hold an agent loop together, but whether its typed decisions can be
trusted by code that acts on them. Accuracy, calibration, latency, cost.

    scripts/ainode-bench.py decide --backend jev --label "jev-latest, 110 items"

    scripts/ainode-bench.py decide --backend chat \\
        --endpoint http://100.122.26.9:3000/v1 \\
        --ainode http://100.122.26.9:3000 \\
        --model ornith-ai/Ornith-1.5-35B-A3B-NVFP4 \\
        --label "Ornith stacked Spark-1, chat fallback"

    scripts/ainode-bench.py decide --backend chat --compare jev ...   # side by side

``--dry-run`` prints the item counts, the backend and one example request and
touches nothing: no request, no file. The API key is never printed, only where it
came from.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from ainode.bench.decide.backends import (
    BACKENDS,
    DEFAULT_API_KEY,
    DEFAULT_TIMEOUT,
    JEV_MODEL,
    BackendError,
    build_backend,
)
from ainode.bench.decide.items import (
    ItemError,
    default_items_path,
    items_file_label,
    load_items,
)
from ainode.bench.decide.runner import (
    DEFAULT_CONCURRENCY,
    SOURCE,
    build_decide_block,
    build_notes,
    build_record,
    print_compare,
    print_table,
    print_wrong,
    reported_model,
    run_items,
)

#: Placement for a hosted backend: there is no node of ours behind it, and the
#: record says so rather than leaving the field empty or naming our own machine.
HOSTED_PLACEMENT = {"node": "typesafe.ai hosted"}


def say(message) -> None:
    """Print a line and flush it: a run is watched, often through a redirect."""
    print(message, flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ainode-bench decide",
        description="measure how well a backend makes typed decisions: accuracy, "
                    "calibration, latency, cost")
    p.add_argument("--backend", default="", choices=list(BACKENDS),
                   help="ainode (POST /v1/decide), chat (any OpenAI-compatible "
                        "engine, lettered options plus logprobs) or jev (TypeSafe "
                        "AI's hosted System One model)")
    p.add_argument("--compare", default="", choices=list(BACKENDS),
                   help="run a second backend over the same items and print the two "
                        "side by side. Each backend still writes its own record")
    p.add_argument("--endpoint", default="",
                   help="OpenAI-compatible base with its /v1 for the ainode and chat "
                        "backends, e.g. http://100.122.26.9:3000/v1")
    p.add_argument("--model", default="",
                   help="model id exactly as served. It belongs to the ainode and "
                        f"chat backends; the jev backend uses {JEV_MODEL} unless it "
                        "is the only backend and --model names one of its aliases")
    p.add_argument("--ainode", default="",
                   help="AINode web base for placement, e.g. http://host:3000. "
                        "Omitted means the record carries no placement rather than a "
                        "guessed one")
    p.add_argument("--label", default="",
                   help="free text: what makes this run distinct")
    p.add_argument("--items", default="",
                   help=f"item file (default {default_items_path()})")
    p.add_argument("--sets", default="",
                   help="comma list of item sets to run (default all)")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                   help=f"items in flight at once (default {DEFAULT_CONCURRENCY})")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                   help=f"seconds per item (default {DEFAULT_TIMEOUT})")
    p.add_argument("--api-key", default="",
                   help="bearer token: the endpoint's for ainode/chat (default "
                        f"{DEFAULT_API_KEY}), TypeSafe's for jev (else "
                        "$TYPESAFE_API_KEY, else ~/.jev_api_key). Never printed and "
                        "never written into a record")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and one example request, write nothing")
    return p


def name_list(text: str) -> list:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def model_for(args, backend_name: str) -> str:
    """Which model id one backend is asked for.

    ``--model`` is the served id of a local model, so the hosted backend keeps its
    own alias whenever a comparison run pairs the two: a ``--compare jev`` next to a
    local model must not post that local id to TypeSafe.
    """
    if backend_name != "jev":
        return args.model
    if args.compare or not args.model:
        return JEV_MODEL
    return args.model


def describe(args, backend):
    """``(model_block, placement, warnings)`` for the record.

    A hosted backend has no placement to resolve: its row says ``typesafe.ai
    hosted`` and its model is the version the API reported. For the local backends
    this is the agentic bench's rule, and for the same reason: the serving node is a
    fact about the cluster, so a run that was not told where the control plane is
    records no placement rather than stamping the endpoint's host as the node.
    """
    model_id = reported_model(backend)
    if backend.name == "jev":
        return ({"id": model_id, "name": model_id, "vendor": "typesafe.ai"},
                dict(HOSTED_PLACEMENT), [])
    if not args.ainode:
        return {"id": model_id}, {}, []
    from ainode.bench.fleet import describe_via_http, resolve_serving_node

    base = args.ainode.rstrip("/")
    node_name, engine_port, gpu_name, resolve_warn = resolve_serving_node(base,
                                                                         args.model)
    model_block, placement, _node_id, warnings = describe_via_http(base, base,
                                                                  args.model)
    if node_name:
        # The fleet view names the node actually serving the model; the master's own
        # description would otherwise stamp the master as the placement.
        placement["node"] = node_name
        placement["port"] = engine_port
        if gpu_name:
            placement["gpu"] = gpu_name
    warnings = list(warnings)
    if resolve_warn:
        warnings.insert(0, resolve_warn)
    return model_block, placement, warnings


def settings_for(args, backend, item_set, names) -> dict:
    return {"backend": backend.name,
            "endpoint": backend.endpoint,
            "model_requested": model_for(args, backend.name),
            "items_file": items_file_label(item_set.path),
            "item_set": item_set.id,
            "items": len(item_set.items),
            "sets": list(names),
            "concurrency": args.concurrency,
            "timeout_s": args.timeout}


def example_items(item_set) -> list:
    """One item per kind in the selection, so a dry run shows every request shape."""
    seen = {}
    for item in item_set.items:
        seen.setdefault(item.kind, item)
    return list(seen.values())


def dry_run(args, item_set, names, out=say) -> int:
    out(f"\n  ainode-bench decide  {args.backend}"
        f"{' + ' + args.compare if args.compare else ''}")
    out(f"  items   : {len(item_set.items)} from {item_set.path} ({item_set.id})")
    for name in names:
        mine = [i for i in item_set.items if i.set == name]
        out(f"    {name:<10} {len(mine):3d}  {mine[0].kind:<6} "
            f"{(item_set.sets.get(name) or {}).get('measures', '')}")
    for backend_name in [args.backend] + ([args.compare] if args.compare else []):
        try:
            backend = build_backend(backend_name, endpoint=args.endpoint,
                                    model=model_for(args, backend_name),
                                    api_key=args.api_key, timeout=args.timeout)
        except BackendError as exc:
            out(f"\n  {backend_name}: not runnable as asked: {exc}")
            continue
        out(f"\n  backend : {backend.name}  model "
            f"{getattr(backend, 'model', '') or 'server default'}")
        if backend.name == "jev":
            out(f"  key     : from {backend.key_source} (never printed)")
        out(f"  cost    : ${backend.input_usd_per_mtok:g}/M input tokens, "
            f"${backend.output_usd_per_mtok:g}/M output")
        for item in example_items(item_set):
            out(f"  example ({item.kind}): {backend.request(item).curl_safe()}")
    out("\n  dry run: nothing was requested and no file was written")
    return 0


def record_path(out_dir, stamp: str, model_id: str, label: str,
                backend_name: str) -> pathlib.Path:
    """``<stamp>-<model-slug>-<label-slug>-decide.json``, with the backend added
    only when two backends of one run would otherwise claim the same name."""
    from ainode.bench.measure import slug

    base = f"{stamp}-{slug(model_id)}-{slug(label)}"
    path = out_dir / f"{base}-decide.json"
    if path.exists():
        return out_dir / f"{base}-{backend_name}-decide.json"
    return path


def run_backend(args, backend_name, item_set, names, out_dir, stamp, log=say):
    """One backend over the item set: the block, the record, the file it landed in."""
    backend = build_backend(backend_name, endpoint=args.endpoint,
                            model=model_for(args, backend_name),
                            api_key=args.api_key, timeout=args.timeout)
    log(f"\n  {backend.name}  {getattr(backend, 'model', '') or 'server default'}"
        f"  {backend.endpoint}")
    if backend.name == "jev":
        log(f"  key from {backend.key_source} (never printed)")

    total = len(item_set.items)
    started = time.time()

    def progress(done, count, row):
        if done == count or done % 10 == 0:
            mark = "err" if row.get("error") else ("ok " if row.get("correct")
                                                  else "MISS")
            log(f"    {done:3d}/{count}  last {row['id']:<12} {mark}")

    rows = run_items(backend, item_set.items, concurrency=args.concurrency,
                     progress=progress)
    seconds = round(time.time() - started)
    block = build_decide_block(backend, item_set, rows, names,
                               concurrency=args.concurrency,
                               model_reported=reported_model(backend))
    model_block, placement, warnings = describe(args, backend)
    for warning in warnings:
        log(f"  warn    : {warning}")
    notes = build_notes(backend, item_set, rows, seconds) + list(warnings)
    record = build_record(args.label, model_block, placement, block,
                          settings_for(args, backend, item_set, names), notes, stamp)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = record_path(out_dir, stamp, model_block.get("id") or backend.name,
                       args.label, backend.name)
    path.write_text(json.dumps(record, indent=1) + "\n")
    title = (f"{backend.name}  {model_block.get('name') or model_block.get('id')}"
             f"  ({total} items in {seconds}s)")
    print_table(block, title, out=log)
    print_wrong(rows, item_set.items, out=log)
    log(f"\n  saved {path}")
    return block, title, path


def main(argv=None, out_dir=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    out_dir = pathlib.Path(out_dir) if out_dir else pathlib.Path.cwd() / "bench" / "results"

    if not args.backend:
        p.error("--backend is required; pick from " + ", ".join(BACKENDS))
    if args.compare == args.backend:
        p.error("--compare names a second backend, not the one --backend already ran")
    if args.concurrency < 1:
        p.error("--concurrency must be at least 1")
    if args.timeout <= 0:
        p.error("--timeout must be positive")
    if not args.dry_run and not args.label:
        p.error("--label is required")

    try:
        item_set = load_items(args.items or None, name_list(args.sets) or None)
    except ItemError as exc:
        return p.error(str(exc))
    names = item_set.set_names

    say(f"\n  ainode-bench decide  {len(item_set.items)} items over "
        f"{len(names)} set(s): {', '.join(names)}")
    say("  metrics : accuracy, Brier, calibration error with its bins, wrong "
        "answers surviving 0.8 and 0.9, p50/p95 latency, tokens, cost")
    if args.dry_run:
        return dry_run(args, item_set, names)

    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    blocks, titles = [], []
    try:
        for backend_name in [args.backend] + ([args.compare] if args.compare else []):
            block, title, _path = run_backend(args, backend_name, item_set, names,
                                              out_dir, stamp)
            blocks.append(block)
            titles.append(f"{backend_name} {block['model_reported'] or ''}".strip())
    except BackendError as exc:
        return p.error(str(exc))
    if len(blocks) == 2:
        print_compare(blocks, titles, out=say)
    say("\n  render the README table with: python3 scripts/render-bench-table.py")
    return 0


def run() -> int:
    try:
        return main()
    except KeyboardInterrupt:
        print("\n  interrupted; nothing was written")
        return 130


__all__ = ["HOSTED_PLACEMENT", "SOURCE", "build_parser", "describe", "dry_run",
           "main", "name_list", "run", "run_backend", "settings_for"]
