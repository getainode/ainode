"""``scripts/ainode-bench.py decide ...`` - the command line for the decision bench.

A fourth subcommand next to ``harness`` and ``agentic``, asking the fourth question:
not how fast the model generates, not whether it can drive a coding agent, not
whether it can hold an agent loop together, but whether its typed decisions can be
trusted by code that acts on them. Accuracy, calibration, latency, cost.

Two measurements live under this one word, because they ask that of the same endpoints
and write the same record. **The Jevals recipe** scores the three public question sets
the independent Jevals boards use, with their formulas, so an AINode-served model can be
read next to Jev and its clones; **the legacy 110-item path** scores AINode's own hand
built set, which is five shapes of the job a router or a triage step actually does.
``--suite``/``--questions`` picks the first and ``--backend`` the second, and mixing them
is an error rather than a guess.

    # the Jevals recipe. Fetch the question sets once; the item text is not committed
    scripts/ainode-bench.py decide download

    scripts/ainode-bench.py decide --suite all --transport decide \\
        --endpoint http://100.122.26.9:3000/v1 \\
        --ainode http://100.122.26.9:3000 \\
        --model ornith-ai/Ornith-1.5-35B-A3B-NVFP4 \\
        --label "Ornith on Spark-1, Jevals 0.1.0"

    # any server that speaks the Jev wire format, TypeSafe's hosted Jev included
    scripts/ainode-bench.py decide --suite all --transport systemone \\
        --endpoint https://api.typesafe.ai/v1 --label "jev-latest, Jevals 0.1.0"

    # a private blind set, in the same shape, never committed
    scripts/ainode-bench.py decide --questions /path/to/blind.json \\
        --transport systemone --endpoint http://kev-host:8080/v1 --label blind-1

    # the legacy 110-item path
    scripts/ainode-bench.py decide --backend jev --label "jev-latest, 110 items"
    scripts/ainode-bench.py decide --backend chat --compare jev ...   # side by side

``--dry-run`` prints the plan, the sets and one example request per question shape and
touches nothing: no request, no file. The API key is never printed, only where it came
from.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from ainode.bench import auth
from ainode.bench.decide import jevals, sets, suite
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
                   help="bearer token: the endpoint's for ainode/chat (else "
                        f"${auth.ENV_API_KEY}, else {DEFAULT_API_KEY}), TypeSafe's "
                        "for jev (else $TYPESAFE_API_KEY, else ~/.jev_api_key). "
                        "Never printed and never written into a record")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and one example request, write nothing")

    # The Jevals-recipe mode. A second measurement in the same subcommand, because it
    # asks the same question of the same endpoints and writes the same record; what it
    # changes is the question sets (the three public ones the independent boards use),
    # the repeats and the formulas. See bench/decide/JEVALS.md.
    jev = p.add_argument_group(
        "the Jevals recipe (suite 0.1.0)",
        "score the same public question sets the independent Jevals boards use, with "
        "their formulas, so an AINode-served model can be read next to Jev and its "
        "clones. `decide download` fetches the item text first; it is not committed")
    jev.add_argument("--suite", default="",
                     help="comma list of " + ", ".join(sets.SUITES) + ", or `all`. "
                          "Turns on the Jevals recipe and needs --transport")
    jev.add_argument("--questions", default="",
                     help="run a question file in the same shape instead of a suite, so "
                          "a private blind set is scored by the same code without being "
                          "committed. Repeatable as a comma list")
    jev.add_argument("--transport", default="", choices=list(suite.TRANSPORTS),
                     help="decide (AINode's POST /v1/decide) or systemone (POST "
                          "/v1/systemone in the Jev wire format: TypeSafe's hosted Jev, "
                          "or any server that speaks it). The key is resolved from the "
                          "endpoint's HOST, so a fleet key can never reach a vendor")
    jev.add_argument("--repeats", type=int, default=jevals.REPEATS,
                     help=f"answers per question (default {jevals.REPEATS}, which is the "
                          "suite's figure; a board listing needs a complete run at 5)")
    jev.add_argument("--limit", type=int, default=0,
                     help="take only the first N questions of each set. A transport "
                          "proof, not a suite result, and the record says so")
    jev.add_argument("--price-in", type=float, default=0.0,
                     help="USD per million input tokens for this endpoint, from its "
                          "posted rate. Without it the cost column reads $0 rather than "
                          "an estimate")
    jev.add_argument("--price-out", type=float, default=0.0,
                     help="USD per million output tokens for this endpoint")
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
    # The node's key, never the hosted backend's: a jev run returns above without
    # touching a control plane of ours, so this can only be the one the local
    # backends are using.
    key = getattr(backend, "api_key", "") if backend.local else ""
    node_name, engine_port, gpu_name, resolve_warn = resolve_serving_node(
        base, args.model, api_key=key)
    model_block, placement, _node_id, warnings = describe_via_http(
        base, base, args.model, api_key=key)
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
    log(f"  key from {backend.key_source} (never printed)")
    if backend.local:
        # Before the items, so a protected node is reported as one rather than as 110
        # items that all came back wrong.
        refused = auth.preflight(args.endpoint, backend.api_key)
        if refused:
            raise auth.EndpointRefused(refused)

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


# --------------------------------------------------------------- the Jevals recipe

def download_main(argv, out=say) -> int:
    """``decide download [suite,...]``: fetch the item text, verify it, write the cache.

    The only networked call in :mod:`ainode.bench.decide.sets`, and the only thing that
    writes under ``bench/decide/cache/``, which is gitignored. Item text is not
    committed: Jevals does not republish it either, the three upstream licences are the
    item text's and not ours to relicense, and every state is checked against its
    published ``state_sha256`` on the way in, so a download proves itself rather than
    being trusted.
    """
    wanted = name_list(argv[0]) if argv and not argv[0].startswith("-") else None
    if wanted == ["all"]:
        wanted = None
    try:
        out("\n  ainode-bench decide download  -> " + str(sets.cache_dir()))
        for suite_id in (wanted or sets.SUITES):
            summary = sets.manifest_summary(suite_id)
            out(f"    {summary['id']:<12} {summary['type']:<7} "
                f"{summary['items']:3d} items  {summary['options']:3d} options  "
                f"{summary['dataset']} {summary['split']} @ "
                f"{summary['hf_revision'][:8]}  {summary['license']}")
        paths = sets.download(wanted, progress=lambda line: out(f"    {line}"))
    except sets.SetError as exc:
        out(f"\n  {exc}")
        return 1
    out(f"\n  {len(paths)} question file(s) written and hash-verified. They are "
        "gitignored on purpose (bench/decide/JEVALS.md).")
    return 0


def suite_docs(args, p):
    """The question files this run scores: named suites, named files, or an error."""
    docs = []
    if args.suite:
        wanted = name_list(args.suite)
        if wanted == ["all"]:
            wanted = list(sets.SUITES)
        docs += sets.load_suite_questions(wanted)
    for path in name_list(args.questions):
        docs.append(sets.load_questions(path))
    if not docs:
        p.error("--suite or --questions names nothing to run")
    seen = set()
    for doc in docs:
        name = doc.get("set") or doc["id"]
        if name in seen:
            p.error(f"two question files both call themselves {name!r}; a set is one "
                    "measurement and two of them cannot share a name")
        seen.add(name)
    return docs


def suite_settings(args, transport, docs) -> dict:
    return {"mode": suite.MODE,
            "transport": transport.name,
            "endpoint": transport.endpoint,
            "model_requested": args.model or "",
            "suite": name_list(args.suite),
            "questions_files": name_list(args.questions),
            "sets": [(doc.get("set") or doc["id"]) for doc in docs],
            "repeats": args.repeats,
            "limit": args.limit or None,
            "concurrency": args.concurrency,
            "timeout_s": args.timeout}


def suite_dry_run(args, transport, docs, out=say) -> int:
    """The plan, one example request per primitive, and what the key source was."""
    work = suite.plan(docs, args.repeats, args.limit)
    out(f"\n  ainode-bench decide  the Jevals recipe, suite {suite.RECIPE['suite']}")
    out(f"  recipe   : {suite.RECIPE['source']} read {suite.RECIPE['read']}, "
        f"recorded in {suite.RECIPE['doc']}")
    out(f"  transport: {transport.name}  {transport.endpoint}")
    out(f"  model    : {transport.model or 'server default'}")
    out(f"  key      : from {transport.key_source or 'the default'} (never printed)")
    out(f"  cost     : ${transport.input_usd_per_mtok:g}/M input, "
        f"${transport.output_usd_per_mtok:g}/M output")
    out(f"  plan     : {len(work)} decisions, {args.repeats} repeats, concurrency "
        f"{args.concurrency}")
    for doc in docs:
        summary = suite.set_summary(doc, args.limit)
        source = summary["source"]
        out(f"    {summary['id']:<12} {summary['type']:<7} "
            f"{summary['questions']:3d} questions  {summary['options']:3d} options  "
            f"seed {summary['seed']}")
        if source:
            out(f"      {source.get('dataset', '')} "
                f"{source.get('split', '')} @ {str(source.get('hf_revision', ''))[:8]}"
                f"  {source.get('license', '')}")
        # One example per primitive the set holds, so a mixed set shows every shape it
        # will really send rather than whichever question happens to be first.
        shown = set()
        for question in doc["questions"]:
            spec = suite.spec_for(doc, question)
            if spec["type"] in shown:
                continue
            shown.add(spec["type"])
            presented = suite.presented_options(spec["options"], question["id"],
                                                doc.get("seed"), 0, spec["type"])
            request = transport.request(doc, question, presented)
            line = request.curl_safe()
            out(f"      example ({spec['type']}): {line[:400]}"
                f"{' ...' if len(line) > 400 else ''}")
            leaks = suite.wire_leaks(request.payload)
            out(f"      wire    : "
                f"{'CARRIES ' + ', '.join(leaks) if leaks else 'no answer key'}")
    out("\n  dry run: nothing was requested and no file was written")
    return 0


def run_suite(args, out_dir, stamp, log=say):
    """One transport over the question sets, five repeats each. Writes one record."""
    p = build_parser()
    docs = suite_docs(args, p)
    transport = suite.build_transport(
        args.transport, endpoint=args.endpoint, model=args.model,
        api_key=args.api_key, timeout=args.timeout,
        input_usd_per_mtok=args.price_in, output_usd_per_mtok=args.price_out)
    names = [(doc.get("set") or doc["id"]) for doc in docs]

    log(f"\n  ainode-bench decide  the Jevals recipe, suite {suite.RECIPE['suite']}: "
        f"{', '.join(names)}")
    log(f"  recipe  : {suite.RECIPE['source']} read {suite.RECIPE['read']} "
        f"({suite.RECIPE['doc']} names every deviation)")
    log("  metrics : accuracy with its guessing floor, Decision Score against the "
        "label prior, ECE over 10 bins with the reliability table, hand-off share at "
        "95%, the published gate, pick flips and confidence swing, p50/p95 latency, "
        "questions per second, malformed answers, cost")
    if args.dry_run:
        return suite_dry_run(args, transport, docs), None
    log(f"  {transport.name}  {transport.model or 'server default'}  "
        f"{transport.endpoint}")
    log(f"  key from {transport.key_source or 'the default'} (never printed)")
    if transport.local:
        refused = auth.preflight(args.endpoint, transport.api_key)
        if refused:
            raise auth.EndpointRefused(refused)

    def progress(done, count, decision):
        if done == count or done % 50 == 0:
            mark = "err " if decision.get("error") else (
                "bad " if decision.get("malformed") else
                ("ok  " if decision.get("pick") == decision.get("label") else "MISS"))
            log(f"    {done:5d}/{count}  last {decision['id']:<16} "
                f"r{decision['repeat']} {mark}")

    decisions, seconds = suite.run(transport, docs, repeats=args.repeats,
                                   concurrency=args.concurrency, limit=args.limit,
                                   progress=progress)
    block = suite.build_decide_block(transport, docs, decisions, seconds,
                                     args.repeats, args.concurrency, args.limit,
                                     model_reported=suite.reported_model(transport))
    model_block, placement, warnings = describe_suite(args, transport)
    for warning in warnings:
        log(f"  warn    : {warning}")
    notes = suite.build_notes(transport, docs, decisions, seconds, args.repeats,
                              args.limit) + list(warnings)
    record = build_record(args.label, model_block, placement, block,
                          suite_settings(args, transport, docs), notes, stamp)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = record_path(out_dir, stamp, model_block.get("id") or transport.name,
                       args.label, transport.name)
    path.write_text(json.dumps(record, indent=1) + "\n")
    title = (f"{transport.name}  "
             f"{model_block.get('name') or model_block.get('id')}  "
             f"({len(decisions)} decisions in {round(seconds)}s)")
    suite.print_table(block["jevals"], title, out=log)
    suite.print_wrong(decisions, out=log)
    log(f"\n  saved {path}")
    return 0, path


def describe_suite(args, transport):
    """``(model_block, placement, warnings)`` for a Jevals-recipe record.

    The same rule the other sections follow: a hosted endpoint gets the one honest
    placement string there is, and a local run that was not told where the control plane
    is records NO placement rather than stamping the endpoint's host as the node.
    """
    model_id = suite.reported_model(transport)
    if not transport.local:
        return ({"id": model_id, "name": model_id, "vendor": "typesafe.ai"},
                dict(HOSTED_PLACEMENT), [])
    if not args.ainode:
        return {"id": model_id}, {}, []
    from ainode.bench.fleet import describe_via_http, resolve_serving_node

    base = args.ainode.rstrip("/")
    key = transport.api_key
    node_name, engine_port, gpu_name, resolve_warn = resolve_serving_node(
        base, args.model, api_key=key)
    model_block, placement, _node_id, warnings = describe_via_http(
        base, base, args.model, api_key=key)
    if node_name:
        placement["node"] = node_name
        placement["port"] = engine_port
        if gpu_name:
            placement["gpu"] = gpu_name
    warnings = list(warnings)
    if resolve_warn:
        warnings.insert(0, resolve_warn)
    return model_block, placement, warnings


def main(argv=None, out_dir=None) -> int:
    words = list(argv) if argv is not None else sys.argv[1:]
    out_dir = pathlib.Path(out_dir) if out_dir else pathlib.Path.cwd() / "bench" / "results"
    # One positional, dispatched before argparse sees it, the way `ainode-bench`
    # dispatches its own subcommands: `decide download` fetches the question sets and
    # asks a model nothing at all, so it shares none of the run's flags.
    if words and words[0] == "download":
        return download_main(words[1:])

    p = build_parser()
    args = p.parse_args(words)

    if args.concurrency < 1:
        p.error("--concurrency must be at least 1")
    if args.timeout <= 0:
        p.error("--timeout must be positive")
    if not args.dry_run and not args.label:
        p.error("--label is required")

    if args.suite or args.questions:
        if args.backend:
            p.error("--suite/--questions run the Jevals recipe and pick their wire with "
                    "--transport; --backend is the legacy 110-item path")
        if not args.transport:
            p.error("--transport is required with --suite/--questions; pick from "
                    + ", ".join(suite.TRANSPORTS))
        if args.repeats < 1:
            p.error("--repeats must be at least 1")
        if args.limit < 0:
            p.error("--limit cannot be negative")
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        try:
            code, path = run_suite(args, out_dir, stamp)
        except (sets.SetError, BackendError) as exc:
            return p.error(str(exc))
        except auth.EndpointRefused as exc:
            # The node refused, before or during the run. Nothing is scored and no
            # record is written: a Decision Score computed over answers nobody gave is
            # worse than no record.
            return auth.stop(str(exc), out=say)
        if path:
            say("\n  render the README table with: python3 "
                "scripts/render-bench-table.py")
        return code
    if args.transport:
        p.error("--transport belongs to --suite/--questions; the legacy 110-item path "
                "picks its wire with --backend")

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
    except auth.EndpointRefused as exc:
        # The node refused, before or during the items. Nothing is scored and no
        # record is written: a partial set of rows would carry an accuracy and a
        # calibration error computed over answers nobody gave.
        return auth.stop(str(exc), out=say)
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


__all__ = ["HOSTED_PLACEMENT", "SOURCE", "build_parser", "describe",
           "describe_suite", "download_main", "dry_run", "main", "name_list", "run",
           "run_backend", "run_suite", "settings_for", "suite_docs", "suite_dry_run",
           "suite_settings"]
