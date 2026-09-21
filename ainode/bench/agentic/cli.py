"""``scripts/ainode-bench.py agentic ...`` - the command line for the agentic rubric.

A third subcommand next to ``harness``, because it is the same question as the other
two asked a third way. The flat flags measure how fast a served model generates;
``harness`` measures whether it can drive a coding agent to green tests; this
measures whether it can be trusted with the small mechanical things an agent loop is
made of: follow a format, call a tool once, call three at once, recover when a tool
returns an error, keep a system rule alive over four turns, find a sentence in a
100k-token prompt. All three write one schema-1 JSON into ``bench/results/``.

    scripts/ainode-bench.py agentic \\
        --endpoint http://100.122.26.9:3000/v1 \\
        --ainode http://100.122.26.9:3000 \\
        --model fraserprice/DeepSeek-V4-Flash-DSpark \\
        --label "DeepSeek TP=2 Spark-2+3, quick" --quick

``--dry-run`` prints the probe list and the request shape and touches nothing: no
request, no file. ``--quick`` is the shape to run first: the 8000-token needle only
and no vision, which is a few minutes instead of a long wait on the 100k prefill.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from ainode.bench import auth
from ainode.bench.agentic.probes import DEFAULT_NEEDLE, GROUPS, all_probes
from ainode.bench.agentic.runner import (
    DEFAULT_API_KEY,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINK_KW,
    DEFAULT_TIMEOUT,
    SOURCE,
    ChatClient,
    build_agentic_block,
    build_notes,
    build_record,
    chat_url,
    group_scores,
    run_probes,
    score,
)

#: What ``--quick`` means: the cheap needle only, and no vision probe.
QUICK_NEEDLE = (8000,)
QUICK_SKIP = ("V",)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ainode-bench agentic",
        description="score a served model on the agentic capability rubric")
    p.add_argument("--endpoint", help="OpenAI-compatible base with its /v1, e.g. "
                                     "http://100.122.26.9:3000/v1")
    p.add_argument("--model", help="model id exactly as served")
    p.add_argument("--label", help="free text: what makes this run distinct")
    p.add_argument("--ainode", default="",
                   help="AINode web base for placement, e.g. http://host:3000. "
                        "Omitted means the record carries no placement rather than "
                        "a guessed one")
    p.add_argument("--groups", default="",
                   help=f"comma list of {','.join(GROUPS)} (default all)")
    p.add_argument("--needle", default=",".join(str(n) for n in DEFAULT_NEEDLE),
                   help="comma list of prompt-token sizes for group E "
                        f"(default {','.join(str(n) for n in DEFAULT_NEEDLE)})")
    p.add_argument("--quick", action="store_true",
                   help="the 8000-token needle only and no vision probe")
    p.add_argument("--no-think-kw", default=DEFAULT_THINK_KW, metavar="NAME",
                   help="chat_template_kwargs switch name for the thinking-off "
                        f"probe (default {DEFAULT_THINK_KW}; both it and `thinking` "
                        "are sent, the way the throughput bench sends them)")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                   help=f"seconds per request (default {DEFAULT_TIMEOUT})")
    p.add_argument("--api-key", default="",
                   help=f"bearer token for the endpoint (default ${auth.ENV_API_KEY}, "
                        f"else the placeholder {DEFAULT_API_KEY} that an open node "
                        f"accepts). Never printed and never written into a record")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                   help=f"temperature for every probe except B2, which pins its own "
                        f"(default {DEFAULT_TEMPERATURE})")
    p.add_argument("--dry-run", action="store_true",
                   help="print the probe list and write nothing")
    return p


def int_list(text: str) -> list:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def wanted_groups(args, parser) -> list:
    """The groups this run asked for, in the order they run."""
    names = [g.strip().upper() for g in args.groups.split(",") if g.strip()]
    bad = [g for g in names if g not in GROUPS]
    if bad:
        parser.error(f"unknown group(s) {', '.join(bad)}; pick from "
                     f"{', '.join(GROUPS)}")
    chosen = names or list(GROUPS)
    if args.quick:
        chosen = [g for g in chosen if g not in QUICK_SKIP]
    return [g for g in GROUPS if g in chosen]


def needle_sizes(args, parser) -> list:
    if args.quick:
        return list(QUICK_NEEDLE)
    try:
        sizes = int_list(args.needle)
    except ValueError:
        return parser.error("--needle takes a comma list of integers")
    if any(n <= 0 for n in sizes):
        parser.error("--needle sizes must be positive")
    return sizes


def dry_run(probes, args, groups, needle, out=print) -> int:
    out(f"\n  POST {chat_url(args.endpoint or 'http://ENDPOINT/v1')}")
    out(f"  model       : {args.model}")
    out(f"  temperature : {args.temperature:g}  (B2 pins 0.2)")
    out(f"  timeout     : {args.timeout:g}s per request")
    switches = ", ".join(f"{name}: false" for name in
                         dict.fromkeys(["enable_thinking", "thinking",
                                        args.no_think_kw]))
    out(f"  thinking off: chat_template_kwargs {{{switches}}}")
    out(f"  needle      : {', '.join(str(n) for n in needle)} prompt tokens")
    out(f"  groups      : {', '.join(groups)}")
    out(f"\n  {len(probes)} probe(s):")
    for probe in probes:
        out(f"    {probe.group}  {probe.id:<20} {type(probe).__name__}")
    out("\n  group C executes the model's own code in a subprocess on this machine")
    out("  dry run: nothing was requested and no file was written")
    return 0


def main(argv=None, out_dir=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    out_dir = pathlib.Path(out_dir) if out_dir else pathlib.Path.cwd() / "bench" / "results"

    for need in ("endpoint", "model", "label"):
        if not getattr(args, need):
            p.error(f"--{need} is required")
    if args.timeout <= 0:
        p.error("--timeout must be positive")
    groups = wanted_groups(args, p)
    if not groups:
        p.error("no groups left to run")
    needle = needle_sizes(args, p)
    probes = all_probes(needle=needle, groups=groups)
    if not probes:
        p.error("that selection holds no probes")
    args.api_key, key_source = auth.key_for(args.api_key, DEFAULT_API_KEY)

    print(f"\n  ainode-bench agentic  {args.model}")
    print(f"  label   : {args.label}")
    print(f"  endpoint: {args.endpoint}")
    print(f"  key     : from {key_source} (never printed)")
    print(f"  probes  : {len(probes)} over groups {', '.join(groups)}"
          f"{' (quick)' if args.quick else ''}")
    print("  protocol: every verdict is mechanical; group C is executed and group G "
          "is a judged tool trace")

    if args.dry_run:
        return dry_run(probes, args, groups, needle)

    refused = auth.preflight(args.endpoint, args.api_key)
    if refused:
        return auth.stop(refused)

    model_block, placement, warnings = describe(args)
    if args.ainode:
        print(f"  node    : {placement.get('node', 'unknown')}  "
              f"engine :{placement.get('port', '?')}  "
              f"{placement.get('gpu', '')}  "
              f"tp={placement.get('tp', '?')}  ainode {placement.get('ainode', '?')}")
    for warning in warnings:
        print(f"  warn    : {warning}")
    print()

    client = ChatClient(args.endpoint, args.model, api_key=args.api_key,
                        timeout=args.timeout, temperature=args.temperature,
                        think_kw=args.no_think_kw)
    started = time.time()
    try:
        runs = run_probes(probes, client)
    except auth.EndpointRefused as exc:
        # Mid-run, which the preflight cannot rule out: 25 probes refused the same
        # way is one refusal, not a score of 0/25.
        return auth.stop(str(exc))
    seconds = round(time.time() - started)

    block = build_agentic_block(runs, probes, args.endpoint, groups, needle,
                               temperature=args.temperature, timeout=args.timeout,
                               think_kw=args.no_think_kw)
    settings = {"groups": groups, "needle_tokens": needle, "quick": bool(args.quick),
                "temperature": args.temperature, "timeout_s": args.timeout,
                "thinking_switch": args.no_think_kw,
                "probes_requested": len(probes)}
    notes = build_notes(runs, seconds, groups) + list(warnings)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    record = build_record(args.label, model_block, placement, block, settings, notes,
                          stamp, SOURCE)

    from ainode.bench.measure import slug

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stamp}-{slug(args.model)}-{slug(args.label)}-agentic.json"
    path.write_text(json.dumps(record, indent=1) + "\n")

    totals = score(runs)
    print(f"\n  saved {path}")
    print(f"  SCORE {totals['pass']}/{totals['total']}  " + "  ".join(
        f"{g} {s['pass']}/{s['total']}" for g, s in group_scores(runs).items()))
    if block["needle"]:
        print("  needle " + "  ".join(f"{size} {'pass' if ok else 'fail'}"
                                      for size, ok in block["needle"].items()))
    if block["structured_output_mode"]:
        print(f"  structured output: {block['structured_output_mode']}")
    return 0


def describe(args):
    """(model_block, placement, warnings) for the record.

    Only asks the fleet anything when ``--ainode`` was given: the serving node is a
    fact about the cluster, and a run that was not told where the control plane is
    records no placement rather than stamping the endpoint's host as the node.
    """
    if not args.ainode:
        return {"id": args.model}, {}, []
    from ainode.bench.fleet import describe_via_http, resolve_serving_node

    base = args.ainode.rstrip("/")
    key = getattr(args, "api_key", "") or ""
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
