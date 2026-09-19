"""Command line for the bench - what ``scripts/ainode-bench.py`` runs.

The script kept its interface and its stdlib-only promise; the measurement moved
into :mod:`ainode.bench.measure` so the web run and the terminal run cannot drift
apart. Everything printed here is the text the script printed before.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from ainode.bench.fleet import describe_via_http
from ainode.bench.measure import (
    Cancelled,
    ConsoleReporter,
    SECTIONS,
    BenchOptions,
    Telemetry,
    build_notes,
    build_record,
    http_nodes_reader,
    int_list,
    measure,
    slug,
)

SOURCE = "scripts/ainode-bench.py"


def build_parser():
    p = argparse.ArgumentParser(
        prog="ainode-bench",
        description="measure what a spec sheet does not, on an AINode-served model",
        epilog="subcommands: `ainode-bench harness --help` measures a model driving "
               "a coding agent to passing tests instead of its throughput, "
               "`ainode-bench agentic --help` scores it on the agentic capability "
               "rubric, `ainode-bench decide --help` measures how well a backend "
               "makes typed decisions (accuracy, calibration, latency, cost), and "
               "`ainode-bench embed --help` measures an embedding model (dimensions, "
               "latency, throughput by batch size, pair ordering)")
    p.add_argument("--url", help="engine or AINode proxy base, e.g. http://host:8000")
    p.add_argument("--model", help="model id exactly as served")
    p.add_argument("--ainode", default="", help="AINode web base, e.g. http://host:3000 "
                                                "(telemetry + placement)")
    p.add_argument("--label", help="free text: what makes this run distinct")
    p.add_argument("--only", default="", help="comma list of " + ",".join(SECTIONS))
    p.add_argument("--depths", default="4000,16000,32000,64000,120000")
    p.add_argument("--streams", default="1,2,4,8,16")
    p.add_argument("--no-think", action="store_true",
                   help="enable_thinking=false for every section except reasoning, "
                        "which always measures both states")
    p.add_argument("--max-tokens", type=int, default=200,
                   help="generation cap for single/prefill/concurrency")
    p.add_argument("--sustained-tokens", type=int, default=1500)
    p.add_argument("--reasoning-tokens", type=int, default=600)
    p.add_argument("--show", help="pretty-print a saved result and exit")
    return p


def main(argv=None, out_dir=None):
    # Four subcommands, dispatched before argparse sees them, so every existing flag
    # keeps working exactly as documented. `harness` measures a different thing
    # (a model driving a coding agent to passing tests), `agentic` a third one
    # (the capability rubric), `decide` a fourth (typed decisions: accuracy,
    # calibration, latency, cost) and `embed` a fifth (an embedding model:
    # dimensions, latency, throughput by batch size, pair ordering); each has its own
    # parser under its own package. Everything else is the throughput bench.
    words = list(sys.argv[1:]) if argv is None else list(argv)
    if words and words[0] == "harness":
        from ainode.bench.harness.cli import main as harness_main

        return harness_main(words[1:], out_dir=out_dir)
    if words and words[0] == "agentic":
        from ainode.bench.agentic.cli import main as agentic_main

        return agentic_main(words[1:], out_dir=out_dir)
    if words and words[0] == "decide":
        from ainode.bench.decide.cli import main as decide_main

        return decide_main(words[1:], out_dir=out_dir)
    if words and words[0] == "embed":
        from ainode.bench.embed.cli import main as embed_main

        return embed_main(words[1:], out_dir=out_dir)

    p = build_parser()
    a = p.parse_args(argv)
    out_dir = pathlib.Path(out_dir) if out_dir else pathlib.Path.cwd() / "bench" / "results"

    if a.show:
        print(json.dumps(json.loads(pathlib.Path(a.show).read_text()), indent=2))
        return 0
    for need in ("url", "model", "label"):
        if not getattr(a, need):
            p.error(f"--{need} is required")
    want = [s.strip() for s in a.only.split(",") if s.strip()] or list(SECTIONS)
    bad = [s for s in want if s not in SECTIONS]
    if bad:
        p.error(f"unknown section(s) {', '.join(bad)}; pick from {', '.join(SECTIONS)}")

    opts = BenchOptions(url=a.url, model=a.model, label=a.label, sections=want,
                        depths=int_list(a.depths), streams=int_list(a.streams),
                        no_think=a.no_think, max_tokens=a.max_tokens,
                        sustained_tokens=a.sustained_tokens,
                        reasoning_tokens=a.reasoning_tokens)

    mb, pl, node_id, warn = describe_via_http(a.ainode, a.url, a.model)
    print(f"\n  ainode-bench  {opts.model}")
    print(f"  label   : {opts.label}")
    print(f"  endpoint: {opts.url}")
    print(f"  node    : {pl.get('node', 'unknown')}  {pl.get('gpu', '')}  "
          f"tp={pl.get('tp', '?')}  ainode {pl.get('ainode', '?')}")
    print(f"  engine  : {pl.get('engine_image', 'unknown image')}  "
          f"kv={pl.get('kv_cache_dtype', '?')}  gmu={pl.get('gpu_memory_utilization', '?')}")
    if pl.get("stacked_with"):
        print(f"  stacked : {', '.join(pl['stacked_with'])}")
    for w in warn:
        print(f"  warn    : {w}")

    tel = Telemetry(read=http_nodes_reader(a.ainode, node_id)).start()
    rep = ConsoleReporter()
    try:
        results, seconds, cpt = measure(opts, rep)
    except Cancelled:
        tel.stop()
        print("\n  cancelled; nothing was written")
        return 130
    tel.stop()
    telemetry = tel.result()
    if telemetry:
        results["telemetry"] = telemetry
        print(f"\n  TELEMETRY  peak GPU {telemetry.get('gpu_util_pct')}%  "
              f"mem {telemetry.get('gpu_mem_used_gb')}/{telemetry.get('gpu_mem_total_gb')} GB  "
              f"{telemetry.get('temp_c')} C  ({telemetry.get('samples')} samples)")

    notes = build_notes(opts, results, pl, warn, seconds, SOURCE)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    rec = build_record(opts, mb, pl, results, cpt, notes, stamp, SOURCE)
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"{stamp}-{slug(opts.model)}-{opts.label}.json"
    f.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"\n  saved {f}")
    print("  render with: python3 bench/report.py")
    return 0


def run():
    try:
        return main()
    except KeyboardInterrupt:
        print("\n  interrupted; nothing was written")
        return 130


if __name__ == "__main__":
    sys.exit(run())
