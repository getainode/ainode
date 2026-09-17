"""``scripts/ainode-bench.py harness ...`` - the command line for the harness bench.

A subcommand rather than a second script, because it is the same question as the
throughput bench asked differently: the existing flat flags measure how fast a
served model generates, and this measures whether it can drive a coding agent to
passing tests. Both write one schema-1 JSON into ``bench/results/``.

    scripts/ainode-bench.py harness \\
        --endpoint http://100.122.26.9:3000/v1 \\
        --model fraserprice/DeepSeek-V4-Flash-DSpark \\
        --harness aider,dsh,pi --tasks 10 --label fleet-flash

``--dry-run`` prints the exact argv, environment overlay and config files for every
task and harness, and touches nothing: no subprocess, no config write, no request.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import time

from ainode.bench.harness import adapters as adapters_mod
from ainode.bench.harness import runner as runner_mod
from ainode.bench.harness.adapters import (
    DEFAULT_API_KEY,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_OUTPUT_TOKENS,
    HarnessRequest,
)
from ainode.bench.harness.runner import (
    DEFAULT_ATTEMPTS,
    DEFAULT_TIMEOUT,
    SOURCE,
    build_harness_block,
    build_notes,
    build_prompt,
    build_record,
    http_metrics_reader,
    run_suite,
)
from ainode.bench.harness.tasks import TaskError, default_tasks_dir, load_tasks, task_set

PROMPT_PREVIEW = 72


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ainode-bench harness",
        description="measure a model driving a coding agent to passing tests")
    p.add_argument("--endpoint", help="OpenAI-compatible base the harness talks to, "
                                     "e.g. http://100.122.26.9:3000/v1")
    p.add_argument("--model", help="model id exactly as served")
    p.add_argument("--label", help="free text: what makes this run distinct")
    p.add_argument("--harness", default="aider",
                   help="comma list of harnesses (default aider)")
    p.add_argument("--tasks", type=int, default=None,
                   help="how many tasks of the set to run, slug order (default all)")
    p.add_argument("--only-tasks", default="",
                   help="comma list of slugs to run instead of a count")
    p.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS,
                   help=f"tries per task; the second sees the failing test output "
                        f"(default {DEFAULT_ATTEMPTS})")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                   help=f"seconds per harness invocation (default {DEFAULT_TIMEOUT})")
    p.add_argument("--ainode", default="",
                   help="AINode web base for placement and token counters "
                        "(default: --endpoint with a trailing /v1 removed)")
    p.add_argument("--tasks-dir", default="", help="task set to use (default the "
                                                  "repo's bench/harness/tasks)")
    p.add_argument("--work-dir", default="", help="where working copies go "
                                                 "(default a fresh temp dir)")
    p.add_argument("--api-key", default=DEFAULT_API_KEY,
                   help="placeholder key for harnesses that insist on one "
                        f"(default {DEFAULT_API_KEY})")
    p.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW,
                   help="context window declared to harnesses that need a catalog "
                        f"entry (default {DEFAULT_CONTEXT_WINDOW})")
    p.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS,
                   help=f"max output tokens declared the same way "
                        f"(default {DEFAULT_MAX_OUTPUT_TOKENS})")
    p.add_argument("--claude-effort", default=None,
                   help="reasoning effort for the claude harness, passed as "
                        "--effort (low, medium, high, xhigh). Unset sends nothing "
                        "and Claude Code's own default stands; a template that "
                        "rejects that default needs this (Qwen3.8-Flash-Next takes "
                        "only xhigh, medium, low)")
    p.add_argument("--no-metrics", action="store_true",
                   help="skip the /api/metrics token window")
    p.add_argument("--dry-run", action="store_true",
                   help="print the per-task commands and write nothing")
    p.add_argument("--list-harnesses", action="store_true",
                   help="print the known harnesses, whether each binary is on PATH, "
                        "and exit")
    return p


def ainode_base(endpoint: str, override: str = "") -> str:
    """The node's web base. Port 3000 serves both the OpenAI proxy and the API, so
    the endpoint minus its ``/v1`` is the control plane."""
    if override:
        return override.rstrip("/")
    base = (endpoint or "").rstrip("/")
    return base[:-3].rstrip("/") if base.endswith("/v1") else base


def effort(args) -> str | None:
    """``--claude-effort`` as the request wants it: a level, or None for "send
    nothing". Blank is the same as unset, so a shell variable that expanded to
    nothing does not become an empty ``--effort`` argument."""
    return (getattr(args, "claude_effort", None) or "").strip() or None


def _preview(arg: str) -> str:
    """One argument, readable: a multi-line prompt collapses to its first line."""
    if "\n" not in arg and len(arg) <= PROMPT_PREVIEW:
        return arg
    head = arg.splitlines()[0][:PROMPT_PREVIEW].rstrip()
    return f"{head} ... [{len(arg)} chars]"


#: Variable names whose value is a credential unless it is our own placeholder.
SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def _mask(env: dict, api_key: str) -> dict:
    """Print the overlay, except a credential we did not put there ourselves.

    Paths are the useful half of this line, so they print. A key-shaped variable
    prints only when it holds our placeholder: the adapters fill in a real key's
    variable only if it was already set in the environment, and that value is
    somebody's actual credential.
    """
    def show(name, value):
        keyish = any(word in name.upper() for word in SECRET_WORDS)
        return value if (value == api_key or not keyish) else "***"

    return {k: show(k, v) for k, v in sorted(env.items())}


def dry_run(tasks, adapters, args, out=print) -> int:
    root = pathlib.Path(args.work_dir) if args.work_dir else pathlib.Path("/tmp/ainode-harness-DRYRUN")
    for adapter in adapters:
        mark = "on PATH" if adapter.available() else "NOT on PATH"
        out(f"\n  {adapter.name} ({adapter.binary}, {mark})")
        for task in tasks:
            workdir = root / adapter.name / task.slug
            req = HarnessRequest(workdir=workdir,
                                 scratch=root / adapter.name / f"{task.slug}.scratch",
                                 prompt=build_prompt(task), entry=task.entry,
                                 endpoint=args.endpoint, model=args.model,
                                 api_key=args.api_key,
                                 context_window=args.context_window,
                                 max_output_tokens=args.max_output_tokens,
                                 claude_effort=effort(args))
            info = adapter.describe(req)
            out(f"\n    task     : {task.slug}")
            out(f"    cwd      : {workdir}")
            out(f"    copied in: {task.instructions_file}, {task.entry}")
            out(f"    hidden   : {', '.join(task.test_names())} (after the harness exits)")
            out(f"    command  : {shlex.join(_preview(a) for a in info['command'])}")
            if info["env"]:
                out(f"    env      : {_mask(info['env'], args.api_key)}")
            if info["git_init"]:
                out(f"    git init : {workdir}")
            for cfg in info["config"]:
                note = " (merged into the existing file)" if cfg["merged"] else ""
                out(f"    config   : {cfg['path']} [{cfg['bytes']} bytes]{note}")
                if not cfg["merged"]:
                    for line in cfg["content"].rstrip().splitlines():
                        out(f"      | {line}")
            out(f"    tests    : {shlex.join(runner_mod.resolve_test_command(task))}")
    out("\n  dry run: nothing was executed and no file was written")
    return 0


def main(argv=None, out_dir=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    out_dir = pathlib.Path(out_dir) if out_dir else pathlib.Path.cwd() / "bench" / "results"
    registry = adapters_mod.registry()

    if args.list_harnesses:
        for name in registry.names():
            adapter = registry.get(name)
            state = "on PATH" if adapter.available() else "not installed"
            print(f"  {name:<10} {adapter.binary:<10} {state}")
        return 0

    for need in ("endpoint", "model", "label"):
        if not getattr(args, need):
            p.error(f"--{need} is required")
    names = [h.strip() for h in args.harness.split(",") if h.strip()]
    if not names:
        p.error("--harness needs at least one name")
    if args.attempts < 1:
        p.error("--attempts must be at least 1")
    if args.timeout <= 0:
        p.error("--timeout must be positive")
    try:
        adapters = [registry.get(n) for n in names]
    except KeyError as exc:
        p.error(str(exc).strip("'"))

    tasks_dir = pathlib.Path(args.tasks_dir) if args.tasks_dir else default_tasks_dir()
    slugs = [s.strip() for s in args.only_tasks.split(",") if s.strip()]
    try:
        tasks = load_tasks(tasks_dir, limit=None if slugs else args.tasks, slugs=slugs or None)
        meta = task_set(tasks_dir)
    except TaskError as exc:
        p.error(str(exc))

    print(f"\n  ainode-bench harness  {args.model}")
    print(f"  label   : {args.label}")
    print(f"  endpoint: {args.endpoint}")
    print(f"  harness : {', '.join(names)}")
    print(f"  tasks   : {len(tasks)} of {meta.get('count')} "
          f"({meta.get('id')}): {', '.join(t.slug for t in tasks)}")
    print(f"  protocol: {args.attempts} attempt(s), {args.timeout:g}s per invocation; "
          "the tests are hidden until the harness exits")
    if effort(args):
        seen = "" if "claude" in names else "  (no claude harness in this run: ignored)"
        print(f"  effort  : claude --effort {effort(args)}{seen}")

    if args.dry_run:
        return dry_run(tasks, adapters, args)

    missing = [a.name for a in adapters if not a.available()]
    if missing:
        p.error(f"not on PATH: {', '.join(missing)}. Install them or drop them from "
                "--harness; run with --dry-run to see what would have been called")

    base = ainode_base(args.endpoint, args.ainode)
    from ainode.bench.fleet import describe_via_http, resolve_serving_node

    node_name, engine_port, gpu_name, resolve_warn = resolve_serving_node(base, args.model)
    model_block, placement, _node_id, warnings = describe_via_http(base, base, args.model)
    if node_name:
        # The fleet view names the serving node; the master's own description
        # would otherwise stamp the master as the placement.
        placement["node"] = node_name
        placement["port"] = engine_port
        if gpu_name:
            placement["gpu"] = gpu_name
    if resolve_warn:
        warnings = [resolve_warn] + list(warnings)
    print(f"  node    : {placement.get('node', 'unknown')}  "
          f"engine :{placement.get('port', '?')}  "
          f"{placement.get('gpu', '')}  "
          f"tp={placement.get('tp', '?')}  ainode {placement.get('ainode', '?')}")
    for warning in warnings:
        print(f"  warn    : {warning}")

    reader = None if args.no_metrics else http_metrics_reader(base)
    started = time.time()
    results = run_suite(tasks, adapters, args.endpoint, args.model,
                        root=pathlib.Path(args.work_dir) if args.work_dir else None,
                        timeout=args.timeout, attempts=args.attempts,
                        api_key=args.api_key, context_window=args.context_window,
                        max_output_tokens=args.max_output_tokens,
                        claude_effort=effort(args), tokens_reader=reader)
    seconds = round(time.time() - started)

    harness_block = build_harness_block(results, tasks, args.endpoint, args.attempts,
                                        args.timeout, tasks_dir)
    settings = {"attempts": args.attempts, "timeout_s": args.timeout,
                "tasks_requested": args.tasks, "harnesses": names,
                "context_window_declared": args.context_window,
                "max_output_tokens_declared": args.max_output_tokens}
    # Only when asked for: absent means the run sent no --effort at all, which is
    # not the same statement as a level of None.
    if effort(args):
        settings["claude_effort"] = effort(args)
    tokens_used = any(t.tokens for r in results for t in r.tasks)
    notes = build_notes(results, tokens_used, seconds) + list(warnings)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    record = build_record(args.label, model_block, placement, harness_block, settings,
                          notes, stamp, SOURCE)

    from ainode.bench.measure import slug

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stamp}-{slug(args.model)}-{slug(args.label)}-harness.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"\n  saved {path}")
    for result in results:
        scores = runner_mod.score(result.tasks)
        print(f"  {result.harness:<10} pass@1 {scores['pass_at_1']:.2f}  "
              f"pass@2 {scores['pass_at_2']:.2f}  mean {scores['mean_wall_s']}s  "
              f"crashes {scores['crashes']}")
    return 0
