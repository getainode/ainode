"""The agentic rubric: can a served model be trusted with the parts of an agent loop.

The throughput bench answers "how fast does this model generate". The harness bench
next door answers "can it drive a coding agent to green tests". Neither says whether
the model can do the small mechanical things every agent loop is made of, and those
are the things that make a local model unusable long before its coding score does:
call one tool with the right arguments, call three at once when three were asked for,
notice a tool came back with an error and try again instead of inventing the answer,
keep a system rule alive for four turns, hold a format, find one sentence in a
100k-token prompt.

    ``probes.py``    the probes and their checkers, one mechanical verdict each
    ``runner.py``    the transport, the loop, the scoring, the record
    ``cli.py``       ``scripts/ainode-bench.py agentic ...``

This started as a hand-run script and two records with a hand-typed ``rubric``
block, which is exactly the shape of number the repo's own rules say not to trust:
"measured numbers live only in bench/results/*.json". So it is the bench's third
section now, it writes a schema-1 record with an ``agentic`` block, and every
verdict is mechanical - group C executes the model's code against asserts it never
saw, group G judges a real tool-loop trace. See ``bench/agentic/README.md``.
"""

from ainode.bench.agentic.probes import (
    DEFAULT_NEEDLE,
    GROUPS,
    MAX_TURNS,
    Probe,
    ProbeResult,
    all_probes,
)
from ainode.bench.agentic.runner import (
    DEFAULT_TEMPERATURE,
    DEFAULT_THINK_KW,
    DEFAULT_TIMEOUT,
    ChatClient,
    ProbeRun,
    Reply,
    build_agentic_block,
    build_record,
    group_scores,
    needle_map,
    run_probes,
    score,
)

__all__ = ["GROUPS", "DEFAULT_NEEDLE", "DEFAULT_TEMPERATURE", "DEFAULT_THINK_KW",
           "DEFAULT_TIMEOUT", "MAX_TURNS", "Probe", "ProbeResult", "ProbeRun",
           "Reply", "ChatClient", "all_probes", "run_probes", "score",
           "group_scores", "needle_map", "build_agentic_block", "build_record"]
