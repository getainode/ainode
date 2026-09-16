"""The harness bench: can a model drive a coding agent to passing tests.

The throughput bench next door answers "how fast does this model generate". This
answers the other half of the question Jason actually has before he puts coding
subagents on his own fleet: given a real agent CLI and a real task, does the model
get to green. They are different numbers and a model can be good at one and
useless at the other.

    ``tasks.py``     the vendored Exercism set and how one task is loaded
    ``adapters/``    one small class per agent CLI: aider, dsh, pi, opencode
    ``runner.py``    the measurement - isolation, two attempts, tests, scoring
    ``cli.py``       ``scripts/ainode-bench.py harness ...``

Same honesty rules as the throughput bench, plus one of its own: the hidden tests
are never in the working directory while the harness is running. See
``bench/harness/README.md``.
"""

from ainode.bench.harness.adapters import HarnessAdapter, HarnessRequest, HarnessRun
from ainode.bench.harness.runner import (
    DEFAULT_ATTEMPTS,
    DEFAULT_TIMEOUT,
    HarnessResult,
    TaskResult,
    build_prompt,
    run_suite,
    run_task,
    score,
)
from ainode.bench.harness.tasks import Task, load_tasks, task_set

__all__ = ["HarnessAdapter", "HarnessRequest", "HarnessRun", "HarnessResult",
           "TaskResult", "Task", "load_tasks", "task_set", "run_task", "run_suite",
           "score", "build_prompt", "DEFAULT_ATTEMPTS", "DEFAULT_TIMEOUT"]
