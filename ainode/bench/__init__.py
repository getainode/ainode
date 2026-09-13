"""The AINode benchmark, as a package the product can run.

``scripts/ainode-bench.py`` used to be the only way to measure a served model, so
the numbers in ``bench/results/`` could only come from somebody with a terminal on
the fabric. This package is the same measurement code, importable and
async-friendly, so the Bench view can point a run at whatever is already loaded on
the fleet and write the same schema-1 JSON the public leaderboard consumes.

Layout:

  ``measure.py``  the timing itself - transport, prompts, the five sections.
                  stdlib only and blocking; the runner drives it off-loop.
  ``fleet.py``    where a run gets its truth: which instance to hit (through the
                  proxy's own routing function) and what placement to record.
  ``runner.py``   job lifecycle - status, progress, log tail, cancel, results dir.
  ``report.py``   the renderer, moved out of ``bench/report.py``.
  ``api_routes.py`` the ``/api/bench`` surface.
  ``cli.py``      the command line ``scripts/ainode-bench.py`` shims to.

Inference only, in every path: a bench never loads, unloads or restarts anything.
"""

from ainode.bench.measure import BenchOptions, SECTION_ORDER, SECTION_TITLES
from ainode.bench.runner import BenchBusy, BenchManager, BenchRun, results_dir

__all__ = ["BenchOptions", "BenchManager", "BenchRun", "BenchBusy",
           "SECTION_ORDER", "SECTION_TITLES", "results_dir"]
