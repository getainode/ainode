#!/usr/bin/env python3
"""bench/report.py - render every bench/results/*.json into one self-contained page.

    python3 bench/report.py                 # -> bench/report.html
    python3 bench/report.py --out /tmp/x.html
    python3 bench/report.py --results ~/.ainode/bench/results --out /tmp/local.html

A shim. The renderer now lives in ``ainode/bench/report.py`` so the product can
serve the same page live at ``/api/bench/report`` instead of the repo owning a
second copy of it. This file keeps the command line and the repo-relative
defaults, which is what the release checklist and AGENTS.md already reference.
stdlib only.
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = HERE / "results"
OUT = HERE / "report.html"

# Run from a bare checkout with no `pip install -e .`: the renderer is a package
# module now, so the repo root has to be importable.
sys.path.insert(0, str(REPO))

from ainode.bench.report import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(results=RESULTS, out=OUT))
