#!/usr/bin/env python3
"""ainode-bench - measure what a spec sheet does not, on an AINode-served model.

    scripts/ainode-bench.py --url http://100.72.9.84:8000 \
        --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
        --ainode http://100.72.9.84:3000 --label dspark-recipe

    scripts/ainode-bench.py ... --only prefill,concurrency --depths 4000,32000
    scripts/ainode-bench.py --show bench/results/<file>.json

A shim, as of the Bench view. The measurement lives in ``ainode/bench/`` so the
product can run the same benchmark from the browser against whatever is loaded on
the fleet, and the numbers from a terminal run and a browser run stay comparable
because they are the same code. The command line, the console output, the honesty
rules and the schema-1 output file are unchanged; ``ainode/bench/measure.py``
documents the rules and ``bench/README.md`` documents the flags.

Still stdlib only: ``ainode.bench.measure`` and ``ainode.bench.fleet`` import
nothing outside the standard library, so this keeps running on a bare python3
with no pip step. It writes into the repo's ``bench/results/``.
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "bench" / "results"

# Run from a bare checkout with no `pip install -e .`: the measurement is a
# package module now, so the repo root has to be importable.
sys.path.insert(0, str(REPO))

from ainode.bench.cli import main  # noqa: E402

if __name__ == "__main__":
    try:
        sys.exit(main(out_dir=OUT))
    except KeyboardInterrupt:
        print("\n  interrupted; nothing was written")
        sys.exit(130)
