"""Tests for scripts/render-bench-table.py, the generated README bench table.

The point of generating the table is that the README can never quote a number the
JSON does not contain, so the drift check is the test that matters: run the
renderer in --check mode against the committed README and fail if someone edited
one without re-running the other.

The script is in scripts/ (not a package), so it is loaded by path.
"""

import importlib.util
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "render-bench-table.py"


def _module():
    spec = importlib.util.spec_from_file_location("render_bench_table", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_readme_table_matches_results():
    """README.md is in sync with bench/results/*.json."""
    proc = subprocess.run([sys.executable, str(SCRIPT), "--check"],
                          capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_every_result_file_has_a_row():
    m = _module()
    runs = m.load_runs(REPO / "bench" / "results")
    table = m.render_table(runs)
    files = sorted(p.name for p in (REPO / "bench" / "results").glob("*.json"))
    assert files, "no bench results to render"
    for name in files:
        assert name in table, f"{name} missing from the rendered table"
    assert len(table.splitlines()) == len(files) + 2  # header + rule + one row each


def test_missing_measurement_is_not_invented(tmp_path):
    """A run with no concurrency sweep and no rubric says so, rather than zero."""
    m = _module()
    (tmp_path / "20260101-000000-bare.json").write_text(json.dumps({
        "schema": 1, "stamp": "20260101-000000", "label": "bare",
        "model": {"id": "x/y", "name": "Bare", "params_b": 7, "active_b": 7,
                  "arch": "dense", "quant": "NVFP4"},
        "placement": {"node": "Spark-9", "gpu": "NVIDIA GB10", "gpus": 1, "tp": 1},
        "results": {"single_stream": {"decode_tok_s": 12.0}},
    }))
    row = m.render_table(m.load_runs(tmp_path)).splitlines()[-1]
    assert "| 12.0 | not measured | not run |" in row
    assert "7B dense" in row
    assert "0.0" not in row.replace("12.0", "")


def test_stacked_runs_are_labelled():
    m = _module()
    runs = m.load_runs(REPO / "bench" / "results")
    stacked = [r for r in runs if (r.get("placement") or {}).get("stacked_with")]
    assert stacked, "expected at least one stacked run in bench/results"
    for run in stacked:
        assert "(stacked)" in m.fmt_placement(run["placement"])
