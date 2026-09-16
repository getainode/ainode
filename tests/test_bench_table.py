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


def test_every_throughput_result_file_has_a_row():
    """One row per run that measured throughput.

    A harness-bench record (a `harness` block and no `single_stream`) measured a
    coding agent instead, so every column here would read "not measured"; those are
    filtered out rather than rendered as a very slow model.
    """
    m = _module()
    runs = m.load_runs(REPO / "bench" / "results")
    table = m.render_table(runs)
    shown = m.throughput_runs(runs)
    skipped = [r["_file"] for r in runs if m.is_harness_run(r)]
    assert shown, "no throughput bench results to render"
    for run in shown:
        assert run["_file"] in table, f"{run['_file']} missing from the rendered table"
    for name in skipped:
        assert name not in table, f"{name} is a harness run and should not be a row"
    assert len(table.splitlines()) == len(shown) + 2  # header + rule + one row each


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


def test_a_moe_row_names_its_active_params_and_a_dense_row_says_dense():
    """The params column is how a reader knows why decode is fast: on GB10 it is
    the active count that sets the ceiling, not the total."""
    m = _module()
    assert m.fmt_params({"params_b": 284.0, "active_b": 13.0, "arch": "moe"}) \
        == "284B / 13B active"
    assert m.fmt_params({"params_b": 27.0, "active_b": 27.0, "arch": "dense"}) \
        == "27B dense"
    # A record that says MoE without an active count is still not dense.
    assert m.fmt_params({"params_b": 504.0, "arch": "moe"}) == "504B MoE"
    assert m.fmt_params({}) == m.NOT_MEASURED


def test_stacked_runs_are_labelled():
    m = _module()
    runs = m.load_runs(REPO / "bench" / "results")
    stacked = [r for r in runs if (r.get("placement") or {}).get("stacked_with")]
    assert stacked, "expected at least one stacked run in bench/results"
    for run in stacked:
        assert "(stacked)" in m.fmt_placement(run["placement"])


def _write(tmp_path, name, obj):
    (tmp_path / name).write_text(json.dumps(obj))


def test_harness_record_renders_a_row(tmp_path):
    """A record whose harness block measured one model on one harness becomes a row."""
    m = _module()
    _write(tmp_path, "20260101-000001-harness.json", {
        "schema": 1, "stamp": "20260101-000001", "label": "my-run",
        "model": {"id": "x/y", "name": "Xeno", "params_b": 7, "arch": "moe"},
        "placement": {"node": "Spark-5", "gpu": "NVIDIA GB10", "gpus": 2, "tp": 2},
        "harness": {"runs": [{
            "harness": "dsh", "version": "0.1.5",
            "scores": {"tasks": 10, "passed_at_1": 8, "passed_at_2": 10,
                       "mean_wall_s": 12.5},
        }]},
    })
    table = m.render_harness_table(m.load_runs(tmp_path))
    lines = table.splitlines()
    assert len(lines) == 3  # header + rule + one row
    assert "| Xeno |" in table
    assert "| dsh 0.1.5 | 8/10 | 10/10 | 12.5 | Spark-5, TP=2 | 2026-01-01 |" in table
    assert ("[my-run]("
            "https://github.com/getainode/ainode/blob/main/bench/results/"
            "20260101-000001-harness.json)") in table


def test_note_marks_harness_rows_not_measured(tmp_path):
    """A note that names a harness alongside 'not measured' blanks its score cells."""
    m = _module()
    _write(tmp_path, "20260101-000001-harness.json", {
        "schema": 1, "stamp": "20260101-000001", "label": "my-run",
        "model": {"id": "x/y", "name": "Xeno", "params_b": 7, "arch": "moe"},
        "placement": {"node": "Spark-5", "gpus": 1, "tp": 1},
        "harness": {"runs": [
            {"harness": "dsh", "version": "0.1.5",
             "scores": {"tasks": 10, "passed_at_1": 8, "passed_at_2": 10,
                        "mean_wall_s": 12.5}},
            {"harness": "broken", "version": "broken 9",
             "scores": {"tasks": 10, "passed_at_1": 3, "passed_at_2": 5,
                        "mean_wall_s": 4.0}},
        ]},
        "notes": ["The broken 9 rows are a host defect; treat its scores as "
                  "not measured."],
    })
    table = m.render_harness_table(m.load_runs(tmp_path))
    assert "| 8/10 | 10/10 | 12.5 |" in table  # the good row still has numbers
    broken = [ln for ln in table.splitlines() if "broken 9" in ln][0]
    assert "| not measured | not measured | not measured |" in broken


def test_later_record_wins_for_the_same_pair(tmp_path):
    """Two records for one (model, harness): the newer stamp supplies the row."""
    m = _module()
    _write(tmp_path, "20260101-000001-earlier.json", {
        "schema": 1, "stamp": "20260101-000001", "label": "early",
        "model": {"id": "x/y", "name": "Xeno", "params_b": 7, "arch": "moe"},
        "placement": {"node": "Spark-5", "gpus": 1, "tp": 1},
        "harness": {"runs": [{
            "harness": "dsh", "version": "0.1.5",
            "scores": {"tasks": 10, "passed_at_1": 2, "passed_at_2": 4,
                       "mean_wall_s": 99.0},
        }]},
    })
    _write(tmp_path, "20260201-000001-later.json", {
        "schema": 1, "stamp": "20260201-000001", "label": "late",
        "model": {"id": "x/y", "name": "Xeno", "params_b": 7, "arch": "moe"},
        "placement": {"node": "Spark-6", "gpus": 1, "tp": 1},
        "harness": {"runs": [{
            "harness": "dsh", "version": "0.2.0",
            "scores": {"tasks": 10, "passed_at_1": 9, "passed_at_2": 10,
                       "mean_wall_s": 30.0},
        }]},
    })
    table = m.render_harness_table(m.load_runs(tmp_path))
    assert "| 9/10 | 10/10 | 30.0 |" in table
    assert "99.0" not in table  # the earlier row is gone, not merged
    assert "| Spark-6, TP=1 |" in table
    assert "20260201-000001-later.json" in table


def test_check_detects_stale_harness_table(tmp_path):
    """--check returns 1 when only the harness table of README.md has drifted."""
    results = tmp_path / "results"
    results.mkdir()
    _write(results, "throughput.json", {
        "schema": 1, "stamp": "20260101-000001", "label": "thr",
        "model": {"id": "x/y", "name": "Thr", "params_b": 7, "active_b": 7,
                  "arch": "dense"},
        "placement": {"node": "N", "gpus": 1, "tp": 1},
        "results": {"single_stream": {"decode_tok_s": 10.0},
                    "concurrency": [{"streams": 16, "aggregate_tok_s": 5.0}]},
    })
    _write(results, "harness.json", {
        "schema": 1, "stamp": "20260101-000002", "label": "har",
        "model": {"id": "a/b", "name": "Har", "params_b": 7, "arch": "dense"},
        "placement": {"node": "Spark-7", "gpus": 1, "tp": 1},
        "harness": {"runs": [{
            "harness": "dsh", "version": "0.1.5",
            "scores": {"tasks": 10, "passed_at_1": 8, "passed_at_2": 10,
                       "mean_wall_s": 20.0},
        }]},
    })

    readme = tmp_path / "README.md"
    m = _module()
    readme.write_text(
        "# Bench\n\n"
        f"{m.BEGIN}\n\n{m.render_table(m.load_runs(results))}\n\n{m.END}\n\n"
        f"{m.HARNESS_BEGIN}\n\n{m.render_harness_table(m.load_runs(results))}\n\n"
        f"{m.HARNESS_END}\n"
    )

    def check():
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--check", "--results", str(results),
             "--readme", str(readme), "--harness-begin", m.HARNESS_BEGIN,
             "--harness-end", m.HARNESS_END],
            capture_output=True, text=True, cwd=REPO)
        return proc.returncode

    assert check() == 0
    text = readme.read_text().replace("8/10", "7/10")
    readme.write_text(text)
    assert check() == 1

