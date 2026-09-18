"""The harness bench refuses to run on an interpreter that cannot import pytest,
and a test run that dies with "No module named pytest" stops the bench instead of
scoring the task as a model failure."""
import subprocess
import sys

import pytest

from ainode.bench.harness import runner
from ainode.bench.harness.runner import (
    NO_PYTEST, HiddenTestsUnavailable, preflight_test_interpreter, run_tests,
)


def _task(tmp_path):
    return runner.Task(slug="bob", language="python", directory=tmp_path, entry="bob.py",
                       instructions_file="instructions.md", test_files=("bob_test.py",),
                       test_command=("python", "-m", "pytest", "-q", "bob_test.py"), source={})


def test_preflight_passes_on_the_interpreter_running_the_tests():
    assert preflight_test_interpreter(sys.executable) is None


def test_preflight_names_an_interpreter_without_pytest(tmp_path):
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\necho 'ModuleNotFoundError: No module named pytest' >&2\nexit 1\n")
    fake.chmod(0o755)
    reason = preflight_test_interpreter(str(fake))
    assert reason and "cannot import pytest" in reason and str(fake) in reason


def test_preflight_reports_an_interpreter_that_cannot_start(tmp_path):
    reason = preflight_test_interpreter(str(tmp_path / "missing-python"))
    assert reason and "could not run" in reason


def test_run_tests_stops_the_bench_when_pytest_is_missing(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="",
                                           stderr=f"/usr/bin/python3: {NO_PYTEST}\n")
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    task = _task(tmp_path)
    with pytest.raises(HiddenTestsUnavailable, match="bob"):
        run_tests(task, tmp_path)


def test_run_tests_still_records_an_ordinary_failure(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="1 failed, 5 passed in 0.02s\n", stderr="")
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    task = _task(tmp_path)
    result = run_tests(task, tmp_path)
    assert result.passed is False and result.exit_code == 1
