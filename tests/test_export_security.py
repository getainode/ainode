"""Keep the publication workflow and exported tree within their intended scope."""

from pathlib import Path
import re
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _workflow_paths():
    directory = ROOT / ".github/workflows"
    return sorted([*directory.glob("*.yml"), *directory.glob("*.yaml")])


def _workflow_or_no_ci(name):
    path = ROOT / ".github/workflows" / name
    if path.exists():
        return path
    assert not _workflow_paths(), "Unexpected workflows in the no-CI snapshot"
    return None


def test_release_artifact_jobs_require_release_environment():
    if _workflow_or_no_ci("publish-image.yml") is None:
        return
    artifact_jobs = []
    for path in _workflow_paths():
        active = "\n".join(
            line for line in path.read_text().splitlines()
            if not line.lstrip().startswith("#")
        )
        job_source = active.partition("\njobs:\n")[2]
        jobs = re.findall(r"^  ([\w-]+):\n((?:[ \t]{4}.*\n|\n)*)", job_source + "\n", re.M)
        for name, body in jobs:
            if not re.search(r"docker/(?:login|build-push)-action@|\bdocker (?:build|push)\b", body):
                continue
            artifact_jobs.append((path.name, name))
            assert re.search(r"^    environment: release\s*$", body, re.M), (path.name, name)
    assert artifact_jobs, "No release artifact jobs found"


def test_publish_image_has_no_docker_hub_steps():
    path = _workflow_or_no_ci("publish-image.yml")
    if path is None:
        return
    workflow = path.read_text()
    active = "\n".join(line for line in workflow.splitlines() if not line.lstrip().startswith("#"))
    assert not re.search(r"docker[._ -]?hub|docker\.io|argentai?os/|REGISTRY_DH", active, re.I)
    registries = re.findall(r"registry:\s*(\S+)", active)
    assert registries == ["ghcr.io"]
    pushes = re.findall(r"docker push\s+([^\n]+)", active)
    assert pushes
    assert all('${REGISTRY_GHCR}/' in push for push in pushes)


@pytest.mark.parametrize("suffix", ["yml", "yaml"])
def test_missing_publish_workflow_requires_no_other_workflows(tmp_path, monkeypatch, suffix):
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    test_release_artifact_jobs_require_release_environment()
    test_publish_image_has_no_docker_hub_steps()
    directory = tmp_path / ".github/workflows"
    directory.mkdir(parents=True)
    # Even a release-gated job under another name cannot stand in for no CI.
    (directory / f"unexpected.{suffix}").write_text(
        "name: synthetic\njobs:\n  publish:\n    environment: release\n"
        "    steps:\n      - uses: docker/build-push-action@v6\n"
    )
    for check in (test_release_artifact_jobs_require_release_environment,
                  test_publish_image_has_no_docker_hub_steps):
        with pytest.raises(AssertionError, match="Unexpected workflows"):
            check()


def test_gitignore_does_not_hide_orca_files():
    rules = (ROOT / ".gitignore").read_text().splitlines()
    assert not any(".orca" in rule for rule in rules if not rule.lstrip().startswith("#"))
