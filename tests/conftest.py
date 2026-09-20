"""Shared pytest fixtures.

Two are global: netdev isolation (per test) and the engine-container guard (per
session, at the bottom of this file).

``ainode.cluster.netdev`` reads the
real host (``ip -o -4 addr show``, ``/sys/class/net``) and caches the answer
per process, so without this fixture the suite would give different results
on a Mac (no ``ip``, no sysfs) than on the Linux CI runner (``ip`` present,
``eth0`` up), and every test that pins ``cluster_interface`` to a DGX Spark
NIC name would see it silently autodetected away to ``eth0``.

So: point the sysfs root at an empty tmpdir, make the command runner return
nothing, and clear the resolution cache around every test. A host with no
detectable interface resolves to the configured name unchanged, which is the
behavior those tests already assert. Tests that WANT detection to fire
monkeypatch these same two seams with their own fakes.
"""

import shutil
import subprocess

import pytest


@pytest.fixture(autouse=True)
def isolate_netdev(monkeypatch, tmp_path_factory):
    from ainode.cluster import netdev

    netdev.reset_cache()
    empty_sysfs = tmp_path_factory.mktemp("sys_class_net_empty")
    monkeypatch.setattr(netdev, "SYS_CLASS_NET", empty_sysfs)
    monkeypatch.setattr(netdev, "_run_command", lambda argv: "")
    yield
    netdev.reset_cache()


def _engine_containers():
    """``{container id: name}`` for every ``ainode-vllm*`` container on this host,
    or None when that cannot be known.

    None means "do not judge": no docker on PATH, no daemon reachable, or the CLI
    answering in a shape we did not ask for. The guard below treats that as a skip
    rather than as an empty mapping, so a machine without docker never reports a
    phantom leak. Keyed by id because that is the stable identity (two runs can
    both produce an ``ainode-vllm-node-solo``); the name is carried along only so
    the failure names something an operator recognizes.
    """
    if shutil.which("docker") is None:
        return None
    try:
        out = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=ainode-vllm",
             "--format", "{{.ID}} {{.Names}}"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    found = {}
    for line in out.stdout.splitlines():
        cid, _, name = line.strip().partition(" ")
        if cid:
            found[cid] = name or cid
    return found


@pytest.fixture(scope="session", autouse=True)
def no_engine_containers_escaped():
    """Fail the session if the unit suite created a real vLLM engine container.

    The suite must never reach the docker boundary: every test that builds an
    engine stubs ``get_backend`` or the ``subprocess`` seam. When one forgets, the
    only evidence is a container left in ``Created`` state on the developer's
    machine, which nothing in a green run points at (#221). So take the census
    before the session and compare after: a new ``ainode-vllm*`` container is a
    test that shelled out to docker for real.

    Deliberately id-based, not a count: a container the operator started by hand
    mid-run is not this suite's doing, and one the suite creates and cleans up
    again is not a leak either.
    """
    before = _engine_containers()
    yield
    if before is None:
        return
    after = _engine_containers()
    if after is None:
        return
    leaked = sorted(after[cid] for cid in after.keys() - before.keys())
    if leaked:
        shown = " ".join(leaked)
        pytest.fail(
            "A test reached the real docker boundary: this run created engine "
            f"container(s) {shown}. The unit suite must stub the engine "
            "(monkeypatch ainode.engine.backends.get_backend, or the subprocess "
            "seam) instead of shelling out to docker. Clean up with "
            f"`docker rm -f {shown}`, then patch the test that launched it.",
            pytrace=False,
        )
