"""Shared pytest fixtures.

Four are global, all there to keep the suite from reading or touching the machine
it runs on: netdev isolation, the metrics-store redirect and the boot-reconcile
guard (per test), and the
engine-container guard (per session, at the bottom of this file).

``isolate_netdev``: ``ainode.cluster.netdev`` reads the
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

``isolate_metrics_store``: ``ainode.metrics.store.default_store_path`` resolves
``AINODE_HOME`` at call time and pytest does not set it, so without this every
test that starts an application would open, sample into and prune the operator's
own ``~/.ainode/metrics.db``. Point it at a temporary file instead. A test that
cares about the path hands one to ``MetricsStore`` directly, which this does not
touch.

``no_boot_reconcile``: ``create_app``'s startup schedules the instance replay,
whose first step asks docker which engine containers this node is already
running and can then relaunch a recorded distributed shape over ssh
(``ainode/engine/reconcile.py``, #179). Neither belongs in a route test that
merely wants an app: on a machine with a real docker and live engines that
background task would adopt real containers into a test app, a run on a head
could try to relaunch its shape, and a test that patches ``subprocess.run`` and
asserts it was not called races it (observed as a flaky
``tests/test_api.py::test_engine_update_unresolvable_version``). The tests that
exercise reconciliation call it directly, which these no-ops do not affect.
"""

import shutil
import subprocess

import pytest


@pytest.fixture(autouse=True)
def isolate_metrics_store(monkeypatch, tmp_path_factory):
    """Keep the on-disk sample store out of the operator's real AINODE_HOME.

    ``ainode.metrics.store.default_store_path`` resolves ``AINODE_HOME`` at call
    time and pytest does not set it, so without this every test that starts an
    application would open, sample into and prune the developer's own
    ``~/.ainode/metrics.db``. Point it at a temporary file per test instead. A
    test that cares about the path hands one to ``MetricsStore`` directly, which
    this does not touch.
    """
    from ainode.metrics import store as metrics_store

    home = tmp_path_factory.mktemp("ainode_metrics_home")
    monkeypatch.setattr(
        metrics_store, "default_store_path", lambda: home / "metrics.db"
    )
    yield


@pytest.fixture(autouse=True)
def isolate_netdev(monkeypatch, tmp_path_factory):
    from ainode.cluster import netdev

    netdev.reset_cache()
    empty_sysfs = tmp_path_factory.mktemp("sys_class_net_empty")
    monkeypatch.setattr(netdev, "SYS_CLASS_NET", empty_sysfs)
    monkeypatch.setattr(netdev, "_run_command", lambda argv: "")
    yield
    netdev.reset_cache()


@pytest.fixture(autouse=True)
def no_boot_reconcile(monkeypatch):
    """The boot replay's reconcile step is a no-op unless a test asks for it.

    Patched where the replay reads them, on ``models.api_routes``, so a direct
    call to ``ainode.engine.reconcile.adopt_running_engines`` (what the tests for
    this behaviour make) still runs the real thing.
    """
    from ainode.engine import reconcile
    from ainode.models import api_routes

    async def _no_adopt(app):
        return []

    async def _no_distributed_replay(app):
        return {"action": "none"}

    monkeypatch.setattr(api_routes, "adopt_running_engines", _no_adopt)
    monkeypatch.setattr(api_routes, "replay_distributed_if_needed",
                        _no_distributed_replay)
    reconcile.reset_state_for_tests()
    yield
    reconcile.reset_state_for_tests()


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
