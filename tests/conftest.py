"""Shared pytest fixtures.

Six are global, all there to keep the suite from reading or touching the machine
it runs on: netdev isolation, the metrics-store redirect, the users-store
redirect, the boot-reconcile guard and the Hub size lookup (per test), and the
engine-container guard (per session, at the bottom of this file).

``isolate_users_store``: ``create_app`` loads the login accounts
(``ainode/auth/accounts.py``) the way it loads ``auth.json``, so without this
every test that starts an application would read, and any test that created an
account would WRITE, the developer's own ``~/.ainode/users.json``. Point the
module constant at a temporary file per test. A test that wants a specific path
passes one to ``UsersStore``, which this does not touch.

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

``no_hub_size_lookup``: the download fit check learns a checkpoint's size from
the Hub's file metadata before it lets a pull start, and remembers the answer
under ``AINODE_HOME``. Both are the machine, so both are redirected: the lookup
reports unknown (which lets the download through, exactly as before the check
existed) and the cache lands in a tmpdir.

``no_bench_api_key``: the bench CLI reads ``$AINODE_API_KEY`` when no ``--api-key``
was passed (#245), and the operator running this suite usually has one exported for
their own fleet. Without this, their key would decide what the harness dry-run
prints and whether a key-source line says "the default", so the tests would pass or
fail depending on whose shell they ran in. Tests that want the variable set do it
themselves with ``monkeypatch.setenv``, which runs after this.

``no_bench_preflight``: every bench section now opens with one GET of the endpoint's
model list to find out whether the node will refuse the run (#245). The endpoints in
these tests are real fleet addresses, so on a machine with the tailnet up that GET
would leave the suite and its answer would decide the test. Stubbed to "not
refused". The tests for the preflight call the real function with a faked
``urlopen``, and the tests for a refusal set this seam to their own answer.
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
def isolate_users_store(monkeypatch, tmp_path):
    """Keep the login accounts out of the operator's real AINODE_HOME.

    ``UsersStore`` resolves ``accounts.USERS_FILE`` at call time rather than in
    ``__init__`` precisely so this one patch reaches a store that already exists,
    the same way the ``auth_home`` fixtures redirect ``AUTH_FILE``.
    """
    from ainode.auth import accounts

    monkeypatch.setattr(accounts, "USERS_FILE", tmp_path / "users.json")
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

    The reconciler's two DOCKER seams are stubbed for the same reason, but on
    ``reconcile`` itself: since #240 the boot adoption decision runs in
    ``cmd_start`` before the sweep, so any test that drives the CLI start path
    would otherwise ask the real docker on the machine running the suite what it
    is serving. A test that exercises adoption fakes those seams itself and its
    own patch wins.
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
    monkeypatch.setattr(reconcile, "inspect_container", lambda name: None)
    monkeypatch.setattr(reconcile, "list_engine_containers", lambda: [])
    reconcile.reset_state_for_tests()
    yield
    reconcile.reset_state_for_tests()


@pytest.fixture(autouse=True)
def no_hub_size_lookup(monkeypatch, tmp_path_factory):
    """The fit check never asks huggingface.co, and never writes the real cache.

    ``ainode/models/fit.py`` learns a checkpoint's size from the Hub before a
    download starts (#184 point 4), so without this every test that posts to a
    download route would make a real HTTP call from the suite, and remembering
    the answer would write into the developer's own ``~/.ainode``. Report unknown
    instead, which is the "nobody could say" branch and the one that lets a
    download through unchanged. The tests for the check hand ``check_fit`` its own
    sizes, or patch this seam with a fake of their own.
    """
    from ainode.models import fit

    home = tmp_path_factory.mktemp("ainode_fit_home")
    monkeypatch.setattr(fit, "fetch_repo_size", lambda *a, **k: fit.RepoSize())
    monkeypatch.setattr(fit, "size_cache_path", lambda: home / "repo-sizes.json")
    yield


@pytest.fixture(autouse=True)
def no_bench_api_key(monkeypatch):
    """The bench never picks up the operator's own API key during the suite.

    ``ainode/bench/auth.py`` falls back to ``$AINODE_API_KEY`` when no ``--api-key``
    was given, so a developer with one exported would change what the CLI tests see:
    a key source of ``$AINODE_API_KEY`` instead of "the default", and a masked
    placeholder in the harness dry-run's env line. A test that wants the variable
    sets it itself.
    """
    from ainode.bench.auth import ENV_API_KEY

    monkeypatch.delenv(ENV_API_KEY, raising=False)
    yield


@pytest.fixture(autouse=True)
def no_bench_preflight(monkeypatch):
    """The bench's opening GET never leaves the suite.

    Each section asks the endpoint for its model list before it measures anything, so
    a refusing node is reported rather than scored (#245). The endpoints these tests
    pass are real addresses on Jason's fleet, and a run of the suite with the tailnet
    up would make that request for real and let the answer decide the test. Report
    "not refused" instead. A test about a refusal sets this same seam to its own
    answer, which wins because it patches later; the tests for the preflight itself
    call the real function with a faked ``urlopen``.
    """
    from ainode.bench import auth

    monkeypatch.setattr(auth, "preflight", lambda *a, **k: None)
    yield


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
