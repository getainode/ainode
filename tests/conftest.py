"""Shared pytest fixtures.

The only global one is netdev isolation. ``ainode.cluster.netdev`` reads the
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
