"""TLS: the certificate, the config block, the two listeners, the doctor line.

What is pinned here, and why each one is worth a test:

* **The SAN list.** A certificate with no matching SAN is refused by every modern
  client, and a node is reached by three different names (hostname, LAN IP,
  tailnet IP). The generated pair has to carry all of them, and the SAN list has
  to reach ``openssl`` rather than being computed and dropped.
* **Key mode 0600.** The pair lives in a bind-mounted directory; a world-readable
  private key there is readable by anything else that mounts it.
* **The config round trip**, including the rule that writing the ``tls`` block
  leaves every other key in config.json alone (``NodeConfig.load`` filters to its
  own fields, so a load-then-save would silently drop what this release does not
  know).
* **Both listeners when TLS is on, and only HTTP when it cannot be.** The failure
  direction matters: a broken certificate must not cost the node its HTTP port,
  because the dashboard is where an operator fixes the certificate.
* **The doctor's branches**, driven with a fake ``certificate_info`` so expiry is
  a value rather than a clock.
* **The host wrapper's ``tls`` path**, run for real with tailscale, docker, sudo,
  systemctl and openssl replaced by recording stubs. ``tailscale cert`` is a HOST
  operation and the CLI is in a container, so which binary is called where and
  with what IS the feature.
* **The renewal decision**, which is a pure function so the timer, the CLI and
  these tests cannot disagree, and its refusals (a self-signed pair, a name
  outside the tailnet, an unreadable file) matter as much as its approvals
  because the caller acts as root.
* **What a node advertises when TLS is on**: ``url`` carries the scheme actually
  served, ``port`` stays the HTTP port every peer uses, and only this node's own
  rows may claim TLS.

Nothing here reaches the network. Certificates are generated for real (openssl is
present on any machine that can build this project) because a certificate the
test writes itself is the only way to check what ends up inside one.
"""

import json
import os
import shutil
import socket
import ssl
import subprocess
import time
from pathlib import Path

import aiohttp
import pytest

from ainode.api.server import listener_plan, ssl_context, start_sites, tls_status_fields
from ainode.cli import doctor as doc
from ainode.core.config import NodeConfig
from ainode.tls import certs as certs_mod
from ainode.tls.certs import (
    certificate_info,
    generate_self_signed,
    host_ip_addresses,
    openssl_config,
    san_entries,
)
from ainode.tls.config import (
    DEFAULT_TLS_PORT,
    TLSConfig,
    cert_paths,
    install_pair,
    load_tls_config,
    save_tls_config,
    tls_dir,
)
from ainode.tls.renew import (
    RENEW_THRESHOLD_DAYS,
    renewal_decision,
    tailnet_name_from_sans,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point AINODE_HOME at a tmpdir so nothing touches the operator's ~/.ainode."""
    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    return tmp_path


@pytest.fixture
def pair(home):
    """A real self-signed pair under <home>/tls, with known names."""
    cert, key = cert_paths()
    made = generate_self_signed(cert, key, hostname="spark-3",
                                ips=["192.168.100.87", "100.80.240.119"])
    return made


# =============================================================================
# Generating a certificate
# =============================================================================

def test_the_san_list_carries_hostname_lan_ip_and_tailnet_ip():
    entries = san_entries(hostname="spark-3",
                          ips=["192.168.100.87", "100.80.240.119"])
    assert ("DNS", "spark-3") in entries
    assert ("IP", "192.168.100.87") in entries
    assert ("IP", "100.80.240.119") in entries
    # localhost is always in it: curl on the node itself is the first thing tried.
    assert ("DNS", "localhost") in entries
    assert ("IP", "127.0.0.1") in entries


def test_the_san_list_is_deduplicated_and_drops_nothing_real():
    entries = san_entries(hostname="spark-3.tail1234.ts.net",
                          ips=["10.0.0.5", "10.0.0.5"])
    assert entries.count(("IP", "10.0.0.5")) == 1
    # A dotted name contributes its short form too, because that is what an
    # operator types into a browser on the LAN.
    assert ("DNS", "spark-3.tail1234.ts.net") in entries
    assert ("DNS", "spark-3") in entries


def test_a_generated_certificate_carries_every_san_and_expires_in_825_days(home):
    cert, key = cert_paths()
    made = generate_self_signed(cert, key, hostname="spark-3",
                                ips=["192.168.100.87", "100.80.240.119"])
    assert made["days"] == certs_mod.CERT_DAYS == 825
    info = certificate_info(cert)
    assert info["exists"] is True
    assert info["error"] == ""
    assert info["self_signed"] is True
    assert "DNS:spark-3" in info["sans"]
    assert "IP:192.168.100.87" in info["sans"]
    assert "IP:100.80.240.119" in info["sans"]
    # 825 days out, allowing a day of slop for the not-before backdate.
    assert 823 <= info["days_left"] <= 826


def test_the_private_key_is_owner_only_and_the_cert_is_not(home):
    cert, key = cert_paths()
    generate_self_signed(cert, key, hostname="spark-3", ips=["10.0.0.5"])
    assert oct(os.stat(key).st_mode & 0o777) == "0o600"
    assert os.stat(cert).st_mode & 0o044  # readable, it is a public document


def test_generation_leaves_no_openssl_config_behind(home):
    cert, key = cert_paths()
    generate_self_signed(cert, key, hostname="spark-3", ips=["10.0.0.5"])
    assert sorted(p.name for p in tls_dir().iterdir()) == ["cert.pem", "key.pem"]


def test_the_san_list_actually_reaches_openssl(home):
    """The SANs are not just computed: they are in the config openssl is given."""
    seen: list = []

    def fake_runner(argv):
        seen.append(list(argv))
        conf = Path(argv[argv.index("-config") + 1])
        seen.append(conf.read_text())
        # openssl would write the pair; the caller only chmods what exists.
        Path(argv[argv.index("-out") + 1]).write_text("cert")
        Path(argv[argv.index("-keyout") + 1]).write_text("key")
        return 0, ""

    cert, key = cert_paths()
    made = generate_self_signed(cert, key, hostname="spark-3",
                                ips=["192.168.100.87"], runner=fake_runner,
                                prefer="openssl")
    argv, conf_text = seen[0], seen[1]
    assert Path(argv[0]).name == "openssl"
    assert argv[1:4] == ["req", "-x509", "-newkey"]
    assert "-days" in argv and argv[argv.index("-days") + 1] == "825"
    assert "DNS.1 = spark-3" in conf_text
    assert "IP.1 = 192.168.100.87" in conf_text
    assert made["tool"] == "openssl"


def test_the_openssl_config_numbers_dns_and_ip_entries_separately():
    text = openssl_config([("DNS", "a"), ("DNS", "b"), ("IP", "1.1.1.1")])
    assert "DNS.1 = a" in text
    assert "DNS.2 = b" in text
    assert "IP.1 = 1.1.1.1" in text
    assert "extendedKeyUsage = serverAuth" in text


def test_generation_without_openssl_or_cryptography_says_so(home, monkeypatch):
    monkeypatch.setattr(certs_mod, "have_cryptography", lambda: False)
    monkeypatch.setattr(certs_mod, "openssl_binary", lambda: None)
    cert, key = cert_paths()
    with pytest.raises(RuntimeError, match="no openssl binary"):
        generate_self_signed(cert, key, hostname="spark-3", ips=["10.0.0.5"])


def test_host_ip_addresses_reads_ip_addr_and_keeps_the_tailnet_address():
    """tailscale0 is exactly the address that must NOT be filtered out."""
    out = (
        "1: lo    inet 127.0.0.1/8 scope host lo\\       valid_lft forever\n"
        "2: enp1s0f0 inet 192.168.100.87/24 brd 192.168.100.255 scope global\n"
        "3: tailscale0 inet 100.80.240.119/32 scope global tailscale0\n"
        "4: docker0 inet 169.254.1.1/16 scope link\n"
    )
    addrs = host_ip_addresses(runner=lambda argv: (0, out))
    assert addrs == ["192.168.100.87", "100.80.240.119"]


# =============================================================================
# tailscale cert
# =============================================================================

def test_the_magicdns_name_comes_off_tailscale_status():
    out = '{"Self": {"DNSName": "spark-3-dgx.tailed10d2.ts.net.", "Online": true}}'
    name = certs_mod.tailscale_dns_name(runner=lambda argv: (0, out))
    # The trailing dot is real in that field and wrong in a certificate.
    assert name == "spark-3-dgx.tailed10d2.ts.net"


@pytest.mark.parametrize("answer", [
    (1, "tailscaled not running"),   # daemon down
    (0, ""),                          # nothing to parse
    (0, "not json"),                  # a future format
    (0, '{"Self": {}}'),              # no name yet
])
def test_no_magicdns_name_is_none_rather_than_a_crash(answer):
    assert certs_mod.tailscale_dns_name(runner=lambda argv: answer) is None


def test_tailscale_cert_writes_into_the_tls_dir_and_hardens_the_key(home):
    seen: list = []

    def fake_runner(argv):
        seen.append(list(argv))
        Path(argv[argv.index("--cert-file") + 1]).write_text("leaf+chain")
        Path(argv[argv.index("--key-file") + 1]).write_text("key")
        return 0, "wrote cert"

    cert, key = cert_paths()
    made = certs_mod.tailscale_cert("spark-3-dgx.tailed10d2.ts.net", cert, key,
                                   runner=fake_runner)
    assert seen[0][:2] == ["tailscale", "cert"]
    assert seen[0][-1] == "spark-3-dgx.tailed10d2.ts.net"
    assert made["self_signed"] is False
    assert oct(os.stat(key).st_mode & 0o777) == "0o600"


def test_a_failed_tailscale_cert_says_so_and_writes_no_config(home):
    with pytest.raises(RuntimeError, match="tailscale cert failed"):
        certs_mod.tailscale_cert("nope.ts.net", *cert_paths(),
                                 runner=lambda argv: (1, "HTTPS is not enabled"))


# =============================================================================
# Reading a certificate
# =============================================================================

def test_certificate_info_on_a_missing_file_is_an_answer_not_an_exception(home):
    info = certificate_info(tls_dir() / "nope.pem")
    assert info["exists"] is False
    assert info["expires"] is None
    assert info["error"]


def test_certificate_info_on_garbage_reports_the_error(home):
    path = home / "garbage.pem"
    path.write_text("this is not a certificate\n")
    info = certificate_info(path, use_cache=False)
    assert info["expires"] is None
    assert info["error"]


def test_certificate_info_recomputes_days_left_against_the_clock(home, pair):
    cert = pair["cert_file"]
    fresh = certificate_info(cert)
    later = certificate_info(cert, now=time.time() + 800 * 86400)
    assert fresh["days_left"] > later["days_left"]
    assert 20 <= later["days_left"] <= 30


# =============================================================================
# The config block
# =============================================================================

def test_the_default_block_is_disabled_on_port_3443():
    tls = TLSConfig()
    assert (tls.enabled, tls.port) == (False, 3443)
    assert DEFAULT_TLS_PORT == 3443
    assert NodeConfig().tls == {}
    assert load_tls_config(NodeConfig()).enabled is False


@pytest.mark.parametrize("block,expected_port", [
    ({}, 3443),
    ({"port": None}, 3443),
    ({"port": "not a port"}, 3443),
    ({"port": 0}, 3443),
    ({"port": 99999}, 3443),
    ({"port": "3480"}, 3480),
    ({"port": 3480}, 3480),
])
def test_a_hand_edited_port_never_takes_the_web_server_down(block, expected_port):
    """Read on the boot path: a typo must not be why a node has no API at all."""
    assert TLSConfig.from_dict(block).port == expected_port


def test_the_block_round_trips_through_config_json(home):
    config_path = home / "config.json"
    config_path.write_text(json.dumps({
        "node_id": "spark-3", "web_port": 3080,
        "some_key_a_future_release_added": "keep me",
    }))
    written = save_tls_config(
        TLSConfig(enabled=True, cert_file="/root/.ainode/tls/cert.pem",
                  key_file="/root/.ainode/tls/key.pem", port=3480),
        config_path=config_path,
    )
    assert written == config_path
    raw = json.loads(config_path.read_text())
    # Every other key survived: the write is one key, not a rewrite.
    assert raw["node_id"] == "spark-3"
    assert raw["some_key_a_future_release_added"] == "keep me"
    assert raw["tls"] == {
        "enabled": True,
        "cert_file": "/root/.ainode/tls/cert.pem",
        "key_file": "/root/.ainode/tls/key.pem",
        "port": 3480,
    }
    # And NodeConfig reads it back as the same block.
    config = NodeConfig(**{k: v for k, v in raw.items()
                          if k in NodeConfig.__dataclass_fields__})
    tls = load_tls_config(config)
    assert (tls.enabled, tls.port, tls.cert_file) == (
        True, 3480, "/root/.ainode/tls/cert.pem")


def test_saving_the_block_into_a_missing_config_file_creates_it(home):
    path = home / "sub" / "config.json"
    save_tls_config(TLSConfig(enabled=True, port=3481), config_path=path)
    assert json.loads(path.read_text())["tls"]["port"] == 3481


def test_an_operator_supplied_pair_is_copied_into_the_tls_dir(home):
    """A path outside AINODE_HOME does not exist inside the container."""
    outside = home / "letsencrypt"
    outside.mkdir()
    (outside / "fullchain.pem").write_text("cert")
    (outside / "privkey.pem").write_text("key")
    cert, key = install_pair(outside / "fullchain.pem", outside / "privkey.pem")
    assert cert == tls_dir() / "fullchain.pem"
    assert key == tls_dir() / "privkey.pem"
    assert cert.read_text() == "cert"
    assert oct(os.stat(key).st_mode & 0o777) == "0o600"


def test_installing_a_pair_that_is_not_there_says_which_file(home):
    with pytest.raises(FileNotFoundError):
        install_pair(home / "nope.pem", home / "nope.key")


def test_usable_needs_the_files_to_be_on_disk(home, pair):
    good = TLSConfig(enabled=True, cert_file=pair["cert_file"],
                     key_file=pair["key_file"])
    assert good.usable() is True
    assert TLSConfig(enabled=False, cert_file=pair["cert_file"],
                     key_file=pair["key_file"]).usable() is False
    assert TLSConfig(enabled=True, cert_file=str(home / "gone.pem"),
                     key_file=pair["key_file"]).usable() is False


# =============================================================================
# The listeners
# =============================================================================

def test_only_http_is_listened_on_when_tls_is_off():
    plan = listener_plan(NodeConfig(host="0.0.0.0", web_port=3000))
    assert plan == [("0.0.0.0", 3000, None)]


def test_both_listeners_are_registered_when_tls_is_on(home, pair):
    config = NodeConfig(host="0.0.0.0", web_port=3080, tls={
        "enabled": True, "port": 3480,
        "cert_file": pair["cert_file"], "key_file": pair["key_file"],
    })
    plan = listener_plan(config)
    assert len(plan) == 2
    assert plan[0] == ("0.0.0.0", 3080, None)
    host, port, context = plan[1]
    assert (host, port) == ("0.0.0.0", 3480)
    assert isinstance(context, ssl.SSLContext)


def test_a_missing_pair_costs_the_tls_port_and_not_the_http_port(home):
    config = NodeConfig(web_port=3080, tls={
        "enabled": True, "port": 3480,
        "cert_file": str(home / "tls" / "gone.pem"),
        "key_file": str(home / "tls" / "gone.key"),
    })
    plan = listener_plan(config)
    assert plan == [(config.host, 3080, None)]


def test_an_unloadable_pair_costs_the_tls_port_and_not_the_http_port(home):
    bad_cert = home / "bad.pem"
    bad_cert.write_text("not a certificate")
    bad_key = home / "bad.key"
    bad_key.write_text("not a key")
    config = NodeConfig(web_port=3080, tls={
        "enabled": True, "port": 3480,
        "cert_file": str(bad_cert), "key_file": str(bad_key),
    })
    assert listener_plan(config) == [(config.host, 3080, None)]


def test_tls_on_the_http_port_is_refused(home, pair):
    """One port cannot serve both, and HTTP is the one every client uses."""
    config = NodeConfig(web_port=3480, tls={
        "enabled": True, "port": 3480,
        "cert_file": pair["cert_file"], "key_file": pair["key_file"],
    })
    assert listener_plan(config) == [(config.host, 3480, None)]


@pytest.mark.asyncio
async def test_a_tls_port_something_else_holds_costs_only_the_tls_port(home, pair):
    """A stale process on 3443 must not take the dashboard down with it."""
    from aiohttp import web

    async def ok(_request):
        return web.json_response({"status": "ok"})

    app = web.Application()
    app.router.add_get("/api/health", ok)
    with socket.socket() as taken, socket.socket() as spare:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        tls_port = taken.getsockname()[1]
        spare.bind(("127.0.0.1", 0))
        http_port = spare.getsockname()[1]
    runner = web.AppRunner(app)
    await runner.setup()
    with socket.socket() as holder:
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", tls_port))
        holder.listen(1)
        context = ssl_context(pair["cert_file"], pair["key_file"])
        plan = [("127.0.0.1", http_port, None), ("127.0.0.1", tls_port, context)]
        try:
            started = await start_sites(runner, plan)
            assert started == 1
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        f"http://127.0.0.1:{http_port}/api/health") as resp:
                    assert resp.status == 200
        finally:
            await runner.cleanup()


@pytest.mark.asyncio
async def test_an_http_port_that_cannot_bind_is_still_fatal(home):
    """Same answer web.run_app gives: a node with no HTTP port is not a node."""
    from aiohttp import web

    runner = web.AppRunner(web.Application())
    await runner.setup()
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        port = holder.getsockname()[1]
        try:
            with pytest.raises(OSError):
                await start_sites(runner, [("127.0.0.1", port, None)])
        finally:
            await runner.cleanup()


# =============================================================================
# /api/status
# =============================================================================

def test_status_says_http_only_when_tls_is_off():
    fields = tls_status_fields(NodeConfig(web_port=3000))
    assert fields["enabled"] is False
    assert fields["port"] == 3443
    assert fields["cert_expires"] is None
    assert "no TLS" in fields["label"]


def test_status_reports_the_certificate_when_tls_is_on(home, pair):
    config = NodeConfig(web_port=3080, tls={
        "enabled": True, "port": 3480,
        "cert_file": pair["cert_file"], "key_file": pair["key_file"],
    })
    fields = tls_status_fields(config)
    assert fields["enabled"] is True
    assert fields["port"] == 3480
    assert fields["self_signed"] is True
    assert fields["cert_expires"].endswith("Z")
    assert fields["cert_days_left"] > 800
    assert fields["label"] == "HTTPS on 3480, self-signed certificate"


def test_status_says_when_an_enabled_certificate_is_not_there(home):
    config = NodeConfig(tls={"enabled": True, "port": 3480,
                            "cert_file": str(home / "gone.pem"),
                            "key_file": str(home / "gone.key")})
    fields = tls_status_fields(config)
    assert fields["cert_expires"] is None
    assert "is not there" in fields["label"]


def test_status_says_when_tls_is_enabled_with_no_certificate_at_all():
    fields = tls_status_fields(NodeConfig(tls={"enabled": True}))
    assert "no certificate configured" in fields["label"]


# =============================================================================
# ainode doctor
# =============================================================================

def _fake_info(days_left, **extra):
    info = {"path": "/root/.ainode/tls/cert.pem", "exists": True,
            "expires": "2027-01-01T00:00:00Z", "expires_epoch": 1.0,
            "days_left": days_left, "self_signed": True, "sans": [], "error": ""}
    info.update(extra)
    return info


def _tls_config(**block):
    base = {"enabled": True, "port": 3443,
            "cert_file": "/root/.ainode/tls/cert.pem",
            "key_file": "/root/.ainode/tls/key.pem"}
    base.update(block)
    return NodeConfig(web_port=3000, tls=base)


def test_doctor_warns_when_there_is_no_tls_and_names_the_command():
    check = doc.check_tls(NodeConfig(web_port=3000))[0]
    assert (check.name, check.status) == ("tls.enabled", doc.WARN)
    assert "ainode tls enable" in check.fix
    assert check.data["enabled"] is False


def test_doctor_is_ok_on_a_certificate_with_time_left(monkeypatch):
    monkeypatch.setattr(doc, "certificate_info", lambda p: _fake_info(400.0))
    check = doc.check_tls(_tls_config())[0]
    assert check.status == doc.OK
    assert "HTTPS on 3443" in check.detail
    assert "HTTP still on 3000" in check.detail


def test_doctor_warns_inside_the_last_two_weeks(monkeypatch):
    monkeypatch.setattr(doc, "certificate_info", lambda p: _fake_info(13.0))
    check = doc.check_tls(_tls_config())[0]
    assert check.status == doc.WARN
    assert "expires in 13.0 days" in check.detail
    assert doc.TLS_EXPIRY_WARN_DAYS == 14


def test_doctor_fails_on_an_expired_certificate(monkeypatch):
    monkeypatch.setattr(doc, "certificate_info", lambda p: _fake_info(-3.0))
    check = doc.check_tls(_tls_config())[0]
    assert check.status == doc.FAIL
    assert "expired 3.0 days ago" in check.detail
    # A FAIL is the only status that makes the command exit non-zero.
    assert doc.exit_code([check]) == 1


def test_doctor_fails_when_the_enabled_certificate_is_gone(monkeypatch):
    monkeypatch.setattr(doc, "certificate_info",
                        lambda p: _fake_info(None, exists=False))
    check = doc.check_tls(_tls_config())[0]
    assert check.status == doc.FAIL
    assert "is not there" in check.detail


def test_doctor_fails_when_tls_is_enabled_with_no_pair_configured():
    check = doc.check_tls(_tls_config(cert_file="", key_file=""))[0]
    assert check.status == doc.FAIL
    assert "no certificate configured" in check.detail


def test_doctor_warns_when_the_certificate_cannot_be_read(monkeypatch):
    monkeypatch.setattr(doc, "certificate_info",
                        lambda p: _fake_info(None, error="could not read cert.pem"))
    check = doc.check_tls(_tls_config())[0]
    assert check.status == doc.WARN
    assert "could not read cert.pem" in check.detail


def test_the_tls_and_ratelimit_checks_are_in_the_real_report(home, monkeypatch):
    """Wiring: a check nobody calls from run_checks is a check nobody sees."""
    monkeypatch.setattr(doc, "check_docker",
                        lambda image="": [doc.Check("engine.docker", doc.OK, "fake",
                                                    data={"reachable": False})])
    monkeypatch.setattr(doc, "check_gpus", lambda gpus=None: [])
    monkeypatch.setattr(doc, "check_peers", lambda c, version="": [
        doc.Check("cluster.peers", doc.OK, "fake", data={"seen": 0})])
    monkeypatch.setattr(doc, "tcp_listening", lambda *a, **k: False)
    monkeypatch.setattr(doc, "udp_listeners", lambda: set())
    names = [c.name for c in doc.run_checks(home, home / "config.json")]
    assert "tls.enabled" in names
    assert "ratelimit.state" in names


def test_info_is_reported_but_never_changes_the_exit_code():
    checks = [doc.Check("ratelimit.state", doc.INFO, "off"),
              doc.Check("tls.enabled", doc.OK, "on")]
    assert doc.exit_code(checks) == 0
    counts = doc.summarize(checks)
    assert counts[doc.INFO] == 1
    assert counts["total"] == 2
    # The old shape is untouched for a report with no INFO in it.
    assert doc.summarize([doc.Check("a", doc.OK, "x")]) == {
        doc.OK: 1, doc.WARN: 0, doc.FAIL: 0, "total": 1}


def test_the_doctor_renders_an_info_line(capsys):
    doc.render([doc.Check("ratelimit.state", doc.INFO, "no per-client limit",
                          fix="set rate_limit in config.json")])
    out = capsys.readouterr().out
    assert "INFO" in out
    assert "no per-client limit" in out
    assert "1 INFO" in out


# =============================================================================
# `tailscale cert` is a HOST operation, so the wrapper owns it
# =============================================================================
#
# The CLI runs inside the container; tailscale runs on the host and is not in the
# image. These tests run the REAL wrapper the installer renders, with tailscale,
# docker, sudo, systemctl and openssl replaced by recording stubs, because the
# whole point of this path is which binary gets called where and with what.

INSTALL_SH = Path(__file__).resolve().parent.parent / "scripts" / "install.sh"

#: A pretty-printed `tailscale status --json`, trimmed, with the real shape:
#: `"DNSName": "name."` with whitespace after the colon and a trailing dot. The
#: wrapper's sed has to survive both; a pattern written against `"DNSName":"x"`
#: matches nothing at all against the real output and the failure is silent.
TS_STATUS_JSON = """{
  "Version": "1.102.2-t6cac91817-g6ff0ddc72",
  "BackendState": "Running",
  "TailscaleIPs": [
    "100.80.240.119",
    "fd7a:115c:a1e0::3835:f077"
  ],
  "Self": {
    "ID": "n2c1AzXDY321CNTRL",
    "HostName": "Spark-3-DGX",
    "DNSName": "spark-3-dgx.tailed10d2.ts.net.",
    "OS": "linux"
  },
  "Peer": {
    "nodekey:aa": {
      "HostName": "Spark-1-DGX",
      "DNSName": "spark-1-dgx.tailed10d2.ts.net.",
      "OS": "linux"
    }
  }
}
"""

TAILNET_NAME = "spark-3-dgx.tailed10d2.ts.net"


def _stub(dirpath: Path, name: str, body: str) -> None:
    path = dirpath / name
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(0o755)


@pytest.fixture
def wrapper(tmp_path):
    """The real host wrapper, rendered by the real installer in --dry-run.

    --dry-run renders config.json, the unit, the renewal timer and the wrapper
    into $AINODE_HOME and stops: no pulls, no systemd, no sudo, no GPU.
    """
    if shutil.which("bash") is None:  # pragma: no cover - every CI image has bash
        pytest.skip("needs bash")
    fake_home = tmp_path / "fake-home"
    ainode_home = fake_home / ".ainode"
    fake_home.mkdir(parents=True, exist_ok=True)
    sysfs = tmp_path / "sys-class-net"
    sysfs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(HOME=str(fake_home), AINODE_HOME=str(ainode_home),
               AINODE_IMAGE="ghcr.io/getainode/ainode:9.9.9",
               SYS_CLASS_NET=str(sysfs))
    env.pop("AINODE_PEERS", None)
    env.pop("HF_TOKEN", None)
    proc = subprocess.run(["bash", str(INSTALL_SH), "--dry-run"],
                          capture_output=True, text=True, timeout=180, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    path = ainode_home / "ainode-wrapper"
    assert path.exists(), "the installer no longer renders a host wrapper"
    return path


@pytest.fixture
def host(tmp_path):
    """A fake host for the wrapper: stub binaries plus the env that reaches it.

    Everything the wrapper can call is a stub that appends its argv to one record
    file, so a test asserts on the sequence of commands rather than on output.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    record = tmp_path / "record"
    record.write_text("")
    enddate = tmp_path / "enddate"
    enddate.write_text("notAfter=Dec 19 23:58:48 2026 GMT\n")
    tsjson = tmp_path / "ts.json"
    tsjson.write_text(TS_STATUS_JSON)
    ainode_home = tmp_path / "home" / ".ainode"
    ainode_home.mkdir(parents=True, exist_ok=True)

    _stub(bindir, "tailscale", f'''
printf '%s\\n' "tailscale $*" >> "{record}"
case "${{1:-}}" in
    status) cat "{tsjson}" ;;
    cert)
        # Emulate the real thing: write both files, and move the expiry on only
        # when the test says tailscale actually issued something new.
        cert=""; key=""
        while [ $# -gt 0 ]; do
            case "$1" in
                --cert-file) cert="$2"; shift 2 ;;
                --key-file) key="$2"; shift 2 ;;
                *) shift ;;
            esac
        done
        [ "${{TS_CERT_FAILS:-0}}" = "1" ] && {{ echo "certificate not available" >&2; exit 1; }}
        printf 'CERT\\n' > "$cert"
        printf 'KEY\\n' > "$key"
        [ -n "${{TS_NEW_ENDDATE:-}}" ] && printf 'notAfter=%s\\n' "$TS_NEW_ENDDATE" > "{enddate}"
        echo "Wrote public cert to $cert"
        ;;
esac
exit 0
''')
    _stub(bindir, "docker", f'''
printf '%s\\n' "docker $* [AINODE_TAILNET_NAME=${{AINODE_TAILNET_NAME:-unset}}]" >> "{record}"
for a in "$@"; do
    if [ "$a" = "--check" ]; then
        printf '%s\\n' "${{AINODE_DECISION:-}}"
        exit "${{AINODE_DECISION_RC:-0}}"
    fi
done
exit "${{DOCKER_RC:-0}}"
''')
    _stub(bindir, "systemctl", f'''
printf '%s\\n' "systemctl $*" >> "{record}"
# `systemctl --user is-enabled ainode.service` decides user mode: say no, so the
# wrapper takes the system path, which is what every Spark runs. Everything else
# succeeds, because a try-restart that fails would mask the assertion.
[ "${{1:-}}" = "--user" ] && exit 1
exit 0
''')
    _stub(bindir, "sudo", f'''
printf '%s\\n' "sudo $*" >> "{record}"
while [ $# -gt 0 ]; do
    case "$1" in -n) shift ;; *) break ;; esac
done
exec "$@"
''')
    _stub(bindir, "openssl", f'''
printf '%s\\n' "openssl $*" >> "{record}"
cat "{enddate}"
exit 0
''')
    env = dict(os.environ)
    # A hermetic PATH, not a prefixed one: /usr/local/bin holds a REAL tailscale
    # on any Mac with the app installed, and "there is no tailscale on this host"
    # is one of the branches under test. Everything the wrapper reaches for in
    # this path lives in /usr/bin or /bin on macOS and on Linux alike.
    env["PATH"] = f"{bindir}:/usr/bin:/bin:/usr/sbin:/sbin"
    env["HOME"] = str(tmp_path / "home")
    env["AINODE_HOME"] = str(ainode_home)
    # Never read a REAL /etc/systemd/system/ainode.service: the suite runs on the
    # Sparks, which have one.
    env["AINODE_UNIT_FILES"] = ""
    env.pop("AINODE_TAILNET_NAME", None)
    env.pop("AINODE_TLS_RENEW_RESTART", None)

    class Host:
        def __init__(self):
            self.env = env
            self.bindir = bindir
            self.record = record
            self.enddate = enddate
            self.ainode_home = ainode_home

        def run(self, wrapper_path, *args, **overrides):
            run_env = dict(self.env)
            run_env.update({k: str(v) for k, v in overrides.items()})
            return subprocess.run(["bash", str(wrapper_path), *args],
                                  capture_output=True, text=True, timeout=60,
                                  env=run_env)

        def calls(self, program=""):
            lines = [ln for ln in self.record.read_text().splitlines() if ln.strip()]
            if program:
                lines = [ln for ln in lines if ln.startswith(program + " ")]
            return lines

        def drop_binary(self, name):
            (self.bindir / name).unlink()

        def existing_pair(self, cert_name, key_name):
            """Seed the pair a renewal replaces, the way a real node already has one."""
            directory = self.ainode_home / "tls"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / cert_name).write_text("CERT\n")
            (directory / key_name).write_text("KEY\n")

    return Host()


class TestWrapperTailscaleEnable:
    """`ainode tls enable --tailscale` on the host, end to end."""

    def test_it_runs_tailscale_cert_into_the_bind_mount_and_then_the_container(
            self, wrapper, host):
        proc = host.run(wrapper, "tls", "enable", "--tailscale")
        assert proc.returncode == 0, proc.stdout + proc.stderr

        cert = host.ainode_home / "tls" / f"{TAILNET_NAME}.crt"
        key = host.ainode_home / "tls" / f"{TAILNET_NAME}.key"
        # The exact command the design calls for: the pair named after the one
        # name it is issued for, inside <AINODE_HOME>/tls, which is the bind
        # mount the containerised server reads.
        assert any(
            f"tailscale cert --cert-file {cert} --key-file {key} {TAILNET_NAME}" == line
            for line in host.calls("tailscale")
        ), host.calls("tailscale")
        assert cert.is_file() and key.is_file()
        # A private key in a bind-mounted directory is readable by anything else
        # that mounts it unless this happens.
        assert oct(key.stat().st_mode & 0o777) == "0o600"

        # And then the container writes the config block, told which name to look
        # for rather than being handed a path it would have to translate.
        forwarded = [ln for ln in host.calls("docker") if "tls" in ln and "enable" in ln]
        assert forwarded, host.calls("docker")
        assert f"AINODE_TAILNET_NAME={TAILNET_NAME}" in forwarded[-1]
        assert "-e AINODE_TAILNET_NAME" in forwarded[-1]

    def test_the_magicdns_name_is_read_out_of_pretty_printed_json(self, wrapper, host):
        """The trailing dot is stripped and the peers' names are not picked up."""
        proc = host.run(wrapper, "tls", "enable", "--tailscale")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        named = [ln for ln in proc.stdout.splitlines() if ln.startswith("==> tailscale cert")]
        assert named == [f"==> tailscale cert for {TAILNET_NAME}"], proc.stdout
        assert "spark-1-dgx" not in proc.stdout, "that is a PEER's name"

    def test_a_host_with_no_tailscale_says_so_and_names_the_alternative(
            self, wrapper, host):
        host.drop_binary("tailscale")
        proc = host.run(wrapper, "tls", "enable", "--tailscale")
        assert proc.returncode == 1
        assert "No tailscale on this host" in proc.stderr
        assert "ainode tls enable" in proc.stderr
        # Nothing was forwarded: a node must not end up with an enabled block and
        # no certificate.
        assert not [ln for ln in host.calls("docker") if "enable" in ln]

    def test_a_refused_certificate_names_the_admin_console_fix(self, wrapper, host):
        proc = host.run(wrapper, "tls", "enable", "--tailscale", TS_CERT_FAILS="1")
        assert proc.returncode == 1
        assert "HTTPS Certificates" in proc.stderr
        assert "--operator=" in proc.stderr, "the other real refusal"
        assert not [ln for ln in host.calls("docker") if "enable" in ln]

    def test_sudo_is_only_used_when_the_plain_call_is_refused(self, wrapper, host):
        """`tailscale set --operator=$USER` makes the plain call work; honour it."""
        proc = host.run(wrapper, "tls", "enable", "--tailscale")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert not host.calls("sudo"), "the plain call succeeded, so sudo is noise"

    def test_every_other_tls_subcommand_is_forwarded_untouched(self, wrapper, host):
        for args in (["tls", "status"], ["tls", "disable"], ["tls", "enable"]):
            host.record.write_text("")
            proc = host.run(wrapper, *args)
            assert proc.returncode == 0, proc.stdout + proc.stderr
            assert not [ln for ln in host.calls("tailscale") if " cert " in ln], args
            assert host.calls("docker"), args

    def test_the_wrapper_refuses_rather_than_writing_the_wrong_ainode(
            self, wrapper, host):
        """Same rule as `update`: WHERE before WHAT (#164)."""
        _stub(host.bindir, "id", "echo 0\n")
        _stub(host.bindir, "getent", "exit 2\n")
        env = dict(host.env)
        env.pop("AINODE_HOME")
        proc = subprocess.run(["bash", str(wrapper), "tls", "enable", "--tailscale"],
                              capture_output=True, text=True, timeout=60,
                              env={**env, "SUDO_USER": "nobodyatall"})
        assert proc.returncode == 1
        assert "Cannot tell which .ainode" in proc.stderr


class TestWrapperRenew:
    """`ainode tls renew`: the container decides, the host acts."""

    DUE = ("renew=yes\n"
           f"name={TAILNET_NAME}\n"
           "days_left=11.5\n"
           "reason=11.5 days left, inside the 14 day window.\n"
           "cert_name=" + TAILNET_NAME + ".crt\n"
           "key_name=" + TAILNET_NAME + ".key\n")
    NOT_DUE = ("renew=no\n"
               f"name={TAILNET_NAME}\n"
               "days_left=61.0\n"
               "reason=61.0 days left, which is more than the 14 day window.\n"
               "cert_name=cert.pem\nkey_name=key.pem\n")

    def test_nothing_is_touched_when_the_node_says_it_is_not_due(self, wrapper, host):
        proc = host.run(wrapper, "tls", "renew",
                        AINODE_DECISION=self.NOT_DUE, AINODE_DECISION_RC="10")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "No renewal needed" in proc.stdout
        assert not [ln for ln in host.calls("tailscale") if " cert " in ln]
        assert not [ln for ln in host.calls("systemctl") if "try-restart" in ln]

    def test_a_due_certificate_is_replaced_and_the_node_restarted(self, wrapper, host):
        host.existing_pair(f"{TAILNET_NAME}.crt", f"{TAILNET_NAME}.key")
        proc = host.run(wrapper, "tls", "renew", AINODE_DECISION=self.DUE,
                        TS_NEW_ENDDATE="Mar 19 23:58:48 2027 GMT")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        cert = host.ainode_home / "tls" / f"{TAILNET_NAME}.crt"
        assert any(f"--cert-file {cert}" in ln for ln in host.calls("tailscale"))
        # The listener and its SSLContext are built at boot, so without this the
        # renewal writes a file nobody serves.
        assert any("try-restart ainode.service" in ln for ln in host.calls("systemctl"))
        assert "Mar 19 23:58:48 2027 GMT" in proc.stdout

    def test_the_files_the_config_points_at_are_the_ones_replaced(self, wrapper, host):
        """A pair from an earlier manual run sits at cert.pem, not <name>.crt."""
        due = self.DUE.replace(f"cert_name={TAILNET_NAME}.crt", "cert_name=cert.pem")
        due = due.replace(f"key_name={TAILNET_NAME}.key", "key_name=key.pem")
        host.existing_pair("cert.pem", "key.pem")
        proc = host.run(wrapper, "tls", "renew", AINODE_DECISION=due,
                        TS_NEW_ENDDATE="Mar 19 23:58:48 2027 GMT")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        cert = host.ainode_home / "tls" / "cert.pem"
        key = host.ainode_home / "tls" / "key.pem"
        assert any(f"--cert-file {cert} --key-file {key}" in ln
                   for ln in host.calls("tailscale")), host.calls("tailscale")

    def test_the_cached_certificate_coming_back_is_not_a_restart(self, wrapper, host):
        """tailscale hands back what it has until it decides to renew."""
        host.existing_pair(f"{TAILNET_NAME}.crt", f"{TAILNET_NAME}.key")
        proc = host.run(wrapper, "tls", "renew", AINODE_DECISION=self.DUE)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "same certificate" in proc.stdout
        assert not [ln for ln in host.calls("systemctl") if "try-restart" in ln]

    def test_the_restart_can_be_left_to_a_human(self, wrapper, host):
        host.existing_pair(f"{TAILNET_NAME}.crt", f"{TAILNET_NAME}.key")
        proc = host.run(wrapper, "tls", "renew", AINODE_DECISION=self.DUE,
                        TS_NEW_ENDDATE="Mar 19 23:58:48 2027 GMT",
                        AINODE_TLS_RENEW_RESTART="0")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "Restart skipped" in proc.stdout
        assert "OLD certificate" in proc.stdout
        assert not [ln for ln in host.calls("systemctl") if "try-restart" in ln]

    def test_a_node_that_is_down_is_not_a_failed_renewal(self, wrapper, host):
        """A daily timer must not mark itself failed all through an outage."""
        proc = host.run(wrapper, "tls", "renew", AINODE_DECISION="", DOCKER_RC="1")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "Could not ask the container" in proc.stdout
        assert not [ln for ln in host.calls("tailscale") if " cert " in ln]

    def test_a_failed_renewal_leaves_the_old_pair_in_place(self, wrapper, host):
        proc = host.run(wrapper, "tls", "renew", AINODE_DECISION=self.DUE,
                        TS_CERT_FAILS="1")
        assert proc.returncode == 1
        assert "still being served" in proc.stderr
        assert not [ln for ln in host.calls("systemctl") if "try-restart" in ln]


class TestRenewalTimer:
    """The actuator: a host timer, because nothing else renews unattended."""

    def test_the_installer_renders_a_daily_timer_that_runs_the_wrapper(self, wrapper):
        home = wrapper.parent
        service = (home / "ainode-tls-renew.service").read_text()
        timer = (home / "ainode-tls-renew.timer").read_text()
        assert f"ExecStart={wrapper} tls renew" in service
        assert f"Environment=AINODE_HOME={home}" in service
        assert "Type=oneshot" in service
        assert "OnCalendar=daily" in timer
        # A node that was off when the timer was due is exactly the case that
        # matters, and a whole fleet asking Let's Encrypt at once is the one to
        # avoid.
        assert "Persistent=true" in timer
        assert "RandomizedDelaySec" in timer
        assert "WantedBy=timers.target" in timer

    def test_the_timer_is_installed_whatever_the_tls_state_is(self):
        """A node that turns TLS on later must not need a reinstall to renew."""
        text = INSTALL_SH.read_text()
        # The rendering is not inside any conditional on the tls block: the only
        # branch it sits under is dry run versus user scope versus system scope.
        assert "systemctl enable --now ainode-tls-renew.timer" in text
        assert "systemctl --user enable --now ainode-tls-renew.timer" in text

    def test_the_two_fourteens_agree(self):
        """The doctor's warning window and the timer's action window are one fact.

        They live in two modules (the doctor warns, ainode/tls/renew.py decides)
        and they must never disagree: an operator told "expires in 13 days, run
        this" about a node whose timer already handled it is being sent to do
        nothing, and the reverse is a certificate nobody renews.
        """
        assert doc.TLS_EXPIRY_WARN_DAYS == RENEW_THRESHOLD_DAYS


# =============================================================================
# The renewal decision itself
# =============================================================================

def _cert_info(days=None, **kw):
    info = {"exists": True, "error": "", "self_signed": False,
            "days_left": days, "sans": [f"DNS:{TAILNET_NAME}"]}
    info.update(kw)
    return info


def test_a_tailnet_certificate_inside_the_window_is_renewed():
    tls = TLSConfig(enabled=True, cert_file="/root/.ainode/tls/c.crt",
                    key_file="/root/.ainode/tls/c.key")
    decision = renewal_decision(tls, _cert_info(11.5))
    assert decision.renew is True
    assert decision.name == TAILNET_NAME
    assert decision.days_left == 11.5
    assert "14 day window" in decision.reason


def test_a_tailnet_certificate_with_time_left_is_left_alone():
    tls = TLSConfig(enabled=True, cert_file="c.crt", key_file="c.key")
    decision = renewal_decision(tls, _cert_info(61.0))
    assert decision.renew is False
    assert "nothing to do yet" in decision.reason


@pytest.mark.parametrize("days,expected", [(14, True), (14.0, True), (14.1, False)])
def test_the_boundary_is_inclusive(days, expected):
    tls = TLSConfig(enabled=True, cert_file="c.crt", key_file="c.key")
    assert renewal_decision(tls, _cert_info(days)).renew is expected


def test_tls_off_is_not_a_renewal():
    assert renewal_decision(TLSConfig(), _cert_info(1.0)).renew is False


def test_a_self_signed_certificate_is_never_swapped_for_a_tailnet_one():
    """Its SANs are the hostname and every address; a tailnet cert has one name.

    Swapping them silently would break every client reaching this node by IP,
    which on a LAN is most of them.
    """
    tls = TLSConfig(enabled=True, cert_file="c.crt", key_file="c.key")
    info = _cert_info(3.0, self_signed=True, sans=["DNS:Spark-3-DGX", "IP:10.0.0.3"])
    decision = renewal_decision(tls, info)
    assert decision.renew is False
    assert "self-signed" in decision.reason
    assert "ainode tls enable" in decision.reason


def test_a_certificate_for_a_name_outside_the_tailnet_says_who_has_to_renew_it():
    tls = TLSConfig(enabled=True, cert_file="c.crt", key_file="c.key")
    info = _cert_info(2.0, sans=["DNS:ainode.example.com"])
    decision = renewal_decision(tls, info)
    assert decision.renew is False
    assert decision.name == ""
    assert "outside the tailnet" in decision.reason


def test_an_unreadable_certificate_is_never_replaced_on_a_guess():
    tls = TLSConfig(enabled=True, cert_file="c.crt", key_file="c.key")
    decision = renewal_decision(tls, _cert_info(None, error="could not read c.crt"))
    assert decision.renew is False
    assert "could not read" in decision.reason


def test_an_enabled_block_whose_pair_vanished_is_repaired():
    """The one missing-file case that means act: the node is serving HTTP only."""
    tls = TLSConfig(enabled=True, cert_file="c.crt", key_file="c.key")
    decision = renewal_decision(tls, {"exists": False, "sans": [f"DNS:{TAILNET_NAME}"]})
    assert decision.renew is True
    assert "HTTP only" in decision.reason

    # With no name to renew for, there is nothing to run, so say so instead.
    nameless = renewal_decision(tls, {"exists": False, "sans": []})
    assert nameless.renew is False
    assert "no tailnet name" in nameless.reason


def test_the_name_comes_off_the_certificate_and_not_off_the_host():
    """A node renamed in the tailnet still renews the name it was issued for."""
    tls = TLSConfig(enabled=True, cert_file="c.crt", key_file="c.key")
    info = _cert_info(2.0, sans=["DNS:old-name.tailed10d2.ts.net"])
    decision = renewal_decision(tls, info, tailnet_name="new-name.tailed10d2.ts.net")
    assert decision.name == "old-name.tailed10d2.ts.net"


def test_the_decision_is_key_value_lines_a_shell_can_read():
    tls = TLSConfig(enabled=True, cert_file="c.crt", key_file="c.key")
    lines = renewal_decision(tls, _cert_info(3.0)).as_lines()
    assert lines[0] == "renew=yes"
    assert f"name={TAILNET_NAME}" in lines
    assert any(ln.startswith("reason=") for ln in lines)
    # One line each, because the wrapper reads them with sed.
    for line in lines:
        assert "\n" not in line


def test_the_san_reader_ignores_ip_entries_and_non_tailnet_names():
    assert tailnet_name_from_sans(["IP:100.80.240.119"]) == ""
    assert tailnet_name_from_sans(["DNS:spark-3", "DNS:a.ts.net"]) == "a.ts.net"
    assert tailnet_name_from_sans([]) == ""
    assert tailnet_name_from_sans(None) == ""


# =============================================================================
# Finding this node's MagicDNS name
# =============================================================================

def test_the_wrappers_variable_wins_because_it_costs_nothing(monkeypatch):
    monkeypatch.setenv("AINODE_TAILNET_NAME", "spark-9.tailed10d2.ts.net.")
    # No runner, no binary, no lookup: route 1 answers before any of that.
    assert certs_mod.tailnet_dns_name() == "spark-9.tailed10d2.ts.net"


def test_tailscale_status_answers_when_there_is_a_binary(monkeypatch):
    monkeypatch.delenv("AINODE_TAILNET_NAME", raising=False)
    calls = []

    def runner(argv):
        calls.append(list(argv))
        return 0, json.dumps({"Self": {"DNSName": f"{TAILNET_NAME}."}})

    assert certs_mod.tailnet_dns_name(runner) == TAILNET_NAME
    assert calls[0][:2] == ["tailscale", "status"]


def test_a_reverse_lookup_of_the_tailnet_address_is_the_containers_route(monkeypatch):
    """No tailscale binary in the image, but --network=host shares MagicDNS."""
    monkeypatch.delenv("AINODE_TAILNET_NAME", raising=False)
    monkeypatch.setattr(certs_mod, "tailscale_binary", lambda: None)
    monkeypatch.setattr(certs_mod, "host_ip_addresses",
                        lambda runner=None: ["192.168.0.13", "100.80.240.119"])
    asked = []

    def gethostbyaddr(addr):
        asked.append(addr)
        return (f"{TAILNET_NAME}.", [], [addr])

    monkeypatch.setattr(certs_mod.socket, "gethostbyaddr", gethostbyaddr)
    assert certs_mod.tailnet_dns_name() == TAILNET_NAME
    # Only the CGNAT-range address is asked about, so nothing else on the box can
    # be mistaken for a tailnet name.
    assert asked == ["100.80.240.119"]


def test_the_lookup_can_be_refused_by_a_caller_that_cannot_afford_to_stall(monkeypatch):
    monkeypatch.delenv("AINODE_TAILNET_NAME", raising=False)
    monkeypatch.setattr(certs_mod, "tailscale_binary", lambda: None)
    monkeypatch.setattr(certs_mod, "host_ip_addresses",
                        lambda runner=None: ["100.80.240.119"])

    def boom(addr):  # pragma: no cover - must never be reached
        raise AssertionError("allow_lookup=False still resolved a name")

    monkeypatch.setattr(certs_mod.socket, "gethostbyaddr", boom)
    assert certs_mod.tailnet_dns_name(allow_lookup=False) is None


def test_a_resolver_that_never_answers_costs_two_seconds_and_not_the_command():
    """gethostbyaddr cannot be interrupted, so it runs where we can walk away."""
    import threading

    started = threading.Event()
    release = threading.Event()

    def hang(addr):
        started.set()
        release.wait(30)  # released in the finally below, so no thread leaks
        return ("never.ts.net", [], [addr])

    real = certs_mod.socket.gethostbyaddr
    certs_mod.socket.gethostbyaddr = hang
    try:
        began = time.monotonic()
        assert certs_mod._reverse_lookup("100.80.240.119", timeout=0.2) == ""
        assert time.monotonic() - began < 5.0
        assert started.is_set(), "the lookup did run, it just was not waited for"
    finally:
        release.set()
        certs_mod.socket.gethostbyaddr = real


def test_an_answer_that_is_not_a_tailnet_name_is_not_one(monkeypatch):
    monkeypatch.delenv("AINODE_TAILNET_NAME", raising=False)
    monkeypatch.setattr(certs_mod, "tailscale_binary", lambda: None)
    monkeypatch.setattr(certs_mod, "host_ip_addresses",
                        lambda runner=None: ["100.80.240.119"])
    monkeypatch.setattr(certs_mod.socket, "gethostbyaddr",
                        lambda addr: ("spark-3.lan", [], [addr]))
    assert certs_mod.tailnet_dns_name() is None


# =============================================================================
# Does this node actually serve HTTPS, and what does it advertise
# =============================================================================

def test_serves_https_answers_from_the_block_alone_when_tls_is_off(home):
    """The fleet's state, and the one that must not cost a stat on /api/status."""
    assert certs_mod.serves_https(NodeConfig(web_port=3000)) == (False, 3000)


def test_serves_https_is_true_for_a_pair_that_loads(home, pair):
    config = NodeConfig(web_port=3000, tls={
        "enabled": True, "port": 3443,
        "cert_file": pair["cert_file"], "key_file": pair["key_file"]})
    assert certs_mod.serves_https(config) == (True, 3443)


def test_serves_https_applies_every_condition_the_listener_does(home, pair):
    """Same list as listener_plan, so the advertised url and the socket agree."""
    good = {"enabled": True, "port": 3443,
            "cert_file": pair["cert_file"], "key_file": pair["key_file"]}
    # The HTTP port: listener_plan refuses it, so nothing may advertise it.
    assert certs_mod.serves_https(
        NodeConfig(web_port=3000, tls={**good, "port": 3000})) == (False, 3000)
    # A missing pair.
    assert certs_mod.serves_https(
        NodeConfig(web_port=3000, tls={**good, "cert_file": "/nope/x.crt"})
    ) == (False, 3000)
    # A pair that cannot be loaded.
    junk = Path(pair["cert_file"]).parent / "junk.pem"
    junk.write_text("-----BEGIN CERTIFICATE-----\nnope\n-----END CERTIFICATE-----\n")
    assert certs_mod.serves_https(
        NodeConfig(web_port=3000, tls={**good, "cert_file": str(junk)})
    ) == (False, 3000)


def test_a_replaced_pair_is_noticed_rather_than_cached_forever(home, pair):
    """The loadability cache is keyed on the files, not on the process."""
    cert = Path(pair["cert_file"])
    key = Path(pair["key_file"])
    assert certs_mod.pair_loadable(cert, key) is True
    cert.write_text("-----BEGIN CERTIFICATE-----\nbroken\n-----END CERTIFICATE-----\n")
    assert certs_mod.pair_loadable(cert, key) is False


def test_the_endpoint_url_carries_the_scheme():
    from ainode.api.server_routes import endpoint_url

    assert endpoint_url("10.0.0.5", 3000) == "http://10.0.0.5:3000"
    assert endpoint_url("10.0.0.5", 3443, True) == "https://10.0.0.5:3443"
    # The loopback rule is unchanged, either way.
    assert endpoint_url("localhost", 3443, True) is None
    assert endpoint_url("", 3443, True) is None


class TestEndpointAdvertisesHttps:
    """What a client reads off a node that has TLS on.

    The rule being pinned: `url` is the address to PREFER, `port` is still the
    HTTP port every peer and every older client uses, and only this node's own
    rows may claim TLS.
    """

    @pytest.fixture
    def rows(self, home, pair, monkeypatch):
        import time as _time

        from ainode.api import server_routes as sr
        from ainode.discovery.broadcast import NodeAnnouncement, NodeStatus
        from ainode.discovery.cluster import ClusterNode, ClusterState

        sr.reset_address_cache()
        monkeypatch.setattr(sr, "own_addresses", lambda config: {"10.0.0.7"})
        monkeypatch.setattr(sr, "lan_address", lambda config: "10.0.0.7")

        def build(tls_block):
            config = NodeConfig(node_id="spark-1", node_name="Spark-1",
                                host="0.0.0.0", web_port=3000, tls=tls_block)
            cluster = ClusterState(local_announcement=NodeAnnouncement(
                node_id="spark-1", node_name="Spark-1", gpu_name="GB10",
                gpu_memory_gb=128.0, unified_memory=True, model="m",
                status="serving", api_port=8000, web_port=3000,
                cluster_id="default", role="master"))
            cluster.add_node(ClusterNode(
                node_id="spark-2", node_name="Spark-2", gpu_name="GB10",
                gpu_memory_gb=128.0, unified_memory=True, model="m",
                status=NodeStatus.ONLINE, api_port=8000, web_port=3000,
                last_seen=_time.time(), cluster_id="default", role="auto",
                peer_ip="10.0.0.8"))
            app = {"config": config, "cluster_state": cluster}

            class _Req:
                def __init__(self):
                    self.app = app
                    self.headers = {}

            return sr, app, _Req()

        yield build
        sr.reset_address_cache()

    def test_a_node_with_tls_on_advertises_https_on_the_tls_port(self, rows, pair):
        sr, app, request = rows({"enabled": True, "port": 3443,
                                 "cert_file": pair["cert_file"],
                                 "key_file": pair["key_file"]})
        mine = [r for r in sr.endpoint_nodes(app, request) if r["name"] == "Spark-1"][0]
        assert mine["tls"] is True
        assert mine["tls_port"] == 3443
        assert mine["url"] == "https://10.0.0.7:3443"
        # Unchanged, and deliberately: the peer proxy and every client older than
        # this release build http://host:port out of these two.
        assert mine["port"] == 3000

    def test_a_peer_row_stays_http_because_its_tls_state_is_not_on_the_wire(
            self, rows, pair):
        sr, app, request = rows({"enabled": True, "port": 3443,
                                 "cert_file": pair["cert_file"],
                                 "key_file": pair["key_file"]})
        peer = [r for r in sr.endpoint_nodes(app, request) if r["name"] == "Spark-2"][0]
        assert peer["tls"] is False
        assert peer["tls_port"] is None
        assert peer["url"] == "http://10.0.0.8:3000"

    def test_tls_off_looks_exactly_as_it_did_before(self, rows):
        sr, app, request = rows(None)
        for row in sr.endpoint_nodes(app, request):
            assert row["tls"] is False
            assert row["tls_port"] is None
            assert row["url"].startswith("http://")

    def test_the_self_and_master_blocks_say_the_same_thing_as_the_rows(
            self, rows, pair):
        sr, app, request = rows({"enabled": True, "port": 3443,
                                 "cert_file": pair["cert_file"],
                                 "key_file": pair["key_file"]})
        payload = sr.endpoint_payload(app, request)
        assert payload["self"]["tls"] is True
        assert payload["self"]["url"] == "https://10.0.0.7:3443"
        assert payload["self"]["port"] == 3000
        # This node IS the master here, so that block says https too.
        assert payload["master"]["tls"] is True
        assert payload["master"]["url"] == "https://10.0.0.7:3443"
        # And the payload still carries nothing but names, addresses, ports,
        # schemes, roles and versions, which is what lets it answer with no key.
        assert "cert_file" not in json.dumps(payload)

    def test_an_enabled_block_with_no_usable_pair_advertises_http(self, rows):
        """A broken certificate must not make the node advertise a dead port."""
        sr, app, request = rows({"enabled": True, "port": 3443,
                                 "cert_file": "/nope/cert.pem",
                                 "key_file": "/nope/key.pem"})
        mine = [r for r in sr.endpoint_nodes(app, request) if r["name"] == "Spark-1"][0]
        assert mine["tls"] is False
        assert mine["url"] == "http://10.0.0.7:3000"


# =============================================================================
# The doctor names THIS node's command
# =============================================================================

def test_the_doctor_names_the_magicdns_name_and_the_exact_command(monkeypatch):
    monkeypatch.setenv("AINODE_TAILNET_NAME", TAILNET_NAME)
    check = doc.check_tls(NodeConfig(web_port=3000))[0]
    assert check.status == doc.WARN
    assert TAILNET_NAME in check.detail
    assert f"ainode tls enable --tailscale (a real certificate for {TAILNET_NAME})" \
        in check.fix
    assert check.data["tailnet_name"] == TAILNET_NAME


def test_a_node_with_no_tailnet_name_gets_the_generic_command(monkeypatch):
    # Patched at the seam the doctor imports, so the machine running the suite
    # (which is on this very tailnet) cannot answer for the node under test.
    monkeypatch.setattr("ainode.tls.certs.tailnet_dns_name",
                        lambda *a, **k: None)
    check = doc.check_tls(NodeConfig(web_port=3000))[0]
    assert check.status == doc.WARN
    assert "ainode tls enable" in check.fix
    assert check.data["tailnet_name"] == ""


def test_the_doctor_points_at_the_renewal_inside_the_window(monkeypatch):
    monkeypatch.setenv("AINODE_TAILNET_NAME", TAILNET_NAME)
    monkeypatch.setattr(doc, "certificate_info",
                        lambda p: _fake_info(9.0, self_signed=False))
    check = doc.check_tls(_tls_config())[0]
    assert check.status == doc.WARN
    assert "ainode tls renew" in check.fix
    assert TAILNET_NAME in check.fix
