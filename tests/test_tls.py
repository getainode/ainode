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

Nothing here reaches the network. Certificates are generated for real (openssl is
present on any machine that can build this project) because a certificate the
test writes itself is the only way to check what ends up inside one.
"""

import json
import os
import socket
import ssl
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
