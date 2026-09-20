"""Make a certificate, read a certificate, build an SSL context.

Two generators, because ``cryptography`` is not an AINode dependency and the
interpreter this runs in is whatever the image happens to have:

* ``cryptography`` when it imports. The shipped image does have it (paramiko
  pulls it in, verified on a node running 0.5.27), so this is the path the fleet
  takes, but nothing may depend on that: it is a transitive dependency of
  something unrelated and could leave with it.
* the ``openssl`` binary otherwise. It is present in the image too (the
  Dockerfile installs ``ca-certificates``, which depends on openssl) and on
  every host that can run docker, which makes it a real fallback rather than a
  theoretical one. It is also the path a ``pip install ainode`` into a bare venv
  takes.

Both produce the same thing: one self-signed leaf, 825 days (the longest span a
current browser or Keychain will accept), ``serverAuth``, with the node's
hostname plus every non-loopback IPv4 on the box as SANs. A certificate with no
matching SAN is a certificate every modern client refuses, and on a node reached
by three different addresses (hostname, LAN IP, tailnet IP) the SAN list is the
whole difference between "works" and "works from one place".

Reading a certificate goes through the same two paths, and the answer is cached
on the file's identity (path, mtime, size) because ``/api/status`` is polled
every few seconds and an ``openssl x509`` fork per poll is not free.
"""

from __future__ import annotations

import datetime as _dt
import ipaddress
import logging
import os
import re
import shutil
import socket
import ssl
import subprocess
from pathlib import Path
from typing import Callable, Optional, Sequence

from ainode.tls.config import ensure_tls_dir, harden_key

logger = logging.getLogger(__name__)

#: How long a generated certificate is valid. 825 days is the ceiling Apple and
#: the CA/Browser Forum settled on, and a self-signed certificate over it is
#: rejected outright by Safari and by anything using the macOS trust store.
CERT_DAYS = 825

#: Addresses that are never worth a SAN entry: loopback (a client reaching a node
#: over loopback is on the node) and IPv4 link-local, which means "no DHCP
#: answered" rather than "here is an address".
_SKIP_IP_PREFIXES = ("127.", "169.254.")

_ADDR_RE = re.compile(r"\binet\s+(\d+\.\d+\.\d+\.\d+)")
_SAN_LINE_RE = re.compile(r"(DNS|IP Address):([^,\s]+)")

CommandRunner = Callable[[Sequence[str]], tuple[int, str]]


def run_command(argv: Sequence[str], timeout: float = 30.0) -> tuple[int, str]:
    """Run ``argv``, return ``(returncode, combined output)``. Never raises.

    The single seam every subprocess here goes through, so a test replaces this
    and touches nothing on the machine.
    """
    try:
        proc = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return 127, str(exc)
    return proc.returncode, ((proc.stdout or "") + (proc.stderr or "")).strip()


def have_cryptography() -> bool:
    """True when the ``cryptography`` package is importable in this interpreter."""
    try:
        import cryptography  # noqa: F401
    except Exception:
        return False
    return True


def openssl_binary() -> Optional[str]:
    """Path to an ``openssl`` binary, or None."""
    return shutil.which("openssl")


# ---------------------------------------------------------------------------
# What goes in the SAN list
# ---------------------------------------------------------------------------

def host_names(hostname: Optional[str] = None) -> list[str]:
    """This node's DNS names, best first, deduplicated.

    The container runs with ``--network=host``, so ``gethostname`` answers with
    the HOST's name, which is the name an operator types.
    """
    names: list[str] = []
    candidates = [hostname] if hostname else [socket.gethostname(), socket.getfqdn()]
    for name in candidates:
        name = (name or "").strip().rstrip(".")
        if not name or name in ("localhost", "localhost.localdomain"):
            continue
        if name not in names:
            names.append(name)
        short = name.split(".", 1)[0]
        if short and short not in names:
            names.append(short)
    return names


def host_ip_addresses(runner: Optional[CommandRunner] = None) -> list[str]:
    """Every non-loopback IPv4 on this host, tailnet address included.

    Read from ``ip -o -4 addr show`` (iproute2 ships in the AINode image), with a
    socket fallback for a host without it. Deliberately NOT
    ``cluster.netdev.list_ipv4_interfaces``: that function excludes VPN and
    overlay devices by prefix, ``tailscale0`` among them, which is exactly the
    address a Mac client with App Transport Security has to reach this node on.
    """
    run = run_command if runner is None else runner
    found: list[str] = []

    code, out = run(["ip", "-o", "-4", "addr", "show"])
    if code == 0:
        for match in _ADDR_RE.finditer(out):
            found.append(match.group(1))

    if not found:
        # No iproute2. The connected-UDP trick names the address the default
        # route would use; no packet is sent.
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(0.5)
                sock.connect(("192.0.2.1", 9))  # RFC 5737 documentation net
                found.append(sock.getsockname()[0])
        except OSError:
            pass
        try:
            _, _, addrs = socket.gethostbyname_ex(socket.gethostname())
            found.extend(addrs)
        except OSError:
            pass

    out_addrs: list[str] = []
    for addr in found:
        if addr.startswith(_SKIP_IP_PREFIXES):
            continue
        try:
            ipaddress.IPv4Address(addr)
        except ValueError:
            continue
        if addr not in out_addrs:
            out_addrs.append(addr)
    return out_addrs


def san_entries(hostname: Optional[str] = None,
                ips: Optional[Sequence[str]] = None,
                runner: Optional[CommandRunner] = None) -> list[tuple[str, str]]:
    """The SAN list as ``[("DNS", name), ("IP", addr), ...]``.

    ``localhost`` and ``127.0.0.1`` are always in it: ``curl
    https://localhost:3443`` on the node itself is the first thing anybody tries.
    """
    entries: list[tuple[str, str]] = []
    for name in host_names(hostname):
        entries.append(("DNS", name))
    entries.append(("DNS", "localhost"))
    addrs = list(ips) if ips is not None else host_ip_addresses(runner)
    for addr in addrs:
        entries.append(("IP", addr))
    entries.append(("IP", "127.0.0.1"))

    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for entry in entries:
        if entry in seen:
            continue
        seen.add(entry)
        unique.append(entry)
    return unique


def _common_name(entries: Sequence[tuple[str, str]]) -> str:
    for kind, value in entries:
        if kind == "DNS":
            return value
    for kind, value in entries:
        if kind == "IP":
            return value
    return "ainode"


# ---------------------------------------------------------------------------
# Generating
# ---------------------------------------------------------------------------

def generate_self_signed(cert_path, key_path,
                         hostname: Optional[str] = None,
                         ips: Optional[Sequence[str]] = None,
                         days: int = CERT_DAYS,
                         runner: Optional[CommandRunner] = None,
                         prefer: str = "") -> dict:
    """Write a self-signed pair and return what was made.

    ``prefer`` forces a generator (``"cryptography"`` / ``"openssl"``) and exists
    so the tests can drive both paths on a machine that only has one of them.
    Raises ``RuntimeError`` when neither is available, because a node that thinks
    it enabled TLS and has no certificate is the worse outcome.
    """
    cert_path = Path(cert_path)
    key_path = Path(key_path)
    entries = san_entries(hostname, ips, runner)
    cert_path.parent.mkdir(parents=True, exist_ok=True)

    tool = prefer
    if not tool:
        tool = "cryptography" if have_cryptography() else "openssl"

    if tool == "cryptography":
        _generate_with_cryptography(cert_path, key_path, entries, days)
    elif tool == "openssl":
        _generate_with_openssl(cert_path, key_path, entries, days, runner)
    else:
        raise RuntimeError(f"unknown certificate generator '{tool}'")

    harden_key(key_path)
    try:
        os.chmod(cert_path, 0o644)
    except OSError:
        pass
    return {
        "cert_file": str(cert_path),
        "key_file": str(key_path),
        "sans": [f"{kind}:{value}" for kind, value in entries],
        "days": days,
        "tool": tool,
        "self_signed": True,
    }


def _generate_with_cryptography(cert_path: Path, key_path: Path,
                                entries: Sequence[tuple[str, str]], days: int) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, _common_name(entries)),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AINode"),
    ])
    alt: list = []
    for kind, value in entries:
        if kind == "DNS":
            alt.append(x509.DNSName(value))
        else:
            alt.append(x509.IPAddress(ipaddress.ip_address(value)))
    now = _dt.datetime.now(_dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _dt.timedelta(minutes=5))
        .not_valid_after(now + _dt.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectAlternativeName(alt), critical=False)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    harden_key(key_path)
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def openssl_config(entries: Sequence[tuple[str, str]]) -> str:
    """The ``openssl req`` config that carries the SAN list.

    A config file rather than ``-addext``: LibreSSL (which is what
    ``/usr/bin/openssl`` is on a Mac) does not have ``-addext``, and this shape
    works on OpenSSL 1.1.1, OpenSSL 3.x and LibreSSL alike.
    """
    dns = [value for kind, value in entries if kind == "DNS"]
    ips = [value for kind, value in entries if kind == "IP"]
    lines = [
        "[req]",
        "distinguished_name = dn",
        "x509_extensions = v3_ainode",
        "prompt = no",
        "[dn]",
        f"CN = {_common_name(entries)}",
        "O = AINode",
        "[v3_ainode]",
        "basicConstraints = critical,CA:FALSE",
        "keyUsage = critical,digitalSignature,keyEncipherment",
        "extendedKeyUsage = serverAuth",
        "subjectAltName = @alt_ainode",
        "[alt_ainode]",
    ]
    for index, value in enumerate(dns, start=1):
        lines.append(f"DNS.{index} = {value}")
    for index, value in enumerate(ips, start=1):
        lines.append(f"IP.{index} = {value}")
    return "\n".join(lines) + "\n"


def _generate_with_openssl(cert_path: Path, key_path: Path,
                           entries: Sequence[tuple[str, str]], days: int,
                           runner: Optional[CommandRunner]) -> None:
    run = run_command if runner is None else runner
    binary = openssl_binary()
    if not binary and runner is None:
        raise RuntimeError(
            "no openssl binary and no cryptography module, so a certificate "
            "cannot be generated here. Install openssl, or pass an existing "
            "pair with --cert and --key."
        )
    conf = cert_path.parent / "openssl.cnf"
    conf.write_text(openssl_config(entries))
    try:
        code, out = run([
            binary or "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-days", str(days),
            "-keyout", str(key_path),
            "-out", str(cert_path),
            "-config", str(conf),
            "-extensions", "v3_ainode",
        ])
    finally:
        try:
            conf.unlink()
        except OSError:
            pass
    if code != 0:
        raise RuntimeError(f"openssl could not write a certificate: {out[-400:]}")


# ---------------------------------------------------------------------------
# tailscale cert: a real Let's Encrypt pair for the tailnet name
# ---------------------------------------------------------------------------

def tailscale_binary() -> Optional[str]:
    """Path to the ``tailscale`` CLI, or None.

    None is the normal answer inside the AINode container: the image does not
    ship tailscale and the daemon socket is not mounted, so ``--tailscale`` has
    to tell the operator what to run on the host instead of failing opaquely.
    """
    return shutil.which("tailscale")


def tailscale_dns_name(runner: Optional[CommandRunner] = None) -> Optional[str]:
    """This node's MagicDNS name, from ``tailscale status --json``."""
    run = run_command if runner is None else runner
    code, out = run(["tailscale", "status", "--json"])
    if code != 0 or not out:
        return None
    try:
        import json

        data = json.loads(out)
    except ValueError:
        return None
    name = ((data.get("Self") or {}).get("DNSName") or "").strip().rstrip(".")
    return name or None


def tailscale_cert(name: str, cert_path, key_path,
                   runner: Optional[CommandRunner] = None) -> dict:
    """Run ``tailscale cert`` for ``name``, writing straight into the TLS dir.

    A real certificate (Let's Encrypt, via the tailnet's own HTTPS support) for
    the MagicDNS name, which is what a client with App Transport Security needs:
    a self-signed pair makes a Mac app refuse the connection with nothing useful
    in the log.
    """
    run = run_command if runner is None else runner
    cert_path = Path(cert_path)
    key_path = Path(key_path)
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    code, out = run([
        "tailscale", "cert",
        "--cert-file", str(cert_path),
        "--key-file", str(key_path),
        name,
    ])
    if code != 0:
        raise RuntimeError(f"tailscale cert failed for {name}: {out[-400:]}")
    harden_key(key_path)
    return {"cert_file": str(cert_path), "key_file": str(key_path),
            "name": name, "self_signed": False}


# ---------------------------------------------------------------------------
# Reading a certificate
# ---------------------------------------------------------------------------

_INFO_CACHE: dict[str, tuple[tuple, dict]] = {}


def _cache_stamp(path: Path) -> Optional[tuple]:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def certificate_info(cert_path, now: Optional[float] = None,
                     runner: Optional[CommandRunner] = None,
                     use_cache: bool = True) -> dict:
    """What a human needs to know about a certificate file.

    ``{"exists", "expires", "days_left", "self_signed", "sans", "error"}``.
    ``expires`` is the EARLIEST expiry in the file: ``tailscale cert`` writes a
    leaf plus its issuer, and the question this answers is "when does HTTPS stop
    working here", which is the first date in the chain to pass.

    Never raises. An unreadable or unparseable file comes back with ``error``
    set and the dates null, because every caller (``/api/status``, the doctor)
    wants to report that rather than fail.
    """
    path = Path(cert_path)
    stamp = _cache_stamp(path)
    if stamp is None:
        return {"path": str(path), "exists": False, "expires": None,
                "days_left": None, "self_signed": None, "sans": [],
                "error": "no such file"}
    if use_cache:
        cached = _INFO_CACHE.get(str(path))
        if cached and cached[0] == stamp:
            return dict(cached[1], **_freshen(cached[1], now))

    info = _read_certificate(path, runner)
    info.update(_freshen(info, now))
    if use_cache:
        _INFO_CACHE[str(path)] = (stamp, info)
    return info


def _freshen(info: dict, now: Optional[float]) -> dict:
    """Recompute ``days_left`` against the clock, leaving the parse cached."""
    epoch = info.get("expires_epoch")
    if not epoch:
        return {}
    reference = _dt.datetime.now(_dt.timezone.utc).timestamp() if now is None else now
    return {"days_left": round((epoch - reference) / 86400.0, 1)}


def _read_certificate(path: Path, runner: Optional[CommandRunner]) -> dict:
    info: dict = {"path": str(path), "exists": True, "expires": None,
                  "expires_epoch": None, "days_left": None, "self_signed": None,
                  "sans": [], "error": ""}
    if have_cryptography():
        try:
            return _read_with_cryptography(path, info)
        except Exception as exc:  # a malformed PEM, a truncated write
            info["error"] = f"could not read {path.name}: {exc}"
            return info
    return _read_with_openssl(path, info, runner)


def _not_after_utc(cert) -> _dt.datetime:
    """A certificate's expiry as an aware UTC datetime, old and new API alike.

    ``not_valid_after_utc`` arrived in cryptography 42; before that
    ``not_valid_after`` answered a naive UTC datetime. The package is not an
    AINode dependency, so whatever version a dev box happens to have is what
    this gets.
    """
    aware = getattr(cert, "not_valid_after_utc", None)
    if aware is not None:
        return aware
    return cert.not_valid_after.replace(tzinfo=_dt.timezone.utc)


def _read_with_cryptography(path: Path, info: dict) -> dict:
    from cryptography import x509

    blob = path.read_bytes()
    certs = x509.load_pem_x509_certificates(blob)
    if not certs:
        info["error"] = f"{path.name} holds no certificate"
        return info
    leaf = certs[0]
    earliest = min(_not_after_utc(c) for c in certs)
    info["expires"] = earliest.strftime("%Y-%m-%dT%H:%M:%SZ")
    info["expires_epoch"] = earliest.timestamp()
    info["self_signed"] = leaf.subject == leaf.issuer
    try:
        ext = leaf.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        for general in ext.value:
            if isinstance(general, x509.DNSName):
                info["sans"].append(f"DNS:{general.value}")
            elif isinstance(general, x509.IPAddress):
                info["sans"].append(f"IP:{general.value}")
    except x509.ExtensionNotFound:
        pass
    return info


def _read_with_openssl(path: Path, info: dict,
                       runner: Optional[CommandRunner]) -> dict:
    run = run_command if runner is None else runner
    code, out = run(["openssl", "x509", "-in", str(path), "-noout",
                     "-enddate", "-subject", "-issuer"])
    if code != 0:
        info["error"] = (
            f"could not read {path.name}: no cryptography module and openssl "
            f"said: {out[-200:]}"
        )
        return info
    subject = issuer = ""
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("notAfter="):
            stamp = line.split("=", 1)[1].strip()
            try:
                epoch = ssl.cert_time_to_seconds(stamp)
            except ValueError:
                continue
            info["expires_epoch"] = float(epoch)
            info["expires"] = _dt.datetime.fromtimestamp(
                epoch, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        elif line.startswith("subject="):
            subject = _normalize_dn(line.split("=", 1)[1])
        elif line.startswith("issuer="):
            issuer = _normalize_dn(line.split("=", 1)[1])
    if subject or issuer:
        info["self_signed"] = subject == issuer
    if info["expires_epoch"] is None:
        info["error"] = f"could not read a notAfter date out of {path.name}"
    info["sans"] = _openssl_sans(path, run)
    return info


def _normalize_dn(text: str) -> str:
    """Compare DNs as text: OpenSSL prints ``CN=a, O=b``, LibreSSL ``/CN=a/O=b``."""
    parts = [p.strip() for p in re.split(r"[,/]", text) if p.strip()]
    return "|".join(sorted(parts))


def _openssl_sans(path: Path, run: CommandRunner) -> list[str]:
    """The SAN list via openssl, tolerating a build without ``-ext``."""
    code, out = run(["openssl", "x509", "-in", str(path), "-noout",
                     "-ext", "subjectAltName"])
    if code != 0:
        # LibreSSL has no -ext. -text carries the same block.
        code, out = run(["openssl", "x509", "-in", str(path), "-noout", "-text"])
        if code != 0:
            return []
        block: list[str] = []
        keep = False
        for line in out.splitlines():
            if "Subject Alternative Name" in line:
                keep = True
                continue
            if keep:
                block.append(line)
                break
        out = "\n".join(block)
    sans: list[str] = []
    for kind, value in _SAN_LINE_RE.findall(out):
        label = "DNS" if kind == "DNS" else "IP"
        entry = f"{label}:{value}"
        if entry not in sans:
            sans.append(entry)
    return sans


# ---------------------------------------------------------------------------
# The listener side
# ---------------------------------------------------------------------------

def ssl_context(cert_file, key_file) -> ssl.SSLContext:
    """A server SSL context for one pair. Raises if either half is unusable."""
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile=str(cert_file), keyfile=str(key_file))
    return context


def ensure_pair(home=None) -> tuple[Path, Path]:
    """The default pair paths, with their directory created."""
    from ainode.tls.config import cert_paths

    ensure_tls_dir(home)
    return cert_paths(home)
