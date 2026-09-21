"""The ``tls`` block in config.json, and where the pair lives on disk.

Two rules this module exists to hold:

1. **The pair lives under ``<AINODE_HOME>/tls/``.** That is the directory the
   host bind-mounts into the container, so it is the only place a file written by
   the CLI is also readable by the server. A ``--cert``/``--key`` pointing
   anywhere else is COPIED in rather than referenced, because a path like
   ``/etc/letsencrypt/live/...`` does not exist inside the container and the
   server would come up with TLS enabled and no certificate.
2. **Writing the block never rewrites the rest of config.json.** ``NodeConfig``
   loads by filtering to its own fields, so a ``load()`` + ``save()`` round trip
   silently drops any key this release does not know about. The TLS CLI edits one
   key in place instead (same approach as ``doctor --fix``).
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from ainode.core.config import AINODE_HOME

#: Where TLS listens when nothing says otherwise. Deliberately not 443: the
#: server does not run as root (and 3000 was moved off 8443/443 for the same
#: reason), and a port above 1024 keeps `ainode tls enable` from needing sudo.
DEFAULT_TLS_PORT = 3443

CERT_NAME = "cert.pem"
KEY_NAME = "key.pem"


def tailnet_pair_paths(name: str, home=None) -> tuple[Path, Path]:
    """``(<AINODE_HOME>/tls/<name>.crt, .../<name>.key)`` for a MagicDNS name.

    Named after the certificate's one name rather than ``cert.pem``, because this
    pair is written by ``tailscale cert`` on the HOST and read by the server
    inside the container: the file name is the only thing carrying which name it
    was issued for across that boundary, and it is what lets the host wrapper and
    the container agree on the pair without either one parsing a certificate.
    """
    directory = tls_dir(home)
    stem = str(name).strip().rstrip(".")
    return directory / f"{stem}.crt", directory / f"{stem}.key"


@dataclass
class TLSConfig:
    """The ``tls`` block, with the defaults a node that never ran the CLI has."""

    enabled: bool = False
    cert_file: str = ""
    key_file: str = ""
    port: int = DEFAULT_TLS_PORT

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "TLSConfig":
        """Read the block, tolerating every shape a hand-edited file can have.

        A missing key, a null, a string port: all of it resolves to the default
        rather than raising, because this is read on the boot path and a typo in
        config.json must not be the reason a node has no web server at all.
        """
        if not isinstance(data, dict):
            return cls()
        try:
            port = int(data.get("port") or DEFAULT_TLS_PORT)
        except (TypeError, ValueError):
            port = DEFAULT_TLS_PORT
        if port <= 0 or port > 65535:
            port = DEFAULT_TLS_PORT
        return cls(
            enabled=bool(data.get("enabled", False)),
            cert_file=str(data.get("cert_file") or ""),
            key_file=str(data.get("key_file") or ""),
            port=port,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def files_present(self) -> bool:
        """True when both halves of the pair are readable files right now."""
        if not self.cert_file or not self.key_file:
            return False
        return Path(self.cert_file).is_file() and Path(self.key_file).is_file()

    def usable(self) -> bool:
        """True when this node should actually open a TLS listener."""
        return bool(self.enabled) and self.files_present()


def ainode_home(home=None) -> Path:
    """``AINODE_HOME``, resolved at call time.

    At call time on purpose: the variable is read from the environment inside the
    container and every test points it at a tmpdir, so a module-level constant
    would freeze whichever value happened to be set at import.
    """
    return Path(home or os.environ.get("AINODE_HOME") or AINODE_HOME)


def tls_dir(home=None) -> Path:
    """``<AINODE_HOME>/tls``, resolved at call time."""
    return ainode_home(home) / "tls"


def cert_paths(home=None) -> tuple[Path, Path]:
    """The default ``(cert, key)`` pair paths under ``<AINODE_HOME>/tls/``."""
    directory = tls_dir(home)
    return directory / CERT_NAME, directory / KEY_NAME


def ensure_tls_dir(home=None) -> Path:
    """Create the TLS directory, owner-only, and return it."""
    directory = tls_dir(home)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(directory, 0o700)
    except OSError:
        pass
    return directory


def harden_key(path) -> None:
    """Make a private key readable only by its owner (0600).

    Called on every path that produces or copies a key: a key mode 0644 in a
    bind-mounted directory is readable by anything else that mounts it.
    """
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def install_pair(cert_src, key_src, home=None) -> tuple[Path, Path]:
    """Copy an operator-supplied pair into ``<AINODE_HOME>/tls/``.

    Returns the installed ``(cert, key)`` paths. Copying rather than referencing
    is the whole point: see this module's docstring. A source already inside the
    TLS directory is left exactly where it is.
    """
    cert_src = Path(cert_src)
    key_src = Path(key_src)
    for path in (cert_src, key_src):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    directory = ensure_tls_dir(home)
    cert_dst = directory / cert_src.name
    key_dst = directory / key_src.name
    if cert_src.resolve() != cert_dst.resolve():
        cert_dst.write_bytes(cert_src.read_bytes())
    if key_src.resolve() != key_dst.resolve():
        key_dst.write_bytes(key_src.read_bytes())
    try:
        os.chmod(cert_dst, 0o644)
    except OSError:
        pass
    harden_key(key_dst)
    return cert_dst, key_dst


def load_tls_config(config) -> TLSConfig:
    """The ``tls`` block off a ``NodeConfig`` (or anything with a ``tls`` attr)."""
    return TLSConfig.from_dict(getattr(config, "tls", None))


def save_tls_config(tls: TLSConfig, config_path=None) -> Path:
    """Write the ``tls`` block into config.json, leaving every other key as it is.

    Atomic (temp file plus rename) so a node reading config.json while this runs
    never sees half a file.
    """
    path = Path(config_path) if config_path else ainode_home() / "config.json"
    data: dict = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict):
                data = loaded
        except ValueError:
            data = {}
    data["tls"] = tls.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tls-tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)
    return path
