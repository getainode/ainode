"""TLS for the AINode API port.

Nothing in the product spoke HTTPS before this: the dashboard, the management
API and the OpenAI-compatible API all lived on one plain HTTP port, so an API
key travelled in clear text and a Mac client with App Transport Security could
not talk to a node at all without an exception.

The shape, in one place:

* HTTP on ``web_port`` (3000) is untouched and stays untouched. Every client in
  the fleet, the peer proxy included, talks to it, so TLS is an ADDITIONAL
  listener on its own port and never a replacement.
* The certificate and key live under ``<AINODE_HOME>/tls/``. That directory is
  the bind mount, so the CLI (which runs on the host, forwarded into the
  container by the installer's wrapper) and the server (which runs inside the
  container) see the same two files under different absolute paths.
* ``ainode tls enable`` writes the config block; the server reads it at boot.
  Turning TLS on is therefore a restart, not a live reconfiguration.

Out of scope here: TLS between peers (the fleet proxy keeps talking HTTP on the
LAN) and ACME over HTTP-01. A real certificate for a tailnet name comes from
``tailscale cert``, which is what ``--tailscale`` runs.
"""

from ainode.tls.certs import (
    CERT_DAYS,
    certificate_info,
    generate_self_signed,
    host_ip_addresses,
    san_entries,
    ssl_context,
    tailscale_cert,
    tailscale_dns_name,
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

__all__ = [
    "CERT_DAYS",
    "DEFAULT_TLS_PORT",
    "TLSConfig",
    "cert_paths",
    "certificate_info",
    "generate_self_signed",
    "host_ip_addresses",
    "install_pair",
    "load_tls_config",
    "san_entries",
    "save_tls_config",
    "ssl_context",
    "tailscale_cert",
    "tailscale_dns_name",
    "tls_dir",
]
