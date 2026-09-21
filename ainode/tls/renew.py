"""When a tailnet certificate has to be replaced, and who does the replacing.

A ``tailscale cert`` pair is Let's Encrypt, so it lives 90 days. Something has
to notice day 76 without being asked. Three candidates were on the table and
only one of them actually renews with nobody watching:

* **The server's own periodic work.** It cannot. The server runs inside the
  AINode container, which ships no ``tailscale`` binary and has no access to the
  tailnet daemon socket, so the process that notices is structurally unable to
  act. It would also have to restart itself to serve the new pair, because the
  listener and its ``SSLContext`` are built at boot.
* **``ainode doctor``.** It already warns inside 14 days, and it is the right
  place for that warning, but it only speaks when a human runs it. A renewal
  that depends on somebody running the doctor in the right fortnight is not a
  renewal, it is a reminder that might arrive.
* **A systemd timer on the HOST, rendered by the installer.** The host is where
  ``tailscale`` is, where sudo is, and where ``systemctl restart ainode`` can be
  run. This is the one that renews without a human, so it is what
  ``scripts/install.sh`` writes: ``ainode-tls-renew.timer`` fires daily, the
  host wrapper asks the container for the decision this module makes, and only
  when the answer is "renew" does it re-run ``tailscale cert`` and restart the
  node.

This module is only the DECISION: a pure function of a certificate reading and
a clock, so the timer, the CLI and the tests all ask the same question and get
the same answer. Nothing here touches the network, the filesystem or systemd.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

#: Renew with this many days left or fewer.
#:
#: 14 is inside Tailscale's own renewal window for a 90-day Let's Encrypt pair,
#: so a ``tailscale cert`` at this point returns a freshly issued certificate
#: rather than the cached one it would hand back earlier in the cycle. It also
#: leaves two weeks of daily retries before anything actually expires, which is
#: the margin that matters when the renewal runs unattended. The doctor warns on
#: the same number, deliberately: the operator and the timer must not disagree
#: about when a certificate is getting old.
RENEW_THRESHOLD_DAYS = 14

#: The suffix every MagicDNS name ends in. A certificate whose names are not in
#: the tailnet was not issued by ``tailscale cert`` and must not be renewed by
#: something that only knows how to call ``tailscale cert``.
TAILNET_SUFFIX = ".ts.net"


@dataclass
class RenewalDecision:
    """Whether to re-run ``tailscale cert``, and the sentence that says why."""

    renew: bool
    reason: str
    #: The MagicDNS name to pass to ``tailscale cert``, or "" when there is none.
    name: str = ""
    days_left: Optional[float] = None

    def as_lines(self) -> list[str]:
        """``key=value`` lines, which is what the host wrapper reads.

        Deliberately not JSON: the consumer is a bash function in the installer's
        wrapper, and ``sed``-ing one key out of a JSON blob is how that sort of
        thing goes wrong quietly.
        """
        return [
            f"renew={'yes' if self.renew else 'no'}",
            f"name={self.name}",
            f"days_left={'' if self.days_left is None else self.days_left}",
            f"reason={self.reason}",
        ]


def tailnet_name_from_sans(sans) -> str:
    """The first MagicDNS name in a SAN list, or "".

    ``tailscale cert`` issues for exactly one name, so this is normally the only
    entry. Reading it back off the certificate rather than re-deriving it means
    the renewal asks for the name the certificate in place was issued for, even
    on a node that has since been renamed in the tailnet.
    """
    for entry in sans or []:
        text = str(entry)
        if not text.upper().startswith("DNS:"):
            continue
        name = text.split(":", 1)[1].strip().rstrip(".")
        if name.lower().endswith(TAILNET_SUFFIX):
            return name
    return ""


def renewal_decision(tls, info,
                     threshold_days: int = RENEW_THRESHOLD_DAYS,
                     tailnet_name: Optional[str] = None) -> RenewalDecision:
    """Decide whether this node's certificate needs replacing right now.

    *tls* is the ``tls`` block (anything with ``enabled`` and ``cert_file``),
    *info* is a ``certs.certificate_info`` reading of the certificate on disk.

    The refusals matter as much as the approvals, because the caller acts as
    root on the host:

    * **A certificate this code cannot read is never replaced.** "I could not
      parse it" is not evidence that it is expiring, and overwriting a working
      pair on a guess is worse than a warning nobody read.
    * **A self-signed certificate is never replaced by a tailnet one.** Its SANs
      are the hostname and every address on the box; a MagicDNS certificate
      carries one name. Swapping them silently would break every client
      reaching this node by IP, which on a LAN is most of them.
    * **A certificate issued for a name outside the tailnet is left alone.**
      ``tailscale cert`` cannot issue for it, so the only honest answer is to
      say who has to renew it.
    """
    enabled = bool(getattr(tls, "enabled", False))
    cert_file = str(getattr(tls, "cert_file", "") or "")
    info = info or {}
    days = info.get("days_left")
    name = tailnet_name_from_sans(info.get("sans")) or (tailnet_name or "")

    if not enabled:
        return RenewalDecision(False, "TLS is off on this node, so there is "
                                      "nothing to renew.", name)
    if not cert_file:
        return RenewalDecision(False, "TLS is on with no certificate configured: "
                                      "run `ainode tls enable --tailscale`.", name)
    if not info.get("exists"):
        # The one case where a missing file means act: the block says this node
        # serves HTTPS, so the pair being gone is a node that came up on HTTP
        # only, and re-running the cert step is exactly the repair.
        return RenewalDecision(bool(name), (
            f"{cert_file} is not there, so this node is serving HTTP only"
            + ("." if name else " and no tailnet name is known for it.")
        ), name)
    if info.get("error"):
        return RenewalDecision(False, f"the certificate cannot be read "
                                      f"({info['error']}), so nothing is replaced "
                                      f"on a guess.", name)
    if info.get("self_signed"):
        return RenewalDecision(False, (
            "the certificate is self-signed, and its names are this node's "
            "hostname and addresses rather than one tailnet name: replace it "
            "with `ainode tls enable` (or move to `ainode tls enable "
            "--tailscale` deliberately)."
        ), name, days)
    if not name:
        return RenewalDecision(False, (
            "the certificate is CA-issued for a name outside the tailnet, so "
            "`tailscale cert` cannot renew it: renew it the way it was issued."
        ), "", days)
    if days is None:
        return RenewalDecision(False, "the certificate carries no expiry this "
                                      "release can read.", name)
    if float(days) > float(threshold_days):
        return RenewalDecision(False, (
            f"{days} days left on {name}, which is more than the {threshold_days} "
            f"day window: nothing to do yet."
        ), name, days)
    return RenewalDecision(True, (
        f"{days} days left on {name}, inside the {threshold_days} day window."
    ), name, days)
