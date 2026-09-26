"""The fleet key: how one AINode node authenticates to another.

Auth was usable by the dashboard (#218) and unusable by a cluster: a node with
``auth.enabled`` refused every request its own peers make, so the federated
union, the load and unload fan-outs, the cluster update and the chat card's peer
config read all answered 401 the moment anybody turned auth on. The fleet ran
open because turning it on split the fleet.

The credential is derived, not stored::

    fleet key = HMAC-SHA256(cluster_secret, "ainode-fleet-key-v1")

Three properties follow from that one line, and they are the reason it is a
derivation rather than a key file:

* **Every node with the secret computes the same key**, so the join flow already
  distributes it: ``POST /api/cluster/join`` hands a joiner the
  ``cluster_secret`` (``api/cluster_join.py``), and from that moment the joiner
  is authenticated to every node in the cluster with no second step, no key to
  copy and no route that hands out credentials.
* **Rotation follows the secret.** Roll ``cluster_secret`` and the fleet key
  rolls with it, at the same moment and on the same nodes, with the same
  "the cluster is only dark for the nodes that disagree" behaviour discovery
  signing already has (``discovery/signing.py::ClusterSecret``). The secret is
  therefore read per request, never captured at startup.
* **No new state on disk.** ``auth.json`` keeps exactly the operator keys it had;
  nothing writes the fleet key anywhere, so there is nothing to leak, back up,
  or forget to revoke. A node with no ``cluster_secret`` has no fleet key, which
  is why ``ainode doctor`` FAILs a node that has auth on and peers but no
  secret: that node cannot talk to its own cluster.

The label is a constant so the fleet key is not the secret itself. The secret
also signs discovery datagrams, and one value used unchanged as both a signing
key and a bearer token means anything that reads an HTTP header (a proxy log, a
crash report) has learned the key the cluster's UDP wire trusts. Domain
separation costs one HMAC.

The middleware accepts it as the caller id ``fleet``
(``auth/middleware.py``), and every outbound node-to-node request sends it
through :func:`fleet_headers`. An engine port is NOT a node-to-node request in
this sense: the inference proxy, the capability probes and the embeddings route
talk straight to a vLLM container, which AINode's middleware never sees.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any, Mapping, Optional

#: What the fleet key is an HMAC over. Versioned in the string so a future
#: derivation can change without a silent mismatch between two releases: a node
#: computing v2 would simply not authenticate against a node computing v1, and
#: the 401 says which key id was expected.
FLEET_KEY_LABEL = b"ainode-fleet-key-v1"

#: The ``api_key_id`` stamped on a request that presented the fleet key. Not a
#: stored key, so it has no entry in ``auth.json`` and cannot be revoked from the
#: dashboard: revoking the fleet's access to a node means changing that node's
#: ``cluster_secret``, which is the same act as removing it from the cluster.
FLEET_KEY_ID = "fleet"
#: Set by a node that hands a decision request (/v1/decide, /v1/systemone) to the
#: node that owns the model. The owner answers it itself and never forwards it
#: again, and its rate limiter does not count it: the node the caller reached has
#: already counted the caller, and every forwarded request arrives under the one
#: fleet key, so counting them here would make the whole fleet one client.
FORWARDED_BY_HEADER = "X-AINode-Forwarded-By"


def fleet_key(secret: Optional[str]) -> str:
    """The fleet key for *secret*, or "" when there is no secret.

    "" is never a valid token: :func:`is_fleet_key` refuses it on both sides, so
    a node with no ``cluster_secret`` neither sends nor accepts a fleet key
    rather than accepting an empty one.
    """
    if not secret:
        return ""
    return hmac.new(
        str(secret).encode("utf-8"), FLEET_KEY_LABEL, hashlib.sha256
    ).hexdigest()


def is_fleet_key(token: str, secret: Optional[str]) -> bool:
    """Is *token* the fleet key derived from *secret*? Constant time."""
    expected = fleet_key(secret)
    if not expected or not token:
        return False
    return hmac.compare_digest(str(token), expected)


def fleet_key_headers(secret: Optional[str]) -> dict:
    """``{"Authorization": "Bearer <fleet key>"}`` for *secret*, or ``{}``.

    The primitive under :func:`fleet_headers`, for the callers that hold a
    ``cluster_secret`` rather than a running app: ``ainode doctor`` reads its own
    node's ``/api/nodes`` and each peer's ``/api/status``, and it has the config
    file, not the application.
    """
    key = fleet_key(secret)
    return {"Authorization": f"Bearer {key}"} if key else {}


def cluster_secret_of(app: Any) -> str:
    """This node's live ``cluster_secret``, or "".

    ``app["cluster_secret"]`` is the :class:`~ainode.discovery.signing.ClusterSecret`
    the discovery sender and listener share, which re-reads ``config.json`` when
    it changes: asking it per request is what makes rotation a config edit rather
    than a fleet-wide restart. It is only installed when clustering is on, so the
    in-memory config is the fallback, which is also the whole answer on a node
    with discovery switched off.
    """
    provider = None
    getter = getattr(app, "get", None)
    if callable(getter):
        provider = getter("cluster_secret")
    if callable(provider):
        try:
            value = provider()
        except Exception:  # pragma: no cover - a broken provider is not fatal
            value = None
        if value:
            return str(value)
    config = getter("config") if callable(getter) else None
    return str(getattr(config, "cluster_secret", "") or "")


def fleet_headers(app: Any,
                  existing: Optional[Mapping[str, str]] = None) -> dict:
    """*existing* plus ``Authorization: Bearer <fleet key>``, when there is one.

    The one helper every node-to-node call site uses, so "does this request carry
    the fleet key" is answerable by reading one name rather than by auditing
    every ``session.post``. Pure: it copies what it was given and never mutates
    it, and it never overwrites an ``Authorization`` the caller set itself (the
    embeddings and inference proxies pass a caller's own headers through, and the
    caller's credential is the one the engine should see).

    With no ``cluster_secret`` this is *existing* unchanged, so a node that never
    had a secret makes exactly the requests it made before.
    """
    headers = {str(k): str(v) for k, v in (existing or {}).items()}
    if any(k.lower() == "authorization" for k in headers):
        return headers
    headers.update(fleet_key_headers(cluster_secret_of(app)))
    return headers
