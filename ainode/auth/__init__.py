"""AINode authentication: API keys for machines, logins for people.

Two credentials live here, plus the one the fleet derives for itself:

* ``middleware.AuthConfig`` and ``api_routes`` are the API keys (``auth.json``),
  which is what the bench, the desktop app and ``curl`` present.
* ``accounts.UsersStore`` and ``session_routes`` are the named accounts and login
  sessions (``users.json``), which is what a person presents (#261).
* ``fleet`` derives a node-to-node key from ``cluster_secret``, so a cluster can
  run with auth on everywhere without a second credential to distribute.
"""
