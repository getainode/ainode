#!/usr/bin/env bash
# AINode container entrypoint.
# Delegates to `ainode start --in-container`. The CLI reads
# ~/.ainode/config.json (mounted via -v <host>/.ainode:/root/.ainode) and
# picks solo vs head via `distributed_mode`.
set -euo pipefail

: "${AINODE_HOME:=/root/.ainode}"
mkdir -p "$AINODE_HOME"/{models,logs,datasets,training}

# SSH keys: the host mounts its ~/.ssh at /host-ssh (read-only). OpenSSH
# refuses to use keys not owned by the invoking user, so copy them into
# /root/.ssh owned by root with the right modes. The caller's docker run
# must use `-v $HOME/.ssh:/host-ssh:ro` (not /root/.ssh — that directory
# is created + populated here).
if [ -d /host-ssh ]; then
    rm -rf /root/.ssh
    mkdir -p /root/.ssh
    cp -rL /host-ssh/. /root/.ssh/ 2>/dev/null || true
    chown -R root:root /root/.ssh
    chmod 700 /root/.ssh
    find /root/.ssh -type f -exec chmod 600 {} +
    find /root/.ssh -type f -name '*.pub' -exec chmod 644 {} + || true

    # The container runs as root but host SSH keys belong to the host user
    # (e.g. `sem`). eugr's launcher and other tools run `ssh <host>` without
    # a username and default to $USER=root — fail. Inject a User directive
    # into ssh_config keyed on the peer IPs from config.json so
    # `ssh 10.0.0.2` means `ssh sem@10.0.0.2`.
    if command -v python3 >/dev/null && [ -f "$AINODE_HOME/config.json" ]; then
        python3 - <<'PY'
import json, os, pathlib
cfg_path = pathlib.Path(os.environ.get("AINODE_HOME", "/root/.ainode")) / "config.json"
try:
    cfg = json.loads(cfg_path.read_text())
except Exception:
    cfg = {}
ssh_user = cfg.get("ssh_user") or ""
peers = cfg.get("peer_ips") or []
if ssh_user and peers:
    ssh_config = pathlib.Path("/root/.ssh/config")
    existing = ssh_config.read_text() if ssh_config.exists() else ""
    block = "\n# Injected by AINode entrypoint for container→peer ssh\n"
    block += f"Host {' '.join(peers)}\n"
    block += f"    User {ssh_user}\n"
    block += "    StrictHostKeyChecking no\n"
    block += "    UserKnownHostsFile /root/.ssh/known_hosts\n"
    ssh_config.write_text(block + existing)
    ssh_config.chmod(0o600)
PY
    fi
fi

exec ainode start --in-container "$@"
