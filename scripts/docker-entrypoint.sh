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
fi

exec ainode start --in-container "$@"
