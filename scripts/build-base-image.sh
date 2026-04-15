#!/usr/bin/env bash
# Build ghcr.io/getainode/ainode-base from eugr/spark-vllm-docker at a pinned
# commit. This is the "base image" that Dockerfile.ainode extends.
#
# End users never run this — only our CI runner (or maintainers rebuilding
# from source). Takes ~12 minutes on a DGX Spark with prebuilt vLLM wheels.
#
# Usage:
#   scripts/build-base-image.sh                 # build, tag locally
#   scripts/build-base-image.sh --push          # build + push to GHCR (requires docker login)
set -euo pipefail

# -- Pinned eugr reference ---------------------------------------------------
# Bumped by editing this file and committing; CI/build reproducibility
# depends on this being explicit.
EUGR_REPO="${EUGR_REPO:-https://github.com/eugr/spark-vllm-docker}"
EUGR_COMMIT="${EUGR_COMMIT:-c026c92bd0c1236f947ac212565b15a33ba1b4e7}"
EUGR_SHORT="${EUGR_COMMIT:0:7}"

# AINode version coupling (kept in sync with pyproject.toml).
AINODE_VERSION="${AINODE_VERSION:-0.4.0}"

REGISTRY="${REGISTRY:-ghcr.io/getainode}"
BASE_NAME="${BASE_NAME:-ainode-base}"

PUSH="false"
for arg in "$@"; do
    case "$arg" in
        --push) PUSH="true" ;;
        -h|--help)
            sed -n '1,30p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$SCRIPT_DIR/_eugr"

echo "==> Preparing eugr worktree at $WORKTREE (commit $EUGR_SHORT)"
if [ -d "$WORKTREE/.git" ]; then
    git -C "$WORKTREE" fetch --depth=1 origin "$EUGR_COMMIT"
    git -C "$WORKTREE" checkout -q "$EUGR_COMMIT"
else
    rm -rf "$WORKTREE"
    git clone --depth=1 "$EUGR_REPO" "$WORKTREE"
    # Shallow clone can't check out an arbitrary sha in one shot; deepen.
    git -C "$WORKTREE" fetch --depth=50 origin "$EUGR_COMMIT" || \
        git -C "$WORKTREE" fetch --unshallow
    git -C "$WORKTREE" checkout -q "$EUGR_COMMIT"
fi

echo "==> Building eugr image (prebuilt vLLM wheels expected; ~12 min)"
pushd "$WORKTREE" >/dev/null
./build-and-copy.sh --tag "vllm-node:${EUGR_SHORT}"
popd >/dev/null

# Tag the base image for our registry.
# Two tags:
#   <registry>/<name>:<eugr-commit>       — the eugr content identity
#   <registry>/<name>:<version>-<eugr>    — correlates ainode version
COMMIT_TAG="$REGISTRY/$BASE_NAME:$EUGR_SHORT"
VERSION_TAG="$REGISTRY/$BASE_NAME:${AINODE_VERSION}-${EUGR_SHORT}"

docker tag "vllm-node:${EUGR_SHORT}" "$COMMIT_TAG"
docker tag "vllm-node:${EUGR_SHORT}" "$VERSION_TAG"
echo "==> Tagged:"
echo "    $COMMIT_TAG"
echo "    $VERSION_TAG"

if [ "$PUSH" = "true" ]; then
    echo "==> Pushing"
    docker push "$COMMIT_TAG"
    docker push "$VERSION_TAG"
fi

echo "==> Done. To build the ainode layer:"
echo "    docker build -f scripts/Dockerfile.ainode -t ainode:${AINODE_VERSION} ."
