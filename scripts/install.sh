#!/bin/bash
# AINode installer — https://ainode.dev
# Usage: curl -fsSL https://ainode.dev/install | bash
#        AINODE_REPO=webdevtodayjason/ainode bash scripts/install.sh   (use a fork)
set -e

AINODE_VERSION="0.3.0"
AINODE_HOME="${AINODE_HOME:-$HOME/.ainode}"
export AINODE_HOME
VENV_DIR="$AINODE_HOME/venv"
AINODE_REPO="${AINODE_REPO:-webdevtodayjason/ainode}"
AINODE_BRANCH="${AINODE_BRANCH:-main}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

banner() {
    echo ""
    echo -e "${CYAN}    ╔══════════════════════════════════════╗${NC}"
    echo -e "${CYAN}    ║            A I N o d e               ║${NC}"
    echo -e "${CYAN}    ║   Your local AI platform for NVIDIA  ║${NC}"
    echo -e "${CYAN}    ╚══════════════════════════════════════╝${NC}"
    echo -e "    ${BLUE}Powered by argentos.ai${NC}"
    echo ""
}

log()   { echo -e "    ${GREEN}✓${NC} $1"; }
info()  { echo -e "    ${BLUE}→${NC} $1"; }
warn()  { echo -e "    ${YELLOW}!${NC} $1"; }
fail()  { echo -e "    ${RED}✗${NC} $1"; exit 1; }
step()  { echo ""; echo -e "    ${CYAN}[$1]${NC} $2"; }

banner
echo "    Installing AINode v${AINODE_VERSION}..."
echo "    Source: ${AINODE_REPO}@${AINODE_BRANCH}"
echo ""

# ── System checks ─────────────────────────────────────────────────────────

[[ "$(uname)" != "Linux" ]] && fail "AINode requires Linux."

command -v python3 &>/dev/null || fail "Python 3 required. Install: sudo apt install python3 python3-venv python3-pip"

PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log "Python ${PYTHON_VER}"

# Ensure python3-dev + venv tools are present (needed for pip-from-source builds)
if ! python3 -c "import sysconfig, ensurepip" 2>/dev/null; then
    info "Installing python3-dev / python3-venv..."
    sudo apt-get install -y python3-dev python3-venv python3-pip 2>/dev/null || warn "apt install skipped — make sure these are present"
fi

command -v git &>/dev/null || { info "Installing git..."; sudo apt-get install -y git 2>/dev/null || fail "git required"; }

# ── GPU + CUDA detection ─────────────────────────────────────────────────

GPU_NAME=""
GPU_COMPUTE=""
CUDA_MAJOR=""
ENGINE_STRATEGY="pip"

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_COMPUTE=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU_NAME" ]; then
        log "GPU: ${GPU_NAME} (sm_${GPU_COMPUTE/./})"
    fi
else
    warn "No NVIDIA GPU detected — AINode will install but inference will be unavailable"
fi

NVCC_PATH=""
for p in /usr/local/cuda/bin/nvcc /usr/local/cuda-13.0/bin/nvcc /usr/local/cuda-12.*/bin/nvcc; do
    [ -x "$p" ] && NVCC_PATH="$p" && break
done

if [ -n "$NVCC_PATH" ]; then
    CUDA_VER=$("$NVCC_PATH" --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    log "CUDA ${CUDA_VER}"
fi

ARCH=$(uname -m)
log "Architecture: ${ARCH}"

# Engine strategy:
#   GB10 / sm_121 / CUDA 13 → Docker container (pip wheels don't target CUDA 13)
#   Everything else        → pip vLLM (pre-built wheels)
if [[ "$GPU_NAME" =~ "GB10" ]] || [[ "$GPU_COMPUTE" == "12.1" ]] || [[ "$CUDA_MAJOR" == "13" ]]; then
    ENGINE_STRATEGY="docker"
    info "Blackwell GB10 / CUDA 13 detected — will use Docker-based vLLM"
fi

# ── Create AINode home + venv ────────────────────────────────────────────

mkdir -p "$AINODE_HOME"/{models,logs,datasets,training}

step "1/4" "Creating virtual environment..."
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip setuptools wheel
log "Virtual environment ready: ${VENV_DIR}"

# ── Install AINode ────────────────────────────────────────────────────────

step "2/4" "Installing AINode from ${AINODE_REPO}..."
pip install --quiet "git+https://github.com/${AINODE_REPO}.git@${AINODE_BRANCH}"
log "AINode installed"

# Embeddings (sentence-transformers) — small, CPU-friendly, enabled by default.
# Failure is non-fatal: AINode runs fine without it, the embeddings endpoint
# just returns a 503 dependency-missing error until it's installed.
info "Installing embedding support (sentence-transformers)..."
if pip install --quiet sentence-transformers 2>&1 | tail -3; then
    log "Embedding support ready"
else
    warn "sentence-transformers install failed — embeddings endpoint will be unavailable"
fi

# Distributed inference (Ray) — required for multi-node tensor-parallel.
# Failure is non-fatal: single-node inference still works.
info "Installing Ray (distributed inference)..."
if pip install --quiet 'ray[default]>=2.9' 2>&1 | tail -3; then
    log "Ray ready"
else
    warn "Ray install failed — multi-node sharding disabled until: pip install 'ray[default]'"
fi

# ── Install inference engine ──────────────────────────────────────────────

if [ "$ENGINE_STRATEGY" = "docker" ]; then
    step "3/4" "Setting up Docker-based vLLM (Blackwell GB10)..."

    if ! command -v docker &>/dev/null; then
        info "Docker not found — installing..."
        curl -fsSL https://get.docker.com | sudo sh 2>&1 | tail -3 || warn "Docker install failed"
        sudo usermod -aG docker "$USER" 2>/dev/null || true
    fi

    if ! sudo docker info &>/dev/null; then
        info "Starting Docker..."
        sudo systemctl reset-failed docker.service 2>/dev/null || true
        sudo rm -rf /var/lib/docker/buildkit 2>/dev/null || true
        sudo systemctl start docker 2>&1 | tail -3 || warn "Could not start Docker — start it manually with: sudo systemctl start docker"
    fi

    if sudo docker info &>/dev/null; then
        log "Docker is running"
        info "Pulling vLLM container for GB10 (a few minutes)..."
        sudo docker pull scitrera/dgx-spark-vllm:0.17.0-t5 2>&1 | tail -3 || warn "Container pull failed — re-run: sudo docker pull scitrera/dgx-spark-vllm:0.17.0-t5"
        log "vLLM container ready: scitrera/dgx-spark-vllm:0.17.0-t5"
    else
        warn "Docker not available — AINode will install without inference engine"
    fi

    # ── Write compose file + .env ────────────────────────────────────────
    info "Writing ~/.ainode/docker-compose.yml..."
    cat > "$AINODE_HOME/docker-compose.yml" <<'COMPOSE_EOF'
services:
  vllm:
    image: scitrera/dgx-spark-vllm:0.17.0-t5
    container_name: ainode-vllm
    restart: unless-stopped
    ipc: host
    shm_size: "16gb"
    ports:
      - "127.0.0.1:8000:8000"
    volumes:
      - ${HOME}/.ainode/models:/models
    environment:
      HF_HOME: /models
      AINODE_MODEL: ${AINODE_MODEL:-meta-llama/Llama-3.2-3B-Instruct}
      AINODE_GPU_MEMORY_UTIL: ${AINODE_GPU_MEMORY_UTIL:-0.9}
      AINODE_TP_SIZE: ${AINODE_TP_SIZE:-1}
      AINODE_RAY_ADDRESS: ${AINODE_RAY_ADDRESS:-}
    command:
      - vllm
      - serve
      - ${AINODE_MODEL:-meta-llama/Llama-3.2-3B-Instruct}
      - --host
      - 0.0.0.0
      - --port
      - "8000"
      - --gpu-memory-utilization
      - ${AINODE_GPU_MEMORY_UTIL:-0.9}
      - --tensor-parallel-size
      - ${AINODE_TP_SIZE:-1}
      - --dtype
      - bfloat16
      - --download-dir
      - /models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
COMPOSE_EOF
    log "docker-compose.yml written"

    info "Writing ~/.ainode/.env..."
    cat > "$AINODE_HOME/.env" <<ENV_EOF
AINODE_MODEL=meta-llama/Llama-3.2-3B-Instruct
AINODE_GPU_MEMORY_UTIL=0.9
AINODE_TP_SIZE=1
AINODE_RAY_ADDRESS=
ENV_EOF
    log ".env written"

    # Seed config.json with docker strategy. Merge with existing config if present.
    info "Seeding ~/.ainode/config.json (engine_strategy=docker)..."
    python3 - <<PYEOF
import json, os
from pathlib import Path
cfg_path = Path(os.environ["AINODE_HOME"]) / "config.json"
existing = {}
if cfg_path.exists():
    try:
        existing = json.loads(cfg_path.read_text())
    except Exception:
        existing = {}
existing.update({
    "engine_strategy": "docker",
    "discovery_port": 5679,
    "cluster_id": "default",
    "onboarded": True,
})
cfg_path.write_text(json.dumps(existing, indent=2))
PYEOF
    log "config.json seeded"

    # ── systemd user unit ────────────────────────────────────────────────
    info "Writing systemd user unit..."
    mkdir -p "$HOME/.config/systemd/user"
    cat > "$HOME/.config/systemd/user/ainode.service" <<UNIT_EOF
[Unit]
Description=AINode
After=network-online.target docker.service

[Service]
Type=simple
ExecStart=%h/.ainode/venv/bin/ainode start
ExecStop=%h/.ainode/venv/bin/ainode stop
Restart=on-failure
RestartSec=5
Environment=PATH=%h/.ainode/venv/bin:/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=default.target
UNIT_EOF
    log "systemd user unit installed"

    systemctl --user daemon-reload 2>&1 | tail -3 || warn "systemctl daemon-reload failed"
    systemctl --user enable --now ainode.service 2>&1 | tail -3 \
        && log "ainode.service enabled and started" \
        || warn "systemctl --user enable --now ainode.service failed — start manually"

    # Linger so the service survives logout / starts at boot. Non-fatal if it fails.
    sudo loginctl enable-linger "$USER" 2>/dev/null \
        && log "Linger enabled for $USER" \
        || warn "loginctl enable-linger failed — service will start on login instead of boot"

    AINODE_SERVICE_STARTED=1
else
    step "3/4" "Installing vLLM (pip)..."
    pip install --quiet vllm 2>&1 | tail -3 || warn "vLLM pip install failed — install it later with: pip install vllm"
    VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null) && \
        log "vLLM ${VLLM_VER}" || warn "vLLM import failed"
fi

# Verify torch
TORCH_AVAIL=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [ "$TORCH_AVAIL" = "True" ]; then
    TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    log "PyTorch CUDA ${TORCH_CUDA}"
fi

# ── PATH setup ────────────────────────────────────────────────────────────

step "4/4" "Finishing up..."

AINODE_BIN="$VENV_DIR/bin"
SHELL_RC=""
[ -f "$HOME/.bashrc" ] && SHELL_RC="$HOME/.bashrc"
[ -f "$HOME/.zshrc" ] && SHELL_RC="$HOME/.zshrc"

if [ -n "$SHELL_RC" ] && ! grep -q "AINode" "$SHELL_RC" 2>/dev/null; then
    {
        echo ""
        echo "# AINode — https://ainode.dev"
        echo "export PATH=\"$AINODE_BIN:\$PATH\""
    } >> "$SHELL_RC"
    log "Added to PATH in ${SHELL_RC}"
fi

# ── systemd service (optional) ────────────────────────────────────────────

INSTALL_SERVICE="${AINODE_SERVICE:-}"
if [ -z "$INSTALL_SERVICE" ] && [ -t 0 ]; then
    echo ""
    echo -n "    Install as systemd service (auto-start on boot)? [y/N] "
    read -r INSTALL_SERVICE
fi

if [[ "$INSTALL_SERVICE" =~ ^[Yy]$ ]]; then
    echo ""
    if [ "$(id -u)" -eq 0 ]; then
        "$AINODE_BIN/ainode" service install
    else
        "$AINODE_BIN/ainode" service install --user
    fi
    log "AINode service installed and enabled"
fi

# ── Done ──────────────────────────────────────────────────────────────────

echo ""
echo -e "    ${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "    ${GREEN}  AINode v${AINODE_VERSION} installed!${NC}"
echo -e "    ${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ "${AINODE_SERVICE_STARTED:-0}" = "1" ]; then
    echo "    AINode service is already running under systemd."
    echo ""
    echo "    Validate:"
    echo ""
    echo -e "      ${CYAN}systemctl --user status ainode.service${NC}"
    echo -e "      ${CYAN}docker ps --filter name=ainode-vllm${NC}"
    echo -e "      ${CYAN}curl http://localhost:3000/api/health${NC}"
    echo ""
    echo "    Open http://localhost:3000 in your browser."
else
    echo "    To start:"
    echo ""
    echo -e "      ${CYAN}source ${SHELL_RC:-~/.bashrc}${NC}"
    echo -e "      ${CYAN}ainode start${NC}"
    echo ""
    echo "    Open http://localhost:3000 in your browser."
fi
echo ""
echo -e "    Powered by ${BLUE}argentos.ai${NC}"
echo ""
