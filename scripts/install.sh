#!/bin/bash
# AINode installer — https://ainode.dev
# Usage: curl -fsSL https://ainode.dev/install | bash
set -e

AINODE_VERSION="0.1.0"
AINODE_HOME="${AINODE_HOME:-$HOME/.ainode}"
VENV_DIR="$AINODE_HOME/venv"

# vLLM source build config (for Blackwell/CUDA 13)
VLLM_COMMIT="66a168a197ba214a5b70a74fa2e713c9eeb3251a"
TRITON_COMMIT="4caa0328bf8df64896dd5f6fb9df41b0eb2e750a"

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

log()     { echo -e "    ${GREEN}✓${NC} $1"; }
info()    { echo -e "    ${BLUE}→${NC} $1"; }
warn()    { echo -e "    ${YELLOW}!${NC} $1"; }
fail()    { echo -e "    ${RED}✗${NC} $1"; exit 1; }
step()    { echo ""; echo -e "    ${CYAN}[$1]${NC} $2"; }

banner
echo "    Installing AINode v${AINODE_VERSION}..."
echo ""

# ── System checks ─────────────────────────────────────────────────────────

[[ "$(uname)" != "Linux" ]] && fail "AINode requires Linux."

command -v python3 &>/dev/null || fail "Python 3 required. Install: sudo apt install python3 python3-venv python3-pip"

# Ensure python3-dev is installed (needed for source builds)
if ! python3 -c "import sysconfig; assert sysconfig.get_path('include')" 2>/dev/null; then
    info "Installing python3-dev..."
    sudo apt-get install -y python3-dev 2>/dev/null || warn "Could not install python3-dev — source builds may fail"
fi

PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log "Python ${PYTHON_VER}"

# ── GPU detection ─────────────────────────────────────────────────────────

GPU_NAME=""
GPU_ARCH=""
CUDA_MAJOR=""
NEEDS_SOURCE_BUILD=false

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_COMPUTE=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU_NAME" ]; then
        log "GPU: ${GPU_NAME} (sm_${GPU_COMPUTE/./})"
    fi
else
    warn "No NVIDIA GPU detected — AINode will run in CPU mode"
fi

# Detect CUDA version
NVCC_PATH=""
for p in /usr/local/cuda/bin/nvcc /usr/local/cuda-13.0/bin/nvcc /usr/local/cuda-12.*/bin/nvcc; do
    if [ -x "$p" ]; then
        NVCC_PATH="$p"
        break
    fi
done

if [ -n "$NVCC_PATH" ]; then
    CUDA_VER=$("$NVCC_PATH" --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    log "CUDA ${CUDA_VER} (${NVCC_PATH})"
    export PATH="$(dirname "$NVCC_PATH"):$PATH"
    export CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
elif command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    log "CUDA ${CUDA_VER}"
fi

# Determine engine strategy:
#   - GB10 / CUDA 13 → Docker container (pip vLLM wheels don't target CUDA 13)
#   - Everything else → pip vLLM
ENGINE_STRATEGY="pip"
if [[ "$GPU_NAME" =~ "GB10" ]] || [[ "$GPU_COMPUTE" == "12.1" ]] || [[ "$CUDA_MAJOR" == "13" ]]; then
    ENGINE_STRATEGY="docker"
    info "Blackwell GB10 / CUDA 13 detected — will use Docker-based vLLM"
fi

ARCH=$(uname -m)
log "Architecture: ${ARCH}"

# ── Create home + venv ────────────────────────────────────────────────────

mkdir -p "$AINODE_HOME"/{models,logs,datasets,training}

step "1/5" "Creating virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip setuptools wheel

# ── Install uv (fast package manager) ────────────────────────────────────

if ! command -v uv &>/dev/null; then
    step "2/5" "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null
    export PATH="$HOME/.local/bin:$PATH"
fi

# ── Install AINode ────────────────────────────────────────────────────────

step "3/5" "Installing AINode..."
pip install --quiet ainode 2>/dev/null || {
    # If not on PyPI yet, install from GitHub
    pip install --quiet "git+https://github.com/getainode/ainode.git"
}
log "AINode installed"

# ── Install inference engine ──────────────────────────────────────────────

if [ "$ENGINE_STRATEGY" = "docker" ]; then
    step "4/5" "Setting up Docker-based vLLM for Blackwell GB10..."

    if ! command -v docker &>/dev/null; then
        info "Docker not found — installing Docker..."
        curl -fsSL https://get.docker.com | sudo sh 2>&1 | tail -3 || warn "Docker install failed"
        sudo usermod -aG docker "$USER" 2>/dev/null || true
    fi

    if ! sudo docker info &>/dev/null; then
        info "Starting Docker..."
        sudo systemctl reset-failed docker.service 2>/dev/null || true
        sudo rm -rf /var/lib/docker/buildkit 2>/dev/null || true
        sudo systemctl start docker 2>&1 | tail -3 || warn "Could not start Docker — you may need to start it manually"
    fi

    if sudo docker info &>/dev/null; then
        log "Docker is running"
        info "Pulling vLLM container for GB10 (this may take several minutes)..."
        sudo docker pull scitrera/dgx-spark-vllm:0.17.0-t5 2>&1 | tail -3 || warn "Container pull failed — AINode will still install"
        log "vLLM container ready"
    else
        warn "Docker not available — AINode will install without inference engine"
        warn "You can install Docker later and run: ainode engine install"
    fi

else
    step "4/5" "Installing vLLM inference engine (pip)..."
    pip install --quiet vllm 2>&1 | tail -3 || warn "vLLM pip install failed — AINode will still install without it"
    VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null) && \
        log "vLLM ${VLLM_VER}" || warn "vLLM import failed — check logs"
fi

# ── PATH setup ────────────────────────────────────────────────────────────

step "5/5" "Finishing up..."

AINODE_BIN="$VENV_DIR/bin"
SHELL_RC=""
[ -f "$HOME/.bashrc" ] && SHELL_RC="$HOME/.bashrc"
[ -f "$HOME/.zshrc" ] && SHELL_RC="$HOME/.zshrc"

if [ -n "$SHELL_RC" ]; then
    if ! grep -q "ainode" "$SHELL_RC" 2>/dev/null; then
        echo "" >> "$SHELL_RC"
        echo "# AINode — https://ainode.dev" >> "$SHELL_RC"
        echo "export PATH=\"$AINODE_BIN:\$PATH\"" >> "$SHELL_RC"
        log "Added to PATH in ${SHELL_RC}"
    fi
fi

# ── Systemd service (optional) ────────────────────────────────────────────

echo ""
echo "    ────────────────────────────────────"

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
echo -e "    ${GREEN}  AINode v${AINODE_VERSION} installed successfully!${NC}"
echo -e "    ${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "    To start:"
echo ""
echo -e "      ${CYAN}source ${SHELL_RC:-~/.bashrc}${NC}"
echo -e "      ${CYAN}ainode start${NC}"
echo ""
echo "    Then open http://localhost:3000 in your browser."
echo ""
echo -e "    Powered by ${BLUE}argentos.ai${NC}"
echo ""
