#!/bin/bash
# AINode installer — https://ainode.dev
# Usage: curl -fsSL https://ainode.dev/install | bash
set -e

AINODE_VERSION="0.1.0"
AINODE_HOME="${AINODE_HOME:-$HOME/.ainode}"
VENV_DIR="$AINODE_HOME/venv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo -e "${BLUE}    ╔══════════════════════════════════════╗${NC}"
echo -e "${BLUE}    ║            A I N o d e               ║${NC}"
echo -e "${BLUE}    ║   Your local AI platform for NVIDIA  ║${NC}"
echo -e "${BLUE}    ╚══════════════════════════════════════╝${NC}"
echo ""
echo "    Installing AINode v${AINODE_VERSION}..."
echo ""

# Check OS
if [[ "$(uname)" != "Linux" ]]; then
    echo -e "${RED}    Error: AINode requires Linux.${NC}"
    echo "    macOS and Windows are not supported."
    exit 1
fi

# Check Python
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}    Error: Python 3 is required.${NC}"
    echo "    Install: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "    ${GREEN}✓${NC} Python ${PYTHON_VERSION}"

# Check NVIDIA GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$GPU_NAME" ]; then
        echo -e "    ${GREEN}✓${NC} GPU: ${GPU_NAME} (${GPU_MEM} MB)"
    fi
else
    echo -e "    ${YELLOW}!${NC} No NVIDIA GPU detected (nvidia-smi not found)"
    echo "    AINode will run in CPU mode."
fi

# Check CUDA
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo -e "    ${GREEN}✓${NC} CUDA ${CUDA_VER}"
fi

# Create AINode home
mkdir -p "$AINODE_HOME"/{models,logs}
echo -e "    ${GREEN}✓${NC} Home: ${AINODE_HOME}"

# Create virtual environment
echo ""
echo "    Creating virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install AINode
echo "    Installing AINode..."
pip install --quiet --upgrade pip
pip install --quiet ainode

# Install vLLM (the heavy dependency)
echo "    Installing vLLM inference engine (this may take a few minutes)..."
pip install --quiet vllm

echo ""
echo -e "    ${GREEN}✓${NC} AINode installed successfully!"
echo ""

# Add to PATH
AINODE_BIN="$VENV_DIR/bin"
SHELL_RC=""
if [ -f "$HOME/.bashrc" ]; then
    SHELL_RC="$HOME/.bashrc"
elif [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
fi

if [ -n "$SHELL_RC" ]; then
    if ! grep -q "ainode" "$SHELL_RC" 2>/dev/null; then
        echo "" >> "$SHELL_RC"
        echo "# AINode" >> "$SHELL_RC"
        echo "export PATH=\"$AINODE_BIN:\$PATH\"" >> "$SHELL_RC"
        echo -e "    ${GREEN}✓${NC} Added to PATH in ${SHELL_RC}"
    fi
fi

echo ""
echo "    ────────────────────────────────────"
echo ""
echo "    To start AINode:"
echo ""
echo "      source ${SHELL_RC}"
echo "      ainode start"
echo ""
echo "    Or run directly:"
echo ""
echo "      $AINODE_BIN/ainode start"
echo ""
echo -e "    Powered by ${BLUE}argentos.ai${NC}"
echo ""
