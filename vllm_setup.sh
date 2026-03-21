#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  vllm_setup.sh  |  vLLM-Omni Setup for Wan 2.2 on RunPod
#  Automates system packages → vLLM installation → model download.
#
#  Usage: bash vllm_setup.sh
# ─────────────────────────────────────────────────────────────────
set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

step() { echo -e "\n${CYAN}▶ $1${NC}"; }
ok()   { echo -e "${GREEN}  ✓ $1${NC}"; }
fail() { echo -e "${RED}  ✗ $1${NC}"; exit 1; }

echo ""
echo "══════════════════════════════════════════════════"
echo "   WAN 2.2 via vLLM-Omni  |  Setup Script"
echo "══════════════════════════════════════════════════"
echo ""

# ── STEP 1: Verify GPU ────────────────────────────────────────────
step "1/5  Verifying GPU..."
if ! nvidia-smi > /dev/null 2>&1; then
    fail "nvidia-smi failed. GPU not available."
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
ok "GPU: $GPU_NAME  |  VRAM: $VRAM"

# ── STEP 2: System packages ───────────────────────────────────────
step "2/5  Installing system packages..."
apt-get update -q
apt-get install -y --no-install-recommends git wget curl ffmpeg file git-lfs > /dev/null 2>&1
git lfs install --skip-smudge > /dev/null 2>&1
ok "System packages installed"

# ── STEP 3: Install vLLM-Omni ─────────────────────────────────────
step "3/5  Installing vLLM-Omni Engine..."
pip install --upgrade pip setuptools wheel -q

# Installing torchvision and torchaudio alongside it forces pip's resolver 
# to upgrade them to match the new PyTorch version vLLM installs, hiding the red ERROR block.
pip install vllm vllm-omni torchvision torchaudio notebook -U
ok "vLLM-Omni installed"

# ── STEP 4: Download model weights ────────────────────────────────
# Removed! You DO NOT need to manual download weights here anymore!
# The vllm_omni engine will automatically fetch blocks from the HuggingFace Hub 
# directly into your standard '~/.cache/huggingface/hub' exactly when you run the model.

# ── Done ─────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo -e "${GREEN}   vLLM-Omni Setup complete!${NC}"
echo "══════════════════════════════════════════════════"
echo ""
echo "  Run inference using the highly-optimized Python script:"
echo ""
echo "    # Text-to-Video (14B MoE on 1 GPU)"
echo "    python3 generate_vllm.py --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \\"
echo "      --prompt \"A cinematic shot of a cyberpunk city\""
echo ""
echo "    # Text-to-Video (14B MoE on 2 GPUs via Tensor Parallelism)"
echo "    python3 generate_vllm.py --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \\"
echo "      --prompt \"A cinematic shot of a cyberpunk city\" --tensor_parallel_size 2"
echo ""
python3 cmd_run.py