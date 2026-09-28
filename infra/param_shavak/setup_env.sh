#!/usr/bin/env bash
# Environment setup for PARAM Shavak (Nirma University GPU server — Ubuntu,
# Xeon Platinum 8180, Quadro GP100). User-level venv only: no sudo, no conda —
# this is a shared machine with a strict "no system-wide installs" policy and
# only ~104GB free disk, so we keep the footprint minimal.
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/venvs/ml}"

echo "Disk space check (only ~104GB is typically free on this box):"
df -h "$HOME"
echo ""

echo "Checking driver's CUDA version..."
nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv || {
  echo "nvidia-smi failed — either no GPU visible in this session or a driver issue."
  echo "This account can't fix driver issues (no sudo) — flag it, don't try to reinstall."
}

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating venv at $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install -q --upgrade pip

# IMPORTANT: pin this exact version. `pip install torch` (any --index-url, including
# cu118) currently resolves to a build with NO sm_60 (Pascal) kernels at all — confirmed
# live on this box: torch.cuda.is_available() reports True but every CUDA op fails with
# "no kernel image is available for execution on the device". torch==2.5.1+cu121 is
# confirmed working (fp16 matmul verified) and satisfies transformers' torch>=2.5
# requirement. No CUDA Toolkit / nvcc install needed — torch ships its own CUDA runtime.
# If this ever needs to change, verify with test_gpu.py's fp16 matmul check before
# trusting a new version — don't assume, the sm_60 drop can reappear silently.
pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers peft bitsandbytes accelerate datasets sentencepiece huggingface_hub

# CPU-side stack (for the baseline/feature-engineering work this box excels at)
pip install -q pandas numpy scikit-learn lightgbm xgboost catboost sentence-transformers Pillow

# Entity resolution (2026 task) — normalization + fuzzy matching, all CPU/pure-Python
pip install -q anyascii cleanco "rapidfuzz>=3.6" pyarrow

python - <<'PY'
import torch
print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("compute capability:", torch.cuda.get_device_capability(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
PY

echo ""
echo "Env ready at $VENV_DIR. Run: source $VENV_DIR/bin/activate && python test_gpu.py"
