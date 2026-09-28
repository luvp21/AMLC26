#!/usr/bin/env bash
# Run this ON the EC2 instance after connecting (Deep Learning AMI already has
# CUDA + a base PyTorch conda env — this adds the fine-tuning stack).
set -euo pipefail

source activate pytorch  # DLAMI's preinstalled conda env; adjust if the AMI names it differently

pip install -q -U transformers peft bitsandbytes accelerate datasets sentencepiece

# Optional: for VLM fine-tuning (2024-style entity extraction), install LLaMA-Factory
# git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git ~/LLaMA-Factory
# pip install -q -e "~/LLaMA-Factory[torch,bitsandbytes]"

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "GPU box ready. Sync your data into ~/mlchallange/data/ and run src/models/finetune_lora.py"
