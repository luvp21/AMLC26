#!/bin/bash
#SBATCH --job-name=amlc2026
#SBATCH --output=logs/%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# Template — only needed if `which sbatch` succeeds on PARAM Shavak (unlikely:
# it's a single workstation, not a cluster). If there's no scheduler, SSH in
# and run the same commands directly instead.
#
# Usage: sbatch slurm_job.sbatch python -m src.models.finetune_lora --model_name ... --train_csv ...

mkdir -p logs
source "$HOME/venvs/ml/bin/activate"

echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

"$@"
