#!/usr/bin/env bash
# Stop (not terminate) a GPU instance — keeps the EBS volume so you can resume
# with `launch_gpu.sh --resume <instance-id>` without losing anything.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
INSTANCE_ID="${1:?usage: stop_gpu.sh <instance-id>}"

aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
echo "Stopping $INSTANCE_ID ... (billing for compute stops once it's fully stopped; EBS storage still bills a small amount)"
aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID" --region "$REGION"
echo "Stopped."
