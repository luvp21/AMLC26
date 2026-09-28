#!/usr/bin/env bash
# Launch (or resume) a GPU dev box on the Deep Learning AMI (PyTorch/CUDA preinstalled).
# Requires: `aws configure` already run, KEY_NAME and SECURITY_GROUP env vars set.
#
# Usage:
#   bash launch_gpu.sh                    # launch a fresh instance
#   bash launch_gpu.sh --resume i-0abcd   # start a previously-stopped instance
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
VOLUME_SIZE_GB="${VOLUME_SIZE_GB:-100}"

if [[ "${1:-}" == "--resume" ]]; then
  INSTANCE_ID="$2"
  echo "Resuming stopped instance $INSTANCE_ID ..."
  aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
  aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
  IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
        --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
  echo "Instance $INSTANCE_ID running at $IP"
  exit 0
fi

: "${KEY_NAME:?set KEY_NAME to your EC2 key pair name}"
: "${SECURITY_GROUP:?set SECURITY_GROUP to a sg-id allowing inbound SSH}"

echo "Looking up latest AWS Deep Learning AMI (PyTorch, Ubuntu)..."
AMI_ID=$(aws ssm get-parameters \
  --names /aws/service/deeplearning/ami/x86_64/pytorch-2.4-gpu-py311-cu124-ubuntu22.04/latest/ami-id \
  --region "$REGION" --query "Parameters[0].Value" --output text)

echo "AMI: $AMI_ID | Instance type: $INSTANCE_TYPE | Region: $REGION"

INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SECURITY_GROUP" \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE_GB,\"VolumeType\":\"gp3\"}}]" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=amazon-ml-challenge-2026}]' \
  --region "$REGION" \
  --query "Instances[0].InstanceId" --output text)

echo "Launched $INSTANCE_ID, waiting for it to boot..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
      --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

echo ""
echo "Instance ID: $INSTANCE_ID"
echo "Public IP:   $IP"
echo "SSH:         ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${IP}"
echo ""
echo "Save the instance ID — you'll need it for stop_gpu.sh and --resume."
