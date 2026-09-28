# AWS GPU workflow (for the $200 credit)

## Current account status (checked 24 Sept 2026, hours before the sprint)

Account `<account-id>` (`vaani-dev`), region `us-east-1`. Setup done:

- `aws configure` — already done, credentials work.
- Budget alarm — created, alerts to `luvvvpatel@gmail.com` at 75%/90% of $200.
- EC2 key pair — created: `amazon-ml-challenge-2026`, private key at `~/.ssh/amazon-ml-challenge-2026.pem`.
- Security group — created: `sg-0d97df7aabf8dfb39`, allows inbound SSH (22) from
  `104.28.241.157/32` (your IP at setup time — **if your IP changes** (new
  network, VPN, hotspot), add a rule for the new IP or the dry-run/launch
  below will hang on connect: `aws ec2 authorize-security-group-ingress
  --group-id sg-0d97df7aabf8dfb39 --protocol tcp --port 22 --cidr
  <new-ip>/32 --region us-east-1`).

**⚠️ Blocker: GPU instance quota is 0.** Both the raw EC2 path
(`g5.xlarge`/`g4dn.xlarge` — quota code `L-DB2E81BA`, "Running On-Demand G
and VT instances") and *every* SageMaker GPU instance quota (checked
g4dn/g5/p3 across notebook/training/endpoint use) are 0 on this account. A
quota increase to 8 vCPUs was requested and is `PENDING` as a support case
(no Premium Support on this account, so it can't be expedited via API —
approval timing for GPU quota bumps on a new account is unpredictable,
sometimes minutes, sometimes 1–2 days). **Don't assume AWS GPU compute will
be available when the sprint starts.**

Practical plan given this:
- **PARAM Shavak (GP100) is the primary/only confirmed GPU right now** — see
  `infra/param_shavak/README.md`. Treat AWS GPU as a bonus if the quota
  clears mid-sprint, not the plan.
- CPU-only AWS still works fine (`Running On-Demand Standard instances`
  quota is 8, `ml.t3.medium` SageMaker notebook quota is 2) if you want a
  second CPU box beyond PARAM Shavak's 56 threads — unlikely to be needed.
- Check quota status anytime: `aws service-quotas
  list-requested-service-quota-change-history-by-quota --service-code ec2
  --quota-code L-DB2E81BA --region us-east-1`. If it's still `CASE_OPENED`
  when the sprint starts, proceed on PARAM Shavak alone for GPU work.

## Free credits — claim these before the sprint, not during it

- **$200 in AWS credits total**: $100 lands on signup, another $100 for completing 5 starter
  activities (e.g. launching an EC2 instance, creating an RDS database). Do the 5 activities
  in the prep window — don't discover this mid-sprint.
- **AWS Builder Center Student Rewards** (builder.aws.com, no credit card): up to **$579**
  extra value — 12 months of Skill Builder Premium ($449) for verified student status, plus
  $10/$20 AWS credits and a $100 cert voucher for earning 7/14/21 badges. Badges come from
  simple daily actions (sign-in, comments, articles). Start earning now — the $10/$20 credits
  arrive fast enough to use during the sprint itself.
- **Top 500 teams** get **+$100 credits** at the 48-hour mark of the challenge (automatic, no action needed).
- **SageMaker free tier** (independent of the $200): 250 hrs of `ml.t3.medium` notebooks, 50 hrs
  of `ml.m5.xlarge` training, 125 hrs of `ml.m5.xlarge` inference — all for ~2 months from account creation.
- Set a **billing alert** immediately after account creation, and use `us-east-1` for best compatibility.

**Budget math** (us-east-1 on-demand prices, check current pricing before you commit):
- `g4dn.xlarge` — 1x T4 16GB, ~$0.53/hr → **~375 hrs** of credit. Fine for QLoRA on 7B text models; tight for 7B vision-language models with large image batches.
- `g5.xlarge` — 1x A10G 24GB, ~$1.01/hr → **~198 hrs** of credit. Recommended default — matches what the 2024 winners needed for Qwen2-VL-7B QLoRA with headroom to spare.

Either instance type gives you far more hours than the 72-hour sprint needs — the real risk isn't running out of credit, it's **forgetting to stop the instance**. Treat `stop_gpu.sh` as mandatory after every session, not optional.

## Workflow

1. **Before the sprint (do this in the prep window, not during the 72 hours):**
   ```bash
   aws configure                     # set up credentials once
   bash infra/aws/budget_alarm.sh    # alert at 75%/90% of $200 so you don't get surprised
   ```
   Then do one dry-run launch + stop to confirm everything works end to end — see step 2–4 below — so there's zero unfamiliar friction once the clock starts.

2. **Launch** (spins up a Deep Learning AMI with PyTorch/CUDA preinstalled — no manual driver setup;
   will fail with a quota error until the GPU quota request above is approved):
   ```bash
   export KEY_NAME=amazon-ml-challenge-2026
   export SECURITY_GROUP=sg-0d97df7aabf8dfb39
   bash infra/aws/launch_gpu.sh
   ```
   Prints the instance ID and public IP when ready (~2-3 min boot).

3. **Connect and sync code:**
   ```bash
   ssh -i ~/.ssh/amazon-ml-challenge-2026.pem ubuntu@<public-ip>
   # on a separate terminal, push your repo up:
   rsync -avz --exclude data --exclude runs -e "ssh -i ~/.ssh/amazon-ml-challenge-2026.pem" \
       /home/luv/Projects/mlchallange/ ubuntu@<public-ip>:~/mlchallange/
   ```
   Then on the instance: `bash setup_gpu.sh` (installs the heavy pip stack from `requirements.txt`'s commented-out section) and run training.

4. **Stop when you're not actively training** — this is the step that protects your $200:
   ```bash
   bash infra/aws/stop_gpu.sh <instance-id>
   ```
   `stop` (not `terminate`) keeps your EBS volume so `launch_gpu.sh --resume <instance-id>` picks up where you left off without re-downloading anything.

## During the 72-hour sprint

- Day 1: keep the GPU box **stopped** while you build the CPU-side baseline (`src/models/baseline_gbm.py`) locally — no need to burn GPU hours on feature engineering.
- Day 2: start the instance, sync data, run fine-tuning (`src/models/finetune_lora.py` or LLaMA-Factory for a VLM task), checkpoint to the attached volume every epoch in case of interruption.
- Day 3: stop the GPU once you've exported final predictions; do ensembling/blending locally.

## Alternative: SageMaker Notebook Instance

The prep session (21 Sep, recording on Twitch — see `docs/CHALLENGE_RULES.md` for the
timeline) demoed training locally inside a SageMaker Notebook Instance instead of a raw EC2
box. It's a valid alternative if you hit EC2 quota issues or want zero driver/CUDA setup:

- SageMaker console → Notebook → Notebook instances → Create → instance type `ml.t3.medium`
  (free tier) → IAM role "Create a new role" (defaults) → Open JupyterLab.
- `sagemaker.Session()` + `get_execution_role()` + `session.default_bucket()` gives you a
  ready S3 bucket and IAM role in 3 lines — no manual VPC/IAM wiring.
- **Local training** (`xgb.train(...)` inside the notebook, data already in memory) needs no
  S3 upload and is what this repo's day-1 baseline (`src/models/baseline_gbm.py`) matches.
  Use a **Training Job** (separate managed instance, pulls from S3) only if you need a bigger
  machine or a GPU than the notebook instance itself has.
- A SageMaker **Endpoint** (24/7 hosted inference) is not needed for this challenge — you
  submit a CSV of predictions, not a live API, and endpoints cost ~$0.12/hr even idle.
- Stop (don't delete) the notebook instance when done for the day — stopped instances don't
  bill, and your files persist. Delete only after the challenge ends.

This repo's scripts (`launch_gpu.sh`, `finetune_lora.py`) target the EC2 path above as the
default since it matches the fine-tuning workflow the 2024 winners used; the notebook path
is documented here as a fallback, not a second workflow to maintain in parallel.
