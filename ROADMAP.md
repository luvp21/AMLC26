# Roadmap — Amazon ML Challenge 2026

Solo, two compute sources: AWS ($200+ credit, g5.xlarge/A10G) and Nirma
University's **PARAM Shavak** GPU server (free, 1x Quadro GP100 16GB, 28c/56t
Xeon Platinum 8180, 250GB RAM, Ubuntu — see `infra/param_shavak/README.md`
for connection steps, the real hardware specs, and its Pascal-specific
caveats: no bf16, no tensor cores, 4-bit quantization support unverified
until tested, and only ~104GB free disk). Official contest rules and
timeline live in `docs/CHALLENGE_RULES.md`. Today: Sept 24, 2026.
Registration closed Sept 22. ML round: 25 Sept 12:00 AM → 27 Sept 11:59 PM
IST. Results Oct 2, finale Oct 7.

## Compute decision (finalized 24 Sept, hours before the sprint)

**PARAM Shavak is primary compute — CPU and GPU both — for the whole sprint. AWS is
backup only, and CPU-only until further notice.** Not a close call: AWS's GPU quota
(EC2 G/VT *and* every SageMaker GPU instance type) is 0 on this account, with a quota
increase pending as an unexpedited support case (no Premium Support) — no ETA, don't
plan around it showing up. PARAM Shavak's GP100 is therefore the only GPU that
actually exists for this sprint unless that changes. AWS's SageMaker free tier
(`ml.t3.medium` notebooks) is real and already usable, but it's CPU-only and PARAM
Shavak's 56 threads/250GB RAM already beats it for CPU work — so there's no reason to
split CPU work across two systems. AWS's role for now: sit on the $200+ credits, and
re-check the GPU quota once at the start of Day 2
(`aws service-quotas list-requested-service-quota-change-history-by-quota
--service-code ec2 --quota-code L-DB2E81BA --region us-east-1`). If it's cleared by
then, move the main fine-tune run to `g5.xlarge` (A10G, bf16, tensor cores — genuinely
faster) and keep PARAM Shavak running a second parallel experiment, per the Day 2 plan
below. If not, stay on PARAM Shavak alone.

## Status

| Item | State |
|---|---|
| Prep repo (`src/`, `infra/aws/`, tests) | Done — 11/11 tests passing |
| AWS account + budget alarm | Done — credentials, budget alarm, key pair, and an SSH security group are all set up (see `infra/aws/README.md`). GPU quota blocked — see the compute decision above. |
| **PARAM Shavak access** | **Verified live, 24 Sept.** GPU works, but the default `pip install torch` was broken (no Pascal kernels at all — fixed by pinning `torch==2.5.1+cu121`, now baked into `setup_env.sh`). fp16 matmul confirmed working. 4-bit QLoRA test was blocked by a firewall issue (large HF downloads get reset on this box, not a GPU problem) — **decision: default to `finetune_lora.py --dtype fp16 --no_4bit` for the sprint**, revisit 4-bit only if there's spare time. Full detail in `infra/param_shavak/README.md`. |

## Prep window (Sept 12 → Sept 24)

**Sept 12–14 — Foundations**
- [x] Repo scaffolded (`src/`, `infra/aws/`, tests green)
- [ ] Register for the challenge on Unstop *now* — don't leave it near the Sept 22 deadline
- [ ] `aws configure`, then `bash infra/aws/budget_alarm.sh` (75%/90% email alerts on the $200)
- [ ] One dry-run: `launch_gpu.sh` → confirm SSH works → `stop_gpu.sh`. Costs pennies, kills all first-time friction before it matters.
- [ ] Get PARAM Shavak access sorted (account, SSH, confirm whether SLURM is present: `which sbatch`) and run `infra/param_shavak/setup_env.sh`

**Sept 15–18 — Battle-test the modules on real proxy data**
Don't rehearse on toy data — the closest real analog to this competition's catalog data is Amazon's own [Amazon Berkeley Objects (ABO) dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) (product images + text + metadata, publicly released by Amazon for research). Pull a few thousand rows and:
- [ ] Run `src/features/text.py` + `src/models/baseline_gbm.py` end-to-end on a text target (e.g. predict a category or a numeric attribute) — confirms the CV harness, not just unit tests
- [ ] Run `src/features/image.py`'s CLIP embedder on a batch of ABO product images — confirms image loading/caching works before you need it under time pressure
- [ ] Build a small VLM SFT set with `src/data/loaders.build_vlm_sft_dataset` from ABO image+attribute pairs
- [ ] Fix whatever breaks — better now than hour 4 of the sprint
- [ ] Run `infra/param_shavak/test_gpu.py` on PARAM Shavak — this is the load-bearing check: it tells you empirically whether 4-bit QLoRA works on the GP100 or whether you need plain fp16 LoRA there instead. Don't skip this.

**Sept 19–20 — Full GPU dry run on both boxes (do this before the registration deadline, not after)**
- [ ] AWS: `launch_gpu.sh`, run `setup_gpu.sh`, install LLaMA-Factory
- [ ] Fine-tune Qwen2-VL-7B (or a smaller Qwen2-VL-2B if time-constrained) via QLoRA on the toy VLM set from the previous step — confirms the *exact* workflow the 2024 winners used works on your account, your AMI, your quota
- [ ] Confirm checkpoint + resume works (stop instance mid-training, resume, continue) — you will need this if AWS interrupts or you need to sleep
- [ ] `stop_gpu.sh` when done
- [ ] PARAM Shavak: run the same toy fine-tune (`src/models/finetune_lora.py --dtype fp16`) and time it against the AWS run, so you know which box to reach for under real time pressure
- [ ] Finalize registration if not already done (deadline: Sept 22, 11:59pm IST)

**Sept 21–23 — Timed rehearsal**
- [ ] Run a compressed 6–8 hour mock sprint solving *any* toy Kaggle regression/classification task end-to-end through this repo (baseline → fine-tune → ensemble → format submission), with a timer. The goal is finding friction, not winning the toy task.
- [x] Approach-document template drafted: `docs/APPROACH_TEMPLATE.md`
- [x] Submission-format checklist drafted: `docs/SUBMISSION_CHECKLIST.md`

**Sept 24 — Rest and final checks**
- [ ] AWS credentials, key pair, security group all still valid
- [ ] Repo committed and pushed somewhere you can pull from on any machine (this includes on PARAM Shavak — `git clone`/`git pull` from there once VPN+SSH is up)
- [x] Full pipeline verified against real proxy data (ABO dataset) right before the sprint: `src.features.text` + `src.models.baseline_gbm` + `src.metrics` + `src.postprocess` + `src.models.ensemble` all run end to end (`data/abo_proxy/battle_test_text_baseline.py`) — 11/11 unit tests pass, all `src/` modules byte-compile clean.
- [ ] Re-skim the winning approaches from 2023–2025 (BERT/RoBERTa min-blend, Qwen2-VL QLoRA, CLIP+DistilBERT fusion) once more
- [ ] Sleep. The sprint is 72 hours — you want to start it rested, not having crunched the night before.

## The 72-hour sprint (25 Sept 12:00 AM → 27 Sept 11:59 PM IST)

**Day 1 — Baseline (CPU, no GPU spend — run this on PARAM Shavak's 56 threads if available, it'll finish faster than laptop or a GPU instance you're paying for)**
1. Read the problem statement. Identify: modality (text/image/both), metric, target type.
2. Implement the exact metric in `src/metrics.py` if it isn't SMAPE/scaled-MAPE/F1 already.
3. EDA via `notebooks/eda_starter.py` — target distribution, missingness, skew.
4. Get *a* submission in via `src/models/baseline_gbm.py` on TF-IDF/basic-stats (+CLIP embeddings if images are involved) within the first 4–6 hours.
5. Decide target transform (log? clip?) from what the EDA showed.

**Day 2 — Fine-tune (both GPUs on, running different experiments in parallel)**
1. AWS: `launch_gpu.sh`, sync data — this is your primary run, especially for a VLM task (bf16 + tensor cores matter most there).
2. PARAM Shavak: kick off a second experiment here in parallel (`--dtype fp16`) — a different model, a hyperparameter variant, or the text-tower fine-tune while AWS handles the image tower. Two experiments running simultaneously beats one, solo or not.
3. Text task → `src/models/finetune_lora.py`. Image/VLM task → LLaMA-Factory + Qwen2-VL QLoRA, dataset built via `build_vlm_sft_dataset`.
4. Checkpoint every epoch on both boxes. If SMAPE/MAPE-scored, apply `postprocess.py` (clip to train range, consider snap-to-nearest-value, consider bias-correct-min if you train two models — which you now likely have, one per box).
5. Keep the GBM baseline's OOF predictions — you'll need them for ensembling regardless of how well the fine-tunes do.

**Day 3 — Ensemble, polish, ship**
1. `src/models/ensemble.py` — blend GBM + both fine-tuned models' OOF predictions, or stack with ridge.
2. Sanity-check final predictions against train distribution (no wild outliers).
3. Write the approach document from your Sept 21–23 template — fill in real numbers.
4. Submit with time to spare. `stop_gpu.sh` on AWS once done (PARAM Shavak has no per-hour cost to worry about, but free the GPU for others).
