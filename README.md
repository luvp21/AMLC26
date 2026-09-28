# Amazon ML Challenge 2026 — prep repo

Reusable modules built ahead of the problem statement (drops Sept 25, 2026),
based on patterns across the 2022–2025 editions. See `ROADMAP.md` for the
day-by-day prep plan and the 72-hour sprint playbook, and
`docs/CHALLENGE_RULES.md` for the official contest rules, timeline, and
submission limits.

## Layout

```
docs/CHALLENGE_RULES.md       official contest rules, prize/eligibility/submission limits
docs/SUBMISSION_CHECKLIST.md  run through this before every submit
docs/APPROACH_TEMPLATE.md     the 1-2 page write-up required for the best submission
data/README.md                what to do the moment train.csv/test.csv land

src/
  metrics.py           SMAPE, scaled-MAPE, exact-match F1 (implement the real
                        one the moment the metric is announced)
  postprocess.py        log-transform, clipping, snap-to-train-value, bias-correct-min
  features/
    text.py             TF-IDF baseline + sentence-embedding features
    image.py             CLIP embeddings + OCR fallback
  models/
    baseline_gbm.py      day-1 LightGBM/XGBoost/CatBoost CV harness
    ensemble.py           blend-weight search + ridge stacking
    finetune_lora.py      QLoRA fine-tuning for a text encoder (regression/classification)
  data/
    loaders.py             CSV loading, splits, LLaMA-Factory-style VLM dataset builder

infra/aws/            launch/stop scripts + budget alarm for the $200 GPU credit
infra/param_shavak/   setup + diagnostics for the uni's PARAM Shavak GPU (see its own README for caveats)
notebooks/eda_starter.py   first-hour EDA template
tests/                 unit tests for metrics.py and postprocess.py — run `pytest`
```

## Compute

Two GPU sources, routed by workload — see `infra/param_shavak/README.md` for
the full breakdown:

| | AWS g5.xlarge | PARAM Shavak |
|---|---|---|
| GPU | A10G 24GB, tensor cores, bf16 | GP100 16GB, Pascal, **fp16 only** |
| Cost | ~$1/hr against $200 credit | Free, but shared/possibly contended |
| Best for | The main VLM QLoRA run | CPU-heavy work (56 threads/250GB RAM), a second GPU experiment in parallel |

`src/models/finetune_lora.py` takes `--dtype bf16|fp16` — use `fp16` on
PARAM Shavak, `bf16` on AWS.

## Quick start

```bash
pip install -r requirements.txt   # CPU stack, safe to run locally
pytest                            # should be green before you rely on any of this mid-sprint
```

For the GPU-heavy path (fine-tuning), see `infra/aws/README.md`.

## When the problem statement drops

1. Drop `train.csv`/`test.csv` into `data/`, update column names in `notebooks/eda_starter.py`.
2. Implement the *exact* announced metric in `src/metrics.py` if it's not already one of the three here.
3. Run the day-1 baseline (`src/models/baseline_gbm.py` on TF-IDF/CLIP features) to get *a* submission in early.
4. Move to the AWS box for fine-tuning (`src/models/finetune_lora.py` for text, LLaMA-Factory for a VLM task).
5. Blend everything with `src/models/ensemble.py` before the deadline.
