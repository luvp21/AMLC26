# Data drop — what to do the moment the problem statement lands

1. Put the released files here: `data/train.csv`, `data/test.csv` (and
   `data/sample_submission.csv` if one is provided). This directory is
   gitignored — nothing here gets committed.
2. Open `notebooks/eda_starter.py` and rename `TARGET_COL` to the real target
   column. Run it top to bottom — target distribution, dtypes, missingness.
3. Implement the *exact* announced metric in `src/metrics.py` if it's not
   already one of `smape` / `mape_score` / `exact_match_f1`.
4. Get a submission in fast: `src/models/baseline_gbm.py` on TF-IDF/basic
   text stats (+ CLIP embeddings from `src/features/image.py` if images are
   involved) — this is the day-1 baseline, don't skip straight to fine-tuning.
5. Before submitting anything, run through `docs/SUBMISSION_CHECKLIST.md`.

`data/abo_proxy/` holds the pre-sprint proxy dataset (Amazon Berkeley
Objects) used to battle-test the pipeline before the real data existed — it's
not part of the actual challenge data, leave it as reference/regression-test
fixtures.
