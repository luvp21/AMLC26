# Business Entity Resolution: reproducible pipeline

Resolves Source 2 / Source 3 records to Source 1 entities and writes
`output/matching_results.tsv` and `output/candidate_pairs.tsv` from the raw
training and test TSVs. `scripts/run_final.sh` reproduces the final submission with its defaults
(GBM `m3_gap` configuration + cross-encoder rescoring).

## Environment

- Python 3.12, Linux, CUDA GPU for the cross-encoder (tested on a Quadro GP100 16 GB, fp16; about 30 min
  fine-tuning per 500k pairs, ~15 min scoring per model). Everything else is CPU.
- `pip install -r requirements.txt` (all pinned; licenses MIT / BSD / Apache-2.0 / ISC).
- Tested on a 56-thread Xeon with 250 GB RAM (predict peaks at ~32 GB RSS; training needs more,
  well within 250 GB) and ~35 GB of free disk for intermediate tables. Use `V2_WORKERS` to set processes/threads (default 32).

## Data layout

```
data_set/student_resource/dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
data_set/student_resource/dataset/test/test_source{1,2,3}.tsv
data_set/student_resource/utils/validate_submission.py      # organisers' validator (optional)
```
Another location: `export V2_DATA_ROOT=/path/to/dataset`. Intermediate tables
go to `runs/v2/` (`V2_OUT_DIR` to change).

## Run end to end

```bash
export PYTHONPATH=.
nohup bash scripts/run_final.sh > runs/final.log 2>&1 &     # ~7-9 h with 32 workers
```

Stages (each one also runnable alone: `bash scripts/run_final.sh <stage> ...`):

| Stage | Module | What it does |
|---|---|---|
| `prepare_raw`, `learn`, `prepare` | `src/v2/prepare.py`, `src/v2/learn.py`, `src/v2/normalize.py` | normalization (anyascii, legal forms, aliases, house numbers, France-only rules); part/word rewrites and native-script dictionary learned from training matches; partition keys; md5 folds |
| `block_train` | `src/v2/blocking.py` | reverse blocking: each S2/S3 record queries S1 of its country, four char TF-IDF indexes (sparse_dot_topn); learns state routing from training matches |
| `prune_train`, `features_train`, `addons_train` | `src/v2/prune.py`, `src/v2/features.py`, `src/v2/house.py`, `src/v2/novel.py`, `src/v2/crowd.py` | cross-fitted pruner; pair features; house-number relation and gap, novel extra words, name crowd |
| `synth` | `src/v2/synth.py` | synthetic look-alikes; skipped by default (`FINAL_SYNTH=0`) |
| `train_model` | `src/v2/train.py`, `src/v2/sibling.py`, `src/v2/decode.py` | 3-fold cross-fitted LightGBM, sibling-evidence second pass, isotonic calibration and decoding tuned on the tune fold; holdout report |
| `block_test` ... `addons_test` | same modules, `--tag test` | test side; countries without training data (France) use label-free routing (`--unsup-routing`) |
| `predict` | `src/v2/predict.py` | GBM-only submission (cross-fit mean scores, look-alike shift 2.5, unseen-country shift 1.5, one-owner decoding) -> `runs/submissions/final_gbm/` |
| `ce_band_train`, `ce_train`, `ce_score_train`, `ce2_*`, `ce3_*`, `ce4_*`, `ce_blend` | `src/v2/ce.py` | cross-encoder rescoring of uncertain pairs (calibrated GBM p in [0.02, 0.98]): four `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` models (ce: 500k positives-first band sample, ce2: 500k natural-ratio band sample, ce3: 1.3M natural-ratio sample covering the whole band, ce4: 1.0M natural-ratio pairs with seed 7 and the pair order swapped), each fine-tuned one epoch on training-fold pairs of raw "name \| address" text (fp16, AdamW 3e-5, batch 64, max length 128); logistic blend of GBM and cross-encoder logits fitted on the tune fold; decoding re-tuned on the tune fold |
| `ce_band_test`, `ce_score_test`, `ce2_score_test`, `ce3_score_test`, `ce4_score_test`, `ce_apply` | `src/v2/ce.py` | test side with the same predict-time shifts -> `output/` (final) |

Final settings (defaults of `scripts/run_final.sh`): `FINAL_SYNTH=0`,
`FINAL_SHIFT=2.5`, `FINAL_UNSEEN_SHIFT=1.5`; matcher flags
`--max-rounds 2000 --lr 0.08 --no-loco --second-pass --house --novel --rules threshold,expected_f`.

## Outputs

- `output/matching_results.tsv`: one row per test S1 entity, comma-separated matched S2/S3 ids (empty for none).
- `output/candidate_pairs.tsv`: the exact candidate set scored by the matcher (after pruning).
- Also copied to `runs/submissions/final/` when the organisers' validator passes.
- Logs per stage in `runs/v2/<tag>/*.log` (resources at start and end of every stage);
  holdout scores and error decomposition in `runs/v2/train/train_final.log`.

## Determinism

Folds use md5 of entity ids; every sampler and LightGBM use seed 42. LightGBM
multi-threading can change the last digits of probabilities between machines,
which may flip a handful of borderline decisions; scores match to about 1e-4.

## Compliance

- No external data, APIs, geocoding or registries. One pre-trained model:
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, ~118M parameters),
  downloaded from the Hugging Face hub on first run and fine-tuned only on the provided training data.
  The fine-tuned weights (4 x ~470 MB) are not in the zip; `ce_train` ... `ce4_train` reproduce them. TF-IDF vocabularies and IDF tables for a country
  without training text (France) are fitted on its own unlabelled test text, and
  France routing is learned from its own unlabelled test addresses.
- Leaderboard feedback was used for two predict-time settings, both documented:
  the look-alike shift 2.5 (from a teammate's probes) and the unseen-country
  shift 1.5 (a France-only paired comparison, 0.97532 vs 0.975281). No
  leaderboard information enters model training.
