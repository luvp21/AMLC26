# Problem Statement — Business Entity Resolution Challenge (2026)

Dropped 25 Sept 2026, start of the 72-hour window. Preserved verbatim-in-substance so it
survives independent of the portal.

## Task

Given business records from 3 independent sources (`source1`/`source2`/`source3`), each with
noisy/inconsistent `business_name`, `business_address`, `country` fields and no shared
identifiers, determine which Source 2/3 records refer to the same real-world business as each
Source 1 record (the deduplicated reference source). A Source 1 entity may match zero, one, or
many records across Source 2 and Source 3.

This is entity resolution / record linkage, **not** the regression/classification/extraction
shape of 2023-2025 — the repo's existing `src/models/baseline_gbm.py` and `finetune_lora.py`
need a blocking + pairwise-matching architecture instead, see `docs/APPROACH_TEMPLATE.md` update
and the new modules under `src/` (blocking, pairwise features, entity matcher).

## Files

All `.tsv`, **must read with `sep="\t"`** — commas appear inside address and ID-list fields, so
reading without an explicit tab separator silently collapses everything into one column.

- `dataset/train/train_source{1,2,3}.tsv` — columns: `entity_id` (prefixed `S1-`/`S2-`/`S3-`),
  `business_name`, `business_address`, `country`.
- `dataset/train/train_ground_truth.tsv` — `source1_entity_id`, `matched_entity_ids`
  (comma-separated `S2-`/`S3-` IDs, empty for singletons).
- `dataset/test/test_source{1,2,3}.tsv` — same schema, no ground truth. Every Source 1 test
  entity must appear in the submission.
- `utils/validate_submission.py` — provided alongside the dataset (stdlib only). Run it before
  every submission:
  ```bash
  python3 utils/validate_submission.py --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv --test-dir dataset/test
  ```

## Critical: country is an open set

Training data covers US and India only. **Test data adds France**, unseen in training. Do not
hardcode, filter, or one-hot-encode the pipeline to `{US, India}` — every test entity, France
included, must get a prediction. Treat `country` as an arbitrary string feature (equality checks,
not a fixed vocabulary).

## Noise to expect

- **Names**: abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited), legal-suffix
  inconsistency, DBA/trade names, punctuation (`&` vs `and`), word-order transposition, typos.
- **Addresses**: abbreviations (Rd/Road, St/Street), transliteration variants, missing components
  (no PIN/state), landmark references ("Near SBI ATM"), municipal numbering, component reorder.

## Output format (both files go in `output/`)

**`matching_results.tsv`** — the only file scored on the leaderboard:
```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00003	
```

**`candidate_pairs.tsv`** — not scored, used to audit blocking quality (recall ceiling,
reduction ratio). This must be the *final* candidate set fed to the matching model for
inference — not an earlier, unfiltered blocking pass. Every ID in `matching_results.tsv` must
appear here (the validator flags a matched ID that was never a candidate as a pipeline bug).

Rules for both: exactly one row per Source 1 test entity, empty string (not omitted row) for no
matches, no duplicate IDs within a list or duplicate `source1_entity_id` rows, IDs must be
Source 2/3 IDs that exist in the test set (no self-matches to Source 1).

## Scoring: macro-averaged F_0.5

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```
Computed **per Source 1 entity**, then averaged across all Source 1 entities (macro, not
micro). **Singletons count**: a true singleton scores 1.0 for a correct empty prediction, 0.0
for any predicted match. Precision is weighted 2× over recall — false merges are punished harder
than missed matches. Implemented in `src/metrics.entity_resolution_f_score` (see worked example
there matching the spec's S1-00001 case: precision 2/3, recall 1.0 → F_0.5 = 0.714).

No ground truth on the test set — hold out a validation split from training (by Source 1 entity,
not by row, to avoid leaking a Source 1 entity's candidates across the split) and self-score with
the same metric.

## Hard constraints

- **No external data lookups. Ever.** No entity-resolution APIs, no government business-registry
  lookups, no geocoding APIs, no internet-sourced augmentation of any kind. This is enforced via
  submission-package review; violations are instant disqualification. This rule applies to me
  (the AI assistant) too — never fetch or suggest fetching external data for this task, even if
  it would obviously help matching accuracy.
- **Model license/size**: final model must be MIT or Apache-2.0 licensed and ≤8B parameters.
  (Rules out Llama-family models — non-OSI custom license. Qwen2/2.5 non-VL text models are
  Apache-2.0 if an LLM/embedding component is used at all; a classical TF-IDF+GBM pairwise
  classifier avoids the license question entirely and is probably sufficient here.)
- Max 5 leaderboard submissions/day (see `docs/CHALLENGE_RULES.md`) — validate locally with
  `utils/validate_submission.py` before every submit, don't burn one on a format rejection.

## Tips from the organizers

- Blocking/candidate-generation quality sets the recall ceiling — invest here first.
- String similarity features: Jaccard, Levenshtein, TF-IDF cosine on name/address.
- Country-specific address patterns matter, but the pipeline itself must generalize (see the
  open-set country note above).
- Correctly predicting "no match" on a singleton is worth a full 1.0 — don't neglect this.
