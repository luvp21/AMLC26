# Approach document — Amazon ML Challenge 2026

Required artefact for the best submission (see `docs/CHALLENGE_RULES.md`): 1–2 pages,
ML approach + models + experiments + conclusion. Fill this in as you go during the
sprint, not from memory at hour 70 — update each section right after the step it
describes.

## 1. Problem framing

- Task type: (regression / classification / entity-value extraction / other)
- Modality: (text-only / image-only / text+image)
- Target column and its distribution (skew, range, missingness)
- Official metric (paste the exact formula once announced, and its `src/metrics.py` implementation)

## 2. Data insights (from EDA)

- Dataset size (train/test rows), missingness patterns
- Key features that correlated with the target during EDA
- Any label/target quirks that shaped preprocessing (outliers dropped, unit normalization, etc.)

## 3. Models used

| Model | Features | Validation score | Notes |
|---|---|---|---|
| GBM baseline (LightGBM/XGBoost/CatBoost) | | | day-1 submission |
| Fine-tuned encoder / VLM | | | |
| Final ensemble | | | |

## 4. Feature engineering

- Text: TF-IDF vs sentence-embedding features used, and why
- Image: CLIP embeddings / OCR fallback, and why
- Any domain-specific features engineered from the raw fields

## 5. Validation strategy

- CV scheme (k-fold, stratification, holdout size)
- Train/val/test split sizes
- Any leakage risks checked (near-duplicate rows, target leaking into a feature)

## 6. Post-processing

- Target transform used (log1p? clip?) and why
- Any snapping/bias-correction tricks applied (see `src/postprocess.py`) and whether they helped on OOF

## 7. Ensembling

- Blend weights or stacking method used (`src/models/ensemble.py`)
- OOF score before vs after ensembling

## 8. Results

- Final public leaderboard score
- Final OOF/CV score
- Gap between the two (overfitting check)

## 9. What we'd try next with more time

- One or two honest next steps — shows judgment even if you didn't get to them.
