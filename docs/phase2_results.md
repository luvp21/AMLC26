# Phase 2 results: exclusivity reference and normalization

Run on PARAM Shavak, 2026-09-25/26, `notebooks/phase2_sweep.py`. Raw reports are in
`runs/phase2/` (not committed). Rows are in `experiments_mtechcse-Precision-7920-Tower.csv`.

## 1. Record-level exclusivity reference (baseline model, no retraining; full scale)

Each S2/S3 record is kept only for its highest-probability S1 entity, then threshold 0.90.
Re-tuning on the tune fold picked 0.90 again.

| Variant | Macro F0.5 | US | India | FP on matched | Singleton FM | Missed retrieved |
|---|---|---|---|---|---|---|
| None (Phase 1) | 0.8240 | 0.8563 | 0.7755 | 0.0449 | 0.0175 | 0.0356 |
| Within-fold (only val entities compete) | 0.8256 | 0.8578 | 0.7773 | 0.0434 | 0.0172 | 0.0357 |
| All S1 entities compete (train fold scored out-of-fold, as on test) | **0.8309** | 0.8616 | 0.7847 | **0.0385** | 0.0163 | 0.0363 |

The within-fold number understates the effect: on test every S1 entity is scored at once,
which is the all-S1 row.

Test, before vs after exclusivity. Conflict rate after is 0 by construction, and the
assignment rate is unchanged.

| Candidate country | Conflict rate before | Predicted pairs removed | S1 predicted-match rate |
|---|---|---|---|
| France | 0.183 | 22.9% | 0.876 → 0.854 |
| India | 0.061 | 2.3% | 0.894 → 0.891 |
| US | 0.024 | 0.6% | 0.948 → 0.947 |

In the France eyeball sample, the pairs removed are the same-street neighbour false
positives seen in Phase 1: "Acef Sportive" (59 rue Alfred de Vigny) is removed from
"BY Sportive" (2 rue Alfred de Vigny), and "Bordeaux SARL Cie" (46 bd Jules Simon) from
"Bordeaux Maison" (59 bd Jules Simon).

## 2. Normalization sweep (subsample, each step on top of the kept set)

Reference: val 0.8227, LOCO average 0.7293.

| Step | Val | Δ | LOCO avg | Δ | Rule verdict |
|---|---|---|---|---|---|
| Fuzzy ratios on normalized tokens | 0.8166 | −0.0061 | 0.7440 | **+0.0147** | revert |
| Digit tokens + letters next to digits | 0.8268 | **+0.0041** | 0.7292 | −0.0001 | revert |
| Legal-token bag (last 3) | 0.8237 | +0.0010 | 0.7327 | +0.0034 | keep |
| Stopwords | 0.8242 | +0.0005 | 0.7313 | −0.0014 | revert |
| Landmarks | 0.8223 | −0.0014 | 0.7324 | −0.0003 | revert |
| Country abbreviations (France only) | 0.8237 | 0 | 0.7327 | 0 | keep (neutral by design) |
| Extracted fields (stored only) | 0.8237 | 0 | 0.7327 | 0 | keep (neutral by design) |

- Normalized-token ratios: US→India goes 0.6607 → 0.6962 and singleton accuracy goes
  0.69 → 0.79. In-distribution drops.
- Digit tokens: India goes 0.7755 → 0.7843.

Full-scale rerun of the kept set (legal_bag, country_abbrev, extract_fields):
- Val: 0.8240 → **0.8236** (US 0.8563 → 0.8563, India 0.7755 → 0.7744).
- LOCO: 0.7301 → 0.7299.

The subsample gain from the legal-token bag did not replicate at full scale.

Candidates are still the cached baseline set, so blocking recall is unchanged (0.8312).
Normalization reaches blocking only in Phase 3.
