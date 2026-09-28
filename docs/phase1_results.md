# Phase 1 results: validation harness and error decomposition

Run on PARAM Shavak, 2026-09-25, `notebooks/phase1_harness.py` (32 workers).
Baseline candidates, features and model unchanged; this phase only measures.

## Headline numbers (full scale; subsample in brackets)

| Metric | ALL | US | India |
|---|---|---|---|
| Macro F0.5, threshold 0.90 tuned on tune fold | 0.8240 (0.8227) | 0.8563 (0.8542) | 0.7755 (0.7755) |
| Oracle on current candidates | 0.9220 (0.9218) | 0.9501 | 0.8797 |
| Singleton accuracy | 0.687 (0.692) | 0.679 | 0.700 |
| Predicted vs true match rate | 0.916 vs 0.944 | 0.936 vs 0.944 | 0.887 vs 0.944 |
| Blocking pair recall @30 | 0.831 | 0.878 | 0.760 |

- Test-mix reweighted (US 45% / India 55%): 0.8119.
- LOCO: US→India 0.6602, India→US 0.8000, average 0.7301 (subsample 0.7293).
- The subsample tracks full scale within 0.0013 on macro F0.5 and 0.001 on LOCO
  average, so Phase 2+ iterates on the subsample.

## Error decomposition (1 − actual, mean per entity)

| Component | ALL | US | India |
|---|---|---|---|
| Blocking loss (1 − oracle) | 0.0780 | 0.0499 | 0.1203 |
| FP on matched entities | 0.0449 | 0.0448 | 0.0450 |
| Missed retrieved matches | 0.0356 | 0.0311 | 0.0424 |
| Singleton false merges | 0.0175 | 0.0180 | 0.0168 |

Blocking is the largest loss and dominates India. Even perfect decisions on
the current candidates cap out at 0.922, so reaching 0.95 requires Phase 3.

## Missed (never retrieved) pairs: 193,290 in validation, 200 sampled per country

| Category | India | US |
|---|---|---|
| other | 92 | 104 |
| chain_crowded | 33 | 54 |
| indic_script | 59 | 0 |
| address_only | 8 | 24 |
| typo_key_token | 8 | 17 |
| name_very_different | 0 | 1 |

Observations from the examples:
- Many chain_crowded and "other" misses have identical or near-identical names
  ("Career Estates Private Limited", "Chennai Systems", "Sai Tech Limited
  Private" at the same plot number). The baseline index covers name tokens
  only and ranks by raw shared-token count. Any entity whose name tokens are
  shared by more than 30 records ties with all of them, and the tie cut is
  arbitrary. IDF weighting plus an address channel should fix this directly.
  This is a hypothesis; Phase 3's forward/reverse/union report will test it.
- indic_script is 30% of India misses: whole names are transliterated
  (हरि सिस्टम्स प्राइवेट लिमिटेड = Hari Systems Private Limited). This is the
  target for the consonant-skeleton channel.
- address_only S3 names are scrambled strings ("Tavozeph", "Yumakor") that
  share only the house or plot number and locality. Only an address channel
  can retrieve them.

## Test unsupervised checks (threshold 0.90)

| Country | Conflict rate | Assignment rate | S1 predicted-match rate |
|---|---|---|---|
| France | 0.183 | 0.489 | 0.876 |
| India | 0.061 | 0.463 | 0.894 |
| US | 0.024 | 0.563 | 0.948 |

- The France conflict rate is 3× India and 8× US. That is the France-specific
  signal.
- Assignment rates are low everywhere (46–56% vs 74% true in train). This is
  consistent with pair recall × precision (US val ≈ 0.80 × 0.74), so it isn't
  specific to France.

The France eyeball sample (50 entities) shows top candidates are usually the
true duplicates (p ≥ 0.98). The false positives are neighbours on the same
street whose names share the city token:
- "Bordeaux Maison", 59 bd Jules Simon, vs "Bordeaux Primaire", 58 bd Jules Simon: p = 0.91
- "Lège-Cap-Ferret Union" vs "Lège-Cap-Ferret Pharmacie", 8 vs 5 b rue du Belvedere: p = 0.99
- "BY Sportive", 2 rue Alfred de Vigny, vs "Acef Sportive", 59 rue Alfred de Vigny: p = 0.975

Causes seen in the normalized tokens:
1. Single-digit house numbers and letter suffixes are dropped by the len ≥ 2
   token filter ("2 rue ..." has no 2, "5 b rue ..." has no 5 or b). Fixed in Phase 2.
2. There is no house-number agreement feature. Planned for Phase 4.
3. French street abbreviations are not expanded: "R." / "R" (rue), "Q." (quai)
   and "Bd" (boulevard) are dropped or left as-is. Fixed in Phase 2.
4. French names are often "<city> <generic word>", so name similarity rewards
   the shared city token. Needs IDF-weighted name overlap and name-tokens-in-
   address features (Phase 4).
