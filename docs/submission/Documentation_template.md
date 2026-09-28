# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Cosmosapiens  
**Team Members:** Mansi Saradva (Institute of Technology, Nirma University, Ahmedabad); Koradiya Nand Sanatkumar (Institute of Technology, Nirma University); Luv Patel (Nirma University, Ahmedabad)  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve Source 2 / Source 3 records to Source 1 entities with a four-stage pipeline: reverse blocking (each S2/S3 record searches S1 records of its own country with char n-gram TF-IDF and learned state routing), label-free region routing for France (no training data), a learned pruner that cuts the pool to ~10-35 candidates per S1, a cross-fitted two-pass LightGBM matcher with ~75 features, cross-encoder rescoring of the 4% uncertain pairs (four fine-tuned multilingual MiniLM models blended with the GBM), and one-owner decoding. Final public leaderboard: 0.9806 (holdout F0.5: US 0.9896, India 0.9884). The key innovations came from studying how the (synthetic) data generator builds noise: look-alike businesses append a real new word to a name while true copies only distort existing words, and empty-address records are ambiguous exactly when their name is shared by several S1 entities. Both became language-independent features, which is what carried the model to France, a country with no training data.

---

## 2. Methodology

### 2.1 Problem Analysis

Verified on the training data:
- Each S2/S3 record belongs to at most one S1 entity (exact over 7.64M links); no cross-country matches.
- 5.6% of S1 entities are singletons; 3.46 matches per S1 on average (max 11).
- About 26% of S2/S3 records match nothing; about 3.4% have an empty address.
- India S2 names are 23.6% native script (Devanagari, Kannada, Telugu, Gurmukhi), S3 13.3%; S1 is 100% Latin.
- Address parts are shuffled ("Rougemont, NC, 506 Medford Oakley Road"; "Karnataka, Bangalore, F-701, ..."), so no positional parsing.
- House numbers in true matches change by a small set of operators: leading zeros (452 -> 0452, 3.3% of true pairs), one digit dropped or added (3.7%), a new unit number prepended while the original appears later (~5.7%), letter affixes (2.0%), one digit substituted or +-1/2 (1.7%).
- Among similar-name candidates, look-alike non-matches are built by appending a new word: "group", "holdings", "care", "public", "clinic", "east", "valley", ... occur thousands of times as extra words in non-matches and never in true matches, while true matches only carry distorted versions of existing words (misspelled legal words such as "limitet"/"piraivet", honorifics, ".com").
- The same holds in France with French words ("Judo Comite SAS" at 13 Avenue Heurteau vs "Judo Comite France SAS" / "Judo Comite Groupe" at 20 Avenue Heurteau).

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned pruner + two-pass gradient-boosted classifier + one-owner decoding  
**Core Innovation:** language-independent features derived from the generator's noise operators (novel-extra-word, house-number operator relation, name-crowd ambiguity), validated with honest full-density cross-fitting and leave-one-country-out (LOCO) as a stand-in for the unseen country.

Validation design: S1 entities are split by md5(entity_id) into 10% holdout, 5% tune, 85% train. Blocking always runs with all training records (no thinning), so holdout entities face the same competition density as test entities. Every pair gets an out-of-sample probability (3-fold cross-fitting over training entities), thresholds and decoding parameters are tuned on the tune fold only, and every change was kept only if the paired bootstrap interval (1000 resamples over S1 entities) of the decision score S = 0.383 F_US + 0.468 F_India + 0.150 LOCO_avg excluded zero.

---

## 3. Candidate Generation (Blocking)

- **Normalization:** anyascii transliteration, casefolding, alias split (dba/aka/...), legal forms in the last three tokens (any order) moved to a legal class, honorifics, landmark phrases ("near X") moved aside, house number / postcode / other number extraction, France-only rules (street-position r/q -> rue/quai, bis/ter, cedex, leading SARL/SAS/Ets). Learned from training matches only: 23 India / 68 US part rewrites (mh -> maharashtra, texas -> tx, transliterated state names), 27 / 91 word rewrites, and a 531-entry native-script word dictionary (support >= 20, precision >= 0.9).
- **Blocking keys used:** reverse direction (each S2/S3 record queries S1 of its country) with four char TF-IDF indexes via sparse_dot_topn: full name (top 5; 30 for empty-address queries), name core + address (top 10), address only (top 10), consonant skeleton for non-Latin names (top 5). Learned state routing (from training matches, 99.7% coverage) restricts each query to its routed states plus key-less S1. Countries without training data get label-free routing learned from their own addresses: each S1's most frequent address part is its region key, a place part (city, department) maps to a region when >= 98% of its co-occurrences in S1 records name that region (extended once through query records), and a query searches only the regions of its places. A self-check routes every S1's own places like a query; routing is used only if fewer than 0.2% lead to another region. Checked against labels on the training countries treated as unseen: India loses 0.04% of true pairs at a 7.4x smaller index (self-check 0.05%), while US (self-check 0.61%, city names competing with states) is correctly left unrouted. On France test: 3 region keys, self-check 0.0, 97% of queries routed (the rest have empty addresses), average index 259k -> 93k S1.
- **Candidate pairs generated:** 34.8M on test after pruning: 9.8 per S1 (US), 23.8 (India), 34.9 (France; 38.7 before label-free routing). Reduction versus all same-country S1 x S2/S3 pairs: > 99.99%.
- **How you ensured true matches were not lost:** blocking pair recall on the full-density holdout is 0.987 (India) / 0.994 (US) before pruning; recall by category is monitored (empty address, non-Latin, scrambled, typo). A cross-fitted LightGBM pruner on blocking signals keeps each training country's own cutoff at a 0.1% recall-loss budget (the least strict cutoff for countries without training data).

---

## 4. Matching Model

**Features used:**
- Name features: ratio, token-set, token-sort, partial, Jaro-Winkler on the legal-free core; full-name token set; best alias match; Jaccard; first/last token equality; acronym; legal-class state; consonant-skeleton ratio; IDF-weighted overlap and one-sided IDF; name core minus own-address tokens (removes city names in names such as "Bordeaux Maison").
- Address features: ratio, token-set, partial, Jaccard, postcode state, number-token Jaccard, landmark state, same-street-with-different-house-number state, IDF overlap.
- Operator features: house-number relation class (equal, leading zeros, digit dropped/added, substituted, +-1/2, permuted, prefix, different, missing), digit edit distance, containment of each house number in the other address; novel extra words (count on each side; an extra word is novel unless it is a Jaro-Winkler variant of a word on the other side or of a legal word); name-crowd features (same-name S1 count in the country, same-name candidates of the query, rank and margin among them).
- Other: blocking scores and ranks per index, pruner score, candidate counts per query and per S1, cross-source corroboration, and a second pass with sibling-evidence features computed from first-pass probabilities (support from the S1's other strong candidates, query margin, S1 rank/sum/count/max).

**Synthetic next-door look-alikes:** test holds 1.6-1.9x more same-name, same-street candidates with a nearby different house number than the training data (label-free shift study; an EM base-rate estimate shows no prior shift for US/India, so the shift is in the covariates). For ~0.6 (US) / 0.2 (India) sampled true pairs per S1, a copy of the S2/S3 record with only the house number changed (moved by 1-25, one digit substituted, +-1/2, or two digits swapped, the operators seen in test) and in 25% of copies a sibling word appended to the name (words learned from training non-matches) is added as a NON-match of that S1. Training, tune and holdout S1 all receive copies, so the holdout becomes a test-like holdout on which models trained with and without synthetic data are compared. **Outcome (not in the final model):** on that test-like holdout the synthetic training gained +0.016 F0.5, but the paired leaderboard submission scored 0.9734 against 0.97528 for the same pipeline without it: our generated look-alikes did not match the real ones well enough, so the final model keeps the house-number gap features plus a look-alike logit shift of 2.5 instead.

**Cross-encoder rescoring of uncertain pairs:** pairs whose calibrated GBM probability is in [0.02, 0.98] (4.1% of pairs; ~27% of them true) are rescored by `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, ~118M parameters) with a fresh one-logit head, fine-tuned for one epoch on 500k training-fold pairs (400k from the band, 100k confident pairs) of raw "name | address" text, fp16, AdamW lr 3e-5, batch 64, max length 128 (28.5 min on a Quadro GP100). A logistic regression on [logit p_gbm, cross-encoder logit] fitted on the tune-fold band replaces the probability inside the band; calibration and decoding are re-tuned on the tune fold. Holdout: F_US 0.9868 -> 0.9888 (+0.0020, CI [0.0017, 0.0022]), F_India 0.9837 -> 0.9872 (+0.0035, CI [0.0032, 0.0039]); it halves singleton false merges (0.0016 -> 0.0008). The first sample kept every band positive (61% positive), so a second model (ce2) was fine-tuned identically on a natural-ratio band sample (26% positive, i.e. mostly hard negatives); blending both [logit p_gbm, ce, ce2] adds US +0.0003 [0.0001, 0.0004] and India +0.0002 [0.0000, 0.0003] over the single cross-encoder (holdout F_US 0.9890, F_India 0.9874; leaderboard 0.9795). A third model (ce3) fine-tuned the same way on a 1.3M natural-ratio sample covering the whole training-fold band (72 min) adds US +0.0004 [0.0003, 0.0006] and India +0.0008 [0.0006, 0.0010] over the two-model blend (final holdout: F_US 0.9894, F_India 0.9882; the tune fold then prefers a plain threshold 0.72 over expected-F0.5). A fourth model (ce4: 1.0M natural-ratio pairs, another seed, candidate text first) adds US +0.0002 [0.0001, 0.0003] and India +0.0003 [0.0001, 0.0004] (holdout F_US 0.9896, F_India 0.9884; expected-F0.5 decoding, alpha 1.5). Widening the band to [0.005, 0.995] added nothing (US -0.0000, India +0.0001, not significant).

**Model type:** LightGBM (binary, 127 leaves, feature and bagging fraction 0.8), 3-fold cross-fitted over training entities with early stopping on the tune fold, two passes; isotonic calibration on the tune fold.  
**Threshold selection method:** one-owner decoding (each S2/S3 record goes only to its highest-probability S1), then the better of a probability threshold and per-entity expected-F0.5 selection, chosen on the tune fold (final: expected-F0.5, parameter 0.75). Predict-time settings: raw logit - 2.5 for candidates whose house numbers are both known and different (look-alike shift), and raw logit - 1.5 for countries without training data (France).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** holdout (full density, 10% of training S1 entities) F_US 0.9868 / F_India 0.9837; public leaderboard **0.9806** (final: GBM + four cross-encoders; holdout F_US 0.9896 / F_India 0.9884). A probe submission with France rows emptied puts test US+India at about 0.98 and France at about 0.94 (France is ~15% of the score).
- **Holdout loss decomposition (1 - F0.5 = 0.0144):** true matches rejected 0.0074, blocking misses 0.0032, false merges of singletons 0.0016, wrong accepts 0.0021.
- **Common false positives (wrong merges):** empty-address records assigned to the wrong one of several same-name S1 entities; sibling businesses (same base name plus a word, other house number on the same street).
- **Common false negatives (missed matches):** empty-address records, true matches whose house number changed by a digit operator, scrambled trade names that match only on address.

Leaderboard history: 0.763 (baseline) -> 0.938 (v2 M1) -> 0.952 (learned rewrites, native-script dictionary, address and skeleton indexes) -> 0.960 (second pass, house features, calibration) -> 0.9678 (novel-extra-word features) -> 0.9726 (look-alike shift 2.5) -> 0.97525 (house-number gap features) -> 0.975281 (label-free France routing) -> 0.97532 (France unseen-country shift 1.5) -> 0.9795 (cross-encoder rescoring of uncertain pairs, two models) -> 0.980176 (third cross-encoder on the whole band) -> **0.9806** (fourth cross-encoder, final).

---

## 6. Conclusion

Honest full-density validation plus features that describe how the noise is generated, rather than which words appear, gave most of the gains; they were also what transferred to France without any French labels. The largest remaining error is intrinsic ambiguity of name-only (empty-address) records among same-name businesses.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`: `src/v2/` holds one module per stage (prepare, learn, blocking, prune, features, house, novel, crowd, sibling, train, decode, predict) with a `python -m src.v2.<stage>` entry point each; `scripts/run_final.sh` runs the full pipeline from the raw TSVs to `output/matching_results.tsv` and `output/candidate_pairs.tsv` and validates them. See README.md for commands, hardware and runtimes.

### B. Additional Results

| Change | Holdout F_US / F_India | Leaderboard |
|---|---|---|
| v2 M1: reverse blocking, cross-fit pruner, 36 features | 0.9615 / 0.9471 | 0.938 |
| M2: learned rewrites, native-script dictionary, address + skeleton indexes | 0.9680 / 0.9624 | 0.952 |
| M3: second pass, house-number operators, calibration, expected-F decoding | 0.9844 / 0.9821 | 0.960 |
| + novel-extra-word features | 0.9858 / 0.9834 | 0.9678 |
| + look-alike logit shift 2.5 (predict time) | same model | 0.9726 |
| + house-number gap features | 0.9868 / 0.9837 | 0.97525 |
| + label-free France routing (test blocking) | same model | 0.975281 |
| + unseen-country shift 1.5 (France only) | same model | **0.97532** |
| + cross-encoder rescoring of uncertain pairs (ce) | 0.9888 / 0.9872 | not uploaded |
| + second cross-encoder (ce2, natural-ratio sample), blend of both | 0.9890 / 0.9874 | 0.9795 |
| + third cross-encoder (ce3, whole band, 1.3M pairs), blend of three | 0.9894 / 0.9882 | 0.980176 |
| + fourth cross-encoder (ce4, 1.0M pairs, seed 7, pair order swapped), blend of four | 0.9896 / 0.9884 | **0.9806** |
| tried: synthetic look-alike training (test-like holdout +0.016) | 0.9868 / 0.9839 on the test-like holdout | 0.9734 (rejected) |
| tried: seed ensemble of the synthetic model | +0.0003 / +0.0001 on the test-like holdout | not kept |

**External data:** none. Only the provided files are used; TF-IDF vocabularies and IDF tables for a country without training text (France) are fitted on its own unlabelled test text. No APIs, geocoding or registries; one pre-trained model (the Apache-2.0 multilingual MiniLM above, fine-tuned only on the provided training data). No leaderboard information enters model training. Two choices did use public-leaderboard feedback and are reported as such: a logit shift of 2.5 for candidates with a known but different house number was taken from a teammate's leaderboard probes (the final submission uses it), the unseen-country shift of 1.5 was chosen by one France-only paired submission (0.97532 vs 0.975281), one probe submission with France rows left empty measured the France and US+India shares of the leaderboard score, and the choice between a few France variants (for example, predicting France with the agreement of two model families) was made with a small number of paired leaderboard submissions that differed only in France rows.
