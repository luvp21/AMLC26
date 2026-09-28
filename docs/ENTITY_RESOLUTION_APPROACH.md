# Entity Resolution — Approach & Validation Notes

Working design doc for the 2026 challenge (see `docs/PROBLEM_STATEMENT.md`). This
captures the research that grounded the architecture and the empirical validation
against real training data — not just theory. Feeds `Documentation_template.md` for
the final submission.

## Data reality check (not just the spec text)

Measured directly from `data_set/student_resource/dataset/`:

| | Source 1 (train/test) | Source 2 | Source 3 |
|---|---|---|---|
| Rows | 2,206,821 / 1,732,545 | 5,034,617 / 4,887,274 | 5,285,604 / 5,082,317 |
| File size | 201MB / 167MB | 467MB / 486MB | 481MB / 483MB |

Ground truth match-count distribution (2,206,821 Source 1 training entities): only
**5.6% are true singletons** (123,247) — most entities have 2-5 matches (mean ~3-4).
Country split: train is US (1.32M) / India (0.88M) only; test adds **France (259,452
rows, ~15% of test Source 1)** — a substantial fraction, not an edge case, confirming
the pipeline must genuinely generalize rather than special-case two countries.

**Critical, non-obvious finding**: Source 1 names are always Latin/romanized script.
Source 2 sometimes has **full Devanagari business names** (e.g.
`राम मार्केटिंग प्राइवेट लिमिटेड`); Source 3 has Tamil/Kannada tokens mixed into
otherwise-Latin text. Pure string-similarity matching cannot bridge scripts without
transliteration first.

## Research cross-check

- **Blocking-then-classify** is the standard architecture for this problem shape —
  confirmed by both academic survey ([Papadakis et al., "Blocking and Filtering
  Techniques for Entity Resolution: A Survey," ACM Computing Surveys 2020](https://dl.acm.org/doi/abs/10.1145/3377455))
  and production blog accounts of company-name matching at multi-million-record scale
  ([Tilores](https://tilores.io/content/fuzzy-matching-explained)).
- **`recordlinkage` (Python Record Linkage Toolkit) ruled out** for the blocking
  engine: its own docs state it's "developed with experimenting in first place and
  performance on the second place... for research and linking of small or medium
  sized files" — a direct admission it isn't built for our 2-5M-row scale.
- **Splink** ([moj-analytical-services/splink](https://github.com/moj-analytical-services/splink),
  MIT, Fellegi-Sunter model) is the recognized production-grade tool for exactly this
  multi-source linkage shape (1M records in <2 min via DuckDB, 100M+ via Spark) —
  documented as the fallback if our hand-rolled blocking's recall ceiling proves
  insufficient, not used by default because its EM-based approach is designed for
  when you *don't* have labels, and we do.
- **`cleanco`** ([psolin/cleanco](https://github.com/psolin/cleanco)) for legal-suffix
  stripping (Ltd/Corp/LLC/Pvt, infers likely country from suffix) instead of a
  hand-curated list.
- **`anyascii`** ([anyascii/anyascii](https://github.com/anyascii/anyascii), ISC
  license) for script transliteration — confirmed better choice than `unidecode`
  (GPL-licensed, license risk for submitted code; also narrower character coverage).
- **`libpostal`** ([openvenues/libpostal](https://github.com/openvenues/libpostal)) —
  a real CRF-based international address parser (99.45% parse accuracy) — considered
  but **deferred**: needs a compiled C build and a ~2GB model download, and PARAM
  Shavak's network resets large binary downloads (confirmed separately, see
  `infra/param_shavak/README.md`). Regex-based address decomposition is the practical
  fallback; revisit only with spare time.
- **Ditto** ([Li et al., VLDB 2020](https://arxiv.org/abs/2004.00584)) — fine-tuning a
  small pretrained LM for the match/no-match pair task reaches 96.5% F1 on
  company-record matching in the paper's benchmark. Validates blocking-then-classify
  as correct even in an LLM-augmented setup (their own blocking step is still
  required); noted as a possible phase-2 ensemble signal, not the day-1 approach
  (avoids the MIT/Apache-2.0 + ≤8B model-license question entirely by defaulting to a
  classical GBM first).

## Empirical blocking validation (the real test)

Ran directly against real training data + real ground truth (not synthetic):
`notebooks/er_blocking_validation.py`, 3,000 sampled Source 1 entities, background
sample of 300k rows/source (all true-match rows guaranteed included so recall is
measured correctly — see the script for the exact caveat on why full 5M-row scale may
shift these numbers slightly).

**Name-token-only blocking** (first attempt):

| top_k | Recall | Avg candidates/entity |
|---|---|---|
| 10 | 69.3% | 9.9 |
| 20 | 72.6% | 19.6 |
| 50 | 74.5% | 47.9 |
| 100 | 76.5% | 94.1 |

Manually inspected 15 random misses at top_k=100 and found three concrete,
fixable causes, not random noise:
1. **Devanagari transliteration mismatch** (dominant cause): `anyascii("प्राइम")` →
   `"praim"`, not `"prime"` — phonetic romanization doesn't recover the original
   English loanword spelling. But the **address field often keeps city/area names in
   Latin script even when the name doesn't** (e.g. both records mention "GHAZIABAD").
2. **Concatenated domain-style names**: Source 3 sometimes stores
   `"patriciodrugsclinic.com"` for `"Patricio Drugs Clinic"` — one token, zero word
   overlap with the tokenized version.
3. **Genuine matches crowded out by ties**: candidates were ranked by raw shared-token
   *count*, so a true match sharing only one common token got buried among many
   unrelated records that coincidentally shared that same one token.

**Fix tested: block on name AND address tokens together** (implemented in
`src/entity_resolution/pipeline.py`):

| top_k | Recall | Avg candidates/entity | Zero-candidate entities |
|---|---|---|---|
| 20 | **89.4%** | 20.0 | 0 |
| 50 | **92.0%** | 50.0 | 0 |

+16-18 points of recall from one change, and it eliminated every zero-candidate
entity (was 2 at name-only). This is the default blocking strategy now.

**Not yet fixed** (documented, not addressed — deprioritized given time budget):
concatenated domain-names (#2) and count-tie crowding (#3) still cost some recall.
Both have known fixes — character n-gram blocking as a third signal, and ranking by
an actual similarity score instead of raw token count — worth doing if there's spare
time before the classifier stage, but the current 89-92% ceiling is a reasonable
foundation to build the classifier on rather than a blocker.

## Full-scale run (real 2.2M/5M/5M rows, on PARAM Shavak, 25 Sept)

The 300k-row-sample estimate above was explicitly flagged as optimistic — confirmed:

| top_k | Recall (full scale) | Recall (300k sample) |
|---|---|---|
| 30 | **83.2%** (6,355,156/7,638,365) | ~90-91% (interpolated) |

A real ~7-8 point drop from more competing false candidates at full corpus size, exactly
as predicted. Also measured for the first time: **candidate generation took ~4 hours**
even after parallelizing across 32 workers — worse than hoped, and the single biggest
time cost in the whole pipeline by far (normalize: ~3 min combined; index build: 82s;
feature computation on the resulting ~66M pairs: ~26 min). If there's time later, this
is the next thing worth optimizing (tighter `max_postings` cap, or reducing `top_k`) —
not done yet because the run needed to produce a result, not be perfect.

The first full-scale run crashed immediately after this point on a missing `lightgbm`
import (the initial `setup_env.sh` run most likely partially failed on that pip line —
plausibly the same class of network reset that hit the Hugging Face download during GPU
setup) before producing a trained classifier or final F_0.5. Since nothing was
checkpointed, that ~4.5 hours of work was lost. Checkpointing was added immediately
after (`runs/checkpoints/candidates.pkl`, `features.pkl`) so this can't happen again —
a re-run after fixing the missing dependency resumes straight past both expensive
stages.

### Final full-scale result (re-run, 25 Sept)

**Macro F_0.5 = 0.8244 at threshold=0.90** — 56,273,910 train pairs (5,397,678 positive),
9,930,720 validation pairs, trained in 218.6s. Meaningfully lower than the 300k-sample
estimate of 0.9287, consistent with the recall drop (83.1% full-scale vs ~90% sampled):
more real data means more coincidental near-miss candidates competing.

Worth naming precisely: at 83.1% blocking recall, a perfect classifier could reach at
best `F_0.5 = 1.25×1.0×0.831/(0.25×1.0+0.831) ≈ 0.961`. The actual 0.8244 is well below
that ceiling, meaning the gap isn't only blocking's fault — the classifier's precision at
full scale has real room to improve too, not just the recall-limiting issues already
documented above (concatenated domain names, count-tie crowding).

Feature importances (full scale): `addr_jaccard` (2026) and `name_jaccard` (1913) now
lead, ahead of `name_ratio` (1786) — consistent with the smaller-scale run's finding that
address-based signals carry real weight (compensating for the Devanagari
name-transliteration gap). `country_match` remains the least useful feature (48) even at
full scale — blocking already implicitly filters to same-country candidates via shared
tokens, so it adds little on top. The best threshold moved from 0.85 (sample) to **0.90**
(full scale) — more competing near-misses means the model needs higher confidence before
committing to a match, exactly what F_0.5's precision weighting predicts.

France (test-only country, zero training examples) smoke-test results: see the end of
`runs/full_scale_run.log` once that stage completes — it can only check the pipeline
produces sane candidate counts, not real accuracy, since no ground truth exists for test.

## What this means for the leaderboard score

A 90%-ish recall ceiling from blocking is the upper bound on what the classifier can
ever recover — it cannot invent matches blocking never surfaced. Since F_0.5 weights
precision 2x over recall, a ~90% recall ceiling combined with a classifier that
achieves high precision on the surfaced candidates should still score well: in the
best case (perfect precision, 90% recall) a matched entity scores
`F_0.5 = (1.25 × 1.0 × 0.9) / (0.25 × 1.0 + 0.9) ≈ 0.978`. The real, harder question is
whether the classifier can hit high precision — that depends on feature quality and
threshold tuning, not yet measured (next step). Singleton handling is a second,
separate risk: blocking may surface spurious candidates for true singletons, and the
classifier must correctly reject *all* of them to preserve that entity's 1.0 score —
worth explicitly checking once the classifier exists, not assuming.

## Architecture (as built and validated)

1. **Normalize** (`src/entity_resolution/normalize.py`): `anyascii` transliteration →
   lowercase → `cleanco` suffix strip (names only) → address abbreviation expansion
   (Rd→road, St→street, etc.) → punctuation strip → tokenize.
2. **Block** (`src/entity_resolution/block.py`, `pipeline.py`): inverted index on name
   AND address tokens, high-doc-frequency tokens purged before indexing, top-K
   ranking by IDF-weighted shared-token score (see below — was raw count, fixed).
3. **Features** (`src/entity_resolution/features.py`): `rapidfuzz` ratio/token-sort/
   token-set-ratio + Jaccard on name and address separately, country-exact-match
   flag, length deltas, shared-token counts — plus optional embedding-similarity
   features (see below).
4. **Classifier** (`src/entity_resolution/matcher.py`): LightGBM on candidate-pair
   features, positives from ground truth, negatives from unmatched blocked candidates.
5. **Threshold** (`matcher.tune_threshold`): swept on a held-out Source-1-entity split
   to directly maximize macro F_0.5 (`src/metrics.entity_resolution_f_score`).
6. **Output writer** (`notebooks/er_generate_submission.py`): `matching_results.tsv` /
   `candidate_pairs.tsv` in the exact required format, verified against the
   organizers' own `utils/validate_submission.py` (both with and without
   `--check-ids`) before every leaderboard submit.

## Fix: rank blocking candidates by IDF weight, not raw shared-token count

Prompted by leaderboard scores of 0.96-0.98 (ours: 0.8244) — at 83.1% blocking recall,
even a perfect classifier caps around F_0.5≈0.961, so teams scoring above that must have
recall well above ours. Diagnosed directly against the full-scale run's saved
`candidates.pkl` checkpoint (`notebooks/er_diagnose_recall_gap.py`, no need to redo the
~4hr candidate generation): of 1,288,558 total misses, a 200-miss sample broke down as
**97% "ranked out"** (present in the unrestricted candidate pool, but buried below the
top_k=30 cutoff — e.g. rank 156,717 of 182,002), 3% purged from the index, 0% zero
overlap. Root cause: ranking by raw shared-token count treats a shared "vision" the same
as a shared "no" — generic connector words create massive ties that bury genuinely
distinctive matches.

Fix: `block.compute_idf_weights` (token's own posting-list length is its document
frequency — no separate tracking needed) + `candidates_for_record` now sums IDF weight
per shared token instead of counting. Unit-tested against the exact failure pattern
found (a candidate sharing one rare token now outranks one sharing two generic tokens).
**Not yet retrained/revalidated at full scale** — the classifier in `features.pkl` was
trained on raw-count-ranked candidates, so `er_generate_submission.py` explicitly pins
`use_weights=False` to reproduce exactly the 0.8244-validated configuration for the
first leaderboard submission. The next full-scale run should use `use_weights=True`
(the new default) and retrain.

## Enhancement: embedding-similarity feature for cross-script names

The Devanagari transliteration gap (see above) can't be fixed by re-weighting tokens —
`anyascii("प्राइम")` → `"praim"` shares literally no characters worth counting with
`"Prime"`. Research confirmed multilingual sentence embeddings are the established
technique for exactly this cross-lingual entity-matching problem. Checked empirically
(not assumed) against real training-data name pairs, comparing two candidate models:

| model | true cross-script match similarity | unrelated-business similarity | separation |
|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 (~470MB) | 0.324 | 0.155 | ~2x |
| BAAI/bge-m3 (MIT, ~2.3GB, newer, benchmarks as SOTA on general retrieval) | 0.810-0.845 | 0.331-0.431 | ~2x |
| **LaBSE (~1.8GB)** | **0.885-0.933** | **0.148-0.160** | **~6x** |

LaBSE wins clearly despite *losing* to BGE-M3 on published general-purpose retrieval
benchmarks — a reminder that a model's benchmark ranking doesn't transfer automatically
to a specific niche task. LaBSE's training objective (bitext mining: finding
translation-equivalent sentence pairs across languages) is a much closer match to "is
this the same business name in two scripts" than BGE-M3's broader multi-functionality
training or a general paraphrase model's. Worth checking empirically rather than trusting
the leaderboard-of-the-week, which is exactly what happened here.

**Considered and not pursued**: `AI4Bharat/IndicXlit`, a small (11M param) model
purpose-built for literal Devanagari↔Roman transliteration — attractive because it could
fix this at the *blocking* (recall) level rather than only as a classifier (precision)
feature. Blocked by a real dependency problem: it requires `fairseq`, which is
unmaintained and fails to install under current pip dependency resolution (confirmed,
not assumed — `ai4bharat-transliteration`'s build fails on missing `fairseq/version.txt`,
and a direct `fairseq` install hits `ResolutionImpossible` on its own `omegaconf` pin).
`IndicTrans2` (AI4Bharat's newer, fairseq-free, HF-`transformers`-native alternative) was
also considered, but it's a full translation model (1B params, or a 200M distilled
variant) rather than literal transliteration — a much heavier integration (new toolkit
dependency, new architecture) for an unproven payoff on this specific loanword-romanization
task, given LaBSE already delivers a strong, low-integration-cost precision fix. Worth
revisiting with more time, not now.

Implementation (`src/entity_resolution/embeddings.py`, wired into
`features.build_full_feature_matrix` as two optional extra columns
`name_embedding_cosine`/`addr_embedding_cosine`): embeds each **unique** name/address
string once (not per candidate pair — millions of pairs share the same string) via
`sentence-transformers`, then does plain vectorized cosine similarity. This is a
classifier *feature* (improves precision on candidates that already survived blocking),
not a blocking signal — blocking's cross-script recall problem is already handled by
falling back to address tokens.

**Not yet integrated into a full-scale run.** Model downloaded and sanity-checked on the
laptop (PARAM Shavak's network resets large HF downloads — same issue as before).
Transfer with:
```bash
rsync -avzL ~/.cache/huggingface/ bce23159@10.1.19.16:~/.cache/huggingface/
```
Then use `HF_HUB_OFFLINE=1` on PARAM Shavak. Embedding ~12.5M unique strings will use
the GP100 GPU (fp16, confirmed working) — the first time this pipeline uses the GPU at
all, since blocking/classical-feature/LightGBM stages are all CPU-only by design.

## Further research: prior art on company-specific record linkage

[Gschwind et al., "Fast Record Linkage for Company Entities" (IBM Research, 2019)](https://arxiv.org/abs/1907.08667)
— directly on-topic (real-world enterprise company-name matching at scale). Findings and
how they relate to what's built here:

- **MinHash/LSH blocking on character bigrams**, not word tokens — a different blocking
  algorithm solving the same class of problem our IDF fix targets (their own example:
  "téléski" vs "teleksi" share almost no exact bigrams due to a typo+diacritic — same
  shape as our Devanagari near-miss problem, different cause). This is a genuine
  alternative architecture, not a tweak — a full rewrite of `block.py`. Not pursued: the
  IDF-weighted ranking fix already directly addresses the diagnosed 97%-of-misses
  "ranked out" problem measured on our actual data; switching blocking algorithms
  entirely on the strength of one paper, without a diagnosed gap it would close, isn't
  justified against the time cost of another architecture + a fresh full-scale
  validation. Documented here as the established alternative if the IDF fix turns out
  insufficient after retraining.
- **CRF-based "short name" extraction** — they train a sequence model to learn which
  words in a company name are the *discriminative* substring (e.g., "Cisco" out of
  "Cisco Systems, Inc.") vs. generic filler, reporting it measurably improves blocking
  efficiency and accuracy. This is conceptually the same goal as our IDF weighting
  (favor discriminative tokens, downweight generic ones) — good independent validation
  that this is the right general idea; our version reaches it via corpus statistics
  (cheap, no training data needed) rather than a trained CRF (needs labeled short-name
  data we don't have and isn't worth building for this).
- **Two cleaning steps we don't currently do**: merging split acronyms ("I.B.M." →
  "IBM") and merging split numbers ("10 20 30" → "102030"). Noted, not implemented —
  unlike the Devanagari/concatenated-domain-name patterns (found by directly inspecting
  real misses), there's no confirmed evidence yet that either pattern is common enough
  in this dataset to be worth the normalization-pipeline change. Worth checking if
  there's spare time, not assumed to matter.

## Further optimization attempts after the sparse-matrix win — two real negative results

After `fast_block.py`'s 10.8x speedup, checked two more established options empirically
(both are legitimate, published/maintained tools — worth knowing they didn't pan out
*here*, not that they're bad in general):

- **`bm25s`** ([arXiv:2407.03618](https://arxiv.org/abs/2407.03618), claims up to 500x
  speedup over standard BM25 implementations via "eager sparse scoring") — installing it
  **downgraded numpy and broke other unrelated packages** (jax, tensorflow, opencv) in
  this environment, and `bm25s` itself then failed to import due to its own `jax`
  dependency conflicting with the numpy version pip resolved. Uninstalled, numpy
  restored, verified the project's own test suite (44/44) still passes throughout.
  Not pursued further — the dependency conflict is a real, reproducible packaging
  problem, not a transient issue, and we already have a working, benchmarked, correct
  10.8x-faster solution that doesn't need it.
- **`BlockingPy`** ([arXiv:2504.04266](https://arxiv.org/abs/2504.04266), MIT-licensed,
  FAISS-backed ANN blocking purpose-built for entity resolution) — benchmarked directly
  against `fast_block.py` on the identical 300k-row sample: **614s vs. our 6.1s — about
  100x slower**, and its default call only returned the single nearest neighbor per
  query rather than a top-30 list (would need reconfiguring, and likely be slower still
  to get comparable coverage). A genuine surprise given it's a purpose-built ANN library
  and general FAISS-based search is usually fast — but general-purpose blocking
  frameworks carry overhead (their own text vectorization/processing pipeline) that a
  narrow, hand-tuned solution for this exact data shape doesn't pay. Uninstalled after
  testing; not adopted.

Also checked and deliberately not pursued: **CuPy/GPU-accelerated sparse matrix
multiplication** — research indicates GPU sparse acceleration pays off mainly for
matrices with >1M nonzeros or genuinely large dense workloads; our sparse matrices are
already fast enough on CPU (6.1s for the 300k-row sample) that the host-device transfer
overhead likely isn't worth it here, and PARAM Shavak's Pascal GPU (no tensor cores) is
better spent on the LaBSE embedding computation, which does clearly benefit from it.
**MinHash LSH** (`datasketch`) — documented in the literature as fast but prone to false
positives, a bad fit for a precision-weighted metric like ours; not tested, ruled out on
this basis alone.

**Net conclusion**: `fast_block.py`'s hand-built sparse TF-IDF approach remains the best
option found, by a wide margin, for this specific dataset and task shape. Worth having
checked properly rather than assumed a fancier/newer/more-cited tool would automatically
win — twice now (BGE-M3 vs LaBSE, and BlockingPy vs fast_block.py) the more famous option
lost to the simpler one on this actual data.
