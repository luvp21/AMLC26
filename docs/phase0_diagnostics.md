# Phase 0 diagnostics (baseline = git tag `baseline`, leaderboard 0.763)

Sources: `notebooks/phase0_local.py` (raw data only) and `notebooks/phase0_remote.py`
(PARAM Shavak cached artifacts). Baseline reproduced exactly from cached features:
validation macro F0.5 = 0.8244 at threshold 0.90.

| # | Finding |
|---|---|
| 1 | True singleton rate 5.58% overall; India 5.59%, US 5.58%. |
| 2 | Matches per entity: 0 = 5.6%, 1 = 5.4%, 2 = 17.0%, 3 = 24.1%, 4-10 = 48.0%, >10 = 37 entities (max 11). 3.67 true pairs per matched entity. |
| 3 | **Zero** S2/S3 IDs appear in more than one S1 match list (7,638,365 IDs, each once). Exclusivity holds exactly. |
| 4 | **Zero** cross-country matches. Blocking can be grouped by country label. |
| 5 | Train S1 60% US / 40% India. Test S1 38.3% US / 46.8% India / 15.0% France. Test S2+S3 pools: India 4.72M, US 3.82M, France 1.43M. 74% of train S2/S3 records are matched to some S1; 26% are unmatched distractors. |
| 6 | Validation (threshold 0.90): US macro F0.5 0.8568 (pred-any 93.6% vs true 94.4%); India 0.7757 (pred-any 88.6% vs true 94.4%). Singleton accuracy 0.69 over 18,512 singletons (31% false merges, worth ~0.017 macro F0.5). |
| 7 | Test pred-any: US 94.8%, India 89.4%, France 87.5%. Share of entities with max p in [0.5, 0.9): US 3.8%, India 5.7%, France 7.9%; max p < 0.5: US 1.4%, India 4.9%, France 4.6%. US/India test histograms are as confident as validation or more (no sign of a test-pool degradation). |
| 8 | Baseline blocking recall (pairs / entities with all matches): k=1 0.222/0.031, k=3 0.544/0.206, k=5 0.702/0.395, k=10 0.783/0.530, k=20 0.815/0.585, k=30 0.831/0.612. |
| 9 | S1 is 0% non-Latin. Indic script: India S2 names 23.6%, addresses 23.8%; India S3 names 13.3%, addresses 22.9% (~8.7% of all test S2+S3 names). No other non-Latin scripts. US S2/S3 names ~6.5% non-ASCII (injected accents). France 16-28% non-ASCII (legitimate accents). |
| 10 | Gap 0.8244 → 0.763 decomposes as: country-mix shift toward India −0.012 (US/India at test mix: 0.8122), France −0.050 (implied France F0.5 ≈ 0.48 if US/India score as on validation). This corrects an earlier hypothesis (validation reusing the S2/S3 pool): item 7 shows no confidence degradation on test for US/India. Caveat: public LB is a subset of test. |
