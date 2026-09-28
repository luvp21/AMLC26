"""Empirical validation of the blocking approach against REAL training data
with real ground truth, before committing to it for the full pipeline.

Answers the one question that matters: what fraction of true matches does
blocking actually find (recall ceiling), at what candidate-set size?
Everything downstream (features, classifier, threshold) can only ever do as
well as this ceiling allows — see docs/PROBLEM_STATEMENT.md.

Key finding (see docs/ENTITY_RESOLUTION_APPROACH.md): name-token-only
blocking recalls only ~73% at top_k=20, because Source 2 sometimes has
Devanagari business names that `anyascii` romanizes phonetically in a way
that doesn't match Source 1's English spelling. Blocking on name AND address
tokens together recovers most of that gap (89.4% at the same top_k) because
addresses often keep city/area names in Latin script even when the name
doesn't.

Uses a background sample of Source 2/3 (not the full ~5M rows each) to keep
this exploratory run fast; guarantees every true-match row is included so
recall is measured correctly. Real block sizes at full 5M-row scale will be
somewhat larger (more competition for top-K slots from generic-name noise),
so treat this as an optimistic-but-informative estimate — rerun at full
scale on PARAM Shavak once the approach is confirmed.
"""
from __future__ import annotations

import time

import pandas as pd

from src.entity_resolution.pipeline import (
    add_normalized_columns,
    build_source_indexes,
    generate_combined_candidates,
)
from src.metrics import parse_id_list

DATA_DIR = "data_set/student_resource/dataset/train"
SAMPLE_N = 3000
BACKGROUND_SAMPLE = 300_000
SEED = 42


def load_source_with_background(path: str, needed_ids: set[str], background_n: int, seed: int) -> pd.DataFrame:
    print(f"Loading {path}...")
    t0 = time.time()
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    print(f"  full: {len(df):,} rows in {time.time() - t0:.1f}s")
    needed_mask = df.entity_id.isin(needed_ids)
    needed_rows = df[needed_mask]
    n_bg = min(background_n, int((~needed_mask).sum()))
    background = df[~needed_mask].sample(n=n_bg, random_state=seed)
    combined = pd.concat([needed_rows, background]).drop_duplicates(subset="entity_id").reset_index(drop=True)
    print(f"  using {len(combined):,} rows ({len(needed_rows):,} guaranteed true-matches + {n_bg:,} background)")
    return combined


def eval_recall(candidates: dict[str, list[str]], true_matches: dict[str, set[str]], label: str) -> None:
    hits = total_true = zero_candidate_entities = 0
    for s1_id, true_set in true_matches.items():
        cand_set = set(candidates.get(s1_id, []))
        if not cand_set:
            zero_candidate_entities += 1
        if not true_set:
            continue
        hits += len(true_set & cand_set)
        total_true += len(true_set)
    recall = hits / total_true if total_true else 0.0
    avg_candidates = sum(len(v) for v in candidates.values()) / len(candidates)
    print(
        f"  {label}: recall={recall:6.1%} ({hits:,}/{total_true:,} true matches found), "
        f"avg candidates/entity={avg_candidates:5.1f}, entities w/ zero candidates={zero_candidate_entities:,}"
    )


def main():
    print("Loading ground truth...")
    gt = pd.read_csv(f"{DATA_DIR}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    sample_gt = gt.sample(n=SAMPLE_N, random_state=SEED)
    true_matches = {row.source1_entity_id: parse_id_list(row.matched_entity_ids) for row in sample_gt.itertuples()}
    sample_s1_ids = set(true_matches.keys())
    needed_match_ids = set().union(*true_matches.values()) if true_matches else set()
    n_singletons = sum(1 for v in true_matches.values() if not v)
    print(
        f"Sampled {len(sample_s1_ids):,} Source 1 entities "
        f"({n_singletons:,} singletons, {len(needed_match_ids):,} true match IDs to find)"
    )

    s1 = pd.read_csv(f"{DATA_DIR}/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s1 = s1[s1.entity_id.isin(sample_s1_ids)].reset_index(drop=True)

    s2 = load_source_with_background(f"{DATA_DIR}/train_source2.tsv", needed_match_ids, BACKGROUND_SAMPLE, SEED)
    s3 = load_source_with_background(f"{DATA_DIR}/train_source3.tsv", needed_match_ids, BACKGROUND_SAMPLE, SEED + 1)

    print("Normalizing (name + address) and building indexes...")
    t0 = time.time()
    for df in (s1, s2, s3):
        add_normalized_columns(df)
    s2_indexes = build_source_indexes(s2)
    s3_indexes = build_source_indexes(s3)
    print(f"  done in {time.time() - t0:.1f}s")

    print("\nRecall at various top_k (name + address blocking):")
    for top_k in (10, 20, 50, 100):
        candidates = generate_combined_candidates(s1, s2_indexes, s3_indexes, top_k=top_k)
        eval_recall(candidates, true_matches, f"top_k={top_k:4d}")


if __name__ == "__main__":
    main()
