"""End-to-end validation: block -> features -> train classifier -> tune
threshold -> measure REAL macro F_0.5 on a held-out split of Source 1
entities, using real training data and real ground truth throughout.

This is the number that actually answers "will this score well on the
leaderboard" — everything before this (docs/ENTITY_RESOLUTION_APPROACH.md's
blocking-recall numbers) was necessary but not sufficient; recall ceiling
sets an upper bound, this measures what we can actually achieve within it.

Train/val split is by Source 1 entity (not by row/pair) to avoid leaking a
Source 1 entity's candidates across the split.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.entity_resolution.features import FEATURE_NAMES, build_feature_matrix
from src.entity_resolution.matcher import select_matches, train_matcher, tune_threshold
from src.entity_resolution.pipeline import (
    add_normalized_columns,
    build_source_indexes,
    generate_combined_candidates,
)
from src.metrics import entity_resolution_f_score, parse_id_list

DATA_DIR = "data_set/student_resource/dataset/train"
SAMPLE_N = 8000
BACKGROUND_SAMPLE = 300_000
TOP_K = 30
SEED = 42


def load_source_with_background(path: str, needed_ids: set[str], background_n: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    needed_mask = df.entity_id.isin(needed_ids)
    needed_rows = df[needed_mask]
    n_bg = min(background_n, int((~needed_mask).sum()))
    background = df[~needed_mask].sample(n=n_bg, random_state=seed)
    return pd.concat([needed_rows, background]).drop_duplicates(subset="entity_id").reset_index(drop=True)


def build_pairs_df(
    candidates: dict[str, list[str]],
    s1_lookup: pd.DataFrame,
    cand_lookup: pd.DataFrame,
    true_matches: dict[str, set[str]],
) -> pd.DataFrame:
    s1_id_col, cand_id_col = [], []
    for s1_id, cand_ids in candidates.items():
        for cid in cand_ids:
            s1_id_col.append(s1_id)
            cand_id_col.append(cid)
    pairs = pd.DataFrame({"s1_id": s1_id_col, "candidate_id": cand_id_col})

    s1_cols = s1_lookup.add_prefix("s1_")
    c_cols = cand_lookup.add_prefix("c_")
    pairs = pairs.merge(s1_cols, left_on="s1_id", right_index=True, how="inner")
    pairs = pairs.merge(c_cols, left_on="candidate_id", right_index=True, how="inner")
    pairs["label"] = [
        1 if cid in true_matches.get(s1_id, ()) else 0
        for s1_id, cid in zip(pairs.s1_id, pairs.candidate_id)
    ]
    return pairs


def main():
    print("Loading ground truth and sampling entities...")
    gt = pd.read_csv(f"{DATA_DIR}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    sample_gt = gt.sample(n=SAMPLE_N, random_state=SEED)
    true_matches = {row.source1_entity_id: parse_id_list(row.matched_entity_ids) for row in sample_gt.itertuples()}
    all_s1_ids = list(true_matches.keys())
    train_ids, val_ids = train_test_split(all_s1_ids, test_size=0.3, random_state=SEED)
    train_ids, val_ids = set(train_ids), set(val_ids)
    needed_match_ids = set().union(*true_matches.values()) if true_matches else set()
    print(f"  {len(train_ids):,} train entities, {len(val_ids):,} val entities")

    print("Loading Source 1 (filtered to sample) + Source 2/3 (background sample)...")
    t0 = time.time()
    s1 = pd.read_csv(f"{DATA_DIR}/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s1 = s1[s1.entity_id.isin(true_matches.keys())].reset_index(drop=True)
    s2 = load_source_with_background(f"{DATA_DIR}/train_source2.tsv", needed_match_ids, BACKGROUND_SAMPLE, SEED)
    s3 = load_source_with_background(f"{DATA_DIR}/train_source3.tsv", needed_match_ids, BACKGROUND_SAMPLE, SEED + 1)
    print(f"  done in {time.time() - t0:.1f}s")

    print("Normalizing + blocking...")
    t0 = time.time()
    for df in (s1, s2, s3):
        add_normalized_columns(df)
    s2_indexes = build_source_indexes(s2)
    s3_indexes = build_source_indexes(s3)
    candidates = generate_combined_candidates(s1, s2_indexes, s3_indexes, top_k=TOP_K)
    print(f"  done in {time.time() - t0:.1f}s")

    print("Building candidate-pair feature table...")
    t0 = time.time()
    lookup_cols = ["business_name", "business_address", "name_tokens", "addr_tokens", "country"]
    rename_map = {"business_name": "name", "business_address": "addr"}
    s1_lookup = s1.set_index("entity_id")[lookup_cols].rename(columns=rename_map)
    cand_lookup = pd.concat([s2, s3]).set_index("entity_id")[lookup_cols].rename(columns=rename_map)

    train_candidates = {k: v for k, v in candidates.items() if k in train_ids}
    val_candidates = {k: v for k, v in candidates.items() if k in val_ids}
    train_pairs = build_pairs_df(train_candidates, s1_lookup, cand_lookup, true_matches)
    val_pairs = build_pairs_df(val_candidates, s1_lookup, cand_lookup, true_matches)
    print(
        f"  train: {len(train_pairs):,} pairs ({train_pairs.label.sum():,} positive), "
        f"val: {len(val_pairs):,} pairs ({val_pairs.label.sum():,} positive), in {time.time() - t0:.1f}s"
    )

    print("Computing features...")
    t0 = time.time()
    X_train = build_feature_matrix(train_pairs)
    X_val = build_feature_matrix(val_pairs)
    y_train = train_pairs.label.values
    print(f"  done in {time.time() - t0:.1f}s, feature matrix shape {X_train.shape}")

    print("Training classifier...")
    t0 = time.time()
    model = train_matcher(X_train, y_train)
    print(f"  done in {time.time() - t0:.1f}s")
    importances = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])
    print("  feature importances:", ", ".join(f"{n}={v}" for n, v in importances))

    print("Scoring validation split...")
    val_proba = model.predict_proba(X_val)[:, 1]
    val_true_matches = {k: v for k, v in true_matches.items() if k in val_ids}

    best_threshold, best_score, all_scores = tune_threshold(
        val_pairs.s1_id.tolist(), val_pairs.candidate_id.tolist(), val_proba, val_true_matches
    )
    print("\nThreshold sweep (macro F_0.5 on held-out validation entities):")
    for t, s in sorted(all_scores.items()):
        marker = "  <-- best" if t == best_threshold else ""
        print(f"  threshold={t:.2f}: F_0.5={s:.4f}{marker}")

    print(f"\n=== VALIDATION RESULT: macro F_0.5 = {best_score:.4f} at threshold={best_threshold:.2f} ===")

    # Sanity baselines for context
    always_empty = entity_resolution_f_score({s1_id: [] for s1_id in val_true_matches}, val_true_matches)
    always_all_candidates = entity_resolution_f_score(val_candidates, val_true_matches)
    print(f"Baseline (always predict no match): F_0.5 = {always_empty:.4f}")
    print(f"Baseline (accept every blocked candidate): F_0.5 = {always_all_candidates:.4f}")


if __name__ == "__main__":
    main()
