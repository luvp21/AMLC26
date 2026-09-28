"""Generate the actual leaderboard submission (matching_results.tsv,
candidate_pairs.tsv) from the classifier validated in er_full_scale_run.py
(macro F_0.5 = 0.8244 at threshold=0.90 — see docs/ENTITY_RESOLUTION_APPROACH.md).

Retrains from the features.pkl checkpoint (fast, ~4 min) rather than needing
a separately-saved model file, since that checkpoint already has the exact
features that produced the validated score.

The expensive part is unavoidable: generating candidates for the REAL test
set (~1.7M Source 1 entities against ~10M Source 2+3 entities) is the same
scale as the training run's candidate generation, which took ~4 hours — a
match can only ever be a subset of what blocking found, so this has to run
against the actual test sources, not reuse the training-side candidates.
Checkpointed the same way as before, for the same reason (a downstream bug
must never cost this again).

IMPORTANT: uses use_weights=False (raw shared-token-count ranking) to
reproduce EXACTLY the blocking behavior the classifier was trained and
validated against. block.py/pipeline.py later added IDF-weighted ranking
(a real fix — see docs/ENTITY_RESOLUTION_APPROACH.md — but not yet
retrained/revalidated against). Mixing the new blocking with the
old-trained classifier here would silently submit an untested
configuration on one of the 5 daily submission slots; don't.
"""
from __future__ import annotations

import os
import pickle
import time

import pandas as pd

from src.entity_resolution.features import build_feature_matrix
from src.entity_resolution.matcher import select_matches, train_matcher
from src.entity_resolution.pipeline import (
    add_normalized_columns_parallel,
    build_source_indexes,
    generate_combined_candidates_parallel,
)

TEST_DIR = "data_set/student_resource/dataset/test"
OUTPUT_DIR = "output"
CHECKPOINT_DIR = "runs/checkpoints"
TOP_K = 30
THRESHOLD = 0.90  # tuned on the full-scale validation split — see docs/ENTITY_RESOLUTION_APPROACH.md


def save_checkpoint(obj, name: str) -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = f"{CHECKPOINT_DIR}/{name}.pkl"
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def load_checkpoint(name: str):
    path = f"{CHECKPOINT_DIR}/{name}.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def make_lookup(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["business_name", "business_address", "name_tokens", "addr_tokens", "country"]
    return df.set_index("entity_id")[cols].rename(columns={"business_name": "name", "business_address": "addr"})


def write_id_list_tsv(path: str, id_col_name: str, list_col_name: str, rows: list[tuple[str, list[str]]]) -> None:
    """Exact format required by docs/PROBLEM_STATEMENT.md: tab-separated,
    comma-joined ID lists with no quoting, empty string (not omitted row)
    for entities with no matches/candidates."""
    with open(path, "w") as f:
        f.write(f"{id_col_name}\t{list_col_name}\n")
        for entity_id, ids in rows:
            f.write(f"{entity_id}\t{','.join(ids)}\n")


def main():
    print("Loading features checkpoint from the validated full-scale run...")
    feature_checkpoint = load_checkpoint("features")
    if feature_checkpoint is None:
        raise SystemExit("runs/checkpoints/features.pkl not found — run er_full_scale_run.py first.")
    X_train = feature_checkpoint["X_train"]
    y_train = feature_checkpoint["y_train"]

    print("Retraining classifier (fast — same features that produced macro F_0.5=0.8244)...")
    t0 = time.time()
    model = train_matcher(X_train, y_train)
    print(f"  done in {time.time() - t0:.1f}s")

    print("Loading TEST data...")
    t0 = time.time()
    test_s1 = load_tsv(f"{TEST_DIR}/test_source1.tsv")
    test_s2 = load_tsv(f"{TEST_DIR}/test_source2.tsv")
    test_s3 = load_tsv(f"{TEST_DIR}/test_source3.tsv")
    print(f"  s1={len(test_s1):,} s2={len(test_s2):,} s3={len(test_s3):,} rows, in {time.time() - t0:.1f}s")

    print("Normalizing TEST data...")
    add_normalized_columns_parallel(test_s1)
    add_normalized_columns_parallel(test_s2)
    add_normalized_columns_parallel(test_s3)

    print("Building TEST indexes...")
    t0 = time.time()
    s2_indexes = build_source_indexes(test_s2)
    s3_indexes = build_source_indexes(test_s3)
    print(f"  done in {time.time() - t0:.1f}s")

    test_candidates = load_checkpoint("test_candidates")
    if test_candidates is not None:
        print(f"Loaded {len(test_candidates):,} entities' TEST candidates from checkpoint.")
    else:
        print(f"Generating candidates for all {len(test_s1):,} TEST Source 1 entities (top_k={TOP_K})...")
        t0 = time.time()
        test_candidates = generate_combined_candidates_parallel(
            test_s1, s2_indexes, s3_indexes, top_k=TOP_K, use_weights=False
        )
        print(f"  done in {time.time() - t0:.1f}s")
        save_checkpoint(test_candidates, "test_candidates")
        print(f"  checkpointed to {CHECKPOINT_DIR}/test_candidates.pkl")

    print("Building candidate-pair feature table for TEST...")
    t0 = time.time()
    s1_lookup = make_lookup(test_s1)
    cand_lookup = pd.concat([make_lookup(test_s2), make_lookup(test_s3)])

    s1_id_col, cand_id_col = [], []
    for s1_id, cand_ids in test_candidates.items():
        for cid in cand_ids:
            s1_id_col.append(s1_id)
            cand_id_col.append(cid)
    pairs = pd.DataFrame({"s1_id": s1_id_col, "candidate_id": cand_id_col})
    pairs = pairs.merge(s1_lookup.add_prefix("s1_"), left_on="s1_id", right_index=True, how="inner")
    pairs = pairs.merge(cand_lookup.add_prefix("c_"), left_on="candidate_id", right_index=True, how="inner")
    print(f"  {len(pairs):,} candidate pairs, in {time.time() - t0:.1f}s")

    print("Computing features + predicting match probabilities...")
    t0 = time.time()
    X_test = build_feature_matrix(pairs)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"  done in {time.time() - t0:.1f}s")

    print(f"Selecting matches at threshold={THRESHOLD}...")
    matches = select_matches(pairs.s1_id.tolist(), pairs.candidate_id.tolist(), proba, THRESHOLD)

    # Every test Source 1 entity needs exactly one row, even with zero candidates.
    all_s1_ids = test_s1.entity_id.tolist()
    for s1_id in all_s1_ids:
        matches.setdefault(s1_id, [])
        test_candidates.setdefault(s1_id, [])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_id_list_tsv(
        f"{OUTPUT_DIR}/matching_results.tsv",
        "source1_entity_id",
        "matched_entity_ids",
        [(s1_id, matches[s1_id]) for s1_id in all_s1_ids],
    )
    write_id_list_tsv(
        f"{OUTPUT_DIR}/candidate_pairs.tsv",
        "source1_entity_id",
        "candidate_entity_ids",
        [(s1_id, test_candidates[s1_id]) for s1_id in all_s1_ids],
    )

    n_matched = sum(1 for s1_id in all_s1_ids if matches[s1_id])
    print(f"\nWrote {OUTPUT_DIR}/matching_results.tsv and {OUTPUT_DIR}/candidate_pairs.tsv")
    print(
        f"  {len(all_s1_ids):,} total test entities, {n_matched:,} with at least one match "
        f"({n_matched / len(all_s1_ids):.1%})"
    )
    print("\nValidate before submitting:")
    print(
        "  python3 data_set/student_resource/utils/validate_submission.py "
        f"--matching {OUTPUT_DIR}/matching_results.tsv --candidate {OUTPUT_DIR}/candidate_pairs.tsv "
        f"--test-dir {TEST_DIR}"
    )


if __name__ == "__main__":
    main()
