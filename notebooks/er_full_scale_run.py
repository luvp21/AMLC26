"""Full-scale entity resolution run — the real 2.2M/5M/5M rows, not a sample.
Run this on PARAM Shavak (250GB RAM, 56 threads), not the laptop (15GB RAM,
already observed swapping on a 600k-row sample).

What this does, in order:
1. Load the FULL training data (all 4 files).
2. Normalize + block on the FULL Source 2 + Source 3 universe (multiprocessing
   across the 56 threads for the normalize step, the most expensive part).
3. Split Source 1 entities 85/15 train/val (by entity, not by row).
4. Train the classifier on the full training split, tune threshold on val.
5. Report the real macro F_0.5 — this supersedes the 300k-row-sample estimate
   of 0.9287 in docs/ENTITY_RESOLUTION_APPROACH.md; full scale has more
   competing false candidates, so expect this number to be somewhat lower.
6. Smoke-test the pipeline against real French test rows (France has zero
   training examples — there's no ground truth to score it against, but this
   at least confirms the pipeline doesn't crash or behave pathologically on
   unseen-country input, e.g. produces a reasonable non-zero candidate count).

Writes results to runs/er_full_scale_report.txt (gitignored) — this file is
the source of truth for what the actual full-scale numbers were, not this
script's docstring, which won't be updated after each run.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.entity_resolution.features import FEATURE_NAMES, build_feature_matrix
from src.entity_resolution.matcher import select_matches, train_matcher, tune_threshold
from src.entity_resolution.normalize import normalize_address, normalize_name, tokenize
from src.entity_resolution.pipeline import build_source_indexes, generate_combined_candidates_parallel
from src.metrics import entity_resolution_f_score, parse_id_list

TRAIN_DIR = "data_set/student_resource/dataset/train"
TEST_DIR = "data_set/student_resource/dataset/test"
TOP_K = 30
SEED = 42
N_WORKERS = min(32, mp.cpu_count())  # leave headroom on a 56-thread shared machine
REPORT_PATH = "runs/er_full_scale_report.txt"
CHECKPOINT_DIR = "runs/checkpoints"


def save_checkpoint(obj, name: str) -> None:
    """Checkpoint expensive intermediate results to disk. Candidate generation
    alone measured ~4 hours at full scale — losing that to a downstream crash
    (as happened the first time this ran, on a missing lightgbm install) is
    unacceptable given the 72-hour clock. Every checkpoint boundary below
    exists because it already cost real time once."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = f"{CHECKPOINT_DIR}/{name}.pkl"
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)  # atomic — a crash mid-write can't corrupt the real checkpoint


def load_checkpoint(name: str):
    path = f"{CHECKPOINT_DIR}/{name}.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _normalize_row(args: tuple[str, str]) -> tuple[list[str], list[str]]:
    name, addr = args
    return tokenize(normalize_name(name)), tokenize(normalize_address(addr))


def add_normalized_columns_parallel(df: pd.DataFrame, label: str) -> pd.DataFrame:
    print(f"  normalizing {label} ({len(df):,} rows) with {N_WORKERS} workers...")
    t0 = time.time()
    pairs = list(zip(df.business_name, df.business_address))
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(_normalize_row, pairs, chunksize=2000)
    df["name_tokens"] = [r[0] for r in results]
    df["addr_tokens"] = [r[1] for r in results]
    print(f"    done in {time.time() - t0:.1f}s")
    return df


def load_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def build_pairs_df(
    candidates: dict[str, list[str]],
    s1_lookup: pd.DataFrame,
    cand_lookup: pd.DataFrame,
    true_matches: dict[str, set[str]] | None,
) -> pd.DataFrame:
    s1_id_col, cand_id_col = [], []
    for s1_id, cand_ids in candidates.items():
        for cid in cand_ids:
            s1_id_col.append(s1_id)
            cand_id_col.append(cid)
    pairs = pd.DataFrame({"s1_id": s1_id_col, "candidate_id": cand_id_col})
    pairs = pairs.merge(s1_lookup.add_prefix("s1_"), left_on="s1_id", right_index=True, how="inner")
    pairs = pairs.merge(cand_lookup.add_prefix("c_"), left_on="candidate_id", right_index=True, how="inner")
    if true_matches is not None:
        pairs["label"] = [
            1 if cid in true_matches.get(s1_id, ()) else 0
            for s1_id, cid in zip(pairs.s1_id, pairs.candidate_id)
        ]
    return pairs


def make_lookup(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["business_name", "business_address", "name_tokens", "addr_tokens", "country"]
    return df.set_index("entity_id")[cols].rename(columns={"business_name": "name", "business_address": "addr"})


def main():
    os.makedirs("runs", exist_ok=True)
    report_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        report_lines.append(msg)

    log(f"Full-scale entity resolution run. Workers={N_WORKERS}")
    log("Loading full training data...")
    t0 = time.time()
    gt = load_tsv(f"{TRAIN_DIR}/train_ground_truth.tsv")
    s1 = load_tsv(f"{TRAIN_DIR}/train_source1.tsv")
    s2 = load_tsv(f"{TRAIN_DIR}/train_source2.tsv")
    s3 = load_tsv(f"{TRAIN_DIR}/train_source3.tsv")
    log(f"  s1={len(s1):,} s2={len(s2):,} s3={len(s3):,} ground_truth={len(gt):,} rows, in {time.time()-t0:.1f}s")

    true_matches = {row.source1_entity_id: parse_id_list(row.matched_entity_ids) for row in gt.itertuples()}

    log("Normalizing (multiprocessing)...")
    add_normalized_columns_parallel(s1, "source1")
    add_normalized_columns_parallel(s2, "source2")
    add_normalized_columns_parallel(s3, "source3")

    log("Building inverted indexes (full Source 2 + Source 3)...")
    t0 = time.time()
    s2_indexes = build_source_indexes(s2)
    s3_indexes = build_source_indexes(s3)
    log(f"  done in {time.time() - t0:.1f}s")

    candidates = load_checkpoint("candidates")
    if candidates is not None:
        log(f"Loaded {len(candidates):,} entities' candidates from checkpoint — skipping regeneration.")
    else:
        log(f"Generating candidates for all {len(s1):,} Source 1 entities (top_k={TOP_K})...")
        t0 = time.time()
        candidates = generate_combined_candidates_parallel(s1, s2_indexes, s3_indexes, top_k=TOP_K)
        log(f"  done in {time.time() - t0:.1f}s")
        save_checkpoint(candidates, "candidates")
        log(f"  checkpointed to {CHECKPOINT_DIR}/candidates.pkl")

    hits = total_true = 0
    for s1_id, true_set in true_matches.items():
        if not true_set:
            continue
        hits += len(true_set & set(candidates.get(s1_id, [])))
        total_true += len(true_set)
    log(f"  FULL-SCALE blocking recall @ top_k={TOP_K}: {hits/total_true:.1%} ({hits:,}/{total_true:,})")

    log("Splitting Source 1 entities 85/15 train/val...")
    all_ids = s1.entity_id.tolist()
    train_ids, val_ids = train_test_split(all_ids, test_size=0.15, random_state=SEED)
    train_ids, val_ids = set(train_ids), set(val_ids)

    s1_lookup = make_lookup(s1)
    cand_lookup = pd.concat([make_lookup(s2), make_lookup(s3)])

    feature_checkpoint = load_checkpoint("features")
    if feature_checkpoint is not None:
        log("Loaded feature matrices from checkpoint — skipping pair-building and feature computation.")
        X_train = feature_checkpoint["X_train"]
        y_train = feature_checkpoint["y_train"]
        X_val = feature_checkpoint["X_val"]
        val_s1_ids = feature_checkpoint["val_s1_ids"]
        val_candidate_ids = feature_checkpoint["val_candidate_ids"]
        log(
            f"  train: {len(y_train):,} pairs ({int(y_train.sum()):,} positive), "
            f"val: {len(val_s1_ids):,} pairs"
        )
    else:
        log("Building candidate-pair feature tables (this is the big one)...")
        t0 = time.time()
        train_candidates = {k: v for k, v in candidates.items() if k in train_ids}
        val_candidates = {k: v for k, v in candidates.items() if k in val_ids}
        train_pairs = build_pairs_df(train_candidates, s1_lookup, cand_lookup, true_matches)
        val_pairs = build_pairs_df(val_candidates, s1_lookup, cand_lookup, true_matches)
        log(
            f"  train: {len(train_pairs):,} pairs ({train_pairs.label.sum():,} positive), "
            f"val: {len(val_pairs):,} pairs ({val_pairs.label.sum():,} positive), in {time.time()-t0:.1f}s"
        )

        log("Computing features...")
        t0 = time.time()
        X_train = build_feature_matrix(train_pairs)
        X_val = build_feature_matrix(val_pairs)
        y_train = train_pairs.label.values
        val_s1_ids = val_pairs.s1_id.tolist()
        val_candidate_ids = val_pairs.candidate_id.tolist()
        log(f"  done in {time.time() - t0:.1f}s")

        save_checkpoint(
            {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "val_s1_ids": val_s1_ids,
                "val_candidate_ids": val_candidate_ids,
            },
            "features",
        )
        log(f"  checkpointed to {CHECKPOINT_DIR}/features.pkl")

    log("Training classifier on full training split...")
    t0 = time.time()
    model = train_matcher(X_train, y_train)
    log(f"  done in {time.time() - t0:.1f}s")
    importances = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])
    log("  feature importances: " + ", ".join(f"{n}={v}" for n, v in importances))

    log("Scoring full-scale validation split...")
    val_proba = model.predict_proba(X_val)[:, 1]
    val_true_matches = {k: v for k, v in true_matches.items() if k in val_ids}
    best_threshold, best_score, all_scores = tune_threshold(
        val_s1_ids, val_candidate_ids, val_proba, val_true_matches
    )
    for t, s in sorted(all_scores.items()):
        marker = "  <-- best" if t == best_threshold else ""
        log(f"  threshold={t:.2f}: F_0.5={s:.4f}{marker}")
    log(f"\n=== FULL-SCALE RESULT: macro F_0.5 = {best_score:.4f} at threshold={best_threshold:.2f} ===")

    log("\nSmoke-testing on real French test data (no ground truth exists — checking the pipeline behaves, not accuracy)...")
    log("  Blocking against TEST Source 2/3 (not train — train has zero France rows, test does).")
    test_s1 = load_tsv(f"{TEST_DIR}/test_source1.tsv")
    test_s2 = load_tsv(f"{TEST_DIR}/test_source2.tsv")
    test_s3 = load_tsv(f"{TEST_DIR}/test_source3.tsv")
    france_s1 = test_s1[test_s1.country == "France"].reset_index(drop=True)
    if len(france_s1) == 0:
        log("  WARNING: no France rows found in test_source1.tsv — check the file.")
    else:
        add_normalized_columns_parallel(test_s2, "test source2")
        add_normalized_columns_parallel(test_s3, "test source3")
        test_s2_indexes = build_source_indexes(test_s2)
        test_s3_indexes = build_source_indexes(test_s3)
        add_normalized_columns_parallel(france_s1, "France test sample")
        france_candidates = generate_combined_candidates_parallel(
            france_s1, test_s2_indexes, test_s3_indexes, top_k=TOP_K
        )
        avg_candidates = sum(len(v) for v in france_candidates.values()) / len(france_candidates)
        zero_candidates = sum(1 for v in france_candidates.values() if not v)
        log(
            f"  France: {len(france_s1):,} entities, avg candidates/entity={avg_candidates:.1f}, "
            f"zero-candidate entities={zero_candidates:,} ({zero_candidates/len(france_s1):.1%})"
        )
        log(
            "  No ground truth exists for test, so this can't measure real accuracy — it only confirms "
            "the pipeline produces sane (non-zero, non-degenerate) candidate sets for a country the "
            "classifier never trained on. A high zero-candidate rate here would be a red flag."
        )

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nFull report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
