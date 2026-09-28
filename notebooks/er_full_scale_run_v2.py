"""Full-scale run v2 — combines the three improvements built after the first
full-scale run (which produced macro F_0.5=0.8244, see
docs/ENTITY_RESOLUTION_APPROACH.md):

1. Vectorized sparse-matrix blocking (fast_block.py) instead of the
   per-entity Python dict loop — benchmarked 10.8x faster AND slightly
   better recall on a 300k-row sample (65.9s -> 6.1s, 94.0% -> 94.7%).
   Candidate generation was ~90%+ of the first run's total time (~4hrs);
   this should turn that into low minutes.
2. IDF-weighted ranking is inherent to fast_block's TF-IDF approach (no
   use_weights flag needed here, unlike pipeline.py's dict-based path —
   TF-IDF *is* the IDF weighting).
3. LaBSE embedding-similarity feature (embeddings.py) for the Devanagari
   cross-script name gap that no amount of token re-weighting can fix.

Uses SEPARATE checkpoint names (candidates_v2, features_v2) from the first
run's (candidates, features) so this can be developed/run without any risk
of colliding with er_generate_submission.py, which reads the ORIGINAL
features.pkl to reproduce the already-validated 0.8244 submission — do not
run this until that submission has been generated and validated, just to
keep the two clearly separate, not because of any actual file conflict.

Model download note: run `rsync -avzL ~/.cache/huggingface/
bce23159@10.1.19.16:~/.cache/huggingface/` from the laptop first, and set
HF_HUB_OFFLINE=1 when running this on PARAM Shavak (see
infra/param_shavak/README.md — large HF downloads get reset on that network).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time

import pandas as pd
from sklearn.model_selection import train_test_split

from src.entity_resolution.fast_block import generate_candidates_fast
from src.entity_resolution.features import FEATURE_NAMES, build_full_feature_matrix
from src.entity_resolution.matcher import select_matches, train_matcher, tune_threshold
from src.entity_resolution.normalize import normalize_address, normalize_name, tokenize
from src.metrics import entity_resolution_f_score, parse_id_list

TRAIN_DIR = "data_set/student_resource/dataset/train"
TEST_DIR = "data_set/student_resource/dataset/test"
TOP_K = 30
SEED = 42
N_WORKERS = min(32, mp.cpu_count())
USE_EMBEDDINGS = True
REPORT_PATH = "runs/er_full_scale_v2_report.txt"
CHECKPOINT_DIR = "runs/checkpoints"


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

    log(f"Full-scale run v2 (fast blocking + IDF + embeddings). Workers={N_WORKERS}, embeddings={USE_EMBEDDINGS}")
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

    candidates = load_checkpoint("candidates_v2")
    if candidates is not None:
        log(f"Loaded {len(candidates):,} entities' candidates from checkpoint — skipping regeneration.")
    else:
        log(f"Generating candidates for all {len(s1):,} Source 1 entities (top_k={TOP_K}, vectorized)...")
        t0 = time.time()
        candidates = generate_candidates_fast(s1, [s2, s3], top_k=TOP_K)
        log(f"  done in {time.time() - t0:.1f}s")
        save_checkpoint(candidates, "candidates_v2")
        log(f"  checkpointed to {CHECKPOINT_DIR}/candidates_v2.pkl")

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

    feature_checkpoint = load_checkpoint("features_v2")
    if feature_checkpoint is not None:
        log("Loaded feature matrices from checkpoint — skipping pair-building and feature computation.")
        X_train = feature_checkpoint["X_train"]
        y_train = feature_checkpoint["y_train"]
        X_val = feature_checkpoint["X_val"]
        val_s1_ids = feature_checkpoint["val_s1_ids"]
        val_candidate_ids = feature_checkpoint["val_candidate_ids"]
        feature_names = feature_checkpoint["feature_names"]
        log(
            f"  train: {len(y_train):,} pairs ({int(y_train.sum()):,} positive), "
            f"val: {len(val_s1_ids):,} pairs"
        )
    else:
        log("Building candidate-pair feature tables...")
        t0 = time.time()
        train_candidates = {k: v for k, v in candidates.items() if k in train_ids}
        val_candidates = {k: v for k, v in candidates.items() if k in val_ids}
        train_pairs = build_pairs_df(train_candidates, s1_lookup, cand_lookup, true_matches)
        val_pairs = build_pairs_df(val_candidates, s1_lookup, cand_lookup, true_matches)
        log(
            f"  train: {len(train_pairs):,} pairs ({train_pairs.label.sum():,} positive), "
            f"val: {len(val_pairs):,} pairs ({val_pairs.label.sum():,} positive), in {time.time()-t0:.1f}s"
        )

        log(f"Computing features (embeddings={USE_EMBEDDINGS})...")
        t0 = time.time()
        X_train, feature_names = build_full_feature_matrix(train_pairs, use_embeddings=USE_EMBEDDINGS)
        X_val, _ = build_full_feature_matrix(val_pairs, use_embeddings=USE_EMBEDDINGS)
        y_train = train_pairs.label.values
        val_s1_ids = val_pairs.s1_id.tolist()
        val_candidate_ids = val_pairs.candidate_id.tolist()
        log(f"  done in {time.time() - t0:.1f}s, features={feature_names}")

        save_checkpoint(
            {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "val_s1_ids": val_s1_ids,
                "val_candidate_ids": val_candidate_ids,
                "feature_names": feature_names,
            },
            "features_v2",
        )
        log(f"  checkpointed to {CHECKPOINT_DIR}/features_v2.pkl")

    log("Training classifier on full training split...")
    t0 = time.time()
    model = train_matcher(X_train, y_train)
    log(f"  done in {time.time() - t0:.1f}s")
    importances = sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1])
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
    log(f"\n=== FULL-SCALE V2 RESULT: macro F_0.5 = {best_score:.4f} at threshold={best_threshold:.2f} ===")
    log("(Compare to v1: macro F_0.5 = 0.8244 at threshold=0.90, ~83.1% blocking recall)")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nFull report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
