"""Phase 0 diagnostics that need PARAM Shavak's cached artifacts:
  - baseline reproduction: retrain from runs/checkpoints/features.pkl and
    re-tune the threshold; must land on macro F0.5 ~0.8244 at 0.90
  - item 6: validation predicted vs true match rate per country
  - item 7: test max-probability histogram per entity per country (recomputes
    test features from runs/checkpoints/test_candidates.pkl with the
    bit-identical fast feature builder)
  - item 8: blocking recall at k in {1,3,5,10,20,30} on validation entities,
    from runs/checkpoints/candidates.pkl (the dict-based raw-count blocking
    that produced the submitted 0.763)

Run from repo root on PARAM Shavak:
  nohup env PYTHONPATH=. python3 -u notebooks/phase0_remote.py > runs/phase0_remote.log 2>&1 &
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.entity_resolution.features import build_feature_matrix_fast
from src.entity_resolution.matcher import select_matches, train_matcher, tune_threshold
from src.entity_resolution.pipeline import add_normalized_columns_parallel
from src.metrics import entity_resolution_f_score, parse_id_list

TRAIN = "data_set/student_resource/dataset/train"
TEST = "data_set/student_resource/dataset/test"
CKPT = "runs/checkpoints"
OUT = "runs/phase0_remote.txt"
SEED = 42
THRESHOLD = 0.90
N_WORKERS = min(32, mp.cpu_count())
HIST_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0000001]

lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    lines.append(msg)


def load(path: str, cols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=cols)


def load_pickle(name: str):
    with open(f"{CKPT}/{name}.pkl", "rb") as f:
        return pickle.load(f)


def max_prob_hist(max_p: pd.Series, country: pd.Series) -> pd.DataFrame:
    binned = pd.cut(max_p, HIST_BINS, right=False, labels=[f"<{b:.2f}" for b in HIST_BINS[1:-1]] + ["<=1.00"])
    return pd.crosstab(binned, country, normalize="columns").round(4)


def main() -> None:
    t_start = time.time()

    # ---------- baseline reproduction ----------
    log("=== Baseline reproduction from cached features ===")
    gt = load(f"{TRAIN}/train_ground_truth.tsv")
    true_matches = {r.source1_entity_id: parse_id_list(r.matched_entity_ids) for r in gt.itertuples()}
    s1 = load(f"{TRAIN}/train_source1.tsv", ["entity_id", "country"])
    country_of = s1.set_index("entity_id").country
    _, val_ids = train_test_split(s1.entity_id.tolist(), test_size=0.15, random_state=SEED)
    val_ids = set(val_ids)
    val_true = {k: v for k, v in true_matches.items() if k in val_ids}

    feats = load_pickle("features")
    t0 = time.time()
    model = train_matcher(feats["X_train"], feats["y_train"])
    log(f"retrained in {time.time() - t0:.1f}s")
    val_p = model.predict_proba(feats["X_val"])[:, 1]
    best_t, best_s, _ = tune_threshold(feats["val_s1_ids"], feats["val_candidate_ids"], val_p, val_true)
    log(f"validation macro F0.5 = {best_s:.4f} at threshold {best_t:.2f}  (baseline record: 0.8244 at 0.90)")

    # ---------- item 6 ----------
    log("\n=== Item 6: validation predicted vs true match rate per country (threshold 0.90) ===")
    pred = select_matches(feats["val_s1_ids"], feats["val_candidate_ids"], val_p, THRESHOLD)
    val_df = pd.DataFrame({"s1": sorted(val_true)})
    val_df["country"] = val_df.s1.map(country_of)
    val_df["true_any"] = val_df.s1.map(lambda s: len(val_true[s]) > 0)
    val_df["pred_any"] = val_df.s1.map(lambda s: len(pred.get(s, [])) > 0)
    val_df["f"] = val_df.s1.map(lambda s: entity_resolution_f_score({s: pred.get(s, [])}, {s: val_true[s]}))
    summary = val_df.groupby("country").agg(
        n=("s1", "size"), true_match_rate=("true_any", "mean"),
        pred_match_rate=("pred_any", "mean"), macro_f05=("f", "mean"),
    ).round(4)
    log(summary.to_string())
    singles = val_df[~val_df.true_any]
    log(f"singleton accuracy (empty prediction on true singletons): {(~singles.pred_any).mean():.4f} "
        f"over {len(singles):,} singletons")

    val_pairs = pd.DataFrame({"s1": feats["val_s1_ids"], "p": val_p})
    val_max = val_pairs.groupby("s1").p.max().reindex(val_df.s1).fillna(0.0)
    log("\nvalidation max-probability-per-entity histogram (share of entities per country):")
    log(max_prob_hist(val_max.reset_index(drop=True), val_df.country).to_string())
    del feats

    # ---------- item 8 ----------
    log("\n=== Item 8: blocking recall at k on validation entities (dict-based raw-count blocking) ===")
    cands = load_pickle("candidates")
    matched_val = {k: v for k, v in val_true.items() if v}
    total_true = sum(len(v) for v in matched_val.values())
    for k in (1, 3, 5, 10, 20, 30):
        hits = full = 0
        for s, tset in matched_val.items():
            got = tset & set(cands.get(s, [])[:k])
            hits += len(got)
            full += len(got) == len(tset)
        log(f"  k={k:2d}: pair recall={hits / total_true:.4f}   entities with ALL matches retrieved={full / len(matched_val):.4f}")
    del cands

    # ---------- item 7 ----------
    log("\n=== Item 7: test max-probability-per-entity histogram per country ===")
    test_s1 = load(f"{TEST}/test_source1.tsv")
    test_s2 = load(f"{TEST}/test_source2.tsv")
    test_s3 = load(f"{TEST}/test_source3.tsv")
    t0 = time.time()
    for df in (test_s1, test_s2, test_s3):
        add_normalized_columns_parallel(df, n_workers=N_WORKERS)
    log(f"normalized test sources in {time.time() - t0:.1f}s")

    test_cands = load_pickle("test_candidates")
    cols = ["business_name", "business_address", "name_tokens", "addr_tokens", "country"]
    ren = {"business_name": "name", "business_address": "addr"}
    s1_look = test_s1.set_index("entity_id")[cols].rename(columns=ren).add_prefix("s1_")
    c_look = pd.concat([test_s2, test_s3]).set_index("entity_id")[cols].rename(columns=ren).add_prefix("c_")
    del test_s2, test_s3
    ids = [(s, c) for s, cs in test_cands.items() for c in cs]
    del test_cands
    pairs = pd.DataFrame(ids, columns=["s1_id", "candidate_id"])
    del ids
    pairs = pairs.merge(s1_look, left_on="s1_id", right_index=True).merge(c_look, left_on="candidate_id", right_index=True)
    log(f"test pairs: {len(pairs):,}")

    t0 = time.time()
    X_test = build_feature_matrix_fast(pairs, n_workers=N_WORKERS)
    log(f"test features ({N_WORKERS} workers) in {time.time() - t0:.1f}s  (previous per-row run: 1306.5s incl. prediction)")
    test_p = model.predict_proba(X_test)[:, 1]
    del X_test

    test_max = pd.Series(test_p).groupby(pairs.s1_id.to_numpy()).max()
    test_max = test_max.reindex(test_s1.entity_id).fillna(0.0).reset_index(drop=True)
    log(max_prob_hist(test_max, test_s1.country).to_string())
    rate = (test_max >= THRESHOLD).groupby(test_s1.country).mean().round(4)
    log("predicted match rate at 0.90 recomputed (should equal the submitted file's): "
        + ", ".join(f"{c}={v:.4f}" for c, v in rate.items()))

    os.makedirs(CKPT, exist_ok=True)
    pd.DataFrame({"s1_id": test_s1.entity_id, "country": test_s1.country, "max_p": test_max}).to_pickle(
        f"{CKPT}/phase0_test_max_p.pkl")
    log(f"\ntotal runtime: {time.time() - t_start:.1f}s")
    with open(OUT, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
