"""Stage 3: learned pruner over the blocking union.

  python -m src.v2.prune --tag train     # cross-fit on training rows, cutoff on tune rows, apply
  python -m src.v2.prune --tag dev       # same, on the dev partitions
  python -m src.v2.prune --tag test      # apply the saved models (mean) and cutoff

LightGBM (300 trees) on blocking signals only, cross-fitted K ways over
training-fold S1 entities (src/v2/crossfit.py): training rows get their
out-of-fold model's score, every other row the mean of the K models, so
prune_p has one distribution in train, holdout and test. Cutoff: per training
country, the highest cutoff losing at most PRUNE_MAX_RECALL_LOSS of the tune
fold's retrieved true pairs; each training country uses its own cutoff and any
label without training data uses the least strict (lowest) one. Also reports a
per-S1 cap of PRUNE_CAP candidates by pruner score. Output
OUT_DIR/<tag>/pairs_pruned.parquet (+ prune_p).
"""
from __future__ import annotations

import argparse
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.v2 import config as C
from src.v2.common import add_context, channel_columns, labels, load_records
from src.v2.crossfit import cross_fit, mean_predict

PRUNE_CONTEXT = ["best", "q_best", "margin", "q_cnt", "s_cnt", "s_first", "q_empty_addr", "q_non_latin", "q_is_s3"]
PRUNE_CAP = 20


def training_rows(candidates: np.ndarray, y: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """All positives plus a seeded sample of negatives, at most `cap` rows."""
    pos, neg = np.flatnonzero(candidates & y), np.flatnonzero(candidates & ~y)
    n_neg = max(0, min(len(neg), cap - len(pos)))
    keep = np.zeros(len(y), bool)
    keep[pos] = True
    keep[np.random.default_rng(seed).choice(neg, n_neg, replace=False)] = True
    return keep


def pruner_model(workers: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63, min_child_samples=100,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=C.SEED,
                              n_jobs=workers, verbosity=-1)


def prune_frame(pairs: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    df = add_context(pairs, len(rec))
    q = df.q.to_numpy()
    return df.assign(q_empty_addr=(rec.addr.to_numpy()[q] == "").astype(np.float32),
                     q_non_latin=rec.non_latin.to_numpy()[q].astype(np.float32),
                     q_is_s3=(rec.source.to_numpy()[q] == 3).astype(np.float32))


def cap_per_s1(s: np.ndarray, p: np.ndarray, cap: int) -> np.ndarray:
    """True for each S1's top-`cap` pairs by p (ties by row order)."""
    order = np.lexsort((-p, s))
    s_sorted = s[order]
    start = np.r_[0, np.flatnonzero(s_sorted[1:] != s_sorted[:-1]) + 1]
    rank = np.arange(len(s)) - np.repeat(start, np.diff(np.r_[start, len(s)]))
    keep = np.zeros(len(s), bool)
    keep[order[rank < cap]] = True
    return keep


def apply_cutoffs(p: np.ndarray, s_country: np.ndarray, cutoffs: dict[str, float], default: float) -> np.ndarray:
    """keep = p >= the country's own cutoff (default for labels without one)."""
    thr = np.full(len(p), default, dtype=np.float64)
    for c, v in cutoffs.items():
        thr[s_country == c] = v
    return p >= thr


def holdout_report(log, rec, s_fold, s_country, y, keep, label) -> None:
    for country in sorted(set(s_country[s_fold == "holdout"])):
        h = (s_fold == "holdout") & (s_country == country)
        n_hold = max(((rec.source.to_numpy() == 1) & (rec.fold.to_numpy() == "holdout")
                      & (rec.country.to_numpy() == country)).sum(), 1)
        log(f"  {label} {country} holdout: retrieved true pairs kept {(keep[h & y]).sum() / max((h & y).sum(), 1):.5f}; "
            f"candidates per S1 {h.sum() / n_hold:.2f} -> {(h & keep).sum() / n_hold:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--workers", type=int, default=C.WORKERS)
    ap.add_argument("--pruner", default=f"{C.MODEL_DIR}/pruner_train.pkl", help="saved pruner used for --tag test")
    a = ap.parse_args()
    out = C.tag_dir(a.tag)
    log = C.log_to(f"{out}/prune.log")
    t0 = C.stage_start(log, f"prune {a.tag}")
    rec = load_records(a.tag, ["source", "country", "addr", "non_latin", "fold", "xgroup", "owner"])
    pairs = pd.read_parquet(f"{out}/pairs_block.parquet")
    df = prune_frame(pairs, rec)
    prune_features = channel_columns(pairs) + PRUNE_CONTEXT
    if a.tag == "test":
        with open(a.pruner, "rb") as f:
            saved = pickle.load(f)
        if saved["features"] != prune_features:
            raise SystemExit(f"test pairs have {prune_features}, pruner was trained on {saved['features']}")
    X = df[prune_features].to_numpy(np.float32)
    s = df.s.to_numpy()
    s_fold, s_country, s_group = rec.fold.to_numpy()[s], rec.country.to_numpy()[s], rec.xgroup.to_numpy()[s]
    path = f"{C.model_dir()}/pruner_{'dev' if a.tag == 'dev' else 'train'}.pkl"

    if a.tag != "test":
        y = labels(df, rec.owner.to_numpy())
        fit_pool = training_rows(s_fold == "train", y, C.PRUNE_TRAIN_ROWS, C.SEED)
        log(f"pruner training pool: {fit_pool.sum():,} rows ({y[fit_pool].sum():,} positive) "
            f"of {(s_fold == 'train').sum():,} training rows")

        def fit(rows, g):
            return pruner_model(a.workers).fit(X[rows & fit_pool], y[rows & fit_pool])

        p, models = cross_fit(X, s_fold == "train", s_group, C.XFIT_K, fit, log, "pruner")
        cut = {}
        for country in sorted(set(s_country[s_fold == "tune"])):
            m = (s_fold == "tune") & (s_country == country) & y
            cut[country] = float(np.quantile(p[m], C.PRUNE_MAX_RECALL_LOSS)) if m.any() else 0.0
        cutoff = min(cut.values())
        for country in cut:
            m = (s_fold == "tune") & (s_country == country) & y
            log(f"cutoff {country}: {cut[country]:.5f} (tune recall loss {(p[m] < cut[country]).mean():.5f}); "
                f"the single lowest cutoff {cutoff:.5f} would lose {(p[m] < cutoff).mean():.5f}")
        log(f"labels without training data use {cutoff:.5f}")
        with open(path, "wb") as f:
            pickle.dump({"models": models, "cutoffs": cut, "cutoff": cutoff, "features": prune_features}, f)
        holdout_report(log, rec, s_fold, s_country, y, p >= cutoff, "single lowest cutoff")
        keep = apply_cutoffs(p, s_country, cut, cutoff)
        holdout_report(log, rec, s_fold, s_country, y, keep, f"per-country budget {C.PRUNE_MAX_RECALL_LOSS:.1%}")
        holdout_report(log, rec, s_fold, s_country, y, cap_per_s1(s, p, PRUNE_CAP), f"cap {PRUNE_CAP}")
    else:
        cutoff = saved["cutoff"]
        p = mean_predict(saved["models"], X)
        keep = apply_cutoffs(p, s_country, saved.get("cutoffs", {}), cutoff)
        log("cutoffs: " + ", ".join(f"{c} {saved.get('cutoffs', {}).get(c, cutoff):.5f}"
                                   + ("" if c in saved.get("cutoffs", {}) else " (no training data)")
                                   for c in sorted(set(s_country))))
        capped = cap_per_s1(s, p, PRUNE_CAP)
        s1_all = rec[rec.source == 1]
        for country, n in s1_all.country.value_counts().items():
            m = s_country == country
            log(f"{country} test: candidates per S1 {m.sum() / n:.2f} -> {(m & keep).sum() / n:.2f} "
                f"(cap {PRUNE_CAP} alternative: {(m & capped).sum() / n:.2f})")

    kept = pairs[keep].assign(prune_p=p[keep].astype(np.float32)).reset_index(drop=True)
    kept.to_parquet(f"{out}/pairs_pruned.parquet", index=False)
    log(f"kept {len(kept):,} of {len(pairs):,} pairs")
    C.stage_end(log, f"prune {a.tag}", t0)


if __name__ == "__main__":
    main()
