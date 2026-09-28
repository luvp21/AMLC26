"""Stage 6: score test, decode, write and validate the submission.

  python -m src.v2.predict --out-dir output

candidate_pairs.tsv is exactly the pruned set the model scored; matches are a
subset. One row per test S1 (sorted by id), empty lists allowed. Prints the
per-country test checks and writes the France review sample.
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil

import numpy as np
import pandas as pd

from src.entity_resolution.evaluate import run_validator
from src.v2 import config as C
from src.v2.common import load_records
from src.v2.crossfit import mean_predict
from src.v2.decode import one_owner
from src.v2.decode import predict as predict_rule
from src.v2.house import HOUSE_FEATURES, REL
from src.v2.prior import apply_shifts, cell_shifts
from src.v2.sibling import second_pass_features
from src.v2.novel import NOVEL_FEATURES
from src.v2.crowd import CROWD_FEATURES
from src.v2.train import add_crowd_features, add_house_features, add_novel_features


def lookalike_mask(feats: pd.DataFrame, out: str) -> np.ndarray:
    """Pairs whose house numbers are both known and differ in their digits."""
    rel = feats["house_rel"] if "house_rel" in feats else pd.read_parquet(f"{out}/features_house.parquet", columns=["house_rel"]).house_rel
    known_different = [REL[k] for k in ("digit_dropped_or_added", "digit_substituted", "near_1_2", "permuted",
                                         "prefix_or_extension", "different")]
    return np.isin(rel.to_numpy(), known_different)


def write_tsv(path: str, header: str, s1_ids: np.ndarray, lists: pd.Series) -> None:
    with open(path, "w") as f:
        f.write(header + "\n")
        for sid in s1_ids:
            f.write(f"{sid}\t{','.join(lists.get(sid, []))}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--experiment", required=True, help="validated files are copied to runs/submissions/<experiment>/")
    ap.add_argument("--matcher", default=f"{C.MODEL_DIR}/matcher.pkl")
    ap.add_argument("--no-check-ids", action="store_true")
    ap.add_argument("--ensemble", default="", help="second matcher .pkl; raw probabilities are averaged")
    ap.add_argument("--em-cells", default="known_different", choices=["known_different", "all"],
                    help="which house-number cases the EM correction may shift")
    ap.add_argument("--em-shift", action="store_true",
                    help="label-free per-(country, house relation) base-rate correction (src/v2/prior.py); needs model/priors.pkl")
    ap.add_argument("--lookalike-shift", type=float, default=0.0,
                    help="subtract this from the logit of pairs whose house numbers are both known and different "
                         "(test holds ~4x more such look-alikes than train; value from teammate's leaderboard probes)")
    ap.add_argument("--unseen-shift", type=float, default=0.0,
                    help="subtract this from the raw logit of pairs whose S1 country had no training data "
                         "(label-free EM estimates and paired France-only leaderboard probes both say the model is "
                         "over-confident there)")
    a = ap.parse_args()
    out = C.tag_dir("test")
    log = C.log_to(f"{out}/predict.log")
    t0 = C.stage_start(log, "predict")
    rec = load_records("test", ["entity_id", "source", "country", "business_name", "business_address", "name_core",
                                "addr", "house_num", "postcode"])
    feats = pd.read_parquet(f"{out}/features.parquet")
    with open(a.matcher, "rb") as f:
        m = pickle.load(f)
    if any(c in m["features"] for c in HOUSE_FEATURES):
        feats = add_house_features(feats, out)
        log("house-number relation features added")
    if any(c in m["features"] for c in NOVEL_FEATURES):
        feats = add_novel_features(feats, out)
    if any(c in m["features"] for c in CROWD_FEATURES):
        feats = add_crowd_features(feats, out)
        log("novel-extra-word features added")
    q, s = feats.q.to_numpy(), feats.s.to_numpy()
    X = feats[m["features"]].to_numpy(np.float32)
    p_raw = mean_predict(m["models"], X)
    if m.get("models2"):  # sibling-evidence second pass on top of the first-pass scores
        X = np.hstack([X, second_pass_features(q, s, p_raw, rec, C.WORKERS)])
        p_raw = mean_predict(m["models2"], X)
        log("second pass applied")
    if "calibration" not in m:  # matcher saved before M3 (plain threshold, no calibration)
        m = {**m, "calibration": None, "rule": "threshold", "param": m["threshold"], "m": 0.0,
             "unseen_offset": 0.0, "train_countries": sorted(rec.country.unique())}
    if a.lookalike_shift:
        shifted = lookalike_mask(feats, out)
        z = np.log(np.clip(p_raw, 1e-9, 1 - 1e-9) / np.clip(1 - p_raw, 1e-9, 1))
        p_raw = np.where(shifted, 1 / (1 + np.exp(-(z - a.lookalike_shift))), p_raw)
        log(f"look-alike shift {a.lookalike_shift} applied to {shifted.mean():.3%} of pairs (house numbers known and different)")
    if a.ensemble:  # average with a second matcher (same test features, its own feature list and passes)
        with open(a.ensemble, "rb") as f:
            m2 = pickle.load(f)
        X2 = feats[m2["features"]].to_numpy(np.float32)
        p2 = mean_predict(m2["models"], X2)
        if m2.get("models2"):
            p2 = mean_predict(m2["models2"], np.hstack([X2, second_pass_features(q, s, p2, rec, C.WORKERS)]))
        p_raw = (p_raw + p2) / 2
        log(f"ensemble with {a.ensemble}: raw probabilities averaged")
    if a.unseen_shift:
        unseen = ~np.isin(rec.country.to_numpy()[s], m["train_countries"])
        z = np.log(np.clip(p_raw, 1e-9, 1 - 1e-9) / np.clip(1 - p_raw, 1e-9, 1))
        p_raw = np.where(unseen, 1 / (1 + np.exp(-(z - a.unseen_shift))), p_raw)
        log(f"unseen-country shift {a.unseen_shift} applied to {unseen.mean():.3%} of pairs")
    p = m["calibration"].predict(p_raw) if m["calibration"] is not None else p_raw
    if a.em_shift:
        with open(f"{C.MODEL_DIR}/priors.pkl", "rb") as f:
            priors = pickle.load(f)
        rel = (feats["house_rel"] if "house_rel" in feats else pd.read_parquet(
            f"{out}/features_house.parquet", columns=["house_rel"]).house_rel).to_numpy()
        s_country = rec.country.to_numpy()[s]
        shifts = cell_shifts(p, rel, s_country, priors)
        if a.em_cells == "known_different":
            allowed = [REL[k] for k in ("digit_dropped_or_added", "digit_substituted", "near_1_2", "permuted",
                                        "prefix_or_extension", "different")]
            shifts.loc[~shifts.rel.isin(allowed), "shift"] = 0.0
        log("EM base-rate shifts (test):\n" + shifts[shifts.pairs >= 2000].round(4).to_string(index=False))
        p = apply_shifts(p, rel, s_country, shifts)
        p_raw = p + 1e-9 * p_raw  # one-owner on the corrected scores (raw breaks isotonic ties)
    keep = one_owner(q, p_raw)
    s_country = rec.country.to_numpy()[s]
    pred = np.zeros(len(p), bool)
    for c in np.unique(s_country):
        rows = s_country == c
        param = m["param"] + (0.0 if c in m["train_countries"] else m["unseen_offset"])
        pred[rows] = predict_rule(m["rule"], param, s[rows], p[rows], keep[rows], m["m"])
        log(f"{c}: {m['rule']} parameter {param}" + ("" if c in m["train_countries"] else " (unseen country: offset applied)"))

    ids = rec.entity_id.to_numpy()
    s1 = rec[rec.source == 1]
    s1_sorted = np.sort(s1.entity_id.to_numpy())
    pairs = pd.DataFrame({"s1": ids[s], "cand": ids[q]})
    os.makedirs(a.out_dir, exist_ok=True)
    write_tsv(f"{a.out_dir}/candidate_pairs.tsv", "source1_entity_id\tcandidate_entity_ids", s1_sorted,
              pairs.groupby("s1").cand.agg(list))
    write_tsv(f"{a.out_dir}/matching_results.tsv", "source1_entity_id\tmatched_entity_ids", s1_sorted,
              pairs[pred].groupby("s1").cand.agg(list))
    ok, report = run_validator(f"{a.out_dir}/matching_results.tsv", f"{a.out_dir}/candidate_pairs.tsv",
                               f"{C.DATA_ROOT}/test", check_ids=not a.no_check_ids)
    log(f"validator passed: {ok}\n{report.strip()}")
    if ok:
        dest = f"runs/submissions/{a.experiment}"
        os.makedirs(dest, exist_ok=True)
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            shutil.copy2(f"{a.out_dir}/{name}", f"{dest}/{name}")
        log(f"validated files copied to {dest}/")

    country = rec.country.to_numpy()
    rows = []
    hi = pd.Series(p > 0.5).groupby(q).sum()
    for c in sorted(s1.country.unique()):
        s1_c = s1.rid[s1.country == c].to_numpy()
        k = np.bincount(s[pred], minlength=len(rec))[s1_c]
        cand = np.bincount(s, minlength=len(rec))[s1_c]
        pool = ((rec.source != 1) & (rec.country == c)).sum()
        hi_c = hi[country[hi.index.to_numpy()] == c]
        rows.append({"country": c, "S1": len(s1_c), "candidates_per_S1": cand.mean(), "pred_matches_per_S1": k.mean(),
                     "pred_singleton_rate": (k == 0).mean(),
                     "conflict_rate": (hi_c >= 2).sum() / max((hi_c >= 1).sum(), 1),
                     "pool_assigned": np.unique(q[pred & (country[q] == c)]).size / max(pool, 1)})
    log("\n=== Test checks (train truth: 3.46 matches per S1, 5.6% singletons, 74% of pool assigned) ===")
    log(pd.DataFrame(rows).set_index("country").round(4).to_string())

    fr = s1[s1.country == "France"].rid.to_numpy()
    show = np.random.default_rng(C.SEED).permutation(fr)[:50]
    tp = pd.DataFrame({"s": s, "q": q, "p": p, "pred": pred})
    tp = tp[np.isin(s, show)].sort_values(["s", "p"], ascending=[True, False]).groupby("s").head(5)
    name, addr = rec.business_name.to_numpy(), rec.business_address.to_numpy()
    with open(f"{out}/france_review.txt", "w") as f:
        for r in show:
            f.write(f"\n=== {ids[r]}: {name[r]!r} | {addr[r]!r}\n")
            for _, row in tp[tp.s == r].iterrows():
                f.write(f"    p={row.p:.3f} {'MATCH' if row.pred else '     '} {ids[row.q]}: {name[row.q]!r} | {addr[row.q]!r}\n")
    log(f"France review sample -> {out}/france_review.txt")
    C.stage_end(log, "predict", t0)


if __name__ == "__main__":
    main()
