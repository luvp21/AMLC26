"""Test-like holdout check of predict-time options (look-alike logit shift, ensemble).

  python -m src.v2.shiftcheck --matcher runs/v2/model/matcher_m3_syn1.pkl \
      --ensemble runs/v2/model/matcher_m3_gap.pkl --shifts 0,1,2.5

Scores the holdout rows of the aug pool (with synthetic next-door look-alikes)
as predict.py scores test: cross-fit mean, second pass, optional look-alike
shift on the raw logit, optional ensemble average, calibration, one-owner and
the saved decoding rule. Reports macro F0.5 per country over holdout S1
(singletons included) and a paired bootstrap against the first variant.
One-owner competes within holdout rows only (a diagnostic, like errors.py).
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from src.v2 import config as C
from src.v2.common import labels, load_records
from src.v2.crossfit import mean_predict
from src.v2.crowd import CROWD_FEATURES
from src.v2.decode import one_owner
from src.v2.decode import predict as predict_rule
from src.v2.house import HOUSE_FEATURES
from src.v2.novel import NOVEL_FEATURES
from src.v2.predict import lookalike_mask
from src.v2.sibling import second_pass_features
from src.v2.train import add_crowd_features, add_house_features, add_novel_features

BOOTSTRAP = 1000


def raw_scores(m: dict, feats: pd.DataFrame, q, s, rec) -> np.ndarray:
    X = feats[m["features"]].to_numpy(np.float32)
    p = mean_predict(m["models"], X)
    if m.get("models2"):
        p = mean_predict(m["models2"], np.hstack([X, second_pass_features(q, s, p, rec, C.WORKERS)]))
    return p


def entity_f05(T, K, TP) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where((K == 0) | (T == 0), 0.0, 1.25 * TP / (0.25 * T + K))
    return np.where((T == 0) & (K == 0), 1.0, f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matcher", required=True, help="decides calibration and decoding (as in predict.py)")
    ap.add_argument("--ensemble", default="", help="second matcher .pkl averaged on raw probabilities")
    ap.add_argument("--shifts", default="0,1,2,2.5")
    ap.add_argument("--tag", default="aug")
    a = ap.parse_args()
    out = C.tag_dir(a.tag)
    rec = load_records(a.tag, ["source", "country", "fold", "owner", "name_core", "addr", "house_num", "postcode"])
    feats = pd.read_parquet(f"{out}/features.parquet")
    fold, owner, country = rec.fold.to_numpy(), rec.owner.to_numpy(), rec.country.to_numpy()
    mats = {"main": a.matcher, **({"ens": a.ensemble} if a.ensemble else {})}
    ms = {}
    for k, path in mats.items():
        with open(path, "rb") as f:
            ms[k] = pickle.load(f)
    need = set().union(*(m["features"] for m in ms.values()))
    if need & set(HOUSE_FEATURES):
        feats = add_house_features(feats, out)
    if need & set(NOVEL_FEATURES):
        feats = add_novel_features(feats, out)
    if need & set(CROWD_FEATURES):
        feats = add_crowd_features(feats, out)
    feats = feats[fold[feats.s.to_numpy()] == "holdout"].reset_index(drop=True)
    q, s = feats.q.to_numpy(), feats.s.to_numpy()
    y = labels(feats, owner)
    m = ms["main"]
    raw = {k: raw_scores(mm, feats, q, s, rec) for k, mm in ms.items()}
    mask = lookalike_mask(feats, out)
    print(f"holdout pairs {len(feats):,}; known-different house numbers {mask.mean():.3%}")

    hold = np.flatnonzero((rec.source == 1).to_numpy() & (fold == "holdout"))
    T = np.bincount(owner[owner >= 0], minlength=len(rec))[hold]
    variants = []
    for shift in [float(x) for x in a.shifts.split(",")]:
        for use_ens in ([False, True] if a.ensemble else [False]):
            p_raw = raw["main"]
            if shift:
                z = np.log(np.clip(p_raw, 1e-9, 1 - 1e-9) / np.clip(1 - p_raw, 1e-9, 1))
                p_raw = np.where(mask, 1 / (1 + np.exp(-(z - shift))), p_raw)
            if use_ens:
                p_raw = (p_raw + raw["ens"]) / 2
            p = m["calibration"].predict(p_raw) if m.get("calibration") is not None else p_raw
            keep = one_owner(q, p_raw)
            pred = predict_rule(m["rule"], m["param"], s, p, keep, m["m"])
            K = np.bincount(s[pred], minlength=len(rec))[hold]
            TP = np.bincount(s[pred & y], minlength=len(rec))[hold]
            variants.append((f"shift {shift:g}" + (" + ensemble" if use_ens else ""), entity_f05(T, K, TP)))

    c_hold = country[hold]
    rng = np.random.default_rng(C.SEED)
    base = variants[0][1]
    rows = []
    for name, f in variants:
        row = {"variant": name}
        for c in sorted(set(c_hold)):
            idx = np.flatnonzero(c_hold == c)
            d = f[idx] - base[idx]
            boots = [d[rng.integers(0, len(idx), len(idx))].mean() for _ in range(BOOTSTRAP)]
            row[f"F_{c}"] = f[idx].mean()
            row[f"d_{c}"] = d.mean()
            row[f"ci_{c}"] = f"[{np.quantile(boots, 0.025):+.4f}, {np.quantile(boots, 0.975):+.4f}]"
        rows.append(row)
    pd.set_option("display.width", 200)
    print(f"test-like holdout ({a.tag}), decoding of {a.matcher}; deltas vs '{variants[0][0]}'")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
