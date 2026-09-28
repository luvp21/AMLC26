"""Error mining on the holdout (M4): which true matches the model rejects and
which false ones it accepts, with the raw records side by side.

  python -m src.v2.errors --matcher runs/v2/model/matcher_m3_all.pkl --n 300

Scores holdout rows with the saved cross-fit models (mean, as on test), applies
the saved calibration, one-owner and decoding rule, then samples false
negatives (true pair retrieved but not predicted), false positives, and missed
pairs (never retrieved), and buckets each by a heuristic noise category. Writes
runs/v2/train/errors_<name>.txt and prints the category table with the F0.5
loss each category costs. Holdout only; nothing feeds training.
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.entity_resolution.evaluate import f05
from src.v2 import config as C
from src.v2.common import labels, load_records
from src.v2.crossfit import mean_predict
from src.v2.decode import one_owner
from src.v2.decode import predict as predict_rule
from src.v2.house import HOUSE_FEATURES, REL, relation
from src.v2.novel import NOVEL_FEATURES, novel_words
from src.v2.prior import cell_shifts
from src.v2.sibling import second_pass_features
from src.v2.crowd import CROWD_FEATURES
from src.v2.train import add_crowd_features, add_house_features, add_novel_features

COLS = ["entity_id", "source", "country", "fold", "owner", "business_name", "business_address", "name_core", "addr",
        "house_num", "postcode", "non_latin", "name_alias", "name"]


def category(q, s) -> str:
    """Noise type of a (query, S1) pair, prefixed with the novel-word and house-number signals."""
    novel = bool(novel_words(q["name_core"], s["name"])[0] or novel_words(s["name_core"], q["name"])[0])
    rel = {v: k for k, v in REL.items()}[relation(q.house_num, s.house_num)]
    return f"{'novel' if novel else 'no_novel'}|{rel}|{_category(q, s)}"


def _category(q, s) -> str:
    """Heuristic noise type for a (query, S1) record pair."""
    name_sim = fuzz.token_set_ratio(q.name_core, s.name_core)
    addr_sim = fuzz.token_set_ratio(q.addr, s.addr) if q.addr and s.addr else -1
    if q.non_latin:
        return "non_latin_name"
    if not q.addr:
        return "empty_address" + ("_name_diff" if name_sim < 80 else "")
    if name_sim < 40:
        return "scrambled_or_trade_name"
    if q.house_num and s.house_num and q.house_num != s.house_num:
        return "house_number_changed" if name_sim >= 80 else "house_and_name_changed"
    if addr_sim != -1 and addr_sim < 60:
        return "address_very_different"
    if name_sim < 80:
        return "name_changed"
    if q.name_core == s.name_core:
        return "same_name_address_noise"
    return "typo_or_small_edit"


P_BUCKETS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0001]  # last bucket includes p = 1 (rejected by one-owner)


def fn_table(fn_rows, q, s, p, R, gain) -> None:
    """Wrong rejects (true pair retrieved, not predicted): share of their total F0.5 loss
    per noise category x calibrated-probability bucket."""
    rows = [(_category(R.loc[q[i]], R.loc[s[i]]), p[i], gain(s[i])) for i in fn_rows]
    df = pd.DataFrame(rows, columns=["category", "p", "loss"])
    df["p_bucket"] = pd.cut(df.p, P_BUCKETS, include_lowest=True, right=False)
    total = df.loss.sum()
    share = df.pivot_table(index="category", columns="p_bucket", values="loss", aggfunc="sum",
                           fill_value=0.0, observed=False) / total
    share["ALL"] = share.sum(axis=1)
    share.loc["ALL"] = share.sum()
    counts = df.groupby("category").size()
    pd.set_option("display.width", 200)
    print(f"wrong rejects: {len(df):,}; summed F0.5 loss {total:.4f} (per-error fix gains, approx. additive)")
    print("share of wrong-reject loss by category x calibrated p bucket:")
    print(share.sort_values("ALL", ascending=False).round(3).to_string())
    print("count per category:\n" + counts.sort_values(ascending=False).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matcher", default=f"{C.MODEL_DIR}/matcher.pkl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--fn-buckets", action="store_true",
                    help="only the wrong-reject loss table by noise category x probability bucket")
    ap.add_argument("--em-check", action="store_true", help="print EM base-rate shifts on the holdout (should be ~0)")
    a = ap.parse_args()
    out = C.tag_dir("train")
    name = os.path.basename(a.matcher).replace(".pkl", "")
    rec = load_records("train", COLS)
    feats = pd.read_parquet(f"{out}/features.parquet")
    with open(a.matcher, "rb") as f:
        m = pickle.load(f)
    if any(c in m["features"] for c in HOUSE_FEATURES):
        feats = add_house_features(feats, out)
    if any(c in m["features"] for c in NOVEL_FEATURES):
        feats = add_novel_features(feats, out)
    if any(c in m["features"] for c in CROWD_FEATURES):
        feats = add_crowd_features(feats, out)
    fold, owner = rec.fold.to_numpy(), rec.owner.to_numpy()
    s_all = feats.s.to_numpy()
    rows = np.flatnonzero(fold[s_all] == "holdout")
    hf = feats.iloc[rows].reset_index(drop=True)
    q, s = hf.q.to_numpy(), hf.s.to_numpy()
    y = labels(hf, owner)
    Xh = hf[m["features"]].to_numpy(np.float32)
    p_raw = mean_predict(m["models"], Xh)
    if m.get("models2"):  # the second pass needs first-pass scores of all candidates of these S1s
        p_raw = mean_predict(m["models2"], np.hstack([Xh, second_pass_features(q, s, p_raw, rec, C.WORKERS)]))
    p = m["calibration"].predict(p_raw) if m.get("calibration") is not None else p_raw
    if a.em_check:
        with open(f"{C.MODEL_DIR}/priors.pkl", "rb") as f:
            priors = pickle.load(f)
        rel = hf["house_rel"].to_numpy() if "house_rel" in hf else pd.read_parquet(
            f"{out}/features_house.parquet", columns=["house_rel"]).house_rel.to_numpy()[rows]
        sh = cell_shifts(p, rel, rec.country.to_numpy()[s], priors)
        print("EM base-rate shifts on the HOLDOUT (train mix; should be close to 0):")
        print(sh[sh.pairs >= 2000].round(4).to_string(index=False))
        return
    keep = one_owner(q, p_raw)  # competition within holdout rows only (a diagnostic, not the reported score)
    pred = predict_rule(m.get("rule", "threshold"), m.get("param", m.get("threshold")), s, p, keep, m.get("m", 0.0))

    hold = rec[(rec.source == 1) & (rec.fold == "holdout")]
    T = np.bincount(owner[owner >= 0], minlength=len(rec))
    K = np.bincount(s[pred], minlength=len(rec))
    TP = np.bincount(s[pred & y], minlength=len(rec))
    fn_rows, fp_rows = np.flatnonzero(y & ~pred), np.flatnonzero(pred & ~y)
    retrieved = set(zip(q[y].tolist(), s[y].tolist()))
    hold_ids = set(hold.rid)
    true_q = rec[(rec.source != 1) & rec.owner.isin(hold_ids)]
    missed = [(qq, ss) for qq, ss in zip(true_q.rid, true_q.owner) if (qq, ss) not in retrieved]

    def f05_delta(sid, dTP, dK):
        t, k, tp = T[sid], K[sid], TP[sid]
        base = 1.0 if t == 0 and k == 0 else (0.0 if k == 0 or t == 0 else 1.25 * tp / (0.25 * t + k))
        k2, tp2 = k + dK, tp + dTP
        new = 1.0 if t == 0 and k2 == 0 else (0.0 if k2 == 0 or t == 0 else 1.25 * tp2 / (0.25 * t + k2))
        return new - base

    R = rec.set_index("rid")
    if a.fn_buckets:
        fn_table(fn_rows, q, s, p, R, lambda ss: f05_delta(ss, 1, 1) / len(hold))
        return
    table, lines = [], []
    rng = np.random.default_rng(C.SEED)
    for kind, items in (("false_negative", [(q[i], s[i]) for i in fn_rows]),
                        ("false_positive", [(q[i], s[i]) for i in fp_rows]), ("never_retrieved", missed)):
        # loss per category over ALL items of this kind (exact F0.5 gain if that one error were fixed)
        for qq, ss in items:
            dTP, dK = (1, 1) if kind != "false_positive" else (0, -1)
            table.append((kind, category(R.loc[qq], R.loc[ss]), f05_delta(ss, dTP, dK) / len(hold)))
        pick = rng.permutation(len(items))[:a.n]
        lines.append(f"\n########## {kind}: {len(items):,} in holdout, showing {len(pick)}")
        for j in pick:
            qq, ss = items[j]
            qr, sr = R.loc[qq], R.loc[ss]
            i_p = np.flatnonzero((q == qq) & (s == ss))
            pp = f"p={p[i_p[0]]:.3f}" if len(i_p) else "not a candidate"
            lines.append(f"[{category(qr, sr)}] {pp} {qr.country}\n"
                         f"   S1 {sr.entity_id}: {sr.business_name!r} | {sr.business_address!r}\n"
                         f"   S{qr.source} {qr.entity_id}: {qr.business_name!r} | {qr.business_address!r}")
    df = pd.DataFrame(table, columns=["kind", "category", "f05_gain_if_fixed"])
    summary = df.groupby(["kind", "category"]).agg(count=("category", "size"), f05_gain_if_fixed=("f05_gain_if_fixed", "sum"))
    summary = summary.sort_values("f05_gain_if_fixed", ascending=False).head(40)
    macro = f05(T[hold.rid], K[hold.rid], TP[hold.rid]).mean()
    text = (f"holdout S1: {len(hold):,}; matcher {name}; macro F0.5 with holdout-only competition {macro:.4f}\n"
            + summary.round(5).to_string())
    print(text)
    with open(f"{out}/errors_{name}.txt", "w") as f:
        f.write(text + "\n" + "\n".join(lines))
    print(f"examples -> {out}/errors_{name}.txt")


if __name__ == "__main__":
    main()
