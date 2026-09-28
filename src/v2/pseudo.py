"""Self-training for countries without training data (France), first pass only.

  python -m src.v2.pseudo loco            # validation: does it help an unseen country?
  python -m src.v2.pseudo test --base runs/submissions/m3_fast --out runs/submissions/m3_fr_pseudo

A cross-fitted first-pass model scores the unlabelled pairs of the target
country; pairs kept by one-owner with p >= HI become positives, pairs with
p <= LO negatives. The model is refit on the labelled rows plus these pseudo
rows and the target country is re-scored. Only provided files are used: no
labels of the target country, no external data, no leaderboard feedback.

loco: for US->India and India->US, the target's training-fold rows are the
unlabelled pool (their labels are never read); the target's holdout is scored
before and after, with decoding tuned on the source country's tune fold.
test: the pool is the France test pairs; US/India rows of --base are kept and
only France rows are replaced (then validated).
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd

from src.entity_resolution.decision import EntityScorer
from src.entity_resolution.evaluate import run_validator
from src.v2 import config as C
from src.v2.common import labels, load_records
from src.v2.crossfit import mean_predict
from src.v2.decode import blocking_misses, fit_calibration, one_owner
from src.v2.decode import predict as predict_rule
from src.v2.decode import tune as tune_rules
from src.v2.features import GROUPS
from src.v2.train import Scoring, add_house_features

HI, LO = 0.98, 0.02


def load_split(tag: str, house: bool, exclude: set[str]):
    out = C.tag_dir(tag)
    feats = pd.read_parquet(f"{out}/features.parquet")
    if house:
        feats = add_house_features(feats, out)
    names = [c for c in feats.columns if c not in ("q", "s") and c not in exclude]
    return feats[["q", "s"]].copy(), feats[names].to_numpy(np.float32), names


def pseudo_rows(q, p, pool):
    """(row mask, pseudo label) for pool rows: one-owner kept and p >= HI -> 1, p <= LO -> 0."""
    keep = np.zeros(len(p), bool)
    keep[pool] = one_owner(q[pool], p[pool])
    pos = pool & keep & (p >= HI)
    neg = pool & (p <= LO)
    return pos | neg, pos


def loco(a) -> None:
    log = C.log_to(f"{C.tag_dir('train')}/pseudo_loco.log")
    t0 = C.stage_start(log, "pseudo loco")
    rec = load_records("train", ["source", "country", "fold", "xgroup", "owner"])
    pairs, X, names = load_split("train", a.house, a.exclude)
    q, s = pairs.q.to_numpy(), pairs.s.to_numpy()
    y = labels(pairs, rec.owner.to_numpy())
    fold, country = rec.fold.to_numpy(), rec.country.to_numpy()
    s_fold, s_country, s_group = fold[s], country[s], rec.xgroup.to_numpy()[s]
    T = np.bincount(rec.owner.to_numpy()[rec.owner.to_numpy() >= 0], minlength=len(rec))
    s1 = np.flatnonzero((rec.source == 1).to_numpy())
    ents = pd.DataFrame({"s1_id": s1, "country": country[s1], "T": T[s1]})
    scorer = EntityScorer(ents, pd.DataFrame({"s1_id": s, "label": y}))
    tune_ids = ents[fold[ents.s1_id] == "tune"]
    hold_ids = ents[fold[ents.s1_id] == "holdout"]
    rows = []
    for src_c, dst_c in (("US", "India"), ("India", "US")):
        train_rows = (s_fold == "train") & (s_country == src_c)
        es_rows = (s_fold == "tune") & (s_country == src_c)
        pool = (s_fold == "train") & (s_country == dst_c)
        tune_src = tune_ids[tune_ids.country == src_c].s1_id
        hold_dst = hold_ids[hold_ids.country == dst_c].s1_id
        results = {}
        for variant in ("base", "pseudo"):
            y_fit = y.copy()
            fit_rows = train_rows.copy()
            if variant == "pseudo":
                extra, pos = pseudo_rows(q, results["base"]["p"], pool)
                y_fit[extra] = pos[extra]  # pseudo labels replace (never read) true labels of the pool
                fit_rows |= extra
                log(f"  {src_c}->{dst_c}: {extra.sum():,} pseudo rows ({pos.sum():,} positive) from {pool.sum():,} "
                    f"unlabelled {dst_c} rows; pseudo-positive precision (diagnostic only) {y[pos].mean():.4f}")
            sc = Scoring(X, y_fit, s_group, C.SEED, a.workers, 2000, log, names, a.lr)
            p, _ = sc.scores(fit_rows, es_rows, f"{src_c}->{dst_c} {variant}")
            iso = fit_calibration(p[es_rows], y[es_rows])
            pc = iso.predict(p)
            keep = one_owner(q, p)
            m = blocking_misses(T, tune_src.to_numpy(), s, y)
            ch = tune_rules(scorer, scorer.positions(tune_src), s, pc, keep, m, a.rules)
            f = scorer.f(scorer.positions(hold_dst), predict_rule(ch["rule"], ch["param"], s, pc, keep, m))
            results[variant] = {"p": p, "f": f}
            log(f"  {src_c}->{dst_c} {variant}: {ch['rule']} {ch['param']}, {dst_c} holdout F0.5 {f.mean():.4f}")
        d = results["pseudo"]["f"] - results["base"]["f"]
        boot = [d[np.random.default_rng(i).integers(0, len(d), len(d))].mean() for i in range(1000)]
        lo_, hi_ = np.quantile(boot, [0.025, 0.975])
        rows.append((f"{src_c}->{dst_c}", results["base"]["f"].mean(), results["pseudo"]["f"].mean(), d.mean(), lo_, hi_))
        log(f"  {src_c}->{dst_c}: delta {d.mean():+.4f} [{lo_:+.4f}, {hi_:+.4f}]")
    table = pd.DataFrame(rows, columns=["direction", "base", "pseudo", "delta", "ci_low", "ci_high"])
    log("\n" + table.round(4).to_string(index=False))
    C.stage_end(log, "pseudo loco", t0)


def test(a) -> None:
    log = C.log_to(f"{C.tag_dir('test')}/pseudo_test.log")
    t0 = C.stage_start(log, "pseudo test")
    rec = load_records("train", ["source", "country", "fold", "xgroup", "owner"])
    trec = load_records("test", ["entity_id", "country"])
    tr_pairs, Xtr, names = load_split("train", a.house, a.exclude)
    te_pairs, Xte, names_te = load_split("test", a.house, a.exclude)
    assert names == names_te, "train and test feature columns differ"
    s = tr_pairs.s.to_numpy()
    y = labels(tr_pairs, rec.owner.to_numpy())
    s_fold, s_group = rec.fold.to_numpy()[s], rec.xgroup.to_numpy()[s]
    fr = trec.country.to_numpy()[te_pairs.s.to_numpy()] == a.country
    qf, sf, Xf = te_pairs.q.to_numpy()[fr], te_pairs.s.to_numpy()[fr], Xte[fr]
    log(f"{a.country} test pairs: {fr.sum():,}")

    sc = Scoring(Xtr, y, s_group, C.SEED, a.workers, 2000, log, names, a.lr)
    _, models0 = sc.scores(s_fold == "train", s_fold == "tune", "first pass (train rows)")
    p0 = mean_predict(models0, Xf)
    extra, pos = pseudo_rows(qf, p0, np.ones(len(p0), bool))
    log(f"pseudo rows: {extra.sum():,} ({pos.sum():,} positive, {(extra & ~pos).sum():,} negative)")
    # refit on train rows + pseudo rows; pseudo rows are always training rows, spread over the groups
    X2 = np.vstack([Xtr, Xf[extra]])
    y2 = np.r_[y, pos[extra]]
    g2 = np.r_[s_group, (sf[extra] % C.XFIT_K).astype(s_group.dtype)]
    train2 = np.r_[s_fold == "train", np.ones(extra.sum(), bool)]
    tune2 = np.r_[s_fold == "tune", np.zeros(extra.sum(), bool)]
    sc2 = Scoring(X2, y2, g2, C.SEED, a.workers, 2000, log, names, a.lr)
    p_tr, models1 = sc2.scores(train2, tune2, "refit with pseudo rows")
    p_tr = p_tr[:len(y)]
    iso = fit_calibration(p_tr[s_fold == "tune"], y[s_fold == "tune"])
    # decoding tuned for this model on the training tune fold, plus the LOCO offset for unseen countries
    T = np.bincount(rec.owner.to_numpy()[rec.owner.to_numpy() >= 0], minlength=len(rec))
    s1 = np.flatnonzero((rec.source == 1).to_numpy())
    ents = pd.DataFrame({"s1_id": s1, "country": rec.country.to_numpy()[s1], "T": T[s1]})
    scorer = EntityScorer(ents, pd.DataFrame({"s1_id": s, "label": y}))
    tune_ids = ents[rec.fold.to_numpy()[ents.s1_id] == "tune"].s1_id
    m = blocking_misses(T, tune_ids.to_numpy(), s, y)
    ch = tune_rules(scorer, scorer.positions(tune_ids), s, iso.predict(p_tr), one_owner(tr_pairs.q.to_numpy(), p_tr), m, a.rules)
    offset = 0.0
    if os.path.exists(a.matcher):  # the LOCO offset applies only if that run chose the same rule
        with open(a.matcher, "rb") as f:
            saved = pickle.load(f)
        offset = saved.get("unseen_offset", 0.0) if saved.get("rule") == ch["rule"] else 0.0
    rule, param = ch["rule"], ch["param"] + offset
    n_fr = max((trec.country == a.country).sum(), 1)
    p1 = mean_predict(models1, Xf)
    pred = predict_rule(rule, param, sf, iso.predict(p1), one_owner(qf, p1), m)
    before = predict_rule(rule, param, sf, iso.predict(p0), one_owner(qf, p0), m)
    log(f"{a.country}: {rule} {ch['param']} + offset {offset}; predicted matches per S1 {pred.sum() / n_fr:.3f} "
        f"(first pass without pseudo rows: {before.sum() / n_fr:.3f}); singleton rate "
        f"{1 - pd.Series(sf[pred]).nunique() / n_fr:.4f}")

    ids = trec.entity_id.to_numpy()
    matched = pd.DataFrame({"s1": ids[sf[pred]], "c": ids[qf[pred]]}).groupby("s1").c.agg(list)
    os.makedirs(a.out, exist_ok=True)
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        b = pd.read_csv(f"{a.base}/{name}", sep="\t", dtype=str, keep_default_na=False)
        if name == "matching_results.tsv":
            fr_rows = b.source1_entity_id.isin(set(ids[trec.country.to_numpy() == a.country]))
            b.loc[fr_rows, "matched_entity_ids"] = [",".join(matched.get(e, [])) for e in b.source1_entity_id[fr_rows]]
        b.to_csv(f"{a.out}/{name}", sep="\t", index=False)
    ok, report = run_validator(f"{a.out}/matching_results.tsv", f"{a.out}/candidate_pairs.tsv", f"{C.DATA_ROOT}/test")
    log(f"validator passed: {ok}\n{report.strip()}")
    C.stage_end(log, "pseudo test", t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["loco", "test"])
    ap.add_argument("--house", action="store_true")
    ap.add_argument("--exclude-groups", default="")
    ap.add_argument("--rules", default="threshold,expected_f")
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--workers", type=int, default=C.WORKERS)
    ap.add_argument("--country", default="France")
    ap.add_argument("--base", default="runs/submissions/m3_fast", help="test: submission whose other rows are kept")
    ap.add_argument("--matcher", default=f"{C.MODEL_DIR}/matcher.pkl", help="test: decoding rule, parameter, offset")
    ap.add_argument("--out", default="runs/submissions/m3_fr_pseudo")
    a = ap.parse_args()
    a.exclude = {f for g in a.exclude_groups.split(",") if g for f in GROUPS[g]}
    a.rules = [r for r in a.rules.split(",") if r]
    loco(a) if a.mode == "loco" else test(a)


if __name__ == "__main__":
    main()
