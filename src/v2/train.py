"""Stage 5: train the matcher, decode, evaluate on the holdout.

  python -m src.v2.train --tag train --experiment v2_m1 [--compare-to v2_prev] [--seeds 42,43,44]

Cross-fitted LightGBM (K = XFIT_K groups of training-fold S1 entities, early
stopping on tune rows): training rows get their out-of-fold model's score,
tune/holdout/test rows the mean of the K models, so one-owner decoding and
threshold tuning compare the same kind of probability everywhere. Decoding
(src/v2/decode.py): isotonic calibration on tune rows, one-owner, then the
better of a threshold and expected-F0.5 selection on the tune fold. LOCO runs
the same procedure with one country's rows only and yields the unseen-country
offset. --exclude-groups leaves feature groups out for ablation.
Logs S, components, error decomposition, bootstrap vs --compare-to.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.entity_resolution.decision import EntityScorer, paired_bootstrap, s_components
from src.entity_resolution.evaluate import error_decomposition, evaluate
from src.entity_resolution.harness_io import append_experiment, experiments_csv_path
from src.v2 import config as C
from src.v2.common import labels, load_records
from src.v2.crossfit import cross_fit
from src.v2.decode import blocking_misses, fit_calibration, one_owner
from src.v2.decode import predict as predict_rule
from src.v2.decode import tune as tune_rules
from src.v2.features import CATEGORICAL, GROUPS
from src.v2.house import HOUSE_CATEGORICAL, HOUSE_FEATURES
from src.v2.crowd import CROWD_FEATURES
from src.v2.novel import NOVEL_FEATURES
from src.v2.sibling import SECOND_PASS_NAMES, second_pass_features

PER_ENTITY_DIR = "runs/perentity"


def make_model(seed: int, rounds: int, workers: int, lr: float = 0.05) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(objective="binary", learning_rate=lr, n_estimators=rounds, num_leaves=127,
                              min_child_samples=100, colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                              random_state=seed, n_jobs=workers, verbosity=-1)


class Scoring:
    def __init__(self, X, y, groups, seed, workers, max_rounds, log, features, lr=0.05):
        self.X, self.y, self.groups, self.lr = X, y, groups, lr
        self.seed, self.workers, self.max_rounds, self.log = seed, workers, max_rounds, log
        self.cat = [features.index(c) for c in CATEGORICAL + HOUSE_CATEGORICAL if c in features]

    def _fit(self, rows, es_rows, lr):
        m = make_model(self.seed, self.max_rounds, self.workers, lr)
        m.fit(self.X[rows], self.y[rows], categorical_feature=self.cat,
              eval_set=[(self.X[es_rows], self.y[es_rows])], callbacks=[lgb.early_stopping(100, verbose=False)])
        return m

    def scores(self, train_rows, es_rows, tag):
        """(p for every row, the K models). If any model's best round is within
        100 of the cap, refit all K with learning rate 0.08."""
        t0 = time.time()
        for lr in sorted({self.lr, 0.08}):
            p, models = cross_fit(self.X, train_rows, self.groups, C.XFIT_K,
                                  lambda rows, g: self._fit(rows, es_rows, lr), self.log, tag)
            rounds = [m.best_iteration_ or self.max_rounds for m in models]
            if max(rounds) < self.max_rounds - 100 or lr == 0.08:
                break
            self.log(f"  {tag}: best round {max(rounds)} within 100 of the cap; refitting with lr 0.08")
        self.log(f"  {tag}: lr {lr}, rounds {rounds}, {train_rows.sum():,} training rows, {time.time() - t0:.0f}s")
        return p, models


def add_addon(feats: pd.DataFrame, out: str, file: str, names: list[str]) -> pd.DataFrame:
    """Append add-on feature columns (rows must align with features.parquet)."""
    add = pd.read_parquet(f"{out}/{file}")
    if len(add) != len(feats) or not ((add.q.to_numpy() == feats.q.to_numpy()).all()
                                      and (add.s.to_numpy() == feats.s.to_numpy()).all()):
        raise SystemExit(f"{file} is not aligned with features.parquet: rebuild it")
    return pd.concat([feats, add[names]], axis=1)


def add_house_features(feats: pd.DataFrame, out: str) -> pd.DataFrame:
    return add_addon(feats, out, "features_house.parquet", HOUSE_FEATURES)


def add_novel_features(feats: pd.DataFrame, out: str) -> pd.DataFrame:
    return add_addon(feats, out, "features_novel.parquet", NOVEL_FEATURES)


def add_crowd_features(feats: pd.DataFrame, out: str) -> pd.DataFrame:
    return add_addon(feats, out, "features_crowd.parquet", CROWD_FEATURES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="train", choices=["train", "dev", "aug"])
    ap.add_argument("--no-synth-train", action="store_true",
                    help="aug tag: keep synthetic look-alikes out of training (baseline for the test-like holdout)")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--config-diff", default="")
    ap.add_argument("--compare-to", default="")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--max-rounds", type=int, default=5000)
    ap.add_argument("--no-loco", action="store_true")
    ap.add_argument("--exclude-groups", default="", help=f"feature groups to leave out: {sorted(GROUPS)}")
    ap.add_argument("--rules", default="threshold,expected_f", help="decoding rules to compare on the tune fold")
    ap.add_argument("--second-pass", action="store_true", help="sibling-evidence second pass (src/v2/sibling.py)")
    ap.add_argument("--house", action="store_true", help="add house-number relation features (src/v2/house.py)")
    ap.add_argument("--novel", action="store_true", help="add novel-extra-word features (src/v2/novel.py)")
    ap.add_argument("--crowd", action="store_true", help="add name-crowd features (src/v2/crowd.py)")
    ap.add_argument("--lr", type=float, default=0.05, help="learning rate to start from (0.08 = no refit when near the cap)")
    ap.add_argument("--workers", type=int, default=C.WORKERS)
    a = ap.parse_args()
    out = C.tag_dir(a.tag)
    log = C.log_to(f"{out}/train_{a.experiment}.log")
    t_all = time.time()

    t_stage = C.stage_start(log, f"train {a.experiment}")
    cols = ["source", "country", "fold", "xgroup", "owner", "name_core", "addr", "house_num", "postcode"]
    rec = load_records(a.tag, cols + (["synthetic"] if a.tag == "aug" else []))
    feats = pd.read_parquet(f"{out}/features.parquet")
    if a.house:
        feats = add_house_features(feats, out)
    if a.novel:
        feats = add_novel_features(feats, out)
    if a.crowd:
        feats = add_crowd_features(feats, out)
    owner, fold, country = rec.owner.to_numpy(), rec.fold.to_numpy(), rec.country.to_numpy()
    q, s = feats.q.to_numpy(), feats.s.to_numpy()
    y = labels(feats, owner)
    excluded = {f for g in a.exclude_groups.split(",") if g for f in GROUPS[g]}
    FEATURES = [c for c in feats.columns if c not in ("q", "s") and c not in excluded]
    rules = [r for r in a.rules.split(",") if r]
    log(f"features ({len(FEATURES)}): {FEATURES}")
    X = feats[FEATURES].to_numpy(np.float32)
    s_fold, s_country, s_group = fold[s], country[s], rec.xgroup.to_numpy()[s]
    if a.tag == "aug":
        syn_q = rec.synthetic.to_numpy()[q]
        log(f"synthetic look-alike pairs: {syn_q.sum():,} (train-fold {(syn_q & (s_fold == 'train')).sum():,}, "
            f"holdout {(syn_q & (s_fold == 'holdout')).sum():,}); in training: {not a.no_synth_train}")
        if a.no_synth_train:  # a training fold of "train" rows without the synthetic ones
            s_fold = np.where(syn_q & (s_fold == "train"), "synthetic_excluded", s_fold)
    del feats

    in_scope = np.zeros(len(rec), bool)
    if a.tag == "dev":
        in_scope[np.unique(pd.read_parquet(f"{out}/pairs_block.parquet", columns=["s"]).s)] = True
    else:
        in_scope[(rec.source == 1).to_numpy()] = True
    T = np.bincount(owner[owner >= 0], minlength=len(rec))
    s1_rids = np.flatnonzero(in_scope)
    ents = pd.DataFrame({"s1_id": s1_rids, "country": country[s1_rids], "T": T[s1_rids]})
    scorer = EntityScorer(ents, pd.DataFrame({"s1_id": s, "label": y}))
    tune = ents[fold[ents.s1_id] == "tune"]
    hold = ents[fold[ents.s1_id] == "holdout"].reset_index(drop=True)
    log(f"{a.experiment} on {a.tag}: {len(X):,} pairs, {y.sum():,} positive; entities tune {len(tune):,} holdout {len(hold):,}")

    tune_rows = s_fold == "tune"

    def decode_with(p_raw, fit_rows, tune_ids, rules):
        """Calibrate on `fit_rows` (tune rows), one-owner on raw p, tune the rule on `tune_ids`."""
        iso = fit_calibration(p_raw[fit_rows], y[fit_rows])
        p_cal = iso.predict(p_raw)
        keep = one_owner(q, p_raw)
        m = blocking_misses(T, tune_ids.to_numpy(), s, y)
        choice = tune_rules(scorer, scorer.positions(tune_ids), s, p_cal, keep, m, rules)
        pred = predict_rule(choice["rule"], choice["param"], s, p_cal, keep, m)
        return {"iso": iso, "p": p_cal, "keep": keep, "m": m, "pred": pred, **choice}

    def two_pass(sc, seed, train_rows, es_rows, tag):
        """First pass; with --second-pass, sibling/context features from its
        (out-of-fold) probabilities and a second cross-fitted model on top."""
        p1, models1 = sc.scores(train_rows, es_rows, tag)
        if not a.second_pass:
            return p1, models1, None
        t0 = time.time()
        X2 = np.hstack([X, second_pass_features(q, s, p1, rec, a.workers)])
        log(f"  {tag}: second-pass features in {time.time() - t0:.0f}s")
        sc2 = Scoring(X2, y, s_group, seed, a.workers, a.max_rounds, log, FEATURES + SECOND_PASS_NAMES, a.lr)
        p2, models2 = sc2.scores(train_rows, es_rows, f"{tag} pass 2")
        return p2, models1, models2

    per_seed, main, loco_params = [], None, {}
    for seed in (int(x) for x in a.seeds.split(",")):
        sc = Scoring(X, y, s_group, seed, a.workers, a.max_rounds, log, FEATURES, a.lr)
        p, models, models2 = two_pass(sc, seed, s_fold == "train", tune_rows, f"main seed {seed}")
        d = decode_with(p, tune_rows, tune.s1_id, rules)
        f_val = scorer.f(scorer.positions(hold.s1_id), d["pred"])
        for rule, r in d["all"].items():
            alt = predict_rule(rule, r["param"], s, d["p"], d["keep"], d["m"])
            log(f"  rule {rule}: param {r['param']}, tune F0.5 {r['f']:.4f}, holdout F0.5 "
                f"{scorer.f(scorer.positions(hold.s1_id), alt).mean():.4f}")
        log(f"  chosen rule {d['rule']} (param {d['param']}); m = {d['m']:.3f} expected blocking misses per S1")
        f_loco = np.full(len(hold), np.nan)
        if not a.no_loco:
            for src_c, dst_c in (("US", "India"), ("India", "US")):
                pa, _, _ = two_pass(sc, seed, (s_fold == "train") & (s_country == src_c), tune_rows & (s_country == src_c),
                                    f"LOCO {src_c}->{dst_c} seed {seed}")
                da = decode_with(pa, tune_rows & (s_country == src_c), tune[tune.country == src_c].s1_id, [d["rule"]])
                is_b = (hold.country == dst_c).to_numpy()
                f_loco[is_b] = scorer.f(scorer.positions(hold.s1_id[is_b]), da["pred"])
                # diagnostic only: the held-out country's own best parameter under the same model
                own = tune_rules(scorer, scorer.positions(tune[tune.country == dst_c].s1_id), s, da["p"], da["keep"],
                                 da["m"], [d["rule"]])
                loco_params[f"{src_c}->{dst_c}"] = (da["param"], own["param"])
                dst_pos = scorer.positions(tune[tune.country == dst_c].s1_id)
                f_in = scorer.f(dst_pos, da["pred"]).mean()
                log(f"  LOCO {src_c}->{dst_c}: in-distribution {d['rule']} param {da['param']} -> held-out tune F0.5 "
                    f"{f_in:.4f}; held-out country's own best {own['param']} -> {own['f']:.4f} (gain {own['f'] - f_in:+.4f})")
        per_seed.append(hold[["s1_id", "country"]].assign(f_val=f_val, f_loco=f_loco))
        comp = s_components(per_seed[-1])
        log(f"seed {seed}: " + ", ".join(f"{k} {v:.4f}" for k, v in comp.items()))
        if main is None:
            main = {"models": models, "models2": models2, **d}

    offset = 0.0
    if loco_params:
        diffs = [own - ind for ind, own in loco_params.values()]
        if all(x > 0 for x in diffs):  # held-out country consistently prefers a stricter setting
            offset = float(np.mean(diffs))
        log(f"unseen-country offset for {main['rule']}: {offset:+.3f} (LOCO in-distribution vs own best: {loco_params})")
    pred_main = main["pred"]
    t_main = main["param"]

    per_entity = per_seed[0][["s1_id", "country"]].copy()
    per_entity["f_val"] = np.mean([d.f_val for d in per_seed], axis=0)
    per_entity["f_loco"] = np.mean([d.f_loco for d in per_seed], axis=0)
    os.makedirs(PER_ENTITY_DIR, exist_ok=True)
    per_entity.to_parquet(f"{PER_ENTITY_DIR}/{a.experiment}.parquet", index=False)
    comp = s_components(per_entity)
    log("\n=== S = 0.383 F_US + 0.468 F_India + 0.150 LOCO_avg ===")
    log(", ".join(f"{k} {v:.4f}" for k, v in comp.items()) + ("  (S pending: run without LOCO)" if a.no_loco else ""))

    hold_rows = np.isin(s, hold.s1_id.to_numpy())
    hp = pd.DataFrame({"s1_id": s[hold_rows], "label": y[hold_rows]})
    report = evaluate(hold, hp, pred_main[hold_rows])
    log(f"\n=== Holdout (seed 0 models, calibrated, one-owner + {main['rule']}) ===")
    log(report.round(4).to_string())
    k_pred = np.bincount(s[pred_main], minlength=len(rec))[hold.s1_id]
    log(f"predicted matches per S1 {k_pred.mean():.3f} vs true {hold['T'].mean():.3f}")
    log("\n=== Error decomposition (holdout; wrong accepts = fp_on_matched, wrong rejects = missed_retrieved) ===")
    log(error_decomposition(hold, hp, pred_main[hold_rows]).round(4).to_string())
    last, names = (main["models2"], FEATURES + SECOND_PASS_NAMES) if main["models2"] else (main["models"], FEATURES)
    gain = pd.Series(np.sum([m.booster_.feature_importance("gain") for m in last], axis=0),
                     index=names).sort_values(ascending=False)
    log("\n=== Top 30 features by gain ===\n" + (gain / gain.sum()).head(30).round(4).to_string())

    boot = None
    if a.compare_to:
        ref = pd.read_parquet(f"{PER_ENTITY_DIR}/{a.compare_to}.parquet")
        boot = paired_bootstrap(per_entity, ref[["s1_id", "country", "f_val", "f_loco"]])
        log(f"\n=== Paired bootstrap vs {a.compare_to} ===\n" + boot.round(4).to_string())

    suffix = "" if a.tag == "train" else "_dev"
    saved = {"models": main["models"], "models2": main["models2"], "features": FEATURES, "calibration": main["iso"], "rule": main["rule"],
             "param": main["param"], "m": main["m"], "unseen_offset": offset,
             "train_countries": sorted(set(country[s1_rids]))}
    for path in (f"{C.model_dir()}/matcher{suffix}.pkl", f"{C.model_dir()}/matcher_{a.experiment}.pkl"):
        with open(path, "wb") as f:
            pickle.dump(saved, f)
    with open(f"{C.model_dir()}/decision{suffix}.json", "w") as f:
        json.dump({"rule": main["rule"], "param": main["param"], "m": main["m"], "unseen_offset": offset,
                   "experiment": a.experiment}, f)

    recall_path = f"{out}/blocking_recall.csv"
    hold_block = pd.read_csv(recall_path) if a.tag == "train" and os.path.exists(recall_path) else None
    row = {"experiment": a.experiment, "config_diff": a.config_diff, "S": round(comp["S"], 4),
           "val_macro_f05": round(float(per_entity.f_val.mean()), 4), "val_f05_us": round(comp["F_US"], 4),
           "val_f05_india": round(comp["F_India"], 4), "loco_us_to_india": round(comp["LOCO_US_to_India"], 4),
           "loco_india_to_us": round(comp["LOCO_India_to_US"], 4), "loco_avg": round(comp["LOCO_avg"], 4),
           "singleton_acc": round(report.loc["ALL", "singleton_acc"], 4),
           "blocking_pair_recall": round(float(np.average(hold_block.pair_recall, weights=hold_block.true_pairs)), 4)
           if hold_block is not None else "",
           "avg_candidates": round(float(np.bincount(s, minlength=len(rec))[hold.s1_id].mean()), 2),
           "runtime": f"{time.time() - t_all:.0f}s train stage", "kept": "pending",
           "notes": f"v2 {a.tag}; {main['rule']} {t_main}; unseen offset {offset:+.3f}; "
                    f"second pass {bool(main['models2'])}; "
                    f"excluded groups {a.exclude_groups or 'none'}", "seeds": a.seeds, "compare_to": a.compare_to}
    if boot is not None:
        for k in ("S", "F_US", "F_India", "LOCO_avg"):
            row[f"d_{k}"] = round(boot.loc[k, "delta"], 4)
        row["d_S_ci"] = f"[{boot.loc['S', 'ci_low']:.4f}, {boot.loc['S', 'ci_high']:.4f}]"
    if a.no_loco:
        for k in ("S", "loco_us_to_india", "loco_india_to_us", "loco_avg", "d_S", "d_LOCO_avg", "d_S_ci"):
            if k in row:
                row[k] = "pending"
    append_experiment(row, experiments_csv_path())
    C.stage_end(log, f"train {a.experiment}", t_stage)


if __name__ == "__main__":
    main()
