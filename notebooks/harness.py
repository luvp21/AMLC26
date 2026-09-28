"""Validation harness. Every pipeline change is a flag (--norm, --feature-text,
...) so each experiment is one run.

Decision step (default from Phase 2 on): every train S1 entity competes for
every candidate record; probabilities are out-of-sample for every pair (rows
the model trained on are scored by 2-fold cross-fit models); each record is
kept only for its best S1 entity; threshold tuned on the tune fold with that
on. The same step is used inside LOCO.

Reports on the validation fold: macro F0.5 per country, S = 0.383 F_US +
0.468 F_India + 0.150 LOCO_avg, singleton accuracy, blocking recall, error
decomposition, missed-pair categories, LOCO both directions, test-side checks
(conflict / exclusivity / France sample), optional submission files. With
--compare-to, a paired bootstrap (over S1 entities) of the deltas. With
--seeds a,b,c, per-entity F is averaged over LightGBM seeds.

Run on PARAM Shavak from repo root:
  nohup env PYTHONPATH=. python3 -u notebooks/harness.py --experiment NAME > runs/NAME.log 2>&1 &
"""
from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import pickle
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from src.entity_resolution.cache import cached_frame, cached_object
from src.entity_resolution.config import Config, FeatureConfig, NormalizationConfig, SplitConfig, config_hash
from src.entity_resolution.decision import EntityScorer, decide, paired_bootstrap, s_components, score_with_oof
from src.entity_resolution.evaluate import (
    assignment_checks,
    blocking_report,
    candidate_stats,
    categorize_missed,
    entity_frame,
    error_decomposition,
    evaluate,
    france_sample_ids,
    make_splits,
    pair_frame,
    run_validator,
)
from src.entity_resolution.features import build_feature_matrix_fast
from src.entity_resolution.harness_io import (
    append_experiment,
    experiments_csv_path,
    feature_input,
    load_source,
    lookup,
    record_exclusive,
)
from src.entity_resolution.matcher import train_matcher
from src.metrics import parse_id_list

NORM_FLAGS = [f for f in NormalizationConfig.__dataclass_fields__ if f != "version"]
PER_ENTITY_DIR = "runs/perentity"

lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    lines.append(str(msg))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="subsample", choices=["subsample", "full"])
    ap.add_argument("--data-root", default="data_set/student_resource/dataset")
    ap.add_argument("--train-candidates", default="runs/checkpoints/candidates.pkl")
    ap.add_argument("--test-candidates", default="runs/checkpoints/test_candidates.pkl")
    ap.add_argument("--n-train", type=int, default=300_000)
    ap.add_argument("--n-tune", type=int, default=50_000)
    ap.add_argument("--n-val", type=int, default=100_000)
    ap.add_argument("--n-missed", type=int, default=200, help="missed pairs sampled per country")
    ap.add_argument("--workers", type=int, default=min(32, mp.cpu_count()))
    ap.add_argument("--seeds", default="42", help="comma list of LightGBM seeds; per-entity F averaged")
    ap.add_argument("--decision", default="exclusive", choices=["exclusive", "threshold"])
    ap.add_argument("--skip-test-checks", action="store_true")
    ap.add_argument("--skip-missed", action="store_true")
    ap.add_argument("--norm", default="", help=f"comma list of normalization flags to enable: {NORM_FLAGS}")
    ap.add_argument("--feature-text", default="raw", choices=["raw", "casefold", "normalized"])
    ap.add_argument("--compare-to", default="", help="experiment_scale whose per-entity F is the bootstrap reference")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--write-submission", default="", help="directory for matching_results.tsv / candidate_pairs.tsv")
    ap.add_argument("--experiment", default="baseline")
    ap.add_argument("--config-diff", default="")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--experiments-csv", default=experiments_csv_path())
    return ap.parse_args()


def norm_config(flags: str) -> NormalizationConfig:
    names = [f for f in flags.split(",") if f]
    unknown = set(names) - set(NORM_FLAGS)
    if unknown:
        raise SystemExit(f"unknown normalization flags {sorted(unknown)}; choose from {NORM_FLAGS}")
    return NormalizationConfig(**{f: True for f in names})


def mask_key(mask: np.ndarray) -> str:
    return hashlib.sha1(np.packbits(mask).tobytes()).hexdigest()[:12]


class Data:
    """Train-side tables: all S1 entities, their candidate pairs and features
    (independent of the split, so cached once per normalization/feature config)."""

    def __init__(self, args, cfg):
        root = args.data_root
        s1, s2, s3 = (load_source(root, "train", k, cfg.normalization, args.workers, log) for k in (1, 2, 3))
        gt = pd.read_csv(f"{root}/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
        self.gt_sets = {s: parse_id_list(m) for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}
        self.country_of = pd.Series(s1.country.to_numpy(), index=s1.entity_id.to_numpy())
        self.s1_look, self.cand_look = lookup(s1), pd.concat([lookup(s2), lookup(s3)])
        with open(args.train_candidates, "rb") as f:
            candidates = pickle.load(f)
        all_ids = s1.entity_id.to_numpy()
        self.ents_all = entity_frame(all_ids, self.country_of, self.gt_sets)
        self.feat_key = config_hash(cfg.normalization, cfg.blocking, cfg.features,
                                    os.path.abspath(args.train_candidates), "feat_v2")
        self.pairs = cached_frame("pairs_all", self.feat_key, lambda: pair_frame(all_ids, candidates, self.gt_sets), log)
        self.X = cached_object("X_all", self.feat_key, lambda: build_feature_matrix_fast(
            feature_input(self.pairs, self.s1_look, self.cand_look, cfg.features.text), n_workers=args.workers), log)
        self.y = self.pairs.label.to_numpy()
        self.s1_arr = self.pairs.s1_id.to_numpy()
        self.cand_arr = self.pairs.candidate_id.to_numpy()
        self.pair_country = self.pairs.s1_id.map(self.country_of).to_numpy()
        self.scorer = EntityScorer(self.ents_all, self.pairs)

        self.splits = make_splits(s1[["entity_id", "country"]], cfg.split)
        self.ents = {k: entity_frame(v, self.country_of, self.gt_sets) for k, v in self.splits.items()}
        fold_of = pd.Series("rest", index=all_ids)
        for k, v in self.splits.items():
            fold_of[v] = k
        self.pair_fold = fold_of.reindex(self.s1_arr).to_numpy()
        self.fold_rows = {k: np.flatnonzero(self.pair_fold == k) for k in self.splits}
        self.fold_pairs = {k: self.pairs.iloc[r].reset_index(drop=True) for k, r in self.fold_rows.items()}


def run_seed(args, cfg, d: Data, seed: int) -> dict:
    """Main model + LOCO for one LightGBM seed, all with the decision step."""
    model_cfg = replace(cfg.model, seed=seed)

    def fitter(tag):
        def fit(mask):
            key = config_hash(d.feat_key, model_cfg, tag, mask_key(mask))
            return cached_object("model", key, lambda: train_matcher(
                d.X[mask], d.y[mask], model_config=model_cfg, n_jobs=args.workers), log)
        return fit

    def decided(train_mask, tag, tune_ids):
        p = score_with_oof(d.X, d.y, d.s1_arr, train_mask, fitter(tag), seed)
        keep = record_exclusive(d.cand_arr, p) if args.decision == "exclusive" else np.ones(len(p), bool)
        t, f_tune = decide(d.scorer, d.scorer.positions(tune_ids), p, keep, cfg.decision.thresholds)
        return p, keep, t, f_tune

    thresholds_only = np.ones(len(d.y), bool)
    train_mask = d.pair_fold == "train"
    p, keep, t, f_tune = decided(train_mask, "main", d.ents["tune"].s1_id)
    # same probabilities, plain threshold: the no-exclusivity number for continuity
    t_plain, _ = decide(d.scorer, d.scorer.positions(d.ents["tune"].s1_id), p, thresholds_only, cfg.decision.thresholds)
    val = d.ents["val"]
    f_val = d.scorer.f(d.scorer.positions(val.s1_id), (p >= t) & keep)
    f_val_plain = d.scorer.f(d.scorer.positions(val.s1_id), p >= t_plain)

    f_loco = np.full(len(val), np.nan)
    loco_t = {}
    for a, b in (("US", "India"), ("India", "US")):
        tune_a = d.ents["tune"][d.ents["tune"].country == a].s1_id
        p_a, keep_a, t_a, _ = decided(train_mask & (d.pair_country == a), f"loco_{a}", tune_a)
        is_b = (val.country == b).to_numpy()
        f_loco[is_b] = d.scorer.f(d.scorer.positions(val.s1_id[is_b]), (p_a >= t_a) & keep_a)
        loco_t[f"{a}->{b}"] = t_a
    per_entity = val[["s1_id", "country"]].assign(f_val=f_val, f_loco=f_loco)
    return {"seed": seed, "p": p, "keep": keep, "t": t, "f_tune": f_tune, "t_plain": t_plain,
            "f_val_plain": f_val_plain, "per_entity": per_entity, "loco_t": loco_t,
            "model": fitter("main")(train_mask)}


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(PER_ENTITY_DIR, exist_ok=True)
    cfg = Config(split=SplitConfig(scale=args.scale, n_train=args.n_train, n_tune=args.n_tune, n_val=args.n_val),
                 normalization=norm_config(args.norm), features=FeatureConfig(text=args.feature_text))
    seeds = [int(s) for s in args.seeds.split(",")]
    name = f"{args.experiment}_{args.scale}"
    timings: dict[str, float] = {}
    t_all = time.time()

    t0 = time.time()
    log(f"Harness — experiment={args.experiment} scale={args.scale} workers={args.workers} seeds={seeds} "
        f"decision={args.decision}")
    log(f"normalization: {cfg.normalization}; feature text: {cfg.features.text}")
    d = Data(args, cfg)
    log("fold sizes: " + ", ".join(f"{k}={len(v):,}" for k, v in d.ents.items())
        + f"; all S1 compete: {len(d.ents_all):,} entities, {len(d.pairs):,} pairs")
    timings["data+features"] = time.time() - t0

    t0 = time.time()
    runs = [run_seed(args, cfg, d, s) for s in seeds]
    timings["models+decision+loco"] = time.time() - t0

    per_entity = runs[0]["per_entity"][["s1_id", "country"]].copy()
    per_entity["f_val"] = np.mean([r["per_entity"].f_val.to_numpy() for r in runs], axis=0)
    per_entity["f_loco"] = np.mean([r["per_entity"].f_loco.to_numpy() for r in runs], axis=0)
    for r in runs:
        per_entity[f"f_val_s{r['seed']}"] = r["per_entity"].f_val.to_numpy()
        per_entity[f"f_loco_s{r['seed']}"] = r["per_entity"].f_loco.to_numpy()
    per_entity.to_parquet(f"{PER_ENTITY_DIR}/{name}.parquet", index=False)
    comp = s_components(per_entity)

    log(f"\n=== Score S = 0.383 F_US + 0.468 F_India + 0.150 LOCO_avg (decision: {args.decision}) ===")
    for r in runs:
        c = s_components(r["per_entity"])
        log(f"seed {r['seed']}: S {c['S']:.4f}  F_US {c['F_US']:.4f}  F_India {c['F_India']:.4f}  "
            f"LOCO US->India {c['LOCO_US_to_India']:.4f}  India->US {c['LOCO_India_to_US']:.4f}  "
            f"(threshold {r['t']:.2f}, LOCO thresholds {r['loco_t']})")
    if len(runs) > 1:
        log(f"seed-averaged: S {comp['S']:.4f}  F_US {comp['F_US']:.4f}  F_India {comp['F_India']:.4f}  "
            f"LOCO avg {comp['LOCO_avg']:.4f}")

    r0 = runs[0]
    val_rows = d.fold_rows["val"]
    pred_val = ((r0["p"] >= r0["t"]) & r0["keep"])[val_rows]
    report = evaluate(d.ents["val"], d.fold_pairs["val"], pred_val)
    log(f"\n=== Validation, seed {r0['seed']} (threshold {r0['t']:.2f} tuned on tune with the decision step; "
        f"tune F0.5 {r0['f_tune']:.4f}) ===")
    log(report.round(4).to_string())
    log(f"without exclusivity (threshold {r0['t_plain']:.2f}): macro F0.5 {r0['f_val_plain'].mean():.4f}")

    log("\n=== Blocking (validation) ===")
    blocking = blocking_report(d.ents["val"], d.fold_pairs["val"])
    log(blocking.round(4).to_string())
    cstats = candidate_stats(d.ents["val"], d.fold_pairs["val"])
    log(f"candidates per entity: avg {cstats['avg']:.1f}, p95 {cstats['p95']:.0f}, max {cstats['max']}")
    log("\n=== Error decomposition (validation, mean per entity; components sum to 1 - actual) ===")
    log(error_decomposition(d.ents["val"], d.fold_pairs["val"], pred_val).round(4).to_string())

    boot = None
    if args.compare_to:
        ref = pd.read_parquet(f"{PER_ENTITY_DIR}/{args.compare_to}.parquet")
        boot = paired_bootstrap(per_entity, ref[["s1_id", "country", "f_val", "f_loco"]], n_boot=args.n_boot)
        log(f"\n=== Paired bootstrap vs {args.compare_to} ({args.n_boot} resamples over S1 entities, 95% interval) ===")
        log(boot.round(4).to_string())

    t0 = time.time()
    if not args.skip_missed:
        missed_analysis(args, d)
    timings["missed-analysis"] = time.time() - t0

    t0 = time.time()
    if not args.skip_test_checks or args.write_submission:
        test_checks(args, cfg, r0["model"], r0["t"])
    timings["test"] = time.time() - t0
    timings["total"] = time.time() - t_all
    log("\nruntime: " + ", ".join(f"{k} {v:.0f}s" for k, v in timings.items()))

    k_max = blocking.index.max()
    row = {
        "experiment": name, "config_diff": args.config_diff,
        "S": round(comp["S"], 4),
        "val_macro_f05": round(float(per_entity.f_val.mean()), 4),
        "val_f05_us": round(comp["F_US"], 4), "val_f05_india": round(comp["F_India"], 4),
        "loco_us_to_india": round(comp["LOCO_US_to_India"], 4), "loco_india_to_us": round(comp["LOCO_India_to_US"], 4),
        "loco_avg": round(comp["LOCO_avg"], 4),
        "singleton_acc": round(report.loc["ALL", "singleton_acc"], 4),
        "blocking_pair_recall": round(blocking.loc[k_max, ("pair_recall", "ALL")], 4),
        "avg_candidates": round(cstats["avg"], 1), "p95_candidates": cstats["p95"],
        "runtime": f"{timings['total']:.0f}s total", "kept": "pending",
        "notes": f"threshold {r0['t']:.2f}; no-exclusivity {r0['f_val_plain'].mean():.4f}",
        "decision": args.decision, "seeds": args.seeds, "compare_to": args.compare_to,
    }
    if boot is not None:
        for k in ("S", "F_US", "F_India", "LOCO_avg"):
            row[f"d_{k}"] = round(boot.loc[k, "delta"], 4)
        row["d_S_ci"] = f"[{boot.loc['S', 'ci_low']:.4f}, {boot.loc['S', 'ci_high']:.4f}]"
    append_experiment(row, args.experiments_csv)
    with open(f"{args.out}/report_{name}.txt", "w") as f:
        f.write("\n".join(lines))


def missed_analysis(args, d: Data) -> None:
    norm_name = d.cand_look.name_tokens.map(" ".join)
    name_freq = pd.Series(1, index=pd.MultiIndex.from_arrays([d.cand_look.country.to_numpy(), norm_name.to_numpy()]))
    name_freq = name_freq.groupby(level=[0, 1]).size()
    missed, sample = categorize_missed(d.ents["val"], d.fold_pairs["val"], d.gt_sets, d.s1_look, d.cand_look,
                                       name_freq, n_per_country=args.n_missed, seed=42)
    log(f"\n=== Missed pairs (never retrieved): {len(missed):,} in validation; sampled {len(sample)} ===")
    log(pd.crosstab(sample.category, sample.country, margins=True).to_string())
    path = f"{args.out}/missed_examples_{args.experiment}.txt"
    with open(path, "w") as f:
        for (country, cat), g in sample.groupby(["country", "category"]):
            f.write(f"\n##### {country} / {cat} ({len(g)})\n")
            for s, c in zip(g.s1_id.head(5), g.candidate_id.head(5)):
                a, b = d.s1_look.loc[s], d.cand_look.loc[c]
                f.write(f"S1 {s}: {a.business_name!r} | {a.business_address!r}\n"
                        f"   {c}: {b.business_name!r} | {b.business_address!r}\n")
    log(f"examples written to {path}")


def write_submission(out_dir, t_ents, t_pairs, pred, test_cands) -> None:
    os.makedirs(out_dir, exist_ok=True)
    matched = t_pairs[pred].groupby("s1_id", sort=False).candidate_id.agg(list)
    with open(f"{out_dir}/matching_results.tsv", "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s in t_ents.s1_id:
            f.write(f"{s}\t{','.join(matched.get(s, []))}\n")
    with open(f"{out_dir}/candidate_pairs.tsv", "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s in t_ents.s1_id:
            f.write(f"{s}\t{','.join(test_cands.get(s, []))}\n")


def test_checks(args, cfg, model, t_star) -> None:
    """Test side: conflict / assignment rates, the decision step (all test S1
    entities compete), the fixed France sample, and optional submission files."""
    t1, t2, t3 = (load_source(args.data_root, "test", k, cfg.normalization, args.workers, log) for k in (1, 2, 3))
    with open(args.test_candidates, "rb") as f:
        test_cands = pickle.load(f)
    t_s1_look, t_cand_look = lookup(t1), pd.concat([lookup(t2), lookup(t3)])
    t_ents = entity_frame(t1.entity_id.to_numpy(), pd.Series(t1.country.to_numpy(), index=t1.entity_id.to_numpy()), {})
    t_pairs = pair_frame(t_ents.s1_id, test_cands, {})
    key = config_hash(cfg.normalization, cfg.features, os.path.abspath(args.test_candidates), "feat_v1")
    X_test = cached_object("X_test", key, lambda: build_feature_matrix_fast(
        feature_input(t_pairs, t_s1_look, t_cand_look, cfg.features.text), n_workers=args.workers), log)
    p_test = model.predict_proba(X_test)[:, 1]
    del X_test
    pred_plain = p_test >= t_star
    keep = record_exclusive(t_pairs.candidate_id, p_test) if args.decision == "exclusive" else np.ones(len(p_test), bool)
    pred = pred_plain & keep
    pool = pd.concat([t2.country, t3.country]).value_counts()
    checks = assignment_checks(t_pairs, p_test, pred_plain, t_cand_look.country, pool)
    s1_idx = pd.Index(t_ents.s1_id).get_indexer(t_pairs.s1_id)
    by_s1 = {}
    for label, pr in (("before", pred_plain), ("after", pred)):
        k = np.bincount(s1_idx, weights=pr, minlength=len(t_ents))
        by_s1[f"S1_pred_match_rate_{label}"] = pd.Series(k > 0).groupby(t_ents.country.to_numpy()).mean()
    cand_country = t_pairs.candidate_id.map(t_cand_look.country).to_numpy()
    checks["pred_pairs_removed_by_excl"] = (pd.Series(pred_plain & ~pred).groupby(cand_country).sum()
                                            / pd.Series(pred_plain).groupby(cand_country).sum())
    log(f"\n=== Test checks (threshold {t_star:.2f}; conflict rate before exclusivity; train true assignment rate 0.74) ===")
    log(checks.round(4).to_string())
    log("S1 predicted-match rate by S1 country, before / after exclusivity:")
    log(pd.DataFrame(by_s1).round(4).to_string())

    show = france_sample_ids(t_ents.s1_id, t_ents.country)
    tp = t_pairs.assign(p=p_test, kept=pred)
    top = tp[tp.s1_id.isin(set(show))].sort_values(["s1_id", "p"], ascending=[True, False]).groupby("s1_id").head(5)
    path = f"{args.out}/france_sample_{args.experiment}.txt"
    with open(path, "w") as f:
        for s in show:
            r = t_s1_look.loc[s]
            f.write(f"\n=== {s}\n  raw : {r.business_name!r} | {r.business_address!r}\n"
                    f"  norm: {r.name_tokens} | {r.addr_tokens}\n")
            g = top[top.s1_id == s]
            for c, p, kept in zip(g.candidate_id, g.p, g.kept):
                q = t_cand_look.loc[c]
                mark = "x" if (p >= t_star and not kept) else " "
                f.write(f"    p={p:.3f}{mark} {c}: {q.business_name!r} | {q.business_address!r}\n"
                        f"          norm: {q.name_tokens} | {q.addr_tokens}\n")
    log(f"France sample (50 entities, top 5; 'x' = above threshold but removed by exclusivity) -> {path}")

    if args.write_submission:
        write_submission(args.write_submission, t_ents, t_pairs, pred, test_cands)
        ok, out = run_validator(f"{args.write_submission}/matching_results.tsv",
                                f"{args.write_submission}/candidate_pairs.tsv", f"{args.data_root}/test")
        log(f"\n=== Submission written to {args.write_submission}; validator passed: {ok} ===")
        log(out.strip())


if __name__ == "__main__":
    main()
