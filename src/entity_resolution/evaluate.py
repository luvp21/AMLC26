"""Validation harness: exact metric, entity-grouped splits, per-country
reports, blocking recall, error decomposition, missed-pair categorization,
unsupervised checks on test, and the organizers' submission validator.

Conventions: `entities` has one row per Source 1 entity (s1_id, country, T =
true match count, including matches blocking never retrieved). `pairs` has one
row per (s1_id, candidate_id) with `rank` (0-based position in the candidate
list) and `label` (true match). Every aggregate is vectorized with bincount
over an entity index, never a per-pair Python loop.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sklearn.model_selection import train_test_split

INDIC_RE = re.compile("[ऀ-ൿ]")


# ---------------------------------------------------------------- metric

def f05(T, K, TP) -> np.ndarray:
    """Per-entity F0.5 = 1.25*TP / (0.25*T + K), with the singleton rules:
    T=0,K=0 -> 1; T=0,K>0 -> 0; T>0,K=0 -> 0."""
    T, K, TP = (np.asarray(x, dtype=np.float64) for x in (T, K, TP))
    with np.errstate(divide="ignore", invalid="ignore"):
        f = 1.25 * TP / (0.25 * T + K)
    f = np.where(K == 0, 0.0, f)
    return np.where(T == 0, np.where(K == 0, 1.0, 0.0), f)


# ---------------------------------------------------------------- splits

def make_splits(s1: pd.DataFrame, cfg) -> dict[str, np.ndarray]:
    """Disjoint Source 1 entity groups train / tune / val (never split by pair).
    subsample: stratified by country with fixed seed. full: val is the
    original baseline 15% split (so full-scale numbers stay comparable), tune
    is carved from the remaining 85%, train is the rest."""
    ids = s1.entity_id.to_numpy()
    country_of = pd.Series(s1.country.to_numpy(), index=ids)
    if cfg.scale == "full":
        rest, val = train_test_split(s1.entity_id.tolist(), test_size=0.15, random_state=cfg.seed)
        rest = np.asarray(rest)
        train, tune = train_test_split(rest, test_size=cfg.n_tune, stratify=country_of.loc[rest], random_state=cfg.seed)
        return {"train": train, "tune": tune, "val": np.asarray(val)}
    n_total = cfg.n_train + cfg.n_tune + cfg.n_val
    pool = ids
    if n_total < len(ids):
        pool, _ = train_test_split(ids, train_size=n_total, stratify=country_of.loc[ids], random_state=cfg.seed)
    rest, val = train_test_split(pool, test_size=cfg.n_val, stratify=country_of.loc[pool], random_state=cfg.seed)
    train, tune = train_test_split(rest, test_size=cfg.n_tune, stratify=country_of.loc[rest], random_state=cfg.seed)
    return {"train": np.asarray(train), "tune": np.asarray(tune), "val": np.asarray(val)}


# ---------------------------------------------------------------- tables

def entity_frame(ids, country_of: pd.Series, gt_sets: dict[str, set[str]]) -> pd.DataFrame:
    df = pd.DataFrame({"s1_id": np.asarray(ids)})
    df["country"] = df.s1_id.map(country_of)
    df["T"] = df.s1_id.map(lambda s: len(gt_sets.get(s, ()))).astype(np.int32)
    return df


def pair_frame(ids, candidates: dict[str, list[str]], gt_sets: dict[str, set[str]]) -> pd.DataFrame:
    s1_col, cand_col, rank_col = [], [], []
    for s in ids:
        cs = candidates.get(s, [])
        s1_col.extend([s] * len(cs))
        cand_col.extend(cs)
        rank_col.extend(range(len(cs)))
    label = np.fromiter((c in gt_sets.get(s, ()) for s, c in zip(s1_col, cand_col)), dtype=bool, count=len(s1_col))
    return pd.DataFrame({"s1_id": s1_col, "candidate_id": cand_col,
                         "rank": np.asarray(rank_col, dtype=np.int16), "label": label})


def _index(entities: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    idx = pd.Index(entities.s1_id).get_indexer(pairs.s1_id)
    if (idx < 0).any():
        raise ValueError("pairs contain s1_ids missing from entities — subset pairs first")
    return idx


def _counts(entities, pairs, pred) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per entity: K predicted, TP correct predicted, R true matches retrieved by blocking."""
    idx, n = _index(entities, pairs), len(entities)
    label = pairs.label.to_numpy()
    pred = np.asarray(pred, dtype=bool)
    K = np.bincount(idx, weights=pred, minlength=n)
    TP = np.bincount(idx, weights=pred & label, minlength=n)
    R = np.bincount(idx, weights=label, minlength=n)
    return K, TP, R


# ---------------------------------------------------------------- reports

def evaluate(entities: pd.DataFrame, pairs: pd.DataFrame, pred) -> pd.DataFrame:
    K, TP, _ = _counts(entities, pairs, pred)
    df = entities.assign(K=K, TP=TP, f=f05(entities["T"], K, TP))
    rows = []
    for name, g in [("ALL", df)] + list(df.groupby("country")):
        singles = g[g["T"] == 0]
        rows.append({
            "country": name, "n": len(g), "macro_f05": g.f.mean(),
            "singleton_acc": (singles.K == 0).mean() if len(singles) else np.nan,
            "singleton_false_merge": (singles.K > 0).mean() if len(singles) else np.nan,
            "pair_precision": g.TP.sum() / max(g.K.sum(), 1),
            "pair_recall": g.TP.sum() / max(g["T"].sum(), 1),
            "pred_match_rate": (g.K > 0).mean(), "true_match_rate": (g["T"] > 0).mean(),
        })
    return pd.DataFrame(rows).set_index("country")


def sweep_threshold(entities, pairs, p, thresholds) -> tuple[float, float, dict[float, float]]:
    idx, n = _index(entities, pairs), len(entities)
    label, T = pairs.label.to_numpy(), entities["T"].to_numpy()
    scores = {}
    for t in thresholds:
        pred = p >= t
        K = np.bincount(idx, weights=pred, minlength=n)
        TP = np.bincount(idx, weights=pred & label, minlength=n)
        scores[t] = float(f05(T, K, TP).mean())
    best = max(scores, key=scores.get)
    return best, scores[best], scores


def blocking_report(entities, pairs, ks=(1, 3, 5, 10, 20, 30)) -> pd.DataFrame:
    idx, n = _index(entities, pairs), len(entities)
    label, rank = pairs.label.to_numpy(), pairs["rank"].to_numpy()
    T = entities["T"].to_numpy()
    country = entities.country.to_numpy()
    rows = []
    for k in ks:
        got = np.bincount(idx, weights=label & (rank < k), minlength=n)
        for name in ["ALL"] + sorted(set(country)):
            m = np.ones(n, bool) if name == "ALL" else country == name
            matched = m & (T > 0)
            rows.append({"k": k, "country": name, "pair_recall": got[m].sum() / max(T[m].sum(), 1),
                         "entity_all_recall": (got[matched] == T[matched]).mean()})
    return pd.DataFrame(rows).pivot(index="k", columns="country")


def candidate_stats(entities, pairs) -> dict[str, float]:
    n_cand = np.bincount(_index(entities, pairs), minlength=len(entities))
    return {"avg": float(n_cand.mean()), "p95": float(np.percentile(n_cand, 95)), "max": int(n_cand.max())}


def error_decomposition(entities, pairs, pred) -> pd.DataFrame:
    """Per country, 1 - actual macro F0.5 split additively into:
    blocking_loss (1 - oracle, where oracle predicts exactly the retrieved true
    matches), singleton_false_merge, fp_on_matched (removing false positives,
    keeping true positives), missed_retrieved (oracle - no-FP version)."""
    K, TP, R = _counts(entities, pairs, pred)
    T = entities["T"].to_numpy()
    actual, oracle, no_fp = f05(T, K, TP), f05(T, R, R), f05(T, TP, TP)
    single = T == 0
    df = entities[["country"]].assign(
        actual=actual, oracle=oracle, blocking_loss=1 - oracle,
        singleton_false_merge=np.where(single, oracle - actual, 0.0),
        fp_on_matched=np.where(single, 0.0, no_fp - actual),
        missed_retrieved=np.where(single, 0.0, oracle - no_fp),
    )
    out = df.groupby("country").mean()
    out.loc["ALL"] = df.drop(columns="country").mean()
    return out


# ---------------------------------------------------------------- missed pairs

def _missed_category(s1, cand, name_freq: pd.Series) -> str:
    n1, n2 = " ".join(s1.name_tokens), " ".join(cand.name_tokens)
    a1, a2 = " ".join(s1.addr_tokens), " ".join(cand.addr_tokens)
    name_sim = fuzz.token_set_ratio(n1, n2)
    if INDIC_RE.search(cand.business_name):
        return "indic_script"
    if name_sim >= 90 and name_freq.get((s1.country, n1), 0) >= 10:
        return "chain_crowded"
    only1 = set(s1.name_tokens) - set(cand.name_tokens)
    only2 = set(cand.name_tokens) - set(s1.name_tokens)
    if any(len(a) >= 4 and len(b) >= 4 and Levenshtein.distance(a, b) <= 2 for a in only1 for b in only2):
        return "typo_key_token"
    if name_sim < 50:
        return "address_only" if fuzz.token_set_ratio(a1, a2) >= 70 else "name_very_different"
    return "other"


def categorize_missed(entities, pairs, gt_sets, s1_lookup, cand_lookup, name_freq, n_per_country=200, seed=42):
    """Sample true pairs that blocking never retrieved and bucket them with
    heuristics (precedence: indic_script, chain_crowded, typo_key_token,
    address_only, name_very_different, other). Heuristic, for triage only.
    name_freq: Series indexed by (country, normalized name) over the S2/S3 pool."""
    retrieved = pairs[pairs.label].groupby("s1_id").candidate_id.agg(set)
    missed = [(s, c, country) for s, country in zip(entities.s1_id, entities.country)
              for c in gt_sets.get(s, ()) if c not in retrieved.get(s, ())]
    missed = pd.DataFrame(missed, columns=["s1_id", "candidate_id", "country"])
    rng = np.random.default_rng(seed)
    parts = [g.iloc[rng.permutation(len(g))[:n_per_country]] for _, g in missed.groupby("country")]
    sample = pd.concat(parts) if parts else missed.iloc[:0].copy()
    sample["category"] = [
        _missed_category(s1_lookup.loc[s], cand_lookup.loc[c], name_freq)
        for s, c in zip(sample.s1_id, sample.candidate_id)
    ]
    return missed, sample


# ---------------------------------------------------------------- test-side checks

def france_sample_ids(s1_ids, countries, seed: int = 42, n: int = 50) -> np.ndarray:
    """The fixed France eyeball sample: n France Source 1 ids, in file order,
    permuted with `seed` (same entities in every phase's before/after print)."""
    fr = np.asarray(s1_ids)[np.asarray(countries) == "France"]
    return fr[np.random.default_rng(seed).permutation(len(fr))[:n]] if len(fr) else fr


def assignment_checks(pairs: pd.DataFrame, p: np.ndarray, pred, cand_country: pd.Series,
                      pool_size_by_country: pd.Series) -> pd.DataFrame:
    """Unsupervised, per candidate country: conflict rate (records with p > 0.5
    for two or more Source 1 entities, before any exclusivity step) and
    assignment rate (share of the country's S2/S3 pool assigned to any entity).
    Train reference for assignment rate: 74% of S2/S3 records are matched."""
    df = pd.DataFrame({"candidate_id": pairs.candidate_id.to_numpy(), "hi": p > 0.5,
                       "pred": np.asarray(pred, dtype=bool)})
    per_record = df.groupby("candidate_id").agg(n_hi=("hi", "sum"), assigned=("pred", "any"))
    per_record["country"] = per_record.index.map(cand_country)
    rows = []
    for country, g in per_record.groupby("country"):
        hi = g[g.n_hi >= 1]
        rows.append({"country": country, "records_with_p>0.5": len(hi),
                     "conflict_rate": (hi.n_hi >= 2).mean() if len(hi) else np.nan,
                     "assignment_rate": g.assigned.sum() / pool_size_by_country[country]})
    return pd.DataFrame(rows).set_index("country")


# ---------------------------------------------------------------- validator

def run_validator(matching: str, candidate: str, test_dir: str,
                  validator: str = "data_set/student_resource/utils/validate_submission.py",
                  check_ids: bool = True) -> tuple[bool | None, str]:
    if not os.path.exists(validator):
        return None, f"validator not found at {validator}"
    cmd = [sys.executable, validator, "--matching", matching, "--candidate", candidate, "--test-dir", test_dir]
    result = subprocess.run(cmd + (["--check-ids"] if check_ids else []), capture_output=True, text=True)
    return result.returncode == 0, result.stdout + result.stderr
