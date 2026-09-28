"""Decoding: calibration, one-owner, and the per-entity selection rule.

Rules (parameters tuned on the tune fold only):
- threshold: predict every one-owner-kept pair with calibrated p >= t.
- expected_f: per S1, sort kept candidates by p; choose K maximizing the
  expected F0.5  1.25 * sum(p_1..p_K) / (0.25 * (sum(p) + m) + K)  for K >= 1,
  against alpha * prod(1 - p_i) for K = 0 (m = expected true matches per S1
  that blocking missed, measured on the tune fold).
Unseen-country offset: if, in both LOCO directions, the held-out country's own
best parameter is stricter than the in-distribution one, the mean difference
is added to the parameter for any country label without training data.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.entity_resolution.harness_io import record_exclusive

THRESHOLDS = tuple(round(0.20 + 0.01 * i, 2) for i in range(76))
ALPHAS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0)
STRICTER = {"threshold": 1.0, "expected_f": 1.0}  # a larger parameter is stricter for both rules


def fit_calibration(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)


def one_owner(q: np.ndarray, p_raw: np.ndarray) -> np.ndarray:
    """Each record kept only for its highest raw-probability S1 (raw p breaks the
    ties that isotonic steps would create)."""
    return record_exclusive(q, p_raw)


def expected_f_pred(s: np.ndarray, p: np.ndarray, keep: np.ndarray, alpha: float, m: float) -> np.ndarray:
    pred = np.zeros(len(p), bool)
    idx = np.flatnonzero(keep & (p > 0))
    if not len(idx):
        return pred
    order = idx[np.lexsort((-p[idx], s[idx]))]
    ss, pp = s[order], p[order]
    start = np.r_[True, ss[1:] != ss[:-1]]
    gid = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    cs = np.cumsum(pp)
    cs_group = cs - np.r_[0.0, cs[first[1:] - 1]][gid]
    total = np.add.reduceat(pp, first)[gid]
    k = np.arange(len(order)) - first[gid] + 1
    score = 1.25 * cs_group / (0.25 * (total + m) + k)
    best = np.maximum.reduceat(score, first)
    empty = alpha * np.exp(np.add.reduceat(np.log1p(-np.clip(pp, 0, 1 - 1e-9)), first))
    at_best = np.flatnonzero(score >= best[gid])
    _, first_best = np.unique(gid[at_best], return_index=True)  # first K reaching the group's max
    k_best = k[at_best[first_best]]
    take = (best > empty)[gid] & (k <= k_best[gid])
    pred[order[take]] = True
    return pred


def predict(rule: str, param: float, s, p, keep, m: float) -> np.ndarray:
    if rule == "threshold":
        return (p >= param) & keep
    return expected_f_pred(s, p, keep, param, m)


def tune(scorer, positions, s, p, keep, m: float, rules=("threshold", "expected_f")) -> dict:
    """Best (rule, parameter) on the given entities, plus every rule's best."""
    results = {}
    for rule in rules:
        grid = THRESHOLDS if rule == "threshold" else ALPHAS
        scores = [(scorer.f(positions, predict(rule, v, s, p, keep, m)).mean(), v) for v in grid]
        f, v = max(scores)
        results[rule] = {"param": v, "f": float(f)}
    best = max(results, key=lambda r: results[r]["f"])
    return {"rule": best, "param": results[best]["param"], "f": results[best]["f"], "all": results}


def blocking_misses(T: np.ndarray, rids: np.ndarray, s: np.ndarray, y: np.ndarray) -> float:
    """Mean over the given S1 record ids of true matches not in the candidate set
    (T and the bincount are indexed by record id)."""
    retrieved = np.bincount(s[y], minlength=len(T))
    return float(np.mean(T[rids] - retrieved[rids])) if len(rids) else 0.0
