"""Second-pass features from first-pass probabilities (sibling evidence + context).

For each S1, its top SIB_TOP candidates by first-pass p act as siblings. For
every candidate c of that S1 (siblings j != c):
  sib_support_{name,addr}  max_j sim(c, j) * p1(j)          (token_set_ratio)
  sib_ratio_{name,addr}    max_j ratio(c, j) * p1(j)         (plain ratio)
  sib_strong_{name,addr}   #j with sim(c, j) >= 0.8 and p1(j) >= 0.5
  sib_mean_{name,addr}     mean sim(c, j) over j with p1(j) >= 0.5 (-1 if none)
  sib_same_source          sib_support_name over siblings from c's source
  sib_house, sib_postcode  max_j [equal and non-empty] * p1(j)
Context: p1 itself, the query's best p minus its second best, rank of p within
the S1, the S1's sum / count(p > 0.5) / max of p, and whether the S1 is the
query's argmax. Computed from inputs and first-pass scores only (no labels),
identically on train, holdout and test.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

SIB_TOP = 8
SIB_NAMES = ["sib_support_name", "sib_support_addr", "sib_ratio_name", "sib_ratio_addr", "sib_strong_name",
             "sib_strong_addr", "sib_mean_name", "sib_mean_addr", "sib_same_source", "sib_house", "sib_postcode"]
CONTEXT2_NAMES = ["p1", "q_p_margin", "s_p_rank", "s_p_sum", "s_p_count", "s_p_max", "q_is_argmax"]
SECOND_PASS_NAMES = SIB_NAMES + CONTEXT2_NAMES

_G: dict = {}


def _groups(bounds):
    lo, hi = bounds
    idx = _G["order"][lo:hi]
    s, q, p = _G["s"][idx], _G["q"][idx], _G["p"][idx]
    core, addr, house, post, src = _G["core"], _G["addr"], _G["house"], _G["post"], _G["src"]
    out = np.zeros((len(idx), len(SIB_NAMES)), np.float32)
    out[:, 6:8] = -1.0
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1, len(idx)]
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a < 2:
            continue
        qs, ps = q[a:b], p[a:b]
        top = np.argsort(-ps)[:SIB_TOP]
        tq, tp = qs[top], ps[top]
        cn, tn = [core[x] for x in qs], [core[x] for x in tq]
        ca, ta = [addr[x] for x in qs], [addr[x] for x in tq]
        sim = {"name": process.cdist(cn, tn, scorer=fuzz.token_set_ratio, dtype=np.float32, workers=1) / 100,
               "addr": process.cdist(ca, ta, scorer=fuzz.token_set_ratio, dtype=np.float32, workers=1) / 100,
               "name_r": process.cdist(cn, tn, scorer=fuzz.ratio, dtype=np.float32, workers=1) / 100,
               "addr_r": process.cdist(ca, ta, scorer=fuzz.ratio, dtype=np.float32, workers=1) / 100}
        self_mask = qs[:, None] == tq[None, :]
        w = np.where(self_mask, 0.0, tp[None, :]).astype(np.float32)
        strong = (tp[None, :] >= 0.5) & ~self_mask
        for k in sim:
            sim[k] = np.where(self_mask, 0.0, sim[k])
        o = out[a:b]
        o[:, 0], o[:, 1] = (sim["name"] * w).max(1), (sim["addr"] * w).max(1)
        o[:, 2], o[:, 3] = (sim["name_r"] * w).max(1), (sim["addr_r"] * w).max(1)
        o[:, 4] = ((sim["name"] >= 0.8) & strong).sum(1)
        o[:, 5] = ((sim["addr"] >= 0.8) & strong).sum(1)
        n_strong = strong.sum(1)
        has = n_strong > 0
        o[has, 6] = (sim["name"] * strong).sum(1)[has] / n_strong[has]
        o[has, 7] = (sim["addr"] * strong).sum(1)[has] / n_strong[has]
        same = np.array([[src[x] == src[y] for y in tq] for x in qs])
        o[:, 8] = (sim["name"] * w * same).max(1)
        hq, ht = [house[x] for x in qs], [house[x] for x in tq]
        pq, pt = [post[x] for x in qs], [post[x] for x in tq]
        eq_h = np.array([[bool(u) and u == v for v in ht] for u in hq])
        eq_p = np.array([[bool(u) and u == v for v in pt] for u in pq])
        o[:, 9], o[:, 10] = (eq_h * w).max(1), (eq_p * w).max(1)
    return idx, out


def _second_best(g: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per row: the second-largest p within its group g (0 if the group has one row)."""
    order = np.lexsort((-p, g))
    gs, ps = g[order], p[order]
    start = np.r_[True, gs[1:] != gs[:-1]]
    first = np.flatnonzero(start)
    size = np.diff(np.r_[first, len(gs)])
    second = np.where(size > 1, ps[np.minimum(first + 1, len(ps) - 1)], 0.0)
    out = np.empty(len(p))
    out[order] = np.repeat(second, size)
    return out


def second_pass_features(q: np.ndarray, s: np.ndarray, p1: np.ndarray, rec: pd.DataFrame, workers: int) -> np.ndarray:
    """(n_pairs, len(SECOND_PASS_NAMES)) float32, rows aligned with q/s/p1."""
    global _G
    n = len(q)
    order = np.lexsort((q, s))
    s_sorted = s[order]
    marks = s_sorted[np.linspace(0, n - 1, max(workers * 8, 1)).astype(int)[1:]] if n else []
    cuts = sorted({0, n, *(int(c) for c in np.searchsorted(s_sorted, marks))})
    _G = {"order": order, "s": s, "q": q, "p": p1, "core": rec.name_core.to_numpy(), "addr": rec.addr.to_numpy(),
          "house": rec.house_num.to_numpy(), "post": rec.postcode.to_numpy(), "src": rec.source.to_numpy()}
    feats = np.zeros((n, len(SECOND_PASS_NAMES)), np.float32)
    with mp.get_context("fork").Pool(workers) as pool:
        for idx, block in pool.imap_unordered(_groups, list(zip(cuts[:-1], cuts[1:]))):
            feats[idx, :len(SIB_NAMES)] = block
    ps = pd.Series(p1)
    q_best = ps.groupby(q).transform("max").to_numpy()
    q_second = _second_best(q, p1)
    j = len(SIB_NAMES)
    feats[:, j] = p1
    feats[:, j + 1] = np.where(p1 >= q_best, p1 - q_second, p1 - q_best)
    feats[:, j + 2] = ps.groupby(s).rank(ascending=False, method="first").to_numpy()
    feats[:, j + 3] = ps.groupby(s).transform("sum").to_numpy()
    feats[:, j + 4] = (ps > 0.5).groupby(s).transform("sum").to_numpy()
    feats[:, j + 5] = ps.groupby(s).transform("max").to_numpy()
    feats[:, j + 6] = (p1 >= q_best).astype(np.float32)
    return feats
