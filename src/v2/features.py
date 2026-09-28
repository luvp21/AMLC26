"""Stage 4: pair features for the pruned candidate set.

  python -m src.v2.features --tag train|dev|test

All strings are already casefolded by normalization; no case-sensitive feature.
Groups (M1): blocking context, name, address, structure. M3 groups (each can
be left out at training with --exclude-groups, for ablation with S):
  idf      IDF-weighted overlap / max shared IDF / one-sided IDF, name and address
  france   similarity on name_core_minus_addr; same street with a different house number
  context2 rank of the pair within its S1 by prune_p; mutual best (query's and S1's top)
  cross    best name similarity to a candidate of the same S1 from the other source,
           and whether that candidate is the S1's top candidate
IDF: per country from the split's own records (unlabelled); test reuses the train
table for countries that have one. Output OUT_DIR/<tag>/features.parquet.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from src.v2 import config as C
from src.v2.common import add_context, channel_columns, load_records

CONTEXT = ["q_best", "margin", "q_cnt", "s_cnt", "s_first", "prune_p"]  # + each channel's score and rank
NAME = ["core_ratio", "core_token_set", "core_token_sort", "core_partial", "core_jw", "full_token_set", "alias_best",
        "core_jaccard", "core_len_diff", "first_tok_eq", "last_tok_eq", "acronym", "legal_state", "skeleton_ratio"]
ADDRESS = ["addr_ratio", "addr_token_set", "addr_partial", "addr_jaccard", "house_state", "postcode_state",
           "num_jaccard", "q_addr_empty", "s_addr_empty", "landmark_state"]
STRUCTURE = ["q_is_s3", "q_non_latin"]
GROUPS = {
    "idf": ["idf_name_overlap", "idf_name_max_shared", "idf_name_onesided", "idf_addr_overlap", "idf_addr_onesided"],
    "france": ["cma_ratio", "cma_token_set", "cma_idf_overlap", "street_house_state"],
    "context2": ["s_side_rank", "mutual_best"],
    "cross": ["cross_best_sim", "cross_best_is_top"],
}
PAIR_FEATURES = NAME + ADDRESS + STRUCTURE + [f for g in GROUPS.values() for f in g]
CATEGORICAL = ["legal_state", "house_state", "postcode_state", "landmark_state", "street_house_state",
               "cross_best_is_top"]

REC_COLS = ["source", "country", "name", "name_alias", "name_core", "name_legal_class", "name_skeleton", "non_latin",
            "addr", "addr_parts", "addr_landmark", "house_num", "postcode", "num_tokens", "name_core_minus_addr"]
_R: dict = {}
_P: dict = {}
_IDF: dict = {}  # country -> (name idf, address idf, default idf)


def _fz(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=1, dtype=np.float32) / 100.0


def _jaccard(a: str, b: str) -> float:
    x, y = set(a.split()), set(b.split())
    return len(x & y) / len(x | y) if (x or y) else 1.0


def _acronym(a: str, b: str) -> float:
    ta, tb = a.split(), b.split()
    ia, ib = "".join(t[0] for t in ta), "".join(t[0] for t in tb)
    ok = (len(ia) >= 2 and (ia == b.replace(" ", "") or ia in tb)) or (len(ib) >= 2 and (ib == a.replace(" ", "") or ib in ta))
    return float(ok)


def _alias_best(q_name, q_alias, s_name, s_alias) -> float:
    if not q_alias and not s_alias:
        return -1.0
    left = [q_name, *(q_alias.split(" | ") if q_alias else [])]
    right = [s_name, *(s_alias.split(" | ") if s_alias else [])]
    return max(fuzz.token_set_ratio(x, y) for x in left for y in right) / 100.0


def _house_state(a: str, b: str) -> float:
    if not a or not b:
        return 3.0
    if a == b:
        return 0.0
    da, db = "".join(c for c in a if c.isdigit()), "".join(c for c in b if c.isdigit())
    return 1.0 if da and da == db else 2.0


def _pair_state(a: str, b: str) -> float:
    if a and b:
        return 0.0 if a == b else 1.0
    return 2.0 if (a or b) else 3.0


def _legal_state(a: str, b: str) -> float:
    if a == "none" or b == "none":
        return 2.0
    return 0.0 if a == b else 1.0


def _street(parts: str) -> frozenset:
    """Non-digit tokens of the first address part that contains a digit."""
    for part in parts.split(" | "):
        if any(c.isdigit() for c in part):
            return frozenset(t for t in part.split() if not any(c.isdigit() for c in t))
    return frozenset()


def _street_house_state(qp, sp, qh, sh) -> float:
    a, b = _street(qp), _street(sp)
    if not a or not b or not qh or not sh:
        return 3.0
    if a != b:
        return 2.0
    return 0.0 if qh == sh else 1.0


def _idf(a: str, b: str, idf: dict, default: float) -> tuple[float, float, float]:
    """(IDF-weighted Jaccard, max IDF of shared tokens, max one-sided IDF sum)."""
    A, B = set(a.split()), set(b.split())
    if not A and not B:
        return -1.0, 0.0, 0.0
    w = {t: idf.get(t, default) for t in A | B}
    union = sum(w.values())
    shared = A & B
    return (sum(w[t] for t in shared) / union if union else 0.0, max((w[t] for t in shared), default=0.0),
            max(sum(w[t] for t in A - B), sum(w[t] for t in B - A)))


def _chunk(bounds):
    lo, hi = bounds
    q, s = _P["q"][lo:hi], _P["s"][lo:hi]
    R = _R

    def col(name, idx):
        return R[name][idx].tolist()

    qc, sc = col("name_core", q), col("name_core", s)
    qa, sa = col("addr", q), col("addr", s)
    out = {
        "core_ratio": _fz(qc, sc, fuzz.ratio), "core_token_set": _fz(qc, sc, fuzz.token_set_ratio),
        "core_token_sort": _fz(qc, sc, fuzz.token_sort_ratio), "core_partial": _fz(qc, sc, fuzz.partial_ratio),
        "core_jw": process.cpdist(qc, sc, scorer=JaroWinkler.normalized_similarity, workers=1, dtype=np.float32),
        "full_token_set": _fz(col("name", q), col("name", s), fuzz.token_set_ratio),
        "skeleton_ratio": _fz(col("name_skeleton", q), col("name_skeleton", s), fuzz.ratio),
        "addr_ratio": _fz(qa, sa, fuzz.ratio), "addr_token_set": _fz(qa, sa, fuzz.token_set_ratio),
        "addr_partial": _fz(qa, sa, fuzz.partial_ratio),
    }
    qal, sal, qn, sn = col("name_alias", q), col("name_alias", s), col("name", q), col("name", s)
    qlc, slc = col("name_legal_class", q), col("name_legal_class", s)
    qh, sh, qp, sp_ = col("house_num", q), col("house_num", s), col("postcode", q), col("postcode", s)
    qnum, snum, ql, sl = col("num_tokens", q), col("num_tokens", s), col("addr_landmark", q), col("addr_landmark", s)
    rows = [(
        _alias_best(qn[i], qal[i], sn[i], sal[i]), _jaccard(qc[i], sc[i]), abs(len(qc[i]) - len(sc[i])),
        float(bool(qc[i]) and bool(sc[i]) and qc[i].split()[0] == sc[i].split()[0]),
        float(bool(qc[i]) and bool(sc[i]) and qc[i].split()[-1] == sc[i].split()[-1]),
        _acronym(qc[i], sc[i]), _legal_state(qlc[i], slc[i]),
        _jaccard(qa[i], sa[i]) if (qa[i] or sa[i]) else -1.0, _house_state(qh[i], sh[i]), _pair_state(qp[i], sp_[i]),
        _jaccard(qnum[i], snum[i]) if (qnum[i] or snum[i]) else -1.0, _pair_state(ql[i], sl[i]),
    ) for i in range(len(q))]
    cols = ["alias_best", "core_jaccard", "core_len_diff", "first_tok_eq", "last_tok_eq", "acronym", "legal_state",
            "addr_jaccard", "house_state", "postcode_state", "num_jaccard", "landmark_state"]
    arr = np.asarray(rows, dtype=np.float32).reshape(len(q), len(cols))
    out.update({c: arr[:, j] for j, c in enumerate(cols)})

    qm, sm = col("name_core_minus_addr", q), col("name_core_minus_addr", s)
    out["cma_ratio"], out["cma_token_set"] = _fz(qm, sm, fuzz.ratio), _fz(qm, sm, fuzz.token_set_ratio)
    country, qpart, spart = col("country", s), col("addr_parts", q), col("addr_parts", s)
    extra = []
    for i in range(len(q)):
        name_idf, addr_idf, default = _IDF.get(country[i], ({}, {}, 1.0))
        n = _idf(qc[i], sc[i], name_idf, default)
        ad = _idf(qa[i], sa[i], addr_idf, default)
        extra.append((*n, ad[0], ad[2], _idf(qm[i], sm[i], name_idf, default)[0],
                      _street_house_state(qpart[i], spart[i], qh[i], sh[i])))
    cols = ["idf_name_overlap", "idf_name_max_shared", "idf_name_onesided", "idf_addr_overlap", "idf_addr_onesided",
            "cma_idf_overlap", "street_house_state"]
    arr = np.asarray(extra, dtype=np.float32).reshape(len(q), len(cols))
    out.update({c: arr[:, j] for j, c in enumerate(cols)})
    return out


def _cross_chunk(bounds):
    """Cross-source corroboration over pairs sorted by S1 (a chunk = whole S1 groups)."""
    lo, hi = bounds
    idx = _P["by_s"][lo:hi]
    s, q, prune = _P["s"][idx], _P["q"][idx], _P["prune_p"][idx]
    core, src = _R["name_core"], _R["source"]
    best_sim = np.full(len(idx), -1.0, np.float32)
    is_top = np.full(len(idx), 2.0, np.float32)
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1, len(idx)]
    for a, b in zip(starts[:-1], starts[1:]):
        qs, ps = q[a:b], prune[a:b]
        top_q = qs[np.argmax(ps)]
        order = np.argsort(-ps)[:12]
        for i in range(a, b):
            others = [qs[j] for j in order if src[qs[j]] != src[q[i]]][:6]
            if not others:
                continue
            sims = [fuzz.token_set_ratio(core[q[i]], core[o]) for o in others]
            k = int(np.argmax(sims))
            best_sim[i] = sims[k] / 100.0
            is_top[i] = float(others[k] == top_q)
    return idx, best_sim, is_top


def learn_idf(rec: pd.DataFrame) -> dict:
    """{country: (name-token idf, address-token idf, idf of an unseen token)} from the split's records."""
    out = {}
    for country, g in rec.groupby("country"):
        n = len(g)
        tables = []
        for col in ("name_core", "addr"):
            cnt = pd.Series([t for text in g[col] for t in set(text.split())]).value_counts()
            tables.append(dict(zip(cnt.index, np.log(n / cnt.to_numpy()))))
        out[country] = (tables[0], tables[1], float(np.log(n)))
    return out


def build(pairs: pd.DataFrame, rec: pd.DataFrame, workers: int, idf: dict) -> pd.DataFrame:
    global _R, _P, _IDF
    _R = {c: rec[c].to_numpy() for c in REC_COLS}
    _R["name_legal_class"] = np.asarray(rec.name_legal_class.fillna("none"), dtype=object)
    _IDF = idf
    df = add_context(pairs, len(rec))
    _P = {"q": df.q.to_numpy(), "s": df.s.to_numpy(), "prune_p": df.prune_p.to_numpy()}
    n = len(df)
    size = max(1, min(200_000, n // (workers * 4) + 1))
    bounds = [(i, min(i + size, n)) for i in range(0, n, size)]
    # cross-source chunks must hold whole S1 groups: sort by S1 and cut at group starts
    by_s = np.lexsort((df.q.to_numpy(), df.s.to_numpy()))
    _P["by_s"] = by_s
    s_sorted = df.s.to_numpy()[by_s]
    marks = s_sorted[np.linspace(0, n - 1, workers * 8).astype(int)[1:]] if n else []
    cuts = sorted({0, n, *(int(c) for c in np.searchsorted(s_sorted, marks))})
    with mp.get_context("fork").Pool(workers) as pool:  # forks after _R/_P/_IDF are set
        parts = pool.map(_chunk, bounds)
        cross = pool.map(_cross_chunk, list(zip(cuts[:-1], cuts[1:])))
    feats = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else {}
    feats["cross_best_sim"] = np.full(n, -1.0, np.float32)
    feats["cross_best_is_top"] = np.full(n, 2.0, np.float32)
    for idx, sim, top in cross:
        feats["cross_best_sim"][idx], feats["cross_best_is_top"][idx] = sim, top
    q, s = df.q.to_numpy(), df.s.to_numpy()
    pp = df.prune_p.to_numpy()
    feats["s_side_rank"] = pd.Series(pp).groupby(s).rank(ascending=False, method="first").to_numpy(np.float32)
    q_top = pd.Series(pp).groupby(q).transform("max").to_numpy() == pp
    s_top = pd.Series(pp).groupby(s).transform("max").to_numpy() == pp
    feats["mutual_best"] = (q_top & s_top).astype(np.float32)
    features = feature_names(pairs)
    out = df[["q", "s"] + channel_columns(pairs) + CONTEXT].copy()
    for k in NAME + ADDRESS + [f for g in GROUPS.values() for f in g]:
        if k in feats:
            out[k] = feats[k]
    out["q_addr_empty"] = (rec.addr.to_numpy()[q] == "").astype(np.float32)
    out["s_addr_empty"] = (rec.addr.to_numpy()[s] == "").astype(np.float32)
    out["q_is_s3"] = (rec.source.to_numpy()[q] == 3).astype(np.float32)
    out["q_non_latin"] = rec.non_latin.to_numpy()[q].astype(np.float32)
    out[features] = out[features].astype(np.float32)
    return out[["q", "s"] + features]


def feature_names(pairs_or_columns) -> list[str]:
    cols = pairs_or_columns.columns if hasattr(pairs_or_columns, "columns") else pairs_or_columns
    chans = [c for c in cols if (c.endswith("_score") or c.endswith("_rank")) and c != "prune_p"]
    return chans + CONTEXT + PAIR_FEATURES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--workers", type=int, default=C.WORKERS)
    a = ap.parse_args()
    out = C.tag_dir(a.tag)
    log = C.log_to(f"{out}/features.log")
    t0 = C.stage_start(log, f"features {a.tag}")
    rec = load_records(a.tag, REC_COLS)
    pairs = pd.read_parquet(f"{out}/pairs_pruned.parquet")
    idf = learn_idf(rec)
    source = {c: "own records" for c in idf}
    idf_path = f"{C.model_dir()}/idf.pkl"
    if a.tag == "train":
        with open(idf_path, "wb") as f:
            pickle.dump(idf, f)
    elif a.tag == "test" and os.path.exists(idf_path):
        with open(idf_path, "rb") as f:
            train_idf = pickle.load(f)
        for c in idf:
            if c in train_idf:
                idf[c], source[c] = train_idf[c], "train table"
    log("IDF tables: " + ", ".join(f"{c} ({v})" for c, v in sorted(source.items())))
    feats = build(pairs, rec, a.workers, idf)
    feats.to_parquet(f"{out}/features.parquet", index=False)
    log(f"{len(feats):,} pairs x {len(feats.columns) - 2} features: {list(feats.columns[2:])}")
    C.stage_end(log, f"features {a.tag}", t0)


if __name__ == "__main__":
    main()
