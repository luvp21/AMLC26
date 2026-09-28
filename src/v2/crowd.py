"""Name-crowd features (M4 error mining round 2: empty-address records).

The largest remaining loss is an empty-address S2/S3 record assigned to the
wrong one of several S1 entities that share its name (a wrong match costs ~3x a
missed one under F0.5). These label-free features tell the model how crowded
the name is, so a name-only match can be trusted when the name is unique and
treated as ambiguous when it is not.

  name_dup_country  S1 entities in the same country with exactly this S1's name core
  name_dup_cand     candidates of this query whose S1 shares this S1's name core
  crowd_rank        rank of this pair's pruner score among those same-name candidates (1 = best)
  crowd_margin      this pair's pruner score minus the best other same-name candidate's
  q_core_equal      the query's name core equals this S1's name core
  addr_dup_country / street_dup_country        S1 entities in the country sharing this S1's address / street
  q_name_dup_country / q_addr_dup_country / q_street_dup_country
                    S1 entities in the country sharing the QUERY's name core / address / street

  python -m src.v2.crowd --tag train|test  -> OUT_DIR/<tag>/features_crowd.parquet (aligned with features.parquet)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.v2 import config as C
from src.v2.common import load_records
from src.v2.sibling import _second_best

CROWD_FEATURES = ["name_dup_country", "name_dup_cand", "crowd_rank", "crowd_margin", "q_core_equal",
                  "addr_dup_country", "street_dup_country", "q_name_dup_country", "q_addr_dup_country", "q_street_dup_country"]


def build(pairs: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    q, s, p = pairs.q.to_numpy(), pairs.s.to_numpy(), pairs.prune_p.to_numpy().astype(np.float64)
    core_code, _ = pd.factorize(rec.name_core.to_numpy())
    core_code = np.where(rec.name_core.to_numpy() == "", -1 - np.arange(len(rec)), core_code)  # empty cores never collide
    country_code, _ = pd.factorize(rec.country.to_numpy())
    s1 = (rec.source == 1).to_numpy()
    dup = pd.Series(1, index=pd.MultiIndex.from_arrays([country_code[s1], core_code[s1]])).groupby(level=[0, 1]).size()
    name_dup_country = dup.reindex(pd.MultiIndex.from_arrays([country_code[s], core_code[s]])).fillna(1).to_numpy()
    key = q.astype(np.int64) * (core_code.max() + len(rec) + 2) + (core_code[s] + len(rec) + 1)
    ps = pd.Series(p)
    name_dup_cand = ps.groupby(key).transform("size").to_numpy()
    crowd_rank = ps.groupby(key).rank(ascending=False, method="first").to_numpy()
    best = ps.groupby(key).transform("max").to_numpy()
    second = _second_best(key, p)
    crowd_margin = np.where(name_dup_cand > 1, np.where(p >= best, p - second, p - best), 1.0)
    q_core_equal = (core_code[q] == core_code[s]).astype(np.float32)

    def s1_count(codes: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """How many S1 entities of the row's country share the code (0 for empty values)."""
        table = pd.Series(1, index=pd.MultiIndex.from_arrays([country_code[s1], codes[s1]])).groupby(level=[0, 1]).size()
        got = table.reindex(pd.MultiIndex.from_arrays([country_code[rows], codes[rows]])).fillna(0).to_numpy()
        return np.where(codes[rows] < 0, 0, got)

    def text_codes(values: np.ndarray) -> np.ndarray:
        codes, _ = pd.factorize(values)
        return np.where(values == "", -1, codes)

    addr_code = text_codes(rec.addr.to_numpy())
    street_code = text_codes(np.array([street_key(p) for p in rec.addr_parts.to_numpy()], dtype=object))
    extra = (s1_count(addr_code, s), s1_count(street_code, s), s1_count(core_code, q), s1_count(addr_code, q),
             s1_count(street_code, q))
    out = pairs[["q", "s"]].reset_index(drop=True)
    for name, arr in zip(CROWD_FEATURES, (name_dup_country, name_dup_cand, crowd_rank, crowd_margin, q_core_equal, *extra)):
        out[name] = np.asarray(arr, dtype=np.float32)
    return out


def street_key(parts: str) -> str:
    """Non-digit tokens of the first address part that contains a number (the street)."""
    for part in parts.split(" | "):
        if any(c.isdigit() for c in part):
            return " ".join(t for t in part.split() if not any(c.isdigit() for c in t))
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["train", "dev", "test"])
    a = ap.parse_args()
    out = C.tag_dir(a.tag)
    log = C.log_to(f"{out}/crowd.log")
    t0 = C.stage_start(log, f"crowd {a.tag}")
    rec = load_records(a.tag, ["source", "country", "name_core", "addr", "addr_parts"])
    pairs = pd.read_parquet(f"{out}/features.parquet", columns=["q", "s", "prune_p"])
    df = build(pairs, rec)
    df.to_parquet(f"{out}/features_crowd.parquet", index=False)
    log(f"{len(df):,} pairs; name_dup_cand > 1 for {(df.name_dup_cand > 1).mean():.3f}; "
        f"S1 name shared in country for {(df.name_dup_country > 1).mean():.3f}")
    C.stage_end(log, f"crowd {a.tag}", t0)


if __name__ == "__main__":
    main()
