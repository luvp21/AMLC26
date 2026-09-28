"""House-number relation features (M4 error mining: the largest loss category).

In true training matches the house number changes by a few fixed operators:
leading zeros added (452 -> 0452), one digit dropped or added (2905 -> 905),
a new unit number prepended while the original appears later in the address,
a letter prefix/suffix, one digit substituted, +-1/2. The M1 house_state only
says same / same digits / different / missing, so these look like a different
building. Features here name the operator and check containment.

  python -m src.v2.house --tag train|test    -> OUT_DIR/<tag>/features_house.parquet
  (rows aligned with features.parquet: q, s + HOUSE_FEATURES)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import re

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from src.v2 import config as C
from src.v2.common import load_records

HOUSE_FEATURES = ["house_rel", "house_digit_edit", "house_in_other", "house_other_in_self", "house_num_overlap",
                  "house_gap", "house_gap_log", "house_gap_small", "house_same_len"]
GAP_SMALL = 20  # next-door numbers: test look-alikes have a median gap of ~11, train's true variants ~300
HOUSE_CATEGORICAL = ["house_rel"]
REL = {"equal": 0, "letters_only": 1, "digit_dropped_or_added": 2, "digit_substituted": 3, "near_1_2": 4,
       "permuted": 5, "prefix_or_extension": 6, "different": 7, "missing": 8}
_DIG = re.compile(r"\d+")
_R: dict = {}


def _digits(h: str) -> str:
    """Digits of a house number with leading zeros stripped ('0452' -> '452', 'd406' -> '406')."""
    d = "".join(_DIG.findall(h))
    return d.lstrip("0") or ("0" if d else "")


def relation(a: str, b: str) -> int:
    if not a or not b:
        return REL["missing"]
    if a == b:
        return REL["equal"]
    da, db = _digits(a), _digits(b)
    if not da or not db:
        return REL["different"]
    if da == db:
        return REL["letters_only"] if a.lstrip("0") != b.lstrip("0") else REL["equal"]
    if abs(len(da) - len(db)) == 1 and Levenshtein.distance(da, db) == 1:
        return REL["digit_dropped_or_added"]
    if len(da) == len(db) and Levenshtein.distance(da, db) == 1:
        return REL["near_1_2"] if abs(int(da) - int(db)) <= 2 else REL["digit_substituted"]
    if sorted(da) == sorted(db):
        return REL["permuted"]
    if da.startswith(db) or db.startswith(da):
        return REL["prefix_or_extension"]
    return REL["different"]


def _numbers(house: str, nums: str, postcode: str) -> set[str]:
    """All digit runs of an address except the postcode, leading zeros stripped."""
    out = {_digits(t) for t in nums.split()} | {_digits(x) for x in _DIG.findall(house)}
    out.discard("")
    out.discard(postcode.lstrip("0"))
    return out


def pair_features(qh, qn, qp, sh, sn, sp) -> tuple:
    rel = relation(qh, sh)
    dq, ds = _digits(qh), _digits(sh)
    edit = Levenshtein.distance(dq, ds) if dq and ds else -1
    q_all, s_all = _numbers(qh, qn, qp), _numbers(sh, sn, sp)
    in_other = float(ds in q_all) if ds else -1.0  # S1's house number anywhere in the query's address
    other_in_self = float(dq in s_all) if dq else -1.0
    union = q_all | s_all
    overlap = len(q_all & s_all) / len(union) if union else -1.0
    if dq and ds and len(dq) <= 9 and len(ds) <= 9:
        gap = abs(int(dq) - int(ds))
        gap_log, small, same_len = float(np.log1p(gap)), float(0 < gap <= GAP_SMALL), float(len(dq) == len(ds))
    else:
        gap, gap_log, small, same_len = -1.0, -1.0, -1.0, -1.0
    return rel, edit, in_other, other_in_self, overlap, gap, gap_log, small, same_len


def _chunk(bounds):
    lo, hi = bounds
    q, s = _R["q"][lo:hi], _R["s"][lo:hi]
    h, n, p = _R["house_num"], _R["num_tokens"], _R["postcode"]
    rows = [pair_features(h[a], n[a], p[a], h[b], n[b], p[b]) for a, b in zip(q, s)]
    return np.asarray(rows, dtype=np.float32).reshape(len(q), len(HOUSE_FEATURES))


def build(pairs: pd.DataFrame, rec: pd.DataFrame, workers: int) -> pd.DataFrame:
    global _R
    _R = {c: rec[c].to_numpy() for c in ("house_num", "num_tokens", "postcode")}
    _R["q"], _R["s"] = pairs.q.to_numpy(), pairs.s.to_numpy()
    n = len(pairs)
    size = max(1, min(500_000, n // (workers * 4) + 1))
    with mp.get_context("fork").Pool(workers) as pool:
        parts = pool.map(_chunk, [(i, min(i + size, n)) for i in range(0, n, size)])
    arr = np.concatenate(parts) if parts else np.zeros((0, len(HOUSE_FEATURES)), np.float32)
    return pd.concat([pairs[["q", "s"]].reset_index(drop=True), pd.DataFrame(arr, columns=HOUSE_FEATURES)], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--workers", type=int, default=C.WORKERS)
    a = ap.parse_args()
    out = C.tag_dir(a.tag)
    log = C.log_to(f"{out}/house.log")
    t0 = C.stage_start(log, f"house {a.tag}")
    rec = load_records(a.tag, ["house_num", "num_tokens", "postcode"])
    pairs = pd.read_parquet(f"{out}/features.parquet", columns=["q", "s"])
    df = build(pairs, rec, a.workers)
    df.to_parquet(f"{out}/features_house.parquet", index=False)
    log(f"{len(df):,} pairs; house_rel distribution: "
        + ", ".join(f"{k}={(df.house_rel == v).mean():.3f}" for k, v in REL.items()))
    C.stage_end(log, f"house {a.tag}", t0)


if __name__ == "__main__":
    main()
