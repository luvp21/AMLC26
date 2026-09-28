"""Novel-extra-word features (M4, from the sibling study on training data).

Among similar-name pairs in training, look-alike businesses are made by
APPENDING a real new word (group, holdings, care, clinic, east, ...: these occur
only in non-matches), while copies of the same business only DISTORT words it
already has (limitet, piraivet for limited/private; honorifics; ".com").
Learned word lists do not carry over to other languages (France: groupe,
france, international, ...), so the features here are language-independent:
an extra word is "novel" if it is not a variant of any word on the other side.

  python -m src.v2.novel --tag train|test   -> OUT_DIR/<tag>/features_novel.parquet
  (rows aligned with features.parquet: q, s + NOVEL_FEATURES)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler

from src.v2 import config as C
from src.v2.common import load_records
from src.v2.normalize import FR_LEAD_LEGAL, FR_TAIL_LEGAL, LEGAL_CLASS_OF

NOVEL_FEATURES = ["novel_q", "novel_s", "novel_any", "extra_q", "extra_s", "novel_q_len"]
VARIANT_JW = 0.75  # similarity to a word of the other name at/above which an extra word is a variant
LEGAL_JW = 0.80  # against the legal vocabulary, with the same first letter (limitet, piraivet pass; sportive does not)
MIN_LEN = 3
_LEGAL = sorted(set(LEGAL_CLASS_OF) | set(FR_LEAD_LEGAL) | set(FR_TAIL_LEGAL))
_R: dict = {}


def novel_words(core_a: str, name_b: str) -> tuple[list[str], int]:
    """(novel extra words of a vs b, number of all extra words of a) for a's name core
    against b's full name (legal words included) plus the legal vocabulary."""
    b_tokens = set(name_b.split())
    extra = [t for t in core_a.split() if t not in b_tokens]

    def is_variant(t):
        return (any(JaroWinkler.normalized_similarity(t, w) >= VARIANT_JW for w in b_tokens)
                or any(t[0] == w[0] and JaroWinkler.normalized_similarity(t, w) >= LEGAL_JW for w in _LEGAL))
    novel = [t for t in extra if len(t) >= MIN_LEN and not is_variant(t)]
    return novel, len(extra)


def pair_features(q_core: str, q_name: str, s_core: str, s_name: str) -> tuple:
    nq, eq = novel_words(q_core, s_name)
    ns, es = novel_words(s_core, q_name)
    return len(nq), len(ns), float(bool(nq or ns)), eq, es, max((len(t) for t in nq), default=0)


def _chunk(bounds):
    lo, hi = bounds
    q, s = _R["q"][lo:hi], _R["s"][lo:hi]
    core, name = _R["name_core"], _R["name"]
    rows = [pair_features(core[a], name[a], core[b], name[b]) for a, b in zip(q, s)]
    return np.asarray(rows, dtype=np.float32).reshape(len(q), len(NOVEL_FEATURES))


def build(pairs: pd.DataFrame, rec: pd.DataFrame, workers: int) -> pd.DataFrame:
    global _R
    _R = {c: rec[c].to_numpy() for c in ("name_core", "name")}
    _R["q"], _R["s"] = pairs.q.to_numpy(), pairs.s.to_numpy()
    n = len(pairs)
    size = max(1, min(500_000, n // (workers * 4) + 1))
    with mp.get_context("fork").Pool(workers) as pool:
        parts = pool.map(_chunk, [(i, min(i + size, n)) for i in range(0, n, size)])
    arr = np.concatenate(parts) if parts else np.zeros((0, len(NOVEL_FEATURES)), np.float32)
    return pd.concat([pairs[["q", "s"]].reset_index(drop=True), pd.DataFrame(arr, columns=NOVEL_FEATURES)], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--workers", type=int, default=C.WORKERS)
    a = ap.parse_args()
    out = C.tag_dir(a.tag)
    log = C.log_to(f"{out}/novel.log")
    t0 = C.stage_start(log, f"novel {a.tag}")
    rec = load_records(a.tag, ["name_core", "name"])
    pairs = pd.read_parquet(f"{out}/features.parquet", columns=["q", "s"])
    df = build(pairs, rec, a.workers)
    df.to_parquet(f"{out}/features_novel.parquet", index=False)
    log(f"{len(df):,} pairs; novel_any rate {df.novel_any.mean():.3f}; mean novel_q {df.novel_q.mean():.3f}")
    C.stage_end(log, f"novel {a.tag}", t0)


if __name__ == "__main__":
    main()
