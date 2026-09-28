"""Helpers shared by v2 stages: record arrays indexed by rid, pair labels and
blocking-context columns (vectorized; pairs are sorted by q)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.v2 import config as C


CONTEXT_COLS = ["q_best", "margin", "q_cnt", "s_cnt", "s_first"]


def split_of(tag: str) -> str:
    return "test" if tag == "test" else "train"


def load_records(tag: str, columns: list[str]) -> pd.DataFrame:
    """records.parquet of the tag (its own, e.g. the synthetic-augmented 'aug'), else of
    the tag's split; row i has rid i."""
    import os
    own = f"{C.OUT_DIR}/{tag}/records.parquet"
    path = own if os.path.exists(own) else f"{C.tag_dir(split_of(tag))}/records.parquet"
    rec = pd.read_parquet(path, columns=["rid", *columns])
    if not (rec.rid.to_numpy() == np.arange(len(rec))).all():
        raise ValueError("records.parquet rows are not in rid order")
    return rec


def labels(pairs: pd.DataFrame, owner: np.ndarray) -> np.ndarray:
    return owner[pairs.q.to_numpy()] == pairs.s.to_numpy()


def channels_of(pairs: pd.DataFrame) -> list[str]:
    """Blocking channels present in a pair table (columns <ch>_score / <ch>_rank)."""
    return [c[:-6] for c in pairs.columns if c.endswith("_score") and f"{c[:-6]}_rank" in pairs.columns]


def channel_columns(pairs: pd.DataFrame) -> list[str]:
    return [col for ch in channels_of(pairs) for col in (f"{ch}_score", f"{ch}_rank")]


def add_context(pairs: pd.DataFrame, n_records: int) -> pd.DataFrame:
    """best / q_best / margin / q_cnt / s_cnt / s_first over the given pair set."""
    q, s = pairs.q.to_numpy(), pairs.s.to_numpy()
    chans = channels_of(pairs)
    best = np.max([pairs[f"{ch}_score"].to_numpy() for ch in chans], axis=0)
    q_best = pd.Series(best).groupby(q).transform("max").to_numpy()
    first = np.any([pairs[f"{ch}_rank"].to_numpy() == 0 for ch in chans], axis=0)
    return pairs.assign(
        best=best, q_best=q_best, margin=best - q_best,
        q_cnt=np.bincount(q, minlength=n_records)[q].astype(np.float32),
        s_cnt=np.bincount(s, minlength=n_records)[s].astype(np.float32),
        s_first=np.bincount(s, weights=first, minlength=n_records)[s].astype(np.float32),
    )
