"""Disk cache for expensive artifacts, keyed by the hash of the config that
produced them. Tables go to parquet; anything else (dicts, arrays, models)
goes to pickle.
"""
from __future__ import annotations

import os
import pickle
from collections.abc import Callable

import pandas as pd

CACHE_DIR = "runs/cache"
LIST_COLUMNS = ("name_tokens", "addr_tokens", "addr_landmark", "postal_all", "house_nums", "name_core")


def _path(name: str, key: str, ext: str) -> str:
    return f"{CACHE_DIR}/{name}_{key}.{ext}"


def cached_frame(name: str, key: str, build: Callable[[], pd.DataFrame], log=print) -> pd.DataFrame:
    path = _path(name, key, "parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        # pyarrow hands list<string> cells back as numpy arrays; downstream code
        # concatenates token lists with `+`, which on arrays means elementwise add.
        for col in LIST_COLUMNS:
            if col in df.columns:
                df[col] = df[col].map(list)
        log(f"  cache hit: {path}")
        return df
    df = build()
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    log(f"  cached: {path}")
    return df


def cached_object(name: str, key: str, build: Callable[[], object], log=print):
    path = _path(name, key, "pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            log(f"  cache hit: {path}")
            return pickle.load(f)
    obj = build()
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path + ".tmp", path)
    log(f"  cached: {path}")
    return obj
