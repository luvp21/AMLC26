"""Shared loading / feature assembly for the validation harness scripts:
normalized sources (cached per NormalizationConfig), id lookups, pair tables
and feature matrices (cached per config hash), and the per-host experiment log.
"""
from __future__ import annotations

import csv
import os
import socket

import numpy as np
import pandas as pd

from src.entity_resolution.cache import cached_frame, cached_object
from src.entity_resolution.config import config_hash
from src.entity_resolution.evaluate import pair_frame
from src.entity_resolution.features import build_feature_matrix_fast
from src.entity_resolution.pipeline import add_normalized_columns_cfg

LOOK_COLS = ["business_name", "business_address", "name_tokens", "addr_tokens", "country"]


def experiments_csv_path() -> str:
    """One experiment log per machine (merged in git), so rsync between the
    laptop and PARAM Shavak can never overwrite results."""
    return f"experiments_{socket.gethostname()}.csv"


def append_experiment(row: dict, path: str) -> None:
    """Append a row; if it brings new columns, rewrite the file with the union."""
    if os.path.exists(path):
        log = pd.read_csv(path, dtype=str, keep_default_na=False)
        if set(row) - set(log.columns):
            log = pd.concat([log, pd.DataFrame([row]).astype(str)], ignore_index=True).fillna("")
            log.to_csv(path, index=False)
            return
        fields = list(log.columns)
    else:
        fields = list(row)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def load_source(root, split, source, norm_cfg, workers, log) -> pd.DataFrame:
    key = config_hash(norm_cfg, os.path.abspath(root), split, source)

    def build():
        df = pd.read_csv(f"{root}/{split}/{split}_source{source}.tsv", sep="\t", dtype=str, keep_default_na=False)
        return add_normalized_columns_cfg(df, norm_cfg, n_workers=workers)

    return cached_frame(f"norm_{split}_s{source}", key, build, log)


def lookup(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("entity_id")[LOOK_COLS]


def feature_input(pairs, s1_look, cand_look, text: str = "raw") -> pd.DataFrame:
    """Columns build_feature_matrix_fast expects. text="raw" feeds the raw
    strings to the fuzzy ratios (baseline); "casefold" the raw strings
    lowercased and nothing else; "normalized" the joined tokens."""
    if text == "raw":
        ren = {"business_name": "name", "business_address": "addr"}
        s1_look, cand_look = s1_look.rename(columns=ren), cand_look.rename(columns=ren)
    elif text == "casefold":
        s1_look, cand_look = (lk.assign(name=lk.business_name.str.lower(), addr=lk.business_address.str.lower())
                              for lk in (s1_look, cand_look))
    elif text == "normalized":
        s1_look, cand_look = (lk.assign(name=lk.name_tokens.map(" ".join), addr=lk.addr_tokens.map(" ".join))
                              for lk in (s1_look, cand_look))
    else:
        raise ValueError(f"unknown feature text mode {text!r}")
    cols = ["name", "addr", "name_tokens", "addr_tokens", "country"]
    left = pairs[["s1_id", "candidate_id"]]
    out = left.merge(s1_look[cols].add_prefix("s1_"), left_on="s1_id", right_index=True, how="left")
    out = out.merge(cand_look[cols].add_prefix("c_"), left_on="candidate_id", right_index=True, how="left")
    if out.c_name.isna().any() or out.s1_name.isna().any():
        raise ValueError("some candidate or source-1 ids are missing from the lookups")
    return out


def fold_features(name, key, ids, candidates, gt_sets, s1_look, cand_look, workers, text, log):
    pairs = cached_frame(f"pairs_{name}", key, lambda: pair_frame(ids, candidates, gt_sets), log)
    X = cached_object(f"X_{name}", key, lambda: build_feature_matrix_fast(
        feature_input(pairs, s1_look, cand_look, text), n_workers=workers), log)
    return pairs, X


def subset(ents, pairs, X, country):
    e = ents[ents.country == country].reset_index(drop=True)
    mask = pairs.s1_id.isin(set(e.s1_id)).to_numpy()
    return e, pairs[mask].reset_index(drop=True), X[mask]


def record_exclusive(candidate_ids, p) -> np.ndarray:
    """Record-level exclusivity: True only for each candidate record's
    highest-probability pair (ties -> first occurrence)."""
    codes = pd.factorize(np.asarray(candidate_ids))[0]
    p = np.asarray(p)
    order = np.lexsort((-p, codes))
    first = np.ones(len(order), dtype=bool)
    first[1:] = codes[order][1:] != codes[order][:-1]
    keep = np.zeros(len(p), dtype=bool)
    keep[order[first]] = True
    return keep
