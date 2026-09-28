"""Stage 1: normalize every record of a split and attach partition keys.

  python -m src.v2.prepare --split train     # also learns the key vocabulary, folds, owners
  python -m src.v2.prepare --split test      # uses the train key vocabulary

Writes OUT_DIR/<split>/records.parquet: one row per record (S1, S2, S3) with a
split-wide integer id `rid`, raw columns, normalized columns, `key` (partition
key), and for train: `fold` (S1) and `owner` (rid of the owning S1, S2/S3).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import time
from collections import Counter

import numpy as np
import pandas as pd

from src.metrics import parse_id_list
from src.v2 import config as C
from src.v2.normalize import normalize_record

_LEARNED: dict = {}


def _norm_row(args):
    name, addr, country = args
    learned = _LEARNED.get(country, {})
    return normalize_record(name, addr, country, **learned)


def normalize_frame(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    rows = list(zip(df.business_name, df.business_address, df.country))
    with mp.get_context("fork").Pool(workers) as pool:
        out = pool.map(_norm_row, rows, chunksize=5000)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def learn_key_vocab(rec: pd.DataFrame) -> dict:
    """{(source, country): Counter(part)} over digit-free address parts."""
    vocab = {}
    for (src, country), g in rec.groupby(["source", "country"]):
        cnt = Counter(p for parts in g.addr_parts for p in parts.split(" | ") if p and not any(c.isdigit() for c in p))
        vocab[(src, country)] = {p: n for p, n in cnt.items() if n >= C.KEY_MIN_COUNT}
    return vocab


def assign_keys(rec: pd.DataFrame, vocab: dict) -> np.ndarray:
    """key = the record's most frequent known part (state, else city); "" if none."""
    keys = np.full(len(rec), "", dtype=object)
    for (src, country), idx in rec.groupby(["source", "country"]).indices.items():
        v = vocab.get((src, country))
        if not v:
            continue
        parts = rec.addr_parts.to_numpy()[idx]
        keys[idx] = [max((p for p in ps.split(" | ") if p in v), key=v.get, default="") for ps in parts]
    return keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--workers", type=int, default=C.WORKERS)
    ap.add_argument("--raw", action="store_true",
                    help="ignore learned tables and write records_raw.parquet (input to src.v2.learn)")
    a = ap.parse_args()
    out = C.tag_dir(a.split)
    log = C.log_to(f"{out}/prepare.log")
    t0 = C.stage_start(log, f"prepare {a.split}")

    learned_path = f"{C.model_dir()}/learned_normalization.pkl"
    if a.raw:
        log("raw pass: learned normalization tables not applied")
    elif os.path.exists(learned_path):
        with open(learned_path, "rb") as f:
            _LEARNED.update(pickle.load(f))
        log("learned normalization tables: " + ", ".join(
            f"{c} ({len(t['part_rewrites'])} part, {len(t['rewrites'])} word, {len(t['word_dict'])} dict)"
            for c, t in sorted(_LEARNED.items())))
    else:
        log("no learned normalization tables")

    frames = []
    for src in (1, 2, 3):
        df = pd.read_csv(f"{C.DATA_ROOT}/{a.split}/{a.split}_source{src}.tsv", sep="\t", dtype=str, keep_default_na=False)
        df["source"] = src
        frames.append(df)
        log(f"source {src}: {len(df):,} rows")
    rec = pd.concat(frames, ignore_index=True)
    rec.insert(0, "rid", np.arange(len(rec), dtype=np.int32))
    rec = normalize_frame(rec, a.workers)
    log(f"normalized {len(rec):,} records in {time.time() - t0:.0f}s")

    vocab_path = f"{C.model_dir()}/key_vocab.pkl"
    if a.split == "train":
        vocab = learn_key_vocab(rec)
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)
    else:
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
    rec["key"] = assign_keys(rec, vocab)
    for (src, country), g in rec.groupby(["source", "country"]):
        log(f"  s{src} {country}: {len(g):,} records, key coverage {(g.key != '').mean():.3f}, "
            f"{g.key.nunique():,} keys; non-latin names {g.non_latin.mean():.3f}; empty address {(g.addr == '').mean():.3f}")

    rec["fold"] = ""
    rec["xgroup"] = np.int8(-1)
    rec["owner"] = np.int32(-1)
    if a.split == "train":
        s1 = rec.source == 1
        rec.loc[s1, "fold"] = C.fold_of(rec.loc[s1, "entity_id"])
        rec.loc[s1, "xgroup"] = C.xgroup_of(rec.loc[s1, "entity_id"])
        gt = pd.read_csv(f"{C.DATA_ROOT}/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
        rid_of = dict(zip(rec.entity_id.to_numpy(), rec.rid.to_numpy()))
        owner = {m: rid_of[s] for s, ms in zip(gt.source1_entity_id, gt.matched_entity_ids) for m in parse_id_list(ms)}
        q = rec.source != 1
        rec.loc[q, "owner"] = rec.loc[q, "entity_id"].map(owner).fillna(-1).astype(np.int32).to_numpy()
        log(f"folds: {rec.loc[s1, 'fold'].value_counts().to_dict()}; owned S2/S3: {(rec.owner >= 0).sum():,}")
    name = "records_raw" if a.raw else "records"
    rec.to_parquet(f"{out}/{name}.parquet", index=False)
    rec[["rid", "entity_id"]].to_parquet(f"{out}/ids.parquet", index=False)  # int32 code <-> entity id
    log(f"wrote {out}/{name}.parquet and ids.parquet")
    C.stage_end(log, f"prepare {a.split}", t0)


if __name__ == "__main__":
    main()
