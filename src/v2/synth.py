"""Synthetic next-door look-alikes (training data only).

Test holds ~1.6-1.9x more same-name, same-street candidates with a nearby,
different house number than train (shift study), and the model scores them too
high (a fixed logit shift of 2.5 and house-gap features both gained far more on
the leaderboard than on holdout). Here, for sampled true pairs (S2/S3 record q,
S1 s) whose record has a house number, a copy of q with only the house number
moved by 1..GAP_MAX (same digit count) becomes a NON-match candidate of s.

  python -m src.v2.synth     -> OUT_DIR/aug/ : records + features (+ house, novel, crowd)
                               with synthetic rows appended (records.synthetic = True)

Training-fold S1s get synthetic negatives for training; tune-fold S1s get them
so calibration and decoding are tuned for test's look-alike density (no fixed
logit shift needed); holdout S1s get them as a "test-like holdout", so models
trained with and without them compare honestly. Synthetic blocking scores,
ranks and context copy the source pair.
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import shutil

import numpy as np
import pandas as pd

from src.v2 import config as C
from src.v2.common import CONTEXT_COLS, channel_columns, labels
from src.v2.features import build as build_features
from src.v2.house import build as build_house
from src.v2.normalize import skeleton
from src.v2.novel import build as build_novel

GAP_MAX = 25
# expected extra look-alikes per S1 (test minus train, known-different cases, shift study)
EXTRA_PER_S1 = {"US": 0.6, "India": 0.2}
_DIGITS = re.compile(r"\d+")


# house-number operators of test's extra look-alikes (shift study: different, one digit substituted, +-1/2, swapped)
OPERATORS = (("move", 0.55), ("substitute", 0.25), ("near", 0.12), ("swap", 0.08))
SIBLING_WORD_SHARE = 0.25  # share of look-alikes that also append a new word to the name


def move_number(house: str, rng: np.random.Generator, op: str = "move") -> str | None:
    """The house number's first digit run changed by one operator, same digit count; None if impossible."""
    m = _DIGITS.search(house)
    if not m:
        return None
    d = m.group(0)
    for _ in range(10):
        if op == "substitute":
            i = int(rng.integers(0, len(d)))
            new_d = d[:i] + str(int(rng.integers(0, 10))) + d[i + 1:]
        elif op == "swap" and len(d) >= 2:
            i = int(rng.integers(0, len(d) - 1))
            new_d = d[:i] + d[i + 1] + d[i] + d[i + 2:]
        else:
            gap = int(rng.integers(1, 3 if op == "near" else GAP_MAX + 1))
            new = int(d) + gap * (1 if rng.random() < 0.5 else -1)
            new_d = str(new).zfill(len(d)) if new > 0 else d
        if new_d != d and len(new_d) == len(d) and (new_d[0] != "0" or d[0] == "0"):
            return house[:m.start()] + new_d + house[m.end():]
    return None


def sibling_words(rec: pd.DataFrame, feats: pd.DataFrame, y: np.ndarray, rng: np.random.Generator,
                  n: int = 1_500_000, top: int = 300) -> list[str]:
    """Words appended to names of look-alike NON-matches and (almost) never in true matches, learned from
    similar-name training pairs (the sibling study); used to build look-alikes with a new word."""
    from collections import Counter

    from rapidfuzz import fuzz, process
    train = np.flatnonzero(rec.fold.to_numpy()[feats.s.to_numpy()] == "train")
    idx = rng.choice(train, min(n, len(train)), replace=False)
    q, s, lab = feats.q.to_numpy()[idx], feats.s.to_numpy()[idx], y[idx]
    core = rec.name_core.to_numpy()
    sim = process.cpdist(core[q].tolist(), core[s].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
    neg, pos = Counter(), Counter()
    for qq, ss, l, sm in zip(q, s, lab, sim):
        if sm < 85:
            continue
        extra = set(core[qq].split()) - set(core[ss].split())
        (pos if l else neg).update(w for w in extra if len(w) >= 3 and w.isalpha())
    scored = [(n_ / (pos.get(w, 0) + 1), w) for w, n_ in neg.items() if n_ >= 50 and pos.get(w, 0) * 20 <= n_]
    return [w for _, w in sorted(scored, reverse=True)[:top]]


def replace_first(text: str, old: str, new: str) -> str:
    return re.sub(rf"(?<!\w){re.escape(old)}(?!\w)", new, text, count=1) if old and old != new else text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=C.WORKERS)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()
    src, out = C.tag_dir("train"), C.tag_dir("aug")
    log = C.log_to(f"{out}/synth.log")
    t0 = C.stage_start(log, "synth")
    rng = np.random.default_rng(a.seed)
    rec = pd.read_parquet(f"{src}/records.parquet")
    feats = pd.read_parquet(f"{src}/features.parquet")
    y = labels(feats, rec.owner.to_numpy())
    q, s = feats.q.to_numpy(), feats.s.to_numpy()
    fold, country, house = rec.fold.to_numpy(), rec.country.to_numpy(), rec.house_num.to_numpy()
    eligible = y & np.array([bool(_DIGITS.search(h)) for h in house[q]])
    # per country: sample true pairs so that each S1 gains EXTRA_PER_S1 look-alikes on average
    take = np.zeros(len(feats), bool)
    for c, extra in EXTRA_PER_S1.items():
        m = eligible & (country[s] == c)
        n_s1 = ((rec.source == 1) & (rec.country == c)).sum()
        prob = min(1.0, extra * n_s1 / max(m.sum(), 1))
        take |= m & (rng.random(len(feats)) < prob)
    rows = np.flatnonzero(take)
    log(f"true pairs with a house number: {eligible.sum():,}; sampled for synthetic look-alikes: {len(rows):,}")

    words = sibling_words(rec, feats, y, rng)
    log(f"sibling words learned from training ({len(words)}): {words[:25]}")
    ops, weights = zip(*OPERATORS)
    new_recs, keep_rows = [], []
    for i in rows:
        r = rec.iloc[q[i]]
        h2 = move_number(r.house_num, rng, str(rng.choice(ops, p=np.array(weights) / sum(weights))))
        if h2 is None:
            continue
        d_old, d_new = _DIGITS.search(r.house_num).group(0), _DIGITS.search(h2).group(0)
        nr = r.copy()
        nr["house_num"] = h2
        nr["addr"] = replace_first(r.addr, d_old, d_new) if d_old in r.addr else r.addr
        nr["addr_parts"] = replace_first(r.addr_parts, d_old, d_new)
        if words and rng.random() < SIBLING_WORD_SHARE:  # sibling business: a new word appended to the name
            w = words[int(rng.integers(0, len(words)))]
            legal = [t for t in r["name"].split() if t not in set(r.name_core.split())]
            core2 = (r.name_core + " " + w).strip()
            nr["name_core"] = core2
            nr["name"] = " ".join([core2, *legal])
            nr["name_skeleton"] = skeleton(core2)
            addr_tokens = set(nr["addr"].split())
            nr["name_core_minus_addr"] = " ".join(t for t in core2.split() if t not in addr_tokens)
        nr["entity_id"] = f"SYN-{len(new_recs)}"
        nr["owner"] = -1
        new_recs.append(nr)
        keep_rows.append(i)
    syn = pd.DataFrame(new_recs)
    syn["rid"] = np.arange(len(rec), len(rec) + len(syn), dtype=np.int32)
    rec["synthetic"] = False
    syn["synthetic"] = True
    rec_ext = pd.concat([rec, syn[rec.columns]], ignore_index=True)
    keep_rows = np.asarray(keep_rows)
    log(f"synthetic records: {len(syn):,} (train-fold S1 {np.isin(fold[s[keep_rows]], ['train']).sum():,}, "
        f"tune {np.isin(fold[s[keep_rows]], ['tune']).sum():,}, holdout {np.isin(fold[s[keep_rows]], ['holdout']).sum():,})")

    base = feats.iloc[keep_rows].reset_index(drop=True)
    pairs_syn = base.copy()
    pairs_syn["q"] = syn.rid.to_numpy()
    with open(f"{C.MODEL_DIR}/idf.pkl", "rb") as f:
        idf = pickle.load(f)
    channel_cols = channel_columns(feats)
    f_syn = build_features(pairs_syn[["q", "s", *channel_cols, "prune_p"]], rec_ext, a.workers, idf)
    for c in CONTEXT_COLS:  # candidate-context columns copy the source pair (its group, not the synthetic subset)
        if c in f_syn.columns:
            f_syn[c] = base[c].to_numpy()
    f_syn = f_syn[feats.columns]
    pd.concat([feats, f_syn], ignore_index=True).to_parquet(f"{out}/features.parquet", index=False)
    rec_ext.to_parquet(f"{out}/records.parquet", index=False)

    for name, builder in (("features_house.parquet", build_house), ("features_novel.parquet", build_novel)):
        orig = pd.read_parquet(f"{src}/{name}")
        add = builder(f_syn[["q", "s"]], rec_ext, a.workers)[orig.columns]
        pd.concat([orig, add], ignore_index=True).to_parquet(f"{out}/{name}", index=False)
    crowd = pd.read_parquet(f"{src}/features_crowd.parquet")
    add = crowd.iloc[keep_rows].reset_index(drop=True).assign(q=syn.rid.to_numpy())
    pd.concat([crowd, add], ignore_index=True).to_parquet(f"{out}/features_crowd.parquet", index=False)
    for extra in ("ids.parquet",):
        if os.path.exists(f"{src}/{extra}"):
            shutil.copy2(f"{src}/{extra}", f"{out}/{extra}")
    log(f"wrote {out}: {len(rec_ext):,} records, {len(feats) + len(f_syn):,} pairs")
    C.stage_end(log, "synth", t0)


if __name__ == "__main__":
    main()
