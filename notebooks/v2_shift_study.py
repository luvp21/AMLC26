"""Train/test shift study (label-free): look-alike density per S1.

For every S1 in the train holdout and in test, count its pruned candidates that
look like a neighbour business: similar name (core token_set >= 90), same street
(non-digit tokens of the house-number part equal), and a house-number relation
from src/v2/house.py. Compares per country and per relation, so the size of the
shift (and whether it differs by country) is known before choosing a correction.

  PYTHONPATH=. python3 notebooks/v2_shift_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from src.v2 import config as C
from src.v2.house import REL

REL_NAME = {v: k for k, v in REL.items()}


def street(parts: str) -> str:
    for p in parts.split(" | "):
        if any(c.isdigit() for c in p):
            return " ".join(sorted(t for t in p.split() if not any(c.isdigit() for c in t)))
    return ""


def study(tag: str, s1_filter) -> pd.DataFrame:
    out = C.tag_dir(tag)
    rec = pd.read_parquet(f"{out}/records.parquet", columns=["rid", "source", "country", "fold", "name_core", "addr_parts"])
    pairs = pd.read_parquet(f"{out}/features.parquet", columns=["q", "s"])
    house = pd.read_parquet(f"{out}/features_house.parquet", columns=["house_rel"]).house_rel.to_numpy()
    keep = s1_filter(rec)[pairs.s.to_numpy()]
    q, s, rel = pairs.q.to_numpy()[keep], pairs.s.to_numpy()[keep], house[keep]
    core = rec.name_core.to_numpy()
    sim = process.cpdist(core[q].tolist(), core[s].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
    parts = rec.addr_parts.to_numpy()
    same_street = np.array([bool(a) and a == b for a, b in
                            zip(map(street, parts[q]), map(street, parts[s]))]) if len(q) else np.zeros(0, bool)
    country = rec.country.to_numpy()[s]
    n_s1 = pd.Series(s1_filter(rec)[rec.rid.to_numpy()] & (rec.source == 1).to_numpy()).groupby(rec.country.to_numpy()).sum()
    df = pd.DataFrame({"country": country, "rel": [REL_NAME[int(r)] for r in rel], "similar": sim >= 90, "street": same_street})
    looks = df[df.similar & df.street]
    table = looks.groupby(["country", "rel"]).size().unstack(fill_value=0)
    table = table.div(n_s1.reindex(table.index), axis=0)  # per S1
    table["similar_same_street_total"] = table.sum(axis=1)
    return table


def main() -> None:
    pd.set_option("display.width", 220)
    hold = study("train", lambda rec: (rec.fold == "holdout").to_numpy())
    test = study("test", lambda rec: np.ones(len(rec), bool))
    print("=== per S1: candidates with similar name + same street, by house-number relation ===")
    print("--- train holdout")
    print(hold.round(3).to_string())
    print("--- test")
    print(test.round(3).to_string())
    common = hold.index.intersection(test.index)
    print("--- test / train ratio (known countries)")
    print((test.loc[common] / hold.loc[common].replace(0, np.nan)).round(2).to_string())


if __name__ == "__main__":
    main()
