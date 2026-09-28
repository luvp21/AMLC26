"""Step 1 study (training data only): how do look-alike non-matches ("siblings")
differ from true matches? For a sample of training-fold S1 entities, every
blocked candidate with a similar name (token_set_ratio >= 85 on name_core) is
split into true matches and non-matches, then compared on extra name words,
house-number relation, same street, legal form and exact name equality.

  PYTHONPATH=. python3 notebooks/v2_sibling_study.py --n 60000
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from src.v2.house import REL, relation

LEGAL = {"private", "pvt", "limited", "ltd", "llp", "inc", "corp", "llc", "co", "company", "sa", "sas", "sarl"}
REL_NAME = {v: k for k, v in REL.items()}


def street(parts: str) -> frozenset:
    for p in parts.split(" | "):
        if any(c.isdigit() for c in p):
            return frozenset(t for t in p.split() if not any(c.isdigit() for c in t))
    return frozenset()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--root", default="runs/v2/train")
    a = ap.parse_args()
    rec = pd.read_parquet(f"{a.root}/records.parquet", columns=["rid", "source", "country", "fold", "owner", "name_core",
                                                               "addr", "addr_parts", "house_num", "name_legal_class"])
    s1 = rec[(rec.source == 1) & (rec.fold == "train")]
    sample = s1.sample(a.n, random_state=42).rid.to_numpy()
    pairs = pd.read_parquet(f"{a.root}/pairs_block.parquet", columns=["q", "s"], filters=[("s", "in", sample.tolist())])
    q, s = pairs.q.to_numpy(), pairs.s.to_numpy()
    core, owner = rec.name_core.to_numpy(), rec.owner.to_numpy()
    sim = process.cpdist(core[q].tolist(), core[s].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
    keep = sim >= 85
    q, s, sim = q[keep], s[keep], sim[keep]
    y = owner[q] == s
    print(f"sampled S1 {len(sample):,}; blocked pairs {len(pairs):,}; similar-name pairs {len(q):,} "
          f"({y.sum():,} true, {(~y).sum():,} look-alike non-matches)")

    parts, house, legal, country = rec.addr_parts.to_numpy(), rec.house_num.to_numpy(), rec.name_legal_class.to_numpy(), rec.country.to_numpy()
    rows = []
    extra_words = {True: Counter(), False: Counter()}
    for qq, ss, lab, sm in zip(q, s, y, sim):
        tq = set(core[qq].split()) - LEGAL
        ts = set(core[ss].split()) - LEGAL
        eq, es = tq - ts, ts - tq
        for w in eq:
            extra_words[bool(lab)][w] += 1
        rows.append((country[ss], lab, len(eq), len(es), core[qq] == core[ss], REL_NAME[relation(house[qq], house[ss])],
                     street(parts[qq]) == street(parts[ss]) and bool(street(parts[ss])), legal[qq] == legal[ss]))
    df = pd.DataFrame(rows, columns=["country", "match", "extra_in_query", "extra_in_s1", "name_equal", "house_rel",
                                     "same_street", "legal_equal"])
    pd.set_option("display.width", 200)

    def rate(col):
        t = df.groupby(["country", col]).match.agg(["size", "mean"]).rename(columns={"size": "pairs", "mean": "P(match)"})
        print(f"\n--- P(match | similar name, {col})\n" + t.round(4).to_string())

    for col in ("name_equal", "extra_in_query", "extra_in_s1", "house_rel", "same_street", "legal_equal"):
        rate(col)
    df["extra_any"] = (df.extra_in_query > 0) | (df.extra_in_s1 > 0)
    t = df.groupby(["country", "extra_any", "house_rel"]).match.agg(["size", "mean"])
    print("\n--- P(match | extra word on either side, house relation)\n" + t[t["size"] >= 200].round(4).to_string())
    t = df[df.same_street].groupby(["country", "extra_any", "house_rel"]).match.agg(["size", "mean"])
    print("\n--- same street only: P(match | extra word, house relation)\n" + t[t["size"] >= 200].round(4).to_string())
    print("\n--- extra words in the query name: most frequent among non-matches vs matches")
    neg, pos = extra_words[False], extra_words[True]
    tot_n, tot_p = max(sum(neg.values()), 1), max(sum(pos.values()), 1)
    print(pd.DataFrame([(w, n, pos.get(w, 0), round((n / tot_n) / ((pos.get(w, 0) + 1) / tot_p), 1))
                        for w, n in neg.most_common(40)], columns=["word", "in_non_matches", "in_matches", "ratio"]).to_string(index=False))


if __name__ == "__main__":
    main()
