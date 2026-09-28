"""Phase 2: print the fixed 50-entity France test sample with normalized
tokens before (baseline) and after (Phase 2 flags), for S1 and its first
candidates. Eyeball check only — nothing here feeds training.

  PYTHONPATH=. python3 notebooks/phase2_france_norm.py --candidates output/candidate_pairs.tsv
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.entity_resolution.config import NormalizationConfig
from src.entity_resolution.evaluate import france_sample_ids
from src.entity_resolution.normalize import normalize_record

AFTER = NormalizationConfig(keep_digits=True, country_abbrev=True, stopwords=True, landmarks=True,
                            legal_bag=True, extract_fields=True)
COLS = ["entity_id", "business_name", "business_address", "country"]


def read_rows(path: str, ids: set[str]) -> pd.DataFrame:
    parts = [c[c.entity_id.isin(ids)] for c in
             pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=COLS, chunksize=1_000_000)]
    return pd.concat(parts).set_index("entity_id")


def show(r, indent: str) -> str:
    before = normalize_record(r.business_name, r.business_address, r.country, NormalizationConfig())
    after = normalize_record(r.business_name, r.business_address, r.country, AFTER)
    extra = {k: after[k] for k in ("addr_landmark", "postal", "house_nums", "name_core") if after.get(k)}
    return (f"{indent}raw   : {r.business_name!r} | {r.business_address!r}\n"
            f"{indent}before: {before['name_tokens']} | {before['addr_tokens']}\n"
            f"{indent}after : {after['name_tokens']} | {after['addr_tokens']}\n"
            f"{indent}fields: {extra}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data_set/student_resource/dataset/test")
    ap.add_argument("--candidates", default="output/candidate_pairs.tsv")
    ap.add_argument("--n-cands", type=int, default=5)
    ap.add_argument("--out", default="runs/phase2_france_norm.txt")
    a = ap.parse_args()

    s1 = pd.read_csv(f"{a.root}/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False, usecols=COLS)
    sample = france_sample_ids(s1.entity_id, s1.country)
    cands = pd.read_csv(a.candidates, sep="\t", dtype=str, keep_default_na=False).set_index("source1_entity_id")
    lists = {s: [c for c in cands.loc[s, "candidate_entity_ids"].split(",") if c][:a.n_cands] for s in sample}
    need = {c for cs in lists.values() for c in cs}
    pool = pd.concat([read_rows(f"{a.root}/test_source{k}.tsv", need) for k in (2, 3)])
    s1 = s1.set_index("entity_id")
    with open(a.out, "w") as f:
        for s in sample:
            f.write(f"\n=== {s}\n" + show(s1.loc[s], "  "))
            for c in lists[s]:
                f.write(f"  -- {c}\n" + show(pool.loc[c], "     "))
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
