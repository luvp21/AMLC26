"""Leaderboard bookkeeping (never feeds training).

  python -m src.v2.leaderboard implied --lb 0.938 --experiment v2_m1 [--note "..."]
      implied France = (LB - 0.383 F_US - 0.468 F_India) / 0.150 from the
      experiment's holdout per-entity scores; appended to leaderboard.csv.
  python -m src.v2.leaderboard splice --base DIR --variant DIR --countries France --out DIR
      a submission equal to --base except the rows of S1 entities in --countries,
      which come from --variant (paired France-only comparisons); validated.
  python -m src.v2.leaderboard intersect --base DIR --variant DIR --countries France --out DIR
      like splice, but the rows of --countries keep only matches predicted by BOTH
      submissions (candidates from --base); a precision-first agreement ensemble.
  python -m src.v2.leaderboard vote --base DIR --variants D1,D2,D3 --min-votes 2 --countries France --out DIR
      rows of --countries keep a match if at least --min-votes of --variants predict it
      (the other rows come from --base); candidates = base candidates plus the kept
      matches, so matches stay a subset of candidates.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import pandas as pd

from src.entity_resolution.evaluate import run_validator
from src.v2 import config as C

LEDGER = "leaderboard.csv"
FILES = ("matching_results.tsv", "candidate_pairs.tsv")


def implied(lb: float, experiment: str, note: str) -> None:
    pe = pd.read_parquet(f"runs/perentity/{experiment}.parquet")
    f_us = pe.loc[pe.country == "US", "f_val"].mean()
    f_in = pe.loc[pe.country == "India", "f_val"].mean()
    france = (lb - C.S_WEIGHTS["US"] * f_us - C.S_WEIGHTS["India"] * f_in) / C.S_WEIGHTS["LOCO"]
    row = {"time": time.strftime("%F %T"), "experiment": experiment, "leaderboard": lb, "holdout_F_US": round(f_us, 4),
           "holdout_F_India": round(f_in, 4), "implied_France": round(france, 4), "note": note}
    new = not os.path.exists(LEDGER)
    with open(LEDGER, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"implied France for {experiment}: ({lb} - 0.383*{f_us:.4f} - 0.468*{f_in:.4f}) / 0.150 = {france:.4f}")


def splice(base: str, variant: str, countries: list[str], out: str) -> None:
    s1 = pd.read_csv(f"{C.DATA_ROOT}/test/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False,
                     usecols=["entity_id", "country"])
    swap = set(s1.entity_id[s1.country.isin(countries)])
    os.makedirs(out, exist_ok=True)
    for name in FILES:
        b = pd.read_csv(f"{base}/{name}", sep="\t", dtype=str, keep_default_na=False)
        v = pd.read_csv(f"{variant}/{name}", sep="\t", dtype=str, keep_default_na=False).set_index(b.columns[0])
        col = b.columns[1]
        rows = b[b.columns[0]].isin(swap)
        b.loc[rows, col] = v.loc[b.loc[rows, b.columns[0]], col].to_numpy()
        b.to_csv(f"{out}/{name}", sep="\t", index=False)
        print(f"{name}: {rows.sum():,} rows from {variant} ({', '.join(countries)}), {(~rows).sum():,} from {base}")
    ok, report = run_validator(f"{out}/matching_results.tsv", f"{out}/candidate_pairs.tsv", f"{C.DATA_ROOT}/test")
    print(f"validator passed: {ok}\n{report.strip()}")


def intersect(base: str, variant: str, countries: list[str], out: str) -> None:
    s1 = pd.read_csv(f"{C.DATA_ROOT}/test/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False,
                     usecols=["entity_id", "country"])
    swap = set(s1.entity_id[s1.country.isin(countries)])
    os.makedirs(out, exist_ok=True)
    b = pd.read_csv(f"{base}/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    v = pd.read_csv(f"{variant}/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False).set_index(
        "source1_entity_id").matched_entity_ids
    rows = b.source1_entity_id.isin(swap)
    before = after = 0
    kept = []
    for e, m in zip(b.source1_entity_id[rows], b.matched_entity_ids[rows]):
        mine, other = [x for x in m.split(",") if x], set(filter(None, v.get(e, "").split(",")))
        both = [x for x in mine if x in other]
        before, after = before + len(mine), after + len(both)
        kept.append(",".join(both))
    b.loc[rows, "matched_entity_ids"] = kept
    b.to_csv(f"{out}/matching_results.tsv", sep="\t", index=False)
    pd.read_csv(f"{base}/candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False).to_csv(
        f"{out}/candidate_pairs.tsv", sep="\t", index=False)
    n = max(rows.sum(), 1)
    print(f"{', '.join(countries)}: {rows.sum():,} S1; matches per S1 {before / n:.3f} -> {after / n:.3f}; "
          f"predicted singletons {np.mean([k == '' for k in kept]):.4f}")
    ok, report = run_validator(f"{out}/matching_results.tsv", f"{out}/candidate_pairs.tsv", f"{C.DATA_ROOT}/test")
    print(f"validator passed: {ok}\n{report.strip()}")


def vote(base: str, variants: list[str], min_votes: int, countries: list[str], out: str) -> None:
    s1 = pd.read_csv(f"{C.DATA_ROOT}/test/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False,
                     usecols=["entity_id", "country"])
    swap = set(s1.entity_id[s1.country.isin(countries)])
    os.makedirs(out, exist_ok=True)

    def ids(path, col):
        d = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        return d.set_index("source1_entity_id")[col]
    votes = [ids(f"{v}/matching_results.tsv", "matched_entity_ids") for v in variants]
    b = pd.read_csv(f"{base}/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    cand = pd.read_csv(f"{base}/candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
    rows = b.source1_entity_id.isin(swap)
    kept, n_pairs = [], 0
    for e in b.source1_entity_id[rows]:
        count: dict[str, int] = {}
        for v in votes:
            for x in filter(None, v.get(e, "").split(",")):
                count[x] = count.get(x, 0) + 1
        keep = sorted(x for x, n in count.items() if n >= min_votes)
        n_pairs += len(keep)
        kept.append(keep)
    b.loc[rows, "matched_entity_ids"] = [",".join(k) for k in kept]
    crow = cand.source1_entity_id.isin(swap)
    extra = dict(zip(b.source1_entity_id[rows], kept))
    cand.loc[crow, "candidate_entity_ids"] = [
        ",".join(dict.fromkeys([x for x in c.split(",") if x] + extra.get(e, [])))
        for e, c in zip(cand.source1_entity_id[crow], cand.candidate_entity_ids[crow])]
    b.to_csv(f"{out}/matching_results.tsv", sep="\t", index=False)
    cand.to_csv(f"{out}/candidate_pairs.tsv", sep="\t", index=False)
    n = max(rows.sum(), 1)
    print(f"{', '.join(countries)}: {rows.sum():,} S1; {len(variants)} variants, >= {min_votes} votes: "
          f"matches per S1 {n_pairs / n:.3f}; predicted singletons {np.mean([len(k) == 0 for k in kept]):.4f}")
    ok, report = run_validator(f"{out}/matching_results.tsv", f"{out}/candidate_pairs.tsv", f"{C.DATA_ROOT}/test")
    print(f"validator passed: {ok}\n{report.strip()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("implied")
    a1.add_argument("--lb", type=float, required=True)
    a1.add_argument("--experiment", required=True)
    a1.add_argument("--note", default="")
    a2 = sub.add_parser("splice")
    a2.add_argument("--base", required=True)
    a2.add_argument("--variant", required=True)
    a2.add_argument("--countries", default="France")
    a2.add_argument("--out", required=True)
    a3 = sub.add_parser("intersect")
    for arg in ("--base", "--variant", "--out"):
        a3.add_argument(arg, required=True)
    a3.add_argument("--countries", default="France")
    a4 = sub.add_parser("vote")
    for arg in ("--base", "--variants", "--out"):
        a4.add_argument(arg, required=True)
    a4.add_argument("--min-votes", type=int, default=2)
    a4.add_argument("--countries", default="France")
    a = ap.parse_args()
    if a.cmd == "implied":
        implied(a.lb, a.experiment, a.note)
    elif a.cmd == "splice":
        splice(a.base, a.variant, a.countries.split(","), a.out)
    elif a.cmd == "intersect":
        intersect(a.base, a.variant, a.countries.split(","), a.out)
    else:
        vote(a.base, a.variants.split(","), a.min_votes, a.countries.split(","), a.out)


if __name__ == "__main__":
    main()
