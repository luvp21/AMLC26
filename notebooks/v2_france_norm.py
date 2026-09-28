"""Print the fixed 50 France test S1 (same sample as predict's france_review.txt)
and their first candidates, normalized with a previous normalize.py (from git)
and the current one. Eyeball check only; nothing feeds training.

  PYTHONPATH=. python3 notebooks/v2_france_norm.py --before-rev de035dc --candidates runs/submissions/v2_m1/candidate_pairs.tsv
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from src.v2 import config as C
from src.v2.normalize import normalize_record as after

COLS = ["entity_id", "business_name", "business_address", "country"]
SHOW = ["name_core", "name_legal_class", "addr", "house_num", "postcode"]


def load_old(rev: str):
    src = subprocess.run(["git", "show", f"{rev}:src/v2/normalize.py"], capture_output=True, text=True, check=True).stdout
    path = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    path.write(src)
    path.close()
    spec = importlib.util.spec_from_file_location("normalize_old", path.name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["normalize_old"] = mod
    spec.loader.exec_module(mod)
    return mod.normalize_record


def fmt(r) -> str:
    return " | ".join(f"{k}={r[k]!r}" for k in SHOW if r.get(k))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before-rev", default="de035dc")
    ap.add_argument("--candidates", default="runs/submissions/v2_m1/candidate_pairs.tsv")
    ap.add_argument("--n-cands", type=int, default=5)
    ap.add_argument("--out", default="runs/v2_france_norm.txt")
    a = ap.parse_args()
    before = load_old(a.before_rev)
    root = f"{C.DATA_ROOT}/test"
    s1 = pd.read_csv(f"{root}/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False, usecols=COLS)
    fr = np.flatnonzero((s1.country == "France").to_numpy())  # rid of S1 row i is i
    show = s1.iloc[np.random.default_rng(C.SEED).permutation(fr)[:50]]
    cands = pd.read_csv(a.candidates, sep="\t", dtype=str, keep_default_na=False).set_index("source1_entity_id")
    lists = {e: [c for c in cands.loc[e, "candidate_entity_ids"].split(",") if c][:a.n_cands] for e in show.entity_id}
    need = {c for cs in lists.values() for c in cs}
    pool = pd.concat([pd.concat([ch[ch.entity_id.isin(need)] for ch in pd.read_csv(
        f"{root}/test_source{k}.tsv", sep="\t", dtype=str, keep_default_na=False, usecols=COLS, chunksize=1_000_000)])
        for k in (2, 3)]).set_index("entity_id")
    changed = 0
    with open(a.out, "w") as f:
        for _, r in show.iterrows():
            rows = [(r.entity_id, r)] + [(c, pool.loc[c]) for c in lists[r.entity_id]]
            for i, (eid, x) in enumerate(rows):
                b, n = before(x.business_name, x.business_address, x.country), after(x.business_name, x.business_address, x.country)
                diff = fmt(b) != fmt(n)
                changed += diff
                ind = "" if i == 0 else "    "
                f.write(f"{'\\n=== ' if i == 0 else ind}{eid}: {x.business_name!r} | {x.business_address!r}\n")
                f.write(f"{ind}   before: {fmt(b)}\n{ind}   after : {fmt(n)}{'   <-- changed' if diff else ''}\n")
    print(f"{changed} of {sum(1 + len(v) for v in lists.values())} records changed -> {a.out}")


if __name__ == "__main__":
    main()
