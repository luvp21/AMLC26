"""Share of S1 whose accepted records ALL have a house number different from the S1's (or missing).

Test: from a submission's matching_results.tsv; holdout: the blend's predictions (ce blend --dump-pred)
and, for reference, the true matches. Label-free on test.
  PYTHONPATH=. python3 notebooks/v2_house_share.py runs/submissions/m3_gap_ce2_fr_u15
"""
import sys

import numpy as np
import pandas as pd

from src.v2 import config as C
from src.v2.house import REL, relation

SAME = {REL["equal"], REL["letters_only"]}


def share(rec: pd.DataFrame, q: np.ndarray, s: np.ndarray, s1_ids: np.ndarray) -> tuple[float, float, int]:
    """(share among S1 with >= 1 accepted record, share among all given S1, n with >= 1)."""
    h = rec.house_num.to_numpy()
    rel = np.array([relation(h[a], h[b]) for a, b in zip(q, s)])
    df = pd.DataFrame({"s": s, "diff": ~np.isin(rel, list(SAME))})
    all_diff = df.groupby("s")["diff"].all()
    all_diff = all_diff[all_diff.index.isin(s1_ids)]
    return all_diff.mean(), all_diff.sum() / len(s1_ids), len(all_diff)


def main() -> None:
    sub = sys.argv[1]
    tr = pd.read_parquet(f"{C.tag_dir('train')}/records.parquet", columns=["source", "country", "fold", "owner", "house_num"])
    hp = pd.read_parquet(f"{C.tag_dir('train')}/ce_hold_pred.parquet")
    te = pd.read_parquet(f"{C.tag_dir('test')}/records.parquet", columns=["entity_id", "source", "country", "house_num"])
    rid = pd.Series(np.arange(len(te)), index=te.entity_id.to_numpy())
    m = pd.read_csv(f"{sub}/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    m = m[m.matched_entity_ids != ""].assign(c=lambda d: d.matched_entity_ids.str.split(",")).explode("c")
    tq, ts = rid[m.c.to_numpy()].to_numpy(), rid[m.source1_entity_id.to_numpy()].to_numpy()
    own = tr.owner.to_numpy()
    true_q = np.flatnonzero(own >= 0)
    print("share of S1 whose accepted records ALL have a different or missing house number")
    print("country  | test (submission)          | holdout (blend predictions)   | holdout (TRUE matches)")
    for c in ("India", "US", "France"):
        te_s1 = np.flatnonzero((te.source == 1).to_numpy() & (te.country == c).to_numpy())
        a = share(te, tq, ts, te_s1)
        if c == "France":
            print(f"{c:8s} | {a[0]:.4f} of matched ({a[1]:.4f} of all) | (no training data)")
            continue
        h_s1 = np.flatnonzero((tr.source == 1).to_numpy() & (tr.country == c).to_numpy() & (tr.fold == "holdout").to_numpy())
        b = share(tr, hp.q.to_numpy(), hp.s.to_numpy(), h_s1)
        keep = np.isin(own[true_q], h_s1)
        t = share(tr, true_q[keep], own[true_q][keep], h_s1)
        print(f"{c:8s} | {a[0]:.4f} of matched ({a[1]:.4f} of all) | {b[0]:.4f} of matched ({b[1]:.4f} of all) | "
              f"{t[0]:.4f} of matched ({t[1]:.4f} of all)")


if __name__ == "__main__":
    main()
