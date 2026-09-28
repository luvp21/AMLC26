"""Build a small real-data replica of the dataset layout for local smoke
tests (this laptop can't hold the full pools). Samples Source 1 entities,
keeps all their true matches, adds background S2/S3 rows, and writes
stand-in candidate pickles via fast_block so harness scripts run unchanged
with --data-root/--*-candidates overrides. Never used for reported numbers.

  PYTHONPATH=. python3 notebooks/make_mini_dataset.py --out runs/mini
"""
from __future__ import annotations

import argparse
import os
import pickle

import pandas as pd

from src.entity_resolution.fast_block import generate_candidates_fast
from src.entity_resolution.pipeline import add_normalized_columns
from src.metrics import parse_id_list

ROOT = "data_set/student_resource/dataset"


def read(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def write(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/mini")
    ap.add_argument("--n-train-s1", type=int, default=8000)
    ap.add_argument("--n-test-s1", type=int, default=4000)
    ap.add_argument("--n-background", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    gt = read(f"{ROOT}/train/train_ground_truth.tsv")
    s1 = read(f"{ROOT}/train/train_source1.tsv").sample(n=a.n_train_s1, random_state=a.seed)
    gt = gt[gt.source1_entity_id.isin(set(s1.entity_id))]
    needed = set().union(*(parse_id_list(m) for m in gt.matched_entity_ids))
    write(s1, f"{a.out}/train/train_source1.tsv")
    write(gt, f"{a.out}/train/train_ground_truth.tsv")
    train_srcs = []
    for k in (2, 3):
        df = read(f"{ROOT}/train/train_source{k}.tsv")
        keep = pd.concat([df[df.entity_id.isin(needed)],
                          df[~df.entity_id.isin(needed)].sample(n=a.n_background, random_state=a.seed + k)])
        write(keep, f"{a.out}/train/train_source{k}.tsv")
        train_srcs.append(keep.reset_index(drop=True))

    t1 = read(f"{ROOT}/test/test_source1.tsv").sample(n=a.n_test_s1, random_state=a.seed)
    write(t1, f"{a.out}/test/test_source1.tsv")
    test_srcs = []
    for k in (2, 3):
        df = read(f"{ROOT}/test/test_source{k}.tsv").sample(n=a.n_background, random_state=a.seed + k)
        write(df, f"{a.out}/test/test_source{k}.tsv")
        test_srcs.append(df.reset_index(drop=True))

    for name, s1_df, srcs in (("candidates", s1, train_srcs), ("test_candidates", t1, test_srcs)):
        s1_df = add_normalized_columns(s1_df.reset_index(drop=True))
        srcs = [add_normalized_columns(d) for d in srcs]
        with open(f"{a.out}/{name}.pkl", "wb") as f:
            pickle.dump(generate_candidates_fast(s1_df, srcs, top_k=30), f)
    print(f"mini dataset written to {a.out}")


if __name__ == "__main__":
    main()
