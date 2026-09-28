"""Apply the experiment decision rule to one or more changes against a reference.

Rule: keep a change if Delta S > 0 and the paired-bootstrap 95% interval on
Delta S excludes 0. If |Delta S| < 0.002 at one seed, rerun the reference and
the change with seeds 42,43,44 (per-entity F averaged) before deciding.

  PYTHONPATH=. python3 -u notebooks/experiment_rule.py --ref "p3pre_raw:" \
      --new "p3pre_casefold:--feature-text casefold" --common "--norm keep_digits,..."
Each spec is "experiment_name:extra harness args".
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

import pandas as pd

from src.entity_resolution.harness_io import experiments_csv_path

PER_ENTITY_DIR = "runs/perentity"
EXPERIMENTS_CSV = experiments_csv_path()
SMALL_DELTA = 0.002
SEEDS3 = "42,43,44"


def run(name: str, extra: str, common: str, scale: str, seeds: str, compare_to: str = "") -> dict:
    tag = f"{name}_{scale}"
    if os.path.exists(f"{PER_ENTITY_DIR}/{tag}.parquet") and not compare_to:
        print(f"reusing {tag}", flush=True)
        return {"experiment": tag}
    cmd = [sys.executable, "-u", "notebooks/harness.py", "--scale", scale, "--experiment", name, "--seeds", seeds,
           f"--config-diff={extra.strip() or 'reference'}", "--experiments-csv", EXPERIMENTS_CSV, *shlex.split(common), *shlex.split(extra)]
    if compare_to:
        cmd += ["--compare-to", compare_to]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    log = pd.read_csv(EXPERIMENTS_CSV, dtype=str, keep_default_na=False)
    return log[log.experiment == tag].iloc[-1].to_dict()


def set_kept(tag: str, verdict: str) -> None:
    path = EXPERIMENTS_CSV
    log = pd.read_csv(path, dtype=str, keep_default_na=False)
    log.loc[log.experiment == tag, "kept"] = verdict
    log.to_csv(path, index=False)


def judge(ref_name, ref_extra, new_name, new_extra, common, scale) -> str:
    run(ref_name, ref_extra, common, scale, "42")
    row = run(new_name, new_extra, common, scale, "42", compare_to=f"{ref_name}_{scale}")
    d_s = float(row["d_S"])
    if abs(d_s) < SMALL_DELTA:
        print(f"|dS| = {abs(d_s):.4f} < {SMALL_DELTA}: rerunning both with seeds {SEEDS3}", flush=True)
        run(f"{ref_name}_3s", ref_extra, common, scale, SEEDS3)
        row = run(f"{new_name}_3s", new_extra, common, scale, SEEDS3, compare_to=f"{ref_name}_3s_{scale}")
        d_s = float(row["d_S"])
    low, high = (float(x) for x in row["d_S_ci"].strip("[]").split(","))
    keep = d_s > 0 and low > 0
    verdict = f"{'yes' if keep else 'no'} (dS {d_s:+.4f}, CI [{low:.4f}, {high:.4f}])"
    set_kept(row["experiment"], verdict)
    print(f"VERDICT {row['experiment']}: {verdict}", flush=True)
    return verdict


def main() -> None:
    global EXPERIMENTS_CSV
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help='"name:extra args"')
    ap.add_argument("--new", action="append", required=True, help='"name:extra args" (repeatable)')
    ap.add_argument("--common", default="", help="harness args shared by every run")
    ap.add_argument("--scale", default="subsample", choices=["subsample", "full"])
    ap.add_argument("--experiments-csv", default=EXPERIMENTS_CSV)
    a = ap.parse_args()
    EXPERIMENTS_CSV = a.experiments_csv
    ref_name, ref_extra = a.ref.split(":", 1)
    verdicts = [judge(ref_name, ref_extra, *spec.split(":", 1), a.common, a.scale) for spec in a.new]
    print("\n" + "\n".join(verdicts))


if __name__ == "__main__":
    main()
