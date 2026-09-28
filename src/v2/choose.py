"""Pick the configuration for the combined test run from a chain of experiments.

  python -m src.v2.choose m3_base m3_sib m3_sib_dec m3_sib_all

Walks the chain in order: a step is kept if its logged Delta S (vs the step
before it) is positive and its bootstrap interval excludes 0; the last kept
step wins (a step that was not kept leaves the choice where it was). Prints
the chosen experiment name on the last line.
"""
from __future__ import annotations

import sys

import pandas as pd

from src.entity_resolution.harness_io import experiments_csv_path


def main() -> None:
    chain = sys.argv[1:]
    log = pd.read_csv(experiments_csv_path(), dtype=str, keep_default_na=False)
    chosen = chain[0]
    for exp in chain[1:]:
        rows = log[log.experiment == exp]
        if rows.empty:
            print(f"{exp}: not run")
            continue
        r = rows.iloc[-1]
        d, ci = r.get("d_S", ""), r.get("d_S_ci", "")
        try:
            lo = float(ci.strip("[]").split(",")[0])
            kept = float(d) > 0 and lo > 0
        except ValueError:
            kept = False
        print(f"{exp}: dS {d} CI {ci} -> {'KEEP' if kept else 'revert'}")
        if kept:
            chosen = exp
    print(chosen)


if __name__ == "__main__":
    main()
