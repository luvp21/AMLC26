"""Phase 2 driver: exclusivity reference on the baseline, then each
normalization step added one at a time on top of the kept set (subsample),
then the kept configuration end to end at subsample and full scale.

Keep rule (per the plan): keep a step only if in-distribution macro F0.5
improves and the LOCO average does not drop. Steps that by construction
cannot move validation (France-only rules, stored-only fields) are kept on
approval when validation is unchanged (within 1e-4).

  nohup env PYTHONPATH=. python3 -u notebooks/phase2_sweep.py > runs/phase2.log 2>&1 &

Historical: this is the Phase 2 run as executed (plain-threshold decision,
LOCO-based keep rule). From Phase 3 on use notebooks/experiment_rule.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import pandas as pd

from src.entity_resolution.harness_io import experiments_csv_path

# (experiment name, --norm flag or feature-text switch, one-line description, change cannot move val)
STEPS = [
    ("p2_feature_text", "feature_text", "fuzzy ratios on normalized tokens instead of raw strings", False),
    ("p2_keep_digits", "keep_digits", "digit tokens any length + single letters next to digits", False),
    ("p2_legal_bag", "legal_bag", "legal-form tokens among last 3 name tokens, any order", False),
    ("p2_stopwords", "stopwords", "generic/landmark stopwords; French articles for France", False),
    ("p2_landmarks", "landmarks", "near/opp/behind X segments -> addr_landmark", False),
    ("p2_country_abbrev", "country_abbrev", "country tables; French table + street-position r/q, bis/ter, cedex", True),
    ("p2_extract_fields", "extract_fields", "postal / house_nums / name_core stored (no current feature uses them)", True),
]


def harness(args, name, flags, text, diff, *extra) -> dict:
    cmd = [sys.executable, "-u", "notebooks/harness.py", "--scale", "subsample", "--experiment", name,
           "--norm", ",".join(flags), "--feature-text", text, "--config-diff", diff, "--out", args.out,
           "--workers", str(args.workers), *extra]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    log = pd.read_csv(experiments_csv_path())
    return log[log.experiment.str.startswith(name + "_")].iloc[-1].to_dict()


def set_kept(name: str, verdict: str) -> None:
    path = experiments_csv_path()
    log = pd.read_csv(path, dtype=str, keep_default_na=False)
    log.loc[log.experiment == name, "kept"] = verdict
    log.to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/phase2")
    ap.add_argument("--workers", default="32")
    ap.add_argument("--skip-exclusivity", action="store_true")
    args = ap.parse_args()
    quick = ["--skip-test-checks", "--skip-missed", "--decision", "threshold"]

    if not args.skip_exclusivity:
        subprocess.run([sys.executable, "-u", "notebooks/harness.py", "--scale", "full", "--experiment", "p2_excl_ref",
                        "--skip-missed", "--out", args.out, "--workers", args.workers,
                        "--config-diff", "baseline pipeline; exclusivity reference run"], check=True)

    flags: list[str] = []
    text = "raw"
    ref = harness(args, "p2_ref", flags, text, "baseline flags (reproduces phase1_baseline_subsample)", *quick)
    decisions = [f"reference: val {ref['val_macro_f05']:.4f}, LOCO avg {ref['loco_avg']:.4f}"]
    for name, step, desc, neutral in STEPS:
        trial_flags = flags if step == "feature_text" else flags + [step]
        trial_text = "normalized" if step == "feature_text" else text
        row = harness(args, name, trial_flags, trial_text, desc, *quick)
        d_val = row["val_macro_f05"] - ref["val_macro_f05"]
        d_loco = row["loco_avg"] - ref["loco_avg"]
        keep = (d_val > 0 and d_loco >= 0) or (neutral and abs(d_val) < 1e-4 and abs(d_loco) < 1e-4)
        verdict = "yes" if keep else "no"
        set_kept(row["experiment"], verdict)
        decisions.append(f"{name:20s} val {row['val_macro_f05']:.4f} ({d_val:+.4f})  LOCO avg {row['loco_avg']:.4f} "
                         f"({d_loco:+.4f})  -> {'KEEP' if keep else 'revert'}  [{desc}]")
        print(decisions[-1], flush=True)
        if keep:
            flags, text, ref = trial_flags, trial_text, row

    kept = ",".join(flags)
    print(f"\nkept normalization flags: {kept or '(none)'}; feature text: {text}", flush=True)
    with open(f"{args.out}/sweep_decisions.txt", "w") as f:
        f.write("\n".join(decisions) + f"\nkept flags: {kept}\nfeature text: {text}\n")

    for scale in ("subsample", "full"):
        subprocess.run([sys.executable, "-u", "notebooks/harness.py", "--scale", scale, "--experiment", "p2_final",
                        "--norm", kept, "--feature-text", text, "--out", args.out,
                        "--workers", args.workers, "--config-diff", f"Phase 2 kept: {kept or 'none'}; text={text}"],
                       check=True)


if __name__ == "__main__":
    main()
