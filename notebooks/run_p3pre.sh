#!/usr/bin/env bash
# Pre-Phase-3 runs on PARAM Shavak (from repo root, inside ~/venvs/ml):
#   1. full-scale run of the approved config with the new decision step, writing
#      and validating the leaderboard submission (expected validation S logged);
#   2. feature-text variants raw / casefold / normalized on the subsample, under
#      the decision rule (bootstrap; 3 seeds when |dS| < 0.002).
#   nohup bash notebooks/run_p3pre.sh > runs/p3pre.log 2>&1 &
set -euo pipefail
export PYTHONPATH=.
COMMON="--norm keep_digits,legal_bag,country_abbrev,extract_fields --out runs/p3pre"

python3 -u notebooks/harness.py --scale full --experiment p3pre_submit $COMMON \
    --config-diff="digits + legal bag + French tables + all-S1 exclusivity; baseline model and candidates" \
    --write-submission runs/submissions/p3pre

python3 -u notebooks/experiment_rule.py --ref "p3pre_raw:" \
    --new "p3pre_casefold:--feature-text casefold" \
    --new "p3pre_normalized:--feature-text normalized" \
    --common "$COMMON --skip-missed --skip-test-checks"
