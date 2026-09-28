#!/usr/bin/env bash
# Tonight's M3 plan on PARAM Shavak (repo root, venv active):
#   nohup bash scripts/run_v2_m3b.sh > runs/v2_m3b.log 2>&1 &
# 1. waits until run_v2_m3.sh has finished Part A and the m3_base training, then stops
#    that script (only our own processes) before its remaining trainings;
# 2. error mining on m3_base;
# 3. m3_sib (+ sibling second pass), m3_sib_dec (+ calibration and expected-F0.5),
#    m3_sib_all (+ remaining M3 feature groups), each with LOCO, compared to the previous;
# 4. picks the last kept step, runs the test side once and writes
#    runs/submissions/m3_combined/ (France normalization is already in the test records).
set -euo pipefail
export PYTHONPATH=. V2_WORKERS="${V2_WORKERS:-32}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
M=runs/v2/model
run() { local n=$1; shift; echo "=== $(date '+%F %T') start $n"; local t=$SECONDS; "$@"; echo "=== $(date '+%F %T') done  $n in $((SECONDS - t))s"; }
train() { run "train $1" python3 -m src.v2.train --tag train --experiment "$1" --max-rounds 2000 --config-diff "$2" "${@:3}"; }

echo "waiting for m3_base in runs/v2_m3.log"
until grep -q "done  train m3_base" runs/v2_m3.log; do sleep 60; done
pkill -u "$USER" -f "scripts/run_v2_m3.sh" || true
pkill -u "$USER" -f "experiment m3_france" || true
sleep 5

run errors_m3_base python3 -m src.v2.errors --matcher $M/matcher_m3_base.pkl --n 300
BASE="--exclude-groups idf,france,context2,cross"
train m3_sib     "m3_base + sibling-evidence second pass" $BASE --rules threshold --second-pass --compare-to m3_base
train m3_sib_dec "+ isotonic calibration, expected-F0.5 vs threshold" $BASE --rules threshold,expected_f --second-pass --compare-to m3_sib
train m3_sib_all "+ idf, france, context2, cross feature groups" --rules threshold,expected_f --second-pass --compare-to m3_sib_dec

CHOSEN=$(python3 -m src.v2.choose m3_base m3_sib m3_sib_dec m3_sib_all | tee /dev/stderr | tail -1)
echo "chosen configuration: $CHOSEN"
run prune_test    python3 -m src.v2.prune --tag test
run features_test python3 -m src.v2.features --tag test
run predict       python3 -m src.v2.predict --matcher $M/matcher_$CHOSEN.pkl --experiment m3_combined --out-dir output_m3
