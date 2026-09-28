#!/usr/bin/env bash
# M3 on PARAM Shavak (repo root, venv active), after M2 finished:
#   nohup bash scripts/run_v2_m3.sh > runs/v2_m3.log 2>&1 &
# Part A  France-only paired submission: M2 models, France normalization on test only.
#         Train has no France rows, so the train side is unchanged by construction.
# Part B  per-country pruner cutoffs, M3 feature groups, then training runs measured
#         with S one change at a time (each --compare-to the previous one).
# Run a subset:  bash scripts/run_v2_m3.sh B      (or A)
set -euo pipefail
export PYTHONPATH=. V2_WORKERS="${V2_WORKERS:-32}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PARTS="${1:-AB}"
M=runs/v2/model
run() { local n=$1; shift; echo "=== $(date '+%F %T') start $n"; local t=$SECONDS; "$@"; echo "=== $(date '+%F %T') done  $n in $((SECONDS - t))s"; }
train() { run "train $1" python3 -m src.v2.train --tag train --experiment "$1" --max-rounds 2000 --config-diff "$2" "${@:3}"; }

if [[ $PARTS == *A* ]]; then
  mkdir -p $M/m2 && { [ -f $M/m2/matcher.pkl ] || cp $M/pruner_train.pkl $M/matcher.pkl $M/m2/; }
  run prepare_test   python3 -m src.v2.prepare --split test
  run block_test     python3 -m src.v2.blocking --tag test
  run prune_test     python3 -m src.v2.prune --tag test --pruner $M/m2/pruner_train.pkl
  run features_test  python3 -m src.v2.features --tag test
  run predict        python3 -m src.v2.predict --matcher $M/m2/matcher.pkl --experiment m2_frnorm --out-dir output_frnorm
  run splice         python3 -m src.v2.leaderboard splice --base runs/submissions/v2_m2 \
                       --variant runs/submissions/m2_frnorm --countries France --out runs/submissions/m2_frnorm_paired
fi
if [[ $PARTS == *B* ]]; then
  run prune_train    python3 -m src.v2.prune --tag train
  run features_train python3 -m src.v2.features --tag train
  train m3_base   "M2 + per-country pruner cutoffs (no M3 groups, threshold)" \
        --exclude-groups idf,france,context2,cross --rules threshold --compare-to v2_m2
  train m3_france "+ France-targeted features" \
        --exclude-groups idf,context2,cross --rules threshold --compare-to m3_base
  train m3_decode "+ calibration and expected-F0.5 decoding" \
        --exclude-groups idf,context2,cross --rules threshold,expected_f --compare-to m3_france
  train m3_all    "+ idf, context2, cross feature groups" \
        --rules threshold,expected_f --compare-to m3_decode
fi
