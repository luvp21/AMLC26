#!/usr/bin/env bash
# v2 end to end on PARAM Shavak (repo root, inside ~/venvs/ml):
#   V2_EXPERIMENT=v2_m2 nohup bash scripts/run_v2.sh > runs/v2_m2.log 2>&1 &
# Resume or run a subset by naming stages:   bash scripts/run_v2.sh train_model predict
# Stages (default order): prepare_raw learn prepare block_train prune_train features_train
#   train_model block_test prune_test features_test predict
#   prepare_raw + learn: first normalization pass and the learned tables (rewrites, dictionary);
#   prepare: final normalization of train and test with those tables.
# Environment:
#   V2_WORKERS    threads/processes for heavy stages (default 32; 24 when the box is busy)
#   V2_EXPERIMENT experiment name (default v2_m1)
#   V2_LOCO       1 to run LOCO in train_model (default 0: first submission faster, S pending)
#   V2_CHANNELS   blocking indexes (default n,b,a,k)
# Each stage also writes its own log under runs/v2/<tag>/ with disk/RAM at start and end.
set -euo pipefail
export PYTHONPATH=.
export V2_WORKERS="${V2_WORKERS:-32}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # threads only where passed explicitly
EXP="${V2_EXPERIMENT:-v2_m2}"
LOCO_FLAG=$([ "${V2_LOCO:-0}" = "1" ] && echo "" || echo "--no-loco")
STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(prepare_raw learn prepare block_train prune_train features_train train_model \
                                    block_test prune_test features_test predict)

run() {
  local name=$1; shift
  echo "=== $(date '+%F %T') start $name"
  local t0=$SECONDS
  "$@"
  echo "=== $(date '+%F %T') done  $name in $((SECONDS - t0))s"
}

echo "experiment $EXP, workers $V2_WORKERS, LOCO ${V2_LOCO:-0}, channels ${V2_CHANNELS:-n,b,a,k}, stages: ${STAGES[*]}"
for stage in "${STAGES[@]}"; do
  case $stage in
    prepare_raw)    run prepare_raw    python3 -m src.v2.prepare --split train --raw ;;
    learn)          run learn          python3 -m src.v2.learn ;;
    prepare)        run prepare        bash -c "python3 -m src.v2.prepare --split train && python3 -m src.v2.prepare --split test" ;;
    block_train)    run block_train    python3 -m src.v2.blocking --tag train ;;
    prune_train)    run prune_train    python3 -m src.v2.prune --tag train ;;
    features_train) run features_train python3 -m src.v2.features --tag train ;;
    train_model)    run train_model    python3 -m src.v2.train --tag train --experiment "$EXP" --max-rounds 2000 $LOCO_FLAG \
                                         --config-diff "v2 $EXP: channels ${V2_CHANNELS:-n,b,a,k}, learned rewrites + dictionary, cross-fit pruner and matcher, one-owner + threshold" ;;
    block_test)     run block_test     python3 -m src.v2.blocking --tag test ;;
    prune_test)     run prune_test     python3 -m src.v2.prune --tag test ;;
    features_test)  run features_test  python3 -m src.v2.features --tag test ;;
    predict)        run predict        python3 -m src.v2.predict --experiment "$EXP" --out-dir output ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
