#!/usr/bin/env bash
# Deadline run (before midnight): every improvement in one model, no LOCO.
#   nohup bash scripts/run_v2_fast.sh > runs/v2_fast.log 2>&1 &
# Stops the slower m3b/m3c chains (our processes only), then runs two tracks in parallel
# with 24 workers each: (T) house features + training, (P) test pruning + features;
# then predicts into runs/submissions/m3_fast/.
set -uo pipefail
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
M=runs/v2/model
run() { local n=$1; shift; echo "=== $(date '+%F %T') start $n"; local t=$SECONDS; "$@" || { echo "=== FAILED $n"; return 1; }; echo "=== $(date '+%F %T') done  $n in $((SECONDS - t))s"; }

pkill -u "$USER" -f "scripts/run_v2_m3b.sh"; pkill -u "$USER" -f "scripts/run_v2_m3c.sh"; pkill -u "$USER" -f "scripts/run_v2_m3.sh"
pkill -u "$USER" -f "src.v2.train"; pkill -u "$USER" -f "src.v2.errors"; sleep 5
echo "stopped earlier chains; our python processes now:"; ps -u "$USER" -o pid,cmd | grep "src.v2" | grep -v grep || true

(
  export V2_WORKERS=24
  run house_train python3 -m src.v2.house --tag train &&
  run "train m3_fast" python3 -m src.v2.train --tag train --experiment m3_fast --max-rounds 2000 --no-loco \
      --second-pass --house --rules threshold,expected_f --compare-to m3_base \
      --config-diff "deadline: per-country cutoffs + all M3 groups + house features + second pass + calibration/expected-F (no LOCO)"
) > runs/v2_fast_train.log 2>&1 &
TRAIN=$!
(
  export V2_WORKERS=24
  run prune_test    python3 -m src.v2.prune --tag test &&
  run features_test python3 -m src.v2.features --tag test &&
  run house_test    python3 -m src.v2.house --tag test
) > runs/v2_fast_test.log 2>&1 &
TEST=$!
wait $TRAIN; T_OK=$?; wait $TEST; P_OK=$?
cat runs/v2_fast_train.log runs/v2_fast_test.log | grep -E "^=== "
[ $T_OK -eq 0 ] && [ $P_OK -eq 0 ] || { echo "a track failed; see runs/v2_fast_train.log / runs/v2_fast_test.log"; exit 1; }
export V2_WORKERS=32
run predict python3 -m src.v2.predict --matcher $M/matcher_m3_fast.pkl --experiment m3_fast --out-dir output_fast
echo "=== FAST DONE"
