#!/usr/bin/env bash
# After run_v2_m3b.sh: house-number relation features on top of the chosen configuration.
#   nohup bash scripts/run_v2_m3c.sh > runs/v2_m3c.log 2>&1 &
# Waits for m3b's predict, builds features_house for train and test, trains m3_house
# (chosen config + --house, with LOCO) against the chosen experiment, and if it is kept
# writes runs/submissions/m3_combined_house/.
set -euo pipefail
export PYTHONPATH=. V2_WORKERS="${V2_WORKERS:-32}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
M=runs/v2/model
run() { local n=$1; shift; echo "=== $(date '+%F %T') start $n"; local t=$SECONDS; "$@"; echo "=== $(date '+%F %T') done  $n in $((SECONDS - t))s"; }
until grep -q "done  predict" runs/v2_m3b.log; do sleep 60; done
CHOSEN=$(grep "chosen configuration:" runs/v2_m3b.log | tail -1 | awk '{print $3}')
case $CHOSEN in
  m3_base)    FLAGS="--exclude-groups idf,france,context2,cross --rules threshold" ;;
  m3_sib)     FLAGS="--exclude-groups idf,france,context2,cross --rules threshold --second-pass" ;;
  m3_sib_dec) FLAGS="--exclude-groups idf,france,context2,cross --rules threshold,expected_f --second-pass" ;;
  m3_sib_all) FLAGS="--rules threshold,expected_f --second-pass" ;;
  *) echo "unknown chosen configuration '$CHOSEN'"; exit 1 ;;
esac
echo "chosen by m3b: $CHOSEN ($FLAGS)"
run house_train python3 -m src.v2.house --tag train
run house_test  python3 -m src.v2.house --tag test
run "train m3_house" python3 -m src.v2.train --tag train --experiment m3_house --max-rounds 2000 $FLAGS --house \
    --compare-to "$CHOSEN" --config-diff "$CHOSEN + house-number relation features"
FINAL=$(python3 -m src.v2.choose "$CHOSEN" m3_house | tee /dev/stderr | tail -1)
echo "final configuration: $FINAL"
if [ "$FINAL" = "m3_house" ]; then
  run predict python3 -m src.v2.predict --matcher $M/matcher_m3_house.pkl --experiment m3_combined_house --out-dir output_m3h
fi
