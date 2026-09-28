#!/usr/bin/env bash
# France self-training. Waits for the LOCO run, then:
#   1. pseudo test: France rows re-predicted by a model refit with France pseudo-labels
#      (US/India rows kept from m3_fast) -> runs/submissions/m3_fr_pseudo/
#   2. pseudo loco: the same procedure inside LOCO, to check it helps an unseen country
#   nohup bash scripts/run_v2_pseudo.sh > runs/v2_pseudo.log 2>&1 &
set -euo pipefail
export PYTHONPATH=. V2_WORKERS=32 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run() { local n=$1; shift; echo "=== $(date '+%F %T') start $n"; local t=$SECONDS; "$@"; echo "=== $(date '+%F %T') done  $n in $((SECONDS - t))s"; }
until grep -q "LOCO DONE" runs/v2_loco.log; do sleep 30; done
run pseudo_test python3 -m src.v2.pseudo test --house --base runs/submissions/m3_fast \
    --matcher runs/v2/model/matcher_m3_loco.pkl --out runs/submissions/m3_fr_pseudo
run pseudo_loco python3 -m src.v2.pseudo loco --house
echo "=== PSEUDO DONE"
