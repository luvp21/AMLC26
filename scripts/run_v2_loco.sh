#!/usr/bin/env bash
# Tonight: m3_fast configuration with LOCO (lr 0.08) -> S and the unseen-country offset,
# then predict (France gets the offset if both LOCO directions prefer a stricter setting).
#   nohup bash scripts/run_v2_loco.sh > runs/v2_loco.log 2>&1 &
set -euo pipefail
export PYTHONPATH=. V2_WORKERS=32 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run() { local n=$1; shift; echo "=== $(date '+%F %T') start $n"; local t=$SECONDS; "$@"; echo "=== $(date '+%F %T') done  $n in $((SECONDS - t))s"; }
run "train m3_loco" python3 -m src.v2.train --tag train --experiment m3_loco --max-rounds 2000 --lr 0.08 \
    --second-pass --house --rules threshold,expected_f --compare-to m3_fast \
    --config-diff "m3_fast configuration with LOCO (lr 0.08): S and unseen-country offset"
run predict python3 -m src.v2.predict --matcher runs/v2/model/matcher_m3_loco.pkl --experiment m3_loco --out-dir output_loco
echo "=== LOCO DONE"
