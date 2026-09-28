#!/usr/bin/env bash
# Label-free routing for countries without training data (France) on test.
#   nohup bash scripts/run_v2_frroute.sh > runs/v2_frroute.log 2>&1 &
# Waits for the synthetic job (runs/v2_syn.log: SYN DONE), moves the unrouted test
# tables to runs/v2/test_unrouted/ (kept, not deleted), re-blocks test with
# --unsup-routing (US/India keep their learned routes, so their candidates are
# unchanged), rebuilds test pruning, features and add-ons, then predicts on the
# routed candidates: m3_gap_la25 and, if the synthetic matcher exists, m3_syn / m3_syn_la25.
set -euo pipefail
export PYTHONPATH=. V2_WORKERS="${V2_WORKERS:-32}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
M=runs/v2/model
T=runs/v2/test
run() { local n=$1; shift; echo "=== $(date '+%F %T') start $n"; local t=$SECONDS; "$@"; echo "=== $(date '+%F %T') done  $n in $((SECONDS - t))s"; }

if [ -f runs/v2_syn.log ]; then
  until grep -qE "SYN DONE|Traceback|aborting" runs/v2_syn.log; do sleep 60; done
fi
mkdir -p runs/v2/test_unrouted
for f in pairs_block.parquet pairs_pruned.parquet features.parquet features_house.parquet features_novel.parquet \
         features_crowd.parquet blocking.log; do
  if [ -f "$T/$f" ]; then mv "$T/$f" runs/v2/test_unrouted/; fi
done
df -h ~ | tail -1

run block_test    python3 -m src.v2.blocking --tag test --unsup-routing
grep -E "unsupervised routing|avg index size|France:" "$T/blocking.log" || true
run prune_test    python3 -m src.v2.prune --tag test
run features_test python3 -m src.v2.features --tag test
run house_test    python3 -m src.v2.house --tag test
run novel_test    python3 -m src.v2.novel --tag test
run crowd_test    python3 -m src.v2.crowd --tag test

run "predict m3_gap_la25_fr" python3 -m src.v2.predict --matcher $M/matcher_m3_gap.pkl --experiment m3_gap_la25_fr \
    --out-dir output_fr --lookalike-shift 2.5
if [ -f $M/matcher_m3_syn1.pkl ]; then
  run "predict m3_syn_fr"      python3 -m src.v2.predict --matcher $M/matcher_m3_syn1.pkl --experiment m3_syn_fr --out-dir output_fr
  run "predict m3_syn_la25_fr" python3 -m src.v2.predict --matcher $M/matcher_m3_syn1.pkl --experiment m3_syn_la25_fr \
      --out-dir output_fr --lookalike-shift 2.5
fi
rm -rf output_fr
echo "FR DONE"
