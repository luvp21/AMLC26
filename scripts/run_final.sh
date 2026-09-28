#!/usr/bin/env bash
# Final pipeline: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv (validated).
#   nohup bash scripts/run_final.sh > runs/final.log 2>&1 &
# Resume or run a subset by naming stages:   bash scripts/run_final.sh train_model predict
#
# Stages (default order):
#   prepare_raw learn prepare   normalization; rewrites and native-script dictionary learned from training matches
#   block_train prune_train features_train addons_train
#                               reverse blocking with learned routing, cross-fitted pruner, pair features,
#                               house-number / novel-word / name-crowd features
#   synth                       synthetic next-door look-alikes (OUT_DIR/aug; only with FINAL_SYNTH=1)
#   train_model                 cross-fitted two-pass LightGBM matcher, calibration, decoding on the tune fold
#   block_test prune_test features_test addons_test
#                               test side; countries without training data use label-free routing
#   predict                     GBM-only submission (one-owner decoding, validation) -> runs/submissions/final_gbm/
#   ce_band_train ce_train ce_score_train ce2_band_train ce2_train ce2_score_train
#   ce3_band_train ce3_train ce3_score_train ce4_band_train ce4_train ce4_score_train ce_blend
#                               cross-encoder rescoring of uncertain pairs (calibrated p in [0.02, 0.98]):
#                               three paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0) models, each fine-tuned
#                               1 epoch (GPU, fp16) on training-fold pairs: ce on a 500k positives-first band sample,
#                               ce2 on a 500k natural-ratio band sample, ce3 on a 1.3M natural-ratio sample (whole
#                               band), ce4 on a 1.0M natural-ratio sample with seed 7 and the pair order swapped; logistic blend of
#                               [GBM logit, ce, ce2, ce3, ce4] fitted on the tune fold
#   ce_band_test ce_score_test ce2_score_test ce3_score_test ce4_score_test ce_apply
#                               test side; blend + decoding with the predict-time shifts -> output/ (final)
# Environment:
#   V2_WORKERS       processes/threads for heavy stages (default 32)
#   FINAL_SYNTH      1 = train on the synthetic look-alike pool (default 0: it scored 0.9734 vs 0.97528 on the leaderboard)
#   FINAL_SHIFT      logit shift for known-different house numbers at predict (default 2.5)
#   FINAL_UNSEEN_SHIFT  logit shift for countries without training data (default 1.5; France-only leaderboard pair: 0.97532 vs 0.975281)
#   FINAL_ENSEMBLE   path of a second matcher .pkl to average with (default none)
# Hardware used: 56-thread Xeon, 250 GB RAM, Quadro GP100 16 GB (cross-encoder, fp16); ~8-10 h end to end.
set -euo pipefail
export PYTHONPATH=.
export V2_WORKERS="${V2_WORKERS:-32}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
M=runs/v2/model
EXP=final
SYNTH="${FINAL_SYNTH:-0}"
SHIFT="${FINAL_SHIFT:-2.5}"
UNSEEN_SHIFT="${FINAL_UNSEEN_SHIFT:-1.5}"
ENSEMBLE="${FINAL_ENSEMBLE:-}"
TRAIN_TAG=$([ "$SYNTH" = "1" ] && echo aug || echo train)
# final submission = m3_gap_la25_fr_u15: second pass, house-number (incl. gap) and novel-word features, no crowd features
FLAGS="--max-rounds 2000 --lr 0.08 --no-loco --second-pass --house --novel --rules threshold,expected_f"
[ "$SYNTH" = "1" ] && FLAGS="$FLAGS --crowd"
STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(prepare_raw learn prepare block_train prune_train features_train addons_train \
                                    synth train_model ce_band_train ce_train ce_score_train \
                                    ce2_band_train ce2_train ce2_score_train \
                                    ce3_band_train ce3_train ce3_score_train \
                                    ce4_band_train ce4_train ce4_score_train ce_blend \
                                    block_test prune_test features_test addons_test predict \
                                    ce_band_test ce_score_test ce2_score_test ce3_score_test ce4_score_test ce_apply)

run() {
  local name=$1; shift
  echo "=== $(date '+%F %T') start $name"
  local t0=$SECONDS
  "$@"
  echo "=== $(date '+%F %T') done  $name in $((SECONDS - t0))s"
}

addons() {
  python3 -m src.v2.house --tag "$1" && python3 -m src.v2.novel --tag "$1" && python3 -m src.v2.crowd --tag "$1"
}

echo "final: workers $V2_WORKERS, synthetic training $SYNTH (tag $TRAIN_TAG), shift $SHIFT, unseen shift $UNSEEN_SHIFT, ensemble '${ENSEMBLE}', stages: ${STAGES[*]}"
for stage in "${STAGES[@]}"; do
  case $stage in
    prepare_raw)    run prepare_raw    python3 -m src.v2.prepare --split train --raw ;;
    learn)          run learn          python3 -m src.v2.learn ;;
    prepare)        run prepare        bash -c "python3 -m src.v2.prepare --split train && python3 -m src.v2.prepare --split test" ;;
    block_train)    run block_train    python3 -m src.v2.blocking --tag train ;;
    prune_train)    run prune_train    python3 -m src.v2.prune --tag train ;;
    features_train) run features_train python3 -m src.v2.features --tag train ;;
    addons_train)   run addons_train   addons train ;;
    synth)          [ "$SYNTH" = "1" ] && run synth python3 -m src.v2.synth || echo "synth skipped (FINAL_SYNTH=$SYNTH)" ;;
    train_model)    run train_model    python3 -m src.v2.train --tag "$TRAIN_TAG" --experiment "$EXP" $FLAGS \
                                         --config-diff "final: synthetic training $SYNTH" ;;
    block_test)     run block_test     python3 -m src.v2.blocking --tag test --unsup-routing ;;
    prune_test)     run prune_test     python3 -m src.v2.prune --tag test ;;
    features_test)  run features_test  python3 -m src.v2.features --tag test ;;
    addons_test)    run addons_test    addons test ;;
    predict)        run predict        python3 -m src.v2.predict --matcher "$M/matcher_$EXP.pkl" --experiment "${EXP}_gbm" \
                                         --out-dir output_gbm --lookalike-shift "$SHIFT" --unseen-shift "$UNSEEN_SHIFT" \
                                         ${ENSEMBLE:+--ensemble "$ENSEMBLE"} ;;
    ce_band_train)  run ce_band_train  python3 -m src.v2.ce band --tag train --matcher "$M/matcher_$EXP.pkl" ;;
    ce_train)       run ce_train       python3 -m src.v2.ce train --tag train ;;
    ce_score_train) run ce_score_train python3 -m src.v2.ce score --tag train ;;
    ce2_band_train) run ce2_band_train python3 -m src.v2.ce band --tag train --matcher "$M/matcher_$EXP.pkl" --reuse \
                                         --sample natural --ce-name ce2 ;;
    ce2_train)      run ce2_train      python3 -m src.v2.ce train --tag train --ce-name ce2 ;;
    ce2_score_train) run ce2_score_train python3 -m src.v2.ce score --tag train --ce-name ce2 ;;
    ce3_band_train) run ce3_band_train python3 -m src.v2.ce band --tag train --matcher "$M/matcher_$EXP.pkl" --reuse \
                                         --sample natural --cap 1300000 --ce-name ce3 ;;
    ce3_train)      run ce3_train      python3 -m src.v2.ce train --tag train --ce-name ce3 ;;
    ce3_score_train) run ce3_score_train python3 -m src.v2.ce score --tag train --ce-name ce3 ;;
    ce4_band_train) run ce4_band_train python3 -m src.v2.ce band --tag train --matcher "$M/matcher_$EXP.pkl" --reuse \
                                         --sample natural --cap 1000000 --seed 7 --ce-name ce4 ;;
    ce4_train)      run ce4_train      python3 -m src.v2.ce train --tag train --ce-name ce4 --seed 7 --swap ;;
    ce4_score_train) run ce4_score_train python3 -m src.v2.ce score --tag train --ce-name ce4 --swap ;;
    ce_blend)       run ce_blend       python3 -m src.v2.ce blend --tag train --ce-names ce,ce2,ce3,ce4 ;;
    ce_band_test)   run ce_band_test   python3 -m src.v2.ce band --tag test --matcher "$M/matcher_$EXP.pkl" ;;
    ce_score_test)  run ce_score_test  python3 -m src.v2.ce score --tag test ;;
    ce2_score_test) run ce2_score_test python3 -m src.v2.ce score --tag test --ce-name ce2 ;;
    ce3_score_test) run ce3_score_test python3 -m src.v2.ce score --tag test --ce-name ce3 ;;
    ce4_score_test) run ce4_score_test python3 -m src.v2.ce score --tag test --ce-name ce4 --swap ;;
    ce_apply)       run ce_apply       python3 -m src.v2.ce apply --matcher "$M/matcher_$EXP.pkl" --experiment "$EXP" --ce-names ce,ce2,ce3,ce4 \
                                         --out-dir output --lookalike-shift "$SHIFT" --unseen-shift "$UNSEEN_SHIFT" ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
