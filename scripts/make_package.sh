#!/usr/bin/env bash
# Assemble <team>_submission.zip in the organisers' layout.
#   bash scripts/make_package.sh <team_name> <dir with the final matching_results.tsv + candidate_pairs.tsv>
set -euo pipefail
TEAM=${1:?team name}; OUTSRC=${2:?directory holding the final two TSV files}
DIST=dist/${TEAM}_submission
CODE=$DIST/code/business_entity_resolution
rm -rf "$DIST" "dist/${TEAM}_submission.zip"
mkdir -p "$DIST/output" "$CODE/src/v2" "$CODE/src/entity_resolution" "$CODE/scripts"
cp "$OUTSRC/matching_results.tsv" "$OUTSRC/candidate_pairs.tsv" "$DIST/output/"
cp src/__init__.py src/metrics.py "$CODE/src/"
cp src/v2/__init__.py src/v2/{blocking,ce,common,config,crossfit,crowd,decode,features,house,learn,normalize,novel,predict,prepare,prior,prune,sibling,synth,train}.py "$CODE/src/v2/"
cp src/entity_resolution/__init__.py src/entity_resolution/{block,cache,config,decision,embeddings,evaluate,features,harness_io,normalize,pipeline}.py "$CODE/src/entity_resolution/"
cp scripts/run_final.sh "$CODE/scripts/"
cp docs/submission/README_code.md "$CODE/README.md"
sed -n '/^# ---- Entity resolution pipeline v2/,/^$/p' requirements.txt > "$CODE/requirements.txt"
cp docs/submission/Documentation_template.md "$DIST/"
# smoke check: the packaged code imports on its own
(cd "$CODE" && PYTHONPATH=. python3 -c "import src.v2.train, src.v2.predict, src.v2.blocking, src.v2.synth, src.v2.ce; print('package imports OK')")
(cd dist && zip -qr "${TEAM}_submission.zip" "${TEAM}_submission")
wc -l "$DIST/output/"*.tsv
du -sh "dist/${TEAM}_submission.zip"; unzip -l "dist/${TEAM}_submission.zip" | tail -1
