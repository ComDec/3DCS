#!/usr/bin/env bash
# Reproduce Table 1 (geometry, rotation dataset) for GemNet and compare with expected.csv.
#
#   bash reproduce/table1_geometry/run.sh            # all 16 shards (compares with expected.csv)
#   QUICK=1 bash reproduce/table1_geometry/run.sh    # shard 1 only (compares with expected_quick.csv)
#
# Environment variables:
#   NJOBS  worker processes (default 24)
#   DATA   data root for downloads (default data)
#   OUT    output directory (default results/reproduce/table1_geometry[_quick])
#   SKIP_DOWNLOAD=1  use data already present under $DATA/hf/rotation and $DATA/embeddings/rotation/gemnet
#
# Downloads: EscheWang/3dcs config "rotation" (~7.5 GB parquet, ~25 GB after save_to_disk) and the
# GemNet rotation embeddings from EscheWang/3dcs-embeddings (16 files, ~5.2 GB).
# Measured on 24 CPU workers (A100 box, cosine space only): about 50-90 s per shard per metric
# version, so both full runs take roughly 30-50 min (QUICK=1: about 3-5 min), plus download time.
# Shard 1 is used for QUICK=1 because it contains one of the two molecules that fail and therefore
# exercises both per-shard offsets and --replicate-offset-drift.
#
# Only GemNet can be recomputed: rotation embeddings for E3FP, UniMol, MolAE and MolSpectra are
# not available, so their expected.csv rows report MISSING (their reference values come from the
# original per-molecule metric outputs).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
NJOBS="${NJOBS:-24}"
DATA="${DATA:-data}"
QUICK="${QUICK:-0}"
PY="${PYTHON:-python}"

if [[ "$QUICK" == "1" ]]; then
  OUT="${OUT:-results/reproduce/table1_geometry_quick}"
  SHARD_ARGS=(--shards 1)
  TABLE="1-shard1"
  EXPECTED="$HERE/expected_quick.csv"
else
  OUT="${OUT:-results/reproduce/table1_geometry}"
  SHARD_ARGS=()
  TABLE="1"
  EXPECTED="$HERE/expected.csv"
fi
mkdir -p "$OUT"

echo "== 1/4 download dataset and embeddings"
if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  $PY -m three_dbench download dataset --task rotation --out "$DATA/hf"
  $PY -m three_dbench download embeddings --task rotation --models gemnet --out "$DATA/embeddings"
fi

for VERSION in paper v2; do
  echo "== 2/4 evaluate GemNet (metric-version $VERSION)"
  EXTRA=()
  [[ "$VERSION" == "paper" ]] && EXTRA=(--replicate-offset-drift)
  $PY -m three_dbench evaluate rotation \
    --dataset-dir "$DATA/hf/rotation" \
    --embeddings "$DATA/embeddings/rotation/gemnet" \
    --layout by-shard \
    --embedding-key gemnet \
    --model-name gemnet \
    --metrics cosine \
    --metric-version "$VERSION" \
    --n-jobs "$NJOBS" \
    ${SHARD_ARGS[@]+"${SHARD_ARGS[@]}"} \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    --output-dir "$OUT/gemnet_$VERSION"
done

echo "== 3/4 collect results"
$PY "$HERE/collect.py" \
  --run paper gemnet "$OUT/gemnet_paper/gemnet_per_key.parquet" \
  --run v2 gemnet "$OUT/gemnet_v2/gemnet_per_key.parquet" \
  --table "$TABLE" \
  --out "$OUT/results.csv"

echo "== 4/4 compare with $(basename "$EXPECTED")"
$PY "$ROOT/reproduce/compare.py" --expected "$EXPECTED" --results "$OUT/results.csv"
