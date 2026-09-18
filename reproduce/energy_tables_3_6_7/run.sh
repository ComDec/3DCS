#!/usr/bin/env bash
# Reproduce the zero-shot energy tables of the 3DCS paper (ICLR 2026):
#   Table 3 (7 models, main text), Table 6 (6 models, mean +/- 95% CI) and Table 7 (energy-jump counts).
#
# Steps: download the float64 rMD17 energies (EscheWang/3dcs, config traj_energies) and the backed-up
# trajectory embeddings (EscheWang/3dcs-embeddings, traj/<model>/), run
# `python -m three_dbench evaluate traj` per model with the published protocol, collect the table cells
# into results.csv and compare them with expected.csv.
#
# Environment variables (all optional):
#   PYTHON=python  OUT=<this dir>/out  N_JOBS=16  MODELS="e3fp gemnet molae molspectra unimol fmg mace"
#   METRIC_VERSIONS="paper"      (add v2 for the corrected definitions: METRIC_VERSIONS="paper v2")
#   ENERGY_DATASET_DIR=...       use an existing save_to_disk copy of traj_energies instead of downloading
#   EMB_ROOT=...                 use existing embeddings: EMB_ROOT/<model>/rmd17_<mol>.{npz,pkl}
#   EMB_LAYOUT=hf|nyubox         nyubox = original backup directory names (molspec, FMG)
#   HF_DATASET_REPO, HF_DATASET_REVISION, HF_EMBEDDINGS_REPO, HF_EMBEDDINGS_REVISION
#   FORCE=1                      recompute models whose summary.csv already exists
#   EXTRA_ARGS="..."             extra CLI options appended last (e.g. "--molecules aspirin --n-samples 2" for a smoke test)
# Runtime: about 1000 windows x 7 models x ~4.5 CPU-s per window, i.e. ~25 min with N_JOBS=24 per metric version.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-python}"
OUT="${OUT:-$HERE/out}"
N_JOBS="${N_JOBS:-16}"
MODELS="${MODELS:-e3fp gemnet molae molspectra unimol fmg mace}"
METRIC_VERSIONS="${METRIC_VERSIONS:-paper}"
HF_DATASET_REPO="${HF_DATASET_REPO:-EscheWang/3dcs}"
HF_DATASET_REVISION="${HF_DATASET_REVISION:-main}"
HF_EMBEDDINGS_REPO="${HF_EMBEDDINGS_REPO:-EscheWang/3dcs-embeddings}"
HF_EMBEDDINGS_REVISION="${HF_EMBEDDINGS_REVISION:-main}"
ENERGY_DATASET_DIR="${ENERGY_DATASET_DIR:-$OUT/data/traj_energies}"
EMB_LAYOUT="${EMB_LAYOUT:-hf}"
FORCE="${FORCE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# One BLAS thread per worker: no oversubscription and reproducible float32 BLAS results.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p "$OUT"

# model id -> "<display name> <nyubox dir> <npz key or ->"
model_spec() {
  case "$1" in
    e3fp) echo "E3FP e3fp -" ;;
    gemnet) echo "GemNet gemnet gemnet" ;;
    molae) echo "MolAE molae arr_0" ;;
    molspectra) echo "MolSpectra molspec arr_0" ;;
    unimol) echo "UniMol unimol arr_0" ;;
    fmg) echo "FMG FMG embeddings" ;;
    mace) echo "MACE mace arr_0" ;;
    *) echo "unknown model $1" >&2; return 1 ;;
  esac
}

# 1) float64 energies
if [ ! -d "$ENERGY_DATASET_DIR" ]; then
  echo "== downloading $HF_DATASET_REPO (traj_energies, revision $HF_DATASET_REVISION)"
  "$PYTHON" - "$HF_DATASET_REPO" "$HF_DATASET_REVISION" "$ENERGY_DATASET_DIR" <<'PY'
import sys
from datasets import load_dataset
repo, revision, out = sys.argv[1:4]
ds = load_dataset(repo, name="traj_energies", split="train", revision=revision)
ds.save_to_disk(out)
print(ds)
PY
fi

# 2) embeddings
if [ -z "${EMB_ROOT:-}" ]; then
  EMB_ROOT="$OUT/embeddings/traj"
  EMB_LAYOUT=hf
  echo "== downloading traj embeddings for: $MODELS from $HF_EMBEDDINGS_REPO"
  "$PYTHON" - "$HF_EMBEDDINGS_REPO" "$HF_EMBEDDINGS_REVISION" "$OUT/embeddings" $MODELS <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, revision, local_dir, *models = sys.argv[1:]
snapshot_download(repo_id=repo, repo_type="dataset", revision=revision, local_dir=local_dir,
                  allow_patterns=[f"traj/{m}/rmd17_*" for m in models])
PY
fi

# 3) evaluate
TIMINGS="$OUT/timings.tsv"
[ -f "$TIMINGS" ] || printf "metric_version\tmodel\tseconds\tn_jobs\n" > "$TIMINGS"
for version in $METRIC_VERSIONS; do
  for model in $MODELS; do
    read -r name nyubox_dir key <<<"$(model_spec "$model")"
    if [ "$EMB_LAYOUT" = "nyubox" ]; then emb_dir="$EMB_ROOT/$nyubox_dir"; else emb_dir="$EMB_ROOT/$model"; fi
    run_dir="$OUT/runs/$version/$model"
    if [ -f "$run_dir/summary.csv" ] && [ "$FORCE" != "1" ]; then
      echo "== $version/$model: summary.csv exists, skipping (FORCE=1 to recompute)"
      continue
    fi
    key_args=()
    [ "$key" != "-" ] && key_args=(--embedding-key "$key")
    echo "== $version/$model ($name) from $emb_dir"
    start=$(date +%s)
    "$PYTHON" -m three_dbench evaluate traj \
      --dataset-dir "$ENERGY_DATASET_DIR" \
      --embeddings "$emb_dir" ${key_args[@]+"${key_args[@]}"} \
      --model-name "$name" \
      --output-dir "$run_dir" \
      --window-scheme legacy --n-samples 100 --window 2000 --random-seed 2025 \
      --metric-version "$version" \
      --n-jobs "$N_JOBS" $EXTRA_ARGS
    printf "%s\t%s\t%s\t%s\n" "$version" "$model" "$(( $(date +%s) - start ))" "$N_JOBS" >> "$TIMINGS"
  done
done

# 4) collect table cells
"$PYTHON" "$HERE/collect.py" --runs "$OUT/runs" --out "$OUT/results.csv" \
  --expected "$HERE/expected.csv" --expected-out "$OUT/expected_selected.csv"

# 5) compare with expected values
if [ -f "$REPO/reproduce/compare.py" ]; then
  "$PYTHON" "$REPO/reproduce/compare.py" --expected "$OUT/expected_selected.csv" --results "$OUT/results.csv"
else
  echo "reproduce/compare.py not found; results written to $OUT/results.csv" >&2
fi
