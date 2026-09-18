#!/usr/bin/env bash
# Reproduce Table 2 (zero-shot chirality) of the 3DCS paper from the published embeddings.
#
#   bash reproduce/table2_chirality/run.sh
#
# Steps: (1) download the HF chirality config of EscheWang/3dcs, (2) download the 7 Table-2
# embedding files from EscheWang/3dcs-embeddings and check their SHA-256 (models.csv), (3) run
# `python -m three_dbench evaluate chirality` for every model and variant, (4) write results.csv
# and compare it with expected.csv (reproduce/compare.py). Only rows present in results.csv can
# pass: restricting VARIANTS/MODELS makes the comparison report the other rows as missing.
#
# Environment variables (all optional):
#   PYTHON        python interpreter                          (default: python)
#   DATASET_DIR   save_to_disk dir of the chirality config    (default: <repo>/data/hf/chirality)
#   EMB_ROOT      root holding chirality/<model>/<file>       (default: <repo>/data/embeddings)
#   OUT_DIR       per-run outputs                              (default: <repo>/results/reproduce/table2_chirality)
#   RESULTS_CSV   collected results                            (default: $OUT_DIR/results.csv)
#   VARIANTS      subset of: euclidean cosine v2_euclidean v2_cosine   (default: all four)
#   MODELS        subset of: e3fp gemnet molae molspectra unimol fmg mace (default: all seven)
#   N_JOBS        worker processes per evaluation              (default: 8)
#   HF_DATASET_REVISION / HF_EMBEDDINGS_REVISION   pin Hub revisions (default: main)
#   SKIP_DOWNLOAD=1  use DATASET_DIR / EMB_ROOT as they are; SKIP_SHA256=1  skip the checksum check
#
# Variant "euclidean" is the published protocol (--distance euclidean --metric-version paper).
# Runtime with N_JOBS=22 on an AMD EPYC 7513: ~1-2 min per model and variant (E3FP ~40 s).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-python}"
DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/data/hf/chirality}"
EMB_ROOT="${EMB_ROOT:-$REPO_ROOT/data/embeddings}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/reproduce/table2_chirality}"
VARIANTS="${VARIANTS:-euclidean cosine v2_euclidean v2_cosine}"
MODELS="${MODELS:-e3fp gemnet molae molspectra unimol fmg mace}"
N_JOBS="${N_JOBS:-8}"
RESULTS_CSV="${RESULTS_CSV:-$OUT_DIR/results.csv}"
export HF_DATASET_REVISION="${HF_DATASET_REVISION:-}" HF_EMBEDDINGS_REVISION="${HF_EMBEDDINGS_REVISION:-}"

# Use the code of this checkout even if another copy of three_dbench is installed.
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -c "import three_dbench, sys; print('three_dbench from', three_dbench.__file__); sys.exit(0)"

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  echo "== [1/4] dataset: EscheWang/3dcs (config chirality) -> $DATASET_DIR"
  DATASET_DIR="$DATASET_DIR" "$PYTHON" - <<'PY'
import os
from pathlib import Path
out = Path(os.environ["DATASET_DIR"])
if out.exists() and any(out.iterdir()):
    print(f"   {out} exists, skipping download")
else:
    from datasets import load_dataset
    kw = {"name": "chirality", "split": "train"}
    if os.environ.get("HF_DATASET_REVISION"):
        kw["revision"] = os.environ["HF_DATASET_REVISION"]
    ds = load_dataset("EscheWang/3dcs", **kw)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    print(f"   saved {len(ds)} rows / {sum(ds['n_conformers'])} conformers")
PY

  echo "== [2/4] embeddings: EscheWang/3dcs-embeddings -> $EMB_ROOT"
  EMB_ROOT="$EMB_ROOT" MODELS="$MODELS" MODELS_CSV="$HERE/models.csv" "$PYTHON" - <<'PY'
import csv, os
from pathlib import Path
root = Path(os.environ["EMB_ROOT"])
wanted = set(os.environ["MODELS"].split())
rows = [r for r in csv.DictReader(open(os.environ["MODELS_CSV"])) if r["model"] in wanted]
missing = [r for r in rows if not (root / r["path"]).exists()]
if not missing:
    print("   all files present, skipping download")
else:
    from huggingface_hub import hf_hub_download
    rev = os.environ.get("HF_EMBEDDINGS_REVISION") or None
    for r in missing:
        print(f"   downloading {r['path']} ({int(r['size_bytes']) / 1e6:.1f} MB)")
        hf_hub_download("EscheWang/3dcs-embeddings", r["path"], repo_type="dataset", revision=rev, local_dir=str(root))
PY
fi

if [[ "${SKIP_SHA256:-0}" != "1" ]]; then
  EMB_ROOT="$EMB_ROOT" MODELS="$MODELS" MODELS_CSV="$HERE/models.csv" "$PYTHON" - <<'PY'
import csv, hashlib, os, sys
from pathlib import Path
root = Path(os.environ["EMB_ROOT"])
wanted = set(os.environ["MODELS"].split())
bad = []
for r in csv.DictReader(open(os.environ["MODELS_CSV"])):
    if r["model"] not in wanted:
        continue
    p = root / r["path"]
    if not p.exists():
        bad.append(f"{p}: missing"); continue
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    if h.hexdigest() != r["sha256"]:
        bad.append(f"{p}: sha256 {h.hexdigest()} != {r['sha256']}")
if bad:
    sys.exit("embedding check failed:\n  " + "\n  ".join(bad))
print("   sha256 OK for", len(wanted), "models")
PY
fi

echo "== [3/4] evaluate (variants: $VARIANTS; models: $MODELS; N_JOBS=$N_JOBS)"
key_of() { awk -F, -v m="$1" '$1==m {print $3}' "$HERE/models.csv"; }
path_of() { awk -F, -v m="$1" '$1==m {print $2}' "$HERE/models.csv"; }
for variant in $VARIANTS; do
  case "$variant" in
    euclidean)    dist=euclidean; version=paper ;;
    cosine)       dist=cosine;    version=paper ;;
    v2_euclidean) dist=euclidean; version=v2 ;;
    v2_cosine)    dist=cosine;    version=v2 ;;
    *) echo "unknown variant $variant" >&2; exit 2 ;;
  esac
  for model in $MODELS; do
    out="$OUT_DIR/$variant/$model"
    if [[ -f "$out/summary.csv" && "${FORCE:-0}" != "1" ]]; then
      echo "   $variant/$model: $out/summary.csv exists, skipping (FORCE=1 to recompute)"; continue
    fi
    mkdir -p "$out"
    start=$(date +%s)
    "$PYTHON" -m three_dbench evaluate chirality \
      --dataset-dir "$DATASET_DIR" \
      --embeddings "$EMB_ROOT/$(path_of "$model")" \
      --embedding-key "$(key_of "$model")" \
      --model-name "$model" \
      --output-dir "$out" \
      --distance "$dist" \
      --metric-version "$version" \
      --unsup-kmax n-1 \
      --n-jobs "$N_JOBS" > "$out/log.txt" 2>&1 || { echo "FAILED: $variant/$model (see $out/log.txt)" >&2; exit 1; }
    echo "   $variant/$model: $(( $(date +%s) - start )) s  $(grep '^ES-AUC' "$out/log.txt" || true)"
  done
done

echo "== [4/4] results.csv"
"$PYTHON" "$HERE/make_results.py" --runs "$OUT_DIR" --out "$RESULTS_CSV"
if [[ -f "$REPO_ROOT/reproduce/compare.py" ]]; then
  "$PYTHON" "$REPO_ROOT/reproduce/compare.py" --expected "$HERE/expected.csv" --results "$RESULTS_CSV"
else
  echo "reproduce/compare.py not found; compare $RESULTS_CSV with $HERE/expected.csv manually"
fi
