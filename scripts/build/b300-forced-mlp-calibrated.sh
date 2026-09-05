#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}";[[ "$N" == 27 ]]||{ echo 'MLP calibrated forced build targets n=27' >&2;exit 2; }
ARCH="${ARCH:-native}"
OUT="$(build_path "${OUT:-b300_forced_mlp_calibrated_n27}")"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
DUALMASK="${DUALMASK:-0}"
CLOSURE_BATCH="${CLOSURE_BATCH:-0}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
[[ "$HIGH_DROP_CHUNK" == 0 || "$HIGH_DROP_CHUNK" == 1 ]]||{ echo 'HIGH_DROP_CHUNK must be 0 or 1' >&2;exit 2; }
[[ "$DUALMASK" == 0 || "$DUALMASK" == 1 ]]||{ echo 'DUALMASK must be 0 or 1' >&2;exit 2; }
case "$CLOSURE_BATCH" in 0|2|4);;*)echo 'CLOSURE_BATCH must be 0,2,4' >&2;exit 2;;esac
[[ "$PTXAS_VERBOSE" == 0 || "$PTXAS_VERBOSE" == 1 ]]||{ echo 'PTXAS_VERBOSE must be 0 or 1' >&2;exit 2; }
command -v nvcc >/dev/null||{ echo 'nvcc required' >&2;exit 2; }

TAG="hd${HIGH_DROP_CHUNK}_dual${DUALMASK}_cb${CLOSURE_BATCH}_$$"
ISO="${MLP_BUILD_DIR:-$ONEESAN_BUILD_DIR/calibrated_mlp/$TAG}"
mkdir -p "$ISO" "$ISO/tmp" "$(dirname "$OUT")"
BASE_BIN="$ISO/core.bin";BASE_OUT="$ISO/core.build.out";BASE_ERR="$ISO/core.build.err"

# Existing calibrated builder owns production proof gates and optional dualmask.
N=27 ARCH="$ARCH" OUT="$BASE_BIN" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" DUALMASK="$DUALMASK" \
BUILD_ERR="$BASE_ERR" CALIBRATED_BUILD_DIR="$ISO/corechain" PTXAS_VERBOSE="$PTXAS_VERBOSE" \
  bash "$ONEESAN_ROOT/scripts/build/b300-forced-calibrated.sh" >"$BASE_OUT" 2>"$ISO/core.driver.err"
cat "$BASE_OUT"
[[ -x "$BASE_BIN" ]]||{ echo 'core calibrated binary missing' >&2;exit 3; }
if [[ "$DUALMASK" == 1 ]];then SRC="$(sed -nE 's/^  source_after_dualmask=(.*)$/\1/p' "$BASE_OUT"|tail -n1)";else SRC="$(sed -nE 's/^  source_after_proof_gates=(.*)$/\1/p' "$BASE_OUT"|tail -n1)";fi
[[ -n "$SRC" && -f "$SRC" ]]||{ echo 'could not resolve core calibrated CUDA source' >&2;exit 3; }

FINAL_ERR="${BUILD_ERR:-${OUT}.build.err}";cp "$BASE_ERR" "$FINAL_ERR"
if [[ "$CLOSURE_BATCH" == 0 ]];then
  cp "$BASE_BIN" "$OUT";chmod +x "$OUT"
  echo "built $OUT"
  echo "  mlp_calibrated_forced=1 high_drop_chunk=$HIGH_DROP_CHUNK dualmask=$DUALMASK closure_batch=0"
  echo "  source_after_core=$SRC"
  echo "  ptxas_log=$FINAL_ERR"
  exit 0
fi

BATCH_SRC="$ISO/final_closure_batch.cu"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-closure-quad.py" "$SRC" "$BATCH_SRC" >"$ISO/closure_batch.transform.out"
grep -Fq 'batch_macro=B300_BLOCK_CLOSURE_BATCH' "$ISO/closure_batch.transform.out"
grep -Fq 'high_rec_groups=' "$BATCH_SRC"
[[ "$DUALMASK" == 0 ]]||grep -Fq 'b300_block_endpoint_masks(d)' "$BATCH_SRC"
PTXAS_FLAGS=();[[ "$PTXAS_VERBOSE" == 1 ]]&&PTXAS_FLAGS+=("-Xptxas=-v")
TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" "${PTXAS_FLAGS[@]}" \
  -DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13 -DB300_BLOCK_CLOSURE_BATCH="$CLOSURE_BATCH" \
  "$BATCH_SRC" -o "$OUT" >"$ISO/closure_batch.compile.out" 2>>"$FINAL_ERR"
[[ -x "$OUT" ]]||{ echo 'MLP calibrated binary missing after compile' >&2;exit 3; }
echo "built $OUT"
echo "  mlp_calibrated_forced=1 high_drop_chunk=$HIGH_DROP_CHUNK dualmask=$DUALMASK closure_batch=$CLOSURE_BATCH"
echo "  source_before_closure_batch=$SRC"
echo "  source_after_closure_batch=$BATCH_SRC"
echo "  production_chain_proof_gates=1 closure_batch_post_transform=1"
echo "  ptxas_log=$FINAL_ERR"
