#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
[[ "$N" == 27 ]] || { echo 'calibrated forced build currently targets n=27' >&2; exit 2; }
ARCH="${ARCH:-native}"
OUT="$(build_path "${OUT:-b300_forced_calibrated_n27}")"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
DUALMASK="${DUALMASK:-0}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
[[ "$HIGH_DROP_CHUNK" == 0 || "$HIGH_DROP_CHUNK" == 1 ]] || { echo 'HIGH_DROP_CHUNK must be 0 or 1' >&2; exit 2; }
[[ "$DUALMASK" == 0 || "$DUALMASK" == 1 ]] || { echo 'DUALMASK must be 0 or 1' >&2; exit 2; }
[[ "$PTXAS_VERBOSE" == 0 || "$PTXAS_VERBOSE" == 1 ]] || { echo 'PTXAS_VERBOSE must be 0 or 1' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }

# Isolate all generated CUDA intermediates from concurrent build sessions.
TAG="mainrec_hd${HIGH_DROP_CHUNK}_dual${DUALMASK}_$$"
ISO="${CALIBRATED_BUILD_DIR:-$ONEESAN_BUILD_DIR/calibrated/$TAG}"
mkdir -p "$ISO" "$ISO/tmp" "$(dirname "$OUT")"
BASE_BIN="$ISO/predualmask.bin"
BASE_OUT="$ISO/base.build.out"
BASE_ERR="$ISO/base.build.err"
FINAL_ERR="${BUILD_ERR:-${OUT}.build.err}"

# Run the ordinary production transform chain first. This also executes every
# proof gate owned by b300-hbm32-batch.sh. The dualmask candidate, when enabled,
# is only one post-transform delta from that exact generated source.
ONEESAN_BUILD_DIR="$ISO" ONEESAN_TMP_DIR="$ISO/tmp" \
N=27 ARCH="$ARCH" OUT="$BASE_BIN" MAIN_PULL_ILP=2 HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" \
LOW_MAIN_RECURRENCE=0 HIGH_MAIN_RECURRENCE=0 MAIN_RECURRENCE=1 \
MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 FAST_SHARD_ADDRESS8=1 \
LOW_DROP_CACHE=1 LOW_DROP_CHUNK=1 LOW_BLOCK_CACHE=1 RUNTIME_THREADS=1 PTXAS_VERBOSE="$PTXAS_VERBOSE" \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh" >"$BASE_OUT" 2>"$BASE_ERR"

cat "$BASE_OUT"
cp "$BASE_ERR" "$FINAL_ERR"
grep -Fq 'main_recurrence=1' "$BASE_OUT"
grep -Fq "high_drop_chunk=$HIGH_DROP_CHUNK" "$BASE_OUT"
grep -Fq 'high_recurrence_p_range=27..15 high_symbol_range=14..27' "$BASE_OUT"
BUILD_SRC="$(sed -nE 's/^  build_source=(.*)$/\1/p' "$BASE_OUT" | tail -n1)"
[[ -n "$BUILD_SRC" && -f "$BUILD_SRC" ]] || { echo "could not resolve final generated CUDA source from $BASE_OUT" >&2; exit 3; }

if [[ "$DUALMASK" == 0 ]]; then
  cp "$BASE_BIN" "$OUT"
  chmod +x "$OUT"
  echo "built $OUT"
  echo "  calibrated_forced=1 main_recurrence=1 high_drop_chunk=$HIGH_DROP_CHUNK dualmask=0"
  echo "  source_after_proof_gates=$BUILD_SRC"
  echo "  ptxas_log=$FINAL_ERR"
  exit 0
fi

bash "$ONEESAN_ROOT/scripts/bench/b300-block-pull-dualmask-proof.sh"
DUAL_SRC="$ISO/final_dualmask.cu"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-pull-dualmask.py" "$BUILD_SRC" "$DUAL_SRC"
grep -Fq 'b300_block_endpoint_masks(d)' "$DUAL_SRC"
grep -Fq 'high_rec_groups=' "$DUAL_SRC"
grep -Fq 'b300_main_trit_get(x,p-14)' "$DUAL_SRC"
if grep -Fq 'const MateValue v=mget(d,q);' "$DUAL_SRC"; then
  echo 'stale closure endpoint mget remains after dualmask transform' >&2; exit 4
fi
if grep -Fq 'b300_main_trit_get(x,p-13)' "$DUAL_SRC"; then
  echo 'stale high recurrence p13 artifact remains' >&2; exit 4
fi

PTXAS_FLAGS=(); [[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")
TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" "${PTXAS_FLAGS[@]}" \
  -DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13 "$DUAL_SRC" -o "$OUT" \
  >"$ISO/dualmask.compile.out" 2>>"$FINAL_ERR"

echo "built $OUT"
echo "  calibrated_forced=1 main_recurrence=1 high_drop_chunk=$HIGH_DROP_CHUNK dualmask=1"
echo "  source_before_dualmask=$BUILD_SRC"
echo "  source_after_dualmask=$DUAL_SRC"
echo "  dualmask_proof_gate=1 production_chain_proof_gates=1"
echo "  ptxas_log=$FINAL_ERR"
