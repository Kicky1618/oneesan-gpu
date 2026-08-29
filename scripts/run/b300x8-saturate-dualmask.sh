#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
MOD="${2:-4294967291}"
[[ "$N" == 27 ]] || { echo 'dualmask saturation wrapper currently targets n=27/W=28' >&2; exit 2; }
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
ROWS="${ROWS:-28}"
THREADS="${GRIDFP_THREADS:-256}"
ARCH="${ARCH:-native}"
REBUILD="${REBUILD:-0}"
TAG="n27_ilp4warp_dualmask"
ISO="${ISO:-$ONEESAN_BUILD_DIR/${TAG}_gen}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_${TAG}}"
mkdir -p "$ISO" "$ISO/tmp"

[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1&&ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$THREADS" =~ ^[0-9]+$ ]] && ((THREADS>=32&&THREADS<=1024&&THREADS%32==0)) || { echo 'GRIDFP_THREADS must be warp multiple in 32..1024' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

if [[ "$REBUILD" == 1 || ! -x "$BIN" ]]; then
  BASE_BIN="$ISO/base_unused"
  BUILD_OUT="$ISO/base.build.out"; BUILD_ERR="$ISO/base.build.err"
  echo '=== build generated ILP4 + closure-warp source ===' >&2
  ONEESAN_BUILD_DIR="$ISO" ONEESAN_TMP_DIR="$ISO/tmp" \
  N=27 ARCH="$ARCH" OUT="$BASE_BIN" FAST_SHARD_ADDRESS8=1 \
  MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 \
  MAIN_PULL_ILP2=0 HEIGHT_CACHE=0 RANK_DELTA_CACHE=1 RANK_STATE_PACKED=1 \
  RANK_STATE_ILP2=0 RANK_STATE_ILP3=0 RANK_STATE_ILP4=1 \
  BLOCK_CLOSURE_QUAD=0 BLOCK_CLOSURE_WARP=1 HOT_DELTA_TABLE=0 \
  CONCURRENT_GROUP_IO=1 MAXRREGCOUNT=0 PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$BUILD_OUT" 2>"$BUILD_ERR"

  BUILD_SRC="$(sed -nE 's/^  build_source=(.*)$/\1/p' "$BUILD_OUT" | tail -n1)"
  [[ -n "$BUILD_SRC" && -f "$BUILD_SRC" ]] || { echo 'could not resolve generated CUDA source' >&2; exit 3; }
  DUAL_SRC="$ISO/final_dualmask.cu"
  bash "$ONEESAN_ROOT/scripts/bench/b300-block-pull-dualmask-proof.sh"
  bash "$ONEESAN_ROOT/scripts/bench/b300-block-closure-warp-batch-proof.sh"
  python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-closure-warp-dualmask.py" "$BUILD_SRC" "$DUAL_SRC"
  grep -Fq 'b300_closure_warp_endpoint_masks(d)' "$DUAL_SRC"
  grep -Fq 'b300_block_closure_warp_kernel' "$DUAL_SRC"

  echo '=== compile ILP4 + closure-warp + dualmask production binary ===' >&2
  TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v \
    -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 \
    -DB300_FAST_SHARD_ADDRESS8=1 -DB300_BLOCK_CLOSURE_QUAD=0 \
    "$DUAL_SRC" -o "$BIN" 2>"$ISO/dualmask.build.err"
  [[ -x "$BIN" ]] || { echo 'candidate binary missing after nvcc' >&2; exit 3; }
fi

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true
echo "B300 x8 saturation-dualmask n=$N rows=$ROWS threads=$THREADS target=${TARGET_MIB}MiB plan=${PLAN_MIB}MiB window=$MAX_WINDOW" >&2
echo "BIN=$BIN" >&2
export B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$THREADS" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB"
exec "$BIN" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8
