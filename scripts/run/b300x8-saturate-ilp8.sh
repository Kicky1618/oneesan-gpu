#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
MOD="${2:-4294967291}"
[[ "$N" == 27 ]] || { echo 'ILP8 saturation wrapper currently targets n=27/W=28' >&2; exit 2; }
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
ROWS="${ROWS:-28}"
THREADS="${GRIDFP_THREADS:-256}"
ARCH="${ARCH:-native}"
RANDOM_CG="${RANDOM_CG:-1}"
WARP_SCAN="${WARP_SCAN:-1}"
CPASYNC="${CPASYNC:-0}"
MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
REBUILD="${REBUILD:-0}"
for x in RANDOM_CG WARP_SCAN CPASYNC REBUILD; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }; done
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] && (( MAXRREGCOUNT == 0 || (MAXRREGCOUNT >= 32 && MAXRREGCOUNT <= 255) )) || { echo 'MAXRREGCOUNT must be 0 or 32..255' >&2; exit 2; }
[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1&&ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$THREADS" =~ ^[0-9]+$ ]] && ((THREADS>=32&&THREADS<=1024&&THREADS%32==0)) || { echo 'GRIDFP_THREADS must be warp multiple in 32..1024' >&2; exit 2; }
# cp.async variant uses 64 bytes dynamic shared memory per thread. Staying at
# <=48 KiB avoids depending on a per-kernel opt-in shared-memory attribute.
if [[ "$CPASYNC" == 1 ]] && (( THREADS > 768 )); then echo 'CPASYNC=1 currently requires GRIDFP_THREADS<=768 (64B/thread dynamic shared)' >&2; exit 2; fi

TAG="n27_mainilp8_warp_dualmask_closuretab"
[[ "$RANDOM_CG" == 1 ]] && TAG="${TAG}_cg"
[[ "$WARP_SCAN" == 1 ]] && TAG="${TAG}_warpscan"
[[ "$CPASYNC" == 1 ]] && TAG="${TAG}_cpasync"
(( MAXRREGCOUNT > 0 )) && TAG="${TAG}_r${MAXRREGCOUNT}"
ISO="${ISO:-$ONEESAN_BUILD_DIR/${TAG}_gen}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_${TAG}}"
mkdir -p "$ISO" "$ISO/tmp"

command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

if [[ "$REBUILD" == 1 || ! -x "$BIN" ]]; then
  BASE_BIN="$ISO/base_unused"
  BUILD_OUT="$ISO/base.build.out"; BUILD_ERR="$ISO/base.build.err"
  echo '=== generate B300 ILP4/closure-warp base for ILP8 post-transform ===' >&2
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
  DUAL_SRC="$ISO/dualmask.cu"
  TABLE_SRC="$ISO/dualmask_closuretab.cu"
  FINAL_SRC="$TABLE_SRC"

  bash "$ONEESAN_ROOT/scripts/bench/b300-ilp8-partition-proof.sh"
  bash "$ONEESAN_ROOT/scripts/bench/b300-block-pull-dualmask-proof.sh"
  bash "$ONEESAN_ROOT/scripts/bench/b300-block-closure-warp-batch-proof.sh"
  bash "$ONEESAN_ROOT/scripts/bench/b300-closure-contrib-table-proof.sh"
  [[ "$WARP_SCAN" == 0 ]] || bash "$ONEESAN_ROOT/scripts/bench/b300-block-closure-warpscan-proof.sh"
  [[ "$CPASYNC" == 0 ]] || ARCH="$ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-gather-proof.sh"

  python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-closure-warp-dualmask.py" "$BUILD_SRC" "$DUAL_SRC"
  python3 "$ONEESAN_ROOT/scripts/build/gen-b300-closure-contrib-table.py" "$DUAL_SRC" "$TABLE_SRC"
  if [[ "$RANDOM_CG" == 1 ]]; then
    CG_SRC="$ISO/random_cg.cu"
    python3 "$ONEESAN_ROOT/scripts/build/gen-b300-rankstate-ilp4-random-cg.py" "$TABLE_SRC" "$CG_SRC"
    FINAL_SRC="$CG_SRC"
  fi
  if [[ "$WARP_SCAN" == 1 ]]; then
    WARP_SCAN_SRC="$ISO/warpscan.cu"
    python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-closure-warpscan.py" "$FINAL_SRC" "$WARP_SCAN_SRC"
    FINAL_SRC="$WARP_SCAN_SRC"
  fi
  ILP8_SRC="$ISO/main_ilp8.cu"
  python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-rankstate-ilp8.py" "$FINAL_SRC" "$ILP8_SRC"
  FINAL_SRC="$ILP8_SRC"
  if [[ "$CPASYNC" == 1 ]]; then
    CPASYNC_SRC="$ISO/final_main_ilp8_cpasync.cu"
    python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-rankstate-ilp8-cpasync.py" "$FINAL_SRC" "$CPASYNC_SRC"
    FINAL_SRC="$CPASYNC_SRC"
  fi

  grep -Fq 'base+=8*grid' "$FINAL_SRC"
  grep -Fq 'b300_main_rankstate_ilp8_blocks' "$FINAL_SRC"
  grep -Fq 'D_BLOCK_CLOSURE_CONTRIB32' "$FINAL_SRC"
  grep -Fq 'b300_block_closure_warp_kernel' "$FINAL_SRC"
  [[ "$RANDOM_CG" == 0 || "$CPASYNC" == 1 ]] || grep -Fq 'b300_rankstate_random_load_cg(in+pj7)' "$FINAL_SRC"
  [[ "$WARP_SCAN" == 0 ]] || grep -Fq 'const int ql=p-2-int(lane)' "$FINAL_SRC"
  if [[ "$CPASYNC" == 1 ]]; then
    grep -Fq 'cp.async.ca.shared.global' "$FINAL_SRC"
    grep -Fq 'size_t(threads)*16u*sizeof(Count)' "$FINAL_SRC"
    [[ "$(grep -Fc 'b300_cpasync_commit();' "$FINAL_SRC")" -eq 2 ]] || { echo 'expected two cp.async groups' >&2; exit 3; }
  fi

  REG_FLAGS=(); (( MAXRREGCOUNT > 0 )) && REG_FLAGS+=("-maxrregcount=$MAXRREGCOUNT")
  echo "=== compile B300 main-ILP8 saturation binary random_cg=$RANDOM_CG warpscan=$WARP_SCAN cpasync=$CPASYNC maxrregcount=$MAXRREGCOUNT ===" >&2
  TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v "${REG_FLAGS[@]}" \
    -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 \
    -DB300_FAST_SHARD_ADDRESS8=1 -DB300_BLOCK_CLOSURE_QUAD=0 \
    "$FINAL_SRC" -o "$BIN" 2>"$ISO/final.build.err"
  [[ -x "$BIN" ]] || { echo 'candidate binary missing after nvcc' >&2; exit 3; }
fi

if [[ -f "$ISO/final.build.err" ]]; then
  echo '=== ILP8 PTXAS resource summary ===' >&2
  grep -E 'ptxas info.*(Function properties|Used [0-9]+ registers|bytes spill|bytes stack frame|bytes cmem|bytes smem)' "$ISO/final.build.err" >&2 || true
fi

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true
echo "B300 x8 saturation-main-ilp8 n=$N rows=$ROWS threads=$THREADS target=${TARGET_MIB}MiB plan=${PLAN_MIB}MiB window=$MAX_WINDOW random_cg=$RANDOM_CG warpscan=$WARP_SCAN cpasync=$CPASYNC maxrregcount=$MAXRREGCOUNT" >&2
features='main_rankstate_ilp8,block_rankstate_ilp4,closure_warp,dualmask,closure_contrib_shift_cross_tables,random_cg_optional,closure_warpscan_optional,concurrent_io'
[[ "$CPASYNC" == 0 ]] || features="${features},main_cpasync_ca_u32_2x8"
echo "features=$features" >&2
echo "BIN=$BIN" >&2
export B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$THREADS" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB"
exec "$BIN" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8
