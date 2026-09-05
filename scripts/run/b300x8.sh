#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
MOD="${2:-4294967291}"
TARGET_MIB_WAS_SET="${TARGET_MIB+x}"
TARGET_MIB="${TARGET_MIB:-16384}"
GRIDFP_PLAN_TARGET_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
NGPU="${NGPU:-8}"
ROWS="${ROWS:-$((N+1))}"
FAST_SHARD_ADDRESS8="${FAST_SHARD_ADDRESS8:-1}"
MAIN_MATE_CACHE="${MAIN_MATE_CACHE:-1}"
MAIN_PULL="${MAIN_PULL:-1}"
BLOCK_PULL="${BLOCK_PULL:-1}"
BLOCK_MATE_CACHE="${BLOCK_MATE_CACHE:-$BLOCK_PULL}"
MAIN_PULL_ILP2="${MAIN_PULL_ILP2:-0}"
HEIGHT_CACHE="${HEIGHT_CACHE:-0}"
RANK_DELTA_CACHE="${RANK_DELTA_CACHE:-1}"
RANK_STATE_PACKED="${RANK_STATE_PACKED:-$RANK_DELTA_CACHE}"
RANK_STATE_ILP2="${RANK_STATE_ILP2:-$RANK_STATE_PACKED}"
RANK_STATE_ILP3="${RANK_STATE_ILP3:-0}"
RANK_STATE_ILP4="${RANK_STATE_ILP4:-0}"
BLOCK_CLOSURE_QUAD="${BLOCK_CLOSURE_QUAD:-0}"
BLOCK_CLOSURE_WARP="${BLOCK_CLOSURE_WARP:-0}"
HOT_DELTA_TABLE="${HOT_DELTA_TABLE:-0}"
CONCURRENT_GROUP_IO="${CONCURRENT_GROUP_IO:-1}"
MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
GRIDFP_PLAN_TARGET_DIVISOR="${GRIDFP_PLAN_TARGET_DIVISOR:-1}"
GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}"
ROW7_TENSOR="${ROW7_TENSOR:-0}"
ROW8_TENSOR="${ROW8_TENSOR:-0}"
ROW8_STRUCTURAL="${ROW8_STRUCTURAL:-0}"
OWNERFUSED="${OWNERFUSED:-auto}"
BLOCKFUSED="${BLOCKFUSED:-0}"
GROUPBATCH="${GROUPBATCH:-0}"
GRIDFP_GROUPBATCH_DYNAMIC="${GRIDFP_GROUPBATCH_DYNAMIC:-1}"
GRIDFP_GROUPBATCH_AFFINITY="${GRIDFP_GROUPBATCH_AFFINITY:-1}"
GRIDFP_GROUPBATCH_STICKY="${GRIDFP_GROUPBATCH_STICKY:-1}"
GRIDFP_GROUPBATCH_RECLAIM="${GRIDFP_GROUPBATCH_RECLAIM:-1}"
GRIDFP_GROUPBATCH_GRAPH="${GRIDFP_GROUPBATCH_GRAPH:-1}"
GRIDFP_GROUPBATCH_P1_DEST="${GRIDFP_GROUPBATCH_P1_DEST:-1}"
GRIDFP_GROUPBATCH_WRAP32="${GRIDFP_GROUPBATCH_WRAP32:-0}"
GRIDFP_GROUPBATCH_MATCHLUT="${GRIDFP_GROUPBATCH_MATCHLUT:-0}"
GRIDFP_GROUPBATCH_BATCHES_PER_GPU="${GRIDFP_GROUPBATCH_BATCHES_PER_GPU:-6}"
if (( GROUPBATCH && N >= 27 )) && [[ -z "$TARGET_MIB_WAS_SET" ]]; then TARGET_MIB=10240; fi
require_uint ROW7_TENSOR "$ROW7_TENSOR" || exit 2
require_uint ROW8_TENSOR "$ROW8_TENSOR" || exit 2
require_uint ROW8_STRUCTURAL "$ROW8_STRUCTURAL" || exit 2
require_uint BLOCKFUSED "$BLOCKFUSED" || exit 2
require_uint GROUPBATCH "$GROUPBATCH" || exit 2
if (( ROW7_TENSOR != 0 && ROW7_TENSOR != 1 )); then echo "ROW7_TENSOR must be 0 or 1" >&2; exit 2; fi
if (( ROW8_TENSOR != 0 && ROW8_TENSOR != 1 )); then echo "ROW8_TENSOR must be 0 or 1" >&2; exit 2; fi
if (( ROW8_STRUCTURAL != 0 && ROW8_STRUCTURAL != 1 )); then echo "ROW8_STRUCTURAL must be 0 or 1" >&2; exit 2; fi
if (( BLOCKFUSED != 0 && BLOCKFUSED != 1 )); then echo "BLOCKFUSED must be 0 or 1" >&2; exit 2; fi
if (( GROUPBATCH != 0 && GROUPBATCH != 1 )); then echo "GROUPBATCH must be 0 or 1" >&2; exit 2; fi
if (( GROUPBATCH )); then
  for spec in "GRIDFP_GROUPBATCH_DYNAMIC:$GRIDFP_GROUPBATCH_DYNAMIC" "GRIDFP_GROUPBATCH_AFFINITY:$GRIDFP_GROUPBATCH_AFFINITY" "GRIDFP_GROUPBATCH_STICKY:$GRIDFP_GROUPBATCH_STICKY" "GRIDFP_GROUPBATCH_RECLAIM:$GRIDFP_GROUPBATCH_RECLAIM" "GRIDFP_GROUPBATCH_GRAPH:$GRIDFP_GROUPBATCH_GRAPH" "GRIDFP_GROUPBATCH_P1_DEST:$GRIDFP_GROUPBATCH_P1_DEST" "GRIDFP_GROUPBATCH_WRAP32:$GRIDFP_GROUPBATCH_WRAP32" "GRIDFP_GROUPBATCH_MATCHLUT:$GRIDFP_GROUPBATCH_MATCHLUT" "GRIDFP_GROUPBATCH_BATCHES_PER_GPU:$GRIDFP_GROUPBATCH_BATCHES_PER_GPU"; do require_uint "${spec%%:*}" "${spec#*:}" || exit 2; done
  if (( GRIDFP_GROUPBATCH_DYNAMIC > 1 || GRIDFP_GROUPBATCH_AFFINITY > 1 || GRIDFP_GROUPBATCH_STICKY > 1 || GRIDFP_GROUPBATCH_RECLAIM > 1 )); then echo "GRIDFP_GROUPBATCH_DYNAMIC/AFFINITY/STICKY/RECLAIM must be 0 or 1" >&2; exit 2; fi
  if (( GRIDFP_GROUPBATCH_GRAPH > 2 )); then echo "GRIDFP_GROUPBATCH_GRAPH must be 0, 1, or 2" >&2; exit 2; fi
  if (( GRIDFP_GROUPBATCH_P1_DEST > 1 )); then echo "GRIDFP_GROUPBATCH_P1_DEST must be 0 or 1" >&2; exit 2; fi
  if (( GRIDFP_GROUPBATCH_WRAP32 > 1 )); then echo "GRIDFP_GROUPBATCH_WRAP32 must be 0 or 1" >&2; exit 2; fi
  if (( GRIDFP_GROUPBATCH_MATCHLUT > 1 )); then echo "GRIDFP_GROUPBATCH_MATCHLUT must be 0 or 1" >&2; exit 2; fi
fi
if (( ROW8_STRUCTURAL && ! ROW8_TENSOR )); then echo "ROW8_STRUCTURAL requires ROW8_TENSOR=1" >&2; exit 2; fi
if (( ROW7_TENSOR && ROW8_TENSOR )); then echo "ROW7_TENSOR and ROW8_TENSOR are mutually exclusive" >&2; exit 2; fi
if [[ "$OWNERFUSED" == "auto" ]]; then
  OWNERFUSED=0
  if (( N >= 20 && N < 27 && ! BLOCKFUSED && ! GROUPBATCH && ! ROW7_TENSOR && ! ROW8_TENSOR )) && grep -Eq "(^|[^0-9])${MOD}u([^0-9]|$)" "$ONEESAN_ROOT/src/cuda/b300/row6_automaton_crt20_generated.hpp"; then OWNERFUSED=1; fi
fi
require_uint OWNERFUSED "$OWNERFUSED" || exit 2
if (( OWNERFUSED != 0 && OWNERFUSED != 1 )); then echo "OWNERFUSED must be 0, 1, or auto" >&2; exit 2; fi
if (( OWNERFUSED + BLOCKFUSED + GROUPBATCH > 1 )); then echo "OWNERFUSED, BLOCKFUSED and GROUPBATCH are mutually exclusive" >&2; exit 2; fi
if (( OWNERFUSED && (ROW7_TENSOR || ROW8_TENSOR) )); then echo "OWNERFUSED is mutually exclusive with tensor modes" >&2; exit 2; fi
if (( BLOCKFUSED && (ROW7_TENSOR || ROW8_TENSOR) )); then echo "BLOCKFUSED is mutually exclusive with tensor modes" >&2; exit 2; fi
if (( GROUPBATCH && (ROW7_TENSOR || ROW8_TENSOR) )); then echo "GROUPBATCH is mutually exclusive with tensor modes" >&2; exit 2; fi
CUSTOM_BIN=0
if [[ -n "${BIN:-}" ]]; then CUSTOM_BIN=1; fi
DEFAULT_BIN="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_batch_n${N}"
if (( OWNERFUSED )); then DEFAULT_BIN="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_ownerfused_batch_n${N}"; fi
if (( BLOCKFUSED )); then DEFAULT_BIN="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_blockfused_batch_n${N}"; fi
if (( GROUPBATCH )); then DEFAULT_BIN="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_groupbatch_batch_n${N}"; fi
if (( ROW7_TENSOR )); then DEFAULT_BIN="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_row7tensor_batch_n${N}"; fi
if (( ROW8_TENSOR )); then DEFAULT_BIN="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_row8tensor_batch_n${N}"; fi
BIN="${BIN:-$DEFAULT_BIN}"

for spec in "N:$N" "MOD:$MOD" "TARGET_MIB:$TARGET_MIB" "MAX_WINDOW:$MAX_WINDOW" "NGPU:$NGPU" "GRIDFP_VRAM_RESERVE_MIB:$GRIDFP_VRAM_RESERVE_MIB"; do
  require_uint "${spec%%:*}" "${spec#*:}" || exit 2
done
REBUILD="${REBUILD:-0}"

for name in FAST_SHARD_ADDRESS8 MAIN_MATE_CACHE MAIN_PULL BLOCK_PULL BLOCK_MATE_CACHE MAIN_PULL_ILP2 HEIGHT_CACHE RANK_DELTA_CACHE RANK_STATE_PACKED RANK_STATE_ILP2 RANK_STATE_ILP3 RANK_STATE_ILP4 BLOCK_CLOSURE_QUAD BLOCK_CLOSURE_WARP HOT_DELTA_TABLE CONCURRENT_GROUP_IO REBUILD; do
  value="${!name}"
  if [[ "$value" != 0 && "$value" != 1 ]]; then echo "$name must be 0 or 1" >&2; exit 2; fi
done
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] && (( MAXRREGCOUNT == 0 || (MAXRREGCOUNT >= 32 && MAXRREGCOUNT <= 255) )) || { echo "MAXRREGCOUNT must be 0 or 32..255" >&2; exit 2; }
[[ "$ROWS" =~ ^[0-9]+$ ]] && (( ROWS >= 1 && ROWS <= N + 1 )) || { echo "ROWS must be 1..$((N+1))" >&2; exit 2; }
[[ "$GRIDFP_PLAN_TARGET_MIB" =~ ^[0-9]+$ ]] && (( GRIDFP_PLAN_TARGET_MIB >= 1 && GRIDFP_PLAN_TARGET_MIB <= TARGET_MIB )) || { echo "GRIDFP_PLAN_TARGET_MIB must be 1..TARGET_MIB" >&2; exit 2; }
[[ "$GRIDFP_PLAN_TARGET_DIVISOR" =~ ^[0-9]+$ ]] && (( GRIDFP_PLAN_TARGET_DIVISOR >= 1 && GRIDFP_PLAN_TARGET_DIVISOR <= 16 )) || { echo "GRIDFP_PLAN_TARGET_DIVISOR must be 1..16" >&2; exit 2; }
if [[ "$MAIN_PULL" == 1 && "$MAIN_MATE_CACHE" != 1 ]]; then echo "MAIN_PULL=1 requires MAIN_MATE_CACHE=1" >&2; exit 2; fi
if [[ "$BLOCK_PULL" == 1 && "$MAIN_PULL" != 1 ]]; then echo "BLOCK_PULL=1 requires MAIN_PULL=1" >&2; exit 2; fi
if [[ "$BLOCK_MATE_CACHE" == 1 && "$BLOCK_PULL" != 1 ]]; then echo "BLOCK_MATE_CACHE=1 requires BLOCK_PULL=1" >&2; exit 2; fi
if (( MAIN_PULL_ILP2 + HEIGHT_CACHE + RANK_DELTA_CACHE > 1 )); then echo "MAIN_PULL_ILP2, HEIGHT_CACHE and RANK_DELTA_CACHE are separate base experiments; use packed rank-state ILP2/3/4 to combine ILP with rank recurrence" >&2; exit 2; fi
for name in MAIN_PULL_ILP2 HEIGHT_CACHE RANK_DELTA_CACHE; do
  if [[ "${!name}" == 1 && ( "$MAIN_PULL" != 1 || "$BLOCK_PULL" != 1 || "$MAIN_MATE_CACHE" != 1 || "$BLOCK_MATE_CACHE" != 1 ) ]]; then echo "$name=1 requires full-pull plus both MateID caches" >&2; exit 2; fi
done
if [[ "$RANK_STATE_PACKED" == 1 && "$RANK_DELTA_CACHE" != 1 ]]; then echo "RANK_STATE_PACKED=1 requires RANK_DELTA_CACHE=1" >&2; exit 2; fi
for name in RANK_STATE_ILP2 RANK_STATE_ILP3 RANK_STATE_ILP4; do
  if [[ "${!name}" == 1 && "$RANK_STATE_PACKED" != 1 ]]; then echo "$name=1 requires RANK_STATE_PACKED=1" >&2; exit 2; fi
done
if (( RANK_STATE_ILP2 + RANK_STATE_ILP3 + RANK_STATE_ILP4 > 1 )); then echo "RANK_STATE_ILP2/3/4 are mutually exclusive" >&2; exit 2; fi
if [[ "$BLOCK_CLOSURE_QUAD" == 1 && "$RANK_STATE_ILP4" != 1 ]]; then echo "BLOCK_CLOSURE_QUAD=1 requires RANK_STATE_ILP4=1" >&2; exit 2; fi
if [[ "$BLOCK_CLOSURE_WARP" == 1 && "$RANK_STATE_ILP4" != 1 ]]; then echo "BLOCK_CLOSURE_WARP=1 requires RANK_STATE_ILP4=1" >&2; exit 2; fi
if (( BLOCK_CLOSURE_QUAD + BLOCK_CLOSURE_WARP > 1 )); then echo "BLOCK_CLOSURE_QUAD and BLOCK_CLOSURE_WARP are separate experiments" >&2; exit 2; fi
if [[ "$HOT_DELTA_TABLE" == 1 && "$RANK_DELTA_CACHE" != 1 ]]; then echo "HOT_DELTA_TABLE=1 requires RANK_DELTA_CACHE=1" >&2; exit 2; fi

BIN_SUFFIX=""
[[ "$FAST_SHARD_ADDRESS8" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_shardaddr8"
[[ "$MAIN_MATE_CACHE" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_matecache"
[[ "$MAIN_PULL" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_mainpull"
[[ "$BLOCK_PULL" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_blockpull"
[[ "$BLOCK_MATE_CACHE" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_blockmate"
[[ "$MAIN_PULL_ILP2" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_ilp2"
[[ "$HEIGHT_CACHE" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_height"
[[ "$RANK_DELTA_CACHE" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_rankdelta"
[[ "$RANK_STATE_PACKED" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_packed56h"
[[ "$RANK_STATE_ILP2" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_rsilp2"
[[ "$RANK_STATE_ILP3" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_rsilp3"
[[ "$RANK_STATE_ILP4" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_rsilp4"
[[ "$BLOCK_CLOSURE_QUAD" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_closureq"
[[ "$BLOCK_CLOSURE_WARP" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_closurewarp"
[[ "$HOT_DELTA_TABLE" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_hotd32"
[[ "$CONCURRENT_GROUP_IO" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_cio"
(( MAXRREGCOUNT > 0 )) && BIN_SUFFIX="${BIN_SUFFIX}_r${MAXRREGCOUNT}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_n${N}${BIN_SUFFIX}}"

if (( MOD < 2 || MOD > 4294967295 )); then echo "HBM32 requires 2 <= modulus <= 4294967295; got $MOD" >&2; exit 2; fi
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if (( visible < NGPU )); then echo "requested $NGPU GPUs, but only $visible are visible" >&2; exit 2; fi
if [[ "$FAST_SHARD_ADDRESS8" == 1 && "$NGPU" != 8 ]]; then echo "FAST_SHARD_ADDRESS8=1 is specialized for NGPU=8" >&2; exit 2; fi

PROVENANCE="${BIN}.provenance.json"
needs_build=0
if [[ ! -x "$BIN" ]]; then
  needs_build=1
elif [[ ! -f "$PROVENANCE" ]] || ! python3 "$ONEESAN_ROOT/scripts/solve/build_provenance.py" verify     "$PROVENANCE" --binary "$BIN" --root "$ONEESAN_ROOT" --verify-sources     --expect-compile-arg="-DTARGET_W=$((N + 1))" >/dev/null 2>&1; then
  if (( CUSTOM_BIN )); then
    echo "custom BIN has missing/stale build provenance: $BIN" >&2
    exit 2
  fi
  needs_build=1
fi
if (( needs_build )); then
  echo "$BIN missing or stale; building batch binary for n=$N" >&2
  N="$N" OUT="$BIN" OWNERFUSED="$OWNERFUSED" BLOCKFUSED="$BLOCKFUSED" GROUPBATCH="$GROUPBATCH" ROW7_TENSOR="$ROW7_TENSOR" ROW8_TENSOR="$ROW8_TENSOR"     "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh"
fi

nvidia-smi -L
nvidia-smi topo -m || true
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

echo "N=$N MOD=$MOD GPUs=$NGPU rows=$ROWS requested_scratch=${TARGET_MIB}MiB planner_target_cap=${GRIDFP_PLAN_TARGET_MIB}MiB planner_divisor=$GRIDFP_PLAN_TARGET_DIVISOR reserve=${GRIDFP_VRAM_RESERVE_MIB}MiB max_window=$MAX_WINDOW fast_shard_address8=$FAST_SHARD_ADDRESS8 main_mate_cache=$MAIN_MATE_CACHE main_pull=$MAIN_PULL block_pull=$BLOCK_PULL block_mate_cache=$BLOCK_MATE_CACHE main_pull_ilp2=$MAIN_PULL_ILP2 height_cache=$HEIGHT_CACHE rank_delta_cache=$RANK_DELTA_CACHE rank_state_packed=$RANK_STATE_PACKED rank_state_ilp2=$RANK_STATE_ILP2 rank_state_ilp3=$RANK_STATE_ILP3 rank_state_ilp4=$RANK_STATE_ILP4 block_closure_quad=$BLOCK_CLOSURE_QUAD block_closure_warp=$BLOCK_CLOSURE_WARP hot_delta_table=$HOT_DELTA_TABLE concurrent_group_io=$CONCURRENT_GROUP_IO maxrregcount=$MAXRREGCOUNT GRIDFP_THREADS=${GRIDFP_THREADS:-256}"
echo "BIN=$BIN"
export GRIDFP_VRAM_RESERVE_MIB
if (( GROUPBATCH )); then
  export GRIDFP_GROUPBATCH_DYNAMIC GRIDFP_GROUPBATCH_AFFINITY GRIDFP_GROUPBATCH_STICKY GRIDFP_GROUPBATCH_RECLAIM GRIDFP_GROUPBATCH_GRAPH GRIDFP_GROUPBATCH_P1_DEST GRIDFP_GROUPBATCH_WRAP32 GRIDFP_GROUPBATCH_MATCHLUT GRIDFP_GROUPBATCH_BATCHES_PER_GPU
  echo "GROUPBATCH dynamic=$GRIDFP_GROUPBATCH_DYNAMIC affinity=$GRIDFP_GROUPBATCH_AFFINITY sticky=$GRIDFP_GROUPBATCH_STICKY reclaim=$GRIDFP_GROUPBATCH_RECLAIM graph=$GRIDFP_GROUPBATCH_GRAPH p1_dest=$GRIDFP_GROUPBATCH_P1_DEST wrap32=$GRIDFP_GROUPBATCH_WRAP32 matchlut=$GRIDFP_GROUPBATCH_MATCHLUT batches_per_gpu=$GRIDFP_GROUPBATCH_BATCHES_PER_GPU" >&2
fi
if (( ROW7_TENSOR )); then
  export GRIDFP_BOUNDED_PREFIX_K=7
  export GRIDFP_DIRECT_ROW7_TENSOR=1
fi
if (( ROW8_TENSOR )); then
  export GRIDFP_BOUNDED_PREFIX_K=8
  export GRIDFP_DIRECT_ROW8_TENSOR=1
  if (( ROW8_STRUCTURAL )); then export GRIDFP_ROW8_STRUCTURAL=1; fi
fi
exec "$BIN" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD"
