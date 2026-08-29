#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
MOD="${2:-4294967291}"
TARGET_MIB="${TARGET_MIB:-65536}"
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
RANK_DELTA_CACHE="${RANK_DELTA_CACHE:-0}"
# Modifier for rank-delta: same 8B/state traffic, but packs signed int32 delta
# plus uint8 height so prefix popcounts disappear too. Groups exceeding int32
# local rank automatically fall back to the exact noncached path.
RANK_STATE_PACKED="${RANK_STATE_PACKED:-0}"
if [[ -z "${GRIDFP_PLAN_TARGET_DIVISOR+x}" ]]; then
  if [[ "$RANK_DELTA_CACHE" == 1 ]]; then GRIDFP_PLAN_TARGET_DIVISOR=3; else GRIDFP_PLAN_TARGET_DIVISOR=1; fi
fi
GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}"
REBUILD="${REBUILD:-0}"

for name in FAST_SHARD_ADDRESS8 MAIN_MATE_CACHE MAIN_PULL BLOCK_PULL BLOCK_MATE_CACHE MAIN_PULL_ILP2 HEIGHT_CACHE RANK_DELTA_CACHE RANK_STATE_PACKED REBUILD; do
  value="${!name}"
  if [[ "$value" != 0 && "$value" != 1 ]]; then echo "$name must be 0 or 1" >&2; exit 2; fi
done
[[ "$ROWS" =~ ^[0-9]+$ ]] && (( ROWS >= 1 && ROWS <= N + 1 )) || { echo "ROWS must be 1..$((N+1))" >&2; exit 2; }
[[ "$GRIDFP_PLAN_TARGET_MIB" =~ ^[0-9]+$ ]] && (( GRIDFP_PLAN_TARGET_MIB >= 1 && GRIDFP_PLAN_TARGET_MIB <= TARGET_MIB )) || { echo "GRIDFP_PLAN_TARGET_MIB must be 1..TARGET_MIB" >&2; exit 2; }
[[ "$GRIDFP_PLAN_TARGET_DIVISOR" =~ ^[0-9]+$ ]] && (( GRIDFP_PLAN_TARGET_DIVISOR >= 1 && GRIDFP_PLAN_TARGET_DIVISOR <= 16 )) || { echo "GRIDFP_PLAN_TARGET_DIVISOR must be 1..16" >&2; exit 2; }
if [[ "$MAIN_PULL" == 1 && "$MAIN_MATE_CACHE" != 1 ]]; then echo "MAIN_PULL=1 requires MAIN_MATE_CACHE=1" >&2; exit 2; fi
if [[ "$BLOCK_PULL" == 1 && "$MAIN_PULL" != 1 ]]; then echo "BLOCK_PULL=1 requires MAIN_PULL=1" >&2; exit 2; fi
if [[ "$BLOCK_MATE_CACHE" == 1 && "$BLOCK_PULL" != 1 ]]; then echo "BLOCK_MATE_CACHE=1 requires BLOCK_PULL=1" >&2; exit 2; fi
if (( MAIN_PULL_ILP2 + HEIGHT_CACHE + RANK_DELTA_CACHE > 1 )); then echo "MAIN_PULL_ILP2, HEIGHT_CACHE and RANK_DELTA_CACHE are separate A/B experiments for now" >&2; exit 2; fi
for name in MAIN_PULL_ILP2 HEIGHT_CACHE RANK_DELTA_CACHE; do
  if [[ "${!name}" == 1 && ( "$MAIN_PULL" != 1 || "$BLOCK_PULL" != 1 || "$MAIN_MATE_CACHE" != 1 || "$BLOCK_MATE_CACHE" != 1 ) ]]; then echo "$name=1 requires full-pull plus both MateID caches" >&2; exit 2; fi
done
if [[ "$RANK_STATE_PACKED" == 1 && "$RANK_DELTA_CACHE" != 1 ]]; then echo "RANK_STATE_PACKED=1 requires RANK_DELTA_CACHE=1" >&2; exit 2; fi

BIN_SUFFIX=""
[[ "$FAST_SHARD_ADDRESS8" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_shardaddr8"
[[ "$MAIN_MATE_CACHE" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_matecache"
[[ "$MAIN_PULL" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_mainpull"
[[ "$BLOCK_PULL" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_blockpull"
[[ "$BLOCK_MATE_CACHE" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_blockmate"
[[ "$MAIN_PULL_ILP2" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_ilp2"
[[ "$HEIGHT_CACHE" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_height"
[[ "$RANK_DELTA_CACHE" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_rankdelta"
[[ "$RANK_STATE_PACKED" == 1 ]] && BIN_SUFFIX="${BIN_SUFFIX}_packedheight"
BIN="${BIN:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_n${N}${BIN_SUFFIX}}"

if (( MOD < 2 || MOD > 4294967295 )); then echo "HBM32 requires 2 <= modulus <= 4294967295; got $MOD" >&2; exit 2; fi
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if (( visible < NGPU )); then echo "requested $NGPU GPUs, but only $visible are visible" >&2; exit 2; fi
if [[ "$FAST_SHARD_ADDRESS8" == 1 && "$NGPU" != 8 ]]; then echo "FAST_SHARD_ADDRESS8=1 is specialized for NGPU=8" >&2; exit 2; fi

if [[ "$REBUILD" == 1 || ! -x "$BIN" ]]; then
  echo "building specialized n=$N binary shardaddr8=$FAST_SHARD_ADDRESS8 matecache=$MAIN_MATE_CACHE mainpull=$MAIN_PULL blockpull=$BLOCK_PULL blockmate=$BLOCK_MATE_CACHE ilp2=$MAIN_PULL_ILP2 height=$HEIGHT_CACHE rankdelta=$RANK_DELTA_CACHE rankstate=$RANK_STATE_PACKED" >&2
  N="$N" FAST_SHARD_ADDRESS8="$FAST_SHARD_ADDRESS8" \
  MAIN_MATE_CACHE="$MAIN_MATE_CACHE" MAIN_PULL="$MAIN_PULL" BLOCK_PULL="$BLOCK_PULL" BLOCK_MATE_CACHE="$BLOCK_MATE_CACHE" MAIN_PULL_ILP2="$MAIN_PULL_ILP2" HEIGHT_CACHE="$HEIGHT_CACHE" RANK_DELTA_CACHE="$RANK_DELTA_CACHE" RANK_STATE_PACKED="$RANK_STATE_PACKED" \
  PTXAS_VERBOSE=1 OUT="$BIN" "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh"
fi

nvidia-smi -L
nvidia-smi topo -m || true
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

echo "N=$N MOD=$MOD GPUs=$NGPU rows=$ROWS requested_scratch=${TARGET_MIB}MiB planner_target_cap=${GRIDFP_PLAN_TARGET_MIB}MiB planner_divisor=$GRIDFP_PLAN_TARGET_DIVISOR reserve=${GRIDFP_VRAM_RESERVE_MIB}MiB max_window=$MAX_WINDOW fast_shard_address8=$FAST_SHARD_ADDRESS8 main_mate_cache=$MAIN_MATE_CACHE main_pull=$MAIN_PULL block_pull=$BLOCK_PULL block_mate_cache=$BLOCK_MATE_CACHE main_pull_ilp2=$MAIN_PULL_ILP2 height_cache=$HEIGHT_CACHE rank_delta_cache=$RANK_DELTA_CACHE rank_state_packed=$RANK_STATE_PACKED GRIDFP_THREADS=${GRIDFP_THREADS:-256}"
echo "BIN=$BIN"
export GRIDFP_VRAM_RESERVE_MIB GRIDFP_PLAN_TARGET_MIB GRIDFP_PLAN_TARGET_DIVISOR
export B300_ROW_LIMIT="$ROWS"
exec "$BIN" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU"
