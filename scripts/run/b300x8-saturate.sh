#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
MOD="${2:-4294967291}"
TARGET_MIB="${TARGET_MIB:-65536}"
GRIDFP_PLAN_TARGET_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
GRIDFP_PLAN_TARGET_DIVISOR="${GRIDFP_PLAN_TARGET_DIVISOR:-1}"
MAX_WINDOW="${MAX_WINDOW:-14}"
ROWS="${ROWS:-$((N+1))}"
GRIDFP_THREADS="${GRIDFP_THREADS:-256}"
MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
REBUILD="${REBUILD:-0}"

# Throughput profile for B300 x8.  Convert serial rank walks into packed HBM
# state, hold four destinations per thread to expose memory-level parallelism,
# and handle closure gathers with one warp issuing up to 32 random HBM loads in
# parallel.  This intentionally spends HBM traffic to remove integer dependency
# chains: the target is higher memory-controller occupancy, not fewer bytes.
export TARGET_MIB GRIDFP_PLAN_TARGET_MIB GRIDFP_PLAN_TARGET_DIVISOR MAX_WINDOW ROWS GRIDFP_THREADS MAXRREGCOUNT REBUILD
export FAST_SHARD_ADDRESS8=1
export MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1
export MAIN_PULL_ILP2=0 HEIGHT_CACHE=0
export RANK_DELTA_CACHE=1 RANK_STATE_PACKED=1
export RANK_STATE_ILP2=0 RANK_STATE_ILP3=0 RANK_STATE_ILP4=1
export BLOCK_CLOSURE_QUAD=0 BLOCK_CLOSURE_WARP=1
export HOT_DELTA_TABLE=0
export CONCURRENT_GROUP_IO=1

exec "$ONEESAN_ROOT/scripts/run/b300x8.sh" "$N" "$MOD"
