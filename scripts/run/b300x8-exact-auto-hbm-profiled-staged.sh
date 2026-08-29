#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo 'profiled staged selector targets n=27' >&2; exit 2; }
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
FORCED_PROFILE_FILE="${FORCED_PROFILE_FILE:-$ONEESAN_ROOT/work/b300_forced_profile_tune27.env}"
TUNE_FORCED="${TUNE_FORCED:-1}"; REBUILD_FORCED="${REBUILD_FORCED:-${REBUILD:-1}}"; REBUILD_BUCKETS="${REBUILD_BUCKETS:-${REBUILD:-1}}"
for x in TUNE_FORCED REBUILD_FORCED REBUILD_BUCKETS; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }; done
[[ -f "$PROFILE_FILE" ]] || { echo "missing n21 profile: $PROFILE_FILE" >&2; echo 'run: bash scripts/bench/b300-hbm-profile-auto21.sh' >&2; exit 2; }

if [[ "$TUNE_FORCED" == 1 || ! -f "$FORCED_PROFILE_FILE" ]]; then
  echo "=== tune forced n27 partial rows=${FORCED_PRESELECT_ROWS:-1} ===" >&2
  PROFILE_OUT="$FORCED_PROFILE_FILE" PREFIX="${FORCED_TUNE_PREFIX:-$ONEESAN_ROOT/work/b300_forced_profile_tune27}" REBUILD="$REBUILD_FORCED" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-forced-profile-tune27.sh"
fi
# shellcheck disable=SC1090
source "$FORCED_PROFILE_FILE"
: "${FORCED_PROFILE:?forced profile file missing FORCED_PROFILE}"
: "${FORCED_PROFILE_BIN:?forced profile file missing FORCED_PROFILE_BIN}"
[[ -x "$FORCED_PROFILE_BIN" ]] || { echo "selected forced binary missing: $FORCED_PROFILE_BIN" >&2; exit 3; }

# Read the n21 profile to identify exactly the bucket binaries that the final
# selector will use. psm=0 means occupancy-derived forward/reverse pools.
# shellcheck disable=SC1090
source "$PROFILE_FILE"
: "${WARP_PROFILE:?n21 profile missing WARP_PROFILE}"
: "${ORBIT_PROFILE:?n21 profile missing ORBIT_PROFILE}"
ORBIT_PRECTX_COMPACT="${ORBIT_PRECTX_COMPACT:-0}"
ORBIT_PRECTX_FLAT_BID="${ORBIT_PRECTX_FLAT_BID:-0}"
ORBIT_PRECTX_FLAT_BID_FUSED="${ORBIT_PRECTX_FLAT_BID_FUSED:-0}"
ORBITCTA_FLAT="${ORBITCTA_FLAT:-0}"
ORBITCTA_FLAT_CHUNK="${ORBITCTA_FLAT_CHUNK:-1}"
ORBITCTA_FLAT_BLOCKS_PER_SM="${ORBITCTA_FLAT_BLOCKS_PER_SM:-0}"
ORBIT_QUAD_MLP="${ORBIT_QUAD_MLP:-0}"
ORBIT_QUAD_OVERLAP_LOCAL="${ORBIT_QUAD_OVERLAP_LOCAL:-0}"
ORBIT_QUAD_LOCAL_DIRECT_MAX="${ORBIT_QUAD_LOCAL_DIRECT_MAX:-0}"
ORBIT_QUAD_SPARSE_DESC_MLP="${ORBIT_QUAD_SPARSE_DESC_MLP:-0}"
ORBIT_PRECTX_FORWARD="${ORBIT_PRECTX_FORWARD:-0}"
ORBIT_PRECTX_REVERSE="${ORBIT_PRECTX_REVERSE:-0}"
[[ "$WARP_PROFILE" =~ ^[A-Za-z0-9_.-]+$ && "$ORBIT_PROFILE" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'unsafe profile name' >&2; exit 3; }
for x in ORBIT_PRECTX_COMPACT ORBIT_PRECTX_FLAT_BID ORBIT_PRECTX_FLAT_BID_FUSED ORBITCTA_FLAT ORBIT_QUAD_MLP ORBIT_QUAD_OVERLAP_LOCAL ORBIT_QUAD_SPARSE_DESC_MLP ORBIT_PRECTX_FORWARD ORBIT_PRECTX_REVERSE; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "bad $x in n21 profile" >&2; exit 3; }
done
case "$ORBITCTA_FLAT_CHUNK" in 1|2|4|8|16|32) ;; *) echo 'bad ORBITCTA_FLAT_CHUNK in n21 profile' >&2; exit 3;; esac
[[ "$ORBITCTA_FLAT_BLOCKS_PER_SM" =~ ^[0-9]+$ ]] || { echo 'bad ORBITCTA_FLAT_BLOCKS_PER_SM in n21 profile' >&2; exit 3; }
[[ "$ORBIT_QUAD_LOCAL_DIRECT_MAX" =~ ^[0-9]+$ ]] && (( ORBIT_QUAD_LOCAL_DIRECT_MAX <= 8 )) || { echo 'bad ORBIT_QUAD_LOCAL_DIRECT_MAX in n21 profile' >&2; exit 3; }
[[ "$ORBITCTA_FLAT" == 1 || "$ORBITCTA_FLAT_CHUNK" == 1 ]] || { echo 'chunked n21 profile requires flat orbit CTA' >&2; exit 3; }
if [[ "$ORBIT_PRECTX_FLAT_BID" == 1 ]]; then
  [[ "$ORBITCTA_FLAT" == 1 && "$ORBITCTA_FLAT_CHUNK" == 1 ]] || { echo 'flat-bid n21 profile requires flat chunk=1' >&2; exit 3; }
  [[ "$ORBIT_PRECTX_COMPACT" == 1 && "$ORBIT_PRECTX_FORWARD" == 1 && "$ORBIT_PRECTX_REVERSE" == 1 ]] || { echo 'flat-bid n21 profile requires compact forward+reverse prectx' >&2; exit 3; }
  [[ "$ORBIT_QUAD_MLP" == 0 ]] || { echo 'flat-bid and quad n21 profile are mutually exclusive' >&2; exit 3; }
else
  [[ "$ORBIT_PRECTX_FLAT_BID_FUSED" == 0 ]] || { echo 'flat-bid-fused requires flat-bid' >&2; exit 3; }
fi
if [[ "$ORBIT_QUAD_MLP" == 1 ]]; then
  [[ "$ORBITCTA_FLAT" == 1 ]] && (( ORBITCTA_FLAT_CHUNK > 1 )) || { echo 'quad n21 profile requires chunked flat orbit CTA' >&2; exit 3; }
  [[ "$ORBIT_PRECTX_FLAT_BID" == 0 && "$ORBIT_PRECTX_FLAT_BID_FUSED" == 0 ]] || { echo 'quad and flat-bid n21 profile are mutually exclusive' >&2; exit 3; }
else
  [[ "$ORBIT_QUAD_OVERLAP_LOCAL" == 0 ]] && (( ORBIT_QUAD_LOCAL_DIRECT_MAX == 0 )) && [[ "$ORBIT_QUAD_SPARSE_DESC_MLP" == 0 ]] || { echo 'quad suboptions require ORBIT_QUAD_MLP=1' >&2; exit 3; }
fi
if [[ "$ORBIT_QUAD_SPARSE_DESC_MLP" == 1 ]]; then
  [[ "$ORBIT_QUAD_OVERLAP_LOCAL" == 1 ]] || { echo 'quad sparse descriptor MLP requires overlap-local' >&2; exit 3; }
fi

FORCED_FIXED="$ONEESAN_BUILD_DIR/b300_profiled_forced_n27"
WARP_FIXED="$ONEESAN_BUILD_DIR/b300_profiled_warp_${WARP_PROFILE}_n27"
ORBIT_FIXED="$ONEESAN_BUILD_DIR/b300_profiled_orbit_${ORBIT_PROFILE}_flat${ORBITCTA_FLAT}_chunk${ORBITCTA_FLAT_CHUNK}_psm${ORBITCTA_FLAT_BLOCKS_PER_SM}_bid${ORBIT_PRECTX_FLAT_BID}_bf${ORBIT_PRECTX_FLAT_BID_FUSED}_quad${ORBIT_QUAD_MLP}_qol${ORBIT_QUAD_OVERLAP_LOCAL}_qld${ORBIT_QUAD_LOCAL_DIRECT_MAX}_qsd${ORBIT_QUAD_SPARSE_DESC_MLP}_n27"
if [[ "$REBUILD_BUCKETS" == 1 ]]; then rm -f "$WARP_FIXED" "$ORBIT_FIXED"; fi
ln -sfn "$FORCED_PROFILE_BIN" "$FORCED_FIXED"

export PROFILE_FILE
export REBUILD=0
export PREFIX="${FINAL_PREFIX:-$ONEESAN_ROOT/work/b300_exact_hbm_profiled_staged_n27}"
echo "profiled staged forced=$FORCED_PROFILE bin=$FORCED_PROFILE_BIN warp=$WARP_PROFILE orbit=$ORBIT_PROFILE flat=$ORBITCTA_FLAT chunk=$ORBITCTA_FLAT_CHUNK flat_blocks_per_sm=$ORBITCTA_FLAT_BLOCKS_PER_SM pool_mode=$([[ "$ORBITCTA_FLAT_BLOCKS_PER_SM" == 0 ]] && echo occupancy || echo per_sm) bid=$ORBIT_PRECTX_FLAT_BID bid_fused=$ORBIT_PRECTX_FLAT_BID_FUSED quad=$ORBIT_QUAD_MLP quad_overlap_local=$ORBIT_QUAD_OVERLAP_LOCAL quad_local_direct_max=$ORBIT_QUAD_LOCAL_DIRECT_MAX quad_sparse_desc_mlp=$ORBIT_QUAD_SPARSE_DESC_MLP rebuild_buckets=$REBUILD_BUCKETS" >&2
exec "$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled.sh" 27 "$@"
