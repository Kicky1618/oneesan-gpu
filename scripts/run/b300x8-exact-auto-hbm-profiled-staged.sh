#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo 'profiled staged selector targets n=27' >&2; exit 2; }
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_tune21.env}"
FORCED_PROFILE_FILE="${FORCED_PROFILE_FILE:-$ONEESAN_ROOT/work/b300_forced_profile_tune27.env}"
TUNE_FORCED="${TUNE_FORCED:-1}"; REBUILD_FORCED="${REBUILD_FORCED:-${REBUILD:-1}}"; REBUILD_BUCKETS="${REBUILD_BUCKETS:-${REBUILD:-1}}"
for x in TUNE_FORCED REBUILD_FORCED REBUILD_BUCKETS; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }; done
[[ -f "$PROFILE_FILE" ]] || { echo "missing n21 profile: $PROFILE_FILE" >&2; echo 'run: bash scripts/bench/b300-hbm-profile-tune21.sh' >&2; exit 2; }

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

# Read profile names only to identify the exact bucket binaries that the final
# selector will use. If rebuilding is requested, remove those outputs so the
# final selector can keep REBUILD=0 (which protects the selected forced symlink)
# while still recompiling bucket candidates.
# shellcheck disable=SC1090
source "$PROFILE_FILE"
: "${WARP_PROFILE:?n21 profile missing WARP_PROFILE}"
: "${ORBIT_PROFILE:?n21 profile missing ORBIT_PROFILE}"
[[ "$WARP_PROFILE" =~ ^[A-Za-z0-9_.-]+$ && "$ORBIT_PROFILE" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'unsafe profile name' >&2; exit 3; }
FORCED_FIXED="$ONEESAN_BUILD_DIR/b300_profiled_forced_n27"
WARP_FIXED="$ONEESAN_BUILD_DIR/b300_profiled_warp_${WARP_PROFILE}_n27"
ORBIT_FIXED="$ONEESAN_BUILD_DIR/b300_profiled_orbit_${ORBIT_PROFILE}_n27"
if [[ "$REBUILD_BUCKETS" == 1 ]]; then rm -f "$WARP_FIXED" "$ORBIT_FIXED"; fi
ln -sfn "$FORCED_PROFILE_BIN" "$FORCED_FIXED"

export PROFILE_FILE
export REBUILD=0
export PREFIX="${FINAL_PREFIX:-$ONEESAN_ROOT/work/b300_exact_hbm_profiled_staged_n27}"
echo "profiled staged forced=$FORCED_PROFILE bin=$FORCED_PROFILE_BIN warp=$WARP_PROFILE orbit=$ORBIT_PROFILE rebuild_buckets=$REBUILD_BUCKETS" >&2
exec "$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled.sh" 27 "$@"
