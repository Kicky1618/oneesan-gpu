#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo 'joint nextgen hybrid8 selector targets n=27' >&2; exit 2; }

PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
ARCH="${ARCH:-native}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_joint_nextgen_hybrid8_n27}"
JOINT_PREFIX="${JOINT_PREFIX:-${PREFIX}.joint}"
PREPARE_ENV="${PREPARE_ENV:-${PREFIX}.joint.prepared.env}"
HYBRID_PREFIX="${HYBRID_PREFIX:-${PREFIX}.hybrid8}"
HYBRID_WINNER_ENV="${HYBRID_WINNER_ENV:-${HYBRID_PREFIX}_winner.env}"
RACE_PREFIX="${RACE_PREFIX:-${PREFIX}.race}"
SELECT_ONLY="${SELECT_ONLY:-1}"
REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
RUN_HYBRID_STAGE="${RUN_HYBRID_STAGE:-1}"

for x in SELECT_ONLY REBUILD_BUCKETS RUN_HYBRID_STAGE; do
  v="${!x}"
  [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
[[ -f "$PROFILE_FILE" ]] || { echo "missing profile: $PROFILE_FILE" >&2; exit 2; }
mkdir -p "$(dirname "$PREPARE_ENV")" "$(dirname "$HYBRID_WINNER_ENV")" "$(dirname "$RACE_PREFIX")"

# Prepare calibrated forced candidates plus the n=27 producer-tuned bucket
# profile, but defer the final complete-prime race until the staged recurrence
# candidate has either passed or been rejected.
echo '=== joint nextgen hybrid8: prepare calibrated forced candidates and bucket profile ===' >&2
PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" PREFIX="$JOINT_PREFIX" PREPARE_ONLY=1 PREPARE_ENV="$PREPARE_ENV" \
  SELECT_ONLY=1 REBUILD_BUCKETS="$REBUILD_BUCKETS" \
  bash "$ONEESAN_ROOT/scripts/run/b300x8-joint-calibrated-select.sh" 27
[[ -s "$PREPARE_ENV" ]] || { echo "joint prepare env missing: $PREPARE_ENV" >&2; exit 3; }
# shellcheck disable=SC1090
source "$PREPARE_ENV"
[[ "${B300_JOINT_PREPARED:-0}" == 1 ]] || { echo 'joint prepared marker missing' >&2; exit 3; }
[[ -x "$FORCED_OVERRIDE_BIN" ]] || { echo 'prepared joint primary missing' >&2; exit 3; }
[[ -f "$PROFILE_FILE" ]] || { echo 'prepared profile missing' >&2; exit 3; }

JOINT_PRIMARY_BIN="$FORCED_OVERRIDE_BIN"
JOINT_PRIMARY_LABEL="$FORCED_OVERRIDE_LABEL"
JOINT_PRIMARY_THREADS="$FORCED_OVERRIDE_THREADS"
JOINT_BASE_BIN="${FORCED_BASE_BIN:-}"
JOINT_BASE_LABEL="${FORCED_BASE_LABEL:-}"
JOINT_BASE_THREADS="${FORCED_BASE_THREADS:-256}"
TARGET_MIB="$FORCED_TARGET_MIB"
PRIME="$SMOKE_PRIME"

if [[ "$RUN_HYBRID_STAGE" == 1 ]]; then
  echo '=== joint nextgen hybrid8: A-D + hybrid8 staged validation ===' >&2
  ARCH="$ARCH" MOD="$PRIME" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
    PREFIX="$HYBRID_PREFIX" FINAL_ENV="$HYBRID_WINNER_ENV" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-staged-calibrate.sh"
fi
[[ -s "$HYBRID_WINNER_ENV" ]] || { echo "hybrid8 winner env missing: $HYBRID_WINNER_ENV" >&2; exit 3; }
# shellcheck disable=SC1090
source "$HYBRID_WINNER_ENV"

if [[ "${B300_HYBRID8_STAGED_VALIDATED:-0}" == 1 && "${B300_HYBRID8_FINAL_ENABLED:-0}" == 1 ]]; then
  [[ -x "${B300_HYBRID8_FINAL_BIN:-}" && -x "${B300_HYBRID8_BASE_BIN:-}" ]] || { echo 'validated hybrid8 final/base binary missing' >&2; exit 3; }
  echo "=== joint nextgen hybrid8: promote threshold=${B300_HYBRID8_FINAL_THRESHOLD:-NA} into unified full-prime race ===" >&2

  # The nextgen promotion owns primary/base. Put the independently calibrated
  # forced family into the two optional slots of the same single-pass race.
  export FORCED_EXTRA_BIN="$JOINT_PRIMARY_BIN"
  export FORCED_EXTRA_LABEL="$JOINT_PRIMARY_LABEL"
  export FORCED_EXTRA_THREADS="$JOINT_PRIMARY_THREADS"
  if [[ -n "$JOINT_BASE_BIN" && "$JOINT_BASE_BIN" != "$JOINT_PRIMARY_BIN" ]]; then
    export FORCED_EXTRA2_BIN="$JOINT_BASE_BIN"
    export FORCED_EXTRA2_LABEL="$JOINT_BASE_LABEL"
    export FORCED_EXTRA2_THREADS="$JOINT_BASE_THREADS"
  else
    unset FORCED_EXTRA2_BIN FORCED_EXTRA2_LABEL FORCED_EXTRA2_THREADS || true
  fi

  exec env PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" SMOKE_PRIME="$PRIME" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
    RUN_STAGED=0 WINNER_ENV="$HYBRID_WINNER_ENV" SELECT_ONLY="$SELECT_ONLY" REBUILD_BUCKETS="$REBUILD_BUCKETS" \
    RACE_PREFIX="$RACE_PREFIX" \
    "$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-staged-fullprime-race.sh" 27 "$@"
fi

# Search/validation miss is a normal fallback, not a fatal optimization error.
echo '=== joint nextgen hybrid8: staged transform rejected; run calibrated joint/bucket race ===' >&2
unset FORCED_EXTRA_BIN FORCED_EXTRA_LABEL FORCED_EXTRA_THREADS FORCED_EXTRA2_BIN FORCED_EXTRA2_LABEL FORCED_EXTRA2_THREADS || true
exec env PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" SMOKE_PRIME="$PRIME" FORCED_TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
  FORCED_OVERRIDE_BIN="$JOINT_PRIMARY_BIN" FORCED_OVERRIDE_LABEL="$JOINT_PRIMARY_LABEL" FORCED_OVERRIDE_THREADS="$JOINT_PRIMARY_THREADS" \
  FORCED_BASE_BIN="$JOINT_BASE_BIN" FORCED_BASE_LABEL="$JOINT_BASE_LABEL" FORCED_BASE_THREADS="$JOINT_BASE_THREADS" \
  SELECT_ONLY="$SELECT_ONLY" REBUILD_BUCKETS="$REBUILD_BUCKETS" PREFIX="$RACE_PREFIX" \
  "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"
