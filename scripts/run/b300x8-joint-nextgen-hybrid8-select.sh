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
HYBRID_MANIFEST="${HYBRID_MANIFEST:-${HYBRID_WINNER_ENV%.env}_promotion-inputs.sha256}"
NEXTSELF_PREFIX="${NEXTSELF_PREFIX:-${PREFIX}.nextself}"
NEXTSELF_WINNER_ENV="${NEXTSELF_WINNER_ENV:-${NEXTSELF_PREFIX}_winner.env}"
NEXTSELF_MANIFEST="${NEXTSELF_MANIFEST:-${NEXTSELF_WINNER_ENV%.env}_promotion-inputs.sha256}"
NEXTSELF_PREPARE_PREFIX="${NEXTSELF_PREPARE_PREFIX:-${PREFIX}.nextself-promotion}"
NEXTSELF_PREPARE_ENV="${NEXTSELF_PREPARE_ENV:-${PREFIX}.nextself.prepared.env}"
RACE_PREFIX="${RACE_PREFIX:-${PREFIX}.race}"
SELECT_ONLY="${SELECT_ONLY:-1}"
REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
RUN_HYBRID_STAGE="${RUN_HYBRID_STAGE:-1}"
RUN_NEXTSELF_STAGE="${RUN_NEXTSELF_STAGE:-1}"
NEXTSELF_THREADS="${NEXTSELF_THREADS:-256}"
NEXTSELF_SEARCH_ROWS="${NEXTSELF_SEARCH_ROWS:-1}"
NEXTSELF_VALIDATE_ROWS="${NEXTSELF_VALIDATE_ROWS:-4 8}"
NEXTSELF_SEARCH_REPEATS="${NEXTSELF_SEARCH_REPEATS:-1}"
NEXTSELF_VALIDATE_REPEATS="${NEXTSELF_VALIDATE_REPEATS:-1}"
NEXTSELF_MIN_SPEEDUP="${NEXTSELF_MIN_SPEEDUP:-1.01}"
PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"

for x in SELECT_ONLY REBUILD_BUCKETS RUN_HYBRID_STAGE RUN_NEXTSELF_STAGE; do
  v="${!x}"
  [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
[[ "$NEXTSELF_THREADS" =~ ^[0-9]+$ ]] && (( NEXTSELF_THREADS >= 32 && NEXTSELF_THREADS <= 768 && NEXTSELF_THREADS % 32 == 0 )) || {
  echo 'NEXTSELF_THREADS must be warp multiple 32..768' >&2; exit 2;
}
[[ -f "$PROFILE_FILE" ]] || { echo "missing profile: $PROFILE_FILE" >&2; exit 2; }
command -v sha256sum >/dev/null || { echo 'sha256sum required' >&2; exit 2; }
mkdir -p "$(dirname "$PREPARE_ENV")" "$(dirname "$HYBRID_WINNER_ENV")" "$(dirname "$HYBRID_MANIFEST")" \
  "$(dirname "$NEXTSELF_WINNER_ENV")" "$(dirname "$NEXTSELF_MANIFEST")" "$(dirname "$NEXTSELF_PREPARE_ENV")" "$(dirname "$RACE_PREFIX")"

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
HYBRID_VALID=0
if [[ "${B300_HYBRID8_STAGED_VALIDATED:-0}" == 1 && "${B300_HYBRID8_FINAL_ENABLED:-0}" == 1 ]]; then
  [[ -x "${B300_HYBRID8_FINAL_BIN:-}" && -x "${B300_HYBRID8_BASE_BIN:-}" ]] || { echo 'validated hybrid8 final/base binary missing' >&2; exit 3; }
  [[ "${B300_HYBRID8_FINAL_SPILL_FREE:-0}" == 1 ]] || { echo 'validated hybrid8 is not spill-free' >&2; exit 3; }
  HYBRID_VALID=1
fi

if [[ "$HYBRID_VALID" == 1 ]]; then
  if [[ "$RUN_HYBRID_STAGE" == 1 ]]; then
    tmp="${HYBRID_MANIFEST}.tmp"
    sha256sum "$HYBRID_WINNER_ENV" "$B300_HYBRID8_FINAL_BIN" "$B300_HYBRID8_BASE_BIN" >"$tmp"
    mv "$tmp" "$HYBRID_MANIFEST"
  else
    [[ -s "$HYBRID_MANIFEST" ]] || { echo "missing reused hybrid8 manifest=$HYBRID_MANIFEST; rerun with RUN_HYBRID_STAGE=1" >&2; exit 3; }
  fi
  if ! sha256sum -c "$HYBRID_MANIFEST" >/dev/null; then
    echo 'joint hybrid8 staged artifact fingerprint mismatch; rerun hybrid stage' >&2
    exit 3
  fi
fi

NEXTSELF_VALID=0
if [[ "$RUN_NEXTSELF_STAGE" == 1 ]]; then
  echo "=== joint nextgen hybrid8: next-self staged validation rows=$NEXTSELF_SEARCH_ROWS -> [$NEXTSELF_VALIDATE_ROWS] ===" >&2
  N=27 ARCH="$ARCH" GRIDFP_THREADS="$NEXTSELF_THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
    SEARCH_ROWS="$NEXTSELF_SEARCH_ROWS" VALIDATE_ROWS="$NEXTSELF_VALIDATE_ROWS" \
    SEARCH_REPEATS="$NEXTSELF_SEARCH_REPEATS" VALIDATE_REPEATS="$NEXTSELF_VALIDATE_REPEATS" MIN_SPEEDUP="$NEXTSELF_MIN_SPEEDUP" \
    PREFIX="$NEXTSELF_PREFIX" WINNER_ENV="$NEXTSELF_WINNER_ENV" \
    bash "$ONEESAN_ROOT/scripts/bench/b300x8-ilp8-nextself-staged-ab.sh"
fi
if [[ -s "$NEXTSELF_WINNER_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$NEXTSELF_WINNER_ENV"
  if [[ "${B300_NEXTSELF_STAGED_VALIDATED:-0}" == 1 ]]; then
    [[ "${B300_NEXTSELF_CONTROL_SPILL_FREE:-0}" == 1 && "${B300_NEXTSELF_SPILL_FREE:-0}" == 1 ]] || { echo 'validated next-self/control is not spill-free' >&2; exit 3; }
    [[ -x "${B300_NEXTSELF_CONTROL_BIN:-}" && -x "${B300_NEXTSELF_BIN:-}" ]] || { echo 'validated next-self/control binary missing' >&2; exit 3; }
    NEXTSELF_VALID=1
  fi
fi

if [[ "$NEXTSELF_VALID" == 1 ]]; then
  if [[ "$RUN_NEXTSELF_STAGE" == 1 ]]; then
    tmp="${NEXTSELF_MANIFEST}.tmp"
    sha256sum "$NEXTSELF_WINNER_ENV" "$B300_NEXTSELF_CONTROL_BIN" "$B300_NEXTSELF_BIN" >"$tmp"
    mv "$tmp" "$NEXTSELF_MANIFEST"
  else
    [[ -s "$NEXTSELF_MANIFEST" ]] || { echo "missing reused next-self manifest=$NEXTSELF_MANIFEST; rerun with RUN_NEXTSELF_STAGE=1" >&2; exit 3; }
  fi
  if ! sha256sum -c "$NEXTSELF_MANIFEST" >/dev/null; then
    echo 'joint next-self staged artifact fingerprint mismatch; rerun next-self stage' >&2
    exit 3
  fi

  PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
    GRIDFP_THREADS="$NEXTSELF_THREADS" RUN_STAGED=0 PREPARE_ONLY=1 WINNER_ENV="$NEXTSELF_WINNER_ENV" \
    RACE_PREFIX="$NEXTSELF_PREPARE_PREFIX" PREPARE_ENV="$NEXTSELF_PREPARE_ENV" \
    bash "$ONEESAN_ROOT/scripts/run/b300x8-ilp8-nextself-staged-fullprime-race.sh" 27
  [[ -s "$NEXTSELF_PREPARE_ENV" ]] || { echo "next-self prepare env missing: $NEXTSELF_PREPARE_ENV" >&2; exit 3; }
  # shellcheck disable=SC1090
  source "$NEXTSELF_PREPARE_ENV"
  [[ "${B300_NEXTSELF_PREPARED:-0}" == 1 ]] || { echo 'next-self prepared marker missing' >&2; exit 3; }
  [[ -x "$B300_NEXTSELF_PREPARED_BIN" && -x "$B300_NEXTSELF_PREPARED_CONTROL_BIN" ]] || { echo 'next-self prepared adapter missing' >&2; exit 3; }
  NEXTSELF_RACE_BIN="$B300_NEXTSELF_PREPARED_BIN"
  NEXTSELF_RACE_LABEL="$B300_NEXTSELF_PREPARED_LABEL"
  NEXTSELF_RACE_THREADS="$B300_NEXTSELF_PREPARED_THREADS"
  NEXTSELF_CONTROL_RACE_BIN="$B300_NEXTSELF_PREPARED_CONTROL_BIN"
  NEXTSELF_CONTROL_RACE_LABEL="$B300_NEXTSELF_PREPARED_CONTROL_LABEL"
  NEXTSELF_CONTROL_RACE_THREADS="$B300_NEXTSELF_PREPARED_CONTROL_THREADS"
fi

echo "JOINT STAGED SUMMARY hybrid8_valid=$HYBRID_VALID nextself_valid=$NEXTSELF_VALID" >&2

if [[ "$HYBRID_VALID" == 1 ]]; then
  echo "=== joint nextgen hybrid8: promote threshold=${B300_HYBRID8_FINAL_THRESHOLD:-NA} into unified full-prime race ===" >&2
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
  if [[ "$NEXTSELF_VALID" == 1 ]]; then
    export FORCED_EXTRA3_BIN="$NEXTSELF_RACE_BIN"
    export FORCED_EXTRA3_LABEL="$NEXTSELF_RACE_LABEL"
    export FORCED_EXTRA3_THREADS="$NEXTSELF_RACE_THREADS"
  else
    unset FORCED_EXTRA3_BIN FORCED_EXTRA3_LABEL FORCED_EXTRA3_THREADS || true
  fi

  exec env PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" SMOKE_PRIME="$PRIME" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
    RUN_STAGED=0 STAGED_PREFIX="$HYBRID_PREFIX" WINNER_ENV="$HYBRID_WINNER_ENV" MANIFEST="$HYBRID_MANIFEST" \
    SELECT_ONLY="$SELECT_ONLY" REBUILD_BUCKETS="$REBUILD_BUCKETS" RACE_PREFIX="$RACE_PREFIX" \
    "$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-staged-fullprime-race.sh" 27 "$@"
fi

if [[ "$NEXTSELF_VALID" == 1 ]]; then
  echo '=== joint nextgen hybrid8: hybrid8 rejected; promote prepared next-self into unified full-prime race ===' >&2
  unset FORCED_EXTRA3_BIN FORCED_EXTRA3_LABEL FORCED_EXTRA3_THREADS || true
  exec env PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" SMOKE_PRIME="$PRIME" FORCED_TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
    FORCED_OVERRIDE_BIN="$NEXTSELF_RACE_BIN" FORCED_OVERRIDE_LABEL="$NEXTSELF_RACE_LABEL" FORCED_OVERRIDE_THREADS="$NEXTSELF_RACE_THREADS" \
    FORCED_BASE_BIN="$NEXTSELF_CONTROL_RACE_BIN" FORCED_BASE_LABEL="$NEXTSELF_CONTROL_RACE_LABEL" FORCED_BASE_THREADS="$NEXTSELF_CONTROL_RACE_THREADS" \
    FORCED_EXTRA_BIN="$JOINT_PRIMARY_BIN" FORCED_EXTRA_LABEL="$JOINT_PRIMARY_LABEL" FORCED_EXTRA_THREADS="$JOINT_PRIMARY_THREADS" \
    FORCED_EXTRA2_BIN="$JOINT_BASE_BIN" FORCED_EXTRA2_LABEL="$JOINT_BASE_LABEL" FORCED_EXTRA2_THREADS="$JOINT_BASE_THREADS" \
    SELECT_ONLY="$SELECT_ONLY" REBUILD_BUCKETS="$REBUILD_BUCKETS" PREFIX="$RACE_PREFIX" \
    "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"
fi

echo '=== joint nextgen hybrid8: staged transforms rejected; run calibrated joint/bucket race ===' >&2
unset FORCED_EXTRA_BIN FORCED_EXTRA_LABEL FORCED_EXTRA_THREADS FORCED_EXTRA2_BIN FORCED_EXTRA2_LABEL FORCED_EXTRA2_THREADS FORCED_EXTRA3_BIN FORCED_EXTRA3_LABEL FORCED_EXTRA3_THREADS || true
exec env PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" SMOKE_PRIME="$PRIME" FORCED_TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
  FORCED_OVERRIDE_BIN="$JOINT_PRIMARY_BIN" FORCED_OVERRIDE_LABEL="$JOINT_PRIMARY_LABEL" FORCED_OVERRIDE_THREADS="$JOINT_PRIMARY_THREADS" \
  FORCED_BASE_BIN="$JOINT_BASE_BIN" FORCED_BASE_LABEL="$JOINT_BASE_LABEL" FORCED_BASE_THREADS="$JOINT_BASE_THREADS" \
  SELECT_ONLY="$SELECT_ONLY" REBUILD_BUCKETS="$REBUILD_BUCKETS" PREFIX="$RACE_PREFIX" \
  "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"
