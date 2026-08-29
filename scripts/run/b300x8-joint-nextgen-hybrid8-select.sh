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
NEXTSELF_PREFIX="${NEXTSELF_PREFIX:-${PREFIX}.nextself}"
NEXTSELF_WINNER_ENV="${NEXTSELF_WINNER_ENV:-${NEXTSELF_PREFIX}_winner.env}"
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
mkdir -p "$(dirname "$PREPARE_ENV")" "$(dirname "$HYBRID_WINNER_ENV")" "$(dirname "$NEXTSELF_WINNER_ENV")" "$(dirname "$RACE_PREFIX")"

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
  HYBRID_VALID=1
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
    [[ -x "${B300_NEXTSELF_BIN:-}" ]] || { echo 'validated next-self binary missing' >&2; exit 3; }
    NEXTSELF_VALID=1
  fi
fi

echo "JOINT STAGED SUMMARY hybrid8_valid=$HYBRID_VALID nextself_valid=$NEXTSELF_VALID" >&2

# The advanced path owns primary/base. Calibrated forced candidates become
# extras. If next-self also survived staged validation, adapt its MOD-first ABI
# into the common forced ABI and keep it as the third extra candidate.
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
    mkdir -p "${RACE_PREFIX}_adapters"
    NEXTSELF_ADAPTER="${RACE_PREFIX}_adapters/nextself_${B300_NEXTSELF_VARIANT:-winner}.sh"
    NEXTSELF_SHA="$(sha256sum "$B300_NEXTSELF_BIN" | awk '{print $1}')"
    cat >"$NEXTSELF_ADAPTER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
# nextself_binary_sha256=$NEXTSELF_SHA
SRC=$(printf '%q' "$B300_NEXTSELF_BIN")
THREADS=$(printf '%q' "$NEXTSELF_THREADS")
PLAN_MIB=$(printf '%q' "$PLAN_MIB")
[[ \$# -ge 5 ]] || { echo 'usage: adapter N TARGET_MIB MAX_WINDOW NGPU MOD' >&2; exit 2; }
N=\$1; TARGET=\$2; MAXW=\$3; NGPU=\$4; MOD=\$5
export B300_ROW_LIMIT=28 GRIDFP_THREADS="\$THREADS" GRIDFP_PLAN_TARGET_MIB="\$PLAN_MIB"
exec "\$SRC" "\$N" "\$MOD" "\$TARGET" "\$MAXW" "\$NGPU"
EOF
    chmod +x "$NEXTSELF_ADAPTER"
    bash -n "$NEXTSELF_ADAPTER"
    export FORCED_EXTRA3_BIN="$NEXTSELF_ADAPTER"
    export FORCED_EXTRA3_LABEL="nextself_${B300_NEXTSELF_VARIANT:-winner}"
    export FORCED_EXTRA3_THREADS="$NEXTSELF_THREADS"
  else
    unset FORCED_EXTRA3_BIN FORCED_EXTRA3_LABEL FORCED_EXTRA3_THREADS || true
  fi

  exec env PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" SMOKE_PRIME="$PRIME" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
    RUN_STAGED=0 WINNER_ENV="$HYBRID_WINNER_ENV" SELECT_ONLY="$SELECT_ONLY" REBUILD_BUCKETS="$REBUILD_BUCKETS" \
    RACE_PREFIX="$RACE_PREFIX" \
    "$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-staged-fullprime-race.sh" 27 "$@"
fi

# If nextgen hybrid8 failed but next-self survived, let next-self own primary/base
# and keep both calibrated forced candidates as extras in the same complete-prime race.
if [[ "$NEXTSELF_VALID" == 1 ]]; then
  echo '=== joint nextgen hybrid8: hybrid8 rejected; promote next-self into unified full-prime race ===' >&2
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
  unset FORCED_EXTRA3_BIN FORCED_EXTRA3_LABEL FORCED_EXTRA3_THREADS || true
  exec env PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" SMOKE_PRIME="$PRIME" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
    GRIDFP_THREADS="$NEXTSELF_THREADS" RUN_STAGED=0 WINNER_ENV="$NEXTSELF_WINNER_ENV" \
    SELECT_ONLY="$SELECT_ONLY" REBUILD_BUCKETS="$REBUILD_BUCKETS" RACE_PREFIX="$RACE_PREFIX" \
    "$ONEESAN_ROOT/scripts/run/b300x8-ilp8-nextself-staged-fullprime-race.sh" 27 "$@"
fi

echo '=== joint nextgen hybrid8: staged transforms rejected; run calibrated joint/bucket race ===' >&2
unset FORCED_EXTRA_BIN FORCED_EXTRA_LABEL FORCED_EXTRA_THREADS FORCED_EXTRA2_BIN FORCED_EXTRA2_LABEL FORCED_EXTRA2_THREADS FORCED_EXTRA3_BIN FORCED_EXTRA3_LABEL FORCED_EXTRA3_THREADS || true
exec env PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" SMOKE_PRIME="$PRIME" FORCED_TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
  FORCED_OVERRIDE_BIN="$JOINT_PRIMARY_BIN" FORCED_OVERRIDE_LABEL="$JOINT_PRIMARY_LABEL" FORCED_OVERRIDE_THREADS="$JOINT_PRIMARY_THREADS" \
  FORCED_BASE_BIN="$JOINT_BASE_BIN" FORCED_BASE_LABEL="$JOINT_BASE_LABEL" FORCED_BASE_THREADS="$JOINT_BASE_THREADS" \
  SELECT_ONLY="$SELECT_ONLY" REBUILD_BUCKETS="$REBUILD_BUCKETS" PREFIX="$RACE_PREFIX" \
  "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"
