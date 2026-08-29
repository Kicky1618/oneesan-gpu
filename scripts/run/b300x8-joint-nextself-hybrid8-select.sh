#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo 'joint next-self+hybrid8 selector targets n=27' >&2; exit 2; }

PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
ARCH="${ARCH:-native}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_joint_nextself_hybrid8_n27}"
JOINT_PREFIX="${JOINT_PREFIX:-${PREFIX}.joint}"
JOINT_PREPARE_ENV="${JOINT_PREPARE_ENV:-${PREFIX}.joint.prepared.env}"
NEXTSELF_PREFIX="${NEXTSELF_PREFIX:-${PREFIX}.nextself}"
NEXTSELF_WINNER_ENV="${NEXTSELF_WINNER_ENV:-${NEXTSELF_PREFIX}_winner.env}"
NEXTSELF_PREPARE_ENV="${NEXTSELF_PREPARE_ENV:-${PREFIX}.nextself.prepared.env}"
NEXTSELF_RACE_PREFIX="${NEXTSELF_RACE_PREFIX:-${PREFIX}.nextself.promote}"
HYBRID_PREFIX="${HYBRID_PREFIX:-${PREFIX}.hybrid8}"
HYBRID_WINNER_ENV="${HYBRID_WINNER_ENV:-${HYBRID_PREFIX}_winner.env}"
HYBRID_PREPARE_ENV="${HYBRID_PREPARE_ENV:-${PREFIX}.hybrid8.prepared.env}"
HYBRID_RACE_PREFIX="${HYBRID_RACE_PREFIX:-${PREFIX}.hybrid8.promote}"
RACE_PREFIX="${RACE_PREFIX:-${PREFIX}.race}"
SELECT_ONLY="${SELECT_ONLY:-1}"
REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
RUN_NEXTSELF_STAGE="${RUN_NEXTSELF_STAGE:-1}"
RUN_HYBRID_STAGE="${RUN_HYBRID_STAGE:-1}"
NEXTSELF_THREADS="${NEXTSELF_THREADS:-256}"
NEXTSELF_MIN_SPEEDUP="${NEXTSELF_MIN_SPEEDUP:-1.01}"
NEXTSELF_SEARCH_ROWS="${NEXTSELF_SEARCH_ROWS:-1}"
NEXTSELF_VALIDATE_ROWS="${NEXTSELF_VALIDATE_ROWS:-4 8}"
NEXTSELF_SEARCH_REPEATS="${NEXTSELF_SEARCH_REPEATS:-1}"
NEXTSELF_VALIDATE_REPEATS="${NEXTSELF_VALIDATE_REPEATS:-1}"
HYBRID_MIN_SPEEDUP="${HYBRID_MIN_SPEEDUP:-1.01}"

for x in SELECT_ONLY REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE; do
  v="${!x}"
  [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
[[ "$NEXTSELF_THREADS" =~ ^[0-9]+$ ]] && (( NEXTSELF_THREADS >= 32 && NEXTSELF_THREADS <= 768 && NEXTSELF_THREADS % 32 == 0 )) || {
  echo 'NEXTSELF_THREADS must be warp multiple 32..768' >&2; exit 2;
}
[[ -f "$PROFILE_FILE" ]] || { echo "missing profile: $PROFILE_FILE" >&2; exit 2; }
mkdir -p \
  "$(dirname "$JOINT_PREPARE_ENV")" \
  "$(dirname "$NEXTSELF_PREPARE_ENV")" \
  "$(dirname "$HYBRID_PREPARE_ENV")" \
  "$(dirname "$RACE_PREFIX")"

echo '=== grand selector: prepare calibrated joint forced candidates and profiled buckets ===' >&2
PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" PREFIX="$JOINT_PREFIX" PREPARE_ONLY=1 PREPARE_ENV="$JOINT_PREPARE_ENV" \
  SELECT_ONLY=1 REBUILD_BUCKETS="$REBUILD_BUCKETS" \
  bash "$ONEESAN_ROOT/scripts/run/b300x8-joint-calibrated-select.sh" 27
[[ -s "$JOINT_PREPARE_ENV" ]] || { echo "joint prepare env missing: $JOINT_PREPARE_ENV" >&2; exit 3; }
# shellcheck disable=SC1090
source "$JOINT_PREPARE_ENV"
[[ "${B300_JOINT_PREPARED:-0}" == 1 ]] || { echo 'joint prepared marker missing' >&2; exit 3; }
[[ -x "$FORCED_OVERRIDE_BIN" ]] || { echo 'prepared joint primary missing' >&2; exit 3; }

JOINT_PRIMARY_BIN="$FORCED_OVERRIDE_BIN"
JOINT_PRIMARY_LABEL="$FORCED_OVERRIDE_LABEL"
JOINT_PRIMARY_THREADS="$FORCED_OVERRIDE_THREADS"
JOINT_BASE_BIN="${FORCED_BASE_BIN:-}"
JOINT_BASE_LABEL="${FORCED_BASE_LABEL:-}"
JOINT_BASE_THREADS="${FORCED_BASE_THREADS:-256}"
TARGET_MIB="$FORCED_TARGET_MIB"
PRIME="$SMOKE_PRIME"

NEXTSELF_OK=0
NEXTSELF_RC=0
echo '=== grand selector: stage/prepare next-self ===' >&2
set +e
MOD="$PRIME" PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
  GRIDFP_THREADS="$NEXTSELF_THREADS" RUN_STAGED="$RUN_NEXTSELF_STAGE" PREPARE_ONLY=1 \
  SEARCH_ROWS="$NEXTSELF_SEARCH_ROWS" VALIDATE_ROWS="$NEXTSELF_VALIDATE_ROWS" \
  SEARCH_REPEATS="$NEXTSELF_SEARCH_REPEATS" VALIDATE_REPEATS="$NEXTSELF_VALIDATE_REPEATS" MIN_SPEEDUP="$NEXTSELF_MIN_SPEEDUP" \
  STAGED_PREFIX="$NEXTSELF_PREFIX" WINNER_ENV="$NEXTSELF_WINNER_ENV" RACE_PREFIX="$NEXTSELF_RACE_PREFIX" PREPARE_ENV="$NEXTSELF_PREPARE_ENV" \
  bash "$ONEESAN_ROOT/scripts/run/b300x8-ilp8-nextself-staged-fullprime-race.sh" 27
NEXTSELF_RC=$?
set -e
if (( NEXTSELF_RC == 0 )); then
  [[ -s "$NEXTSELF_PREPARE_ENV" ]] || { echo 'next-self prepare env missing after success' >&2; exit 3; }
  # shellcheck disable=SC1090
  source "$NEXTSELF_PREPARE_ENV"
  [[ "${B300_NEXTSELF_PREPARED:-0}" == 1 ]] || { echo 'next-self prepared marker missing' >&2; exit 3; }
  [[ -x "$B300_NEXTSELF_PREPARED_BIN" && -x "$B300_NEXTSELF_PREPARED_CONTROL_BIN" ]] || { echo 'next-self prepared binaries missing' >&2; exit 3; }
  NEXTSELF_OK=1
elif (( NEXTSELF_RC == 4 )); then
  echo 'grand selector: next-self rejected by staged gates' >&2
else
  echo "grand selector: next-self preparation failed rc=$NEXTSELF_RC" >&2
  exit "$NEXTSELF_RC"
fi

HYBRID_OK=0
HYBRID_RC=0
echo '=== grand selector: stage/prepare nextgen hybrid8 ===' >&2
set +e
MOD="$PRIME" PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
  RUN_STAGED="$RUN_HYBRID_STAGE" PREPARE_ONLY=1 HYBRID_MIN_SPEEDUP="$HYBRID_MIN_SPEEDUP" \
  STAGED_PREFIX="$HYBRID_PREFIX" WINNER_ENV="$HYBRID_WINNER_ENV" RACE_PREFIX="$HYBRID_RACE_PREFIX" PREPARE_ENV="$HYBRID_PREPARE_ENV" \
  bash "$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-staged-fullprime-race.sh" 27
HYBRID_RC=$?
set -e
if (( HYBRID_RC == 0 )); then
  [[ -s "$HYBRID_PREPARE_ENV" ]] || { echo 'hybrid8 prepare env missing after success' >&2; exit 3; }
  # shellcheck disable=SC1090
  source "$HYBRID_PREPARE_ENV"
  [[ "${B300_HYBRID8_PREPARED:-0}" == 1 ]] || { echo 'hybrid8 prepared marker missing' >&2; exit 3; }
  [[ -x "$B300_HYBRID8_PREPARED_BIN" && -x "$B300_HYBRID8_PREPARED_BASE_BIN" ]] || { echo 'hybrid8 prepared binaries missing' >&2; exit 3; }
  HYBRID_OK=1
elif (( HYBRID_RC == 4 )); then
  echo 'grand selector: hybrid8 rejected by staged gates' >&2
else
  echo "grand selector: hybrid8 preparation failed rc=$HYBRID_RC" >&2
  exit "$HYBRID_RC"
fi

# Candidate budget of b300x8-race-external-forced-profiled-once.sh:
# primary + base + extra1 + extra2 + extra3, plus profiled warp/orbit.
# When both transforms survive, spend all five forced slots on:
# next-self, exact next-self control, hybrid8, exact A-D base, joint primary.
# The lower-priority joint fallback is intentionally omitted in that case.
P_BIN=""; P_LABEL=""; P_THREADS=256
B_BIN=""; B_LABEL=""; B_THREADS=256
E1_BIN=""; E1_LABEL=""; E1_THREADS=256
E2_BIN=""; E2_LABEL=""; E2_THREADS=256
E3_BIN=""; E3_LABEL=""; E3_THREADS=256
MODE=""

if (( NEXTSELF_OK && HYBRID_OK )); then
  MODE=nextself_hybrid8_joint
  P_BIN="$B300_NEXTSELF_PREPARED_BIN"; P_LABEL="$B300_NEXTSELF_PREPARED_LABEL"; P_THREADS="$B300_NEXTSELF_PREPARED_THREADS"
  B_BIN="$B300_NEXTSELF_PREPARED_CONTROL_BIN"; B_LABEL="$B300_NEXTSELF_PREPARED_CONTROL_LABEL"; B_THREADS="$B300_NEXTSELF_PREPARED_CONTROL_THREADS"
  E1_BIN="$B300_HYBRID8_PREPARED_BIN"; E1_LABEL="$B300_HYBRID8_PREPARED_LABEL"; E1_THREADS="$B300_HYBRID8_PREPARED_THREADS"
  E2_BIN="$B300_HYBRID8_PREPARED_BASE_BIN"; E2_LABEL="$B300_HYBRID8_PREPARED_BASE_LABEL"; E2_THREADS="$B300_HYBRID8_PREPARED_BASE_THREADS"
  E3_BIN="$JOINT_PRIMARY_BIN"; E3_LABEL="$JOINT_PRIMARY_LABEL"; E3_THREADS="$JOINT_PRIMARY_THREADS"
elif (( NEXTSELF_OK )); then
  MODE=nextself_joint
  P_BIN="$B300_NEXTSELF_PREPARED_BIN"; P_LABEL="$B300_NEXTSELF_PREPARED_LABEL"; P_THREADS="$B300_NEXTSELF_PREPARED_THREADS"
  B_BIN="$B300_NEXTSELF_PREPARED_CONTROL_BIN"; B_LABEL="$B300_NEXTSELF_PREPARED_CONTROL_LABEL"; B_THREADS="$B300_NEXTSELF_PREPARED_CONTROL_THREADS"
  E1_BIN="$JOINT_PRIMARY_BIN"; E1_LABEL="$JOINT_PRIMARY_LABEL"; E1_THREADS="$JOINT_PRIMARY_THREADS"
  if [[ -n "$JOINT_BASE_BIN" && "$JOINT_BASE_BIN" != "$JOINT_PRIMARY_BIN" ]]; then
    E2_BIN="$JOINT_BASE_BIN"; E2_LABEL="$JOINT_BASE_LABEL"; E2_THREADS="$JOINT_BASE_THREADS"
  fi
elif (( HYBRID_OK )); then
  MODE=hybrid8_joint
  P_BIN="$B300_HYBRID8_PREPARED_BIN"; P_LABEL="$B300_HYBRID8_PREPARED_LABEL"; P_THREADS="$B300_HYBRID8_PREPARED_THREADS"
  B_BIN="$B300_HYBRID8_PREPARED_BASE_BIN"; B_LABEL="$B300_HYBRID8_PREPARED_BASE_LABEL"; B_THREADS="$B300_HYBRID8_PREPARED_BASE_THREADS"
  E1_BIN="$JOINT_PRIMARY_BIN"; E1_LABEL="$JOINT_PRIMARY_LABEL"; E1_THREADS="$JOINT_PRIMARY_THREADS"
  if [[ -n "$JOINT_BASE_BIN" && "$JOINT_BASE_BIN" != "$JOINT_PRIMARY_BIN" ]]; then
    E2_BIN="$JOINT_BASE_BIN"; E2_LABEL="$JOINT_BASE_LABEL"; E2_THREADS="$JOINT_BASE_THREADS"
  fi
else
  MODE=joint_fallback
  P_BIN="$JOINT_PRIMARY_BIN"; P_LABEL="$JOINT_PRIMARY_LABEL"; P_THREADS="$JOINT_PRIMARY_THREADS"
  if [[ -n "$JOINT_BASE_BIN" && "$JOINT_BASE_BIN" != "$JOINT_PRIMARY_BIN" ]]; then
    B_BIN="$JOINT_BASE_BIN"; B_LABEL="$JOINT_BASE_LABEL"; B_THREADS="$JOINT_BASE_THREADS"
  fi
fi

[[ -n "$P_BIN" && -x "$P_BIN" ]] || { echo 'grand selector primary candidate missing' >&2; exit 3; }
for t in "$P_THREADS" "$B_THREADS" "$E1_THREADS" "$E2_THREADS" "$E3_THREADS"; do
  [[ "$t" =~ ^[0-9]+$ ]] && ((t>=32 && t<=1024 && t%32==0)) || { echo "bad candidate threads=$t" >&2; exit 3; }
done

SUMMARY_ENV="${RACE_PREFIX}_grand.env"
{
  printf 'B300_GRAND_PREPARED=1\n'
  printf 'B300_GRAND_MODE=%q\n' "$MODE"
  printf 'B300_GRAND_NEXTSELF_OK=%q\n' "$NEXTSELF_OK"
  printf 'B300_GRAND_HYBRID8_OK=%q\n' "$HYBRID_OK"
  printf 'B300_GRAND_PRIMARY_BIN=%q\n' "$P_BIN"
  printf 'B300_GRAND_PRIMARY_LABEL=%q\n' "$P_LABEL"
  printf 'B300_GRAND_PRIMARY_THREADS=%q\n' "$P_THREADS"
  printf 'B300_GRAND_BASE_BIN=%q\n' "$B_BIN"
  printf 'B300_GRAND_BASE_LABEL=%q\n' "$B_LABEL"
  printf 'B300_GRAND_BASE_THREADS=%q\n' "$B_THREADS"
  printf 'B300_GRAND_EXTRA_BIN=%q\n' "$E1_BIN"
  printf 'B300_GRAND_EXTRA_LABEL=%q\n' "$E1_LABEL"
  printf 'B300_GRAND_EXTRA_THREADS=%q\n' "$E1_THREADS"
  printf 'B300_GRAND_EXTRA2_BIN=%q\n' "$E2_BIN"
  printf 'B300_GRAND_EXTRA2_LABEL=%q\n' "$E2_LABEL"
  printf 'B300_GRAND_EXTRA2_THREADS=%q\n' "$E2_THREADS"
  printf 'B300_GRAND_EXTRA3_BIN=%q\n' "$E3_BIN"
  printf 'B300_GRAND_EXTRA3_LABEL=%q\n' "$E3_LABEL"
  printf 'B300_GRAND_EXTRA3_THREADS=%q\n' "$E3_THREADS"
  printf 'B300_GRAND_DROPPED_JOINT_BASE_WHEN_BOTH=%q\n' "$([[ "$MODE" == nextself_hybrid8_joint ]] && echo 1 || echo 0)"
} >"$SUMMARY_ENV"
cat "$SUMMARY_ENV" >&2

echo "=== grand full-prime race mode=$MODE nextself=$NEXTSELF_OK hybrid8=$HYBRID_OK ===" >&2
exec env \
  PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" SMOKE_PRIME="$PRIME" FORCED_TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
  FORCED_OVERRIDE_BIN="$P_BIN" FORCED_OVERRIDE_LABEL="$P_LABEL" FORCED_OVERRIDE_THREADS="$P_THREADS" \
  FORCED_BASE_BIN="$B_BIN" FORCED_BASE_LABEL="$B_LABEL" FORCED_BASE_THREADS="$B_THREADS" \
  FORCED_EXTRA_BIN="$E1_BIN" FORCED_EXTRA_LABEL="$E1_LABEL" FORCED_EXTRA_THREADS="$E1_THREADS" \
  FORCED_EXTRA2_BIN="$E2_BIN" FORCED_EXTRA2_LABEL="$E2_LABEL" FORCED_EXTRA2_THREADS="$E2_THREADS" \
  FORCED_EXTRA3_BIN="$E3_BIN" FORCED_EXTRA3_LABEL="$E3_LABEL" FORCED_EXTRA3_THREADS="$E3_THREADS" \
  SELECT_ONLY="$SELECT_ONLY" REBUILD_BUCKETS="$REBUILD_BUCKETS" PREFIX="$RACE_PREFIX" \
  "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"
