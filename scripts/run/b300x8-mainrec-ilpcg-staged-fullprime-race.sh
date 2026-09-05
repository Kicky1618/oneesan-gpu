#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'staged mainrec full-prime race targets n=27' >&2; exit 2; }

PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
[[ -f "$PROFILE_FILE" ]] || { echo "missing PROFILE_FILE=$PROFILE_FILE" >&2; exit 2; }
ARCH="${ARCH:-native}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
HIGHDROP_LIST="${HIGHDROP_LIST:-0 1}"
SEARCH_ROWS="${SEARCH_ROWS:-1}"
VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"
SEARCH_REPEATS="${SEARCH_REPEATS:-1}"
VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"
TRANSFORM_MIN_SPEEDUP="${TRANSFORM_MIN_SPEEDUP:-1.01}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
RUN_STAGED="${RUN_STAGED:-1}"
SELECT_ONLY="${SELECT_ONLY:-1}"
REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
STAGED_PREFIX="${STAGED_PREFIX:-$ONEESAN_ROOT/work/b300_mainrec_ilpcg_staged}"
WINNER_ENV="${WINNER_ENV:-${STAGED_PREFIX}_winner.env}"
RACE_PREFIX="${RACE_PREFIX:-$ONEESAN_ROOT/work/b300_mainrec_ilpcg_staged_fullprime_n27}"
for x in RUN_STAGED SELECT_ONLY REBUILD_BUCKETS; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done

if [[ "$RUN_STAGED" == 1 ]]; then
  echo '=== staged mainrec calibration: search -> rows 4/8 validation ===' >&2
  ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" THREADS_LIST="$THREADS_LIST" HIGHDROP_LIST="$HIGHDROP_LIST" \
    SEARCH_ROWS="$SEARCH_ROWS" VALIDATE_ROWS="$VALIDATE_ROWS" SEARCH_REPEATS="$SEARCH_REPEATS" VALIDATE_REPEATS="$VALIDATE_REPEATS" \
    TRANSFORM_MIN_SPEEDUP="$TRANSFORM_MIN_SPEEDUP" SAMPLE_INTERVAL="$SAMPLE_INTERVAL" PREFIX="$STAGED_PREFIX" WINNER_ENV="$WINNER_ENV" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-mainrec-ilpcg-calibrate-staged.sh"
fi

[[ -f "$WINNER_ENV" ]] || { echo "missing staged WINNER_ENV=$WINNER_ENV" >&2; exit 3; }
# shellcheck disable=SC1090
source "$WINNER_ENV"
for n in B300_MAINREC_STAGED_VALIDATED B300_MAINREC_TRANSFORMED B300_MAINREC_HIGH_DROP_CHUNK B300_MAINREC_MODE \
  B300_MAINREC_THREADS B300_MAINREC_BIN B300_MAINREC_BASE_BIN B300_MAINREC_BASE_THREADS \
  B300_MAINREC_SPILL_STORE_BYTES B300_MAINREC_SPILL_LOAD_BYTES; do
  [[ -n "${!n+x}" ]] || { echo "staged winner env missing $n" >&2; exit 3; }
done
[[ "$B300_MAINREC_STAGED_VALIDATED" == 0 || "$B300_MAINREC_STAGED_VALIDATED" == 1 ]] || exit 3
[[ "$B300_MAINREC_TRANSFORMED" == 0 || "$B300_MAINREC_TRANSFORMED" == 1 ]] || exit 3
[[ "$B300_MAINREC_HIGH_DROP_CHUNK" == 0 || "$B300_MAINREC_HIGH_DROP_CHUNK" == 1 ]] || exit 3
[[ "$B300_MAINREC_MODE" =~ ^ilp(2|4|8)(cg)?$ ]] || exit 3
[[ "$B300_MAINREC_THREADS" =~ ^[0-9]+$ && "$B300_MAINREC_BASE_THREADS" =~ ^[0-9]+$ ]] || exit 3
[[ -x "$B300_MAINREC_BIN" && -x "$B300_MAINREC_BASE_BIN" ]] || { echo 'staged winner/base binary missing' >&2; exit 3; }
if [[ "$B300_MAINREC_TRANSFORMED" == 1 ]]; then
  [[ "$B300_MAINREC_STAGED_VALIDATED" == 1 ]] || { echo 'refusing unvalidated transformed candidate' >&2; exit 4; }
  [[ "$B300_MAINREC_SPILL_STORE_BYTES" == 0 && "$B300_MAINREC_SPILL_LOAD_BYTES" == 0 ]] || {
    echo "refusing spilling staged candidate store=$B300_MAINREC_SPILL_STORE_BYTES load=$B300_MAINREC_SPILL_LOAD_BYTES" >&2
    exit 4
  }
fi

label="mainrec_staged_${B300_MAINREC_MODE}_hd${B300_MAINREC_HIGH_DROP_CHUNK}"
echo "=== full-prime race staged=$B300_MAINREC_STAGED_VALIDATED transformed=$B300_MAINREC_TRANSFORMED candidate=$label threads=$B300_MAINREC_THREADS ===" >&2
PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" MAX_WINDOW="$MAX_WINDOW" FORCED_TARGET_MIB="$TARGET_MIB" \
  FORCED_OVERRIDE_BIN="$B300_MAINREC_BIN" FORCED_OVERRIDE_LABEL="$label" FORCED_OVERRIDE_THREADS="$B300_MAINREC_THREADS" \
  FORCED_BASE_BIN="$B300_MAINREC_BASE_BIN" FORCED_BASE_LABEL=mainrec_staged_ilp2_base FORCED_BASE_THREADS="$B300_MAINREC_BASE_THREADS" \
  REBUILD_BUCKETS="$REBUILD_BUCKETS" SELECT_ONLY="$SELECT_ONLY" PREFIX="$RACE_PREFIX" \
  exec "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"
