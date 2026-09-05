#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo 'mainrec ILP/CG full-prime race targets n=27' >&2; exit 2; }
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
[[ -f "$PROFILE_FILE" ]] || { echo "missing PROFILE_FILE=$PROFILE_FILE" >&2; exit 2; }
SWEEP_PREFIX="${SWEEP_PREFIX:-$ONEESAN_ROOT/work/b300_mainrec_ilpcg_hd${HIGH_DROP_CHUNK:-0}_row${ROWS:-1}}"
WINNER_ENV="${WINNER_ENV:-${SWEEP_PREFIX}_winner.env}"
RUN_SWEEP="${RUN_SWEEP:-1}"; ROWS="${ROWS:-1}"; REPEATS="${REPEATS:-1}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; ARCH="${ARCH:-native}"
SELECT_ONLY="${SELECT_ONLY:-1}"; REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
RACE_PREFIX="${RACE_PREFIX:-$ONEESAN_ROOT/work/b300_mainrec_ilpcg_fullprime_race_n27}"
for x in RUN_SWEEP SELECT_ONLY REBUILD_BUCKETS; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] || { echo 'ROWS must be positive' >&2; exit 2; }

if [[ "$RUN_SWEEP" == 1 ]]; then
  echo "=== main recurrence ILP2/4/8 x default/CG partial-row calibration ===" >&2
  ARCH="$ARCH" ROWS="$ROWS" REPEATS="$REPEATS" THREADS_LIST="$THREADS_LIST" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" PREFIX="$SWEEP_PREFIX" WINNER_ENV="$WINNER_ENV" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-mainrec-ilp-cg-sweep.sh"
fi
[[ -f "$WINNER_ENV" ]] || { echo "missing WINNER_ENV=$WINNER_ENV" >&2; exit 3; }
# shellcheck disable=SC1090
source "$WINNER_ENV"
for n in B300_MAINREC_WINNER_MODE B300_MAINREC_WINNER_BIN B300_MAINREC_WINNER_THREADS B300_MAINREC_BASE_BIN B300_MAINREC_BASE_THREADS B300_MAINREC_WINNER_SPILL_STORE_BYTES B300_MAINREC_WINNER_SPILL_LOAD_BYTES; do
  [[ -n "${!n+x}" ]] || { echo "winner env missing $n" >&2; exit 3; }
done
[[ -x "$B300_MAINREC_WINNER_BIN" && -x "$B300_MAINREC_BASE_BIN" ]] || { echo 'winner/base binary missing' >&2; exit 3; }
[[ "$B300_MAINREC_WINNER_THREADS" =~ ^[0-9]+$ && "$B300_MAINREC_BASE_THREADS" =~ ^[0-9]+$ ]] || exit 3
[[ "$B300_MAINREC_WINNER_SPILL_STORE_BYTES" == 0 && "$B300_MAINREC_WINNER_SPILL_LOAD_BYTES" == 0 ]] || {
  echo "refusing full-prime promotion of spilling winner store=$B300_MAINREC_WINNER_SPILL_STORE_BYTES load=$B300_MAINREC_WINNER_SPILL_LOAD_BYTES" >&2
  exit 4
}

label="mainrec_${B300_MAINREC_WINNER_MODE}"
echo "=== full-prime race winner=$label threads=$B300_MAINREC_WINNER_THREADS mc_partial=${B300_MAINREC_WINNER_MC_AVG_PCT:-NA}% vs ilp2 baseline + profiled warp/orbit ===" >&2
PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" MAX_WINDOW="$MAX_WINDOW" FORCED_TARGET_MIB="$TARGET_MIB" \
FORCED_OVERRIDE_BIN="$B300_MAINREC_WINNER_BIN" FORCED_OVERRIDE_LABEL="$label" FORCED_OVERRIDE_THREADS="$B300_MAINREC_WINNER_THREADS" \
FORCED_BASE_BIN="$B300_MAINREC_BASE_BIN" FORCED_BASE_LABEL=mainrec_ilp2_base FORCED_BASE_THREADS="$B300_MAINREC_BASE_THREADS" \
REBUILD_BUCKETS="$REBUILD_BUCKETS" SELECT_ONLY="$SELECT_ONLY" PREFIX="$RACE_PREFIX" \
  exec "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"
