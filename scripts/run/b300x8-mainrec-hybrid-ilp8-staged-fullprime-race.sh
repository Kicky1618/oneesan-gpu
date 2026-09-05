#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo 'mainrec hybrid staged full-prime race targets n=27' >&2; exit 2; }

PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
[[ -f "$PROFILE_FILE" ]] || { echo "missing PROFILE_FILE=$PROFILE_FILE" >&2; exit 2; }
ARCH="${ARCH:-native}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
RUN_CALIBRATION="${RUN_CALIBRATION:-1}"
SELECT_ONLY="${SELECT_ONLY:-1}"
REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
SEARCH_ROWS="${SEARCH_ROWS:-1}"
VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"
SEARCH_REPEATS="${SEARCH_REPEATS:-1}"
VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.01}"
ILP8_THRESHOLDS="${ILP8_THRESHOLDS:-0 262144 524288 1048576 2097152 4194304 8388608 16777216}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
RANDOM_CG="${RANDOM_CG:-0}"
RANDOM_CG_L2_FETCH_BYTES="${RANDOM_CG_L2_FETCH_BYTES:-0}"
DUALMASK="${DUALMASK:-0}"
MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
CAL_PREFIX="${CAL_PREFIX:-$ONEESAN_ROOT/work/b300_mainrec_hybrid8_staged}"
WINNER_ENV="${WINNER_ENV:-${CAL_PREFIX}_winner.env}"
MANIFEST="${MANIFEST:-${CAL_PREFIX}_winner.sha256}"
RACE_PREFIX="${RACE_PREFIX:-$ONEESAN_ROOT/work/b300_mainrec_hybrid8_staged_fullprime_n27}"

for x in RUN_CALIBRATION SELECT_ONLY REBUILD_BUCKETS; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
command -v sha256sum >/dev/null || { echo 'sha256sum required' >&2; exit 2; }

if [[ "$RUN_CALIBRATION" == 1 ]]; then
  echo '=== staged recurrence ILP2/8 hybrid calibration ===' >&2
  N=27 ARCH="$ARCH" SEARCH_ROWS="$SEARCH_ROWS" VALIDATE_ROWS="$VALIDATE_ROWS" \
    SEARCH_REPEATS="$SEARCH_REPEATS" VALIDATE_REPEATS="$VALIDATE_REPEATS" MIN_SPEEDUP="$MIN_SPEEDUP" \
    ILP8_THRESHOLDS="$ILP8_THRESHOLDS" THREADS_LIST="$THREADS_LIST" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
    HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" RANDOM_CG="$RANDOM_CG" RANDOM_CG_L2_FETCH_BYTES="$RANDOM_CG_L2_FETCH_BYTES" \
    DUALMASK="$DUALMASK" MAXRREGCOUNT="$MAXRREGCOUNT" PREFIX="$CAL_PREFIX" WINNER_ENV="$WINNER_ENV" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid-ilp8-calibrate-staged.sh"
fi

[[ -f "$WINNER_ENV" ]] || { echo "missing staged winner env=$WINNER_ENV" >&2; exit 3; }
# shellcheck disable=SC1090
source "$WINNER_ENV"
if [[ "${B300_MAINREC_HYBRID_STAGED_VALIDATED:-0}" != 1 ]]; then
  echo "staged hybrid did not validate; no full-prime promotion (search_mode=${B300_MAINREC_HYBRID_SEARCH_WINNER_MODE:-unknown} threshold=${B300_MAINREC_HYBRID_SEARCH_WINNER_THRESHOLD:-NA} speedup=${B300_MAINREC_HYBRID_SEARCH_WINNER_SPEEDUP:-NA})" >&2
  exit 5
fi

for k in B300_MAINREC_HYBRID_SELECTED_THRESHOLD B300_MAINREC_HYBRID_SELECTED_THREADS B300_MAINREC_HYBRID_SELECTED_BIN B300_MAINREC_HYBRID_SELECTED_BASE_BIN B300_MAINREC_HYBRID_SELECTED_RESIDUE B300_MAINREC_HYBRID_SELECTED_SPEEDUP B300_MAINREC_HYBRID_SELECTED_SPILL_FREE; do
  [[ -n "${!k+x}" ]] || { echo "winner env missing $k" >&2; exit 3; }
done
[[ "$B300_MAINREC_HYBRID_SELECTED_SPILL_FREE" == 1 ]] || { echo 'refusing full-prime promotion of spilling hybrid' >&2; exit 4; }
[[ -x "$B300_MAINREC_HYBRID_SELECTED_BIN" && -x "$B300_MAINREC_HYBRID_SELECTED_BASE_BIN" ]] || { echo 'staged winner/base binary missing' >&2; exit 3; }
[[ "$B300_MAINREC_HYBRID_SELECTED_THREADS" =~ ^[0-9]+$ ]] && ((B300_MAINREC_HYBRID_SELECTED_THREADS>=32 && B300_MAINREC_HYBRID_SELECTED_THREADS<=1024 && B300_MAINREC_HYBRID_SELECTED_THREADS%32==0)) || { echo 'bad selected threads' >&2; exit 3; }

if [[ "$RUN_CALIBRATION" == 1 ]]; then
  tmp="${MANIFEST}.tmp"
  sha256sum "$WINNER_ENV" "$B300_MAINREC_HYBRID_SELECTED_BIN" "$B300_MAINREC_HYBRID_SELECTED_BASE_BIN" >"$tmp"
  mv "$tmp" "$MANIFEST"
else
  [[ -f "$MANIFEST" ]] || { echo "missing calibration manifest=$MANIFEST; rerun with RUN_CALIBRATION=1" >&2; exit 3; }
fi
if ! sha256sum -c "$MANIFEST" >/dev/null; then
  echo 'staged hybrid calibration fingerprint mismatch; rerun calibration' >&2
  exit 3
fi

label="mainrec_hybrid8_t${B300_MAINREC_HYBRID_SELECTED_THRESHOLD}"
[[ "${B300_MAINREC_HYBRID_RANDOM_CG:-0}" == 0 ]] || label="${label}_cg${B300_MAINREC_HYBRID_RANDOM_CG_L2_FETCH_BYTES:-0}"
[[ "${B300_MAINREC_HYBRID_DUALMASK:-0}" == 0 ]] || label="${label}_dual"
[[ "${B300_MAINREC_HYBRID_MAXRREGCOUNT:-0}" == 0 ]] || label="${label}_r${B300_MAINREC_HYBRID_MAXRREGCOUNT}"

echo "=== complete-prime race staged hybrid=$label threads=$B300_MAINREC_HYBRID_SELECTED_THREADS staged_speedup=${B300_MAINREC_HYBRID_SELECTED_SPEEDUP} vs production ILP2 + profiled warp/orbit ===" >&2
echo "calibration_manifest=$MANIFEST" >&2

PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" MAX_WINDOW="$MAX_WINDOW" FORCED_TARGET_MIB="$TARGET_MIB" \
FORCED_OVERRIDE_BIN="$B300_MAINREC_HYBRID_SELECTED_BIN" FORCED_OVERRIDE_LABEL="$label" FORCED_OVERRIDE_THREADS="$B300_MAINREC_HYBRID_SELECTED_THREADS" \
FORCED_BASE_BIN="$B300_MAINREC_HYBRID_SELECTED_BASE_BIN" FORCED_BASE_LABEL=mainrec_ilp2_staged_base FORCED_BASE_THREADS="$B300_MAINREC_HYBRID_SELECTED_THREADS" \
REBUILD_BUCKETS="$REBUILD_BUCKETS" SELECT_ONLY="$SELECT_ONLY" PREFIX="$RACE_PREFIX" \
  exec "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"
