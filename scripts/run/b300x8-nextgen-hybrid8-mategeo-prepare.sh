#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'mate-geometry prepare wrapper targets n=27' >&2; exit 2; }
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
[[ -f "$PROFILE_FILE" ]] || { echo "missing PROFILE_FILE=$PROFILE_FILE" >&2; exit 2; }
INPUT_ENV="${INPUT_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
[[ -s "$INPUT_ENV" ]] || { echo "missing INPUT_ENV=$INPUT_ENV" >&2; exit 2; }
ARCH="${ARCH:-native}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
SELF_EVICT="${SELF_EVICT:-default}"
MATE_EVICT="${MATE_EVICT:-default}"
MATE_WIDTH_LIST="${MATE_WIDTH_LIST:-1 2 4 8}"
MATE_DISTANCE_LIST="${MATE_DISTANCE_LIST:-1 2 4}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"
RUN_STAGED="${RUN_STAGED:-1}"
STAGED_PREFIX="${STAGED_PREFIX:-$ONEESAN_ROOT/work/b300_mategeo_staged}"
RAW_WINNER_ENV="${RAW_WINNER_ENV:-${STAGED_PREFIX}_winner.env}"
RAW_RACE_PREFIX="${RAW_RACE_PREFIX:-${STAGED_PREFIX}.promote}"
RAW_PREPARE_ENV="${RAW_PREPARE_ENV:-${STAGED_PREFIX}.raw-prepared.env}"
PREPARE_ENV="${PREPARE_ENV:-${STAGED_PREFIX}.prepared.env}"
for x in RUN_STAGED; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || exit 2; done
for e in "$SELF_EVICT" "$MATE_EVICT"; do case "$e" in default|normal|last) ;; *) echo "bad eviction hint=$e" >&2; exit 2;; esac; done
for w in $MATE_WIDTH_LIST; do case "$w" in 1|2|4|8) ;; *) echo "bad mate width=$w" >&2; exit 2;; esac; done
for d in $MATE_DISTANCE_LIST; do case "$d" in 1|2|4) ;; *) echo "bad mate distance=$d" >&2; exit 2;; esac; done
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY
mkdir -p "$(dirname "$PREPARE_ENV")" "$(dirname "$RAW_PREPARE_ENV")"

set +e
PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" MOD="${MOD:-4294967291}" \
  INPUT_ENV="$INPUT_ENV" SELF_EVICT="$SELF_EVICT" MATE_EVICT="$MATE_EVICT" \
  MATE_WIDTH_LIST="$MATE_WIDTH_LIST" MATE_DISTANCE_LIST="$MATE_DISTANCE_LIST" \
  RUN_STAGED="$RUN_STAGED" PREPARE_ONLY=1 MIN_SPEEDUP="$MIN_SPEEDUP" \
  STAGED_PREFIX="$STAGED_PREFIX" WINNER_ENV="$RAW_WINNER_ENV" RACE_PREFIX="$RAW_RACE_PREFIX" PREPARE_ENV="$RAW_PREPARE_ENV" \
  bash "$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextmate-geometry-staged-fullprime-race.sh" 27 "$@"
rc=$?
set -e
if ((rc==4)); then
  {
    printf 'B300_MATEGEO_PREPARED=0\n'
    printf 'B300_MATEGEO_REJECTED=1\n'
    printf 'B300_MATEGEO_INPUT_ENV=%q\n' "$INPUT_ENV"
    printf 'B300_MATEGEO_SELF_EVICT=%q\n' "$SELF_EVICT"
    printf 'B300_MATEGEO_MATE_EVICT=%q\n' "$MATE_EVICT"
  } >"$PREPARE_ENV"
  cat "$PREPARE_ENV"
  exit 4
fi
((rc==0)) || exit "$rc"
[[ -s "$RAW_PREPARE_ENV" ]] || { echo 'raw mate-geometry prepare env missing' >&2; exit 3; }
# shellcheck disable=SC1090
source "$RAW_PREPARE_ENV"
for k in B300_STAGEI_PREPARED B300_STAGEI_PREPARED_SELF_WIDTH B300_STAGEI_PREPARED_SELF_DISTANCE B300_STAGEI_PREPARED_MATE_WIDTH B300_STAGEI_PREPARED_MATE_DISTANCE B300_STAGEI_PREPARED_BIN B300_STAGEI_PREPARED_LABEL B300_STAGEI_PREPARED_THREADS B300_STAGEI_PREPARED_CONTROL_BIN B300_STAGEI_PREPARED_CONTROL_LABEL B300_STAGEI_PREPARED_CONTROL_THREADS B300_STAGEI_PREPARED_STAGED_SPEEDUP B300_STAGEI_PREPARED_MANIFEST; do
  [[ -n "${!k+x}" ]] || { echo "raw mate-geometry prepare missing $k" >&2; exit 3; }
done
[[ "$B300_STAGEI_PREPARED" == 1 ]] || exit 3
for w in "$B300_STAGEI_PREPARED_SELF_WIDTH" "$B300_STAGEI_PREPARED_MATE_WIDTH"; do case "$w" in 1|2|4|8) ;; *) exit 3;; esac; done
for d in "$B300_STAGEI_PREPARED_SELF_DISTANCE" "$B300_STAGEI_PREPARED_MATE_DISTANCE"; do case "$d" in 1|2|4) ;; *) exit 3;; esac; done
[[ -x "$B300_STAGEI_PREPARED_BIN" && -x "$B300_STAGEI_PREPARED_CONTROL_BIN" && -s "$B300_STAGEI_PREPARED_MANIFEST" ]] || exit 3
INPUT_SHA="$(sha256sum "$INPUT_ENV" | awk '{print $1}')"
RAW_SHA="$(sha256sum "$RAW_PREPARE_ENV" | awk '{print $1}')"
{
  printf 'B300_MATEGEO_PREPARED=1\n'
  printf 'B300_MATEGEO_REJECTED=0\n'
  printf 'B300_MATEGEO_SELF_WIDTH=%q\n' "$B300_STAGEI_PREPARED_SELF_WIDTH"
  printf 'B300_MATEGEO_SELF_DISTANCE=%q\n' "$B300_STAGEI_PREPARED_SELF_DISTANCE"
  printf 'B300_MATEGEO_MATE_WIDTH=%q\n' "$B300_STAGEI_PREPARED_MATE_WIDTH"
  printf 'B300_MATEGEO_MATE_DISTANCE=%q\n' "$B300_STAGEI_PREPARED_MATE_DISTANCE"
  printf 'B300_MATEGEO_SELF_EVICT=%q\n' "$SELF_EVICT"
  printf 'B300_MATEGEO_MATE_EVICT=%q\n' "$MATE_EVICT"
  printf 'B300_MATEGEO_BIN=%q\n' "$B300_STAGEI_PREPARED_BIN"
  printf 'B300_MATEGEO_LABEL=%q\n' "$B300_STAGEI_PREPARED_LABEL"
  printf 'B300_MATEGEO_THREADS=%q\n' "$B300_STAGEI_PREPARED_THREADS"
  printf 'B300_MATEGEO_CONTROL_BIN=%q\n' "$B300_STAGEI_PREPARED_CONTROL_BIN"
  printf 'B300_MATEGEO_CONTROL_LABEL=%q\n' "$B300_STAGEI_PREPARED_CONTROL_LABEL"
  printf 'B300_MATEGEO_CONTROL_THREADS=%q\n' "$B300_STAGEI_PREPARED_CONTROL_THREADS"
  printf 'B300_MATEGEO_STAGED_SPEEDUP=%q\n' "$B300_STAGEI_PREPARED_STAGED_SPEEDUP"
  printf 'B300_MATEGEO_MANIFEST=%q\n' "$B300_STAGEI_PREPARED_MANIFEST"
  printf 'B300_MATEGEO_INPUT_ENV=%q\n' "$INPUT_ENV"
  printf 'B300_MATEGEO_INPUT_ENV_SHA256=%q\n' "$INPUT_SHA"
  printf 'B300_MATEGEO_RAW_PREPARE_ENV=%q\n' "$RAW_PREPARE_ENV"
  printf 'B300_MATEGEO_RAW_PREPARE_ENV_SHA256=%q\n' "$RAW_SHA"
  printf 'B300_MATEGEO_RAW_WINNER_ENV=%q\n' "$RAW_WINNER_ENV"
} >"$PREPARE_ENV"
cat "$PREPARE_ENV"
echo "B300 MATEGEO PREPARED self=w${B300_STAGEI_PREPARED_SELF_WIDTH}d${B300_STAGEI_PREPARED_SELF_DISTANCE} mate=w${B300_STAGEI_PREPARED_MATE_WIDTH}d${B300_STAGEI_PREPARED_MATE_DISTANCE} self_evict=$SELF_EVICT mate_evict=$MATE_EVICT speedup=${B300_STAGEI_PREPARED_STAGED_SPEEDUP}x env=$PREPARE_ENV" >&2