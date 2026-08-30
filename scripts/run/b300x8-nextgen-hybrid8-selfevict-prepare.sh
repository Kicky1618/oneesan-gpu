#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'self-eviction prepare wrapper targets n=27' >&2; exit 2; }

PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
[[ -f "$PROFILE_FILE" ]] || { echo "missing PROFILE_FILE=$PROFILE_FILE" >&2; exit 2; }
INPUT_ENV="${INPUT_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
[[ -s "$INPUT_ENV" ]] || { echo "missing INPUT_ENV=$INPUT_ENV" >&2; exit 2; }
ARCH="${ARCH:-native}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"
RUN_STAGED="${RUN_STAGED:-1}"
STAGED_PREFIX="${STAGED_PREFIX:-$ONEESAN_ROOT/work/b300_selfevict_staged}"
RAW_WINNER_ENV="${RAW_WINNER_ENV:-${STAGED_PREFIX}_winner.env}"
RAW_RACE_PREFIX="${RAW_RACE_PREFIX:-${STAGED_PREFIX}.promote}"
RAW_PREPARE_ENV="${RAW_PREPARE_ENV:-${STAGED_PREFIX}.raw-prepared.env}"
PREPARE_ENV="${PREPARE_ENV:-${STAGED_PREFIX}.prepared.env}"
for x in RUN_STAGED; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY
mkdir -p "$(dirname "$PREPARE_ENV")" "$(dirname "$RAW_PREPARE_ENV")"

set +e
PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" MOD="${MOD:-4294967291}" \
  INPUT_ENV="$INPUT_ENV" RUN_STAGED="$RUN_STAGED" PREPARE_ONLY=1 MIN_SPEEDUP="$MIN_SPEEDUP" \
  STAGED_PREFIX="$STAGED_PREFIX" WINNER_ENV="$RAW_WINNER_ENV" RACE_PREFIX="$RAW_RACE_PREFIX" PREPARE_ENV="$RAW_PREPARE_ENV" \
  bash "$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextself-evict-staged-fullprime-race.sh" 27 "$@"
rc=$?
set -e
if ((rc==4)); then
  {
    printf 'B300_EVICT_PREPARED=0\n'
    printf 'B300_EVICT_REJECTED=1\n'
    printf 'B300_EVICT_INPUT_ENV=%q\n' "$INPUT_ENV"
    printf 'B300_EVICT_RAW_WINNER_ENV=%q\n' "$RAW_WINNER_ENV"
  } >"$PREPARE_ENV"
  cat "$PREPARE_ENV"
  exit 4
fi
((rc==0)) || exit "$rc"
[[ -s "$RAW_PREPARE_ENV" ]] || { echo 'raw self-eviction prepare env missing' >&2; exit 3; }
# shellcheck disable=SC1090
source "$RAW_PREPARE_ENV"
for k in B300_STAGEI_PREPARED B300_STAGEI_PREPARED_HINT B300_STAGEI_PREPARED_WIDTH B300_STAGEI_PREPARED_DISTANCE B300_STAGEI_PREPARED_BIN B300_STAGEI_PREPARED_LABEL B300_STAGEI_PREPARED_THREADS B300_STAGEI_PREPARED_CONTROL_BIN B300_STAGEI_PREPARED_CONTROL_LABEL B300_STAGEI_PREPARED_CONTROL_THREADS B300_STAGEI_PREPARED_MANIFEST; do
  [[ -n "${!k+x}" ]] || { echo "raw self-eviction prepare missing $k" >&2; exit 3; }
done
[[ "$B300_STAGEI_PREPARED" == 1 ]] || exit 3
case "$B300_STAGEI_PREPARED_HINT" in normal|last) ;; *) echo "bad self-eviction hint=$B300_STAGEI_PREPARED_HINT" >&2; exit 3;; esac
case "$B300_STAGEI_PREPARED_WIDTH" in 1|2|4|8) ;; *) exit 3;; esac
case "$B300_STAGEI_PREPARED_DISTANCE" in 1|2|4) ;; *) exit 3;; esac
[[ -x "$B300_STAGEI_PREPARED_BIN" && -x "$B300_STAGEI_PREPARED_CONTROL_BIN" && -s "$B300_STAGEI_PREPARED_MANIFEST" ]] || exit 3
INPUT_SHA="$(sha256sum "$INPUT_ENV" | awk '{print $1}')"
RAW_SHA="$(sha256sum "$RAW_PREPARE_ENV" | awk '{print $1}')"
{
  printf 'B300_EVICT_PREPARED=1\n'
  printf 'B300_EVICT_REJECTED=0\n'
  printf 'B300_EVICT_HINT=%q\n' "$B300_STAGEI_PREPARED_HINT"
  printf 'B300_EVICT_WIDTH=%q\n' "$B300_STAGEI_PREPARED_WIDTH"
  printf 'B300_EVICT_DISTANCE=%q\n' "$B300_STAGEI_PREPARED_DISTANCE"
  printf 'B300_EVICT_BIN=%q\n' "$B300_STAGEI_PREPARED_BIN"
  printf 'B300_EVICT_LABEL=%q\n' "$B300_STAGEI_PREPARED_LABEL"
  printf 'B300_EVICT_THREADS=%q\n' "$B300_STAGEI_PREPARED_THREADS"
  printf 'B300_EVICT_CONTROL_BIN=%q\n' "$B300_STAGEI_PREPARED_CONTROL_BIN"
  printf 'B300_EVICT_CONTROL_LABEL=%q\n' "$B300_STAGEI_PREPARED_CONTROL_LABEL"
  printf 'B300_EVICT_CONTROL_THREADS=%q\n' "$B300_STAGEI_PREPARED_CONTROL_THREADS"
  printf 'B300_EVICT_MANIFEST=%q\n' "$B300_STAGEI_PREPARED_MANIFEST"
  printf 'B300_EVICT_INPUT_ENV=%q\n' "$INPUT_ENV"
  printf 'B300_EVICT_INPUT_ENV_SHA256=%q\n' "$INPUT_SHA"
  printf 'B300_EVICT_RAW_PREPARE_ENV=%q\n' "$RAW_PREPARE_ENV"
  printf 'B300_EVICT_RAW_PREPARE_ENV_SHA256=%q\n' "$RAW_SHA"
  printf 'B300_EVICT_RAW_WINNER_ENV=%q\n' "$RAW_WINNER_ENV"
} >"$PREPARE_ENV"
cat "$PREPARE_ENV"
echo "B300 EVICT PREPARED hint=$B300_STAGEI_PREPARED_HINT geometry=w${B300_STAGEI_PREPARED_WIDTH}d${B300_STAGEI_PREPARED_DISTANCE} env=$PREPARE_ENV" >&2
