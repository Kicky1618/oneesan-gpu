#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'nextgen hybrid8 staged full-prime race targets n=27' >&2; exit 2; }

PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
[[ -f "$PROFILE_FILE" ]] || { echo "missing PROFILE_FILE=$PROFILE_FILE" >&2; exit 2; }
ARCH="${ARCH:-native}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
RUN_STAGED="${RUN_STAGED:-1}"
SELECT_ONLY="${SELECT_ONLY:-1}"
REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
STAGED_PREFIX="${STAGED_PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_staged}"
WINNER_ENV="${WINNER_ENV:-${STAGED_PREFIX}_winner.env}"
RACE_PREFIX="${RACE_PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_staged_fullprime_n27}"
for x in RUN_STAGED SELECT_ONLY REBUILD_BUCKETS; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done

if [[ "$RUN_STAGED" == 1 ]]; then
  echo '=== nextgen hybrid8 staged calibration ===' >&2
  PREFIX="$STAGED_PREFIX" FINAL_ENV="$WINNER_ENV" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-staged-calibrate.sh"
fi

[[ -s "$WINNER_ENV" ]] || { echo "missing staged WINNER_ENV=$WINNER_ENV" >&2; exit 3; }
# shellcheck disable=SC1090
source "$WINNER_ENV"
for key in \
  B300_HYBRID8_STAGED_VALIDATED B300_HYBRID8_FINAL_ENABLED B300_HYBRID8_FINAL_THRESHOLD \
  B300_HYBRID8_FINAL_BIN B300_HYBRID8_FINAL_THREADS B300_HYBRID8_FINAL_SPEEDUP_VS_BASE B300_HYBRID8_FINAL_SPILL_FREE \
  B300_HYBRID8_BASE_BIN B300_HYBRID8_BASE_THREADS B300_HYBRID8_RESIDUE; do
  [[ -n "${!key+x}" ]] || { echo "winner env missing $key" >&2; exit 3; }
done
[[ "$B300_HYBRID8_STAGED_VALIDATED" == 1 && "$B300_HYBRID8_FINAL_ENABLED" == 1 ]] || {
  echo 'hybrid8 did not survive ROWS=1/4/8 validation; refusing transformed full-prime promotion' >&2
  exit 4
}
[[ "$B300_HYBRID8_FINAL_SPILL_FREE" == 1 ]] || {
  echo 'hybrid8 staged winner is not explicitly spill-free; refusing full-prime promotion' >&2
  exit 4
}
[[ "$B300_HYBRID8_FINAL_THRESHOLD" =~ ^[0-9]+$ ]] || exit 3
for key in B300_HYBRID8_FINAL_THREADS B300_HYBRID8_BASE_THREADS; do
  v="${!key}"; [[ "$v" =~ ^[0-9]+$ ]] && ((v>=32 && v<=1024 && v%32==0)) || { echo "bad $key=$v" >&2; exit 3; }
done
[[ -x "$B300_HYBRID8_FINAL_BIN" && -x "$B300_HYBRID8_BASE_BIN" ]] || { echo 'hybrid8 final/base binary missing' >&2; exit 3; }
python3 - "$B300_HYBRID8_FINAL_SPEEDUP_VS_BASE" "${HYBRID_MIN_SPEEDUP:-1.01}" <<'PY'
import sys
speed,need=map(float,sys.argv[1:])
if speed<need: raise SystemExit(f'staged hybrid8 speedup {speed:.9f}x below {need:.9f}x')
PY

FINAL_SHA="$(sha256sum "$B300_HYBRID8_FINAL_BIN" | awk '{print $1}')"
BASE_SHA="$(sha256sum "$B300_HYBRID8_BASE_BIN" | awk '{print $1}')"
ENV_SHA="$(sha256sum "$WINNER_ENV" | awk '{print $1}')"
cat >"${RACE_PREFIX}_promotion.env" <<EOF
B300_HYBRID8_PROMOTION_VALIDATED=1
B300_HYBRID8_PROMOTION_SPILL_FREE=1
B300_HYBRID8_PROMOTION_THRESHOLD=$B300_HYBRID8_FINAL_THRESHOLD
B300_HYBRID8_PROMOTION_BIN=$(printf '%q' "$B300_HYBRID8_FINAL_BIN")
B300_HYBRID8_PROMOTION_THREADS=$B300_HYBRID8_FINAL_THREADS
B300_HYBRID8_PROMOTION_SHA256=$FINAL_SHA
B300_HYBRID8_PROMOTION_BASE_BIN=$(printf '%q' "$B300_HYBRID8_BASE_BIN")
B300_HYBRID8_PROMOTION_BASE_THREADS=$B300_HYBRID8_BASE_THREADS
B300_HYBRID8_PROMOTION_BASE_SHA256=$BASE_SHA
B300_HYBRID8_PROMOTION_WINNER_ENV_SHA256=$ENV_SHA
B300_HYBRID8_PROMOTION_PARTIAL_RESIDUE=$B300_HYBRID8_RESIDUE
B300_HYBRID8_PROMOTION_STAGED_SPEEDUP=$B300_HYBRID8_FINAL_SPEEDUP_VS_BASE
EOF

label="nextgen_hybrid8_t${B300_HYBRID8_FINAL_THRESHOLD}"
echo "=== full-prime race: $label vs nextgen A-D base vs profiled warp/orbit ===" >&2
echo "staged_speedup=${B300_HYBRID8_FINAL_SPEEDUP_VS_BASE}x hybrid_sha=${FINAL_SHA:0:12} base_sha=${BASE_SHA:0:12}" >&2

exec env \
  PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" MAX_WINDOW="$MAX_WINDOW" FORCED_TARGET_MIB="$TARGET_MIB" \
  FORCED_OVERRIDE_BIN="$B300_HYBRID8_FINAL_BIN" FORCED_OVERRIDE_LABEL="$label" FORCED_OVERRIDE_THREADS="$B300_HYBRID8_FINAL_THREADS" \
  FORCED_BASE_BIN="$B300_HYBRID8_BASE_BIN" FORCED_BASE_LABEL=nextgen_abcd_base FORCED_BASE_THREADS="$B300_HYBRID8_BASE_THREADS" \
  REBUILD_BUCKETS="$REBUILD_BUCKETS" SELECT_ONLY="$SELECT_ONLY" PREFIX="$RACE_PREFIX" \
  "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"
