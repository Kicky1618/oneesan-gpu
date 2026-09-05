#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo 'ILP8 async full-prime race targets n=27' >&2; exit 2; }
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
[[ -f "$PROFILE_FILE" ]] || { echo "missing PROFILE_FILE=$PROFILE_FILE" >&2; exit 2; }
CAL_ROWS="${CAL_ROWS:-1}"; CAL_REBUILD="${CAL_REBUILD:-0}"; CAL_THREADS="${CAL_THREADS:-256}"
CAL_LOGDIR="${CAL_LOGDIR:-$ONEESAN_ROOT/work/b300_ilp8_async_cal_n27}"
CAL_SUMMARY="${CAL_SUMMARY:-$CAL_LOGDIR/summary.tsv}"; WINNER_ENV="${WINNER_ENV:-$CAL_LOGDIR/winner.env}"
TARGET_MIB="${TARGET_MIB:-65536}"; PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
SELECT_ONLY="${SELECT_ONLY:-1}"; REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"; ARCH="${ARCH:-native}"
RACE_PREFIX="${RACE_PREFIX:-$ONEESAN_ROOT/work/b300_ilp8_async_fullprime_race_n27}"
RUN_CAL="${RUN_CAL:-1}"
for x in CAL_REBUILD SELECT_ONLY REBUILD_BUCKETS RUN_CAL; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
[[ "$CAL_ROWS" =~ ^[1-9][0-9]*$ ]] && ((CAL_ROWS<=28)) || { echo 'CAL_ROWS must be 1..28' >&2; exit 2; }
[[ "$CAL_THREADS" =~ ^[0-9]+$ ]] && ((CAL_THREADS>=32&&CAL_THREADS<=768&&CAL_THREADS%32==0)) || { echo 'CAL_THREADS must be warp multiple 32..768 so cp.async candidate is legal' >&2; exit 2; }

if [[ "$RUN_CAL" == 1 ]]; then
  echo "=== calibrate sync/cp.async ILP8 rows=$CAL_ROWS threads=$CAL_THREADS ===" >&2
  N=27 ROWS="$CAL_ROWS" GRIDFP_THREADS="$CAL_THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" ARCH="$ARCH" REBUILD="$CAL_REBUILD" LOGDIR="$CAL_LOGDIR" SUMMARY="$CAL_SUMMARY" WINNER_ENV="$WINNER_ENV" \
    bash "$ONEESAN_ROOT/scripts/bench/b300x8-saturate-ilp8-ab.sh"
fi
[[ -f "$WINNER_ENV" && -f "$CAL_SUMMARY" ]] || { echo 'missing ILP8 calibration outputs' >&2; exit 3; }
# shellcheck disable=SC1090
source "$WINNER_ENV"
for n in B300_ILP8_WINNER_PROFILE B300_ILP8_WINNER_BIN B300_ILP8_WINNER_THREADS B300_ILP8_WINNER_SPILL_STORE_MAIN_BYTES B300_ILP8_WINNER_SPILL_LOAD_MAIN_BYTES; do [[ -n "${!n+x}" ]] || { echo "winner env missing $n" >&2; exit 3; }; done
[[ -x "$B300_ILP8_WINNER_BIN" ]] || { echo 'ILP8 winner binary missing' >&2; exit 3; }
[[ "$B300_ILP8_WINNER_SPILL_STORE_MAIN_BYTES" == 0 && "$B300_ILP8_WINNER_SPILL_LOAD_MAIN_BYTES" == 0 ]] || { echo 'refusing spilling ILP8 winner' >&2; exit 4; }

read -r BASE_BIN BASE_WALL < <(python3 - "$CAL_SUMMARY" <<'PY'
import csv,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
b=next((x for x in r if x['profile']=='ilp4'),None)
if not b: raise SystemExit('missing ilp4 calibration baseline')
print(b['binary'],b['wall_s'])
PY
)
[[ -x "$BASE_BIN" ]] || { echo 'ILP4 baseline binary missing' >&2; exit 3; }
mkdir -p "$(dirname "$RACE_PREFIX")" "${RACE_PREFIX}_adapters"

make_adapter(){
  local src="$1" threads="$2" label="$3" out="${RACE_PREFIX}_adapters/${label}.sh" sha
  sha="$(sha256sum "$src" | awk '{print $1}')"
  cat >"$out" <<EOF
#!/usr/bin/env bash
set -euo pipefail
# underlying_sha256=$sha
SRC=$(printf '%q' "$src")
THREADS=$(printf '%q' "$threads")
[[ \$# -ge 5 ]] || { echo 'usage: adapter N TARGET_MIB MAX_WINDOW NGPU MOD' >&2; exit 2; }
N=\$1; TARGET=\$2; MAXW=\$3; NGPU=\$4; MOD=\$5
export B300_ROW_LIMIT=28 GRIDFP_THREADS="\$THREADS" GRIDFP_PLAN_TARGET_MIB=$(printf '%q' "$PLAN_MIB")
exec "\$SRC" "\$N" "\$MOD" "\$TARGET" "\$MAXW" "\$NGPU"
EOF
  chmod +x "$out"
  printf '%s\n' "$out"
}
WIN_ADAPTER="$(make_adapter "$B300_ILP8_WINNER_BIN" "$B300_ILP8_WINNER_THREADS" "winner_${B300_ILP8_WINNER_PROFILE}")"
BASE_ADAPTER="$(make_adapter "$BASE_BIN" "$CAL_THREADS" ilp4_base)"

echo "=== full-prime race saturation winner=$B300_ILP8_WINNER_PROFILE partial_wall=${B300_ILP8_WINNER_WALL_S:-NA} partial_mc=${B300_ILP8_WINNER_MC_AVG_PCT:-NA}% vs ILP4 base + profiled warp/orbit ===" >&2
PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" MAX_WINDOW="$MAX_WINDOW" FORCED_TARGET_MIB="$TARGET_MIB" \
FORCED_OVERRIDE_BIN="$WIN_ADAPTER" FORCED_OVERRIDE_LABEL="sat_${B300_ILP8_WINNER_PROFILE}" FORCED_OVERRIDE_THREADS="$B300_ILP8_WINNER_THREADS" \
FORCED_BASE_BIN="$BASE_ADAPTER" FORCED_BASE_LABEL=sat_ilp4_base FORCED_BASE_THREADS="$CAL_THREADS" \
REBUILD_BUCKETS="$REBUILD_BUCKETS" SELECT_ONLY="$SELECT_ONLY" PREFIX="$RACE_PREFIX" \
  exec "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"
