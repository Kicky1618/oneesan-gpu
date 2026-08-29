#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'hybrid ILP8 staged race targets n=27' >&2; exit 2; }

PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
[[ -f "$PROFILE_FILE" ]] || { echo "missing PROFILE_FILE=$PROFILE_FILE" >&2; exit 2; }
ARCH="${ARCH:-native}"
MOD="${MOD:-4294967291}"
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${GRIDFP_THREADS:-256}"
RANDOM_CG="${RANDOM_CG:-1}"
WARP_SCAN="${WARP_SCAN:-1}"
SEARCH_THRESHOLDS="${ILP8_THRESHOLDS:-0 262144 524288 1048576 2097152 4194304 8388608}"
SEARCH_ROWS="${SEARCH_ROWS:-1}"
VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"
SEARCH_REPEATS="${SEARCH_REPEATS:-1}"
VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.01}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
REBUILD="${REBUILD:-0}"
RUN_STAGED="${RUN_STAGED:-1}"
SELECT_ONLY="${SELECT_ONLY:-1}"
REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_hybrid_ilp8_staged}"
FINAL_ENV="${FINAL_ENV:-${PREFIX}_winner.env}"
RACE_PREFIX="${RACE_PREFIX:-$ONEESAN_ROOT/work/b300_hybrid_ilp8_staged_fullprime_n27}"

for x in RANDOM_CG WARP_SCAN REBUILD RUN_STAGED SELECT_ONLY REBUILD_BUCKETS; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
[[ "$THREADS" =~ ^[0-9]+$ ]] && ((THREADS>=32&&THREADS<=1024&&THREADS%32==0)) || { echo 'GRIDFP_THREADS must be warp multiple 32..1024' >&2; exit 2; }
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY

run_stage(){
  local rows="$1" thresholds="$2" repeats="$3" tag="$4"
  local dir="${PREFIX}_${tag}" result="$dir/results.tsv" envfile="$dir/winner.env"
  mkdir -p "$dir"
  echo "=== hybrid staged rows=$rows thresholds=[$thresholds] repeats=$repeats ===" >&2
  N=27 MOD="$MOD" ROWS="$rows" GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
    RANDOM_CG="$RANDOM_CG" WARP_SCAN="$WARP_SCAN" ILP8_THRESHOLDS="$thresholds" REPEATS="$repeats" SAMPLE_INTERVAL="$SAMPLE_INTERVAL" REBUILD="$REBUILD" \
    LOGDIR="$dir" RESULT="$result" bash "$ONEESAN_ROOT/scripts/bench/b300x8-saturate-hybrid-ilp8-threshold-ab.sh"
  python3 "$ONEESAN_ROOT/scripts/bench/b300-hybrid-ilp8-export-winner.py" "$result" "$envfile" \
    --build-dir "$ONEESAN_BUILD_DIR" --threads "$THREADS" --random-cg "$RANDOM_CG" --warp-scan "$WARP_SCAN"
  printf '%s\n' "$envfile"
}

passes_margin(){
  python3 - "$1" "$MIN_SPEEDUP" <<'PY'
import sys
print(1 if float(sys.argv[1]) >= float(sys.argv[2]) else 0)
PY
}

threshold_pair(){
  local t="$1"
  if [[ "$t" == 0 ]]; then printf '0\n'; else printf '0 %s\n' "$t"; fi
}

if [[ "$RUN_STAGED" == 1 ]]; then
  search_env="$(run_stage "$SEARCH_ROWS" "$SEARCH_THRESHOLDS" "$SEARCH_REPEATS" search)"
  # shellcheck disable=SC1090
  source "$search_env"
  CURRENT_ENV="$search_env"
  TRANSFORM_OK="$B300_HYBRID_WINNER_TRANSFORMED"
  if [[ "$TRANSFORM_OK" == 1 && "$(passes_margin "$B300_HYBRID_WINNER_SPEEDUP_VS_ILP4")" != 1 ]]; then TRANSFORM_OK=0; fi

  if [[ "$TRANSFORM_OK" == 1 ]]; then
    current_threshold="$B300_HYBRID_WINNER_THRESHOLD"
    stage_index=0
    for rows in $VALIDATE_ROWS; do
      ((stage_index+=1))
      thresholds="$(threshold_pair "$current_threshold")"
      stage_env="$(run_stage "$rows" "$thresholds" "$VALIDATE_REPEATS" "validate${stage_index}_r${rows}")"
      # shellcheck disable=SC1090
      source "$stage_env"
      CURRENT_ENV="$stage_env"
      if [[ "$B300_HYBRID_WINNER_TRANSFORMED" != 1 || "$(passes_margin "$B300_HYBRID_WINNER_SPEEDUP_VS_ILP4")" != 1 ]]; then
        TRANSFORM_OK=0
        break
      fi
      [[ "$B300_HYBRID_WINNER_SPILL_FREE" == 1 ]] || { echo 'staged hybrid selector returned spilling transform' >&2; exit 4; }
      current_threshold="$B300_HYBRID_WINNER_THRESHOLD"
    done
  fi

  # shellcheck disable=SC1090
  source "$CURRENT_ENV"
  if [[ "$TRANSFORM_OK" == 0 ]]; then
    B300_HYBRID_WINNER_MODE=ilp4
    B300_HYBRID_WINNER_THRESHOLD=NA
    B300_HYBRID_WINNER_BIN="$B300_HYBRID_BASE_BIN"
    B300_HYBRID_WINNER_THREADS="$B300_HYBRID_BASE_THREADS"
    B300_HYBRID_WINNER_TRANSFORMED=0
    B300_HYBRID_WINNER_SPILL_FREE=0
  fi
  {
    printf 'B300_HYBRID_STAGED_VALIDATED=%q\n' "$TRANSFORM_OK"
    printf 'B300_HYBRID_WINNER_MODE=%q\n' "$B300_HYBRID_WINNER_MODE"
    printf 'B300_HYBRID_WINNER_THRESHOLD=%q\n' "$B300_HYBRID_WINNER_THRESHOLD"
    printf 'B300_HYBRID_WINNER_BIN=%q\n' "$B300_HYBRID_WINNER_BIN"
    printf 'B300_HYBRID_WINNER_THREADS=%q\n' "$B300_HYBRID_WINNER_THREADS"
    printf 'B300_HYBRID_WINNER_TRANSFORMED=%q\n' "$B300_HYBRID_WINNER_TRANSFORMED"
    printf 'B300_HYBRID_WINNER_SPILL_FREE=%q\n' "$B300_HYBRID_WINNER_SPILL_FREE"
    printf 'B300_HYBRID_BASE_BIN=%q\n' "$B300_HYBRID_BASE_BIN"
    printf 'B300_HYBRID_BASE_THREADS=%q\n' "$B300_HYBRID_BASE_THREADS"
    printf 'B300_HYBRID_MIN_SPEEDUP=%q\n' "$MIN_SPEEDUP"
  } >"$FINAL_ENV"
fi

[[ -f "$FINAL_ENV" ]] || { echo "missing FINAL_ENV=$FINAL_ENV" >&2; exit 3; }
# shellcheck disable=SC1090
source "$FINAL_ENV"
for n in B300_HYBRID_STAGED_VALIDATED B300_HYBRID_WINNER_MODE B300_HYBRID_WINNER_BIN B300_HYBRID_WINNER_THREADS B300_HYBRID_WINNER_TRANSFORMED B300_HYBRID_BASE_BIN B300_HYBRID_BASE_THREADS; do
  [[ -n "${!n+x}" ]] || { echo "final hybrid env missing $n" >&2; exit 3; }
done
[[ -x "$B300_HYBRID_WINNER_BIN" && -x "$B300_HYBRID_BASE_BIN" ]] || { echo 'hybrid final/base binary missing' >&2; exit 3; }
if [[ "$B300_HYBRID_WINNER_TRANSFORMED" == 1 ]]; then
  [[ "$B300_HYBRID_STAGED_VALIDATED" == 1 && "$B300_HYBRID_WINNER_SPILL_FREE" == 1 ]] || { echo 'refusing unvalidated/spilling hybrid transform' >&2; exit 4; }
fi

mkdir -p "${RACE_PREFIX}_adapters"
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

WIN_ADAPTER="$(make_adapter "$B300_HYBRID_WINNER_BIN" "$B300_HYBRID_WINNER_THREADS" "winner_${B300_HYBRID_WINNER_MODE}_${B300_HYBRID_WINNER_THRESHOLD}")"
BASE_ADAPTER="$(make_adapter "$B300_HYBRID_BASE_BIN" "$B300_HYBRID_BASE_THREADS" ilp4_base)"
label="hybrid_${B300_HYBRID_WINNER_MODE}_t${B300_HYBRID_WINNER_THRESHOLD}"
echo "=== hybrid full-prime race staged=$B300_HYBRID_STAGED_VALIDATED candidate=$label ===" >&2

base_args=()
if [[ "$B300_HYBRID_WINNER_TRANSFORMED" == 1 ]]; then
  base_args=(FORCED_BASE_BIN="$BASE_ADAPTER" FORCED_BASE_LABEL=sat_ilp4_base FORCED_BASE_THREADS="$B300_HYBRID_BASE_THREADS")
fi
env PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" MAX_WINDOW="$MAX_WINDOW" FORCED_TARGET_MIB="$TARGET_MIB" \
  FORCED_OVERRIDE_BIN="$WIN_ADAPTER" FORCED_OVERRIDE_LABEL="$label" FORCED_OVERRIDE_THREADS="$B300_HYBRID_WINNER_THREADS" \
  "${base_args[@]}" REBUILD_BUCKETS="$REBUILD_BUCKETS" SELECT_ONLY="$SELECT_ONLY" PREFIX="$RACE_PREFIX" \
  exec "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"
