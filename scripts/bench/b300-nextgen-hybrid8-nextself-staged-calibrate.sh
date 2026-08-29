#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
MOD="${MOD:-4294967291}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
SEARCH_ROWS="${SEARCH_ROWS:-1}"
VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"
SEARCH_REPEATS="${SEARCH_REPEATS:-1}"
VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.01}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
RUN_HYBRID_STAGE="${RUN_HYBRID_STAGE:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged}"
HYBRID_PREFIX="${HYBRID_PREFIX:-${PREFIX}.hybrid8}"
HYBRID_WINNER_ENV="${HYBRID_WINNER_ENV:-${HYBRID_PREFIX}_winner.env}"
FINAL_ENV="${FINAL_ENV:-${PREFIX}_winner.env}"
mkdir -p "$(dirname "$PREFIX")" "$(dirname "$FINAL_ENV")"

[[ "$RUN_HYBRID_STAGE" == 0 || "$RUN_HYBRID_STAGE" == 1 ]] || { echo 'RUN_HYBRID_STAGE must be 0/1' >&2; exit 2; }
for rows in "$SEARCH_ROWS" $VALIDATE_ROWS; do
  [[ "$rows" =~ ^[1-9][0-9]*$ ]] && ((rows<=28)) || { echo "bad stage rows=$rows" >&2; exit 2; }
done
for n in SEARCH_REPEATS VALIDATE_REPEATS; do
  v="${!n}"; [[ "$v" =~ ^[1-9][0-9]*$ ]] || { echo "$n must be >=1" >&2; exit 2; }
done
python3 - "$MIN_SPEEDUP" "$SAMPLE_INTERVAL" <<'PY'
import sys
if float(sys.argv[1])<1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
if float(sys.argv[2])<=0: raise SystemExit('SAMPLE_INTERVAL must be >0')
PY

if [[ "$RUN_HYBRID_STAGE" == 1 ]]; then
  echo '=== Stage F prerequisite: calibrate/validate hybrid8 threshold ===' >&2
  PREFIX="$HYBRID_PREFIX" FINAL_ENV="$HYBRID_WINNER_ENV" ARCH="$ARCH" MOD="$MOD" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-staged-calibrate.sh"
fi
[[ -s "$HYBRID_WINNER_ENV" ]] || { echo "missing hybrid Stage-E env=$HYBRID_WINNER_ENV" >&2; exit 3; }
# shellcheck disable=SC1090
source "$HYBRID_WINNER_ENV"
for key in \
  B300_HYBRID8_STAGED_VALIDATED B300_HYBRID8_FINAL_ENABLED B300_HYBRID8_FINAL_THRESHOLD B300_HYBRID8_FINAL_SPILL_FREE \
  B300_HYBRID8_CORE_ROWS B300_HYBRID8_RESIDUE B300_HYBRID8_FINAL_STAGE_ROWS B300_HYBRID8_FINAL_STAGE_RESIDUE \
  B300_HYBRID8_HIGH_DROP_CHUNK B300_HYBRID8_RANDOM_CG B300_HYBRID8_RANDOM_CG_L2_FETCH_BYTES \
  B300_HYBRID8_PREFETCH_L2 B300_HYBRID8_DUALMASK B300_HYBRID8_CLOSURE_BATCH B300_HYBRID8_MAXRREGCOUNT; do
  [[ -n "${!key+x}" ]] || { echo "hybrid Stage-E env missing $key" >&2; exit 3; }
done
[[ "$B300_HYBRID8_STAGED_VALIDATED" == 1 && "$B300_HYBRID8_FINAL_ENABLED" == 1 && "$B300_HYBRID8_FINAL_SPILL_FREE" == 1 ]] || {
  echo 'hybrid8 did not survive Stage E; Stage F is not applicable' >&2
  exit 4
}
for rows in "$B300_HYBRID8_CORE_ROWS" "$B300_HYBRID8_FINAL_STAGE_ROWS"; do
  [[ "$rows" =~ ^[1-9][0-9]*$ ]] && ((rows<=28)) || { echo "bad Stage-E row proof=$rows" >&2; exit 3; }
done
[[ "$B300_HYBRID8_RESIDUE" =~ ^[0-9]+$ && "$B300_HYBRID8_FINAL_STAGE_RESIDUE" =~ ^[0-9]+$ ]] || {
  echo 'bad Stage-E residue proof' >&2; exit 3;
}
THRESHOLD="$B300_HYBRID8_FINAL_THRESHOLD"
H="$B300_HYBRID8_HIGH_DROP_CHUNK"
CG="$B300_HYBRID8_RANDOM_CG"
CGL2="$B300_HYBRID8_RANDOM_CG_L2_FETCH_BYTES"
PRE="$B300_HYBRID8_PREFETCH_L2"
DUAL="$B300_HYBRID8_DUALMASK"
BATCH="$B300_HYBRID8_CLOSURE_BATCH"
CAP="$B300_HYBRID8_MAXRREGCOUNT"

# Never let a custom validation list accidentally omit the largest Stage-E
# slice that justified promotion of the plain hybrid control.
stage_validate_rows=()
for rows in $VALIDATE_ROWS "$B300_HYBRID8_FINAL_STAGE_ROWS"; do
  [[ "$rows" == "$SEARCH_ROWS" ]] && continue
  seen=0
  for old in "${stage_validate_rows[@]}"; do [[ "$old" == "$rows" ]] && seen=1; done
  ((seen)) || stage_validate_rows+=("$rows")
done

check_stage_e_residue(){
  local rows="$1" got="$2"
  if [[ "$rows" == "$B300_HYBRID8_CORE_ROWS" && "$got" != "$B300_HYBRID8_RESIDUE" ]]; then
    echo "FATAL Stage-F/core Stage-E residue mismatch rows=$rows got=$got expected=$B300_HYBRID8_RESIDUE" >&2
    exit 4
  fi
  if [[ "$rows" == "$B300_HYBRID8_FINAL_STAGE_ROWS" && "$got" != "$B300_HYBRID8_FINAL_STAGE_RESIDUE" ]]; then
    echo "FATAL Stage-F/final Stage-E residue mismatch rows=$rows got=$got expected=$B300_HYBRID8_FINAL_STAGE_RESIDUE" >&2
    exit 4
  fi
}

run_stage(){
  local rows="$1" threads="$2" repeats="$3" tag="$4"
  local p="${PREFIX}.${tag}.r${rows}" log="${p}.log" env="${p}_winner.env"
  echo "=== Stage F hybrid8 next-self rows=$rows threads=[$threads] repeats=$repeats threshold=$THRESHOLD ===" >&2
  ARCH="$ARCH" MOD="$MOD" ROWS="$rows" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
    HIGH_DROP_CHUNK="$H" HYBRID_THRESHOLD="$THRESHOLD" RANDOM_CG="$CG" RANDOM_CG_L2_FETCH_BYTES="$CGL2" \
    PREFETCH_L2="$PRE" DUALMASK="$DUAL" CLOSURE_BATCH="$BATCH" MAXRREGCOUNT="$CAP" \
    THREADS_LIST="$threads" REPEATS="$repeats" SAMPLE_INTERVAL="$SAMPLE_INTERVAL" PREFIX="$p" WINNER_ENV="$env" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-nextself-ab.sh" | tee "$log" >&2
  grep -Fq 'b300_nextgen_hybrid8_nextself_exact_intermediate_match=1' "$log" || { echo "Stage F exact gate missing rows=$rows" >&2; exit 4; }
  [[ -s "$env" ]] || { echo "Stage F env missing rows=$rows" >&2; exit 4; }
  printf '%s\n' "$env"
}
passes(){
  python3 - "$1" "$MIN_SPEEDUP" <<'PY'
import sys
print(1 if float(sys.argv[1])>=float(sys.argv[2]) else 0)
PY
}

SEARCH_ENV="$(run_stage "$SEARCH_ROWS" "$THREADS_LIST" "$SEARCH_REPEATS" search)"
# shellcheck disable=SC1090
source "$SEARCH_ENV"
check_stage_e_residue "$SEARCH_ROWS" "$B300_HYBRID8_NEXTSELF_RESIDUE"
VALIDATED=0
CURRENT_ENV="$SEARCH_ENV"
if [[ "$B300_HYBRID8_NEXTSELF_BEST_ENABLED" == 1 && "$B300_HYBRID8_NEXTSELF_CONTROL_SPILL_FREE" == 1 && "$B300_HYBRID8_NEXTSELF_SPILL_FREE" == 1 && "$(passes "$B300_HYBRID8_NEXTSELF_SPEEDUP")" == 1 ]]; then
  VALIDATED=1
  control_threads="$B300_HYBRID8_NEXTSELF_CONTROL_THREADS"
  test_threads="$B300_HYBRID8_NEXTSELF_THREADS"
  validation_threads="$control_threads"
  [[ "$test_threads" == "$control_threads" ]] || validation_threads+=" $test_threads"
  stage=0
  for rows in "${stage_validate_rows[@]}"; do
    ((stage+=1))
    CURRENT_ENV="$(run_stage "$rows" "$validation_threads" "$VALIDATE_REPEATS" "validate${stage}")"
    # shellcheck disable=SC1090
    source "$CURRENT_ENV"
    check_stage_e_residue "$rows" "$B300_HYBRID8_NEXTSELF_RESIDUE"
    if [[ "$B300_HYBRID8_NEXTSELF_BEST_ENABLED" != 1 || "$B300_HYBRID8_NEXTSELF_CONTROL_SPILL_FREE" != 1 || "$B300_HYBRID8_NEXTSELF_SPILL_FREE" != 1 || "$(passes "$B300_HYBRID8_NEXTSELF_SPEEDUP")" != 1 ]]; then
      VALIDATED=0
      break
    fi
    control_threads="$B300_HYBRID8_NEXTSELF_CONTROL_THREADS"
    test_threads="$B300_HYBRID8_NEXTSELF_THREADS"
    validation_threads="$control_threads"
    [[ "$test_threads" == "$control_threads" ]] || validation_threads+=" $test_threads"
  done
fi

# shellcheck disable=SC1090
source "$CURRENT_ENV"
if [[ "$VALIDATED" == 1 ]]; then
  FINAL_ENABLED=1
  FINAL_BIN="$B300_HYBRID8_NEXTSELF_BIN"
  FINAL_THREADS="$B300_HYBRID8_NEXTSELF_THREADS"
  FINAL_WALL="$B300_HYBRID8_NEXTSELF_WALL_S"
  FINAL_SPEED="$B300_HYBRID8_NEXTSELF_SPEEDUP"
else
  FINAL_ENABLED=0
  FINAL_BIN="$B300_HYBRID8_NEXTSELF_CONTROL_BIN"
  FINAL_THREADS="$B300_HYBRID8_NEXTSELF_CONTROL_THREADS"
  FINAL_WALL="$B300_HYBRID8_NEXTSELF_CONTROL_WALL_S"
  FINAL_SPEED=1.000000000
fi

{
  printf 'B300_HYBRID8_NEXTSELF_STAGED_VALIDATED=%q\n' "$VALIDATED"
  printf 'B300_HYBRID8_NEXTSELF_FINAL_ENABLED=%q\n' "$FINAL_ENABLED"
  printf 'B300_HYBRID8_NEXTSELF_FINAL_BIN=%q\n' "$FINAL_BIN"
  printf 'B300_HYBRID8_NEXTSELF_FINAL_THREADS=%q\n' "$FINAL_THREADS"
  printf 'B300_HYBRID8_NEXTSELF_FINAL_WALL_S=%q\n' "$FINAL_WALL"
  printf 'B300_HYBRID8_NEXTSELF_FINAL_SPEEDUP_VS_HYBRID8=%q\n' "$FINAL_SPEED"
  printf 'B300_HYBRID8_NEXTSELF_FINAL_SPILL_FREE=1\n'
  printf 'B300_HYBRID8_NEXTSELF_CONTROL_BIN=%q\n' "$B300_HYBRID8_NEXTSELF_CONTROL_BIN"
  printf 'B300_HYBRID8_NEXTSELF_CONTROL_THREADS=%q\n' "$B300_HYBRID8_NEXTSELF_CONTROL_THREADS"
  printf 'B300_HYBRID8_NEXTSELF_CONTROL_WALL_S=%q\n' "$B300_HYBRID8_NEXTSELF_CONTROL_WALL_S"
  printf 'B300_HYBRID8_NEXTSELF_CONTROL_SPILL_FREE=1\n'
  printf 'B300_HYBRID8_NEXTSELF_FINAL_STAGE_ROWS=%q\n' "$B300_HYBRID8_NEXTSELF_ROWS"
  printf 'B300_HYBRID8_NEXTSELF_FINAL_STAGE_RESIDUE=%q\n' "$B300_HYBRID8_NEXTSELF_RESIDUE"
  printf 'B300_HYBRID8_NEXTSELF_STAGE_E_CORE_ROWS=%q\n' "$B300_HYBRID8_CORE_ROWS"
  printf 'B300_HYBRID8_NEXTSELF_STAGE_E_CORE_RESIDUE=%q\n' "$B300_HYBRID8_RESIDUE"
  printf 'B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS=%q\n' "$B300_HYBRID8_FINAL_STAGE_ROWS"
  printf 'B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_RESIDUE=%q\n' "$B300_HYBRID8_FINAL_STAGE_RESIDUE"
  printf 'B300_HYBRID8_NEXTSELF_THRESHOLD=%q\n' "$THRESHOLD"
  printf 'B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK=%q\n' "$H"
  printf 'B300_HYBRID8_NEXTSELF_RANDOM_CG=%q\n' "$CG"
  printf 'B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES=%q\n' "$CGL2"
  printf 'B300_HYBRID8_NEXTSELF_PREFETCH_L2=%q\n' "$PRE"
  printf 'B300_HYBRID8_NEXTSELF_DUALMASK=%q\n' "$DUAL"
  printf 'B300_HYBRID8_NEXTSELF_CLOSURE_BATCH=%q\n' "$BATCH"
  printf 'B300_HYBRID8_NEXTSELF_MAXRREGCOUNT=%q\n' "$CAP"
  printf 'B300_HYBRID8_NEXTSELF_MIN_SPEEDUP=%q\n' "$MIN_SPEEDUP"
  printf 'B300_HYBRID8_NEXTSELF_STAGE_E_ENV=%q\n' "$HYBRID_WINNER_ENV"
} >"$FINAL_ENV"

cat "$FINAL_ENV"
echo "b300-nextgen-hybrid8-nextself-staged-calibrate OK validated=$VALIDATED threshold=$THRESHOLD final_rows=$B300_HYBRID8_NEXTSELF_ROWS winner_env=$FINAL_ENV spill_proof=1 stage_e_crosscheck=1" >&2
