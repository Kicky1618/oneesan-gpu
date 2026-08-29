#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"
DISTANCE_LIST="${DISTANCE_LIST:-1 2 4}"; SEARCH_ROWS="${SEARCH_ROWS:-1}"; VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"
SEARCH_REPEATS="${SEARCH_REPEATS:-1}"; VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"; MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"
RUN_STAGE_F="${RUN_STAGE_F:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_distance_staged}"
STAGE_F_PREFIX="${STAGE_F_PREFIX:-${PREFIX}.stagef}"; STAGE_F_ENV="${STAGE_F_ENV:-${STAGE_F_PREFIX}_winner.env}"
FINAL_ENV="${FINAL_ENV:-${PREFIX}_winner.env}"
mkdir -p "$(dirname "$PREFIX")" "$(dirname "$FINAL_ENV")"
[[ "$RUN_STAGE_F" == 0 || "$RUN_STAGE_F" == 1 ]] || exit 2
for n in SEARCH_REPEATS VALIDATE_REPEATS; do v="${!n}"; [[ "$v" =~ ^[1-9][0-9]*$ ]] || exit 2; done
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY
if [[ "$RUN_STAGE_F" == 1 ]]; then
  echo '=== Stage G prerequisite: Stage F width calibration ===' >&2
  ARCH="$ARCH" MOD="$MOD" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" PREFIX="$STAGE_F_PREFIX" FINAL_ENV="$STAGE_F_ENV" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-nextself-staged-calibrate.sh"
fi
[[ -s "$STAGE_F_ENV" ]] || { echo "missing Stage-F env=$STAGE_F_ENV" >&2; exit 3; }
# shellcheck disable=SC1090
source "$STAGE_F_ENV"
for k in B300_HYBRID8_NEXTSELF_STAGED_VALIDATED B300_HYBRID8_NEXTSELF_FINAL_ENABLED B300_HYBRID8_NEXTSELF_FINAL_WIDTH \
  B300_HYBRID8_NEXTSELF_STAGE_E_CORE_ROWS B300_HYBRID8_NEXTSELF_STAGE_E_CORE_RESIDUE B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_RESIDUE \
  B300_HYBRID8_NEXTSELF_THRESHOLD B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK B300_HYBRID8_NEXTSELF_RANDOM_CG B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES \
  B300_HYBRID8_NEXTSELF_PREFETCH_L2 B300_HYBRID8_NEXTSELF_DUALMASK B300_HYBRID8_NEXTSELF_CLOSURE_BATCH B300_HYBRID8_NEXTSELF_MAXRREGCOUNT; do
  [[ -n "${!k+x}" ]] || { echo "Stage-F env missing $k" >&2; exit 3; }
done
[[ "$B300_HYBRID8_NEXTSELF_STAGED_VALIDATED" == 1 && "$B300_HYBRID8_NEXTSELF_FINAL_ENABLED" == 1 ]] || { echo 'Stage F not promotable' >&2; exit 4; }
WIDTH="$B300_HYBRID8_NEXTSELF_FINAL_WIDTH"; case "$WIDTH" in 1|2|4|8);;*) exit 3;; esac
THRESHOLD="$B300_HYBRID8_NEXTSELF_THRESHOLD"; H="$B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK"; CG="$B300_HYBRID8_NEXTSELF_RANDOM_CG"; CGL2="$B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES"; PRE="$B300_HYBRID8_NEXTSELF_PREFETCH_L2"; DUAL="$B300_HYBRID8_NEXTSELF_DUALMASK"; BATCH="$B300_HYBRID8_NEXTSELF_CLOSURE_BATCH"; CAP="$B300_HYBRID8_NEXTSELF_MAXRREGCOUNT"

stage_validate_rows=()
for rows in $VALIDATE_ROWS "$B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS"; do
  [[ "$rows" == "$SEARCH_ROWS" ]] && continue
  seen=0; for old in "${stage_validate_rows[@]}"; do [[ "$old" == "$rows" ]] && seen=1; done; ((seen)) || stage_validate_rows+=("$rows")
done
check_residue(){
  local rows="$1" got="$2"
  if [[ "$rows" == "$B300_HYBRID8_NEXTSELF_STAGE_E_CORE_ROWS" && "$got" != "$B300_HYBRID8_NEXTSELF_STAGE_E_CORE_RESIDUE" ]]; then echo "FATAL Stage-G/core residue mismatch rows=$rows got=$got" >&2; exit 4; fi
  if [[ "$rows" == "$B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS" && "$got" != "$B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_RESIDUE" ]]; then echo "FATAL Stage-G/final residue mismatch rows=$rows got=$got" >&2; exit 4; fi
}
run_stage(){
  local rows="$1" threads="$2" repeats="$3" tag="$4" distances="$5"
  local p="${PREFIX}.${tag}.r${rows}" log="${p}.log" env="${p}_winner.env"
  echo "=== Stage G width=$WIDTH rows=$rows distances=[$distances] threads=[$threads] repeats=$repeats ===" >&2
  ARCH="$ARCH" MOD="$MOD" ROWS="$rows" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" HIGH_DROP_CHUNK="$H" HYBRID_THRESHOLD="$THRESHOLD" NEXTSELF_WIDTH="$WIDTH" DISTANCE_LIST="$distances" \
    RANDOM_CG="$CG" RANDOM_CG_L2_FETCH_BYTES="$CGL2" PREFETCH_L2="$PRE" DUALMASK="$DUAL" CLOSURE_BATCH="$BATCH" MAXRREGCOUNT="$CAP" \
    THREADS_LIST="$threads" REPEATS="$repeats" PREFIX="$p" WINNER_ENV="$env" bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-nextself-distance-sweep.sh" | tee "$log" >&2
  grep -Fq 'b300_nextgen_hybrid8_nextself_distance_exact_match=1' "$log" || exit 4
  [[ -s "$env" ]] || exit 4
  printf '%s\n' "$env"
}
passes(){ python3 - "$1" "$MIN_SPEEDUP" <<'PY'
import sys
print(1 if float(sys.argv[1])>=float(sys.argv[2]) else 0)
PY
}

SEARCH_ENV="$(run_stage "$SEARCH_ROWS" '128 256 512' "$SEARCH_REPEATS" search "$DISTANCE_LIST")"
# shellcheck disable=SC1090
source "$SEARCH_ENV"
check_residue "$SEARCH_ROWS" "$B300_HYBRID8_NEXTSELF_DISTANCE_RESIDUE"
SELECTED_DISTANCE="$B300_HYBRID8_NEXTSELF_DISTANCE_BEST"
VALIDATED=0
case "$SELECTED_DISTANCE" in 1) ;; 2|4)
  if [[ "$B300_HYBRID8_NEXTSELF_DISTANCE_REFERENCE_SPILL_FREE" == 1 && "$B300_HYBRID8_NEXTSELF_DISTANCE_SPILL_FREE" == 1 && "$(passes "$B300_HYBRID8_NEXTSELF_DISTANCE_SPEEDUP_VS_D1")" == 1 ]]; then VALIDATED=1; fi
  ;; *) echo 'invalid Stage-G selected distance' >&2; exit 4;; esac
CURRENT_ENV="$SEARCH_ENV"
if [[ "$VALIDATED" == 1 ]]; then
  ref_threads="$B300_HYBRID8_NEXTSELF_DISTANCE_REFERENCE_THREADS"; test_threads="$B300_HYBRID8_NEXTSELF_DISTANCE_THREADS"; validation_threads="$ref_threads"; [[ "$test_threads" == "$ref_threads" ]] || validation_threads+=" $test_threads"
  stage=0
  for rows in "${stage_validate_rows[@]}"; do
    ((stage+=1)); CURRENT_ENV="$(run_stage "$rows" "$validation_threads" "$VALIDATE_REPEATS" "validate${stage}" "1 $SELECTED_DISTANCE")"
    # shellcheck disable=SC1090
    source "$CURRENT_ENV"; check_residue "$rows" "$B300_HYBRID8_NEXTSELF_DISTANCE_RESIDUE"
    if [[ "$B300_HYBRID8_NEXTSELF_DISTANCE_BEST" != "$SELECTED_DISTANCE" || "$B300_HYBRID8_NEXTSELF_DISTANCE_REFERENCE_SPILL_FREE" != 1 || "$B300_HYBRID8_NEXTSELF_DISTANCE_SPILL_FREE" != 1 || "$(passes "$B300_HYBRID8_NEXTSELF_DISTANCE_SPEEDUP_VS_D1")" != 1 ]]; then VALIDATED=0; break; fi
    ref_threads="$B300_HYBRID8_NEXTSELF_DISTANCE_REFERENCE_THREADS"; test_threads="$B300_HYBRID8_NEXTSELF_DISTANCE_THREADS"; validation_threads="$ref_threads"; [[ "$test_threads" == "$ref_threads" ]] || validation_threads+=" $test_threads"
  done
fi
# shellcheck disable=SC1090
source "$CURRENT_ENV"
if [[ "$VALIDATED" == 1 ]]; then FINAL_DISTANCE="$SELECTED_DISTANCE"; FINAL_BIN="$B300_HYBRID8_NEXTSELF_DISTANCE_BIN"; FINAL_THREADS="$B300_HYBRID8_NEXTSELF_DISTANCE_THREADS"; FINAL_WALL="$B300_HYBRID8_NEXTSELF_DISTANCE_WALL_S"; FINAL_SPEED="$B300_HYBRID8_NEXTSELF_DISTANCE_SPEEDUP_VS_D1"; else FINAL_DISTANCE=1; FINAL_BIN="$B300_HYBRID8_NEXTSELF_DISTANCE_REFERENCE_BIN"; FINAL_THREADS="$B300_HYBRID8_NEXTSELF_DISTANCE_REFERENCE_THREADS"; FINAL_WALL="$B300_HYBRID8_NEXTSELF_DISTANCE_REFERENCE_WALL_S"; FINAL_SPEED=1.000000000; fi
{
 printf 'B300_HYBRID8_NEXTSELF_DISTANCE_STAGED_VALIDATED=%q\n' "$VALIDATED"
 printf 'B300_HYBRID8_NEXTSELF_DISTANCE_FINAL_DISTANCE=%q\n' "$FINAL_DISTANCE"
 printf 'B300_HYBRID8_NEXTSELF_DISTANCE_FINAL_WIDTH=%q\n' "$WIDTH"
 printf 'B300_HYBRID8_NEXTSELF_DISTANCE_FINAL_BIN=%q\n' "$FINAL_BIN"
 printf 'B300_HYBRID8_NEXTSELF_DISTANCE_FINAL_THREADS=%q\n' "$FINAL_THREADS"
 printf 'B300_HYBRID8_NEXTSELF_DISTANCE_FINAL_WALL_S=%q\n' "$FINAL_WALL"
 printf 'B300_HYBRID8_NEXTSELF_DISTANCE_FINAL_SPEEDUP_VS_D1=%q\n' "$FINAL_SPEED"
 printf 'B300_HYBRID8_NEXTSELF_DISTANCE_FINAL_SPILL_FREE=1\n'
 printf 'B300_HYBRID8_NEXTSELF_DISTANCE_REFERENCE_BIN=%q\n' "$B300_HYBRID8_NEXTSELF_DISTANCE_REFERENCE_BIN"
 printf 'B300_HYBRID8_NEXTSELF_DISTANCE_REFERENCE_THREADS=%q\n' "$B300_HYBRID8_NEXTSELF_DISTANCE_REFERENCE_THREADS"
 printf 'B300_HYBRID8_NEXTSELF_DISTANCE_FINAL_STAGE_ROWS=%q\n' "$B300_HYBRID8_NEXTSELF_DISTANCE_ROWS"
 printf 'B300_HYBRID8_NEXTSELF_DISTANCE_FINAL_STAGE_RESIDUE=%q\n' "$B300_HYBRID8_NEXTSELF_DISTANCE_RESIDUE"
 printf 'B300_HYBRID8_NEXTSELF_DISTANCE_THRESHOLD=%q\n' "$THRESHOLD"
 printf 'B300_HYBRID8_NEXTSELF_DISTANCE_STAGE_F_ENV=%q\n' "$STAGE_F_ENV"
 printf 'B300_HYBRID8_NEXTSELF_DISTANCE_MIN_SPEEDUP=%q\n' "$MIN_SPEEDUP"
 printf 'B300_HYBRID8_NEXTSELF_DISTANCE_SEARCH_LIST=%q\n' "$DISTANCE_LIST"
} >"$FINAL_ENV"
cat "$FINAL_ENV"
echo "b300-nextgen-hybrid8-nextself-distance-staged-calibrate OK validated=$VALIDATED width=$WIDTH distance=$FINAL_DISTANCE speedup_vs_d1=$FINAL_SPEED final_rows=$B300_HYBRID8_NEXTSELF_DISTANCE_ROWS"
