#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

INPUT_ENV="${INPUT_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
ARCH="${ARCH:-native}"
MOD="${MOD:-4294967291}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
SEARCH_ROWS="${SEARCH_ROWS:-1}"
VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"
SEARCH_REPEATS="${SEARCH_REPEATS:-1}"
VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"
MATE_WIDTH_LIST="${MATE_WIDTH_LIST:-1 2 4 8}"
MATE_DISTANCE_LIST="${MATE_DISTANCE_LIST:-1 2 4}"
SELF_EVICT="${SELF_EVICT:-default}"
MATE_EVICT="${MATE_EVICT:-default}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextmate_geometry_staged}"
FINAL_ENV="${FINAL_ENV:-${PREFIX}_winner.env}"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-nextmate-geometry-sweep.sh"
mkdir -p "$(dirname "$PREFIX")" "$(dirname "$FINAL_ENV")"
[[ -s "$INPUT_ENV" ]] || { echo "missing Stage-F INPUT_ENV=$INPUT_ENV" >&2; exit 2; }
for n in SEARCH_REPEATS VALIDATE_REPEATS; do v="${!n}"; [[ "$v" =~ ^[1-9][0-9]*$ ]] || { echo "$n must be >=1" >&2; exit 2; }; done
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY
for ev in SELF_EVICT MATE_EVICT; do case "${!ev}" in default|normal|last) ;; *) exit 2;; esac; done

# shellcheck disable=SC1090
source "$INPUT_ENV"
for k in \
  B300_HYBRID8_NEXTSELF_STAGED_VALIDATED B300_HYBRID8_NEXTSELF_FINAL_ENABLED \
  B300_HYBRID8_NEXTSELF_FINAL_WIDTH B300_HYBRID8_NEXTSELF_FINAL_DISTANCE \
  B300_HYBRID8_NEXTSELF_FINAL_STAGE_ROWS B300_HYBRID8_NEXTSELF_FINAL_STAGE_RESIDUE \
  B300_HYBRID8_NEXTSELF_STAGE_E_CORE_ROWS B300_HYBRID8_NEXTSELF_STAGE_E_CORE_RESIDUE \
  B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_RESIDUE; do
  [[ -n "${!k+x}" ]] || { echo "Stage-F env missing $k" >&2; exit 3; }
done
[[ "$B300_HYBRID8_NEXTSELF_STAGED_VALIDATED" == 1 && "$B300_HYBRID8_NEXTSELF_FINAL_ENABLED" == 1 ]] || {
  echo 'Stage F not promotable' >&2; exit 4;
}
SELF_W="$B300_HYBRID8_NEXTSELF_FINAL_WIDTH"
SELF_D="$B300_HYBRID8_NEXTSELF_FINAL_DISTANCE"

stage_validate_rows=()
for rows in $VALIDATE_ROWS "$B300_HYBRID8_NEXTSELF_FINAL_STAGE_ROWS" "$B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS"; do
  [[ "$rows" == "$SEARCH_ROWS" ]] && continue
  [[ "$rows" =~ ^[1-9][0-9]*$ ]] && ((rows<=28)) || { echo "bad validation rows=$rows" >&2; exit 2; }
  seen=0; for old in "${stage_validate_rows[@]}"; do [[ "$old" == "$rows" ]] && seen=1; done
  ((seen)) || stage_validate_rows+=("$rows")
done
check_known_residue(){
  local rows="$1" got="$2"
  if [[ "$rows" == "$B300_HYBRID8_NEXTSELF_STAGE_E_CORE_ROWS" && "$got" != "$B300_HYBRID8_NEXTSELF_STAGE_E_CORE_RESIDUE" ]]; then
    echo "FATAL Stage-I/core residue mismatch rows=$rows got=$got" >&2; exit 4
  fi
  if [[ "$rows" == "$B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS" && "$got" != "$B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_RESIDUE" ]]; then
    echo "FATAL Stage-I/Stage-E-final residue mismatch rows=$rows got=$got" >&2; exit 4
  fi
  if [[ "$rows" == "$B300_HYBRID8_NEXTSELF_FINAL_STAGE_ROWS" && "$got" != "$B300_HYBRID8_NEXTSELF_FINAL_STAGE_RESIDUE" ]]; then
    echo "FATAL Stage-I/Stage-F-final residue mismatch rows=$rows got=$got" >&2; exit 4
  fi
}
passes(){ python3 - "$1" "$MIN_SPEEDUP" <<'PY'
import sys
print(1 if float(sys.argv[1]) >= float(sys.argv[2]) else 0)
PY
}
run_stage(){
  local rows="$1" widths="$2" distances="$3" threads="$4" repeats="$5" tag="$6"
  local p="${PREFIX}.${tag}.r${rows}" log="${p}.log" env="${p}_winner.env"
  INPUT_ENV="$INPUT_ENV" ARCH="$ARCH" MOD="$MOD" ROWS="$rows" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
    MATE_WIDTH_LIST="$widths" MATE_DISTANCE_LIST="$distances" SELF_EVICT="$SELF_EVICT" MATE_EVICT="$MATE_EVICT" \
    THREADS_LIST="$threads" REPEATS="$repeats" PREFIX="$p" WINNER_ENV="$env" bash "$SWEEP" | tee "$log" >&2
  grep -Fq 'b300_stagei_exact_match=1' "$log" || { echo "Stage-I exact gate missing rows=$rows" >&2; exit 4; }
  [[ -s "$env" ]] || exit 4
  printf '%s\n' "$env"
}

SEARCH_ENV="$(run_stage "$SEARCH_ROWS" "$MATE_WIDTH_LIST" "$MATE_DISTANCE_LIST" '128 256 512' "$SEARCH_REPEATS" search)"
# shellcheck disable=SC1090
source "$SEARCH_ENV"
check_known_residue "$SEARCH_ROWS" "$B300_STAGEI_RESIDUE"
[[ "$B300_STAGEI_SELF_WIDTH" == "$SELF_W" && "$B300_STAGEI_SELF_DISTANCE" == "$SELF_D" ]] || {
  echo 'Stage-I self geometry drifted from Stage F' >&2; exit 4;
}
SELECTED_MATE_W="$B300_STAGEI_MATE_WIDTH"
SELECTED_MATE_D="$B300_STAGEI_MATE_DISTANCE"
VALIDATED=0
CURRENT_ENV="$SEARCH_ENV"
if [[ "$B300_STAGEI_BEST_ENABLED" == 1 && "$B300_STAGEI_CONTROL_SPILL_FREE" == 1 && "$B300_STAGEI_SPILL_FREE" == 1 && "$(passes "$B300_STAGEI_SPEEDUP")" == 1 ]]; then
  VALIDATED=1
  control_threads="$B300_STAGEI_CONTROL_THREADS"
  mate_threads="$B300_STAGEI_THREADS"
  validation_threads="$control_threads"; [[ "$mate_threads" == "$control_threads" ]] || validation_threads+=" $mate_threads"
  stage=0
  for rows in "${stage_validate_rows[@]}"; do
    ((stage+=1))
    CURRENT_ENV="$(run_stage "$rows" "$SELECTED_MATE_W" "$SELECTED_MATE_D" "$validation_threads" "$VALIDATE_REPEATS" "validate${stage}")"
    # shellcheck disable=SC1090
    source "$CURRENT_ENV"
    check_known_residue "$rows" "$B300_STAGEI_RESIDUE"
    if [[ "$B300_STAGEI_SELF_WIDTH" != "$SELF_W" || "$B300_STAGEI_SELF_DISTANCE" != "$SELF_D" || \
          "$B300_STAGEI_MATE_WIDTH" != "$SELECTED_MATE_W" || "$B300_STAGEI_MATE_DISTANCE" != "$SELECTED_MATE_D" ]]; then
      echo "FATAL Stage-I geometry changed during validation rows=$rows" >&2; exit 4
    fi
    if [[ "$B300_STAGEI_BEST_ENABLED" != 1 || "$B300_STAGEI_CONTROL_SPILL_FREE" != 1 || "$B300_STAGEI_SPILL_FREE" != 1 || "$(passes "$B300_STAGEI_SPEEDUP")" != 1 ]]; then
      VALIDATED=0; break
    fi
    control_threads="$B300_STAGEI_CONTROL_THREADS"
    mate_threads="$B300_STAGEI_THREADS"
    validation_threads="$control_threads"; [[ "$mate_threads" == "$control_threads" ]] || validation_threads+=" $mate_threads"
  done
fi

# shellcheck disable=SC1090
source "$CURRENT_ENV"
if [[ "$VALIDATED" == 1 ]]; then
  FINAL_BIN="$B300_STAGEI_BIN"; FINAL_THREADS="$B300_STAGEI_THREADS"; FINAL_WALL="$B300_STAGEI_WALL_S"; FINAL_HIGH="$B300_STAGEI_HIGH_S"; FINAL_SPEED="$B300_STAGEI_SPEEDUP"
else
  FINAL_BIN="$B300_STAGEI_CONTROL_BIN"; FINAL_THREADS="$B300_STAGEI_CONTROL_THREADS"; FINAL_WALL="$B300_STAGEI_CONTROL_WALL_S"; FINAL_HIGH="$B300_STAGEI_CONTROL_HIGH_S"; FINAL_SPEED=1.000000000
fi
{
  printf 'B300_STAGEI_STAGED_VALIDATED=%q\n' "$VALIDATED"
  printf 'B300_STAGEI_FINAL_ENABLED=%q\n' "$VALIDATED"
  printf 'B300_STAGEI_SELF_WIDTH=%q\n' "$SELF_W"
  printf 'B300_STAGEI_SELF_DISTANCE=%q\n' "$SELF_D"
  printf 'B300_STAGEI_SELF_EVICT=%q\n' "$SELF_EVICT"
  printf 'B300_STAGEI_FINAL_MATE_WIDTH=%q\n' "$SELECTED_MATE_W"
  printf 'B300_STAGEI_FINAL_MATE_DISTANCE=%q\n' "$SELECTED_MATE_D"
  printf 'B300_STAGEI_FINAL_MATE_EVICT=%q\n' "$MATE_EVICT"
  printf 'B300_STAGEI_FINAL_BIN=%q\n' "$FINAL_BIN"
  printf 'B300_STAGEI_FINAL_THREADS=%q\n' "$FINAL_THREADS"
  printf 'B300_STAGEI_FINAL_WALL_S=%q\n' "$FINAL_WALL"
  printf 'B300_STAGEI_FINAL_HIGH_S=%q\n' "$FINAL_HIGH"
  printf 'B300_STAGEI_FINAL_SPEEDUP_VS_SELF=%q\n' "$FINAL_SPEED"
  printf 'B300_STAGEI_FINAL_SPILL_FREE=1\n'
  printf 'B300_STAGEI_CONTROL_BIN=%q\n' "$B300_STAGEI_CONTROL_BIN"
  printf 'B300_STAGEI_CONTROL_THREADS=%q\n' "$B300_STAGEI_CONTROL_THREADS"
  printf 'B300_STAGEI_CONTROL_WALL_S=%q\n' "$B300_STAGEI_CONTROL_WALL_S"
  printf 'B300_STAGEI_FINAL_STAGE_ROWS=%q\n' "$B300_STAGEI_ROWS"
  printf 'B300_STAGEI_FINAL_STAGE_RESIDUE=%q\n' "$B300_STAGEI_RESIDUE"
  printf 'B300_STAGEI_INPUT_STAGE_F_ENV=%q\n' "$INPUT_ENV"
  printf 'B300_STAGEI_SEARCH_MATE_WIDTHS=%q\n' "$MATE_WIDTH_LIST"
  printf 'B300_STAGEI_SEARCH_MATE_DISTANCES=%q\n' "$MATE_DISTANCE_LIST"
  printf 'B300_STAGEI_MIN_SPEEDUP=%q\n' "$MIN_SPEEDUP"
} >"$FINAL_ENV"
cat "$FINAL_ENV"
echo "b300-nextgen-hybrid8-nextmate-geometry-staged-calibrate OK validated=$VALIDATED self=w${SELF_W}d${SELF_D} mate=w${SELECTED_MATE_W}d${SELECTED_MATE_D} speedup=$FINAL_SPEED final_rows=$B300_STAGEI_ROWS" >&2
