#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

INPUT_ENV="${INPUT_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
SELF_EVICT="${SELF_EVICT:-default}"
ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"
SEARCH_ROWS="${SEARCH_ROWS:-1}"; VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"; SEARCH_REPEATS="${SEARCH_REPEATS:-1}"; VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"; MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextmate_staged}"; FINAL_ENV="${FINAL_ENV:-${PREFIX}_winner.env}"
mkdir -p "$(dirname "$PREFIX")" "$(dirname "$FINAL_ENV")"
[[ -s "$INPUT_ENV" ]] || { echo "missing Stage-F INPUT_ENV=$INPUT_ENV" >&2; exit 2; }
case "$SELF_EVICT" in default|normal|last) ;; *) echo 'SELF_EVICT must be default,normal,last' >&2; exit 2;; esac
for n in SEARCH_REPEATS VALIDATE_REPEATS; do v="${!n}"; [[ "$v" =~ ^[1-9][0-9]*$ ]] || exit 2; done
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1])<1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY
# shellcheck disable=SC1090
source "$INPUT_ENV"
for k in B300_HYBRID8_NEXTSELF_STAGED_VALIDATED B300_HYBRID8_NEXTSELF_FINAL_ENABLED B300_HYBRID8_NEXTSELF_FINAL_WIDTH B300_HYBRID8_NEXTSELF_FINAL_DISTANCE \
 B300_HYBRID8_NEXTSELF_FINAL_STAGE_ROWS B300_HYBRID8_NEXTSELF_FINAL_STAGE_RESIDUE B300_HYBRID8_NEXTSELF_STAGE_E_CORE_ROWS B300_HYBRID8_NEXTSELF_STAGE_E_CORE_RESIDUE \
 B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_RESIDUE; do [[ -n "${!k+x}" ]] || { echo "Stage-F env missing $k" >&2; exit 3; }; done
[[ "$B300_HYBRID8_NEXTSELF_STAGED_VALIDATED" == 1 && "$B300_HYBRID8_NEXTSELF_FINAL_ENABLED" == 1 ]] || { echo 'Stage F not promotable' >&2; exit 4; }
W="$B300_HYBRID8_NEXTSELF_FINAL_WIDTH"; D="$B300_HYBRID8_NEXTSELF_FINAL_DISTANCE"
case "$W" in 1|2|4|8);;*) exit 3;; esac; case "$D" in 1|2|4);;*) exit 3;; esac

stage_validate_rows=()
for rows in $VALIDATE_ROWS "$B300_HYBRID8_NEXTSELF_FINAL_STAGE_ROWS" "$B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS"; do
  [[ "$rows" == "$SEARCH_ROWS" ]] && continue
  [[ "$rows" =~ ^[1-9][0-9]*$ ]] && ((rows<=28)) || exit 2
  seen=0; for old in "${stage_validate_rows[@]}"; do [[ "$old" == "$rows" ]] && seen=1; done; ((seen)) || stage_validate_rows+=("$rows")
done
check_known_residue(){
  local rows="$1" got="$2"
  if [[ "$rows" == "$B300_HYBRID8_NEXTSELF_STAGE_E_CORE_ROWS" && "$got" != "$B300_HYBRID8_NEXTSELF_STAGE_E_CORE_RESIDUE" ]]; then echo "FATAL Stage-H/core residue mismatch rows=$rows got=$got" >&2; exit 4; fi
  if [[ "$rows" == "$B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS" && "$got" != "$B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_RESIDUE" ]]; then echo "FATAL Stage-H/Stage-E-final residue mismatch rows=$rows got=$got" >&2; exit 4; fi
  if [[ "$rows" == "$B300_HYBRID8_NEXTSELF_FINAL_STAGE_ROWS" && "$got" != "$B300_HYBRID8_NEXTSELF_FINAL_STAGE_RESIDUE" ]]; then echo "FATAL Stage-H/Stage-F-final residue mismatch rows=$rows got=$got" >&2; exit 4; fi
}
run_stage(){
  local rows="$1" threads="$2" repeats="$3" tag="$4"
  local p="${PREFIX}.${tag}.r${rows}" log="${p}.log" env="${p}_winner.env"
  INPUT_ENV="$INPUT_ENV" SELF_EVICT="$SELF_EVICT" ARCH="$ARCH" MOD="$MOD" ROWS="$rows" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" THREADS_LIST="$threads" REPEATS="$repeats" PREFIX="$p" WINNER_ENV="$env" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-nextmate-ab.sh" | tee "$log" >&2
  grep -Fq 'b300_stageh_exact_match=1' "$log" || { echo "Stage H exact gate missing rows=$rows" >&2; exit 4; }
  grep -Fq "b300_stageh_self_evict=$SELF_EVICT" "$log" || { echo "Stage H self-evict marker mismatch rows=$rows" >&2; exit 4; }
  [[ -s "$env" ]] || exit 4
  printf '%s\n' "$env"
}
passes(){ python3 - "$1" "$MIN_SPEEDUP" <<'PY'
import sys
print(1 if float(sys.argv[1])>=float(sys.argv[2]) else 0)
PY
}
SEARCH_ENV="$(run_stage "$SEARCH_ROWS" '128 256 512' "$SEARCH_REPEATS" search)"
# shellcheck disable=SC1090
source "$SEARCH_ENV"; check_known_residue "$SEARCH_ROWS" "$B300_STAGEH_RESIDUE"
[[ "$B300_STAGEH_SELF_EVICT" == "$SELF_EVICT" ]] || { echo 'Stage-H search self eviction changed' >&2; exit 4; }
VALIDATED=0; CURRENT_ENV="$SEARCH_ENV"
if [[ "$B300_STAGEH_BEST_ENABLED" == 1 && "$B300_STAGEH_SELF_SPILL_FREE" == 1 && "$B300_STAGEH_MATE_SPILL_FREE" == 1 && "$(passes "$B300_STAGEH_SPEEDUP")" == 1 ]]; then
  VALIDATED=1; self_threads="$B300_STAGEH_SELF_THREADS"; mate_threads="$B300_STAGEH_MATE_THREADS"; validation_threads="$self_threads"; [[ "$mate_threads" == "$self_threads" ]] || validation_threads+=" $mate_threads"
  stage=0
  for rows in "${stage_validate_rows[@]}"; do
    ((stage+=1)); CURRENT_ENV="$(run_stage "$rows" "$validation_threads" "$VALIDATE_REPEATS" "validate${stage}")"
    # shellcheck disable=SC1090
    source "$CURRENT_ENV"; check_known_residue "$rows" "$B300_STAGEH_RESIDUE"
    [[ "$B300_STAGEH_SELF_EVICT" == "$SELF_EVICT" ]] || { echo "FATAL Stage-H self eviction changed rows=$rows" >&2; exit 4; }
    if [[ "$B300_STAGEH_BEST_ENABLED" != 1 || "$B300_STAGEH_SELF_SPILL_FREE" != 1 || "$B300_STAGEH_MATE_SPILL_FREE" != 1 || "$(passes "$B300_STAGEH_SPEEDUP")" != 1 ]]; then VALIDATED=0; break; fi
    self_threads="$B300_STAGEH_SELF_THREADS"; mate_threads="$B300_STAGEH_MATE_THREADS"; validation_threads="$self_threads"; [[ "$mate_threads" == "$self_threads" ]] || validation_threads+=" $mate_threads"
  done
fi
# shellcheck disable=SC1090
source "$CURRENT_ENV"
if [[ "$VALIDATED" == 1 ]]; then FINAL_BIN="$B300_STAGEH_MATE_BIN"; FINAL_THREADS="$B300_STAGEH_MATE_THREADS"; FINAL_WALL="$B300_STAGEH_MATE_WALL_S"; FINAL_HIGH="$B300_STAGEH_MATE_HIGH_S"; FINAL_SPEED="$B300_STAGEH_SPEEDUP"; else FINAL_BIN="$B300_STAGEH_SELF_BIN"; FINAL_THREADS="$B300_STAGEH_SELF_THREADS"; FINAL_WALL="$B300_STAGEH_SELF_WALL_S"; FINAL_HIGH="$B300_STAGEH_SELF_HIGH_S"; FINAL_SPEED=1.000000000; fi
{
 printf 'B300_STAGEH_STAGED_VALIDATED=%q\n' "$VALIDATED"
 printf 'B300_STAGEH_FINAL_ENABLED=%q\n' "$VALIDATED"
 printf 'B300_STAGEH_FINAL_WIDTH=%q\n' "$W"
 printf 'B300_STAGEH_FINAL_DISTANCE=%q\n' "$D"
 printf 'B300_STAGEH_FINAL_SELF_EVICT=%q\n' "$SELF_EVICT"
 printf 'B300_STAGEH_FINAL_BIN=%q\n' "$FINAL_BIN"
 printf 'B300_STAGEH_FINAL_THREADS=%q\n' "$FINAL_THREADS"
 printf 'B300_STAGEH_FINAL_WALL_S=%q\n' "$FINAL_WALL"
 printf 'B300_STAGEH_FINAL_HIGH_S=%q\n' "$FINAL_HIGH"
 printf 'B300_STAGEH_FINAL_SPEEDUP_VS_SELF=%q\n' "$FINAL_SPEED"
 printf 'B300_STAGEH_FINAL_SPILL_FREE=1\n'
 printf 'B300_STAGEH_CONTROL_BIN=%q\n' "$B300_STAGEH_SELF_BIN"
 printf 'B300_STAGEH_CONTROL_THREADS=%q\n' "$B300_STAGEH_SELF_THREADS"
 printf 'B300_STAGEH_CONTROL_WALL_S=%q\n' "$B300_STAGEH_SELF_WALL_S"
 printf 'B300_STAGEH_FINAL_STAGE_ROWS=%q\n' "$B300_STAGEH_ROWS"
 printf 'B300_STAGEH_FINAL_STAGE_RESIDUE=%q\n' "$B300_STAGEH_RESIDUE"
 printf 'B300_STAGEH_INPUT_STAGE_F_ENV=%q\n' "$INPUT_ENV"
 printf 'B300_STAGEH_MIN_SPEEDUP=%q\n' "$MIN_SPEEDUP"
} >"$FINAL_ENV"
cat "$FINAL_ENV"
echo "b300-nextgen-hybrid8-nextmate-staged-calibrate OK validated=$VALIDATED width=$W distance=$D self_evict=$SELF_EVICT speedup=$FINAL_SPEED final_rows=$B300_STAGEH_ROWS" >&2
