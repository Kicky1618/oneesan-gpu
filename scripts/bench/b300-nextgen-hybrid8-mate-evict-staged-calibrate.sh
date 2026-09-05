#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
STAGEJ_WINNER_ENV="${STAGEJ_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextmate_geometry_stagej_staged_winner.env}"
STAGEJ_PREPARE_ENV="${STAGEJ_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextmate_geometry_stagej_fullprime_n27_prepared.env}"
ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
SEARCH_ROWS="${SEARCH_ROWS:-1}"; VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"; SEARCH_REPEATS="${SEARCH_REPEATS:-1}"; VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"; MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"
EVICT_LIST="${EVICT_LIST:-default normal last}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_mate_evict_stagek_staged_g${NGPU}}"; FINAL_ENV="${FINAL_ENV:-${PREFIX}_winner.env}"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-mate-evict-sweep.sh"
mkdir -p "$(dirname "$PREFIX")" "$(dirname "$FINAL_ENV")"
for f in "$STAGE_F_ENV" "$STAGEJ_WINNER_ENV" "$STAGEJ_PREPARE_ENV"; do [[ -s "$f" ]] || { echo "missing Stage-K input=$f" >&2; exit 2; }; done
[[ "$NGPU" =~ ^[1-8]$ ]] || { echo 'NGPU must be 1..8' >&2; exit 2; }
for n in SEARCH_REPEATS VALIDATE_REPEATS; do v="${!n}"; [[ "$v" =~ ^[1-9][0-9]*$ ]] || exit 2; done
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY
# shellcheck disable=SC1090
source "$STAGE_F_ENV"
for k in B300_HYBRID8_NEXTSELF_STAGE_E_CORE_ROWS B300_HYBRID8_NEXTSELF_STAGE_E_CORE_RESIDUE B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_RESIDUE B300_HYBRID8_NEXTSELF_FINAL_STAGE_ROWS B300_HYBRID8_NEXTSELF_FINAL_STAGE_RESIDUE; do
  [[ -n "${!k+x}" ]] || { echo "Stage-F env missing $k" >&2; exit 3; }
done
# shellcheck disable=SC1090
source "$STAGEJ_WINNER_ENV"
for k in B300_STAGEJ_STAGED_VALIDATED B300_STAGEJ_FINAL_ENABLED B300_STAGEJ_SELF_WIDTH B300_STAGEJ_SELF_DISTANCE B300_STAGEJ_SELF_EVICT B300_STAGEJ_FINAL_MATE_WIDTH B300_STAGEJ_FINAL_MATE_DISTANCE B300_STAGEJ_FINAL_MATE_EVICT B300_STAGEJ_FINAL_STAGE_ROWS B300_STAGEJ_FINAL_STAGE_RESIDUE; do
  [[ -n "${!k+x}" ]] || { echo "Stage-J winner env missing $k" >&2; exit 3; }
done
[[ "$B300_STAGEJ_STAGED_VALIDATED" == 1 && "$B300_STAGEJ_FINAL_ENABLED" == 1 ]] || { echo 'Stage J not promotable' >&2; exit 4; }
SW="$B300_STAGEJ_SELF_WIDTH"; SD="$B300_STAGEJ_SELF_DISTANCE"; SE="$B300_STAGEJ_SELF_EVICT"; MW="$B300_STAGEJ_FINAL_MATE_WIDTH"; MD="$B300_STAGEJ_FINAL_MATE_DISTANCE"; BASE_EV="$B300_STAGEJ_FINAL_MATE_EVICT"
for w in "$SW" "$MW"; do case "$w" in 1|2|4|8) ;; *) exit 3;; esac; done
for d in "$SD" "$MD"; do case "$d" in 1|2|4) ;; *) exit 3;; esac; done
for ev in "$SE" "$BASE_EV"; do case "$ev" in default|normal|last) ;; *) exit 3;; esac; done

stage_validate_rows=()
for rows in $VALIDATE_ROWS "$B300_STAGEJ_FINAL_STAGE_ROWS" "$B300_HYBRID8_NEXTSELF_FINAL_STAGE_ROWS" "$B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS"; do
  [[ "$rows" == "$SEARCH_ROWS" ]] && continue
  [[ "$rows" =~ ^[1-9][0-9]*$ ]] && ((rows<=28)) || exit 2
  seen=0; for old in "${stage_validate_rows[@]}"; do [[ "$old" == "$rows" ]] && seen=1; done; ((seen)) || stage_validate_rows+=("$rows")
done
check_known_residue(){
  local rows="$1" got="$2"
  if [[ "$rows" == "$B300_HYBRID8_NEXTSELF_STAGE_E_CORE_ROWS" && "$got" != "$B300_HYBRID8_NEXTSELF_STAGE_E_CORE_RESIDUE" ]]; then echo "FATAL Stage-K/core residue mismatch rows=$rows got=$got" >&2; exit 4; fi
  if [[ "$rows" == "$B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS" && "$got" != "$B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_RESIDUE" ]]; then echo "FATAL Stage-K/Stage-E-final residue mismatch rows=$rows got=$got" >&2; exit 4; fi
  if [[ "$rows" == "$B300_HYBRID8_NEXTSELF_FINAL_STAGE_ROWS" && "$got" != "$B300_HYBRID8_NEXTSELF_FINAL_STAGE_RESIDUE" ]]; then echo "FATAL Stage-K/Stage-F-final residue mismatch rows=$rows got=$got" >&2; exit 4; fi
  if [[ "$rows" == "$B300_STAGEJ_FINAL_STAGE_ROWS" && "$got" != "$B300_STAGEJ_FINAL_STAGE_RESIDUE" ]]; then echo "FATAL Stage-K/Stage-J-final residue mismatch rows=$rows got=$got" >&2; exit 4; fi
}
passes(){ python3 - "$1" "$MIN_SPEEDUP" <<'PY'
import sys
print(1 if float(sys.argv[1]) >= float(sys.argv[2]) else 0)
PY
}
run_stage(){
  local rows="$1" hints="$2" threads="$3" repeats="$4" tag="$5"
  local p="${PREFIX}.${tag}.r${rows}" log="${p}.log" env="${p}_winner.env"
  STAGE_F_ENV="$STAGE_F_ENV" STAGEJ_PREPARE_ENV="$STAGEJ_PREPARE_ENV" ARCH="$ARCH" MOD="$MOD" ROWS="$rows" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" NGPU="$NGPU" \
    EVICT_LIST="$hints" THREADS_LIST="$threads" REPEATS="$repeats" PREFIX="$p" WINNER_ENV="$env" bash "$SWEEP" | tee "$log" >&2
  grep -Fq 'b300_stagek_exact_match=1' "$log" || { echo "Stage-K exact gate missing rows=$rows" >&2; exit 4; }
  grep -Fq "b300_stagek_ngpu=$NGPU" "$log" || { echo "Stage-K NGPU marker missing rows=$rows" >&2; exit 4; }
  [[ -s "$env" ]] || exit 4
  printf '%s\n' "$env"
}

SEARCH_ENV="$(run_stage "$SEARCH_ROWS" "$EVICT_LIST" '128 256 512' "$SEARCH_REPEATS" search)"
# shellcheck disable=SC1090
source "$SEARCH_ENV"
check_stage_ngpu(){ [[ "${B300_STAGEK_NGPU:-}" == "$NGPU" ]] || { echo "FATAL Stage-K GPU count drift expected=$NGPU got=${B300_STAGEK_NGPU:-missing}" >&2; exit 4; }; }
check_stage_ngpu
check_known_residue "$SEARCH_ROWS" "$B300_STAGEK_RESIDUE"
[[ "$B300_STAGEK_SELF_WIDTH" == "$SW" && "$B300_STAGEK_SELF_DISTANCE" == "$SD" && "$B300_STAGEK_SELF_EVICT" == "$SE" && "$B300_STAGEK_MATE_WIDTH" == "$MW" && "$B300_STAGEK_MATE_DISTANCE" == "$MD" && "$B300_STAGEK_BASE_EVICT" == "$BASE_EV" ]] || { echo 'Stage-K fixed geometry drifted' >&2; exit 4; }
SELECTED_EV="$B300_STAGEK_EVICT"; VALIDATED=0; CURRENT_ENV="$SEARCH_ENV"
if [[ "$B300_STAGEK_BEST_ENABLED" == 1 && "$B300_STAGEK_BASE_SPILL_FREE" == 1 && "$B300_STAGEK_SPILL_FREE" == 1 && "$(passes "$B300_STAGEK_SPEEDUP")" == 1 ]]; then
  case "$SELECTED_EV" in default|normal|last) ;; *) exit 4;; esac
  VALIDATED=1
  base_threads="$B300_STAGEK_BASE_THREADS"; test_threads="$B300_STAGEK_THREADS"; validation_threads="$base_threads"; [[ "$test_threads" == "$base_threads" ]] || validation_threads+=" $test_threads"
  stage=0
  for rows in "${stage_validate_rows[@]}"; do
    ((stage+=1))
    CURRENT_ENV="$(run_stage "$rows" "$BASE_EV $SELECTED_EV" "$validation_threads" "$VALIDATE_REPEATS" "validate${stage}")"
    # shellcheck disable=SC1090
    source "$CURRENT_ENV"
    check_stage_ngpu
    check_known_residue "$rows" "$B300_STAGEK_RESIDUE"
    if [[ "$B300_STAGEK_EVICT" != "$SELECTED_EV" || "$B300_STAGEK_BASE_EVICT" != "$BASE_EV" ]]; then echo "FATAL Stage-K hint changed during validation rows=$rows" >&2; exit 4; fi
    if [[ "$B300_STAGEK_BEST_ENABLED" != 1 || "$B300_STAGEK_BASE_SPILL_FREE" != 1 || "$B300_STAGEK_SPILL_FREE" != 1 || "$(passes "$B300_STAGEK_SPEEDUP")" != 1 ]]; then VALIDATED=0; break; fi
    base_threads="$B300_STAGEK_BASE_THREADS"; test_threads="$B300_STAGEK_THREADS"; validation_threads="$base_threads"; [[ "$test_threads" == "$base_threads" ]] || validation_threads+=" $test_threads"
  done
fi
# shellcheck disable=SC1090
source "$CURRENT_ENV"
check_stage_ngpu
if [[ "$VALIDATED" == 1 ]]; then FINAL_EV="$SELECTED_EV"; FINAL_BIN="$B300_STAGEK_BIN"; FINAL_THREADS="$B300_STAGEK_THREADS"; FINAL_WALL="$B300_STAGEK_WALL_S"; FINAL_HIGH="$B300_STAGEK_HIGH_S"; FINAL_SPEED="$B300_STAGEK_SPEEDUP"; else FINAL_EV="$BASE_EV"; FINAL_BIN="$B300_STAGEK_BASE_BIN"; FINAL_THREADS="$B300_STAGEK_BASE_THREADS"; FINAL_WALL="$B300_STAGEK_BASE_WALL_S"; FINAL_HIGH="$B300_STAGEK_BASE_HIGH_S"; FINAL_SPEED=1.000000000; fi
{
  printf 'B300_STAGEK_STAGED_VALIDATED=%q\n' "$VALIDATED"
  printf 'B300_STAGEK_FINAL_ENABLED=%q\n' "$VALIDATED"
  printf 'B300_STAGEK_NGPU=%q\n' "$NGPU"
  printf 'B300_STAGEK_SELF_WIDTH=%q\n' "$SW"
  printf 'B300_STAGEK_SELF_DISTANCE=%q\n' "$SD"
  printf 'B300_STAGEK_SELF_EVICT=%q\n' "$SE"
  printf 'B300_STAGEK_MATE_WIDTH=%q\n' "$MW"
  printf 'B300_STAGEK_MATE_DISTANCE=%q\n' "$MD"
  printf 'B300_STAGEK_BASE_MATE_EVICT=%q\n' "$BASE_EV"
  printf 'B300_STAGEK_FINAL_MATE_EVICT=%q\n' "$FINAL_EV"
  printf 'B300_STAGEK_FINAL_BIN=%q\n' "$FINAL_BIN"
  printf 'B300_STAGEK_FINAL_THREADS=%q\n' "$FINAL_THREADS"
  printf 'B300_STAGEK_FINAL_WALL_S=%q\n' "$FINAL_WALL"
  printf 'B300_STAGEK_FINAL_HIGH_S=%q\n' "$FINAL_HIGH"
  printf 'B300_STAGEK_FINAL_SPEEDUP=%q\n' "$FINAL_SPEED"
  printf 'B300_STAGEK_FINAL_SPILL_FREE=1\n'
  printf 'B300_STAGEK_CONTROL_BIN=%q\n' "$B300_STAGEK_BASE_BIN"
  printf 'B300_STAGEK_CONTROL_THREADS=%q\n' "$B300_STAGEK_BASE_THREADS"
  printf 'B300_STAGEK_FINAL_STAGE_ROWS=%q\n' "$B300_STAGEK_ROWS"
  printf 'B300_STAGEK_FINAL_STAGE_RESIDUE=%q\n' "$B300_STAGEK_RESIDUE"
  printf 'B300_STAGEK_STAGE_F_ENV=%q\n' "$STAGE_F_ENV"
  printf 'B300_STAGEK_STAGEJ_WINNER_ENV=%q\n' "$STAGEJ_WINNER_ENV"
  printf 'B300_STAGEK_STAGEJ_PREPARE_ENV=%q\n' "$STAGEJ_PREPARE_ENV"
  printf 'B300_STAGEK_SEARCH_HINTS=%q\n' "$EVICT_LIST"
  printf 'B300_STAGEK_MIN_SPEEDUP=%q\n' "$MIN_SPEEDUP"
} >"$FINAL_ENV"
cat "$FINAL_ENV"
echo "b300-nextgen-hybrid8-mate-evict-staged-calibrate OK validated=$VALIDATED ngpu=$NGPU self=w${SW}d${SD}/$SE mate=w${MW}d${MD} mate_evict=$FINAL_EV speedup=$FINAL_SPEED final_rows=$B300_STAGEK_ROWS" >&2
