#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
STAGEL_GUARD_ENV="${STAGEL_GUARD_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_prefetch_guard_stagel_g${NGPU}_winner.env}"
SEARCH_ROWS="${SEARCH_ROWS:-1}"; VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"; SEARCH_REPEATS="${SEARCH_REPEATS:-1}"; VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"; POLICY_LIST="${POLICY_LIST:-default cg cs}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_mate_load_stagem_staged_g${NGPU}}"; FINAL_ENV="${FINAL_ENV:-${PREFIX}_winner.env}"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-mate-load-policy-sweep.sh"
mkdir -p "$(dirname "$PREFIX")" "$(dirname "$FINAL_ENV")"
for f in "$STAGE_F_ENV" "$STAGEL_GUARD_ENV" "$SWEEP"; do [[ -s "$f" ]] || { echo "missing Stage-M input=$f" >&2; exit 2; }; done
for x in SEARCH_REPEATS VALIDATE_REPEATS; do v="${!x}"; [[ "$v" =~ ^[1-9][0-9]*$ ]] || exit 2; done
[[ "$SEARCH_ROWS" =~ ^[1-9][0-9]*$ ]] && ((SEARCH_ROWS<=28)) || exit 2
[[ "$NGPU" =~ ^[1-8]$ ]] || exit 2
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY

# Stage L is the immediate upstream policy owner. Preserve its geometry,
# eviction and self/mate guard modes exactly throughout Stage M.
# shellcheck disable=SC1090
source "$STAGEL_GUARD_ENV"
for k in B300_STAGEL_FINAL_SPILL_FREE B300_STAGEL_NGPU B300_STAGEL_SELF_WIDTH B300_STAGEL_SELF_DISTANCE B300_STAGEL_SELF_EVICT B300_STAGEL_MATE_WIDTH B300_STAGEL_MATE_DISTANCE B300_STAGEL_MATE_EVICT B300_STAGEL_FINAL_SELF_GUARD B300_STAGEL_FINAL_MATE_GUARD B300_STAGEL_FINAL_BIN B300_STAGEL_FINAL_THREADS B300_STAGEL_FINAL_STAGE_ROWS B300_STAGEL_FINAL_STAGE_RESIDUE; do
  [[ -n "${!k+x}" ]] || { echo "Stage-L guard env missing $k" >&2; exit 3; }
done
[[ "$B300_STAGEL_NGPU" == "$NGPU" ]] || { echo "Stage-L GPU count drift expected=$NGPU got=$B300_STAGEL_NGPU" >&2; exit 3; }
[[ "$B300_STAGEL_FINAL_SPILL_FREE" == 1 ]] || exit 4
L_ROWS="$B300_STAGEL_FINAL_STAGE_ROWS"; L_RES="$B300_STAGEL_FINAL_STAGE_RESIDUE"
SW="$B300_STAGEL_SELF_WIDTH"; SD="$B300_STAGEL_SELF_DISTANCE"; SE="$B300_STAGEL_SELF_EVICT"; SG="$B300_STAGEL_FINAL_SELF_GUARD"
MW="$B300_STAGEL_MATE_WIDTH"; MD="$B300_STAGEL_MATE_DISTANCE"; ME="$B300_STAGEL_MATE_EVICT"; MG="$B300_STAGEL_FINAL_MATE_GUARD"
# shellcheck disable=SC1090
source "$STAGE_F_ENV"
F_ROWS="${B300_HYBRID8_NEXTSELF_FINAL_STAGE_ROWS:-0}"; F_RES="${B300_HYBRID8_NEXTSELF_FINAL_STAGE_RESIDUE:-}"
E_ROWS="${B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS:-0}"; E_RES="${B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_RESIDUE:-}"

check_known_residue(){
  local rows="$1" got="$2"
  if [[ "$rows" == "$L_ROWS" && "$got" != "$L_RES" ]]; then echo "FATAL Stage-M/Stage-L residue mismatch rows=$rows got=$got expected=$L_RES" >&2; exit 4; fi
  if [[ "$rows" == "$F_ROWS" && -n "$F_RES" && "$got" != "$F_RES" ]]; then echo "FATAL Stage-M/Stage-F residue mismatch rows=$rows got=$got expected=$F_RES" >&2; exit 4; fi
  if [[ "$rows" == "$E_ROWS" && -n "$E_RES" && "$got" != "$E_RES" ]]; then echo "FATAL Stage-M/Stage-E residue mismatch rows=$rows got=$got expected=$E_RES" >&2; exit 4; fi
}
passes(){ python3 - "$1" "$MIN_SPEEDUP" <<'PY'
import sys
print(1 if float(sys.argv[1]) >= float(sys.argv[2]) else 0)
PY
}
run_stage(){
  local rows="$1" policies="$2" threads="$3" repeats="$4" tag="$5"
  local p="${PREFIX}.${tag}.r${rows}" log="${p}.log" env="${p}_winner.env"
  STAGE_F_ENV="$STAGE_F_ENV" STAGEL_GUARD_ENV="$STAGEL_GUARD_ENV" \
    ARCH="$ARCH" MOD="$MOD" ROWS="$rows" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" NGPU="$NGPU" \
    POLICY_LIST="$policies" THREADS_LIST="$threads" REPEATS="$repeats" PREFIX="$p" WINNER_ENV="$env" \
    bash "$SWEEP" | tee "$log" >&2
  grep -Fq 'b300_stagem_exact_match=1' "$log" || { echo "Stage-M exact gate missing rows=$rows" >&2; exit 4; }
  grep -Fq "b300_stagem_ngpu=$NGPU" "$log" || { echo "Stage-M GPU count gate missing rows=$rows" >&2; exit 4; }
  [[ -s "$env" ]] || exit 4
  printf '%s\n' "$env"
}
check_configuration(){
  [[ "$B300_STAGEM_NGPU" == "$NGPU" && \
      "$B300_STAGEM_SELF_WIDTH" == "$SW" && "$B300_STAGEM_SELF_DISTANCE" == "$SD" && "$B300_STAGEM_SELF_EVICT" == "$SE" && "$B300_STAGEM_SELF_GUARD" == "$SG" && \
      "$B300_STAGEM_MATE_WIDTH" == "$MW" && "$B300_STAGEM_MATE_DISTANCE" == "$MD" && "$B300_STAGEM_MATE_EVICT" == "$ME" && "$B300_STAGEM_MATE_GUARD" == "$MG" ]] || {
    echo 'FATAL Stage-M upstream configuration drift' >&2
    exit 4
  }
}

SEARCH_ENV="$(run_stage "$SEARCH_ROWS" "$POLICY_LIST" "$THREADS_LIST" "$SEARCH_REPEATS" search)"
# shellcheck disable=SC1090
source "$SEARCH_ENV"
check_configuration
check_known_residue "$SEARCH_ROWS" "$B300_STAGEM_RESIDUE"
VALIDATED=0; CURRENT_ENV="$SEARCH_ENV"; SELECTED_POLICY="$B300_STAGEM_POLICY"
if [[ "$B300_STAGEM_BEST_ENABLED" == 1 && "$B300_STAGEM_CONTROL_SPILL_FREE" == 1 && "$B300_STAGEM_SPILL_FREE" == 1 && "$SELECTED_POLICY" != default && "$(passes "$B300_STAGEM_SPEEDUP")" == 1 ]]; then
  case "$SELECTED_POLICY" in cg|cs) ;; *) exit 4;; esac
  VALIDATED=1
  control_threads="$B300_STAGEM_CONTROL_THREADS"; test_threads="$B300_STAGEM_THREADS"
  validation_threads="$control_threads"; [[ "$test_threads" == "$control_threads" ]] || validation_threads+=" $test_threads"
  validate_rows=()
  for rows in $VALIDATE_ROWS "$L_ROWS" "$F_ROWS" "$E_ROWS"; do
    [[ "$rows" =~ ^[1-9][0-9]*$ ]] || continue
    [[ "$rows" == "$SEARCH_ROWS" ]] && continue
    seen=0; for old in "${validate_rows[@]}"; do [[ "$old" == "$rows" ]] && seen=1; done
    ((seen)) || validate_rows+=("$rows")
  done
  stage=0
  for rows in "${validate_rows[@]}"; do
    ((stage+=1))
    CURRENT_ENV="$(run_stage "$rows" "default $SELECTED_POLICY" "$validation_threads" "$VALIDATE_REPEATS" "validate${stage}")"
    # shellcheck disable=SC1090
    source "$CURRENT_ENV"
    check_configuration
    check_known_residue "$rows" "$B300_STAGEM_RESIDUE"
    if [[ "$B300_STAGEM_POLICY" != "$SELECTED_POLICY" || "$B300_STAGEM_BEST_ENABLED" != 1 || "$B300_STAGEM_CONTROL_SPILL_FREE" != 1 || "$B300_STAGEM_SPILL_FREE" != 1 || "$(passes "$B300_STAGEM_SPEEDUP")" != 1 ]]; then
      VALIDATED=0
      break
    fi
    control_threads="$B300_STAGEM_CONTROL_THREADS"; test_threads="$B300_STAGEM_THREADS"
    validation_threads="$control_threads"; [[ "$test_threads" == "$control_threads" ]] || validation_threads+=" $test_threads"
  done
fi

# shellcheck disable=SC1090
source "$CURRENT_ENV"
check_configuration
if [[ "$VALIDATED" == 1 ]]; then
  FINAL_POLICY="$B300_STAGEM_POLICY"; FINAL_BIN="$B300_STAGEM_BIN"; FINAL_THREADS="$B300_STAGEM_THREADS"; FINAL_WALL="$B300_STAGEM_WALL_S"; FINAL_HIGH="$B300_STAGEM_HIGH_S"; FINAL_SPEED="$B300_STAGEM_SPEEDUP"
else
  FINAL_POLICY=default; FINAL_BIN="$B300_STAGEM_CONTROL_BIN"; FINAL_THREADS="$B300_STAGEM_CONTROL_THREADS"; FINAL_WALL="$B300_STAGEM_CONTROL_WALL_S"; FINAL_HIGH="$B300_STAGEM_CONTROL_HIGH_S"; FINAL_SPEED=1.000000000
fi
{
  printf 'B300_STAGEM_STAGED_VALIDATED=%q\n' "$VALIDATED"
  printf 'B300_STAGEM_FINAL_ENABLED=%q\n' "$VALIDATED"
  printf 'B300_STAGEM_NGPU=%q\n' "$NGPU"
  printf 'B300_STAGEM_POLICY=%q\n' "$FINAL_POLICY"
  printf 'B300_STAGEM_FINAL_BIN=%q\n' "$FINAL_BIN"
  printf 'B300_STAGEM_FINAL_THREADS=%q\n' "$FINAL_THREADS"
  printf 'B300_STAGEM_FINAL_WALL_S=%q\n' "$FINAL_WALL"
  printf 'B300_STAGEM_FINAL_HIGH_S=%q\n' "$FINAL_HIGH"
  printf 'B300_STAGEM_FINAL_SPEEDUP=%q\n' "$FINAL_SPEED"
  printf 'B300_STAGEM_FINAL_SPILL_FREE=1\n'
  printf 'B300_STAGEM_CONTROL_BIN=%q\n' "$B300_STAGEM_CONTROL_BIN"
  printf 'B300_STAGEM_CONTROL_THREADS=%q\n' "$B300_STAGEM_CONTROL_THREADS"
  printf 'B300_STAGEM_SELF_WIDTH=%q\n' "$SW"
  printf 'B300_STAGEM_SELF_DISTANCE=%q\n' "$SD"
  printf 'B300_STAGEM_SELF_EVICT=%q\n' "$SE"
  printf 'B300_STAGEM_SELF_GUARD=%q\n' "$SG"
  printf 'B300_STAGEM_MATE_WIDTH=%q\n' "$MW"
  printf 'B300_STAGEM_MATE_DISTANCE=%q\n' "$MD"
  printf 'B300_STAGEM_MATE_EVICT=%q\n' "$ME"
  printf 'B300_STAGEM_MATE_GUARD=%q\n' "$MG"
  printf 'B300_STAGEM_FINAL_STAGE_ROWS=%q\n' "$B300_STAGEM_ROWS"
  printf 'B300_STAGEM_FINAL_STAGE_RESIDUE=%q\n' "$B300_STAGEM_RESIDUE"
  printf 'B300_STAGEM_SEARCH_POLICIES=%q\n' "$POLICY_LIST"
  printf 'B300_STAGEM_MIN_SPEEDUP=%q\n' "$MIN_SPEEDUP"
  printf 'B300_STAGEM_INPUT_STAGE_F_ENV=%q\n' "$STAGE_F_ENV"
  printf 'B300_STAGEM_INPUT_STAGEL_GUARD_ENV=%q\n' "$STAGEL_GUARD_ENV"
} >"$FINAL_ENV"
cat "$FINAL_ENV"
echo "b300-nextgen-hybrid8-mate-load-policy-staged-calibrate OK stage=M validated=$VALIDATED policy=$FINAL_POLICY guards=$SG/$MG speedup=${FINAL_SPEED}x final_rows=$B300_STAGEM_ROWS ngpu=$NGPU" >&2
