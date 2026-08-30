#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
SEARCH_ROWS="${SEARCH_ROWS:-1}"; VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"; SEARCH_REPEATS="${SEARCH_REPEATS:-1}"; VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"; MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"
SELF_WIDTH="${SELF_WIDTH:?SELF_WIDTH required}"; SELF_DISTANCE="${SELF_DISTANCE:?SELF_DISTANCE required}"; SELF_EVICT="${SELF_EVICT:?SELF_EVICT required}"
MATE_WIDTH="${MATE_WIDTH:?MATE_WIDTH required}"; MATE_DISTANCE="${MATE_DISTANCE:?MATE_DISTANCE required}"; MATE_EVICT="${MATE_EVICT:?MATE_EVICT required}"
UPSTREAM_ROWS="${UPSTREAM_ROWS:?UPSTREAM_ROWS required}"; UPSTREAM_RESIDUE="${UPSTREAM_RESIDUE:?UPSTREAM_RESIDUE required}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_guard_staged_g${NGPU}}"; FINAL_ENV="${FINAL_ENV:-${PREFIX}_winner.env}"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-guard-sweep.sh"
mkdir -p "$(dirname "$PREFIX")" "$(dirname "$FINAL_ENV")"
[[ -s "$STAGE_F_ENV" ]] || { echo "missing STAGE_F_ENV=$STAGE_F_ENV" >&2; exit 2; }
[[ "$NGPU" =~ ^[1-8]$ ]] || exit 2
[[ "$UPSTREAM_ROWS" =~ ^[1-9][0-9]*$ ]] && (( UPSTREAM_ROWS <= 28 )) || { echo 'UPSTREAM_ROWS must be 1..28' >&2; exit 2; }
[[ "$UPSTREAM_RESIDUE" =~ ^[0-9]+$ ]] || { echo 'UPSTREAM_RESIDUE must be nonnegative integer' >&2; exit 2; }
for n in SEARCH_REPEATS VALIDATE_REPEATS; do v="${!n}"; [[ "$v" =~ ^[1-9][0-9]*$ ]] || exit 2; done
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY
for w in "$SELF_WIDTH" "$MATE_WIDTH"; do case "$w" in 1|2|4|8) ;; *) exit 2;; esac; done
for d in "$SELF_DISTANCE" "$MATE_DISTANCE"; do case "$d" in 1|2|4) ;; *) exit 2;; esac; done
for e in "$SELF_EVICT" "$MATE_EVICT"; do case "$e" in default|normal|last) ;; *) exit 2;; esac; done

# shellcheck disable=SC1090
source "$STAGE_F_ENV"
for k in B300_HYBRID8_NEXTSELF_STAGE_E_CORE_ROWS B300_HYBRID8_NEXTSELF_STAGE_E_CORE_RESIDUE \
  B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_RESIDUE \
  B300_HYBRID8_NEXTSELF_FINAL_STAGE_ROWS B300_HYBRID8_NEXTSELF_FINAL_STAGE_RESIDUE \
  B300_HYBRID8_NEXTSELF_FINAL_WIDTH B300_HYBRID8_NEXTSELF_FINAL_DISTANCE; do
  [[ -n "${!k+x}" ]] || { echo "Stage-F env missing $k" >&2; exit 3; }
done
[[ "$SELF_WIDTH" == "$B300_HYBRID8_NEXTSELF_FINAL_WIDTH" && "$SELF_DISTANCE" == "$B300_HYBRID8_NEXTSELF_FINAL_DISTANCE" ]] || { echo 'Stage-L fixed self geometry differs from Stage F' >&2; exit 3; }

stage_validate_rows=()
for rows in $VALIDATE_ROWS "$UPSTREAM_ROWS" "$B300_HYBRID8_NEXTSELF_FINAL_STAGE_ROWS" "$B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS"; do
  [[ "$rows" == "$SEARCH_ROWS" ]] && continue
  [[ "$rows" =~ ^[1-9][0-9]*$ ]] && (( rows <= 28 )) || exit 2
  seen=0; for old in "${stage_validate_rows[@]}"; do [[ "$old" == "$rows" ]] && seen=1; done
  (( seen )) || stage_validate_rows+=("$rows")
done
check_known_residue(){
  local rows="$1" got="$2"
  if [[ "$rows" == "$UPSTREAM_ROWS" && "$got" != "$UPSTREAM_RESIDUE" ]]; then echo "FATAL Stage-L/upstream residue mismatch rows=$rows got=$got expected=$UPSTREAM_RESIDUE" >&2; exit 4; fi
  if [[ "$rows" == "$B300_HYBRID8_NEXTSELF_STAGE_E_CORE_ROWS" && "$got" != "$B300_HYBRID8_NEXTSELF_STAGE_E_CORE_RESIDUE" ]]; then echo "FATAL Stage-L/core residue mismatch rows=$rows" >&2; exit 4; fi
  if [[ "$rows" == "$B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS" && "$got" != "$B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_RESIDUE" ]]; then echo "FATAL Stage-L/Stage-E-final residue mismatch rows=$rows" >&2; exit 4; fi
  if [[ "$rows" == "$B300_HYBRID8_NEXTSELF_FINAL_STAGE_ROWS" && "$got" != "$B300_HYBRID8_NEXTSELF_FINAL_STAGE_RESIDUE" ]]; then echo "FATAL Stage-L/Stage-F-final residue mismatch rows=$rows" >&2; exit 4; fi
}
passes(){ python3 - "$1" "$MIN_SPEEDUP" <<'PY'
import sys
print(1 if float(sys.argv[1]) >= float(sys.argv[2]) else 0)
PY
}
run_stage(){
  local rows="$1" selfguards="$2" mateguards="$3" threads="$4" repeats="$5" tag="$6"
  local p="${PREFIX}.${tag}.r${rows}" log="${p}.log" env="${p}_winner.env"
  STAGE_F_ENV="$STAGE_F_ENV" ARCH="$ARCH" MOD="$MOD" ROWS="$rows" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" NGPU="$NGPU" \
    SELF_WIDTH="$SELF_WIDTH" SELF_DISTANCE="$SELF_DISTANCE" SELF_EVICT="$SELF_EVICT" MATE_WIDTH="$MATE_WIDTH" MATE_DISTANCE="$MATE_DISTANCE" MATE_EVICT="$MATE_EVICT" \
    SELF_GUARD_LIST="$selfguards" MATE_GUARD_LIST="$mateguards" THREADS_LIST="$threads" REPEATS="$repeats" PREFIX="$p" WINNER_ENV="$env" \
    bash "$SWEEP" | tee "$log" >&2
  grep -Fq 'b300_stagel_exact_match=1' "$log" || { echo "Stage-L exact gate missing rows=$rows" >&2; exit 4; }
  grep -Fq "b300_stagel_ngpu=$NGPU" "$log" || { echo "Stage-L NGPU gate missing rows=$rows" >&2; exit 4; }
  [[ -s "$env" ]] || exit 4
  printf '%s\n' "$env"
}

SEARCH_ENV="$(run_stage "$SEARCH_ROWS" 'branch predicated' 'branch predicated' '128 256 512' "$SEARCH_REPEATS" search)"
# shellcheck disable=SC1090
source "$SEARCH_ENV"
[[ "$B300_STAGEL_NGPU" == "$NGPU" ]] || exit 4
check_known_residue "$SEARCH_ROWS" "$B300_STAGEL_RESIDUE"
SELECTED_SG="$B300_STAGEL_SELF_GUARD"; SELECTED_MG="$B300_STAGEL_MATE_GUARD"
VALIDATED=0; CURRENT_ENV="$SEARCH_ENV"
if [[ "$B300_STAGEL_BEST_ENABLED" == 1 && "$B300_STAGEL_CONTROL_SPILL_FREE" == 1 && "$B300_STAGEL_SPILL_FREE" == 1 && "$(passes "$B300_STAGEL_SPEEDUP")" == 1 ]]; then
  [[ "$SELECTED_SG" == branch || "$SELECTED_SG" == predicated ]] || exit 4
  [[ "$SELECTED_MG" == branch || "$SELECTED_MG" == predicated ]] || exit 4
  VALIDATED=1
  control_threads="$B300_STAGEL_CONTROL_THREADS"; test_threads="$B300_STAGEL_THREADS"
  validation_threads="$control_threads"; [[ "$test_threads" == "$control_threads" ]] || validation_threads+=" $test_threads"
  sg_list=branch; [[ "$SELECTED_SG" == predicated ]] && sg_list='branch predicated'
  mg_list=branch; [[ "$SELECTED_MG" == predicated ]] && mg_list='branch predicated'
  stage=0
  for rows in "${stage_validate_rows[@]}"; do
    ((stage+=1))
    CURRENT_ENV="$(run_stage "$rows" "$sg_list" "$mg_list" "$validation_threads" "$VALIDATE_REPEATS" "validate${stage}")"
    # shellcheck disable=SC1090
    source "$CURRENT_ENV"
    [[ "$B300_STAGEL_NGPU" == "$NGPU" ]] || { echo 'FATAL Stage-L GPU count drift' >&2; exit 4; }
    check_known_residue "$rows" "$B300_STAGEL_RESIDUE"
    if [[ "$B300_STAGEL_SELF_GUARD" != "$SELECTED_SG" || "$B300_STAGEL_MATE_GUARD" != "$SELECTED_MG" ]]; then
      echo "FATAL Stage-L guard policy changed during validation rows=$rows selected=$SELECTED_SG/$SELECTED_MG got=$B300_STAGEL_SELF_GUARD/$B300_STAGEL_MATE_GUARD" >&2; exit 4
    fi
    if [[ "$B300_STAGEL_BEST_ENABLED" != 1 || "$B300_STAGEL_CONTROL_SPILL_FREE" != 1 || "$B300_STAGEL_SPILL_FREE" != 1 || "$(passes "$B300_STAGEL_SPEEDUP")" != 1 ]]; then VALIDATED=0; break; fi
    control_threads="$B300_STAGEL_CONTROL_THREADS"; test_threads="$B300_STAGEL_THREADS"
    validation_threads="$control_threads"; [[ "$test_threads" == "$control_threads" ]] || validation_threads+=" $test_threads"
  done
fi
# shellcheck disable=SC1090
source "$CURRENT_ENV"
[[ "$B300_STAGEL_NGPU" == "$NGPU" ]] || exit 4
if [[ "$VALIDATED" == 1 ]]; then
  FINAL_SG="$SELECTED_SG"; FINAL_MG="$SELECTED_MG"; FINAL_BIN="$B300_STAGEL_BIN"; FINAL_THREADS="$B300_STAGEL_THREADS"; FINAL_WALL="$B300_STAGEL_WALL_S"; FINAL_HIGH="$B300_STAGEL_HIGH_S"; FINAL_SPEED="$B300_STAGEL_SPEEDUP"
else
  FINAL_SG=branch; FINAL_MG=branch; FINAL_BIN="$B300_STAGEL_CONTROL_BIN"; FINAL_THREADS="$B300_STAGEL_CONTROL_THREADS"; FINAL_WALL="$B300_STAGEL_CONTROL_WALL_S"; FINAL_HIGH="$B300_STAGEL_CONTROL_HIGH_S"; FINAL_SPEED=1.000000000
fi
{
  printf 'B300_STAGEL_STAGED_VALIDATED=%q\n' "$VALIDATED"
  printf 'B300_STAGEL_FINAL_ENABLED=%q\n' "$VALIDATED"
  printf 'B300_STAGEL_NGPU=%q\n' "$NGPU"
  printf 'B300_STAGEL_SELF_WIDTH=%q\n' "$SELF_WIDTH"
  printf 'B300_STAGEL_SELF_DISTANCE=%q\n' "$SELF_DISTANCE"
  printf 'B300_STAGEL_SELF_EVICT=%q\n' "$SELF_EVICT"
  printf 'B300_STAGEL_MATE_WIDTH=%q\n' "$MATE_WIDTH"
  printf 'B300_STAGEL_MATE_DISTANCE=%q\n' "$MATE_DISTANCE"
  printf 'B300_STAGEL_MATE_EVICT=%q\n' "$MATE_EVICT"
  printf 'B300_STAGEL_FINAL_SELF_GUARD=%q\n' "$FINAL_SG"
  printf 'B300_STAGEL_FINAL_MATE_GUARD=%q\n' "$FINAL_MG"
  printf 'B300_STAGEL_FINAL_BIN=%q\n' "$FINAL_BIN"
  printf 'B300_STAGEL_FINAL_THREADS=%q\n' "$FINAL_THREADS"
  printf 'B300_STAGEL_FINAL_WALL_S=%q\n' "$FINAL_WALL"
  printf 'B300_STAGEL_FINAL_HIGH_S=%q\n' "$FINAL_HIGH"
  printf 'B300_STAGEL_FINAL_SPEEDUP=%q\n' "$FINAL_SPEED"
  printf 'B300_STAGEL_FINAL_SPILL_FREE=1\n'
  printf 'B300_STAGEL_CONTROL_BIN=%q\n' "$B300_STAGEL_CONTROL_BIN"
  printf 'B300_STAGEL_CONTROL_THREADS=%q\n' "$B300_STAGEL_CONTROL_THREADS"
  printf 'B300_STAGEL_FINAL_STAGE_ROWS=%q\n' "$B300_STAGEL_ROWS"
  printf 'B300_STAGEL_FINAL_STAGE_RESIDUE=%q\n' "$B300_STAGEL_RESIDUE"
  printf 'B300_STAGEL_STAGE_F_ENV=%q\n' "$STAGE_F_ENV"
  printf 'B300_STAGEL_UPSTREAM_ROWS=%q\n' "$UPSTREAM_ROWS"
  printf 'B300_STAGEL_UPSTREAM_RESIDUE=%q\n' "$UPSTREAM_RESIDUE"
  printf 'B300_STAGEL_MIN_SPEEDUP=%q\n' "$MIN_SPEEDUP"
} >"$FINAL_ENV"
cat "$FINAL_ENV"
echo "b300-nextgen-hybrid8-guard-staged-calibrate OK validated=$VALIDATED ngpu=$NGPU guards=$FINAL_SG/$FINAL_MG speedup=$FINAL_SPEED final_rows=$B300_STAGEL_ROWS" >&2