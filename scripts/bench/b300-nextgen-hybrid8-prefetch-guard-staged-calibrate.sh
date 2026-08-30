#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
UPSTREAM_PREPARE_ENV="${UPSTREAM_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_mate_evict_stagek_fullprime_n27_prepared.env}"
UPSTREAM_WINNER_ENV="${UPSTREAM_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_mate_evict_stagek_staged_winner.env}"
ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
SEARCH_ROWS="${SEARCH_ROWS:-1}"; VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"; SEARCH_REPEATS="${SEARCH_REPEATS:-1}"; VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"; MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_prefetch_guard_stagel_g${NGPU}}"; FINAL_ENV="${FINAL_ENV:-${PREFIX}_winner.env}"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-guard-sweep.sh"
mkdir -p "$(dirname "$PREFIX")" "$(dirname "$FINAL_ENV")"
for f in "$STAGE_F_ENV" "$UPSTREAM_PREPARE_ENV" "$UPSTREAM_WINNER_ENV" "$SWEEP"; do [[ -s "$f" ]] || { echo "missing Stage-L input=$f" >&2; exit 2; }; done
[[ "$NGPU" =~ ^[1-8]$ ]] || exit 2
for n in SEARCH_REPEATS VALIDATE_REPEATS; do v="${!n}"; [[ "$v" =~ ^[1-9][0-9]*$ ]] || exit 2; done
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY
# shellcheck disable=SC1090
source "$STAGE_F_ENV"
for k in B300_HYBRID8_NEXTSELF_STAGE_E_CORE_ROWS B300_HYBRID8_NEXTSELF_STAGE_E_CORE_RESIDUE B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_RESIDUE B300_HYBRID8_NEXTSELF_FINAL_STAGE_ROWS B300_HYBRID8_NEXTSELF_FINAL_STAGE_RESIDUE; do [[ -n "${!k+x}" ]] || { echo "Stage-F env missing $k" >&2; exit 3; }; done
# shellcheck disable=SC1090
source "$UPSTREAM_PREPARE_ENV"
UPSTREAM_KIND=""
if [[ "${B300_STAGEK_PREPARED:-0}" == 1 ]]; then
  UPSTREAM_KIND=stagek; SW="$B300_STAGEK_PREPARED_SELF_WIDTH"; SD="$B300_STAGEK_PREPARED_SELF_DISTANCE"; SE="$B300_STAGEK_PREPARED_SELF_EVICT"; MW="$B300_STAGEK_PREPARED_MATE_WIDTH"; MD="$B300_STAGEK_PREPARED_MATE_DISTANCE"; ME="$B300_STAGEK_PREPARED_MATE_EVICT"
elif [[ "${B300_STAGEJ_PREPARED:-0}" == 1 ]]; then
  UPSTREAM_KIND=stagej; SW="$B300_STAGEJ_PREPARED_SELF_WIDTH"; SD="$B300_STAGEJ_PREPARED_SELF_DISTANCE"; SE="$B300_STAGEJ_PREPARED_SELF_EVICT"; MW="$B300_STAGEJ_PREPARED_MATE_WIDTH"; MD="$B300_STAGEJ_PREPARED_MATE_DISTANCE"; ME="$B300_STAGEJ_PREPARED_MATE_EVICT"
else echo 'Stage-L upstream prepare is not Stage J/K' >&2; exit 3; fi
for w in "$SW" "$MW"; do case "$w" in 1|2|4|8) ;; *) exit 3;; esac; done
for d in "$SD" "$MD"; do case "$d" in 1|2|4) ;; *) exit 3;; esac; done
for e in "$SE" "$ME"; do case "$e" in default|normal|last) ;; *) exit 3;; esac; done
# shellcheck disable=SC1090
source "$UPSTREAM_WINNER_ENV"
if [[ "$UPSTREAM_KIND" == stagek ]]; then
  for k in B300_STAGEK_STAGED_VALIDATED B300_STAGEK_FINAL_STAGE_ROWS B300_STAGEK_FINAL_STAGE_RESIDUE; do [[ -n "${!k+x}" ]] || exit 3; done
  [[ "$B300_STAGEK_STAGED_VALIDATED" == 1 ]] || exit 4; UP_ROWS="$B300_STAGEK_FINAL_STAGE_ROWS"; UP_RES="$B300_STAGEK_FINAL_STAGE_RESIDUE"
else
  for k in B300_STAGEJ_STAGED_VALIDATED B300_STAGEJ_FINAL_STAGE_ROWS B300_STAGEJ_FINAL_STAGE_RESIDUE; do [[ -n "${!k+x}" ]] || exit 3; done
  [[ "$B300_STAGEJ_STAGED_VALIDATED" == 1 ]] || exit 4; UP_ROWS="$B300_STAGEJ_FINAL_STAGE_ROWS"; UP_RES="$B300_STAGEJ_FINAL_STAGE_RESIDUE"
fi
stage_validate_rows=()
for rows in $VALIDATE_ROWS "$UP_ROWS" "$B300_HYBRID8_NEXTSELF_FINAL_STAGE_ROWS" "$B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS"; do
  [[ "$rows" == "$SEARCH_ROWS" ]] && continue
  [[ "$rows" =~ ^[1-9][0-9]*$ ]] && ((rows<=28)) || exit 2
  seen=0; for old in "${stage_validate_rows[@]}"; do [[ "$old" == "$rows" ]] && seen=1; done; ((seen)) || stage_validate_rows+=("$rows")
done
check_residue(){
  local rows="$1" got="$2"
  if [[ "$rows" == "$B300_HYBRID8_NEXTSELF_STAGE_E_CORE_ROWS" && "$got" != "$B300_HYBRID8_NEXTSELF_STAGE_E_CORE_RESIDUE" ]]; then echo "FATAL Stage-L/core residue mismatch rows=$rows" >&2; exit 4; fi
  if [[ "$rows" == "$B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS" && "$got" != "$B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_RESIDUE" ]]; then echo "FATAL Stage-L/Stage-E-final residue mismatch rows=$rows" >&2; exit 4; fi
  if [[ "$rows" == "$B300_HYBRID8_NEXTSELF_FINAL_STAGE_ROWS" && "$got" != "$B300_HYBRID8_NEXTSELF_FINAL_STAGE_RESIDUE" ]]; then echo "FATAL Stage-L/Stage-F-final residue mismatch rows=$rows" >&2; exit 4; fi
  if [[ "$rows" == "$UP_ROWS" && "$got" != "$UP_RES" ]]; then echo "FATAL Stage-L/upstream-final residue mismatch rows=$rows" >&2; exit 4; fi
}
passes(){ python3 - "$1" "$MIN_SPEEDUP" <<'PY'
import sys
print(1 if float(sys.argv[1]) >= float(sys.argv[2]) else 0)
PY
}
profile_of(){
  if [[ "$1" == branch && "$2" == branch ]]; then echo bb
  elif [[ "$1" == predicated && "$2" == branch ]]; then echo pb
  elif [[ "$1" == branch && "$2" == predicated ]]; then echo bp
  elif [[ "$1" == predicated && "$2" == predicated ]]; then echo pp
  else return 2; fi
}
run_stage(){
  local rows="$1" self_guards="$2" mate_guards="$3" threads="$4" repeats="$5" tag="$6" p="${PREFIX}.${tag}.r${rows}" log="${p}.log" envf="${p}_winner.env"
  STAGE_F_ENV="$STAGE_F_ENV" ARCH="$ARCH" MOD="$MOD" ROWS="$rows" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" NGPU="$NGPU" SELF_WIDTH="$SW" SELF_DISTANCE="$SD" SELF_EVICT="$SE" MATE_WIDTH="$MW" MATE_DISTANCE="$MD" MATE_EVICT="$ME" SELF_GUARD_LIST="$self_guards" MATE_GUARD_LIST="$mate_guards" THREADS_LIST="$threads" REPEATS="$repeats" PREFIX="$p" WINNER_ENV="$envf" bash "$SWEEP" | tee "$log" >&2
  grep -Fq 'b300_stagel_exact_match=1' "$log" || { echo "Stage-L exact gate missing rows=$rows" >&2; exit 4; }
  grep -Fq "b300_stagel_ngpu=$NGPU" "$log" || exit 4
  [[ -s "$envf" ]] || exit 4
  printf '%s\n' "$envf"
}
SEARCH_ENV="$(run_stage "$SEARCH_ROWS" 'branch predicated' 'branch predicated' '128 256 512' "$SEARCH_REPEATS" search)"
# shellcheck disable=SC1090
source "$SEARCH_ENV"; check_residue "$SEARCH_ROWS" "$B300_STAGEL_RESIDUE"
[[ "$B300_STAGEL_NGPU" == "$NGPU" ]] || exit 4
SELECTED="$(profile_of "$B300_STAGEL_SELF_GUARD" "$B300_STAGEL_MATE_GUARD")"; VALIDATED=0; CURRENT_ENV="$SEARCH_ENV"
if [[ "$B300_STAGEL_BEST_ENABLED" == 1 && "$B300_STAGEL_CONTROL_SPILL_FREE" == 1 && "$B300_STAGEL_SPILL_FREE" == 1 && "$(passes "$B300_STAGEL_SPEEDUP")" == 1 ]]; then
  [[ "$SELECTED" != bb ]] || exit 4
  VALIDATED=1; bt="$B300_STAGEL_CONTROL_THREADS"; tt="$B300_STAGEL_THREADS"; vt="$bt"; [[ "$tt" == "$bt" ]] || vt+=" $tt"
  if [[ "$B300_STAGEL_SELF_GUARD" == predicated ]]; then VSG='branch predicated'; else VSG=branch; fi
  if [[ "$B300_STAGEL_MATE_GUARD" == predicated ]]; then VMG='branch predicated'; else VMG=branch; fi
  stage=0
  for rows in "${stage_validate_rows[@]}"; do
    ((stage+=1)); CURRENT_ENV="$(run_stage "$rows" "$VSG" "$VMG" "$vt" "$VALIDATE_REPEATS" "validate${stage}")"
    # shellcheck disable=SC1090
    source "$CURRENT_ENV"; check_residue "$rows" "$B300_STAGEL_RESIDUE"
    got="$(profile_of "$B300_STAGEL_SELF_GUARD" "$B300_STAGEL_MATE_GUARD")"
    [[ "$B300_STAGEL_NGPU" == "$NGPU" ]] || exit 4
    if [[ "$B300_STAGEL_BEST_ENABLED" != 1 || "$got" != "$SELECTED" || "$B300_STAGEL_CONTROL_SPILL_FREE" != 1 || "$B300_STAGEL_SPILL_FREE" != 1 || "$(passes "$B300_STAGEL_SPEEDUP")" != 1 ]]; then VALIDATED=0; break; fi
    bt="$B300_STAGEL_CONTROL_THREADS"; tt="$B300_STAGEL_THREADS"; vt="$bt"; [[ "$tt" == "$bt" ]] || vt+=" $tt"
  done
fi
# shellcheck disable=SC1090
source "$CURRENT_ENV"
if [[ "$VALIDATED" == 1 ]]; then FINAL_PROFILE="$SELECTED"; FINAL_SG="$B300_STAGEL_SELF_GUARD"; FINAL_MG="$B300_STAGEL_MATE_GUARD"; FINAL_BIN="$B300_STAGEL_BIN"; FINAL_THREADS="$B300_STAGEL_THREADS"; FINAL_WALL="$B300_STAGEL_WALL_S"; FINAL_HIGH="$B300_STAGEL_HIGH_S"; FINAL_SPEED="$B300_STAGEL_SPEEDUP"; else FINAL_PROFILE=bb; FINAL_SG=branch; FINAL_MG=branch; FINAL_BIN="$B300_STAGEL_CONTROL_BIN"; FINAL_THREADS="$B300_STAGEL_CONTROL_THREADS"; FINAL_WALL="$B300_STAGEL_CONTROL_WALL_S"; FINAL_HIGH="$B300_STAGEL_CONTROL_HIGH_S"; FINAL_SPEED=1.000000000; fi
{
  printf 'B300_STAGEL_STAGED_VALIDATED=%q\n' "$VALIDATED"
  printf 'B300_STAGEL_FINAL_ENABLED=%q\n' "$VALIDATED"
  printf 'B300_STAGEL_NGPU=%q\n' "$NGPU"
  printf 'B300_STAGEL_UPSTREAM_KIND=%q\n' "$UPSTREAM_KIND"
  printf 'B300_STAGEL_SELF_WIDTH=%q\n' "$SW"
  printf 'B300_STAGEL_SELF_DISTANCE=%q\n' "$SD"
  printf 'B300_STAGEL_SELF_EVICT=%q\n' "$SE"
  printf 'B300_STAGEL_MATE_WIDTH=%q\n' "$MW"
  printf 'B300_STAGEL_MATE_DISTANCE=%q\n' "$MD"
  printf 'B300_STAGEL_MATE_EVICT=%q\n' "$ME"
  printf 'B300_STAGEL_FINAL_PROFILE=%q\n' "$FINAL_PROFILE"
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
  printf 'B300_STAGEL_UPSTREAM_PREPARE_ENV=%q\n' "$UPSTREAM_PREPARE_ENV"
  printf 'B300_STAGEL_UPSTREAM_WINNER_ENV=%q\n' "$UPSTREAM_WINNER_ENV"
  printf 'B300_STAGEL_MIN_SPEEDUP=%q\n' "$MIN_SPEEDUP"
} >"$FINAL_ENV"
cat "$FINAL_ENV"
echo "b300-nextgen-hybrid8-prefetch-guard-staged-calibrate OK canonical_sweep=1 validated=$VALIDATED ngpu=$NGPU upstream=$UPSTREAM_KIND profile=$FINAL_PROFILE guards=$FINAL_SG/$FINAL_MG speedup=$FINAL_SPEED final_rows=$B300_STAGEL_ROWS" >&2
