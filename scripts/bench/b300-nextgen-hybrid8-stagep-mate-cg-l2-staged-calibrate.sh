#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
STAGEN_WINNER_ENV="${STAGEN_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_staged_g${NGPU}_winner.env}"
STAGEN_PREPARE_ENV="${STAGEN_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_fullprime_n27_prepared.env}"
STAGEO_WINNER_ENV="${STAGEO_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageo_cg_l2_staged_g${NGPU}_winner.env}"
STAGEO_PREPARE_ENV="${STAGEO_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageo_cg_l2_fullprime_n27_prepared.env}"
UPSTREAM_KIND="${UPSTREAM_KIND:-auto}"; SEARCH_ROWS="${SEARCH_ROWS:-1}"; VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"; SEARCH_REPEATS="${SEARCH_REPEATS:-1}"; VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"; MATE_L2_LIST="${MATE_L2_LIST:-0 64 128 256}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stagep_mate_cg_l2_staged_g${NGPU}}"; FINAL_ENV="${FINAL_ENV:-${PREFIX}_winner.env}"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-stagep-mate-cg-l2-sweep.sh"
mkdir -p "$(dirname "$PREFIX")" "$(dirname "$FINAL_ENV")"
for f in "$STAGE_F_ENV" "$STAGEN_WINNER_ENV" "$STAGEN_PREPARE_ENV" "$SWEEP"; do [[ -s "$f" ]] || { echo "missing Stage-P staged input=$f" >&2; exit 2; }; done
case "$UPSTREAM_KIND" in auto|stagen|stageo) ;; *) exit 2;; esac
[[ "$SEARCH_ROWS" =~ ^[1-9][0-9]*$ ]] && ((SEARCH_ROWS<=28)) || exit 2; [[ "$NGPU" =~ ^[1-8]$ ]] || exit 2
for x in SEARCH_REPEATS VALIDATE_REPEATS; do [[ "${!x}" =~ ^[1-9][0-9]*$ ]] || exit 2; done
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY
passes(){ python3 - "$1" "$MIN_SPEEDUP" <<'PY'
import sys
print(1 if float(sys.argv[1]) >= float(sys.argv[2]) else 0)
PY
}
run_stage(){
  local rows="$1" upstream="$2" l2list="$3" threads="$4" repeats="$5" tag="$6"
  local p="${PREFIX}.${tag}.r${rows}" log="${p}.log" envf="${p}_winner.env"
  STAGE_F_ENV="$STAGE_F_ENV" STAGEN_WINNER_ENV="$STAGEN_WINNER_ENV" STAGEN_PREPARE_ENV="$STAGEN_PREPARE_ENV" STAGEO_WINNER_ENV="$STAGEO_WINNER_ENV" STAGEO_PREPARE_ENV="$STAGEO_PREPARE_ENV" \
    UPSTREAM_KIND="$upstream" ARCH="$ARCH" MOD="$MOD" ROWS="$rows" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" NGPU="$NGPU" MATE_L2_LIST="$l2list" THREADS_LIST="$threads" REPEATS="$repeats" PREFIX="$p" WINNER_ENV="$envf" \
    bash "$SWEEP" | tee "$log" >&2
  grep -Fq 'b300_stagep_exact_match=1' "$log" || { echo "Stage-P exact gate missing rows=$rows" >&2; exit 4; }
  [[ -s "$envf" ]] || exit 4; printf '%s\n' "$envf"
}

SEARCH_ENV="$(run_stage "$SEARCH_ROWS" "$UPSTREAM_KIND" "$MATE_L2_LIST" "$THREADS_LIST" "$SEARCH_REPEATS" search)"
# shellcheck disable=SC1090
source "$SEARCH_ENV"
RESOLVED_UPSTREAM="$B300_STAGEP_COUNT_UPSTREAM"; UP_ROWS="$B300_STAGEP_UPSTREAM_ROWS"; UP_RES="$B300_STAGEP_UPSTREAM_RESIDUE"
check_config(){ [[ "$B300_STAGEP_COUNT_UPSTREAM" == "$RESOLVED_UPSTREAM" && "$B300_STAGEP_NGPU" == "$NGPU" && "$B300_STAGEP_MATE_LOAD_POLICY" == cg ]] || { echo 'Stage-P upstream/GPU/mate-policy drift' >&2; exit 4; }; }
check_residue(){ local rows="$1" got="$2"; if [[ "$rows" == "$UP_ROWS" && "$got" != "$UP_RES" ]]; then echo "FATAL Stage-P/upstream residue mismatch rows=$rows got=$got expected=$UP_RES" >&2; exit 4; fi; }
check_config; check_residue "$SEARCH_ROWS" "$B300_STAGEP_RESIDUE"
VALIDATED=0; CURRENT_ENV="$SEARCH_ENV"; SELECTED_L2="$B300_STAGEP_MATE_L2_BYTES"
if [[ "$B300_STAGEP_BEST_ENABLED" == 1 && "$B300_STAGEP_CONTROL_SPILL_FREE" == 1 && "$B300_STAGEP_SPILL_FREE" == 1 && "$(passes "$B300_STAGEP_SPEEDUP")" == 1 ]]; then
  VALIDATED=1
  control_threads="$B300_STAGEP_CONTROL_THREADS"; test_threads="$B300_STAGEP_THREADS"; validation_threads="$control_threads"; [[ "$test_threads" == "$control_threads" ]] || validation_threads+=" $test_threads"
  validation_l2="0"; [[ "$SELECTED_L2" == 0 ]] || validation_l2+=" $SELECTED_L2"
  validate_rows=(); for rows in $VALIDATE_ROWS "$UP_ROWS"; do [[ "$rows" =~ ^[1-9][0-9]*$ ]] || continue; [[ "$rows" == "$SEARCH_ROWS" ]] && continue; seen=0; for old in "${validate_rows[@]}"; do [[ "$old" == "$rows" ]] && seen=1; done; ((seen)) || validate_rows+=("$rows"); done
  stage=0
  for rows in "${validate_rows[@]}"; do
    ((stage+=1)); CURRENT_ENV="$(run_stage "$rows" "$RESOLVED_UPSTREAM" "$validation_l2" "$validation_threads" "$VALIDATE_REPEATS" "validate${stage}")"
    # shellcheck disable=SC1090
    source "$CURRENT_ENV"; check_config; check_residue "$rows" "$B300_STAGEP_RESIDUE"
    if [[ "$B300_STAGEP_MATE_L2_BYTES" != "$SELECTED_L2" || "$B300_STAGEP_BEST_ENABLED" != 1 || "$B300_STAGEP_CONTROL_SPILL_FREE" != 1 || "$B300_STAGEP_SPILL_FREE" != 1 || "$(passes "$B300_STAGEP_SPEEDUP")" != 1 ]]; then VALIDATED=0; break; fi
    control_threads="$B300_STAGEP_CONTROL_THREADS"; test_threads="$B300_STAGEP_THREADS"; validation_threads="$control_threads"; [[ "$test_threads" == "$control_threads" ]] || validation_threads+=" $test_threads"
  done
fi
# shellcheck disable=SC1090
source "$CURRENT_ENV"; check_config
if [[ "$VALIDATED" == 1 ]]; then FINAL_L2="$B300_STAGEP_MATE_L2_BYTES"; FINAL_BIN="$B300_STAGEP_BIN"; FINAL_THREADS="$B300_STAGEP_THREADS"; FINAL_WALL="$B300_STAGEP_WALL_S"; FINAL_HIGH="$B300_STAGEP_HIGH_S"; FINAL_SPEED="$B300_STAGEP_SPEEDUP"; else FINAL_L2=0; FINAL_BIN="$B300_STAGEP_CONTROL_BIN"; FINAL_THREADS="$B300_STAGEP_CONTROL_THREADS"; FINAL_WALL="$B300_STAGEP_CONTROL_WALL_S"; FINAL_HIGH="$B300_STAGEP_CONTROL_HIGH_S"; FINAL_SPEED=1.000000000; fi
{
  printf 'B300_STAGEP_STAGED_VALIDATED=%q\n' "$VALIDATED"; printf 'B300_STAGEP_FINAL_ENABLED=%q\n' "$VALIDATED"; printf 'B300_STAGEP_NGPU=%q\n' "$NGPU"; printf 'B300_STAGEP_COUNT_UPSTREAM=%q\n' "$RESOLVED_UPSTREAM"
  printf 'B300_STAGEP_PAIR_POLICY=%q\n' "$B300_STAGEP_PAIR_POLICY"; printf 'B300_STAGEP_BLOCK_POLICY=%q\n' "$B300_STAGEP_BLOCK_POLICY"; printf 'B300_STAGEP_MATE_LOAD_POLICY=cg\n'
  printf 'B300_STAGEP_BASE_CG_L2_BYTES=%q\n' "$B300_STAGEP_BASE_CG_L2_BYTES"; printf 'B300_STAGEP_PAIR_CG_L2_BYTES=%q\n' "$B300_STAGEP_PAIR_CG_L2_BYTES"; printf 'B300_STAGEP_BLOCK_CG_L2_BYTES=%q\n' "$B300_STAGEP_BLOCK_CG_L2_BYTES"; printf 'B300_STAGEP_BASE_MATE_L2_BYTES=0\n'; printf 'B300_STAGEP_MATE_L2_BYTES=%q\n' "$FINAL_L2"
  printf 'B300_STAGEP_FINAL_BIN=%q\n' "$FINAL_BIN"; printf 'B300_STAGEP_FINAL_THREADS=%q\n' "$FINAL_THREADS"; printf 'B300_STAGEP_FINAL_WALL_S=%q\n' "$FINAL_WALL"; printf 'B300_STAGEP_FINAL_HIGH_S=%q\n' "$FINAL_HIGH"; printf 'B300_STAGEP_FINAL_SPEEDUP=%q\n' "$FINAL_SPEED"; printf 'B300_STAGEP_FINAL_SPILL_FREE=1\n'
  printf 'B300_STAGEP_CONTROL_BIN=%q\n' "$B300_STAGEP_CONTROL_BIN"; printf 'B300_STAGEP_CONTROL_THREADS=%q\n' "$B300_STAGEP_CONTROL_THREADS"; printf 'B300_STAGEP_FINAL_STAGE_ROWS=%q\n' "$B300_STAGEP_ROWS"; printf 'B300_STAGEP_FINAL_STAGE_RESIDUE=%q\n' "$B300_STAGEP_RESIDUE"
  printf 'B300_STAGEP_UPSTREAM_ROWS=%q\n' "$UP_ROWS"; printf 'B300_STAGEP_UPSTREAM_RESIDUE=%q\n' "$UP_RES"; printf 'B300_STAGEP_SEARCH_MATE_L2=%q\n' "$MATE_L2_LIST"; printf 'B300_STAGEP_MIN_SPEEDUP=%q\n' "$MIN_SPEEDUP"
  printf 'B300_STAGEP_INPUT_STAGE_F_ENV=%q\n' "$STAGE_F_ENV"; printf 'B300_STAGEP_INPUT_STAGEN_WINNER_ENV=%q\n' "$STAGEN_WINNER_ENV"; printf 'B300_STAGEP_INPUT_STAGEN_PREPARE_ENV=%q\n' "$STAGEN_PREPARE_ENV"; printf 'B300_STAGEP_INPUT_STAGEO_WINNER_ENV=%q\n' "$STAGEO_WINNER_ENV"; printf 'B300_STAGEP_INPUT_STAGEO_PREPARE_ENV=%q\n' "$STAGEO_PREPARE_ENV"
} >"$FINAL_ENV"
cat "$FINAL_ENV"
echo "b300-nextgen-hybrid8-stagep-mate-cg-l2-staged-calibrate OK stage=P validated=$VALIDATED count_upstream=$RESOLVED_UPSTREAM mate_l2=$FINAL_L2 speedup=${FINAL_SPEED}x final_rows=$B300_STAGEP_ROWS ngpu=$NGPU" >&2
