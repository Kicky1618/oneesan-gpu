#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
STAGEN_WINNER_ENV="${STAGEN_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_staged_g${NGPU}_winner.env}"
STAGEN_PREPARE_ENV="${STAGEN_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_fullprime_n27_prepared.env}"
STAGEO_WINNER_ENV="${STAGEO_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageo_cg_l2_staged_g${NGPU}_winner.env}"
STAGEO_PREPARE_ENV="${STAGEO_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageo_cg_l2_fullprime_n27_prepared.env}"
STAGEP_WINNER_ENV="${STAGEP_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stagep_mate_cg_l2_staged_g${NGPU}_winner.env}"
STAGEP_PREPARE_ENV="${STAGEP_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stagep_mate_cg_l2_fullprime_n27_prepared.env}"
UPSTREAM_KIND="${UPSTREAM_KIND:-auto}"; SEARCH_ROWS="${SEARCH_ROWS:-1}"; VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"; SEARCH_REPEATS="${SEARCH_REPEATS:-1}"; VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"; PAIR_L2_LIST="${PAIR_L2_LIST:-0 64 128 256}"; BLOCK_L2_LIST="${BLOCK_L2_LIST:-0 64 128 256}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageq_ilp8_count_cg_l2_staged_g${NGPU}}"; FINAL_ENV="${FINAL_ENV:-${PREFIX}_winner.env}"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-stageq-ilp8-count-cg-l2-sweep.sh"
mkdir -p "$(dirname "$PREFIX")" "$(dirname "$FINAL_ENV")"
for f in "$STAGE_F_ENV" "$STAGEN_WINNER_ENV" "$STAGEN_PREPARE_ENV" "$SWEEP"; do [[ -s "$f" ]] || { echo "missing Stage-Q staged input=$f" >&2; exit 2; }; done
case "$UPSTREAM_KIND" in auto|stagen|stageo|stagep) ;; *) echo 'bad Stage-Q UPSTREAM_KIND' >&2; exit 2;; esac
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
  local rows="$1" upstream="$2" pair_list="$3" block_list="$4" threads="$5" repeats="$6" tag="$7"
  local p="${PREFIX}.${tag}.r${rows}" log="${p}.log" envf="${p}_winner.env"
  STAGE_F_ENV="$STAGE_F_ENV" STAGEN_WINNER_ENV="$STAGEN_WINNER_ENV" STAGEN_PREPARE_ENV="$STAGEN_PREPARE_ENV" \
    STAGEO_WINNER_ENV="$STAGEO_WINNER_ENV" STAGEO_PREPARE_ENV="$STAGEO_PREPARE_ENV" STAGEP_WINNER_ENV="$STAGEP_WINNER_ENV" STAGEP_PREPARE_ENV="$STAGEP_PREPARE_ENV" \
    UPSTREAM_KIND="$upstream" ARCH="$ARCH" MOD="$MOD" ROWS="$rows" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" NGPU="$NGPU" \
    PAIR_L2_LIST="$pair_list" BLOCK_L2_LIST="$block_list" THREADS_LIST="$threads" REPEATS="$repeats" PREFIX="$p" WINNER_ENV="$envf" \
    bash "$SWEEP" | tee "$log" >&2
  grep -Fq 'b300_stageq_exact_match=1' "$log" || { echo "Stage-Q exact gate missing rows=$rows" >&2; exit 4; }
  [[ -s "$envf" ]] || exit 4; printf '%s\n' "$envf"
}

SEARCH_ENV="$(run_stage "$SEARCH_ROWS" "$UPSTREAM_KIND" "$PAIR_L2_LIST" "$BLOCK_L2_LIST" "$THREADS_LIST" "$SEARCH_REPEATS" search)"
# shellcheck disable=SC1090
source "$SEARCH_ENV"
RESOLVED_UPSTREAM="$B300_STAGEQ_UPSTREAM_KIND"; RESOLVED_STAGEP_COUNT="$B300_STAGEQ_STAGEP_COUNT_UPSTREAM"
UP_PAIR="$B300_STAGEQ_UPSTREAM_PAIR_L2_BYTES"; UP_BLOCK="$B300_STAGEQ_UPSTREAM_BLOCK_L2_BYTES"; UP_ROWS="$B300_STAGEQ_UPSTREAM_ROWS"; UP_RES="$B300_STAGEQ_UPSTREAM_RESIDUE"; UP_MANIFEST="$B300_STAGEQ_UPSTREAM_MANIFEST"
LOCK_PAIR_POLICY="$B300_STAGEQ_PAIR_POLICY"; LOCK_BLOCK_POLICY="$B300_STAGEQ_BLOCK_POLICY"; LOCK_MATE_POLICY="$B300_STAGEQ_MATE_LOAD_POLICY"; LOCK_MATE_L2="$B300_STAGEQ_MATE_CG_L2_BYTES"
LOCK_BASE_L2="$B300_STAGEQ_BASE_CG_L2_BYTES"; LOCK_BUILDER_PAIR="$B300_STAGEQ_BUILDER_PAIR_CG_L2_BYTES"; LOCK_BUILDER_BLOCK="$B300_STAGEQ_BUILDER_BLOCK_CG_L2_BYTES"
SELECTED_PAIR="$B300_STAGEQ_PAIR_L2_BYTES"; SELECTED_BLOCK="$B300_STAGEQ_BLOCK_L2_BYTES"
check_config(){
  [[ "$B300_STAGEQ_UPSTREAM_KIND" == "$RESOLVED_UPSTREAM" && "$B300_STAGEQ_STAGEP_COUNT_UPSTREAM" == "$RESOLVED_STAGEP_COUNT" && "$B300_STAGEQ_NGPU" == "$NGPU" ]] || { echo 'Stage-Q upstream/GPU drift' >&2; exit 4; }
  [[ "$B300_STAGEQ_PAIR_POLICY" == "$LOCK_PAIR_POLICY" && "$B300_STAGEQ_BLOCK_POLICY" == "$LOCK_BLOCK_POLICY" && "$B300_STAGEQ_MATE_LOAD_POLICY" == "$LOCK_MATE_POLICY" ]] || { echo 'Stage-Q policy drift' >&2; exit 4; }
  [[ "$B300_STAGEQ_MATE_CG_L2_BYTES" == "$LOCK_MATE_L2" && "$B300_STAGEQ_BASE_CG_L2_BYTES" == "$LOCK_BASE_L2" && "$B300_STAGEQ_BUILDER_PAIR_CG_L2_BYTES" == "$LOCK_BUILDER_PAIR" && "$B300_STAGEQ_BUILDER_BLOCK_CG_L2_BYTES" == "$LOCK_BUILDER_BLOCK" ]] || { echo 'Stage-Q upstream compile knob drift' >&2; exit 4; }
  [[ "$B300_STAGEQ_UPSTREAM_PAIR_L2_BYTES" == "$UP_PAIR" && "$B300_STAGEQ_UPSTREAM_BLOCK_L2_BYTES" == "$UP_BLOCK" ]] || { echo 'Stage-Q exact upstream L2 tuple drift' >&2; exit 4; }
  [[ "$B300_STAGEQ_UPSTREAM_MANIFEST" == "$UP_MANIFEST" ]] || { echo 'Stage-Q upstream manifest drift' >&2; exit 4; }
}
check_residue(){ local rows="$1" got="$2"; if [[ "$rows" == "$UP_ROWS" && "$got" != "$UP_RES" ]]; then echo "FATAL Stage-Q/upstream residue mismatch rows=$rows got=$got expected=$UP_RES" >&2; exit 4; fi; }
check_config; check_residue "$SEARCH_ROWS" "$B300_STAGEQ_RESIDUE"

VALIDATED=0; CURRENT_ENV="$SEARCH_ENV"
if [[ "$B300_STAGEQ_BEST_ENABLED" == 1 && "$B300_STAGEQ_CONTROL_SPILL_FREE" == 1 && "$B300_STAGEQ_SPILL_FREE" == 1 && "$(passes "$B300_STAGEQ_SPEEDUP")" == 1 && ( "$SELECTED_PAIR" != "$UP_PAIR" || "$SELECTED_BLOCK" != "$UP_BLOCK" ) ]]; then
  VALIDATED=1
  control_threads="$B300_STAGEQ_CONTROL_THREADS"; test_threads="$B300_STAGEQ_THREADS"; validation_threads="$control_threads"; [[ "$test_threads" == "$control_threads" ]] || validation_threads+=" $test_threads"
  validation_pair="$UP_PAIR"; [[ "$SELECTED_PAIR" == "$UP_PAIR" ]] || validation_pair+=" $SELECTED_PAIR"
  validation_block="$UP_BLOCK"; [[ "$SELECTED_BLOCK" == "$UP_BLOCK" ]] || validation_block+=" $SELECTED_BLOCK"
  validate_rows=(); for rows in $VALIDATE_ROWS "$UP_ROWS"; do [[ "$rows" =~ ^[1-9][0-9]*$ ]] || continue; [[ "$rows" == "$SEARCH_ROWS" ]] && continue; seen=0; for old in "${validate_rows[@]}"; do [[ "$old" == "$rows" ]] && seen=1; done; ((seen)) || validate_rows+=("$rows"); done
  stage=0
  for rows in "${validate_rows[@]}"; do
    ((stage+=1)); CURRENT_ENV="$(run_stage "$rows" "$RESOLVED_UPSTREAM" "$validation_pair" "$validation_block" "$validation_threads" "$VALIDATE_REPEATS" "validate${stage}")"
    # shellcheck disable=SC1090
    source "$CURRENT_ENV"; check_config; check_residue "$rows" "$B300_STAGEQ_RESIDUE"
    if [[ "$B300_STAGEQ_PAIR_L2_BYTES" != "$SELECTED_PAIR" || "$B300_STAGEQ_BLOCK_L2_BYTES" != "$SELECTED_BLOCK" || "$B300_STAGEQ_BEST_ENABLED" != 1 || "$B300_STAGEQ_CONTROL_SPILL_FREE" != 1 || "$B300_STAGEQ_SPILL_FREE" != 1 || "$(passes "$B300_STAGEQ_SPEEDUP")" != 1 ]]; then VALIDATED=0; break; fi
    control_threads="$B300_STAGEQ_CONTROL_THREADS"; test_threads="$B300_STAGEQ_THREADS"; validation_threads="$control_threads"; [[ "$test_threads" == "$control_threads" ]] || validation_threads+=" $test_threads"
  done
fi
# shellcheck disable=SC1090
source "$CURRENT_ENV"; check_config
if [[ "$VALIDATED" == 1 ]]; then
  FINAL_PAIR="$B300_STAGEQ_PAIR_L2_BYTES"; FINAL_BLOCK="$B300_STAGEQ_BLOCK_L2_BYTES"; FINAL_BIN="$B300_STAGEQ_BIN"; FINAL_THREADS="$B300_STAGEQ_THREADS"; FINAL_WALL="$B300_STAGEQ_WALL_S"; FINAL_HIGH="$B300_STAGEQ_HIGH_S"; FINAL_SPEED="$B300_STAGEQ_SPEEDUP"
else
  FINAL_PAIR="$UP_PAIR"; FINAL_BLOCK="$UP_BLOCK"; FINAL_BIN="$B300_STAGEQ_CONTROL_BIN"; FINAL_THREADS="$B300_STAGEQ_CONTROL_THREADS"; FINAL_WALL="$B300_STAGEQ_CONTROL_WALL_S"; FINAL_HIGH="$B300_STAGEQ_CONTROL_HIGH_S"; FINAL_SPEED=1.000000000
fi
{
  printf 'B300_STAGEQ_STAGED_VALIDATED=%q\n' "$VALIDATED"; printf 'B300_STAGEQ_FINAL_ENABLED=%q\n' "$VALIDATED"; printf 'B300_STAGEQ_NGPU=%q\n' "$NGPU"
  printf 'B300_STAGEQ_UPSTREAM_KIND=%q\n' "$RESOLVED_UPSTREAM"; printf 'B300_STAGEQ_STAGEP_COUNT_UPSTREAM=%q\n' "$RESOLVED_STAGEP_COUNT"
  printf 'B300_STAGEQ_PAIR_POLICY=%q\n' "$LOCK_PAIR_POLICY"; printf 'B300_STAGEQ_BLOCK_POLICY=%q\n' "$LOCK_BLOCK_POLICY"; printf 'B300_STAGEQ_MATE_LOAD_POLICY=%q\n' "$LOCK_MATE_POLICY"
  printf 'B300_STAGEQ_BASE_CG_L2_BYTES=%q\n' "$LOCK_BASE_L2"; printf 'B300_STAGEQ_BUILDER_PAIR_CG_L2_BYTES=%q\n' "$LOCK_BUILDER_PAIR"; printf 'B300_STAGEQ_BUILDER_BLOCK_CG_L2_BYTES=%q\n' "$LOCK_BUILDER_BLOCK"; printf 'B300_STAGEQ_MATE_CG_L2_BYTES=%q\n' "$LOCK_MATE_L2"
  printf 'B300_STAGEQ_UPSTREAM_PAIR_L2_BYTES=%q\n' "$UP_PAIR"; printf 'B300_STAGEQ_UPSTREAM_BLOCK_L2_BYTES=%q\n' "$UP_BLOCK"; printf 'B300_STAGEQ_PAIR_L2_BYTES=%q\n' "$FINAL_PAIR"; printf 'B300_STAGEQ_BLOCK_L2_BYTES=%q\n' "$FINAL_BLOCK"
  printf 'B300_STAGEQ_FINAL_BIN=%q\n' "$FINAL_BIN"; printf 'B300_STAGEQ_FINAL_THREADS=%q\n' "$FINAL_THREADS"; printf 'B300_STAGEQ_FINAL_WALL_S=%q\n' "$FINAL_WALL"; printf 'B300_STAGEQ_FINAL_HIGH_S=%q\n' "$FINAL_HIGH"; printf 'B300_STAGEQ_FINAL_SPEEDUP=%q\n' "$FINAL_SPEED"; printf 'B300_STAGEQ_FINAL_SPILL_FREE=1\n'
  printf 'B300_STAGEQ_CONTROL_BIN=%q\n' "$B300_STAGEQ_CONTROL_BIN"; printf 'B300_STAGEQ_CONTROL_THREADS=%q\n' "$B300_STAGEQ_CONTROL_THREADS"; printf 'B300_STAGEQ_FINAL_STAGE_ROWS=%q\n' "$B300_STAGEQ_ROWS"; printf 'B300_STAGEQ_FINAL_STAGE_RESIDUE=%q\n' "$B300_STAGEQ_RESIDUE"
  printf 'B300_STAGEQ_UPSTREAM_ROWS=%q\n' "$UP_ROWS"; printf 'B300_STAGEQ_UPSTREAM_RESIDUE=%q\n' "$UP_RES"; printf 'B300_STAGEQ_UPSTREAM_MANIFEST=%q\n' "$UP_MANIFEST"
  printf 'B300_STAGEQ_SEARCH_PAIR_L2=%q\n' "$PAIR_L2_LIST"; printf 'B300_STAGEQ_SEARCH_BLOCK_L2=%q\n' "$BLOCK_L2_LIST"; printf 'B300_STAGEQ_MIN_SPEEDUP=%q\n' "$MIN_SPEEDUP"
  printf 'B300_STAGEQ_INPUT_STAGE_F_ENV=%q\n' "$STAGE_F_ENV"; printf 'B300_STAGEQ_INPUT_STAGEN_WINNER_ENV=%q\n' "$STAGEN_WINNER_ENV"; printf 'B300_STAGEQ_INPUT_STAGEN_PREPARE_ENV=%q\n' "$STAGEN_PREPARE_ENV"
  printf 'B300_STAGEQ_INPUT_STAGEO_WINNER_ENV=%q\n' "$STAGEO_WINNER_ENV"; printf 'B300_STAGEQ_INPUT_STAGEO_PREPARE_ENV=%q\n' "$STAGEO_PREPARE_ENV"; printf 'B300_STAGEQ_INPUT_STAGEP_WINNER_ENV=%q\n' "$STAGEP_WINNER_ENV"; printf 'B300_STAGEQ_INPUT_STAGEP_PREPARE_ENV=%q\n' "$STAGEP_PREPARE_ENV"
} >"$FINAL_ENV"
cat "$FINAL_ENV"
echo "b300-nextgen-hybrid8-stageq-ilp8-count-cg-l2-staged-calibrate OK stage=Q validated=$VALIDATED upstream=$RESOLVED_UPSTREAM pair_l2=$FINAL_PAIR block_l2=$FINAL_BLOCK speedup=${FINAL_SPEED}x final_rows=$B300_STAGEQ_ROWS ngpu=$NGPU" >&2
