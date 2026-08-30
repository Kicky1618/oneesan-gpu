#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"; STAGEN_WINNER_ENV="${STAGEN_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_staged_g${NGPU}_winner.env}"; STAGEN_PREPARE_ENV="${STAGEN_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_fullprime_n27_prepared.env}"
STAGEO_PREPARE_ENV="${STAGEO_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageo_cg_l2_fullprime_n27_prepared.env}"; STAGEP_PREPARE_ENV="${STAGEP_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stagep_mate_cg_l2_fullprime_n27_prepared.env}"; STAGEQ_PREPARE_ENV="${STAGEQ_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageq_ilp8_count_cg_l2_fullprime_n27_prepared.env}"
STAGER_WINNER_ENV="${STAGER_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stager_ilp2_load_staged_g${NGPU}_winner.env}"; STAGER_PREPARE_ENV="${STAGER_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stager_ilp2_load_fullprime_n27_prepared.env}"; STAGES_WINNER_ENV="${STAGES_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stages_ilp2_cg_l2_staged_g${NGPU}_winner.env}"; STAGES_PREPARE_ENV="${STAGES_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stages_ilp2_cg_l2_fullprime_n27_prepared.env}"
UPSTREAM_KIND="${UPSTREAM_KIND:-auto}"; SEARCH_ROWS="${SEARCH_ROWS:-1}"; VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"; SEARCH_REPEATS="${SEARCH_REPEATS:-1}"; VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"; MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"; L2_LIST="${L2_LIST:-0 64 128 256}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageu_ilp2_mate_joint_staged_g${NGPU}}"; FINAL_ENV="${FINAL_ENV:-${PREFIX}_winner.env}"; SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-stageu-ilp2-mate-cg-l2-joint-sweep.sh"
mkdir -p "$(dirname "$PREFIX")" "$(dirname "$FINAL_ENV")"; [[ -s "$SWEEP" ]] || exit 2
case "$UPSTREAM_KIND" in auto|stager|stages) ;; *) exit 2;; esac
[[ "$SEARCH_ROWS" =~ ^[1-9][0-9]*$ ]] && ((SEARCH_ROWS<=28)) || exit 2; [[ "$NGPU" =~ ^[1-8]$ ]] || exit 2
for x in SEARCH_REPEATS VALIDATE_REPEATS; do [[ "${!x}" =~ ^[1-9][0-9]*$ ]] || exit 2; done
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1])<1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY
l2s=(); has0=0; nonzero=0
for b in $L2_LIST; do case "$b" in 0|64|128|256) ;; *) echo "bad Stage-U L2=$b" >&2; exit 2;; esac; [[ "$b" == 0 ]] && has0=1 || nonzero=1; seen=0; for old in "${l2s[@]}"; do [[ "$old" == "$b" ]] && seen=1; done; ((seen)) || l2s+=("$b"); done
((has0&&nonzero)) || { echo 'L2_LIST must include 0 and at least one nonzero hint' >&2; exit 2; }
passes(){ python3 - "$1" "$MIN_SPEEDUP" <<'PY'
import sys
print(1 if float(sys.argv[1])>=float(sys.argv[2]) else 0)
PY
}
run_stage(){
  local rows="$1" l2list="$2" threads="$3" repeats="$4" tag="$5" fixed_up="$6"
  local p="${PREFIX}.${tag}.r${rows}" log="${p}.log" envf="${p}_winner.env"
  STAGE_F_ENV="$STAGE_F_ENV" STAGEN_WINNER_ENV="$STAGEN_WINNER_ENV" STAGEN_PREPARE_ENV="$STAGEN_PREPARE_ENV" STAGEO_PREPARE_ENV="$STAGEO_PREPARE_ENV" STAGEP_PREPARE_ENV="$STAGEP_PREPARE_ENV" STAGEQ_PREPARE_ENV="$STAGEQ_PREPARE_ENV" STAGER_WINNER_ENV="$STAGER_WINNER_ENV" STAGER_PREPARE_ENV="$STAGER_PREPARE_ENV" STAGES_WINNER_ENV="$STAGES_WINNER_ENV" STAGES_PREPARE_ENV="$STAGES_PREPARE_ENV" \
    UPSTREAM_KIND="$fixed_up" ARCH="$ARCH" MOD="$MOD" ROWS="$rows" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" NGPU="$NGPU" L2_LIST="$l2list" THREADS_LIST="$threads" REPEATS="$repeats" PREFIX="$p" WINNER_ENV="$envf" \
    bash "$SWEEP" | tee "$log" >&2
  grep -Fq 'b300_stageu_exact_match=1' "$log" || { echo "Stage-U exact gate missing rows=$rows" >&2; exit 4; }
  [[ -s "$envf" ]] || exit 4
  printf '%s\n' "$envf"
}
SEARCH_ENV="$(run_stage "$SEARCH_ROWS" "$L2_LIST" "$THREADS_LIST" "$SEARCH_REPEATS" search "$UPSTREAM_KIND")"
# shellcheck disable=SC1090
source "$SEARCH_ENV"
RES_UP="$B300_STAGEU_UPSTREAM_KIND"; R_UP="$B300_STAGEU_STAGER_UPSTREAM_KIND"; LOW_PAIR="$B300_STAGEU_LOW_PAIR_POLICY"; LOW_BLOCK="$B300_STAGEU_LOW_BLOCK_POLICY"; LOW_PL2="$B300_STAGEU_LOW_PAIR_L2_BYTES"; LOW_BL2="$B300_STAGEU_LOW_BLOCK_L2_BYTES"; HIGH_PAIR="$B300_STAGEU_HIGH_PAIR_POLICY"; HIGH_BLOCK="$B300_STAGEU_HIGH_BLOCK_POLICY"; HIGH_PL2="$B300_STAGEU_HIGH_PAIR_L2_BYTES"; HIGH_BL2="$B300_STAGEU_HIGH_BLOCK_L2_BYTES"; HIGH_MATE="$B300_STAGEU_HIGH_MATE_POLICY"; HIGH_MATE_L2="$B300_STAGEU_HIGH_MATE_L2_BYTES"; UP_ROWS="$B300_STAGEU_UPSTREAM_ROWS"; UP_RES="$B300_STAGEU_UPSTREAM_RESIDUE"; UP_MANIFEST="$B300_STAGEU_UPSTREAM_MANIFEST"; SELECTED_L2="$B300_STAGEU_JOINT_BEST_L2_BYTES"; SELECTED_LABEL="$B300_STAGEU_JOINT_BEST_LABEL"
case "$RES_UP" in stager|stages) ;; *) echo 'Stage-U search resolved invalid immediate upstream' >&2; exit 4;; esac
case "$SELECTED_L2" in 0|64|128|256) ;; *) exit 4;; esac
check_config(){
  [[ "$B300_STAGEU_UPSTREAM_KIND" == "$RES_UP" && "$B300_STAGEU_STAGER_UPSTREAM_KIND" == "$R_UP" && "$B300_STAGEU_NGPU" == "$NGPU" ]] || { echo 'Stage-U upstream/GPU drift' >&2; exit 4; }
  [[ "$B300_STAGEU_LOW_PAIR_POLICY" == "$LOW_PAIR" && "$B300_STAGEU_LOW_BLOCK_POLICY" == "$LOW_BLOCK" && "$B300_STAGEU_LOW_PAIR_L2_BYTES" == "$LOW_PL2" && "$B300_STAGEU_LOW_BLOCK_L2_BYTES" == "$LOW_BL2" ]] || { echo 'Stage-U low Count provenance drift' >&2; exit 4; }
  [[ "$B300_STAGEU_HIGH_PAIR_POLICY" == "$HIGH_PAIR" && "$B300_STAGEU_HIGH_BLOCK_POLICY" == "$HIGH_BLOCK" && "$B300_STAGEU_HIGH_PAIR_L2_BYTES" == "$HIGH_PL2" && "$B300_STAGEU_HIGH_BLOCK_L2_BYTES" == "$HIGH_BL2" && "$B300_STAGEU_HIGH_MATE_POLICY" == "$HIGH_MATE" && "$B300_STAGEU_HIGH_MATE_L2_BYTES" == "$HIGH_MATE_L2" ]] || { echo 'Stage-U high-state provenance drift' >&2; exit 4; }
  [[ "$B300_STAGEU_CONTROL_POLICY" == default && "$B300_STAGEU_UPSTREAM_MANIFEST" == "$UP_MANIFEST" ]] || { echo 'Stage-U exact control provenance drift' >&2; exit 4; }
}
check_residue(){ local rows="$1" got="$2"; if [[ "$rows" == "$UP_ROWS" && "$got" != "$UP_RES" ]]; then echo "FATAL Stage-U/upstream residue mismatch rows=$rows got=$got expected=$UP_RES" >&2; exit 4; fi; }
check_config; check_residue "$SEARCH_ROWS" "$B300_STAGEU_RESIDUE"
VALIDATED=0; CURRENT_ENV="$SEARCH_ENV"; LAST_ROWS="$SEARCH_ROWS"
if [[ "$B300_STAGEU_BEST_ENABLED" == 1 && "$B300_STAGEU_CONTROL_SPILL_FREE" == 1 && "$B300_STAGEU_SPILL_FREE" == 1 && "$SELECTED_LABEL" == u_cg_l2_* && "$SELECTED_L2" != 0 && "$(passes "$B300_STAGEU_SPEEDUP")" == 1 ]]; then
  VALIDATED=1
  control_threads="$B300_STAGEU_CONTROL_THREADS"; test_threads="$B300_STAGEU_THREADS"; validation_threads="$control_threads"; [[ "$test_threads" == "$control_threads" ]] || validation_threads+=" $test_threads"
  validate_rows=(); for rows in $VALIDATE_ROWS "$UP_ROWS"; do [[ "$rows" =~ ^[1-9][0-9]*$ ]] || continue; [[ "$rows" == "$SEARCH_ROWS" ]] && continue; seen=0; for old in "${validate_rows[@]}"; do [[ "$old" == "$rows" ]] && seen=1; done; ((seen)) || validate_rows+=("$rows"); done
  stage=0
  for rows in "${validate_rows[@]}"; do
    ((stage+=1)); CURRENT_ENV="$(run_stage "$rows" "0 $SELECTED_L2" "$validation_threads" "$VALIDATE_REPEATS" "validate${stage}" "$RES_UP")"; LAST_ROWS="$rows"
    # shellcheck disable=SC1090
    source "$CURRENT_ENV"; check_config; check_residue "$rows" "$B300_STAGEU_RESIDUE"
    if [[ "$B300_STAGEU_BEST_ENABLED" != 1 || "$B300_STAGEU_JOINT_BEST_LABEL" != "$SELECTED_LABEL" || "$B300_STAGEU_JOINT_BEST_POLICY" != cg || "$B300_STAGEU_JOINT_BEST_L2_BYTES" != "$SELECTED_L2" || "$B300_STAGEU_CONTROL_SPILL_FREE" != 1 || "$B300_STAGEU_SPILL_FREE" != 1 || "$(passes "$B300_STAGEU_SPEEDUP")" != 1 ]]; then VALIDATED=0; break; fi
    control_threads="$B300_STAGEU_CONTROL_THREADS"; test_threads="$B300_STAGEU_THREADS"; validation_threads="$control_threads"; [[ "$test_threads" == "$control_threads" ]] || validation_threads+=" $test_threads"
  done
fi
# shellcheck disable=SC1090
source "$CURRENT_ENV"; check_config
if [[ "$VALIDATED" == 1 ]]; then
  FINAL_LABEL="$B300_STAGEU_JOINT_BEST_LABEL"; FINAL_POLICY=cg; FINAL_L2="$B300_STAGEU_JOINT_BEST_L2_BYTES"; FINAL_BIN="$B300_STAGEU_BIN"; FINAL_THREADS="$B300_STAGEU_THREADS"; FINAL_WALL="$B300_STAGEU_WALL_S"; FINAL_HIGH="$B300_STAGEU_HIGH_S"; FINAL_SPEED="$B300_STAGEU_SPEEDUP"
else
  FINAL_LABEL=control; FINAL_POLICY=default; FINAL_L2=0; FINAL_BIN="$B300_STAGEU_CONTROL_BIN"; FINAL_THREADS="$B300_STAGEU_CONTROL_THREADS"; FINAL_WALL="$B300_STAGEU_CONTROL_WALL_S"; FINAL_HIGH="$B300_STAGEU_CONTROL_HIGH_S"; FINAL_SPEED=1.000000000
fi
{
  printf 'B300_STAGEU_STAGED_VALIDATED=%q\n' "$VALIDATED"; printf 'B300_STAGEU_FINAL_ENABLED=%q\n' "$VALIDATED"; printf 'B300_STAGEU_NGPU=%q\n' "$NGPU"; printf 'B300_STAGEU_UPSTREAM_KIND=%q\n' "$RES_UP"; printf 'B300_STAGEU_STAGER_UPSTREAM_KIND=%q\n' "$R_UP"
  printf 'B300_STAGEU_LOW_PAIR_POLICY=%q\n' "$LOW_PAIR"; printf 'B300_STAGEU_LOW_BLOCK_POLICY=%q\n' "$LOW_BLOCK"; printf 'B300_STAGEU_LOW_PAIR_L2_BYTES=%q\n' "$LOW_PL2"; printf 'B300_STAGEU_LOW_BLOCK_L2_BYTES=%q\n' "$LOW_BL2"; printf 'B300_STAGEU_HIGH_PAIR_POLICY=%q\n' "$HIGH_PAIR"; printf 'B300_STAGEU_HIGH_BLOCK_POLICY=%q\n' "$HIGH_BLOCK"; printf 'B300_STAGEU_HIGH_PAIR_L2_BYTES=%q\n' "$HIGH_PL2"; printf 'B300_STAGEU_HIGH_BLOCK_L2_BYTES=%q\n' "$HIGH_BL2"; printf 'B300_STAGEU_HIGH_MATE_POLICY=%q\n' "$HIGH_MATE"; printf 'B300_STAGEU_HIGH_MATE_L2_BYTES=%q\n' "$HIGH_MATE_L2"
  printf 'B300_STAGEU_CONTROL_POLICY=default\n'; printf 'B300_STAGEU_FINAL_LABEL=%q\n' "$FINAL_LABEL"; printf 'B300_STAGEU_FINAL_POLICY=%q\n' "$FINAL_POLICY"; printf 'B300_STAGEU_FINAL_L2_BYTES=%q\n' "$FINAL_L2"; printf 'B300_STAGEU_FINAL_BIN=%q\n' "$FINAL_BIN"; printf 'B300_STAGEU_FINAL_THREADS=%q\n' "$FINAL_THREADS"; printf 'B300_STAGEU_FINAL_WALL_S=%q\n' "$FINAL_WALL"; printf 'B300_STAGEU_FINAL_HIGH_S=%q\n' "$FINAL_HIGH"; printf 'B300_STAGEU_FINAL_SPEEDUP=%q\n' "$FINAL_SPEED"; printf 'B300_STAGEU_FINAL_SPILL_FREE=1\n'; printf 'B300_STAGEU_CONTROL_BIN=%q\n' "$B300_STAGEU_CONTROL_BIN"; printf 'B300_STAGEU_CONTROL_THREADS=%q\n' "$B300_STAGEU_CONTROL_THREADS"; printf 'B300_STAGEU_FINAL_STAGE_ROWS=%q\n' "$LAST_ROWS"; printf 'B300_STAGEU_FINAL_STAGE_RESIDUE=%q\n' "$B300_STAGEU_RESIDUE"; printf 'B300_STAGEU_UPSTREAM_ROWS=%q\n' "$UP_ROWS"; printf 'B300_STAGEU_UPSTREAM_RESIDUE=%q\n' "$UP_RES"; printf 'B300_STAGEU_UPSTREAM_MANIFEST=%q\n' "$UP_MANIFEST"; printf 'B300_STAGEU_SEARCH_L2=%q\n' "$L2_LIST"; printf 'B300_STAGEU_SELECTED_L2_BYTES=%q\n' "$SELECTED_L2"; printf 'B300_STAGEU_SELECTED_LABEL=%q\n' "$SELECTED_LABEL"; printf 'B300_STAGEU_MIN_SPEEDUP=%q\n' "$MIN_SPEEDUP"
} >"$FINAL_ENV"
cat "$FINAL_ENV"
echo "b300-nextgen-hybrid8-stageu-joint-staged-calibrate OK stage=U validated=$VALIDATED upstream=$RES_UP r_upstream=$R_UP selected_l2=$SELECTED_L2 final_l2=$FINAL_L2 speedup=${FINAL_SPEED}x final_rows=$LAST_ROWS ngpu=$NGPU" >&2
