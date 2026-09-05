#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
STAGEN_WINNER_ENV="${STAGEN_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_staged_g${NGPU}_winner.env}"; STAGEN_PREPARE_ENV="${STAGEN_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_fullprime_n27_prepared.env}"
STAGEO_PREPARE_ENV="${STAGEO_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageo_cg_l2_fullprime_n27_prepared.env}"; STAGEP_PREPARE_ENV="${STAGEP_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stagep_mate_cg_l2_fullprime_n27_prepared.env}"
STAGEQ_WINNER_ENV="${STAGEQ_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageq_ilp8_count_cg_l2_staged_g${NGPU}_winner.env}"; STAGEQ_PREPARE_ENV="${STAGEQ_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageq_ilp8_count_cg_l2_fullprime_n27_prepared.env}"
UPSTREAM_KIND="${UPSTREAM_KIND:-auto}"; SEARCH_ROWS="${SEARCH_ROWS:-1}"; VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"; SEARCH_REPEATS="${SEARCH_REPEATS:-1}"; VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"; PAIR_POLICY_LIST="${PAIR_POLICY_LIST:-default cg cs}"; BLOCK_POLICY_LIST="${BLOCK_POLICY_LIST:-default cg cs}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stager_ilp2_load_staged_g${NGPU}}"; FINAL_ENV="${FINAL_ENV:-${PREFIX}_winner.env}"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-stager-ilp2-load-policy-sweep.sh"
mkdir -p "$(dirname "$PREFIX")" "$(dirname "$FINAL_ENV")"; [[ -s "$SWEEP" ]] || exit 2
case "$UPSTREAM_KIND" in auto|stagen|stageo|stagep|stageq) ;; *) exit 2;; esac
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
  STAGE_F_ENV="$STAGE_F_ENV" STAGEN_WINNER_ENV="$STAGEN_WINNER_ENV" STAGEN_PREPARE_ENV="$STAGEN_PREPARE_ENV" STAGEO_PREPARE_ENV="$STAGEO_PREPARE_ENV" STAGEP_PREPARE_ENV="$STAGEP_PREPARE_ENV" STAGEQ_WINNER_ENV="$STAGEQ_WINNER_ENV" STAGEQ_PREPARE_ENV="$STAGEQ_PREPARE_ENV" \
    UPSTREAM_KIND="$upstream" ARCH="$ARCH" MOD="$MOD" ROWS="$rows" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" NGPU="$NGPU" PAIR_POLICY_LIST="$pair_list" BLOCK_POLICY_LIST="$block_list" THREADS_LIST="$threads" REPEATS="$repeats" PREFIX="$p" WINNER_ENV="$envf" bash "$SWEEP" | tee "$log" >&2
  grep -Fq 'b300_stager_exact_match=1' "$log" || { echo "Stage-R exact gate missing rows=$rows" >&2; exit 4; }; [[ -s "$envf" ]] || exit 4; printf '%s\n' "$envf"
}
SEARCH_ENV="$(run_stage "$SEARCH_ROWS" "$UPSTREAM_KIND" "$PAIR_POLICY_LIST" "$BLOCK_POLICY_LIST" "$THREADS_LIST" "$SEARCH_REPEATS" search)"
# shellcheck disable=SC1090
source "$SEARCH_ENV"
RESOLVED="$B300_STAGER_UPSTREAM_KIND"; Q_UP="$B300_STAGER_STAGEQ_UPSTREAM_KIND"; BASE_PAIR="$B300_STAGER_UPSTREAM_PAIR_POLICY"; BASE_BLOCK="$B300_STAGER_UPSTREAM_BLOCK_POLICY"; HIGH_PAIR="$B300_STAGER_HIGH_PAIR_POLICY"; HIGH_BLOCK="$B300_STAGER_HIGH_BLOCK_POLICY"; HIGH_PL2="$B300_STAGER_HIGH_PAIR_L2_BYTES"; HIGH_BL2="$B300_STAGER_HIGH_BLOCK_L2_BYTES"; UP_ROWS="$B300_STAGER_UPSTREAM_ROWS"; UP_RES="$B300_STAGER_UPSTREAM_RESIDUE"; UP_MANIFEST="$B300_STAGER_UPSTREAM_MANIFEST"; SELECTED_PAIR="$B300_STAGER_PAIR_POLICY"; SELECTED_BLOCK="$B300_STAGER_BLOCK_POLICY"
check_config(){
  [[ "$B300_STAGER_UPSTREAM_KIND" == "$RESOLVED" && "$B300_STAGER_STAGEQ_UPSTREAM_KIND" == "$Q_UP" && "$B300_STAGER_NGPU" == "$NGPU" ]] || { echo 'Stage-R upstream/GPU drift' >&2; exit 4; }
  [[ "$B300_STAGER_UPSTREAM_PAIR_POLICY" == "$BASE_PAIR" && "$B300_STAGER_UPSTREAM_BLOCK_POLICY" == "$BASE_BLOCK" && "$B300_STAGER_HIGH_PAIR_POLICY" == "$HIGH_PAIR" && "$B300_STAGER_HIGH_BLOCK_POLICY" == "$HIGH_BLOCK" && "$B300_STAGER_HIGH_PAIR_L2_BYTES" == "$HIGH_PL2" && "$B300_STAGER_HIGH_BLOCK_L2_BYTES" == "$HIGH_BL2" ]] || { echo 'Stage-R high/low baseline drift' >&2; exit 4; }
  [[ "$B300_STAGER_UPSTREAM_MANIFEST" == "$UP_MANIFEST" ]] || { echo 'Stage-R upstream manifest drift' >&2; exit 4; }
}
check_residue(){ local rows="$1" got="$2"; if [[ "$rows" == "$UP_ROWS" && "$got" != "$UP_RES" ]]; then echo "FATAL Stage-R/upstream residue mismatch rows=$rows got=$got expected=$UP_RES" >&2; exit 4; fi; }
check_config; check_residue "$SEARCH_ROWS" "$B300_STAGER_RESIDUE"
VALIDATED=0; CURRENT_ENV="$SEARCH_ENV"
if [[ "$B300_STAGER_BEST_ENABLED" == 1 && "$B300_STAGER_CONTROL_SPILL_FREE" == 1 && "$B300_STAGER_SPILL_FREE" == 1 && "$(passes "$B300_STAGER_SPEEDUP")" == 1 && ( "$SELECTED_PAIR" != "$BASE_PAIR" || "$SELECTED_BLOCK" != "$BASE_BLOCK" ) ]]; then
  VALIDATED=1; control_threads="$B300_STAGER_CONTROL_THREADS"; test_threads="$B300_STAGER_THREADS"; validation_threads="$control_threads"; [[ "$test_threads" == "$control_threads" ]] || validation_threads+=" $test_threads"
  validation_pair="$BASE_PAIR"; [[ "$SELECTED_PAIR" == "$BASE_PAIR" ]] || validation_pair+=" $SELECTED_PAIR"; validation_block="$BASE_BLOCK"; [[ "$SELECTED_BLOCK" == "$BASE_BLOCK" ]] || validation_block+=" $SELECTED_BLOCK"
  validate_rows=(); for rows in $VALIDATE_ROWS "$UP_ROWS"; do [[ "$rows" =~ ^[1-9][0-9]*$ ]] || continue; [[ "$rows" == "$SEARCH_ROWS" ]] && continue; seen=0; for old in "${validate_rows[@]}"; do [[ "$old" == "$rows" ]] && seen=1; done; ((seen)) || validate_rows+=("$rows"); done
  stage=0
  for rows in "${validate_rows[@]}"; do
    ((stage+=1)); CURRENT_ENV="$(run_stage "$rows" "$RESOLVED" "$validation_pair" "$validation_block" "$validation_threads" "$VALIDATE_REPEATS" "validate${stage}")"; source "$CURRENT_ENV"; check_config; check_residue "$rows" "$B300_STAGER_RESIDUE"
    if [[ "$B300_STAGER_PAIR_POLICY" != "$SELECTED_PAIR" || "$B300_STAGER_BLOCK_POLICY" != "$SELECTED_BLOCK" || "$B300_STAGER_BEST_ENABLED" != 1 || "$B300_STAGER_CONTROL_SPILL_FREE" != 1 || "$B300_STAGER_SPILL_FREE" != 1 || "$(passes "$B300_STAGER_SPEEDUP")" != 1 ]]; then VALIDATED=0; break; fi
    control_threads="$B300_STAGER_CONTROL_THREADS"; test_threads="$B300_STAGER_THREADS"; validation_threads="$control_threads"; [[ "$test_threads" == "$control_threads" ]] || validation_threads+=" $test_threads"
  done
fi
source "$CURRENT_ENV"; check_config
if [[ "$VALIDATED" == 1 ]]; then FINAL_PAIR="$B300_STAGER_PAIR_POLICY"; FINAL_BLOCK="$B300_STAGER_BLOCK_POLICY"; FINAL_BIN="$B300_STAGER_BIN"; FINAL_THREADS="$B300_STAGER_THREADS"; FINAL_WALL="$B300_STAGER_WALL_S"; FINAL_HIGH="$B300_STAGER_HIGH_S"; FINAL_SPEED="$B300_STAGER_SPEEDUP"; else FINAL_PAIR="$BASE_PAIR"; FINAL_BLOCK="$BASE_BLOCK"; FINAL_BIN="$B300_STAGER_CONTROL_BIN"; FINAL_THREADS="$B300_STAGER_CONTROL_THREADS"; FINAL_WALL="$B300_STAGER_CONTROL_WALL_S"; FINAL_HIGH="$B300_STAGER_CONTROL_HIGH_S"; FINAL_SPEED=1.000000000; fi
{
  printf 'B300_STAGER_STAGED_VALIDATED=%q\n' "$VALIDATED"; printf 'B300_STAGER_FINAL_ENABLED=%q\n' "$VALIDATED"; printf 'B300_STAGER_NGPU=%q\n' "$NGPU"; printf 'B300_STAGER_UPSTREAM_KIND=%q\n' "$RESOLVED"; printf 'B300_STAGER_STAGEQ_UPSTREAM_KIND=%q\n' "$Q_UP"
  printf 'B300_STAGER_UPSTREAM_PAIR_POLICY=%q\n' "$BASE_PAIR"; printf 'B300_STAGER_UPSTREAM_BLOCK_POLICY=%q\n' "$BASE_BLOCK"; printf 'B300_STAGER_HIGH_PAIR_POLICY=%q\n' "$HIGH_PAIR"; printf 'B300_STAGER_HIGH_BLOCK_POLICY=%q\n' "$HIGH_BLOCK"; printf 'B300_STAGER_HIGH_PAIR_L2_BYTES=%q\n' "$HIGH_PL2"; printf 'B300_STAGER_HIGH_BLOCK_L2_BYTES=%q\n' "$HIGH_BL2"
  printf 'B300_STAGER_PAIR_POLICY=%q\n' "$FINAL_PAIR"; printf 'B300_STAGER_BLOCK_POLICY=%q\n' "$FINAL_BLOCK"; printf 'B300_STAGER_FINAL_BIN=%q\n' "$FINAL_BIN"; printf 'B300_STAGER_FINAL_THREADS=%q\n' "$FINAL_THREADS"; printf 'B300_STAGER_FINAL_WALL_S=%q\n' "$FINAL_WALL"; printf 'B300_STAGER_FINAL_HIGH_S=%q\n' "$FINAL_HIGH"; printf 'B300_STAGER_FINAL_SPEEDUP=%q\n' "$FINAL_SPEED"; printf 'B300_STAGER_FINAL_SPILL_FREE=1\n'
  printf 'B300_STAGER_CONTROL_BIN=%q\n' "$B300_STAGER_CONTROL_BIN"; printf 'B300_STAGER_CONTROL_THREADS=%q\n' "$B300_STAGER_CONTROL_THREADS"; printf 'B300_STAGER_FINAL_STAGE_ROWS=%q\n' "$B300_STAGER_ROWS"; printf 'B300_STAGER_FINAL_STAGE_RESIDUE=%q\n' "$B300_STAGER_RESIDUE"; printf 'B300_STAGER_UPSTREAM_ROWS=%q\n' "$UP_ROWS"; printf 'B300_STAGER_UPSTREAM_RESIDUE=%q\n' "$UP_RES"; printf 'B300_STAGER_UPSTREAM_MANIFEST=%q\n' "$UP_MANIFEST"
  printf 'B300_STAGER_SEARCH_PAIR_POLICIES=%q\n' "$PAIR_POLICY_LIST"; printf 'B300_STAGER_SEARCH_BLOCK_POLICIES=%q\n' "$BLOCK_POLICY_LIST"; printf 'B300_STAGER_MIN_SPEEDUP=%q\n' "$MIN_SPEEDUP"
} >"$FINAL_ENV"
cat "$FINAL_ENV"; echo "b300-nextgen-hybrid8-stager-ilp2-load-policy-staged-calibrate OK stage=R validated=$VALIDATED upstream=$RESOLVED low_pair=$FINAL_PAIR low_block=$FINAL_BLOCK high_pair=$HIGH_PAIR high_block=$HIGH_BLOCK q_l2=${HIGH_PL2}/${HIGH_BL2} speedup=${FINAL_SPEED}x final_rows=$B300_STAGER_ROWS ngpu=$NGPU" >&2
