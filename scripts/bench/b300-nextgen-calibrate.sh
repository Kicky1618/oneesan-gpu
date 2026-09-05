#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}";MOD="${MOD:-4294967291}";ROWS="${ROWS:-1}";TARGET_MIB="${TARGET_MIB:-65536}";MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS_LIST="${THREADS_LIST:-128 256 512}";HIGHDROP_LIST="${HIGHDROP_LIST:-0 1}";REPEATS="${REPEATS:-1}"
MAIN_MIN_SPEEDUP="${MAIN_MIN_SPEEDUP:-1.01}";BLOCK_MIN_SPEEDUP="${BLOCK_MIN_SPEEDUP:-1.01}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_calibrate_row${ROWS}}";MAIN_LOG="${MAIN_LOG:-${PREFIX}.main.log}";BLOCK_LOG="${BLOCK_LOG:-${PREFIX}.block.log}"
mkdir -p "$(dirname "$PREFIX")"
getv(){ local k="$1" f="$2";sed -nE "s/^${k}=([^[:space:]]+).*/\\1/p" "$f"|tail -n1; }

# Stage A: choose highdrop + zero-extra-state main MLP policy.
echo '=== nextgen calibration stage A: main ILP/CG ===' >&2
ARCH="$ARCH" MOD="$MOD" ROWS="$ROWS" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" THREADS_LIST="$THREADS_LIST" HIGHDROP_LIST="$HIGHDROP_LIST" REPEATS="$REPEATS" TRANSFORM_MIN_SPEEDUP="$MAIN_MIN_SPEEDUP" PREFIX="${PREFIX}.main" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-mainrec-ilpcg-calibrate.sh" | tee "$MAIN_LOG"
[[ "$(getv b300_mainrec_ilpcg_calibrate_exact_gates "$MAIN_LOG")" == 1 ]]||{ echo 'main calibration exact gate missing' >&2;exit 3; }
RES_A="$(getv b300_mainrec_ilpcg_calibrate_residue "$MAIN_LOG")"
MAIN_HIGH="$(getv b300_mainrec_ilpcg_calibrate_final_high_drop_chunk "$MAIN_LOG")";MAIN_ILP="$(getv b300_mainrec_ilpcg_calibrate_final_ilp "$MAIN_LOG")";MAIN_CG="$(getv b300_mainrec_ilpcg_calibrate_final_random_cg "$MAIN_LOG")";MAIN_THREADS_A="$(getv b300_mainrec_ilpcg_calibrate_final_threads "$MAIN_LOG")"
GLOBAL_BASE_HIGH="$(getv b300_mainrec_ilpcg_calibrate_global_base_high_drop "$MAIN_LOG")";GLOBAL_BASE_THREADS="$(getv b300_mainrec_ilpcg_calibrate_global_base_threads "$MAIN_LOG")";GLOBAL_BASE_WALL="$(getv b300_mainrec_ilpcg_calibrate_global_base_wall_s "$MAIN_LOG")"

# Stage B: around that main policy, retune runtime threads while evaluating only
# block transforms. The Stage-B base is therefore the main-only candidate that
# should be retained for the full-prime arbitration.
echo "=== nextgen calibration stage B: block transforms around hd=$MAIN_HIGH ilp=$MAIN_ILP cg=$MAIN_CG ===" >&2
ARCH="$ARCH" MOD="$MOD" ROWS="$ROWS" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" HIGH_DROP_CHUNK="$MAIN_HIGH" RECURRENCE_ILP="$MAIN_ILP" RANDOM_CG="$MAIN_CG" THREADS_LIST="$THREADS_LIST" REPEATS="$REPEATS" PREFIX="${PREFIX}.block" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-block-combo-sweep.sh" | tee "$BLOCK_LOG"
[[ "$(getv b300_nextgen_block_combo_exact_intermediate_match "$BLOCK_LOG")" == 1 ]]||{ echo 'block calibration exact gate missing' >&2;exit 3; }
RES_B="$(getv b300_nextgen_block_combo_residue "$BLOCK_LOG")";[[ "$RES_A" == "$RES_B" ]]||{ echo "FATAL stage residue mismatch main=$RES_A block=$RES_B" >&2;exit 4; }
MAIN_ONLY_THREADS="$(getv b300_nextgen_block_combo_base_best_threads "$BLOCK_LOG")";MAIN_ONLY_WALL="$(getv b300_nextgen_block_combo_base_best_wall_s "$BLOCK_LOG")"
BEST_MODE="$(getv b300_nextgen_block_combo_best_mode "$BLOCK_LOG")";BEST_DUAL="$(getv b300_nextgen_block_combo_best_dualmask "$BLOCK_LOG")";BEST_BATCH="$(getv b300_nextgen_block_combo_best_closure_batch "$BLOCK_LOG")";BEST_THREADS="$(getv b300_nextgen_block_combo_best_threads "$BLOCK_LOG")";BEST_WALL="$(getv b300_nextgen_block_combo_best_wall_s "$BLOCK_LOG")";BLOCK_SPEED="$(getv b300_nextgen_block_combo_speedup_vs_base "$BLOCK_LOG")";BLOCK_SPEED="${BLOCK_SPEED%x}"

BLOCK_DECISION="$(python3 - "$BLOCK_SPEED" "$BLOCK_MIN_SPEEDUP" "$BEST_MODE" "$BEST_DUAL" "$BEST_BATCH" "$BEST_THREADS" "$BEST_WALL" "$MAIN_ONLY_THREADS" "$MAIN_ONLY_WALL" <<'PY'
import sys
sp,cut=map(float,sys.argv[1:3]);mode,dual,batch,threads,wall,bt,bw=sys.argv[3:10]
adopt=mode!='base' and sp>=cut
if adopt:print('\t'.join([dual,batch,threads,wall,mode,'1']))
else:print('\t'.join(['0','0',bt,bw,'base','0']))
PY
)"
IFS=$'\t' read -r FINAL_DUAL FINAL_BATCH FINAL_THREADS FINAL_WALL FINAL_BLOCK_MODE ADOPT_BLOCK<<<"$BLOCK_DECISION"
TOTAL_SPEED="$(python3 - "$GLOBAL_BASE_WALL" "$FINAL_WALL" <<'PY'
import sys
print(f'{float(sys.argv[1])/float(sys.argv[2]):.9f}')
PY
)"

printf 'b300_nextgen_calibrate_exact_gates=1\n'
printf 'b300_nextgen_calibrate_residue=%s\n' "$RES_A"
printf 'b300_nextgen_calibrate_global_base_high_drop=%s\n' "$GLOBAL_BASE_HIGH"
printf 'b300_nextgen_calibrate_global_base_ilp=2\n'
printf 'b300_nextgen_calibrate_global_base_random_cg=0\n'
printf 'b300_nextgen_calibrate_global_base_dualmask=0\n'
printf 'b300_nextgen_calibrate_global_base_closure_batch=0\n'
printf 'b300_nextgen_calibrate_global_base_threads=%s\n' "$GLOBAL_BASE_THREADS"
printf 'b300_nextgen_calibrate_global_base_wall_s=%s\n' "$GLOBAL_BASE_WALL"
printf 'b300_nextgen_calibrate_main_high_drop=%s\n' "$MAIN_HIGH"
printf 'b300_nextgen_calibrate_main_ilp=%s\n' "$MAIN_ILP"
printf 'b300_nextgen_calibrate_main_random_cg=%s\n' "$MAIN_CG"
printf 'b300_nextgen_calibrate_main_only_threads=%s\n' "$MAIN_ONLY_THREADS"
printf 'b300_nextgen_calibrate_main_only_wall_s=%s\n' "$MAIN_ONLY_WALL"
printf 'b300_nextgen_calibrate_block_min_speedup=%sx\n' "$BLOCK_MIN_SPEEDUP"
printf 'b300_nextgen_calibrate_block_measured_speedup=%sx\n' "$BLOCK_SPEED"
printf 'b300_nextgen_calibrate_adopt_block_transform=%s\n' "$ADOPT_BLOCK"
printf 'b300_nextgen_calibrate_final_block_mode=%s\n' "$FINAL_BLOCK_MODE"
printf 'b300_nextgen_calibrate_final_high_drop=%s\n' "$MAIN_HIGH"
printf 'b300_nextgen_calibrate_final_ilp=%s\n' "$MAIN_ILP"
printf 'b300_nextgen_calibrate_final_random_cg=%s\n' "$MAIN_CG"
printf 'b300_nextgen_calibrate_final_dualmask=%s\n' "$FINAL_DUAL"
printf 'b300_nextgen_calibrate_final_closure_batch=%s\n' "$FINAL_BATCH"
printf 'b300_nextgen_calibrate_final_threads=%s\n' "$FINAL_THREADS"
printf 'b300_nextgen_calibrate_final_wall_s=%s\n' "$FINAL_WALL"
printf 'b300_nextgen_calibrate_final_speedup_vs_global_base=%sx\n' "$TOTAL_SPEED"
printf 'b300_nextgen_calibrate_main_log=%s\n' "$MAIN_LOG"
printf 'b300_nextgen_calibrate_block_log=%s\n' "$BLOCK_LOG"
printf 'b300_nextgen_calibrate_note=partial-row calibration retains global ILP2 baseline and main-only candidate for complete-prime arbitration\n'
