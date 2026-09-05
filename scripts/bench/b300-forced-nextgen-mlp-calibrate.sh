#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
ROWS="${ROWS:-1}"
MOD="${MOD:-4294967291}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
HIGHDROP_LIST="${HIGHDROP_LIST:-0 1}"
SAMPLE_LOG2="${SAMPLE_LOG2:-20}"
REPEATS="${REPEATS:-1}"
BATCH_LIST="${BATCH_LIST:-2 4}"
DUALMASK_MIN_SPEEDUP="${DUALMASK_MIN_SPEEDUP:-1.01}"
CLOSURE_BATCH_MIN_SPEEDUP="${CLOSURE_BATCH_MIN_SPEEDUP:-1.01}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_forced_nextgen_mlp_row${ROWS}}"
CORE_PREFIX="${CORE_PREFIX:-${PREFIX}.core}"
CORE_LOG="${CORE_LOG:-${PREFIX}.core.log}"
BATCH_PREFIX="${BATCH_PREFIX:-${PREFIX}.closure_batch}"
BATCH_LOG="${BATCH_LOG:-${PREFIX}.closure_batch.log}"
mkdir -p "$(dirname "$PREFIX")"

getv(){ local k="$1" f="$2";sed -nE "s/^${k}=([^[:space:]]+).*/\\1/p" "$f"|tail -n1; }

echo '=== forced MLP stage 1: recurrence/highdrop/dualmask calibration ===' >&2
ROWS="$ROWS" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" MOD="$MOD" \
THREADS_LIST="$THREADS_LIST" HIGHDROP_LIST="$HIGHDROP_LIST" SAMPLE_LOG2="$SAMPLE_LOG2" PREFIX="$CORE_PREFIX" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-forced-nextgen-calibrate.sh" | tee "$CORE_LOG"
[[ "$(getv b300_forced_nextgen_exact_calibration "$CORE_LOG")" == 1 ]]||{ echo 'core forced calibration exact gate failed' >&2;exit 3; }
BEST_THREADS="$(getv b300_forced_nextgen_best_threads "$CORE_LOG")"
BEST_HIGH="$(getv b300_forced_nextgen_best_high_drop_chunk "$CORE_LOG")"
DUAL_SPEED_RAW="$(getv b300_forced_nextgen_dualmask_speedup "$CORE_LOG")";DUAL_SPEED="${DUAL_SPEED_RAW%x}"
[[ "$BEST_THREADS" =~ ^[0-9]+$ ]]||{ echo 'missing best threads' >&2;exit 3; }
[[ "$BEST_HIGH" == 0 || "$BEST_HIGH" == 1 ]]||{ echo 'missing best highdrop' >&2;exit 3; }
ADOPT_DUAL="$(python3 - "$DUAL_SPEED" "$DUALMASK_MIN_SPEEDUP" <<'PY'
import sys
x,m=map(float,sys.argv[1:]);print(1 if x>=m else 0)
PY
)"
echo "CORE WINNER threads=$BEST_THREADS highdrop=$BEST_HIGH dualmask=$ADOPT_DUAL measured_dual=${DUAL_SPEED}x" >&2

echo '=== forced MLP stage 2: closure source-load batching ===' >&2
# Re-sweep threads because closure batching changes block kernel register pressure
# and can move the occupancy optimum away from the main-recurrence-only winner.
ROWS="$ROWS" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" MOD="$MOD" \
THREADS_LIST="$THREADS_LIST" BATCH_LIST="$BATCH_LIST" REPEATS="$REPEATS" HIGH_DROP_CHUNK="$BEST_HIGH" DUALMASK="$ADOPT_DUAL" PREFIX="$BATCH_PREFIX" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-mainrec-closure-batch-sweep.sh" | tee "$BATCH_LOG"
[[ "$(getv b300_mainrec_closure_batch_exact_intermediate_match "$BATCH_LOG")" == 1 ]]||{ echo 'closure batch exact gate failed' >&2;exit 4; }
BATCH="$(getv b300_mainrec_closure_batch_best_batch "$BATCH_LOG")"
BATCH_THREADS="$(getv b300_mainrec_closure_batch_best_threads "$BATCH_LOG")"
BATCH_WALL="$(getv b300_mainrec_closure_batch_best_wall_s "$BATCH_LOG")"
BATCH_SPEED_RAW="$(getv b300_mainrec_closure_batch_speedup_vs_global_base_best "$BATCH_LOG")";BATCH_SPEED="${BATCH_SPEED_RAW%x}"
[[ "$BATCH" == 2 || "$BATCH" == 4 ]]||{ echo 'closure batch winner missing' >&2;exit 4; }
[[ "$BATCH_THREADS" =~ ^[0-9]+$ ]]||{ echo 'closure batch threads missing' >&2;exit 4; }
ADOPT_BATCH="$(python3 - "$BATCH_SPEED" "$CLOSURE_BATCH_MIN_SPEEDUP" <<'PY'
import sys
x,m=map(float,sys.argv[1:]);print(1 if x>=m else 0)
PY
)"
FINAL_BATCH=0;FINAL_THREADS="$BEST_THREADS"
if [[ "$ADOPT_BATCH" == 1 ]];then FINAL_BATCH="$BATCH";FINAL_THREADS="$BATCH_THREADS";fi

printf 'b300_forced_nextgen_mlp_exact_gates=1\n'
printf 'b300_forced_nextgen_mlp_high_drop_chunk=%s\n' "$BEST_HIGH"
printf 'b300_forced_nextgen_mlp_dualmask=%s\n' "$ADOPT_DUAL"
printf 'b300_forced_nextgen_mlp_dualmask_speedup=%sx\n' "$DUAL_SPEED"
printf 'b300_forced_nextgen_mlp_dualmask_min_speedup=%sx\n' "$DUALMASK_MIN_SPEEDUP"
printf 'b300_forced_nextgen_mlp_closure_batch_measured=%s\n' "$BATCH"
printf 'b300_forced_nextgen_mlp_closure_batch_threads=%s\n' "$BATCH_THREADS"
printf 'b300_forced_nextgen_mlp_closure_batch_wall_s=%s\n' "$BATCH_WALL"
printf 'b300_forced_nextgen_mlp_closure_batch_speedup=%sx\n' "$BATCH_SPEED"
printf 'b300_forced_nextgen_mlp_closure_batch_min_speedup=%sx\n' "$CLOSURE_BATCH_MIN_SPEEDUP"
printf 'b300_forced_nextgen_mlp_adopt_closure_batch=%s\n' "$ADOPT_BATCH"
printf 'b300_forced_nextgen_mlp_final_threads=%s\n' "$FINAL_THREADS"
printf 'b300_forced_nextgen_mlp_final_closure_batch=%s\n' "$FINAL_BATCH"
printf 'b300_forced_nextgen_mlp_core_log=%s\n' "$CORE_LOG"
printf 'b300_forced_nextgen_mlp_batch_log=%s\n' "$BATCH_LOG"
printf 'b300_forced_nextgen_mlp_note=closure batching is adopted only on global baseline wall win; if rejected final threads revert to the core recurrence winner\n'
