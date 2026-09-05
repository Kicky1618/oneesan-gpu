#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ROWS="${ROWS:-1}"
ARCH="${ARCH:-native}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
MOD="${MOD:-4294967291}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
HIGHDROP_LIST="${HIGHDROP_LIST:-0 1}"
SAMPLE_LOG2="${SAMPLE_LOG2:-20}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_forced_nextgen_calibrate_row${ROWS}}"
mkdir -p "$(dirname "$PREFIX")"

SWEEP_LOG="${PREFIX}.mainrec_sweep.log"
PROFILE_LOG="${PREFIX}.block_profile.log"
DUAL_LOG="${PREFIX}.mainrec_dualmask.log"

echo '=== stage 1: MAIN_RECURRENCE threads/high-drop sweep ===' >&2
ROWS="$ROWS" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" MOD="$MOD" \
THREADS_LIST="$THREADS_LIST" HIGHDROP_LIST="$HIGHDROP_LIST" PREFIX="${PREFIX}.mainrec" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-main-recurrence-thread-highdrop-sweep.sh" | tee "$SWEEP_LOG"

getv(){ local k="$1" f="$2"; sed -nE "s/^${k}=([^[:space:]]+).*/\\1/p" "$f" | tail -n1; }
BEST_HIGH="$(getv b300_mainrec_sweep_best_high_drop_chunk "$SWEEP_LOG")"
BEST_THREADS="$(getv b300_mainrec_sweep_best_threads "$SWEEP_LOG")"
BEST_WALL="$(getv b300_mainrec_sweep_best_wall_s "$SWEEP_LOG")"
MATCH="$(getv b300_mainrec_sweep_exact_intermediate_match "$SWEEP_LOG")"
[[ "$MATCH" == 1 ]] || { echo 'main recurrence sweep did not prove intermediate equality' >&2; exit 3; }
[[ "$BEST_HIGH" == 0 || "$BEST_HIGH" == 1 ]] || { echo "invalid best high-drop=$BEST_HIGH" >&2; exit 3; }
[[ "$BEST_THREADS" =~ ^[0-9]+$ ]] || { echo "invalid best threads=$BEST_THREADS" >&2; exit 3; }

echo "=== stage 2: sampled block workload high=$BEST_HIGH threads=$BEST_THREADS ===" >&2
ROWS="$ROWS" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" MOD="$MOD" \
THREADS="$BEST_THREADS" HIGH_DROP_CHUNK="$BEST_HIGH" SAMPLE_LOG2="$SAMPLE_LOG2" PREFIX="${PREFIX}.blockprof" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-block-pull-profile-row1.sh" | tee "$PROFILE_LOG"
[[ "$(getv b300_block_profile_exact_intermediate_match "$PROFILE_LOG")" == 1 ]] || { echo 'block profiler residue check failed' >&2; exit 4; }

echo "=== stage 3: production-like MAIN_RECURRENCE + dualmask A/B ===" >&2
ROWS="$ROWS" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" MOD="$MOD" \
THREADS="$BEST_THREADS" HIGH_DROP_CHUNK="$BEST_HIGH" PREFIX="${PREFIX}.dualmask" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-mainrec-dualmask-row1-ab.sh" | tee "$DUAL_LOG"
[[ "$(getv b300_mainrec_dualmask_exact_intermediate_match "$DUAL_LOG")" == 1 ]] || { echo 'mainrec dualmask residue check failed' >&2; exit 5; }

DUAL_SPEED="$(getv b300_mainrec_dualmask_wall_speedup "$DUAL_LOG")"
DUAL_SPEED_NUM="${DUAL_SPEED%x}"
ADOPT_DUAL="$(python3 - "$DUAL_SPEED_NUM" <<'PY'
import sys
x=float(sys.argv[1]);print(1 if x>1.0 else 0)
PY
)"

printf 'b300_forced_nextgen_exact_calibration=1\n'
printf 'b300_forced_nextgen_best_threads=%s\n' "$BEST_THREADS"
printf 'b300_forced_nextgen_best_high_drop_chunk=%s\n' "$BEST_HIGH"
printf 'b300_forced_nextgen_mainrec_row_wall_s=%s\n' "$BEST_WALL"
printf 'b300_forced_nextgen_dualmask_speedup=%s\n' "$DUAL_SPEED"
printf 'b300_forced_nextgen_adopt_dualmask=%s\n' "$ADOPT_DUAL"
printf 'b300_forced_nextgen_high_left_iters_per_closure=%s\n' "$(getv b300_block_profile_high_left_iters_per_closure "$PROFILE_LOG")"
printf 'b300_forced_nextgen_high_right_iters_per_closure=%s\n' "$(getv b300_block_profile_high_right_iters_per_closure "$PROFILE_LOG")"
printf 'b300_forced_nextgen_low_left_iters_per_closure=%s\n' "$(getv b300_block_profile_low_left_iters_per_closure "$PROFILE_LOG")"
printf 'b300_forced_nextgen_low_right_iters_per_closure=%s\n' "$(getv b300_block_profile_low_right_iters_per_closure "$PROFILE_LOG")"
printf 'b300_forced_nextgen_sweep_log=%s\n' "$SWEEP_LOG"
printf 'b300_forced_nextgen_profile_log=%s\n' "$PROFILE_LOG"
printf 'b300_forced_nextgen_dualmask_log=%s\n' "$DUAL_LOG"
printf 'b300_forced_nextgen_note=full exact intentionally not launched; use calibrated winner only after reviewing row-limited wall results\n'
