#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
MOD="${MOD:-4294967291}"
ROWS="${ROWS:-1}"
PROFILE_THREADS="${PROFILE_THREADS:-256}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
REPEATS="${REPEATS:-1}"
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
SAMPLE_LOG2="${SAMPLE_LOG2:-20}"
HYBRID_MIN_SPEEDUP="${HYBRID_MIN_SPEEDUP:-1.01}"
DUALMASK_MIN_SPEEDUP="${DUALMASK_MIN_SPEEDUP:-1.01}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_closure_hybrid_auto_row${ROWS}}"
PROFILE_PREFIX="${PROFILE_PREFIX:-${PREFIX}.profile}"
HYBRID_PREFIX="${HYBRID_PREFIX:-${PREFIX}.hybrid}"
DUAL_PREFIX="${DUAL_PREFIX:-${PREFIX}.dual}"
PROFILE_LOG="${PROFILE_LOG:-${PREFIX}.profile.log}"
HYBRID_LOG="${HYBRID_LOG:-${PREFIX}.hybrid.log}"
DUAL_LOG="${DUAL_LOG:-${PREFIX}.dual.log}"
mkdir -p "$(dirname "$PREFIX")"

getv(){ local k="$1" f="$2"; sed -nE "s/^${k}=([^[:space:]]+).*/\\1/p" "$f" | tail -n1; }

echo '=== closure auto stage 1: sampled workload profile ===' >&2
MOD="$MOD" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
ROWS="$ROWS" THREADS="$PROFILE_THREADS" SAMPLE_LOG2="$SAMPLE_LOG2" PREFIX="$PROFILE_PREFIX" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-rankstate-closure-warp-profile.sh" | tee "$PROFILE_LOG"
[[ "$(getv b300_closure_warp_profile_exact_intermediate_match "$PROFILE_LOG")" == 1 ]] || { echo 'closure profile exact gate failed' >&2; exit 3; }
THRESHOLDS="$(sed -nE 's/^b300_closure_warp_profile_thresholds="(.*)"$/\1/p' "$PROFILE_LOG" | tail -n1)"
[[ -n "$THRESHOLDS" ]] || { echo 'profile did not produce threshold seed set' >&2; exit 3; }
echo "PROFILE-DERIVED THRESHOLDS: $THRESHOLDS" >&2

echo '=== closure auto stage 2: all-warp vs profile-derived hybrid thresholds ===' >&2
MOD="$MOD" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
ROWS="$ROWS" THREADS_LIST="$THREADS_LIST" THRESHOLDS="$THRESHOLDS" REPEATS="$REPEATS" PREFIX="$HYBRID_PREFIX" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-rankstate-closure-warp-hybrid-ab.sh" | tee "$HYBRID_LOG"
[[ "$(getv b300_closure_warp_hybrid_exact_intermediate_match "$HYBRID_LOG")" == 1 ]] || { echo 'hybrid exact gate failed' >&2; exit 4; }
BEST_TH="$(getv b300_closure_warp_hybrid_best_threshold "$HYBRID_LOG")"
BEST_THREADS="$(getv b300_closure_warp_hybrid_best_threads "$HYBRID_LOG")"
HYBRID_WALL="$(getv b300_closure_warp_hybrid_best_wall_s "$HYBRID_LOG")"
HYBRID_SPEED_RAW="$(getv b300_closure_warp_hybrid_speedup_vs_global_base_best "$HYBRID_LOG")"
HYBRID_SPEED="${HYBRID_SPEED_RAW%x}"
[[ "$BEST_TH" =~ ^[0-9]+$ && "$BEST_THREADS" =~ ^[0-9]+$ ]] || { echo 'hybrid winner metadata missing' >&2; exit 4; }

echo "HYBRID WINNER threshold=$BEST_TH threads=$BEST_THREADS speedup=${HYBRID_SPEED}x" >&2

echo '=== closure auto stage 3: incremental warp dualmask at selected threshold ===' >&2
MOD="$MOD" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
ROWS="$ROWS" THREADS_LIST="$BEST_THREADS" THRESHOLDS="$BEST_TH" REPEATS="$REPEATS" PREFIX="$DUAL_PREFIX" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-rankstate-closure-warp-hybrid-dualmask-ab.sh" | tee "$DUAL_LOG"
[[ "$(getv b300_closure_warp_hybrid_dualmask_exact_intermediate_match "$DUAL_LOG")" == 1 ]] || { echo 'hybrid dualmask exact gate failed' >&2; exit 5; }
DUAL_SPEED_KEY="b300_closure_warp_hybrid_dualmask_threshold_${BEST_TH}_threads_${BEST_THREADS}_speedup"
DUAL_WALL_KEY="b300_closure_warp_hybrid_dualmask_threshold_${BEST_TH}_threads_${BEST_THREADS}_dual_wall_s"
DUAL_BASE_KEY="b300_closure_warp_hybrid_dualmask_threshold_${BEST_TH}_threads_${BEST_THREADS}_base_wall_s"
DUAL_SPEED_RAW="$(getv "$DUAL_SPEED_KEY" "$DUAL_LOG")"; DUAL_SPEED="${DUAL_SPEED_RAW%x}"
DUAL_WALL="$(getv "$DUAL_WALL_KEY" "$DUAL_LOG")"; DUAL_BASE_WALL="$(getv "$DUAL_BASE_KEY" "$DUAL_LOG")"
[[ -n "$DUAL_SPEED" && -n "$DUAL_WALL" && -n "$DUAL_BASE_WALL" ]] || { echo 'selected dualmask result missing' >&2; exit 5; }

BASELINE_DRIFT="$(python3 - "$HYBRID_WALL" "$DUAL_BASE_WALL" <<'PY'
import sys
x,y=map(float,sys.argv[1:]);print(f'{abs(x-y)/min(x,y):.9f}' if min(x,y)>0 else '999')
PY
)"
PROMOTE_HYBRID="$(python3 - "$HYBRID_SPEED" "$HYBRID_MIN_SPEEDUP" "$BASELINE_DRIFT" <<'PY'
import sys
x,m,d=map(float,sys.argv[1:]);print(1 if x>=m and d<=.05 else 0)
PY
)"
ADOPT_DUAL="$(python3 - "$DUAL_SPEED" "$DUALMASK_MIN_SPEEDUP" <<'PY'
import sys
x,m=map(float,sys.argv[1:]);print(1 if x>=m else 0)
PY
)"
PORT="$(python3 - "$PROMOTE_HYBRID" "$ADOPT_DUAL" <<'PY'
import sys
h,d=map(int,sys.argv[1:]);print(1 if h else 0)
PY
)"

printf 'b300_closure_hybrid_auto_exact_gates=1\n'
printf 'b300_closure_hybrid_auto_profile_threads=%s\n' "$PROFILE_THREADS"
printf 'b300_closure_hybrid_auto_profile_thresholds="%s"\n' "$THRESHOLDS"
printf 'b300_closure_hybrid_auto_threshold=%s\n' "$BEST_TH"
printf 'b300_closure_hybrid_auto_threads=%s\n' "$BEST_THREADS"
printf 'b300_closure_hybrid_auto_hybrid_wall_s=%s\n' "$HYBRID_WALL"
printf 'b300_closure_hybrid_auto_hybrid_speedup=%sx\n' "$HYBRID_SPEED"
printf 'b300_closure_hybrid_auto_hybrid_min_speedup=%sx\n' "$HYBRID_MIN_SPEEDUP"
printf 'b300_closure_hybrid_auto_dualmask_base_wall_s=%s\n' "$DUAL_BASE_WALL"
printf 'b300_closure_hybrid_auto_dualmask_wall_s=%s\n' "$DUAL_WALL"
printf 'b300_closure_hybrid_auto_dualmask_speedup=%sx\n' "$DUAL_SPEED"
printf 'b300_closure_hybrid_auto_dualmask_min_speedup=%sx\n' "$DUALMASK_MIN_SPEEDUP"
printf 'b300_closure_hybrid_auto_baseline_drift=%s\n' "$BASELINE_DRIFT"
printf 'b300_closure_hybrid_auto_promote_hybrid=%s\n' "$PROMOTE_HYBRID"
printf 'b300_closure_hybrid_auto_adopt_dualmask=%s\n' "$ADOPT_DUAL"
printf 'b300_closure_hybrid_batch_port_candidate=%s\n' "$PORT"
printf 'b300_closure_hybrid_batch_port_dualmask=%s\n' "$ADOPT_DUAL"
printf 'b300_closure_hybrid_auto_profile_log=%s\n' "$PROFILE_LOG"
printf 'b300_closure_hybrid_auto_hybrid_log=%s\n' "$HYBRID_LOG"
printf 'b300_closure_hybrid_auto_dual_log=%s\n' "$DUAL_LOG"
printf 'b300_closure_hybrid_auto_note=profile selects sweep seeds only; production promotion still requires exact residues, >= configured global wall win, and <=5%% cross-run baseline drift\n'
