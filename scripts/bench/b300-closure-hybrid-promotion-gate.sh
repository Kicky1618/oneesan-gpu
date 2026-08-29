#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
MOD="${MOD:-4294967291}"
ROWS="${ROWS:-1}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
THRESHOLDS="${THRESHOLDS:-4 8 12}"
REPEATS="${REPEATS:-1}"
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
HYBRID_MIN_SPEEDUP="${HYBRID_MIN_SPEEDUP:-1.01}"
DUALMASK_MIN_SPEEDUP="${DUALMASK_MIN_SPEEDUP:-1.01}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_closure_hybrid_promotion_row${ROWS}}"
HYBRID_PREFIX="${HYBRID_PREFIX:-${PREFIX}.hybrid}"
DUAL_PREFIX="${DUAL_PREFIX:-${PREFIX}.hybrid_dual}"
HYBRID_LOG="${HYBRID_LOG:-${PREFIX}.hybrid.log}"
DUAL_LOG="${DUAL_LOG:-${PREFIX}.hybrid_dual.log}"
RUN_BENCH="${RUN_BENCH:-1}"
[[ "$RUN_BENCH" == 0 || "$RUN_BENCH" == 1 ]] || { echo 'RUN_BENCH must be 0 or 1' >&2; exit 2; }
mkdir -p "$(dirname "$PREFIX")"

if [[ "$RUN_BENCH" == 1 ]]; then
  echo '=== hybrid promotion gate: all-warp vs scalar-small/warp-large ===' >&2
  MOD="$MOD" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
    ROWS="$ROWS" THREADS_LIST="$THREADS_LIST" THRESHOLDS="$THRESHOLDS" REPEATS="$REPEATS" PREFIX="$HYBRID_PREFIX" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-rankstate-closure-warp-hybrid-ab.sh" | tee "$HYBRID_LOG"

  echo '=== hybrid promotion gate: selected hybrid family vs warp-dualmask composition ===' >&2
  MOD="$MOD" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
    ROWS="$ROWS" THREADS_LIST="$THREADS_LIST" THRESHOLDS="$THRESHOLDS" REPEATS="$REPEATS" PREFIX="$DUAL_PREFIX" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-rankstate-closure-warp-hybrid-dualmask-ab.sh" | tee "$DUAL_LOG"
fi
[[ -s "$HYBRID_LOG" && -s "$DUAL_LOG" ]] || { echo 'hybrid promotion logs missing' >&2; exit 3; }

gv(){ local k="$1" f="$2"; sed -nE "s/^${k}=([^[:space:]]+).*/\\1/p" "$f" | tail -n1; }
[[ "$(gv b300_closure_warp_hybrid_exact_intermediate_match "$HYBRID_LOG")" == 1 ]] || { echo 'hybrid residue gate failed' >&2; exit 4; }
[[ "$(gv b300_closure_warp_hybrid_dualmask_exact_intermediate_match "$DUAL_LOG")" == 1 ]] || { echo 'hybrid-dualmask residue gate failed' >&2; exit 4; }

HYBRID_BEST_TH="$(gv b300_closure_warp_hybrid_best_threshold "$HYBRID_LOG")"
HYBRID_BEST_THREADS="$(gv b300_closure_warp_hybrid_best_threads "$HYBRID_LOG")"
HYBRID_BEST_WALL="$(gv b300_closure_warp_hybrid_best_wall_s "$HYBRID_LOG")"
HYBRID_GLOBAL_SPEED_RAW="$(gv b300_closure_warp_hybrid_speedup_vs_global_base_best "$HYBRID_LOG")"
HYBRID_GLOBAL_SPEED="${HYBRID_GLOBAL_SPEED_RAW%x}"
[[ "$HYBRID_BEST_TH" =~ ^[0-9]+$ && "$HYBRID_BEST_THREADS" =~ ^[0-9]+$ ]] || { echo 'invalid hybrid winner metadata' >&2; exit 4; }

# Do not accept a dualmask win from another threshold/thread. Promotion must be
# measured at the exact hybrid winner selected against the global all-warp base.
DUAL_BASE_KEY="b300_closure_warp_hybrid_dualmask_threshold_${HYBRID_BEST_TH}_threads_${HYBRID_BEST_THREADS}_base_wall_s"
DUAL_WALL_KEY="b300_closure_warp_hybrid_dualmask_threshold_${HYBRID_BEST_TH}_threads_${HYBRID_BEST_THREADS}_dual_wall_s"
DUAL_SPEED_KEY="b300_closure_warp_hybrid_dualmask_threshold_${HYBRID_BEST_TH}_threads_${HYBRID_BEST_THREADS}_speedup"
DUAL_BASE_WALL="$(gv "$DUAL_BASE_KEY" "$DUAL_LOG")"
DUAL_WALL="$(gv "$DUAL_WALL_KEY" "$DUAL_LOG")"
DUAL_SPEED_RAW="$(gv "$DUAL_SPEED_KEY" "$DUAL_LOG")"
DUAL_SPEED="${DUAL_SPEED_RAW%x}"
[[ -n "$DUAL_BASE_WALL" && -n "$DUAL_WALL" && -n "$DUAL_SPEED" ]] || { echo "missing dualmask result at hybrid winner threshold=$HYBRID_BEST_TH threads=$HYBRID_BEST_THREADS" >&2; exit 4; }

# Cross-run consistency guard. The hybrid-only and hybrid+dualmask scripts each
# rebuild independently; reject promotion if their measured hybrid baselines
# differ materially (>5%), which usually signals noise/thermal/run-order issues.
CONSISTENCY="$(python3 - "$HYBRID_BEST_WALL" "$DUAL_BASE_WALL" <<'PY'
import sys
x,y=map(float,sys.argv[1:]); d=abs(x-y)/min(x,y) if min(x,y)>0 else 999.; print(f'{d:.9f}')
PY
)"
PROMOTE="$(python3 - "$HYBRID_GLOBAL_SPEED" "$HYBRID_MIN_SPEEDUP" "$DUAL_SPEED" "$DUALMASK_MIN_SPEEDUP" "$CONSISTENCY" <<'PY'
import sys
h,hm,d,dm,c=map(float,sys.argv[1:]);print(1 if h>=hm and d>=dm and c<=0.05 else 0)
PY
)"

printf 'b300_closure_hybrid_promotion_exact_gates=1\n'
printf 'b300_closure_hybrid_promotion_threshold=%s\n' "$HYBRID_BEST_TH"
printf 'b300_closure_hybrid_promotion_threads=%s\n' "$HYBRID_BEST_THREADS"
printf 'b300_closure_hybrid_promotion_hybrid_wall_s=%s\n' "$HYBRID_BEST_WALL"
printf 'b300_closure_hybrid_promotion_hybrid_global_speedup=%sx\n' "$HYBRID_GLOBAL_SPEED"
printf 'b300_closure_hybrid_promotion_hybrid_min_speedup=%sx\n' "$HYBRID_MIN_SPEEDUP"
printf 'b300_closure_hybrid_promotion_dualmask_base_wall_s=%s\n' "$DUAL_BASE_WALL"
printf 'b300_closure_hybrid_promotion_dualmask_wall_s=%s\n' "$DUAL_WALL"
printf 'b300_closure_hybrid_promotion_dualmask_speedup=%sx\n' "$DUAL_SPEED"
printf 'b300_closure_hybrid_promotion_dualmask_min_speedup=%sx\n' "$DUALMASK_MIN_SPEEDUP"
printf 'b300_closure_hybrid_promotion_cross_run_baseline_delta=%s\n' "$CONSISTENCY"
printf 'b300_closure_hybrid_batch_port_candidate=%s\n' "$PROMOTE"
printf 'b300_closure_hybrid_promotion_note=requires exact residues, hybrid global wall win, incremental dualmask win at same threshold/thread, and <=5%% independent-baseline drift\n'
