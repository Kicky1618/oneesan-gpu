#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
MOD="${MOD:-4294967291}"
ROWS="${ROWS:-1}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
REPEATS="${REPEATS:-1}"
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
WARP_MIN_SPEEDUP="${WARP_MIN_SPEEDUP:-1.01}"
DUALMASK_MIN_SPEEDUP="${DUALMASK_MIN_SPEEDUP:-1.01}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_closure_warp_promotion_row${ROWS}}"
RANK_PREFIX="${RANK_PREFIX:-${PREFIX}.rankstate}"
DUAL_PREFIX="${DUAL_PREFIX:-${PREFIX}.dualmask}"
RANK_LOG="${RANK_LOG:-${PREFIX}.rankstate.log}"
DUAL_LOG="${DUAL_LOG:-${PREFIX}.dualmask.log}"
RUN_BENCH="${RUN_BENCH:-1}"
[[ "$RUN_BENCH" == 0 || "$RUN_BENCH" == 1 ]] || { echo 'RUN_BENCH must be 0 or 1' >&2; exit 2; }
mkdir -p "$(dirname "$PREFIX")"

if [[ "$RUN_BENCH" == 1 ]]; then
  echo '=== promotion gate: rank-state ILP4 vs closure-warp ===' >&2
  N=27 MOD="$MOD" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
    ROWS="$ROWS" THREADS_LIST="$THREADS_LIST" REPEATS="$REPEATS" MODES='ilp4 ilp4warp' PREFIX="$RANK_PREFIX" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-rankstate-ilp24-ab.sh" | tee "$RANK_LOG"

  echo '=== promotion gate: closure-warp baseline vs closure-warp dualmask ===' >&2
  MOD="$MOD" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
    ROWS="$ROWS" THREADS_LIST="$THREADS_LIST" REPEATS="$REPEATS" PREFIX="$DUAL_PREFIX" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-rankstate-closure-warp-dualmask-ab.sh" | tee "$DUAL_LOG"
fi
[[ -s "$RANK_LOG" && -s "$DUAL_LOG" ]] || { echo 'promotion gate logs missing' >&2; exit 3; }

gv(){ local k="$1" f="$2"; sed -nE "s/^${k}=([^[:space:]]+).*/\\1/p" "$f" | tail -n1; }
[[ "$(gv b300_rankstate_mlp_residue_match "$RANK_LOG")" == 1 ]] || { echo 'rank-state residue gate failed' >&2; exit 4; }
[[ "$(gv b300_closure_warp_dualmask_exact_intermediate_match "$DUAL_LOG")" == 1 ]] || { echo 'dualmask residue gate failed' >&2; exit 4; }

RANK_TSV="${RANK_PREFIX}.tsv"
[[ -s "$RANK_TSV" ]] || { echo "rank-state result missing: $RANK_TSV" >&2; exit 4; }
RANK_COMPARE="$(python3 - "$RANK_TSV" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
by={}
for r in rows:
    if r['mode'] not in ('ilp4','ilp4warp'):continue
    by.setdefault((r['mode'],int(r['threads'])),[]).append(float(r['wall_s']))
med={k:statistics.median(v) for k,v in by.items()}
best=None
for t in sorted({t for _,t in med}):
    a=med.get(('ilp4',t));w=med.get(('ilp4warp',t))
    if a is None or w is None:continue
    sp=a/w
    if best is None or w<best[0]:best=(w,t,sp,a)
if best is None:raise SystemExit('no paired ilp4/ilp4warp rows')
print('\t'.join(map(str,best)))
PY
)"
IFS=$'\t' read -r WARP_WALL WARP_THREADS WARP_SPEEDUP ILP4_WALL <<<"$RANK_COMPARE"
DUAL_SPEED_RAW="$(gv b300_closure_warp_dualmask_best_thread_speedup "$DUAL_LOG")"
DUAL_THREADS="$(gv b300_closure_warp_dualmask_best_threads "$DUAL_LOG")"
DUAL_WALL="$(gv b300_closure_warp_dualmask_best_candidate_wall_s "$DUAL_LOG")"
DUAL_SPEED="${DUAL_SPEED_RAW%x}"

PORT="$(python3 - "$WARP_SPEEDUP" "$WARP_MIN_SPEEDUP" "$DUAL_SPEED" "$DUALMASK_MIN_SPEEDUP" <<'PY'
import sys
w,wm,d,dm=map(float,sys.argv[1:]);print(1 if w>=wm and d>=dm else 0)
PY
)"
printf 'b300_closure_warp_promotion_exact_gates=1\n'
printf 'b300_closure_warp_promotion_ilp4_wall_s=%s\n' "$ILP4_WALL"
printf 'b300_closure_warp_promotion_warp_wall_s=%s\n' "$WARP_WALL"
printf 'b300_closure_warp_promotion_warp_threads=%s\n' "$WARP_THREADS"
printf 'b300_closure_warp_promotion_warp_speedup=%sx\n' "$WARP_SPEEDUP"
printf 'b300_closure_warp_promotion_warp_min_speedup=%sx\n' "$WARP_MIN_SPEEDUP"
printf 'b300_closure_warp_promotion_dualmask_wall_s=%s\n' "$DUAL_WALL"
printf 'b300_closure_warp_promotion_dualmask_threads=%s\n' "$DUAL_THREADS"
printf 'b300_closure_warp_promotion_dualmask_speedup=%sx\n' "$DUAL_SPEED"
printf 'b300_closure_warp_promotion_dualmask_min_speedup=%sx\n' "$DUALMASK_MIN_SPEEDUP"
printf 'b300_closure_warp_batch_port_candidate=%s\n' "$PORT"
printf 'b300_closure_warp_promotion_note=port to batch only when both independent wall gates pass; no production transform is enabled by this script\n'
