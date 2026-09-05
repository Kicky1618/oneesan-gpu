#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; ROWS="${ROWS:-1}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"; HIGHDROP_LIST="${HIGHDROP_LIST:-0 1}"; REPEATS="${REPEATS:-1}"; REGCAP_LIST="${REGCAP_LIST:-0 96 128 160 192 224}"
MAIN_MIN_SPEEDUP="${MAIN_MIN_SPEEDUP:-1.01}"; BLOCK_MIN_SPEEDUP="${BLOCK_MIN_SPEEDUP:-1.01}"; LATENCY_MIN_SPEEDUP="${LATENCY_MIN_SPEEDUP:-1.01}"; CGL2_MIN_SPEEDUP="${CGL2_MIN_SPEEDUP:-1.01}"
L2_SIZES="${L2_SIZES:-0 64 128 256}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_cgl2_calibrate_row${ROWS}}"; ABC_LOG="${ABC_LOG:-${PREFIX}.abc.log}"; D_LOG="${D_LOG:-${PREFIX}.cgl2.log}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}.cgl2-winner.env}"
mkdir -p "$(dirname "$PREFIX")"
getv(){ local k="$1" f="$2"; sed -nE "s/^${k}=([^[:space:]]+).*/\\1/p" "$f" | tail -n1; }

# A/B/C: highdrop, recurrent main MLP, block transforms, prefetch/register cap.
echo '=== nextgen calibration stages A/B/C ===' >&2
ARCH="$ARCH" MOD="$MOD" ROWS="$ROWS" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" THREADS_LIST="$THREADS_LIST" HIGHDROP_LIST="$HIGHDROP_LIST" REPEATS="$REPEATS" REGCAP_LIST="$REGCAP_LIST" MAIN_MIN_SPEEDUP="$MAIN_MIN_SPEEDUP" BLOCK_MIN_SPEEDUP="$BLOCK_MIN_SPEEDUP" LATENCY_MIN_SPEEDUP="$LATENCY_MIN_SPEEDUP" PREFIX="${PREFIX}.abc" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-calibrate-latency.sh" | tee "$ABC_LOG"
[[ "$(getv b300_nextgen_latency_calibrate_exact_gates "$ABC_LOG")" == 1 ]] || { echo 'ABC exact gate missing' >&2; exit 3; }
RES_ABC="$(getv b300_nextgen_latency_calibrate_residue "$ABC_LOG")"
H="$(getv b300_nextgen_latency_calibrate_final_high_drop "$ABC_LOG")"; ILP="$(getv b300_nextgen_latency_calibrate_final_ilp "$ABC_LOG")"; CG_ABC="$(getv b300_nextgen_latency_calibrate_final_random_cg "$ABC_LOG")"; PRE="$(getv b300_nextgen_latency_calibrate_final_prefetch_l2 "$ABC_LOG")"; DUAL="$(getv b300_nextgen_latency_calibrate_final_dualmask "$ABC_LOG")"; BATCH="$(getv b300_nextgen_latency_calibrate_final_closure_batch "$ABC_LOG")"; CAP="$(getv b300_nextgen_latency_calibrate_final_maxrregcount "$ABC_LOG")"; THREADS_ABC="$(getv b300_nextgen_latency_calibrate_final_threads "$ABC_LOG")"; WALL_ABC="$(getv b300_nextgen_latency_calibrate_final_wall_s "$ABC_LOG")"

# D: vary only random-load cache policy. INCLUDE_NOCG=1 means the exact Stage-C
# no-CG policy remains in the pool; L2_SIZES includes CG baseline 0 by default.
echo "=== nextgen calibration stage D: CG L2 fetch size h=$H ilp=$ILP pre=$PRE dual=$DUAL batch=$BATCH cap=$CAP ===" >&2
ARCH="$ARCH" MOD="$MOD" ROWS="$ROWS" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" HIGH_DROP_CHUNK="$H" RECURRENCE_ILP="$ILP" PREFETCH_L2="$PRE" DUALMASK="$DUAL" CLOSURE_BATCH="$BATCH" MAXRREGCOUNT="$CAP" L2_SIZES="$L2_SIZES" INCLUDE_NOCG=1 THREADS_LIST="$THREADS_LIST" REPEATS="$REPEATS" PREFIX="${PREFIX}.d" WINNER_ENV="$WINNER_ENV" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-cg-l2size-sweep.sh" | tee "$D_LOG"
[[ "$(getv b300_nextgen_cgl2_exact_intermediate_match "$D_LOG")" == 1 ]] || { echo 'CG-L2 exact gate missing' >&2; exit 3; }
RES_D="$(getv b300_nextgen_cgl2_residue "$D_LOG")"; [[ "$RES_D" == "$RES_ABC" ]] || { echo "FATAL ABC/D residue mismatch abc=$RES_ABC d=$RES_D" >&2; exit 4; }
[[ -s "$WINNER_ENV" ]] || { echo 'CG-L2 winner env missing' >&2; exit 4; }
# shellcheck disable=SC1090
source "$WINNER_ENV"
CG_D="${B300_CGL2_WINNER_RANDOM_CG:?}"; L2_D="${B300_CGL2_WINNER_L2_FETCH_BYTES:?}"; THREADS_D="${B300_CGL2_WINNER_THREADS:?}"; WALL_D="${B300_CGL2_WINNER_WALL_S:?}"
[[ "$CG_D" == 0 || "$CG_D" == 1 ]] || exit 4; case "$L2_D" in 0|64|128|256);;*)exit 4;;esac
[[ "$THREADS_D" =~ ^[0-9]+$ ]] || exit 4

read -r FINAL_CG FINAL_L2 FINAL_THREADS FINAL_WALL ADOPT_D POLICY_CHANGED SPEED <<<"$(python3 - "$CG_ABC" "$THREADS_ABC" "$WALL_ABC" "$CG_D" "$L2_D" "$THREADS_D" "$WALL_D" "$CGL2_MIN_SPEEDUP" <<'PY'
import sys
cg0,t0,w0,cg1,l2,t1,w1=sys.argv[1:8];cut=float(sys.argv[8]);speed=float(w0)/float(w1)
changed=(cg1!=cg0) or (l2!='0')
# If the cache policy itself changes, demand margin. If policy is identical,
# keep Stage-D's remeasured thread choice when it is no slower.
adopt=(changed and speed>=cut) or ((not changed) and float(w1)<=float(w0))
if adopt: print(cg1,l2,t1,w1,1,int(changed),f'{speed:.9f}')
else: print(cg0,0,t0,w0,0,int(changed),f'{speed:.9f}')
PY
)"
GLOBAL_BASE_WALL="$(getv b300_nextgen_latency_calibrate_global_base_wall_s "$ABC_LOG")"
TOTAL_SPEED="$(python3 - "$GLOBAL_BASE_WALL" "$FINAL_WALL" <<'PY'
import sys
print(f'{float(sys.argv[1])/float(sys.argv[2]):.9f}')
PY
)"

# Re-export all fallback configurations needed by complete-prime arbitration.
for key in \
  global_base_high_drop global_base_threads \
  main_high_drop main_ilp main_random_cg main_threads \
  uncapped_high_drop uncapped_ilp uncapped_random_cg uncapped_dualmask uncapped_closure_batch uncapped_threads \
  final_high_drop final_ilp final_prefetch_l2 final_dualmask final_closure_batch final_maxrregcount; do
  v="$(getv b300_nextgen_latency_calibrate_${key} "$ABC_LOG")"
  printf 'b300_nextgen_cgl2_calibrate_%s=%s\n' "$key" "$v"
done
printf 'b300_nextgen_cgl2_calibrate_exact_gates=1\n'
printf 'b300_nextgen_cgl2_calibrate_residue=%s\n' "$RES_ABC"
printf 'b300_nextgen_cgl2_calibrate_stage_c_random_cg=%s\n' "$CG_ABC"
printf 'b300_nextgen_cgl2_calibrate_stage_c_threads=%s\n' "$THREADS_ABC"
printf 'b300_nextgen_cgl2_calibrate_stage_c_wall_s=%s\n' "$WALL_ABC"
printf 'b300_nextgen_cgl2_calibrate_cgl2_min_speedup=%sx\n' "$CGL2_MIN_SPEEDUP"
printf 'b300_nextgen_cgl2_calibrate_cgl2_measured_speedup=%sx\n' "$SPEED"
printf 'b300_nextgen_cgl2_calibrate_cgl2_policy_changed=%s\n' "$POLICY_CHANGED"
printf 'b300_nextgen_cgl2_calibrate_adopt_cgl2=%s\n' "$ADOPT_D"
printf 'b300_nextgen_cgl2_calibrate_final_random_cg=%s\n' "$FINAL_CG"
printf 'b300_nextgen_cgl2_calibrate_final_random_cg_l2_fetch_bytes=%s\n' "$FINAL_L2"
printf 'b300_nextgen_cgl2_calibrate_final_threads=%s\n' "$FINAL_THREADS"
printf 'b300_nextgen_cgl2_calibrate_final_wall_s=%s\n' "$FINAL_WALL"
printf 'b300_nextgen_cgl2_calibrate_final_speedup_vs_global_base=%sx\n' "$TOTAL_SPEED"
printf 'b300_nextgen_cgl2_calibrate_abc_log=%s\n' "$ABC_LOG"
printf 'b300_nextgen_cgl2_calibrate_d_log=%s\n' "$D_LOG"
printf 'b300_nextgen_cgl2_calibrate_winner_env=%s\n' "$WINNER_ENV"
printf 'b300_nextgen_cgl2_calibrate_note=Stage D varies random cache policy only; policy changes require margin; complete-prime arbitration retains Stage-C candidate and conservative fallbacks\n'
