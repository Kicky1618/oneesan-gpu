#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}";MOD="${MOD:-4294967291}";ROWS="${ROWS:-1}";TARGET_MIB="${TARGET_MIB:-65536}";MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS_LIST="${THREADS_LIST:-128 256 512}";HIGHDROP_LIST="${HIGHDROP_LIST:-0 1}";REPEATS="${REPEATS:-1}";REGCAP_LIST="${REGCAP_LIST:-0 96 128 160 192 224}"
MAIN_MIN_SPEEDUP="${MAIN_MIN_SPEEDUP:-1.01}";BLOCK_MIN_SPEEDUP="${BLOCK_MIN_SPEEDUP:-1.01}";LATENCY_MIN_SPEEDUP="${LATENCY_MIN_SPEEDUP:-1.01}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_latency_calibrate_row${ROWS}}";CORE_LOG="${CORE_LOG:-${PREFIX}.core.log}";LAT_LOG="${LAT_LOG:-${PREFIX}.latency.log}"
mkdir -p "$(dirname "$PREFIX")"
getv(){ local k="$1" f="$2";sed -nE "s/^${k}=([^[:space:]]+).*/\\1/p" "$f"|tail -n1; }

echo '=== nextgen latency calibration: stages A+B ===' >&2
ARCH="$ARCH" MOD="$MOD" ROWS="$ROWS" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" THREADS_LIST="$THREADS_LIST" HIGHDROP_LIST="$HIGHDROP_LIST" REPEATS="$REPEATS" MAIN_MIN_SPEEDUP="$MAIN_MIN_SPEEDUP" BLOCK_MIN_SPEEDUP="$BLOCK_MIN_SPEEDUP" PREFIX="${PREFIX}.core" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-calibrate.sh" | tee "$CORE_LOG"
[[ "$(getv b300_nextgen_calibrate_exact_gates "$CORE_LOG")" == 1 ]]||{ echo 'core nextgen exact gate missing' >&2;exit 3; }
RES_CORE="$(getv b300_nextgen_calibrate_residue "$CORE_LOG")"
H="$(getv b300_nextgen_calibrate_final_high_drop "$CORE_LOG")";ILP="$(getv b300_nextgen_calibrate_final_ilp "$CORE_LOG")";CG="$(getv b300_nextgen_calibrate_final_random_cg "$CORE_LOG")";DUAL="$(getv b300_nextgen_calibrate_final_dualmask "$CORE_LOG")";BATCH="$(getv b300_nextgen_calibrate_final_closure_batch "$CORE_LOG")"
GLOBAL_BASE_HIGH="$(getv b300_nextgen_calibrate_global_base_high_drop "$CORE_LOG")";GLOBAL_BASE_THREADS="$(getv b300_nextgen_calibrate_global_base_threads "$CORE_LOG")";GLOBAL_BASE_WALL="$(getv b300_nextgen_calibrate_global_base_wall_s "$CORE_LOG")"
MAIN_HIGH="$(getv b300_nextgen_calibrate_main_high_drop "$CORE_LOG")";MAIN_ILP="$(getv b300_nextgen_calibrate_main_ilp "$CORE_LOG")";MAIN_CG="$(getv b300_nextgen_calibrate_main_random_cg "$CORE_LOG")";MAIN_THREADS="$(getv b300_nextgen_calibrate_main_only_threads "$CORE_LOG")"

echo "=== nextgen latency calibration: stage C h=$H ilp=$ILP cg=$CG dual=$DUAL batch=$BATCH ===" >&2
ARCH="$ARCH" MOD="$MOD" ROWS="$ROWS" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" HIGH_DROP_CHUNK="$H" RECURRENCE_ILP="$ILP" RANDOM_CG="$CG" DUALMASK="$DUAL" CLOSURE_BATCH="$BATCH" REGCAP_LIST="$REGCAP_LIST" THREADS_LIST="$THREADS_LIST" REPEATS="$REPEATS" PREFIX="${PREFIX}.latency" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-latency-regcap-sweep.sh" | tee "$LAT_LOG"
[[ "$(getv b300_nextgen_latency_exact_intermediate_match "$LAT_LOG")" == 1 ]]||{ echo 'latency exact gate missing' >&2;exit 3; }
RES_LAT="$(getv b300_nextgen_latency_residue "$LAT_LOG")";[[ "$RES_CORE" == "$RES_LAT" ]]||{ echo "FATAL core/latency residue mismatch $RES_CORE $RES_LAT" >&2;exit 4; }
UNCAP_THREADS="$(getv b300_nextgen_latency_base_threads "$LAT_LOG")";UNCAP_WALL="$(getv b300_nextgen_latency_base_wall_s "$LAT_LOG")"
BEST_PRE="$(getv b300_nextgen_latency_best_prefetch_l2 "$LAT_LOG")";BEST_CAP="$(getv b300_nextgen_latency_best_cap "$LAT_LOG")";BEST_THREADS="$(getv b300_nextgen_latency_best_threads "$LAT_LOG")";BEST_WALL="$(getv b300_nextgen_latency_best_wall_s "$LAT_LOG")";LAT_SPEED="$(getv b300_nextgen_latency_speedup_vs_uncapped_sync "$LAT_LOG")";LAT_SPEED="${LAT_SPEED%x}"
read -r FINAL_PRE FINAL_CAP FINAL_THREADS FINAL_WALL ADOPT_LAT <<<"$(python3 - "$BEST_PRE" "$BEST_CAP" "$BEST_THREADS" "$BEST_WALL" "$LAT_SPEED" "$LATENCY_MIN_SPEEDUP" "$UNCAP_THREADS" "$UNCAP_WALL" <<'PY'
import sys
pre,cap,t,w=sys.argv[1:5];sp,cut=map(float,sys.argv[5:7]);bt,bw=sys.argv[7:9]
changed=(pre!='0' or cap!='0');adopt=changed and sp>=cut
print(pre,cap,t,w,1) if adopt else print(0,0,bt,bw,0)
PY
)"
TOTAL_SPEED="$(python3 - "$GLOBAL_BASE_WALL" "$FINAL_WALL" <<'PY'
import sys
print(f'{float(sys.argv[1])/float(sys.argv[2]):.9f}')
PY
)"

printf 'b300_nextgen_latency_calibrate_exact_gates=1\n'
printf 'b300_nextgen_latency_calibrate_residue=%s\n' "$RES_CORE"
printf 'b300_nextgen_latency_calibrate_global_base_high_drop=%s\n' "$GLOBAL_BASE_HIGH"
printf 'b300_nextgen_latency_calibrate_global_base_threads=%s\n' "$GLOBAL_BASE_THREADS"
printf 'b300_nextgen_latency_calibrate_global_base_wall_s=%s\n' "$GLOBAL_BASE_WALL"
printf 'b300_nextgen_latency_calibrate_main_high_drop=%s\n' "$MAIN_HIGH"
printf 'b300_nextgen_latency_calibrate_main_ilp=%s\n' "$MAIN_ILP"
printf 'b300_nextgen_latency_calibrate_main_random_cg=%s\n' "$MAIN_CG"
printf 'b300_nextgen_latency_calibrate_main_threads=%s\n' "$MAIN_THREADS"
printf 'b300_nextgen_latency_calibrate_uncapped_high_drop=%s\n' "$H"
printf 'b300_nextgen_latency_calibrate_uncapped_ilp=%s\n' "$ILP"
printf 'b300_nextgen_latency_calibrate_uncapped_random_cg=%s\n' "$CG"
printf 'b300_nextgen_latency_calibrate_uncapped_dualmask=%s\n' "$DUAL"
printf 'b300_nextgen_latency_calibrate_uncapped_closure_batch=%s\n' "$BATCH"
printf 'b300_nextgen_latency_calibrate_uncapped_threads=%s\n' "$UNCAP_THREADS"
printf 'b300_nextgen_latency_calibrate_uncapped_wall_s=%s\n' "$UNCAP_WALL"
printf 'b300_nextgen_latency_calibrate_latency_min_speedup=%sx\n' "$LATENCY_MIN_SPEEDUP"
printf 'b300_nextgen_latency_calibrate_latency_measured_speedup=%sx\n' "$LAT_SPEED"
printf 'b300_nextgen_latency_calibrate_adopt_latency=%s\n' "$ADOPT_LAT"
printf 'b300_nextgen_latency_calibrate_final_high_drop=%s\n' "$H"
printf 'b300_nextgen_latency_calibrate_final_ilp=%s\n' "$ILP"
printf 'b300_nextgen_latency_calibrate_final_random_cg=%s\n' "$CG"
printf 'b300_nextgen_latency_calibrate_final_prefetch_l2=%s\n' "$FINAL_PRE"
printf 'b300_nextgen_latency_calibrate_final_dualmask=%s\n' "$DUAL"
printf 'b300_nextgen_latency_calibrate_final_closure_batch=%s\n' "$BATCH"
printf 'b300_nextgen_latency_calibrate_final_maxrregcount=%s\n' "$FINAL_CAP"
printf 'b300_nextgen_latency_calibrate_final_threads=%s\n' "$FINAL_THREADS"
printf 'b300_nextgen_latency_calibrate_final_wall_s=%s\n' "$FINAL_WALL"
printf 'b300_nextgen_latency_calibrate_final_speedup_vs_global_base=%sx\n' "$TOTAL_SPEED"
printf 'b300_nextgen_latency_calibrate_core_log=%s\n' "$CORE_LOG"
printf 'b300_nextgen_latency_calibrate_latency_log=%s\n' "$LAT_LOG"
printf 'b300_nextgen_latency_calibrate_note=full-prime arbitration should retain final latency candidate, uncapped block candidate, main-only candidate and global ILP2 baseline\n'
