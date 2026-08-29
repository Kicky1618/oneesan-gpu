#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"; (($#==0)) || shift
[[ "$N" == 27 ]] || { echo 'nextgen selector targets n=27' >&2; exit 2; }
ARCH="${ARCH:-native}"; PRIME="${SMOKE_PRIME:-4294967291}"; TARGET_MIB="${FORCED_TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"; [[ -f "$PROFILE_FILE" ]] || { echo "missing profile $PROFILE_FILE" >&2; exit 2; }
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_select_n27}"; CAL_LOG="${CAL_LOG:-${PREFIX}.calibration.log}"; CAL_PREFIX="${CAL_PREFIX:-${PREFIX}.calibration}"
RECALIBRATE="${RECALIBRATE:-1}"; SELECT_ONLY="${SELECT_ONLY:-1}"; REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
N27_PRODUCER_WEIGHT_RACE="${N27_PRODUCER_WEIGHT_RACE:-0}"; N27_PRODUCER_ADAPTIVE_RACE="${N27_PRODUCER_ADAPTIVE_RACE:-0}"
for x in RECALIBRATE SELECT_ONLY REBUILD_BUCKETS N27_PRODUCER_WEIGHT_RACE N27_PRODUCER_ADAPTIVE_RACE; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || exit 2; done
mkdir -p "$(dirname "$PREFIX")"
getv(){ local k="$1" f="$2";sed -nE "s/^${k}=([^[:space:]]+).*/\\1/p" "$f"|tail -n1; }

if [[ "$RECALIBRATE" == 1 || ! -s "$CAL_LOG" ]]; then
  echo '=== nextgen selector: partial-row forced calibration A/B/C ===' >&2
  ARCH="$ARCH" MOD="$PRIME" ROWS="${CAL_ROWS:-1}" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" THREADS_LIST="${THREADS_LIST:-128 256 512}" HIGHDROP_LIST="${HIGHDROP_LIST:-0 1}" REPEATS="${REPEATS:-1}" REGCAP_LIST="${REGCAP_LIST:-0 96 128 160 192 224}" MAIN_MIN_SPEEDUP="${MAIN_MIN_SPEEDUP:-1.01}" BLOCK_MIN_SPEEDUP="${BLOCK_MIN_SPEEDUP:-1.01}" LATENCY_MIN_SPEEDUP="${LATENCY_MIN_SPEEDUP:-1.01}" PREFIX="$CAL_PREFIX" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-calibrate-latency.sh" | tee "$CAL_LOG"
fi
[[ "$(getv b300_nextgen_latency_calibrate_exact_gates "$CAL_LOG")" == 1 ]] || { echo 'nextgen latency calibration exact gate missing' >&2; exit 3; }
RES="$(getv b300_nextgen_latency_calibrate_residue "$CAL_LOG")"
FINAL_HIGH="$(getv b300_nextgen_latency_calibrate_final_high_drop "$CAL_LOG")"; FINAL_ILP="$(getv b300_nextgen_latency_calibrate_final_ilp "$CAL_LOG")"; FINAL_CG="$(getv b300_nextgen_latency_calibrate_final_random_cg "$CAL_LOG")"; FINAL_PRE="$(getv b300_nextgen_latency_calibrate_final_prefetch_l2 "$CAL_LOG")"; FINAL_DUAL="$(getv b300_nextgen_latency_calibrate_final_dualmask "$CAL_LOG")"; FINAL_BATCH="$(getv b300_nextgen_latency_calibrate_final_closure_batch "$CAL_LOG")"; FINAL_CAP="$(getv b300_nextgen_latency_calibrate_final_maxrregcount "$CAL_LOG")"; FINAL_THREADS="$(getv b300_nextgen_latency_calibrate_final_threads "$CAL_LOG")"
UNCAP_HIGH="$(getv b300_nextgen_latency_calibrate_uncapped_high_drop "$CAL_LOG")"; UNCAP_ILP="$(getv b300_nextgen_latency_calibrate_uncapped_ilp "$CAL_LOG")"; UNCAP_CG="$(getv b300_nextgen_latency_calibrate_uncapped_random_cg "$CAL_LOG")"; UNCAP_DUAL="$(getv b300_nextgen_latency_calibrate_uncapped_dualmask "$CAL_LOG")"; UNCAP_BATCH="$(getv b300_nextgen_latency_calibrate_uncapped_closure_batch "$CAL_LOG")"; UNCAP_THREADS="$(getv b300_nextgen_latency_calibrate_uncapped_threads "$CAL_LOG")"
MAIN_HIGH="$(getv b300_nextgen_latency_calibrate_main_high_drop "$CAL_LOG")"; MAIN_ILP="$(getv b300_nextgen_latency_calibrate_main_ilp "$CAL_LOG")"; MAIN_CG="$(getv b300_nextgen_latency_calibrate_main_random_cg "$CAL_LOG")"; MAIN_THREADS="$(getv b300_nextgen_latency_calibrate_main_threads "$CAL_LOG")"
BASE_HIGH="$(getv b300_nextgen_latency_calibrate_global_base_high_drop "$CAL_LOG")"; BASE_THREADS="$(getv b300_nextgen_latency_calibrate_global_base_threads "$CAL_LOG")"

# Optional expensive n=27 bucket refinements. Disabled by default; if enabled,
# they run before the single-pass final arbitration.
BUCKET_PROFILE="$PROFILE_FILE"
if [[ "$N27_PRODUCER_WEIGHT_RACE" == 1 ]]; then
  OUT="${PREFIX}.producer-weight.env"
  PROFILE_FILE="$BUCKET_PROFILE" PROFILE_OUT="$OUT" PREFIX="${PREFIX}.producer-weight" WEIGHT_RACE_ONLY=1 WEIGHT_REBUILD="$REBUILD_BUCKETS" ARCH="$ARCH" SMOKE_PRIME="$PRIME" MAX_WINDOW="$MAX_WINDOW" \
    bash "$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled-producer-weight-race.sh" 27
  [[ -s "$OUT" ]] || exit 4; BUCKET_PROFILE="$OUT"
fi
if [[ "$N27_PRODUCER_ADAPTIVE_RACE" == 1 ]]; then
  OUT="${PREFIX}.producer-adaptive.env"
  PROFILE_FILE="$BUCKET_PROFILE" PROFILE_OUT="$OUT" PREFIX="${PREFIX}.producer-adaptive" ADAPTIVE_RACE_ONLY=1 ADAPTIVE_REBUILD="$REBUILD_BUCKETS" ARCH="$ARCH" SMOKE_PRIME="$PRIME" MAX_WINDOW="$MAX_WINDOW" \
    bash "$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled-producer-adaptive-race.sh" 27
  [[ -s "$OUT" ]] || exit 4; BUCKET_PROFILE="$OUT"
fi

NORMAL_PROFILE="${PREFIX}.profile.normalized.env"
python3 - "$BUCKET_PROFILE" "$NORMAL_PROFILE" <<'PY'
import sys
src,out=sys.argv[1:];lines=open(src).read().splitlines();vals={}
for line in lines:
 if '=' in line:k,v=line.split('=',1);vals[k]=v
pac=vals.get('ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_ADAPTIVE_COLS',vals.get('ORBIT_N27_PRODUCER_ADAPTIVE_COLS','0'))
try:
 if int(pac)<0:raise ValueError
except:raise SystemExit('invalid producer adaptive cols '+pac)
key='ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_ADAPTIVE_COLS';dst=[];done=False
for line in lines:
 if line.startswith(key+'='):
  if not done:dst.append(key+'='+pac);done=True
 else:dst.append(line)
if not done:dst.append(key+'='+pac)
open(out,'w').write('\n'.join(dst)+'\n')
PY
BUCKET_PROFILE="$NORMAL_PROFILE"

FORCED_SET="${PREFIX}.forced.tsv"; printf 'label\tbinary\tthreads\n' >"$FORCED_SET"
declare -A SEEN_CFG=()
build_candidate(){
  local role="$1" h="$2" ilp="$3" cg="$4" pre="$5" dual="$6" batch="$7" cap="$8" t="$9"
  local cfg="h${h}_i${ilp}_c${cg}_p${pre}_d${dual}_b${batch}_r${cap}"; [[ -z "${SEEN_CFG[$cfg]+x}" ]] || { echo "skip duplicate forced config role=$role cfg=$cfg" >&2; return 0; }; SEEN_CFG[$cfg]=1
  local label="${role}_${cfg}_t${t}" bin="$ONEESAN_BUILD_DIR/b300_nextgen_${cfg}_n27" bout="${PREFIX}.${role}.build.out" berr="${PREFIX}.${role}.build.err"
  echo "=== nextgen selector: build $label ===" >&2
  N=27 ARCH="$ARCH" OUT="$bin" HIGH_DROP_CHUNK="$h" RECURRENCE_ILP="$ilp" RANDOM_CG="$cg" PREFETCH_L2="$pre" DUALMASK="$dual" CLOSURE_BATCH="$batch" MAXRREGCOUNT="$cap" BUILD_ERR="$berr" PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen.sh" >"$bout" 2>"${PREFIX}.${role}.build.driver.err"
  [[ -x "$bin" ]] || { echo "missing built candidate $bin" >&2; exit 5; }
  grep -Fq "nextgen_forced=1 high_drop_chunk=$h recurrence_ilp=$ilp random_cg=$cg prefetch_l2=$pre dualmask=$dual closure_batch=$batch maxrregcount=$cap" "$bout"
  printf '%s\t%s\t%s\n' "$label" "$bin" "$t" >>"$FORCED_SET"
}
# Priority order provides automatic dedupe while retaining progressively more
# conservative fallbacks for the complete-prime race.
build_candidate final "$FINAL_HIGH" "$FINAL_ILP" "$FINAL_CG" "$FINAL_PRE" "$FINAL_DUAL" "$FINAL_BATCH" "$FINAL_CAP" "$FINAL_THREADS"
build_candidate uncapped "$UNCAP_HIGH" "$UNCAP_ILP" "$UNCAP_CG" 0 "$UNCAP_DUAL" "$UNCAP_BATCH" 0 "$UNCAP_THREADS"
build_candidate main "$MAIN_HIGH" "$MAIN_ILP" "$MAIN_CG" 0 0 0 0 "$MAIN_THREADS"
build_candidate base "$BASE_HIGH" 2 0 0 0 0 0 "$BASE_THREADS"
FORCED_COUNT=$(( $(wc -l <"$FORCED_SET") - 1 )); ((FORCED_COUNT>=1)) || exit 5

echo "NEXTGEN FORCED SET candidates=$FORCED_COUNT calibration_residue=$RES" >&2; cat "$FORCED_SET" >&2
echo "NEXTGEN BUCKET PROFILE profile=$BUCKET_PROFILE n27_weight_race=$N27_PRODUCER_WEIGHT_RACE n27_adaptive_race=$N27_PRODUCER_ADAPTIVE_RACE" >&2
export FORCED_SET_FILE="$FORCED_SET" PROFILE_FILE="$BUCKET_PROFILE" ARCH SMOKE_PRIME="$PRIME" FORCED_TARGET_MIB="$TARGET_MIB" MAX_WINDOW REBUILD_BUCKETS SELECT_ONLY
export PREFIX="${RACE_PREFIX:-${PREFIX}.race}"
exec "$ONEESAN_ROOT/scripts/run/b300x8-race-forced-set-profiled-once.sh" 27 "$@"
