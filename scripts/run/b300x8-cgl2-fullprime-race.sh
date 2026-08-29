#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo 'CG-L2 full-prime race targets n=27' >&2; exit 2; }
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"; [[ -f "$PROFILE_FILE" ]] || { echo "missing PROFILE_FILE=$PROFILE_FILE" >&2; exit 2; }
ROWS="${ROWS:-1}"; HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"; RECURRENCE_ILP="${RECURRENCE_ILP:-4}"; PREFETCH_L2="${PREFETCH_L2:-0}"; DUALMASK="${DUALMASK:-0}"; CLOSURE_BATCH="${CLOSURE_BATCH:-0}"; MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"; L2_SIZES="${L2_SIZES:-0 64 128 256}"; REPEATS="${REPEATS:-1}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; ARCH="${ARCH:-native}"; RUN_SWEEP="${RUN_SWEEP:-1}"; SELECT_ONLY="${SELECT_ONLY:-1}"; REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
SWEEP_PREFIX="${SWEEP_PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_cgl2_fullprime_cal}"
SWEEP_LOGDIR="${SWEEP_LOGDIR:-${SWEEP_PREFIX}_logs}"; SWEEP_RESULT="${SWEEP_RESULT:-${SWEEP_PREFIX}.tsv}"; SWEEP_RESOURCE="${SWEEP_RESOURCE:-${SWEEP_PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${SWEEP_PREFIX}_winner.env}"
RACE_PREFIX="${RACE_PREFIX:-$ONEESAN_ROOT/work/b300_cgl2_fullprime_race_n27}"
for x in PREFETCH_L2 DUALMASK RUN_SWEEP SELECT_ONLY REBUILD_BUCKETS; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done

if [[ "$RUN_SWEEP" == 1 ]]; then
  echo '=== partial-row CG load-level L2 fetch-size calibration ===' >&2
  ARCH="$ARCH" ROWS="$ROWS" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" RECURRENCE_ILP="$RECURRENCE_ILP" PREFETCH_L2="$PREFETCH_L2" DUALMASK="$DUALMASK" CLOSURE_BATCH="$CLOSURE_BATCH" MAXRREGCOUNT="$MAXRREGCOUNT" THREADS_LIST="$THREADS_LIST" L2_SIZES="$L2_SIZES" INCLUDE_NOCG=1 REPEATS="$REPEATS" SAMPLE_INTERVAL="$SAMPLE_INTERVAL" PREFIX="$SWEEP_PREFIX" LOGDIR="$SWEEP_LOGDIR" RESULT="$SWEEP_RESULT" RESOURCE="$SWEEP_RESOURCE" WINNER_ENV="$WINNER_ENV" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-cg-l2size-sweep.sh"
fi
for f in "$WINNER_ENV" "$SWEEP_RESULT" "$SWEEP_RESOURCE" "$SWEEP_LOGDIR/binaries.tsv"; do [[ -f "$f" ]] || { echo "missing calibration artifact=$f" >&2; exit 3; }; done
# shellcheck disable=SC1090
source "$WINNER_ENV"
for n in B300_CGL2_WINNER_PROFILE B300_CGL2_WINNER_BIN B300_CGL2_WINNER_THREADS B300_CGL2_WINNER_RANDOM_CG B300_CGL2_WINNER_L2_FETCH_BYTES B300_CGL2_WINNER_SPILL_STORE_BYTES B300_CGL2_WINNER_SPILL_LOAD_BYTES; do [[ -n "${!n+x}" ]] || { echo "winner env missing $n" >&2; exit 3; }; done
[[ -x "$B300_CGL2_WINNER_BIN" ]] || { echo 'CG-L2 winner binary missing' >&2; exit 3; }
[[ "$B300_CGL2_WINNER_SPILL_STORE_BYTES" == 0 && "$B300_CGL2_WINNER_SPILL_LOAD_BYTES" == 0 ]] || { echo 'refusing spilling CG-L2 winner' >&2; exit 4; }

CONTROL="$(python3 - "$SWEEP_RESULT" "$SWEEP_RESOURCE" "$SWEEP_LOGDIR/binaries.tsv" "$B300_CGL2_WINNER_PROFILE" "$B300_CGL2_WINNER_RANDOM_CG" "$B300_CGL2_WINNER_L2_FETCH_BYTES" <<'PY'
import csv,statistics,sys,math
result,resource,bins_path,wp,wcg,wl2=sys.argv[1:]; wcg=int(wcg); wl2=int(wl2)
rows=list(csv.DictReader(open(result),delimiter='\t')); rr=list(csv.DictReader(open(resource),delimiter='\t')); bins={r['profile']:r for r in csv.DictReader(open(bins_path),delimiter='\t')}
res={}
for r in rr:
 try:z=(int(r['registers']),int(r['spill_store_bytes']),int(r['spill_load_bytes']))
 except:continue
 old=res.get(r['profile'],(-1,-1,-1));res[r['profile']]=(max(old[0],z[0]),max(old[1],z[1]),max(old[2],z[2]))
by={}
for r in rows:by.setdefault((r['profile'],int(r['random_cg']),int(r['l2_fetch_bytes']),int(r['threads'])),[]).append(r)
c=[]
for (p,cg,l2,t),g in by.items():
 regs,ss,sl=res.get(p,(-1,-1,-1))
 if regs<0 or ss or sl or l2!=0:continue
 w=statistics.median(float(x['wall_s']) for x in g);mv=[float(x['mc_avg_pct']) for x in g if x['mc_avg_pct']!='nan'];mc=statistics.median(mv) if mv else math.nan
 # If winner uses a fetch-size hint, cg0 is the direct control. Otherwise use
 # the opposite no-hint CG policy when available to avoid racing the same binary.
 direct = (wl2>0 and cg==1) or (wl2==0 and cg!=wcg)
 if p==wp:continue
 c.append((0 if direct else 1,w,-mc if not math.isnan(mc) else math.inf,p,t,bins[p]['binary']))
if not c:print('NONE')
else:
 x=min(c);print('\t'.join([x[3],str(x[4]),x[5]]))
PY
)"

BASE_ARGS=()
if [[ "$CONTROL" != NONE ]]; then
  IFS=$'\t' read -r CONTROL_PROFILE CONTROL_THREADS CONTROL_BIN <<<"$CONTROL"
  [[ -x "$CONTROL_BIN" ]] || { echo 'CG-L2 control binary missing' >&2; exit 3; }
  BASE_ARGS=(FORCED_BASE_BIN="$CONTROL_BIN" FORCED_BASE_LABEL="cgl2_control_${CONTROL_PROFILE}" FORCED_BASE_THREADS="$CONTROL_THREADS")
  echo "CG-L2 full-prime control=$CONTROL_PROFILE threads=$CONTROL_THREADS" >&2
else
  echo 'CG-L2 full-prime control unavailable; racing winner against profiled families only' >&2
fi

echo "=== full-prime CG-L2 race winner=$B300_CGL2_WINNER_PROFILE threads=$B300_CGL2_WINNER_THREADS l2=${B300_CGL2_WINNER_L2_FETCH_BYTES}B partial_mc=${B300_CGL2_WINNER_MC_AVG_PCT:-NA}% ===" >&2
env PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" MAX_WINDOW="$MAX_WINDOW" FORCED_TARGET_MIB="$TARGET_MIB" \
  FORCED_OVERRIDE_BIN="$B300_CGL2_WINNER_BIN" FORCED_OVERRIDE_LABEL="cgl2_${B300_CGL2_WINNER_PROFILE}" FORCED_OVERRIDE_THREADS="$B300_CGL2_WINNER_THREADS" \
  REBUILD_BUCKETS="$REBUILD_BUCKETS" SELECT_ONLY="$SELECT_ONLY" PREFIX="$RACE_PREFIX" "${BASE_ARGS[@]}" \
  exec "$ONEESAN_ROOT/scripts/run/b300x8-race-external-forced-profiled-once.sh" 27 "$@"
