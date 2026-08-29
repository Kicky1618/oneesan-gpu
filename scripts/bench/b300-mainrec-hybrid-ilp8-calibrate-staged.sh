#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
[[ "$N" == 27 ]] || { echo 'mainrec hybrid staged calibration targets n=27' >&2; exit 2; }
ARCH="${ARCH:-native}"
MOD="${MOD:-4294967291}"
SEARCH_ROWS="${SEARCH_ROWS:-1}"
VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"
SEARCH_REPEATS="${SEARCH_REPEATS:-1}"
VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.01}"
ILP8_THRESHOLDS="${ILP8_THRESHOLDS:-0 262144 524288 1048576 2097152 4194304 8388608 16777216}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
RANDOM_CG="${RANDOM_CG:-0}"
RANDOM_CG_L2_FETCH_BYTES="${RANDOM_CG_L2_FETCH_BYTES:-0}"
DUALMASK="${DUALMASK:-0}"
MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_mainrec_hybrid8_staged}"
WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
mkdir -p "$(dirname "$WINNER_ENV")"

for rows in "$SEARCH_ROWS" $VALIDATE_ROWS; do
  [[ "$rows" =~ ^[0-9]+$ ]] && ((rows>=1 && rows<=28)) || { echo "bad stage rows=$rows" >&2; exit 2; }
done
for r in SEARCH_REPEATS VALIDATE_REPEATS; do v="${!r}"; [[ "$v" =~ ^[0-9]+$ ]] && ((v>=1)) || { echo "$r must be >=1" >&2; exit 2; }; done
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
v=float(sys.argv[1])
if v < 1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY

run_sweep(){
  local rows="$1" repeats="$2" thresholds="$3" threads="$4" tag="$5"
  local p="${PREFIX}_${tag}_r${rows}"
  echo "=== mainrec hybrid stage=$tag rows=$rows repeats=$repeats thresholds=[$thresholds] threads=[$threads] ===" >&2
  N=27 MOD="$MOD" ROWS="$rows" ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
    ILP8_THRESHOLDS="$thresholds" THREADS_LIST="$threads" REPEATS="$repeats" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" \
    RANDOM_CG="$RANDOM_CG" RANDOM_CG_L2_FETCH_BYTES="$RANDOM_CG_L2_FETCH_BYTES" DUALMASK="$DUALMASK" MAXRREGCOUNT="$MAXRREGCOUNT" \
    PREFIX="$p" LOGDIR="${p}_logs" RESULT="${p}.tsv" RESOURCE="${p}_ptxas.tsv" WINNER_ENV="${p}_winner.env" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid-ilp8-threshold-ab.sh" >"${p}.stdout" 2>"${p}.stderr"
  cat "${p}.stderr" >&2
  printf '%s\n' "$p"
}

stage_eval(){
  local p="$1" threshold="$2" threads="$3" out="$4"
  python3 - "${p}.tsv" "${p}_ptxas.tsv" "${p}_logs/binaries.tsv" "$threshold" "$threads" "$MIN_SPEEDUP" "$out" <<'PY'
import csv,statistics,sys,shlex
result,resource,binaries,threshold,threads,minsp,out=sys.argv[1:]
threads=int(threads); minsp=float(minsp)
rows=list(csv.DictReader(open(result),delimiter='\t'))
rr=list(csv.DictReader(open(resource),delimiter='\t'))
bins=list(csv.DictReader(open(binaries),delimiter='\t'))
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL staged residue mismatch')

def grp(mode,t):
    return [r for r in rows if r['mode']==mode and r['threshold']==t and int(r['threads'])==threads]
base=grp('ilp2','NA'); cand=grp('hybrid',threshold)
if not base: raise SystemExit('missing staged ILP2 baseline rows')
if not cand: raise SystemExit(f'missing staged hybrid threshold={threshold} rows')
bw=statistics.median(float(r['wall_s']) for r in base)
cw=statistics.median(float(r['wall_s']) for r in cand)
speed=bw/cw

resources=[r for r in rr if r['mode']=='hybrid' and r['threshold']==threshold]
vals=[]
for r in resources:
    try: vals.append((int(r['registers']),int(r['spill_store_bytes']),int(r['spill_load_bytes'])))
    except (ValueError,TypeError,KeyError): pass
known=len(vals)>=2
regs=max((x[0] for x in vals),default=-1)
ss=max((x[1] for x in vals),default=-1)
sl=max((x[2] for x in vals),default=-1)
spill_free=known and ss==0 and sl==0
valid=spill_free and speed>=minsp
bmap={(r['mode'],r['threshold']):r['binary'] for r in bins}
cb=bmap.get(('hybrid',threshold),''); bb=bmap.get(('ilp2','NA'),'')
if not cb or not bb: raise SystemExit('staged binary lookup failed')
def q(v): return shlex.quote(str(v))
with open(out,'w') as f:
    f.write('B300_MAINREC_HYBRID_STAGE_THRESHOLD='+q(threshold)+'\n')
    f.write('B300_MAINREC_HYBRID_STAGE_THREADS='+q(threads)+'\n')
    f.write('B300_MAINREC_HYBRID_STAGE_RESIDUE='+q(next(iter(res)))+'\n')
    f.write('B300_MAINREC_HYBRID_STAGE_BASE_WALL_S='+q(f'{bw:.9f}')+'\n')
    f.write('B300_MAINREC_HYBRID_STAGE_WALL_S='+q(f'{cw:.9f}')+'\n')
    f.write('B300_MAINREC_HYBRID_STAGE_SPEEDUP='+q(f'{speed:.9f}')+'\n')
    f.write('B300_MAINREC_HYBRID_STAGE_REGISTERS='+q(regs)+'\n')
    f.write('B300_MAINREC_HYBRID_STAGE_SPILL_STORE_BYTES='+q(ss)+'\n')
    f.write('B300_MAINREC_HYBRID_STAGE_SPILL_LOAD_BYTES='+q(sl)+'\n')
    f.write('B300_MAINREC_HYBRID_STAGE_SPILL_FREE='+q(int(spill_free))+'\n')
    f.write('B300_MAINREC_HYBRID_STAGE_VALID='+q(int(valid))+'\n')
    f.write('B300_MAINREC_HYBRID_STAGE_BIN='+q(cb)+'\n')
    f.write('B300_MAINREC_HYBRID_STAGE_BASE_BIN='+q(bb)+'\n')
print('MAINREC_HYBRID_STAGE',f'threshold={threshold}',f'threads={threads}',f'base_wall={bw:.9f}',f'hybrid_wall={cw:.9f}',f'speedup={speed:.6f}x',f'regs={regs}',f'spill_store={ss}',f'spill_load={sl}',f'spill_free={int(spill_free)}',f'valid={int(valid)}',f'residue={next(iter(res))}')
PY
}

SEARCH_PREFIX="$(run_sweep "$SEARCH_ROWS" "$SEARCH_REPEATS" "$ILP8_THRESHOLDS" "$THREADS_LIST" search)"
SEARCH_WINNER="${SEARCH_PREFIX}_winner.env"
[[ -f "$SEARCH_WINNER" ]] || { echo 'search winner env missing' >&2; exit 3; }
# shellcheck disable=SC1090
source "$SEARCH_WINNER"

VALIDATED=0
FINAL_STAGE_ENV=""
if [[ "$B300_MAINREC_HYBRID_WINNER_MODE" == hybrid ]]; then
  SELECTED_THRESHOLD="$B300_MAINREC_HYBRID_WINNER_THRESHOLD"
  SELECTED_THREADS="$B300_MAINREC_HYBRID_WINNER_THREADS"
  if python3 - "$B300_MAINREC_HYBRID_SPEEDUP_VS_ILP2" "$MIN_SPEEDUP" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)
PY
  then
    VALIDATED=1
    stage_i=0
    for rows in $VALIDATE_ROWS; do
      ((stage_i+=1))
      # threshold=0 is required by the lower-level sweep as the always-ILP8
      # reference. When the selected threshold is nonzero keep both, but the
      # evaluator below judges only the fixed search winner.
      if [[ "$SELECTED_THRESHOLD" == 0 ]]; then vt="0"; else vt="0 $SELECTED_THRESHOLD"; fi
      p="$(run_sweep "$rows" "$VALIDATE_REPEATS" "$vt" "$SELECTED_THREADS" "validate${stage_i}")"
      envfile="${p}_fixed.env"
      stage_eval "$p" "$SELECTED_THRESHOLD" "$SELECTED_THREADS" "$envfile" >"${p}_fixed.out"
      cat "${p}_fixed.out" >&2
      # shellcheck disable=SC1090
      source "$envfile"
      FINAL_STAGE_ENV="$envfile"
      if [[ "$B300_MAINREC_HYBRID_STAGE_VALID" != 1 ]]; then
        VALIDATED=0
        break
      fi
    done
  else
    echo "search winner failed MIN_SPEEDUP=$MIN_SPEEDUP; falling back to ILP2 without error" >&2
  fi
else
  echo 'search selected ILP2 baseline; hybrid staged validation skipped' >&2
fi

if [[ "$VALIDATED" == 1 && -n "$FINAL_STAGE_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$FINAL_STAGE_ENV"
  {
    printf 'B300_MAINREC_HYBRID_STAGED_VALIDATED=1\n'
    printf 'B300_MAINREC_HYBRID_SELECTED_THRESHOLD=%q\n' "$B300_MAINREC_HYBRID_STAGE_THRESHOLD"
    printf 'B300_MAINREC_HYBRID_SELECTED_THREADS=%q\n' "$B300_MAINREC_HYBRID_STAGE_THREADS"
    printf 'B300_MAINREC_HYBRID_SELECTED_BIN=%q\n' "$B300_MAINREC_HYBRID_STAGE_BIN"
    printf 'B300_MAINREC_HYBRID_SELECTED_BASE_BIN=%q\n' "$B300_MAINREC_HYBRID_STAGE_BASE_BIN"
    printf 'B300_MAINREC_HYBRID_SELECTED_RESIDUE=%q\n' "$B300_MAINREC_HYBRID_STAGE_RESIDUE"
    printf 'B300_MAINREC_HYBRID_SELECTED_WALL_S=%q\n' "$B300_MAINREC_HYBRID_STAGE_WALL_S"
    printf 'B300_MAINREC_HYBRID_SELECTED_BASE_WALL_S=%q\n' "$B300_MAINREC_HYBRID_STAGE_BASE_WALL_S"
    printf 'B300_MAINREC_HYBRID_SELECTED_SPEEDUP=%q\n' "$B300_MAINREC_HYBRID_STAGE_SPEEDUP"
    printf 'B300_MAINREC_HYBRID_SELECTED_SPILL_FREE=1\n'
    printf 'B300_MAINREC_HYBRID_MIN_SPEEDUP=%q\n' "$MIN_SPEEDUP"
    printf 'B300_MAINREC_HYBRID_SEARCH_ROWS=%q\n' "$SEARCH_ROWS"
    printf 'B300_MAINREC_HYBRID_VALIDATE_ROWS=%q\n' "$VALIDATE_ROWS"
    printf 'B300_MAINREC_HYBRID_HIGH_DROP_CHUNK=%q\n' "$HIGH_DROP_CHUNK"
    printf 'B300_MAINREC_HYBRID_RANDOM_CG=%q\n' "$RANDOM_CG"
    printf 'B300_MAINREC_HYBRID_RANDOM_CG_L2_FETCH_BYTES=%q\n' "$RANDOM_CG_L2_FETCH_BYTES"
    printf 'B300_MAINREC_HYBRID_DUALMASK=%q\n' "$DUALMASK"
    printf 'B300_MAINREC_HYBRID_MAXRREGCOUNT=%q\n' "$MAXRREGCOUNT"
  } >"$WINNER_ENV"
else
  {
    printf 'B300_MAINREC_HYBRID_STAGED_VALIDATED=0\n'
    printf 'B300_MAINREC_HYBRID_MIN_SPEEDUP=%q\n' "$MIN_SPEEDUP"
    printf 'B300_MAINREC_HYBRID_SEARCH_WINNER_MODE=%q\n' "${B300_MAINREC_HYBRID_WINNER_MODE:-unknown}"
    printf 'B300_MAINREC_HYBRID_SEARCH_WINNER_THRESHOLD=%q\n' "${B300_MAINREC_HYBRID_WINNER_THRESHOLD:-NA}"
    printf 'B300_MAINREC_HYBRID_SEARCH_WINNER_SPEEDUP=%q\n' "${B300_MAINREC_HYBRID_SPEEDUP_VS_ILP2:-NA}"
  } >"$WINNER_ENV"
fi

cat "$WINNER_ENV"
echo "b300-mainrec-hybrid-ilp8-calibrate-staged OK validated=$VALIDATED winner_env=$WINNER_ENV" >&2
