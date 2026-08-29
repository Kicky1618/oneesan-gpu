#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
[[ "$N" == 27 ]] || { echo 'next-self staged A/B targets n=27' >&2; exit 2; }
WAIT_VARIANTS="${WAIT_VARIANTS:-pairfirst blockfirst}"
SEARCH_ROWS="${SEARCH_ROWS:-1}"
VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"
SEARCH_REPEATS="${SEARCH_REPEATS:-1}"
VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.01}"
THREADS="${GRIDFP_THREADS:-256}"
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"
RANDOM_CG="${RANDOM_CG:-1}"
WARP_SCAN="${WARP_SCAN:-1}"
MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300x8_ilp8_nextself_staged}"
WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
mkdir -p "$(dirname "$WINNER_ENV")"

python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY
for rows in "$SEARCH_ROWS" $VALIDATE_ROWS; do
  [[ "$rows" =~ ^[0-9]+$ ]] && ((rows>=1 && rows<=28)) || { echo "bad stage rows=$rows" >&2; exit 2; }
done
for variant in $WAIT_VARIANTS; do
  case "$variant" in pairfirst|blockfirst) ;; *) echo "bad WAIT_VARIANTS entry=$variant" >&2; exit 2;; esac
done

stage_eval(){
  local summary="$1" variant="$2" envfile="$3" control_bin="$4" next_bin="$5"
  python3 - "$summary" "$variant" "$envfile" "$control_bin" "$next_bin" "$MIN_SPEEDUP" <<'PY'
import csv,shlex,statistics,sys
summary,variant,out,control_bin,next_bin,minsp=sys.argv[1:]
minsp=float(minsp)
rows=list(csv.DictReader(open(summary),delimiter='\t'))
if not rows: raise SystemExit('no next-self rows')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL next-self staged residue mismatch')
def grp(name): return [r for r in rows if r['profile']==name]
def med(name,key): return statistics.median(float(r[key]) for r in grp(name))
def maxint(name,key):
    vals=[]
    for r in grp(name):
        try: vals.append(int(r[key]))
        except ValueError: pass
    return max(vals) if vals else -1
cw=med('control','wall_s'); nw=med('nextself','wall_s')
cm=med('control','mc_avg_pct'); nm=med('nextself','mc_avg_pct')
css=maxint('control','spill_store_main_bytes'); csl=maxint('control','spill_load_main_bytes')
nss=maxint('nextself','spill_store_main_bytes'); nsl=maxint('nextself','spill_load_main_bytes')
speed=cw/nw
known=css>=0 and csl>=0 and nss>=0 and nsl>=0
spill_free=known and css==0 and csl==0 and nss==0 and nsl==0
valid=spill_free and speed>=minsp
def q(v): return shlex.quote(str(v))
vals={
 'B300_NEXTSELF_VARIANT':variant,
 'B300_NEXTSELF_RESIDUE':next(iter(res)),
 'B300_NEXTSELF_CONTROL_WALL_S':f'{cw:.9f}',
 'B300_NEXTSELF_WALL_S':f'{nw:.9f}',
 'B300_NEXTSELF_SPEEDUP':f'{speed:.9f}',
 'B300_NEXTSELF_CONTROL_MC_AVG_PCT':f'{cm:.3f}',
 'B300_NEXTSELF_MC_AVG_PCT':f'{nm:.3f}',
 'B300_NEXTSELF_MC_DELTA_PP':f'{nm-cm:.3f}',
 'B300_NEXTSELF_CONTROL_SPILL_FREE':int(css==0 and csl==0),
 'B300_NEXTSELF_SPILL_FREE':int(nss==0 and nsl==0),
 'B300_NEXTSELF_STAGE_VALID':int(valid),
 'B300_NEXTSELF_CONTROL_BIN':control_bin,
 'B300_NEXTSELF_BIN':next_bin,
}
with open(out,'w') as f:
    for k,v in vals.items(): f.write(f'{k}={q(v)}\n')
print(f"NEXTSELF_STAGE variant={variant} control_wall={cw:.9f} nextself_wall={nw:.9f} speedup={speed:.6f}x control_mc={cm:.3f} nextself_mc={nm:.3f} mc_delta={nm-cm:.3f}pp spill_free={int(spill_free)} valid={int(valid)} exact=1")
PY
}

run_stage(){
  local rows="$1" variant="$2" repeats="$3" tag="$4"
  local dir="${PREFIX}_${tag}_${variant}_r${rows}" summary="$dir/summary.tsv" envfile="$dir/stage.env"
  mkdir -p "$dir"
  echo "=== next-self stage rows=$rows variant=$variant repeats=$repeats ===" >&2
  N=27 ROWS="$rows" GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
    ARCH="$ARCH" RANDOM_CG="$RANDOM_CG" WARP_SCAN="$WARP_SCAN" MAXRREGCOUNT="$MAXRREGCOUNT" WAIT_VARIANT="$variant" REPEATS="$repeats" SAMPLE_INTERVAL="$SAMPLE_INTERVAL" \
    LOGDIR="$dir" SUMMARY="$summary" \
    bash "$ONEESAN_ROOT/scripts/bench/b300x8-ilp8-nextself-delta-ab.sh" >"$dir/delta.out"
  cat "$dir/delta.out" >&2
  stage_eval "$summary" "$variant" "$envfile" "$dir/control.bin" "$dir/nextself.bin" >"$dir/eval.out"
  cat "$dir/eval.out" >&2
  printf '%s\n' "$envfile"
}

SEARCH_TABLE="${PREFIX}_search.tsv"
printf 'variant\tvalid\tnextself_wall_s\tspeedup\tenv\n' >"$SEARCH_TABLE"
for variant in $WAIT_VARIANTS; do
  envfile="$(run_stage "$SEARCH_ROWS" "$variant" "$SEARCH_REPEATS" search)"
  # shellcheck disable=SC1090
  source "$envfile"
  printf '%s\t%s\t%s\t%s\t%s\n' "$variant" "$B300_NEXTSELF_STAGE_VALID" "$B300_NEXTSELF_WALL_S" "$B300_NEXTSELF_SPEEDUP" "$envfile" >>"$SEARCH_TABLE"
done

SELECTED_ENV="$(python3 - "$SEARCH_TABLE" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
valid=[r for r in rows if r['valid']=='1']
if not valid:
    print('')
else:
    print(min(valid,key=lambda r:(float(r['nextself_wall_s']),-float(r['speedup'])))['env'])
PY
)"

VALIDATED=0
CURRENT_ENV=""
if [[ -n "$SELECTED_ENV" ]]; then
  CURRENT_ENV="$SELECTED_ENV"
  # shellcheck disable=SC1090
  source "$CURRENT_ENV"
  SELECTED_VARIANT="$B300_NEXTSELF_VARIANT"
  VALIDATED=1
  stage_index=0
  for rows in $VALIDATE_ROWS; do
    ((stage_index+=1))
    CURRENT_ENV="$(run_stage "$rows" "$SELECTED_VARIANT" "$VALIDATE_REPEATS" "validate${stage_index}")"
    # shellcheck disable=SC1090
    source "$CURRENT_ENV"
    if [[ "$B300_NEXTSELF_STAGE_VALID" != 1 ]]; then VALIDATED=0; break; fi
  done
fi

if [[ "$VALIDATED" == 1 ]]; then
  # shellcheck disable=SC1090
  source "$CURRENT_ENV"
  {
    printf 'B300_NEXTSELF_STAGED_VALIDATED=1\n'
    cat "$CURRENT_ENV"
    printf 'B300_NEXTSELF_MIN_SPEEDUP=%q\n' "$MIN_SPEEDUP"
  } >"$WINNER_ENV"
else
  {
    printf 'B300_NEXTSELF_STAGED_VALIDATED=0\n'
    printf 'B300_NEXTSELF_MIN_SPEEDUP=%q\n' "$MIN_SPEEDUP"
  } >"$WINNER_ENV"
fi

cat "$SEARCH_TABLE" >&2
cat "$WINNER_ENV"
echo "b300x8-ilp8-nextself-staged-ab OK validated=$VALIDATED winner_env=$WINNER_ENV" >&2
