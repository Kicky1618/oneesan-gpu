#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
MOD="${MOD:-4294967291}"
ROWS="${ROWS:-1}"
THREADS="${GRIDFP_THREADS:-256}"
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
RANDOM_CG="${RANDOM_CG:-1}"
WARP_SCAN="${WARP_SCAN:-1}"
THRESHOLDS="${ILP8_THRESHOLDS:-0 262144 524288 1048576 2097152 4194304 8388608}"
REPEATS="${REPEATS:-1}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
REBUILD="${REBUILD:-0}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300x8_hybrid_ilp8_threshold_ab_n${N}_rows${ROWS}}"
RESULT="${RESULT:-$LOGDIR/results.tsv}"
mkdir -p "$LOGDIR"

[[ "$N" == 27 ]] || { echo 'hybrid ILP8 threshold A/B currently targets n=27' >&2; exit 2; }
[[ "$ROWS" =~ ^[0-9]+$ ]] && (( ROWS >= 1 && ROWS <= 28 )) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$THREADS" =~ ^[0-9]+$ ]] && (( THREADS >= 32 && THREADS <= 1024 && THREADS % 32 == 0 )) || { echo 'GRIDFP_THREADS must be warp multiple 32..1024' >&2; exit 2; }
[[ "$REPEATS" =~ ^[0-9]+$ ]] && (( REPEATS >= 1 )) || { echo 'REPEATS must be >=1' >&2; exit 2; }
for x in RANDOM_CG WARP_SCAN REBUILD; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
python3 - "$SAMPLE_INTERVAL" <<'PY'
import sys
if float(sys.argv[1]) <= 0: raise SystemExit('SAMPLE_INTERVAL must be >0')
PY

read -r -a threshold_list <<<"$THRESHOLDS"
(( ${#threshold_list[@]} > 0 )) || { echo 'ILP8_THRESHOLDS is empty' >&2; exit 2; }
declare -A seen=()
for t in "${threshold_list[@]}"; do
  [[ "$t" =~ ^[0-9]+$ ]] || { echo "invalid threshold=$t" >&2; exit 2; }
  [[ -z "${seen[$t]+x}" ]] || { echo "duplicate threshold=$t" >&2; exit 2; }
  seen[$t]=1
done
[[ -n "${seen[0]+x}" ]] || { echo 'ILP8_THRESHOLDS must include 0 (always-ILP8 baseline)' >&2; exit 2; }

field(){
  local key="$1" line="$2"
  sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1
}

sample_mem(){
  local pid="$1" out="$2"
  : >"$out"
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits 2>/dev/null |
      awk '{s+=$1;n++}END{if(n)printf "%.6f\n",s/n}' >>"$out" || true
    sleep "$SAMPLE_INTERVAL"
  done
}

printf 'mode\tthreshold\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tmc_avg_pct\tmc_max_pct\tmc_samples\tregs_max\tspill_store_max_bytes\tspill_load_max_bytes\n' >"$RESULT"

run_one(){
  local mode="$1" threshold="$2" rep="$3"; shift 3
  local tag="${mode}_t${threshold}_r${rep}"
  local log="$LOGDIR/${tag}.log" memlog="$LOGDIR/${tag}.mem"
  echo "=== hybrid ILP8 threshold A/B mode=$mode threshold=$threshold repeat=$rep ===" >&2
  set +e
  "$@" >"$log" 2>&1 &
  local pid=$!
  sample_mem "$pid" "$memlog" & local mpid=$!
  wait "$pid"; local rc=$?
  set -e
  wait "$mpid" 2>/dev/null || true
  (( rc == 0 )) || { tail -n 120 "$log" >&2 || true; echo "$tag failed rc=$rc" >&2; return "$rc"; }

  local line residue wall active_max active_sum mem_avg mem_max mem_n regs_max spill_store spill_load
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$log" | tail -n1 || true)"
  [[ -n "$line" ]] || line="$(grep '^backend=gridfp-b300-hbm32' "$log" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$tag missing backend result line" >&2; return 4; }
  residue="$(field residue "$line")"
  wall="$(field wall_s "$line")"
  active_max="$(field active_max_s "$line")"
  active_sum="$(field active_sum_s "$line")"
  [[ -n "$residue" && -n "$wall" ]] || { echo "$tag missing residue/wall_s" >&2; return 4; }
  read -r mem_avg mem_max mem_n < <(awk '{s+=$1;n++;if($1>m)m=$1}END{if(n)printf "%.3f %.3f %d\n",s/n,m,n;else print "nan nan 0"}' "$memlog")
  read -r regs_max spill_store spill_load < <(python3 - "$log" <<'PY'
import re,sys
text=open(sys.argv[1],encoding='utf-8',errors='replace').read()
regs=[int(x) for x in re.findall(r'Used\s+(\d+)\s+registers',text)]
spill=[(int(a),int(b)) for a,b in re.findall(r'(\d+)\s+bytes spill stores,\s*(\d+)\s+bytes spill loads',text)]
print(max(regs) if regs else 'nan', max((x for x,_ in spill),default='nan'), max((y for _,y in spill),default='nan'))
PY
)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$mode" "$threshold" "$rep" "$residue" "$wall" "${active_max:-nan}" "${active_sum:-nan}" "$mem_avg" "$mem_max" "$mem_n" "$regs_max" "$spill_store" "$spill_load" >>"$RESULT"
}

for (( rep=1; rep<=REPEATS; ++rep )); do
  if (( rep % 2 == 1 )); then
    run_one ilp4 NA "$rep" \
      env ROWS="$ROWS" GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" RANDOM_CG="$RANDOM_CG" WARP_SCAN="$WARP_SCAN" REBUILD="$REBUILD" \
      "$ONEESAN_ROOT/scripts/run/b300x8-saturate-dualmask.sh" "$N" "$MOD"
    for t in "${threshold_list[@]}"; do
      run_one hybrid "$t" "$rep" \
        env ROWS="$ROWS" GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" RANDOM_CG="$RANDOM_CG" WARP_SCAN="$WARP_SCAN" ILP8_MIN_STATES="$t" REBUILD="$REBUILD" \
        "$ONEESAN_ROOT/scripts/run/b300x8-saturate-hybrid-ilp8.sh" "$N" "$MOD"
    done
  else
    for (( j=${#threshold_list[@]}-1; j>=0; --j )); do
      t="${threshold_list[$j]}"
      run_one hybrid "$t" "$rep" \
        env ROWS="$ROWS" GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" RANDOM_CG="$RANDOM_CG" WARP_SCAN="$WARP_SCAN" ILP8_MIN_STATES="$t" REBUILD="$REBUILD" \
        "$ONEESAN_ROOT/scripts/run/b300x8-saturate-hybrid-ilp8.sh" "$N" "$MOD"
    done
    run_one ilp4 NA "$rep" \
      env ROWS="$ROWS" GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" RANDOM_CG="$RANDOM_CG" WARP_SCAN="$WARP_SCAN" REBUILD="$REBUILD" \
      "$ONEESAN_ROOT/scripts/run/b300x8-saturate-dualmask.sh" "$N" "$MOD"
  fi
done

python3 - "$RESULT" <<'PY'
import csv,math,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
if not rows: raise SystemExit('no hybrid threshold rows')
res={r['residue'] for r in rows}
if len(res)!=1:
    raise SystemExit('FATAL hybrid ILP4/ILP8 residue mismatch '+repr({(r['mode'],r['threshold'],r['repeat']):r['residue'] for r in rows}))
by={}
for r in rows:
    by.setdefault((r['mode'],r['threshold']),[]).append(r)
def median_num(rs,key):
    vals=[float(r[key]) for r in rs if r[key] not in ('','nan')]
    return statistics.median(vals) if vals else math.nan
def spill_free(rs):
    vals=[]
    for r in rs:
        try: vals.append((int(r['spill_store_max_bytes']),int(r['spill_load_max_bytes'])))
        except ValueError: return False
    return bool(vals) and all(a==0 and b==0 for a,b in vals)
agg=[]
for (mode,t),rs in by.items():
    agg.append({
        'mode':mode,'threshold':t,
        'wall':median_num(rs,'wall_s'),'mc':median_num(rs,'mc_avg_pct'),
        'mc_samples':sum(int(r['mc_samples']) for r in rs),
        'regs':max((int(r['regs_max']) for r in rs if r['regs_max'].isdigit()),default=-1),
        'spill_free':spill_free(rs),
    })
base=next((x for x in agg if x['mode']=='ilp4'),None)
if base is None: raise SystemExit('missing pure ILP4 baseline')
hybrid=[x for x in agg if x['mode']=='hybrid']
clean=[x for x in hybrid if x['spill_free']]
pool=[base]+clean
key=lambda x:(x['wall'],-x['mc'] if not math.isnan(x['mc']) else math.inf)
best=min(pool,key=key)
for x in sorted(agg,key=key):
    print('HYBRID_ILP8_CANDIDATE',f"mode={x['mode']}",f"threshold={x['threshold']}",f"median_wall_s={x['wall']:.9f}",f"speedup_vs_ilp4={base['wall']/x['wall']:.6f}x",f"mc_avg_pct={x['mc']:.3f}",f"mc_samples={x['mc_samples']}",f"regs_max={x['regs']}",f"spill_free={int(x['spill_free'])}",file=sys.stderr)
print('HYBRID_ILP8_SELECTED',
      f"mode={best['mode']}",f"threshold={best['threshold']}",
      f"median_wall_s={best['wall']:.9f}",f"speedup_vs_ilp4={base['wall']/best['wall']:.6f}x",
      f"mc_avg_pct={best['mc']:.3f}",f"mc_delta_vs_ilp4={best['mc']-base['mc']:.3f}pp",
      f"regs_max={best['regs']}",f"spill_free={int(best['spill_free'])}",
      f"spill_free_hybrid_pool={len(clean)}",f"residue={next(iter(res))}",f'exact_gate=1',file=sys.stderr)
PY

cat "$RESULT"
echo "hybrid ILP8 threshold A/B OK result=$RESULT rows=$ROWS repeats=$REPEATS sample_interval=$SAMPLE_INTERVAL" >&2
