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
ILP8_REGS="${ILP8_REGS:-0 96 128 160}"
ILP8_CPASYNC_REGS="${ILP8_CPASYNC_REGS:-0 96 128 160}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
REBUILD="${REBUILD:-0}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300x8_saturate_ilp8_ab_n${N}}"
SUMMARY="${SUMMARY:-$LOGDIR/summary.tsv}"
mkdir -p "$LOGDIR"

command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
python3 - "$SAMPLE_INTERVAL" <<'PY'
import sys
if float(sys.argv[1]) <= 0: raise SystemExit('SAMPLE_INTERVAL must be >0')
PY

printf 'profile\tresidue\twall_s\tactive_max_s\tactive_sum_s\tmc_avg_pct\tmc_max_pct\tmc_samples\tregs_max\tspill_store_max_bytes\tspill_load_max_bytes\n' >"$SUMMARY"
cat "$SUMMARY"

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

run_one(){
  local name="$1";shift
  local log="$LOGDIR/${name}.log" memlog="$LOGDIR/${name}.mem"
  echo "=== $name ===" >&2
  set +e
  "$@" > >(tee "$log") 2>&1 &
  local pid=$!
  sample_mem "$pid" "$memlog" & local mpid=$!
  wait "$pid"; local rc=$?
  set -e
  wait "$mpid" 2>/dev/null || true
  (( rc==0 )) || return "$rc"

  local line residue wall active_max active_sum mem_avg mem_max mem_n regs_max spill_store spill_load row
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$log" | tail -n1 || true)"
  [[ -n "$line" ]] || line="$(grep '^backend=gridfp-b300-hbm32' "$log" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$name missing backend result line" >&2; return 4; }
  residue="$(field residue "$line")"
  wall="$(field wall_s "$line")"
  active_max="$(field active_max_s "$line")"
  active_sum="$(field active_sum_s "$line")"
  [[ -n "$residue" && -n "$wall" ]] || { echo "$name missing residue/wall_s" >&2; return 4; }
  read -r mem_avg mem_max mem_n < <(awk '
    {s+=$1;n++;if($1>m)m=$1}
    END{if(n)printf "%.3f %.3f %d\n",s/n,m,n;else print "nan nan 0"}
  ' "$memlog")
  read -r regs_max spill_store spill_load < <(python3 - "$log" <<'PY'
import re,sys
text=open(sys.argv[1],encoding='utf-8',errors='replace').read()
regs=[int(x) for x in re.findall(r'Used\s+(\d+)\s+registers',text)]
spill=[(int(a),int(b)) for a,b in re.findall(r'(\d+)\s+bytes spill stores,\s*(\d+)\s+bytes spill loads',text)]
print(max(regs) if regs else 'nan', max((x for x,_ in spill),default='nan'), max((y for _,y in spill),default='nan'))
PY
)
  row="$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' "$name" "$residue" "$wall" "${active_max:-nan}" "${active_sum:-nan}" "$mem_avg" "$mem_max" "$mem_n" "$regs_max" "$spill_store" "$spill_load")"
  printf '%s\n' "$row" | tee -a "$SUMMARY"
}

run_one ilp4 \
  env ROWS="$ROWS" GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
      RANDOM_CG="$RANDOM_CG" WARP_SCAN="$WARP_SCAN" REBUILD="$REBUILD" \
  "$ONEESAN_ROOT/scripts/run/b300x8-saturate-dualmask.sh" "$N" "$MOD"

for r in $ILP8_REGS; do
  run_one "ilp8_sync_r${r}" \
    env ROWS="$ROWS" GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
        RANDOM_CG="$RANDOM_CG" WARP_SCAN="$WARP_SCAN" CPASYNC=0 MAXRREGCOUNT="$r" REBUILD="$REBUILD" \
    "$ONEESAN_ROOT/scripts/run/b300x8-saturate-ilp8.sh" "$N" "$MOD"
done

for r in $ILP8_CPASYNC_REGS; do
  run_one "ilp8_cpasync_r${r}" \
    env ROWS="$ROWS" GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
        RANDOM_CG="$RANDOM_CG" WARP_SCAN="$WARP_SCAN" CPASYNC=1 MAXRREGCOUNT="$r" REBUILD="$REBUILD" \
    "$ONEESAN_ROOT/scripts/run/b300x8-saturate-ilp8.sh" "$N" "$MOD"
done

python3 - "$SUMMARY" <<'PY'
import csv,math,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
if not rows:
    raise SystemExit('no ILP saturation rows')
res={r['residue'] for r in rows}
if len(res)!=1:
    raise SystemExit('FATAL ILP4/ILP8 residue mismatch '+repr({r['profile']:r['residue'] for r in rows}))
ilp8=[r for r in rows if r['profile'].startswith('ilp8_') and r['wall_s'] not in ('','nan')]
if not ilp8:
    raise SystemExit('no successful ILP8 candidate')
def num(r,k,default=math.inf):
    try:return float(r[k])
    except (ValueError,TypeError):return default
def spill_free(r):
    return num(r,'spill_store_max_bytes',1)==0 and num(r,'spill_load_max_bytes',1)==0
clean=[r for r in ilp8 if spill_free(r)]
pool=clean or ilp8
best=min(pool,key=lambda r:(num(r,'wall_s'),-num(r,'mc_avg_pct',-math.inf)))
base=next((r for r in rows if r['profile']=='ilp4' and r['wall_s'] not in ('','nan')),None)
common=next(iter(res))
sync=[r for r in ilp8 if r['profile'].startswith('ilp8_sync_')]
asyncs=[r for r in ilp8 if r['profile'].startswith('ilp8_cpasync_')]
def bestof(xs):
    ys=[r for r in xs if spill_free(r)] or xs
    return min(ys,key=lambda r:(num(r,'wall_s'),-num(r,'mc_avg_pct',-math.inf))) if ys else None
bs,ba=bestof(sync),bestof(asyncs)
if bs and ba:
    print(f"ILP8_ASYNC_COMPARE sync={bs['profile']} sync_wall={bs['wall_s']} sync_mc={bs['mc_avg_pct']} sync_mc_samples={bs['mc_samples']} async={ba['profile']} async_wall={ba['wall_s']} async_mc={ba['mc_avg_pct']} async_mc_samples={ba['mc_samples']} async_speedup={num(bs,'wall_s')/num(ba,'wall_s'):.6f}x exact_gate=1",file=sys.stderr)
if base:
    print(f"ILP8_SELECTED profile={best['profile']} residue={common} wall_s={best['wall_s']} speedup={num(base,'wall_s')/num(best,'wall_s'):.6f}x mc_avg_pct={best['mc_avg_pct']} mc_samples={best['mc_samples']} regs_max={best['regs_max']} spill_store_max={best['spill_store_max_bytes']} spill_load_max={best['spill_load_max_bytes']} spill_free_pool={int(bool(clean))} exact_gate=1",file=sys.stderr)
else:
    print(f"ILP8_SELECTED profile={best['profile']} residue={common} wall_s={best['wall_s']} mc_avg_pct={best['mc_avg_pct']} mc_samples={best['mc_samples']} regs_max={best['regs_max']} spill_store_max={best['spill_store_max_bytes']} spill_load_max={best['spill_load_max_bytes']} spill_free_pool={int(bool(clean))} exact_gate=1",file=sys.stderr)
PY

echo "summary=$SUMMARY logs=$LOGDIR sample_interval=$SAMPLE_INTERVAL" >&2
