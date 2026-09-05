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
ILP8_PREFETCH_REGS="${ILP8_PREFETCH_REGS:-0 96 128 160}"
ILP8_CPASYNC_REGS="${ILP8_CPASYNC_REGS:-0 96 128 160}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
REBUILD="${REBUILD:-0}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300x8_saturate_ilp8_ab_n${N}}"
SUMMARY="${SUMMARY:-$LOGDIR/summary.tsv}"
WINNER_ENV="${WINNER_ENV:-$LOGDIR/winner.env}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR"

command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
python3 - "$SAMPLE_INTERVAL" <<'PY'
import sys
if float(sys.argv[1]) <= 0: raise SystemExit('SAMPLE_INTERVAL must be >0')
PY
[[ "$THREADS" =~ ^[0-9]+$ ]] && ((THREADS>=32&&THREADS<=1024&&THREADS%32==0)) || { echo 'GRIDFP_THREADS must be warp multiple 32..1024' >&2; exit 2; }

printf 'profile\tresidue\twall_s\tactive_max_s\tactive_sum_s\tmc_avg_pct\tmc_max_pct\tmc_samples\tregs_main\tspill_store_main_bytes\tspill_load_main_bytes\tbinary\n' >"$SUMMARY"
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
  local log="$LOGDIR/${name}.log" memlog="$LOGDIR/${name}.mem" res="$LOGDIR/${name}.main.ptxas.tsv"
  echo "=== $name ===" >&2
  set +e
  "$@" >"$log" 2>&1 &
  local pid=$!
  sample_mem "$pid" "$memlog" & local mpid=$!
  wait "$pid"; local rc=$?
  set -e
  wait "$mpid" 2>/dev/null || true
  cat "$log"
  (( rc==0 )) || return "$rc"

  local line residue wall active_max active_sum mem_avg mem_max mem_n regs_main spill_store spill_load row bin
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

  : >"$res"
  if python3 "$PARSER" "$log" --label "$name" --contains b300_main_pull_rankstate_ilp4_kernel >"$res" 2>"$res.err"; then
    read -r regs_main spill_store spill_load < <(python3 - "$res" <<'PY'
import csv,sys
vals=[]
for x in csv.reader(open(sys.argv[1]),delimiter='\t'):
    if len(x)<8: continue
    try: vals.append((int(x[2]),int(x[4]),int(x[5])))
    except ValueError: pass
print('nan nan nan' if not vals else f'{max(x[0] for x in vals)} {max(x[1] for x in vals)} {max(x[2] for x in vals)}')
PY
)
  else
    regs_main=nan; spill_store=nan; spill_load=nan
  fi
  bin="$(sed -nE 's/^BIN=(.*)$/\1/p' "$log" | tail -n1)"
  [[ -n "$bin" && -x "$bin" ]] || { echo "$name missing executable BIN marker" >&2; return 5; }
  row="$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' "$name" "$residue" "$wall" "${active_max:-nan}" "${active_sum:-nan}" "$mem_avg" "$mem_max" "$mem_n" "$regs_main" "$spill_store" "$spill_load" "$bin")"
  printf '%s\n' "$row" | tee -a "$SUMMARY"
}

common_env=(ROWS="$ROWS" GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" RANDOM_CG="$RANDOM_CG" WARP_SCAN="$WARP_SCAN" REBUILD="$REBUILD")

run_one ilp4 env "${common_env[@]}" "$ONEESAN_ROOT/scripts/run/b300x8-saturate-dualmask.sh" "$N" "$MOD"

for r in $ILP8_REGS; do
  run_one "ilp8_sync_r${r}" env "${common_env[@]}" PREFETCH_L2=0 CPASYNC=0 MAXRREGCOUNT="$r" \
    "$ONEESAN_ROOT/scripts/run/b300x8-saturate-ilp8.sh" "$N" "$MOD"
done

for r in $ILP8_PREFETCH_REGS; do
  run_one "ilp8_prefetch_r${r}" env "${common_env[@]}" PREFETCH_L2=1 CPASYNC=0 MAXRREGCOUNT="$r" \
    "$ONEESAN_ROOT/scripts/run/b300x8-saturate-ilp8.sh" "$N" "$MOD"
done

for r in $ILP8_CPASYNC_REGS; do
  run_one "ilp8_cpasync_r${r}" env "${common_env[@]}" PREFETCH_L2=0 CPASYNC=1 MAXRREGCOUNT="$r" \
    "$ONEESAN_ROOT/scripts/run/b300x8-saturate-ilp8.sh" "$N" "$MOD"
done

python3 - "$SUMMARY" "$WINNER_ENV" "$THREADS" <<'PY'
import csv,math,sys,shlex,re
summary,winner,threads=sys.argv[1:]
rows=list(csv.DictReader(open(summary),delimiter='\t'))
if not rows: raise SystemExit('no ILP saturation rows')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL ILP4/ILP8 residue mismatch '+repr({r['profile']:r['residue'] for r in rows}))
ilp8=[r for r in rows if r['profile'].startswith('ilp8_') and r['wall_s'] not in ('','nan')]
if not ilp8: raise SystemExit('no successful ILP8 candidate')
def num(r,k,default=math.inf):
    try:return float(r[k])
    except (ValueError,TypeError):return default
def spill_known_free(r):
    return num(r,'regs_main',-1)>=0 and num(r,'spill_store_main_bytes',1)==0 and num(r,'spill_load_main_bytes',1)==0
clean=[r for r in ilp8 if spill_known_free(r)]
if not clean: raise SystemExit('no ILP8 candidate has known spill-free main-kernel ptxas data; rebuild/resource log required')
key=lambda r:(num(r,'wall_s'),-num(r,'mc_avg_pct',-math.inf))
best=min(clean,key=key)
base=next((r for r in rows if r['profile']=='ilp4' and r['wall_s'] not in ('','nan')),None)
common=next(iter(res))
def classbest(prefix):
    x=[r for r in clean if r['profile'].startswith(prefix)]
    return min(x,key=key) if x else None
bs=classbest('ilp8_sync_');bp=classbest('ilp8_prefetch_');ba=classbest('ilp8_cpasync_')
for label,r in (('sync',bs),('prefetch',bp),('cpasync',ba)):
    if r: print(f"ILP8_CLASS_BEST class={label} profile={r['profile']} wall_s={r['wall_s']} mc_avg_pct={r['mc_avg_pct']} mc_max_pct={r['mc_max_pct']} mc_samples={r['mc_samples']} regs_main={r['regs_main']} spill_store={r['spill_store_main_bytes']} spill_load={r['spill_load_main_bytes']}",file=sys.stderr)
if bs and bp:
    print(f"ILP8_PREFETCH_COMPARE sync={bs['profile']} prefetch={bp['profile']} speedup={num(bs,'wall_s')/num(bp,'wall_s'):.6f}x mc_delta={num(bp,'mc_avg_pct',0)-num(bs,'mc_avg_pct',0):.3f} exact_gate=1",file=sys.stderr)
if bs and ba:
    print(f"ILP8_ASYNC_COMPARE sync={bs['profile']} async={ba['profile']} speedup={num(bs,'wall_s')/num(ba,'wall_s'):.6f}x mc_delta={num(ba,'mc_avg_pct',0)-num(bs,'mc_avg_pct',0):.3f} exact_gate=1",file=sys.stderr)
if base:
    print(f"ILP8_SELECTED profile={best['profile']} residue={common} wall_s={best['wall_s']} speedup={num(base,'wall_s')/num(best,'wall_s'):.6f}x mc_avg_pct={best['mc_avg_pct']} mc_samples={best['mc_samples']} regs_main={best['regs_main']} spill_store_main={best['spill_store_main_bytes']} spill_load_main={best['spill_load_main_bytes']} spill_free_pool=1 exact_gate=1",file=sys.stderr)
else:
    print(f"ILP8_SELECTED profile={best['profile']} residue={common} wall_s={best['wall_s']} mc_avg_pct={best['mc_avg_pct']} mc_samples={best['mc_samples']} regs_main={best['regs_main']} spill_store_main={best['spill_store_main_bytes']} spill_load_main={best['spill_load_main_bytes']} spill_free_pool=1 exact_gate=1",file=sys.stderr)
prefetch=int(best['profile'].startswith('ilp8_prefetch_'))
cpasync=int(best['profile'].startswith('ilp8_cpasync_'))
m=re.search(r'_r(\d+)$',best['profile']); maxr=int(m.group(1)) if m else 0
def q(x): return shlex.quote(str(x))
with open(winner,'w') as f:
    f.write('B300_ILP8_WINNER_PROFILE='+q(best['profile'])+'\n')
    f.write('B300_ILP8_WINNER_BIN='+q(best['binary'])+'\n')
    f.write('B300_ILP8_WINNER_THREADS='+q(threads)+'\n')
    f.write('B300_ILP8_WINNER_PREFETCH_L2='+q(prefetch)+'\n')
    f.write('B300_ILP8_WINNER_CPASYNC='+q(cpasync)+'\n')
    f.write('B300_ILP8_WINNER_MAXRREGCOUNT='+q(maxr)+'\n')
    f.write('B300_ILP8_WINNER_RESIDUE='+q(common)+'\n')
    f.write('B300_ILP8_WINNER_WALL_S='+q(best['wall_s'])+'\n')
    f.write('B300_ILP8_WINNER_MC_AVG_PCT='+q(best['mc_avg_pct'])+'\n')
    f.write('B300_ILP8_WINNER_REGS_MAIN='+q(best['regs_main'])+'\n')
    f.write('B300_ILP8_WINNER_SPILL_STORE_MAIN_BYTES='+q(best['spill_store_main_bytes'])+'\n')
    f.write('B300_ILP8_WINNER_SPILL_LOAD_MAIN_BYTES='+q(best['spill_load_main_bytes'])+'\n')
print('B300_ILP8_WINNER_ENV='+winner,file=sys.stderr)
PY

echo "summary=$SUMMARY winner_env=$WINNER_ENV logs=$LOGDIR sample_interval=$SAMPLE_INTERVAL" >&2
