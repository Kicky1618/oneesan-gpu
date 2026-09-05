#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
MOD="${MOD:-4294967291}"
ROWS="${ROWS:-1}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
HYBRID_THRESHOLD="${HYBRID_THRESHOLD:-1048576}"
RANDOM_CG="${RANDOM_CG:-0}"
RANDOM_CG_L2_FETCH_BYTES="${RANDOM_CG_L2_FETCH_BYTES:-0}"
PREFETCH_L2="${PREFETCH_L2:-0}"
DUALMASK="${DUALMASK:-0}"
CLOSURE_BATCH="${CLOSURE_BATCH:-0}"
MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
REPEATS="${REPEATS:-1}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_t${HYBRID_THRESHOLD}_row${ROWS}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESULT="${RESULT:-${PREFIX}.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")" "$(dirname "$WINNER_ENV")"

[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$HYBRID_THRESHOLD" =~ ^[0-9]+$ ]] || { echo 'HYBRID_THRESHOLD must be non-negative integer' >&2; exit 2; }
[[ "$HIGH_DROP_CHUNK" == 0 || "$HIGH_DROP_CHUNK" == 1 ]] || exit 2
for x in RANDOM_CG PREFETCH_L2 DUALMASK; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
case "$RANDOM_CG_L2_FETCH_BYTES" in 0|64|128|256) ;; *) exit 2;; esac
[[ "$RANDOM_CG" == 1 || "$RANDOM_CG_L2_FETCH_BYTES" == 0 ]] || { echo 'CG L2 fetch hint requires RANDOM_CG=1' >&2; exit 2; }
case "$CLOSURE_BATCH" in 0|2|4) ;; *) exit 2;; esac
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] && ((MAXRREGCOUNT==0 || (MAXRREGCOUNT>=32 && MAXRREGCOUNT<=255))) || exit 2
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || exit 2
python3 - "$SAMPLE_INTERVAL" <<'PY'
import sys
if float(sys.argv[1])<=0: raise SystemExit('SAMPLE_INTERVAL must be >0')
PY
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

printf 'nextself\tbinary\tbuild_err\n' >"$LOGDIR/binaries.tsv"
for nextself in 0 1; do
  bin="$ONEESAN_BUILD_DIR/b300_nextgen_hybrid8_nextself${nextself}_t${HYBRID_THRESHOLD}_h${HIGH_DROP_CHUNK}_cg${RANDOM_CG}_l2${RANDOM_CG_L2_FETCH_BYTES}_pre${PREFETCH_L2}_d${DUALMASK}_b${CLOSURE_BATCH}_r${MAXRREGCOUNT}_n27"
  err="$LOGDIR/nextself${nextself}.build.err"
  out="$LOGDIR/nextself${nextself}.build.out"
  N=27 ARCH="$ARCH" OUT="$bin" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" RECURRENCE_ILP=2 \
    RECURRENCE_HYBRID_ILP8=1 RECURRENCE_HYBRID_ILP8_MIN_STATES="$HYBRID_THRESHOLD" RECURRENCE_HYBRID_ILP8_NEXTSELF="$nextself" \
    RANDOM_CG="$RANDOM_CG" RANDOM_CG_L2_FETCH_BYTES="$RANDOM_CG_L2_FETCH_BYTES" PREFETCH_L2="$PREFETCH_L2" \
    DUALMASK="$DUALMASK" CLOSURE_BATCH="$CLOSURE_BATCH" MAXRREGCOUNT="$MAXRREGCOUNT" PTXAS_VERBOSE=1 BUILD_ERR="$err" \
    bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen.sh" >"$out" 2>"$LOGDIR/nextself${nextself}.build.driver.err"
  [[ -x "$bin" ]] || { echo "nextself=$nextself binary missing" >&2; exit 3; }
  grep -Fq "recurrence_hybrid_ilp8=1 recurrence_hybrid_ilp8_min_states=$HYBRID_THRESHOLD recurrence_hybrid_ilp8_nextself=$nextself" "$out" || {
    echo "nextself=$nextself build summary mismatch" >&2; exit 3;
  }
  if [[ "$nextself" == 1 ]]; then
    grep -Fq 'b300_mainrec_hybrid8_next_self_prefetch=1' "$LOGDIR/nextself1.build.driver.err" "$out" "$err" 2>/dev/null || true
  fi
  printf '%s\t%s\t%s\n' "$nextself" "$bin" "$err" >>"$LOGDIR/binaries.tsv"
done

printf 'nextself\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
while IFS=$'\t' read -r nextself bin err; do
  [[ "$nextself" == nextself ]] && continue
  python3 "$PARSER" "$err" --label "$nextself" --contains main_pull_kernel_ilp2 >>"$RESOURCE" || true
  python3 "$PARSER" "$err" --label "$nextself" --contains main_pull_kernel_ilp8_hybrid >>"$RESOURCE" || true
done <"$LOGDIR/binaries.tsv"

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample_mem(){
  local pid="$1" out="$2"; : >"$out"
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits 2>/dev/null |
      awk '{s+=$1;n++;if($1>m)m=$1}END{if(n)printf "%.6f %.6f\n",s/n,m}' >>"$out" || true
    sleep "$SAMPLE_INTERVAL"
  done
}

printf 'nextself\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\thigh_rec_groups\thigh_rec_fallback_groups\tmc_avg_pct\tmc_max_pct\tmc_samples\n' >"$RESULT"
run_one(){
  local nextself="$1" bin="$2" threads="$3" repeat="$4"
  local so="$LOGDIR/ns${nextself}_t${threads}_r${repeat}.out" se="$LOGDIR/ns${nextself}_t${threads}_r${repeat}.err" mem="$LOGDIR/ns${nextself}_t${threads}_r${repeat}.mem"
  set +e
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$threads" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se" &
  local pid=$!; sample_mem "$pid" "$mem" & local mpid=$!
  wait "$pid"; local rc=$?
  set -e
  wait "$mpid" 2>/dev/null || true
  ((rc==0)) || { echo "nextself=$nextself threads=$threads repeat=$repeat failed rc=$rc" >&2; tail -n 80 "$se" >&2 || true; return "$rc"; }
  local line hg hf ma mm mn
  line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "nextself=$nextself result missing" >&2; return 4; }
  hg="$(field high_rec_groups "$line")"; hf="$(field high_rec_fallback_groups "$line")"
  [[ "$hg" =~ ^[0-9]+$ ]] && ((hg>0)) || return 5
  read -r ma mm mn < <(awk '{sa+=$1;if($2>mm)mm=$2;n++}END{if(n)printf "%.3f %.3f %d\n",sa/n,mm,n;else print "nan nan 0"}' "$mem")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$nextself" "$threads" "$repeat" "$(field residue "$line")" "$(field wall_s "$line")" \
    "$(field active_max_s "$line")" "$(field active_sum_s "$line")" "$hg" "$hf" "$ma" "$mm" "$mn" >>"$RESULT"
}

for threads in $THREADS_LIST; do
  [[ "$threads" =~ ^[0-9]+$ ]] && ((threads>=32 && threads<=1024 && threads%32==0)) || { echo "bad threads=$threads" >&2; exit 2; }
  while IFS=$'\t' read -r nextself bin err; do
    [[ "$nextself" == nextself ]] && continue
    for ((repeat=1; repeat<=REPEATS; ++repeat)); do
      echo "=== hybrid8-nextself nextself=$nextself threads=$threads repeat=$repeat rows=$ROWS ===" >&2
      run_one "$nextself" "$bin" "$threads" "$repeat"
    done
  done <"$LOGDIR/binaries.tsv"
done

python3 - "$RESULT" "$RESOURCE" "$LOGDIR/binaries.tsv" "$WINNER_ENV" "$ROWS" <<'PY'
import csv,math,shlex,statistics,sys
result,resource,bins_path,winner,rows_arg=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t'))
rr=list(csv.DictReader(open(resource),delimiter='\t'))
bins={int(r['nextself']):r for r in csv.DictReader(open(bins_path),delimiter='\t')}
if not rows: raise SystemExit('no hybrid8 next-self results')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL hybrid8 next-self partial residue mismatch '+repr({(r['nextself'],r['threads']):r['residue'] for r in rows}))
resource_rows={0:[],1:[]}
for r in rr:
    try: resource_rows[int(r['nextself'])].append((int(r['registers']),int(r['spill_store_bytes']),int(r['spill_load_bytes'])))
    except (ValueError,KeyError): pass
by={}
for r in rows:
    key=(int(r['nextself']),int(r['threads']))
    by.setdefault(key,[]).append(r)
agg=[]
for (ns,t),rs in by.items():
    wall=statistics.median(float(r['wall_s']) for r in rs)
    mv=[float(r['mc_avg_pct']) for r in rs if r['mc_avg_pct']!='nan']
    mc=statistics.median(mv) if mv else math.nan
    rv=resource_rows[ns]
    regs=max((x[0] for x in rv),default=-1);ss=max((x[1] for x in rv),default=-1);sl=max((x[2] for x in rv),default=-1)
    resource_ok=len(rv)>=2
    clean=resource_ok and ss==0 and sl==0
    agg.append((wall,ns,t,mc,regs,ss,sl,clean,len(rv)))
def rank(x): return (x[0],-x[3] if not math.isnan(x[3]) else math.inf)
for x in sorted(agg,key=rank):
    print(f'HYBRID8_NEXTSELF nextself={x[1]} threads={x[2]} wall_s={x[0]:.9f} mc_avg_pct={x[3]:.3f} regs_max={x[4]} spill_store={x[5]} spill_load={x[6]} spill_free={int(x[7])} resource_rows={x[8]}',file=sys.stderr)
control=min((x for x in agg if x[1]==0 and x[7]),default=None,key=rank)
test=min((x for x in agg if x[1]==1 and x[7]),default=None,key=rank)
if control is None: raise SystemExit('hybrid8 next-self control lacks known spill-free ILP2/ILP8 ptxas')
if test is None: raise SystemExit('hybrid8 next-self candidate lacks known spill-free ILP2/ILP8 ptxas')
speed=control[0]/test[0]
best=test if rank(test)<rank(control) else control
def q(x): return shlex.quote(str(x))
vals={
 'B300_HYBRID8_NEXTSELF_ROWS':rows_arg,
 'B300_HYBRID8_NEXTSELF_RESIDUE':next(iter(res)),
 'B300_HYBRID8_NEXTSELF_CONTROL_BIN':bins[0]['binary'],
 'B300_HYBRID8_NEXTSELF_CONTROL_THREADS':control[2],
 'B300_HYBRID8_NEXTSELF_CONTROL_WALL_S':f'{control[0]:.9f}',
 'B300_HYBRID8_NEXTSELF_BIN':bins[1]['binary'],
 'B300_HYBRID8_NEXTSELF_THREADS':test[2],
 'B300_HYBRID8_NEXTSELF_WALL_S':f'{test[0]:.9f}',
 'B300_HYBRID8_NEXTSELF_SPEEDUP':f'{speed:.9f}',
 'B300_HYBRID8_NEXTSELF_CONTROL_SPILL_FREE':1,
 'B300_HYBRID8_NEXTSELF_SPILL_FREE':1,
 'B300_HYBRID8_NEXTSELF_BEST_ENABLED':best[1],
 'B300_HYBRID8_NEXTSELF_BEST_BIN':bins[best[1]]['binary'],
 'B300_HYBRID8_NEXTSELF_BEST_THREADS':best[2],
 'B300_HYBRID8_NEXTSELF_BEST_WALL_S':f'{best[0]:.9f}',
}
with open(winner,'w') as f:
    for k,v in vals.items(): f.write(f'{k}={q(v)}\n')
print('b300_nextgen_hybrid8_nextself_exact_intermediate_match=1')
print(f'b300_nextgen_hybrid8_nextself_rows={rows_arg}')
print(f'b300_nextgen_hybrid8_nextself_residue={next(iter(res))}')
print(f'b300_nextgen_hybrid8_nextself_control_threads={control[2]}')
print(f'b300_nextgen_hybrid8_nextself_control_wall_s={control[0]:.9f}')
print(f'b300_nextgen_hybrid8_nextself_threads={test[2]}')
print(f'b300_nextgen_hybrid8_nextself_wall_s={test[0]:.9f}')
print(f'b300_nextgen_hybrid8_nextself_speedup={speed:.9f}x')
print(f'b300_nextgen_hybrid8_nextself_best_enabled={best[1]}')
print(f'b300_nextgen_hybrid8_nextself_winner_env={winner}')
PY

cat "$RESULT"
cat "$RESOURCE"
echo "b300-nextgen-hybrid8-nextself-ab OK rows=$ROWS result=$RESULT resources=$RESOURCE winner_env=$WINNER_ENV" >&2
