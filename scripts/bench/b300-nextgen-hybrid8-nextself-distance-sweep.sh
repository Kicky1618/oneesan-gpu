#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; ROWS="${ROWS:-1}"
TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"; HYBRID_THRESHOLD="${HYBRID_THRESHOLD:-1048576}"
NEXTSELF_WIDTH="${NEXTSELF_WIDTH:-8}"; DISTANCE_LIST="${DISTANCE_LIST:-1 2 4}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"; REPEATS="${REPEATS:-1}"
RANDOM_CG="${RANDOM_CG:-0}"; RANDOM_CG_L2_FETCH_BYTES="${RANDOM_CG_L2_FETCH_BYTES:-0}"; PREFETCH_L2="${PREFETCH_L2:-0}"
DUALMASK="${DUALMASK:-0}"; CLOSURE_BATCH="${CLOSURE_BATCH:-0}"; MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_distance_w${NEXTSELF_WIDTH}_t${HYBRID_THRESHOLD}_row${ROWS}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")" "$(dirname "$WINNER_ENV")"
case "$NEXTSELF_WIDTH" in 1|2|4|8) ;; *) echo 'NEXTSELF_WIDTH must be 1,2,4,8' >&2; exit 2;; esac
[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || exit 2
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || exit 2
distances=(); for d in $DISTANCE_LIST; do case "$d" in 1|2|4);;*) echo "bad distance=$d" >&2; exit 2;; esac; seen=0; for old in "${distances[@]}"; do [[ "$old" == "$d" ]] && seen=1; done; ((seen)) || distances+=("$d"); done
((${#distances[@]})) || exit 2
has1=0; for d in "${distances[@]}"; do [[ "$d" == 1 ]] && has1=1; done; ((has1)) || { echo 'DISTANCE_LIST must include 1 reference' >&2; exit 2; }
command -v nvcc >/dev/null || exit 2
command -v nvidia-smi >/dev/null || exit 2
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

printf 'distance\tbinary\tbuild_err\n' >"$LOGDIR/binaries.tsv"
for d in "${distances[@]}"; do
  bin="$ONEESAN_BUILD_DIR/b300_nextgen_hybrid8_ns_w${NEXTSELF_WIDTH}_d${d}_t${HYBRID_THRESHOLD}_h${HIGH_DROP_CHUNK}_cg${RANDOM_CG}_l2${RANDOM_CG_L2_FETCH_BYTES}_pre${PREFETCH_L2}_dual${DUALMASK}_cb${CLOSURE_BATCH}_r${MAXRREGCOUNT}_n27"
  err="$LOGDIR/d${d}.build.err"; out="$LOGDIR/d${d}.build.out"
  N=27 ARCH="$ARCH" OUT="$bin" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" HYBRID_THRESHOLD="$HYBRID_THRESHOLD" NEXTSELF_WIDTH="$NEXTSELF_WIDTH" NEXTSELF_DISTANCE="$d" \
    RANDOM_CG="$RANDOM_CG" RANDOM_CG_L2_FETCH_BYTES="$RANDOM_CG_L2_FETCH_BYTES" PREFETCH_L2="$PREFETCH_L2" DUALMASK="$DUALMASK" CLOSURE_BATCH="$CLOSURE_BATCH" MAXRREGCOUNT="$MAXRREGCOUNT" \
    BUILD_ERR="$err" PTXAS_VERBOSE=1 bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-nextself-distance.sh" >"$out" 2>"$LOGDIR/d${d}.driver.err"
  [[ -x "$bin" ]] || exit 3
  grep -Fq "recurrence_hybrid_ilp8_nextself_width=$NEXTSELF_WIDTH recurrence_hybrid_ilp8_nextself_distance=$d" "$out" || exit 3
  printf '%s\t%s\t%s\n' "$d" "$bin" "$err" >>"$LOGDIR/binaries.tsv"
done

printf 'distance\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
while IFS=$'\t' read -r d bin err; do [[ "$d" == distance ]] && continue; python3 "$PARSER" "$err" --label "$d" --contains main_pull_kernel_ilp2 >>"$RESOURCE" || true; python3 "$PARSER" "$err" --label "$d" --contains main_pull_kernel_ilp8_hybrid >>"$RESOURCE" || true; done <"$LOGDIR/binaries.tsv"
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
printf 'distance\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tforward_high_s\treverse_high_s\n' >"$RESULT"
run_one(){
  local d="$1" bin="$2" t="$3" r="$4" so="$LOGDIR/d${1}_t${3}_r${4}.out" se="$LOGDIR/d${1}_t${3}_r${4}.err"
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$t" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se"
  line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { tail -n 80 "$se" >&2 || true; return 4; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$d" "$t" "$r" "$(field residue "$line")" "$(field wall_s "$line")" "$(field active_max_s "$line")" "$(field active_sum_s "$line")" "$(field forward_high_s "$line")" "$(field reverse_high_s "$line")" >>"$RESULT"
}
for t in $THREADS_LIST; do [[ "$t" =~ ^[0-9]+$ ]] && ((t>=32&&t<=1024&&t%32==0)) || exit 2; while IFS=$'\t' read -r d bin err; do [[ "$d" == distance ]] && continue; for ((r=1;r<=REPEATS;++r)); do echo "=== Stage-G distance=$d width=$NEXTSELF_WIDTH threads=$t repeat=$r rows=$ROWS ===" >&2; run_one "$d" "$bin" "$t" "$r"; done; done <"$LOGDIR/binaries.tsv"; done

python3 - "$RESULT" "$RESOURCE" "$LOGDIR/binaries.tsv" "$WINNER_ENV" "$ROWS" "$NEXTSELF_WIDTH" <<'PY'
import csv,statistics,sys,shlex,math
result,resource,bins_path,winner,rows_arg,width_arg=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t')); rr=list(csv.DictReader(open(resource),delimiter='\t')); bins={int(r['distance']):r for r in csv.DictReader(open(bins_path),delimiter='\t')}
if not rows: raise SystemExit('no distance sweep rows')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL distance residue mismatch '+repr({(r['distance'],r['threads'],r['repeat']):r['residue'] for r in rows}))
resources={d:[] for d in bins}
for r in rr:
 try: resources[int(r['distance'])].append((int(r['registers']),int(r['spill_store_bytes']),int(r['spill_load_bytes'])))
 except (ValueError,KeyError): pass
agg=[]
for d,t in {(int(r['distance']),int(r['threads'])) for r in rows}:
 rs=[r for r in rows if int(r['distance'])==d and int(r['threads'])==t]; wall=statistics.median(float(r['wall_s']) for r in rs)
 hs=[]
 for r in rs:
  try: hs.append(float(r['forward_high_s'])+float(r['reverse_high_s']))
  except ValueError: pass
 high=statistics.median(hs) if hs else math.nan
 rv=resources.get(d,[]); regs=max((x[0] for x in rv),default=-1); ss=max((x[1] for x in rv),default=-1); sl=max((x[2] for x in rv),default=-1); clean=len(rv)>=2 and ss==0 and sl==0
 agg.append((wall,d,t,high,regs,ss,sl,clean,len(rv)))
def rank(x): return (x[0],x[3] if not math.isnan(x[3]) else math.inf,x[1],x[2])
for x in sorted(agg,key=rank): print(f'HYBRID8_NEXTSELF_DISTANCE distance={x[1]} width={width_arg} threads={x[2]} wall_s={x[0]:.9f} high_s={x[3]:.9f} regs={x[4]} spill_store={x[5]} spill_load={x[6]} spill_free={int(x[7])}',file=sys.stderr)
ref=min((x for x in agg if x[1]==1 and x[7]),default=None,key=rank); tests=[x for x in agg if x[7]]
if ref is None or not tests: raise SystemExit('missing spill-free distance=1 reference/candidate')
best=min(tests,key=rank); speed=ref[0]/best[0]; q=lambda x:shlex.quote(str(x))
vals={'B300_HYBRID8_NEXTSELF_DISTANCE_ROWS':rows_arg,'B300_HYBRID8_NEXTSELF_DISTANCE_RESIDUE':next(iter(res)),'B300_HYBRID8_NEXTSELF_DISTANCE_WIDTH':int(width_arg),'B300_HYBRID8_NEXTSELF_DISTANCE_REFERENCE_BIN':bins[1]['binary'],'B300_HYBRID8_NEXTSELF_DISTANCE_REFERENCE_THREADS':ref[2],'B300_HYBRID8_NEXTSELF_DISTANCE_REFERENCE_WALL_S':f'{ref[0]:.9f}','B300_HYBRID8_NEXTSELF_DISTANCE_BEST':best[1],'B300_HYBRID8_NEXTSELF_DISTANCE_BIN':bins[best[1]]['binary'],'B300_HYBRID8_NEXTSELF_DISTANCE_THREADS':best[2],'B300_HYBRID8_NEXTSELF_DISTANCE_WALL_S':f'{best[0]:.9f}','B300_HYBRID8_NEXTSELF_DISTANCE_SPEEDUP_VS_D1':f'{speed:.9f}','B300_HYBRID8_NEXTSELF_DISTANCE_REFERENCE_SPILL_FREE':1,'B300_HYBRID8_NEXTSELF_DISTANCE_SPILL_FREE':1}
with open(winner,'w') as f:
 for k,v in vals.items(): f.write(k+'='+q(v)+'\n')
print('b300_nextgen_hybrid8_nextself_distance_exact_match=1'); print('b300_nextgen_hybrid8_nextself_distance_sweep=1'); print('b300_nextgen_hybrid8_nextself_distance_residue='+next(iter(res))); print(f'b300_nextgen_hybrid8_nextself_distance_best={best[1]}'); print(f'b300_nextgen_hybrid8_nextself_distance_speedup_vs_d1={speed:.9f}x')
PY
cat "$RESULT"; cat "$RESOURCE"
echo "b300-nextgen-hybrid8-nextself-distance-sweep OK width=$NEXTSELF_WIDTH distances=${distances[*]} rows=$ROWS winner_env=$WINNER_ENV" >&2
