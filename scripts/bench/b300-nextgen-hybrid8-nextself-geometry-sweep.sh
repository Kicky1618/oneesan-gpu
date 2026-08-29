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
WIDTH_LIST="${WIDTH_LIST:-1 2 4 8}"
DISTANCE_LIST="${DISTANCE_LIST:-1 2 4}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
REPEATS="${REPEATS:-1}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
RANDOM_CG="${RANDOM_CG:-0}"
RANDOM_CG_L2_FETCH_BYTES="${RANDOM_CG_L2_FETCH_BYTES:-0}"
PREFETCH_L2="${PREFETCH_L2:-0}"
DUALMASK="${DUALMASK:-0}"
CLOSURE_BATCH="${CLOSURE_BATCH:-0}"
MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_geometry_t${HYBRID_THRESHOLD}_row${ROWS}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESULT="${RESULT:-${PREFIX}.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-next-self-prefetch.py"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")" "$(dirname "$WINNER_ENV")"

[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$HYBRID_THRESHOLD" =~ ^[0-9]+$ ]] || { echo 'HYBRID_THRESHOLD must be non-negative' >&2; exit 2; }
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

widths=()
for w in $WIDTH_LIST; do
  case "$w" in 1|2|4|8) ;; *) echo "bad width=$w" >&2; exit 2;; esac
  seen=0; for old in "${widths[@]}"; do [[ "$old" == "$w" ]] && seen=1; done
  ((seen)) || widths+=("$w")
done
distances=()
for d in $DISTANCE_LIST; do
  case "$d" in 1|2|4) ;; *) echo "bad distance=$d" >&2; exit 2;; esac
  seen=0; for old in "${distances[@]}"; do [[ "$old" == "$d" ]] && seen=1; done
  ((seen)) || distances+=("$d")
done
((${#widths[@]} && ${#distances[@]})) || { echo 'WIDTH_LIST and DISTANCE_LIST must be non-empty' >&2; exit 2; }
for t in $THREADS_LIST; do [[ "$t" =~ ^[0-9]+$ ]] && ((t>=32&&t<=1024&&t%32==0)) || { echo "bad threads=$t" >&2; exit 2; }; done
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

# Build the exact plain-hybrid control once. Every geometry candidate is derived
# from this same final CUDA source, so cache/block transforms and compile flags
# are held constant while only next-self geometry changes.
CONTROL_BIN="$ONEESAN_BUILD_DIR/b300_nextgen_hybrid8_geometry_control_t${HYBRID_THRESHOLD}_h${HIGH_DROP_CHUNK}_cg${RANDOM_CG}_l2${RANDOM_CG_L2_FETCH_BYTES}_pre${PREFETCH_L2}_d${DUALMASK}_b${CLOSURE_BATCH}_r${MAXRREGCOUNT}_n27"
CONTROL_ERR="$LOGDIR/control.build.err"
CONTROL_OUT="$LOGDIR/control.build.out"
N=27 ARCH="$ARCH" OUT="$CONTROL_BIN" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" RECURRENCE_ILP=2 \
  RECURRENCE_HYBRID_ILP8=1 RECURRENCE_HYBRID_ILP8_MIN_STATES="$HYBRID_THRESHOLD" RECURRENCE_HYBRID_ILP8_NEXTSELF=0 \
  RANDOM_CG="$RANDOM_CG" RANDOM_CG_L2_FETCH_BYTES="$RANDOM_CG_L2_FETCH_BYTES" PREFETCH_L2="$PREFETCH_L2" \
  DUALMASK="$DUALMASK" CLOSURE_BATCH="$CLOSURE_BATCH" MAXRREGCOUNT="$MAXRREGCOUNT" PTXAS_VERBOSE=1 BUILD_ERR="$CONTROL_ERR" \
  bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen.sh" >"$CONTROL_OUT" 2>"$LOGDIR/control.build.driver.err"
[[ -x "$CONTROL_BIN" ]] || { echo 'geometry control binary missing' >&2; exit 3; }
BASE_SRC="$(sed -nE 's/^  source_after_all=(.*)$/\1/p' "$CONTROL_OUT" | tail -n1)"
[[ -n "$BASE_SRC" && -f "$BASE_SRC" ]] || { echo 'geometry control source missing' >&2; exit 3; }
grep -Fq 'main_pull_kernel_ilp8_hybrid' "$BASE_SRC"

printf 'enabled\twidth\tdistance\tbinary\tbuild_err\n' >"$LOGDIR/binaries.tsv"
printf '0\t0\t0\t%s\t%s\n' "$CONTROL_BIN" "$CONTROL_ERR" >>"$LOGDIR/binaries.tsv"
PTXAS_FLAGS=(-Xptxas=-v)
REG_FLAGS=(); ((MAXRREGCOUNT>0)) && REG_FLAGS+=("-maxrregcount=$MAXRREGCOUNT")
DEFS=(-DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13)
[[ "$CLOSURE_BATCH" == 0 ]] || DEFS+=("-DB300_BLOCK_CLOSURE_BATCH=$CLOSURE_BATCH")
for w in "${widths[@]}"; do
  for d in "${distances[@]}"; do
    tag="w${w}d${d}"
    src="$LOGDIR/${tag}.cu"; tr="$LOGDIR/${tag}.transform.out"
    bin="$ONEESAN_BUILD_DIR/b300_nextgen_hybrid8_ns_${tag}_t${HYBRID_THRESHOLD}_h${HIGH_DROP_CHUNK}_cg${RANDOM_CG}_l2${RANDOM_CG_L2_FETCH_BYTES}_pre${PREFETCH_L2}_dual${DUALMASK}_cb${CLOSURE_BATCH}_r${MAXRREGCOUNT}_n27"
    err="$LOGDIR/${tag}.build.err"
    python3 "$GEN" "$BASE_SRC" "$src" "$w" "$d" >"$tr"
    grep -Fq "prefetch_width=$w" "$tr"; grep -Fq "prefetch_distance_iterations=$d" "$tr"
    : >"$err"
    TMPDIR="$LOGDIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" "${PTXAS_FLAGS[@]}" "${REG_FLAGS[@]}" "${DEFS[@]}" "$src" -o "$bin" 2>>"$err"
    [[ -x "$bin" ]] || { echo "geometry binary missing w=$w d=$d" >&2; exit 3; }
    printf '1\t%s\t%s\t%s\t%s\n' "$w" "$d" "$bin" "$err" >>"$LOGDIR/binaries.tsv"
  done
done

printf 'config\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
while IFS=$'\t' read -r enabled w d bin err; do
  [[ "$enabled" == enabled ]] && continue
  label="${enabled}:${w}:${d}"
  python3 "$PARSER" "$err" --label "$label" --contains main_pull_kernel_ilp2 >>"$RESOURCE" || true
  python3 "$PARSER" "$err" --label "$label" --contains main_pull_kernel_ilp8_hybrid >>"$RESOURCE" || true
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
printf 'enabled\twidth\tdistance\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tforward_high_s\treverse_high_s\thigh_rec_groups\thigh_rec_fallback_groups\tmc_avg_pct\tmc_max_pct\tmc_samples\n' >"$RESULT"
run_one(){
  local enabled="$1" w="$2" d="$3" bin="$4" t="$5" r="$6" tag="e${1}_w${2}_d${3}_t${5}_r${6}"
  local so="$LOGDIR/${tag}.out" se="$LOGDIR/${tag}.err" mem="$LOGDIR/${tag}.mem"
  set +e
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$t" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se" &
  local pid=$!; sample_mem "$pid" "$mem" & local mpid=$!
  wait "$pid"; local rc=$?; set -e
  wait "$mpid" 2>/dev/null || true
  ((rc==0)) || { echo "$tag failed rc=$rc" >&2; tail -n 80 "$se" >&2 || true; return "$rc"; }
  local line hg hf ma mm mn
  line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$tag result missing" >&2; return 4; }
  hg="$(field high_rec_groups "$line")"; hf="$(field high_rec_fallback_groups "$line")"
  [[ "$hg" =~ ^[0-9]+$ ]] && ((hg>0)) || { echo "$tag hybrid path inactive" >&2; return 5; }
  read -r ma mm mn < <(awk '{sa+=$1;if($2>mm)mm=$2;n++}END{if(n)printf "%.3f %.3f %d\n",sa/n,mm,n;else print "nan nan 0"}' "$mem")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$enabled" "$w" "$d" "$t" "$r" "$(field residue "$line")" "$(field wall_s "$line")" "$(field active_max_s "$line")" "$(field active_sum_s "$line")" \
    "$(field forward_high_s "$line")" "$(field reverse_high_s "$line")" "$hg" "$hf" "$ma" "$mm" "$mn" >>"$RESULT"
}
for t in $THREADS_LIST; do
  while IFS=$'\t' read -r enabled w d bin err; do
    [[ "$enabled" == enabled ]] && continue
    for ((r=1;r<=REPEATS;++r)); do
      echo "=== geometry enabled=$enabled width=$w distance=$d threads=$t repeat=$r rows=$ROWS ===" >&2
      run_one "$enabled" "$w" "$d" "$bin" "$t" "$r"
    done
  done <"$LOGDIR/binaries.tsv"
done

python3 - "$RESULT" "$RESOURCE" "$LOGDIR/binaries.tsv" "$WINNER_ENV" "$ROWS" <<'PY'
import csv,math,shlex,statistics,sys
result,resource,bins_path,winner,rows_arg=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t'))
rr=list(csv.DictReader(open(resource),delimiter='\t'))
bins={(int(r['enabled']),int(r['width']),int(r['distance'])):r for r in csv.DictReader(open(bins_path),delimiter='\t')}
if not rows: raise SystemExit('no geometry results')
res={r['residue'] for r in rows}
if len(res)!=1:
    raise SystemExit('FATAL geometry residue mismatch '+repr({(r['enabled'],r['width'],r['distance'],r['threads']):r['residue'] for r in rows}))
resources={k:[] for k in bins}
for r in rr:
    try:
        e,w,d=map(int,r['config'].split(':'))
        resources.setdefault((e,w,d),[]).append((int(r['registers']),int(r['spill_store_bytes']),int(r['spill_load_bytes'])))
    except (ValueError,KeyError): pass
by={}
for r in rows:
    key=(int(r['enabled']),int(r['width']),int(r['distance']),int(r['threads']))
    by.setdefault(key,[]).append(r)
agg=[]
for (e,w,d,t),rs in by.items():
    wall=statistics.median(float(r['wall_s']) for r in rs)
    mv=[float(r['mc_avg_pct']) for r in rs if r['mc_avg_pct']!='nan']
    mc=statistics.median(mv) if mv else math.nan
    hs=[]
    for r in rs:
        try: hs.append(float(r['forward_high_s'])+float(r['reverse_high_s']))
        except (TypeError,ValueError): pass
    high=statistics.median(hs) if hs else math.nan
    rv=resources.get((e,w,d),[])
    regs=max((x[0] for x in rv),default=-1); ss=max((x[1] for x in rv),default=-1); sl=max((x[2] for x in rv),default=-1)
    clean=len(rv)>=2 and ss==0 and sl==0
    agg.append((wall,e,w,d,t,mc,high,regs,ss,sl,clean,len(rv)))
def finite(v,default): return v if not math.isnan(v) else default
def rank(x): return (x[0],-finite(x[5],-math.inf),finite(x[6],math.inf),x[2],x[3],x[4])
for x in sorted(agg,key=rank):
    print(f'HYBRID8_NEXTSELF_GEOMETRY enabled={x[1]} width={x[2]} distance={x[3]} threads={x[4]} wall_s={x[0]:.9f} mc_avg_pct={x[5]:.3f} high_s={x[6]:.9f} regs={x[7]} spill_store={x[8]} spill_load={x[9]} spill_free={int(x[10])} resource_rows={x[11]}',file=sys.stderr)
control=min((x for x in agg if x[1]==0 and x[10]),default=None,key=rank)
tests=[x for x in agg if x[1]==1 and x[10]]
if control is None: raise SystemExit('geometry control lacks known spill-free ILP2/ILP8 ptxas')
if not tests: raise SystemExit('geometry candidates lack known spill-free ILP2/ILP8 ptxas')
test=min(tests,key=rank); speed=control[0]/test[0]
best=test if rank(test)<rank(control) else control
q=lambda x:shlex.quote(str(x))
vals={
 'B300_HYBRID8_NEXTSELF_ROWS':rows_arg,
 'B300_HYBRID8_NEXTSELF_RESIDUE':next(iter(res)),
 'B300_HYBRID8_NEXTSELF_CONTROL_BIN':bins[(0,0,0)]['binary'],
 'B300_HYBRID8_NEXTSELF_CONTROL_THREADS':control[4],
 'B300_HYBRID8_NEXTSELF_CONTROL_WALL_S':f'{control[0]:.9f}',
 'B300_HYBRID8_NEXTSELF_BIN':bins[(1,test[2],test[3])]['binary'],
 'B300_HYBRID8_NEXTSELF_WIDTH':test[2],
 'B300_HYBRID8_NEXTSELF_DISTANCE':test[3],
 'B300_HYBRID8_NEXTSELF_THREADS':test[4],
 'B300_HYBRID8_NEXTSELF_WALL_S':f'{test[0]:.9f}',
 'B300_HYBRID8_NEXTSELF_SPEEDUP':f'{speed:.9f}',
 'B300_HYBRID8_NEXTSELF_CONTROL_SPILL_FREE':1,
 'B300_HYBRID8_NEXTSELF_SPILL_FREE':1,
 'B300_HYBRID8_NEXTSELF_BEST_ENABLED':best[1],
 'B300_HYBRID8_NEXTSELF_BEST_WIDTH':best[2] if best[1] else 0,
 'B300_HYBRID8_NEXTSELF_BEST_DISTANCE':best[3] if best[1] else 0,
 'B300_HYBRID8_NEXTSELF_BEST_BIN':bins[(best[1],best[2],best[3])]['binary'],
 'B300_HYBRID8_NEXTSELF_BEST_THREADS':best[4],
 'B300_HYBRID8_NEXTSELF_BEST_WALL_S':f'{best[0]:.9f}',
}
with open(winner,'w') as f:
    for k,v in vals.items(): f.write(k+'='+q(v)+'\n')
print('b300_nextgen_hybrid8_nextself_exact_intermediate_match=1')
print('b300_nextgen_hybrid8_nextself_geometry_sweep=1')
print(f'b300_nextgen_hybrid8_nextself_residue={next(iter(res))}')
print(f'b300_nextgen_hybrid8_nextself_width={test[2]}')
print(f'b300_nextgen_hybrid8_nextself_distance={test[3]}')
print(f'b300_nextgen_hybrid8_nextself_speedup={speed:.9f}x')
print(f'b300_nextgen_hybrid8_nextself_best_enabled={best[1]}')
print(f'b300_nextgen_hybrid8_nextself_best_width={best[2] if best[1] else 0}')
print(f'b300_nextgen_hybrid8_nextself_best_distance={best[3] if best[1] else 0}')
print(f'b300_nextgen_hybrid8_nextself_winner_env={winner}')
PY

cat "$RESULT"
cat "$RESOURCE"
echo "b300-nextgen-hybrid8-nextself-geometry-sweep OK widths=${widths[*]} distances=${distances[*]} rows=$ROWS winner_env=$WINNER_ENV" >&2
