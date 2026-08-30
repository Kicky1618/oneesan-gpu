#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
INPUT_ENV="${INPUT_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; ROWS="${ROWS:-1}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"
EVICT_LIST="${EVICT_LIST:-default normal last}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"; REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_evict_row${ROWS}}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"; mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")" "$(dirname "$WINNER_ENV")"
[[ -s "$INPUT_ENV" ]] || exit 2; [[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || exit 2; [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || exit 2
# shellcheck disable=SC1090
source "$INPUT_ENV"
for k in B300_HYBRID8_NEXTSELF_STAGED_VALIDATED B300_HYBRID8_NEXTSELF_FINAL_ENABLED B300_HYBRID8_NEXTSELF_FINAL_WIDTH B300_HYBRID8_NEXTSELF_FINAL_DISTANCE B300_HYBRID8_NEXTSELF_THRESHOLD B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK B300_HYBRID8_NEXTSELF_RANDOM_CG B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES B300_HYBRID8_NEXTSELF_PREFETCH_L2 B300_HYBRID8_NEXTSELF_DUALMASK B300_HYBRID8_NEXTSELF_CLOSURE_BATCH B300_HYBRID8_NEXTSELF_MAXRREGCOUNT; do [[ -n "${!k+x}" ]] || { echo "Stage-F env missing $k" >&2; exit 3; }; done
[[ "$B300_HYBRID8_NEXTSELF_STAGED_VALIDATED" == 1 && "$B300_HYBRID8_NEXTSELF_FINAL_ENABLED" == 1 ]] || exit 4
W="$B300_HYBRID8_NEXTSELF_FINAL_WIDTH"; D="$B300_HYBRID8_NEXTSELF_FINAL_DISTANCE"; T="$B300_HYBRID8_NEXTSELF_THRESHOLD"; H="$B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK"; CG="$B300_HYBRID8_NEXTSELF_RANDOM_CG"; CGL2="$B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES"; PRE="$B300_HYBRID8_NEXTSELF_PREFETCH_L2"; DUAL="$B300_HYBRID8_NEXTSELF_DUALMASK"; BATCH="$B300_HYBRID8_NEXTSELF_CLOSURE_BATCH"; CAP="$B300_HYBRID8_NEXTSELF_MAXRREGCOUNT"
evicts=(); for ev in $EVICT_LIST; do case "$ev" in default|normal|last);;*) echo "bad evict=$ev" >&2; exit 2;; esac; seen=0; for old in "${evicts[@]}"; do [[ "$old" == "$ev" ]] && seen=1; done; ((seen)) || evicts+=("$ev"); done
has_default=0; for ev in "${evicts[@]}"; do [[ "$ev" == default ]] && has_default=1; done; ((has_default)) || { echo 'EVICT_LIST must include default' >&2; exit 2; }
for th in $THREADS_LIST; do [[ "$th" =~ ^[0-9]+$ ]] && ((th>=32&&th<=1024&&th%32==0)) || exit 2; done
command -v nvcc >/dev/null || exit 2; command -v nvidia-smi >/dev/null || exit 2; (( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }
printf 'evict\tbinary\tbuild_err\n' >"$LOGDIR/binaries.tsv"
for ev in "${evicts[@]}"; do
  bin="$ONEESAN_BUILD_DIR/b300_evict_${ev}_w${W}_d${D}_t${T}_n27"; err="$LOGDIR/${ev}.build.err"; out="$LOGDIR/${ev}.build.out"
  N=27 ARCH="$ARCH" OUT="$bin" HIGH_DROP_CHUNK="$H" HYBRID_THRESHOLD="$T" NEXTSELF_WIDTH="$W" NEXTSELF_DISTANCE="$D" NEXTSELF_EVICT="$ev" RANDOM_CG="$CG" RANDOM_CG_L2_FETCH_BYTES="$CGL2" PREFETCH_L2="$PRE" DUALMASK="$DUAL" CLOSURE_BATCH="$BATCH" MAXRREGCOUNT="$CAP" PTXAS_VERBOSE=1 BUILD_ERR="$err" bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-nextself-distance.sh" >"$out" 2>"$LOGDIR/${ev}.driver.err"
  [[ -x "$bin" ]] || exit 3; grep -Fq "recurrence_hybrid_ilp8_nextself_evict=$ev" "$out" || exit 3
  printf '%s\t%s\t%s\n' "$ev" "$bin" "$err" >>"$LOGDIR/binaries.tsv"
done
printf 'evict\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
while IFS=$'\t' read -r ev bin err; do [[ "$ev" == evict ]] && continue; python3 "$PARSER" "$err" --label "$ev" --contains main_pull_kernel_ilp2 >>"$RESOURCE" || true; python3 "$PARSER" "$err" --label "$ev" --contains main_pull_kernel_ilp8_hybrid >>"$RESOURCE" || true; done <"$LOGDIR/binaries.tsv"
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
printf 'evict\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tforward_high_s\treverse_high_s\n' >"$RESULT"
run_one(){ local ev="$1" bin="$2" th="$3" r="$4" so="$LOGDIR/${ev}_t${th}_r${r}.out" se="$LOGDIR/${ev}_t${th}_r${r}.err"; B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$th" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se"; line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || return 4; printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ev" "$th" "$r" "$(field residue "$line")" "$(field wall_s "$line")" "$(field active_max_s "$line")" "$(field active_sum_s "$line")" "$(field forward_high_s "$line")" "$(field reverse_high_s "$line")" >>"$RESULT"; }
for th in $THREADS_LIST; do while IFS=$'\t' read -r ev bin err; do [[ "$ev" == evict ]] && continue; for ((r=1;r<=REPEATS;++r)); do echo "=== eviction=$ev threads=$th repeat=$r rows=$ROWS w=$W d=$D ===" >&2; run_one "$ev" "$bin" "$th" "$r"; done; done <"$LOGDIR/binaries.tsv"; done
python3 - "$RESULT" "$RESOURCE" "$LOGDIR/binaries.tsv" "$WINNER_ENV" "$ROWS" "$W" "$D" <<'PY'
import csv,math,statistics,sys,shlex
result,resource,bins_path,winner,rows_arg,w,d=sys.argv[1:]; rows=list(csv.DictReader(open(result),delimiter='\t')); rr=list(csv.DictReader(open(resource),delimiter='\t')); bins={r['evict']:r for r in csv.DictReader(open(bins_path),delimiter='\t')}
if not rows: raise SystemExit('no eviction rows')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL eviction residue mismatch '+repr({(r['evict'],r['threads'],r['repeat']):r['residue'] for r in rows}))
resources={k:[] for k in bins}
for r in rr:
 try: resources[r['evict']].append((int(r['registers']),int(r['spill_store_bytes']),int(r['spill_load_bytes'])))
 except (ValueError,KeyError): pass
agg=[]
for ev,t in {(r['evict'],int(r['threads'])) for r in rows}:
 rs=[r for r in rows if r['evict']==ev and int(r['threads'])==t]; wall=statistics.median(float(r['wall_s']) for r in rs); hs=[]
 for r in rs:
  try: hs.append(float(r['forward_high_s'])+float(r['reverse_high_s']))
  except ValueError: pass
 high=statistics.median(hs) if hs else math.nan; rv=resources[ev]; regs=max((x[0] for x in rv),default=-1); ss=max((x[1] for x in rv),default=-1); sl=max((x[2] for x in rv),default=-1); clean=len(rv)>=2 and ss==0 and sl==0; agg.append((wall,ev,t,high,regs,ss,sl,clean))
def rank(x): return (x[0],x[3] if not math.isnan(x[3]) else math.inf,x[4],x[2])
for x in sorted(agg,key=rank): print(f'EVICT evict={x[1]} threads={x[2]} wall_s={x[0]:.9f} high_s={x[3]:.9f} regs={x[4]} spill_store={x[5]} spill_load={x[6]} spill_free={int(x[7])}',file=sys.stderr)
base=min((x for x in agg if x[1]=='default' and x[7]),default=None,key=rank); tests=[x for x in agg if x[1]!='default' and x[7]]
if base is None or not tests: raise SystemExit('eviction sweep needs spill-free default and hinted candidate')
test=min(tests,key=rank); speed=base[0]/test[0]; enabled=int(rank(test)<rank(base)); q=lambda x:shlex.quote(str(x))
vals={'B300_EVICT_ROWS':rows_arg,'B300_EVICT_RESIDUE':next(iter(res)),'B300_EVICT_WIDTH':int(w),'B300_EVICT_DISTANCE':int(d),'B300_EVICT_DEFAULT_BIN':bins['default']['binary'],'B300_EVICT_DEFAULT_THREADS':base[2],'B300_EVICT_DEFAULT_WALL_S':f'{base[0]:.9f}','B300_EVICT_DEFAULT_HIGH_S':f'{base[3]:.9f}','B300_EVICT_DEFAULT_SPILL_FREE':1,'B300_EVICT_HINT':test[1],'B300_EVICT_BIN':bins[test[1]]['binary'],'B300_EVICT_THREADS':test[2],'B300_EVICT_WALL_S':f'{test[0]:.9f}','B300_EVICT_HIGH_S':f'{test[3]:.9f}','B300_EVICT_SPILL_FREE':1,'B300_EVICT_SPEEDUP':f'{speed:.9f}','B300_EVICT_BEST_ENABLED':enabled}
with open(winner,'w') as f:
 for k,v in vals.items(): f.write(k+'='+q(v)+'\n')
print('b300_evict_exact_match=1'); print(f'b300_evict_hint={test[1]}'); print(f'b300_evict_speedup={speed:.9f}x'); print(f'b300_evict_best_enabled={enabled}')
PY
cat "$RESULT"; cat "$RESOURCE"
echo "b300-nextgen-hybrid8-nextself-evict-sweep OK rows=$ROWS width=$W distance=$D hints=${evicts[*]} winner_env=$WINNER_ENV" >&2
