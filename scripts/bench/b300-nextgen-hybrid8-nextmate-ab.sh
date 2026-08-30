#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

INPUT_ENV="${INPUT_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
SELF_EVICT="${SELF_EVICT:-default}"
ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; ROWS="${ROWS:-1}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"; REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextmate_row${ROWS}}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"; mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")" "$(dirname "$WINNER_ENV")"
[[ -s "$INPUT_ENV" ]] || { echo "missing INPUT_ENV=$INPUT_ENV" >&2; exit 2; }
[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || exit 2
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || exit 2
case "$SELF_EVICT" in default|normal|last) ;; *) echo "SELF_EVICT must be default,normal,last" >&2; exit 2;; esac
# shellcheck disable=SC1090
source "$INPUT_ENV"
for k in B300_HYBRID8_NEXTSELF_STAGED_VALIDATED B300_HYBRID8_NEXTSELF_FINAL_ENABLED B300_HYBRID8_NEXTSELF_FINAL_WIDTH B300_HYBRID8_NEXTSELF_FINAL_DISTANCE \
 B300_HYBRID8_NEXTSELF_THRESHOLD B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK B300_HYBRID8_NEXTSELF_RANDOM_CG B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES \
 B300_HYBRID8_NEXTSELF_PREFETCH_L2 B300_HYBRID8_NEXTSELF_DUALMASK B300_HYBRID8_NEXTSELF_CLOSURE_BATCH B300_HYBRID8_NEXTSELF_MAXRREGCOUNT; do [[ -n "${!k+x}" ]] || { echo "input Stage-F env missing $k" >&2; exit 3; }; done
[[ "$B300_HYBRID8_NEXTSELF_STAGED_VALIDATED" == 1 && "$B300_HYBRID8_NEXTSELF_FINAL_ENABLED" == 1 ]] || { echo 'Stage-F geometry not promotable' >&2; exit 4; }
W="$B300_HYBRID8_NEXTSELF_FINAL_WIDTH"; D="$B300_HYBRID8_NEXTSELF_FINAL_DISTANCE"; T="$B300_HYBRID8_NEXTSELF_THRESHOLD"; H="$B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK"; CG="$B300_HYBRID8_NEXTSELF_RANDOM_CG"; CGL2="$B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES"; PRE="$B300_HYBRID8_NEXTSELF_PREFETCH_L2"; DUAL="$B300_HYBRID8_NEXTSELF_DUALMASK"; BATCH="$B300_HYBRID8_NEXTSELF_CLOSURE_BATCH"; CAP="$B300_HYBRID8_NEXTSELF_MAXRREGCOUNT"
case "$W" in 1|2|4|8);;*) exit 3;; esac; case "$D" in 1|2|4);;*) exit 3;; esac
for th in $THREADS_LIST; do [[ "$th" =~ ^[0-9]+$ ]] && ((th>=32&&th<=1024&&th%32==0)) || exit 2; done
command -v nvcc >/dev/null || exit 2; command -v nvidia-smi >/dev/null || exit 2; (( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

SELF_BIN="$ONEESAN_BUILD_DIR/b300_stageh_self_w${W}_d${D}_ev${SELF_EVICT}_t${T}_n27"; SELF_ERR="$LOGDIR/self.build.err"; SELF_OUT="$LOGDIR/self.build.out"
MATE_BIN="$ONEESAN_BUILD_DIR/b300_stageh_mate_w${W}_d${D}_sev${SELF_EVICT}_t${T}_n27"; MATE_ERR="$LOGDIR/mate.build.err"; MATE_OUT="$LOGDIR/mate.build.out"
COMMON=(N=27 ARCH="$ARCH" HIGH_DROP_CHUNK="$H" HYBRID_THRESHOLD="$T" RANDOM_CG="$CG" RANDOM_CG_L2_FETCH_BYTES="$CGL2" PREFETCH_L2="$PRE" DUALMASK="$DUAL" CLOSURE_BATCH="$BATCH" MAXRREGCOUNT="$CAP" PTXAS_VERBOSE=1)
env "${COMMON[@]}" NEXTSELF_WIDTH="$W" NEXTSELF_DISTANCE="$D" NEXTSELF_EVICT="$SELF_EVICT" OUT="$SELF_BIN" BUILD_ERR="$SELF_ERR" \
  bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-nextself-distance.sh" >"$SELF_OUT" 2>"$LOGDIR/self.driver.err"
env "${COMMON[@]}" SELF_WIDTH="$W" SELF_DISTANCE="$D" MATE_WIDTH="$W" MATE_DISTANCE="$D" SELF_EVICT="$SELF_EVICT" MATE_EVICT=default OUT="$MATE_BIN" BUILD_ERR="$MATE_ERR" \
  bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-self-mate-geometry.sh" >"$MATE_OUT" 2>"$LOGDIR/mate.driver.err"
[[ -x "$SELF_BIN" && -x "$MATE_BIN" ]] || exit 3
grep -Fq "recurrence_hybrid_ilp8_nextself_width=$W recurrence_hybrid_ilp8_nextself_distance=$D" "$SELF_OUT" || exit 3
grep -Fq "recurrence_hybrid_ilp8_nextself_evict=$SELF_EVICT" "$SELF_OUT" || exit 3
grep -Fq "self_geometry width=$W distance=$D evict=$SELF_EVICT" "$MATE_OUT" || exit 3
grep -Fq "mate_geometry width=$W distance=$D evict=default" "$MATE_OUT" || exit 3
grep -Fq 'next_mate_transform=1' "$MATE_OUT" || exit 3

printf 'profile\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
for pair in "self:$SELF_ERR" "mate:$MATE_ERR"; do name="${pair%%:*}"; err="${pair#*:}"; python3 "$PARSER" "$err" --label "$name" --contains main_pull_kernel_ilp2 >>"$RESOURCE" || true; python3 "$PARSER" "$err" --label "$name" --contains main_pull_kernel_ilp8_hybrid >>"$RESOURCE" || true; done
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
printf 'profile\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tforward_high_s\treverse_high_s\n' >"$RESULT"
run_one(){ local name="$1" bin="$2" th="$3" r="$4" so="$LOGDIR/${name}_t${th}_r${r}.out" se="$LOGDIR/${name}_t${th}_r${r}.err"; B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$th" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se"; line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { tail -n 80 "$se" >&2 || true; return 4; }; printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$th" "$r" "$(field residue "$line")" "$(field wall_s "$line")" "$(field active_max_s "$line")" "$(field active_sum_s "$line")" "$(field forward_high_s "$line")" "$(field reverse_high_s "$line")" >>"$RESULT"; }
for th in $THREADS_LIST; do for ((r=1;r<=REPEATS;++r)); do echo "=== Stage H self t=$th r=$r rows=$ROWS w=$W d=$D evict=$SELF_EVICT ===" >&2; run_one self "$SELF_BIN" "$th" "$r"; done; for ((r=1;r<=REPEATS;++r)); do echo "=== Stage H mate t=$th r=$r rows=$ROWS w=$W d=$D self_evict=$SELF_EVICT ===" >&2; run_one mate "$MATE_BIN" "$th" "$r"; done; done

python3 - "$RESULT" "$RESOURCE" "$WINNER_ENV" "$SELF_BIN" "$MATE_BIN" "$ROWS" "$W" "$D" "$SELF_EVICT" <<'PY'
import csv,math,statistics,sys,shlex
result,resource,winner,selfbin,matebin,rows_arg,w,d,self_evict=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t')); rr=list(csv.DictReader(open(resource),delimiter='\t'))
if not rows: raise SystemExit('no Stage-H rows')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL Stage-H residue mismatch '+repr({(r['profile'],r['threads'],r['repeat']):r['residue'] for r in rows}))
resources={'self':[],'mate':[]}
for r in rr:
 try: resources[r['profile']].append((int(r['registers']),int(r['spill_store_bytes']),int(r['spill_load_bytes'])))
 except (ValueError,KeyError): pass
agg=[]
for name,t in {(r['profile'],int(r['threads'])) for r in rows}:
 rs=[r for r in rows if r['profile']==name and int(r['threads'])==t]; wall=statistics.median(float(r['wall_s']) for r in rs); hs=[]
 for r in rs:
  try: hs.append(float(r['forward_high_s'])+float(r['reverse_high_s']))
  except ValueError: pass
 high=statistics.median(hs) if hs else math.nan; rv=resources[name]; regs=max((x[0] for x in rv),default=-1); ss=max((x[1] for x in rv),default=-1); sl=max((x[2] for x in rv),default=-1); clean=len(rv)>=2 and ss==0 and sl==0; agg.append((wall,name,t,high,regs,ss,sl,clean,len(rv)))
def rank(x): return (x[0],x[3] if not math.isnan(x[3]) else math.inf,x[4],x[2])
for x in sorted(agg,key=rank): print(f'STAGE_H profile={x[1]} threads={x[2]} wall_s={x[0]:.9f} high_s={x[3]:.9f} regs={x[4]} spill_store={x[5]} spill_load={x[6]} spill_free={int(x[7])} self_evict={self_evict}',file=sys.stderr)
selfx=min((x for x in agg if x[1]=='self' and x[7]),default=None,key=rank); matex=min((x for x in agg if x[1]=='mate' and x[7]),default=None,key=rank)
if selfx is None or matex is None: raise SystemExit('Stage-H candidates need known spill-free ILP2+ILP8 resources')
speed=selfx[0]/matex[0]; enabled=int(rank(matex)<rank(selfx)); q=lambda x:shlex.quote(str(x))
vals={'B300_STAGEH_ROWS':rows_arg,'B300_STAGEH_RESIDUE':next(iter(res)),'B300_STAGEH_WIDTH':int(w),'B300_STAGEH_DISTANCE':int(d),'B300_STAGEH_SELF_EVICT':self_evict,'B300_STAGEH_SELF_BIN':selfbin,'B300_STAGEH_SELF_THREADS':selfx[2],'B300_STAGEH_SELF_WALL_S':f'{selfx[0]:.9f}','B300_STAGEH_SELF_HIGH_S':f'{selfx[3]:.9f}','B300_STAGEH_SELF_SPILL_FREE':1,'B300_STAGEH_MATE_BIN':matebin,'B300_STAGEH_MATE_THREADS':matex[2],'B300_STAGEH_MATE_WALL_S':f'{matex[0]:.9f}','B300_STAGEH_MATE_HIGH_S':f'{matex[3]:.9f}','B300_STAGEH_MATE_SPILL_FREE':1,'B300_STAGEH_SPEEDUP':f'{speed:.9f}','B300_STAGEH_BEST_ENABLED':enabled}
with open(winner,'w') as f:
 for k,v in vals.items(): f.write(k+'='+q(v)+'\n')
print('b300_stageh_exact_match=1'); print(f'b300_stageh_self_evict={self_evict}'); print(f'b300_stageh_speedup={speed:.9f}x'); print(f'b300_stageh_best_enabled={enabled}')
PY
cat "$RESULT"; cat "$RESOURCE"
echo "b300-nextgen-hybrid8-nextmate-ab OK rows=$ROWS width=$W distance=$D self_evict=$SELF_EVICT winner_env=$WINNER_ENV" >&2
