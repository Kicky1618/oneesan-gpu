#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
ARCH="${ARCH:-native}"
MOD="${MOD:-4294967291}"
ROWS="${ROWS:-1}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
NGPU="${NGPU:-8}"
SELF_WIDTH="${SELF_WIDTH:?SELF_WIDTH required}"
SELF_DISTANCE="${SELF_DISTANCE:?SELF_DISTANCE required}"
SELF_EVICT="${SELF_EVICT:?SELF_EVICT required}"
MATE_WIDTH="${MATE_WIDTH:?MATE_WIDTH required}"
MATE_DISTANCE="${MATE_DISTANCE:?MATE_DISTANCE required}"
MATE_EVICT="${MATE_EVICT:?MATE_EVICT required}"
SELF_GUARD_LIST="${SELF_GUARD_LIST:-branch predicated}"
MATE_GUARD_LIST="${MATE_GUARD_LIST:-branch predicated}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_guard_row${ROWS}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESULT="${RESULT:-${PREFIX}.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-self-mate-geometry.sh"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"

mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")" "$(dirname "$WINNER_ENV")"
[[ -s "$STAGE_F_ENV" ]] || { echo "missing STAGE_F_ENV=$STAGE_F_ENV" >&2; exit 2; }
[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && (( ROWS <= 28 )) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$NGPU" =~ ^[1-8]$ ]] || { echo 'NGPU must be 1..8' >&2; exit 2; }
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo 'REPEATS must be >=1' >&2; exit 2; }
for w in "$SELF_WIDTH" "$MATE_WIDTH"; do case "$w" in 1|2|4|8) ;; *) echo "bad width=$w" >&2; exit 2;; esac; done
for d in "$SELF_DISTANCE" "$MATE_DISTANCE"; do case "$d" in 1|2|4) ;; *) echo "bad distance=$d" >&2; exit 2;; esac; done
for e in "$SELF_EVICT" "$MATE_EVICT"; do case "$e" in default|normal|last) ;; *) echo "bad eviction=$e" >&2; exit 2;; esac; done

normalize_guards(){
  local raw="$1" out=() g old seen
  for g in $raw; do
    case "$g" in branch|predicated) ;; *) echo "bad guard=$g" >&2; exit 2;; esac
    seen=0; for old in "${out[@]}"; do [[ "$old" == "$g" ]] && seen=1; done
    (( seen )) || out+=("$g")
  done
  ((${#out[@]})) || { echo 'guard list must not be empty' >&2; exit 2; }
  printf '%s' "${out[*]}"
}
SELF_GUARD_LIST="$(normalize_guards "$SELF_GUARD_LIST")"
MATE_GUARD_LIST="$(normalize_guards "$MATE_GUARD_LIST")"
case " $SELF_GUARD_LIST " in *' branch '*) ;; *) echo 'SELF_GUARD_LIST must include branch control' >&2; exit 2;; esac
case " $MATE_GUARD_LIST " in *' branch '*) ;; *) echo 'MATE_GUARD_LIST must include branch control' >&2; exit 2;; esac
threads=()
for th in $THREADS_LIST; do
  [[ "$th" =~ ^[0-9]+$ ]] && (( th>=32 && th<=1024 && th%32==0 )) || { echo "bad threads=$th" >&2; exit 2; }
  seen=0; for old in "${threads[@]}"; do [[ "$old" == "$th" ]] && seen=1; done
  (( seen )) || threads+=("$th")
done
((${#threads[@]})) || exit 2

# shellcheck disable=SC1090
source "$STAGE_F_ENV"
for k in B300_HYBRID8_NEXTSELF_STAGED_VALIDATED B300_HYBRID8_NEXTSELF_FINAL_ENABLED \
  B300_HYBRID8_NEXTSELF_FINAL_WIDTH B300_HYBRID8_NEXTSELF_FINAL_DISTANCE \
  B300_HYBRID8_NEXTSELF_THRESHOLD B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK \
  B300_HYBRID8_NEXTSELF_RANDOM_CG B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES \
  B300_HYBRID8_NEXTSELF_PREFETCH_L2 B300_HYBRID8_NEXTSELF_DUALMASK \
  B300_HYBRID8_NEXTSELF_CLOSURE_BATCH B300_HYBRID8_NEXTSELF_MAXRREGCOUNT; do
  [[ -n "${!k+x}" ]] || { echo "Stage-F env missing $k" >&2; exit 3; }
done
[[ "$B300_HYBRID8_NEXTSELF_STAGED_VALIDATED" == 1 && "$B300_HYBRID8_NEXTSELF_FINAL_ENABLED" == 1 ]] || { echo 'Stage F not promotable' >&2; exit 4; }
[[ "$B300_HYBRID8_NEXTSELF_FINAL_WIDTH" == "$SELF_WIDTH" && "$B300_HYBRID8_NEXTSELF_FINAL_DISTANCE" == "$SELF_DISTANCE" ]] || {
  echo 'Stage-L self geometry does not match Stage F' >&2; exit 3;
}
T="$B300_HYBRID8_NEXTSELF_THRESHOLD"; H="$B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK"
CG="$B300_HYBRID8_NEXTSELF_RANDOM_CG"; CGL2="$B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES"
PRE="$B300_HYBRID8_NEXTSELF_PREFETCH_L2"; DUAL="$B300_HYBRID8_NEXTSELF_DUALMASK"
BATCH="$B300_HYBRID8_NEXTSELF_CLOSURE_BATCH"; CAP="$B300_HYBRID8_NEXTSELF_MAXRREGCOUNT"

command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }

COMMON=(N=27 ARCH="$ARCH" HIGH_DROP_CHUNK="$H" HYBRID_THRESHOLD="$T" RANDOM_CG="$CG" RANDOM_CG_L2_FETCH_BYTES="$CGL2" PREFETCH_L2="$PRE" DUALMASK="$DUAL" CLOSURE_BATCH="$BATCH" MAXRREGCOUNT="$CAP" PTXAS_VERBOSE=1)
BINS="$LOGDIR/binaries.tsv"
printf 'candidate\tself_guard\tmate_guard\tbinary\tbuild_err\n' >"$BINS"
for sg in $SELF_GUARD_LIST; do
  for mg in $MATE_GUARD_LIST; do
    name="sg_${sg}_mg_${mg}"
    bin="$ONEESAN_BUILD_DIR/b300_stagel_${name}_sw${SELF_WIDTH}_sd${SELF_DISTANCE}_mw${MATE_WIDTH}_md${MATE_DISTANCE}_sev${SELF_EVICT}_mev${MATE_EVICT}_t${T}_n27"
    err="$LOGDIR/${name}.build.err"; out="$LOGDIR/${name}.build.out"
    env "${COMMON[@]}" OUT="$bin" SELF_WIDTH="$SELF_WIDTH" SELF_DISTANCE="$SELF_DISTANCE" MATE_WIDTH="$MATE_WIDTH" MATE_DISTANCE="$MATE_DISTANCE" \
      SELF_EVICT="$SELF_EVICT" MATE_EVICT="$MATE_EVICT" SELF_GUARD="$sg" MATE_GUARD="$mg" BUILD_ERR="$err" \
      bash "$BUILDER" >"$out" 2>"$LOGDIR/${name}.driver.err"
    [[ -x "$bin" ]] || { echo "Stage-L binary missing: $name" >&2; exit 3; }
    grep -Fq "self_geometry width=$SELF_WIDTH distance=$SELF_DISTANCE evict=$SELF_EVICT guard=$sg" "$out" || exit 3
    grep -Fq "mate_geometry width=$MATE_WIDTH distance=$MATE_DISTANCE evict=$MATE_EVICT guard=$mg" "$out" || exit 3
    grep -Fq "guard_modes=$sg/$mg" "$out" || exit 3
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$sg" "$mg" "$bin" "$err" >>"$BINS"
  done
done

printf 'candidate\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
while IFS=$'\t' read -r name sg mg bin err; do
  [[ "$name" == candidate ]] && continue
  python3 "$PARSER" "$err" --label "$name" --contains main_pull_kernel_ilp2 >>"$RESOURCE" || true
  python3 "$PARSER" "$err" --label "$name" --contains main_pull_kernel_ilp8_hybrid >>"$RESOURCE" || true
done <"$BINS"

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
printf 'candidate\tself_guard\tmate_guard\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tforward_high_s\treverse_high_s\n' >"$RESULT"
run_one(){
  local name="$1" sg="$2" mg="$3" bin="$4" th="$5" r="$6" so se line
  so="$LOGDIR/${name}_t${th}_r${r}.out"; se="$LOGDIR/${name}_t${th}_r${r}.err"
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$th" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { tail -n 80 "$se" >&2 || true; return 4; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$sg" "$mg" "$th" "$r" \
    "$(field residue "$line")" "$(field wall_s "$line")" "$(field active_max_s "$line")" "$(field active_sum_s "$line")" \
    "$(field forward_high_s "$line")" "$(field reverse_high_s "$line")" >>"$RESULT"
}
for th in "${threads[@]}"; do
  while IFS=$'\t' read -r name sg mg bin err; do
    [[ "$name" == candidate ]] && continue
    for ((r=1;r<=REPEATS;++r)); do
      echo "=== Stage L $name threads=$th repeat=$r rows=$ROWS self=w${SELF_WIDTH}d${SELF_DISTANCE}/$SELF_EVICT mate=w${MATE_WIDTH}d${MATE_DISTANCE}/$MATE_EVICT ===" >&2
      run_one "$name" "$sg" "$mg" "$bin" "$th" "$r"
    done
  done <"$BINS"
done

python3 - "$RESULT" "$RESOURCE" "$BINS" "$WINNER_ENV" "$ROWS" "$NGPU" "$SELF_WIDTH" "$SELF_DISTANCE" "$SELF_EVICT" "$MATE_WIDTH" "$MATE_DISTANCE" "$MATE_EVICT" <<'PY'
import csv,math,statistics,sys,shlex
result,resource,bins_path,winner,rows_arg,ngpu,sw,sd,sev,mw,md,mev=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t')); rr=list(csv.DictReader(open(resource),delimiter='\t')); bins={r['candidate']:r for r in csv.DictReader(open(bins_path),delimiter='\t')}
if not rows: raise SystemExit('no Stage-L rows')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL Stage-L residue mismatch '+repr({(r['candidate'],r['threads'],r['repeat']):r['residue'] for r in rows}))
resources={n:[] for n in bins}
for r in rr:
    try: resources[r['candidate']].append((int(r['registers']),int(r['spill_store_bytes']),int(r['spill_load_bytes'])))
    except (ValueError,KeyError): pass
agg=[]
for name,t in {(r['candidate'],int(r['threads'])) for r in rows}:
    rs=[r for r in rows if r['candidate']==name and int(r['threads'])==t]
    wall=statistics.median(float(r['wall_s']) for r in rs); hs=[]
    for r in rs:
        try: hs.append(float(r['forward_high_s'])+float(r['reverse_high_s']))
        except ValueError: pass
    high=statistics.median(hs) if hs else math.nan
    rv=resources[name]; regs=max((x[0] for x in rv),default=-1); ss=max((x[1] for x in rv),default=-1); sl=max((x[2] for x in rv),default=-1)
    clean=len(rv)>=2 and ss==0 and sl==0
    agg.append((wall,name,t,high,regs,ss,sl,clean,len(rv)))
def rank(x): return (x[0],x[3] if not math.isnan(x[3]) else math.inf,x[4],x[2])
for x in sorted(agg,key=rank): print(f'STAGE_L candidate={x[1]} threads={x[2]} wall_s={x[0]:.9f} high_s={x[3]:.9f} regs={x[4]} spill_store={x[5]} spill_load={x[6]} spill_free={int(x[7])} resource_rows={x[8]}',file=sys.stderr)
control=min((x for x in agg if x[1]=='sg_branch_mg_branch' and x[7]),default=None,key=rank)
clean=[x for x in agg if x[7]]
if control is None or not clean: raise SystemExit('Stage-L requires known spill-free branch/branch control and candidate')
best=min(clean,key=rank); meta=bins[best[1]]; speed=control[0]/best[0]; enabled=int(best[1]!='sg_branch_mg_branch' and rank(best)<rank(control))
q=lambda x:shlex.quote(str(x))
vals={
 'B300_STAGEL_ROWS':rows_arg,'B300_STAGEL_NGPU':ngpu,'B300_STAGEL_RESIDUE':next(iter(res)),
 'B300_STAGEL_SELF_WIDTH':int(sw),'B300_STAGEL_SELF_DISTANCE':int(sd),'B300_STAGEL_SELF_EVICT':sev,
 'B300_STAGEL_MATE_WIDTH':int(mw),'B300_STAGEL_MATE_DISTANCE':int(md),'B300_STAGEL_MATE_EVICT':mev,
 'B300_STAGEL_CONTROL_BIN':bins['sg_branch_mg_branch']['binary'],'B300_STAGEL_CONTROL_THREADS':control[2],
 'B300_STAGEL_CONTROL_WALL_S':f'{control[0]:.9f}','B300_STAGEL_CONTROL_HIGH_S':f'{control[3]:.9f}','B300_STAGEL_CONTROL_SPILL_FREE':1,
 'B300_STAGEL_SELF_GUARD':meta['self_guard'],'B300_STAGEL_MATE_GUARD':meta['mate_guard'],'B300_STAGEL_BIN':meta['binary'],
 'B300_STAGEL_THREADS':best[2],'B300_STAGEL_WALL_S':f'{best[0]:.9f}','B300_STAGEL_HIGH_S':f'{best[3]:.9f}','B300_STAGEL_SPILL_FREE':1,
 'B300_STAGEL_SPEEDUP':f'{speed:.9f}','B300_STAGEL_BEST_ENABLED':enabled,
}
with open(winner,'w') as f:
    for k,v in vals.items(): f.write(k+'='+q(v)+'\n')
print('b300_stagel_exact_match=1'); print(f'b300_stagel_ngpu={ngpu}'); print(f'b300_stagel_speedup={speed:.9f}x'); print(f'b300_stagel_best_enabled={enabled}'); print(f'b300_stagel_best_guard={meta["self_guard"]}/{meta["mate_guard"]}')
PY
cat "$RESULT"
cat "$RESOURCE"
echo "b300-nextgen-hybrid8-guard-sweep OK rows=$ROWS ngpu=$NGPU self_guard_list=[$SELF_GUARD_LIST] mate_guard_list=[$MATE_GUARD_LIST] winner_env=$WINNER_ENV" >&2