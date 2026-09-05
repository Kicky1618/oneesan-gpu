#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
STAGEJ_PREPARE_ENV="${STAGEJ_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextmate_geometry_stagej_fullprime_n27_prepared.env}"
ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; ROWS="${ROWS:-1}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
EVICT_LIST="${EVICT_LIST:-default normal last}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"; REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_mate_evict_row${ROWS}_g${NGPU}}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")" "$(dirname "$WINNER_ENV")"

[[ -s "$STAGE_F_ENV" ]] || { echo "missing Stage-F env=$STAGE_F_ENV" >&2; exit 2; }
[[ -s "$STAGEJ_PREPARE_ENV" ]] || { echo "missing Stage-J prepare env=$STAGEJ_PREPARE_ENV" >&2; exit 2; }
[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || exit 2
[[ "$NGPU" =~ ^[1-8]$ ]] || { echo 'NGPU must be 1..8' >&2; exit 2; }
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || exit 2
# shellcheck disable=SC1090
source "$STAGE_F_ENV"
for k in B300_HYBRID8_NEXTSELF_STAGED_VALIDATED B300_HYBRID8_NEXTSELF_FINAL_ENABLED B300_HYBRID8_NEXTSELF_THRESHOLD B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK B300_HYBRID8_NEXTSELF_RANDOM_CG B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES B300_HYBRID8_NEXTSELF_PREFETCH_L2 B300_HYBRID8_NEXTSELF_DUALMASK B300_HYBRID8_NEXTSELF_CLOSURE_BATCH B300_HYBRID8_NEXTSELF_MAXRREGCOUNT; do
  [[ -n "${!k+x}" ]] || { echo "Stage-F env missing $k" >&2; exit 3; }
done
[[ "$B300_HYBRID8_NEXTSELF_STAGED_VALIDATED" == 1 && "$B300_HYBRID8_NEXTSELF_FINAL_ENABLED" == 1 ]] || exit 4
T="$B300_HYBRID8_NEXTSELF_THRESHOLD"; H="$B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK"; CG="$B300_HYBRID8_NEXTSELF_RANDOM_CG"; CGL2="$B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES"; PRE="$B300_HYBRID8_NEXTSELF_PREFETCH_L2"; DUAL="$B300_HYBRID8_NEXTSELF_DUALMASK"; BATCH="$B300_HYBRID8_NEXTSELF_CLOSURE_BATCH"; CAP="$B300_HYBRID8_NEXTSELF_MAXRREGCOUNT"
# shellcheck disable=SC1090
source "$STAGEJ_PREPARE_ENV"
for k in B300_STAGEJ_PREPARED B300_STAGEJ_PREPARED_SELF_WIDTH B300_STAGEJ_PREPARED_SELF_DISTANCE B300_STAGEJ_PREPARED_SELF_EVICT B300_STAGEJ_PREPARED_MATE_WIDTH B300_STAGEJ_PREPARED_MATE_DISTANCE B300_STAGEJ_PREPARED_MATE_EVICT; do
  [[ -n "${!k+x}" ]] || { echo "Stage-J prepare env missing $k" >&2; exit 3; }
done
[[ "$B300_STAGEJ_PREPARED" == 1 ]] || exit 4
SW="$B300_STAGEJ_PREPARED_SELF_WIDTH"; SD="$B300_STAGEJ_PREPARED_SELF_DISTANCE"; SE="$B300_STAGEJ_PREPARED_SELF_EVICT"; MW="$B300_STAGEJ_PREPARED_MATE_WIDTH"; MD="$B300_STAGEJ_PREPARED_MATE_DISTANCE"; BASE_ME="$B300_STAGEJ_PREPARED_MATE_EVICT"
for w in "$SW" "$MW"; do case "$w" in 1|2|4|8) ;; *) exit 3;; esac; done
for d in "$SD" "$MD"; do case "$d" in 1|2|4) ;; *) exit 3;; esac; done
for ev in "$SE" "$BASE_ME"; do case "$ev" in default|normal|last) ;; *) exit 3;; esac; done

evicts=()
for ev in $EVICT_LIST; do
  case "$ev" in default|normal|last) ;; *) echo "bad mate eviction=$ev" >&2; exit 2;; esac
  seen=0; for old in "${evicts[@]}"; do [[ "$old" == "$ev" ]] && seen=1; done; ((seen)) || evicts+=("$ev")
done
has_base=0; for ev in "${evicts[@]}"; do [[ "$ev" == "$BASE_ME" ]] && has_base=1; done
((has_base)) || { echo "EVICT_LIST must include Stage-J baseline hint=$BASE_ME" >&2; exit 2; }
for th in $THREADS_LIST; do [[ "$th" =~ ^[0-9]+$ ]] && ((th>=32&&th<=1024&&th%32==0)) || exit 2; done
command -v nvcc >/dev/null || exit 2
command -v nvidia-smi >/dev/null || exit 2
VISIBLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( VISIBLE_GPUS >= NGPU )) || { echo "need $NGPU visible GPU(s), found $VISIBLE_GPUS" >&2; exit 2; }

printf 'evict\tbinary\tbuild_err\n' >"$LOGDIR/binaries.tsv"
for ev in "${evicts[@]}"; do
  bin="$ONEESAN_BUILD_DIR/b300_mate_evict_${ev}_selfw${SW}_selfd${SD}_matew${MW}_mated${MD}_t${T}_n27"
  err="$LOGDIR/${ev}.build.err"; out="$LOGDIR/${ev}.build.out"
  N=27 ARCH="$ARCH" OUT="$bin" HIGH_DROP_CHUNK="$H" HYBRID_THRESHOLD="$T" \
    SELF_WIDTH="$SW" SELF_DISTANCE="$SD" MATE_WIDTH="$MW" MATE_DISTANCE="$MD" SELF_EVICT="$SE" MATE_EVICT="$ev" \
    RANDOM_CG="$CG" RANDOM_CG_L2_FETCH_BYTES="$CGL2" PREFETCH_L2="$PRE" DUALMASK="$DUAL" CLOSURE_BATCH="$BATCH" MAXRREGCOUNT="$CAP" PTXAS_VERBOSE=1 BUILD_ERR="$err" \
    bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-self-mate-geometry.sh" >"$out" 2>"$LOGDIR/${ev}.driver.err"
  [[ -x "$bin" ]] || { echo "mate eviction binary missing evict=$ev" >&2; exit 3; }
  grep -Fq "self_geometry width=$SW distance=$SD evict=$SE" "$out" || exit 3
  grep -Fq "mate_geometry width=$MW distance=$MD evict=$ev" "$out" || exit 3
  printf '%s\t%s\t%s\n' "$ev" "$bin" "$err" >>"$LOGDIR/binaries.tsv"
done

printf 'evict\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
while IFS=$'\t' read -r ev bin err; do
  [[ "$ev" == evict ]] && continue
  python3 "$PARSER" "$err" --label "$ev" --contains main_pull_kernel_ilp2 >>"$RESOURCE" || true
  python3 "$PARSER" "$err" --label "$ev" --contains main_pull_kernel_ilp8_hybrid >>"$RESOURCE" || true
done <"$LOGDIR/binaries.tsv"
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
printf 'evict\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tforward_high_s\treverse_high_s\n' >"$RESULT"
run_one(){
  local ev="$1" bin="$2" th="$3" r="$4" so="$LOGDIR/${ev}_t${th}_r${r}.out" se="$LOGDIR/${ev}_t${th}_r${r}.err"
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$th" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  local line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || return 4
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ev" "$th" "$r" "$(field residue "$line")" "$(field wall_s "$line")" "$(field active_max_s "$line")" "$(field active_sum_s "$line")" "$(field forward_high_s "$line")" "$(field reverse_high_s "$line")" >>"$RESULT"
}
for th in $THREADS_LIST; do
  while IFS=$'\t' read -r ev bin err; do
    [[ "$ev" == evict ]] && continue
    for ((r=1;r<=REPEATS;++r)); do
      echo "=== mate-evict=$ev threads=$th repeat=$r rows=$ROWS ngpu=$NGPU self=w${SW}d${SD}/$SE mate=w${MW}d${MD} ===" >&2
      run_one "$ev" "$bin" "$th" "$r"
    done
  done <"$LOGDIR/binaries.tsv"
done

python3 - "$RESULT" "$RESOURCE" "$LOGDIR/binaries.tsv" "$WINNER_ENV" "$ROWS" "$SW" "$SD" "$SE" "$MW" "$MD" "$BASE_ME" "$NGPU" <<'PY'
import csv,math,statistics,sys,shlex
result,resource,bins_path,winner,rows_arg,sw,sd,se,mw,md,base_ev,ngpu=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t')); rr=list(csv.DictReader(open(resource),delimiter='\t')); bins={r['evict']:r for r in csv.DictReader(open(bins_path),delimiter='\t')}
if not rows: raise SystemExit('no mate-eviction rows')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL mate-eviction residue mismatch '+repr({(r['evict'],r['threads'],r['repeat']):r['residue'] for r in rows}))
resources={k:[] for k in bins}
for r in rr:
    try: resources[r['evict']].append((int(r['registers']),int(r['spill_store_bytes']),int(r['spill_load_bytes'])))
    except (ValueError,KeyError): pass
agg=[]
for ev,t in {(r['evict'],int(r['threads'])) for r in rows}:
    rs=[r for r in rows if r['evict']==ev and int(r['threads'])==t]
    wall=statistics.median(float(r['wall_s']) for r in rs); hs=[]
    for r in rs:
        try: hs.append(float(r['forward_high_s'])+float(r['reverse_high_s']))
        except ValueError: pass
    high=statistics.median(hs) if hs else math.nan
    rv=resources[ev]; regs=max((x[0] for x in rv),default=-1); ss=max((x[1] for x in rv),default=-1); sl=max((x[2] for x in rv),default=-1); clean=len(rv)>=2 and ss==0 and sl==0
    agg.append((wall,ev,t,high,regs,ss,sl,clean))
def rank(x): return (x[0],x[3] if not math.isnan(x[3]) else math.inf,x[4],x[2])
for x in sorted(agg,key=rank): print(f'MATE_EVICT evict={x[1]} threads={x[2]} wall_s={x[0]:.9f} high_s={x[3]:.9f} regs={x[4]} spill_store={x[5]} spill_load={x[6]} spill_free={int(x[7])} ngpu={ngpu}',file=sys.stderr)
base=min((x for x in agg if x[1]==base_ev and x[7]),default=None,key=rank)
tests=[x for x in agg if x[1]!=base_ev and x[7]]
if base is None or not tests: raise SystemExit('mate-eviction sweep needs spill-free baseline and alternate candidate')
test=min(tests,key=rank); speed=base[0]/test[0]; enabled=int(rank(test)<rank(base)); q=lambda x:shlex.quote(str(x))
vals={
 'B300_STAGEK_ROWS':rows_arg,'B300_STAGEK_NGPU':int(ngpu),'B300_STAGEK_RESIDUE':next(iter(res)),'B300_STAGEK_SELF_WIDTH':int(sw),'B300_STAGEK_SELF_DISTANCE':int(sd),'B300_STAGEK_SELF_EVICT':se,
 'B300_STAGEK_MATE_WIDTH':int(mw),'B300_STAGEK_MATE_DISTANCE':int(md),'B300_STAGEK_BASE_EVICT':base_ev,'B300_STAGEK_BASE_BIN':bins[base_ev]['binary'],'B300_STAGEK_BASE_THREADS':base[2],
 'B300_STAGEK_BASE_WALL_S':f'{base[0]:.9f}','B300_STAGEK_BASE_HIGH_S':f'{base[3]:.9f}','B300_STAGEK_BASE_SPILL_FREE':1,
 'B300_STAGEK_EVICT':test[1],'B300_STAGEK_BIN':bins[test[1]]['binary'],'B300_STAGEK_THREADS':test[2],'B300_STAGEK_WALL_S':f'{test[0]:.9f}','B300_STAGEK_HIGH_S':f'{test[3]:.9f}',
 'B300_STAGEK_SPILL_FREE':1,'B300_STAGEK_SPEEDUP':f'{speed:.9f}','B300_STAGEK_BEST_ENABLED':enabled,
}
with open(winner,'w') as f:
    for k,v in vals.items(): f.write(k+'='+q(v)+'\n')
print('b300_stagek_exact_match=1'); print(f'b300_stagek_ngpu={ngpu}'); print(f'b300_stagek_mate_evict={test[1]}'); print(f'b300_stagek_speedup={speed:.9f}x'); print(f'b300_stagek_best_enabled={enabled}')
PY
cat "$RESULT"; cat "$RESOURCE"
echo "b300-nextgen-hybrid8-mate-evict-sweep OK rows=$ROWS ngpu=$NGPU self=w${SW}d${SD}/$SE mate=w${MW}d${MD} baseline=$BASE_ME hints=${evicts[*]} winner_env=$WINNER_ENV" >&2
