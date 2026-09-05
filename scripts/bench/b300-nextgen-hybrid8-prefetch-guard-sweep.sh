#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
UPSTREAM_PREPARE_ENV="${UPSTREAM_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_mate_evict_stagek_fullprime_n27_prepared.env}"
ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; ROWS="${ROWS:-1}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
GUARD_LIST="${GUARD_LIST:-bb pb bp pp}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"; REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_prefetch_guard_row${ROWS}_g${NGPU}}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"; BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-self-mate-geometry.sh"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")" "$(dirname "$WINNER_ENV")"
[[ -s "$STAGE_F_ENV" ]] || { echo "missing Stage-F env=$STAGE_F_ENV" >&2; exit 2; }
[[ -s "$UPSTREAM_PREPARE_ENV" ]] || { echo "missing upstream prepare env=$UPSTREAM_PREPARE_ENV" >&2; exit 2; }
[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || exit 2
[[ "$NGPU" =~ ^[1-8]$ ]] || exit 2
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || exit 2
# shellcheck disable=SC1090
source "$STAGE_F_ENV"
for k in B300_HYBRID8_NEXTSELF_STAGED_VALIDATED B300_HYBRID8_NEXTSELF_FINAL_ENABLED B300_HYBRID8_NEXTSELF_THRESHOLD B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK B300_HYBRID8_NEXTSELF_RANDOM_CG B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES B300_HYBRID8_NEXTSELF_PREFETCH_L2 B300_HYBRID8_NEXTSELF_DUALMASK B300_HYBRID8_NEXTSELF_CLOSURE_BATCH B300_HYBRID8_NEXTSELF_MAXRREGCOUNT; do
  [[ -n "${!k+x}" ]] || { echo "Stage-F env missing $k" >&2; exit 3; }
done
[[ "$B300_HYBRID8_NEXTSELF_STAGED_VALIDATED" == 1 && "$B300_HYBRID8_NEXTSELF_FINAL_ENABLED" == 1 ]] || exit 4
T="$B300_HYBRID8_NEXTSELF_THRESHOLD"; H="$B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK"; CG="$B300_HYBRID8_NEXTSELF_RANDOM_CG"; CGL2="$B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES"; PRE="$B300_HYBRID8_NEXTSELF_PREFETCH_L2"; DUAL="$B300_HYBRID8_NEXTSELF_DUALMASK"; BATCH="$B300_HYBRID8_NEXTSELF_CLOSURE_BATCH"; CAP="$B300_HYBRID8_NEXTSELF_MAXRREGCOUNT"
# shellcheck disable=SC1090
source "$UPSTREAM_PREPARE_ENV"
UPSTREAM_KIND=""
if [[ "${B300_STAGEK_PREPARED:-0}" == 1 ]]; then
  UPSTREAM_KIND=stagek
  SW="$B300_STAGEK_PREPARED_SELF_WIDTH"; SD="$B300_STAGEK_PREPARED_SELF_DISTANCE"; SE="$B300_STAGEK_PREPARED_SELF_EVICT"; MW="$B300_STAGEK_PREPARED_MATE_WIDTH"; MD="$B300_STAGEK_PREPARED_MATE_DISTANCE"; ME="$B300_STAGEK_PREPARED_MATE_EVICT"
elif [[ "${B300_STAGEJ_PREPARED:-0}" == 1 ]]; then
  UPSTREAM_KIND=stagej
  SW="$B300_STAGEJ_PREPARED_SELF_WIDTH"; SD="$B300_STAGEJ_PREPARED_SELF_DISTANCE"; SE="$B300_STAGEJ_PREPARED_SELF_EVICT"; MW="$B300_STAGEJ_PREPARED_MATE_WIDTH"; MD="$B300_STAGEJ_PREPARED_MATE_DISTANCE"; ME="$B300_STAGEJ_PREPARED_MATE_EVICT"
else
  echo 'upstream prepare env is neither Stage J nor Stage K' >&2; exit 3
fi
for w in "$SW" "$MW"; do case "$w" in 1|2|4|8) ;; *) exit 3;; esac; done
for d in "$SD" "$MD"; do case "$d" in 1|2|4) ;; *) exit 3;; esac; done
for ev in "$SE" "$ME"; do case "$ev" in default|normal|last) ;; *) exit 3;; esac; done

guards=()
for g in $GUARD_LIST; do
  case "$g" in bb|pb|bp|pp) ;; *) echo "bad guard profile=$g" >&2; exit 2;; esac
  seen=0; for old in "${guards[@]}"; do [[ "$old" == "$g" ]] && seen=1; done; ((seen)) || guards+=("$g")
done
has_bb=0; for g in "${guards[@]}"; do [[ "$g" == bb ]] && has_bb=1; done
((has_bb)) || { echo 'GUARD_LIST must include bb baseline' >&2; exit 2; }
for th in $THREADS_LIST; do [[ "$th" =~ ^[0-9]+$ ]] && ((th>=32&&th<=1024&&th%32==0)) || exit 2; done
command -v nvcc >/dev/null || exit 2; command -v nvidia-smi >/dev/null || exit 2
VISIBLE="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; ((VISIBLE>=NGPU)) || { echo "need $NGPU visible GPU(s), found $VISIBLE" >&2; exit 2; }

printf 'profile\tself_guard\tmate_guard\tbinary\tbuild_err\n' >"$LOGDIR/binaries.tsv"
for g in "${guards[@]}"; do
  case "$g" in bb) SG=branch; MG=branch;; pb) SG=predicated; MG=branch;; bp) SG=branch; MG=predicated;; pp) SG=predicated; MG=predicated;; esac
  bin="$ONEESAN_BUILD_DIR/b300_guard_${g}_selfw${SW}_d${SD}_matew${MW}_d${MD}_sev${SE}_mev${ME}_t${T}_n27"; err="$LOGDIR/${g}.build.err"; out="$LOGDIR/${g}.build.out"
  N=27 ARCH="$ARCH" OUT="$bin" HIGH_DROP_CHUNK="$H" HYBRID_THRESHOLD="$T" SELF_WIDTH="$SW" SELF_DISTANCE="$SD" MATE_WIDTH="$MW" MATE_DISTANCE="$MD" SELF_EVICT="$SE" MATE_EVICT="$ME" SELF_GUARD="$SG" MATE_GUARD="$MG" RANDOM_CG="$CG" RANDOM_CG_L2_FETCH_BYTES="$CGL2" PREFETCH_L2="$PRE" DUALMASK="$DUAL" CLOSURE_BATCH="$BATCH" MAXRREGCOUNT="$CAP" PTXAS_VERBOSE=1 BUILD_ERR="$err" bash "$BUILDER" >"$out" 2>"$LOGDIR/${g}.driver.err"
  [[ -x "$bin" ]] || { echo "guard binary missing profile=$g" >&2; exit 3; }
  grep -Fq "self_geometry width=$SW distance=$SD evict=$SE guard=$SG" "$out" || exit 3
  grep -Fq "mate_geometry width=$MW distance=$MD evict=$ME guard=$MG" "$out" || exit 3
  printf '%s\t%s\t%s\t%s\t%s\n' "$g" "$SG" "$MG" "$bin" "$err" >>"$LOGDIR/binaries.tsv"
done

printf 'profile\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
while IFS=$'\t' read -r g sg mg bin err; do
  [[ "$g" == profile ]] && continue
  python3 "$PARSER" "$err" --label "$g" --contains main_pull_kernel_ilp2 >>"$RESOURCE" || true
  python3 "$PARSER" "$err" --label "$g" --contains main_pull_kernel_ilp8_hybrid >>"$RESOURCE" || true
done <"$LOGDIR/binaries.tsv"
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
printf 'profile\tself_guard\tmate_guard\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tforward_high_s\treverse_high_s\n' >"$RESULT"
run_one(){
  local g="$1" sg="$2" mg="$3" bin="$4" th="$5" r="$6" so="$LOGDIR/${g}_t${th}_r${r}.out" se="$LOGDIR/${g}_t${th}_r${r}.err"
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$th" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  local line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || return 4
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$g" "$sg" "$mg" "$th" "$r" "$(field residue "$line")" "$(field wall_s "$line")" "$(field active_max_s "$line")" "$(field active_sum_s "$line")" "$(field forward_high_s "$line")" "$(field reverse_high_s "$line")" >>"$RESULT"
}
for th in $THREADS_LIST; do
  while IFS=$'\t' read -r g sg mg bin err; do
    [[ "$g" == profile ]] && continue
    for ((r=1;r<=REPEATS;++r)); do echo "=== guard=$g threads=$th repeat=$r rows=$ROWS ngpu=$NGPU ===" >&2; run_one "$g" "$sg" "$mg" "$bin" "$th" "$r"; done
  done <"$LOGDIR/binaries.tsv"
done

python3 - "$RESULT" "$RESOURCE" "$LOGDIR/binaries.tsv" "$WINNER_ENV" "$ROWS" "$NGPU" "$UPSTREAM_KIND" "$SW" "$SD" "$SE" "$MW" "$MD" "$ME" <<'PY'
import csv,math,statistics,sys,shlex
result,resource,bins_path,winner,rows_arg,ngpu,upstream,sw,sd,se,mw,md,me=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t')); rr=list(csv.DictReader(open(resource),delimiter='\t')); bins={r['profile']:r for r in csv.DictReader(open(bins_path),delimiter='\t')}
if not rows: raise SystemExit('no guard rows')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL guard residue mismatch '+repr({(r['profile'],r['threads'],r['repeat']):r['residue'] for r in rows}))
resources={k:[] for k in bins}
for r in rr:
    try: resources[r['profile']].append((int(r['registers']),int(r['spill_store_bytes']),int(r['spill_load_bytes'])))
    except (ValueError,KeyError): pass
agg=[]
for g,t in {(r['profile'],int(r['threads'])) for r in rows}:
    rs=[r for r in rows if r['profile']==g and int(r['threads'])==t]; wall=statistics.median(float(r['wall_s']) for r in rs); hs=[]
    for r in rs:
        try: hs.append(float(r['forward_high_s'])+float(r['reverse_high_s']))
        except ValueError: pass
    high=statistics.median(hs) if hs else math.nan; rv=resources[g]; regs=max((x[0] for x in rv),default=-1); ss=max((x[1] for x in rv),default=-1); sl=max((x[2] for x in rv),default=-1); clean=len(rv)>=2 and ss==0 and sl==0
    agg.append((wall,g,t,high,regs,ss,sl,clean))
def rank(x): return (x[0],x[3] if not math.isnan(x[3]) else math.inf,x[4],x[2])
for x in sorted(agg,key=rank): print(f'PREFETCH_GUARD profile={x[1]} threads={x[2]} wall_s={x[0]:.9f} high_s={x[3]:.9f} regs={x[4]} spill_store={x[5]} spill_load={x[6]} spill_free={int(x[7])} ngpu={ngpu}',file=sys.stderr)
base=min((x for x in agg if x[1]=='bb' and x[7]),default=None,key=rank); tests=[x for x in agg if x[1]!='bb' and x[7]]
if base is None or not tests: raise SystemExit('guard sweep needs spill-free bb baseline and alternate candidate')
test=min(tests,key=rank); speed=base[0]/test[0]; enabled=int(rank(test)<rank(base)); b=bins[test[1]]; q=lambda x:shlex.quote(str(x))
vals={'B300_STAGEL_ROWS':rows_arg,'B300_STAGEL_NGPU':int(ngpu),'B300_STAGEL_RESIDUE':next(iter(res)),'B300_STAGEL_UPSTREAM_KIND':upstream,'B300_STAGEL_SELF_WIDTH':int(sw),'B300_STAGEL_SELF_DISTANCE':int(sd),'B300_STAGEL_SELF_EVICT':se,'B300_STAGEL_MATE_WIDTH':int(mw),'B300_STAGEL_MATE_DISTANCE':int(md),'B300_STAGEL_MATE_EVICT':me,'B300_STAGEL_BASE_PROFILE':'bb','B300_STAGEL_BASE_BIN':bins['bb']['binary'],'B300_STAGEL_BASE_THREADS':base[2],'B300_STAGEL_BASE_WALL_S':f'{base[0]:.9f}','B300_STAGEL_BASE_HIGH_S':f'{base[3]:.9f}','B300_STAGEL_BASE_SPILL_FREE':1,'B300_STAGEL_PROFILE':test[1],'B300_STAGEL_SELF_GUARD':b['self_guard'],'B300_STAGEL_MATE_GUARD':b['mate_guard'],'B300_STAGEL_BIN':b['binary'],'B300_STAGEL_THREADS':test[2],'B300_STAGEL_WALL_S':f'{test[0]:.9f}','B300_STAGEL_HIGH_S':f'{test[3]:.9f}','B300_STAGEL_SPILL_FREE':1,'B300_STAGEL_SPEEDUP':f'{speed:.9f}','B300_STAGEL_BEST_ENABLED':enabled}
with open(winner,'w') as f:
    for k,v in vals.items(): f.write(k+'='+q(v)+'\n')
print('b300_stagel_exact_match=1'); print(f'b300_stagel_ngpu={ngpu}'); print(f'b300_stagel_profile={test[1]}'); print(f'b300_stagel_speedup={speed:.9f}x'); print(f'b300_stagel_best_enabled={enabled}')
PY
cat "$RESULT"; cat "$RESOURCE"
echo "b300-nextgen-hybrid8-prefetch-guard-sweep OK rows=$ROWS ngpu=$NGPU upstream=$UPSTREAM_KIND self=w${SW}d${SD}/$SE mate=w${MW}d${MD}/$ME profiles=${guards[*]} winner_env=$WINNER_ENV" >&2
