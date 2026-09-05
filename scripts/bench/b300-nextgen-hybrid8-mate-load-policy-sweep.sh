#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; ROWS="${ROWS:-1}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
STAGEL_GUARD_ENV="${STAGEL_GUARD_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_prefetch_guard_stagel_g${NGPU}_winner.env}"
POLICY_LIST="${POLICY_LIST:-default cg cs}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"; REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_mate_load_stagem_row${ROWS}_g${NGPU}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-mate-load-policy.sh"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")" "$(dirname "$WINNER_ENV")"
for f in "$STAGE_F_ENV" "$STAGEL_GUARD_ENV" "$BUILDER"; do [[ -s "$f" ]] || { echo "missing Stage-M input=$f" >&2; exit 2; }; done
[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || exit 2
[[ "$NGPU" =~ ^[1-8]$ ]] || exit 2
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || exit 2

policies=()
for p in $POLICY_LIST; do
  case "$p" in default|cg|cs) ;; *) echo "bad POLICY_LIST entry=$p" >&2; exit 2;; esac
  seen=0; for old in "${policies[@]}"; do [[ "$old" == "$p" ]] && seen=1; done
  ((seen)) || policies+=("$p")
done
((${#policies[@]})) || exit 2
has_default=0; for p in "${policies[@]}"; do [[ "$p" == default ]] && has_default=1; done
((has_default)) || { echo 'POLICY_LIST must include default' >&2; exit 2; }
threads=()
for th in $THREADS_LIST; do
  [[ "$th" =~ ^[0-9]+$ ]] && ((th>=32&&th<=1024&&th%32==0)) || exit 2
  seen=0; for old in "${threads[@]}"; do [[ "$old" == "$th" ]] && seen=1; done
  ((seen)) || threads+=("$th")
done
((${#threads[@]})) || exit 2
command -v nvcc >/dev/null || exit 2
command -v nvidia-smi >/dev/null || exit 2
VISIBLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( VISIBLE_GPUS >= NGPU )) || { echo "need $NGPU visible GPU(s), found $VISIBLE_GPUS" >&2; exit 2; }

# Stage F owns the recurrence/build policy. Stage L owns the exact geometry,
# eviction hints and prefetch guard policy. Stage M is allowed to change only
# the eight ILP8 mate loads.
# shellcheck disable=SC1090
source "$STAGE_F_ENV"
for k in B300_HYBRID8_NEXTSELF_THRESHOLD B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK B300_HYBRID8_NEXTSELF_RANDOM_CG B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES B300_HYBRID8_NEXTSELF_PREFETCH_L2 B300_HYBRID8_NEXTSELF_DUALMASK B300_HYBRID8_NEXTSELF_CLOSURE_BATCH B300_HYBRID8_NEXTSELF_MAXRREGCOUNT; do
  [[ -n "${!k+x}" ]] || { echo "Stage-F env missing $k" >&2; exit 3; }
done
T="$B300_HYBRID8_NEXTSELF_THRESHOLD"; H="$B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK"; CG="$B300_HYBRID8_NEXTSELF_RANDOM_CG"; CGL2="$B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES"; PRE="$B300_HYBRID8_NEXTSELF_PREFETCH_L2"; DUAL="$B300_HYBRID8_NEXTSELF_DUALMASK"; BATCH="$B300_HYBRID8_NEXTSELF_CLOSURE_BATCH"; CAP="$B300_HYBRID8_NEXTSELF_MAXRREGCOUNT"
# shellcheck disable=SC1090
source "$STAGEL_GUARD_ENV"
for k in B300_STAGEL_STAGED_VALIDATED B300_STAGEL_FINAL_ENABLED B300_STAGEL_FINAL_SPILL_FREE B300_STAGEL_NGPU B300_STAGEL_SELF_WIDTH B300_STAGEL_SELF_DISTANCE B300_STAGEL_SELF_EVICT B300_STAGEL_MATE_WIDTH B300_STAGEL_MATE_DISTANCE B300_STAGEL_MATE_EVICT B300_STAGEL_FINAL_SELF_GUARD B300_STAGEL_FINAL_MATE_GUARD B300_STAGEL_FINAL_BIN B300_STAGEL_FINAL_THREADS B300_STAGEL_FINAL_STAGE_ROWS B300_STAGEL_FINAL_STAGE_RESIDUE; do
  [[ -n "${!k+x}" ]] || { echo "Stage-L guard env missing $k" >&2; exit 3; }
done
[[ "$B300_STAGEL_NGPU" == "$NGPU" ]] || { echo "Stage-L GPU count drift expected=$NGPU got=$B300_STAGEL_NGPU" >&2; exit 3; }
[[ "$B300_STAGEL_FINAL_SPILL_FREE" == 1 ]] || { echo 'Stage-L final candidate is not spill-free' >&2; exit 4; }
SW="$B300_STAGEL_SELF_WIDTH"; SD="$B300_STAGEL_SELF_DISTANCE"; SE="$B300_STAGEL_SELF_EVICT"; MW="$B300_STAGEL_MATE_WIDTH"; MD="$B300_STAGEL_MATE_DISTANCE"; ME="$B300_STAGEL_MATE_EVICT"; SG="$B300_STAGEL_FINAL_SELF_GUARD"; MG="$B300_STAGEL_FINAL_MATE_GUARD"
CONTROL_BIN="$B300_STAGEL_FINAL_BIN"; CONTROL_THREADS="$B300_STAGEL_FINAL_THREADS"
[[ -x "$CONTROL_BIN" ]] || { echo "Stage-L final binary missing: $CONTROL_BIN" >&2; exit 3; }
for w in "$SW" "$MW"; do case "$w" in 1|2|4|8) ;; *) exit 3;; esac; done
for d in "$SD" "$MD"; do case "$d" in 1|2|4) ;; *) exit 3;; esac; done
for ev in "$SE" "$ME"; do case "$ev" in default|normal|last) ;; *) exit 3;; esac; done
for g in "$SG" "$MG"; do case "$g" in branch|predicated) ;; *) exit 3;; esac; done

BINS="$LOGDIR/binaries.tsv"
printf 'policy\tbinary\tbuild_err\n' >"$BINS"
printf 'default\t%s\t-\n' "$CONTROL_BIN" >>"$BINS"
for policy in "${policies[@]}"; do
  [[ "$policy" == default ]] && continue
  bin="$ONEESAN_BUILD_DIR/b300_stagem_${policy}_selfw${SW}_selfd${SD}_matew${MW}_mated${MD}_sev${SE}_mev${ME}_sg${SG}_mg${MG}_t${T}_n27"
  err="$LOGDIR/${policy}.build.err"; out="$LOGDIR/${policy}.build.out"
  N=27 ARCH="$ARCH" OUT="$bin" MATE_LOAD_POLICY="$policy" HIGH_DROP_CHUNK="$H" HYBRID_THRESHOLD="$T" \
    SELF_WIDTH="$SW" SELF_DISTANCE="$SD" MATE_WIDTH="$MW" MATE_DISTANCE="$MD" SELF_EVICT="$SE" MATE_EVICT="$ME" SELF_GUARD="$SG" MATE_GUARD="$MG" \
    RANDOM_CG="$CG" RANDOM_CG_L2_FETCH_BYTES="$CGL2" PREFETCH_L2="$PRE" DUALMASK="$DUAL" CLOSURE_BATCH="$BATCH" MAXRREGCOUNT="$CAP" PTXAS_VERBOSE=1 BUILD_ERR="$err" \
    bash "$BUILDER" >"$out" 2>"$LOGDIR/${policy}.driver.err"
  [[ -x "$bin" ]] || exit 3
  grep -Fq "mate_load_policy=$policy mate_load_scope=ilp8_only" "$out" || exit 3
  grep -Fq "self_geometry width=$SW distance=$SD evict=$SE guard=$SG" "$out" || { echo 'Stage-M self policy drift' >&2; exit 3; }
  grep -Fq "mate_geometry width=$MW distance=$MD evict=$ME guard=$MG" "$out" || { echo 'Stage-M mate policy drift' >&2; exit 3; }
  grep -Fq 'guard_policy_preserved=1' "$out" || exit 3
  printf '%s\t%s\t%s\n' "$policy" "$bin" "$err" >>"$BINS"
done

printf 'policy\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
while IFS=$'\t' read -r policy bin err; do
  [[ "$policy" == policy || "$policy" == default ]] && continue
  python3 "$PARSER" "$err" --label "$policy" --contains main_pull_kernel_ilp2 >>"$RESOURCE" || true
  python3 "$PARSER" "$err" --label "$policy" --contains main_pull_kernel_ilp8_hybrid >>"$RESOURCE" || true
done <"$BINS"
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
printf 'policy\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tforward_high_s\treverse_high_s\n' >"$RESULT"
run_one(){
  local policy="$1" bin="$2" th="$3" r="$4" so="$LOGDIR/${policy}_t${th}_r${r}.out" se="$LOGDIR/${policy}_t${th}_r${r}.err" line
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$th" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { tail -n 80 "$se" >&2 || true; return 4; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$policy" "$th" "$r" "$(field residue "$line")" "$(field wall_s "$line")" "$(field active_max_s "$line")" "$(field active_sum_s "$line")" "$(field forward_high_s "$line")" "$(field reverse_high_s "$line")" >>"$RESULT"
}
while IFS=$'\t' read -r policy bin err; do
  [[ "$policy" == policy ]] && continue
  for th in "${threads[@]}"; do
    for ((r=1;r<=REPEATS;++r)); do
      echo "=== Stage M policy=$policy threads=$th repeat=$r rows=$ROWS gpus=$NGPU guards=$SG/$MG ===" >&2
      run_one "$policy" "$bin" "$th" "$r"
    done
  done
done <"$BINS"

python3 - "$RESULT" "$RESOURCE" "$BINS" "$WINNER_ENV" "$ROWS" "$NGPU" "$SW" "$SD" "$SE" "$MW" "$MD" "$ME" "$SG" "$MG" "$CONTROL_THREADS" <<'PY'
import csv,math,statistics,sys,shlex
result,resource,bins_path,winner,rows_arg,ngpu,sw,sd,se,mw,md,me,sg,mg,control_threads=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t')); rr=list(csv.DictReader(open(resource),delimiter='\t')); bins={r['policy']:r['binary'] for r in csv.DictReader(open(bins_path),delimiter='\t')}
if not rows: raise SystemExit('no Stage-M rows')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL Stage-M residue mismatch '+repr({(r['policy'],r['threads'],r['repeat']):r['residue'] for r in rows}))
resources={p:[] for p in bins}
for r in rr:
    try: resources[r['policy']].append((int(r['registers']),int(r['spill_store_bytes']),int(r['spill_load_bytes'])))
    except (ValueError,KeyError): pass
agg=[]
for p,t in {(r['policy'],int(r['threads'])) for r in rows}:
    rs=[r for r in rows if r['policy']==p and int(r['threads'])==t]
    wall=statistics.median(float(r['wall_s']) for r in rs); hs=[]
    for r in rs:
        try: hs.append(float(r['forward_high_s'])+float(r['reverse_high_s']))
        except ValueError: pass
    high=statistics.median(hs) if hs else math.nan
    if p=='default': regs=-1; ss=sl=0; clean=True
    else:
        rv=resources[p]; regs=max((x[0] for x in rv),default=-1); ss=max((x[1] for x in rv),default=-1); sl=max((x[2] for x in rv),default=-1); clean=len(rv)>=2 and ss==0 and sl==0
    agg.append((wall,p,t,high,regs,ss,sl,clean))
def rank(x): return (x[0],x[3] if not math.isnan(x[3]) else math.inf,x[4] if x[4]>=0 else math.inf,x[2])
for x in sorted(agg,key=rank): print(f'STAGE_M policy={x[1]} threads={x[2]} wall_s={x[0]:.9f} high_s={x[3]:.9f} regs={x[4]} spill_store={x[5]} spill_load={x[6]} spill_free={int(x[7])}',file=sys.stderr)
base=min((x for x in agg if x[1]=='default'),key=rank); tests=[x for x in agg if x[1]!='default' and x[7]]
best=min(tests,key=rank) if tests else base
speed=base[0]/best[0]; enabled=int(best[1]!='default' and rank(best)<rank(base)); q=lambda x:shlex.quote(str(x))
vals={
 'B300_STAGEM_ROWS':rows_arg,'B300_STAGEM_NGPU':ngpu,'B300_STAGEM_RESIDUE':next(iter(res)),
 'B300_STAGEM_SELF_WIDTH':sw,'B300_STAGEM_SELF_DISTANCE':sd,'B300_STAGEM_SELF_EVICT':se,'B300_STAGEM_SELF_GUARD':sg,
 'B300_STAGEM_MATE_WIDTH':mw,'B300_STAGEM_MATE_DISTANCE':md,'B300_STAGEM_MATE_EVICT':me,'B300_STAGEM_MATE_GUARD':mg,
 'B300_STAGEM_CONTROL_BIN':bins['default'],'B300_STAGEM_CONTROL_THREADS':base[2],'B300_STAGEM_CONTROL_SOURCE_THREADS':control_threads,
 'B300_STAGEM_CONTROL_WALL_S':f'{base[0]:.9f}','B300_STAGEM_CONTROL_HIGH_S':f'{base[3]:.9f}','B300_STAGEM_CONTROL_SPILL_FREE':1,
 'B300_STAGEM_POLICY':best[1],'B300_STAGEM_BIN':bins[best[1]],'B300_STAGEM_THREADS':best[2],
 'B300_STAGEM_WALL_S':f'{best[0]:.9f}','B300_STAGEM_HIGH_S':f'{best[3]:.9f}','B300_STAGEM_SPILL_FREE':int(best[7]),
 'B300_STAGEM_SPEEDUP':f'{speed:.9f}','B300_STAGEM_BEST_ENABLED':enabled,
}
with open(winner,'w') as f:
    for k,v in vals.items(): f.write(k+'='+q(v)+'\n')
print('b300_stagem_exact_match=1'); print(f'b300_stagem_ngpu={ngpu}'); print(f'b300_stagem_policy={best[1]}'); print(f'b300_stagem_speedup={speed:.9f}x'); print(f'b300_stagem_best_enabled={enabled}')
PY
cat "$RESULT"; cat "$RESOURCE"
echo "b300-nextgen-hybrid8-mate-load-policy-sweep OK stage=M rows=$ROWS ngpu=$NGPU guards=$SG/$MG policies=${policies[*]} winner_env=$WINNER_ENV" >&2
