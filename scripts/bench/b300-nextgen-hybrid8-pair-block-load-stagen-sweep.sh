#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
STAGEM_WINNER_ENV="${STAGEM_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_mate_load_stagem_g8_winner.env}"
STAGEM_PREPARE_ENV="${STAGEM_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_mate_load_stagem_fullprime_n27_prepared.env}"
ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; ROWS="${ROWS:-1}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
PAIR_POLICY_LIST="${PAIR_POLICY_LIST:-default cg cs}"; BLOCK_POLICY_LIST="${BLOCK_POLICY_LIST:-default cg cs}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"; REPEATS="${REPEATS:-1}"
STAGEN_CG_L2_FETCH_BYTES="${STAGEN_CG_L2_FETCH_BYTES:-128}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_stagen_row${ROWS}_g${NGPU}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-pair-block-load-policy.sh"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"

mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")" "$(dirname "$WINNER_ENV")"
for f in "$STAGE_F_ENV" "$STAGEM_WINNER_ENV" "$STAGEM_PREPARE_ENV"; do [[ -s "$f" ]] || { echo "missing Stage-N input=$f" >&2; exit 2; }; done
[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$NGPU" =~ ^[1-8]$ ]] || { echo 'NGPU must be 1..8' >&2; exit 2; }
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo 'REPEATS must be >=1' >&2; exit 2; }
case "$STAGEN_CG_L2_FETCH_BYTES" in 0|64|128|256) ;; *) echo 'STAGEN_CG_L2_FETCH_BYTES must be 0,64,128,256' >&2; exit 2;; esac
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
command -v sha256sum >/dev/null || { echo 'sha256sum required' >&2; exit 2; }
VISIBLE="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; ((VISIBLE>=NGPU)) || { echo "need $NGPU visible GPU(s), found $VISIBLE" >&2; exit 2; }

policies(){
  local raw="$1" out=() x old seen
  for x in $raw; do
    case "$x" in default|cg|cs) ;; *) echo "bad Stage-N policy=$x" >&2; exit 2;; esac
    seen=0; for old in "${out[@]}"; do [[ "$old" == "$x" ]] && seen=1; done; ((seen)) || out+=("$x")
  done
  ((${#out[@]})) || { echo 'Stage-N policy list must not be empty' >&2; exit 2; }
  printf '%s\n' "${out[@]}"
}
mapfile -t pair_policies < <(policies "$PAIR_POLICY_LIST")
mapfile -t block_policies < <(policies "$BLOCK_POLICY_LIST")
has_pair_default=0; for p in "${pair_policies[@]}"; do [[ "$p" == default ]] && has_pair_default=1; done
has_block_default=0; for p in "${block_policies[@]}"; do [[ "$p" == default ]] && has_block_default=1; done
((has_pair_default&&has_block_default)) || { echo 'Stage-N search must include default/default control' >&2; exit 2; }
threads=()
for th in $THREADS_LIST; do
  [[ "$th" =~ ^[0-9]+$ ]] && ((th>=32&&th<=1024&&th%32==0)) || { echo "bad THREADS_LIST entry=$th" >&2; exit 2; }
  seen=0; for old in "${threads[@]}"; do [[ "$old" == "$th" ]] && seen=1; done; ((seen)) || threads+=("$th")
done
((${#threads[@]})) || exit 2

# Stage F supplies all non-L/M transform knobs.
# shellcheck disable=SC1090
source "$STAGE_F_ENV"
for k in B300_HYBRID8_NEXTSELF_STAGED_VALIDATED B300_HYBRID8_NEXTSELF_FINAL_ENABLED \
  B300_HYBRID8_NEXTSELF_THRESHOLD B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK \
  B300_HYBRID8_NEXTSELF_RANDOM_CG B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES \
  B300_HYBRID8_NEXTSELF_PREFETCH_L2 B300_HYBRID8_NEXTSELF_DUALMASK \
  B300_HYBRID8_NEXTSELF_CLOSURE_BATCH B300_HYBRID8_NEXTSELF_MAXRREGCOUNT; do
  [[ -n "${!k+x}" ]] || { echo "Stage-F env missing $k" >&2; exit 3; }
done
[[ "$B300_HYBRID8_NEXTSELF_STAGED_VALIDATED" == 1 && "$B300_HYBRID8_NEXTSELF_FINAL_ENABLED" == 1 ]] || { echo 'Stage F not promotable' >&2; exit 4; }
T="$B300_HYBRID8_NEXTSELF_THRESHOLD"; H="$B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK"; CG="$B300_HYBRID8_NEXTSELF_RANDOM_CG"; CGL2="$B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES"; PRE="$B300_HYBRID8_NEXTSELF_PREFETCH_L2"; DUAL="$B300_HYBRID8_NEXTSELF_DUALMASK"; BATCH="$B300_HYBRID8_NEXTSELF_CLOSURE_BATCH"; CAP="$B300_HYBRID8_NEXTSELF_MAXRREGCOUNT"

# The exact Stage-M prepared binary is the Stage-N default/default control.
# shellcheck disable=SC1090
source "$STAGEM_PREPARE_ENV"
for k in B300_STAGEM_PREPARED B300_STAGEM_PREPARED_MOD B300_STAGEM_PREPARED_NGPU B300_STAGEM_PREPARED_POLICY \
  B300_STAGEM_PREPARED_SELF_WIDTH B300_STAGEM_PREPARED_SELF_DISTANCE B300_STAGEM_PREPARED_SELF_EVICT B300_STAGEM_PREPARED_SELF_GUARD \
  B300_STAGEM_PREPARED_MATE_WIDTH B300_STAGEM_PREPARED_MATE_DISTANCE B300_STAGEM_PREPARED_MATE_EVICT B300_STAGEM_PREPARED_MATE_GUARD \
  B300_STAGEM_PREPARED_BIN B300_STAGEM_PREPARED_THREADS B300_STAGEM_PREPARED_MANIFEST; do
  [[ -n "${!k+x}" ]] || { echo "Stage-M prepare missing $k" >&2; exit 3; }
done
[[ "$B300_STAGEM_PREPARED" == 1 && "$B300_STAGEM_PREPARED_MOD" == "$MOD" && "$B300_STAGEM_PREPARED_NGPU" == "$NGPU" ]] || { echo 'Stage-N/M modulus or GPU-count mismatch' >&2; exit 3; }
[[ -x "$B300_STAGEM_PREPARED_BIN" && -s "$B300_STAGEM_PREPARED_MANIFEST" ]] || exit 3
sha256sum -c "$B300_STAGEM_PREPARED_MANIFEST" >/dev/null || { echo 'Stage-M promotion manifest failed before Stage N' >&2; exit 3; }
CONTROL_BIN="$B300_STAGEM_PREPARED_BIN"; MP="$B300_STAGEM_PREPARED_POLICY"
SW="$B300_STAGEM_PREPARED_SELF_WIDTH"; SD="$B300_STAGEM_PREPARED_SELF_DISTANCE"; SE="$B300_STAGEM_PREPARED_SELF_EVICT"; SG="$B300_STAGEM_PREPARED_SELF_GUARD"
MW="$B300_STAGEM_PREPARED_MATE_WIDTH"; MD="$B300_STAGEM_PREPARED_MATE_DISTANCE"; ME="$B300_STAGEM_PREPARED_MATE_EVICT"; MG="$B300_STAGEM_PREPARED_MATE_GUARD"
case "$MP" in cg|cs) ;; *) echo 'Stage N requires promoted Stage-M cg/cs upstream' >&2; exit 4;; esac

# Winner env proves the upstream binary was already staged spill-free.
# shellcheck disable=SC1090
source "$STAGEM_WINNER_ENV"
for k in B300_STAGEM_STAGED_VALIDATED B300_STAGEM_FINAL_ENABLED B300_STAGEM_FINAL_SPILL_FREE B300_STAGEM_FINAL_BIN B300_STAGEM_FINAL_STAGE_ROWS B300_STAGEM_FINAL_STAGE_RESIDUE; do
  [[ -n "${!k+x}" ]] || { echo "Stage-M winner missing $k" >&2; exit 3; }
done
[[ "$B300_STAGEM_STAGED_VALIDATED" == 1 && "$B300_STAGEM_FINAL_ENABLED" == 1 && "$B300_STAGEM_FINAL_SPILL_FREE" == 1 && "$B300_STAGEM_FINAL_BIN" == "$CONTROL_BIN" ]] || { echo 'Stage-M spill-free winner/control proof mismatch' >&2; exit 3; }

BINS="$LOGDIR/binaries.tsv"
printf 'profile\tpair_policy\tblock_policy\tbinary\tbuild_err\n' >"$BINS"
printf 'dd\tdefault\tdefault\t%s\t-\n' "$CONTROL_BIN" >>"$BINS"
for pp in "${pair_policies[@]}"; do
  for bp in "${block_policies[@]}"; do
    [[ "$pp" == default && "$bp" == default ]] && continue
    profile="${pp:0:1}${bp:0:1}"
    # default/cg both begin with d/c, while cs also c; retain full names in binary.
    profile="p${pp}_b${bp}"
    bin="$ONEESAN_BUILD_DIR/b300_stagen_${profile}_mp${MP}_sw${SW}d${SD}_mw${MW}d${MD}_n27"
    err="$LOGDIR/${profile}.build.err"; out="$LOGDIR/${profile}.build.out"
    N=27 ARCH="$ARCH" OUT="$bin" MATE_LOAD_POLICY="$MP" PAIR_LOAD_POLICY="$pp" BLOCK_LOAD_POLICY="$bp" \
      STAGEN_CG_L2_FETCH_BYTES="$STAGEN_CG_L2_FETCH_BYTES" HIGH_DROP_CHUNK="$H" HYBRID_THRESHOLD="$T" \
      SELF_WIDTH="$SW" SELF_DISTANCE="$SD" MATE_WIDTH="$MW" MATE_DISTANCE="$MD" SELF_EVICT="$SE" MATE_EVICT="$ME" SELF_GUARD="$SG" MATE_GUARD="$MG" \
      RANDOM_CG="$CG" RANDOM_CG_L2_FETCH_BYTES="$CGL2" PREFETCH_L2="$PRE" DUALMASK="$DUAL" CLOSURE_BATCH="$BATCH" MAXRREGCOUNT="$CAP" \
      PTXAS_VERBOSE=1 BUILD_ERR="$err" bash "$BUILDER" >"$out" 2>"$LOGDIR/${profile}.driver.err"
    [[ -x "$bin" ]] || { echo "Stage-N binary missing profile=$profile" >&2; exit 3; }
    grep -Fq "pair_load_policy=$pp block_load_policy=$bp" "$out" || exit 3
    grep -Fq "mate_load_policy=$MP" "$out" || exit 3
    grep -Fq "self_geometry width=$SW distance=$SD evict=$SE guard=$SG" "$out" || exit 3
    grep -Fq "mate_geometry width=$MW distance=$MD evict=$ME guard=$MG" "$out" || exit 3
    printf '%s\t%s\t%s\t%s\t%s\n' "$profile" "$pp" "$bp" "$bin" "$err" >>"$BINS"
  done
done

printf 'profile\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
while IFS=$'\t' read -r profile pp bp bin err; do
  [[ "$profile" == profile || "$profile" == dd ]] && continue
  python3 "$PARSER" "$err" --label "$profile" --contains main_pull_kernel_ilp2 >>"$RESOURCE" || true
  python3 "$PARSER" "$err" --label "$profile" --contains main_pull_kernel_ilp8_hybrid >>"$RESOURCE" || true
done <"$BINS"

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
printf 'profile\tpair_policy\tblock_policy\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tforward_high_s\treverse_high_s\n' >"$RESULT"
run_one(){
  local profile="$1" pp="$2" bp="$3" bin="$4" th="$5" r="$6"
  local so="$LOGDIR/${profile}_t${th}_r${r}.out" se="$LOGDIR/${profile}_t${th}_r${r}.err" line
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$th" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { tail -n 80 "$se" >&2 || true; return 4; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$profile" "$pp" "$bp" "$th" "$r" \
    "$(field residue "$line")" "$(field wall_s "$line")" "$(field active_max_s "$line")" "$(field active_sum_s "$line")" "$(field forward_high_s "$line")" "$(field reverse_high_s "$line")" >>"$RESULT"
}
for th in "${threads[@]}"; do
  while IFS=$'\t' read -r profile pp bp bin err; do
    [[ "$profile" == profile ]] && continue
    for ((r=1;r<=REPEATS;++r)); do
      echo "=== Stage N profile=$profile pair=$pp block=$bp threads=$th repeat=$r rows=$ROWS ngpu=$NGPU ===" >&2
      run_one "$profile" "$pp" "$bp" "$bin" "$th" "$r"
    done
  done <"$BINS"
done

python3 - "$RESULT" "$RESOURCE" "$BINS" "$WINNER_ENV" "$ROWS" "$NGPU" "$MP" "$SW" "$SD" "$SE" "$SG" "$MW" "$MD" "$ME" "$MG" "$CONTROL_BIN" "$STAGEN_CG_L2_FETCH_BYTES" <<'PY'
import csv,math,statistics,sys,shlex
(result,resource,bins_path,winner,rows_arg,ngpu,mp,sw,sd,se,sg,mw,md,me,mg,control_bin,cgl2)=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t')); rr=list(csv.DictReader(open(resource),delimiter='\t')); bins={r['profile']:r for r in csv.DictReader(open(bins_path),delimiter='\t')}
if not rows: raise SystemExit('no Stage-N rows')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL Stage-N residue mismatch '+repr({(r['profile'],r['threads'],r['repeat']):r['residue'] for r in rows}))
resources={k:[] for k in bins}
for r in rr:
    try: resources[r['profile']].append((int(r['registers']),int(r['spill_store_bytes']),int(r['spill_load_bytes'])))
    except (ValueError,KeyError): pass
agg=[]
for profile,t in {(r['profile'],int(r['threads'])) for r in rows}:
    rs=[r for r in rows if r['profile']==profile and int(r['threads'])==t]
    wall=statistics.median(float(r['wall_s']) for r in rs)
    hs=[]
    for r in rs:
        try: hs.append(float(r['forward_high_s'])+float(r['reverse_high_s']))
        except ValueError: pass
    high=statistics.median(hs) if hs else math.nan
    if profile=='dd': regs=ss=sl=-1; clean=True
    else:
        rv=resources[profile]; regs=max((x[0] for x in rv),default=-1); ss=max((x[1] for x in rv),default=-1); sl=max((x[2] for x in rv),default=-1); clean=len(rv)>=2 and ss==0 and sl==0
    agg.append((wall,profile,t,high,regs,ss,sl,clean))
def rank(x): return (x[0],x[3] if not math.isnan(x[3]) else math.inf,x[4] if x[4]>=0 else math.inf,x[2])
for x in sorted(agg,key=rank): print(f'STAGE_N profile={x[1]} threads={x[2]} wall_s={x[0]:.9f} high_s={x[3]:.9f} regs={x[4]} spill_store={x[5]} spill_load={x[6]} spill_free={int(x[7])} ngpu={ngpu}',file=sys.stderr)
base=min((x for x in agg if x[1]=='dd'),key=rank)
tests=[x for x in agg if x[1]!='dd' and x[7]]
if not tests: raise SystemExit('Stage-N needs a spill-free alternate candidate')
best=min(tests,key=rank); meta=bins[best[1]]; speed=base[0]/best[0]; enabled=int(rank(best)<rank(base)); q=lambda x:shlex.quote(str(x))
vals={
 'B300_STAGEN_ROWS':rows_arg,'B300_STAGEN_NGPU':int(ngpu),'B300_STAGEN_RESIDUE':next(iter(res)),
 'B300_STAGEN_MATE_LOAD_POLICY':mp,'B300_STAGEN_SELF_WIDTH':int(sw),'B300_STAGEN_SELF_DISTANCE':int(sd),'B300_STAGEN_SELF_EVICT':se,'B300_STAGEN_SELF_GUARD':sg,
 'B300_STAGEN_MATE_WIDTH':int(mw),'B300_STAGEN_MATE_DISTANCE':int(md),'B300_STAGEN_MATE_EVICT':me,'B300_STAGEN_MATE_GUARD':mg,
 'B300_STAGEN_CG_L2_FETCH_BYTES':int(cgl2),
 'B300_STAGEN_BASE_PAIR_POLICY':'default','B300_STAGEN_BASE_BLOCK_POLICY':'default','B300_STAGEN_BASE_BIN':control_bin,'B300_STAGEN_BASE_THREADS':base[2],'B300_STAGEN_BASE_WALL_S':f'{base[0]:.9f}','B300_STAGEN_BASE_HIGH_S':f'{base[3]:.9f}','B300_STAGEN_BASE_SPILL_FREE':1,
 'B300_STAGEN_PROFILE':best[1],'B300_STAGEN_PAIR_POLICY':meta['pair_policy'],'B300_STAGEN_BLOCK_POLICY':meta['block_policy'],'B300_STAGEN_BIN':meta['binary'],'B300_STAGEN_THREADS':best[2],'B300_STAGEN_WALL_S':f'{best[0]:.9f}','B300_STAGEN_HIGH_S':f'{best[3]:.9f}','B300_STAGEN_SPILL_FREE':1,'B300_STAGEN_SPEEDUP':f'{speed:.9f}','B300_STAGEN_BEST_ENABLED':enabled,
}
with open(winner,'w') as f:
    for k,v in vals.items(): f.write(f'{k}={q(v)}\n')
print(f'b300_stagen_exact_match=1 b300_stagen_ngpu={ngpu} rows={rows_arg} residue={next(iter(res))} base_wall_s={base[0]:.9f} best={best[1]} pair={meta["pair_policy"]} block={meta["block_policy"]} wall_s={best[0]:.9f} speedup={speed:.9f} enabled={enabled} spill_free=1 cgl2={cgl2}')
PY
cat "$WINNER_ENV"
