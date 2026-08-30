#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; ROWS="${ROWS:-1}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
STAGEN_WINNER_ENV="${STAGEN_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_staged_g${NGPU}_winner.env}"
STAGEN_PREPARE_ENV="${STAGEN_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_fullprime_n27_prepared.env}"
STAGEO_PREPARE_ENV="${STAGEO_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageo_cg_l2_fullprime_n27_prepared.env}"
STAGEP_PREPARE_ENV="${STAGEP_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stagep_mate_cg_l2_fullprime_n27_prepared.env}"
STAGEQ_WINNER_ENV="${STAGEQ_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageq_ilp8_count_cg_l2_staged_g${NGPU}_winner.env}"
STAGEQ_PREPARE_ENV="${STAGEQ_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageq_ilp8_count_cg_l2_fullprime_n27_prepared.env}"
UPSTREAM_KIND="${UPSTREAM_KIND:-auto}"; PAIR_POLICY_LIST="${PAIR_POLICY_LIST:-default cg cs}"; BLOCK_POLICY_LIST="${BLOCK_POLICY_LIST:-default cg cs}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"; REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stager_ilp2_load_row${ROWS}_g${NGPU}}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-stager-ilp2-load-policy.sh"; PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")" "$(dirname "$WINNER_ENV")"
for f in "$STAGE_F_ENV" "$STAGEN_WINNER_ENV" "$STAGEN_PREPARE_ENV" "$BUILDER"; do [[ -s "$f" ]] || { echo "missing Stage-R input=$f" >&2; exit 2; }; done
case "$UPSTREAM_KIND" in auto|stagen|stageo|stagep|stageq) ;; *) echo 'UPSTREAM_KIND must be auto,stagen,stageo,stagep,stageq' >&2; exit 2;; esac
[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || exit 2; [[ "$NGPU" =~ ^[1-8]$ ]] || exit 2; [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || exit 2
norm_policy(){ local raw="$1" out=() p old seen; for p in $raw; do case "$p" in default|cg|cs) ;; *) echo "bad Stage-R policy=$p" >&2; exit 2;; esac; seen=0; for old in "${out[@]}"; do [[ "$old" == "$p" ]] && seen=1; done; ((seen)) || out+=("$p"); done; ((${#out[@]})) || exit 2; printf '%s' "${out[*]}"; }
PAIR_POLICY_LIST="$(norm_policy "$PAIR_POLICY_LIST")"; BLOCK_POLICY_LIST="$(norm_policy "$BLOCK_POLICY_LIST")"
threads=(); for th in $THREADS_LIST; do [[ "$th" =~ ^[0-9]+$ ]] && ((th>=32&&th<=1024&&th%32==0)) || exit 2; seen=0; for old in "${threads[@]}"; do [[ "$old" == "$th" ]] && seen=1; done; ((seen)) || threads+=("$th"); done; ((${#threads[@]})) || exit 2
command -v nvcc >/dev/null || exit 2; command -v nvidia-smi >/dev/null || exit 2; command -v sha256sum >/dev/null || exit 2
VISIBLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; ((VISIBLE_GPUS>=NGPU)) || { echo "need $NGPU visible GPU(s), found $VISIBLE_GPUS" >&2; exit 2; }

# Stage F fixes recurrence compile knobs.
# shellcheck disable=SC1090
source "$STAGE_F_ENV"
for k in B300_HYBRID8_NEXTSELF_THRESHOLD B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK B300_HYBRID8_NEXTSELF_RANDOM_CG B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES B300_HYBRID8_NEXTSELF_PREFETCH_L2 B300_HYBRID8_NEXTSELF_DUALMASK B300_HYBRID8_NEXTSELF_CLOSURE_BATCH B300_HYBRID8_NEXTSELF_MAXRREGCOUNT; do [[ -n "${!k+x}" ]] || { echo "Stage-F env missing $k" >&2; exit 3; }; done
T="$B300_HYBRID8_NEXTSELF_THRESHOLD"; H="$B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK"; CG="$B300_HYBRID8_NEXTSELF_RANDOM_CG"; CGL2="$B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES"; PRE="$B300_HYBRID8_NEXTSELF_PREFETCH_L2"; DUAL="$B300_HYBRID8_NEXTSELF_DUALMASK"; BATCH="$B300_HYBRID8_NEXTSELF_CLOSURE_BATCH"; CAP="$B300_HYBRID8_NEXTSELF_MAXRREGCOUNT"
# Stage N owns geometry and high-state load policies.
# shellcheck disable=SC1090
source "$STAGEN_WINNER_ENV"
for k in B300_STAGEN_STAGED_VALIDATED B300_STAGEN_FINAL_ENABLED B300_STAGEN_NGPU B300_STAGEN_PAIR_POLICY B300_STAGEN_BLOCK_POLICY B300_STAGEN_MATE_LOAD_POLICY B300_STAGEN_SELF_WIDTH B300_STAGEN_SELF_DISTANCE B300_STAGEN_SELF_EVICT B300_STAGEN_SELF_GUARD B300_STAGEN_MATE_WIDTH B300_STAGEN_MATE_DISTANCE B300_STAGEN_MATE_EVICT B300_STAGEN_MATE_GUARD B300_STAGEN_FINAL_STAGE_ROWS B300_STAGEN_FINAL_STAGE_RESIDUE; do [[ -n "${!k+x}" ]] || { echo "Stage-N winner missing $k" >&2; exit 3; }; done
[[ "$B300_STAGEN_STAGED_VALIDATED" == 1 && "$B300_STAGEN_FINAL_ENABLED" == 1 && "$B300_STAGEN_NGPU" == "$NGPU" ]] || { echo 'Stage N not promotable for Stage R' >&2; exit 4; }
PAIR_POLICY="$B300_STAGEN_PAIR_POLICY"; BLOCK_POLICY="$B300_STAGEN_BLOCK_POLICY"; MATE_POLICY="$B300_STAGEN_MATE_LOAD_POLICY"; SW="$B300_STAGEN_SELF_WIDTH"; SD="$B300_STAGEN_SELF_DISTANCE"; SE="$B300_STAGEN_SELF_EVICT"; SG="$B300_STAGEN_SELF_GUARD"; MW="$B300_STAGEN_MATE_WIDTH"; MD="$B300_STAGEN_MATE_DISTANCE"; ME="$B300_STAGEN_MATE_EVICT"; MG="$B300_STAGEN_MATE_GUARD"
# shellcheck disable=SC1090
source "$STAGEN_PREPARE_ENV"
[[ "${B300_STAGEN_PREPARED:-0}" == 1 && "${B300_STAGEN_PREPARED_MOD:-}" == "$MOD" && "${B300_STAGEN_PREPARED_NGPU:-0}" == "$NGPU" ]] || exit 3
sha256sum -c "$B300_STAGEN_PREPARED_MANIFEST" >/dev/null || exit 3

RESOLVED=stagen; CONTROL_BIN="$B300_STAGEN_PREPARED_BIN"; CONTROL_THREADS="$B300_STAGEN_PREPARED_THREADS"; UP_ROWS="$B300_STAGEN_PREPARED_FINAL_STAGE_ROWS"; UP_RES="$B300_STAGEN_PREPARED_FINAL_STAGE_RESIDUE"; UP_MANIFEST="$B300_STAGEN_PREPARED_MANIFEST"
STAGEQ_UPSTREAM=stagen; STAGEP_COUNT=stagen; BUILDER_PAIR_L2=0; BUILDER_BLOCK_L2=0; MATE_L2=0; Q_PAIR_L2=0; Q_BLOCK_L2=0
EFF_PAIR_L2=0; [[ "$PAIR_POLICY" == cg ]] && EFF_PAIR_L2="$CGL2"; EFF_BLOCK_L2=0; [[ "$BLOCK_POLICY" == cg ]] && EFF_BLOCK_L2="$CGL2"
O_VALID=0; if [[ -s "$STAGEO_PREPARE_ENV" ]] && grep -Fq 'B300_STAGEO_PREPARED=1' "$STAGEO_PREPARE_ENV"; then O_VALID=1; fi
P_VALID=0; if [[ -s "$STAGEP_PREPARE_ENV" ]] && grep -Fq 'B300_STAGEP_PREPARED=1' "$STAGEP_PREPARE_ENV"; then P_VALID=1; fi
Q_VALID=0; if [[ -s "$STAGEQ_WINNER_ENV" && -s "$STAGEQ_PREPARE_ENV" ]] && grep -Fq 'B300_STAGEQ_STAGED_VALIDATED=1' "$STAGEQ_WINNER_ENV" && grep -Fq 'B300_STAGEQ_PREPARED=1' "$STAGEQ_PREPARE_ENV"; then Q_VALID=1; fi
if [[ "$UPSTREAM_KIND" == stageo || "$UPSTREAM_KIND" == stagep || "$UPSTREAM_KIND" == stageq || ( "$UPSTREAM_KIND" == auto && "$O_VALID" == 1 ) ]]; then
  if ((O_VALID)); then source "$STAGEO_PREPARE_ENV"; sha256sum -c "$B300_STAGEO_PREPARED_MANIFEST" >/dev/null || exit 3; RESOLVED=stageo; CONTROL_BIN="$B300_STAGEO_PREPARED_BIN"; CONTROL_THREADS="$B300_STAGEO_PREPARED_THREADS"; UP_ROWS="$B300_STAGEO_PREPARED_FINAL_STAGE_ROWS"; UP_RES="$B300_STAGEO_PREPARED_FINAL_STAGE_RESIDUE"; UP_MANIFEST="$B300_STAGEO_PREPARED_MANIFEST"; BUILDER_PAIR_L2="$B300_STAGEO_PREPARED_PAIR_L2_BYTES"; BUILDER_BLOCK_L2="$B300_STAGEO_PREPARED_BLOCK_L2_BYTES"; EFF_PAIR_L2="$B300_STAGEO_PREPARED_PAIR_L2_BYTES"; EFF_BLOCK_L2="$B300_STAGEO_PREPARED_BLOCK_L2_BYTES"; fi
fi
if [[ "$UPSTREAM_KIND" == stagep || "$UPSTREAM_KIND" == stageq || ( "$UPSTREAM_KIND" == auto && "$P_VALID" == 1 ) ]]; then
  if ((P_VALID)); then source "$STAGEP_PREPARE_ENV"; sha256sum -c "$B300_STAGEP_PREPARED_MANIFEST" >/dev/null || exit 3; RESOLVED=stagep; CONTROL_BIN="$B300_STAGEP_PREPARED_BIN"; CONTROL_THREADS="$B300_STAGEP_PREPARED_THREADS"; UP_ROWS="$B300_STAGEP_PREPARED_FINAL_STAGE_ROWS"; UP_RES="$B300_STAGEP_PREPARED_FINAL_STAGE_RESIDUE"; UP_MANIFEST="$B300_STAGEP_PREPARED_MANIFEST"; STAGEP_COUNT="$B300_STAGEP_PREPARED_COUNT_UPSTREAM"; MATE_POLICY=cg; MATE_L2="$B300_STAGEP_PREPARED_MATE_L2_BYTES"; if [[ "$STAGEP_COUNT" == stageo ]]; then BUILDER_PAIR_L2="$B300_STAGEP_PREPARED_PAIR_CG_L2_BYTES"; BUILDER_BLOCK_L2="$B300_STAGEP_PREPARED_BLOCK_CG_L2_BYTES"; EFF_PAIR_L2="$BUILDER_PAIR_L2"; EFF_BLOCK_L2="$BUILDER_BLOCK_L2"; else BUILDER_PAIR_L2=0; BUILDER_BLOCK_L2=0; EFF_PAIR_L2=0; [[ "$PAIR_POLICY" == cg ]] && EFF_PAIR_L2="$CGL2"; EFF_BLOCK_L2=0; [[ "$BLOCK_POLICY" == cg ]] && EFF_BLOCK_L2="$CGL2"; fi; fi
fi
if [[ "$UPSTREAM_KIND" == stageq || ( "$UPSTREAM_KIND" == auto && "$Q_VALID" == 1 ) ]]; then
  ((Q_VALID)) || { echo 'Stage R requested Stage Q but unavailable' >&2; exit 3; }
  source "$STAGEQ_WINNER_ENV"; source "$STAGEQ_PREPARE_ENV"; sha256sum -c "$B300_STAGEQ_PREPARED_MANIFEST" >/dev/null || exit 3
  RESOLVED=stageq; CONTROL_BIN="$B300_STAGEQ_PREPARED_BIN"; CONTROL_THREADS="$B300_STAGEQ_PREPARED_THREADS"; UP_ROWS="$B300_STAGEQ_PREPARED_FINAL_STAGE_ROWS"; UP_RES="$B300_STAGEQ_PREPARED_FINAL_STAGE_RESIDUE"; UP_MANIFEST="$B300_STAGEQ_PREPARED_MANIFEST"; STAGEQ_UPSTREAM="$B300_STAGEQ_PREPARED_UPSTREAM_KIND"; Q_PAIR_L2="$B300_STAGEQ_PREPARED_PAIR_L2_BYTES"; Q_BLOCK_L2="$B300_STAGEQ_PREPARED_BLOCK_L2_BYTES"; EFF_PAIR_L2="$Q_PAIR_L2"; EFF_BLOCK_L2="$Q_BLOCK_L2"; STAGEP_COUNT="$B300_STAGEQ_PREPARED_STAGEP_COUNT_UPSTREAM"; BUILDER_PAIR_L2="$B300_STAGEQ_PREPARED_BUILDER_PAIR_CG_L2_BYTES"; BUILDER_BLOCK_L2="$B300_STAGEQ_PREPARED_BUILDER_BLOCK_CG_L2_BYTES"; MATE_POLICY="$B300_STAGEQ_PREPARED_MATE_LOAD_POLICY"; MATE_L2="$B300_STAGEQ_PREPARED_MATE_CG_L2_BYTES"
fi
case "$UPSTREAM_KIND" in stagen) RESOLVED=stagen;; stageo) [[ "$RESOLVED" == stageo ]] || exit 3;; stagep) [[ "$RESOLVED" == stagep ]] || exit 3;; stageq) [[ "$RESOLVED" == stageq ]] || exit 3;; esac
[[ -x "$CONTROL_BIN" ]] || exit 3
# Exact upstream ILP2 policy is Stage-N pair/block policy because O/P/Q leave ILP2 unchanged.
BASE_PAIR="$PAIR_POLICY"; BASE_BLOCK="$BLOCK_POLICY"
case " $PAIR_POLICY_LIST " in *" $BASE_PAIR "*) ;; *) echo "PAIR_POLICY_LIST omits exact upstream $BASE_PAIR" >&2; exit 2;; esac
case " $BLOCK_POLICY_LIST " in *" $BASE_BLOCK "*) ;; *) echo "BLOCK_POLICY_LIST omits exact upstream $BASE_BLOCK" >&2; exit 2;; esac

BINS="$LOGDIR/binaries.tsv"; printf 'label\tpair_policy\tblock_policy\tbinary\tbuild_err\n' >"$BINS"; printf 'control\t%s\t%s\t%s\t-\n' "$BASE_PAIR" "$BASE_BLOCK" "$CONTROL_BIN" >>"$BINS"
for lp in $PAIR_POLICY_LIST; do for lb in $BLOCK_POLICY_LIST; do
  [[ "$lp" == "$BASE_PAIR" && "$lb" == "$BASE_BLOCK" ]] && continue
  label="p${lp}_b${lb}"; bin="$ONEESAN_BUILD_DIR/b300_stager_${label}_${RESOLVED}_n27"; err="$LOGDIR/${label}.build.err"; bout="$LOGDIR/${label}.build.out"
  N=27 ARCH="$ARCH" OUT="$bin" UPSTREAM_KIND="$RESOLVED" STAGEQ_UPSTREAM_KIND="$STAGEQ_UPSTREAM" STAGEP_COUNT_UPSTREAM="$STAGEP_COUNT" MATE_LOAD_POLICY="$MATE_POLICY" PAIR_LOAD_POLICY="$PAIR_POLICY" BLOCK_LOAD_POLICY="$BLOCK_POLICY" ILP2_PAIR_LOAD_POLICY="$lp" ILP2_BLOCK_LOAD_POLICY="$lb" BASE_CG_L2_BYTES="$CGL2" PAIR_CG_L2_BYTES="$BUILDER_PAIR_L2" BLOCK_CG_L2_BYTES="$BUILDER_BLOCK_L2" MATE_CG_L2_BYTES="$MATE_L2" ILP8_PAIR_CG_L2_BYTES="$Q_PAIR_L2" ILP8_BLOCK_CG_L2_BYTES="$Q_BLOCK_L2" HIGH_DROP_CHUNK="$H" HYBRID_THRESHOLD="$T" SELF_WIDTH="$SW" SELF_DISTANCE="$SD" MATE_WIDTH="$MW" MATE_DISTANCE="$MD" SELF_EVICT="$SE" MATE_EVICT="$ME" SELF_GUARD="$SG" MATE_GUARD="$MG" RANDOM_CG="$CG" RANDOM_CG_L2_FETCH_BYTES="$CGL2" PREFETCH_L2="$PRE" DUALMASK="$DUAL" CLOSURE_BATCH="$BATCH" MAXRREGCOUNT="$CAP" PTXAS_VERBOSE=1 BUILD_ERR="$err" bash "$BUILDER" >"$bout" 2>"$LOGDIR/${label}.driver.err"
  [[ -x "$bin" ]] || exit 3; grep -Fq "stage_r_ilp2_load_policy=1 upstream_kind=$RESOLVED" "$bout" || exit 3; grep -Fq "ilp2_pair_load_policy=$lp ilp2_block_load_policy=$lb" "$bout" || exit 3; grep -Fq 'stage_r_scope=ilp2_pair_block_load_policy_only ilp8_exact_upstream=1' "$bout" || exit 3
  printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$lp" "$lb" "$bin" "$err" >>"$BINS"
done; done
printf 'label\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
while IFS=$'\t' read -r label lp lb bin err; do [[ "$label" == label || "$label" == control ]] && continue; python3 "$PARSER" "$err" --label "$label" --contains main_pull_kernel_ilp2 >>"$RESOURCE" || true; python3 "$PARSER" "$err" --label "$label" --contains main_pull_kernel_ilp8_hybrid >>"$RESOURCE" || true; done <"$BINS"
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
printf 'label\tpair_policy\tblock_policy\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tforward_high_s\treverse_high_s\n' >"$RESULT"
run_one(){ local label="$1" lp="$2" lb="$3" bin="$4" th="$5" r="$6" so="$LOGDIR/${label}_t${th}_r${r}.out" se="$LOGDIR/${label}_t${th}_r${r}.err" line; B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$th" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"; line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || return 4; printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$lp" "$lb" "$th" "$r" "$(field residue "$line")" "$(field wall_s "$line")" "$(field active_max_s "$line")" "$(field active_sum_s "$line")" "$(field forward_high_s "$line")" "$(field reverse_high_s "$line")" >>"$RESULT"; }
while IFS=$'\t' read -r label lp lb bin err; do [[ "$label" == label ]] && continue; for th in "${threads[@]}"; do for ((r=1;r<=REPEATS;++r)); do echo "=== Stage R label=$label low_pair=$lp low_block=$lb threads=$th rows=$ROWS upstream=$RESOLVED ===" >&2; run_one "$label" "$lp" "$lb" "$bin" "$th" "$r"; done; done; done <"$BINS"

python3 - "$RESULT" "$RESOURCE" "$BINS" "$WINNER_ENV" "$ROWS" "$NGPU" "$RESOLVED" "$STAGEQ_UPSTREAM" "$BASE_PAIR" "$BASE_BLOCK" "$PAIR_POLICY" "$BLOCK_POLICY" "$EFF_PAIR_L2" "$EFF_BLOCK_L2" "$CONTROL_THREADS" "$UP_ROWS" "$UP_RES" "$UP_MANIFEST" <<'PY'
import csv,math,statistics,sys,shlex
result,resource,bins_path,winner,rows_arg,ngpu,upstream,q_up,base_pair,base_block,high_pair,high_block,high_pl2,high_bl2,control_src_threads,up_rows,up_res,up_manifest=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t')); rr=list(csv.DictReader(open(resource),delimiter='\t')); bins={r['label']:r for r in csv.DictReader(open(bins_path),delimiter='\t')}
if not rows: raise SystemExit('no Stage-R rows')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL Stage-R residue mismatch '+repr({(r['label'],r['threads'],r['repeat']):r['residue'] for r in rows}))
only_res=next(iter(res))
if rows_arg==up_rows and only_res!=up_res:
    raise SystemExit(f'FATAL Stage-R/upstream residue mismatch rows={rows_arg} got={only_res} expected={up_res}')
resources={k:[] for k in bins}
for r in rr:
    try: resources[r['label']].append((int(r['registers']),int(r['spill_store_bytes']),int(r['spill_load_bytes'])))
    except (ValueError,KeyError): pass
agg=[]
for label,t in {(r['label'],int(r['threads'])) for r in rows}:
    rs=[r for r in rows if r['label']==label and int(r['threads'])==t]; wall=statistics.median(float(r['wall_s']) for r in rs); hs=[]
    for r in rs:
        try: hs.append(float(r['forward_high_s'])+float(r['reverse_high_s']))
        except ValueError: pass
    high=statistics.median(hs) if hs else math.inf
    if label=='control': regs=-1; ss=sl=0; clean=True
    else:
        rv=resources[label]; regs=max((x[0] for x in rv),default=-1); ss=max((x[1] for x in rv),default=-1); sl=max((x[2] for x in rv),default=-1); clean=len(rv)>=2 and ss==0 and sl==0
    agg.append((wall,label,t,high,regs,ss,sl,clean))
def rank(x): return (x[0],x[3],x[4] if x[4]>=0 else math.inf,x[2])
base=min((x for x in agg if x[1]=='control'),key=rank); tests=[x for x in agg if x[1]!='control' and x[7]]; best=min(tests,key=rank) if tests else base; speed=base[0]/best[0]; enabled=int(best[1]!='control' and rank(best)<rank(base)); bb=bins[best[1]]; q=lambda x:shlex.quote(str(x))
for x in sorted(agg,key=rank):
    b=bins[x[1]]; print(f'STAGE_R label={x[1]} pair={b["pair_policy"]} block={b["block_policy"]} threads={x[2]} wall_s={x[0]:.9f} high_s={x[3]:.9f} regs={x[4]} spill_store={x[5]} spill_load={x[6]} spill_free={int(x[7])}',file=sys.stderr)
vals={'B300_STAGER_ROWS':rows_arg,'B300_STAGER_NGPU':ngpu,'B300_STAGER_RESIDUE':only_res,'B300_STAGER_UPSTREAM_KIND':upstream,'B300_STAGER_STAGEQ_UPSTREAM_KIND':q_up,'B300_STAGER_HIGH_PAIR_POLICY':high_pair,'B300_STAGER_HIGH_BLOCK_POLICY':high_block,'B300_STAGER_HIGH_PAIR_L2_BYTES':high_pl2,'B300_STAGER_HIGH_BLOCK_L2_BYTES':high_bl2,'B300_STAGER_UPSTREAM_PAIR_POLICY':base_pair,'B300_STAGER_UPSTREAM_BLOCK_POLICY':base_block,'B300_STAGER_CONTROL_BIN':bins['control']['binary'],'B300_STAGER_CONTROL_THREADS':base[2],'B300_STAGER_CONTROL_SOURCE_THREADS':control_src_threads,'B300_STAGER_CONTROL_WALL_S':f'{base[0]:.9f}','B300_STAGER_CONTROL_HIGH_S':f'{base[3]:.9f}','B300_STAGER_CONTROL_SPILL_FREE':1,'B300_STAGER_PAIR_POLICY':bb['pair_policy'],'B300_STAGER_BLOCK_POLICY':bb['block_policy'],'B300_STAGER_BIN':bb['binary'],'B300_STAGER_THREADS':best[2],'B300_STAGER_WALL_S':f'{best[0]:.9f}','B300_STAGER_HIGH_S':f'{best[3]:.9f}','B300_STAGER_SPILL_FREE':int(best[7]),'B300_STAGER_SPEEDUP':f'{speed:.9f}','B300_STAGER_BEST_ENABLED':enabled,'B300_STAGER_UPSTREAM_ROWS':up_rows,'B300_STAGER_UPSTREAM_RESIDUE':up_res,'B300_STAGER_UPSTREAM_MANIFEST':up_manifest}
with open(winner,'w') as f:
    for k,v in vals.items(): f.write(f'{k}={q(v)}\n')
print(f'b300_stager_exact_match=1 b300_stager_ngpu={ngpu} upstream={upstream} base_pair={base_pair} base_block={base_block} best_pair={bb["pair_policy"]} best_block={bb["block_policy"]} speedup={speed:.9f} residue={only_res}')
PY
cat "$WINNER_ENV"
