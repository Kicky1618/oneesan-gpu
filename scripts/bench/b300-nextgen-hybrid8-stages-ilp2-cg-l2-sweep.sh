#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; ROWS="${ROWS:-1}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
STAGEN_WINNER_ENV="${STAGEN_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_staged_g${NGPU}_winner.env}"
STAGEN_PREPARE_ENV="${STAGEN_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_fullprime_n27_prepared.env}"
STAGEO_PREPARE_ENV="${STAGEO_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageo_cg_l2_fullprime_n27_prepared.env}"
STAGEP_PREPARE_ENV="${STAGEP_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stagep_mate_cg_l2_fullprime_n27_prepared.env}"
STAGEQ_PREPARE_ENV="${STAGEQ_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageq_ilp8_count_cg_l2_fullprime_n27_prepared.env}"
STAGER_WINNER_ENV="${STAGER_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stager_ilp2_load_staged_g${NGPU}_winner.env}"
STAGER_PREPARE_ENV="${STAGER_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stager_ilp2_load_fullprime_n27_prepared.env}"
PAIR_L2_LIST="${PAIR_L2_LIST:-0 64 128 256}"; BLOCK_L2_LIST="${BLOCK_L2_LIST:-0 64 128 256}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"; REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stages_ilp2_cg_l2_row${ROWS}_g${NGPU}}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-stages-ilp2-cg-l2-policy.sh"; PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")" "$(dirname "$WINNER_ENV")"
for f in "$STAGE_F_ENV" "$STAGEN_WINNER_ENV" "$STAGEN_PREPARE_ENV" "$STAGER_WINNER_ENV" "$STAGER_PREPARE_ENV" "$BUILDER"; do [[ -s "$f" ]] || { echo "missing Stage-S input=$f" >&2; exit 2; }; done
[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || exit 2; [[ "$NGPU" =~ ^[1-8]$ ]] || exit 2; [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || exit 2
norm_l2(){ local raw="$1" out=() b old seen; for b in $raw; do case "$b" in 0|64|128|256) ;; *) echo "bad Stage-S L2 bytes=$b" >&2; exit 2;; esac; seen=0; for old in "${out[@]}"; do [[ "$old" == "$b" ]] && seen=1; done; ((seen)) || out+=("$b"); done; ((${#out[@]})) || exit 2; printf '%s' "${out[*]}"; }
PAIR_L2_LIST="$(norm_l2 "$PAIR_L2_LIST")"; BLOCK_L2_LIST="$(norm_l2 "$BLOCK_L2_LIST")"
threads=(); for th in $THREADS_LIST; do [[ "$th" =~ ^[0-9]+$ ]] && ((th>=32&&th<=1024&&th%32==0)) || exit 2; seen=0; for old in "${threads[@]}"; do [[ "$old" == "$th" ]] && seen=1; done; ((seen)) || threads+=("$th"); done; ((${#threads[@]})) || exit 2
command -v nvcc >/dev/null || exit 2; command -v nvidia-smi >/dev/null || exit 2; command -v sha256sum >/dev/null || exit 2
VISIBLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; ((VISIBLE_GPUS>=NGPU)) || { echo "need $NGPU visible GPU(s), found $VISIBLE_GPUS" >&2; exit 2; }

# Stage F fixes recurrence compile knobs.
# shellcheck disable=SC1090
source "$STAGE_F_ENV"
for k in B300_HYBRID8_NEXTSELF_THRESHOLD B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK B300_HYBRID8_NEXTSELF_RANDOM_CG B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES B300_HYBRID8_NEXTSELF_PREFETCH_L2 B300_HYBRID8_NEXTSELF_DUALMASK B300_HYBRID8_NEXTSELF_CLOSURE_BATCH B300_HYBRID8_NEXTSELF_MAXRREGCOUNT; do [[ -n "${!k+x}" ]] || { echo "Stage-F env missing $k" >&2; exit 3; }; done
T="$B300_HYBRID8_NEXTSELF_THRESHOLD"; H="$B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK"; CG="$B300_HYBRID8_NEXTSELF_RANDOM_CG"; CGL2="$B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES"; PRE="$B300_HYBRID8_NEXTSELF_PREFETCH_L2"; DUAL="$B300_HYBRID8_NEXTSELF_DUALMASK"; BATCH="$B300_HYBRID8_NEXTSELF_CLOSURE_BATCH"; CAP="$B300_HYBRID8_NEXTSELF_MAXRREGCOUNT"
# Stage N owns geometry, high-state policy, and mate policy baseline.
# shellcheck disable=SC1090
source "$STAGEN_WINNER_ENV"
for k in B300_STAGEN_STAGED_VALIDATED B300_STAGEN_FINAL_ENABLED B300_STAGEN_NGPU B300_STAGEN_PAIR_POLICY B300_STAGEN_BLOCK_POLICY B300_STAGEN_MATE_LOAD_POLICY B300_STAGEN_SELF_WIDTH B300_STAGEN_SELF_DISTANCE B300_STAGEN_SELF_EVICT B300_STAGEN_SELF_GUARD B300_STAGEN_MATE_WIDTH B300_STAGEN_MATE_DISTANCE B300_STAGEN_MATE_EVICT B300_STAGEN_MATE_GUARD; do [[ -n "${!k+x}" ]] || { echo "Stage-N winner missing $k" >&2; exit 3; }; done
[[ "$B300_STAGEN_STAGED_VALIDATED" == 1 && "$B300_STAGEN_FINAL_ENABLED" == 1 && "$B300_STAGEN_NGPU" == "$NGPU" ]] || { echo 'Stage N not promotable for Stage S' >&2; exit 4; }
HIGH_PAIR="$B300_STAGEN_PAIR_POLICY"; HIGH_BLOCK="$B300_STAGEN_BLOCK_POLICY"; MATE_POLICY="$B300_STAGEN_MATE_LOAD_POLICY"; SW="$B300_STAGEN_SELF_WIDTH"; SD="$B300_STAGEN_SELF_DISTANCE"; SE="$B300_STAGEN_SELF_EVICT"; SG="$B300_STAGEN_SELF_GUARD"; MW="$B300_STAGEN_MATE_WIDTH"; MD="$B300_STAGEN_MATE_DISTANCE"; ME="$B300_STAGEN_MATE_EVICT"; MG="$B300_STAGEN_MATE_GUARD"
# Verify Stage N base promotion as a provenance root.
# shellcheck disable=SC1090
source "$STAGEN_PREPARE_ENV"
[[ "${B300_STAGEN_PREPARED:-0}" == 1 && "${B300_STAGEN_PREPARED_MOD:-}" == "$MOD" && "${B300_STAGEN_PREPARED_NGPU:-0}" == "$NGPU" ]] || exit 3
sha256sum -c "$B300_STAGEN_PREPARED_MANIFEST" >/dev/null || exit 3

# Stage R is the exact Stage-S control. S is only a load-hint refinement on R's ILP2 cg lanes.
# shellcheck disable=SC1090
source "$STAGER_WINNER_ENV"
for k in B300_STAGER_STAGED_VALIDATED B300_STAGER_FINAL_ENABLED B300_STAGER_NGPU B300_STAGER_UPSTREAM_KIND B300_STAGER_HIGH_PAIR_POLICY B300_STAGER_HIGH_BLOCK_POLICY B300_STAGER_HIGH_PAIR_L2_BYTES B300_STAGER_HIGH_BLOCK_L2_BYTES B300_STAGER_PAIR_POLICY B300_STAGER_BLOCK_POLICY B300_STAGER_FINAL_SPILL_FREE B300_STAGER_FINAL_STAGE_ROWS B300_STAGER_FINAL_STAGE_RESIDUE; do [[ -n "${!k+x}" ]] || { echo "Stage-R winner missing $k" >&2; exit 3; }; done
[[ "$B300_STAGER_STAGED_VALIDATED" == 1 && "$B300_STAGER_FINAL_ENABLED" == 1 && "$B300_STAGER_FINAL_SPILL_FREE" == 1 && "$B300_STAGER_NGPU" == "$NGPU" ]] || { echo 'Stage R not promotable for Stage S' >&2; exit 4; }
R_UP="$B300_STAGER_UPSTREAM_KIND"; LOW_PAIR="$B300_STAGER_PAIR_POLICY"; LOW_BLOCK="$B300_STAGER_BLOCK_POLICY"; R_HIGH_PAIR="$B300_STAGER_HIGH_PAIR_POLICY"; R_HIGH_BLOCK="$B300_STAGER_HIGH_BLOCK_POLICY"; R_HIGH_PL2="$B300_STAGER_HIGH_PAIR_L2_BYTES"; R_HIGH_BL2="$B300_STAGER_HIGH_BLOCK_L2_BYTES"
[[ "$R_HIGH_PAIR" == "$HIGH_PAIR" && "$R_HIGH_BLOCK" == "$HIGH_BLOCK" ]] || { echo 'Stage-R high policy drift from Stage N' >&2; exit 3; }
[[ "$LOW_PAIR" == cg || "$LOW_BLOCK" == cg ]] || { echo 'Stage S not applicable: Stage-R selected no ILP2 cg axis' >&2; exit 4; }
if [[ "$LOW_PAIR" == cg ]]; then case " $PAIR_L2_LIST " in *' 0 '*) ;; *) echo 'PAIR_L2_LIST must include exact Stage-R baseline 0' >&2; exit 2;; esac; else PAIR_L2_LIST=0; fi
if [[ "$LOW_BLOCK" == cg ]]; then case " $BLOCK_L2_LIST " in *' 0 '*) ;; *) echo 'BLOCK_L2_LIST must include exact Stage-R baseline 0' >&2; exit 2;; esac; else BLOCK_L2_LIST=0; fi
# shellcheck disable=SC1090
source "$STAGER_PREPARE_ENV"
for k in B300_STAGER_PREPARED B300_STAGER_PREPARED_MOD B300_STAGER_PREPARED_NGPU B300_STAGER_PREPARED_UPSTREAM_KIND B300_STAGER_PREPARED_HIGH_PAIR_POLICY B300_STAGER_PREPARED_HIGH_BLOCK_POLICY B300_STAGER_PREPARED_HIGH_PAIR_L2_BYTES B300_STAGER_PREPARED_HIGH_BLOCK_L2_BYTES B300_STAGER_PREPARED_PAIR_POLICY B300_STAGER_PREPARED_BLOCK_POLICY B300_STAGER_PREPARED_BIN B300_STAGER_PREPARED_THREADS B300_STAGER_PREPARED_FINAL_STAGE_ROWS B300_STAGER_PREPARED_FINAL_STAGE_RESIDUE B300_STAGER_PREPARED_MANIFEST; do [[ -n "${!k+x}" ]] || { echo "Stage-R prepare missing $k" >&2; exit 3; }; done
[[ "$B300_STAGER_PREPARED" == 1 && "$B300_STAGER_PREPARED_MOD" == "$MOD" && "$B300_STAGER_PREPARED_NGPU" == "$NGPU" && "$B300_STAGER_PREPARED_UPSTREAM_KIND" == "$R_UP" ]] || exit 3
[[ "$B300_STAGER_PREPARED_PAIR_POLICY" == "$LOW_PAIR" && "$B300_STAGER_PREPARED_BLOCK_POLICY" == "$LOW_BLOCK" ]] || { echo 'Stage-R prepared low policy drift' >&2; exit 3; }
[[ "$B300_STAGER_PREPARED_HIGH_PAIR_POLICY" == "$R_HIGH_PAIR" && "$B300_STAGER_PREPARED_HIGH_BLOCK_POLICY" == "$R_HIGH_BLOCK" && "$B300_STAGER_PREPARED_HIGH_PAIR_L2_BYTES" == "$R_HIGH_PL2" && "$B300_STAGER_PREPARED_HIGH_BLOCK_L2_BYTES" == "$R_HIGH_BL2" ]] || { echo 'Stage-R prepared high-state drift' >&2; exit 3; }
sha256sum -c "$B300_STAGER_PREPARED_MANIFEST" >/dev/null || { echo 'Stage-R manifest failed before Stage S' >&2; exit 3; }
CONTROL_BIN="$B300_STAGER_PREPARED_BIN"; CONTROL_THREADS="$B300_STAGER_PREPARED_THREADS"; UP_ROWS="$B300_STAGER_PREPARED_FINAL_STAGE_ROWS"; UP_RES="$B300_STAGER_PREPARED_FINAL_STAGE_RESIDUE"; UP_MANIFEST="$B300_STAGER_PREPARED_MANIFEST"
[[ -x "$CONTROL_BIN" ]] || exit 3
[[ "$UP_ROWS" == "$B300_STAGER_FINAL_STAGE_ROWS" && "$UP_RES" == "$B300_STAGER_FINAL_STAGE_RESIDUE" ]] || { echo 'Stage-R winner/prepare residue provenance drift' >&2; exit 3; }

# Reconstruct the exact high-state/mate cache lineage used by the Stage-R binary.
STAGEQ_UPSTREAM=stagen; STAGEP_COUNT=stagen; BUILDER_PAIR_L2=0; BUILDER_BLOCK_L2=0; MATE_L2=0; Q_PAIR_L2=0; Q_BLOCK_L2=0
EFF_PAIR_L2=0; [[ "$HIGH_PAIR" == cg ]] && EFF_PAIR_L2="$CGL2"; EFF_BLOCK_L2=0; [[ "$HIGH_BLOCK" == cg ]] && EFF_BLOCK_L2="$CGL2"
case "$R_UP" in
  stagen) ;;
  stageo)
    [[ -s "$STAGEO_PREPARE_ENV" ]] || exit 3; source "$STAGEO_PREPARE_ENV"; sha256sum -c "$B300_STAGEO_PREPARED_MANIFEST" >/dev/null || exit 3
    BUILDER_PAIR_L2="$B300_STAGEO_PREPARED_PAIR_L2_BYTES"; BUILDER_BLOCK_L2="$B300_STAGEO_PREPARED_BLOCK_L2_BYTES"; EFF_PAIR_L2="$BUILDER_PAIR_L2"; EFF_BLOCK_L2="$BUILDER_BLOCK_L2"
    ;;
  stagep)
    [[ -s "$STAGEP_PREPARE_ENV" ]] || exit 3; source "$STAGEP_PREPARE_ENV"; sha256sum -c "$B300_STAGEP_PREPARED_MANIFEST" >/dev/null || exit 3
    STAGEP_COUNT="$B300_STAGEP_PREPARED_COUNT_UPSTREAM"; MATE_POLICY=cg; MATE_L2="$B300_STAGEP_PREPARED_MATE_L2_BYTES"
    if [[ "$STAGEP_COUNT" == stageo ]]; then BUILDER_PAIR_L2="$B300_STAGEP_PREPARED_PAIR_CG_L2_BYTES"; BUILDER_BLOCK_L2="$B300_STAGEP_PREPARED_BLOCK_CG_L2_BYTES"; EFF_PAIR_L2="$BUILDER_PAIR_L2"; EFF_BLOCK_L2="$BUILDER_BLOCK_L2"; fi
    ;;
  stageq)
    [[ -s "$STAGEQ_PREPARE_ENV" ]] || exit 3; source "$STAGEQ_PREPARE_ENV"; sha256sum -c "$B300_STAGEQ_PREPARED_MANIFEST" >/dev/null || exit 3
    STAGEQ_UPSTREAM="$B300_STAGEQ_PREPARED_UPSTREAM_KIND"; STAGEP_COUNT="$B300_STAGEQ_PREPARED_STAGEP_COUNT_UPSTREAM"; BUILDER_PAIR_L2="$B300_STAGEQ_PREPARED_BUILDER_PAIR_CG_L2_BYTES"; BUILDER_BLOCK_L2="$B300_STAGEQ_PREPARED_BUILDER_BLOCK_CG_L2_BYTES"; MATE_POLICY="$B300_STAGEQ_PREPARED_MATE_LOAD_POLICY"; MATE_L2="$B300_STAGEQ_PREPARED_MATE_CG_L2_BYTES"; Q_PAIR_L2="$B300_STAGEQ_PREPARED_PAIR_L2_BYTES"; Q_BLOCK_L2="$B300_STAGEQ_PREPARED_BLOCK_L2_BYTES"; EFF_PAIR_L2="$Q_PAIR_L2"; EFF_BLOCK_L2="$Q_BLOCK_L2"
    ;;
  *) echo "bad Stage-R upstream kind=$R_UP" >&2; exit 3;;
esac
[[ "$EFF_PAIR_L2" == "$R_HIGH_PL2" && "$EFF_BLOCK_L2" == "$R_HIGH_BL2" ]] || { echo "Stage-S reconstructed high L2 drift got=$EFF_PAIR_L2/$EFF_BLOCK_L2 expected=$R_HIGH_PL2/$R_HIGH_BL2" >&2; exit 3; }

BINS="$LOGDIR/binaries.tsv"; printf 'label\tpair_l2\tblock_l2\tbinary\tbuild_err\n' >"$BINS"; printf 'control\t0\t0\t%s\t-\n' "$CONTROL_BIN" >>"$BINS"
for pl2 in $PAIR_L2_LIST; do for bl2 in $BLOCK_L2_LIST; do
  [[ "$pl2" == 0 && "$bl2" == 0 ]] && continue
  label="p${pl2}_b${bl2}"; bin="$ONEESAN_BUILD_DIR/b300_stages_${label}_up${R_UP}_n27"; err="$LOGDIR/${label}.build.err"; bout="$LOGDIR/${label}.build.out"
  N=27 ARCH="$ARCH" OUT="$bin" UPSTREAM_KIND="$R_UP" STAGEQ_UPSTREAM_KIND="$STAGEQ_UPSTREAM" STAGEP_COUNT_UPSTREAM="$STAGEP_COUNT" MATE_LOAD_POLICY="$MATE_POLICY" PAIR_LOAD_POLICY="$HIGH_PAIR" BLOCK_LOAD_POLICY="$HIGH_BLOCK" ILP2_PAIR_LOAD_POLICY="$LOW_PAIR" ILP2_BLOCK_LOAD_POLICY="$LOW_BLOCK" ILP2_PAIR_CG_L2_BYTES="$pl2" ILP2_BLOCK_CG_L2_BYTES="$bl2" BASE_CG_L2_BYTES="$CGL2" PAIR_CG_L2_BYTES="$BUILDER_PAIR_L2" BLOCK_CG_L2_BYTES="$BUILDER_BLOCK_L2" MATE_CG_L2_BYTES="$MATE_L2" ILP8_PAIR_CG_L2_BYTES="$Q_PAIR_L2" ILP8_BLOCK_CG_L2_BYTES="$Q_BLOCK_L2" HIGH_DROP_CHUNK="$H" HYBRID_THRESHOLD="$T" SELF_WIDTH="$SW" SELF_DISTANCE="$SD" MATE_WIDTH="$MW" MATE_DISTANCE="$MD" SELF_EVICT="$SE" MATE_EVICT="$ME" SELF_GUARD="$SG" MATE_GUARD="$MG" RANDOM_CG="$CG" RANDOM_CG_L2_FETCH_BYTES="$CGL2" PREFETCH_L2="$PRE" DUALMASK="$DUAL" CLOSURE_BATCH="$BATCH" MAXRREGCOUNT="$CAP" PTXAS_VERBOSE=1 BUILD_ERR="$err" bash "$BUILDER" >"$bout" 2>"$LOGDIR/${label}.driver.err"
  [[ -x "$bin" ]] || exit 3; grep -Fq "stage_s_ilp2_cg_l2=1 upstream_kind=$R_UP" "$bout" || exit 3; grep -Fq "ilp2_pair_load_policy=$LOW_PAIR ilp2_block_load_policy=$LOW_BLOCK ilp2_pair_cg_l2_bytes=$pl2 ilp2_block_cg_l2_bytes=$bl2" "$bout" || exit 3; grep -Fq 'stage_s_scope=ilp2_cg_l2_hint_only ilp8_exact_upstream=1 stage_r_policy_preserved=1' "$bout" || exit 3
  printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$pl2" "$bl2" "$bin" "$err" >>"$BINS"
done; done
printf 'label\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
while IFS=$'\t' read -r label pl2 bl2 bin err; do [[ "$label" == label || "$label" == control ]] && continue; python3 "$PARSER" "$err" --label "$label" --contains main_pull_kernel_ilp2 >>"$RESOURCE" || true; python3 "$PARSER" "$err" --label "$label" --contains main_pull_kernel_ilp8_hybrid >>"$RESOURCE" || true; done <"$BINS"
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\2/p" <<<"$l" | tail -n1; }
printf 'label\tpair_l2\tblock_l2\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tforward_high_s\treverse_high_s\n' >"$RESULT"
run_one(){ local label="$1" pl2="$2" bl2="$3" bin="$4" th="$5" r="$6" so="$LOGDIR/${label}_t${th}_r${r}.out" se="$LOGDIR/${label}_t${th}_r${r}.err" line; B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$th" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"; line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || return 4; printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$pl2" "$bl2" "$th" "$r" "$(field residue "$line")" "$(field wall_s "$line")" "$(field active_max_s "$line")" "$(field active_sum_s "$line")" "$(field forward_high_s "$line")" "$(field reverse_high_s "$line")" >>"$RESULT"; }
while IFS=$'\t' read -r label pl2 bl2 bin err; do [[ "$label" == label ]] && continue; for th in "${threads[@]}"; do for ((r=1;r<=REPEATS;++r)); do echo "=== Stage S label=$label low_pair=$LOW_PAIR low_block=$LOW_BLOCK pair_l2=$pl2 block_l2=$bl2 threads=$th rows=$ROWS r_up=$R_UP ===" >&2; run_one "$label" "$pl2" "$bl2" "$bin" "$th" "$r"; done; done; done <"$BINS"

python3 - "$RESULT" "$RESOURCE" "$BINS" "$WINNER_ENV" "$ROWS" "$NGPU" "$R_UP" "$LOW_PAIR" "$LOW_BLOCK" "$HIGH_PAIR" "$HIGH_BLOCK" "$R_HIGH_PL2" "$R_HIGH_BL2" "$CONTROL_THREADS" "$UP_ROWS" "$UP_RES" "$UP_MANIFEST" <<'PY'
import csv,math,statistics,sys,shlex
result,resource,bins_path,winner,rows_arg,ngpu,r_up,low_pair,low_block,high_pair,high_block,high_pl2,high_bl2,control_src_threads,up_rows,up_res,up_manifest=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t')); rr=list(csv.DictReader(open(resource),delimiter='\t')); bins={r['label']:r for r in csv.DictReader(open(bins_path),delimiter='\t')}
if not rows: raise SystemExit('no Stage-S rows')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL Stage-S residue mismatch '+repr({(r['label'],r['threads'],r['repeat']):r['residue'] for r in rows}))
only_res=next(iter(res))
if rows_arg==up_rows and only_res!=up_res: raise SystemExit(f'FATAL Stage-S/Stage-R residue mismatch rows={rows_arg} got={only_res} expected={up_res}')
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
    b=bins[x[1]]; print(f'STAGE_S label={x[1]} pair_l2={b["pair_l2"]} block_l2={b["block_l2"]} threads={x[2]} wall_s={x[0]:.9f} high_s={x[3]:.9f} regs={x[4]} spill_store={x[5]} spill_load={x[6]} spill_free={int(x[7])}',file=sys.stderr)
vals={'B300_STAGES_ROWS':rows_arg,'B300_STAGES_NGPU':ngpu,'B300_STAGES_RESIDUE':only_res,'B300_STAGES_STAGER_UPSTREAM_KIND':r_up,'B300_STAGES_LOW_PAIR_POLICY':low_pair,'B300_STAGES_LOW_BLOCK_POLICY':low_block,'B300_STAGES_HIGH_PAIR_POLICY':high_pair,'B300_STAGES_HIGH_BLOCK_POLICY':high_block,'B300_STAGES_HIGH_PAIR_L2_BYTES':high_pl2,'B300_STAGES_HIGH_BLOCK_L2_BYTES':high_bl2,'B300_STAGES_BASE_PAIR_L2_BYTES':0,'B300_STAGES_BASE_BLOCK_L2_BYTES':0,'B300_STAGES_CONTROL_BIN':bins['control']['binary'],'B300_STAGES_CONTROL_THREADS':base[2],'B300_STAGES_CONTROL_SOURCE_THREADS':control_src_threads,'B300_STAGES_CONTROL_WALL_S':f'{base[0]:.9f}','B300_STAGES_CONTROL_HIGH_S':f'{base[3]:.9f}','B300_STAGES_CONTROL_SPILL_FREE':1,'B300_STAGES_PAIR_L2_BYTES':bb['pair_l2'],'B300_STAGES_BLOCK_L2_BYTES':bb['block_l2'],'B300_STAGES_BIN':bb['binary'],'B300_STAGES_THREADS':best[2],'B300_STAGES_WALL_S':f'{best[0]:.9f}','B300_STAGES_HIGH_S':f'{best[3]:.9f}','B300_STAGES_SPILL_FREE':int(best[7]),'B300_STAGES_SPEEDUP':f'{speed:.9f}','B300_STAGES_BEST_ENABLED':enabled,'B300_STAGES_UPSTREAM_ROWS':up_rows,'B300_STAGES_UPSTREAM_RESIDUE':up_res,'B300_STAGES_UPSTREAM_MANIFEST':up_manifest}
with open(winner,'w') as f:
    for k,v in vals.items(): f.write(f'{k}={q(v)}\n')
print(f'b300_stages_exact_match=1 b300_stages_ngpu={ngpu} r_upstream={r_up} low_pair={low_pair} low_block={low_block} best_pair_l2={bb["pair_l2"]} best_block_l2={bb["block_l2"]} speedup={speed:.9f} residue={only_res}')
PY
cat "$WINNER_ENV"
