#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; ROWS="${ROWS:-1}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"; STAGEN_WINNER_ENV="${STAGEN_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_staged_g${NGPU}_winner.env}"; STAGEN_PREPARE_ENV="${STAGEN_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_fullprime_n27_prepared.env}"
STAGEO_PREPARE_ENV="${STAGEO_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageo_cg_l2_fullprime_n27_prepared.env}"; STAGEP_PREPARE_ENV="${STAGEP_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stagep_mate_cg_l2_fullprime_n27_prepared.env}"; STAGEQ_PREPARE_ENV="${STAGEQ_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageq_ilp8_count_cg_l2_fullprime_n27_prepared.env}"
STAGER_WINNER_ENV="${STAGER_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stager_ilp2_load_staged_g${NGPU}_winner.env}"; STAGER_PREPARE_ENV="${STAGER_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stager_ilp2_load_fullprime_n27_prepared.env}"; STAGES_WINNER_ENV="${STAGES_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stages_ilp2_cg_l2_staged_g${NGPU}_winner.env}"; STAGES_PREPARE_ENV="${STAGES_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stages_ilp2_cg_l2_fullprime_n27_prepared.env}"
UPSTREAM_KIND="${UPSTREAM_KIND:-auto}"; L2_LIST="${L2_LIST:-0 64 128 256}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"; REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageu_ilp2_mate_joint_row${ROWS}_g${NGPU}}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
TBUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-staget-ilp2-mate-load-policy.sh"; UBUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-stageu-ilp2-mate-cg-l2-policy.sh"; PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")" "$(dirname "$WINNER_ENV")"
for f in "$STAGE_F_ENV" "$STAGEN_WINNER_ENV" "$STAGEN_PREPARE_ENV" "$STAGER_WINNER_ENV" "$STAGER_PREPARE_ENV" "$TBUILDER" "$UBUILDER"; do [[ -s "$f" ]] || { echo "missing Stage-U input=$f" >&2; exit 2; }; done
case "$UPSTREAM_KIND" in auto|stager|stages) ;; *) exit 2;; esac; [[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || exit 2; [[ "$NGPU" =~ ^[1-8]$ ]] || exit 2; [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || exit 2
l2s=(); has0=0; for b in $L2_LIST; do case "$b" in 0|64|128|256) ;; *) echo "bad Stage-U L2=$b" >&2; exit 2;; esac; [[ "$b" == 0 ]] && has0=1; seen=0; for old in "${l2s[@]}"; do [[ "$old" == "$b" ]] && seen=1; done; ((seen)) || l2s+=("$b"); done; ((has0&&${#l2s[@]}>1)) || { echo 'L2_LIST must include 0 and at least one nonzero hint' >&2; exit 2; }
threads=(); for th in $THREADS_LIST; do [[ "$th" =~ ^[0-9]+$ ]] && ((th>=32&&th<=1024&&th%32==0)) || exit 2; seen=0; for old in "${threads[@]}"; do [[ "$old" == "$th" ]] && seen=1; done; ((seen)) || threads+=("$th"); done; ((${#threads[@]})) || exit 2
command -v nvcc >/dev/null || exit 2; command -v nvidia-smi >/dev/null || exit 2; command -v sha256sum >/dev/null || exit 2; VISIBLE="$(nvidia-smi --query-gpu=index --format=csv,noheader|wc -l)"; ((VISIBLE>=NGPU)) || exit 2
source "$STAGE_F_ENV"; T="$B300_HYBRID8_NEXTSELF_THRESHOLD"; H="$B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK"; CG="$B300_HYBRID8_NEXTSELF_RANDOM_CG"; CGL2="$B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES"; PRE="$B300_HYBRID8_NEXTSELF_PREFETCH_L2"; DUAL="$B300_HYBRID8_NEXTSELF_DUALMASK"; BATCH="$B300_HYBRID8_NEXTSELF_CLOSURE_BATCH"; CAP="$B300_HYBRID8_NEXTSELF_MAXRREGCOUNT"
source "$STAGEN_WINNER_ENV"; [[ "$B300_STAGEN_STAGED_VALIDATED" == 1 && "$B300_STAGEN_FINAL_ENABLED" == 1 && "$B300_STAGEN_NGPU" == "$NGPU" ]] || { echo 'Stage N not promotable for Stage U' >&2; exit 4; }
HIGH_PAIR="$B300_STAGEN_PAIR_POLICY"; HIGH_BLOCK="$B300_STAGEN_BLOCK_POLICY"; HIGH_MATE="$B300_STAGEN_MATE_LOAD_POLICY"; SW="$B300_STAGEN_SELF_WIDTH"; SD="$B300_STAGEN_SELF_DISTANCE"; SE="$B300_STAGEN_SELF_EVICT"; SG="$B300_STAGEN_SELF_GUARD"; MW="$B300_STAGEN_MATE_WIDTH"; MD="$B300_STAGEN_MATE_DISTANCE"; ME="$B300_STAGEN_MATE_EVICT"; MG="$B300_STAGEN_MATE_GUARD"
source "$STAGER_WINNER_ENV"; [[ "$B300_STAGER_STAGED_VALIDATED" == 1 && "$B300_STAGER_FINAL_ENABLED" == 1 && "$B300_STAGER_FINAL_SPILL_FREE" == 1 && "$B300_STAGER_NGPU" == "$NGPU" ]] || { echo 'Stage R not promotable for Stage U' >&2; exit 4; }
R_UP="$B300_STAGER_UPSTREAM_KIND"; Q_UP="$B300_STAGER_STAGEQ_UPSTREAM_KIND"; LOW_PAIR="$B300_STAGER_PAIR_POLICY"; LOW_BLOCK="$B300_STAGER_BLOCK_POLICY"; HIGH_PL2="$B300_STAGER_HIGH_PAIR_L2_BYTES"; HIGH_BL2="$B300_STAGER_HIGH_BLOCK_L2_BYTES"
source "$STAGER_PREPARE_ENV"; [[ "$B300_STAGER_PREPARED" == 1 && "$B300_STAGER_PREPARED_MOD" == "$MOD" && "$B300_STAGER_PREPARED_NGPU" == "$NGPU" && "$B300_STAGER_PREPARED_BIN" == "$B300_STAGER_FINAL_BIN" ]] || { echo 'Stage-U/R prepare drift' >&2; exit 3; }; sha256sum -c "$B300_STAGER_PREPARED_MANIFEST" >/dev/null || exit 3
RESOLVED=stager; CONTROL_BIN="$B300_STAGER_PREPARED_BIN"; CONTROL_THREADS="$B300_STAGER_PREPARED_THREADS"; UP_ROWS="$B300_STAGER_PREPARED_FINAL_STAGE_ROWS"; UP_RES="$B300_STAGER_PREPARED_FINAL_STAGE_RESIDUE"; UP_MANIFEST="$B300_STAGER_PREPARED_MANIFEST"; LOW_PL2=0; LOW_BL2=0
S_VALID=0
if [[ -s "$STAGES_WINNER_ENV" && -s "$STAGES_PREPARE_ENV" ]] && grep -Fq 'B300_STAGES_STAGED_VALIDATED=1' "$STAGES_WINNER_ENV" && grep -Fq 'B300_STAGES_FINAL_ENABLED=1' "$STAGES_WINNER_ENV"; then S_VALID=1; fi
if [[ "$UPSTREAM_KIND" == stages || ( "$UPSTREAM_KIND" == auto && "$S_VALID" == 1 ) ]]; then
  ((S_VALID)) || { echo 'Stage U requested Stage S but it is unavailable' >&2; exit 3; }
  source "$STAGES_PREPARE_ENV"; [[ "$B300_STAGES_PREPARED" == 1 && "$B300_STAGES_PREPARED_MOD" == "$MOD" && "$B300_STAGES_PREPARED_NGPU" == "$NGPU" && "$B300_STAGES_PREPARED_CONTROL_BIN" == "$B300_STAGER_PREPARED_BIN" ]] || { echo 'Stage-U/S prepare drift' >&2; exit 3; }; sha256sum -c "$B300_STAGES_PREPARED_MANIFEST" >/dev/null || exit 3
  [[ "$B300_STAGES_PREPARED_STAGER_UPSTREAM_KIND" == "$R_UP" && "$B300_STAGES_PREPARED_LOW_PAIR_POLICY" == "$LOW_PAIR" && "$B300_STAGES_PREPARED_LOW_BLOCK_POLICY" == "$LOW_BLOCK" && "$B300_STAGES_PREPARED_HIGH_PAIR_L2_BYTES" == "$HIGH_PL2" && "$B300_STAGES_PREPARED_HIGH_BLOCK_L2_BYTES" == "$HIGH_BL2" ]] || { echo 'Stage-U lost R provenance through S' >&2; exit 3; }
  RESOLVED=stages; CONTROL_BIN="$B300_STAGES_PREPARED_BIN"; CONTROL_THREADS="$B300_STAGES_PREPARED_THREADS"; UP_ROWS="$B300_STAGES_PREPARED_FINAL_STAGE_ROWS"; UP_RES="$B300_STAGES_PREPARED_FINAL_STAGE_RESIDUE"; UP_MANIFEST="$B300_STAGES_PREPARED_MANIFEST"; LOW_PL2="$B300_STAGES_PREPARED_PAIR_L2_BYTES"; LOW_BL2="$B300_STAGES_PREPARED_BLOCK_L2_BYTES"
fi
[[ -x "$CONTROL_BIN" ]] || exit 3
STAGEP_COUNT=stagen; BUILDER_PL2=0; BUILDER_BL2=0; MATE_L2=0; Q_PL2=0; Q_BL2=0
case "$R_UP" in
 stagen) ;;
 stageo) source "$STAGEO_PREPARE_ENV"; sha256sum -c "$B300_STAGEO_PREPARED_MANIFEST" >/dev/null || exit 3; BUILDER_PL2="$B300_STAGEO_PREPARED_PAIR_L2_BYTES"; BUILDER_BL2="$B300_STAGEO_PREPARED_BLOCK_L2_BYTES" ;;
 stagep) source "$STAGEP_PREPARE_ENV"; sha256sum -c "$B300_STAGEP_PREPARED_MANIFEST" >/dev/null || exit 3; STAGEP_COUNT="$B300_STAGEP_PREPARED_COUNT_UPSTREAM"; MATE_L2="$B300_STAGEP_PREPARED_MATE_L2_BYTES"; [[ "$STAGEP_COUNT" == stageo ]] && { BUILDER_PL2="$B300_STAGEP_PREPARED_PAIR_CG_L2_BYTES"; BUILDER_BL2="$B300_STAGEP_PREPARED_BLOCK_CG_L2_BYTES"; } ;;
 stageq) source "$STAGEQ_PREPARE_ENV"; sha256sum -c "$B300_STAGEQ_PREPARED_MANIFEST" >/dev/null || exit 3; [[ "$B300_STAGEQ_PREPARED_UPSTREAM_KIND" == "$Q_UP" ]] || exit 3; STAGEP_COUNT="$B300_STAGEQ_PREPARED_STAGEP_COUNT_UPSTREAM"; BUILDER_PL2="$B300_STAGEQ_PREPARED_BUILDER_PAIR_CG_L2_BYTES"; BUILDER_BL2="$B300_STAGEQ_PREPARED_BUILDER_BLOCK_CG_L2_BYTES"; MATE_L2="$B300_STAGEQ_PREPARED_MATE_CG_L2_BYTES"; Q_PL2="$B300_STAGEQ_PREPARED_PAIR_L2_BYTES"; Q_BL2="$B300_STAGEQ_PREPARED_BLOCK_L2_BYTES" ;;
 *) exit 3;;
esac
EXPECTED_HIGH_MATE_L2=0; [[ "$HIGH_MATE" == cg ]] && EXPECTED_HIGH_MATE_L2="$MATE_L2"
common=(N=27 ARCH="$ARCH" UPSTREAM_KIND="$RESOLVED" STAGER_UPSTREAM_KIND="$R_UP" STAGEQ_UPSTREAM_KIND="$Q_UP" STAGEP_COUNT_UPSTREAM="$STAGEP_COUNT" MATE_LOAD_POLICY="$HIGH_MATE" PAIR_LOAD_POLICY="$HIGH_PAIR" BLOCK_LOAD_POLICY="$HIGH_BLOCK" ILP2_PAIR_LOAD_POLICY="$LOW_PAIR" ILP2_BLOCK_LOAD_POLICY="$LOW_BLOCK" ILP2_PAIR_CG_L2_BYTES="$LOW_PL2" ILP2_BLOCK_CG_L2_BYTES="$LOW_BL2" BASE_CG_L2_BYTES="$CGL2" PAIR_CG_L2_BYTES="$BUILDER_PL2" BLOCK_CG_L2_BYTES="$BUILDER_BL2" MATE_CG_L2_BYTES="$MATE_L2" ILP8_PAIR_CG_L2_BYTES="$Q_PL2" ILP8_BLOCK_CG_L2_BYTES="$Q_BL2" HIGH_DROP_CHUNK="$H" HYBRID_THRESHOLD="$T" SELF_WIDTH="$SW" SELF_DISTANCE="$SD" MATE_WIDTH="$MW" MATE_DISTANCE="$MD" SELF_EVICT="$SE" MATE_EVICT="$ME" SELF_GUARD="$SG" MATE_GUARD="$MG" RANDOM_CG="$CG" RANDOM_CG_L2_FETCH_BYTES="$CGL2" PREFETCH_L2="$PRE" DUALMASK="$DUAL" CLOSURE_BATCH="$BATCH" MAXRREGCOUNT="$CAP" PTXAS_VERBOSE=1)
BINS="$LOGDIR/binaries.tsv"; printf 'label\tpolicy\tl2_bytes\tbinary\tbuild_err\n' >"$BINS"; printf 'control\tdefault\t0\t%s\t-\n' "$CONTROL_BIN" >>"$BINS"
# Exact Stage-T comparator candidates are included even though Stage U does not require T acceptance.
for p in cg cs; do label="t_${p}"; bin="$ONEESAN_BUILD_DIR/b300_stageu_cmp_${p}_up${RESOLVED}_rup${R_UP}_n27"; err="$LOGDIR/${label}.build.err"; bout="$LOGDIR/${label}.build.out"; env "${common[@]}" OUT="$bin" ILP2_MATE_LOAD_POLICY="$p" BUILD_ERR="$err" bash "$TBUILDER" >"$bout" 2>"$LOGDIR/${label}.driver.err"; [[ -x "$bin" ]] || exit 3; grep -Fq "ilp2_mate_load_policy=$p" "$bout" || exit 3; printf '%s\t%s\t0\t%s\t%s\n' "$label" "$p" "$bin" "$err" >>"$BINS"; done
for b in "${l2s[@]}"; do [[ "$b" == 0 ]] && continue; label="u_cg_l2_${b}"; bin="$ONEESAN_BUILD_DIR/b300_stageu_cg_l2_${b}_up${RESOLVED}_rup${R_UP}_n27"; err="$LOGDIR/${label}.build.err"; bout="$LOGDIR/${label}.build.out"; env "${common[@]}" OUT="$bin" ILP2_MATE_CG_L2_BYTES="$b" BUILD_ERR="$err" bash "$UBUILDER" >"$bout" 2>"$LOGDIR/${label}.driver.err"; [[ -x "$bin" ]] || exit 3; grep -Fq "stage_u_ilp2_mate_cg_l2=1 upstream_kind=$RESOLVED ilp2_mate_policy=cg ilp2_mate_cg_l2_bytes=$b" "$bout" || exit 3; grep -Fq 'accepted_staget_winner_required=0' "$bout" || exit 3; printf '%s\tcg\t%s\t%s\t%s\n' "$label" "$b" "$bin" "$err" >>"$BINS"; done
printf 'label\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"; while IFS=$'\t' read -r label policy l2 bin err; do [[ "$label" == label || "$label" == control ]] && continue; python3 "$PARSER" "$err" --label "$label" --contains main_pull_kernel_ilp2 >>"$RESOURCE" || true; python3 "$PARSER" "$err" --label "$label" --contains main_pull_kernel_ilp8_hybrid >>"$RESOURCE" || true; done <"$BINS"
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l"|tail -n1; }
printf 'label\tpolicy\tl2_bytes\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tforward_high_s\treverse_high_s\n' >"$RESULT"
run_one(){ local label="$1" policy="$2" l2="$3" bin="$4" th="$5" r="$6" so="$LOGDIR/${label}_t${th}_r${r}.out" se="$LOGDIR/${label}_t${th}_r${r}.err" line; B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$th" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"; line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so"|tail -n1||true)"; [[ -n "$line" ]] || return 4; printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$policy" "$l2" "$th" "$r" "$(field residue "$line")" "$(field wall_s "$line")" "$(field active_max_s "$line")" "$(field active_sum_s "$line")" "$(field forward_high_s "$line")" "$(field reverse_high_s "$line")" >>"$RESULT"; }
while IFS=$'\t' read -r label policy l2 bin err; do [[ "$label" == label ]] && continue; for th in "${threads[@]}"; do for ((r=1;r<=REPEATS;++r)); do echo "=== Stage U label=$label policy=$policy l2=$l2 threads=$th rows=$ROWS upstream=$RESOLVED ===" >&2; run_one "$label" "$policy" "$l2" "$bin" "$th" "$r"; done; done; done <"$BINS"
python3 - "$RESULT" "$RESOURCE" "$BINS" "$WINNER_ENV" "$ROWS" "$NGPU" "$RESOLVED" "$R_UP" "$LOW_PAIR" "$LOW_BLOCK" "$LOW_PL2" "$LOW_BL2" "$HIGH_PAIR" "$HIGH_BLOCK" "$HIGH_PL2" "$HIGH_BL2" "$HIGH_MATE" "$EXPECTED_HIGH_MATE_L2" "$CONTROL_THREADS" "$UP_ROWS" "$UP_RES" "$UP_MANIFEST" <<'PY'
import csv,math,statistics,sys,shlex
(result,resource,bins_path,winner,rows_arg,ngpu,upstream,r_up,low_pair,low_block,low_pl2,low_bl2,high_pair,high_block,high_pl2,high_bl2,high_mate,high_mate_l2,control_src_threads,up_rows,up_res,up_manifest)=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t')); rr=list(csv.DictReader(open(resource),delimiter='\t')); bins={r['label']:r for r in csv.DictReader(open(bins_path),delimiter='\t')}
if not rows: raise SystemExit('no Stage-U rows')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL Stage-U residue mismatch '+repr({(r['label'],r['threads'],r['repeat']):r['residue'] for r in rows}))
residue=next(iter(res))
if rows_arg==up_rows and residue!=up_res: raise SystemExit(f'FATAL Stage-U/upstream residue mismatch rows={rows_arg} got={residue} expected={up_res}')
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
base=min((x for x in agg if x[1]=='control'),key=rank); tests=[x for x in agg if x[1]!='control' and x[7]]; best=min(tests,key=rank) if tests else base; bb=bins[best[1]]; speed=base[0]/best[0]; novel=int(best[1].startswith('u_cg_l2_') and int(bb['l2_bytes'])>0 and rank(best)<rank(base)); q=lambda x:shlex.quote(str(x))
for x in sorted(agg,key=rank): print(f'STAGE_U label={x[1]} policy={bins[x[1]]["policy"]} l2={bins[x[1]]["l2_bytes"]} threads={x[2]} wall_s={x[0]:.9f} high_s={x[3]:.9f} regs={x[4]} spill_store={x[5]} spill_load={x[6]} spill_free={int(x[7])}',file=sys.stderr)
vals={'B300_STAGEU_ROWS':rows_arg,'B300_STAGEU_NGPU':ngpu,'B300_STAGEU_RESIDUE':residue,'B300_STAGEU_UPSTREAM_KIND':upstream,'B300_STAGEU_STAGER_UPSTREAM_KIND':r_up,'B300_STAGEU_LOW_PAIR_POLICY':low_pair,'B300_STAGEU_LOW_BLOCK_POLICY':low_block,'B300_STAGEU_LOW_PAIR_L2_BYTES':low_pl2,'B300_STAGEU_LOW_BLOCK_L2_BYTES':low_bl2,'B300_STAGEU_HIGH_PAIR_POLICY':high_pair,'B300_STAGEU_HIGH_BLOCK_POLICY':high_block,'B300_STAGEU_HIGH_PAIR_L2_BYTES':high_pl2,'B300_STAGEU_HIGH_BLOCK_L2_BYTES':high_bl2,'B300_STAGEU_HIGH_MATE_POLICY':high_mate,'B300_STAGEU_HIGH_MATE_L2_BYTES':high_mate_l2,'B300_STAGEU_CONTROL_POLICY':'default','B300_STAGEU_CONTROL_BIN':bins['control']['binary'],'B300_STAGEU_CONTROL_THREADS':base[2],'B300_STAGEU_CONTROL_SOURCE_THREADS':control_src_threads,'B300_STAGEU_CONTROL_WALL_S':f'{base[0]:.9f}','B300_STAGEU_CONTROL_HIGH_S':f'{base[3]:.9f}','B300_STAGEU_CONTROL_SPILL_FREE':1,'B300_STAGEU_JOINT_BEST_LABEL':best[1],'B300_STAGEU_JOINT_BEST_POLICY':bb['policy'],'B300_STAGEU_JOINT_BEST_L2_BYTES':bb['l2_bytes'],'B300_STAGEU_BIN':bb['binary'],'B300_STAGEU_THREADS':best[2],'B300_STAGEU_WALL_S':f'{best[0]:.9f}','B300_STAGEU_HIGH_S':f'{best[3]:.9f}','B300_STAGEU_SPILL_FREE':int(best[7]),'B300_STAGEU_SPEEDUP':f'{speed:.9f}','B300_STAGEU_BEST_ENABLED':novel,'B300_STAGEU_UPSTREAM_ROWS':up_rows,'B300_STAGEU_UPSTREAM_RESIDUE':up_res,'B300_STAGEU_UPSTREAM_MANIFEST':up_manifest}
with open(winner,'w') as f:
 for k,v in vals.items(): f.write(f'{k}={q(v)}\n')
print(f'b300_stageu_exact_match=1 b300_stageu_ngpu={ngpu} upstream={upstream} r_upstream={r_up} joint_best={best[1]} policy={bb["policy"]} l2={bb["l2_bytes"]} novel={novel} speedup={speed:.9f} residue={residue}')
PY
cat "$WINNER_ENV"
