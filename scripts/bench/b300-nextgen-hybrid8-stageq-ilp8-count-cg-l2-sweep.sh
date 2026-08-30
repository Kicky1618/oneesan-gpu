#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; ROWS="${ROWS:-1}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
STAGEN_WINNER_ENV="${STAGEN_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_staged_g${NGPU}_winner.env}"
STAGEN_PREPARE_ENV="${STAGEN_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_fullprime_n27_prepared.env}"
STAGEO_WINNER_ENV="${STAGEO_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageo_cg_l2_staged_g${NGPU}_winner.env}"
STAGEO_PREPARE_ENV="${STAGEO_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageo_cg_l2_fullprime_n27_prepared.env}"
STAGEP_WINNER_ENV="${STAGEP_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stagep_mate_cg_l2_staged_g${NGPU}_winner.env}"
STAGEP_PREPARE_ENV="${STAGEP_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stagep_mate_cg_l2_fullprime_n27_prepared.env}"
UPSTREAM_KIND="${UPSTREAM_KIND:-auto}"; PAIR_L2_LIST="${PAIR_L2_LIST:-0 64 128 256}"; BLOCK_L2_LIST="${BLOCK_L2_LIST:-0 64 128 256}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"; REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageq_ilp8_count_cg_l2_row${ROWS}_g${NGPU}}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-stageq-ilp8-count-cg-l2-policy.sh"; PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")" "$(dirname "$WINNER_ENV")"
for f in "$STAGE_F_ENV" "$STAGEN_WINNER_ENV" "$STAGEN_PREPARE_ENV" "$BUILDER"; do [[ -s "$f" ]] || { echo "missing Stage-Q input=$f" >&2; exit 2; }; done
case "$UPSTREAM_KIND" in auto|stagen|stageo|stagep) ;; *) echo 'UPSTREAM_KIND must be auto,stagen,stageo,stagep' >&2; exit 2;; esac
[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || exit 2; [[ "$NGPU" =~ ^[1-8]$ ]] || exit 2; [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || exit 2
norm_bytes(){ local raw="$1" out=() v old seen; for v in $raw; do case "$v" in 0|64|128|256) ;; *) echo "bad Stage-Q L2 bytes=$v" >&2; exit 2;; esac; seen=0; for old in "${out[@]}"; do [[ "$old" == "$v" ]] && seen=1; done; ((seen)) || out+=("$v"); done; ((${#out[@]})) || exit 2; printf '%s' "${out[*]}"; }
PAIR_L2_LIST="$(norm_bytes "$PAIR_L2_LIST")"; BLOCK_L2_LIST="$(norm_bytes "$BLOCK_L2_LIST")"
threads=(); for th in $THREADS_LIST; do [[ "$th" =~ ^[0-9]+$ ]] && ((th>=32&&th<=1024&&th%32==0)) || exit 2; seen=0; for old in "${threads[@]}"; do [[ "$old" == "$th" ]] && seen=1; done; ((seen)) || threads+=("$th"); done; ((${#threads[@]})) || exit 2
command -v nvcc >/dev/null || exit 2; command -v nvidia-smi >/dev/null || exit 2; command -v sha256sum >/dev/null || exit 2
VISIBLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; ((VISIBLE_GPUS>=NGPU)) || { echo "need $NGPU visible GPU(s), found $VISIBLE_GPUS" >&2; exit 2; }

# Stage F fixes recurrence-level compile knobs and the Stage-N shared CG baseline.
# shellcheck disable=SC1090
source "$STAGE_F_ENV"
for k in B300_HYBRID8_NEXTSELF_THRESHOLD B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK B300_HYBRID8_NEXTSELF_RANDOM_CG B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES B300_HYBRID8_NEXTSELF_PREFETCH_L2 B300_HYBRID8_NEXTSELF_DUALMASK B300_HYBRID8_NEXTSELF_CLOSURE_BATCH B300_HYBRID8_NEXTSELF_MAXRREGCOUNT; do [[ -n "${!k+x}" ]] || { echo "Stage-F env missing $k" >&2; exit 3; }; done
T="$B300_HYBRID8_NEXTSELF_THRESHOLD"; H="$B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK"; CG="$B300_HYBRID8_NEXTSELF_RANDOM_CG"; CGL2="$B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES"; PRE="$B300_HYBRID8_NEXTSELF_PREFETCH_L2"; DUAL="$B300_HYBRID8_NEXTSELF_DUALMASK"; BATCH="$B300_HYBRID8_NEXTSELF_CLOSURE_BATCH"; CAP="$B300_HYBRID8_NEXTSELF_MAXRREGCOUNT"
case "$CGL2" in 0|64|128|256) ;; *) exit 3;; esac

# Stage N owns geometry and all three load policies.
# shellcheck disable=SC1090
source "$STAGEN_WINNER_ENV"
for k in B300_STAGEN_STAGED_VALIDATED B300_STAGEN_FINAL_ENABLED B300_STAGEN_NGPU B300_STAGEN_PAIR_POLICY B300_STAGEN_BLOCK_POLICY B300_STAGEN_MATE_LOAD_POLICY B300_STAGEN_FINAL_BIN B300_STAGEN_FINAL_THREADS B300_STAGEN_FINAL_SPILL_FREE B300_STAGEN_SELF_WIDTH B300_STAGEN_SELF_DISTANCE B300_STAGEN_SELF_EVICT B300_STAGEN_SELF_GUARD B300_STAGEN_MATE_WIDTH B300_STAGEN_MATE_DISTANCE B300_STAGEN_MATE_EVICT B300_STAGEN_MATE_GUARD B300_STAGEN_FINAL_STAGE_ROWS B300_STAGEN_FINAL_STAGE_RESIDUE; do [[ -n "${!k+x}" ]] || { echo "Stage-N winner missing $k" >&2; exit 3; }; done
[[ "$B300_STAGEN_STAGED_VALIDATED" == 1 && "$B300_STAGEN_FINAL_ENABLED" == 1 && "$B300_STAGEN_FINAL_SPILL_FREE" == 1 && "$B300_STAGEN_NGPU" == "$NGPU" ]] || { echo 'Stage N not promotable for Stage Q' >&2; exit 4; }
PAIR_POLICY="$B300_STAGEN_PAIR_POLICY"; BLOCK_POLICY="$B300_STAGEN_BLOCK_POLICY"; MATE_POLICY="$B300_STAGEN_MATE_LOAD_POLICY"; SW="$B300_STAGEN_SELF_WIDTH"; SD="$B300_STAGEN_SELF_DISTANCE"; SE="$B300_STAGEN_SELF_EVICT"; SG="$B300_STAGEN_SELF_GUARD"; MW="$B300_STAGEN_MATE_WIDTH"; MD="$B300_STAGEN_MATE_DISTANCE"; ME="$B300_STAGEN_MATE_EVICT"; MG="$B300_STAGEN_MATE_GUARD"
for p in "$PAIR_POLICY" "$BLOCK_POLICY" "$MATE_POLICY"; do case "$p" in default|cg|cs) ;; *) exit 3;; esac; done
[[ "$PAIR_POLICY" == cg || "$BLOCK_POLICY" == cg ]] || { echo 'Stage Q not applicable: selected Stage N has no CG Count axis' >&2; exit 4; }
# shellcheck disable=SC1090
source "$STAGEN_PREPARE_ENV"
[[ "${B300_STAGEN_PREPARED:-0}" == 1 && "${B300_STAGEN_PREPARED_MOD:-}" == "$MOD" && "${B300_STAGEN_PREPARED_NGPU:-0}" == "$NGPU" ]] || { echo 'Stage-Q/Stage-N prepare mismatch' >&2; exit 3; }
[[ "$B300_STAGEN_PREPARED_BIN" == "$B300_STAGEN_FINAL_BIN" && "$B300_STAGEN_PREPARED_PAIR_POLICY" == "$PAIR_POLICY" && "$B300_STAGEN_PREPARED_BLOCK_POLICY" == "$BLOCK_POLICY" && "$B300_STAGEN_PREPARED_MATE_LOAD_POLICY" == "$MATE_POLICY" ]] || { echo 'Stage-Q Stage-N prepared/winner drift' >&2; exit 3; }
[[ -s "${B300_STAGEN_PREPARED_MANIFEST:-}" ]] || exit 3; sha256sum -c "$B300_STAGEN_PREPARED_MANIFEST" >/dev/null || { echo 'Stage-N manifest mismatch before Stage Q' >&2; exit 3; }

RESOLVED_UPSTREAM=stagen; STAGEP_COUNT_UPSTREAM=stagen; CONTROL_BIN="$B300_STAGEN_PREPARED_BIN"; CONTROL_THREADS="$B300_STAGEN_PREPARED_THREADS"; BUILDER_PAIR_L2=0; BUILDER_BLOCK_L2=0; MATE_L2=0; UP_ROWS="$B300_STAGEN_FINAL_STAGE_ROWS"; UP_RES="$B300_STAGEN_FINAL_STAGE_RESIDUE"; UP_MANIFEST="$B300_STAGEN_PREPARED_MANIFEST"
EFF_PAIR_L2=0; [[ "$PAIR_POLICY" == cg ]] && EFF_PAIR_L2="$CGL2"; EFF_BLOCK_L2=0; [[ "$BLOCK_POLICY" == cg ]] && EFF_BLOCK_L2="$CGL2"

O_VALID=0
if [[ -s "$STAGEO_WINNER_ENV" && -s "$STAGEO_PREPARE_ENV" ]] && grep -Fq 'B300_STAGEO_STAGED_VALIDATED=1' "$STAGEO_WINNER_ENV" && grep -Fq 'B300_STAGEO_FINAL_ENABLED=1' "$STAGEO_WINNER_ENV"; then O_VALID=1; fi
if [[ "$UPSTREAM_KIND" == stageo || "$UPSTREAM_KIND" == stagep || ( "$UPSTREAM_KIND" == auto && "$O_VALID" == 1 ) ]]; then
  if ((O_VALID)); then
    # shellcheck disable=SC1090
    source "$STAGEO_PREPARE_ENV"
    [[ "${B300_STAGEO_PREPARED:-0}" == 1 && "${B300_STAGEO_PREPARED_MOD:-}" == "$MOD" && "${B300_STAGEO_PREPARED_NGPU:-0}" == "$NGPU" ]] || { echo 'Stage-Q/Stage-O prepare mismatch' >&2; exit 3; }
    [[ "$B300_STAGEO_PREPARED_CONTROL_BIN" == "$B300_STAGEN_PREPARED_BIN" && "$B300_STAGEO_PREPARED_PAIR_POLICY" == "$PAIR_POLICY" && "$B300_STAGEO_PREPARED_BLOCK_POLICY" == "$BLOCK_POLICY" && "$B300_STAGEO_PREPARED_MATE_LOAD_POLICY" == "$MATE_POLICY" ]] || { echo 'Stage-Q Stage-O control/policy drift' >&2; exit 3; }
    sha256sum -c "$B300_STAGEO_PREPARED_MANIFEST" >/dev/null || { echo 'Stage-O manifest mismatch before Stage Q' >&2; exit 3; }
    RESOLVED_UPSTREAM=stageo; CONTROL_BIN="$B300_STAGEO_PREPARED_BIN"; CONTROL_THREADS="$B300_STAGEO_PREPARED_THREADS"; BUILDER_PAIR_L2="$B300_STAGEO_PREPARED_PAIR_L2_BYTES"; BUILDER_BLOCK_L2="$B300_STAGEO_PREPARED_BLOCK_L2_BYTES"; EFF_PAIR_L2="$BUILDER_PAIR_L2"; EFF_BLOCK_L2="$BUILDER_BLOCK_L2"; UP_ROWS="$B300_STAGEO_PREPARED_FINAL_STAGE_ROWS"; UP_RES="$B300_STAGEO_PREPARED_FINAL_STAGE_RESIDUE"; UP_MANIFEST="$B300_STAGEO_PREPARED_MANIFEST"
  elif [[ "$UPSTREAM_KIND" == stageo ]]; then echo 'Stage-Q requested Stage O but it is unavailable' >&2; exit 3; fi
fi

P_VALID=0
if [[ -s "$STAGEP_WINNER_ENV" && -s "$STAGEP_PREPARE_ENV" ]] && grep -Fq 'B300_STAGEP_STAGED_VALIDATED=1' "$STAGEP_WINNER_ENV" && grep -Fq 'B300_STAGEP_FINAL_ENABLED=1' "$STAGEP_WINNER_ENV"; then P_VALID=1; fi
if [[ "$UPSTREAM_KIND" == stagep || ( "$UPSTREAM_KIND" == auto && "$P_VALID" == 1 ) ]]; then
  ((P_VALID)) || { echo 'Stage-Q requested Stage P but it is unavailable' >&2; exit 3; }
  # shellcheck disable=SC1090
  source "$STAGEP_PREPARE_ENV"
  [[ "${B300_STAGEP_PREPARED:-0}" == 1 && "${B300_STAGEP_PREPARED_MOD:-}" == "$MOD" && "${B300_STAGEP_PREPARED_NGPU:-0}" == "$NGPU" ]] || { echo 'Stage-Q/Stage-P prepare mismatch' >&2; exit 3; }
  [[ "$B300_STAGEP_PREPARED_PAIR_POLICY" == "$PAIR_POLICY" && "$B300_STAGEP_PREPARED_BLOCK_POLICY" == "$BLOCK_POLICY" && "$B300_STAGEP_PREPARED_MATE_LOAD_POLICY" == cg ]] || { echo 'Stage-Q Stage-P policy drift' >&2; exit 3; }
  sha256sum -c "$B300_STAGEP_PREPARED_MANIFEST" >/dev/null || { echo 'Stage-P manifest mismatch before Stage Q' >&2; exit 3; }
  STAGEP_COUNT_UPSTREAM="$B300_STAGEP_PREPARED_COUNT_UPSTREAM"
  case "$STAGEP_COUNT_UPSTREAM" in stagen)
      [[ "$B300_STAGEP_PREPARED_CONTROL_BIN" == "$B300_STAGEN_PREPARED_BIN" ]] || { echo 'Stage-P control is not exact Stage N' >&2; exit 3; }
      BUILDER_PAIR_L2=0; BUILDER_BLOCK_L2=0; EFF_PAIR_L2=0; [[ "$PAIR_POLICY" == cg ]] && EFF_PAIR_L2="$CGL2"; EFF_BLOCK_L2=0; [[ "$BLOCK_POLICY" == cg ]] && EFF_BLOCK_L2="$CGL2"
      ;;
    stageo)
      ((O_VALID)) || { echo 'Stage-P claims Stage O but Stage O is unavailable' >&2; exit 3; }
      [[ "$B300_STAGEP_PREPARED_CONTROL_BIN" == "$B300_STAGEO_PREPARED_BIN" ]] || { echo 'Stage-P control is not exact Stage O' >&2; exit 3; }
      BUILDER_PAIR_L2="$B300_STAGEP_PREPARED_PAIR_CG_L2_BYTES"; BUILDER_BLOCK_L2="$B300_STAGEP_PREPARED_BLOCK_CG_L2_BYTES"; EFF_PAIR_L2="$BUILDER_PAIR_L2"; EFF_BLOCK_L2="$BUILDER_BLOCK_L2"
      ;;
    *) echo 'bad Stage-P Count upstream' >&2; exit 3;;
  esac
  RESOLVED_UPSTREAM=stagep; CONTROL_BIN="$B300_STAGEP_PREPARED_BIN"; CONTROL_THREADS="$B300_STAGEP_PREPARED_THREADS"; MATE_POLICY=cg; MATE_L2="$B300_STAGEP_PREPARED_MATE_L2_BYTES"; UP_ROWS="$B300_STAGEP_PREPARED_FINAL_STAGE_ROWS"; UP_RES="$B300_STAGEP_PREPARED_FINAL_STAGE_RESIDUE"; UP_MANIFEST="$B300_STAGEP_PREPARED_MANIFEST"
fi
if [[ "$UPSTREAM_KIND" == stagen ]]; then RESOLVED_UPSTREAM=stagen; fi
if [[ "$UPSTREAM_KIND" == stageo && "$RESOLVED_UPSTREAM" != stageo ]]; then exit 3; fi
if [[ "$UPSTREAM_KIND" == stagep && "$RESOLVED_UPSTREAM" != stagep ]]; then exit 3; fi
[[ -x "$CONTROL_BIN" ]] || { echo "Stage-Q control binary missing=$CONTROL_BIN" >&2; exit 3; }

# Non-CG axes are pinned to zero. CG axes must contain the exact upstream L2 value.
if [[ "$PAIR_POLICY" != cg ]]; then PAIR_L2_LIST=0; else case " $PAIR_L2_LIST " in *" $EFF_PAIR_L2 "*) ;; *) echo "PAIR_L2_LIST omits exact upstream $EFF_PAIR_L2" >&2; exit 2;; esac; fi
if [[ "$BLOCK_POLICY" != cg ]]; then BLOCK_L2_LIST=0; else case " $BLOCK_L2_LIST " in *" $EFF_BLOCK_L2 "*) ;; *) echo "BLOCK_L2_LIST omits exact upstream $EFF_BLOCK_L2" >&2; exit 2;; esac; fi

BINS="$LOGDIR/binaries.tsv"; printf 'label\tpair_l2\tblock_l2\tbinary\tbuild_err\n' >"$BINS"; printf 'control\t%s\t%s\t%s\t-\n' "$EFF_PAIR_L2" "$EFF_BLOCK_L2" "$CONTROL_BIN" >>"$BINS"
for pl2 in $PAIR_L2_LIST; do for bl2 in $BLOCK_L2_LIST; do
  [[ "$pl2" == "$EFF_PAIR_L2" && "$bl2" == "$EFF_BLOCK_L2" ]] && continue
  label="p${pl2}_b${bl2}"; bin="$ONEESAN_BUILD_DIR/b300_stageq_${label}_${RESOLVED_UPSTREAM}_pp${PAIR_POLICY}${BUILDER_PAIR_L2}_bp${BLOCK_POLICY}${BUILDER_BLOCK_L2}_mp${MATE_POLICY}${MATE_L2}_n27"; err="$LOGDIR/${label}.build.err"; bout="$LOGDIR/${label}.build.out"
  N=27 ARCH="$ARCH" OUT="$bin" UPSTREAM_KIND="$RESOLVED_UPSTREAM" STAGEP_COUNT_UPSTREAM="$STAGEP_COUNT_UPSTREAM" MATE_LOAD_POLICY="$MATE_POLICY" PAIR_LOAD_POLICY="$PAIR_POLICY" BLOCK_LOAD_POLICY="$BLOCK_POLICY" BASE_CG_L2_BYTES="$CGL2" PAIR_CG_L2_BYTES="$BUILDER_PAIR_L2" BLOCK_CG_L2_BYTES="$BUILDER_BLOCK_L2" MATE_CG_L2_BYTES="$MATE_L2" ILP8_PAIR_CG_L2_BYTES="$pl2" ILP8_BLOCK_CG_L2_BYTES="$bl2" HIGH_DROP_CHUNK="$H" HYBRID_THRESHOLD="$T" SELF_WIDTH="$SW" SELF_DISTANCE="$SD" MATE_WIDTH="$MW" MATE_DISTANCE="$MD" SELF_EVICT="$SE" MATE_EVICT="$ME" SELF_GUARD="$SG" MATE_GUARD="$MG" RANDOM_CG="$CG" RANDOM_CG_L2_FETCH_BYTES="$CGL2" PREFETCH_L2="$PRE" DUALMASK="$DUAL" CLOSURE_BATCH="$BATCH" MAXRREGCOUNT="$CAP" PTXAS_VERBOSE=1 BUILD_ERR="$err" bash "$BUILDER" >"$bout" 2>"$LOGDIR/${label}.driver.err"
  [[ -x "$bin" ]] || exit 3; grep -Fq "stage_q_ilp8_count_cg_l2=1 upstream_kind=$RESOLVED_UPSTREAM" "$bout" || exit 3; grep -Fq "ilp8_pair_cg_l2_bytes=$pl2 ilp8_block_cg_l2_bytes=$bl2" "$bout" || exit 3; grep -Fq 'stage_q_scope=ilp8_count_cg_l2_only ilp2_exact_upstream=1' "$bout" || exit 3
  printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$pl2" "$bl2" "$bin" "$err" >>"$BINS"
done; done

printf 'label\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
while IFS=$'\t' read -r label pl2 bl2 bin err; do [[ "$label" == label || "$label" == control ]] && continue; python3 "$PARSER" "$err" --label "$label" --contains main_pull_kernel_ilp2 >>"$RESOURCE" || true; python3 "$PARSER" "$err" --label "$label" --contains main_pull_kernel_ilp8_hybrid >>"$RESOURCE" || true; done <"$BINS"
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
printf 'label\tpair_l2\tblock_l2\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tforward_high_s\treverse_high_s\n' >"$RESULT"
run_one(){ local label="$1" pl2="$2" bl2="$3" bin="$4" th="$5" r="$6" so="$LOGDIR/${label}_t${th}_r${r}.out" se="$LOGDIR/${label}_t${th}_r${r}.err" line; B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$th" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"; line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { tail -n 80 "$se" >&2 || true; return 4; }; printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$pl2" "$bl2" "$th" "$r" "$(field residue "$line")" "$(field wall_s "$line")" "$(field active_max_s "$line")" "$(field active_sum_s "$line")" "$(field forward_high_s "$line")" "$(field reverse_high_s "$line")" >>"$RESULT"; }
while IFS=$'\t' read -r label pl2 bl2 bin err; do [[ "$label" == label ]] && continue; for th in "${threads[@]}"; do for ((r=1;r<=REPEATS;++r)); do echo "=== Stage Q label=$label pair_l2=$pl2 block_l2=$bl2 threads=$th rows=$ROWS upstream=$RESOLVED_UPSTREAM ===" >&2; run_one "$label" "$pl2" "$bl2" "$bin" "$th" "$r"; done; done; done <"$BINS"

python3 - "$RESULT" "$RESOURCE" "$BINS" "$WINNER_ENV" "$ROWS" "$NGPU" "$RESOLVED_UPSTREAM" "$STAGEP_COUNT_UPSTREAM" "$PAIR_POLICY" "$BLOCK_POLICY" "$MATE_POLICY" "$CGL2" "$BUILDER_PAIR_L2" "$BUILDER_BLOCK_L2" "$MATE_L2" "$EFF_PAIR_L2" "$EFF_BLOCK_L2" "$CONTROL_THREADS" "$UP_ROWS" "$UP_RES" "$UP_MANIFEST" <<'PY'
import csv,math,statistics,sys,shlex
(result,resource,bins_path,winner,rows_arg,ngpu,upstream,stagep_count,pair_policy,block_policy,mate_policy,base_l2,builder_pair,builder_block,mate_l2,eff_pair,eff_block,control_src_threads,up_rows,up_res,up_manifest)=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t')); rr=list(csv.DictReader(open(resource),delimiter='\t')); bins={r['label']:r for r in csv.DictReader(open(bins_path),delimiter='\t')}
if not rows: raise SystemExit('no Stage-Q rows')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL Stage-Q residue mismatch '+repr({(r['label'],r['threads'],r['repeat']):r['residue'] for r in rows}))
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
for x in sorted(agg,key=rank):
    b=bins[x[1]]; print(f'STAGE_Q label={x[1]} pair_l2={b["pair_l2"]} block_l2={b["block_l2"]} threads={x[2]} wall_s={x[0]:.9f} high_s={x[3]:.9f} regs={x[4]} spill_store={x[5]} spill_load={x[6]} spill_free={int(x[7])}',file=sys.stderr)
base=min((x for x in agg if x[1]=='control'),key=rank); tests=[x for x in agg if x[1]!='control' and x[7]]; best=min(tests,key=rank) if tests else base; speed=base[0]/best[0]; enabled=int(best[1]!='control' and rank(best)<rank(base)); bb=bins[best[1]]; q=lambda x:shlex.quote(str(x))
vals={'B300_STAGEQ_ROWS':rows_arg,'B300_STAGEQ_NGPU':ngpu,'B300_STAGEQ_RESIDUE':next(iter(res)),'B300_STAGEQ_UPSTREAM_KIND':upstream,'B300_STAGEQ_STAGEP_COUNT_UPSTREAM':stagep_count,'B300_STAGEQ_PAIR_POLICY':pair_policy,'B300_STAGEQ_BLOCK_POLICY':block_policy,'B300_STAGEQ_MATE_LOAD_POLICY':mate_policy,'B300_STAGEQ_BASE_CG_L2_BYTES':base_l2,'B300_STAGEQ_BUILDER_PAIR_CG_L2_BYTES':builder_pair,'B300_STAGEQ_BUILDER_BLOCK_CG_L2_BYTES':builder_block,'B300_STAGEQ_MATE_CG_L2_BYTES':mate_l2,'B300_STAGEQ_UPSTREAM_PAIR_L2_BYTES':eff_pair,'B300_STAGEQ_UPSTREAM_BLOCK_L2_BYTES':eff_block,'B300_STAGEQ_CONTROL_BIN':bins['control']['binary'],'B300_STAGEQ_CONTROL_THREADS':base[2],'B300_STAGEQ_CONTROL_SOURCE_THREADS':control_src_threads,'B300_STAGEQ_CONTROL_WALL_S':f'{base[0]:.9f}','B300_STAGEQ_CONTROL_HIGH_S':f'{base[3]:.9f}','B300_STAGEQ_CONTROL_SPILL_FREE':1,'B300_STAGEQ_PAIR_L2_BYTES':bb['pair_l2'],'B300_STAGEQ_BLOCK_L2_BYTES':bb['block_l2'],'B300_STAGEQ_BIN':bb['binary'],'B300_STAGEQ_THREADS':best[2],'B300_STAGEQ_WALL_S':f'{best[0]:.9f}','B300_STAGEQ_HIGH_S':f'{best[3]:.9f}','B300_STAGEQ_SPILL_FREE':int(best[7]),'B300_STAGEQ_SPEEDUP':f'{speed:.9f}','B300_STAGEQ_BEST_ENABLED':enabled,'B300_STAGEQ_UPSTREAM_ROWS':up_rows,'B300_STAGEQ_UPSTREAM_RESIDUE':up_res,'B300_STAGEQ_UPSTREAM_MANIFEST':up_manifest}
with open(winner,'w') as f:
    for k,v in vals.items(): f.write(f'{k}={q(v)}\n')
print(f'b300_stageq_exact_match=1 b300_stageq_ngpu={ngpu} upstream={upstream} pair_policy={pair_policy} block_policy={block_policy} upstream_pair_l2={eff_pair} upstream_block_l2={eff_block} best_pair_l2={bb["pair_l2"]} best_block_l2={bb["block_l2"]} speedup={speed:.9f} residue={next(iter(res))}')
PY
cat "$WINNER_ENV"
