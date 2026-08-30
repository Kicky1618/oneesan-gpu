#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; ROWS="${ROWS:-1}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
STAGEL_GUARD_ENV="${STAGEL_GUARD_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_prefetch_guard_stagel_g${NGPU}_winner.env}"
STAGEM_WINNER_ENV="${STAGEM_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_mate_load_stagem_g${NGPU}_winner.env}"
UPSTREAM_KIND="${UPSTREAM_KIND:-auto}"
PAIR_POLICY_LIST="${PAIR_POLICY_LIST:-default cg cs}"; BLOCK_POLICY_LIST="${BLOCK_POLICY_LIST:-default cg cs}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"; REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_row${ROWS}_g${NGPU}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"; BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-pair-block-load-policy.sh"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")" "$(dirname "$WINNER_ENV")"
for f in "$STAGE_F_ENV" "$STAGEL_GUARD_ENV" "$BUILDER"; do [[ -s "$f" ]] || { echo "missing Stage-N input=$f" >&2; exit 2; }; done
case "$UPSTREAM_KIND" in auto|stagel|stagem) ;; *) echo 'UPSTREAM_KIND must be auto,stagel,stagem' >&2; exit 2;; esac
[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || exit 2
[[ "$NGPU" =~ ^[1-8]$ ]] || exit 2
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || exit 2
normalize_policies(){ local raw="$1" out=() p old seen; for p in $raw; do case "$p" in default|cg|cs) ;; *) echo "bad Stage-N policy=$p" >&2; exit 2;; esac; seen=0; for old in "${out[@]}"; do [[ "$old" == "$p" ]] && seen=1; done; ((seen)) || out+=("$p"); done; ((${#out[@]})) || exit 2; printf '%s' "${out[*]}"; }
PAIR_POLICY_LIST="$(normalize_policies "$PAIR_POLICY_LIST")"; BLOCK_POLICY_LIST="$(normalize_policies "$BLOCK_POLICY_LIST")"
threads=(); for th in $THREADS_LIST; do [[ "$th" =~ ^[0-9]+$ ]] && ((th>=32&&th<=1024&&th%32==0)) || exit 2; seen=0; for old in "${threads[@]}"; do [[ "$old" == "$th" ]] && seen=1; done; ((seen)) || threads+=("$th"); done; ((${#threads[@]})) || exit 2
command -v nvcc >/dev/null || exit 2; command -v nvidia-smi >/dev/null || exit 2
VISIBLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; ((VISIBLE_GPUS>=NGPU)) || { echo "need $NGPU visible GPU(s), found $VISIBLE_GPUS" >&2; exit 2; }

# Stage F owns recurrence-level build knobs and the inherited global random-load policy.
# shellcheck disable=SC1090
source "$STAGE_F_ENV"
for k in B300_HYBRID8_NEXTSELF_THRESHOLD B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK B300_HYBRID8_NEXTSELF_RANDOM_CG B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES B300_HYBRID8_NEXTSELF_PREFETCH_L2 B300_HYBRID8_NEXTSELF_DUALMASK B300_HYBRID8_NEXTSELF_CLOSURE_BATCH B300_HYBRID8_NEXTSELF_MAXRREGCOUNT; do [[ -n "${!k+x}" ]] || { echo "Stage-F env missing $k" >&2; exit 3; }; done
T="$B300_HYBRID8_NEXTSELF_THRESHOLD"; H="$B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK"; CG="$B300_HYBRID8_NEXTSELF_RANDOM_CG"; CGL2="$B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES"; PRE="$B300_HYBRID8_NEXTSELF_PREFETCH_L2"; DUAL="$B300_HYBRID8_NEXTSELF_DUALMASK"; BATCH="$B300_HYBRID8_NEXTSELF_CLOSURE_BATCH"; CAP="$B300_HYBRID8_NEXTSELF_MAXRREGCOUNT"
BASE_COUNT_POLICY=default; [[ "$CG" == 1 ]] && BASE_COUNT_POLICY=cg

# Stage L owns geometry/eviction/guard state and is always the semantic anchor.
# shellcheck disable=SC1090
source "$STAGEL_GUARD_ENV"
for k in B300_STAGEL_STAGED_VALIDATED B300_STAGEL_FINAL_ENABLED B300_STAGEL_FINAL_SPILL_FREE B300_STAGEL_NGPU B300_STAGEL_SELF_WIDTH B300_STAGEL_SELF_DISTANCE B300_STAGEL_SELF_EVICT B300_STAGEL_MATE_WIDTH B300_STAGEL_MATE_DISTANCE B300_STAGEL_MATE_EVICT B300_STAGEL_FINAL_SELF_GUARD B300_STAGEL_FINAL_MATE_GUARD B300_STAGEL_FINAL_BIN B300_STAGEL_FINAL_THREADS B300_STAGEL_FINAL_STAGE_ROWS B300_STAGEL_FINAL_STAGE_RESIDUE; do [[ -n "${!k+x}" ]] || { echo "Stage-L env missing $k" >&2; exit 3; }; done
[[ "$B300_STAGEL_NGPU" == "$NGPU" && "$B300_STAGEL_FINAL_SPILL_FREE" == 1 ]] || exit 4
SW="$B300_STAGEL_SELF_WIDTH"; SD="$B300_STAGEL_SELF_DISTANCE"; SE="$B300_STAGEL_SELF_EVICT"; MW="$B300_STAGEL_MATE_WIDTH"; MD="$B300_STAGEL_MATE_DISTANCE"; ME="$B300_STAGEL_MATE_EVICT"; SG="$B300_STAGEL_FINAL_SELF_GUARD"; MG="$B300_STAGEL_FINAL_MATE_GUARD"
L_ROWS="$B300_STAGEL_FINAL_STAGE_ROWS"; L_RES="$B300_STAGEL_FINAL_STAGE_RESIDUE"
CONTROL_BIN="$B300_STAGEL_FINAL_BIN"; CONTROL_THREADS="$B300_STAGEL_FINAL_THREADS"; MATE_POLICY=default; ACTUAL_UPSTREAM=stagel; U_ROWS="$L_ROWS"; U_RES="$L_RES"

use_m=0
if [[ "$UPSTREAM_KIND" == stagem ]]; then use_m=1
elif [[ "$UPSTREAM_KIND" == auto && -s "$STAGEM_WINNER_ENV" ]]; then
  # Probe the marker in a subshell-like parse without trusting partial files.
  if grep -Fq 'B300_STAGEM_STAGED_VALIDATED=1' "$STAGEM_WINNER_ENV" && grep -Fq 'B300_STAGEM_FINAL_ENABLED=1' "$STAGEM_WINNER_ENV"; then use_m=1; fi
fi
if ((use_m)); then
  [[ -s "$STAGEM_WINNER_ENV" ]] || { echo 'Stage-N requested Stage-M upstream but winner env missing' >&2; exit 3; }
  # shellcheck disable=SC1090
  source "$STAGEM_WINNER_ENV"
  for k in B300_STAGEM_STAGED_VALIDATED B300_STAGEM_FINAL_ENABLED B300_STAGEM_NGPU B300_STAGEM_POLICY B300_STAGEM_FINAL_BIN B300_STAGEM_FINAL_THREADS B300_STAGEM_FINAL_SPILL_FREE B300_STAGEM_SELF_WIDTH B300_STAGEM_SELF_DISTANCE B300_STAGEM_SELF_EVICT B300_STAGEM_SELF_GUARD B300_STAGEM_MATE_WIDTH B300_STAGEM_MATE_DISTANCE B300_STAGEM_MATE_EVICT B300_STAGEM_MATE_GUARD B300_STAGEM_FINAL_STAGE_ROWS B300_STAGEM_FINAL_STAGE_RESIDUE; do [[ -n "${!k+x}" ]] || { echo "Stage-M env missing $k" >&2; exit 3; }; done
  [[ "$B300_STAGEM_STAGED_VALIDATED" == 1 && "$B300_STAGEM_FINAL_ENABLED" == 1 && "$B300_STAGEM_FINAL_SPILL_FREE" == 1 && "$B300_STAGEM_NGPU" == "$NGPU" ]] || exit 4
  [[ "$B300_STAGEM_SELF_WIDTH" == "$SW" && "$B300_STAGEM_SELF_DISTANCE" == "$SD" && "$B300_STAGEM_SELF_EVICT" == "$SE" && "$B300_STAGEM_SELF_GUARD" == "$SG" && "$B300_STAGEM_MATE_WIDTH" == "$MW" && "$B300_STAGEM_MATE_DISTANCE" == "$MD" && "$B300_STAGEM_MATE_EVICT" == "$ME" && "$B300_STAGEM_MATE_GUARD" == "$MG" ]] || { echo 'Stage-N Stage-M/Stage-L policy drift' >&2; exit 3; }
  case "$B300_STAGEM_POLICY" in cg|cs) ;; *) exit 3;; esac
  CONTROL_BIN="$B300_STAGEM_FINAL_BIN"; CONTROL_THREADS="$B300_STAGEM_FINAL_THREADS"; MATE_POLICY="$B300_STAGEM_POLICY"; ACTUAL_UPSTREAM=stagem; U_ROWS="$B300_STAGEM_FINAL_STAGE_ROWS"; U_RES="$B300_STAGEM_FINAL_STAGE_RESIDUE"
elif [[ "$UPSTREAM_KIND" == stagem ]]; then exit 3
fi
[[ -x "$CONTROL_BIN" ]] || { echo "Stage-N control binary missing=$CONTROL_BIN" >&2; exit 3; }

BINS="$LOGDIR/binaries.tsv"; printf 'label\tpair\tblock\tbinary\tbuild_err\n' >"$BINS"
printf 'control\t%s\t%s\t%s\t-\n' "$BASE_COUNT_POLICY" "$BASE_COUNT_POLICY" "$CONTROL_BIN" >>"$BINS"
for pair in $PAIR_POLICY_LIST; do
  for block in $BLOCK_POLICY_LIST; do
    [[ "$pair" == "$BASE_COUNT_POLICY" && "$block" == "$BASE_COUNT_POLICY" ]] && continue
    label="p${pair}_b${block}"; bin="$ONEESAN_BUILD_DIR/b300_stagen_${label}_mp${MATE_POLICY}_sw${SW}d${SD}_mw${MW}d${MD}_n27"; err="$LOGDIR/${label}.build.err"; bout="$LOGDIR/${label}.build.out"
    N=27 ARCH="$ARCH" OUT="$bin" MATE_LOAD_POLICY="$MATE_POLICY" PAIR_LOAD_POLICY="$pair" BLOCK_LOAD_POLICY="$block" STAGEN_CG_L2_FETCH_BYTES="$CGL2" HIGH_DROP_CHUNK="$H" HYBRID_THRESHOLD="$T" \
      SELF_WIDTH="$SW" SELF_DISTANCE="$SD" MATE_WIDTH="$MW" MATE_DISTANCE="$MD" SELF_EVICT="$SE" MATE_EVICT="$ME" SELF_GUARD="$SG" MATE_GUARD="$MG" RANDOM_CG="$CG" RANDOM_CG_L2_FETCH_BYTES="$CGL2" PREFETCH_L2="$PRE" DUALMASK="$DUAL" CLOSURE_BATCH="$BATCH" MAXRREGCOUNT="$CAP" PTXAS_VERBOSE=1 BUILD_ERR="$err" \
      bash "$BUILDER" >"$bout" 2>"$LOGDIR/${label}.driver.err"
    [[ -x "$bin" ]] || exit 3
    grep -Fq "pair_load_policy=$pair block_load_policy=$block" "$bout" || exit 3
    grep -Fq "mate_load_policy=$MATE_POLICY" "$bout" || exit 3
    grep -Fq 'stage_n_scope=pair_block_count_reads_only' "$bout" || exit 3
    printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$pair" "$block" "$bin" "$err" >>"$BINS"
  done
done

printf 'label\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
while IFS=$'\t' read -r label pair block bin err; do
  [[ "$label" == label || "$label" == control ]] && continue
  python3 "$PARSER" "$err" --label "$label" --contains main_pull_kernel_ilp2 >>"$RESOURCE" || true
  python3 "$PARSER" "$err" --label "$label" --contains main_pull_kernel_ilp8_hybrid >>"$RESOURCE" || true
done <"$BINS"
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
printf 'label\tpair\tblock\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tforward_high_s\treverse_high_s\n' >"$RESULT"
run_one(){ local label="$1" pair="$2" block="$3" bin="$4" th="$5" r="$6" so="$LOGDIR/${label}_t${th}_r${r}.out" se="$LOGDIR/${label}_t${th}_r${r}.err" line; B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$th" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"; line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { tail -n 80 "$se" >&2 || true; return 4; }; printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$pair" "$block" "$th" "$r" "$(field residue "$line")" "$(field wall_s "$line")" "$(field active_max_s "$line")" "$(field active_sum_s "$line")" "$(field forward_high_s "$line")" "$(field reverse_high_s "$line")" >>"$RESULT"; }
while IFS=$'\t' read -r label pair block bin err; do [[ "$label" == label ]] && continue; for th in "${threads[@]}"; do for ((r=1;r<=REPEATS;++r)); do echo "=== Stage N label=$label pair=$pair block=$block threads=$th repeat=$r rows=$ROWS upstream=$ACTUAL_UPSTREAM mate_load=$MATE_POLICY ===" >&2; run_one "$label" "$pair" "$block" "$bin" "$th" "$r"; done; done; done <"$BINS"

python3 - "$RESULT" "$RESOURCE" "$BINS" "$WINNER_ENV" "$ROWS" "$NGPU" "$ACTUAL_UPSTREAM" "$MATE_POLICY" "$BASE_COUNT_POLICY" "$CONTROL_THREADS" "$SW" "$SD" "$SE" "$SG" "$MW" "$MD" "$ME" "$MG" <<'PY'
import csv,math,statistics,sys,shlex
result,resource,bins_path,winner,rows_arg,ngpu,upstream,mate_policy,base_policy,control_source_threads,sw,sd,se,sg,mw,md,me,mg=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t')); rr=list(csv.DictReader(open(resource),delimiter='\t')); bins={r['label']:r for r in csv.DictReader(open(bins_path),delimiter='\t')}
if not rows: raise SystemExit('no Stage-N benchmark rows')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL Stage-N residue mismatch '+repr({(r['label'],r['threads'],r['repeat']):r['residue'] for r in rows}))
resources={k:[] for k in bins}
for r in rr:
    try: resources[r['label']].append((int(r['registers']),int(r['spill_store_bytes']),int(r['spill_load_bytes'])))
    except (ValueError,KeyError): pass
agg=[]
for label,t in {(r['label'],int(r['threads'])) for r in rows}:
    rs=[r for r in rows if r['label']==label and int(r['threads'])==t]
    wall=statistics.median(float(r['wall_s']) for r in rs); hs=[]
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
    b=bins[x[1]]; print(f'STAGE_N label={x[1]} pair={b["pair"]} block={b["block"]} threads={x[2]} wall_s={x[0]:.9f} high_s={x[3]:.9f} regs={x[4]} spill_store={x[5]} spill_load={x[6]} spill_free={int(x[7])}',file=sys.stderr)
base=min((x for x in agg if x[1]=='control'),key=rank); tests=[x for x in agg if x[1]!='control' and x[7]]; best=min(tests,key=rank) if tests else base
speed=base[0]/best[0]; enabled=int(best[1]!='control' and rank(best)<rank(base)); bb=bins[best[1]]; q=lambda x:shlex.quote(str(x))
vals={
 'B300_STAGEN_ROWS':rows_arg,'B300_STAGEN_NGPU':ngpu,'B300_STAGEN_RESIDUE':next(iter(res)),'B300_STAGEN_UPSTREAM_KIND':upstream,'B300_STAGEN_MATE_LOAD_POLICY':mate_policy,'B300_STAGEN_BASE_COUNT_POLICY':base_policy,
 'B300_STAGEN_SELF_WIDTH':sw,'B300_STAGEN_SELF_DISTANCE':sd,'B300_STAGEN_SELF_EVICT':se,'B300_STAGEN_SELF_GUARD':sg,'B300_STAGEN_MATE_WIDTH':mw,'B300_STAGEN_MATE_DISTANCE':md,'B300_STAGEN_MATE_EVICT':me,'B300_STAGEN_MATE_GUARD':mg,
 'B300_STAGEN_CONTROL_BIN':bins['control']['binary'],'B300_STAGEN_CONTROL_THREADS':base[2],'B300_STAGEN_CONTROL_SOURCE_THREADS':control_source_threads,'B300_STAGEN_CONTROL_WALL_S':f'{base[0]:.9f}','B300_STAGEN_CONTROL_HIGH_S':f'{base[3]:.9f}','B300_STAGEN_CONTROL_SPILL_FREE':1,
 'B300_STAGEN_PAIR_POLICY':bb['pair'],'B300_STAGEN_BLOCK_POLICY':bb['block'],'B300_STAGEN_BIN':bb['binary'],'B300_STAGEN_THREADS':best[2],'B300_STAGEN_WALL_S':f'{best[0]:.9f}','B300_STAGEN_HIGH_S':f'{best[3]:.9f}','B300_STAGEN_SPILL_FREE':int(best[7]),'B300_STAGEN_SPEEDUP':f'{speed:.9f}','B300_STAGEN_BEST_ENABLED':enabled,
}
with open(winner,'w') as f:
    for k,v in vals.items(): f.write(f'{k}={q(v)}\n')
print(f'b300_stagen_exact_match=1 b300_stagen_ngpu={ngpu} upstream={upstream} mate_load={mate_policy} base_count={base_policy} best_pair={bb["pair"]} best_block={bb["block"]} speedup={speed:.9f} residue={next(iter(res))}')
PY
cat "$WINNER_ENV"
