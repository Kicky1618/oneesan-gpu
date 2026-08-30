#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; ROWS="${ROWS:-1}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
STAGEN_WINNER_ENV="${STAGEN_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_staged_g${NGPU}_winner.env}"
STAGEN_PREPARE_ENV="${STAGEN_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_fullprime_n27_prepared.env}"
L2_LIST="${L2_LIST:-0 64 128 256}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"; REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_stageno_cgl2_row${ROWS}_g${NGPU}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-pair-block-load-policy.sh"; PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")" "$(dirname "$WINNER_ENV")"
for f in "$STAGE_F_ENV" "$STAGEN_WINNER_ENV" "$STAGEN_PREPARE_ENV" "$BUILDER"; do [[ -s "$f" ]] || { echo "missing Stage-O input=$f" >&2; exit 2; }; done
[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || exit 2; [[ "$NGPU" =~ ^[1-8]$ ]] || exit 2; [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || exit 2
command -v nvcc >/dev/null || exit 2; command -v nvidia-smi >/dev/null || exit 2; command -v sha256sum >/dev/null || exit 2
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }

# Stage F fixes every cache policy except Stage-O's pair/block .cg L2 hint.
# shellcheck disable=SC1090
source "$STAGE_F_ENV"
for k in B300_HYBRID8_NEXTSELF_THRESHOLD B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK B300_HYBRID8_NEXTSELF_RANDOM_CG B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES B300_HYBRID8_NEXTSELF_PREFETCH_L2 B300_HYBRID8_NEXTSELF_DUALMASK B300_HYBRID8_NEXTSELF_CLOSURE_BATCH B300_HYBRID8_NEXTSELF_MAXRREGCOUNT; do [[ -n "${!k+x}" ]] || exit 3; done
T="$B300_HYBRID8_NEXTSELF_THRESHOLD"; H="$B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK"; CG="$B300_HYBRID8_NEXTSELF_RANDOM_CG"; BASE_L2="$B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES"; PRE="$B300_HYBRID8_NEXTSELF_PREFETCH_L2"; DUAL="$B300_HYBRID8_NEXTSELF_DUALMASK"; BATCH="$B300_HYBRID8_NEXTSELF_CLOSURE_BATCH"; CAP="$B300_HYBRID8_NEXTSELF_MAXRREGCOUNT"
case "$BASE_L2" in 0|64|128|256) ;; *) exit 3;; esac

# Stage N fixes pair/block/mate policies and geometry; its prepared binary is the exact control.
# shellcheck disable=SC1090
source "$STAGEN_PREPARE_ENV"
for k in B300_STAGEN_PREPARED B300_STAGEN_PREPARED_MOD B300_STAGEN_PREPARED_NGPU B300_STAGEN_PREPARED_MATE_LOAD_POLICY B300_STAGEN_PREPARED_BASE_COUNT_POLICY B300_STAGEN_PREPARED_PAIR_POLICY B300_STAGEN_PREPARED_BLOCK_POLICY B300_STAGEN_PREPARED_BIN B300_STAGEN_PREPARED_THREADS B300_STAGEN_PREPARED_MANIFEST; do [[ -n "${!k+x}" ]] || { echo "Stage-N prepare missing $k" >&2; exit 3; }; done
[[ "$B300_STAGEN_PREPARED" == 1 && "$B300_STAGEN_PREPARED_MOD" == "$MOD" && "$B300_STAGEN_PREPARED_NGPU" == "$NGPU" ]] || exit 3
[[ -x "$B300_STAGEN_PREPARED_BIN" && -s "$B300_STAGEN_PREPARED_MANIFEST" ]] || exit 3; sha256sum -c "$B300_STAGEN_PREPARED_MANIFEST" >/dev/null || { echo 'Stage-N manifest failed before Stage O' >&2; exit 3; }
MP="$B300_STAGEN_PREPARED_MATE_LOAD_POLICY"; PP="$B300_STAGEN_PREPARED_PAIR_POLICY"; BP="$B300_STAGEN_PREPARED_BLOCK_POLICY"; BASE_COUNT="$B300_STAGEN_PREPARED_BASE_COUNT_POLICY"; CONTROL_BIN="$B300_STAGEN_PREPARED_BIN"; CONTROL_THREADS="$B300_STAGEN_PREPARED_THREADS"
for p in "$MP" "$PP" "$BP" "$BASE_COUNT"; do case "$p" in default|cg|cs) ;; *) exit 3;; esac; done
[[ "$PP" == cg || "$BP" == cg ]] || { echo 'Stage O not applicable: Stage-N winner has no cg pair/block load' >&2; exit 4; }
# shellcheck disable=SC1090
source "$STAGEN_WINNER_ENV"
for k in B300_STAGEN_STAGED_VALIDATED B300_STAGEN_FINAL_ENABLED B300_STAGEN_FINAL_SPILL_FREE B300_STAGEN_FINAL_BIN B300_STAGEN_SELF_WIDTH B300_STAGEN_SELF_DISTANCE B300_STAGEN_SELF_EVICT B300_STAGEN_SELF_GUARD B300_STAGEN_MATE_WIDTH B300_STAGEN_MATE_DISTANCE B300_STAGEN_MATE_EVICT B300_STAGEN_MATE_GUARD B300_STAGEN_FINAL_STAGE_ROWS B300_STAGEN_FINAL_STAGE_RESIDUE; do [[ -n "${!k+x}" ]] || exit 3; done
[[ "$B300_STAGEN_STAGED_VALIDATED" == 1 && "$B300_STAGEN_FINAL_ENABLED" == 1 && "$B300_STAGEN_FINAL_SPILL_FREE" == 1 && "$B300_STAGEN_FINAL_BIN" == "$CONTROL_BIN" ]] || exit 3
SW="$B300_STAGEN_SELF_WIDTH"; SD="$B300_STAGEN_SELF_DISTANCE"; SE="$B300_STAGEN_SELF_EVICT"; SG="$B300_STAGEN_SELF_GUARD"; MW="$B300_STAGEN_MATE_WIDTH"; MD="$B300_STAGEN_MATE_DISTANCE"; ME="$B300_STAGEN_MATE_EVICT"; MG="$B300_STAGEN_MATE_GUARD"; N_ROWS="$B300_STAGEN_FINAL_STAGE_ROWS"; N_RES="$B300_STAGEN_FINAL_STAGE_RESIDUE"

l2s=(); for x in $L2_LIST; do case "$x" in 0|64|128|256) ;; *) echo "bad Stage-O L2 hint=$x" >&2; exit 2;; esac; seen=0; for old in "${l2s[@]}"; do [[ "$old" == "$x" ]] && seen=1; done; ((seen)) || l2s+=("$x"); done
has_base=0; for x in "${l2s[@]}"; do [[ "$x" == "$BASE_L2" ]] && has_base=1; done; ((has_base)) || { echo "L2_LIST must include inherited baseline=$BASE_L2" >&2; exit 2; }
threads=(); for th in $THREADS_LIST; do [[ "$th" =~ ^[0-9]+$ ]] && ((th>=32&&th<=1024&&th%32==0)) || exit 2; seen=0; for old in "${threads[@]}"; do [[ "$old" == "$th" ]] && seen=1; done; ((seen)) || threads+=("$th"); done

BINS="$LOGDIR/binaries.tsv"; printf 'l2\tbinary\tbuild_err\n' >"$BINS"; printf '%s\t%s\t-\n' "$BASE_L2" "$CONTROL_BIN" >>"$BINS"
for l2 in "${l2s[@]}"; do
  [[ "$l2" == "$BASE_L2" ]] && continue
  bin="$ONEESAN_BUILD_DIR/b300_stageo_l2${l2}_p${PP}_b${BP}_mp${MP}_n27"; err="$LOGDIR/l2${l2}.build.err"; bout="$LOGDIR/l2${l2}.build.out"
  N=27 ARCH="$ARCH" OUT="$bin" MATE_LOAD_POLICY="$MP" PAIR_LOAD_POLICY="$PP" BLOCK_LOAD_POLICY="$BP" STAGEN_CG_L2_FETCH_BYTES="$l2" HIGH_DROP_CHUNK="$H" HYBRID_THRESHOLD="$T" \
    SELF_WIDTH="$SW" SELF_DISTANCE="$SD" MATE_WIDTH="$MW" MATE_DISTANCE="$MD" SELF_EVICT="$SE" MATE_EVICT="$ME" SELF_GUARD="$SG" MATE_GUARD="$MG" RANDOM_CG="$CG" RANDOM_CG_L2_FETCH_BYTES="$BASE_L2" PREFETCH_L2="$PRE" DUALMASK="$DUAL" CLOSURE_BATCH="$BATCH" MAXRREGCOUNT="$CAP" PTXAS_VERBOSE=1 BUILD_ERR="$err" \
    bash "$BUILDER" >"$bout" 2>"$LOGDIR/l2${l2}.driver.err"
  [[ -x "$bin" ]] || exit 3; grep -Fq "pair_load_policy=$PP block_load_policy=$BP" "$bout" || exit 3; grep -Fq "stagen_cg_l2_fetch_bytes=$l2" "$bout" || exit 3
  printf '%s\t%s\t%s\n' "$l2" "$bin" "$err" >>"$BINS"
done

printf 'l2\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
while IFS=$'\t' read -r l2 bin err; do [[ "$l2" == l2 || "$l2" == "$BASE_L2" ]] && continue; python3 "$PARSER" "$err" --label "$l2" --contains main_pull_kernel_ilp2 >>"$RESOURCE" || true; python3 "$PARSER" "$err" --label "$l2" --contains main_pull_kernel_ilp8_hybrid >>"$RESOURCE" || true; done <"$BINS"
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
printf 'l2\tthreads\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\n' >"$RESULT"
run_one(){ local l2="$1" bin="$2" th="$3" r="$4" so="$LOGDIR/l2${l2}_t${th}_r${r}.out" se="$LOGDIR/l2${l2}_t${th}_r${r}.err" line; B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$th" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"; line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || return 4; printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$l2" "$th" "$r" "$(field residue "$line")" "$(field wall_s "$line")" "$(field forward_high_s "$line")" "$(field reverse_high_s "$line")" >>"$RESULT"; }
while IFS=$'\t' read -r l2 bin err; do [[ "$l2" == l2 ]] && continue; for th in "${threads[@]}"; do for ((r=1;r<=REPEATS;++r)); do echo "=== Stage O l2=$l2 threads=$th repeat=$r rows=$ROWS pair=$PP block=$BP ===" >&2; run_one "$l2" "$bin" "$th" "$r"; done; done; done <"$BINS"

python3 - "$RESULT" "$RESOURCE" "$BINS" "$WINNER_ENV" "$ROWS" "$NGPU" "$BASE_L2" "$PP" "$BP" "$MP" "$CONTROL_THREADS" "$N_ROWS" "$N_RES" <<'PY'
import csv,math,statistics,sys,shlex
result,resource,bins_path,winner,rows_arg,ngpu,base_l2,pp,bp,mp,control_source_threads,n_rows,n_res=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t')); rr=list(csv.DictReader(open(resource),delimiter='\t')); bins={r['l2']:r for r in csv.DictReader(open(bins_path),delimiter='\t')}
if not rows: raise SystemExit('no Stage-O rows')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL Stage-O residue mismatch '+repr({(r['l2'],r['threads'],r['repeat']):r['residue'] for r in rows}))
if rows_arg==n_rows and next(iter(res))!=n_res: raise SystemExit('FATAL Stage-O/Stage-N residue mismatch')
resources={k:[] for k in bins}
for r in rr:
    try: resources[r['l2']].append((int(r['registers']),int(r['spill_store_bytes']),int(r['spill_load_bytes'])))
    except (ValueError,KeyError): pass
agg=[]
for l2,t in {(r['l2'],int(r['threads'])) for r in rows}:
    rs=[r for r in rows if r['l2']==l2 and int(r['threads'])==t]; wall=statistics.median(float(r['wall_s']) for r in rs); high=statistics.median(float(r['forward_high_s'])+float(r['reverse_high_s']) for r in rs)
    if l2==base_l2: regs=-1; ss=sl=0; clean=True
    else:
        rv=resources[l2]; regs=max((x[0] for x in rv),default=-1); ss=max((x[1] for x in rv),default=-1); sl=max((x[2] for x in rv),default=-1); clean=len(rv)>=2 and ss==0 and sl==0
    agg.append((wall,l2,t,high,regs,ss,sl,clean))
def rank(x): return (x[0],x[3],x[4] if x[4]>=0 else math.inf,x[2])
base=min((x for x in agg if x[1]==base_l2),key=rank); tests=[x for x in agg if x[1]!=base_l2 and x[7]]; best=min(tests,key=rank) if tests else base; speed=base[0]/best[0]; enabled=int(best[1]!=base_l2 and rank(best)<rank(base)); q=lambda x:shlex.quote(str(x))
vals={'B300_STAGEO_ROWS':rows_arg,'B300_STAGEO_NGPU':ngpu,'B300_STAGEO_RESIDUE':next(iter(res)),'B300_STAGEO_PAIR_POLICY':pp,'B300_STAGEO_BLOCK_POLICY':bp,'B300_STAGEO_MATE_LOAD_POLICY':mp,'B300_STAGEO_BASE_L2_BYTES':base_l2,'B300_STAGEO_CONTROL_BIN':bins[base_l2]['binary'],'B300_STAGEO_CONTROL_THREADS':base[2],'B300_STAGEO_CONTROL_SOURCE_THREADS':control_source_threads,'B300_STAGEO_CONTROL_WALL_S':f'{base[0]:.9f}','B300_STAGEO_CONTROL_SPILL_FREE':1,'B300_STAGEO_L2_BYTES':best[1],'B300_STAGEO_BIN':bins[best[1]]['binary'],'B300_STAGEO_THREADS':best[2],'B300_STAGEO_WALL_S':f'{best[0]:.9f}','B300_STAGEO_SPILL_FREE':int(best[7]),'B300_STAGEO_SPEEDUP':f'{speed:.9f}','B300_STAGEO_BEST_ENABLED':enabled}
with open(winner,'w') as f:
    for k,v in vals.items(): f.write(f'{k}={q(v)}\n')
print(f'b300_stageo_exact_match=1 base_l2={base_l2} best_l2={best[1]} speedup={speed:.9f} residue={next(iter(res))}')
PY
cat "$WINNER_ENV"
