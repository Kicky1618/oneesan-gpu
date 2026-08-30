#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; ROWS="${ROWS:-1}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"
STAGE_F_ENV="${STAGE_F_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
STAGEN_WINNER_ENV="${STAGEN_WINNER_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_staged_g${NGPU}_winner.env}"
STAGEN_PREPARE_ENV="${STAGEN_PREPARE_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_pair_block_load_stagen_fullprime_n27_prepared.env}"
FETCH_LIST="${FETCH_LIST:-0 64 128 256}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"; REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_count_cg_l2_stageo_row${ROWS}_g${NGPU}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-pair-block-load-policy.sh"; PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"

mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")" "$(dirname "$WINNER_ENV")"
for f in "$STAGE_F_ENV" "$STAGEN_WINNER_ENV" "$STAGEN_PREPARE_ENV" "$BUILDER"; do [[ -s "$f" ]] || { echo "missing Stage-O input=$f" >&2; exit 2; }; done
[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || exit 2
[[ "$NGPU" =~ ^[1-8]$ ]] || exit 2
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || exit 2
command -v nvcc >/dev/null || exit 2; command -v nvidia-smi >/dev/null || exit 2; command -v sha256sum >/dev/null || exit 2
VISIBLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; ((VISIBLE_GPUS>=NGPU)) || { echo "need $NGPU visible GPU(s), found $VISIBLE_GPUS" >&2; exit 2; }

fetches=()
for fb in $FETCH_LIST; do
  case "$fb" in 0|64|128|256) ;; *) echo "bad Stage-O fetch bytes=$fb" >&2; exit 2;; esac
  seen=0; for old in "${fetches[@]}"; do [[ "$old" == "$fb" ]] && seen=1; done; ((seen)) || fetches+=("$fb")
done
((${#fetches[@]})) || exit 2
threads=()
for th in $THREADS_LIST; do
  [[ "$th" =~ ^[0-9]+$ ]] && ((th>=32&&th<=1024&&th%32==0)) || { echo "bad Stage-O threads=$th" >&2; exit 2; }
  seen=0; for old in "${threads[@]}"; do [[ "$old" == "$th" ]] && seen=1; done; ((seen)) || threads+=("$th")
done
((${#threads[@]})) || exit 2

# Stage F supplies recurrence build knobs and the inherited CG fetch-size baseline.
# shellcheck disable=SC1090
source "$STAGE_F_ENV"
for k in B300_HYBRID8_NEXTSELF_THRESHOLD B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK B300_HYBRID8_NEXTSELF_RANDOM_CG B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES B300_HYBRID8_NEXTSELF_PREFETCH_L2 B300_HYBRID8_NEXTSELF_DUALMASK B300_HYBRID8_NEXTSELF_CLOSURE_BATCH B300_HYBRID8_NEXTSELF_MAXRREGCOUNT; do
  [[ -n "${!k+x}" ]] || { echo "Stage-F env missing $k" >&2; exit 3; }
done
T="$B300_HYBRID8_NEXTSELF_THRESHOLD"; H="$B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK"; CG="$B300_HYBRID8_NEXTSELF_RANDOM_CG"; BASE_FETCH="$B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES"; PRE="$B300_HYBRID8_NEXTSELF_PREFETCH_L2"; DUAL="$B300_HYBRID8_NEXTSELF_DUALMASK"; BATCH="$B300_HYBRID8_NEXTSELF_CLOSURE_BATCH"; CAP="$B300_HYBRID8_NEXTSELF_MAXRREGCOUNT"
case "$BASE_FETCH" in 0|64|128|256) ;; *) echo "bad inherited Stage-O baseline fetch=$BASE_FETCH" >&2; exit 3;; esac
has_base=0; for fb in "${fetches[@]}"; do [[ "$fb" == "$BASE_FETCH" ]] && has_base=1; done
((has_base)) || { echo "Stage-O FETCH_LIST must include inherited baseline=$BASE_FETCH" >&2; exit 2; }

# Stage N winner freezes pair/block policy and all geometry/guard state.
# shellcheck disable=SC1090
source "$STAGEN_WINNER_ENV"
for k in B300_STAGEN_STAGED_VALIDATED B300_STAGEN_FINAL_ENABLED B300_STAGEN_NGPU B300_STAGEN_UPSTREAM_KIND B300_STAGEN_MATE_LOAD_POLICY B300_STAGEN_BASE_COUNT_POLICY B300_STAGEN_PAIR_POLICY B300_STAGEN_BLOCK_POLICY B300_STAGEN_FINAL_BIN B300_STAGEN_FINAL_THREADS B300_STAGEN_FINAL_SPILL_FREE B300_STAGEN_SELF_WIDTH B300_STAGEN_SELF_DISTANCE B300_STAGEN_SELF_EVICT B300_STAGEN_SELF_GUARD B300_STAGEN_MATE_WIDTH B300_STAGEN_MATE_DISTANCE B300_STAGEN_MATE_EVICT B300_STAGEN_MATE_GUARD B300_STAGEN_FINAL_STAGE_ROWS B300_STAGEN_FINAL_STAGE_RESIDUE; do
  [[ -n "${!k+x}" ]] || { echo "Stage-N winner missing $k" >&2; exit 3; }
done
[[ "$B300_STAGEN_STAGED_VALIDATED" == 1 && "$B300_STAGEN_FINAL_ENABLED" == 1 && "$B300_STAGEN_FINAL_SPILL_FREE" == 1 && "$B300_STAGEN_NGPU" == "$NGPU" ]] || { echo 'Stage N not promotable for Stage O' >&2; exit 4; }
PAIR="$B300_STAGEN_PAIR_POLICY"; BLOCK="$B300_STAGEN_BLOCK_POLICY"; MP="$B300_STAGEN_MATE_LOAD_POLICY"
case "$PAIR" in default|cg|cs) ;; *) exit 3;; esac; case "$BLOCK" in default|cg|cs) ;; *) exit 3;; esac; case "$MP" in default|cg|cs) ;; *) exit 3;; esac
if [[ "$PAIR" != cg && "$BLOCK" != cg ]]; then
  echo 'Stage O not applicable: Stage N selected no Count cg load' >&2
  exit 4
fi
SW="$B300_STAGEN_SELF_WIDTH"; SD="$B300_STAGEN_SELF_DISTANCE"; SE="$B300_STAGEN_SELF_EVICT"; SG="$B300_STAGEN_SELF_GUARD"
MW="$B300_STAGEN_MATE_WIDTH"; MD="$B300_STAGEN_MATE_DISTANCE"; ME="$B300_STAGEN_MATE_EVICT"; MG="$B300_STAGEN_MATE_GUARD"
U_ROWS="$B300_STAGEN_FINAL_STAGE_ROWS"; U_RES="$B300_STAGEN_FINAL_STAGE_RESIDUE"

# Stage N prepare gives the exact binary to use as the unrebuilt baseline.
# shellcheck disable=SC1090
source "$STAGEN_PREPARE_ENV"
for k in B300_STAGEN_PREPARED B300_STAGEN_PREPARED_MOD B300_STAGEN_PREPARED_NGPU B300_STAGEN_PREPARED_PAIR_POLICY B300_STAGEN_PREPARED_BLOCK_POLICY B300_STAGEN_PREPARED_BIN B300_STAGEN_PREPARED_THREADS B300_STAGEN_PREPARED_MANIFEST; do
  [[ -n "${!k+x}" ]] || { echo "Stage-N prepare missing $k" >&2; exit 3; }
done
[[ "$B300_STAGEN_PREPARED" == 1 && "$B300_STAGEN_PREPARED_MOD" == "$MOD" && "$B300_STAGEN_PREPARED_NGPU" == "$NGPU" ]] || { echo 'Stage-O/Stage-N prepare mismatch' >&2; exit 3; }
[[ "$B300_STAGEN_PREPARED_PAIR_POLICY" == "$PAIR" && "$B300_STAGEN_PREPARED_BLOCK_POLICY" == "$BLOCK" ]] || { echo 'Stage-O Stage-N winner/prepare policy drift' >&2; exit 3; }
[[ "$B300_STAGEN_PREPARED_BIN" == "$B300_STAGEN_FINAL_BIN" && -x "$B300_STAGEN_PREPARED_BIN" && -s "$B300_STAGEN_PREPARED_MANIFEST" ]] || { echo 'Stage-O exact Stage-N control identity mismatch' >&2; exit 3; }
sha256sum -c "$B300_STAGEN_PREPARED_MANIFEST" >/dev/null || { echo 'Stage-N promotion manifest failed before Stage O' >&2; exit 3; }
CONTROL_BIN="$B300_STAGEN_PREPARED_BIN"

BINS="$LOGDIR/binaries.tsv"; printf 'label\tfetch_bytes\tbinary\tbuild_err\n' >"$BINS"
printf 'control\t%s\t%s\t-\n' "$BASE_FETCH" "$CONTROL_BIN" >>"$BINS"
for fb in "${fetches[@]}"; do
  [[ "$fb" == "$BASE_FETCH" ]] && continue
  label="l2${fb}"; bin="$ONEESAN_BUILD_DIR/b300_stageo_${label}_pair${PAIR}_block${BLOCK}_mp${MP}_sw${SW}d${SD}_mw${MW}d${MD}_n27"; err="$LOGDIR/${label}.build.err"; bout="$LOGDIR/${label}.build.out"
  N=27 ARCH="$ARCH" OUT="$bin" MATE_LOAD_POLICY="$MP" PAIR_LOAD_POLICY="$PAIR" BLOCK_LOAD_POLICY="$BLOCK" STAGEN_CG_L2_FETCH_BYTES="$fb" \
    HIGH_DROP_CHUNK="$H" HYBRID_THRESHOLD="$T" SELF_WIDTH="$SW" SELF_DISTANCE="$SD" MATE_WIDTH="$MW" MATE_DISTANCE="$MD" SELF_EVICT="$SE" MATE_EVICT="$ME" SELF_GUARD="$SG" MATE_GUARD="$MG" \
    RANDOM_CG="$CG" RANDOM_CG_L2_FETCH_BYTES="$BASE_FETCH" PREFETCH_L2="$PRE" DUALMASK="$DUAL" CLOSURE_BATCH="$BATCH" MAXRREGCOUNT="$CAP" PTXAS_VERBOSE=1 BUILD_ERR="$err" \
    bash "$BUILDER" >"$bout" 2>"$LOGDIR/${label}.driver.err"
  [[ -x "$bin" ]] || { echo "Stage-O binary missing fetch=$fb" >&2; exit 3; }
  grep -Fq "pair_load_policy=$PAIR block_load_policy=$BLOCK stagen_cg_l2_fetch_bytes=$fb" "$bout" || exit 3
  grep -Fq "mate_load_policy=$MP" "$bout" || exit 3
  grep -Fq "self_geometry width=$SW distance=$SD evict=$SE guard=$SG" "$bout" || exit 3
  grep -Fq "mate_geometry width=$MW distance=$MD evict=$ME guard=$MG" "$bout" || exit 3
  printf '%s\t%s\t%s\t%s\n' "$label" "$fb" "$bin" "$err" >>"$BINS"
done

printf 'label\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
while IFS=$'\t' read -r label fb bin err; do
  [[ "$label" == label || "$label" == control ]] && continue
  python3 "$PARSER" "$err" --label "$label" --contains main_pull_kernel_ilp2 >>"$RESOURCE" || true
  python3 "$PARSER" "$err" --label "$label" --contains main_pull_kernel_ilp8_hybrid >>"$RESOURCE" || true
done <"$BINS"

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
printf 'label\tfetch_bytes\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tforward_high_s\treverse_high_s\n' >"$RESULT"
run_one(){
  local label="$1" fb="$2" bin="$3" th="$4" r="$5" so="$LOGDIR/${label}_t${th}_r${r}.out" se="$LOGDIR/${label}_t${th}_r${r}.err" line
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$th" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { tail -n 80 "$se" >&2 || true; return 4; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$fb" "$th" "$r" "$(field residue "$line")" "$(field wall_s "$line")" "$(field active_max_s "$line")" "$(field active_sum_s "$line")" "$(field forward_high_s "$line")" "$(field reverse_high_s "$line")" >>"$RESULT"
}
for th in "${threads[@]}"; do
  while IFS=$'\t' read -r label fb bin err; do
    [[ "$label" == label ]] && continue
    for ((r=1;r<=REPEATS;++r)); do echo "=== Stage O fetch=$fb threads=$th repeat=$r rows=$ROWS ===" >&2; run_one "$label" "$fb" "$bin" "$th" "$r"; done
  done <"$BINS"
done

python3 - "$RESULT" "$RESOURCE" "$BINS" "$WINNER_ENV" "$ROWS" "$NGPU" "$BASE_FETCH" "$PAIR" "$BLOCK" "$MP" "$SW" "$SD" "$SE" "$SG" "$MW" "$MD" "$ME" "$MG" "$CONTROL_BIN" "$U_ROWS" "$U_RES" <<'PY'
import csv,math,statistics,sys,shlex
(result,resource,bins_path,winner,rows_arg,ngpu,base_fetch,pair,block,mp,sw,sd,se,sg,mw,md,me,mg,control_bin,urows,ures)=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t')); rr=list(csv.DictReader(open(resource),delimiter='\t')); bins={r['label']:r for r in csv.DictReader(open(bins_path),delimiter='\t')}
if not rows: raise SystemExit('no Stage-O rows')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL Stage-O residue mismatch '+repr({(r['label'],r['threads'],r['repeat']):r['residue'] for r in rows}))
if rows_arg==urows and next(iter(res))!=ures: raise SystemExit(f'FATAL Stage-O/Stage-N residue mismatch rows={rows_arg} got={next(iter(res))} expected={ures}')
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
    high=statistics.median(hs) if hs else math.nan
    if label=='control': regs=ss=sl=-1; clean=True
    else:
        rv=resources[label]; regs=max((x[0] for x in rv),default=-1); ss=max((x[1] for x in rv),default=-1); sl=max((x[2] for x in rv),default=-1); clean=len(rv)>=2 and ss==0 and sl==0
    agg.append((wall,label,t,high,regs,ss,sl,clean))
def rank(x): return (x[0],x[3] if not math.isnan(x[3]) else math.inf,x[4] if x[4]>=0 else math.inf,x[2])
for x in sorted(agg,key=rank): print(f'STAGE_O label={x[1]} threads={x[2]} wall_s={x[0]:.9f} high_s={x[3]:.9f} regs={x[4]} spill_store={x[5]} spill_load={x[6]} spill_free={int(x[7])}',file=sys.stderr)
base=min((x for x in agg if x[1]=='control'),key=rank); tests=[x for x in agg if x[1]!='control' and x[7]]
if not tests: raise SystemExit('Stage-O needs a spill-free alternate fetch candidate')
best=min(tests,key=rank); meta=bins[best[1]]; speed=base[0]/best[0]; enabled=int(rank(best)<rank(base)); q=lambda x:shlex.quote(str(x))
vals={
 'B300_STAGEO_ROWS':rows_arg,'B300_STAGEO_NGPU':int(ngpu),'B300_STAGEO_RESIDUE':next(iter(res)),'B300_STAGEO_BASE_FETCH_BYTES':int(base_fetch),'B300_STAGEO_FETCH_BYTES':int(meta['fetch_bytes']),
 'B300_STAGEO_PAIR_POLICY':pair,'B300_STAGEO_BLOCK_POLICY':block,'B300_STAGEO_MATE_LOAD_POLICY':mp,
 'B300_STAGEO_SELF_WIDTH':int(sw),'B300_STAGEO_SELF_DISTANCE':int(sd),'B300_STAGEO_SELF_EVICT':se,'B300_STAGEO_SELF_GUARD':sg,'B300_STAGEO_MATE_WIDTH':int(mw),'B300_STAGEO_MATE_DISTANCE':int(md),'B300_STAGEO_MATE_EVICT':me,'B300_STAGEO_MATE_GUARD':mg,
 'B300_STAGEO_CONTROL_BIN':control_bin,'B300_STAGEO_CONTROL_THREADS':base[2],'B300_STAGEO_CONTROL_WALL_S':f'{base[0]:.9f}','B300_STAGEO_CONTROL_HIGH_S':f'{base[3]:.9f}','B300_STAGEO_CONTROL_SPILL_FREE':1,
 'B300_STAGEO_BIN':meta['binary'],'B300_STAGEO_THREADS':best[2],'B300_STAGEO_WALL_S':f'{best[0]:.9f}','B300_STAGEO_HIGH_S':f'{best[3]:.9f}','B300_STAGEO_SPILL_FREE':1,'B300_STAGEO_SPEEDUP':f'{speed:.9f}','B300_STAGEO_BEST_ENABLED':enabled,
}
with open(winner,'w') as f:
    for k,v in vals.items(): f.write(f'{k}={q(v)}\n')
print(f'b300_stageo_exact_match=1 b300_stageo_ngpu={ngpu} rows={rows_arg} residue={next(iter(res))} base_fetch={base_fetch} best_fetch={meta["fetch_bytes"]} speedup={speed:.9f} enabled={enabled} spill_free=1')
PY
cat "$WINNER_ENV"
