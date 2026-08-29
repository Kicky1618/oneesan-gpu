#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
MOD="${MOD:-4294967291}"
ROWS="${ROWS:-1}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
BASE_RECURRENCE_ILP="${BASE_RECURRENCE_ILP:-2}"
RANDOM_CG="${RANDOM_CG:-0}"
RANDOM_CG_L2_FETCH_BYTES="${RANDOM_CG_L2_FETCH_BYTES:-0}"
PREFETCH_L2="${PREFETCH_L2:-0}"
DUALMASK="${DUALMASK:-0}"
CLOSURE_BATCH="${CLOSURE_BATCH:-0}"
MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
THRESHOLDS="${HYBRID_ILP8_THRESHOLDS:-0 262144 524288 1048576 2097152 4194304 8388608}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
REPEATS="${REPEATS:-1}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_h${HIGH_DROP_CHUNK}_basei${BASE_RECURRENCE_ILP}_row${ROWS}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESULT="${RESULT:-${PREFIX}.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")" "$(dirname "$WINNER_ENV")"

[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$HIGH_DROP_CHUNK" == 0 || "$HIGH_DROP_CHUNK" == 1 ]] || exit 2
case "$BASE_RECURRENCE_ILP" in 2|4|8) ;; *) echo 'BASE_RECURRENCE_ILP must be 2,4,8' >&2; exit 2;; esac
for x in RANDOM_CG PREFETCH_L2 DUALMASK; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
case "$RANDOM_CG_L2_FETCH_BYTES" in 0|64|128|256) ;; *) exit 2;; esac
[[ "$RANDOM_CG" == 1 || "$RANDOM_CG_L2_FETCH_BYTES" == 0 ]] || { echo 'CG L2 fetch hint requires RANDOM_CG=1' >&2; exit 2; }
case "$CLOSURE_BATCH" in 0|2|4) ;; *) exit 2;; esac
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] && ((MAXRREGCOUNT==0 || (MAXRREGCOUNT>=32 && MAXRREGCOUNT<=255))) || exit 2
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || exit 2
python3 - "$SAMPLE_INTERVAL" <<'PY'
import sys
if float(sys.argv[1])<=0: raise SystemExit('SAMPLE_INTERVAL must be >0')
PY
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

read -r -a threshold_list <<<"$THRESHOLDS"
(( ${#threshold_list[@]} > 0 )) || { echo 'HYBRID_ILP8_THRESHOLDS is empty' >&2; exit 2; }
declare -A seen_threshold=()
for threshold in "${threshold_list[@]}"; do
  [[ "$threshold" =~ ^[0-9]+$ ]] || { echo "bad threshold=$threshold" >&2; exit 2; }
  [[ -z "${seen_threshold[$threshold]+x}" ]] || { echo "duplicate threshold=$threshold" >&2; exit 2; }
  seen_threshold[$threshold]=1
done

printf 'profile\thybrid\tthreshold\trecurrence_ilp\tbinary\tbuild_err\n' >"$LOGDIR/binaries.tsv"
build_one(){
  local profile="$1" hybrid="$2" threshold="$3" ilp="$4"
  local bin="$ONEESAN_BUILD_DIR/b300_nextgen_hybrid8_${profile}_h${HIGH_DROP_CHUNK}_cg${RANDOM_CG}_l2${RANDOM_CG_L2_FETCH_BYTES}_pre${PREFETCH_L2}_d${DUALMASK}_b${CLOSURE_BATCH}_r${MAXRREGCOUNT}_n27"
  local err="$LOGDIR/${profile}.build.err" out="$LOGDIR/${profile}.build.out"
  N=27 ARCH="$ARCH" OUT="$bin" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" RECURRENCE_ILP="$ilp" \
    RECURRENCE_HYBRID_ILP8="$hybrid" RECURRENCE_HYBRID_ILP8_MIN_STATES="$threshold" \
    RANDOM_CG="$RANDOM_CG" RANDOM_CG_L2_FETCH_BYTES="$RANDOM_CG_L2_FETCH_BYTES" PREFETCH_L2="$PREFETCH_L2" \
    DUALMASK="$DUALMASK" CLOSURE_BATCH="$CLOSURE_BATCH" MAXRREGCOUNT="$MAXRREGCOUNT" PTXAS_VERBOSE=1 BUILD_ERR="$err" \
    bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen.sh" >"$out" 2>"$LOGDIR/${profile}.build.driver.err"
  [[ -x "$bin" ]] || { echo "$profile binary missing" >&2; exit 3; }
  grep -Fq "recurrence_hybrid_ilp8=$hybrid" "$out" || { echo "$profile hybrid build marker mismatch" >&2; exit 3; }
  if [[ "$hybrid" == 1 ]]; then
    grep -Fq "recurrence_hybrid_ilp8_min_states=$threshold" "$out" || exit 3
    grep -Fq 'main_pull_kernel_ilp8_hybrid' "$out" "$LOGDIR/${profile}.build.driver.err" "$err" 2>/dev/null || true
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$profile" "$hybrid" "$threshold" "$ilp" "$bin" "$err" >>"$LOGDIR/binaries.tsv"
}

build_one base 0 0 "$BASE_RECURRENCE_ILP"
for threshold in "${threshold_list[@]}"; do build_one "hybrid_t${threshold}" 1 "$threshold" 2; done

printf 'profile\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
while IFS=$'\t' read -r profile hybrid threshold ilp bin err; do
  [[ "$profile" == profile ]] && continue
  python3 "$PARSER" "$err" --label "$profile" --contains main_pull_kernel_ilp2 >>"$RESOURCE" || true
  if [[ "$hybrid" == 1 ]]; then
    python3 "$PARSER" "$err" --label "$profile" --contains main_pull_kernel_ilp8_hybrid >>"$RESOURCE" || true
  fi
done <"$LOGDIR/binaries.tsv"

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample_mem(){
  local pid="$1" out="$2"; : >"$out"
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits 2>/dev/null |
      awk '{s+=$1;n++}END{if(n)printf "%.6f\n",s/n}' >>"$out" || true
    sleep "$SAMPLE_INTERVAL"
  done
}

printf 'profile\thybrid\tthreshold\trecurrence_ilp\tthreads\trepeat\tresidue\twall_s\thigh_rec_groups\thigh_rec_fallback_groups\tmc_avg_pct\tmc_max_pct\tmc_samples\n' >"$RESULT"
run_one(){
  local profile="$1" hybrid="$2" threshold="$3" ilp="$4" bin="$5" threads="$6" repeat="$7"
  local so="$LOGDIR/${profile}_t${threads}_r${repeat}.out" se="$LOGDIR/${profile}_t${threads}_r${repeat}.err" mem="$LOGDIR/${profile}_t${threads}_r${repeat}.mem"
  set +e
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$threads" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se" &
  local pid=$!; sample_mem "$pid" "$mem" & local mpid=$!
  wait "$pid"; local rc=$?
  set -e
  wait "$mpid" 2>/dev/null || true
  ((rc==0)) || { echo "$profile threads=$threads repeat=$repeat failed rc=$rc" >&2; tail -n 80 "$se" >&2 || true; return "$rc"; }
  local line hg hf ma mm mn
  line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || return 4
  hg="$(field high_rec_groups "$line")"; hf="$(field high_rec_fallback_groups "$line")"
  [[ "$hg" =~ ^[0-9]+$ ]] && ((hg>0)) || return 5
  read -r ma mm mn < <(awk '{s+=$1;n++;if($1>m)m=$1}END{if(n)printf "%.3f %.3f %d\n",s/n,m,n;else print "nan nan 0"}' "$mem")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$profile" "$hybrid" "$threshold" "$ilp" "$threads" "$repeat" "$(field residue "$line")" "$(field wall_s "$line")" "$hg" "$hf" "$ma" "$mm" "$mn" >>"$RESULT"
}

for threads in $THREADS_LIST; do
  [[ "$threads" =~ ^[0-9]+$ ]] && ((threads>=32 && threads<=1024 && threads%32==0)) || { echo "bad threads=$threads" >&2; exit 2; }
  while IFS=$'\t' read -r profile hybrid threshold ilp bin err; do
    [[ "$profile" == profile ]] && continue
    for ((repeat=1; repeat<=REPEATS; ++repeat)); do
      echo "=== hybrid8 $profile threads=$threads repeat=$repeat ===" >&2
      run_one "$profile" "$hybrid" "$threshold" "$ilp" "$bin" "$threads" "$repeat"
    done
  done <"$LOGDIR/binaries.tsv"
done

python3 - "$RESULT" "$RESOURCE" "$LOGDIR/binaries.tsv" "$WINNER_ENV" <<'PY'
import csv,math,shlex,statistics,sys
result,resource,bins_path,winner=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t'))
rr=list(csv.DictReader(open(resource),delimiter='\t'))
bins={r['profile']:r for r in csv.DictReader(open(bins_path),delimiter='\t')}
if not rows: raise SystemExit('no hybrid ILP8 results')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL hybrid ILP8 partial residue mismatch '+repr({(r['profile'],r['threads']):r['residue'] for r in rows}))
resource_by={}
for r in rr:
    try: cur=(int(r['registers']),int(r['spill_store_bytes']),int(r['spill_load_bytes']))
    except (ValueError,KeyError): continue
    old=resource_by.get(r['profile'],(-1,-1,-1))
    resource_by[r['profile']]=tuple(max(a,b) for a,b in zip(old,cur))
by={}
for r in rows:
    key=(r['profile'],int(r['hybrid']),int(r['threshold']),int(r['recurrence_ilp']),int(r['threads']))
    by.setdefault(key,[]).append(r)
agg=[]
for (p,h,threshold,ilp,t),rs in by.items():
    wall=statistics.median(float(r['wall_s']) for r in rs)
    mcvals=[float(r['mc_avg_pct']) for r in rs if r['mc_avg_pct']!='nan']
    mc=statistics.median(mcvals) if mcvals else math.nan
    samples=sum(int(r['mc_samples']) for r in rs)
    regs,ss,sl=resource_by.get(p,(-1,-1,-1))
    agg.append((wall,p,h,threshold,ilp,t,mc,samples,regs,ss,sl))
def key(x): return (x[0],-x[6] if not math.isnan(x[6]) else math.inf)
for x in sorted(agg,key=key):
    print(f'HYBRID8 profile={x[1]} hybrid={x[2]} threshold={x[3]} ilp={x[4]} threads={x[5]} wall_s={x[0]:.9f} mc={x[6]:.3f} mc_samples={x[7]} regs={x[8]} spill_store={x[9]} spill_load={x[10]}',file=sys.stderr)
base_pool=[x for x in agg if x[2]==0]
if not base_pool: raise SystemExit('missing nextgen base candidate')
base=min(base_pool,key=key)
clean=[x for x in agg if x[8]>=0 and x[9]==0 and x[10]==0]
if not clean: raise SystemExit('no candidate has known spill-free recurrence ptxas')
best=min(clean,key=key)
b=bins[best[1]]
def q(v): return shlex.quote(str(v))
vals={
 'B300_HYBRID8_WINNER_PROFILE':best[1],
 'B300_HYBRID8_WINNER_BIN':b['binary'],
 'B300_HYBRID8_WINNER_THREADS':best[5],
 'B300_HYBRID8_WINNER_ENABLED':best[2],
 'B300_HYBRID8_WINNER_THRESHOLD':best[3],
 'B300_HYBRID8_WINNER_RECURRENCE_ILP':best[4],
 'B300_HYBRID8_WINNER_WALL_S':f'{best[0]:.9f}',
 'B300_HYBRID8_WINNER_MC_AVG_PCT':f'{best[6]:.3f}',
 'B300_HYBRID8_WINNER_MC_SAMPLES':best[7],
 'B300_HYBRID8_WINNER_REGISTERS':best[8],
 'B300_HYBRID8_WINNER_SPILL_STORE_BYTES':best[9],
 'B300_HYBRID8_WINNER_SPILL_LOAD_BYTES':best[10],
 'B300_HYBRID8_BASE_PROFILE':base[1],
 'B300_HYBRID8_BASE_BIN':bins[base[1]]['binary'],
 'B300_HYBRID8_BASE_THREADS':base[5],
 'B300_HYBRID8_BASE_WALL_S':f'{base[0]:.9f}',
 'B300_HYBRID8_SPEEDUP_VS_BASE':f'{base[0]/best[0]:.9f}',
 'B300_HYBRID8_RESIDUE':next(iter(res)),
}
with open(winner,'w') as f:
    for k,v in vals.items(): f.write(f'{k}={q(v)}\n')
print('b300_nextgen_hybrid8_exact_intermediate_match=1')
print(f'b300_nextgen_hybrid8_residue={next(iter(res))}')
print(f'b300_nextgen_hybrid8_base_profile={base[1]}')
print(f'b300_nextgen_hybrid8_base_threads={base[5]}')
print(f'b300_nextgen_hybrid8_base_wall_s={base[0]:.9f}')
print(f'b300_nextgen_hybrid8_best_profile={best[1]}')
print(f'b300_nextgen_hybrid8_best_enabled={best[2]}')
print(f'b300_nextgen_hybrid8_best_threshold={best[3]}')
print(f'b300_nextgen_hybrid8_best_threads={best[5]}')
print(f'b300_nextgen_hybrid8_best_wall_s={best[0]:.9f}')
print(f'b300_nextgen_hybrid8_best_mc_avg_pct={best[6]:.3f}')
print(f'b300_nextgen_hybrid8_best_spill_store_bytes={best[9]}')
print(f'b300_nextgen_hybrid8_best_spill_load_bytes={best[10]}')
print(f'b300_nextgen_hybrid8_speedup_vs_base={base[0]/best[0]:.9f}x')
print(f'b300_nextgen_hybrid8_winner_env={winner}')
PY

cat "$RESULT"
cat "$RESOURCE"
echo "b300-nextgen-hybrid-ilp8-threshold-sweep OK result=$RESULT resources=$RESOURCE winner_env=$WINNER_ENV sample_interval=$SAMPLE_INTERVAL" >&2
