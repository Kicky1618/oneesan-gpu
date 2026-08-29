#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; ROWS="${ROWS:-1}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"; BASE_RECURRENCE_ILP="${BASE_RECURRENCE_ILP:-2}"; RANDOM_CG="${RANDOM_CG:-0}"; RANDOM_CG_L2_FETCH_BYTES="${RANDOM_CG_L2_FETCH_BYTES:-0}"; PREFETCH_L2="${PREFETCH_L2:-0}"; DUALMASK="${DUALMASK:-0}"; CLOSURE_BATCH="${CLOSURE_BATCH:-0}"; MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
THRESHOLDS="${ILP8_THRESHOLDS:-0 262144 524288 1048576 2097152 4194304 8388608}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"; REPEATS="${REPEATS:-1}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_h${HIGH_DROP_CHUNK}_basei${BASE_RECURRENCE_ILP}_cg${RANDOM_CG}_l2${RANDOM_CG_L2_FETCH_BYTES}_pre${PREFETCH_L2}_d${DUALMASK}_b${CLOSURE_BATCH}_r${MAXRREGCOUNT}_row${ROWS}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")"

[[ "$HIGH_DROP_CHUNK" == 0 || "$HIGH_DROP_CHUNK" == 1 ]] || exit 2
case "$BASE_RECURRENCE_ILP" in 2|4|8) ;; *) echo 'BASE_RECURRENCE_ILP must be 2,4,8' >&2; exit 2;; esac
for x in RANDOM_CG PREFETCH_L2 DUALMASK; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
case "$RANDOM_CG_L2_FETCH_BYTES" in 0|64|128|256) ;; *) exit 2;; esac
(( RANDOM_CG_L2_FETCH_BYTES == 0 )) || [[ "$RANDOM_CG" == 1 ]] || { echo 'RANDOM_CG_L2_FETCH_BYTES>0 requires RANDOM_CG=1' >&2; exit 2; }
case "$CLOSURE_BATCH" in 0|2|4) ;; *) exit 2;; esac
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] && ((MAXRREGCOUNT==0 || (MAXRREGCOUNT>=32 && MAXRREGCOUNT<=255))) || exit 2
[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || exit 2
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || exit 2
python3 - "$SAMPLE_INTERVAL" <<'PY'
import sys
if float(sys.argv[1]) <= 0: raise SystemExit('SAMPLE_INTERVAL must be >0')
PY
read -r -a threshold_list <<<"$THRESHOLDS"
(( ${#threshold_list[@]} > 0 )) || { echo 'ILP8_THRESHOLDS is empty' >&2; exit 2; }
declare -A seen_threshold=()
for z in "${threshold_list[@]}"; do
  [[ "$z" =~ ^[0-9]+$ ]] || { echo "bad threshold=$z" >&2; exit 2; }
  [[ -z "${seen_threshold[$z]+x}" ]] || { echo "duplicate threshold=$z" >&2; exit 2; }
  seen_threshold[$z]=1
done
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

printf 'profile\tmode\tthreshold\trecurrence_ilp\tbinary\tbuild_err\n' >"$LOGDIR/binaries.tsv"
build_one(){
  local profile="$1" mode="$2" threshold="$3" ilp="$4"
  local hybrid=0; [[ "$mode" == hybrid ]] && hybrid=1
  local bin="$ONEESAN_BUILD_DIR/b300_nextgen_${profile}_h${HIGH_DROP_CHUNK}_i${ilp}_cg${RANDOM_CG}_l2${RANDOM_CG_L2_FETCH_BYTES}_pre${PREFETCH_L2}_d${DUALMASK}_b${CLOSURE_BATCH}_r${MAXRREGCOUNT}_n27"
  local err="$LOGDIR/${profile}.build.err" out="$LOGDIR/${profile}.build.out"
  N=27 ARCH="$ARCH" OUT="$bin" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" RECURRENCE_ILP="$ilp" RECURRENCE_HYBRID_ILP8="$hybrid" RECURRENCE_HYBRID_ILP8_MIN_STATES="$threshold" RANDOM_CG="$RANDOM_CG" RANDOM_CG_L2_FETCH_BYTES="$RANDOM_CG_L2_FETCH_BYTES" PREFETCH_L2="$PREFETCH_L2" DUALMASK="$DUALMASK" CLOSURE_BATCH="$CLOSURE_BATCH" MAXRREGCOUNT="$MAXRREGCOUNT" BUILD_ERR="$err" PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen.sh" >"$out" 2>"$LOGDIR/${profile}.build.driver.err"
  [[ -x "$bin" ]] || { echo "$profile binary missing" >&2; exit 3; }
  grep -Fq "recurrence_ilp=$ilp recurrence_hybrid_ilp8=$hybrid recurrence_hybrid_ilp8_min_states=$threshold" "$out" || { echo "$profile hybrid build summary mismatch" >&2; exit 3; }
  grep -Fq "random_cg=$RANDOM_CG" "$out"; grep -Fq "random_cg_l2_fetch_bytes=$RANDOM_CG_L2_FETCH_BYTES" "$out"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$profile" "$mode" "$threshold" "$ilp" "$bin" "$err" >>"$LOGDIR/binaries.tsv"
}

build_one baseline baseline 0 "$BASE_RECURRENCE_ILP"
for z in "${threshold_list[@]}"; do build_one "hybrid_t${z}" hybrid "$z" 2; done

printf 'profile\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
while IFS=$'\t' read -r profile mode threshold ilp bin err; do
  [[ "$profile" == profile ]] && continue
  python3 "$PARSER" "$err" --label "$profile" --contains main_pull_kernel_ilp2 >>"$RESOURCE" || true
  if [[ "$mode" == hybrid ]]; then python3 "$PARSER" "$err" --label "$profile" --contains main_pull_kernel_ilp8_hybrid >>"$RESOURCE" || true; fi
done <"$LOGDIR/binaries.tsv"

sample_mem(){
  local pid="$1" out="$2"; : >"$out"
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1;n++;if($1>m)m=$1}END{if(n)printf "%.6f %.6f\n",s/n,m}' >>"$out" || true
    sleep "$SAMPLE_INTERVAL"
  done
}
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
printf 'profile\tmode\tthreshold\trecurrence_ilp\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\thigh_rec_groups\thigh_rec_fallback_groups\tmc_avg_pct\tmc_max_pct\tmc_samples\n' >"$RESULT"
run_one(){
  local profile="$1" mode="$2" threshold="$3" ilp="$4" bin="$5" t="$6" r="$7"
  local so="$LOGDIR/${profile}_t${t}_r${r}.out" se="$LOGDIR/${profile}_t${t}_r${r}.err" mem="$LOGDIR/${profile}_t${t}_r${r}.mem"
  set +e
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$t" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se" & local pid=$!
  sample_mem "$pid" "$mem" & local sp=$!
  wait "$pid"; local rc=$?
  set -e
  wait "$sp" 2>/dev/null || true
  ((rc==0)) || { echo "$profile t=$t failed rc=$rc" >&2; tail -n 100 "$se" >&2 || true; return "$rc"; }
  local line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$profile missing backend line" >&2; return 4; }
  local hg="$(field high_rec_groups "$line")" hf="$(field high_rec_fallback_groups "$line")"; [[ "$hg" =~ ^[0-9]+$ ]] && ((hg>0)) || return 5
  local ma mm mn; read -r ma mm mn < <(awk '{sa+=$1;if($2>mm)mm=$2;n++}END{if(n)printf "%.3f %.3f %d\n",sa/n,mm,n;else print "nan nan 0"}' "$mem")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$profile" "$mode" "$threshold" "$ilp" "$t" "$r" "$(field residue "$line")" "$(field wall_s "$line")" "$(field active_max_s "$line")" "$(field active_sum_s "$line")" "$hg" "$hf" "$ma" "$mm" "$mn" >>"$RESULT"
}

for t in $THREADS_LIST; do
  [[ "$t" =~ ^[0-9]+$ ]] && ((t>=32&&t<=1024&&t%32==0)) || exit 2
  while IFS=$'\t' read -r profile mode threshold ilp bin err; do
    [[ "$profile" == profile ]] && continue
    for ((r=1;r<=REPEATS;++r)); do
      echo "=== hybrid-ILP8 profile=$profile mode=$mode threshold=$threshold ilp=$ilp threads=$t repeat=$r ===" >&2
      run_one "$profile" "$mode" "$threshold" "$ilp" "$bin" "$t" "$r"
    done
  done <"$LOGDIR/binaries.tsv"
done

python3 - "$RESULT" "$RESOURCE" "$LOGDIR/binaries.tsv" "$WINNER_ENV" <<'PY'
import csv,math,statistics,sys,shlex
result,resource,bins_path,winner=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t')); rr=list(csv.DictReader(open(resource),delimiter='\t'))
if not rows: raise SystemExit('no hybrid ILP8 results')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL hybrid ILP8 partial residue mismatch '+repr({(r['profile'],r['threads'],r['repeat']):r['residue'] for r in rows}))
bins={r['profile']:r for r in csv.DictReader(open(bins_path),delimiter='\t')}
resource_rows={}
for r in rr:
    try: z=(int(r['registers']),int(r['spill_store_bytes']),int(r['spill_load_bytes']))
    except (ValueError,TypeError,KeyError): continue
    resource_rows.setdefault(r['profile'],[]).append(z)
by={}
for r in rows: by.setdefault((r['profile'],r['mode'],int(r['threshold']),int(r['recurrence_ilp']),int(r['threads'])),[]).append(r)
med=[]
for (p,mode,thr,ilp,t),rs in by.items():
    wall=statistics.median(float(r['wall_s']) for r in rs)
    mv=[float(r['mc_avg_pct']) for r in rs if r['mc_avg_pct']!='nan']; mc=statistics.median(mv) if mv else math.nan
    rv=resource_rows.get(p,[]); regs=max((x[0] for x in rv),default=-1); ss=max((x[1] for x in rv),default=-1); sl=max((x[2] for x in rv),default=-1)
    # Hybrid candidates must expose resources for both ILP2 and ILP8 kernels.
    resource_ok=(len(rv)>=2 if mode=='hybrid' else len(rv)>=1)
    spill_free=resource_ok and ss==0 and sl==0
    med.append((wall,p,mode,thr,ilp,t,mc,regs,ss,sl,spill_free,len(rv)))
base=min((x for x in med if x[2]=='baseline' and x[10]),default=None,key=lambda x:x[0])
if base is None: raise SystemExit('baseline lacks known spill-free recurrence ptxas')
clean=[x for x in med if x[10]]
if not clean: raise SystemExit('no spill-free hybrid ILP8 candidate')
key=lambda x:(x[0],-x[6] if not math.isnan(x[6]) else math.inf)
best=min(clean,key=key)
for x in sorted(med,key=key):
    print('HYBRID8',f'profile={x[1]}',f'mode={x[2]}',f'threshold={x[3]}',f'ilp={x[4]}',f'threads={x[5]}',f'wall_s={x[0]:.9f}',f'speedup_vs_baseline={base[0]/x[0]:.6f}x',f'mc_avg_pct={x[6]:.3f}',f'regs_max={x[7]}',f'spill_store={x[8]}',f'spill_load={x[9]}',f'spill_free={int(x[10])}',f'resource_rows={x[11]}',file=sys.stderr)
b=bins[best[1]]
def q(x): return shlex.quote(str(x))
with open(winner,'w') as f:
    for k,v in (
        ('B300_HYBRID8_WINNER_PROFILE',best[1]),('B300_HYBRID8_WINNER_BIN',b['binary']),('B300_HYBRID8_WINNER_MODE',best[2]),
        ('B300_HYBRID8_WINNER_THRESHOLD',best[3]),('B300_HYBRID8_WINNER_RECURRENCE_ILP',best[4]),('B300_HYBRID8_WINNER_THREADS',best[5]),
        ('B300_HYBRID8_WINNER_WALL_S',f'{best[0]:.9f}'),('B300_HYBRID8_WINNER_SPEEDUP_VS_BASELINE',f'{base[0]/best[0]:.9f}'),
        ('B300_HYBRID8_WINNER_MC_AVG_PCT',f'{best[6]:.3f}'),('B300_HYBRID8_WINNER_REGISTERS_MAX',best[7]),
        ('B300_HYBRID8_WINNER_SPILL_STORE_BYTES',best[8]),('B300_HYBRID8_WINNER_SPILL_LOAD_BYTES',best[9]),('B300_HYBRID8_RESIDUE',next(iter(res)))):
        f.write(k+'='+q(v)+'\n')
print('b300_nextgen_hybrid8_exact_intermediate_match=1')
print('b300_nextgen_hybrid8_residue='+next(iter(res)))
print(f'b300_nextgen_hybrid8_baseline_profile={base[1]}')
print(f'b300_nextgen_hybrid8_baseline_wall_s={base[0]:.9f}')
print(f'b300_nextgen_hybrid8_best_profile={best[1]}')
print(f'b300_nextgen_hybrid8_best_mode={best[2]}')
print(f'b300_nextgen_hybrid8_best_threshold={best[3]}')
print(f'b300_nextgen_hybrid8_best_recurrence_ilp={best[4]}')
print(f'b300_nextgen_hybrid8_best_threads={best[5]}')
print(f'b300_nextgen_hybrid8_best_wall_s={best[0]:.9f}')
print(f'b300_nextgen_hybrid8_speedup_vs_baseline={base[0]/best[0]:.9f}x')
print(f'b300_nextgen_hybrid8_best_mc_avg_pct={best[6]:.3f}')
print('b300_nextgen_hybrid8_winner_env='+winner)
PY
cat "$RESOURCE"
echo "b300-nextgen-hybrid-ilp8-sweep OK result=$RESULT resources=$RESOURCE winner_env=$WINNER_ENV" >&2
