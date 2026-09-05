#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; ROWS="${ROWS:-1}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"; RECURRENCE_ILP="${RECURRENCE_ILP:-4}"; PREFETCH_L2="${PREFETCH_L2:-0}"; DUALMASK="${DUALMASK:-0}"; CLOSURE_BATCH="${CLOSURE_BATCH:-0}"; MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
L2_SIZES="${L2_SIZES:-0 64 128 256}"; INCLUDE_NOCG="${INCLUDE_NOCG:-1}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"; REPEATS="${REPEATS:-1}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_cgl2_h${HIGH_DROP_CHUNK}_i${RECURRENCE_ILP}_pre${PREFETCH_L2}_d${DUALMASK}_b${CLOSURE_BATCH}_r${MAXRREGCOUNT}_row${ROWS}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")"

[[ "$HIGH_DROP_CHUNK" == 0 || "$HIGH_DROP_CHUNK" == 1 ]] || exit 2
case "$RECURRENCE_ILP" in 2|4|8) ;; *) echo 'RECURRENCE_ILP must be 2,4,8' >&2; exit 2;; esac
for x in PREFETCH_L2 DUALMASK INCLUDE_NOCG; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
case "$CLOSURE_BATCH" in 0|2|4) ;; *) exit 2;; esac
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] && ((MAXRREGCOUNT==0 || (MAXRREGCOUNT>=32 && MAXRREGCOUNT<=255))) || exit 2
[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || exit 2
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || exit 2
python3 - "$SAMPLE_INTERVAL" <<'PY'
import sys
if float(sys.argv[1]) <= 0: raise SystemExit('SAMPLE_INTERVAL must be >0')
PY
for z in $L2_SIZES; do case "$z" in 0|64|128|256) ;; *) echo "bad L2 size=$z" >&2; exit 2;; esac; done
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }; command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

printf 'profile\trandom_cg\tl2_fetch_bytes\tbinary\tbuild_err\n' >"$LOGDIR/binaries.tsv"
build_one(){
  local profile="$1" cg="$2" l2="$3" bin="$ONEESAN_BUILD_DIR/b300_nextgen_${profile}_h${HIGH_DROP_CHUNK}_i${RECURRENCE_ILP}_pre${PREFETCH_L2}_d${DUALMASK}_b${CLOSURE_BATCH}_r${MAXRREGCOUNT}_n27" err="$LOGDIR/${profile}.build.err" out="$LOGDIR/${profile}.build.out"
  N=27 ARCH="$ARCH" OUT="$bin" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" RECURRENCE_ILP="$RECURRENCE_ILP" RANDOM_CG="$cg" RANDOM_CG_L2_FETCH_BYTES="$l2" PREFETCH_L2="$PREFETCH_L2" DUALMASK="$DUALMASK" CLOSURE_BATCH="$CLOSURE_BATCH" MAXRREGCOUNT="$MAXRREGCOUNT" BUILD_ERR="$err" PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen.sh" >"$out" 2>"$LOGDIR/${profile}.build.driver.err"
  [[ -x "$bin" ]] || { echo "$profile binary missing" >&2; exit 3; }
  grep -Fq "random_cg=$cg random_cg_l2_fetch_bytes=$l2 prefetch_l2=$PREFETCH_L2" "$out" || { echo "$profile build summary mismatch" >&2; exit 3; }
  printf '%s\t%s\t%s\t%s\t%s\n' "$profile" "$cg" "$l2" "$bin" "$err" >>"$LOGDIR/binaries.tsv"
}
if [[ "$INCLUDE_NOCG" == 1 ]]; then build_one nocg 0 0; fi
declare -A seen=()
for z in $L2_SIZES; do [[ -z "${seen[$z]+x}" ]] || continue; seen[$z]=1; build_one "cg${z}" 1 "$z"; done

printf 'profile\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
while IFS=$'\t' read -r profile cg l2 bin err; do
  [[ "$profile" == profile ]] && continue
  python3 "$PARSER" "$err" --label "$profile" --contains main_pull_kernel_ilp2 >>"$RESOURCE" || true
done <"$LOGDIR/binaries.tsv"

sample_mem(){
  local pid="$1" out="$2"; : >"$out"
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1;n++}END{if(n)printf "%.6f\n",s/n}' >>"$out" || true
    sleep "$SAMPLE_INTERVAL"
  done
}
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
printf 'profile\trandom_cg\tl2_fetch_bytes\tthreads\trepeat\tresidue\twall_s\thigh_rec_groups\thigh_rec_fallback_groups\tmc_avg_pct\tmc_max_pct\tmc_samples\n' >"$RESULT"
run_one(){
  local profile="$1" cg="$2" l2="$3" bin="$4" t="$5" r="$6" so="$LOGDIR/${profile}_t${t}_r${r}.out" se="$LOGDIR/${profile}_t${t}_r${r}.err" mem="$LOGDIR/${profile}_t${t}_r${r}.mem"
  set +e
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$t" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se" & local pid=$!
  sample_mem "$pid" "$mem" & local sp=$!
  wait "$pid"; local rc=$?
  set -e
  wait "$sp" 2>/dev/null || true
  ((rc==0)) || { echo "$profile t=$t failed rc=$rc" >&2; tail -n 100 "$se" >&2 || true; return "$rc"; }
  local line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || return 4
  local hg="$(field high_rec_groups "$line")" hf="$(field high_rec_fallback_groups "$line")"; [[ "$hg" =~ ^[0-9]+$ ]] && ((hg>0)) || return 5
  local ma mm mn; read -r ma mm mn < <(awk '{s+=$1;n++;if($1>m)m=$1}END{if(n)printf "%.3f %.3f %d\n",s/n,m,n;else print "nan nan 0"}' "$mem")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$profile" "$cg" "$l2" "$t" "$r" "$(field residue "$line")" "$(field wall_s "$line")" "$hg" "$hf" "$ma" "$mm" "$mn" >>"$RESULT"
}
for t in $THREADS_LIST; do
  [[ "$t" =~ ^[0-9]+$ ]] && ((t>=32&&t<=1024&&t%32==0)) || exit 2
  while IFS=$'\t' read -r profile cg l2 bin err; do
    [[ "$profile" == profile ]] && continue
    for ((r=1;r<=REPEATS;++r)); do echo "=== CG-L2 $profile threads=$t repeat=$r ===" >&2; run_one "$profile" "$cg" "$l2" "$bin" "$t" "$r"; done
  done <"$LOGDIR/binaries.tsv"
done

python3 - "$RESULT" "$RESOURCE" "$LOGDIR/binaries.tsv" "$WINNER_ENV" "$HIGH_DROP_CHUNK" "$RECURRENCE_ILP" "$PREFETCH_L2" "$DUALMASK" "$CLOSURE_BATCH" "$MAXRREGCOUNT" <<'PY'
import csv,math,statistics,sys,shlex
result,resource,bins_path,winner,high,ilp,prefetch,dual,batch,cap=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t')); rr=list(csv.DictReader(open(resource),delimiter='\t'))
if not rows: raise SystemExit('no CG L2 results')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL CG L2 partial residue mismatch '+repr({(r['profile'],r['threads']):r['residue'] for r in rows}))
bins={r['profile']:r for r in csv.DictReader(open(bins_path),delimiter='\t')}
resources={}
for r in rr:
    try: z=(int(r['registers']),int(r['spill_store_bytes']),int(r['spill_load_bytes']))
    except (ValueError,TypeError,KeyError): continue
    old=resources.get(r['profile'],(-1,-1,-1))
    resources[r['profile']] = (max(old[0],z[0]),max(old[1],z[1]),max(old[2],z[2]))
by={}
for r in rows: by.setdefault((r['profile'],int(r['random_cg']),int(r['l2_fetch_bytes']),int(r['threads'])),[]).append(r)
med=[]
for (p,cg,l2,t),rs in by.items():
    wall=statistics.median(float(r['wall_s']) for r in rs)
    mv=[float(r['mc_avg_pct']) for r in rs if r['mc_avg_pct']!='nan']; mc=statistics.median(mv) if mv else math.nan
    regs,ss,sl=resources.get(p,(-1,-1,-1)); med.append((wall,p,cg,l2,t,mc,regs,ss,sl))
for x in sorted(med): print(f'CG_L2 profile={x[1]} cg={x[2]} l2={x[3]} threads={x[4]} wall_s={x[0]:.9f} mc={x[5]:.3f} regs={x[6]} spill_store={x[7]} spill_load={x[8]}',file=sys.stderr)
clean=[x for x in med if x[6]>=0 and x[7]==0 and x[8]==0]
if not clean: raise SystemExit('no CG L2 candidate has known spill-free main recurrence ptxas')
key=lambda x:(x[0],-x[5] if not math.isnan(x[5]) else math.inf)
best=min(clean,key=key); b=bins[best[1]]
def q(x): return shlex.quote(str(x))
with open(winner,'w') as f:
    for k,v in (
        ('B300_CGL2_WINNER_PROFILE',best[1]),('B300_CGL2_WINNER_BIN',b['binary']),('B300_CGL2_WINNER_THREADS',best[4]),
        ('B300_CGL2_WINNER_RANDOM_CG',best[2]),('B300_CGL2_WINNER_L2_FETCH_BYTES',best[3]),('B300_CGL2_HIGH_DROP_CHUNK',high),
        ('B300_CGL2_RECURRENCE_ILP',ilp),('B300_CGL2_PREFETCH_L2',prefetch),('B300_CGL2_DUALMASK',dual),('B300_CGL2_CLOSURE_BATCH',batch),
        ('B300_CGL2_MAXRREGCOUNT',cap),('B300_CGL2_WINNER_WALL_S',f'{best[0]:.9f}'),('B300_CGL2_WINNER_MC_AVG_PCT',f'{best[5]:.3f}'),
        ('B300_CGL2_WINNER_REGISTERS',best[6]),('B300_CGL2_WINNER_SPILL_STORE_BYTES',best[7]),('B300_CGL2_WINNER_SPILL_LOAD_BYTES',best[8])):
        f.write(k+'='+q(v)+'\n')
print('b300_nextgen_cgl2_exact_intermediate_match=1')
print('b300_nextgen_cgl2_residue='+next(iter(res)))
print(f'b300_nextgen_cgl2_best_profile={best[1]}')
print(f'b300_nextgen_cgl2_best_threads={best[4]}')
print(f'b300_nextgen_cgl2_best_l2_fetch_bytes={best[3]}')
print(f'b300_nextgen_cgl2_best_wall_s={best[0]:.9f}')
print(f'b300_nextgen_cgl2_best_mc_avg_pct={best[5]:.3f}')
print('b300_nextgen_cgl2_winner_env='+winner)
PY
cat "$RESOURCE"
echo "b300-nextgen-cg-l2size-sweep OK result=$RESULT resources=$RESOURCE winner_env=$WINNER_ENV" >&2
