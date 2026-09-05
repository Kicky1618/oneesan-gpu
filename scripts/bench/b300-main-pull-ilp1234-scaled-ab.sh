#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
MOD="${MOD:-4294967291}"
ROWS="${ROWS:-1}"
ARCH="${ARCH:-native}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
ILP_LIST="${ILP_LIST:-1 2 3 4}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
REPEATS="${REPEATS:-1}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_main_pull_ilp1234_scaled_row${ROWS}_hd${HIGH_DROP_CHUNK}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESULT="${RESULT:-${PREFIX}.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")"

[[ "$N" == 27 ]] || { echo 'scaled main-pull ILP A/B currently targets n=27' >&2; exit 2; }
[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1 && ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$REPEATS" =~ ^[0-9]+$ ]] && ((REPEATS>=1)) || { echo 'REPEATS must be >=1' >&2; exit 2; }
[[ "$HIGH_DROP_CHUNK" == 0 || "$HIGH_DROP_CHUNK" == 1 ]] || { echo 'HIGH_DROP_CHUNK must be 0/1' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

read -r -a ilps <<<"$ILP_LIST"
read -r -a threads_values <<<"$THREADS_LIST"
(( ${#ilps[@]} > 0 && ${#threads_values[@]} > 0 )) || exit 2
declare -A seen_ilp=()
for ilp in "${ilps[@]}"; do
  case "$ilp" in 1|2|3|4) ;; *) echo "ILP_LIST contains invalid value: $ilp" >&2; exit 2;; esac
  [[ -z "${seen_ilp[$ilp]+x}" ]] || { echo "duplicate ILP=$ilp" >&2; exit 2; }
  seen_ilp[$ilp]=1
done
for t in "${threads_values[@]}"; do
  [[ "$t" =~ ^[0-9]+$ ]] && ((t>=32 && t<=1024 && t%32==0)) || { echo "invalid threads=$t" >&2; exit 2; }
done
[[ -n "${seen_ilp[1]+x}" ]] || { echo 'ILP_LIST must include ILP=1 baseline' >&2; exit 2; }

# Production launch proofs are part of every ILP>1 build, but run the combined
# proof explicitly once so the benchmark log records the regression gate.
python3 "$ONEESAN_ROOT/scripts/bench/b300-main-pull-ilp-production-launch-proof.py" >"$LOGDIR/production-launch-proof.out"
grep -Fq 'b300-main-pull-ilp-production-launch-proof OK scaled_launch=1 exact=1' "$LOGDIR/production-launch-proof.out"

printf 'ilp\tbinary\tbuild_err\n' >"$LOGDIR/binaries.tsv"
for ilp in "${ilps[@]}"; do
  bin="$ONEESAN_BUILD_DIR/b300_main_pull_scaled_ilp${ilp}_hd${HIGH_DROP_CHUNK}_n27"
  bout="$LOGDIR/ilp${ilp}.build.out"
  berr="$LOGDIR/ilp${ilp}.build.err"
  echo "=== build scaled main-pull ILP$ilp ===" >&2
  N=27 ARCH="$ARCH" OUT="$bin" MAIN_PULL_ILP="$ilp" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" \
    RUNTIME_THREADS=1 PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh" >"$bout" 2>"$berr"
  [[ -x "$bin" ]] || { echo "missing ILP$ilp binary: $bin" >&2; exit 3; }
  if (( ilp > 1 )); then
    grep -Fq "launch_mlp_fixed=1" "$bout" || {
      echo "ILP$ilp build did not expose launch_mlp_fixed=1; refusing stale transform" >&2
      tail -n 120 "$bout" >&2 || true
      exit 3
    }
  fi
  printf '%s\t%s\t%s\n' "$ilp" "$bin" "$berr" >>"$LOGDIR/binaries.tsv"
done

PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
printf 'ilp\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
while IFS=$'\t' read -r ilp bin berr; do
  [[ "$ilp" == ilp ]] && continue
  python3 "$PARSER" "$berr" --label "ilp${ilp}" --contains main_pull_kernel >>"$RESOURCE" || true
done <"$LOGDIR/binaries.tsv"

field(){
  local k="$1" l="$2"
  sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1
}

printf 'ilp\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tmc_avg_pct\tmc_max_pct\n' >"$RESULT"
run_one(){
  local ilp="$1" bin="$2" t="$3" rep="$4"
  local tag="ilp${ilp}_t${t}_r${rep}" so="$LOGDIR/${tag}.out" se="$LOGDIR/${tag}.err" dm="$LOGDIR/${tag}.dmon"
  echo "=== scaled main-pull ILP$ilp threads=$t repeat=$rep ===" >&2
  : >"$dm"
  nvidia-smi dmon -s u -d 1 >"$dm" 2>&1 & local mpid=$!
  set +e
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$t" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se"
  local rc=$?
  set -e
  kill "$mpid" 2>/dev/null || true
  wait "$mpid" 2>/dev/null || true
  if (( rc != 0 )); then
    echo "ILP$ilp threads=$t failed rc=$rc" >&2
    tail -n 120 "$se" >&2 || true
    return "$rc"
  fi
  local line residue wall active_max active_sum mc_avg mc_max
  line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$tag missing backend result" >&2; return 4; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"
  active_max="$(field active_max_s "$line")"; active_sum="$(field active_sum_s "$line")"
  [[ -n "$residue" && -n "$wall" ]] || { echo "$tag missing residue/wall_s" >&2; return 4; }
  read -r mc_avg mc_max < <(awk '$1~/^[0-9]+$/&&$3~/^[0-9]+$/{s+=$3;n++;if($3>m)m=$3}END{if(n)printf "%.3f %d\n",s/n,m;else print "nan nan"}' "$dm")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$ilp" "$t" "$rep" "$residue" "$wall" "${active_max:-nan}" "${active_sum:-nan}" "$mc_avg" "$mc_max" >>"$RESULT"
}

# Alternate both candidate and thread order between repeats to reduce monotonic
# thermal/clock drift in a long B300 sweep.
for ((rep=1; rep<=REPEATS; ++rep)); do
  if (( rep % 2 == 1 )); then
    ilp_order=("${ilps[@]}"); thread_order=("${threads_values[@]}")
  else
    ilp_order=(); for ((i=${#ilps[@]}-1;i>=0;--i)); do ilp_order+=("${ilps[$i]}"); done
    thread_order=(); for ((i=${#threads_values[@]}-1;i>=0;--i)); do thread_order+=("${threads_values[$i]}"); done
  fi
  for t in "${thread_order[@]}"; do
    for ilp in "${ilp_order[@]}"; do
      bin="$(awk -F '\t' -v x="$ilp" '$1==x{print $2}' "$LOGDIR/binaries.tsv")"
      [[ -x "$bin" ]] || exit 3
      run_one "$ilp" "$bin" "$t" "$rep"
    done
  done
done

python3 - "$RESULT" "$RESOURCE" "$LOGDIR/binaries.tsv" "$WINNER_ENV" <<'PY'
import csv,math,statistics,sys,shlex
result,resource,bins_path,winner=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t'))
if not rows: raise SystemExit('no scaled main-pull ILP rows')
res={r['residue'] for r in rows}
if len(res)!=1:
    raise SystemExit('FATAL scaled main-pull ILP residue mismatch '+repr({(r['ilp'],r['threads'],r['repeat']):r['residue'] for r in rows}))

resources=list(csv.DictReader(open(resource),delimiter='\t'))
rb={}
for r in resources:
    label=r.get('ilp','')
    if not label.startswith('ilp'): continue
    try: ilp=int(label[3:]); regs=int(r['registers']); ss=int(r['spill_store_bytes']); sl=int(r['spill_load_bytes'])
    except (ValueError,TypeError,KeyError): continue
    old=rb.get(ilp,(0,0,0)); rb[ilp]=(max(old[0],regs),max(old[1],ss),max(old[2],sl))

bins={int(a):b for a,b,_ in list(csv.reader(open(bins_path),delimiter='\t'))[1:]}
by={}
for r in rows: by.setdefault((int(r['ilp']),int(r['threads'])),[]).append(r)
med=[]
for (ilp,t),rs in by.items():
    wall=statistics.median(float(r['wall_s']) for r in rs)
    mc=[float(r['mc_avg_pct']) for r in rs if r['mc_avg_pct']!='nan']
    mc_med=statistics.median(mc) if mc else float('nan')
    regs,ss,sl=rb.get(ilp,(-1,-1,-1))
    med.append((wall,ilp,t,mc_med,regs,ss,sl))

def key(x):
    return (x[0], -x[3] if not math.isnan(x[3]) else math.inf)
def spill_free(x): return x[4]>=0 and x[5]==0 and x[6]==0
base=min((x for x in med if x[1]==1),key=key)
clean=[x for x in med if spill_free(x)]
pool=clean or med
best=min(pool,key=key)
for x in sorted(med,key=key):
    print('MAIN_PULL_SCALED_CANDIDATE',
          f'ilp={x[1]}',f'threads={x[2]}',f'median_wall_s={x[0]:.9f}',
          f'speedup_vs_ilp1={base[0]/x[0]:.6f}x',f'mc_avg_pct={x[3]:.3f}',
          f'regs={x[4]}',f'spill_store={x[5]}',f'spill_load={x[6]}',file=sys.stderr)
print('MAIN_PULL_SCALED_SELECTED',
      f'ilp={best[1]}',f'threads={best[2]}',f'median_wall_s={best[0]:.9f}',
      f'speedup_vs_ilp1={base[0]/best[0]:.6f}x',f'residue={next(iter(res))}',
      f'spill_free_pool={int(bool(clean))}',f'exact_gate=1',file=sys.stderr)
if best[1] not in bins: raise SystemExit('winner binary lookup failed')
def q(x): return shlex.quote(str(x))
with open(winner,'w') as f:
    f.write('B300_MAIN_PULL_SCALED_WINNER_ILP='+q(best[1])+'\n')
    f.write('B300_MAIN_PULL_SCALED_WINNER_THREADS='+q(best[2])+'\n')
    f.write('B300_MAIN_PULL_SCALED_WINNER_BIN='+q(bins[best[1]])+'\n')
    f.write('B300_MAIN_PULL_SCALED_WINNER_WALL_S='+q(f'{best[0]:.9f}')+'\n')
    f.write('B300_MAIN_PULL_SCALED_WINNER_REGISTERS='+q(best[4])+'\n')
    f.write('B300_MAIN_PULL_SCALED_WINNER_SPILL_STORE_BYTES='+q(best[5])+'\n')
    f.write('B300_MAIN_PULL_SCALED_WINNER_SPILL_LOAD_BYTES='+q(best[6])+'\n')
    f.write('B300_MAIN_PULL_SCALED_RESIDUE='+q(next(iter(res)))+'\n')
print('b300_main_pull_scaled_winner_env='+winner)
PY

cat "$RESOURCE"
cat "$RESULT"
echo "b300-main-pull-ilp1234-scaled-ab OK result=$RESULT resources=$RESOURCE winner_env=$WINNER_ENV" >&2
