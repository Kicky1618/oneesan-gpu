#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; [[ "$N" == 27 ]] || { echo 'race currently targets n=27' >&2; exit 2; }
MOD="${MOD:-4294967291}"; ARCH="${ARCH:-native}"; NGPU=8
TARGET_MIB="${TARGET_MIB:-65536}"; PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ROWS="${ROWS:-1}"; THREADS_LIST="${THREADS_LIST:-256}"; REPEATS="${REPEATS:-1}"; SAMPLE_MS="${SAMPLE_MS:-200}"
CONCURRENT_GROUP_IO="${CONCURRENT_GROUP_IO:-1}"; MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_rankstate_ilp4_hotd32_race_row${ROWS}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")" "$ONEESAN_BUILD_DIR"

[[ "$ROWS" =~ ^[0-9]+$ ]] && (( ROWS>=1 && ROWS<=28 )) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$REPEATS" =~ ^[0-9]+$ ]] && (( REPEATS>=1 )) || { echo 'REPEATS must be >=1' >&2; exit 2; }
[[ "$CONCURRENT_GROUP_IO" == 0 || "$CONCURRENT_GROUP_IO" == 1 ]] || { echo 'CONCURRENT_GROUP_IO must be 0/1' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

bash "$ONEESAN_ROOT/scripts/bench/b300-ilp2-partition-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-ilp4-partition-proof.sh"

PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
printf 'mode\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
printf 'mode\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tmem_avg_pct\tmem_max_pct\tmem_busy_avg_pct\tsm_avg_pct\tpower_avg_w\tsamples\n' >"$RESULT"

declare -A BIN
build_one(){
  local mode="$1" ilp2="$2" ilp4="$3" hot="$4"
  local bin="$ONEESAN_BUILD_DIR/b300_rankstate_${mode}_n27" bout="$LOGDIR/${mode}.build.out" berr="$LOGDIR/${mode}.build.err"
  N=27 ARCH="$ARCH" OUT="$bin" FAST_SHARD_ADDRESS8=1 \
    MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 MAIN_PULL_ILP2=0 HEIGHT_CACHE=0 \
    RANK_DELTA_CACHE=1 RANK_STATE_PACKED=1 RANK_STATE_ILP2="$ilp2" RANK_STATE_ILP4="$ilp4" \
    HOT_DELTA_TABLE="$hot" CONCURRENT_GROUP_IO="$CONCURRENT_GROUP_IO" MAXRREGCOUNT="$MAXRREGCOUNT" PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$bout" 2>"$berr"
  [[ -x "$bin" ]] || { echo "missing binary $bin" >&2; exit 3; }
  grep -Fq "rank_state_ilp2=$ilp2 rank_state_ilp4=$ilp4 hot_delta_table=$hot" "$bout" || {
    echo "$mode build metadata mismatch" >&2; tail -n 40 "$bout" >&2; exit 4;
  }
  if [[ "$hot" == 1 ]]; then
    grep -Fq 'b300_hot_delta_table=1 delta_bits=32 constant_bytes_added=17400' "$bout" || {
      echo "$mode missing compact hot-delta transform" >&2; exit 4;
    }
  fi
  python3 "$PARSER" "$berr" --label "$mode" >>"$RESOURCE" || true
  BIN[$mode]="$bin"
}

build_one ilp2 1 0 0
build_one ilp4 0 1 0
build_one ilp4_hotd32 0 1 1

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
summarize_gpu(){ python3 - "$1" <<'PY'
import csv,sys
mem=[];busy=[];sm=[];power=[]
for r in csv.reader(open(sys.argv[1])):
    if len(r)<5: continue
    try:s=float(r[2]);m=float(r[3]);p=float(r[4])
    except ValueError: continue
    sm.append(s);mem.append(m);power.append(p)
    if s>=50: busy.append(m)
def avg(x):return sum(x)/len(x) if x else float('nan')
print(f'{avg(mem):.6f} {max(mem) if mem else float("nan"):.6f} {avg(busy):.6f} {avg(sm):.6f} {avg(power):.6f} {len(mem)}')
PY
}

run_one(){
  local mode="$1" threads="$2" rep="$3" bin="${BIN[$mode]}" tag="${mode}_t${threads}_r${rep}"
  local out="$LOGDIR/$tag.out" err="$LOGDIR/$tag.err" tele="$LOGDIR/$tag.gpu.csv"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits -lms "$SAMPLE_MS" >"$tele" 2>/dev/null & local mon=$!
  sleep 1
  set +e
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$threads" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" GRIDFP_PLAN_TARGET_DIVISOR=1 GRIDFP_VRAM_RESERVE_MIB=8192 \
    "$bin" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$out" 2>"$err"
  local rc=$?; set -e
  kill "$mon" 2>/dev/null || true; wait "$mon" 2>/dev/null || true
  (( rc==0 )) || { echo "$tag failed rc=$rc" >&2; tail -n 100 "$err" >&2 || true; exit "$rc"; }
  local line residue wall active asum stats
  line="$(grep '^backend=gridfp-b300-hbm32' "$out" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$tag missing backend line" >&2; exit 5; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"; active="$(field active_max_s "$line")"; asum="$(field active_sum_s "$line")"
  stats="$(summarize_gpu "$tele")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$threads" "$rep" "$residue" "$wall" "$active" "$asum" "$stats" >>"$RESULT"
}

for t in $THREADS_LIST; do
  [[ "$t" =~ ^[0-9]+$ ]] && (( t>=32 && t<=1024 && t%32==0 )) || { echo "bad threads=$t" >&2; exit 2; }
  for ((r=1;r<=REPEATS;++r)); do
    for mode in ilp2 ilp4 ilp4_hotd32; do
      echo "=== $mode threads=$t repeat=$r ===" >&2
      run_one "$mode" "$t" "$r"
    done
  done
done

python3 - "$RESULT" "$SUMMARY" "$WINNER_ENV" <<'PY'
import csv,statistics,sys
src,summary,winner=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t'))
if not rows: raise SystemExit('no results')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL ILP2/ILP4/hotd32 residue mismatch '+repr(sorted(res)))
num=('wall_s','active_max_s','active_sum_s','mem_avg_pct','mem_max_pct','mem_busy_avg_pct','sm_avg_pct','power_avg_w')
out=[]
for mode in ('ilp2','ilp4','ilp4_hotd32'):
  for threads in sorted({int(r['threads']) for r in rows if r['mode']==mode}):
    g=[r for r in rows if r['mode']==mode and int(r['threads'])==threads]
    z={'mode':mode,'threads':threads,'repeats':len(g),'residue':g[0]['residue']}
    for k in num:z[k]=statistics.median(float(r[k]) for r in g)
    out.append(z)
cols=('mode','threads','repeats','residue')+num
with open(summary,'w') as f:
  f.write('\t'.join(cols)+'\n')
  for z in out:f.write('\t'.join(str(z[k]) for k in cols)+'\n')
base=min((z for z in out if z['mode']=='ilp2'),key=lambda z:z['wall_s'])
for z in sorted(out,key=lambda z:z['wall_s']):
  print(z['mode'],f"threads={z['threads']}",f"wall_s={z['wall_s']:.9f}",f"speedup_vs_ilp2={base['wall_s']/z['wall_s']:.6f}x",f"mem_busy={z['mem_busy_avg_pct']:.3f}%",f"mem_avg={z['mem_avg_pct']:.3f}%",f"sm_avg={z['sm_avg_pct']:.3f}%")
best=min(out,key=lambda z:z['wall_s'])
with open(winner,'w') as f:
  f.write(f'RANK_STATE_ILP2={1 if best["mode"]=="ilp2" else 0}\n')
  f.write(f'RANK_STATE_ILP4={1 if best["mode"]!="ilp2" else 0}\n')
  f.write(f'HOT_DELTA_TABLE={1 if best["mode"]=="ilp4_hotd32" else 0}\n')
  f.write(f'GRIDFP_THREADS={best["threads"]}\n')
  f.write(f'RANKSTATE_RACE_WALL_S={best["wall_s"]:.9f}\n')
  f.write(f'RANKSTATE_RACE_MEM_BUSY_PCT={best["mem_busy_avg_pct"]:.6f}\n')
print('rankstate_race_residue_match=1')
print('WINNER='+best['mode'],f"threads={best['threads']}",f"wall_s={best['wall_s']:.9f}",f"winner_env={winner}")
PY
cat "$RESULT"
cat "$RESOURCE"
echo "b300 rankstate ILP4/hotd32 race OK result=$RESULT summary=$SUMMARY winner=$WINNER_ENV" >&2
