#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; MOD="${MOD:-4294967291}"; THREADS="${GRIDFP_THREADS:-256}"
TARGET_MIB="${TARGET_MIB:-65536}"; PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ROWS="${ROWS:-1}"; REPEATS="${REPEATS:-2}"; SAMPLE_S="${SAMPLE_S:-0.20}"
CONCURRENT_GROUP_IO="${CONCURRENT_GROUP_IO:-1}"; MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300x8_hot_delta_ab_n${N}}"
RESULT="${RESULT:-$LOGDIR/result.tsv}"; SUMMARY="${SUMMARY:-$LOGDIR/summary.tsv}"
RESOURCE="${RESOURCE:-$LOGDIR/ptxas.tsv}"
mkdir -p "$LOGDIR"

for x in CONCURRENT_GROUP_IO; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
[[ "$ROWS" =~ ^[0-9]+$ ]] && (( ROWS >= 1 && ROWS <= N + 1 )) || { echo "ROWS must be 1..$((N+1))" >&2; exit 2; }
[[ "$REPEATS" =~ ^[0-9]+$ ]] && (( REPEATS >= 1 )) || { echo "REPEATS must be >=1" >&2; exit 2; }
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi required" >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo "need 8 visible GPUs" >&2; exit 2; }

PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
printf 'variant\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tavg_gpu_pct\tavg_memctrl_pct\tpeak_memctrl_pct\n' >"$RESULT"
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"

build_one(){
  local hot="$1" name="$2" bin="$LOGDIR/$name"
  N="$N" OUT="$bin" FAST_SHARD_ADDRESS8=1 MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 \
    MAIN_PULL_ILP2=0 HEIGHT_CACHE=0 RANK_DELTA_CACHE=1 RANK_STATE_PACKED=1 RANK_STATE_ILP2=1 \
    HOT_DELTA_TABLE="$hot" CONCURRENT_GROUP_IO="$CONCURRENT_GROUP_IO" MAXRREGCOUNT="$MAXRREGCOUNT" PTXAS_VERBOSE=1 \
    "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$LOGDIR/$name.build.log" 2>&1
  [[ -x "$bin" ]] || { echo "missing binary $bin" >&2; exit 3; }
  if [[ "$hot" == 1 ]]; then
    grep -q 'b300_hot_delta_table=1 delta_bits=32 constant_bytes_added=17400' "$LOGDIR/$name.build.log" || {
      echo "hot-delta generator marker missing" >&2; exit 4;
    }
    grep -q 'step_n_stored=0' "$LOGDIR/$name.build.log" || {
      echo "compact hot-step marker missing" >&2; exit 4;
    }
  fi
  python3 "$PARSER" "$LOGDIR/$name.build.log" --label "$name" >>"$RESOURCE" || true
  printf '%s' "$bin"
}

sample_process(){
  local pid="$1" out="$2"; : >"$out"
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F',' '{g=$1+0;m=$2+0;print g "," m}' >>"$out" || true
    sleep "$SAMPLE_S"
  done
}

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
run_one(){
  local name="$1" bin="$2" rep="$3" log="$LOGDIR/${name}_r${rep}.run.log" tele="$LOGDIR/${name}_r${rep}.mem.csv"
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$THREADS" GRIDFP_VRAM_RESERVE_MIB=8192 \
    GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" GRIDFP_PLAN_TARGET_DIVISOR=1 \
    "$bin" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$log" 2>&1 &
  local pid=$!; sample_process "$pid" "$tele" & local sampler=$!
  set +e; wait "$pid"; local rc=$?; set -e
  wait "$sampler" || true
  (( rc == 0 )) || { echo "$name repeat=$rep failed rc=$rc" >&2; exit "$rc"; }
  local line residue wall active asum gu mu peak
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$log" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$name repeat=$rep missing result line" >&2; exit 5; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"; active="$(field active_max_s "$line")"; asum="$(field active_sum_s "$line")"
  read -r gu mu peak < <(awk -F, '{sg+=$1;sm+=$2;n++;if($2>p)p=$2}END{if(n)printf "%.3f %.3f %.0f\n",sg/n,sm/n,p;else print "nan nan nan"}' "$tele")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$rep" "$residue" "$wall" "$active" "$asum" "$gu" "$mu" "$peak" >>"$RESULT"
}

base_bin="$(build_one 0 base)"
hot_bin="$(build_one 1 hotd32)"
for ((r=1;r<=REPEATS;++r)); do run_one base "$base_bin" "$r"; done
for ((r=1;r<=REPEATS;++r)); do run_one hotd32 "$hot_bin" "$r"; done

cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL residue mismatch '+repr(sorted(res)))
keys=('wall_s','active_max_s','active_sum_s','avg_gpu_pct','avg_memctrl_pct','peak_memctrl_pct')
out=[]
for variant in ('base','hotd32'):
    g=[r for r in rows if r['variant']==variant]
    z={'variant':variant,'repeats':len(g),'residue':g[0]['residue']}
    for k in keys:z[k]=statistics.median(float(r[k]) for r in g)
    out.append(z)
with open(sys.argv[2],'w') as f:
    cols=('variant','repeats','residue')+keys
    f.write('\t'.join(cols)+'\n')
    for z in out:f.write('\t'.join(str(z[k]) for k in cols)+'\n')
q={z['variant']:z for z in out}
print(f'hotd32_wall_speedup={q["base"]["wall_s"]/q["hotd32"]["wall_s"]:.6f}x')
print(f'hotd32_active_speedup={q["base"]["active_max_s"]/q["hotd32"]["active_max_s"]:.6f}x')
print(f'hotd32_memctrl_delta={q["hotd32"]["avg_memctrl_pct"]-q["base"]["avg_memctrl_pct"]:.6f}pp')
print('delta_bits=32 constant_bytes_added=17400 step_n_stored=0 exact_residue_match=1')
PY

echo "b300x8-hot-delta-ab OK result=$RESULT summary=$SUMMARY resource=$RESOURCE rows=$ROWS" >&2
