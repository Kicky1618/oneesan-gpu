#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; [[ "$N" == 27 ]] || { echo 'rank-state MLP sweep currently targets n=27' >&2; exit 2; }
MOD="${MOD:-4294967291}"; ARCH="${ARCH:-native}"; NGPU=8
TARGET_MIB="${TARGET_MIB:-65536}"; PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ROWS="${ROWS:-1}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"; REPEATS="${REPEATS:-1}"
MODES="${MODES:-ilp2 ilp4 ilp4q ilp4qhot}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_rankstate_mlp_row${ROWS}}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")" "$ONEESAN_BUILD_DIR"
[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1&&ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$REPEATS" =~ ^[0-9]+$ ]] && ((REPEATS>=1)) || { echo 'REPEATS must be >=1' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

bash "$ONEESAN_ROOT/scripts/bench/b300-ilp2-partition-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-ilp4-partition-proof.sh"

build_one(){
  local mode="$1" i2=0 i4=0 cq=0 hot=0
  case "$mode" in
    ilp2) i2=1 ;;
    ilp4) i4=1 ;;
    ilp4q) i4=1; cq=1 ;;
    ilp4qhot) i4=1; cq=1; hot=1 ;;
    *) echo "unknown mode=$mode" >&2; return 2 ;;
  esac
  local bin="$ONEESAN_BUILD_DIR/b300_rankstate_${mode}_n27"
  echo "=== build $mode ===" >&2
  N=27 ARCH="$ARCH" OUT="$bin" FAST_SHARD_ADDRESS8=1 \
    MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 \
    MAIN_PULL_ILP2=0 HEIGHT_CACHE=0 RANK_DELTA_CACHE=1 RANK_STATE_PACKED=1 \
    RANK_STATE_ILP2="$i2" RANK_STATE_ILP4="$i4" BLOCK_CLOSURE_QUAD="$cq" \
    HOT_DELTA_TABLE="$hot" CONCURRENT_GROUP_IO=1 MAXRREGCOUNT=0 PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$LOGDIR/$mode.build.out" 2>"$LOGDIR/$mode.build.err"
  [[ -x "$bin" ]] || { echo "missing binary $bin" >&2; return 3; }
  printf '%s' "$bin"
}

PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
printf 'mode\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
for mode in $MODES; do
  bin="$(build_one "$mode")"
  printf '%s\t%s\n' "$mode" "$bin" >>"$LOGDIR/binaries.tsv"
  python3 "$PARSER" "$LOGDIR/$mode.build.err" --label "$mode" --contains rankstate >>"$RESOURCE" || true
done

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
summarize(){ python3 - "$1" <<'PY'
import csv,sys
mem=[];busy=[];sm=[];power=[]
for r in csv.reader(open(sys.argv[1])):
    if len(r)<5: continue
    try:s=float(r[2]);m=float(r[3]);p=float(r[4])
    except ValueError: continue
    sm.append(s);mem.append(m);power.append(p)
    if s>=50: busy.append(m)
def avg(x): return sum(x)/len(x) if x else float('nan')
print(f'{avg(mem):.6f} {max(mem) if mem else float("nan"):.6f} {avg(busy):.6f} {avg(sm):.6f} {avg(power):.6f} {len(mem)}')
PY
}

printf 'mode\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tmem_avg_pct\tmem_max_pct\tmem_busy_avg_pct\tsm_avg_pct\tpower_avg_w\tsamples\n' >"$RESULT"
run_one(){
  local mode="$1" bin="$2" threads="$3" rep="$4" tag="${mode}_t${threads}_r${rep}"
  local out="$LOGDIR/$tag.out" err="$LOGDIR/$tag.err" tele="$LOGDIR/$tag.gpu.csv"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits -lms 200 >"$tele" 2>/dev/null & local mon=$!
  sleep 1
  set +e
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$threads" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" \
    "$bin" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$out" 2>"$err"
  local rc=$?; set -e
  kill "$mon" 2>/dev/null || true; wait "$mon" 2>/dev/null || true
  ((rc==0)) || { echo "$tag failed rc=$rc" >&2; tail -n 120 "$err" >&2 || true; return "$rc"; }
  local line residue wall active stats
  line="$(grep '^backend=gridfp-b300-hbm32' "$out" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$tag missing backend line" >&2; return 4; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"; active="$(field active_max_s "$line")"; stats="$(summarize "$tele")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$threads" "$rep" "$residue" "$wall" "${active:-NA}" "$stats" >>"$RESULT"
}

for t in $THREADS_LIST; do
  [[ "$t" =~ ^[0-9]+$ ]] && ((t>=32&&t<=1024&&t%32==0)) || { echo "bad threads=$t" >&2; exit 2; }
  for mode in $MODES; do
    bin="$(awk -F '\t' -v m="$mode" '$1==m{print $2}' "$LOGDIR/binaries.tsv" | tail -n1)"
    [[ -x "$bin" ]] || { echo "missing mode binary $mode" >&2; exit 3; }
    for ((r=1;r<=REPEATS;++r)); do echo "=== $mode threads=$t repeat=$r ===" >&2; run_one "$mode" "$bin" "$t" "$r"; done
  done
done

python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
if not rows: raise SystemExit('no results')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL rank-state MLP residue mismatch '+repr(sorted(res)))
by={}
for r in rows: by.setdefault((r['mode'],int(r['threads'])),[]).append(r)
out=[]
for (m,t),g in by.items():
    wall=statistics.median(float(x['wall_s']) for x in g)
    busy=statistics.median(float(x['mem_busy_avg_pct']) for x in g)
    mem=statistics.median(float(x['mem_avg_pct']) for x in g)
    sm=statistics.median(float(x['sm_avg_pct']) for x in g)
    out.append((wall,m,t,busy,mem,sm))
for wall,m,t,busy,mem,sm in sorted(out): print(f'{m} threads={t} wall_s={wall:.9f} mem_busy={busy:.3f}% mem_avg={mem:.3f}% sm_avg={sm:.3f}%')
b=min(out)
print('b300_rankstate_mlp_residue_match=1')
print(f'b300_rankstate_mlp_best_mode={b[1]}')
print(f'b300_rankstate_mlp_best_threads={b[2]}')
print(f'b300_rankstate_mlp_best_wall_s={b[0]:.9f}')
print(f'b300_rankstate_mlp_best_mem_busy_pct={b[3]:.3f}')
PY
cat "$RESOURCE"
echo "b300-rankstate-ilp24-ab OK result=$RESULT resources=$RESOURCE logs=$LOGDIR" >&2
