#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; [[ "$N" == 27 ]] || { echo 'ILP2/4 A/B currently targets n=27' >&2; exit 2; }
MOD="${MOD:-4294967291}"; ARCH="${ARCH:-native}"; NGPU=8
TARGET_MIB="${TARGET_MIB:-65536}"; PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ROWS="${ROWS:-1}"; THREADS_LIST="${THREADS_LIST:-128 256 512}"; REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_rankstate_ilp24_row${ROWS}}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")" "$ONEESAN_BUILD_DIR"
[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1&&ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

bash "$ONEESAN_ROOT/scripts/bench/b300-ilp4-partition-proof.sh"

ILP2_BIN="$ONEESAN_BUILD_DIR/b300_rankstate_ilp2_ab_n27"
ILP4_BIN="$ONEESAN_BUILD_DIR/b300_rankstate_ilp4_ab_n27"

# Build the exact current ILP2 production candidate. This also leaves the packed
# rank-state intermediate source before ILP2/concurrent-I/O lowering.
N=27 ARCH="$ARCH" OUT="$ILP2_BIN" FAST_SHARD_ADDRESS8=1 \
  MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 \
  MAIN_PULL_ILP2=0 HEIGHT_CACHE=0 RANK_DELTA_CACHE=1 RANK_STATE_PACKED=1 RANK_STATE_ILP2=1 \
  HOT_DELTA_TABLE=0 CONCURRENT_GROUP_IO=1 MAXRREGCOUNT=0 PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$LOGDIR/ilp2.build.out" 2>"$LOGDIR/ilp2.build.err"

PACKED_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n27_rank_state_packed.cu"
[[ -f "$PACKED_SRC" ]] || { echo "missing packed source: $PACKED_SRC" >&2; exit 3; }
MAIN4_SRC="$ONEESAN_BUILD_DIR/b300_rankstate_ilp4_main_n27.cu"
BOTH4_SRC="$ONEESAN_BUILD_DIR/b300_rankstate_ilp4_both_n27.cu"
CIO4_SRC="$ONEESAN_BUILD_DIR/b300_rankstate_ilp4_cio_n27.cu"
ROW4_SRC="$ONEESAN_BUILD_DIR/b300_rankstate_ilp4_row_n27.cu"
THREAD4_SRC="$ONEESAN_BUILD_DIR/b300_rankstate_ilp4_threads_n27.cu"
FINAL4_SRC="$ONEESAN_BUILD_DIR/b300_rankstate_ilp4_plan_n27.cu"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-rank-state-ilp4.py" "$PACKED_SRC" "$MAIN4_SRC" >"$LOGDIR/ilp4.main.transform.out"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-rank-state-ilp4.py" "$MAIN4_SRC" "$BOTH4_SRC" >"$LOGDIR/ilp4.block.transform.out"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-concurrent-group-io.py" "$BOTH4_SRC" "$CIO4_SRC" >"$LOGDIR/ilp4.cio.transform.out"
cp "$CIO4_SRC" "$ROW4_SRC"
python3 "$ONEESAN_ROOT/scripts/build/lower-b300-row-limit.py" "$ROW4_SRC" "$ROW4_SRC" >"$LOGDIR/ilp4.row.transform.out"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-runtime-threads.py" "$ROW4_SRC" "$THREAD4_SRC" >"$LOGDIR/ilp4.threads.transform.out"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-plan-target.py" "$THREAD4_SRC" "$FINAL4_SRC" >"$LOGDIR/ilp4.plan.transform.out"

TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v \
  -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 -DB300_FAST_SHARD_ADDRESS8=1 \
  "$FINAL4_SRC" -o "$ILP4_BIN" >"$LOGDIR/ilp4.build.out" 2>"$LOGDIR/ilp4.build.err"

PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
printf 'mode\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
python3 "$PARSER" "$LOGDIR/ilp2.build.err" --label ilp2 --contains rankstate_ilp2 >>"$RESOURCE" || true
python3 "$PARSER" "$LOGDIR/ilp4.build.err" --label ilp4 --contains rankstate_ilp4 >>"$RESOURCE" || true

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

printf 'mode\tthreads\trepeat\tresidue\twall_s\tmem_avg_pct\tmem_max_pct\tmem_busy_avg_pct\tsm_avg_pct\tpower_avg_w\tsamples\n' >"$RESULT"
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
  ((rc==0)) || { echo "$tag failed rc=$rc" >&2; tail -n 100 "$err" >&2 || true; return "$rc"; }
  local line residue wall stats
  line="$(grep '^backend=gridfp-b300-hbm32' "$out" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$tag missing backend line" >&2; return 4; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"
  stats="$(summarize "$tele")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$threads" "$rep" "$residue" "$wall" "$stats" >>"$RESULT"
}

for t in $THREADS_LIST; do
  [[ "$t" =~ ^[0-9]+$ ]] && ((t>=32&&t<=1024&&t%32==0)) || { echo "bad threads=$t" >&2; exit 2; }
  for ((r=1;r<=REPEATS;++r)); do
    echo "=== ILP2 threads=$t repeat=$r ===" >&2; run_one ilp2 "$ILP2_BIN" "$t" "$r"
    echo "=== ILP4 threads=$t repeat=$r ===" >&2; run_one ilp4 "$ILP4_BIN" "$t" "$r"
  done
done

python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
if not rows: raise SystemExit('no results')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL ILP2/4 residue mismatch '+repr(sorted(res)))
by={}
for r in rows: by.setdefault((r['mode'],int(r['threads'])),[]).append(r)
out=[]
for (m,t),g in by.items():
    wall=statistics.median(float(x['wall_s']) for x in g)
    busy=statistics.median(float(x['mem_busy_avg_pct']) for x in g)
    mem=statistics.median(float(x['mem_avg_pct']) for x in g)
    sm=statistics.median(float(x['sm_avg_pct']) for x in g)
    out.append((wall,m,t,busy,mem,sm))
for wall,m,t,busy,mem,sm in sorted(out):
    print(f'{m} threads={t} wall_s={wall:.9f} mem_busy={busy:.3f}% mem_avg={mem:.3f}% sm_avg={sm:.3f}%')
b=min(out)
print('b300_rankstate_ilp24_residue_match=1')
print(f'b300_rankstate_ilp24_best_mode={b[1]}')
print(f'b300_rankstate_ilp24_best_threads={b[2]}')
print(f'b300_rankstate_ilp24_best_wall_s={b[0]:.9f}')
print(f'b300_rankstate_ilp24_best_mem_busy_pct={b[3]:.3f}')
PY
cat "$RESOURCE"
echo "b300-rankstate-ilp24-ab OK result=$RESULT resources=$RESOURCE logs=$LOGDIR" >&2
