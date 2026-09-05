#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N=27; NGPU=8
MOD="${MOD:-4294967291}"
ARCH="${ARCH:-native}"
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
ROWS="${ROWS:-1}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
THRESHOLDS="${THRESHOLDS:-4 8 12}"
REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_rankstate_closure_warp_hybrid_row${ROWS}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESULT="${RESULT:-${PREFIX}.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
ISO="${ISO:-$ONEESAN_BUILD_DIR/rankstate_closure_warp_hybrid_$$}"
mkdir -p "$LOGDIR" "$ISO" "$ISO/tmp" "$(dirname "$RESULT")"
[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1&&ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$REPEATS" =~ ^[0-9]+$ ]] && ((REPEATS>=1)) || { echo 'REPEATS must be >=1' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo 'need 8 visible GPUs' >&2; exit 2; }
for t in $THRESHOLDS; do [[ "$t" =~ ^[0-9]+$ ]] && ((t>=1&&t<=28)) || { echo "bad threshold=$t" >&2; exit 2; }; done

bash "$ONEESAN_ROOT/scripts/bench/b300-block-closure-warp-batch-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-block-closure-warp-hybrid-proof.sh"

BASE_BIN="$ONEESAN_BUILD_DIR/b300_rankstate_ilp4warp_hybrid_base_n27"
BASE_BUILD_OUT="$LOGDIR/base.build.out"; BASE_BUILD_ERR="$LOGDIR/base.build.err"
echo '=== build all-warp baseline ===' >&2
ONEESAN_BUILD_DIR="$ISO" ONEESAN_TMP_DIR="$ISO/tmp" \
N=27 ARCH="$ARCH" OUT="$BASE_BIN" FAST_SHARD_ADDRESS8=1 \
MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 \
MAIN_PULL_ILP2=0 HEIGHT_CACHE=0 RANK_DELTA_CACHE=1 RANK_STATE_PACKED=1 \
RANK_STATE_ILP2=0 RANK_STATE_ILP3=0 RANK_STATE_ILP4=1 BLOCK_CLOSURE_QUAD=0 BLOCK_CLOSURE_WARP=1 \
HOT_DELTA_TABLE=0 CONCURRENT_GROUP_IO=1 MAXRREGCOUNT=0 PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$BASE_BUILD_OUT" 2>"$BASE_BUILD_ERR"
[[ -x "$BASE_BIN" ]] || { echo 'baseline binary missing' >&2; exit 3; }
BUILD_SRC="$(sed -nE 's/^  build_source=(.*)$/\1/p' "$BASE_BUILD_OUT" | tail -n1)"
[[ -n "$BUILD_SRC" && -f "$BUILD_SRC" ]] || { echo 'could not resolve generated baseline source' >&2; exit 3; }

HYBRID_SRC="$ISO/final_closure_warp_hybrid.cu"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-closure-warp-hybrid.py" "$BUILD_SRC" "$HYBRID_SRC" >"$LOGDIR/hybrid.transform.out"
grep -Fq 'B300_BLOCK_CLOSURE_WARP_MIN_ENDPOINTS' "$HYBRID_SRC"
grep -Fq 'b300_block_closure_warp_use_warp' "$HYBRID_SRC"

: >"$LOGDIR/binaries.tsv"
printf 'base\t%s\n' "$BASE_BIN" >>"$LOGDIR/binaries.tsv"
for th in $THRESHOLDS; do
  bin="$ONEESAN_BUILD_DIR/b300_rankstate_ilp4warp_hybrid_t${th}_n27"
  err="$LOGDIR/t${th}.build.err"
  echo "=== compile hybrid threshold=$th ===" >&2
  TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v \
    -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 -DB300_FAST_SHARD_ADDRESS8=1 -DB300_BLOCK_CLOSURE_QUAD=0 \
    -DB300_BLOCK_CLOSURE_WARP_MIN_ENDPOINTS="$th" "$HYBRID_SRC" -o "$bin" >"$LOGDIR/t${th}.build.out" 2>"$err"
  [[ -x "$bin" ]] || { echo "candidate binary missing threshold=$th" >&2; exit 3; }
  printf 't%s\t%s\n' "$th" "$bin" >>"$LOGDIR/binaries.tsv"
done

PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
printf 'mode\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
python3 "$PARSER" "$BASE_BUILD_ERR" --label base --contains closure_warp >>"$RESOURCE" || true
for th in $THRESHOLDS; do
  python3 "$PARSER" "$LOGDIR/t${th}.build.err" --label "t${th}" --contains closure_warp >>"$RESOURCE" || true
  python3 "$PARSER" "$LOGDIR/t${th}.build.err" --label "t${th}_scalar" --contains rankstate_ilp4 >>"$RESOURCE" || true
done

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
summarize(){ python3 - "$1" <<'PY'
import csv,sys
mem=[];busy=[];sm=[];power=[]
for r in csv.reader(open(sys.argv[1])):
    if len(r)<5:continue
    try:s=float(r[2]);m=float(r[3]);p=float(r[4])
    except ValueError:continue
    sm.append(s);mem.append(m);power.append(p)
    if s>=50:busy.append(m)
def avg(x):return sum(x)/len(x) if x else float('nan')
print(f'{avg(mem):.6f} {max(mem) if mem else float("nan"):.6f} {avg(busy):.6f} {avg(sm):.6f} {avg(power):.6f} {len(mem)}')
PY
}

printf 'mode\tthreshold\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tmem_avg_pct\tmem_max_pct\tmem_busy_avg_pct\tsm_avg_pct\tpower_avg_w\tsamples\n' >"$RESULT"
run_one(){
  local mode="$1" threshold="$2" bin="$3" threads="$4" rep="$5"
  local tag="${mode}_t${threads}_r${rep}"
  local out="$LOGDIR/$tag.out" err="$LOGDIR/$tag.err" tele="$LOGDIR/$tag.gpu.csv"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits -lms 200 >"$tele" 2>/dev/null & local mon=$!
  sleep 1; set +e
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$threads" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" \
    "$bin" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$out" 2>"$err"
  local rc=$?; set -e
  kill "$mon" 2>/dev/null || true; wait "$mon" 2>/dev/null || true
  ((rc==0)) || { echo "$tag failed rc=$rc" >&2; tail -n 160 "$err" >&2 || true; return "$rc"; }
  local line="$(grep '^backend=gridfp-b300-hbm32' "$out" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$tag missing backend line" >&2; return 4; }
  local stats="$(summarize "$tele")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$threshold" "$threads" "$rep" "$(field residue "$line")" "$(field wall_s "$line")" "$(field active_max_s "$line")" "$stats" >>"$RESULT"
}

for threads in $THREADS_LIST; do
  [[ "$threads" =~ ^[0-9]+$ ]] && ((threads>=32&&threads<=1024&&threads%32==0)) || { echo "bad threads=$threads" >&2; exit 2; }
  for ((r=1;r<=REPEATS;++r)); do
    if ((r&1)); then run_one base 0 "$BASE_BIN" "$threads" "$r"; fi
    for th in $THRESHOLDS; do
      bin="$(awk -F '\t' -v m="t$th" '$1==m{print $2}' "$LOGDIR/binaries.tsv")"
      run_one "hybrid${th}" "$th" "$bin" "$threads" "$r"
    done
    if ((!(r&1))); then run_one base 0 "$BASE_BIN" "$threads" "$r"; fi
  done
done

python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
if not rows:raise SystemExit('no results')
res={r['residue'] for r in rows}
if len(res)!=1:raise SystemExit('FATAL closure-warp hybrid residue mismatch '+repr(sorted(res)))
by={}
for r in rows:by.setdefault((r['mode'],int(r['threads'])),[]).append(float(r['wall_s']))
med={k:statistics.median(v) for k,v in by.items()}
base_best=min((v,t) for (m,t),v in med.items() if m=='base')
print('b300_closure_warp_hybrid_exact_intermediate_match=1')
print(f'b300_closure_warp_hybrid_residue={next(iter(res))}')
print(f'b300_closure_warp_hybrid_base_best_wall_s={base_best[0]:.9f}')
print(f'b300_closure_warp_hybrid_base_best_threads={base_best[1]}')
best=None
for (m,t),c in sorted(med.items()):
    if not m.startswith('hybrid'):continue
    b=med.get(('base',t));
    if b is None:continue
    th=int(m.removeprefix('hybrid'));sp=b/c
    print(f'b300_closure_warp_hybrid_threshold_{th}_threads_{t}_wall_s={c:.9f}')
    print(f'b300_closure_warp_hybrid_threshold_{th}_threads_{t}_speedup_vs_same_thread_base={sp:.9f}x')
    if best is None or c<best[0]:best=(c,th,t,sp)
if best:
    print(f'b300_closure_warp_hybrid_best_threshold={best[1]}')
    print(f'b300_closure_warp_hybrid_best_threads={best[2]}')
    print(f'b300_closure_warp_hybrid_best_wall_s={best[0]:.9f}')
    print(f'b300_closure_warp_hybrid_best_same_thread_speedup={best[3]:.9f}x')
    print(f'b300_closure_warp_hybrid_speedup_vs_global_base_best={base_best[0]/best[0]:.9f}x')
PY
cat "$RESOURCE"
printf 'b300_closure_warp_hybrid_rows=%s\n' "$ROWS"
printf 'b300_closure_warp_hybrid_note=experimental scalar-small/warp-large partition; promote only on exact match plus reproducible global wall win\n'
