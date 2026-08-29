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
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_rankstate_closure_warp_hybrid_dualmask_row${ROWS}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESULT="${RESULT:-${PREFIX}.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
ISO="${ISO:-$ONEESAN_BUILD_DIR/rankstate_closure_warp_hybrid_dualmask_$$}"
mkdir -p "$LOGDIR" "$ISO" "$ISO/tmp" "$(dirname "$RESULT")"
[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1&&ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$REPEATS" =~ ^[0-9]+$ ]] && ((REPEATS>=1)) || { echo 'REPEATS must be >=1' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo 'need 8 visible GPUs' >&2; exit 2; }
for th in $THRESHOLDS; do [[ "$th" =~ ^[0-9]+$ ]] && ((th>=1&&th<=28)) || { echo "bad threshold=$th" >&2; exit 2; }; done

bash "$ONEESAN_ROOT/scripts/bench/b300-block-closure-warp-batch-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-block-closure-warp-hybrid-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-block-pull-dualmask-proof.sh"

BASE_BIN="$ISO/base.bin"
BASE_OUT="$LOGDIR/base.build.out"; BASE_ERR="$LOGDIR/base.build.err"
echo '=== build closure-warp production source ===' >&2
ONEESAN_BUILD_DIR="$ISO" ONEESAN_TMP_DIR="$ISO/tmp" \
N=27 ARCH="$ARCH" OUT="$BASE_BIN" FAST_SHARD_ADDRESS8=1 \
MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 \
MAIN_PULL_ILP2=0 HEIGHT_CACHE=0 RANK_DELTA_CACHE=1 RANK_STATE_PACKED=1 \
RANK_STATE_ILP2=0 RANK_STATE_ILP3=0 RANK_STATE_ILP4=1 BLOCK_CLOSURE_QUAD=0 BLOCK_CLOSURE_WARP=1 \
HOT_DELTA_TABLE=0 CONCURRENT_GROUP_IO=1 MAXRREGCOUNT=0 PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$BASE_OUT" 2>"$BASE_ERR"
BUILD_SRC="$(sed -nE 's/^  build_source=(.*)$/\1/p' "$BASE_OUT" | tail -n1)"
[[ -n "$BUILD_SRC" && -f "$BUILD_SRC" ]] || { echo 'could not resolve generated closure-warp source' >&2; exit 3; }

HYBRID_SRC="$ISO/hybrid.cu"
HYBRID_DUAL_SRC="$ISO/hybrid_dualmask.cu"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-closure-warp-hybrid.py" "$BUILD_SRC" "$HYBRID_SRC" >"$LOGDIR/hybrid.transform.out"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-closure-warp-dualmask.py" "$HYBRID_SRC" "$HYBRID_DUAL_SRC" >"$LOGDIR/dualmask.transform.out"
grep -Fq 'b300_block_closure_warp_use_warp' "$HYBRID_DUAL_SRC"
grep -Fq 'b300_closure_warp_endpoint_masks(d)' "$HYBRID_DUAL_SRC"

: >"$LOGDIR/binaries.tsv"
for th in $THRESHOLDS; do
  for dual in 0 1; do
    src="$HYBRID_SRC"; [[ "$dual" == 1 ]] && src="$HYBRID_DUAL_SRC"
    mode="h${th}d${dual}"; bin="$ONEESAN_BUILD_DIR/b300_rankstate_closure_${mode}_n27"; err="$LOGDIR/$mode.build.err"
    echo "=== compile threshold=$th warp_dualmask=$dual ===" >&2
    TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v \
      -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 -DB300_FAST_SHARD_ADDRESS8=1 -DB300_BLOCK_CLOSURE_QUAD=0 \
      -DB300_BLOCK_CLOSURE_WARP_MIN_ENDPOINTS="$th" "$src" -o "$bin" >"$LOGDIR/$mode.build.out" 2>"$err"
    [[ -x "$bin" ]] || { echo "missing binary $mode" >&2; exit 3; }
    printf '%s\t%s\t%s\t%s\n' "$mode" "$th" "$dual" "$bin" >>"$LOGDIR/binaries.tsv"
  done
done

PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
printf 'mode\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
while IFS=$'\t' read -r mode _ _ _; do
  python3 "$PARSER" "$LOGDIR/$mode.build.err" --label "$mode" --contains closure_warp >>"$RESOURCE" || true
done <"$LOGDIR/binaries.tsv"

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

printf 'mode\tthreshold\tdualmask\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tmem_avg_pct\tmem_max_pct\tmem_busy_avg_pct\tsm_avg_pct\tpower_avg_w\tsamples\n' >"$RESULT"
run_one(){
  local mode="$1" th="$2" dual="$3" bin="$4" threads="$5" rep="$6" tag="${mode}_t${threads}_r${rep}"
  local out="$LOGDIR/$tag.out" err="$LOGDIR/$tag.err" tele="$LOGDIR/$tag.gpu.csv"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits -lms 200 >"$tele" 2>/dev/null & local mon=$!
  sleep 1; set +e
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$threads" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" \
    "$bin" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$out" 2>"$err"
  local rc=$?; set -e; kill "$mon" 2>/dev/null || true; wait "$mon" 2>/dev/null || true
  ((rc==0)) || { echo "$tag failed rc=$rc" >&2; tail -n 160 "$err" >&2 || true; return "$rc"; }
  local line="$(grep '^backend=gridfp-b300-hbm32' "$out" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$tag missing backend line" >&2; return 4; }
  local stats="$(summarize "$tele")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$th" "$dual" "$threads" "$rep" "$(field residue "$line")" "$(field wall_s "$line")" "$(field active_max_s "$line")" "$stats" >>"$RESULT"
}

for threads in $THREADS_LIST; do
  [[ "$threads" =~ ^[0-9]+$ ]] && ((threads>=32&&threads<=1024&&threads%32==0)) || { echo "bad threads=$threads" >&2; exit 2; }
  for ((r=1;r<=REPEATS;++r)); do
    while IFS=$'\t' read -r mode th dual bin; do run_one "$mode" "$th" "$dual" "$bin" "$threads" "$r"; done <"$LOGDIR/binaries.tsv"
  done
done

python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
if not rows:raise SystemExit('no results')
res={r['residue'] for r in rows}
if len(res)!=1:raise SystemExit('FATAL hybrid-dualmask residue mismatch '+repr(sorted(res)))
by={}
for r in rows:by.setdefault((int(r['threshold']),int(r['dualmask']),int(r['threads'])),[]).append(float(r['wall_s']))
med={k:statistics.median(v) for k,v in by.items()}
print('b300_closure_warp_hybrid_dualmask_exact_intermediate_match=1')
print(f'b300_closure_warp_hybrid_dualmask_residue={next(iter(res))}')
best=None
for th in sorted({k[0] for k in med}):
  for t in sorted({k[2] for k in med if k[0]==th}):
    a=med.get((th,0,t));d=med.get((th,1,t))
    if a is None or d is None:continue
    sp=a/d
    print(f'b300_closure_warp_hybrid_dualmask_threshold_{th}_threads_{t}_base_wall_s={a:.9f}')
    print(f'b300_closure_warp_hybrid_dualmask_threshold_{th}_threads_{t}_dual_wall_s={d:.9f}')
    print(f'b300_closure_warp_hybrid_dualmask_threshold_{th}_threads_{t}_speedup={sp:.9f}x')
    if best is None or d<best[0]:best=(d,th,t,sp)
if best:
    print(f'b300_closure_warp_hybrid_dualmask_best_threshold={best[1]}')
    print(f'b300_closure_warp_hybrid_dualmask_best_threads={best[2]}')
    print(f'b300_closure_warp_hybrid_dualmask_best_wall_s={best[0]:.9f}')
    print(f'b300_closure_warp_hybrid_dualmask_best_incremental_speedup={best[3]:.9f}x')
PY
cat "$RESOURCE"
printf 'b300_closure_warp_hybrid_dualmask_rows=%s\n' "$ROWS"
printf 'b300_closure_warp_hybrid_dualmask_note=dualmask is restricted to the large-closure warp rank generator; scalar-small closure helper remains unchanged\n'
