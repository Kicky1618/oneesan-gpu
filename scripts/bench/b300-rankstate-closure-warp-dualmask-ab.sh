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
REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_rankstate_closure_warp_dualmask_row${ROWS}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESULT="${RESULT:-${PREFIX}.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
ISO="${ISO:-$ONEESAN_BUILD_DIR/rankstate_closure_warp_dualmask_$$}"
mkdir -p "$LOGDIR" "$ISO" "$ISO/tmp" "$(dirname "$RESULT")"
[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1&&ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$REPEATS" =~ ^[0-9]+$ ]] && ((REPEATS>=1)) || { echo 'REPEATS must be >=1' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

bash "$ONEESAN_ROOT/scripts/bench/b300-block-closure-warp-batch-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-block-pull-dualmask-proof.sh"

BASE_BIN="$ONEESAN_BUILD_DIR/b300_rankstate_ilp4warp_dualmask_base_n27"
CAND_BIN="$ONEESAN_BUILD_DIR/b300_rankstate_ilp4warp_dualmask_n27"
BASE_BUILD_OUT="$LOGDIR/base.build.out"; BASE_BUILD_ERR="$LOGDIR/base.build.err"

echo '=== build rank-state ILP4 closure-warp baseline ===' >&2
ONEESAN_BUILD_DIR="$ISO" ONEESAN_TMP_DIR="$ISO/tmp" \
N=27 ARCH="$ARCH" OUT="$BASE_BIN" FAST_SHARD_ADDRESS8=1 \
MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 \
MAIN_PULL_ILP2=0 HEIGHT_CACHE=0 RANK_DELTA_CACHE=1 RANK_STATE_PACKED=1 \
RANK_STATE_ILP2=0 RANK_STATE_ILP3=0 RANK_STATE_ILP4=1 BLOCK_CLOSURE_QUAD=0 BLOCK_CLOSURE_WARP=1 \
HOT_DELTA_TABLE=0 CONCURRENT_GROUP_IO=1 MAXRREGCOUNT=0 PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$BASE_BUILD_OUT" 2>"$BASE_BUILD_ERR"
[[ -x "$BASE_BIN" ]] || { echo 'baseline binary missing' >&2; exit 3; }
grep -Fq 'block_closure_warp=1' "$BASE_BUILD_OUT" || true
BUILD_SRC="$(sed -nE 's/^  build_source=(.*)$/\1/p' "$BASE_BUILD_OUT" | tail -n1)"
[[ -n "$BUILD_SRC" && -f "$BUILD_SRC" ]] || { echo 'could not resolve generated baseline CUDA source' >&2; exit 3; }
grep -Fq 'b300_block_closure_warp_kernel' "$BUILD_SRC"

DUAL_SRC="$ISO/final_closure_warp_dualmask.cu"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-closure-warp-dualmask.py" "$BUILD_SRC" "$DUAL_SRC" >"$LOGDIR/dualmask.transform.out"
grep -Fq 'b300_closure_warp_endpoint_masks(d)' "$DUAL_SRC"
grep -Fq 'b300_block_closure_warp_kernel' "$DUAL_SRC"

CAND_BUILD_ERR="$LOGDIR/cand.build.err"
echo '=== compile rank-state ILP4 closure-warp + dualmask ===' >&2
TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v \
  -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 -DB300_FAST_SHARD_ADDRESS8=1 -DB300_BLOCK_CLOSURE_QUAD=0 \
  "$DUAL_SRC" -o "$CAND_BIN" >"$LOGDIR/cand.build.out" 2>"$CAND_BUILD_ERR"
[[ -x "$CAND_BIN" ]] || { echo 'candidate binary missing' >&2; exit 3; }

PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
printf 'mode\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
python3 "$PARSER" "$BASE_BUILD_ERR" --label base --contains closure_warp >>"$RESOURCE" || true
python3 "$PARSER" "$CAND_BUILD_ERR" --label dualmask --contains closure_warp >>"$RESOURCE" || true

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

printf 'mode\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tmem_avg_pct\tmem_max_pct\tmem_busy_avg_pct\tsm_avg_pct\tpower_avg_w\tsamples\n' >"$RESULT"
run_one(){
  local mode="$1" bin="$2" threads="$3" rep="$4" tag="${mode}_t${threads}_r${rep}"
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
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$threads" "$rep" "$(field residue "$line")" "$(field wall_s "$line")" "$(field active_max_s "$line")" "$stats" >>"$RESULT"
}

for t in $THREADS_LIST; do
  [[ "$t" =~ ^[0-9]+$ ]] && ((t>=32&&t<=1024&&t%32==0)) || { echo "bad threads=$t" >&2; exit 2; }
  for ((r=1;r<=REPEATS;++r)); do
    # Alternate order across repeats to reduce thermal/order bias.
    if ((r&1)); then
      run_one base "$BASE_BIN" "$t" "$r"; run_one dualmask "$CAND_BIN" "$t" "$r"
    else
      run_one dualmask "$CAND_BIN" "$t" "$r"; run_one base "$BASE_BIN" "$t" "$r"
    fi
  done
done

python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
if not rows:raise SystemExit('no results')
res={r['residue'] for r in rows}
if len(res)!=1:raise SystemExit('FATAL closure-warp dualmask residue mismatch '+repr(sorted(res)))
by={}
for r in rows:by.setdefault((r['mode'],int(r['threads'])),[]).append(float(r['wall_s']))
med={(m,t):statistics.median(v) for (m,t),v in by.items()}
print('b300_closure_warp_dualmask_exact_intermediate_match=1')
print(f'b300_closure_warp_dualmask_residue={next(iter(res))}')
best=None
for t in sorted({t for _,t in med}):
    b=med.get(('base',t));c=med.get(('dualmask',t))
    if b is None or c is None:continue
    sp=b/c
    print(f'b300_closure_warp_dualmask_threads_{t}_base_wall_s={b:.9f}')
    print(f'b300_closure_warp_dualmask_threads_{t}_candidate_wall_s={c:.9f}')
    print(f'b300_closure_warp_dualmask_threads_{t}_speedup={sp:.9f}x')
    if best is None or c<best[0]:best=(c,t,sp)
if best:
    print(f'b300_closure_warp_dualmask_best_threads={best[1]}')
    print(f'b300_closure_warp_dualmask_best_candidate_wall_s={best[0]:.9f}')
    print(f'b300_closure_warp_dualmask_best_thread_speedup={best[2]:.9f}x')
PY
cat "$RESOURCE"
printf 'b300_closure_warp_dualmask_rows=%s\n' "$ROWS"
printf 'b300_closure_warp_dualmask_note=experimental post-transform; promote only on exact match plus reproducible wall-time win\n'
