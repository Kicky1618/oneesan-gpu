#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N=27; W=28; GPUS=8
MOD="${MOD:-4294967291}"
ARCH="${ARCH:-native}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
ROWS="${ROWS:-1}"
THREADS="${THREADS:-256}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
SAMPLE_LOG2="${SAMPLE_LOG2:-20}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_block_pull_profile_row${ROWS}_high${HIGH_DROP_CHUNK}}"
TMP="${TMP:-$ONEESAN_BUILD_DIR/b300_block_pull_profile_gen}"
BASE_BIN="$ONEESAN_BUILD_DIR/b300_block_pull_profile_base_n27_high${HIGH_DROP_CHUNK}"
PROF_BIN="$ONEESAN_BUILD_DIR/b300_block_pull_profile_n27_high${HIGH_DROP_CHUNK}_s${SAMPLE_LOG2}"
mkdir -p "$(dirname "$PREFIX")" "$TMP" "$ONEESAN_BUILD_DIR"
[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1&&ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$THREADS" =~ ^[0-9]+$ ]] && ((THREADS>=32&&THREADS<=1024)) || { echo 'THREADS must be 32..1024' >&2; exit 2; }
[[ "$HIGH_DROP_CHUNK" == 0 || "$HIGH_DROP_CHUNK" == 1 ]] || { echo 'HIGH_DROP_CHUNK must be 0 or 1' >&2; exit 2; }
[[ "$SAMPLE_LOG2" =~ ^[0-9]+$ ]] && ((SAMPLE_LOG2>=8&&SAMPLE_LOG2<=30)) || { echo 'SAMPLE_LOG2 must be 8..30' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= GPUS )) || { echo "need $GPUS visible GPUs" >&2; exit 2; }

# Build an uninstrumented reference through the normal production build path.
echo "=== build reference high_drop_chunk=$HIGH_DROP_CHUNK ===" >&2
N=27 ARCH="$ARCH" OUT="$BASE_BIN" MAIN_PULL_ILP=2 HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" \
  LOW_MAIN_RECURRENCE=0 HIGH_MAIN_RECURRENCE=0 MAIN_RECURRENCE=0 \
  MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 FAST_SHARD_ADDRESS8=1 \
  LOW_DROP_CACHE=1 LOW_DROP_CHUNK=1 LOW_BLOCK_CACHE=1 RUNTIME_THREADS=1 PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh" >"${PREFIX}.base.build.out" 2>"${PREFIX}.base.build.err"
grep -Fq 'main_pull_ilp=2' "${PREFIX}.base.build.out"
grep -Fq "high_drop_chunk=$HIGH_DROP_CHUNK" "${PREFIX}.base.build.out"

# Reproduce the same transform chain and add only the sampled profiler before
# shard/row/thread lowering. Production build scripts remain profiler-free.
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_forced2window_opt_batch.cu"
S1="$TMP/01_main_mate.cu"; S2="$TMP/02_main_pull.cu"; S3="$TMP/03_block_pull.cu"
S4="$TMP/04_block_mate.cu"; S5="$TMP/05_low_cache.cu"; S6="$TMP/06_low_chunk.cu"
S7="$TMP/07_high_chunk.cu"; S8="$TMP/08_low_block.cu"; S9="$TMP/09_ilp2.cu"
S10="$TMP/10_profile.cu"; S11="$TMP/11_shard8.cu"; S12="$TMP/12_rowlimit.cu"; S13="$TMP/13_threads.cu"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-mate-cache.py" "$SRC" "$S1"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-pull.py" "$S1" "$S2"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-pull.py" "$S2" "$S3"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-mate-cache.py" "$S3" "$S4"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-low-window-drop-cache.py" "$S4" "$S5"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-low-drop-chunk.py" "$S5" "$S6"
BUILD_SRC="$S6"
if [[ "$HIGH_DROP_CHUNK" == 1 ]]; then
  python3 "$ONEESAN_ROOT/scripts/build/gen-b300-high-drop-chunk.py" "$BUILD_SRC" "$S7"
  BUILD_SRC="$S7"
fi
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-low-block-cache.py" "$BUILD_SRC" "$S8"; BUILD_SRC="$S8"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-pull-ilp2.py" "$BUILD_SRC" "$S9"; BUILD_SRC="$S9"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-pull-profile.py" "$BUILD_SRC" "$S10"; BUILD_SRC="$S10"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-batch-shard-address8.py" "$BUILD_SRC" "$S11"; BUILD_SRC="$S11"
python3 "$ONEESAN_ROOT/scripts/build/lower-b300-batch-row-limit.py" "$BUILD_SRC" "$S12"; BUILD_SRC="$S12"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-runtime-threads.py" "$BUILD_SRC" "$S13"; BUILD_SRC="$S13"

grep -Fq 'B300_BLOCK_PROF[2][B300_BP_METRICS]' "$BUILD_SRC"
grep -Fq 'b300_block_profile modulus=' "$BUILD_SRC"
grep -Fq 'batch_row_limit_env' "${PREFIX}.base.build.out"

echo "=== compile sampled block profiler log2=$SAMPLE_LOG2 ===" >&2
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v \
  -DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13 -DB300_BLOCK_PROF_LOG2="$SAMPLE_LOG2" \
  "$BUILD_SRC" -o "$PROF_BIN" >"${PREFIX}.prof.build.out" 2>"${PREFIX}.prof.build.err"

run_one(){
  local mode="$1" bin="$2" out="${PREFIX}.${1}.out" err="${PREFIX}.${1}.err"
  echo "=== run $mode rows=$ROWS threads=$THREADS ===" >&2
  set +e
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$THREADS" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$out" 2>"$err"
  local rc=$?; set -e
  ((rc==0)) || { echo "$mode failed rc=$rc" >&2; tail -n 160 "$err" >&2 || true; return "$rc"; }
  local line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$mode missing backend result" >&2; return 4; }
  grep -Fq " rows=$ROWS calibration=$((ROWS<28?1:0)) " <<<"$line" || { echo "$mode row metadata mismatch" >&2; return 5; }
  printf '%s\n' "$line"
}
BASE_LINE="$(run_one base "$BASE_BIN")"
PROF_LINE="$(run_one prof "$PROF_BIN")"
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
BR="$(field residue "$BASE_LINE")"; PR="$(field residue "$PROF_LINE")"
[[ "$BR" == "$PR" ]] || { echo "FATAL block profiler residue mismatch base=$BR prof=$PR" >&2; exit 6; }
P="$(grep '^b300_block_profile ' "${PREFIX}.prof.out" | tail -n1 || true)"
[[ -n "$P" ]] || { echo 'profile result line missing' >&2; exit 7; }

python3 - "$P" "$(field wall_s "$BASE_LINE")" "$(field wall_s "$PROF_LINE")" <<'PY'
import sys
line=sys.argv[1];base=float(sys.argv[2]);prof=float(sys.argv[3])
d={}
for x in line.split()[1:]:
    if '=' in x:
        k,v=x.split('=',1);d[k]=int(v)
print('b300_block_profile_exact_intermediate_match=1')
print(f'b300_block_profile_sample_log2={d["sample_log2"]}')
print(f'b300_block_profile_reference_wall_s={base:.9f}')
print(f'b300_block_profile_instrumented_wall_s={prof:.9f}')
print(f'b300_block_profile_instrumentation_slowdown={prof/base:.6f}x')
for w in ('high','low'):
    s=d[f'{w}_sampled'];n=d[f'{w}_look_n'];e=d[f'{w}_look_endpoint']
    li=d[f'{w}_left_iters'];ri=d[f'{w}_right_iters'];lc=d[f'{w}_left_candidates'];rc=d[f'{w}_right_candidates'];ep=d[f'{w}_endpoint_popcnt'];hp=d[f'{w}_hpos']
    def q(a,b):return a/b if b else float('nan')
    print(f'b300_block_profile_{w}_sampled={s}')
    print(f'b300_block_profile_{w}_look_n_pct={100*q(n,s):.6f}')
    print(f'b300_block_profile_{w}_look_endpoint_pct={100*q(e,s):.6f}')
    print(f'b300_block_profile_{w}_hpos_per_closure={q(hp,n):.9f}')
    print(f'b300_block_profile_{w}_endpoint_popcnt_per_closure={q(ep,n):.9f}')
    print(f'b300_block_profile_{w}_left_iters_per_closure={q(li,n):.9f}')
    print(f'b300_block_profile_{w}_right_iters_per_closure={q(ri,n):.9f}')
    print(f'b300_block_profile_{w}_left_candidates_per_closure={q(lc,n):.9f}')
    print(f'b300_block_profile_{w}_right_candidates_per_closure={q(rc,n):.9f}')
    print(f'b300_block_profile_{w}_candidate_per_scan={q(lc+rc,li+ri):.9f}')
print('b300_block_profile_note=sampled structural workload only; instrumented wall time is not a production performance result')
PY
printf 'b300_block_profile_rows=%s\n' "$ROWS"
printf 'b300_block_profile_threads=%s\n' "$THREADS"
printf 'b300_block_profile_high_drop_chunk=%s\n' "$HIGH_DROP_CHUNK"
printf 'b300_block_profile_residue=%s\n' "$BR"
printf 'b300_block_profile_raw=%s\n' "$P"
