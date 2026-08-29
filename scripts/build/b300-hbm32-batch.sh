#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}\")/../lib/common.sh"

N="${N:-27}"
W=$((N + 1))
ARCH="${ARCH:-native}"
SRC="${SRC:-}"
if [[ -z "$SRC" ]]; then
  if (( N >= 27 )); then
    SRC="src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_forced2window_opt_batch.cu"
  else
    SRC="src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN_batch.cu"
  fi
fi
SRC="$(repo_path "$SRC")"
OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_hbm32_batch_n${N}}")"
LOW_LUT_K="${LOW_LUT_K:-}"
HIGH_LUT_K="${HIGH_LUT_K:-}"
MAIN_MATE_CACHE="${MAIN_MATE_CACHE:-1}"
MAIN_PULL="${MAIN_PULL:-1}"
BLOCK_PULL="${BLOCK_PULL:-1}"
BLOCK_MATE_CACHE="${BLOCK_MATE_CACHE:-1}"
RUNTIME_THREADS="${RUNTIME_THREADS:-1}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"

for name in MAIN_MATE_CACHE MAIN_PULL BLOCK_PULL BLOCK_MATE_CACHE RUNTIME_THREADS PTXAS_VERBOSE; do
  value="${!name}"
  [[ "$value" == 0 || "$value" == 1 ]] || { echo "$name must be 0 or 1" >&2; exit 2; }
done
if [[ "$MAIN_PULL" == 1 && "$MAIN_MATE_CACHE" != 1 ]]; then
  echo "MAIN_PULL=1 requires MAIN_MATE_CACHE=1" >&2; exit 2
fi
if [[ "$BLOCK_PULL" == 1 && "$MAIN_PULL" != 1 ]]; then
  echo "BLOCK_PULL=1 requires MAIN_PULL=1" >&2; exit 2
fi
if [[ "$BLOCK_MATE_CACHE" == 1 && "$BLOCK_PULL" != 1 ]]; then
  echo "BLOCK_MATE_CACHE=1 requires BLOCK_PULL=1" >&2; exit 2
fi

if [[ -z "$LOW_LUT_K" ]]; then
  if (( N >= 27 )); then LOW_LUT_K=14; else LOW_LUT_K=0; fi
fi
if [[ -z "$HIGH_LUT_K" ]]; then
  if (( N >= 27 )); then HIGH_LUT_K=13; else HIGH_LUT_K=0; fi
fi

if [[ "$MAIN_PULL" == 1 ]]; then
  bash "$ONEESAN_ROOT/scripts/bench/b300-main-pull-operator-proof.sh"
fi
if [[ "$BLOCK_PULL" == 1 ]]; then
  bash "$ONEESAN_ROOT/scripts/bench/b300-block-pull-operator-proof.sh"
fi

BUILD_SRC="$SRC"
if [[ "$MAIN_MATE_CACHE" == 1 ]]; then
  BUILD_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n${N}_main_mate_cache.cu"
  python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-mate-cache.py" "$SRC" "$BUILD_SRC"
fi
if [[ "$MAIN_PULL" == 1 ]]; then
  PULL_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n${N}_main_pull.cu"
  python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-pull.py" "$BUILD_SRC" "$PULL_SRC"
  BUILD_SRC="$PULL_SRC"
fi
if [[ "$BLOCK_PULL" == 1 ]]; then
  FULL_PULL_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n${N}_full_pull.cu"
  python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-pull.py" "$BUILD_SRC" "$FULL_PULL_SRC"
  BUILD_SRC="$FULL_PULL_SRC"
fi
if [[ "$BLOCK_MATE_CACHE" == 1 ]]; then
  BLOCK_CACHE_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n${N}_full_pull_block_cache.cu"
  python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-mate-cache.py" "$BUILD_SRC" "$BLOCK_CACHE_SRC"
  BUILD_SRC="$BLOCK_CACHE_SRC"
fi
if [[ "$RUNTIME_THREADS" == 1 ]]; then
  THREAD_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n${N}_runtime_threads.cu"
  python3 "$ONEESAN_ROOT/scripts/build/gen-b300-runtime-threads.py" "$BUILD_SRC" "$THREAD_SRC"
  BUILD_SRC="$THREAD_SRC"
fi

PTXAS_FLAGS=()
[[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")
TMPDIR="$ONEESAN_TMP_DIR" nvcc \
  -O3 -std=c++17 -lineinfo \
  -arch="$ARCH" \
  "${PTXAS_FLAGS[@]}" \
  -DTARGET_W="$W" \
  -DLOW_LUT_K="$LOW_LUT_K" \
  -DHIGH_LUT_K="$HIGH_LUT_K" \
  "$BUILD_SRC" -o "$OUT"

echo "built $OUT"
echo "  source=$SRC"
echo "  build_source=$BUILD_SRC"
echo "  n=$N width=$W arch=$ARCH low_lut_k=$LOW_LUT_K high_lut_k=$HIGH_LUT_K main_mate_cache=$MAIN_MATE_CACHE main_pull=$MAIN_PULL block_pull=$BLOCK_PULL block_mate_cache=$BLOCK_MATE_CACHE runtime_threads=$RUNTIME_THREADS"
