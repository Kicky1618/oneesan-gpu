#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-native}"
PTX_ARCH="${PTX_ARCH:-sm_80}"
W="${W:-28}"
BLOCKS="${BLOCKS:-256}"
THREADS="${THREADS:-256}"
ITERS="${ITERS:-4096}"
REPEATS="${REPEATS:-9}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"

if (( W < 8 || W > 28 || W % 2 != 0 || BLOCKS < 1 || THREADS < 1 || THREADS > 1024 || ITERS < 1 || REPEATS < 1 )); then
  echo "invalid W/BLOCKS/THREADS/ITERS/REPEATS" >&2
  exit 2
fi
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }

bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-owner-u32limb-ngpu8-proof.sh"
if [[ "$W" == 28 ]]; then
  bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-owner-w28-ngpu8-direct-proof.sh"
  ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-owner-u32limb-ngpu8-w28-ptx-proof.sh"
fi

SRC="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_runtime_owner_u32limb_ngpu8_microprobe.cu"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_owner_u32limb_ngpu8_microprobe}"
BUILD_LOG="${BUILD_LOG:-$ONEESAN_BUILD_DIR/gridfp_runtime_owner_u32limb_ngpu8_microprobe.ptxas.log}"
mkdir -p "$(dirname "$BIN")"
PTXAS_FLAGS=()
[[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")
"$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" "${PTXAS_FLAGS[@]}" \
  "$SRC" -o "$BIN" 2> >(tee "$BUILD_LOG" >&2)

out="$($BIN "$W" "$BLOCKS" "$THREADS" "$ITERS" "$REPEATS")"
printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-owner-u32limb-ngpu8-microprobe OK' <<<"$out"
grep -Fq 'ngpu=8' <<<"$out"
grep -Fq 'exact=OK' <<<"$out"
grep -Fq 'ngpu_mul_old=1 ngpu_mul_new=0 shift_bias=3' <<<"$out"

if [[ "$PTXAS_VERBOSE" == 1 ]]; then
  echo '--- ptxas ngpu8 owner microprobe ---' >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' "$BUILD_LOG" >&2 || true
fi

echo "gridfp-runtime-owner-u32limb-ngpu8-microprobe runner OK W=$W blocks=$BLOCKS threads=$THREADS iters=$ITERS repeats=$REPEATS direct_w28=$([[ "$W" == 28 ]] && echo 1 || echo 0) ptx_gate=$([[ "$W" == 28 ]] && echo 1 || echo 0)" >&2
