#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-sm_80}"
RUN_HOST_PROOF="${RUN_HOST_PROOF:-1}"
if [[ "$RUN_HOST_PROOF" != 0 && "$RUN_HOST_PROOF" != 1 ]]; then echo "RUN_HOST_PROOF must be 0 or 1" >&2; exit 2; fi
if ! command -v nvcc >/dev/null; then echo "nvcc not found" >&2; exit 2; fi
if [[ "$RUN_HOST_PROOF" == 1 ]]; then
  bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-basepair64-proof.sh"
  bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-basepair64-warp-proof.sh"
fi
SRC="$ONEESAN_ROOT/src/cuda/gridfp/probes/ramstream32_rankchunk32_basepair64_warp_selftest.cu"
for align in 0 1; do
  BIN="${BIN_PREFIX:-$ONEESAN_BUILD_DIR/rankchunk32_basepair64_warp_cuda_selftest}_align${align}"
  mkdir -p "$(dirname "$BIN")"
  TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
    -DTARGET_W=10 -DLOW_LUT_K=5 -DHIGH_LUT_K=4 \
    -DP10DC_RANKCHUNK32_BYTEPACK=1 -DP10DC_RANKCHUNK32_ALIGN32="$align" \
    -DP10DC_RANKCHUNK32_BLOCK64=0 -DP10DC_RANKCHUNK32_ONESHFL=1 \
    "$SRC" -o "$BIN"
  out="$($BIN)"
  printf '%s\n' "$out"
  grep -Eq 'rankchunk32-basepair64-warp-cuda-selftest (OK|SKIP no CUDA device)' <<<"$out"
  if grep -Fq ' OK' <<<"$out"; then
    grep -Fq 'cases=2048 starts=64 partial_widths=32' <<<"$out"
    grep -Fq "align=$align bytepack=1 pair_decode_exact=1 row_pointer_exact=1" <<<"$out"
    grep -Fq 'base_bits=22 delta_bits=8' <<<"$out"
  fi
done

echo "rankchunk32-basepair64-warp-cuda-selftest runner OK arch=$ARCH align_modes=0,1" >&2
