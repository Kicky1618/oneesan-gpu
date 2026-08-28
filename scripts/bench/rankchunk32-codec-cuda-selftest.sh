#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
ARCH="${ARCH:-sm_80}"
SRC="$ONEESAN_ROOT/src/cuda/gridfp/probes/ramstream32_rankchunk32_codec_selftest.cu"
BIN="${BIN:-$ONEESAN_BUILD_DIR/rankchunk32_codec_cuda_selftest}"
if ! command -v nvcc >/dev/null; then echo "nvcc not found" >&2; exit 2; fi
mkdir -p "$(dirname "$BIN")"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13 \
  "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Eq 'rankchunk32-codec-cuda-selftest (OK|SKIP no CUDA device)' <<<"$out"
if grep -Fq ' OK' <<<"$out"; then
  grep -Fq 'keys=4782969 chunk_bits=23 prefix_bits=9 block=32' <<<"$out"
  grep -Fq 'chunk0_bits=8 chunk1_bits=8 chunk2_bits=7' <<<"$out"
  grep -Fq 'device_decode_exact=1' <<<"$out"
fi
echo "rankchunk32-codec-cuda-selftest OK arch=$ARCH" >&2
