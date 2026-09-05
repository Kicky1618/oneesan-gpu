#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
ARCH="${ARCH:-sm_80}"
BYTEPACK="${RANKCHUNK32_BYTEPACK:-both}"
SRC="$ONEESAN_ROOT/src/cuda/gridfp/probes/ramstream32_rankchunk32_codec_selftest.cu"
if ! command -v nvcc >/dev/null; then echo "nvcc not found" >&2; exit 2; fi
case "$BYTEPACK" in both) MODES=(0 1) ;; 0|1) MODES=("$BYTEPACK") ;; *) echo "RANKCHUNK32_BYTEPACK must be 0, 1, or both" >&2; exit 2;; esac
bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-bytepack-proof.sh"
for mode in "${MODES[@]}"; do
  BIN="${BIN:-$ONEESAN_BUILD_DIR/rankchunk32_codec_cuda_selftest_bytepack${mode}}"
  mkdir -p "$(dirname "$BIN")"
  TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
    -DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13 \
    -DP10DC_RANKCHUNK32_BYTEPACK="$mode" \
    "$SRC" -o "$BIN"
  out="$($BIN)"
  printf '%s\n' "$out"
  grep -Eq 'rankchunk32-codec-cuda-selftest (OK|SKIP no CUDA device)' <<<"$out"
  if grep -Fq ' OK' <<<"$out"; then
    if [[ "$mode" == 1 ]]; then cb=24; pb=8; else cb=23; pb=9; fi
    grep -Fq "keys=4782969 chunk_bits=$cb prefix_bits=$pb block=32" <<<"$out"
    grep -Fq "byte_aligned_chunks=$mode prefix_isolation_exact=1 device_decode_exact=1" <<<"$out"
  fi
done
echo "rankchunk32-codec-cuda-selftest OK arch=$ARCH bytepack=$BYTEPACK" >&2