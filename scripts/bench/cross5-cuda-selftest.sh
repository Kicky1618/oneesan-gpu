#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

W="${W:-10}"; ARCH="${ARCH:-sm_80}"; PM_ACCUM="${PM_ACCUM:-0}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"; HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
if (( W > 12 || LOW_LUT_K <= 0 || LOW_LUT_K > 6 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then
  echo "cross5 CUDA selftest requires valid W<=12 split and LOW_LUT_K<=6" >&2; exit 2
fi
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then echo "PM_ACCUM must be 0 or 1" >&2; exit 2; fi
if ! command -v nvcc >/dev/null; then echo "nvcc not found" >&2; exit 2; fi
SRC="$ONEESAN_ROOT/src/cuda/gridfp/probes/ramstream32_bucket_cross5_selftest.cu"
BIN="${BIN:-$ONEESAN_BUILD_DIR/cross5_cuda_selftest_w${W}_pm${PM_ACCUM}}"
mkdir -p "$(dirname "$BIN")"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DGPU_DIRECT_PM_ACCUM="$PM_ACCUM" \
  "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Eq "bucket-cross5-selftest (OK W=$W|SKIP no CUDA device)" <<<"$out"
if grep -Fq "OK W=$W" <<<"$out"; then
  grep -Fq 'mismatches=0' <<<"$out"
  grep -Fq 'table_bytes=6561' <<<"$out"
  grep -Fq 'scalar_equivalent=1' <<<"$out"
  grep -Fq 'prekey_equivalent=1' <<<"$out"
  grep -Fq "pm_accum=$PM_ACCUM" <<<"$out"
fi
echo "cross5-cuda-selftest OK W=$W low=$LOW_LUT_K high=$HIGH_LUT_K pm_accum=$PM_ACCUM prekey=1" >&2
