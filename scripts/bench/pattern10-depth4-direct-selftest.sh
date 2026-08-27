#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

W="${W:-10}"
ARCH="${ARCH:-sm_80}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"
HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
PM_ACCUM="${PM_ACCUM:-0}"
SRC="$ONEESAN_ROOT/src/cuda/gridfp/probes/ramstream32_bucket_pattern10_depth4_direct_plan.cu"
BIN="${BIN:-$ONEESAN_BUILD_DIR/pattern10_depth4_direct_plan_w${W}_pm${PM_ACCUM}}"

if (( W > 12 || LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then
  echo "invalid small-width split" >&2
  exit 2
fi
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then
  echo "PM_ACCUM must be 0 or 1" >&2
  exit 2
fi
if ! command -v nvcc >/dev/null; then
  echo "nvcc not found" >&2
  exit 2
fi

mkdir -p "$(dirname "$BIN")"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" \
  -DLOW_LUT_K="$LOW_LUT_K" \
  -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DGPU_DIRECT_PM_ACCUM="$PM_ACCUM" \
  "$SRC" -o "$BIN"

"$BIN"
