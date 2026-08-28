#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

W="${W:-10}"; ARCH="${ARCH:-sm_80}"; PM_ACCUM="${PM_ACCUM:-0}"; DECODE_LOAD="${DECODE_LOAD:-global}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"; HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
if (( W > 12 || LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then echo "depthcode selftest requires valid W<=12 split" >&2; exit 2; fi
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then echo "PM_ACCUM must be 0 or 1" >&2; exit 2; fi
case "$DECODE_LOAD" in global) P10DC_DECODE_LDG=0 ;; ldg) P10DC_DECODE_LDG=1 ;; *) echo "DECODE_LOAD must be global or ldg" >&2; exit 2;; esac
SRC="$ONEESAN_ROOT/src/cuda/gridfp/probes/ramstream32_bucket_orbit_closure_pattern10_depthcode_selftest.cu"
BIN="${BIN:-$ONEESAN_BUILD_DIR/pattern10_depthcode_selftest_w${W}_pm${PM_ACCUM}_${DECODE_LOAD}}"
mkdir -p "$(dirname "$BIN")"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" -DGPU_DIRECT_PM_ACCUM="$PM_ACCUM" \
  -DP10DC_DECODE_LDG="$P10DC_DECODE_LDG" \
  "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Eq "bucket-closure-pattern10-depthcode-selftest (OK W=$W|SKIP no CUDA device)" <<<"$out"
if grep -Fq "OK W=$W" <<<"$out"; then
  grep -Fq 'sidecar_bytes_per_orbit=0' <<<"$out"
  grep -Fq 'temporary_depth_bytes=0' <<<"$out"
  grep -Fq 'decode_unrank=0' <<<"$out"
  grep -Fq 'payload_masks=1' <<<"$out"
  grep -Fq 'high_ctx=thread,resolved,warp,warpstriped' <<<"$out"
  grep -Fq "decode_load=$DECODE_LOAD" <<<"$out"
  grep -Fq 'warpctx_dynamic_smem=1' <<<"$out"
  grep -Fq 'warpctx_smem_bytes_256=' <<<"$out"
  grep -Fq 'warpstriped_threads=256' <<<"$out"
  grep -Fq 'warpstriped_full_warp_required=1' <<<"$out"
  grep -Fq 'delta_plan_exact=1' <<<"$out"
  grep -Eq 'pattern10-depthcode-delta-plan-equivalence checked=[1-9][0-9]* mismatches=0 .* plan_exact=1' <<<"$out"
fi
echo "pattern10-depthcode-selftest OK W=$W pm_accum=$PM_ACCUM high_ctx=thread,resolved,warp,warpstriped decode_load=$DECODE_LOAD warpctx_dynamic_smem=1 delta_plan_exact=1" >&2
