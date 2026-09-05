#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

W="${W:-10}"
ARCH="${ARCH:-sm_80}"
PM_ACCUM="${PM_ACCUM:-0}"
DECODE_LOAD="${DECODE_LOAD:-ldg}"
RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-constant}"
RANKCHUNK32_FUSED16="${RANKCHUNK32_FUSED16:-0}"
RUN_LAYOUT_PROOF="${RUN_LAYOUT_PROOF:-1}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"
HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"

if (( W > 12 || LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then
  echo "rankchunk32 basepair64 CROSS5 selftest requires valid W<=12 split" >&2
  exit 2
fi
for x in PM_ACCUM RANKCHUNK32_FUSED16 RUN_LAYOUT_PROOF; do
  v="${!x}"
  if [[ "$v" != 0 && "$v" != 1 ]]; then echo "$x must be 0 or 1" >&2; exit 2; fi
done
case "$DECODE_LOAD" in
  global) P10DC_DECODE_LDG=0 ;;
  ldg) P10DC_DECODE_LDG=1 ;;
  *) echo "DECODE_LOAD must be global or ldg" >&2; exit 2;;
esac
P10DC_RANKSTREAM_LUT_LDG=0
P10DC_RANKSTREAM_LUT_PAD256=0
case "$RANKSTREAM_LUT_LOAD" in
  constant) ;;
  ldg) P10DC_RANKSTREAM_LUT_LDG=1 ;;
  ldg256) P10DC_RANKSTREAM_LUT_LDG=1; P10DC_RANKSTREAM_LUT_PAD256=1 ;;
  *) echo "RANKSTREAM_LUT_LOAD must be constant, ldg, or ldg256" >&2; exit 2;;
esac

if [[ "$RUN_LAYOUT_PROOF" == 1 ]]; then
  bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-bytepack-proof.sh"
  bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-align32-proof.sh"
  bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-basepair64-proof.sh"
fi

SRC="$ONEESAN_ROOT/src/cuda/gridfp/probes/ramstream32_bucket_orbit_closure_pattern10_depthcode_rankchunk32_basepair64_cross5_selftest.cu"
BIN="${BIN:-$ONEESAN_BUILD_DIR/pattern10_depthcode_rankchunk32_basepair64_cross5_selftest_w${W}_pm${PM_ACCUM}_${DECODE_LOAD}_ranklut${RANKSTREAM_LUT_LOAD}_fused16${RANKCHUNK32_FUSED16}}"
mkdir -p "$(dirname "$BIN")"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DGPU_DIRECT_PM_ACCUM="$PM_ACCUM" -DP10DC_DECODE_LDG="$P10DC_DECODE_LDG" \
  -DP10DC_RANKSTREAM_LUT_LDG="$P10DC_RANKSTREAM_LUT_LDG" \
  -DP10DC_RANKSTREAM_LUT_PAD256="$P10DC_RANKSTREAM_LUT_PAD256" \
  -DP10DC_RANKCHUNK32_ONESHFL=1 \
  -DP10DC_RANKCHUNK32_FUSED16="$RANKCHUNK32_FUSED16" \
  -DP10DC_RANKCHUNK32_BYTEPACK=1 \
  -DP10DC_RANKCHUNK32_ALIGN32=1 \
  -DP10DC_RANKCHUNK32_BLOCK64=0 \
  "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Eq "bucket-closure-pattern10-depthcode-rankchunk32-basepair64-cross5-selftest (OK W=$W|SKIP no CUDA device)" <<<"$out"
if grep -Fq "OK W=$W" <<<"$out"; then
  grep -Fq 'forward_exact=1 reverse_exact=1 rankchunk32_table_exact=1 pair_table_exact=1 padding_exact=1' <<<"$out"
  grep -Fq 'chunk_bits=24 prefix_bits=8 block=32 height_align=32' <<<"$out"
  grep -Fq 'base_bits=22 delta_bits=8 codes_per_pair=64 block_base_bytes_per_code=0.0625' <<<"$out"
  grep -Fq 'block_base_loads_per_warp_max=1 byte_aligned_chunks=1' <<<"$out"
  grep -Fq 'cross_runtime_div=0 cross_runtime_mod=0 cross_runtime_direct_lookup=0' <<<"$out"
  grep -Fq 'old_prekey_offset_arrays_freed=1 fallback_structurally_unreachable=1' <<<"$out"
fi

echo "pattern10-depthcode-rankchunk32-basepair64-cross5-selftest OK W=$W pm_accum=$PM_ACCUM decode_load=$DECODE_LOAD rankstream_lut_load=$RANKSTREAM_LUT_LOAD fused16=$RANKCHUNK32_FUSED16 layout_proof=$RUN_LAYOUT_PROOF" >&2
