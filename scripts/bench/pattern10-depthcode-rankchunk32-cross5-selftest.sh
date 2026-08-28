#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

W="${W:-10}"; ARCH="${ARCH:-sm_80}"; PM_ACCUM="${PM_ACCUM:-0}"; DECODE_LOAD="${DECODE_LOAD:-ldg}"; RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-constant}"; RANKCHUNK32_ONESHFL="${RANKCHUNK32_ONESHFL:-1}"; RANKCHUNK32_FUSED16="${RANKCHUNK32_FUSED16:-0}"; RANKCHUNK32_BYTEPACK="${RANKCHUNK32_BYTEPACK:-0}"; RANKCHUNK32_ALIGN32="${RANKCHUNK32_ALIGN32:-0}"; RANKCHUNK32_BLOCK64="${RANKCHUNK32_BLOCK64:-0}"; RUN_LAYOUT_PROOF="${RUN_LAYOUT_PROOF:-1}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"; HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
if (( W > 12 || LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then echo "rankchunk32 CROSS5 selftest requires valid W<=12 split" >&2; exit 2; fi
for x in PM_ACCUM RANKCHUNK32_ONESHFL RANKCHUNK32_FUSED16 RANKCHUNK32_BYTEPACK RANKCHUNK32_ALIGN32 RANKCHUNK32_BLOCK64 RUN_LAYOUT_PROOF; do
  v="${!x}"; if [[ "$v" != 0 && "$v" != 1 ]]; then echo "$x must be 0 or 1" >&2; exit 2; fi
done
if [[ "$RANKCHUNK32_BLOCK64" == 1 && "$RANKCHUNK32_BYTEPACK" == 1 ]]; then echo "BLOCK64 requires BYTEPACK=0" >&2; exit 2; fi
case "$DECODE_LOAD" in global) P10DC_DECODE_LDG=0 ;; ldg) P10DC_DECODE_LDG=1 ;; *) echo "DECODE_LOAD must be global or ldg" >&2; exit 2;; esac
P10DC_RANKSTREAM_LUT_LDG=0; P10DC_RANKSTREAM_LUT_PAD256=0
case "$RANKSTREAM_LUT_LOAD" in
  constant) ;;
  ldg) P10DC_RANKSTREAM_LUT_LDG=1 ;;
  ldg256) P10DC_RANKSTREAM_LUT_LDG=1; P10DC_RANKSTREAM_LUT_PAD256=1 ;;
  *) echo "RANKSTREAM_LUT_LOAD must be constant, ldg, or ldg256" >&2; exit 2;;
esac

if [[ "$RANKCHUNK32_BYTEPACK" == 1 ]]; then EXPECT_CHUNK_BITS=24; EXPECT_PREFIX_BITS=8; PACK_NAME=24bit_chunks_8bit_prefix
else EXPECT_CHUNK_BITS=23; EXPECT_PREFIX_BITS=9; PACK_NAME=23bit_chunks_9bit_prefix; fi
if [[ "$RANKCHUNK32_BLOCK64" == 1 ]]; then EXPECT_BLOCK=64; BLOCK_NAME=block64
else EXPECT_BLOCK=32; BLOCK_NAME=block32; fi
if [[ "$RANKCHUNK32_ALIGN32" == 1 ]]; then EXPECT_ALIGN="$EXPECT_BLOCK"; EXPECT_LOADS=1; ALIGN_NAME="align${EXPECT_BLOCK}"
else EXPECT_ALIGN=1; EXPECT_LOADS=2; ALIGN_NAME=packed_heights; fi
LAYOUT_NAME="${PACK_NAME}_${BLOCK_NAME}_${ALIGN_NAME}"

if [[ "$RUN_LAYOUT_PROOF" == 1 ]]; then
  bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-warpbase-proof.sh"
  if [[ "$RANKCHUNK32_BYTEPACK" == 1 ]]; then bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-bytepack-proof.sh"; fi
  if [[ "$RANKCHUNK32_BLOCK64" == 1 ]]; then bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-block64-proof.sh"; fi
  if [[ "$RANKCHUNK32_ALIGN32" == 1 && "$RANKCHUNK32_BLOCK64" == 0 ]]; then bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-align32-proof.sh"; fi
  if [[ "$RANKCHUNK32_ALIGN32" == 1 && "$RANKCHUNK32_BLOCK64" == 1 ]]; then bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-block64-align-proof.sh"; fi
fi

SRC="$ONEESAN_ROOT/src/cuda/gridfp/probes/ramstream32_bucket_orbit_closure_pattern10_depthcode_rankchunk32_cross5_selftest.cu"
BIN="${BIN:-$ONEESAN_BUILD_DIR/pattern10_depthcode_rankchunk32_cross5_selftest_w${W}_pm${PM_ACCUM}_${DECODE_LOAD}_ranklut${RANKSTREAM_LUT_LOAD}_oneshfl${RANKCHUNK32_ONESHFL}_fused16${RANKCHUNK32_FUSED16}_bytepack${RANKCHUNK32_BYTEPACK}_align${RANKCHUNK32_ALIGN32}_block64${RANKCHUNK32_BLOCK64}}"
mkdir -p "$(dirname "$BIN")"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DGPU_DIRECT_PM_ACCUM="$PM_ACCUM" -DP10DC_DECODE_LDG="$P10DC_DECODE_LDG" \
  -DP10DC_RANKSTREAM_LUT_LDG="$P10DC_RANKSTREAM_LUT_LDG" \
  -DP10DC_RANKSTREAM_LUT_PAD256="$P10DC_RANKSTREAM_LUT_PAD256" \
  -DP10DC_RANKCHUNK32_ONESHFL="$RANKCHUNK32_ONESHFL" \
  -DP10DC_RANKCHUNK32_FUSED16="$RANKCHUNK32_FUSED16" \
  -DP10DC_RANKCHUNK32_BYTEPACK="$RANKCHUNK32_BYTEPACK" \
  -DP10DC_RANKCHUNK32_ALIGN32="$RANKCHUNK32_ALIGN32" \
  -DP10DC_RANKCHUNK32_BLOCK64="$RANKCHUNK32_BLOCK64" \
  "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Eq "bucket-closure-pattern10-depthcode-rankchunk32-cross5-selftest (OK W=$W|SKIP no CUDA device)" <<<"$out"
if grep -Fq "OK W=$W" <<<"$out"; then
  grep -Fq 'forward_exact=1 reverse_exact=1 rankchunk32_table_exact=1 padding_exact=1' <<<"$out"
  grep -Fq "chunk_bits=$EXPECT_CHUNK_BITS prefix_bits=$EXPECT_PREFIX_BITS block=$EXPECT_BLOCK height_align=$EXPECT_ALIGN" <<<"$out"
  grep -Fq "block_base_loads_per_warp_max=$EXPECT_LOADS" <<<"$out"
  grep -Fq 'cross_runtime_div=0 cross_runtime_mod=0 cross_runtime_direct_lookup=0' <<<"$out"
  grep -Fq 'old_prekey_offset_arrays_freed=1 fallback_structurally_unreachable=1' <<<"$out"
fi
echo "pattern10-depthcode-rankchunk32-cross5-selftest OK W=$W pm_accum=$PM_ACCUM decode_load=$DECODE_LOAD rankstream_lut_load=$RANKSTREAM_LUT_LOAD layout=$LAYOUT_NAME rankchunk32_bytepack=$RANKCHUNK32_BYTEPACK rankchunk32_align32=$RANKCHUNK32_ALIGN32 rankchunk32_block64=$RANKCHUNK32_BLOCK64 prefix_bound_proved=1 rankchunk32_oneshfl=$RANKCHUNK32_ONESHFL rankchunk32_fused16=$RANKCHUNK32_FUSED16 layout_proof=$RUN_LAYOUT_PROOF" >&2