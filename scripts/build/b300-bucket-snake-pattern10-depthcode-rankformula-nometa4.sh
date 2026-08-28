#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; W=$((N + 1)); ARCH="${ARCH:-native}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"; HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
DEPTHCODE_DECODE_LOAD="${DEPTHCODE_DECODE_LOAD:-ldg}"
RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-ldg}"
PM_ACCUM="${PM_ACCUM:-0}"; TERNARY_KEY4="${TERNARY_KEY4:-1}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-0}"

if (( LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W || LOW_LUT_K > 14 || HIGH_LUT_K > 14 )); then
  echo "invalid pattern10 depthcode split" >&2; exit 2
fi
case "$TRANSPOSE_MODE" in
  sync) SRC_NAME="oneesan_cuda_gridfp_b300_bucket_snake_onepass_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_graph_batch.cu" ;;
  events) SRC_NAME="oneesan_cuda_gridfp_b300_bucket_snake_onepass_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_graph_batch_events.cu" ;;
  pipeline) SRC_NAME="oneesan_cuda_gridfp_b300_bucket_snake_onepass_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_graph_batch_pipeline.cu" ;;
  *) echo "TRANSPOSE_MODE must be sync, events, or pipeline" >&2; exit 2 ;;
esac
case "$DEPTHCODE_DECODE_LOAD" in global) P10DC_DECODE_LDG=0 ;; ldg) P10DC_DECODE_LDG=1 ;; *) exit 2 ;; esac
P10DC_RANKSTREAM_LUT_LDG=0; P10DC_RANKSTREAM_LUT_PAD256=0
case "$RANKSTREAM_LUT_LOAD" in
  constant) ;;
  ldg) P10DC_RANKSTREAM_LUT_LDG=1 ;;
  ldg256) P10DC_RANKSTREAM_LUT_LDG=1; P10DC_RANKSTREAM_LUT_PAD256=1 ;;
  *) echo "RANKSTREAM_LUT_LOAD must be constant, ldg, or ldg256" >&2; exit 2 ;;
esac
for x in PM_ACCUM TERNARY_KEY4 PTXAS_VERBOSE; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }; done

SRC="$(repo_path "src/cuda/b300/$SRC_NAME")"
OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_rankformula_nometa4_${TRANSPOSE_MODE}_n${N}}")"
NVCC_EXTRA=(); [[ "$PTXAS_VERBOSE" == 1 ]] && NVCC_EXTRA+=("-Xptxas=-v")
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  "${NVCC_EXTRA[@]}" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DGPU_DIRECT_PM_ACCUM="$PM_ACCUM" -DBKCZ_TERNARY_KEY4="$TERNARY_KEY4" \
  -DP10DC_DECODE_LDG="$P10DC_DECODE_LDG" \
  -DP10DC_RANKSTREAM_LUT_LDG="$P10DC_RANKSTREAM_LUT_LDG" \
  -DP10DC_RANKSTREAM_LUT_PAD256="$P10DC_RANKSTREAM_LUT_PAD256" \
  -DP10DC_RANKCHUNK32_ONESHFL=1 -DP10DC_RANKCHUNK32_FUSED16=0 \
  -DP10DC_RANKCHUNK32_BYTEPACK=0 -DP10DC_RANKCHUNK32_ALIGN32=0 -DP10DC_RANKCHUNK32_BLOCK64=0 \
  -DP10DC_RANKDELTA8_FUSED13=1 \
  -DP10DC_RANKFORMULA_SPARSE_BASE=1 -DP10DC_RANKFORMULA_RAWCODE=1 \
  -DP10DC_RANKFORMULA_INLINE_CROSS=1 -DP10DC_RANKFORMULA_BASE_DELTA=0 \
  -DP10DC_RANKFORMULA_SLOTMETA=0 -DP10DC_RANKFORMULA_SLOTROW32=0 \
  "$SRC" -o "$OUT"

echo "built $OUT (closure=pattern10-depthcode high_ctx=rankformula-nometa4 block=4 per_code_metadata_bytes=0 ballot_unrank=1 transpose=$TRANSPOSE_MODE decode_load=$DEPTHCODE_DECODE_LOAD rankstream_lut_load=$RANKSTREAM_LUT_LOAD pm_accum=$PM_ACCUM ptxas_verbose=$PTXAS_VERBOSE)"
