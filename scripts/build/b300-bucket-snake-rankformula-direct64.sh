#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; W=$((N+1)); ARCH="${ARCH:-native}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"; HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
DEPTHCODE_DECODE_LOAD="${DEPTHCODE_DECODE_LOAD:-ldg}"; RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-ldg}"
PM_ACCUM="${PM_ACCUM:-0}"; TERNARY_KEY4="${TERNARY_KEY4:-1}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-0}"
if (( LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W || LOW_LUT_K > 14 || HIGH_LUT_K > 14 )); then
  echo "invalid LOW/HIGH split W=$W low=$LOW_LUT_K high=$HIGH_LUT_K" >&2; exit 2
fi
for x in PM_ACCUM TERNARY_KEY4 PTXAS_VERBOSE; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || exit 2; done
case "$DEPTHCODE_DECODE_LOAD" in global) P10DC_DECODE_LDG=0 ;; ldg) P10DC_DECODE_LDG=1 ;; *) exit 2;; esac
P10DC_RANKSTREAM_LUT_LDG=0; P10DC_RANKSTREAM_LUT_PAD256=0
case "$RANKSTREAM_LUT_LOAD" in constant) ;; ldg) P10DC_RANKSTREAM_LUT_LDG=1 ;; ldg256) P10DC_RANKSTREAM_LUT_LDG=1; P10DC_RANKSTREAM_LUT_PAD256=1 ;; *) exit 2;; esac

SRC="$(repo_path src/cuda/b300/oneesan_cuda_gridfp_b300_bucket_snake_onepass_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_direct64_graph_batch_pipeline.cu)"
OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_rankformula_direct64_pipeline_n${N}}")"
NVCC="${NVCC:-nvcc}"; command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
NVCC_EXTRA=(); [[ "$PTXAS_VERBOSE" == 1 ]] && NVCC_EXTRA+=("-Xptxas=-v")
TMPDIR="$ONEESAN_TMP_DIR" "$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  "${NVCC_EXTRA[@]}" -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DGPU_DIRECT_PM_ACCUM="$PM_ACCUM" -DBKCZ_TERNARY_KEY4="$TERNARY_KEY4" \
  -DP10DC_DECODE_LDG="$P10DC_DECODE_LDG" -DP10DC_RANKSTREAM_LUT_LDG="$P10DC_RANKSTREAM_LUT_LDG" \
  -DP10DC_RANKSTREAM_LUT_PAD256="$P10DC_RANKSTREAM_LUT_PAD256" \
  -DP10DC_RANKFORMULA_NOMETA_BLOCK=16 -DP10DC_RANKFORMULA_NOMETA_WARPSHARE=1 \
  -DP10DC_RANKFORMULA_NOMETA_COOPGROUP=1 -DP10DC_RANKFORMULA_NOMETA_COOP_UNROLL=0 \
  -DP10DC_RANKFORMULA_NOMETA_GROUP56=0 -DP10DC_RANKFORMULA_NOMETA_GROUP61=1 \
  -DP10DC_RANKFORMULA_ABSTRACT_SELECT8=1 -DP10DC_RANKFORMULA_ABSTRACT_DEPTH4=1 \
  -DP10DC_RANKFORMULA_ABSTRACT_SRCPACK10=1 \
  -DP10DC_RANKCHUNK32_ONESHFL=1 -DP10DC_RANKCHUNK32_FUSED16=0 -DP10DC_RANKCHUNK32_BYTEPACK=0 \
  -DP10DC_RANKCHUNK32_ALIGN32=0 -DP10DC_RANKCHUNK32_BLOCK64=0 -DP10DC_RANKDELTA8_FUSED13=1 \
  -DP10DC_RANKFORMULA_SPARSE_BASE=1 -DP10DC_RANKFORMULA_RAWCODE=1 -DP10DC_RANKFORMULA_INLINE_CROSS=1 \
  -DP10DC_RANKFORMULA_BASE_DELTA=0 -DP10DC_RANKFORMULA_SLOTMETA=0 -DP10DC_RANKFORMULA_SLOTROW32=0 \
  "$SRC" -o "$OUT"

echo "built $OUT"
echo "  backend=rankformula-direct64 pipeline=1 n=$N W=$W low=$LOW_LUT_K high=$HIGH_LUT_K"
echo "  locator=direct64 hot_locator_loads=1 hot_shuffle=0 hot_ballot=0 hot_scan=0"
echo "  semantic_base=group61 block=16 select8=1 depth4=1 srcpack10=1"
