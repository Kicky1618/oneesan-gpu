#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; W=$((N + 1)); ARCH="${ARCH:-native}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"; HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
COL_ILP="${COL_ILP:-2}"; PM_ACCUM="${PM_ACCUM:-1}"
DEPTHMAJOR="${DEPTHMAJOR:-1}"; FORCE7="${FORCE7:-0}"; MLP_WINDOW4="${MLP_WINDOW4:-0}"
PREFETCH_NEXT="${PREFETCH_NEXT:-0}"
MAXRREGCOUNT="${MAXRREGCOUNT:-0}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
OUT="${OUT:-oneesan_cuda_gridfp_b300_directgather_colilp${COL_ILP}_n${N}}"

if (( LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W || LOW_LUT_K > 14 || HIGH_LUT_K > 14 )); then
  echo "invalid LOW/HIGH split W=$W low=$LOW_LUT_K high=$HIGH_LUT_K" >&2; exit 2
fi
case "$COL_ILP" in 1|2|4) ;; *) echo "COL_ILP must be 1, 2, or 4" >&2; exit 2;; esac
for x in PM_ACCUM DEPTHMAJOR FORCE7 MLP_WINDOW4 PREFETCH_NEXT PTXAS_VERBOSE; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
if [[ "$FORCE7" == 1 && "$MLP_WINDOW4" == 1 ]]; then echo "FORCE7 and MLP_WINDOW4 are mutually exclusive" >&2; exit 2; fi
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] || { echo "MAXRREGCOUNT must be non-negative integer" >&2; exit 2; }
if (( MAXRREGCOUNT != 0 && (MAXRREGCOUNT < 32 || MAXRREGCOUNT > 255) )); then
  echo "MAXRREGCOUNT must be 0 or 32..255" >&2; exit 2
fi

case "$TRANSPOSE_MODE" in
  sync) SRC_NAME="oneesan_cuda_gridfp_b300_bucket_snake_onepass_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_graph_batch.cu" ;;
  events) SRC_NAME="oneesan_cuda_gridfp_b300_bucket_snake_onepass_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_graph_batch_events.cu" ;;
  pipeline) SRC_NAME="oneesan_cuda_gridfp_b300_bucket_snake_onepass_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_graph_batch_pipeline.cu" ;;
  *) echo "TRANSPOSE_MODE must be sync/events/pipeline" >&2; exit 2;;
esac
SRC="$ONEESAN_ROOT/src/cuda/b300/$SRC_NAME"
BIN="$(build_path "$OUT")"
NVCC_EXTRA=()
[[ "$PTXAS_VERBOSE" == 1 ]] && NVCC_EXTRA+=("-Xptxas=-v")
(( MAXRREGCOUNT )) && NVCC_EXTRA+=("-maxrregcount=$MAXRREGCOUNT")

TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  "${NVCC_EXTRA[@]}" -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DGPU_DIRECT_PM_ACCUM="$PM_ACCUM" -DBKCZ_TERNARY_KEY4=1 \
  -DP10DC_DECODE_LDG=1 -DP10DC_RANKSTREAM_LUT_LDG=1 -DP10DC_RANKSTREAM_LUT_PAD256=0 \
  -DP10DC_RANKFORMULA_NOMETA_BLOCK=16 \
  -DP10DC_RANKFORMULA_NOMETA_WARPSHARE=1 \
  -DP10DC_RANKFORMULA_NOMETA_COOPGROUP=1 \
  -DP10DC_RANKFORMULA_NOMETA_COOP_UNROLL=0 \
  -DP10DC_RANKFORMULA_NOMETA_GROUP56=0 \
  -DP10DC_RANKFORMULA_NOMETA_GROUP61=1 \
  -DP10DC_RANKFORMULA_NOMETA_DIRECTMAP=1 \
  -DP10DC_RANKFORMULA_DIRECTGATHER=1 \
  -DP10DC_RANKFORMULA_DIRECTGATHER_DEPTHMAJOR="$DEPTHMAJOR" \
  -DP10DC_RANKFORMULA_DIRECTGATHER_FORCE7="$FORCE7" \
  -DP10DC_RANKFORMULA_MLP_WINDOW4="$MLP_WINDOW4" \
  -DP10DC_RANKFORMULA_PREFETCH_NEXT="$PREFETCH_NEXT" \
  -DP10DC_RANKFORMULA_ABSTRACT_SELECT8=1 \
  -DP10DC_RANKFORMULA_ABSTRACT_DEPTH4=1 \
  -DP10DC_RANKFORMULA_ABSTRACT_SRCPACK10=1 \
  -DP10DC_RANKFORMULA_GATHER_MLP=1 \
  -DP10DC_WARPSTRIPED_COL_ILP="$COL_ILP" \
  -DP10DC_RANKCHUNK32_ONESHFL=1 -DP10DC_RANKCHUNK32_FUSED16=0 -DP10DC_RANKCHUNK32_BYTEPACK=0 \
  -DP10DC_RANKCHUNK32_ALIGN32=0 -DP10DC_RANKCHUNK32_BLOCK64=0 -DP10DC_RANKDELTA8_FUSED13=1 \
  -DP10DC_RANKFORMULA_SPARSE_BASE=1 -DP10DC_RANKFORMULA_RAWCODE=1 -DP10DC_RANKFORMULA_INLINE_CROSS=1 \
  -DP10DC_RANKFORMULA_BASE_DELTA=0 -DP10DC_RANKFORMULA_SLOTMETA=0 -DP10DC_RANKFORMULA_SLOTROW32=0 \
  "$SRC" -o "$BIN"

echo "built $BIN (directmap=1 directgather=1 depthmajor=$DEPTHMAJOR force7=$FORCE7 mlp_window4=$MLP_WINDOW4 prefetch_next=$PREFETCH_NEXT gather_mlp=1 group61=1 block=16 col_ilp=$COL_ILP pm_accum=$PM_ACCUM maxrregcount=$MAXRREGCOUNT transpose=$TRANSPOSE_MODE)" >&2
echo "run example:" >&2
echo "  BUCKET_THREADS=256 BUCKET_GRID_X=32 BUCKET_GRID_Y=8 $BIN $N <target_mib> <max_window> 8 <mod>" >&2
