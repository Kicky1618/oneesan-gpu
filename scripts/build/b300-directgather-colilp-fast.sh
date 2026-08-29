#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; W=$((N + 1)); ARCH="${ARCH:-native}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"; HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
COL_ILP="${COL_ILP:-2}"; PM_ACCUM="${PM_ACCUM:-1}"
DEPTHMAJOR="${DEPTHMAJOR:-1}"; FORCE7="${FORCE7:-0}"; MLP_WINDOW4="${MLP_WINDOW4:-0}"
PAIR_MLP="${PAIR_MLP:-0}"; QUAD_MLP="${QUAD_MLP:-0}"; QUAD_LOCAL_DIRECT_MAX="${QUAD_LOCAL_DIRECT_MAX:-0}"; QUAD_OVERLAP_LOCAL="${QUAD_OVERLAP_LOCAL:-0}"; QUAD_SPARSE_DESC_MLP="${QUAD_SPARSE_DESC_MLP:-0}"
CPASYNC_PAIR="${CPASYNC_PAIR:-0}"; CPASYNC_LOCAL_PAIR="${CPASYNC_LOCAL_PAIR:-0}"; CPASYNC_OVERLAP_LOCAL_PAIR="${CPASYNC_OVERLAP_LOCAL_PAIR:-0}"; CPASYNC_OVERLAP_LOCAL_PIPE2="${CPASYNC_OVERLAP_LOCAL_PIPE2:-0}"; OVERLAP_LOCAL_CG="${OVERLAP_LOCAL_CG:-0}"; SORTED="${SORTED:-0}"
PRECTX_FORWARD="${PRECTX_FORWARD:-0}"; PRECTX_REVERSE="${PRECTX_REVERSE:-0}"; PRECTX_COMPACT="${PRECTX_COMPACT:-0}"
PREFETCH_NEXT="${PREFETCH_NEXT:-0}"; DIRECTGATHER64="${DIRECTGATHER64:-0}"; DIRECTGATHER_SPARSE64="${DIRECTGATHER_SPARSE64:-0}"
HIGH_PLAN_PROFILE="${HIGH_PLAN_PROFILE:-0}"
MAXRREGCOUNT="${MAXRREGCOUNT:-0}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
OUT="${OUT:-oneesan_cuda_gridfp_b300_directgather_colilp${COL_ILP}_pair${PAIR_MLP}_quad${QUAD_MLP}_qld${QUAD_LOCAL_DIRECT_MAX}_qol${QUAD_OVERLAP_LOCAL}_qsd${QUAD_SPARSE_DESC_MLP}_cpa${CPASYNC_PAIR}_cpalocal${CPASYNC_LOCAL_PAIR}_overlap${CPASYNC_OVERLAP_LOCAL_PAIR}_pipe2${CPASYNC_OVERLAP_LOCAL_PIPE2}_cg${OVERLAP_LOCAL_CG}_sort${SORTED}_prectxf${PRECTX_FORWARD}r${PRECTX_REVERSE}c${PRECTX_COMPACT}_dg64${DIRECTGATHER64}_sp64${DIRECTGATHER_SPARSE64}_prof${HIGH_PLAN_PROFILE}_n${N}}"

if (( LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W || LOW_LUT_K > 14 || HIGH_LUT_K > 14 )); then
  echo "invalid LOW/HIGH split W=$W low=$LOW_LUT_K high=$HIGH_LUT_K" >&2; exit 2
fi
case "$COL_ILP" in 1|2|4) ;; *) echo "COL_ILP must be 1, 2, or 4" >&2; exit 2;; esac
for x in PM_ACCUM DEPTHMAJOR FORCE7 MLP_WINDOW4 PAIR_MLP QUAD_MLP QUAD_OVERLAP_LOCAL QUAD_SPARSE_DESC_MLP CPASYNC_PAIR CPASYNC_LOCAL_PAIR CPASYNC_OVERLAP_LOCAL_PAIR CPASYNC_OVERLAP_LOCAL_PIPE2 OVERLAP_LOCAL_CG SORTED PRECTX_FORWARD PRECTX_REVERSE PRECTX_COMPACT PREFETCH_NEXT DIRECTGATHER64 DIRECTGATHER_SPARSE64 HIGH_PLAN_PROFILE PTXAS_VERBOSE; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
[[ "$QUAD_LOCAL_DIRECT_MAX" =~ ^[0-9]+$ ]] || { echo "QUAD_LOCAL_DIRECT_MAX must be integer 0..8" >&2; exit 2; }
(( QUAD_LOCAL_DIRECT_MAX <= 8 )) || { echo "QUAD_LOCAL_DIRECT_MAX must be 0..8" >&2; exit 2; }
if [[ "$PRECTX_COMPACT" == 1 ]]; then
  [[ "$PRECTX_FORWARD" == 1 || "$PRECTX_REVERSE" == 1 ]] || { echo "PRECTX_COMPACT requires PRECTX_FORWARD=1 and/or PRECTX_REVERSE=1" >&2; exit 2; }
fi
if [[ "$FORCE7" == 1 && "$MLP_WINDOW4" == 1 ]]; then echo "FORCE7 and MLP_WINDOW4 are mutually exclusive" >&2; exit 2; fi
if [[ "$PAIR_MLP" == 1 ]]; then
  [[ "$DEPTHMAJOR" == 1 ]] || { echo "PAIR_MLP requires DEPTHMAJOR=1" >&2; exit 2; }
  [[ "$MLP_WINDOW4" == 1 ]] || { echo "PAIR_MLP requires MLP_WINDOW4=1" >&2; exit 2; }
  [[ "$COL_ILP" == 2 || "$COL_ILP" == 4 ]] || { echo "PAIR_MLP requires COL_ILP=2 or 4" >&2; exit 2; }
  [[ "$FORCE7" == 0 ]] || { echo "PAIR_MLP does not combine with FORCE7" >&2; exit 2; }
fi
if [[ "$QUAD_MLP" == 1 ]]; then
  [[ "$DIRECTGATHER64" == 1 ]] || { echo "QUAD_MLP requires DIRECTGATHER64=1" >&2; exit 2; }
  [[ "$PAIR_MLP" == 1 ]] || { echo "QUAD_MLP requires PAIR_MLP=1 for tail fallback" >&2; exit 2; }
  [[ "$COL_ILP" == 4 ]] || { echo "QUAD_MLP requires COL_ILP=4" >&2; exit 2; }
  [[ "$MLP_WINDOW4" == 1 ]] || { echo "QUAD_MLP requires MLP_WINDOW4=1" >&2; exit 2; }
  [[ "$CPASYNC_LOCAL_PAIR" == 0 ]] || { echo "QUAD_MLP currently requires CPASYNC_LOCAL_PAIR=0" >&2; exit 2; }
  [[ "$CPASYNC_OVERLAP_LOCAL_PAIR" == 0 ]] || { echo "QUAD_MLP currently requires CPASYNC_OVERLAP_LOCAL_PAIR=0" >&2; exit 2; }
  [[ "$CPASYNC_OVERLAP_LOCAL_PIPE2" == 0 ]] || { echo "QUAD_MLP currently requires CPASYNC_OVERLAP_LOCAL_PIPE2=0" >&2; exit 2; }
else
  (( QUAD_LOCAL_DIRECT_MAX == 0 )) || { echo "QUAD_LOCAL_DIRECT_MAX requires QUAD_MLP=1" >&2; exit 2; }
  [[ "$QUAD_OVERLAP_LOCAL" == 0 ]] || { echo "QUAD_OVERLAP_LOCAL requires QUAD_MLP=1" >&2; exit 2; }
fi
if [[ "$QUAD_OVERLAP_LOCAL" == 1 ]]; then
  [[ "$CPASYNC_PAIR" == 1 ]] || { echo "QUAD_OVERLAP_LOCAL requires CPASYNC_PAIR=1" >&2; exit 2; }
  [[ "$DIRECTGATHER64" == 1 ]] || { echo "QUAD_OVERLAP_LOCAL requires DIRECTGATHER64=1" >&2; exit 2; }
  (( QUAD_LOCAL_DIRECT_MAX == 0 )) || { echo "QUAD_OVERLAP_LOCAL already uses direct local loads; set QUAD_LOCAL_DIRECT_MAX=0" >&2; exit 2; }
fi
if [[ "$QUAD_SPARSE_DESC_MLP" == 1 ]]; then
  [[ "$QUAD_MLP" == 1 && "$QUAD_OVERLAP_LOCAL" == 1 ]] || { echo "QUAD_SPARSE_DESC_MLP requires QUAD_MLP=1 and QUAD_OVERLAP_LOCAL=1" >&2; exit 2; }
  [[ "$DIRECTGATHER_SPARSE64" == 1 ]] || { echo "QUAD_SPARSE_DESC_MLP requires DIRECTGATHER_SPARSE64=1" >&2; exit 2; }
fi
if [[ "$CPASYNC_PAIR" == 1 ]]; then
  [[ "$PAIR_MLP" == 1 ]] || { echo "CPASYNC_PAIR requires PAIR_MLP=1" >&2; exit 2; }
  [[ "$DEPTHMAJOR" == 1 ]] || { echo "CPASYNC_PAIR requires DEPTHMAJOR=1" >&2; exit 2; }
  [[ "$MLP_WINDOW4" == 1 ]] || { echo "CPASYNC_PAIR requires MLP_WINDOW4=1" >&2; exit 2; }
  [[ "$FORCE7" == 0 ]] || { echo "CPASYNC_PAIR does not combine with FORCE7" >&2; exit 2; }
  [[ "$PREFETCH_NEXT" == 0 ]] || { echo "CPASYNC_PAIR prefetch combination is intentionally isolated; use PREFETCH_NEXT=0" >&2; exit 2; }
fi
if [[ "$CPASYNC_LOCAL_PAIR" == 1 ]]; then
  [[ "$CPASYNC_PAIR" == 1 ]] || { echo "CPASYNC_LOCAL_PAIR requires CPASYNC_PAIR=1" >&2; exit 2; }
  [[ "$PAIR_MLP" == 1 ]] || { echo "CPASYNC_LOCAL_PAIR requires PAIR_MLP=1" >&2; exit 2; }
fi
if [[ "$CPASYNC_OVERLAP_LOCAL_PAIR" == 1 ]]; then
  [[ "$CPASYNC_PAIR" == 1 ]] || { echo "CPASYNC_OVERLAP_LOCAL_PAIR requires CPASYNC_PAIR=1" >&2; exit 2; }
  [[ "$PAIR_MLP" == 1 ]] || { echo "CPASYNC_OVERLAP_LOCAL_PAIR requires PAIR_MLP=1" >&2; exit 2; }
  [[ "$DIRECTGATHER64" == 1 ]] || { echo "CPASYNC_OVERLAP_LOCAL_PAIR requires DIRECTGATHER64=1" >&2; exit 2; }
  [[ "$CPASYNC_LOCAL_PAIR" == 0 ]] || { echo "CPASYNC_OVERLAP_LOCAL_PAIR is isolated from CPASYNC_LOCAL_PAIR" >&2; exit 2; }
fi
if [[ "$CPASYNC_OVERLAP_LOCAL_PIPE2" == 1 ]]; then
  [[ "$CPASYNC_OVERLAP_LOCAL_PAIR" == 1 ]] || { echo "CPASYNC_OVERLAP_LOCAL_PIPE2 requires CPASYNC_OVERLAP_LOCAL_PAIR=1" >&2; exit 2; }
  [[ "$QUAD_MLP" == 0 ]] || { echo "CPASYNC_OVERLAP_LOCAL_PIPE2 is isolated from QUAD_MLP" >&2; exit 2; }
fi
if [[ "$OVERLAP_LOCAL_CG" == 1 && "$CPASYNC_OVERLAP_LOCAL_PAIR" != 1 ]]; then
  echo "OVERLAP_LOCAL_CG requires CPASYNC_OVERLAP_LOCAL_PAIR=1" >&2; exit 2
fi
if [[ "$SORTED" == 1 ]]; then
  [[ "$DEPTHMAJOR" == 1 ]] || { echo "SORTED currently targets DEPTHMAJOR=1" >&2; exit 2; }
fi
if [[ "$PRECTX_FORWARD" == 1 || "$PRECTX_REVERSE" == 1 ]]; then
  [[ "$DEPTHMAJOR" == 1 ]] || { echo "PRECTX currently targets DEPTHMAJOR=1" >&2; exit 2; }
fi
if [[ "$DIRECTGATHER64" == 1 ]]; then
  [[ "$DEPTHMAJOR" == 1 ]] || { echo "DIRECTGATHER64 requires DEPTHMAJOR=1 for this B300 path" >&2; exit 2; }
  [[ "$FORCE7" == 0 ]] || { echo "DIRECTGATHER64 does not combine with FORCE7" >&2; exit 2; }
  [[ "$PREFETCH_NEXT" == 0 ]] || { echo "DIRECTGATHER64 prefetch path is intentionally isolated; use PREFETCH_NEXT=0" >&2; exit 2; }
fi
if [[ "$DIRECTGATHER_SPARSE64" == 1 ]]; then
  [[ "$DIRECTGATHER64" == 1 ]] || { echo "DIRECTGATHER_SPARSE64 requires DIRECTGATHER64=1" >&2; exit 2; }
  [[ "$DEPTHMAJOR" == 1 ]] || { echo "DIRECTGATHER_SPARSE64 requires DEPTHMAJOR=1" >&2; exit 2; }
fi
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] || { echo "$MAXRREGCOUNT must be non-negative integer" >&2; exit 2; }
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
  -DP10DC_RANKFORMULA_DIRECTGATHER_SORTED="$SORTED" \
  -DP10DC_RANKFORMULA_DIRECTGATHER_FORCE7="$FORCE7" \
  -DP10DC_RANKFORMULA_DIRECTGATHER64="$DIRECTGATHER64" \
  -DP10DC_RANKFORMULA_DIRECTGATHER_SPARSE64="$DIRECTGATHER_SPARSE64" \
  -DP10DC_RANKFORMULA_MLP_WINDOW4="$MLP_WINDOW4" \
  -DP10DC_RANKFORMULA_PAIR_MLP="$PAIR_MLP" \
  -DP10DC_RANKFORMULA_QUAD_MLP="$QUAD_MLP" \
  -DP10DC_RANKFORMULA_QUAD_LOCAL_DIRECT_MAX="$QUAD_LOCAL_DIRECT_MAX" \
  -DP10DC_RANKFORMULA_QUAD_OVERLAP_LOCAL="$QUAD_OVERLAP_LOCAL" \
  -DP10DC_RANKFORMULA_QUAD_SPARSE_DESC_MLP="$QUAD_SPARSE_DESC_MLP" \
  -DP10DC_RANKFORMULA_CPASYNC_PAIR="$CPASYNC_PAIR" \
  -DP10DC_RANKFORMULA_CPASYNC_LOCAL_PAIR="$CPASYNC_LOCAL_PAIR" \
  -DP10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PAIR="$CPASYNC_OVERLAP_LOCAL_PAIR" \
  -DP10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PIPE2="$CPASYNC_OVERLAP_LOCAL_PIPE2" \
  -DP10DC_RANKFORMULA_OVERLAP_LOCAL_CG="$OVERLAP_LOCAL_CG" \
  -DP10DC_RANKFORMULA_HIGH_PLAN_PROFILE="$HIGH_PLAN_PROFILE" \
  -DP10DC_RANKFORMULA_PRECTX_FORWARD="$PRECTX_FORWARD" \
  -DP10DC_RANKFORMULA_PRECTX_REVERSE="$PRECTX_REVERSE" \
  -DP10DC_RANKFORMULA_PRECTX_COMPACT="$PRECTX_COMPACT" \
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

echo "built $BIN (directmap=1 directgather=1 directgather64=$DIRECTGATHER64 sparse64=$DIRECTGATHER_SPARSE64 depthmajor=$DEPTHMAJOR sorted=$SORTED prectx_forward=$PRECTX_FORWARD prectx_reverse=$PRECTX_REVERSE prectx_compact=$PRECTX_COMPACT force7=$FORCE7 mlp_window4=$MLP_WINDOW4 pair_mlp=$PAIR_MLP quad_mlp=$QUAD_MLP quad_local_direct_max=$QUAD_LOCAL_DIRECT_MAX quad_overlap_local=$QUAD_OVERLAP_LOCAL quad_sparse_desc_mlp=$QUAD_SPARSE_DESC_MLP cpasync_pair=$CPASYNC_PAIR cpasync_local_pair=$CPASYNC_LOCAL_PAIR overlap_local_pair=$CPASYNC_OVERLAP_LOCAL_PAIR overlap_local_pipe2=$CPASYNC_OVERLAP_LOCAL_PIPE2 overlap_local_cg=$OVERLAP_LOCAL_CG high_plan_profile=$HIGH_PLAN_PROFILE prefetch_next=$PREFETCH_NEXT gather_mlp=1 group61=1 block=16 col_ilp=$COL_ILP pm_accum=$PM_ACCUM maxrregcount=$MAXRREGCOUNT transpose=$TRANSPOSE_MODE)" >&2
echo "run example:" >&2
echo "  BUCKET_THREADS=256 BUCKET_GRID_X=32 BUCKET_GRID_Y=8 $BIN $N <target_mib> <max_window> 8 <mod>" >&2
