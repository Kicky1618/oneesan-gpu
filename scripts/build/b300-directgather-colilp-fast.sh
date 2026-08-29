#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; W=$((N + 1)); ARCH="${ARCH:-native}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"; HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
COL_ILP="${COL_ILP:-2}"; PM_ACCUM="${PM_ACCUM:-1}"
DEPTHMAJOR="${DEPTHMAJOR:-1}"; FORCE7="${FORCE7:-0}"; MLP_WINDOW4="${MLP_WINDOW4:-0}"
PAIR_MLP="${PAIR_MLP:-0}"; QUAD_MLP="${QUAD_MLP:-0}"
CPASYNC_PAIR="${CPASYNC_PAIR:-0}"; CPASYNC_LOCAL_PAIR="${CPASYNC_LOCAL_PAIR:-0}"; SORTED="${SORTED:-0}"
PRECTX_FORWARD="${PRECTX_FORWARD:-0}"; PRECTX_REVERSE="${PRECTX_REVERSE:-0}"
PREFETCH_NEXT="${PREFETCH_NEXT:-0}"; DIRECTGATHER64="${DIRECTGATHER64:-0}"; DIRECTGATHER_SPARSE64="${DIRECTGATHER_SPARSE64:-0}"
MAXRREGCOUNT="${MAXRREGCOUNT:-0}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
OUT="${OUT:-oneesan_cuda_gridfp_b300_directgather_colilp${COL_ILP}_pair${PAIR_MLP}_quad${QUAD_MLP}_cpa${CPASYNC_PAIR}_cpalocal${CPASYNC_LOCAL_PAIR}_sort${SORTED}_prectxf${PRECTX_FORWARD}r${PRECTX_REVERSE}_dg64${DIRECTGATHER64}_sp64${DIRECTGATHER_SPARSE64}_n${N}}"

if (( LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W || LOW_LUT_K > 14 || HIGH_LUT_K > 14 )); then
  echo "invalid LOW/HIGH split W=$W low=$LOW_LUT_K high=$HIGH_LUT_K" >&2; exit 2
fi
case "$COL_ILP" in 1|2|4) ;; *) echo "COL_ILP must be 1, 2, or 4" >&2; exit 2;; esac
for x in PM_ACCUM DEPTHMAJOR FORCE7 MLP_WINDOW4 PAIR_MLP QUAD_MLP CPASYNC_PAIR CPASYNC_LOCAL_PAIR SORTED PRECTX_FORWARD PRECTX_REVERSE PREFETCH_NEXT DIRECTGATHER64 DIRECTGATHER_SPARSE64 PTXAS_VERBOSE; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
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
  [[ "$CPASYNC_PAIR" == 0 ]] || { echo "QUAD_MLP register experiment currently requires CPASYNC_PAIR=0" >&2; exit 2; }
  [[ "$CPASYNC_LOCAL_PAIR" == 0 ]] || { echo "QUAD_MLP currently requires CPASYNC_LOCAL_PAIR=0" >&2; exit 2; }
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
  -DP10DC_RANKFORMULA_CPASYNC_PAIR="$CPASYNC_PAIR" \
  -DP10DC_RANKFORMULA_CPASYNC_LOCAL_PAIR="$CPASYNC_LOCAL_PAIR" \
  -DP10DC_RANKFORMULA_PRECTX_FORWARD="$PRECTX_FORWARD" \
  -DP10DC_RANKFORMULA_PRECTX_REVERSE="$PRECTX_REVERSE" \
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

echo "built $BIN (directmap=1 directgather=1 directgather64=$DIRECTGATHER64 sparse64=$DIRECTGATHER_SPARSE64 depthmajor=$DEPTHMAJOR sorted=$SORTED prectx_forward=$PRECTX_FORWARD prectx_reverse=$PRECTX_REVERSE force7=$FORCE7 mlp_window4=$MLP_WINDOW4 pair_mlp=$PAIR_MLP quad_mlp=$QUAD_MLP cpasync_pair=$CPASYNC_PAIR cpasync_local_pair=$CPASYNC_LOCAL_PAIR prefetch_next=$PREFETCH_NEXT gather_mlp=1 group61=1 block=16 col_ilp=$COL_ILP pm_accum=$PM_ACCUM maxrregcount=$MAXRREGCOUNT transpose=$TRANSPOSE_MODE)" >&2
echo "run example:" >&2
echo "  BUCKET_THREADS=256 BUCKET_GRID_X=32 BUCKET_GRID_Y=8 $BIN $N <target_mib> <max_window> 8 <mod>" >&2
