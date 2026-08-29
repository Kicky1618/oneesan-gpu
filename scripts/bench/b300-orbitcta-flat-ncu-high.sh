#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-sm_103}"; PM_ACCUM="${PM_ACCUM:-1}"; THREADS="${BUCKET_THREADS:-256}"
LOW_GX="${BUCKET_LOW_GRID_X:-16}"; LOW_GY="${BUCKET_LOW_GRID_Y:-8}"
DEVICE="${DEVICE:-0}"; LAUNCH_SKIP="${LAUNCH_SKIP:-0}"; LAUNCH_COUNT="${LAUNCH_COUNT:-1}"
SPARSE64="${DIRECTGATHER_SPARSE64:-1}"; SPARSE64_WARP_INDEX="${SPARSE64_WARP_INDEX:-0}"
COL_ILP="${ORBITCTA_COL_ILP:-2}"; CPASYNC_PAIR="${CPASYNC_PAIR:-0}"
DYNAMIC="${ORBITCTA_FLAT_DYNAMIC:-1}"; DYNAMIC_BATCH="${ORBITCTA_FLAT_DYNAMIC_BATCH:-1}"
DYNAMIC_FUSE_PREP="${ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP:-1}"
DYNAMIC_ADAPTIVE_WAVES="${ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES:-0}"
DYNAMIC_PIPE2="${ORBITCTA_FLAT_DYNAMIC_PIPE2:-0}"
PRECTX_FLAT_BID="${PRECTX_FLAT_BID:-1}"; PRECTX_FLAT_BID_FUSED="${PRECTX_FLAT_BID_FUSED:-0}"
FLAT_PER_SM="${BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM:-}"; FLAT_BLOCKS="${BUCKET_ORBITCTA_FLAT_BLOCKS:-}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_orbitcta_flat_ncu_n${N}}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_orbitcta_flat_ncu_n${N}_wi${SPARSE64_WARP_INDEX}_dyn${DYNAMIC}_b${DYNAMIC_BATCH}_p2${DYNAMIC_PIPE2}}"
mkdir -p "$(dirname "$PREFIX")"

for c in ncu nvcc nvidia-smi; do command -v "$c" >/dev/null || { echo "$c required" >&2; exit 2; }; done
for x in PM_ACCUM SPARSE64 SPARSE64_WARP_INDEX CPASYNC_PAIR DYNAMIC DYNAMIC_FUSE_PREP DYNAMIC_PIPE2 PRECTX_FLAT_BID PRECTX_FLAT_BID_FUSED; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
case "$COL_ILP" in 2|4) ;; *) echo 'ORBITCTA_COL_ILP must be 2 or 4' >&2; exit 2;; esac
case "$DYNAMIC_BATCH" in 1|2|4|8|16) ;; *) echo 'dynamic batch must be 1,2,4,8,16' >&2; exit 2;; esac
case "$DYNAMIC_ADAPTIVE_WAVES" in 0|1|2|4) ;; *) echo 'adaptive waves must be 0,1,2,4' >&2; exit 2;; esac
[[ "$SPARSE64_WARP_INDEX" == 0 || "$SPARSE64" == 1 ]] || { echo 'warp index requires sparse64' >&2; exit 2; }
if [[ "$DYNAMIC" == 0 ]]; then
  [[ "$DYNAMIC_BATCH" == 1 && "$DYNAMIC_FUSE_PREP" == 0 && "$DYNAMIC_ADAPTIVE_WAVES" == 0 && "$DYNAMIC_PIPE2" == 0 ]] || { echo 'static flat requires dynamic extras disabled' >&2; exit 2; }
else
  if [[ "$DYNAMIC_PIPE2" == 1 ]]; then
    [[ "$DYNAMIC_FUSE_PREP" == 0 ]] || { echo 'pipe2 replaces fuse-lease-prep; set DYNAMIC_FUSE_PREP=0' >&2; exit 2; }
  fi
fi
if (( DYNAMIC_ADAPTIVE_WAVES != 0 )); then (( DYNAMIC_BATCH > 1 )) || { echo 'adaptive waves require batch>1' >&2; exit 2; }; fi
[[ "$PRECTX_FLAT_BID_FUSED" == 0 || "$PRECTX_FLAT_BID" == 1 ]] || { echo 'fused prectx requires flat-bid' >&2; exit 2; }
[[ -z "$FLAT_PER_SM" || -z "$FLAT_BLOCKS" ]] || { echo 'set at most one flat pool override' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }

PRE_F=0; PRE_R=0; PRE_C=0
if [[ "$PRECTX_FLAT_BID" == 1 ]]; then PRE_F=1; PRE_R=1; PRE_C=1; fi
N="$N" ARCH="$ARCH" OUT="$BIN" PM_ACCUM="$PM_ACCUM" \
DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$SPARSE64" SPARSE64_WARP_INDEX="$SPARSE64_WARP_INDEX" DIRECTGATHER_SORT_RANKS=0 \
RANKFORMULA_MLP_WINDOW4=1 PAIR_MLP=1 CPASYNC_PAIR="$CPASYNC_PAIR" \
ORBITCTA_FLAT=1 ORBITCTA_FLAT_CHUNK=1 ORBITCTA_FLAT_DYNAMIC="$DYNAMIC" \
ORBITCTA_FLAT_DYNAMIC_BATCH="$DYNAMIC_BATCH" ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP="$DYNAMIC_FUSE_PREP" \
ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES="$DYNAMIC_ADAPTIVE_WAVES" ORBITCTA_FLAT_DYNAMIC_PIPE2="$DYNAMIC_PIPE2" ORBITCTA_COL_ILP="$COL_ILP" \
QUAD_MLP=0 QUAD_OVERLAP_LOCAL=0 QUAD_LOCAL_DIRECT_MAX=0 QUAD_SPARSE_DESC_MLP=0 QUAD_OVERLAP_BYPASS_LOCAL0=0 QUAD_CPASYNC_PREFETCH_BYTES=0 QUAD_CPASYNC_GROUP_COLS=1 \
PRECTX_FORWARD="$PRE_F" PRECTX_REVERSE="$PRE_R" PRECTX_COMPACT="$PRE_C" PRECTX_FLAT_BID="$PRECTX_FLAT_BID" PRECTX_FLAT_BID_FUSED="$PRECTX_FLAT_BID_FUSED" PRECTX_WARPCOOP=0 \
PTXAS_VERBOSE=1 bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" \
  >"${PREFIX}.build.out" 2>"${PREFIX}.build.err"

grep -q "sparse64=$SPARSE64 sparse64_warp_index=$SPARSE64_WARP_INDEX" "${PREFIX}.build.err" || { echo 'build sparse64 marker mismatch' >&2; exit 4; }
grep -q "flat_dynamic=$DYNAMIC flat_dynamic_batch=$DYNAMIC_BATCH" "${PREFIX}.build.err" || { echo 'build scheduler marker mismatch' >&2; exit 4; }
grep -q "flat_dynamic_pipe2=$DYNAMIC_PIPE2" "${PREFIX}.build.err" || { echo 'build pipe2 marker mismatch' >&2; exit 4; }

RUNENV=(BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY")
[[ -z "$FLAT_PER_SM" ]] || RUNENV+=(BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$FLAT_PER_SM")
[[ -z "$FLAT_BLOCKS" ]] || RUNENV+=(BUCKET_ORBITCTA_FLAT_BLOCKS="$FLAT_BLOCKS")

# Select forward/reverse HIGH worker kernels only. This catches static,
# compact-prectx, dynamic and pipe2 variants while excluding queue reset.
KREGEX='regex:.*bucket_.*high.*orbitcta_flat.*kernel.*'

env "${RUNENV[@]}" \
  ncu --devices "$DEVICE" --filter-mode per-gpu \
      --kernel-name-base function \
      --kernel-name "$KREGEX" \
      --launch-skip "$LAUNCH_SKIP" --launch-count "$LAUNCH_COUNT" \
      --section SpeedOfLight \
      --section Occupancy \
      --section WarpStateStats \
      --section MemoryWorkloadAnalysis \
      --page details --force-overwrite -o "$PREFIX" \
      "$BIN" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" \
      >"${PREFIX}.out" 2>"${PREFIX}.err"

echo "Nsight report: ${PREFIX}.ncu-rep" >&2
echo "config: flat=1 dynamic=$DYNAMIC batch=$DYNAMIC_BATCH adaptive_waves=$DYNAMIC_ADAPTIVE_WAVES fuse_lease_prep=$DYNAMIC_FUSE_PREP pipe2=$DYNAMIC_PIPE2 sparse64=$SPARSE64 warp_index=$SPARSE64_WARP_INDEX prectx_flat_bid=$PRECTX_FLAT_BID prectx_fused=$PRECTX_FLAT_BID_FUSED ilp=$COL_ILP cpasync_pair=$CPASYNC_PAIR" >&2
echo 'inspect: DRAM throughput, L2 throughput/hit rate, achieved occupancy, issue-active, Long Scoreboard, memory sectors/request, and SM throughput' >&2
