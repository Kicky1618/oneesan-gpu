#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"
W=$((N + 1))
ARCH="${ARCH:-native}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"
HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
HIGH_CTX="${HIGH_CTX:-thread}"
DEPTHCODE_DECODE_LOAD="${DEPTHCODE_DECODE_LOAD:-global}"
DEPTHCODE_SOURCE_KEY="${DEPTHCODE_SOURCE_KEY:-scalar}"
PM_ACCUM="${PM_ACCUM:-0}"
TERNARY_KEY4="${TERNARY_KEY4:-1}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-0}"

if (( LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then echo "invalid factor split" >&2; exit 2; fi
if (( LOW_LUT_K > 14 || HIGH_LUT_K > 14 )); then echo "pattern10 depthcode requires half widths <=14" >&2; exit 2; fi
case "$HIGH_CTX" in thread|resolved|warp|warpstriped) ;; *) echo "HIGH_CTX must be thread, resolved, warp, or warpstriped" >&2; exit 2;; esac
case "$DEPTHCODE_DECODE_LOAD" in global|ldg) ;; *) echo "DEPTHCODE_DECODE_LOAD must be global or ldg" >&2; exit 2;; esac
case "$DEPTHCODE_SOURCE_KEY" in scalar|key4) ;; *) echo "DEPTHCODE_SOURCE_KEY must be scalar or key4" >&2; exit 2;; esac
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then echo "PM_ACCUM must be 0 or 1" >&2; exit 2; fi
if [[ "$TERNARY_KEY4" != 0 && "$TERNARY_KEY4" != 1 ]]; then echo "TERNARY_KEY4 must be 0 or 1" >&2; exit 2; fi
if [[ "$PTXAS_VERBOSE" != 0 && "$PTXAS_VERBOSE" != 1 ]]; then echo "PTXAS_VERBOSE must be 0 or 1" >&2; exit 2; fi

base="oneesan_cuda_gridfp_b300_bucket_snake_onepass_pattern10_depthcode"
[[ "$HIGH_CTX" == resolved ]] && base="${base}_resolved"
[[ "$HIGH_CTX" == warp ]] && base="${base}_warpctx"
[[ "$HIGH_CTX" == warpstriped ]] && base="${base}_warpstriped"
case "$TRANSPOSE_MODE" in
  sync) SRC_NAME="${base}_graph_batch.cu" ;;
  events) SRC_NAME="${base}_graph_batch_events.cu" ;;
  pipeline) SRC_NAME="${base}_graph_batch_pipeline.cu" ;;
  *) echo "TRANSPOSE_MODE must be sync, events, or pipeline" >&2; exit 2 ;;
esac

P10DC_DECODE_LDG=0
[[ "$DEPTHCODE_DECODE_LOAD" == ldg ]] && P10DC_DECODE_LDG=1
P10DC_SOURCE_KEY4=0
[[ "$DEPTHCODE_SOURCE_KEY" == key4 ]] && P10DC_SOURCE_KEY4=1
SUFFIX="_payload_${HIGH_CTX}_${TRANSPOSE_MODE}"
[[ "$DEPTHCODE_DECODE_LOAD" == ldg ]] && SUFFIX="${SUFFIX}_ldg"
[[ "$DEPTHCODE_SOURCE_KEY" == key4 ]] && SUFFIX="${SUFFIX}_srckey4"
[[ "$PM_ACCUM" == 1 ]] && SUFFIX="${SUFFIX}_pm"
[[ "$TERNARY_KEY4" == 0 ]] && SUFFIX="${SUFFIX}_keyscalar"
SRC="$(repo_path "src/cuda/b300/$SRC_NAME")"
OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_bucket_snake_onepass_pattern10_depthcode_graph_batch${SUFFIX}_n${N}}")"
NVCC_EXTRA=()
[[ "$PTXAS_VERBOSE" == 1 ]] && NVCC_EXTRA+=("-Xptxas=-v")

TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  "${NVCC_EXTRA[@]}" \
  -DTARGET_W="$W" \
  -DLOW_LUT_K="$LOW_LUT_K" \
  -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DGPU_DIRECT_PM_ACCUM="$PM_ACCUM" \
  -DBKCZ_TERNARY_KEY4="$TERNARY_KEY4" \
  -DP10DC_DECODE_LDG="$P10DC_DECODE_LDG" \
  -DP10DC_SOURCE_KEY4="$P10DC_SOURCE_KEY4" \
  "$SRC" -o "$OUT"

echo "built $OUT (closure=pattern10-depthcode sidecar_bytes_per_orbit=0 temporary_depth_bytes=0 decode=payload-masks runtime_unrank=0 high_ctx=$HIGH_CTX decode_load=$DEPTHCODE_DECODE_LOAD source_key=$DEPTHCODE_SOURCE_KEY window=graph transpose=$TRANSPOSE_MODE pm_accum=$PM_ACCUM ternary_key4=$TERNARY_KEY4 ptxas_verbose=$PTXAS_VERBOSE)"
