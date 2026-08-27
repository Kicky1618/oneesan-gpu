#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"; if (( $# > 0 )); then shift; fi
NGPU="${NGPU:-8}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"; HIGH_CTX="${HIGH_CTX:-thread}"; DEPTHCODE_DECODE_LOAD="${DEPTHCODE_DECODE_LOAD:-global}"; PM_ACCUM="${PM_ACCUM:-0}"; TERNARY_KEY4="${TERNARY_KEY4:-1}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"; BUCKET_GRID_X="${BUCKET_GRID_X:-16}"; BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
BUCKET_TRANSPOSE_CHUNK_MIB="${BUCKET_TRANSPOSE_CHUNK_MIB:-1024}"; BUCKET_RESERVE_MIB="${BUCKET_RESERVE_MIB:-8192}"
case "$TRANSPOSE_MODE" in sync|events|pipeline) ;; *) echo "TRANSPOSE_MODE must be sync, events, or pipeline" >&2; exit 2;; esac
case "$HIGH_CTX" in thread|resolved|warp|warpstriped) ;; *) echo "HIGH_CTX must be thread, resolved, warp, or warpstriped" >&2; exit 2;; esac
case "$DEPTHCODE_DECODE_LOAD" in global|ldg) ;; *) echo "DEPTHCODE_DECODE_LOAD must be global or ldg" >&2; exit 2;; esac
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then echo "PM_ACCUM must be 0 or 1" >&2; exit 2; fi
if [[ "$TERNARY_KEY4" != 0 && "$TERNARY_KEY4" != 1 ]]; then echo "TERNARY_KEY4 must be 0 or 1" >&2; exit 2; fi
if (( BUCKET_THREADS < 1 || BUCKET_THREADS > 1024 )); then echo "BUCKET_THREADS must be in [1,1024]" >&2; exit 2; fi
if (( BUCKET_GRID_X < 1 || BUCKET_GRID_Y < 1 )); then echo "BUCKET_GRID_X/Y must be >=1" >&2; exit 2; fi
if [[ "$HIGH_CTX" == warpstriped ]] && (( BUCKET_THREADS < 32 || BUCKET_THREADS % 32 != 0 )); then echo "warpstriped requires BUCKET_THREADS multiple of 32 in [32,1024]" >&2; exit 2; fi
if (( NGPU != 8 )); then echo "pattern10 depthcode graph backend currently requires NGPU=8" >&2; exit 2; fi

SUFFIX="_payload_${HIGH_CTX}_${TRANSPOSE_MODE}"; [[ "$DEPTHCODE_DECODE_LOAD" == ldg ]] && SUFFIX="${SUFFIX}_ldg"; [[ "$PM_ACCUM" == 1 ]] && SUFFIX="${SUFFIX}_pm"; [[ "$TERNARY_KEY4" == 0 ]] && SUFFIX="${SUFFIX}_keyscalar"
BIN="${BIN:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_bucket_snake_onepass_pattern10_depthcode_graph_batch${SUFFIX}_n${N}}"
WORK_TAG="pattern10_depthcode_graph_payload_${HIGH_CTX}_${TRANSPOSE_MODE}"; [[ "$DEPTHCODE_DECODE_LOAD" == ldg ]] && WORK_TAG="${WORK_TAG}_ldg"; [[ "$PM_ACCUM" == 1 ]] && WORK_TAG="${WORK_TAG}_pm"; if [[ "$TERNARY_KEY4" == 0 ]]; then WORK_TAG="${WORK_TAG}_keyscalar"; else WORK_TAG="${WORK_TAG}_key4"; fi; WORK_TAG="${WORK_TAG}_t${BUCKET_THREADS}_gx${BUCKET_GRID_X}_gy${BUCKET_GRID_Y}"
WORK_DIR="${WORK_DIR:-$ONEESAN_ROOT/work/b300_bucket_snake_${WORK_TAG}_exact_n${N}}"

if ! command -v nvidia-smi >/dev/null; then echo "nvidia-smi not found" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; if (( visible < NGPU )); then echo "requested $NGPU GPUs, but only $visible are visible" >&2; exit 2; fi
if [[ ! -x "$BIN" ]]; then
  echo "$BIN not found; building pattern10 depthcode Graph batch n=$N decode=payload high_ctx=$HIGH_CTX decode_load=$DEPTHCODE_DECODE_LOAD transpose=$TRANSPOSE_MODE pm=$PM_ACCUM ternary_key4=$TERNARY_KEY4 threads=$BUCKET_THREADS gx=$BUCKET_GRID_X gy=$BUCKET_GRID_Y" >&2
  N="$N" OUT="$BIN" HIGH_CTX="$HIGH_CTX" DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh"
fi

export BUCKET_TRANSPOSE_CHUNK_MIB BUCKET_RESERVE_MIB BUCKET_THREADS BUCKET_GRID_X BUCKET_GRID_Y
exec python3 "$ONEESAN_ROOT/scripts/solve/solve_b300_exact_batch.py" "$N" \
  --binary "$BIN" --work-dir "$WORK_DIR" --target-mib "$TARGET_MIB" --max-window "$MAX_WINDOW" --gpus "$NGPU" "$@"
