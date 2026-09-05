#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${1:-27}"; if (( $# > 0 )); then shift; fi
NGPU="${NGPU:-8}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"; PM_ACCUM="${PM_ACCUM:-0}"; ZERO_HIGH_PLAN="${ZERO_HIGH_PLAN:-thread}"; TERNARY_KEY4="${TERNARY_KEY4:-1}"
BUCKET_TRANSPOSE_CHUNK_MIB="${BUCKET_TRANSPOSE_CHUNK_MIB:-1024}"; BUCKET_RESERVE_MIB="${BUCKET_RESERVE_MIB:-8192}"
case "$ZERO_HIGH_PLAN" in thread|shared) ;; *) echo "ZERO_HIGH_PLAN must be thread or shared" >&2; exit 2;; esac
if [[ "$TERNARY_KEY4" != 0 && "$TERNARY_KEY4" != 1 ]]; then echo "TERNARY_KEY4 must be 0 or 1" >&2; exit 2; fi
SUFFIX="_${ZERO_HIGH_PLAN}_${TRANSPOSE_MODE}"; if [[ "$PM_ACCUM" == 1 ]]; then SUFFIX="${SUFFIX}_pm"; fi; if [[ "$TERNARY_KEY4" == 0 ]]; then SUFFIX="${SUFFIX}_keyscalar"; fi
BIN="${BIN:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_bucket_snake_onepass_zero_graph_batch${SUFFIX}_n${N}}"
WORK_TAG="zero_graph_${ZERO_HIGH_PLAN}_${TRANSPOSE_MODE}"; if [[ "$PM_ACCUM" == 1 ]]; then WORK_TAG="${WORK_TAG}_pm"; fi; if [[ "$TERNARY_KEY4" == 0 ]]; then WORK_TAG="${WORK_TAG}_keyscalar"; fi
WORK_DIR="${WORK_DIR:-$ONEESAN_ROOT/work/b300_bucket_snake_${WORK_TAG}_exact_n${N}}"
if (( NGPU != 8 )); then echo "closure-zero graph snake backend currently requires NGPU=8" >&2; exit 2; fi
if ! command -v nvidia-smi >/dev/null; then echo "nvidia-smi not found" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; if (( visible < NGPU )); then echo "requested $NGPU GPUs, but only $visible are visible" >&2; exit 2; fi
if [[ ! -x "$BIN" ]]; then
  echo "$BIN not found; building closure-zero graph batch binary n=$N high_plan=$ZERO_HIGH_PLAN transpose=$TRANSPOSE_MODE pm=$PM_ACCUM ternary_key4=$TERNARY_KEY4" >&2
  N="$N" OUT="$BIN" ZERO_HIGH_PLAN="$ZERO_HIGH_PLAN" TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-zero-graph-batch.sh"
fi
export BUCKET_TRANSPOSE_CHUNK_MIB BUCKET_RESERVE_MIB
exec python3 "$ONEESAN_ROOT/scripts/solve/solve_b300_exact_batch.py" "$N" \
  --binary "$BIN" --work-dir "$WORK_DIR" --target-mib "$TARGET_MIB" --max-window "$MAX_WINDOW" --gpus "$NGPU" "$@"
