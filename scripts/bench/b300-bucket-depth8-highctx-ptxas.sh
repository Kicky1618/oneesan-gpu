#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"
ARCH="${ARCH:-native}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-sync}"
PM_ACCUM="${PM_ACCUM:-0}"
TERNARY_KEY4="${TERNARY_KEY4:-1}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300_bucket_depth8_highctx_ptxas_n${N}}"

if ! command -v nvcc >/dev/null; then
  echo "nvcc not found" >&2
  exit 2
fi
case "$TRANSPOSE_MODE" in
  sync|events|pipeline) ;;
  *) echo "TRANSPOSE_MODE must be sync, events, or pipeline" >&2; exit 2 ;;
esac
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then
  echo "PM_ACCUM must be 0 or 1" >&2
  exit 2
fi
if [[ "$TERNARY_KEY4" != 0 && "$TERNARY_KEY4" != 1 ]]; then
  echo "TERNARY_KEY4 must be 0 or 1" >&2
  exit 2
fi

mkdir -p "$LOGDIR"

for high_ctx in thread shared warp; do
  bin="$ONEESAN_BUILD_DIR/ptxas_depth8_${high_ctx}_${TRANSPOSE_MODE}_n${N}"
  log="$LOGDIR/${high_ctx}.log"
  echo "=== ptxas $high_ctx ===" >&2
  N="$N" \
  ARCH="$ARCH" \
  OUT="$bin" \
  HIGH_CTX="$high_ctx" \
  TRANSPOSE_MODE="$TRANSPOSE_MODE" \
  PM_ACCUM="$PM_ACCUM" \
  TERNARY_KEY4="$TERNARY_KEY4" \
  PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depth8-graph-batch.sh" \
      >"$log" 2>&1

done

for high_ctx in thread shared warp; do
  log="$LOGDIR/${high_ctx}.log"
  echo
  echo "===== $high_ctx ====="
  grep -E \
    'Compiling entry function|Function properties|Used [0-9]+ registers|bytes smem|spill stores|spill loads' \
    "$log" || true
done

echo >&2
echo "ptxas HIGH-context comparison complete: $LOGDIR" >&2
