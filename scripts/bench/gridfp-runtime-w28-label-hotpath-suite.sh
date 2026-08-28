#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BLOCKS="${BLOCKS:-4096}"
THREADS="${THREADS:-256}"
ITERS="${ITERS:-128}"
REPEATS="${REPEATS:-7}"
WARMUP="${WARMUP:-2}"
CASES="${CASES:-motzkin 25 50 75 100}"
ARCH="${ARCH:-native}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
RUN_OWNER_TREE="${RUN_OWNER_TREE:-1}"
RUN_TURN_TREE="${RUN_TURN_TREE:-1}"
RUN_TURN_NONN="${RUN_TURN_NONN:-1}"

for name in RUN_OWNER_TREE RUN_TURN_TREE RUN_TURN_NONN; do
  value="${!name}"
  if [[ "$value" != 0 && "$value" != 1 ]]; then
    echo "$name must be 0 or 1" >&2
    exit 2
  fi
done
if [[ "$RUN_OWNER_TREE" == 0 && "$RUN_TURN_TREE" == 0 && "$RUN_TURN_NONN" == 0 ]]; then
  echo "nothing selected" >&2
  exit 2
fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then
  echo "nvcc and nvidia-smi are required" >&2
  exit 2
fi

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_runtime_w28_label_hotpath_suite}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.txt}"
mkdir -p "$LOGDIR" "$(dirname "$SUMMARY")"
: >"$SUMMARY"

run_case() {
  local name="$1" script="$2"
  shift 2
  local out="$LOGDIR/${name}.out"
  local err="$LOGDIR/${name}.err"
  echo "=== $name ===" >&2
  env \
    BLOCKS="$BLOCKS" THREADS="$THREADS" ITERS="$ITERS" \
    REPEATS="$REPEATS" WARMUP="$WARMUP" ARCH="$ARCH" \
    PTXAS_VERBOSE="$PTXAS_VERBOSE" "$@" \
    bash "$ONEESAN_ROOT/scripts/bench/$script" >"$out" 2>"$err"
  {
    echo "[$name]"
    grep -E '(^|_)(speedup|delta_pct|actual_nonn_fraction)=' "$out" || true
    grep -E ' OK ' "$err" | tail -n1 || true
    echo
  } >>"$SUMMARY"
  cat "$out"
}

if [[ "$RUN_OWNER_TREE" == 1 ]]; then
  run_case owner_local_sector_w28_tree \
    gridfp-runtime-owner-local-sector-w28-tree-microprobe.sh
fi
if [[ "$RUN_TURN_TREE" == 1 ]]; then
  run_case turn_local_sector_w28_tree \
    gridfp-runtime-turn-local-sector-w28-tree-microprobe.sh
fi
if [[ "$RUN_TURN_NONN" == 1 ]]; then
  run_case turn_discovery_nonn_scan \
    gridfp-runtime-turn-discovery-nonn-scan-microprobe.sh \
    CASES="$CASES"
fi

echo "--- W28 label hotpath suite summary ---"
cat "$SUMMARY"
echo "gridfp-runtime-w28-label-hotpath-suite OK summary=$SUMMARY" >&2
