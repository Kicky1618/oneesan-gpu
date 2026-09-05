#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"; THREADS="${THREADS:-256}"; DEVICE="${DEVICE:-0}"
MAIN_PULL_ILP="${MAIN_PULL_ILP:-1}"; HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
MAIN_SKIP="${MAIN_SKIP:-0}"; BLOCK_SKIP="${BLOCK_SKIP:-0}"
command -v ncu >/dev/null || { echo "ncu required" >&2; exit 2; }
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }

BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_exact_ncu_n${N}_ilp${MAIN_PULL_ILP}}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_exact_ncu_n${N}_ilp${MAIN_PULL_ILP}_t${THREADS}}"
mkdir -p "$(dirname "$PREFIX")"

N="$N" ARCH="$ARCH" OUT="$BIN" MAIN_PULL_ILP="$MAIN_PULL_ILP" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh" \
  >"${PREFIX}.build.out" 2>"${PREFIX}.build.err"

profile(){
  local label="$1" regex="$2" skip="$3"
  local out="${PREFIX}.${label}"
  GRIDFP_THREADS="$THREADS" \
    ncu --devices "$DEVICE" --filter-mode per-gpu \
      --kernel-name-base function \
      --kernel-name "regex:${regex}" \
      --launch-skip "$skip" --launch-count 1 \
      --section SpeedOfLight --section Occupancy --section WarpStateStats \
      --section MemoryWorkloadAnalysis \
      --page details --force-overwrite -o "$out" \
      "$BIN" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" \
      >"${out}.out" 2>"${out}.err"
  echo "Nsight report: ${out}.ncu-rep" >&2
}

if [[ "$MAIN_PULL_ILP" == 1 ]]; then
  profile main '.*main_pull_kernel.*' "$MAIN_SKIP"
else
  profile main ".*main_pull_kernel_ilp${MAIN_PULL_ILP}.*" "$MAIN_SKIP"
fi
profile block '.*block_pull_kernel.*' "$BLOCK_SKIP"

PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
RESOURCE="${PREFIX}.ptxas.tsv"
python3 "$PARSER" "${PREFIX}.build.err" --label exact --header \
  --contains main_pull_kernel --contains block_pull_kernel >"$RESOURCE" || true

echo "config: exact-full-pull n=$N threads=$THREADS main_pull_ilp=$MAIN_PULL_ILP high_drop_chunk=$HIGH_DROP_CHUNK device=$DEVICE" >&2
echo "inspect: DRAM Throughput, L2 Throughput, Long Scoreboard, Issue Active, Achieved Occupancy, registers/spills" >&2
echo "ptxas=$RESOURCE" >&2
