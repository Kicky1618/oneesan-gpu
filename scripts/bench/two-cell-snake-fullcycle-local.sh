#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NVCC="${NVCC:-nvcc}"
CUDA_ARCH="${CUDA_ARCH:-sm_90}"
W="${W:-8}"
MAX_CLUSTER="${MAX_CLUSTER:-8}"
SHARED_KIB="${SHARED_KIB:-228}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build/two-cell-snake}"

mkdir -p "$BUILD_DIR"

COMMON=(
  -std=c++17
  -O3
  -arch="$CUDA_ARCH"
  -rdc=true
  -lineinfo
)

SOURCES=(
  "$ROOT/src/cuda/gridfp/two_cell_snake_stage_forward.cu"
  "$ROOT/src/cuda/gridfp/two_cell_snake_stage_right.cu"
  "$ROOT/src/cuda/gridfp/two_cell_snake_stage_reverse.cu"
  "$ROOT/src/cuda/gridfp/two_cell_snake_stage_left.cu"
  "$ROOT/src/cuda/gridfp/two_cell_snake_fullcycle_microprobe.cu"
)

OBJECTS=()
for src in "${SOURCES[@]}"; do
  base="$(basename "$src" .cu)"
  obj="$BUILD_DIR/$base.o"
  echo "+ $NVCC ${COMMON[*]} -c $src -o $obj"
  "$NVCC" "${COMMON[@]}" -c "$src" -o "$obj"
  OBJECTS+=("$obj")
done

BIN="$BUILD_DIR/two-cell-snake-fullcycle"
echo "+ $NVCC ${COMMON[*]} ${OBJECTS[*]} -o $BIN"
"$NVCC" "${COMMON[@]}" "${OBJECTS[@]}" -o "$BIN"

echo "built: $BIN"
echo "run:   $BIN $W $MAX_CLUSTER $SHARED_KIB"

if [[ "${RUN:-1}" == "1" ]]; then
  "$BIN" "$W" "$MAX_CLUSTER" "$SHARED_KIB"
fi
