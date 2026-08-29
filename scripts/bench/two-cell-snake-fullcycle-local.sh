#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NVCC="${NVCC:-nvcc}"
OBJCOPY="${OBJCOPY:-objcopy}"
CUDA_ARCH="${CUDA_ARCH:-sm_90}"
W="${W:-8}"
MAX_CLUSTER="${MAX_CLUSTER:-8}"
SHARED_KIB="${SHARED_KIB:-228}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build/two-cell-snake}"

command -v "$NVCC" >/dev/null || {
  echo "nvcc not found: $NVCC" >&2
  exit 2
}
command -v "$OBJCOPY" >/dev/null || {
  echo "objcopy not found: $OBJCOPY" >&2
  exit 2
}

mkdir -p "$BUILD_DIR"

COMMON=(
  -std=c++17
  -O3
  -arch="$CUDA_ARCH"
  -rdc=true
  -lineinfo
)

STAGE_SOURCES=(
  "$ROOT/src/cuda/gridfp/two_cell_snake_stage_forward.cu"
  "$ROOT/src/cuda/gridfp/two_cell_snake_stage_right.cu"
  "$ROOT/src/cuda/gridfp/two_cell_snake_stage_reverse.cu"
  "$ROOT/src/cuda/gridfp/two_cell_snake_stage_left.cu"
)
DRIVER="$ROOT/src/cuda/gridfp/two_cell_snake_fullcycle_microprobe.cu"

OBJECTS=()
for src in "${STAGE_SOURCES[@]}"; do
  base="$(basename "$src" .cu)"
  obj="$BUILD_DIR/$base.o"
  echo "+ $NVCC ${COMMON[*]} -c $src -o $obj"
  "$NVCC" "${COMMON[@]}" -c "$src" -o "$obj"

  # Standalone microprobe files recursively rename their nested `main`
  # functions to symbols containing `main_unused`.  Several stage translation
  # units intentionally include common probe ancestors, so those dead host
  # symbols would otherwise collide at the final link.  Localize only those
  # synthetic probe-entry symbols; kernels and the exported stage API stay
  # global.
  echo "+ $OBJCOPY --wildcard --localize-symbol=*main_unused* $obj"
  "$OBJCOPY" --wildcard --localize-symbol='*main_unused*' "$obj"
  OBJECTS+=("$obj")
done

DRIVER_OBJ="$BUILD_DIR/two_cell_snake_fullcycle_microprobe.o"
echo "+ $NVCC ${COMMON[*]} -c $DRIVER -o $DRIVER_OBJ"
"$NVCC" "${COMMON[@]}" -c "$DRIVER" -o "$DRIVER_OBJ"
OBJECTS+=("$DRIVER_OBJ")

BIN="$BUILD_DIR/two-cell-snake-fullcycle"
echo "+ $NVCC ${COMMON[*]} ${OBJECTS[*]} -o $BIN"
"$NVCC" "${COMMON[@]}" "${OBJECTS[@]}" -o "$BIN"

echo "built: $BIN"
echo "run:   $BIN $W $MAX_CLUSTER $SHARED_KIB"

if [[ "${RUN:-1}" == "1" ]]; then
  "$BIN" "$W" "$MAX_CLUSTER" "$SHARED_KIB"
fi
