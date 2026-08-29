#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
NVCC="${NVCC:-nvcc}"
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_codec_table_physical_compile_probe.cu"
[[ -f "$SRC" ]] || { echo "missing physical compile probe: $SRC" >&2; exit 2; }
PREFIX="${PREFIX:-$ONEESAN_BUILD_DIR/gridfp_codec_table_physical_compile}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/gridfp_codec_table_physical_compile_matrix_logs}"
mkdir -p "$LOGDIR" "$(dirname "$PREFIX")"
PTXAS_FLAGS=(); [[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")

compiled=0
for choose in 0 1 2 3; do
  for primitive in 0 1 2; do
    out="${PREFIX}_c${choose}_p${primitive}"
    bout="$LOGDIR/c${choose}_p${primitive}.out"
    berr="$LOGDIR/c${choose}_p${primitive}.err"
    echo "=== physical codec compile choose=$choose primitive=$primitive ===" >&2
    TMPDIR="$ONEESAN_TMP_DIR" "$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" \
      "${PTXAS_FLAGS[@]}" \
      -DRP_EXPERIMENTAL_CODEC_CHOOSE_PHYSICAL_MODE="$choose" \
      -DRP_EXPERIMENTAL_CODEC_PRIMITIVE_PHYSICAL_MODE="$primitive" \
      "$SRC" -o "$out" >"$bout" 2>"$berr"
    [[ -x "$out" ]] || { echo "physical codec compile produced no binary c=$choose p=$primitive" >&2; exit 3; }
    ((compiled += 1))
    if [[ "$PTXAS_VERBOSE" == 1 ]]; then
      echo "--- ptxas physical codec c=$choose p=$primitive ---" >&2
      grep -E 'Used .* registers|bytes smem|bytes cmem' "$berr" >&2 || true
    fi
  done
done
[[ "$compiled" == 12 ]] || { echo "unexpected physical compile count=$compiled" >&2; exit 4; }

# Nonzero physical layout + old preinclude remap must be rejected at compile
# time so a benchmark cannot accidentally retain the legacy constant table.
PRE="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_runtime_codec_tables_sym_u32_preinclude.cuh"
conflict_out="${PREFIX}_conflict_should_not_exist"
if TMPDIR="$ONEESAN_TMP_DIR" "$NVCC" -O2 -std=c++17 -arch="$ARCH" \
    -include "$PRE" \
    -DRP_RUNTIME_CODEC_CHOOSE_U32_MODE=1 \
    -DRP_RUNTIME_CODEC_PRIMITIVE_U32_MODE=1 \
    -DRP_EXPERIMENTAL_CODEC_CHOOSE_PHYSICAL_MODE=1 \
    -DRP_EXPERIMENTAL_CODEC_PRIMITIVE_PHYSICAL_MODE=1 \
    "$SRC" -o "$conflict_out" >"$LOGDIR/conflict.out" 2>"$LOGDIR/conflict.err"; then
  echo "physical codec compile unexpectedly accepted preinclude conflict" >&2
  exit 5
fi
grep -Eq 'physical (choose|primitive) layout cannot be combined' "$LOGDIR/conflict.err" || {
  echo "physical codec conflict failed for an unexpected reason" >&2
  cat "$LOGDIR/conflict.err" >&2
  exit 6
}

echo "gridfp-codec-table-physical-compile-matrix OK combinations=$compiled preinclude_conflict_rejected=1 arch=$ARCH exact=1"
