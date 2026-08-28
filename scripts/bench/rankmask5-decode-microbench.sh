#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-sm_80}"
N="${N:-4194304}"
REPEATS="${REPEATS:-20}"
TRIALS="${TRIALS:-5}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-0}"

if (( N <= 0 || REPEATS <= 0 || TRIALS <= 0 )); then
  echo "N, REPEATS, and TRIALS must be positive" >&2
  exit 2
fi
if [[ "$PTXAS_VERBOSE" != 0 && "$PTXAS_VERBOSE" != 1 ]]; then
  echo "PTXAS_VERBOSE must be 0 or 1" >&2
  exit 2
fi

SRC="$ONEESAN_ROOT/src/cuda/gridfp/probes/rankmask5_decode_microbench.cu"
BIN="${BIN:-$ONEESAN_BUILD_DIR/rankmask5_decode_microbench_${ARCH}}"
mkdir -p "$(dirname "$BIN")"

NVCC_FLAGS=(-O3 -std=c++17 -lineinfo -arch="$ARCH")
if [[ "$PTXAS_VERBOSE" == 1 ]]; then
  NVCC_FLAGS+=(-Xptxas=-v)
fi
TMPDIR="$ONEESAN_TMP_DIR" nvcc "${NVCC_FLAGS[@]}" "$SRC" -o "$BIN"

out="$($BIN "$N" "$REPEATS" "$TRIALS")"
printf '%s\n' "$out"
grep -Eq 'rankmask5-decode-microbench (OK|SKIP no CUDA device)' <<<"$out"
if grep -Fq 'rankmask5-decode-microbench OK' <<<"$out"; then
  grep -Fq 'table_cases=6075' <<<"$out"
  grep -Fq 'checksum_exact=1' <<<"$out"
  grep -Fq 'decode_model=rank16_then_source32' <<<"$out"
  grep -Fq 'ffs_to_unrolled_speedup=' <<<"$out"
fi

echo "rankmask5-decode-microbench done arch=$ARCH n=$N repeats=$REPEATS trials=$TRIALS ptxas_verbose=$PTXAS_VERBOSE" >&2
