#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-sm_80}"
N="${N:-4194304}"
REPEATS="${REPEATS:-20}"
TRIALS="${TRIALS:-5}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-0}"
MASK_ORDER="${MASK_ORDER:-shuffled}"

if (( N <= 0 || REPEATS <= 0 || TRIALS <= 0 )); then
  echo "N, REPEATS, and TRIALS must be positive" >&2
  exit 2
fi
if [[ "$PTXAS_VERBOSE" != 0 && "$PTXAS_VERBOSE" != 1 ]]; then
  echo "PTXAS_VERBOSE must be 0 or 1" >&2
  exit 2
fi
if [[ "$MASK_ORDER" != ordered && "$MASK_ORDER" != shuffled ]]; then
  echo "MASK_ORDER must be ordered or shuffled" >&2
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

out="$($BIN "$N" "$REPEATS" "$TRIALS" "$MASK_ORDER")"
printf '%s\n' "$out"
grep -Eq 'rankmask5-decode-microbench (OK|SKIP no CUDA device)' <<<"$out"
if grep -Fq 'rankmask5-decode-microbench OK' <<<"$out"; then
  grep -Fq "mask_order=$MASK_ORDER" <<<"$out"
  grep -Fq 'table_cases=6075' <<<"$out"
  grep -Fq 'popcount_hist=5855,187,32,1,0,0' <<<"$out"
  grep -Fq 'rankmask_or=0x07 upper_bits_zero=1 checksum_exact=1' <<<"$out"
  grep -Fq 'decode_model=rank16_then_source32' <<<"$out"
  grep -Fq 'direct3_ordinals=0,1,2' <<<"$out"
  grep -Fq 'direct3_guard=rankmask_nonzero_outer_guard' <<<"$out"
  grep -Fq 'direct3_guard_ms=' <<<"$out"
  grep -Fq 'ffs_to_unrolled5_speedup=' <<<"$out"
  grep -Fq 'ffs_to_direct3_speedup=' <<<"$out"
  grep -Fq 'ffs_to_direct3_guard_speedup=' <<<"$out"
  grep -Fq 'unrolled5_to_direct3_speedup=' <<<"$out"
  grep -Fq 'direct3_to_guard_speedup=' <<<"$out"
fi

echo "rankmask5-decode-microbench done arch=$ARCH n=$N repeats=$REPEATS trials=$TRIALS mask_order=$MASK_ORDER modes=ffs,unrolled5,direct3,direct3_guard ptxas_verbose=$PTXAS_VERBOSE" >&2
