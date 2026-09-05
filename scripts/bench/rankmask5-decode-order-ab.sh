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
BIN="${BIN:-$ONEESAN_BUILD_DIR/rankmask5_decode_order_ab_${ARCH}}"
mkdir -p "$(dirname "$BIN")"

NVCC_FLAGS=(-O3 -std=c++17 -lineinfo -arch="$ARCH")
if [[ "$PTXAS_VERBOSE" == 1 ]]; then
  NVCC_FLAGS+=(-Xptxas=-v)
fi
TMPDIR="$ONEESAN_TMP_DIR" nvcc "${NVCC_FLAGS[@]}" "$SRC" -o "$BIN"

metric() {
  local key="$1"
  awk -v key="$key" '
    {
      for (i = 1; i <= NF; ++i) {
        split($i, kv, "=")
        if (kv[1] == key) {
          print kv[2]
          exit
        }
      }
    }
  '
}

validate_output() {
  local order="$1"
  local out="$2"
  grep -Fq 'rankmask5-decode-microbench OK' <<<"$out"
  grep -Fq "mask_order=$order" <<<"$out"
  grep -Fq 'table_cases=6075' <<<"$out"
  grep -Fq 'popcount_hist=5855,187,32,1,0,0' <<<"$out"
  grep -Fq 'rankmask_or=0x07 upper_bits_zero=1 checksum_exact=1' <<<"$out"
  grep -Fq 'direct3_guard=rankmask_nonzero_outer_guard' <<<"$out"
  grep -Fq 'direct3_guard_ms=' <<<"$out"
  grep -Fq 'ffs_to_direct3_guard_speedup=' <<<"$out"
  grep -Fq 'direct3_to_guard_speedup=' <<<"$out"
}

ordered_out="$($BIN "$N" "$REPEATS" "$TRIALS" ordered)"
printf '%s\n' "$ordered_out"
if grep -Fq 'rankmask5-decode-microbench SKIP no CUDA device' <<<"$ordered_out"; then
  echo "rankmask5-decode-order-ab SKIP no CUDA device" >&2
  exit 0
fi
validate_output ordered "$ordered_out"

shuffled_out="$($BIN "$N" "$REPEATS" "$TRIALS" shuffled)"
printf '%s\n' "$shuffled_out"
validate_output shuffled "$shuffled_out"

ordered_direct3_ms="$(metric direct3_ms <<<"$ordered_out")"
ordered_guard_ms="$(metric direct3_guard_ms <<<"$ordered_out")"
ordered_guard_speedup="$(metric direct3_to_guard_speedup <<<"$ordered_out")"
ordered_ffs_guard_speedup="$(metric ffs_to_direct3_guard_speedup <<<"$ordered_out")"
shuffled_direct3_ms="$(metric direct3_ms <<<"$shuffled_out")"
shuffled_guard_ms="$(metric direct3_guard_ms <<<"$shuffled_out")"
shuffled_guard_speedup="$(metric direct3_to_guard_speedup <<<"$shuffled_out")"
shuffled_ffs_guard_speedup="$(metric ffs_to_direct3_guard_speedup <<<"$shuffled_out")"

for x in \
  "$ordered_direct3_ms" "$ordered_guard_ms" "$ordered_guard_speedup" "$ordered_ffs_guard_speedup" \
  "$shuffled_direct3_ms" "$shuffled_guard_ms" "$shuffled_guard_speedup" "$shuffled_ffs_guard_speedup"; do
  if [[ -z "$x" ]]; then
    echo "failed to parse rankmask5 decode metric" >&2
    exit 3
  fi
done

guard_order_ratio="$(awk -v a="$ordered_guard_speedup" -v b="$shuffled_guard_speedup" 'BEGIN { if (b == 0) print "inf"; else printf "%.6f", a / b }')"

printf 'rankmask5-decode-order-ab OK arch=%s n=%s repeats=%s trials=%s\n' \
  "$ARCH" "$N" "$REPEATS" "$TRIALS"
printf 'ordered direct3_ms=%s direct3_guard_ms=%s direct3_to_guard_speedup=%s ffs_to_direct3_guard_speedup=%s\n' \
  "$ordered_direct3_ms" "$ordered_guard_ms" "$ordered_guard_speedup" "$ordered_ffs_guard_speedup"
printf 'shuffled direct3_ms=%s direct3_guard_ms=%s direct3_to_guard_speedup=%s ffs_to_direct3_guard_speedup=%s\n' \
  "$shuffled_direct3_ms" "$shuffled_guard_ms" "$shuffled_guard_speedup" "$shuffled_ffs_guard_speedup"
printf 'guard_order_sensitivity_ratio=%s note=ordered_speedup_over_shuffled_speedup\n' "$guard_order_ratio"
