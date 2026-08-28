#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_sparse_base.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_sparse_base}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankformula-sparse-base OK' <<<"$out"
grep -Fq 'owner_masks=2047,2047,2047,2048,2048,2049,2048,2050' <<<"$out"
grep -Fq 'max_owner_masks=2050' <<<"$out"
grep -Fq 'mask_slot_bytes=32768' <<<"$out"
grep -Fq 'max_sparse_base_bytes=65600' <<<"$out"
grep -Fq 'max_sparse_total_bytes=98368' <<<"$out"
grep -Fq 'dense_base_bytes=524288' <<<"$out"
echo 'rankformula-sparse-base-proof OK max_masks=2050 sparse_max_bytes=98368 dense_bytes=524288' >&2
