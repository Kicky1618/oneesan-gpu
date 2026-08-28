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
field(){ sed -nE "s/.*(^|[[:space:]])$1=([^[:space:]]+).*/\\2/p" <<<"$out" | tail -n1; }
max_masks="$(field max_owner_masks)"
slot_bytes="$(field mask_slot_bytes)"
base_bytes="$(field max_sparse_base_bytes)"
total_bytes="$(field max_sparse_total_bytes)"
dense_bytes="$(field dense_base_bytes)"
[[ -n "$max_masks" && -n "$slot_bytes" && -n "$base_bytes" && -n "$total_bytes" && -n "$dense_bytes" ]]
(( max_masks < 4096 ))
(( slot_bytes == 32768 ))
(( total_bytes == slot_bytes + base_bytes ))
(( total_bytes < 100000 ))
(( dense_bytes == 524288 ))
(( total_bytes * 5 < dense_bytes ))
echo "rankformula-sparse-base-proof OK max_masks=$max_masks sparse_max_bytes=$total_bytes dense_bytes=$dense_bytes reduction_gt_5x=1" >&2
