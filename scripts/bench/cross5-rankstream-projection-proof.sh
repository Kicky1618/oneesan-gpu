#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
if ! command -v "$CXX" >/dev/null; then
  echo "$CXX not found" >&2
  exit 2
fi

SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_cross5_rankstream_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_cross5_rankstream_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"

grep -Fq 'gridfp-cross5-rankstream-proof OK' <<<"$out"
grep -Fq 'fused16_cases=6075' <<<"$out"
grep -Fq 'chunk_state_bounds=15,20,25' <<<"$out"
grep -Fq 'states=26 max_depth=15 max_factor=14' <<<"$out"
grep -Fq 'rank_projection_exact=1 halt_projection_exact=1 partial_chunk_zero_prefix_exact=1' <<<"$out"
grep -Fq 'fused16_entry_exact=1 fused16_meta_exact=1 fused16_byte_isolation_exact=1' <<<"$out"
grep -Fq 'fallback_structurally_unreachable=1' <<<"$out"

echo 'cross5-rankstream-projection-proof OK chunk_state_bounds=15,20,25 fused16_cases=6075 production_state_checks=0 production_fallback=0' >&2
