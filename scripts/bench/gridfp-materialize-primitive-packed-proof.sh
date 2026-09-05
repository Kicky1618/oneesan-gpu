#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-c++}"
command -v "$CXX" >/dev/null || { echo "C++ compiler is required" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_materialize_primitive_packed_proof.cpp"
BIN="$ONEESAN_BUILD_DIR/gridfp_materialize_primitive_packed_proof"
mkdir -p "$ONEESAN_BUILD_DIR"
"$CXX" -O3 -std=c++17 -Wall -Wextra -pedantic "$SRC" -o "$BIN"
out="$($BIN)"
echo "$out"
grep -Fq 'gridfp-materialize-primitive-packed-proof OK' <<<"$out"
grep -Fq 'all_primitive_ranks=3707851' <<<"$out"
grep -Fq 'reachable_threshold_cells=104' <<<"$out"
grep -Fq 'packed_entries=104' <<<"$out"
grep -Fq 'packed_bytes=416' <<<"$out"
grep -Fq 'full_u64_bytes=6960' <<<"$out"
grep -Fq 'packed_max_value=742900' <<<"$out"
grep -Fq 'threshold_exact=1' <<<"$out"
grep -Fq 'final_R_assumed=1' <<<"$out"
echo 'gridfp-materialize-primitive-packed-proof runner OK' >&2
