#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-c++}"
command -v "$CXX" >/dev/null || {
  echo "C++ compiler is required" >&2
  exit 2
}

SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_materialize_primitive_last_r_proof.cpp"
BIN="$ONEESAN_BUILD_DIR/gridfp_materialize_primitive_last_r_proof"
mkdir -p "$ONEESAN_BUILD_DIR"
"$CXX" -O3 -std=c++17 -Wall -Wextra -pedantic "$SRC" -o "$BIN"
out="$($BIN)"
echo "$out"
grep -Fq 'gridfp-materialize-primitive-last-r-proof OK' <<<"$out"
grep -Fq 'all_primitive_ranks=3707851' <<<"$out"
grep -Fq 'final_pre_state_h=1' <<<"$out"
grep -Fq 'final_pre_state_rank=0' <<<"$out"
grep -Fq 'final_symbol=R' <<<"$out"
grep -Fq 'output_exact=1' <<<"$out"
grep -Fq 'saved_primitive_table_loads_per_call=1' <<<"$out"
grep -Fq 'saved_rank_branch_per_call=1' <<<"$out"
echo 'gridfp-materialize-primitive-last-r-proof runner OK' >&2
