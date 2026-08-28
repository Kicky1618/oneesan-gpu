#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_abstract_lazy_load.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_abstract_lazy_load}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankformula-abstract-lazy-load OK' <<<"$out"
grep -Fq 'production_codes=1201917 depths=25 calls=30047925' <<<"$out"
grep -Fq 'eager_source_loads=88482757 lazy_source_loads=2492769 removed_source_loads=85989988' <<<"$out"
grep -Fq 'depth1_eager=1497681 depth1_lazy=891345' <<<"$out"
grep -Fq 'depth13_eager=3720805 depth13_lazy=1 depth14_lazy=0' <<<"$out"
grep -Fq 'source_load_only_on_state1=1 uniform_depth_model=1' <<<"$out"
echo 'rankformula-abstract-lazy-load-proof OK removed=85989988 uniform_depths=1..25' >&2
