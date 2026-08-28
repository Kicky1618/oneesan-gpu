#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_shared_budget_proof.cpp"; BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_shared_budget_proof}"; mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"; out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-shared-budget-proof OK' <<<"$out"
grep -Fq 'baseline_bytes=24064 fast_no_index_bytes=16384' <<<"$out"
grep -Fq 'index16_bytes=1024 fast16_bytes=17408 delta16_bytes=-6656' <<<"$out"
grep -Fq 'index32_bytes=2048 fast32_bytes=18432 delta32_bytes=-5632' <<<"$out"
grep -Fq 'index64_bytes=4096 fast64_bytes=20480 delta64_bytes=-3584' <<<"$out"
grep -Fq 'packed_key_bytes=8 device_key_bytes=16 context_bytes=24 exact=1' <<<"$out"
echo 'gridfp-runtime-shared-budget-proof OK exact=1' >&2
