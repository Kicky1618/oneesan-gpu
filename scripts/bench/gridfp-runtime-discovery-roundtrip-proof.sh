#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_discovery_roundtrip_proof.cpp"; BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_discovery_roundtrip_proof}"; mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"; out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-discovery-roundtrip-proof OK' <<<"$out"
grep -Fq 'exhaustive_W_max=10' <<<"$out"
grep -Fq 'random_cases=500000' <<<"$out"
grep -Fq 'rl_prefix_exact=1 include_recheck_required=0' <<<"$out"
grep -Fq 'production_W_max=28 exact=1' <<<"$out"
echo 'gridfp-runtime-discovery-roundtrip-proof OK exact=1' >&2
