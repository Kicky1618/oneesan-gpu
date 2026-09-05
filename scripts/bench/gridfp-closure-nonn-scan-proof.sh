#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_closure_nonn_scan_proof.cpp"; BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_closure_nonn_scan_proof}"; mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"; out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-closure-nonn-scan-proof OK' <<<"$out"
grep -Fq 'random_cases=1000000 production_width_max=28' <<<"$out"
grep -Fq 'non_n_scan_exact=1 include_exact=1' <<<"$out"
echo 'gridfp-closure-nonn-scan-proof OK exact=1' >&2
