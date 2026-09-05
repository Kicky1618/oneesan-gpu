#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; MAX_W="${MAX_W:-11}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
if (( MAX_W < 5 || MAX_W > 12 )); then echo "MAX_W must be 5..12" >&2; exit 2; fi
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_find_signature_filter_model.cpp"; BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_find_signature_filter_model}"; mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN "$MAX_W")"; printf '%s\n' "$out"
grep -Fq 'ALL_OK runtime_find_signature_filter_model=1 hash=xor_shift_7_14 bits=64 false_negative=0' <<<"$out"
if (( MAX_W >= 10 )); then
  grep -Eq '^W=10 .*comparison_ratio=0\.(5[0-9]|6[0-4])' <<<"$out"
fi
echo "gridfp-runtime-find-signature-filter-model OK maxW=$MAX_W" >&2
