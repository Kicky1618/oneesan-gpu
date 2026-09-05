#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_height_recurrence_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_height_recurrence_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-height-recurrence-proof OK' <<<"$out"
grep -Fq 'exhaustive_width_max=12' <<<"$out"
grep -Fq 'storage_bytes=1 main_update=mp block_update=pm1 exact=1' <<<"$out"
echo 'b300-height-recurrence-proof OK exact=1' >&2
