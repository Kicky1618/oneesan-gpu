#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_find_recent_unrolled_proof.cpp"; BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_find_recent_unrolled_proof}"; mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"; out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-find-recent-unrolled-proof OK' <<<"$out"
grep -Fq 'max_pairs=20 cases=4200000 exact=1 order=recent_first loop_control=fallthrough_switch' <<<"$out"
echo 'gridfp-runtime-find-recent-unrolled-proof OK exact=1' >&2
