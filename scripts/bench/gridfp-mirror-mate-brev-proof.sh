#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_mirror_mate_brev_proof.cpp"; BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_mirror_mate_brev_proof}"; mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"; out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-mirror-mate-brev-proof OK' <<<"$out"
grep -Fq 'random_cases=1000000 width_max=32' <<<"$out"
grep -Fq 'bit_reverse_exact=1 rl_swap_implicit=1' <<<"$out"
echo 'gridfp-mirror-mate-brev-proof OK exact=1' >&2
