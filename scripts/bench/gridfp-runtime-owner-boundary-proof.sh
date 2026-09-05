#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
if ! command -v "$CXX" >/dev/null; then
  echo "$CXX not found" >&2
  exit 2
fi
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_owner_boundary_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_owner_boundary_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-owner-boundary-proof OK' <<<"$out"
grep -Fq 'configs=165 group_cases=245640 empty_owner_slots=54 active_nonzero_owner_at_zero=0' <<<"$out"
grep -Fq 'W_min=8 W_max=28 ngpu_min=2 ngpu_max=16 exact=1' <<<"$out"
echo 'gridfp-runtime-owner-boundary-proof OK exact=1' >&2
