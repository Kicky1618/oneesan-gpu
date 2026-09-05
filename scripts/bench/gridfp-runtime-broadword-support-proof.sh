#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
if ! command -v "$CXX" >/dev/null; then
  echo "$CXX not found" >&2
  exit 2
fi

SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_broadword_support_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_broadword_support_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"

grep -Fq 'gridfp-runtime-broadword-support-proof OK' <<<"$out"
grep -Fq 'random_cases=1000000' <<<"$out"
grep -Fq 'max_runtime_len=28 broadword_stages=6 exact=1' <<<"$out"

echo 'gridfp-runtime-broadword-support-proof OK exact=1' >&2
