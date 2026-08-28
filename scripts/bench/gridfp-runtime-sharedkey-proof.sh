#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
if ! command -v "$CXX" >/dev/null; then
  echo "$CXX not found" >&2
  exit 2
fi

SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_sharedkey_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_sharedkey_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"

grep -Fq 'gridfp-runtime-sharedkey-proof OK' <<<"$out"
grep -Fq 'max_w=28 mate_bits_max=56 blocked_bit=63' <<<"$out"
grep -Fq 'devicekey_model_bytes=16 packed_key_bytes=8' <<<"$out"
grep -Fq 'shared_key_entries_per_block=1280' <<<"$out"
grep -Fq 'unpacked_shared_key_bytes_per_block=20480 packed_shared_key_bytes_per_block=10240' <<<"$out"
grep -Fq 'shared_key_bytes_saved_per_block=10240 shared_key_reduction_pct=50' <<<"$out"
grep -Fq 'roundtrip_exact=1' <<<"$out"

echo 'gridfp-runtime-sharedkey-proof OK packed_shared_key_bytes_per_block=10240 saved=10240' >&2
