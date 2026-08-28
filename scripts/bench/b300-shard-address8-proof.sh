#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_b300_shard_address8_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_b300_shard_address8_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-b300-shard-address8-proof OK' <<<"$out"
grep -Fq 'ngpu_min=1 ngpu_max=8' <<<"$out"
grep -Fq 'main_states=385719506620 block_states=135015505407' <<<"$out"
grep -Fq 'main_chunk=48214938328 block_chunk=16876938176' <<<"$out"
grep -Fq 'max_chunk_bits=61 compare_stages=3 max_subtractions=3' <<<"$out"
grep -Fq 'owner_mul=0 div64=0 mod64=0 exact=1' <<<"$out"
echo 'gridfp-b300-shard-address8-proof OK exact=1' >&2
