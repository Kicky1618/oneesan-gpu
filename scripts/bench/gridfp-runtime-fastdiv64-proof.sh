#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
if ! command -v "$CXX" >/dev/null; then
  echo "$CXX not found" >&2
  exit 2
fi

mkdir -p "$ONEESAN_BUILD_DIR"

run_proof() {
  local src="$1" bin="$2" marker="$3"
  "$CXX" -O3 -std=c++17 "$src" -o "$bin"
  local out
  out="$($bin)"
  grep -Fq "$marker" <<<"$out"
  printf '%s\n' "$out"
}

FAST_SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_fastdiv64_proof.cpp"
FAST_BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_fastdiv64_proof}"
fast_out="$(run_proof "$FAST_SRC" "$FAST_BIN" 'gridfp-runtime-fastdiv64-proof OK')"
printf '%s\n' "$fast_out"
grep -Fq 'primitive_magic_entries=29 primitive_nonzero_divisors=14 primitive_magic_exact=1' <<<"$fast_out"
grep -Fq 'small_bits=12' <<<"$fast_out"
grep -Fq 'random64=2000000' <<<"$fast_out"
grep -Fq 'production_max_numerator=473397057701' <<<"$fast_out"
grep -Fq 'quotient_error_bound=1 product_overflow_checked=1 exact=1' <<<"$fast_out"

OWNER_SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_owner_group_magic_proof.cpp"
OWNER_BIN="$ONEESAN_BUILD_DIR/gridfp_runtime_owner_group_magic_proof"
owner_out="$(run_proof "$OWNER_SRC" "$OWNER_BIN" 'gridfp-runtime-owner-group-magic-proof OK')"
printf '%s\n' "$owner_out"
grep -Fq 'slots=154 active_divisors=99 max_divisor=448876754 table_bytes=1232 magic_exact=1' <<<"$owner_out"

TURN_SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_turn_compress_group_magic_proof.cpp"
TURN_BIN="$ONEESAN_BUILD_DIR/gridfp_runtime_turn_compress_group_magic_proof"
turn_out="$(run_proof "$TURN_SRC" "$TURN_BIN" 'gridfp-runtime-turn-compress-group-magic-proof OK')"
printf '%s\n' "$turn_out"
grep -Fq 'slots=154 active_divisors=99 max_divisor=510468519 table_bytes=1232 magic_exact=1' <<<"$turn_out"

echo 'gridfp-runtime-fastdiv64-proof OK primitive_magic_exact=1 owner_group_magic_exact=1 turn_compress_group_magic_exact=1 exact_divmod64=1' >&2
