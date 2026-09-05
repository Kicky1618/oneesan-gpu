#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BIN="${1:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_multigpu_mmap}"
N="${2:-17}"
MOD="${3:-2305843009213693951}"
TARGET_MIB="${4:-8}"
MAX_WINDOW="${5:-8}"

for spec in "N:$N" "MOD:$MOD" "TARGET_MIB:$TARGET_MIB" "MAX_WINDOW:$MAX_WINDOW"; do
  require_uint "${spec%%:*}" "${spec#*:}" || exit 2
done

KNOWN_EXACT=""
case "$N" in
  1) KNOWN_EXACT=2 ;;
  2) KNOWN_EXACT=12 ;;
  3) KNOWN_EXACT=184 ;;
  4) KNOWN_EXACT=8512 ;;
  5) KNOWN_EXACT=1262816 ;;
  6) KNOWN_EXACT=575780564 ;;
  9) KNOWN_EXACT=41044208702632496804 ;;
esac

if [[ ! -x "$BIN" ]]; then
  OUT="$BIN" "$ONEESAN_ROOT/scripts/build/gridfp-multigpu-mmap.sh"
fi
VISIBLE="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
BASE="${TMPDIR:-/tmp}/gridfp-mg-regression-$$"
mkdir -p "$BASE"
trap 'rm -rf "$BASE"' EXIT

runs=(1)
for g in 2 4 8; do
  if (( VISIBLE >= g )); then runs+=("$g"); fi
done

ref=""
for g in "${runs[@]}"; do
  dir="$BASE/g$g"
  mkdir -p "$dir"
  echo "=== ${g} GPU ==="
  "$BIN" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" "$g" "$dir" | tee "$BASE/g$g.out"
  result_line="$(grep 'residue=' "$BASE/g$g.out" | tail -n 1)"
  residue="$(sed -n 's/.* residue=\([0-9][0-9]*\).*/\1/p' <<<"$result_line")"
  echoed_mod="$(sed -n 's/.* modulus=\([0-9][0-9]*\).*/\1/p' <<<"$result_line")"
  if [[ -z "$residue" || "$echoed_mod" != "$MOD" ]]; then
    echo "could not validate solver residue/modulus echo: $result_line" >&2
    exit 3
  fi
  if [[ -n "$KNOWN_EXACT" ]]; then
    expected="$(python3 -c 'import sys; print(int(sys.argv[1]) % int(sys.argv[2]))' "$KNOWN_EXACT" "$MOD")"
    if [[ "$residue" != "$expected" ]]; then
      echo "golden mismatch for n=$N: residue=$residue expected=$expected (mod $MOD)" >&2
      exit 4
    fi
    echo "golden residue: MATCH (n=$N exact mod p=$expected)"
  fi
  if [[ -z "$ref" ]]; then
    ref="$dir"
  else
    cmp "$ref/main.bin" "$dir/main.bin"
    cmp "$ref/blocked.bin" "$dir/blocked.bin"
    echo "state files: IDENTICAL to 1-GPU reference"
  fi
done

sha256sum "$ref/main.bin" "$ref/blocked.bin"
echo "PASS: ${runs[*]} GPU results are byte-identical"
