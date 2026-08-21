#!/usr/bin/env bash
set -euo pipefail
BIN="${1:-./oneesan_cuda_gridfp_multigpu}"
N="${2:-17}"
MOD="${3:-2305843009213693951}"
TARGET_MIB="${4:-8}"
MAX_WINDOW="${5:-8}"
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
