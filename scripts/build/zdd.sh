#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

RAPIDD_ROOT="${RAPIDD_ROOT:-/home/kicky/ダウンロード/rapidd}"
CXX="${CXX:-g++}"
OUT="$(build_path "${OUT:-oneesan_zdd}")"

if [[ ! -f "$RAPIDD_ROOT/include/rapidd/rapidd.hpp" ]]; then
  echo "RAPiDD headers not found: $RAPIDD_ROOT/include/rapidd/rapidd.hpp" >&2
  echo "set RAPIDD_ROOT=/path/to/rapidd" >&2
  exit 1
fi

"$CXX" -O3 -std=c++20 -march=native \
  -I"$RAPIDD_ROOT/include" \
  "$ONEESAN_ROOT/src/cpp/oneesan_zdd.cpp" \
  -o "$OUT"

echo "built $OUT"
echo "  RAPIDD_ROOT=$RAPIDD_ROOT"
