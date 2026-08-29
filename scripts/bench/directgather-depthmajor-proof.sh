#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_directgather_depthmajor.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_directgather_depthmajor_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
"$BIN"
