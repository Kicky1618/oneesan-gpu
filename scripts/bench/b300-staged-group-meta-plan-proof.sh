#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}";command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_staged_group_meta_plan_proof.cpp";BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_staged_group_meta_plan_proof}"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
text="$($BIN)";printf '%s\n' "$text"
grep -Fq 'total_groups=16384' <<<"$text"
grep -Fq 'staged_mib_per_gpu=217.75' <<<"$text"
grep -Fq 'staged_total_h2d_gib=1.701171875' <<<"$text"
grep -Fq 'old_meta_h2d_gib=5.9541015625 h2d_reduction=3.5x' <<<"$text"
grep -Fq 'total_group_processings=458752' <<<"$text"
grep -Fq 'exact=1' <<<"$text"
