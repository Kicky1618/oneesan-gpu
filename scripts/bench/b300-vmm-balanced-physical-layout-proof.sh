#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_vmm_balanced_physical_layout_proof.cpp"
OUT="${OUT:-$ONEESAN_BUILD_DIR/b300_vmm_balanced_physical_layout_proof}"
"$CXX" -O3 -std=c++17 "$SRC" -o "$OUT"
text="$($OUT)"
printf '%s\n' "$text"
grep -Fq 'combined_imbalance_le_one_granularity=1' <<<"$text"
grep -Fq 'logical_shard_views_preserved=1 exact=1' <<<"$text"
