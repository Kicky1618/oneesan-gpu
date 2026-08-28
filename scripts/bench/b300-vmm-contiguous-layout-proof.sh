#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || exit 2
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_vmm_contiguous_layout_proof.cpp"
OUT="${OUT:-$ONEESAN_BUILD_DIR/b300_vmm_contiguous_layout_proof}"
"$CXX" -O3 -std=c++17 "$SRC" -o "$OUT"
text="$($OUT)"; printf '%s\n' "$text"
grep -Fq 'gran=2097152' <<<"$text"
grep -Fq 'internal_padding=0 tail_padding_only=1 exact=1' <<<"$text"
grep -Fq 'max_segment_imbalance_one_granularity=1 internal_padding=0 tail_padding_lt_granularity=1 direct_base_index=1 exact=1' <<<"$text"
