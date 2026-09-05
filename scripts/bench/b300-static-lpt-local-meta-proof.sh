#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}";command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_static_lpt_local_meta_proof.cpp";BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_static_lpt_local_meta_proof}"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
text="$($BIN)";printf '%s\n' "$text"
grep -Fq 'total_groups=16384' <<<"$text"
grep -Fq 'gpu_group_min=2044 gpu_group_max=2052' <<<"$text"
grep -Fq 'local_meta_ids_contiguous=1 local_meta_ids_unique=1 group_assignment_exactly_once=1' <<<"$text"
grep -Fq 'rows=28 total_group_processings=458752' <<<"$text"
grep -Fq 'gpu_process_min=57232 gpu_process_max=57456' <<<"$text"
grep -Fq 'max_meta_bytes_per_gpu=28596672' <<<"$text"
grep -Fq 'total_h2d_gib=0.212646484375 metadata_replication=0 exact=1' <<<"$text"
