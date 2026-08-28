#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}";command -v "$CXX" >/dev/null||{ echo "$CXX not found" >&2;exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_static_lpt_interval_staging_proof.cpp";BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_static_lpt_interval_staging_proof}"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
text="$($BIN)";printf '%s\n' "$text"
grep -Fq 'groups=16384 descriptor_bytes=24' <<<"$text"
grep -Fq 'interval_groups_main=8192 interval_groups_block=8192' <<<"$text"
grep -Fq 'main_descriptors=6067328 block_descriptors=2386190 total_descriptors=8453518' <<<"$text"
grep -Fq 'gpu_descriptor_min=1056256 gpu_descriptor_max=1057352' <<<"$text"
grep -Fq 'staged_total_bytes=202884432 staged_total_gib=0.188950851560' <<<"$text"
grep -Fq 'staged_max_bytes_per_gpu=25376448 staged_max_mib_per_gpu=24.200866699' <<<"$text"
grep -Fq 'repeated_h2d_bytes=5680764096 repeated_h2d_gib=5.290623843670 h2d_reduction=28x' <<<"$text"
grep -Fq 'metadata_plus_interval_max_mib_per_gpu=51.472778320' <<<"$text"
grep -Fq 'assignment=static_lpt exact=1' <<<"$text"
