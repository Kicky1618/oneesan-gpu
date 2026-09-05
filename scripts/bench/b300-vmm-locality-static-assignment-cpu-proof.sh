#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}";command -v "$CXX" >/dev/null||exit 2
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_vmm_locality_static_assignment_proof.cpp";BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_vmm_locality_static_assignment_proof}"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
text="$($BIN)";printf '%s\n' "$text"
grep -Fq 'gran_kib=64 ' <<<"$text";grep -Fq 'gran_kib=2048 ' <<<"$text";grep -Fq 'gran_kib=16384 ' <<<"$text";grep -Fq 'gran_kib=65536 ' <<<"$text"
grep -Fq 'baseline_local=130943942319 locality_local=355140504787 total=1041470024054' <<<"$text"
grep -Fq 'remote_reduction_gt=24.60pct exact_prefix_dp=1' <<<"$text"
echo 'b300-vmm-locality-static-assignment-cpu-proof OK'
