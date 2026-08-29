#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cuda/probes/b300_cpasync_gather_proof.cu"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_cpasync_gather_proof}"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -arch="${ARCH:-native}" "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-cpasync-gather-proof OK' <<<"$out"
grep -Fq 'cp_bytes=4 exact=1' <<<"$out"
echo 'b300-cpasync-gather-proof OK exact=1' >&2
