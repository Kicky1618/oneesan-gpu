#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
SRC="$ONEESAN_ROOT/src/cuda/b300/probes/cpasync_ignore_src_microprobe.cu"
OUT="${OUT:-$ONEESAN_BUILD_DIR/cpasync_ignore_src_microprobe}"
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }

nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" "$SRC" -o "$OUT"
text="$($OUT)"
printf '%s\n' "$text"
grep -q 'cp_async_ignore_src=OK zero_fill=OK exact=OK' <<<"$text" || {
  echo 'cp.async ignore-src microprobe failed' >&2
  exit 3
}
echo "b300-cpasync-ignore-src-microprobe OK arch=$ARCH out=$OUT" >&2
