#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-sm_103}"
NGPU="${NGPU:-8}"
VALUES="${VALUES:-4096}"
THREADS="${THREADS:-256}"
BLOCKS="${BLOCKS:-64}"
OUT="${OUT:-$ONEESAN_BUILD_DIR/b300_cpasync_remote_peer_microprobe}"
SRC="$ONEESAN_ROOT/src/cuda/b300/probes/cpasync_remote_peer_microprobe.cu"

command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || {
  echo "need at least $NGPU visible GPUs" >&2; exit 2;
}

TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -Xptxas=-v "$SRC" -o "$OUT"

"$OUT" "$NGPU" "$VALUES" "$THREADS" "$BLOCKS"

echo "b300-cpasync-remote-peer-microprobe OK arch=$ARCH ngpu=$NGPU values=$VALUES threads=$THREADS blocks=$BLOCKS" >&2
