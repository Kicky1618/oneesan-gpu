#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}";ARCH="${ARCH:-native}";GPUS="${GPUS:-8}";((GPUS>=2&&GPUS<=8))||exit 2
require_nvcc_version_at_least "$NVCC" 13 0 "B300 cross-device event wait preflight"
command -v nvidia-smi >/dev/null||exit 2;visible="$(nvidia-smi --query-gpu=index --format=csv,noheader|wc -l)";((visible>=GPUS))||{ echo "need $GPUS visible GPUs, have $visible" >&2;exit 2; }
SRC="$ONEESAN_ROOT/src/cuda/b300/probes/cross_device_event_wait_probe.cu";BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_cross_device_event_wait_probe}"
TMPDIR="$ONEESAN_TMP_DIR" "$NVCC" -O3 -std=c++17 -arch="$ARCH" "$SRC" -o "$BIN"
text="$($BIN "$GPUS")";printf '%s\n' "$text"
grep -Fq "gpus=$GPUS" <<<"$text";grep -Fq "foreign_event_waits=$((GPUS-1))" <<<"$text";grep -Fq 'gpu0_host_syncs=1 exact=1' <<<"$text"
echo "b300-cross-device-event-wait-preflight OK gpus=$GPUS"
