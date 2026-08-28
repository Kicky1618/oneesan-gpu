#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-native}"
GPUS="${GPUS:-8}"
ELEMS="${ELEMS:-8388731}"
SPAN="${SPAN:-65536}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_vmm_cross_boundary_rw_microprobe}"
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= GPUS )) || { echo "requested $GPUS GPUs, visible=$visible" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cuda/b300/probes/vmm_cross_boundary_rw_microprobe.cu"
"$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" -I"$ONEESAN_ROOT/src/cuda/b300" "$SRC" -lcuda -o "$BIN"
text="$($BIN "$GPUS" "$ELEMS" "$SPAN")"
printf '%s\n' "$text"
grep -Fq 'cross_physical_boundary_rw_all_gpu=OK exact=OK' <<<"$text"
grep -Fq 'all_gpu_write=OK all_gpu_read=OK restore=OK shard_owner_ops=0' <<<"$text"
