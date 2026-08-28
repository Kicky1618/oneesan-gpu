#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-native}"
GPUS="${GPUS:-8}"
ELEMS="${ELEMS:-8388731}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_vmm_storage_helper_microprobe}"
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= GPUS )) || { echo "requested $GPUS GPUs, visible=$visible" >&2; exit 2; }
bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-balanced-physical-layout-proof.sh"
SRC="$ONEESAN_ROOT/src/cuda/b300/probes/vmm_contiguous_storage_helper_microprobe.cu"
"$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" -I"$ONEESAN_ROOT/src/cuda/b300" "$SRC" -lcuda -o "$BIN"
text="$($BIN "$GPUS" "$ELEMS")"
printf '%s\n' "$text"
grep -Fq 'all_gpu_read=OK exact=OK' <<<"$text"
grep -Fq 'direct_base_index=1 logical_shard_views=1' <<<"$text"
