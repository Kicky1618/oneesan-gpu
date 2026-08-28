#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
ARCH="${ARCH:-native}"
PTX_ARCH="${PTX_ARCH:-sm_103}"
GPUS="${GPUS:-8}"
[[ "$GPUS" == 8 ]] || { echo "VMM production preflight currently requires GPUS=8" >&2; exit 2; }

bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-production-generate-proof.sh"
ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-production-ptx-proof.sh"
GPUS=8 ARCH="$ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-storage-helper-microprobe.sh"
GPUS=8 ARCH="$ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-cross-boundary-rw-microprobe.sh"

echo "b300-vmm-production-preflight OK layout=1 ptx=1 stale_shard_symbols=0 stale_width_symbols=0 logical_shard_chunks=0 logical_shard_views=0 compact_interval_bytes=24 device0_direct_remote_memcpy=1 cross_boundary_rw_all_gpu=1 shard_free_interval_io=1"
