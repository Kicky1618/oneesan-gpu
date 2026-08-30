#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
MOD="${2:-4294967291}"
STORAGE="${ONEESAN_STORAGE:-device-vmm}"

case "$STORAGE" in
  device-vmm|managed-host) ;;
  *) echo "ONEESAN_STORAGE must be device-vmm or managed-host; got $STORAGE" >&2; exit 2 ;;
esac

BASE_SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
ADAPTIVE_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n${N}_adaptive_storage_base.cu"
BIN="${BIN:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_adaptive_n${N}}"

python3 "$ONEESAN_ROOT/scripts/build/gen-b300-adaptive-storage.py" "$BASE_SRC" "$ADAPTIVE_SRC"

echo "adaptive runner: n=$N storage=$STORAGE ngpu=${NGPU:-auto} target_mib=${TARGET_MIB:-auto} planner_target_mib=${GRIDFP_PLAN_TARGET_MIB:-auto} reserve_mib=${GRIDFP_VRAM_RESERVE_MIB:-auto} max_window=${MAX_WINDOW:-auto}" >&2

export SRC="$ADAPTIVE_SRC"
export BIN
export FAST_SHARD_ADDRESS8=0
export ONEESAN_STORAGE="$STORAGE"
exec "$ONEESAN_ROOT/scripts/run/b300x8.sh" "$N" "$MOD"
