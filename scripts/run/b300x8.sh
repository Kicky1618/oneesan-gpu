#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
MOD="${2:-4294967291}"
TARGET_MIB="${TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
NGPU="${NGPU:-8}"
FAST_SHARD_ADDRESS8="${FAST_SHARD_ADDRESS8:-0}"
GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}"
if [[ "$FAST_SHARD_ADDRESS8" != 0 && "$FAST_SHARD_ADDRESS8" != 1 ]]; then
  echo "FAST_SHARD_ADDRESS8 must be 0 or 1" >&2
  exit 2
fi
BIN_SUFFIX=""
[[ "$FAST_SHARD_ADDRESS8" == 1 ]] && BIN_SUFFIX="_shardaddr8"
BIN="${BIN:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_n${N}${BIN_SUFFIX}}"

if (( MOD < 2 || MOD > 4294967295 )); then
  echo "HBM32 requires 2 <= modulus <= 4294967295; got $MOD" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found" >&2
  exit 2
fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if (( visible < NGPU )); then
  echo "requested $NGPU GPUs, but only $visible are visible" >&2
  exit 2
fi

if [[ ! -x "$BIN" ]]; then
  echo "$BIN not found; building specialized n=$N binary fast_shard_address8=$FAST_SHARD_ADDRESS8" >&2
  N="$N" FAST_SHARD_ADDRESS8="$FAST_SHARD_ADDRESS8" OUT="$BIN" "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh"
fi

nvidia-smi -L
nvidia-smi topo -m || true
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

echo "N=$N MOD=$MOD GPUs=$NGPU requested_scratch=${TARGET_MIB}MiB reserve=${GRIDFP_VRAM_RESERVE_MIB}MiB max_window=$MAX_WINDOW fast_shard_address8=$FAST_SHARD_ADDRESS8"
echo "BIN=$BIN"
export GRIDFP_VRAM_RESERVE_MIB
exec "$BIN" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU"
