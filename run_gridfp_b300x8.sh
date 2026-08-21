#!/usr/bin/env bash
set -euo pipefail

N="${1:-27}"
MOD="${2:-4294967291}"
TARGET_MIB="${TARGET_MIB:-16384}"       # requested scratch cap; solver auto-caps to free HBM
MAX_WINDOW="${MAX_WINDOW:-14}"
NGPU="${NGPU:-8}"
GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}"
BIN="${BIN:-./oneesan_cuda_gridfp_b300_hbm32_n${N}}"

if (( MOD < 2 || MOD > 4294967295 )); then
  echo "HBM32 requires 2 <= modulus <= 4294967295; got $MOD" >&2
  exit 2
fi

if [[ ! -x "$BIN" ]]; then
  echo "$BIN not found; building specialized n=$N binary" >&2
  N="$N" OUT="${BIN#./}" ./build_gridfp_b300_hbm32.sh
fi

command -v nvidia-smi >/dev/null && nvidia-smi -L
command -v nvidia-smi >/dev/null && nvidia-smi topo -m || true
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

echo "N=$N MOD=$MOD GPUs=$NGPU requested_scratch=${TARGET_MIB}MiB reserve=${GRIDFP_VRAM_RESERVE_MIB}MiB max_window=$MAX_WINDOW bin=$BIN"
export GRIDFP_VRAM_RESERVE_MIB
exec "$BIN" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU"
