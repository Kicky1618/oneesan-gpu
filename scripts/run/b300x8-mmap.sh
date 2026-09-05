#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-26}"
MOD="${2:-2305843009213693951}"
STORE="${3:-$ONEESAN_ROOT/work/gridfp_n${N}_${MOD}}"
TARGET_MIB="${TARGET_MIB:-245760}"
MAX_WINDOW="${MAX_WINDOW:-14}"
NGPU="${NGPU:-8}"
GRIDFP_RESUME="${GRIDFP_RESUME:-1}"
GRIDFP_FRESH="${GRIDFP_FRESH:-0}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_multigpu_mmap}"

for spec in "N:$N" "MOD:$MOD" "TARGET_MIB:$TARGET_MIB" "MAX_WINDOW:$MAX_WINDOW" "NGPU:$NGPU" "GRIDFP_RESUME:$GRIDFP_RESUME" "GRIDFP_FRESH:$GRIDFP_FRESH"; do
  require_uint "${spec%%:*}" "${spec#*:}" || exit 2
done

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
  ARCH="${ARCH:-native}" OUT="$BIN" "$ONEESAN_ROOT/scripts/build/gridfp-multigpu-mmap.sh"
fi

nvidia-smi -L
nvidia-smi topo -m || true
mkdir -p "$STORE"
echo "N=$N MOD=$MOD GPUs=$NGPU target=${TARGET_MIB}MiB/GPU max_window=$MAX_WINDOW store=$STORE resume=$GRIDFP_RESUME fresh=$GRIDFP_FRESH"
df -h "$STORE" || true
export GRIDFP_RESUME GRIDFP_FRESH
exec "$BIN" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$STORE"
