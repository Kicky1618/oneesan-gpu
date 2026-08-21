#!/usr/bin/env bash
set -euo pipefail
N="${1:-26}"
MOD="${2:-2305843009213693951}"
STORE="${3:-/workspace/gridfp_n${N}_${MOD}}"
TARGET_MIB="${TARGET_MIB:-245760}"   # 240 GiB/GPU, leaves B300 headroom
MAX_WINDOW="${MAX_WINDOW:-14}"
NGPU="${NGPU:-8}"
BIN="${BIN:-./oneesan_cuda_gridfp_multigpu}"

command -v nvidia-smi >/dev/null && nvidia-smi -L
command -v nvidia-smi >/dev/null && nvidia-smi topo -m || true
mkdir -p "$STORE"
echo "N=$N MOD=$MOD GPUs=$NGPU target=${TARGET_MIB}MiB/GPU max_window=$MAX_WINDOW store=$STORE"
df -h "$STORE" || true
exec "$BIN" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$STORE"
