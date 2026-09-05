#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-sm_86}"
NGPU="${NGPU:-1}"
ROWS="${ROWS:-1}"
TARGET_MIB="${TARGET_MIB:-1024}"
MAX_WINDOW="${MAX_WINDOW:-14}"
MOD="${MOD:-4294967291}"
ILP8_THRESHOLDS="${ILP8_THRESHOLDS:-0 262144 1048576 4194304}"
THREADS_LIST="${THREADS_LIST:-128 256}"
REPEATS="${REPEATS:-1}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_local_sm86_hybrid8_sweep_r${ROWS}_g${NGPU}}"

[[ "$NGPU" =~ ^[1-8]$ ]] || { echo 'NGPU must be 1..8' >&2; exit 2; }
[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$TARGET_MIB" =~ ^[1-9][0-9]*$ ]] || { echo 'TARGET_MIB must be positive' >&2; exit 2; }

exec env \
  ARCH="$ARCH" NGPU="$NGPU" ROWS="$ROWS" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" MOD="$MOD" \
  ILP8_THRESHOLDS="$ILP8_THRESHOLDS" THREADS_LIST="$THREADS_LIST" REPEATS="$REPEATS" SAMPLE_INTERVAL="$SAMPLE_INTERVAL" \
  PREFIX="$PREFIX" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid-ilp8-sweep.sh"
