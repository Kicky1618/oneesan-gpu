#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
MOD="${MOD:-4294967291}"
THREADS="${THREADS:-128 256 512}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-$N}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300x8_fullpull_thread_sweep_n${N}}"
mkdir -p "$LOGDIR"

first=1
printf 'threads\twall_s\tactive_max_s\tactive_sum_s\n'
for t in $THREADS; do
  if (( t < 32 || t > 1024 || t % 32 != 0 )); then
    echo "invalid thread count: $t" >&2
    exit 2
  fi
  log="$LOGDIR/t${t}.log"
  echo "=== B300 x8 full-pull one-row threads=$t ===" >&2
  rebuild=0
  if (( first )); then rebuild=1; first=0; fi
  B300_ROW_LIMIT=1 GRIDFP_THREADS="$t" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
  REBUILD="$rebuild" FAST_SHARD_ADDRESS8=1 MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 \
  "$ONEESAN_ROOT/scripts/run/b300x8.sh" "$N" "$MOD" 2>&1 | tee "$log"
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$log" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "missing backend result for threads=$t" >&2; exit 3; }
  wall="$(sed -nE 's/.* wall_s=([^ ]+).*/\1/p' <<<"$line")"
  active_max="$(sed -nE 's/.* active_max_s=([^ ]+).*/\1/p' <<<"$line")"
  active_sum="$(sed -nE 's/.* active_sum_s=([^ ]+).*/\1/p' <<<"$line")"
  printf '%s\t%s\t%s\t%s\n' "$t" "$wall" "$active_max" "$active_sum"
done

echo "logs=$LOGDIR" >&2
