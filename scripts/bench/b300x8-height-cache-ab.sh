#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
MOD="${MOD:-4294967291}"
THREADS="${GRIDFP_THREADS:-256}"
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300x8_height_cache_ab_n${N}}"
mkdir -p "$LOGDIR"

printf 'height_cache\twall_s\tactive_max_s\tactive_sum_s\n'
for mode in 0 1; do
  log="$LOGDIR/height${mode}.log"
  echo "=== B300 x8 height-cache=$mode one-row ===" >&2
  ROWS=1 GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
  HEIGHT_CACHE="$mode" RANK_DELTA_CACHE=0 REBUILD=1 \
  "$ONEESAN_ROOT/scripts/run/b300x8.sh" "$N" "$MOD" 2>&1 | tee "$log"
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$log" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "missing backend result height_cache=$mode" >&2; exit 3; }
  get(){ sed -nE "s/.* $1=([^ ]+).*/\\1/p" <<<"$line"; }
  printf '%s\t%s\t%s\t%s\n' "$mode" "$(get wall_s)" "$(get active_max_s)" "$(get active_sum_s)"
done

echo "logs=$LOGDIR" >&2
