#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
MOD="${MOD:-4294967291}"
THREADS="${GRIDFP_THREADS:-256}"
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300x8_rank_delta_ab_n${N}}"
mkdir -p "$LOGDIR"

printf 'rank_delta\twall_s\tactive_max_s\tactive_sum_s\tcache_groups\tfallback_groups\n'
for mode in 0 1; do
  log="$LOGDIR/rankdelta${mode}.log"
  echo "=== B300 x8 rank-delta=$mode one-row ===" >&2
  ROWS=1 GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
  RANK_DELTA_CACHE="$mode" REBUILD=1 \
  "$ONEESAN_ROOT/scripts/run/b300x8.sh" "$N" "$MOD" 2>&1 | tee "$log"
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$log" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "missing backend result rank_delta=$mode" >&2; exit 3; }
  get(){ sed -nE "s/.* $1=([^ ]+).*/\\1/p" <<<"$line"; }
  wall="$(get wall_s)";amax="$(get active_max_s)";asum="$(get active_sum_s)"
  groups="-";fallback="-"
  if [[ "$mode" == 1 ]]; then
    groups="$(get rank_delta_groups)";fallback="$(get rank_delta_fallback_groups)"
    [[ -n "$groups" && -n "$fallback" ]] || { echo "rank-delta coverage counters missing" >&2; exit 4; }
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$wall" "$amax" "$asum" "$groups" "$fallback"
done

echo "logs=$LOGDIR" >&2
