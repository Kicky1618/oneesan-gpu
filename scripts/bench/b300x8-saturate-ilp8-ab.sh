#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
MOD="${MOD:-4294967291}"
ROWS="${ROWS:-1}"
THREADS="${GRIDFP_THREADS:-256}"
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
RANDOM_CG="${RANDOM_CG:-1}"
WARP_SCAN="${WARP_SCAN:-1}"
ILP8_REGS="${ILP8_REGS:-0 96 128 160}"
REBUILD="${REBUILD:-0}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300x8_saturate_ilp8_ab_n${N}}"
mkdir -p "$LOGDIR"

command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }

run_one(){
  local name="$1";shift
  local log="$LOGDIR/${name}.log" dmon="$LOGDIR/${name}.dmon"
  echo "=== $name ===" >&2
  : >"$dmon"
  nvidia-smi dmon -s u -d 1 >"$dmon" 2>&1 &
  local mpid=$!
  set +e
  "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  kill "$mpid" 2>/dev/null || true
  wait "$mpid" 2>/dev/null || true
  (( rc==0 )) || return "$rc"

  local line wall active_max active_sum mem_avg mem_max
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$log" | tail -n1 || true)"
  [[ -n "$line" ]] || line="$(grep '^backend=gridfp-b300-hbm32' "$log" | tail -n1 || true)"
  wall="$(sed -nE 's/.* wall_s=([^ ]+).*/\1/p' <<<"$line")"
  active_max="$(sed -nE 's/.* active_max_s=([^ ]+).*/\1/p' <<<"$line")"
  active_sum="$(sed -nE 's/.* active_sum_s=([^ ]+).*/\1/p' <<<"$line")"
  read -r mem_avg mem_max < <(awk '
    $1 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ {s+=$3;n++;if($3>m)m=$3}
    END{if(n)printf "%.3f %d\n",s/n,m;else print "nan nan"}
  ' "$dmon")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "${wall:-nan}" "${active_max:-nan}" "${active_sum:-nan}" "$mem_avg" "$mem_max"
}

printf 'profile\twall_s\tactive_max_s\tactive_sum_s\tmc_avg_pct\tmc_max_pct\n'

run_one ilp4 \
  env ROWS="$ROWS" GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
      RANDOM_CG="$RANDOM_CG" WARP_SCAN="$WARP_SCAN" REBUILD="$REBUILD" \
  "$ONEESAN_ROOT/scripts/run/b300x8-saturate-dualmask.sh" "$N" "$MOD"

for r in $ILP8_REGS; do
  run_one "ilp8_r${r}" \
    env ROWS="$ROWS" GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
        RANDOM_CG="$RANDOM_CG" WARP_SCAN="$WARP_SCAN" MAXRREGCOUNT="$r" REBUILD="$REBUILD" \
    "$ONEESAN_ROOT/scripts/run/b300x8-saturate-ilp8.sh" "$N" "$MOD"
done

echo "logs=$LOGDIR" >&2
