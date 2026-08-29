#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
MOD="${MOD:-4294967291}"
THREADS="${GRIDFP_THREADS:-256}"
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
SAMPLE_S="${SAMPLE_S:-0.25}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300x8_rankstate_ilp2_ab_n${N}}"
mkdir -p "$LOGDIR"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }

build_one(){
  local ilp="$1" name="$2" bin="$LOGDIR/$name"
  N="$N" OUT="$bin" FAST_SHARD_ADDRESS8=1 MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 \
    MAIN_PULL_ILP2=0 HEIGHT_CACHE=0 RANK_DELTA_CACHE=1 RANK_STATE_PACKED=1 RANK_STATE_ILP2="$ilp" PTXAS_VERBOSE=1 \
    "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$LOGDIR/$name.build.log" 2>&1
  echo "$bin"
}

run_one(){
  local name="$1" bin="$2" log="$LOGDIR/$name.run.log" util="$LOGDIR/$name.mem.csv"
  : >"$util"
  (
    export B300_ROW_LIMIT=1 GRIDFP_THREADS="$THREADS" GRIDFP_VRAM_RESERVE_MIB=8192
    export GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" GRIDFP_PLAN_TARGET_DIVISOR=1
    "$bin" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8
  ) >"$log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -v ts="$(date +%s.%N)" '{gsub(/ /,"");if($1~/^[0-9]+$/)print ts "," $1}' >>"$util" || true
    sleep "$SAMPLE_S"
  done
  wait "$pid"
  local line wall active avg peak rg rf
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$log" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "missing backend line: $log" >&2; return 3; }
  wall="$(sed -nE 's/.* wall_s=([^ ]+).*/\1/p' <<<"$line")"
  active="$(sed -nE 's/.* active_max_s=([^ ]+).*/\1/p' <<<"$line")"
  rg="$(sed -nE 's/.* rank_delta_groups=([^ ]+).*/\1/p' <<<"$line")";rg="${rg:-0}"
  rf="$(sed -nE 's/.* rank_delta_fallback_groups=([^ ]+).*/\1/p' <<<"$line")";rf="${rf:-0}"
  read -r avg peak < <(awk -F, 'BEGIN{s=0;n=0;m=0}{v=$2+0;s+=v;n++;if(v>m)m=v}END{if(n)printf "%.3f %.0f\n",s/n,m;else print "nan nan"}' "$util")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$wall" "$active" "$avg" "$peak" "$rg" "$rf"
}

base="$(build_one 0 packed56)"
ilp="$(build_one 1 packed56_ilp2)"
printf 'variant\twall_s\tactive_max_s\tmemory_util_avg_pct\tmemory_util_peak_pct\trank_delta_groups\trank_delta_fallback_groups\n'
run_one packed56 "$base"
run_one packed56_ilp2 "$ilp"
echo "logs=$LOGDIR" >&2
