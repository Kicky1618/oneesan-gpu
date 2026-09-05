#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
MOD="${MOD:-4294967291}"
THREADS="${GRIDFP_THREADS:-256}"
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
SAMPLE_S="${SAMPLE_S:-0.5}"
VARIANTS="${VARIANTS:-base ilp2 height rankdelta rankstate rankstate_ilp2}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300x8_memory_path_sweep_n${N}}"
mkdir -p "$LOGDIR"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }

build_variant(){
  local name="$1" ilp=0 height=0 rank=0 packed=0 rsilp=0 divisor=1
  case "$name" in
    base) ;;
    ilp2) ilp=1 ;;
    height) height=1 ;;
    rankdelta) rank=1 ;;
    rankstate) rank=1;packed=1 ;;
    rankstate_ilp2) rank=1;packed=1;rsilp=1 ;;
    *) echo "unknown variant: $name" >&2; return 2 ;;
  esac
  local bin="$LOGDIR/n${N}_${name}"
  echo "=== build variant=$name ===" >&2
  N="$N" OUT="$bin" FAST_SHARD_ADDRESS8=1 MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 \
    MAIN_PULL_ILP2="$ilp" HEIGHT_CACHE="$height" RANK_DELTA_CACHE="$rank" RANK_STATE_PACKED="$packed" RANK_STATE_ILP2="$rsilp" PTXAS_VERBOSE=1 \
    "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$LOGDIR/${name}.build.log" 2>&1
  printf '%s %s\n' "$bin" "$divisor"
}

sample_run(){
  local name="$1" bin="$2" divisor="$3"
  local log="$LOGDIR/${name}.run.log" util="$LOGDIR/${name}.memory.csv"
  : >"$util"
  echo "=== run variant=$name row=1 threads=$THREADS ===" >&2
  (
    export B300_ROW_LIMIT=1 GRIDFP_THREADS="$THREADS" GRIDFP_VRAM_RESERVE_MIB=8192
    export GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" GRIDFP_PLAN_TARGET_DIVISOR="$divisor"
    "$bin" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8
  ) >"$log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -v ts="$(date +%s.%N)" '{gsub(/ /,""); if($1~/^[0-9]+$/) print ts "," $1}' >>"$util" || true
    sleep "$SAMPLE_S"
  done
  wait "$pid"
  local line
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$log" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "missing backend result variant=$name; see $log" >&2; return 3; }
  local wall active avg max samples rg rf
  wall="$(sed -nE 's/.* wall_s=([^ ]+).*/\1/p' <<<"$line")"
  active="$(sed -nE 's/.* active_max_s=([^ ]+).*/\1/p' <<<"$line")"
  rg="$(sed -nE 's/.* rank_delta_groups=([^ ]+).*/\1/p' <<<"$line")"; rg="${rg:-0}"
  rf="$(sed -nE 's/.* rank_delta_fallback_groups=([^ ]+).*/\1/p' <<<"$line")"; rf="${rf:-0}"
  read -r avg max samples < <(awk -F, 'BEGIN{s=0;n=0;m=0}{v=$2+0;s+=v;n++;if(v>m)m=v}END{if(n)printf "%.3f %.0f %d\n",s/n,m,n;else print "nan nan 0"}' "$util")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$wall" "$active" "$avg" "$max" "$samples" "$rg" "$rf"
}

printf 'variant\twall_s\tactive_max_s\tmemory_util_avg_pct\tmemory_util_max_pct\tsamples_gpu\trank_delta_groups\trank_delta_fallback_groups\n'
for name in $VARIANTS; do
  read -r bin divisor < <(build_variant "$name")
  sample_run "$name" "$bin" "$divisor"
done

echo "logs=$LOGDIR" >&2
