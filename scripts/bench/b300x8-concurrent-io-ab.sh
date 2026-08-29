#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${N:-27}";MOD="${MOD:-4294967291}";THREADS="${GRIDFP_THREADS:-256}"
TARGET_MIB="${TARGET_MIB:-65536}";PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}";MAX_WINDOW="${MAX_WINDOW:-14}"
SAMPLE_S="${SAMPLE_S:-0.25}";LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300x8_concurrent_io_ab_n${N}}";mkdir -p "$LOGDIR"
command -v nvidia-smi >/dev/null || exit 2

build(){ local cio="$1" name="$2" bin="$LOGDIR/$name";N="$N" OUT="$bin" FAST_SHARD_ADDRESS8=1 MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 MAIN_PULL_ILP2=0 HEIGHT_CACHE=0 RANK_DELTA_CACHE=1 RANK_STATE_PACKED=1 RANK_STATE_ILP2=1 CONCURRENT_GROUP_IO="$cio" PTXAS_VERBOSE=1 "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$LOGDIR/$name.build.log" 2>&1;echo "$bin"; }
run(){ local name="$1" bin="$2" log="$LOGDIR/$name.run.log" tele="$LOGDIR/$name.mem.csv";:>"$tele";(B300_ROW_LIMIT=1 GRIDFP_THREADS="$THREADS" GRIDFP_VRAM_RESERVE_MIB=8192 GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" GRIDFP_PLAN_TARGET_DIVISOR=1 "$bin" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8)>"$log" 2>&1&pid=$!;while kill -0 "$pid" 2>/dev/null;do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null|awk -F, '{g=$1+0;m=$2+0;print g "," m}'>>"$tele"||true;sleep "$SAMPLE_S";done;wait "$pid";line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$log"|tail -n1)";wall="$(sed -nE 's/.* wall_s=([^ ]+).*/\1/p'<<<"$line")";active="$(sed -nE 's/.* active_max_s=([^ ]+).*/\1/p'<<<"$line")";read -r gu mu peak < <(awk -F, '{sg+=$1;sm+=$2;n++;if($2>p)p=$2}END{if(n)printf "%.3f %.3f %.0f\n",sg/n,sm/n,p;else print "nan nan nan"}' "$tele");printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$wall" "$active" "$gu" "$mu" "$peak";}
serial="$(build 0 serial_io)";concurrent="$(build 1 concurrent_io)"
printf 'variant\twall_s\tactive_max_s\tavg_gpu_pct\tavg_memctrl_pct\tpeak_memctrl_pct\n'
run serial_io "$serial";run concurrent_io "$concurrent"
echo "logs=$LOGDIR" >&2
