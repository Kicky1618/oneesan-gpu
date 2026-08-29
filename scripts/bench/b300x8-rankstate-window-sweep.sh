#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${N:-27}";MOD="${MOD:-4294967291}";TARGET_MIB="${TARGET_MIB:-65536}";PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
WINDOWS="${WINDOWS:-14 18 22 27}";THREADS="${GRIDFP_THREADS:-256}";SAMPLE_S="${SAMPLE_S:-0.25}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300x8_rankstate_window_n${N}}";mkdir -p "$LOGDIR"
BIN="${BIN:-$LOGDIR/rankstate_ilp2_cio}"
command -v nvidia-smi >/dev/null || exit 2

if [[ ! -x "$BIN" || "${REBUILD:-0}" == 1 ]]; then
  N="$N" OUT="$BIN" FAST_SHARD_ADDRESS8=1 MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 MAIN_PULL_ILP2=0 HEIGHT_CACHE=0 RANK_DELTA_CACHE=1 RANK_STATE_PACKED=1 RANK_STATE_ILP2=1 CONCURRENT_GROUP_IO=1 PTXAS_VERBOSE=1 "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$LOGDIR/build.log" 2>&1
fi

printf 'max_window\twall_s\tactive_max_s\twindows\tmax_groups\tavg_gpu_pct\tavg_memctrl_pct\tpeak_memctrl_pct\trank_delta_fallback_groups\n'
for w in $WINDOWS;do
  (( w>=2 && w<=N ))||{ echo "bad max_window=$w" >&2;exit 2; }
  log="$LOGDIR/w${w}.run.log";tele="$LOGDIR/w${w}.util.csv";:>"$tele"
  (B300_ROW_LIMIT=1 GRIDFP_THREADS="$THREADS" GRIDFP_VRAM_RESERVE_MIB=8192 GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" GRIDFP_PLAN_TARGET_DIVISOR=1 "$BIN" "$N" "$MOD" "$TARGET_MIB" "$w" 8)>"$log" 2>&1&pid=$!
  while kill -0 "$pid" 2>/dev/null;do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null|awk -F, '{print $1+0 "," $2+0}'>>"$tele"||true;sleep "$SAMPLE_S";done
  wait "$pid"
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$log"|tail -n1)";[[ -n "$line" ]]||exit 3
  f(){ sed -nE "s/.* $1=([^ ]+).*/\\1/p"<<<"$line"; }
  wall="$(f wall_s)";active="$(f active_max_s)";wins="$(f windows)";groups="$(f max_groups)";rf="$(f rank_delta_fallback_groups)";rf="${rf:-0}"
  read -r gu mu peak < <(awk -F, '{sg+=$1;sm+=$2;n++;if($2>p)p=$2}END{if(n)printf "%.3f %.3f %.0f\n",sg/n,sm/n,p;else print "nan nan nan"}' "$tele")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$w" "$wall" "$active" "$wins" "$groups" "$gu" "$mu" "$peak" "$rf"
done
echo "logs=$LOGDIR" >&2
