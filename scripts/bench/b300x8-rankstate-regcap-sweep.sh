#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
MOD="${MOD:-4294967291}"
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
CAPS="${CAPS:-0 96 128 160 192}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
SAMPLE_S="${SAMPLE_S:-0.25}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300x8_rankstate_regcap_n${N}}"
mkdir -p "$LOGDIR"
command -v nvcc >/dev/null || { echo "nvcc not found" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }

parse_ptxas(){
  python3 - "$1" <<'PY'
import re,sys
s=open(sys.argv[1],errors='ignore').read()
regs=[int(x) for x in re.findall(r'Used\s+(\d+)\s+registers',s)]
sp=[tuple(map(int,x)) for x in re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',s)]
print(max(regs) if regs else -1, sum(a for a,b in sp), sum(b for a,b in sp))
PY
}

sample_mem(){
  local pid="$1" file="$2"; : >"$file"
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F, -v ts="$(date +%s.%N)" '{g=$1+0;m=$2+0;print ts "," g "," m}' >>"$file" || true
    sleep "$SAMPLE_S"
  done
}

printf 'cap\tthreads\twall_s\tactive_max_s\tavg_gpu_pct\tavg_memctrl_pct\tpeak_memctrl_pct\tmax_regs\tspill_store_bytes\tspill_load_bytes\trank_delta_groups\trank_delta_fallback_groups\n'
for cap in $CAPS; do
  [[ "$cap" =~ ^[0-9]+$ ]] || { echo "bad cap=$cap" >&2; exit 2; }
  bin="$LOGDIR/n${N}_cap${cap}"
  build="$LOGDIR/cap${cap}.build.log"
  echo "=== build packed56+ILP2 cap=$cap ===" >&2
  N="$N" OUT="$bin" FAST_SHARD_ADDRESS8=1 MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 \
    MAIN_PULL_ILP2=0 HEIGHT_CACHE=0 RANK_DELTA_CACHE=1 RANK_STATE_PACKED=1 RANK_STATE_ILP2=1 \
    MAXRREGCOUNT="$cap" PTXAS_VERBOSE=1 "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$build" 2>&1
  read -r regs ss sl < <(parse_ptxas "$build")
  for th in $THREADS_LIST; do
    (( th>=32 && th<=1024 && th%32==0 )) || { echo "bad threads=$th" >&2; exit 2; }
    tag="cap${cap}_t${th}"; log="$LOGDIR/$tag.run.log"; util="$LOGDIR/$tag.util.csv"
    echo "=== run cap=$cap threads=$th row=1 ===" >&2
    (
      export B300_ROW_LIMIT=1 GRIDFP_THREADS="$th" GRIDFP_VRAM_RESERVE_MIB=8192
      export GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" GRIDFP_PLAN_TARGET_DIVISOR=1
      "$bin" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8
    ) >"$log" 2>&1 &
    pid=$!; sample_mem "$pid" "$util" & sampler=$!
    set +e; wait "$pid"; rc=$?; set -e; wait "$sampler" || true
    (( rc==0 )) || { echo "run failed cap=$cap threads=$th rc=$rc log=$log" >&2; exit "$rc"; }
    line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$log" | tail -n1 || true)"
    [[ -n "$line" ]] || { echo "missing backend line: $log" >&2; exit 3; }
    wall="$(sed -nE 's/.* wall_s=([^ ]+).*/\1/p' <<<"$line")"
    active="$(sed -nE 's/.* active_max_s=([^ ]+).*/\1/p' <<<"$line")"
    rg="$(sed -nE 's/.* rank_delta_groups=([^ ]+).*/\1/p' <<<"$line")"; rg="${rg:-0}"
    rf="$(sed -nE 's/.* rank_delta_fallback_groups=([^ ]+).*/\1/p' <<<"$line")"; rf="${rf:-0}"
    read -r gu mu mp < <(awk -F, 'BEGIN{sg=sm=n=pm=0}{g=$2+0;m=$3+0;sg+=g;sm+=m;n++;if(m>pm)pm=m}END{if(n)printf "%.3f %.3f %.0f\n",sg/n,sm/n,pm;else print "nan nan nan"}' "$util")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cap" "$th" "$wall" "$active" "$gu" "$mu" "$mp" "$regs" "$ss" "$sl" "$rg" "$rf"
  done
done

echo "logs=$LOGDIR" >&2
