#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
NGPU="${NGPU:-8}"
VALUES="${VALUES:-67108864}"       # 256 MiB/GPU, power of two
ROUNDS="${ROUNDS:-64}"
LAUNCHES="${LAUNCHES:-16}"
WARMUP="${WARMUP:-2}"
CONFIGS="${CONFIGS:-128:512 256:256 256:512 512:256}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.05}"
RUN_CORRECTNESS="${RUN_CORRECTNESS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_cpasync_remote_peer_bw}"
RESULT="${RESULT:-${PREFIX}.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"

case "$RUN_CORRECTNESS" in 0|1) ;; *) echo "RUN_CORRECTNESS must be 0 or 1" >&2; exit 2;; esac
[[ "$VALUES" =~ ^[0-9]+$ ]] || { echo "VALUES must be integer" >&2; exit 2; }
(( VALUES >= 2 && (VALUES & (VALUES - 1)) == 0 )) || { echo "VALUES must be a power of two" >&2; exit 2; }
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi required" >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

SRC_BOTH="$ONEESAN_ROOT/src/cuda/b300/probes/cpasync_remote_peer_bandwidth.cu"
SRC_MODE="$ONEESAN_ROOT/src/cuda/b300/probes/cpasync_remote_peer_bandwidth_mode.cu"
BIN_BOTH="$ONEESAN_BUILD_DIR/b300_cpasync_remote_peer_bandwidth_both"
BIN_MODE="$ONEESAN_BUILD_DIR/b300_cpasync_remote_peer_bandwidth_mode"

TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v \
  "$SRC_BOTH" -o "$BIN_BOTH" >"$LOGDIR/both.build.out" 2>"$LOGDIR/both.build.err"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v \
  "$SRC_MODE" -o "$BIN_MODE" >"$LOGDIR/mode.build.out" 2>"$LOGDIR/mode.build.err"

if [[ "$RUN_CORRECTNESS" == 1 ]]; then
  "$BIN_BOTH" "$NGPU" 4096 128 16 2 1 0 >"$LOGDIR/correctness.out" 2>"$LOGDIR/correctness.err"
  grep -Fq 'exact_direct_vs_cpasync=1' "$LOGDIR/correctness.out"
fi

field() {
  local key="$1" line="$2"
  sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1
}

sample_process() {
  local pid="$1" out="$2"
  : >"$out"
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>mg)mg=g;if(m>mm)mm=m;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm}' \
      >>"$out" || true
    sleep "$SAMPLE_INTERVAL"
  done
}

printf 'mode\tthreads\tblocks\tmax_ms\tavg_ms\tper_gpu_gbs\taggregate_gbs\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_gpu_util_pct\tmax_memctrl_util_pct\n' >"$RESULT"

run_one() {
  local mode="$1" threads="$2" blocks="$3"
  local tag="${mode}_t${threads}_b${blocks}"
  local so="$LOGDIR/${tag}.out" se="$LOGDIR/${tag}.err" util="$LOGDIR/${tag}.util"
  "$BIN_MODE" "$NGPU" "$VALUES" "$threads" "$blocks" "$ROUNDS" "$LAUNCHES" "$WARMUP" "$mode" \
    >"$so" 2>"$se" &
  local pid=$!
  sample_process "$pid" "$util" & local sampler=$!
  set +e
  wait "$pid"; local rc=$?
  set -e
  wait "$sampler" || true
  (( rc == 0 )) || { echo "$tag failed rc=$rc" >&2; return "$rc"; }

  local line max_ms avg_ms pg ag ug um mg mm
  line="$(grep "^mode=${mode} " "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$tag missing result line" >&2; return 3; }
  max_ms="$(field max_ms "$line")"; avg_ms="$(field avg_ms "$line")"
  pg="$(field per_gpu_gbs "$line")"; ag="$(field aggregate_gbs "$line")"
  read -r ug um mg mm < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$mode" "$threads" "$blocks" "$max_ms" "$avg_ms" "$pg" "$ag" "$ug" "$um" "$mg" "$mm" >>"$RESULT"
}

for cfg in $CONFIGS; do
  IFS=: read -r threads blocks <<<"$cfg"
  [[ "$threads" =~ ^[0-9]+$ && "$blocks" =~ ^[0-9]+$ ]] || { echo "bad CONFIGS entry $cfg" >&2; exit 2; }
  (( threads >= 32 && threads <= 1024 && blocks >= 1 )) || { echo "bad CONFIGS entry $cfg" >&2; exit 2; }
  echo "=== peer random gather direct threads=$threads blocks=$blocks ===" >&2
  run_one direct "$threads" "$blocks"
  echo "=== peer random gather cpasync threads=$threads blocks=$blocks ===" >&2
  run_one cpasync "$threads" "$blocks"
done

cat "$RESULT"
python3 - "$RESULT" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
for r in rows:
    r['max_ms']=float(r['max_ms']); r['aggregate_gbs']=float(r['aggregate_gbs'])
    r['avg_memctrl_util_pct']=None if r['avg_memctrl_util_pct']=='NA' else float(r['avg_memctrl_util_pct'])
for mode in ('direct','cpasync'):
    g=[r for r in rows if r['mode']==mode]
    if not g: continue
    best=min(g,key=lambda r:r['max_ms'])
    print(f'best_{mode}=threads{best["threads"]}_blocks{best["blocks"]}')
    print(f'best_{mode}_max_ms={best["max_ms"]:.6f}')
    print(f'best_{mode}_aggregate_gbs={best["aggregate_gbs"]:.6f}')
    print(f'best_{mode}_avg_memctrl_pct={best["avg_memctrl_util_pct"]}')
by={(r['threads'],r['blocks'],r['mode']):r for r in rows}
for t,b,_ in sorted({(r['threads'],r['blocks'],'') for r in rows}, key=lambda x:(int(x[0]),int(x[1]))):
    a=by.get((t,b,'direct')); c=by.get((t,b,'cpasync'))
    if a and c:
        print(f'cpasync_speedup_t{t}_b{b}={a["max_ms"]/c["max_ms"]:.6f}x')
        if a['avg_memctrl_util_pct'] is not None and c['avg_memctrl_util_pct'] is not None:
            print(f'cpasync_memctrl_delta_t{t}_b{b}={c["avg_memctrl_util_pct"]-a["avg_memctrl_util_pct"]:.6f}pp')
print('traffic=8gpu_peer_ring_random_u32')
print('loads_per_thread_round=14')
print('working_set_exceeds_l2_recommended=1')
PY

echo "b300-cpasync-remote-peer-bandwidth OK arch=$ARCH ngpu=$NGPU values=$VALUES rounds=$ROUNDS launches=$LAUNCHES configs='$CONFIGS' result=$RESULT" >&2
