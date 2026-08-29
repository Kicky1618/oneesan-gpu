#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N=27
MOD="${MOD:-4294967291}"
SCRATCH_MIB="${SCRATCH_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-27}"
GPUS=8
ARCH="${ARCH:-native}"
BACKEND="${BACKEND:-vmm}"
case "$BACKEND" in vmm|sharded) ;; *) echo "BACKEND must be vmm or sharded" >&2; exit 2;; esac
OUT="${OUT:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_n27_x8_bandwidth_recovery_${BACKEND}}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_hbm32_n27_x8_bandwidth_recovery_${BACKEND}}"
BUILD_OUT="${BUILD_OUT:-${PREFIX}.build.out}"
BUILD_ERR="${BUILD_ERR:-${PREFIX}.build.err}"
RUN_OUT="${RUN_OUT:-${PREFIX}.out}"
RUN_ERR="${RUN_ERR:-${PREFIX}.err}"
TELEMETRY="${TELEMETRY:-${PREFIX}.gpu.csv}"
mkdir -p "$(dirname "$OUT")" "$(dirname "$RUN_OUT")"

command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= GPUS )) || { echo "need $GPUS GPUs, visible=$visible" >&2; exit 2; }

# The current ~2% memory-controller symptom is compute/control bound. This
# candidate deliberately trades HBM traffic for less integer/control work:
#   * K=13 rank/unrank LUTs
#   * materialized MateID reused by the main transition kernels
#   * p>1 destination-pull main update: no identity D2D copy, no main CAS scatter,
#     and no blocked->main scatter kernel
#   * VMM direct-global authoritative addressing by default
#   * wide scratch request so large windows stay resident; the executable caps
#     the request against cudaMemGetInfo()-reserve.
echo "=== build B300 n=27 bandwidth-recovery candidate backend=$BACKEND ===" >&2
if [[ "$BACKEND" == vmm ]]; then
  N="$N" ARCH="$ARCH" OUT="$OUT" MAIN_MATE_CACHE=1 MAIN_PULL=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm.sh" >"$BUILD_OUT" 2>"$BUILD_ERR"
  grep -Fq 'main_mate_cache=1 main_pull=1' "$BUILD_OUT" || {
    echo "VMM bandwidth-recovery build did not report MAIN_MATE_CACHE=1 MAIN_PULL=1" >&2
    cat "$BUILD_OUT" >&2; exit 3; }
  grep -Fq 'authoritative_storage=contiguous_multi_gpu_vmm' "$BUILD_OUT" || {
    echo "VMM bandwidth-recovery build lost contiguous VMM storage" >&2; exit 3; }
else
  N="$N" ARCH="$ARCH" OUT="$OUT" FAST_SHARD_ADDRESS8=1 MAIN_MATE_CACHE=1 MAIN_PULL=1 PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$BUILD_OUT" 2>"$BUILD_ERR"
  grep -Fq 'main_mate_cache=1 main_pull=1' "$BUILD_OUT" || {
    echo "sharded bandwidth-recovery build did not report MAIN_MATE_CACHE=1 MAIN_PULL=1" >&2
    cat "$BUILD_OUT" >&2; exit 3; }
fi

: >"$TELEMETRY"
monitor_pid=""
cleanup_monitor(){
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    monitor_pid=""
  fi
}
trap cleanup_monitor EXIT INT TERM

# utilization.memory is the percentage of the sample interval during which the
# memory controller is busy; this is the metric that exposed the ~2% problem.
nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw,memory.used,memory.total \
  --format=csv,noheader,nounits -l 1 >"$TELEMETRY" 2>/dev/null &
monitor_pid=$!
sleep 1

echo "=== run B300 x8 n=27 bandwidth-recovery candidate ===" >&2
echo "backend=$BACKEND scratch_mib=$SCRATCH_MIB max_window=$MAX_WINDOW gpus=$GPUS telemetry=$TELEMETRY" >&2
set +e
"$OUT" "$N" "$MOD" "$SCRATCH_MIB" "$MAX_WINDOW" "$GPUS" >"$RUN_OUT" 2>"$RUN_ERR"
rc=$?
set -e
cleanup_monitor

python3 - "$TELEMETRY" <<'PY'
import csv, sys
from collections import defaultdict
p=sys.argv[1]
rows=defaultdict(list)
with open(p,newline='') as f:
    for r in csv.reader(f):
        if len(r)<7: continue
        try:
            gpu=int(r[1].strip()); sm=float(r[2].strip()); mem=float(r[3].strip()); power=float(r[4].strip())
        except ValueError: continue
        rows[gpu].append((sm,mem,power))
if not rows:
    print('b300_bandwidth_recovery_telemetry_samples=0'); raise SystemExit(0)
all_mem=[]; all_sm=[]
for gpu in sorted(rows):
    xs=rows[gpu]; sm=[x[0] for x in xs]; mem=[x[1] for x in xs]; power=[x[2] for x in xs]
    all_mem += mem; all_sm += sm
    print(f'b300_bandwidth_recovery_gpu{gpu}_samples={len(xs)}')
    print(f'b300_bandwidth_recovery_gpu{gpu}_sm_avg_pct={sum(sm)/len(sm):.3f}')
    print(f'b300_bandwidth_recovery_gpu{gpu}_sm_max_pct={max(sm):.3f}')
    print(f'b300_bandwidth_recovery_gpu{gpu}_memctl_avg_pct={sum(mem)/len(mem):.3f}')
    print(f'b300_bandwidth_recovery_gpu{gpu}_memctl_max_pct={max(mem):.3f}')
    print(f'b300_bandwidth_recovery_gpu{gpu}_power_avg_w={sum(power)/len(power):.3f}')
print(f'b300_bandwidth_recovery_allgpu_memctl_avg_pct={sum(all_mem)/len(all_mem):.3f}')
print(f'b300_bandwidth_recovery_allgpu_memctl_max_pct={max(all_mem):.3f}')
print(f'b300_bandwidth_recovery_allgpu_sm_avg_pct={sum(all_sm)/len(all_sm):.3f}')
PY

printf 'b300_bandwidth_recovery_backend=%s\n' "$BACKEND"
printf 'b300_bandwidth_recovery_exit_code=%s\n' "$rc"
printf 'b300_bandwidth_recovery_binary=%s\n' "$OUT"
printf 'b300_bandwidth_recovery_stdout=%s\n' "$RUN_OUT"
printf 'b300_bandwidth_recovery_stderr=%s\n' "$RUN_ERR"
printf 'b300_bandwidth_recovery_telemetry=%s\n' "$TELEMETRY"
if (( rc != 0 )); then
  echo "bandwidth-recovery candidate exited rc=$rc" >&2
  tail -n 80 "$RUN_ERR" >&2 || true
  exit "$rc"
fi
