#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"; ARCH="${ARCH:-native}"
THREADS="${BUCKET_THREADS:-256}"; HIGH_GX="${BUCKET_HIGH_GRID_X:-32}"; HIGH_GY="${BUCKET_HIGH_GRID_Y:-8}"
LOW_GX="${BUCKET_LOW_GRID_X:-16}"; LOW_GY="${BUCKET_LOW_GRID_Y:-8}"
OUT="${OUT:-$ONEESAN_BUILD_DIR/b300_ncu_high_n${N}}"; PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_ncu_high}"
CSV="${CSV:-${PREFIX}.csv}"; REPORT="${REPORT:-${PREFIX}.ncu-rep}"; QUERY="${QUERY:-${PREFIX}.metrics.txt}"
command -v ncu >/dev/null || { echo "ncu not found" >&2; exit 2; }
command -v nvcc >/dev/null || { echo "nvcc not found" >&2; exit 2; }

if [[ "${BUILD:-1}" == 1 ]]; then
  N="$N" ARCH="$ARCH" OUT="$OUT" COL_ILP="${COL_ILP:-2}" DEPTHMAJOR=1 PAIR_MLP="${PAIR_MLP:-1}" MLP_WINDOW4="${MLP_WINDOW4:-1}" PREFETCH_NEXT="${PREFETCH_NEXT:-0}" PM_ACCUM="${PM_ACCUM:-1}" MAXRREGCOUNT="${MAXRREGCOUNT:-0}" PTXAS_VERBOSE=1 bash "$ONEESAN_ROOT/scripts/build/b300-rankformula-hbm-max.sh"
fi

ncu --query-metrics --query-metrics-mode suffix --devices 0 >"$QUERY" 2>/dev/null || ncu --query-metrics --devices 0 >"$QUERY"
candidates=(
  gpu__time_duration.sum
  dram__throughput.avg.pct_of_peak_sustained_elapsed
  lts__throughput.avg.pct_of_peak_sustained_elapsed
  l1tex__throughput.avg.pct_of_peak_sustained_elapsed
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  sm__warps_active.avg.pct_of_peak_sustained_active
  smsp__issue_active.avg.pct_of_peak_sustained_active
  smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
  smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
  smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
  smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
  smsp__warp_issue_stalled_not_selected_per_warp_active.pct
  smsp__warp_issue_stalled_wait_per_warp_active.pct
  dram__bytes_read.sum.per_second
  dram__bytes_write.sum.per_second
  lts__t_sectors_srcunit_tex_op_read.sum.per_second
)
metrics=()
for m in "${candidates[@]}"; do grep -Fq "$m" "$QUERY" && metrics+=("$m"); done
((${#metrics[@]}>=5)) || { echo "too few requested metrics available (${#metrics[@]}), see $QUERY" >&2; exit 3; }
metric_csv="$(IFS=,;echo "${metrics[*]}")"
echo "NCU metrics=$metric_csv" >&2
rm -f "$REPORT" "$CSV"
set +e
BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" \
BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" BUCKET_HIGH_GRID_X="$HIGH_GX" BUCKET_HIGH_GRID_Y="$HIGH_GY" \
ncu --devices 0 --filter-mode global --kernel-name-base function \
    --kernel-name regex:".*high.*rankformula_nometa4_abstract_kernel" \
    --launch-count 1 --replay-mode application --apply-rules no \
    --metrics "$metric_csv" --page raw --csv --export "$REPORT" \
    "$OUT" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$CSV" 2>"${PREFIX}.err"
rc=$?
set -e
((rc==0)) || { echo "ncu failed rc=$rc stderr=${PREFIX}.err" >&2; exit "$rc"; }

python3 - "$CSV" <<'PY'
import csv,sys,re
path=sys.argv[1]
rows=[]
with open(path,errors='ignore') as f:
    # ncu may prepend non-CSV status lines; start at the header containing Metric Name.
    lines=f.readlines()
start=next((i for i,x in enumerate(lines) if 'Metric Name' in x and 'Metric Value' in x),None)
if start is None:
    print('NCU_SUMMARY parse_failed; inspect',path);raise SystemExit(0)
r=list(csv.DictReader(lines[start:]))
for x in r:
    name=x.get('Metric Name','').strip(); val=x.get('Metric Value','').strip(); unit=x.get('Metric Unit','').strip()
    if name: rows.append((name,val,unit))
print('NCU_HIGH_BOTTLENECK')
for name,val,unit in rows: print(f'{name}={val}{unit}')
# Coarse automatic interpretation only when the standard metrics are present.
d={n:(v,u) for n,v,u in rows}
def num(n):
    try:return float(d[n][0].replace(',',''))
    except:return None
dram=num('dram__throughput.avg.pct_of_peak_sustained_elapsed')
l2=num('lts__throughput.avg.pct_of_peak_sustained_elapsed')
occ=num('sm__warps_active.avg.pct_of_peak_sustained_active')
score=num('smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct')
issue=num('smsp__issue_active.avg.pct_of_peak_sustained_active')
if dram is not None and l2 is not None:
    if dram < 20 and l2 >= 60: print('diagnosis=L2_or_cache_fabric_bound_not_HBM')
    elif dram < 20 and score is not None and score >= 30: print('diagnosis=memory_latency_or_insufficient_MLP')
    elif dram >= 60: print('diagnosis=HBM_bandwidth_pressure')
if occ is not None and occ < 35: print('diagnosis_extra=low_resident_warp_occupancy')
if issue is not None and issue < 25 and (score or 0) < 20: print('diagnosis_extra=front_end_or_dependency_underissue')
PY

echo "b300-ncu-high-bottleneck OK csv=$CSV report=$REPORT" >&2
