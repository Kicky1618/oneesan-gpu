#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then
  ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"

N="${N:-27}"
ARCH="${ARCH:-native}"
MODULUS="${MODULUS:-4294967291}"
GPU_TARGET_MIB="${GPU_TARGET_MIB:-12288}"
CPU_WORKERS="${CPU_WORKERS:-32}"
CPU_HIGH_WORKERS="${CPU_HIGH_WORKERS:-$CPU_WORKERS}"
CPU_HIGH_MODE="${CPU_HIGH_MODE:-direct}"
CPU_HIGH_OVERLAP="${CPU_HIGH_OVERLAP:-1}"
CPU_HIGH_MAX_MIB="${CPU_HIGH_MAX_MIB:-256}"
CPU_HIGH_GROUPS_FILE="${CPU_HIGH_GROUPS_FILE:-}"
CPU_HIGH_CPU_LIST="${CPU_HIGH_CPU_LIST:-}"
CPU_LOW_CPU_LIST="${CPU_LOW_CPU_LIST:-}"
REPEATS="${REPEATS:-6}"
BUILD="${BUILD:-1}"
EXPECTED_RESIDUE="${EXPECTED_RESIDUE:-}"
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_low_schedule_compare}"

if (( N < 2 || N > 27 || GPU_TARGET_MIB <= 0 || CPU_WORKERS <= 0 || CPU_HIGH_WORKERS <= 0 || REPEATS <= 0 )); then
  echo "invalid benchmark parameters" >&2
  exit 2
fi
if [[ "$CPU_HIGH_MODE" != scratch && "$CPU_HIGH_MODE" != direct ]]; then
  echo "CPU_HIGH_MODE must be scratch or direct" >&2
  exit 2
fi
if [[ "$CPU_HIGH_OVERLAP" != 0 && "$CPU_HIGH_OVERLAP" != 1 ]]; then
  echo "CPU_HIGH_OVERLAP must be 0 or 1" >&2
  exit 2
fi
if [[ -n "$CPU_HIGH_GROUPS_FILE" && ! -f "$CPU_HIGH_GROUPS_FILE" ]]; then
  echo "missing CPU_HIGH_GROUPS_FILE: $CPU_HIGH_GROUPS_FILE" >&2
  exit 2
fi
if [[ -z "$EXPECTED_RESIDUE" && "$N:$MODULUS" == "21:4294967291" ]]; then
  EXPECTED_RESIDUE=998035516
fi

if [[ "$BUILD" != 0 ]]; then
  N="$N" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32-factorized-hybrid-sparse.sh
fi
bin="$ROOT/build/oneesan_cuda_gridfp_ramstream32_factorized_hybrid_sparse_n${N}"
[[ -x "$bin" ]] || { echo "missing executable: $bin" >&2; exit 3; }

mkdir -p "$OUT_DIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="$OUT_DIR/low-schedule-compare-n${N}-${ts}.tsv"
meta="$OUT_DIR/low-schedule-compare-n${N}-${ts}.meta"

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
field() {
  local line="$1" key="$2" token
  for token in $line; do
    if [[ "$token" == "$key="* ]]; then
      printf '%s\n' "${token#*=}"
      return 0
    fi
  done
  return 1
}

cat >"$meta" <<EOF
commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
host=$(hostname 2>/dev/null || echo unknown)
gpu=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader 2>/dev/null | paste -sd ';' - || echo unknown)
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || echo unknown)
n=$N
arch=$ARCH
modulus=$MODULUS
gpu_target_mib=$GPU_TARGET_MIB
cpu_workers=$CPU_WORKERS
cpu_high_workers=$CPU_HIGH_WORKERS
cpu_high_mode=$CPU_HIGH_MODE
cpu_high_overlap=$CPU_HIGH_OVERLAP
cpu_high_max_mib=$CPU_HIGH_MAX_MIB
cpu_high_groups_file=${CPU_HIGH_GROUPS_FILE:-none}
cpu_high_cpu_list=${CPU_HIGH_CPU_LIST:-none}
cpu_low_cpu_list=${CPU_LOW_CPU_LIST:-none}
repeats=$REPEATS
schedules=dynamic,sticky,contiguous
order_design=cyclic-latin-3
numa_sampling=disabled
expected_residue=${EXPECTED_RESIDUE:-unknown}
binary_sha256=$(file_sha256 "$bin")
EOF
if [[ -n "$CPU_HIGH_GROUPS_FILE" ]]; then
  echo "cpu_high_groups_file_sha256=$(file_sha256 "$CPU_HIGH_GROUPS_FILE")" >>"$meta"
fi

printf 'repeat\torder\tposition\tschedule\tresidue\twall_s\tcpu_low_wall_s\tcpu_low_kernel_sum_s\tcpu_low_schedule_build_s\tcpu_low_contiguous_optimal_cap\tcpu_low_worker_start_s\tcpu_high_wall_s\th2d_s\tgpu_kernel_s\td2h_s\traw\n' >"$out"

run_one() {
  local repeat="$1" order="$2" position="$3" schedule="$4"
  local line residue got_schedule
  line="$(CPU_HIGH_MODE="$CPU_HIGH_MODE" \
    CPU_HIGH_OVERLAP="$CPU_HIGH_OVERLAP" \
    CPU_HIGH_MAX_MIB="$CPU_HIGH_MAX_MIB" \
    CPU_HIGH_GROUPS_FILE="$CPU_HIGH_GROUPS_FILE" \
    CPU_HIGH_WORKERS="$CPU_HIGH_WORKERS" \
    CPU_HIGH_CPU_LIST="$CPU_HIGH_CPU_LIST" \
    CPU_LOW_CPU_LIST="$CPU_LOW_CPU_LIST" \
    CPU_LOW_SCHEDULE="$schedule" \
    RAMSTREAM_NUMA_SAMPLE_MIB=0 \
    "$bin" "$N" "$MODULUS" "$GPU_TARGET_MIB" "$CPU_WORKERS" | tail -n1)"
  residue="$(field "$line" residue)"
  got_schedule="$(field "$line" cpu_low_schedule)"
  if [[ "$got_schedule" != "$schedule" ]]; then
    echo "schedule provenance mismatch requested=$schedule got=$got_schedule" >&2
    exit 7
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$repeat" "$order" "$position" "$schedule" "$residue" \
    "$(field "$line" wall_s)" "$(field "$line" cpu_low_wall_s)" \
    "$(field "$line" cpu_low_kernel_sum_s)" "$(field "$line" cpu_low_schedule_build_s)" \
    "$(field "$line" cpu_low_contiguous_optimal_cap)" \
    "$(field "$line" cpu_low_worker_start_s)" "$(field "$line" cpu_high_wall_s)" \
    "$(field "$line" h2d_s)" "$(field "$line" gpu_kernel_s)" \
    "$(field "$line" d2h_s)" "$line" >>"$out"
  printf '%s\n' "$residue"
}

reference_residue=""
for ((r=1; r<=REPEATS; ++r)); do
  case $(((r - 1) % 3)) in
    0) order=dynamic-sticky-contiguous; schedules=(dynamic sticky contiguous) ;;
    1) order=sticky-contiguous-dynamic; schedules=(sticky contiguous dynamic) ;;
    2) order=contiguous-dynamic-sticky; schedules=(contiguous dynamic sticky) ;;
  esac
  echo "repeat $r/$REPEATS ($order)" >&2
  position=0
  for schedule in "${schedules[@]}"; do
    ((position+=1))
    echo "  position=$position schedule=$schedule" >&2
    residue="$(run_one "$r" "$order" "$position" "$schedule")"
    if [[ -z "$reference_residue" ]]; then reference_residue="$residue"; fi
    if [[ "$residue" != "$reference_residue" ]]; then
      echo "residue mismatch schedule=$schedule got=$residue reference=$reference_residue" >&2
      exit 5
    fi
    if [[ -n "$EXPECTED_RESIDUE" && "$residue" != "$EXPECTED_RESIDUE" ]]; then
      echo "unexpected residue schedule=$schedule got=$residue expected=$EXPECTED_RESIDUE" >&2
      exit 6
    fi
  done
done

awk -F '\t' '
  NR==1 {next}
  {
    s=$4; n[s]++;
    wall[s]+=$6; low[s]+=$7; kernel[s]+=$8; build[s]+=$9;
    start[s]+=$11; high[s]+=$12;
    pos[s,$3]++;
  }
  END {
    modes[1]="dynamic"; modes[2]="sticky"; modes[3]="contiguous";
    for (i=1;i<=3;i++) {
      s=modes[i];
      if (!n[s]) continue;
      printf("summary schedule=%s runs=%d mean_wall_s=%.9f mean_cpu_low_wall_s=%.9f mean_cpu_low_kernel_sum_s=%.9f mean_schedule_build_s=%.9f mean_worker_start_s=%.9f mean_cpu_high_wall_s=%.9f positions=%d/%d/%d\n",
             s,n[s],wall[s]/n[s],low[s]/n[s],kernel[s]/n[s],build[s]/n[s],start[s]/n[s],high[s]/n[s],pos[s,1]+0,pos[s,2]+0,pos[s,3]+0);
    }
    if (n["dynamic"] && n["sticky"] && n["contiguous"]) {
      dw=wall["dynamic"]/n["dynamic"]; sw=wall["sticky"]/n["sticky"]; cw=wall["contiguous"]/n["contiguous"];
      dl=low["dynamic"]/n["dynamic"]; sl=low["sticky"]/n["sticky"]; cl=low["contiguous"]/n["contiguous"];
      printf("comparison dynamic_wall_s=%.9f sticky_wall_s=%.9f contiguous_wall_s=%.9f sticky_vs_dynamic_wall=%.9fx contiguous_vs_dynamic_wall=%.9fx contiguous_vs_sticky_wall=%.9fx dynamic_low_s=%.9f sticky_low_s=%.9f contiguous_low_s=%.9f sticky_vs_dynamic_low=%.9fx contiguous_vs_dynamic_low=%.9fx contiguous_vs_sticky_low=%.9fx\n",
             dw,sw,cw,dw/sw,dw/cw,sw/cw,dl,sl,cl,dl/sl,dl/cl,sl/cl);
    }
  }
' "$out"

echo "results=$out"
echo "metadata=$meta"
