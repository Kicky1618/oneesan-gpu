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
REPEATS="${REPEATS:-4}"
BUILD="${BUILD:-1}"
EXPECTED_RESIDUE="${EXPECTED_RESIDUE:-}"
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_low_schedule_ab}"

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
out="$OUT_DIR/low-schedule-ab-n${N}-${ts}.tsv"
meta="$OUT_DIR/low-schedule-ab-n${N}-${ts}.meta"

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
numa_sampling=disabled
expected_residue=${EXPECTED_RESIDUE:-unknown}
binary_sha256=$(file_sha256 "$bin")
EOF
if [[ -n "$CPU_HIGH_GROUPS_FILE" ]]; then
  echo "cpu_high_groups_file_sha256=$(file_sha256 "$CPU_HIGH_GROUPS_FILE")" >>"$meta"
fi

printf 'repeat\torder\tschedule\tresidue\twall_s\tcpu_low_wall_s\tcpu_low_kernel_sum_s\tcpu_low_schedule_build_s\tcpu_low_worker_start_s\tcpu_high_wall_s\th2d_s\tgpu_kernel_s\td2h_s\traw\n' >"$out"

run_one() {
  local repeat="$1" order="$2" schedule="$3"
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
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$repeat" "$order" "$schedule" "$residue" \
    "$(field "$line" wall_s)" "$(field "$line" cpu_low_wall_s)" \
    "$(field "$line" cpu_low_kernel_sum_s)" "$(field "$line" cpu_low_schedule_build_s)" \
    "$(field "$line" cpu_low_worker_start_s)" "$(field "$line" cpu_high_wall_s)" \
    "$(field "$line" h2d_s)" "$(field "$line" gpu_kernel_s)" \
    "$(field "$line" d2h_s)" "$line" >>"$out"
  printf '%s\n' "$residue"
}

reference_residue=""
for ((r=1; r<=REPEATS; ++r)); do
  if ((r % 2 == 1)); then
    order=dynamic-first
    schedules=(dynamic sticky)
  else
    order=sticky-first
    schedules=(sticky dynamic)
  fi
  echo "repeat $r/$REPEATS ($order)" >&2
  for schedule in "${schedules[@]}"; do
    echo "  schedule=$schedule" >&2
    residue="$(run_one "$r" "$order" "$schedule")"
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
    n[$3]++;
    wall[$3]+=$5; low[$3]+=$6; kernel[$3]+=$7;
    build[$3]+=$8; start[$3]+=$9; high[$3]+=$10;
  }
  END {
    for (s in n)
      printf("summary schedule=%s runs=%d mean_wall_s=%.9f mean_cpu_low_wall_s=%.9f mean_cpu_low_kernel_sum_s=%.9f mean_schedule_build_s=%.9f mean_worker_start_s=%.9f mean_cpu_high_wall_s=%.9f\n",
             s,n[s],wall[s]/n[s],low[s]/n[s],kernel[s]/n[s],build[s]/n[s],start[s]/n[s],high[s]/n[s]);
    if (n["dynamic"] && n["sticky"]) {
      dw=wall["dynamic"]/n["dynamic"];
      sw=wall["sticky"]/n["sticky"];
      dl=low["dynamic"]/n["dynamic"];
      sl=low["sticky"]/n["sticky"];
      printf("comparison dynamic_mean_wall_s=%.9f sticky_mean_wall_s=%.9f sticky_wall_speedup=%.9fx sticky_wall_saved_s=%.9f dynamic_mean_cpu_low_wall_s=%.9f sticky_mean_cpu_low_wall_s=%.9f sticky_low_speedup=%.9fx sticky_low_saved_s=%.9f\n",
             dw,sw,dw/sw,dw-sw,dl,sl,dl/sl,dl-sl);
    }
  }
' "$out" | sort

echo "results=$out"
echo "metadata=$meta"
