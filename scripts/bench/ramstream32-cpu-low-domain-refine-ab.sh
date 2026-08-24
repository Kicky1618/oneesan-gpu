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
CPU_WORKERS="${CPU_WORKERS:-64}"
CPU_HIGH_WORKERS="${CPU_HIGH_WORKERS:-32}"
CPU_HIGH_MODE="${CPU_HIGH_MODE:-direct}"
CPU_HIGH_OVERLAP="${CPU_HIGH_OVERLAP:-1}"
CPU_HIGH_MAX_MIB="${CPU_HIGH_MAX_MIB:-256}"
CPU_HIGH_GROUPS_FILE="${CPU_HIGH_GROUPS_FILE:-}"
CPU_HIGH_CPU_LIST="${CPU_HIGH_CPU_LIST:-}"
CPU_LOW_CPU_LIST="${CPU_LOW_CPU_LIST:-}"
CPU_LOW_DOMAIN_SIZE="${CPU_LOW_DOMAIN_SIZE:-32}"
REPEATS="${REPEATS:-4}"
BUILD="${BUILD:-1}"
EXPECTED_RESIDUE="${EXPECTED_RESIDUE:-}"
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_low_domain_refine_ab}"

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
if [[ ! "$CPU_LOW_DOMAIN_SIZE" =~ ^[1-9][0-9]*$ ]] || (( CPU_LOW_DOMAIN_SIZE > CPU_WORKERS )); then
  echo "CPU_LOW_DOMAIN_SIZE must be a positive integer <= CPU_WORKERS" >&2
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
out="$OUT_DIR/domain-refine-ab-n${N}-${ts}.tsv"
meta="$OUT_DIR/domain-refine-ab-n${N}-${ts}.meta"
log_dir="$OUT_DIR/domain-refine-ab-n${N}-${ts}.logs"
mkdir -p "$log_dir"

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
cpu_low_schedule=domain
cpu_low_domain_size=$CPU_LOW_DOMAIN_SIZE
variants=refine0,refine1
order_design=alternating-pairs
repeats=$REPEATS
numa_sampling=disabled
expected_residue=${EXPECTED_RESIDUE:-unknown}
binary_sha256=$(file_sha256 "$bin")
EOF
if [[ -n "$CPU_HIGH_GROUPS_FILE" ]]; then
  echo "cpu_high_groups_file_sha256=$(file_sha256 "$CPU_HIGH_GROUPS_FILE")" >>"$meta"
fi

printf 'repeat\torder\trefine\tresidue\twall_s\tcpu_low_wall_s\tcpu_low_kernel_sum_s\tcpu_low_schedule_build_s\tcpu_low_domain_outer_normalized_cap\tcpu_low_domain_active_domains\tcpu_low_domain_refined_boundaries\tcpu_low_domain_refined_job_moves\tcpu_high_wall_s\th2d_s\tgpu_kernel_s\td2h_s\tstderr_log\traw\n' >"$out"

run_one() {
  local repeat="$1" order="$2" refine="$3"
  local stderr_log="$log_dir/repeat${repeat}-${order}-refine${refine}.stderr.txt"
  local line residue got_schedule got_domain got_refine boundaries moves

  line="$(CPU_HIGH_MODE="$CPU_HIGH_MODE" \
    CPU_HIGH_OVERLAP="$CPU_HIGH_OVERLAP" \
    CPU_HIGH_MAX_MIB="$CPU_HIGH_MAX_MIB" \
    CPU_HIGH_GROUPS_FILE="$CPU_HIGH_GROUPS_FILE" \
    CPU_HIGH_WORKERS="$CPU_HIGH_WORKERS" \
    CPU_HIGH_CPU_LIST="$CPU_HIGH_CPU_LIST" \
    CPU_LOW_CPU_LIST="$CPU_LOW_CPU_LIST" \
    CPU_LOW_SCHEDULE=domain \
    CPU_LOW_DOMAIN_SIZE="$CPU_LOW_DOMAIN_SIZE" \
    CPU_LOW_DOMAIN_REFINE="$refine" \
    RAMSTREAM_NUMA_SAMPLE_MIB=0 \
    "$bin" "$N" "$MODULUS" "$GPU_TARGET_MIB" "$CPU_WORKERS" \
      2>"$stderr_log" | tail -n1)"

  residue="$(field "$line" residue)"
  got_schedule="$(field "$line" cpu_low_schedule)"
  got_domain="$(field "$line" cpu_low_domain_size)"
  got_refine="$(field "$line" cpu_low_domain_refine)"
  [[ "$got_schedule" == domain ]] || {
    echo "schedule provenance mismatch got=$got_schedule" >&2; exit 7;
  }
  [[ "$got_domain" == "$CPU_LOW_DOMAIN_SIZE" ]] || {
    echo "domain provenance mismatch requested=$CPU_LOW_DOMAIN_SIZE got=$got_domain" >&2; exit 7;
  }
  [[ "$got_refine" == "$refine" ]] || {
    echo "refine stdout provenance mismatch requested=$refine got=$got_refine" >&2; exit 7;
  }
  grep -Eq "cpu_low_domain_schedule .* refine=${refine}( |$)" "$stderr_log" || {
    echo "refinement stderr provenance mismatch refine=$refine log=$stderr_log" >&2
    exit 7
  }

  boundaries="$(field "$line" cpu_low_domain_refined_boundaries)"
  moves="$(field "$line" cpu_low_domain_refined_job_moves)"
  if [[ "$refine" == 0 && ( "$boundaries" != 0 || "$moves" != 0 ) ]]; then
    echo "refinement disabled but moves were reported boundaries=$boundaries moves=$moves" >&2
    exit 8
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$repeat" "$order" "$refine" "$residue" \
    "$(field "$line" wall_s)" "$(field "$line" cpu_low_wall_s)" \
    "$(field "$line" cpu_low_kernel_sum_s)" "$(field "$line" cpu_low_schedule_build_s)" \
    "$(field "$line" cpu_low_domain_outer_normalized_cap)" \
    "$(field "$line" cpu_low_domain_active_domains)" "$boundaries" "$moves" \
    "$(field "$line" cpu_high_wall_s)" "$(field "$line" h2d_s)" \
    "$(field "$line" gpu_kernel_s)" "$(field "$line" d2h_s)" \
    "$stderr_log" "$line" >>"$out"
  printf '%s\n' "$residue"
}

reference_residue=""
for ((r=1; r<=REPEATS; ++r)); do
  if ((r % 2 == 1)); then
    order=off-first
    variants=(0 1)
  else
    order=on-first
    variants=(1 0)
  fi
  echo "repeat $r/$REPEATS domain_size=$CPU_LOW_DOMAIN_SIZE ($order)" >&2
  for refine in "${variants[@]}"; do
    echo "  CPU_LOW_DOMAIN_REFINE=$refine" >&2
    residue="$(run_one "$r" "$order" "$refine")"
    if [[ -z "$reference_residue" ]]; then reference_residue="$residue"; fi
    if [[ "$residue" != "$reference_residue" ]]; then
      echo "residue mismatch refine=$refine got=$residue reference=$reference_residue" >&2
      exit 5
    fi
    if [[ -n "$EXPECTED_RESIDUE" && "$residue" != "$EXPECTED_RESIDUE" ]]; then
      echo "unexpected residue refine=$refine got=$residue expected=$EXPECTED_RESIDUE" >&2
      exit 6
    fi
  done
done

awk -F '\t' '
  NR==1 {next}
  {
    r=$3; n[r]++;
    wall[r]+=$5; low[r]+=$6; kernel[r]+=$7; build[r]+=$8;
    boundaries[r]+=$11; moves[r]+=$12; high[r]+=$13;
  }
  END {
    for (r=0; r<=1; ++r) if (n[r])
      printf("summary refine=%d runs=%d mean_wall_s=%.9f mean_cpu_low_wall_s=%.9f mean_cpu_low_kernel_sum_s=%.9f mean_schedule_build_s=%.9f mean_refined_boundaries=%.3f mean_refined_job_moves=%.3f mean_cpu_high_wall_s=%.9f\n",
             r,n[r],wall[r]/n[r],low[r]/n[r],kernel[r]/n[r],build[r]/n[r],boundaries[r]/n[r],moves[r]/n[r],high[r]/n[r]);
    if (n[0] && n[1]) {
      w0=wall[0]/n[0]; w1=wall[1]/n[1]; l0=low[0]/n[0]; l1=low[1]/n[1];
      b0=build[0]/n[0]; b1=build[1]/n[1];
      printf("comparison refine0_wall_s=%.9f refine1_wall_s=%.9f refine_speedup=%.9fx refine_saved_s=%.9f refine0_low_s=%.9f refine1_low_s=%.9f refine_low_speedup=%.9fx refine_low_saved_s=%.9f refine0_build_s=%.9f refine1_build_s=%.9f refine_extra_build_s=%.9f\n",
             w0,w1,w0/w1,w0-w1,l0,l1,l0/l1,l0-l1,b0,b1,b1-b0);
    }
  }
' "$out"

echo "results=$out"
echo "metadata=$meta"
echo "logs=$log_dir"
