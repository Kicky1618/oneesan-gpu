#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then
  ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"

BASE_REF="${BASE_REF:-c8de8d73af1c44f075aee937bc8f37e8b7b79d27}"
HEAD_REF="${HEAD_REF:-HEAD}"
N="${N:-21}"
ARCH="${ARCH:-native}"
MODULUS="${MODULUS:-4294967291}"
GPU_TARGET_MIB="${GPU_TARGET_MIB:-12288}"
CPU_WORKERS="${CPU_WORKERS:-4}"
REPEATS="${REPEATS:-3}"
EXPECTED_RESIDUE="${EXPECTED_RESIDUE:-}"
PERF="${PERF:-0}"
PERF_EVENTS="${PERF_EVENTS:-cycles,instructions,branches,branch-misses,cache-misses}"
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_sparse_v46_v47}"

if (( N < 2 || N > 27 )); then
  echo "N must be in [2, 27]" >&2
  exit 2
fi
if (( GPU_TARGET_MIB <= 0 || CPU_WORKERS <= 0 || REPEATS <= 0 )); then
  echo "GPU_TARGET_MIB, CPU_WORKERS and REPEATS must be positive" >&2
  exit 2
fi

base_commit="$(git rev-parse "$BASE_REF^{commit}")"
head_commit="$(git rev-parse "$HEAD_REF^{commit}")"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="$OUT_DIR/n${N}-${ts}"
base_tree="$run_dir/v46-tree"
head_tree="$run_dir/v47-tree"
mkdir -p "$run_dir"

cleanup() {
  git worktree remove --force "$base_tree" >/dev/null 2>&1 || true
  git worktree remove --force "$head_tree" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ "$PERF" != 0 ]] && ! command -v perf >/dev/null 2>&1; then
  echo "PERF=$PERF requested but perf is unavailable" >&2
  exit 3
fi

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
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

echo "creating detached worktrees" >&2
git worktree add --detach "$base_tree" "$base_commit" >/dev/null
git worktree add --detach "$head_tree" "$head_commit" >/dev/null

build_one() {
  local tree="$1"
  (
    cd "$tree"
    N="$N" ARCH="$ARCH" \
      bash scripts/build/gridfp-ramstream32-factorized-hybrid-sparse.sh
  ) >&2
}

build_one "$base_tree"
build_one "$head_tree"

base_bin="$base_tree/build/oneesan_cuda_gridfp_ramstream32_factorized_hybrid_sparse_n${N}"
head_bin="$head_tree/build/oneesan_cuda_gridfp_ramstream32_factorized_hybrid_sparse_n${N}"
for bin in "$base_bin" "$head_bin"; do
  if [[ ! -x "$bin" ]]; then
    echo "missing executable: $bin" >&2
    exit 4
  fi
done

meta="$run_dir/metadata.txt"
out="$run_dir/results.tsv"
host="$(hostname 2>/dev/null || echo unknown)"
gpu="$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader 2>/dev/null | paste -sd ';' - || true)"
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
nvcc="$(nvcc --version 2>/dev/null | tail -n1 || true)"
[[ -n "$gpu" ]] || gpu=unknown
[[ -n "$driver" ]] || driver=unknown
[[ -n "$nvcc" ]] || nvcc=unknown

cat >"$meta" <<EOF
base_ref=$BASE_REF
base_commit=$base_commit
head_ref=$HEAD_REF
head_commit=$head_commit
host=$host
gpu=$gpu
driver=$driver
nvcc=$nvcc
n=$N
arch=$ARCH
modulus=$MODULUS
gpu_target_mib=$GPU_TARGET_MIB
cpu_workers=$CPU_WORKERS
repeats=$REPEATS
expected_residue=${EXPECTED_RESIDUE:-unknown}
perf=$PERF
perf_events=$PERF_EVENTS
base_binary_sha256=$(file_sha256 "$base_bin")
head_binary_sha256=$(file_sha256 "$head_bin")
EOF

printf 'repeat\torder\tvariant\tbackend\tresidue\twall_s\tcpu_wall_s\tcpu_kernel_sum_s\th2d_s\tgpu_kernel_s\td2h_s\traw\n' >"$out"

run_one() {
  local repeat="$1" order="$2" variant="$3" bin="$4"
  local line perf_file backend residue wall cpu_wall cpu_kernel h2d gpu_kernel d2h
  perf_file="$run_dir/perf-${variant}-r${repeat}.csv"

  if [[ "$PERF" != 0 ]]; then
    line="$(perf stat -x, -o "$perf_file" -e "$PERF_EVENTS" \
      "$bin" "$N" "$MODULUS" "$GPU_TARGET_MIB" "$CPU_WORKERS" | tail -n1)"
  else
    line="$("$bin" "$N" "$MODULUS" "$GPU_TARGET_MIB" "$CPU_WORKERS" | tail -n1)"
  fi

  backend="$(field "$line" backend)"
  residue="$(field "$line" residue)"
  wall="$(field "$line" wall_s)"
  cpu_wall="$(field "$line" cpu_wall_s)"
  cpu_kernel="$(field "$line" cpu_kernel_sum_s)"
  h2d="$(field "$line" h2d_s)"
  gpu_kernel="$(field "$line" gpu_kernel_s)"
  d2h="$(field "$line" d2h_s)"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$repeat" "$order" "$variant" "$backend" "$residue" "$wall" \
    "$cpu_wall" "$cpu_kernel" "$h2d" "$gpu_kernel" "$d2h" "$line" >>"$out"
  printf '%s\n' "$residue"
}

for ((r = 1; r <= REPEATS; ++r)); do
  if (( r % 2 == 1 )); then
    order=v46-first
    echo "repeat $r/$REPEATS: v4.6 -> v4.7" >&2
    r46="$(run_one "$r" "$order" v4.6 "$base_bin")"
    r47="$(run_one "$r" "$order" v4.7 "$head_bin")"
  else
    order=v47-first
    echo "repeat $r/$REPEATS: v4.7 -> v4.6" >&2
    r47="$(run_one "$r" "$order" v4.7 "$head_bin")"
    r46="$(run_one "$r" "$order" v4.6 "$base_bin")"
  fi

  if [[ "$r46" != "$r47" ]]; then
    echo "residue mismatch repeat=$r v4.6=$r46 v4.7=$r47" >&2
    exit 5
  fi
  if [[ -n "$EXPECTED_RESIDUE" && "$r46" != "$EXPECTED_RESIDUE" ]]; then
    echo "unexpected residue repeat=$r got=$r46 expected=$EXPECTED_RESIDUE" >&2
    exit 6
  fi
done

awk -F '\t' '
  NR == 1 { next }
  {
    n[$3]++
    wall[$3] += $6
    cpuwall[$3] += $7
    cpukernel[$3] += $8
  }
  END {
    for (v in n) {
      printf("summary variant=%s runs=%d mean_wall_s=%.9f mean_cpu_wall_s=%.9f mean_cpu_kernel_sum_s=%.9f\n",
             v, n[v], wall[v]/n[v], cpuwall[v]/n[v], cpukernel[v]/n[v])
    }
  }
' "$out"

mean_field() {
  local variant="$1" col="$2"
  awk -F '\t' -v v="$variant" -v c="$col" \
    'NR>1 && $3==v {s+=$c; n++} END {if(n) printf "%.12f", s/n}' "$out"
}

v46_cpu="$(mean_field v4.6 7)"
v47_cpu="$(mean_field v4.7 7)"
v46_wall="$(mean_field v4.6 6)"
v47_wall="$(mean_field v4.7 6)"
if [[ -n "$v46_cpu" && -n "$v47_cpu" ]]; then
  awk -v b="$v46_cpu" -v h="$v47_cpu" 'BEGIN {
    printf("cpu_wall_speedup=%.6fx cpu_wall_reduction=%.3f%%\n", b/h, 100.0*(1.0-h/b));
  }'
fi
if [[ -n "$v46_wall" && -n "$v47_wall" ]]; then
  awk -v b="$v46_wall" -v h="$v47_wall" 'BEGIN {
    printf("full_wall_speedup=%.6fx full_wall_reduction=%.3f%%\n", b/h, 100.0*(1.0-h/b));
  }'
fi

echo "results=$out"
echo "metadata=$meta"
if [[ "$PERF" != 0 ]]; then
  echo "perf_files=$run_dir/perf-*.csv"
fi
