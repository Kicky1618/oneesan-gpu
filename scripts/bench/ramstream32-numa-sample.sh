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
NUMA_SAMPLE_MIB="${NUMA_SAMPLE_MIB:-64}"
BUILD="${BUILD:-1}"
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_numa_sample}"

if (( N < 2 || N > 27 || GPU_TARGET_MIB <= 0 || CPU_WORKERS <= 0 || CPU_HIGH_WORKERS <= 0 )); then
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
if [[ ! "$NUMA_SAMPLE_MIB" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || [[ "$NUMA_SAMPLE_MIB" == 0 ]]; then
  echo "NUMA_SAMPLE_MIB must be positive" >&2
  exit 2
fi
if [[ -n "$CPU_HIGH_GROUPS_FILE" && ! -f "$CPU_HIGH_GROUPS_FILE" ]]; then
  echo "missing CPU_HIGH_GROUPS_FILE: $CPU_HIGH_GROUPS_FILE" >&2
  exit 2
fi

if [[ "$BUILD" != 0 ]]; then
  N="$N" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32-factorized-hybrid-sparse.sh
fi
bin="$ROOT/build/oneesan_cuda_gridfp_ramstream32_factorized_hybrid_sparse_n${N}"
[[ -x "$bin" ]] || { echo "missing executable: $bin" >&2; exit 3; }

mkdir -p "$OUT_DIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
base="$OUT_DIR/numa-n${N}-${ts}"
stdout_file="$base.stdout.txt"
stderr_file="$base.stderr.txt"
analysis_file="$base.analysis.txt"
meta_file="$base.meta"

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

cat >"$meta_file" <<EOF
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
numa_sample_mib=$NUMA_SAMPLE_MIB
binary_sha256=$(file_sha256 "$bin")
EOF
if [[ -n "$CPU_HIGH_GROUPS_FILE" ]]; then
  echo "cpu_high_groups_file_sha256=$(file_sha256 "$CPU_HIGH_GROUPS_FILE")" >>"$meta_file"
fi

CPU_HIGH_MODE="$CPU_HIGH_MODE" \
CPU_HIGH_OVERLAP="$CPU_HIGH_OVERLAP" \
CPU_HIGH_MAX_MIB="$CPU_HIGH_MAX_MIB" \
CPU_HIGH_GROUPS_FILE="$CPU_HIGH_GROUPS_FILE" \
CPU_HIGH_WORKERS="$CPU_HIGH_WORKERS" \
CPU_HIGH_CPU_LIST="$CPU_HIGH_CPU_LIST" \
CPU_LOW_CPU_LIST="$CPU_LOW_CPU_LIST" \
RAMSTREAM_NUMA_SAMPLE_MIB="$NUMA_SAMPLE_MIB" \
  "$bin" "$N" "$MODULUS" "$GPU_TARGET_MIB" "$CPU_WORKERS" \
  >"$stdout_file" 2>"$stderr_file"

python3 scripts/tools/analyze_ramstream_numa_samples.py "$stderr_file" \
  | tee "$analysis_file"

echo "result=$(tail -n1 "$stdout_file")"
echo "stdout=$stdout_file"
echo "stderr=$stderr_file"
echo "analysis=$analysis_file"
echo "metadata=$meta_file"
echo "note=NUMA sampling is diagnostic and should not be used as a clean performance timing run"
