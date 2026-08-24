#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MOD="${MOD:-4294967291}"
EXPECTED="${EXPECTED:-998035516}"
ARCH="${ARCH:-native}"
THREADS="${THREADS:-256}"
GEOMETRIES="${GEOMETRIES:-16x8}"
mkdir -p build

run_checked() {
  local tag="$1"; shift
  local log="build/${tag}.log"
  echo "== $tag =="
  "$@" 2>&1 | tee "$log"
  local residue
  residue="$(sed -n 's/.* residue=\([0-9][0-9]*\).*/\1/p' "$log" | tail -n1)"
  if [[ -z "$residue" ]]; then
    echo "$tag: no residue in output" >&2
    exit 20
  fi
  if [[ "$residue" != "$EXPECTED" ]]; then
    echo "$tag: residue mismatch got=$residue expected=$EXPECTED" >&2
    exit 21
  fi
}

echo "[1/4] compile+run W10 exhaustive gather selftest"
nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W=10 -DLOW_LUT_K=5 -DHIGH_LUT_K=4 \
  src/cuda/gridfp/probes/ramstream32_gpu_direct_gather_selftest.cu \
  -o build/ramstream32_gpu_direct_gather_selftest_w10
build/ramstream32_gpu_direct_gather_selftest_w10

echo "[2/4] build n21 atomic direct"
N=21 ARCH="$ARCH" OUT=oneesan_cuda_gridfp_gpu_direct_n21 \
  bash scripts/build/gridfp-gpu-direct.sh

echo "[3/4] build n21 destination-gather direct"
N=21 ARCH="$ARCH" OUT=oneesan_cuda_gridfp_gpu_direct_gather_n21 \
  bash scripts/build/gridfp-gpu-direct-gather.sh

ATOMIC=build/oneesan_cuda_gridfp_gpu_direct_n21
GATHER=build/oneesan_cuda_gridfp_gpu_direct_gather_n21

echo "[plan] gather"
"$GATHER" 21 "$MOD" "$THREADS" 16 8 --plan-only

echo "[4/4] A/B runs; geometries: $GEOMETRIES"
printf 'backend\tgeometry\twall_s\thigh_s\tlow_s\n' | tee build/gpu-direct-gather-ab.tsv
for geom in $GEOMETRIES; do
  gx="${geom%x*}"
  gy="${geom#*x}"
  if [[ -z "$gx" || -z "$gy" || "$gx" == "$geom" ]]; then
    echo "invalid geometry '$geom' (expected e.g. 16x8)" >&2
    exit 22
  fi
  run_checked "atomic-${geom}" "$ATOMIC" 21 "$MOD" "$THREADS" "$gx" "$gy"
  run_checked "gather-${geom}" "$GATHER" 21 "$MOD" "$THREADS" "$gx" "$gy"
  for backend in atomic gather; do
    log="build/${backend}-${geom}.log"
    wall="$(sed -n 's/.* wall_s=\([^ ]*\).*/\1/p' "$log" | tail -n1)"
    high="$(sed -n 's/.* high_s=\([^ ]*\).*/\1/p' "$log" | tail -n1)"
    low="$(sed -n 's/.* low_s=\([^ ]*\).*/\1/p' "$log" | tail -n1)"
    printf '%s\t%s\t%s\t%s\t%s\n' "$backend" "$geom" "$wall" "$high" "$low" \
      | tee -a build/gpu-direct-gather-ab.tsv
  done
done

echo "A/B summary: build/gpu-direct-gather-ab.tsv"
cat build/gpu-direct-gather-ab.tsv
