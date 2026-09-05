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

run_checked(){
  local tag="$1";shift;local log="build/${tag}.log";echo "== $tag ==";"$@" 2>&1|tee "$log"
  local r;r="$(sed -n 's/.* residue=\([0-9][0-9]*\).*/\1/p' "$log"|tail -n1)"
  [[ "$r" == "$EXPECTED" ]] || { echo "$tag residue=$r expected=$EXPECTED" >&2;exit 21; }
}

echo '[1/5] host CROSS inverse exhaustive test'
g++ -O3 -std=c++17 src/cpp/probes/cross_inverse_selftest.cpp -o build/cross_inverse_selftest
build/cross_inverse_selftest

echo '[2/5] W10 fully atomic-free GPU exhaustive selftest'
nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W=10 -DLOW_LUT_K=5 -DHIGH_LUT_K=4 \
  src/cuda/gridfp/probes/ramstream32_gpu_direct_atomicfree_selftest.cu \
  -o build/ramstream32_gpu_direct_atomicfree_selftest_w10
build/ramstream32_gpu_direct_atomicfree_selftest_w10

echo '[3/5] build n21 backends'
N=21 ARCH="$ARCH" OUT=oneesan_cuda_gridfp_gpu_direct_n21 bash scripts/build/gridfp-gpu-direct.sh
N=21 ARCH="$ARCH" OUT=oneesan_cuda_gridfp_gpu_direct_gather_n21 bash scripts/build/gridfp-gpu-direct-gather.sh
N=21 ARCH="$ARCH" OUT=oneesan_cuda_gridfp_gpu_direct_atomicfree_n21 bash scripts/build/gridfp-gpu-direct-atomicfree.sh

ATOMIC=build/oneesan_cuda_gridfp_gpu_direct_n21
GATHER=build/oneesan_cuda_gridfp_gpu_direct_gather_n21
FREE=build/oneesan_cuda_gridfp_gpu_direct_atomicfree_n21

echo '[4/5] plans'
"$GATHER" 21 "$MOD" "$THREADS" 16 8 --plan-only
"$FREE" 21 "$MOD" "$THREADS" 16 8 --plan-only

echo '[5/5] three-way A/B'
printf 'backend\tgeometry\twall_s\thigh_s\tlow_s\n'|tee build/gpu-direct-atomicfree-ab.tsv
for geom in $GEOMETRIES;do
  gx="${geom%x*}";gy="${geom#*x}";[[ "$gx" != "$geom" ]]||exit 22
  run_checked "atomic-${geom}" "$ATOMIC" 21 "$MOD" "$THREADS" "$gx" "$gy"
  run_checked "gather-${geom}" "$GATHER" 21 "$MOD" "$THREADS" "$gx" "$gy"
  run_checked "atomicfree-${geom}" "$FREE" 21 "$MOD" "$THREADS" "$gx" "$gy"
  for backend in atomic gather atomicfree;do
    log="build/${backend}-${geom}.log"
    wall="$(sed -n 's/.* wall_s=\([^ ]*\).*/\1/p' "$log"|tail -n1)"
    high="$(sed -n 's/.* high_s=\([^ ]*\).*/\1/p' "$log"|tail -n1)"
    low="$(sed -n 's/.* low_s=\([^ ]*\).*/\1/p' "$log"|tail -n1)"
    printf '%s\t%s\t%s\t%s\t%s\n' "$backend" "$geom" "$wall" "$high" "$low"|tee -a build/gpu-direct-atomicfree-ab.tsv
  done
done
cat build/gpu-direct-atomicfree-ab.tsv
