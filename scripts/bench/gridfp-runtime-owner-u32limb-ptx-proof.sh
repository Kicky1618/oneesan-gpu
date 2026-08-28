#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_80}"
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_runtime_owner_u32limb_ptx_probe.cu"
PTX="${PTX:-$ONEESAN_BUILD_DIR/gridfp_runtime_owner_u32limb_ptx_probe.ptx}"
mkdir -p "$(dirname "$PTX")"

"$NVCC" -O3 -std=c++17 -ptx -arch="$ARCH" "$SRC" -o "$PTX"

if grep -Eq '\bmul\.(lo|hi)\.u64\b|\bdiv\.u64\b|\brem\.u64\b' "$PTX"; then
  echo 'unexpected 64-bit integer multiply/divide in u32-limb owner PTX' >&2
  grep -En '\bmul\.(lo|hi)\.u64\b|\bdiv\.u64\b|\brem\.u64\b' "$PTX" >&2 || true
  exit 3
fi
grep -Eq '\bmul\.hi\.u32\b' "$PTX" || { echo 'missing mul.hi.u32 in u32-limb owner PTX' >&2; exit 4; }
grep -Eq '\b(mul|mad)\.lo\.u32\b' "$PTX" || { echo 'missing 32-bit low multiply in u32-limb owner PTX' >&2; exit 5; }

wide="$(grep -Ec '\b(mul|mad)\.wide\.u32\b' "$PTX" || true)"
hi="$(grep -Ec '\bmul\.hi\.u32\b' "$PTX" || true)"
lo32="$(grep -Ec '\b(mul|mad)\.lo\.u32\b' "$PTX" || true)"
echo "gridfp-runtime-owner-u32limb-ptx-proof OK arch=$ARCH low_mul_u32=$lo32 mul_hi_u32=$hi wide_u32_total=$wide mul_u64=0 div_u64=0 rem_u64=0"
