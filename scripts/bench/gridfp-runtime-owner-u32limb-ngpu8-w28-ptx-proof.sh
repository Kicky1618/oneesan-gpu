#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_80}"
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_runtime_owner_u32limb_ngpu8_w28_ptx_probe.cu"
PTX="${PTX:-$ONEESAN_BUILD_DIR/gridfp_runtime_owner_u32limb_ngpu8_w28_ptx_probe.ptx}"
GENERIC="${PTX}.generic"
SPECIAL="${PTX}.ngpu8"
mkdir -p "$(dirname "$PTX")"

"$NVCC" -O3 -std=c++17 -ptx -arch="$ARCH" "$SRC" -o "$PTX"

extract_entry() {
  local name="$1" out="$2"
  awk -v name="$name" '
    $0 ~ "\\.entry[[:space:]]+" name "\\(" {in_entry=1}
    in_entry {print}
    in_entry && /^}/ {exit}
  ' "$PTX" >"$out"
  [[ -s "$out" ]] || { echo "missing PTX entry $name" >&2; exit 3; }
}
extract_entry owner_generic_w28_probe "$GENERIC"
extract_entry owner_ngpu8_w28_probe "$SPECIAL"

for f in "$GENERIC" "$SPECIAL"; do
  if grep -Eq '\bmul\.(lo|hi)\.u64\b|\bdiv\.u64\b|\brem\.u64\b' "$f"; then
    echo "unexpected u64 multiply/divide in $f" >&2
    exit 4
  fi
  grep -Eq '\bmul\.hi\.u32\b' "$f" || { echo "missing mul.hi.u32 in $f" >&2; exit 5; }
done

low_count() { grep -Ec '\b(mul|mad)\.lo\.u32\b' "$1" || true; }
generic_low="$(low_count "$GENERIC")"
special_low="$(low_count "$SPECIAL")"
if (( generic_low <= special_low )); then
  echo "expected W28 ngpu8 specialization to reduce low u32 multiplies: generic=$generic_low specialized=$special_low" >&2
  exit 6
fi

generic_hi="$(grep -Ec '\bmul\.hi\.u32\b' "$GENERIC" || true)"
special_hi="$(grep -Ec '\bmul\.hi\.u32\b' "$SPECIAL" || true)"
echo "gridfp-runtime-owner-u32limb-ngpu8-w28-ptx-proof OK arch=$ARCH generic_low_mul_u32=$generic_low ngpu8_low_mul_u32=$special_low low_mul_saved=$((generic_low-special_low)) generic_mul_hi_u32=$generic_hi ngpu8_mul_hi_u32=$special_hi ngpu_mul_new=0 shift_bias=3 mul_u64=0 div_u64=0 rem_u64=0"
