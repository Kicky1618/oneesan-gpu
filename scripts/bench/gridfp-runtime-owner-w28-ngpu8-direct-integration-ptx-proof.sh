#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_80}"
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }

bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-owner-w28-ngpu8-direct-proof.sh"

SRC="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_runtime_owner_w28_ngpu8_direct_integration_ptx_probe.cu"
PTX="${PTX:-$ONEESAN_BUILD_DIR/gridfp_runtime_owner_w28_ngpu8_direct_integration.ptx}"
BODY="${BODY:-$ONEESAN_BUILD_DIR/gridfp_runtime_owner_w28_ngpu8_direct_integration.body.ptx}"
mkdir -p "$(dirname "$PTX")"
"$NVCC" -O3 -std=c++17 -ptx -arch="$ARCH" "$SRC" -o "$PTX"

awk '
  /\.entry[[:space:]]+gridfp_runtime_owner_w28_ngpu8_direct_integration_probe/ { inside=1 }
  inside { print }
  inside && /^}/ { exit }
' "$PTX" > "$BODY"
[[ -s "$BODY" ]] || { echo "integration kernel body not found in PTX" >&2; exit 3; }

if grep -Eq '\bmul\.(lo|hi)\.u64\b|\bdiv\.u64\b|\brem\.u64\b' "$BODY"; then
  echo "unexpected 64-bit integer multiply/divide in direct W28 x8 integration kernel" >&2
  grep -En '\bmul\.(lo|hi)\.u64\b|\bdiv\.u64\b|\brem\.u64\b' "$BODY" >&2 || true
  exit 4
fi
if grep -Eq '\bld\.const\b|RP_RUNTIME_OWNER_U32_META' "$BODY"; then
  echo "direct W28 x8 integration kernel still loads owner metadata" >&2
  grep -En '\bld\.const\b|RP_RUNTIME_OWNER_U32_META' "$BODY" >&2 || true
  exit 5
fi
grep -Eq '\bmul\.hi\.u32\b' "$BODY" || { echo "missing mul.hi.u32 in direct W28 x8 integration kernel" >&2; exit 6; }
grep -Eq '\b(mul|mad)\.lo\.u32\b' "$BODY" || { echo "missing low 32-bit multiply in direct W28 x8 integration kernel" >&2; exit 7; }
grep -Eq '\bshr\.u32\b' "$BODY" || { echo "missing final 32-bit shift in direct W28 x8 integration kernel" >&2; exit 8; }

hi="$(grep -Ec '\bmul\.hi\.u32\b' "$BODY" || true)"
lo="$(grep -Ec '\b(mul|mad)\.lo\.u32\b' "$BODY" || true)"
shr="$(grep -Ec '\bshr\.u32\b' "$BODY" || true)"
echo "gridfp-runtime-owner-w28-ngpu8-direct-integration-ptx-proof OK arch=$ARCH meta_loads=0 variable_owner_meta=0 mul_hi_u32=$hi low_mul_u32=$lo shr_u32=$shr mul_u64=0 div_u64=0 rem_u64=0 exact_proof=1"
