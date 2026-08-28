#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_80}"
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
bash "$ONEESAN_ROOT/scripts/bench/b300-shard-address8-proof.sh"

SRC="$ONEESAN_ROOT/src/cuda/b300/probes/shard_address8_integration_ptx_probe.cu"
OUTDIR="${OUTDIR:-$ONEESAN_BUILD_DIR/b300_shard_address8_ptx}"
mkdir -p "$OUTDIR"

compile_one() {
  local fast="$1" ptx="$OUTDIR/fast_${fast}.ptx" body="$OUTDIR/fast_${fast}.body.ptx"
  "$NVCC" -O3 -std=c++17 -ptx -arch="$ARCH" \
    -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 \
    -DB300_FAST_SHARD_ADDRESS8="$fast" "$SRC" -o "$ptx"
  awk '
    /\.entry[[:space:]]+gridfp_b300_shard_address8_integration_probe/ { inside=1 }
    inside { print }
    inside && /^}/ { exit }
  ' "$ptx" > "$body"
  [[ -s "$body" ]] || { echo "probe body missing fast=$fast" >&2; exit 3; }
}
compile_one 0
compile_one 1

slow="$OUTDIR/fast_0.body.ptx"
fast="$OUTDIR/fast_1.body.ptx"
slow_div="$(grep -Ec '\b(div|rem)\.u64\b' "$slow" || true)"
slow_mul64="$(grep -Ec '\bmul\.(lo|hi|wide)\.u64\b' "$slow" || true)"
fast_div="$(grep -Ec '\b(div|rem)\.u64\b' "$fast" || true)"
fast_mul64="$(grep -Ec '\bmul\.(lo|hi|wide)\.u64\b' "$fast" || true)"
fast_setp="$(grep -Ec '\bsetp\..*\.u64\b' "$fast" || true)"
fast_sub="$(grep -Ec '\bsub\.u64\b' "$fast" || true)"

if (( fast_div != 0 )); then
  echo "fast shard integration still contains u64 divide/remainder" >&2
  grep -En '\b(div|rem)\.u64\b' "$fast" >&2 || true
  exit 4
fi
if (( fast_mul64 != 0 )); then
  echo "fast shard integration still contains u64 multiply" >&2
  grep -En '\bmul\.(lo|hi|wide)\.u64\b' "$fast" >&2 || true
  exit 5
fi
if (( fast_setp < 3 || fast_sub < 1 )); then
  echo "fast shard integration missing expected compare/subtract structure" >&2
  exit 6
fi

echo "gridfp-b300-shard-address8-integration-ptx-proof OK arch=$ARCH slow_divrem_u64=$slow_div slow_mul_u64=$slow_mul64 fast_divrem_u64=$fast_div fast_mul_u64=$fast_mul64 fast_setp_u64=$fast_setp fast_sub_u64=$fast_sub compare_stages=3 exact_proof=1"
