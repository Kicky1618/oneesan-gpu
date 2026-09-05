#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_80}"
REQUIRE_BRANCHLESS="${REQUIRE_BRANCHLESS:-0}"
if [[ "$REQUIRE_BRANCHLESS" != 0 && "$REQUIRE_BRANCHLESS" != 1 ]]; then
  echo "REQUIRE_BRANCHLESS must be 0 or 1" >&2
  exit 2
fi
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
bash "$ONEESAN_ROOT/scripts/bench/b300-shard-address8-proof.sh"

SRC="$ONEESAN_ROOT/src/cuda/b300/probes/shard_address8_integration_ptx_probe.cu"
HELPER_SRC="$ONEESAN_ROOT/src/cuda/b300/probes/shard_address8_helper_ptx_probe.cu"
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

helper_ptx="$OUTDIR/helper.ptx"
helper_body="$OUTDIR/helper.body.ptx"
"$NVCC" -O3 -std=c++17 -ptx -arch="$ARCH" \
  -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 \
  -DB300_FAST_SHARD_ADDRESS8=1 "$HELPER_SRC" -o "$helper_ptx"
awk '
  /\.entry[[:space:]]+gridfp_b300_shard_address8_helper_probe/ { inside=1 }
  inside { print }
  inside && /^}/ { exit }
' "$helper_ptx" > "$helper_body"
[[ -s "$helper_body" ]] || { echo "helper probe body missing" >&2; exit 3; }

slow="$OUTDIR/fast_0.body.ptx"
fast="$OUTDIR/fast_1.body.ptx"
slow_div="$(grep -Ec '\b(div|rem)\.u64\b' "$slow" || true)"
slow_mul64="$(grep -Ec '\bmul\.(lo|hi|wide)\.u64\b' "$slow" || true)"
fast_div="$(grep -Ec '\b(div|rem)\.u64\b' "$fast" || true)"
fast_mul64="$(grep -Ec '\bmul\.(lo|hi|wide)\.u64\b' "$fast" || true)"
fast_setp="$(grep -Ec '\bsetp\..*\.u64\b' "$fast" || true)"
fast_sub="$(grep -Ec '\bsub\.u64\b' "$fast" || true)"
helper_div="$(grep -Ec '\b(div|rem)\.u64\b' "$helper_body" || true)"
helper_mul64="$(grep -Ec '\bmul\.(lo|hi|wide)\.u64\b' "$helper_body" || true)"
helper_setp="$(grep -Ec '\bsetp\..*\.u64\b' "$helper_body" || true)"
helper_sub="$(grep -Ec '\bsub\.u64\b' "$helper_body" || true)"
helper_bra="$(grep -Ec '(^|[[:space:]])@?%?p?[0-9]*[[:space:]]+bra([.;[:space:]]|$)|(^|[[:space:]])bra([.;[:space:]]|$)' "$helper_body" || true)"
helper_selp="$(grep -Ec '\bselp\.' "$helper_body" || true)"

if (( fast_div != 0 || helper_div != 0 )); then
  echo "fast shard integration/helper still contains u64 divide/remainder" >&2
  grep -En '\b(div|rem)\.u64\b' "$fast" "$helper_body" >&2 || true
  exit 4
fi
if (( fast_mul64 != 0 || helper_mul64 != 0 )); then
  echo "fast shard integration/helper still contains u64 multiply" >&2
  grep -En '\bmul\.(lo|hi|wide)\.u64\b' "$fast" "$helper_body" >&2 || true
  exit 5
fi
if (( fast_setp < 3 || fast_sub < 1 || helper_setp < 3 || helper_sub < 1 )); then
  echo "fast shard integration/helper missing expected compare/subtract structure" >&2
  exit 6
fi
if (( REQUIRE_BRANCHLESS == 1 && helper_bra != 0 )); then
  echo "shard helper is not branchless: bra=$helper_bra" >&2
  grep -En '\bbra\b' "$helper_body" >&2 || true
  exit 7
fi

echo "gridfp-b300-shard-address8-integration-ptx-proof OK arch=$ARCH slow_divrem_u64=$slow_div slow_mul_u64=$slow_mul64 fast_divrem_u64=$fast_div fast_mul_u64=$fast_mul64 fast_setp_u64=$fast_setp fast_sub_u64=$fast_sub helper_divrem_u64=$helper_div helper_mul_u64=$helper_mul64 helper_setp_u64=$helper_setp helper_sub_u64=$helper_sub helper_bra=$helper_bra helper_selp=$helper_selp require_branchless=$REQUIRE_BRANCHLESS compare_stages=3 exact_proof=1"
