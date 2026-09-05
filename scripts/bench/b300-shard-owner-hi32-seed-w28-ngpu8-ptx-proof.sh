#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_80}"
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
bash "$ONEESAN_ROOT/scripts/bench/b300-shard-owner-hi32-seed-w28-ngpu8-proof.sh"

SRC="$ONEESAN_ROOT/src/cuda/b300/probes/shard_owner_hi32_seed_w28_ngpu8_ptx_probe.cu"
OUTDIR="${OUTDIR:-$ONEESAN_BUILD_DIR/b300_shard_owner_hi32_seed_w28_ngpu8_ptx}"
mkdir -p "$OUTDIR"
PTX="$OUTDIR/probe.ptx"
"$NVCC" -O3 -std=c++17 -ptx -arch="$ARCH" \
  -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 \
  -DB300_FAST_SHARD_ADDRESS8=1 "$SRC" -o "$PTX"

extract(){
  local kernel="$1" out="$2"
  awk -v k="$kernel" '
    $0 ~ "\\.entry[[:space:]]+" k { inside=1 }
    inside { print }
    inside && /^}/ { exit }
  ' "$PTX" > "$out"
  [[ -s "$out" ]] || { echo "PTX body missing for $kernel" >&2; exit 3; }
}
metric(){ local re="$1" file="$2"; grep -Ec "$re" "$file" || true; }

extract gridfp_b300_shard_owner_hi32_compare_probe "$OUTDIR/compare.body.ptx"
extract gridfp_b300_shard_owner_hi32_mul_probe "$OUTDIR/seed_mul.body.ptx"
extract gridfp_b300_shard_owner_hi32_shiftadd_probe "$OUTDIR/seed_shiftadd.body.ptx"

for mode in compare seed_mul seed_shiftadd; do
  body="$OUTDIR/${mode}.body.ptx"
  div64="$(metric '\b(div|rem)\.u64\b' "$body")"
  mulhi64="$(metric '\bmul\.hi\.u64\b' "$body")"
  mullo64="$(metric '\bmul\.lo\.u64\b' "$body")"
  mulwide64="$(metric '\bmul\.wide\.u64\b' "$body")"
  mul32="$(metric '\bmul\.(lo|hi|wide)\.u32\b|\bmad\.lo\.u32\b' "$body")"
  setp64="$(metric '\bsetp\..*\.u64\b' "$body")"
  setp32="$(metric '\bsetp\..*\.u32\b' "$body")"
  shl32="$(metric '\bshl\.b32\b' "$body")"
  sub64="$(metric '\bsub\.u64\b' "$body")"
  bra="$(metric '\bbra\b' "$body")"
  selp="$(metric '\bselp\.' "$body")"
  if (( div64 != 0 || mulhi64 != 0 || mullo64 != 0 || mulwide64 != 0 )); then
    echo "$mode unexpectedly contains u64 divide/multiply" >&2
    grep -En '\b(div|rem)\.u64\b|\bmul\.(hi|lo|wide)\.u64\b' "$body" >&2 || true
    exit 4
  fi
  printf '%s_divrem_u64=%s\n%s_mulhi_u64=%s\n%s_mullo_u64=%s\n%s_mulwide_u64=%s\n%s_mul_u32=%s\n%s_setp_u64=%s\n%s_setp_u32=%s\n%s_shl_b32=%s\n%s_sub_u64=%s\n%s_bra=%s\n%s_selp=%s\n' \
    "$mode" "$div64" "$mode" "$mulhi64" "$mode" "$mullo64" "$mode" "$mulwide64" \
    "$mode" "$mul32" "$mode" "$setp64" "$mode" "$setp32" "$mode" "$shl32" \
    "$mode" "$sub64" "$mode" "$bra" "$mode" "$selp"
done

compare_setp="$(metric '\bsetp\..*\.u64\b' "$OUTDIR/compare.body.ptx")"
mul_setp="$(metric '\bsetp\..*\.u64\b' "$OUTDIR/seed_mul.body.ptx")"
shift_setp="$(metric '\bsetp\..*\.u64\b' "$OUTDIR/seed_shiftadd.body.ptx")"
if (( compare_setp < 3 )); then
  echo "compare path missing expected three u64 comparisons" >&2
  exit 5
fi
if (( mul_setp < 1 || shift_setp < 1 || mul_setp >= compare_setp || shift_setp >= compare_setp )); then
  echo "hi32 seed did not reduce u64 comparison count" >&2
  exit 6
fi

echo "gridfp-b300-shard-owner-hi32-seed-w28-ngpu8-ptx-proof OK arch=$ARCH compare_setp_u64=$compare_setp seed_mul_setp_u64=$mul_setp seed_shiftadd_setp_u64=$shift_setp main_seed=(hi32*365)>>12 correction_max=1 device_div64=0 device_mul64=0 exact_proof=1"
