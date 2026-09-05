#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_80}"
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
bash "$ONEESAN_ROOT/scripts/bench/b300-shard-owner-hi32-u32addr-w28-ngpu8-proof.sh"

SRC="$ONEESAN_ROOT/src/cuda/b300/probes/shard_owner_hi32_u32addr_w28_ngpu8_ptx_probe.cu"
OUTDIR="${OUTDIR:-$ONEESAN_BUILD_DIR/b300_shard_owner_hi32_u32addr_w28_ngpu8_ptx}"
PTX="$OUTDIR/probe.ptx"
mkdir -p "$OUTDIR"
"$NVCC" -O3 -std=c++17 -ptx -arch="$ARCH" \
  -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 -DB300_FAST_SHARD_ADDRESS8=1 \
  "$SRC" -o "$PTX"

extract(){ local kernel="$1" out="$2"; awk -v k="$kernel" '$0 ~ "\\.entry[[:space:]]+" k{inside=1} inside{print} inside&&/^}/{exit}' "$PTX" > "$out"; [[ -s "$out" ]] || { echo "missing PTX body $kernel" >&2; exit 3; }; }
metric(){ local re="$1" file="$2"; grep -Ec "$re" "$file" || true; }
extract b300_hi32_u32addr_compare_probe "$OUTDIR/compare.body.ptx"
extract b300_hi32_u32addr_u64corr_probe "$OUTDIR/u64corr.body.ptx"
extract b300_hi32_u32addr_full_probe "$OUTDIR/full_u32.body.ptx"

for mode in compare u64corr full_u32; do
  body="$OUTDIR/${mode}.body.ptx"
  div64="$(metric '\b(div|rem)\.u64\b' "$body")"
  mul64="$(metric '\bmul\.(hi|lo|wide)\.u64\b' "$body")"
  setp64="$(metric '\bsetp\..*\.u64\b' "$body")"
  sub64="$(metric '\bsub\.u64\b' "$body")"
  setp32="$(metric '\bsetp\..*\.u32\b' "$body")"
  sub32="$(metric '\bsub\.u32\b' "$body")"
  add32="$(metric '\badd\.u32\b|\badd\.s32\b' "$body")"
  shl32="$(metric '\bshl\.b32\b' "$body")"
  if (( div64 != 0 || mul64 != 0 )); then echo "$mode contains forbidden u64 div/mul" >&2; exit 4; fi
  printf '%s_divrem_u64=%s\n%s_mul_u64=%s\n%s_setp_u64=%s\n%s_sub_u64=%s\n%s_setp_u32=%s\n%s_sub_u32=%s\n%s_add_u32=%s\n%s_shl_b32=%s\n' \
    "$mode" "$div64" "$mode" "$mul64" "$mode" "$setp64" "$mode" "$sub64" \
    "$mode" "$setp32" "$mode" "$sub32" "$mode" "$add32" "$mode" "$shl32"
done

compare_setp="$(metric '\bsetp\..*\.u64\b' "$OUTDIR/compare.body.ptx")"
u64corr_setp="$(metric '\bsetp\..*\.u64\b' "$OUTDIR/u64corr.body.ptx")"
full_setp="$(metric '\bsetp\..*\.u64\b' "$OUTDIR/full_u32.body.ptx")"
full_sub="$(metric '\bsub\.u64\b' "$OUTDIR/full_u32.body.ptx")"
if (( compare_setp < 3 )); then echo "compare path missing 3 u64 compares" >&2; exit 5; fi
if (( u64corr_setp < 1 || u64corr_setp >= compare_setp )); then echo "u64 correction did not reduce compare count" >&2; exit 6; fi
if (( full_setp != 0 || full_sub != 0 )); then
  echo "fully-u32 path still contains setp/sub u64 setp=$full_setp sub=$full_sub" >&2
  grep -En '\bsetp\..*\.u64\b|\bsub\.u64\b' "$OUTDIR/full_u32.body.ptx" >&2 || true
  exit 7
fi

echo "b300-shard-owner-hi32-u32addr-w28-ngpu8-ptx-proof OK arch=$ARCH compare_setp_u64=$compare_setp u64corr_setp_u64=$u64corr_setp full_setp_u64=$full_setp full_sub_u64=$full_sub device_div64=0 device_mul64=0 exact_proof=1"
