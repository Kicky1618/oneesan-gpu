#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_80}"
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
bash "$ONEESAN_ROOT/scripts/bench/b300-shard-owner-mulhi-w28-ngpu8-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-shard-owner-u32limb-w28-ngpu8-proof.sh"

SRC="$ONEESAN_ROOT/src/cuda/b300/probes/shard_owner_mulhi_w28_ngpu8_ptx_probe.cu"
OUTDIR="${OUTDIR:-$ONEESAN_BUILD_DIR/b300_shard_owner_mulhi_w28_ngpu8_ptx}"
mkdir -p "$OUTDIR"
PTX="$OUTDIR/probe.ptx"
"$NVCC" -O3 -std=c++17 -ptx -arch="$ARCH" \
  -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 \
  -DB300_FAST_SHARD_ADDRESS8=1 "$SRC" -o "$PTX"

extract() {
  local kernel="$1" out="$2"
  awk -v k="$kernel" '
    $0 ~ "\\.entry[[:space:]]+" k { inside=1 }
    inside { print }
    inside && /^}/ { exit }
  ' "$PTX" > "$out"
  [[ -s "$out" ]] || { echo "PTX body missing for $kernel" >&2; exit 3; }
}
extract gridfp_b300_shard_owner_branchy_probe "$OUTDIR/branchy.body.ptx"
extract gridfp_b300_shard_owner_mulhi_mul_probe "$OUTDIR/mulhi_mul.body.ptx"
extract gridfp_b300_shard_owner_mulhi_table_probe "$OUTDIR/mulhi_table.body.ptx"
extract gridfp_b300_shard_owner_mulhi_mask_probe "$OUTDIR/mulhi_mask.body.ptx"
extract gridfp_b300_shard_owner_u32limb_mask_probe "$OUTDIR/u32limb_mask.body.ptx"

metric(){ local re="$1" file="$2"; grep -Ec "$re" "$file" || true; }
for mode in branchy mulhi_mul mulhi_table mulhi_mask u32limb_mask; do
  body="$OUTDIR/${mode}.body.ptx"
  div="$(metric '\b(div|rem)\.u64\b' "$body")"
  mulhi64="$(metric '\bmul\.hi\.u64\b' "$body")"
  mullo64="$(metric '\bmul\.lo\.u64\b' "$body")"
  mulwide64="$(metric '\bmul\.wide\.u64\b' "$body")"
  mulhi32="$(metric '\bmul\.hi\.u32\b' "$body")"
  mullo32="$(metric '\bmul\.lo\.u32\b|\bmad\.lo\.u32\b' "$body")"
  setp32="$(metric '\bsetp\..*\.u32\b' "$body")"
  setp64="$(metric '\bsetp\..*\.u64\b' "$body")"
  sub64="$(metric '\bsub\.u64\b' "$body")"
  bra="$(metric '\bbra\b' "$body")"
  selp="$(metric '\bselp\.' "$body")"
  ldconst="$(metric '\bld\.const\..*\.u64\b' "$body")"
  if (( div != 0 )); then
    echo "$mode still contains u64 divide/remainder" >&2
    exit 4
  fi
  if [[ "$mode" == mulhi_mul || "$mode" == mulhi_table || "$mode" == mulhi_mask ]] && (( mulhi64 < 1 )); then
    echo "$mode missing expected mul.hi.u64" >&2
    exit 5
  fi
  if [[ "$mode" == mulhi_mask ]] && (( mullo64 != 0 || mulwide64 != 0 || ldconst != 0 )); then
    echo "mulhi_mask unexpectedly uses u64 low/wide multiply or constant-table load" >&2
    exit 6
  fi
  if [[ "$mode" == u32limb_mask ]]; then
    if (( mulhi64 != 0 || mullo64 != 0 || mulwide64 != 0 || ldconst != 0 )); then
      echo "u32limb_mask unexpectedly uses u64 multiply or constant-table load" >&2
      grep -En '\bmul\.(hi|lo|wide)\.u64\b|\bld\.const\..*\.u64\b' "$body" >&2 || true
      exit 7
    fi
    if (( mulhi32 < 3 )); then
      echo "u32limb_mask missing expected mul.hi.u32 limbs" >&2
      exit 8
    fi
  fi
  printf '%s_divrem_u64=%s\n%s_mulhi_u64=%s\n%s_mullo_u64=%s\n%s_mulwide_u64=%s\n%s_mulhi_u32=%s\n%s_mullo_u32=%s\n%s_setp_u32=%s\n%s_setp_u64=%s\n%s_sub_u64=%s\n%s_bra=%s\n%s_selp=%s\n%s_ldconst_u64=%s\n' \
    "$mode" "$div" "$mode" "$mulhi64" "$mode" "$mullo64" "$mode" "$mulwide64" \
    "$mode" "$mulhi32" "$mode" "$mullo32" "$mode" "$setp32" "$mode" "$setp64" \
    "$mode" "$sub64" "$mode" "$bra" "$mode" "$selp" "$mode" "$ldconst"
done

echo "gridfp-b300-shard-owner-mulhi-w28-ngpu8-ptx-proof OK arch=$ARCH main_magic=195888106327 main_high_shift=9 block_magic=139905900989 block_high_shift=7 masked_base_mul64=0 masked_base_table=0 u32limb_device_mul64=0 u32limb_umulhi32=3 exact_proof=1"