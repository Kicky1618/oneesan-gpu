#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_80}"
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
bash "$ONEESAN_ROOT/scripts/bench/b300-shard-owner-mulhi-w28-ngpu8-proof.sh"

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

metric(){ local re="$1" file="$2"; grep -Ec "$re" "$file" || true; }
for mode in branchy mulhi_mul mulhi_table; do
  body="$OUTDIR/${mode}.body.ptx"
  div="$(metric '\b(div|rem)\.u64\b' "$body")"
  mulhi="$(metric '\bmul\.hi\.u64\b' "$body")"
  mullo="$(metric '\bmul\.lo\.u64\b' "$body")"
  mulwide="$(metric '\bmul\.wide\.u64\b' "$body")"
  setp="$(metric '\bsetp\..*\.u64\b' "$body")"
  sub="$(metric '\bsub\.u64\b' "$body")"
  bra="$(metric '\bbra\b' "$body")"
  selp="$(metric '\bselp\.' "$body")"
  ldconst="$(metric '\bld\.const\..*\.u64\b' "$body")"
  if (( div != 0 )); then
    echo "$mode still contains u64 divide/remainder" >&2
    exit 4
  fi
  if [[ "$mode" != branchy ]] && (( mulhi < 1 )); then
    echo "$mode missing expected mul.hi.u64" >&2
    exit 5
  fi
  printf '%s_divrem_u64=%s\n%s_mulhi_u64=%s\n%s_mullo_u64=%s\n%s_mulwide_u64=%s\n%s_setp_u64=%s\n%s_sub_u64=%s\n%s_bra=%s\n%s_selp=%s\n%s_ldconst_u64=%s\n' \
    "$mode" "$div" "$mode" "$mulhi" "$mode" "$mullo" "$mode" "$mulwide" \
    "$mode" "$setp" "$mode" "$sub" "$mode" "$bra" "$mode" "$selp" "$mode" "$ldconst"
done

echo "gridfp-b300-shard-owner-mulhi-w28-ngpu8-ptx-proof OK arch=$ARCH main_magic=195888106327 main_high_shift=9 block_magic=139905900989 block_high_shift=7 exact_proof=1"
