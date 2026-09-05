#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_80}"
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
bash "$ONEESAN_ROOT/scripts/bench/b300-shard-address8-proof.sh"

SRC="$ONEESAN_ROOT/src/cuda/b300/probes/shard_address8_select_ptx_probe.cu"
OUTDIR="${OUTDIR:-$ONEESAN_BUILD_DIR/b300_shard_address8_select_ptx}"
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
BRANCHY="$OUTDIR/branchy.body.ptx"
SELECT="$OUTDIR/select.body.ptx"
extract gridfp_b300_shard_address8_branchy_probe "$BRANCHY"
extract gridfp_b300_shard_address8_select_probe "$SELECT"

metric(){ local re="$1" file="$2"; grep -Ec "$re" "$file" || true; }
for mode in branchy select; do
  [[ "$mode" == branchy ]] && body="$BRANCHY" || body="$SELECT"
  div="$(metric '\b(div|rem)\.u64\b' "$body")"
  mul="$(metric '\bmul\.(lo|hi|wide)\.u64\b' "$body")"
  setp="$(metric '\bsetp\..*\.u64\b' "$body")"
  sub="$(metric '\bsub\.u64\b' "$body")"
  bra="$(metric '\bbra\b' "$body")"
  selp="$(metric '\bselp\.' "$body")"
  if (( div != 0 || mul != 0 || setp < 3 || sub < 1 )); then
    echo "$mode shard PTX violates arithmetic contract div=$div mul=$mul setp=$setp sub=$sub" >&2
    exit 4
  fi
  printf '%s_divrem_u64=%s\n%s_mul_u64=%s\n%s_setp_u64=%s\n%s_sub_u64=%s\n%s_bra=%s\n%s_selp=%s\n' \
    "$mode" "$div" "$mode" "$mul" "$mode" "$setp" "$mode" "$sub" "$mode" "$bra" "$mode" "$selp"
done

echo "gridfp-b300-shard-address8-select-ptx-proof OK arch=$ARCH exact_proof=1"
