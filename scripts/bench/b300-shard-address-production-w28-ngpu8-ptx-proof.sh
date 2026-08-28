#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_80}"
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }

bash "$ONEESAN_ROOT/scripts/bench/b300-shard-owner-mulhi-w28-ngpu8-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-shard-owner-u32limb-w28-ngpu8-proof.sh"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
OUTDIR="${OUTDIR:-$ONEESAN_BUILD_DIR/b300_shard_address_production_w28_ngpu8_ptx}"
GENSRC="$OUTDIR/generated.cu"
PROBE="$OUTDIR/probe.cu"
mkdir -p "$OUTDIR"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-shard-address-mode.py" "$SRC" "$GENSRC"

python3 - "$GENSRC" "$PROBE" <<'PY'
import json, pathlib, sys
src = pathlib.Path(sys.argv[1]).resolve()
out = pathlib.Path(sys.argv[2])
inc = json.dumps(src.as_posix())
out.write_text(f'''#pragma push_macro("main")
#undef main
#define main gridfp_b300_generated_main_unused
#include {inc}
#pragma pop_macro("main")

extern "C" __global__ void b300_generated_shard_main_probe(const Code* g, Count* out) {{
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    out[i]=global_load_main(g[i]);
}}
extern "C" __global__ void b300_generated_shard_block_probe(const Code* g, Count* out) {{
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    out[i]=global_load_block(g[i]);
}}
''')
PY

extract(){
  local ptx="$1" kernel="$2" out="$3"
  awk -v k="$kernel" '
    $0 ~ "\\.entry[[:space:]]+" k { inside=1 }
    inside { print }
    inside && /^}/ { exit }
  ' "$ptx" > "$out"
  [[ -s "$out" ]] || { echo "missing PTX body for $kernel" >&2; exit 3; }
}
metric(){ local re="$1" file="$2"; grep -Ec "$re" "$file" || true; }

for mode in 2 3; do
  ptx="$OUTDIR/mode_${mode}.ptx"
  "$NVCC" -O3 -std=c++17 -ptx -arch="$ARCH" \
    -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 \
    -DB300_FAST_SHARD_ADDRESS8=0 -DB300_SHARD_ADDRESS_MODE="$mode" \
    "$PROBE" -o "$ptx"
  extract "$ptx" b300_generated_shard_main_probe "$OUTDIR/mode_${mode}_main.body.ptx"
  extract "$ptx" b300_generated_shard_block_probe "$OUTDIR/mode_${mode}_block.body.ptx"
  for kind in main block; do
    body="$OUTDIR/mode_${mode}_${kind}.body.ptx"
    div="$(metric '\b(div|rem)\.u64\b' "$body")"
    mulhi64="$(metric '\bmul\.hi\.u64\b' "$body")"
    mullo64="$(metric '\bmul\.lo\.u64\b' "$body")"
    mulwide64="$(metric '\bmul\.wide\.u64\b' "$body")"
    mulhi32="$(metric '\bmul\.hi\.u32\b' "$body")"
    mullo32="$(metric '\bmul\.lo\.u32\b|\bmad\.lo\.u32\b' "$body")"
    bra="$(metric '\bbra\b' "$body")"
    selp="$(metric '\bselp\.' "$body")"
    if (( div != 0 )); then
      echo "mode=$mode $kind generated production shard path still contains u64 divide/remainder" >&2
      exit 4
    fi
    if [[ "$mode" == 2 ]] && (( mulhi64 < 1 )); then
      echo "mode=2 $kind missing expected mul.hi.u64" >&2
      exit 5
    fi
    if [[ "$mode" == 3 ]]; then
      if (( mulhi64 != 0 )); then
        echo "mode=3 $kind still contains mul.hi.u64" >&2
        exit 6
      fi
      if (( mulhi32 < 3 )); then
        echo "mode=3 $kind missing expected mul.hi.u32 limbs" >&2
        exit 7
      fi
    fi
    printf 'mode%s_%s_divrem_u64=%s\nmode%s_%s_mulhi_u64=%s\nmode%s_%s_mullo_u64=%s\nmode%s_%s_mulwide_u64=%s\nmode%s_%s_mulhi_u32=%s\nmode%s_%s_mullo_u32=%s\nmode%s_%s_bra=%s\nmode%s_%s_selp=%s\n' \
      "$mode" "$kind" "$div" "$mode" "$kind" "$mulhi64" "$mode" "$kind" "$mullo64" \
      "$mode" "$kind" "$mulwide64" "$mode" "$kind" "$mulhi32" "$mode" "$kind" "$mullo32" \
      "$mode" "$kind" "$bra" "$mode" "$kind" "$selp"
  done
done

grep -Fq 'B300_SHARD_ADDRESS_MODE>=2 requires exactly 8 GPUs' "$GENSRC"
grep -Fq 'B300_SHARD_ADDRESS_MODE>=2 W28x8 constants mismatch' "$GENSRC"
grep -Fq '__umul64hi(g,magic)>>9' "$GENSRC"
grep -Fq '__umul64hi(g,magic)>>7' "$GENSRC"
grep -Fq 'shard_owner8_u32limb<2614578007u,45u,9>' "$GENSRC"
grep -Fq 'shard_owner8_u32limb<2466947517u,32u,7>' "$GENSRC"

echo "b300-shard-address-production-w28-ngpu8-ptx-proof OK arch=$ARCH generated_source_guard=1 runtime_ngpu_guard=8 chunk_guard=1 mode2_mulhi64=1 mode3_mulhi64=0 mode3_umulhi32=3 exact_proof=1"
