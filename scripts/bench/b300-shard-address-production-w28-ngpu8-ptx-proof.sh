#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_80}"
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }

bash "$ONEESAN_ROOT/scripts/bench/b300-shard-owner-mulhi-w28-ngpu8-proof.sh"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
OUTDIR="${OUTDIR:-$ONEESAN_BUILD_DIR/b300_shard_address_production_w28_ngpu8_ptx}"
GENSRC="$OUTDIR/generated.cu"
PROBE="$OUTDIR/probe.cu"
PTX="$OUTDIR/probe.ptx"
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

"$NVCC" -O3 -std=c++17 -ptx -arch="$ARCH" \
  -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 \
  -DB300_FAST_SHARD_ADDRESS8=0 -DB300_SHARD_ADDRESS_MODE=2 \
  "$PROBE" -o "$PTX"

extract(){
  local kernel="$1" out="$2"
  awk -v k="$kernel" '
    $0 ~ "\\.entry[[:space:]]+" k { inside=1 }
    inside { print }
    inside && /^}/ { exit }
  ' "$PTX" > "$out"
  [[ -s "$out" ]] || { echo "missing PTX body for $kernel" >&2; exit 3; }
}
metric(){ local re="$1" file="$2"; grep -Ec "$re" "$file" || true; }
extract b300_generated_shard_main_probe "$OUTDIR/main.body.ptx"
extract b300_generated_shard_block_probe "$OUTDIR/block.body.ptx"

for kind in main block; do
  body="$OUTDIR/$kind.body.ptx"
  div="$(metric '\b(div|rem)\.u64\b' "$body")"
  mulhi="$(metric '\bmul\.hi\.u64\b' "$body")"
  mullo="$(metric '\bmul\.lo\.u64\b' "$body")"
  mulwide="$(metric '\bmul\.wide\.u64\b' "$body")"
  bra="$(metric '\bbra\b' "$body")"
  selp="$(metric '\bselp\.' "$body")"
  if (( div != 0 )); then
    echo "$kind generated production shard path still contains u64 divide/remainder" >&2
    grep -En '\b(div|rem)\.u64\b' "$body" >&2 || true
    exit 4
  fi
  if (( mulhi < 1 )); then
    echo "$kind generated production shard path missing mul.hi.u64" >&2
    exit 5
  fi
  printf '%s_divrem_u64=%s\n%s_mulhi_u64=%s\n%s_mullo_u64=%s\n%s_mulwide_u64=%s\n%s_bra=%s\n%s_selp=%s\n' \
    "$kind" "$div" "$kind" "$mulhi" "$kind" "$mullo" "$kind" "$mulwide" "$kind" "$bra" "$kind" "$selp"
done

grep -Fq 'B300_SHARD_ADDRESS_MODE=2 requires exactly 8 GPUs' "$GENSRC"
grep -Fq 'B300_SHARD_ADDRESS_MODE=2 W28x8 constants mismatch' "$GENSRC"
grep -Fq '__umul64hi(g,magic)>>9' "$GENSRC"
grep -Fq '__umul64hi(g,magic)>>7' "$GENSRC"

echo "b300-shard-address-production-w28-ngpu8-ptx-proof OK arch=$ARCH generated_source_guard=1 runtime_ngpu_guard=8 chunk_guard=1 main_magic=195888106327 block_magic=139905900989 exact_proof=1"
